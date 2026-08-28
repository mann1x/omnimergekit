"""MPE-100 cap x pass cross-tab.

omk's own capped_correct is 0/0 on MPE because its _correct() reads the samples
file, which carries no pass flag -- MPE pass status lives in results/*.results.json.
So the cross-tab was never actually built. This builds it, using omk's OWN cap
definition (eval/omk_eval.py:1546): len >= max_gen_toks - TOL, TOL=16, measured on
_text(s) = the `completion` field, tokenized with add_special_tokens=False.
"""
import json, os, glob, sys
from transformers import AutoTokenizer

ROOT = "/srv/ml/eval_results/ream_arms/multipl_e_100"
TOKDIR = "/mnt/sdc/ream-work/armJ"
MAX_GEN, TOL = 1024, 16
CUT = MAX_GEN - TOL
CELLS = [("armJ pre-b604 (VOID)", "hybrid_p24_ourssal_reapfloor"),
         ("armJ post-b604",       "hybrid_p24_ourssal_reapfloor_b604"),
         ("gepo1 post-b604",      "a3b_gepo1")]

tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)

STATUS_KEYS = ("status", "exit_code", "passed", "result")


def passed(entry):
    """True iff every executed program for this problem succeeded."""
    res = entry.get("results") or []
    if not res:
        return None
    oks = []
    for r in res:
        if not isinstance(r, dict):
            continue
        if "status" in r:
            oks.append(str(r["status"]).upper() in ("OK", "PASS", "PASSED"))
        elif "exit_code" in r:
            oks.append(r["exit_code"] == 0)
        elif "passed" in r:
            oks.append(bool(r["passed"]))
    return all(oks) if oks else None


def cell(cellname):
    d = os.path.join(ROOT, cellname)
    # pass status per task
    st = {}
    for p in glob.glob(os.path.join(d, "results", "*", "*.results.json")):
        if p.endswith("_summary.json"):
            continue
        e = json.load(open(p))
        st["%s::%s" % (e.get("language"), e.get("name"))] = passed(e)
    # completion length per task, omk's definition
    ln = {}
    with open(os.path.join(d, "mpe_result.samples.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("completion")
            if not (isinstance(t, str) and t):
                rs = r.get("resps") or []
                t = (rs[0][0] if rs and isinstance(rs[0], list) else "") or ""
            ln[r.get("task_id")] = len(tok(t, add_special_tokens=False).input_ids)
    return st, ln


print("cap definition: completion tokens >= %d  (max_gen_toks %d - TOL %d), omk eval/omk_eval.py:1546"
      % (CUT, MAX_GEN, TOL))
print()
hdr = ("%-22s %6s %8s %9s %9s %10s %10s %11s"
       % ("arm", "n", "capped", "cap PASS", "cap FAIL", "uncapped", "unc PASS", "unc pass%"))
print(hdr); print("-" * len(hdr))
store = {}
for tag, cn in CELLS:
    st, ln = cell(cn)
    keys = [k for k in ln if k in st]
    capped = [k for k in keys if ln[k] >= CUT]
    unc = [k for k in keys if ln[k] < CUT]
    cp = sum(1 for k in capped if st[k])
    up = sum(1 for k in unc if st[k])
    store[tag] = (len(keys), len(capped), cp, len(unc), up)
    print("%-22s %6d %8d %9d %9d %10d %10d %10.2f%%"
          % (tag, len(keys), len(capped), cp, len(capped) - cp, len(unc), up,
             100.0 * up / max(len(unc), 1)))

a = store.get("armJ post-b604"); b = store.get("gepo1 post-b604")
if a and b:
    print()
    print("  ON THE MATCHED (post-b604) PAIR:")
    print("    capped:          armJ %d   gepo1 %d   (gepo1 %+d)" % (a[1], b[1], b[1] - a[1]))
    print("    capped that PASS: armJ %d   gepo1 %d" % (a[2], b[2]))
    print("    overall pass:    armJ %d/%d   gepo1 %d/%d   (%+d problems)"
          % (a[2] + a[4], a[0], b[2] + b[4], b[0], (b[2] + b[4]) - (a[2] + a[4])))
    ra, rb = a[4] / max(a[3], 1), b[4] / max(b[3], 1)
    print("    UNCAPPED-ONLY pass rate: armJ %.4f (%d/%d)   gepo1 %.4f (%d/%d)   delta %+.2f pp"
          % (ra, a[4], a[3], rb, b[4], b[3], (rb - ra) * 100))
