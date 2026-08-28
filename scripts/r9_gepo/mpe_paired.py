"""Paired MPE cell: restrict to problems UNCAPPED IN BOTH arms (matched post-b604)."""
import json, os, glob
from transformers import AutoTokenizer

ROOT = "/srv/ml/eval_results/ream_arms/multipl_e_100"
MAX_GEN, TOL = 1024, 16
CUT = MAX_GEN - TOL
A = ("armJ",  "hybrid_p24_ourssal_reapfloor_b604")
B = ("gepo1", "a3b_gepo1")
tok = AutoTokenizer.from_pretrained("/mnt/sdc/ream-work/armJ", trust_remote_code=True)


def passed(e):
    res = e.get("results") or []
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


def cell(cn):
    d = os.path.join(ROOT, cn)
    st = {}
    for p in glob.glob(os.path.join(d, "results", "*", "*.results.json")):
        e = json.load(open(p))
        st["%s::%s" % (e.get("language"), e.get("name"))] = passed(e)
    ln = {}
    for line in open(os.path.join(d, "mpe_result.samples.jsonl")):
        r = json.loads(line)
        t = r.get("completion") or ""
        ln[r["task_id"]] = len(tok(t, add_special_tokens=False).input_ids)
    return st, ln


sa, la = cell(A[1])
sb, lb = cell(B[1])
keys = [k for k in la if k in lb and k in sa and k in sb]
both = [k for k in keys if la[k] < CUT and lb[k] < CUT]
pa = sum(1 for k in both if sa[k])
pb = sum(1 for k in both if sb[k])

print("all problems:                 %d" % len(keys))
print("uncapped in BOTH arms:        %d   (dropped %d where either arm hit the 1024 ceiling)"
      % (len(both), len(keys) - len(both)))
print()
print("PAIRED cell (identical problem set, neither arm truncated):")
print("  armJ  %d/%d = %.4f" % (pa, len(both), pa / len(both)))
print("  gepo1 %d/%d = %.4f" % (pb, len(both), pb / len(both)))
print("  delta %+.2f pp  (%+d problems)" % ((pb - pa) / len(both) * 100, pb - pa))
print()
# McNemar discordant pairs
b_only = sum(1 for k in both if sb[k] and not sa[k])
a_only = sum(1 for k in both if sa[k] and not sb[k])
print("  discordant pairs: gepo1-only wins %d, armJ-only wins %d  (McNemar b=%d c=%d)"
      % (b_only, a_only, b_only, a_only))
print()
print("per-language on the paired set:")
for lang in ("rs", "java", "js"):
    ks = [k for k in both if k.startswith(lang + "::")]
    if not ks:
        continue
    xa = sum(1 for k in ks if sa[k]); xb = sum(1 for k in ks if sb[k])
    print("  %-5s n=%3d   armJ %.3f   gepo1 %.3f   %+.1f pp"
          % (lang, len(ks), xa / len(ks), xb / len(ks), (xb - xa) / len(ks) * 100))
