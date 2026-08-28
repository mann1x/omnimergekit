"""MPE-100 cap cross-tab + paired both-uncapped comparison.

WHY THIS EXISTS
---------------
omk records finish_reasons {'None': 300} for the native MultiPL-E backend -- the
runner does not surface a finish_reason, so `trunc` reads 0/300 on every arm and the
truncation signal is BLIND on this bench (bug-592). Meanwhile max=1023 against
--max-tokens 1024 proves at least one completion WAS clipped.

That matters because a truncated program does not compile, so on MPE capped => fail,
always. If the cap binds asymmetrically across arms, the headline score delta is
measuring OUTPUT LENGTH, not coding capability. Measured 2026-08-24 on exactly these
two arms: armJ 0.7000 / gepo1 0.7267, gepo1 capped 8 fewer times and won by exactly 8
problems; on the 256 uncapped in BOTH arms it was 207 vs 207, +0.00 pp.

So the score is only a capability claim on the problems uncapped in both arms. This
script builds that pairing.

CAP DEFINITION: omk's own -- tokens of the completion, add_special_tokens=False,
>= max_gen_toks - 16. Reusing it is what makes these numbers reconcile with omk's.
PASS: results/<lang>/<name>.results.json -> results[].status == "OK". It is NOT in
the samples file; omk never built this cross-tab.
"""
import json
import os
import sys
from collections import defaultdict

from transformers import AutoTokenizer

RES = "/srv/ml/eval_results/ream_arms/multipl_e_100"
TOKENIZER = "/mnt/sdc/ream-work/armJ"
MAX_GEN_TOKS = 1024
CAP_TOL = 16
ARMS = [("armJ_b604", "hybrid_p24_ourssal_reapfloor_b604"),
        ("armJ_pre", "hybrid_p24_ourssal_reapfloor"),
        ("gepo1", "a3b_gepo1"),
        ("gepo2", "a3b_gepo2")]

tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def load_arm(name):
    """-> {task_id: (n_tokens, capped, passed)}"""
    root = os.path.join(RES, name)
    out = {}
    with open(os.path.join(root, "mpe_result.samples.jsonl")) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            tid = d.get("task_id") or d.get("doc_id")
            resps = d.get("resps") or [[""]]
            text = resps[0][0] if resps and resps[0] else ""
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            out[tid] = [n, n >= MAX_GEN_TOKS - CAP_TOL, None]

    # pass status lives per-problem in the results tree, keyed lang::problem
    rroot = os.path.join(root, "results")
    for lang_dir in sorted(os.listdir(rroot)):
        lang = lang_dir.replace("humaneval-", "")
        for fn in sorted(os.listdir(os.path.join(rroot, lang_dir))):
            if not fn.endswith(".results.json"):
                continue
            prob = fn[: -len(".results.json")]
            tid = f"{lang}::{prob}"
            with open(os.path.join(rroot, lang_dir, fn)) as fh:
                r = json.load(fh)
            sts = [x.get("status") for x in (r.get("results") or [])]
            ok = bool(sts) and all(s == "OK" for s in sts)
            if tid in out:
                out[tid][2] = ok
            else:
                out[tid] = [None, None, ok]
    return out


arms = {tag: load_arm(name) for tag, name in ARMS}

print(f"{'arm':<7} {'n':>5} {'pass':>6} {'score':>8} {'capped':>7} "
      f"{'capped&pass':>12} {'p50 tok':>8} {'mean tok':>9}")
print("-" * 70)
for tag, _ in ARMS:
    d = arms[tag]
    toks = sorted(v[0] for v in d.values() if v[0] is not None)
    npass = sum(1 for v in d.values() if v[2])
    ncap = sum(1 for v in d.values() if v[1])
    cappass = sum(1 for v in d.values() if v[1] and v[2])
    p50 = toks[len(toks) // 2] if toks else 0
    mean = sum(toks) / len(toks) if toks else 0
    print(f"{tag:<7} {len(d):>5} {npass:>6} {npass/len(d):>8.4f} {ncap:>7} "
          f"{cappass:>12} {p50:>8} {mean:>9.0f}")

print()
print("=== paired: problems UNCAPPED IN BOTH ARMS (the capability question) ===")
for a, b in (("armJ_b604", "gepo1"), ("armJ_b604", "gepo2"), ("gepo1", "gepo2"), ("armJ_pre", "armJ_b604")):
    da, db = arms[a], arms[b]
    common = sorted(set(da) & set(db))
    both = [t for t in common if not da[t][1] and not db[t][1]]
    pa = sum(1 for t in both if da[t][2])
    pb = sum(1 for t in both if db[t][2])
    # McNemar discordant pairs: b=c means the difference is symmetric = noise
    disc_a = sum(1 for t in both if da[t][2] and not db[t][2])
    disc_b = sum(1 for t in both if db[t][2] and not da[t][2])
    n = len(both)
    if not n:
        print(f"{a} vs {b}: no commonly-uncapped problems"); continue
    print(f"{a:>5} vs {b:<5}  n_both_uncapped={n:>3}  "
          f"{a}={pa}/{n} ({pa/n:.4f})  {b}={pb}/{n} ({pb/n:.4f})  "
          f"delta={((pb-pa)/n)*100:+.2f}pp  discordant {a}-only={disc_a} {b}-only={disc_b}")
    # and the raw all-problems delta, for contrast
    ra = sum(1 for t in common if da[t][2]) / len(common)
    rb = sum(1 for t in common if db[t][2]) / len(common)
    print(f"{'':>5}    {'':<5}  raw all n={len(common)}: {ra:.4f} vs {rb:.4f} "
          f"= {((rb-ra))*100:+.2f}pp   <- how much of this is the cap?")

print()
print("=== cap asymmetry ===")
for tag, _ in ARMS:
    d = arms[tag]
    print(f"  {tag}: capped {sum(1 for v in d.values() if v[1])}/{len(d)}")
print("A capped completion cannot compile, so capped => fail. If the arms differ in")
print("cap count, that difference alone moves the headline score.")
