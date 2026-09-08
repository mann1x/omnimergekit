import json, glob, collections

R = "/srv/ml/eval_results/qwen_suite"
CELLS = {
 ("HE",  "pub"):  R+"/humaneval_full_think/qwencodermpe_q6k",
 ("HE",  "armJ"): R+"/humaneval_full_think/qwenhybridp24_q6k",
 ("HE+", "pub"):  R+"/humanevalplus_full_think/qwencodermpe_q6k",
 ("HE+", "armJ"): R+"/humanevalplus_full_think/qwenhybridp24_q6k",
}

def load(d):
    fs = sorted(glob.glob(d+"/lm_eval_out/*/samples_*.jsonl"))
    assert len(fs) == 1, (d, fs)
    out = {}
    for line in open(fs[0]):
        r = json.loads(line)
        doc = r.get("doc") or {}
        tid = doc.get("task_id") or r.get("doc_id")
        # pass@1 lives under the metric key; be tolerant about its exact name
        p = None
        for k in ("pass@1", "pass@1,extract_chat", "pass_at_1"):
            if k in r: p = r[k]; break
        if p is None:
            for k, v in r.items():
                if k.startswith("pass@1"): p = v; break
        resps = r.get("filtered_resps") or r.get("resps") or []
        txt = resps[0] if resps and isinstance(resps[0], str) else (
              resps[0][0] if resps and isinstance(resps[0], (list, tuple)) and resps[0] else "")
        out[tid] = (float(p), txt)
    return out

D = {k: load(v) for k, v in CELLS.items()}
for k, v in D.items():
    print("%-4s %-5s n=%d  pass=%d  score=%.4f" % (k[0], k[1], len(v),
          sum(1 for p, _ in v.values() if p >= 1.0),
          sum(p for p, _ in v.values())/len(v)))

ids = sorted(set(D[("HE","pub")]) & set(D[("HE","armJ")])
           & set(D[("HE+","pub")]) & set(D[("HE+","armJ")]))
print("\ncommon task_ids: %d" % len(ids))

# ---- are HE and HE+ the SAME generations, or independent draws? ----
for arm in ("pub", "armJ"):
    same = sum(1 for i in ids if D[("HE",arm)][i][1].strip() == D[("HE+",arm)][i][1].strip())
    print("  %-5s HE vs HE+ identical completions: %d/%d" % (arm, same, len(ids)))

# ---- per-bench arm cross-tab ----
for b in ("HE", "HE+"):
    c = collections.Counter()
    for i in ids:
        c[(D[(b,"pub")][i][0] >= 1.0, D[(b,"armJ")][i][0] >= 1.0)] += 1
    both, ponly, jonly, neither = c[(1,1)], c[(1,0)], c[(0,1)], c[(0,0)]
    print("\n%s  both=%d  pub-only=%d  armJ-only=%d  neither=%d   net armJ-pub = %+d"
          % (b, both, ponly, jonly, neither, jonly - ponly))
    if ponly: print("   pub-only  (armJ fails): %s" % ", ".join(i for i in ids if D[(b,"pub")][i][0]>=1 and D[(b,"armJ")][i][0]<1))
    if jonly: print("   armJ-only (pub  fails): %s" % ", ".join(i for i in ids if D[(b,"pub")][i][0]<1 and D[(b,"armJ")][i][0]>=1))

# ---- does HE+ strictly subsume HE? (extra tests can only turn pass->fail) ----
print()
for arm in ("pub", "armJ"):
    up = [i for i in ids if D[("HE",arm)][i][0] < 1 <= D[("HE+",arm)][i][0]]
    dn = [i for i in ids if D[("HE+",arm)][i][0] < 1 <= D[("HE",arm)][i][0]]
    print("%-5s HE fail -> HE+ pass: %d %s" % (arm, len(up), up[:8]))
    print("%-5s HE pass -> HE+ fail: %d %s" % (arm, len(dn), dn[:8]))
