"""Mid-run LCB check, paired on the task_ids completed SO FAR -- pass AND length.

Channel verified 2026-08-26 by reading the producer: eval/lcb/lcb_llama_server.py
stores "completion": completion (line 349/368) and logs chars=len(completion)
(line 373) with no reassignment between. So the log's chars IS the samples file's
completion length -- same channel, comparable. (Do not compare either to lm-eval
`resps`, which is content-only; that was bug-635.)

Pass AND length are joined on the completed ids only. Comparing gepo2's running p50
to a ref's all-77 p50 would be a composition artifact, since LCB is id-ordered and
not difficulty- or length-balanced.
"""
import json, os, re
from math import comb

LOG = "/mnt/sdc/ml/brevity/gepo/suite_gepo2.log"
R = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q"
REFS = [("armJ", "qwenhybridp24_q6k"), ("gepo1", "qwena3bgepo1_q6k")]

done, chars = {}, {}
for ln in open(LOG):
    m = re.match(r"\[(\d+)/(\d+)\]\s+(\S+)\s+(PASS|FAIL)\s+([\d.]+)s\s+chars=(\d+)", ln.strip())
    if m:
        done[m.group(3)] = (m.group(4) == "PASS")
        chars[m.group(3)] = int(m.group(6))
if not done:
    raise SystemExit("REFUSE: no completed problems parsed")


def med(v):
    v = sorted(v); return v[len(v) // 2] if v else 0


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print("gepo2 partial: %d done, %d PASS (%.4f)  chars p50=%d"
      % (len(done), sum(done.values()), sum(done.values()) / len(done), med(chars.values())))

for tag, name in REFS:
    p = os.path.join(R, name, "lcb_result.samples.jsonl")
    ref = {}
    for ln in open(p):
        if ln.strip():
            d = json.loads(ln)
            ref[d.get("task_id") or d.get("doc_id")] = (
                bool(d.get("passed")), len(d.get("completion") or ""),
                d.get("finish_reason"), d.get("completion_tokens"))
    common = sorted(t for t in done if t in ref)
    if not common:
        print("  %-6s: 0 overlap" % tag); continue
    a = sum(1 for t in common if ref[t][0]); b = sum(1 for t in common if done[t])
    d1 = sum(1 for t in common if ref[t][0] and not done[t])
    d2 = sum(1 for t in common if done[t] and not ref[t][0])
    rc = [ref[t][1] for t in common]; gc = [chars[t] for t in common]
    # paired per-problem length delta -- the only length read that is not a composition artifact
    delta = [chars[t] - ref[t][1] for t in common]
    shorter = sum(1 for d in delta if d < 0)
    ntr = sum(1 for t in common if ref[t][2] == "length")
    print("  %-6s n=%d | PASS %s=%d gepo2=%d disc %d/%d p=%.4f"
          % (tag, len(common), tag, a, b, d1, d2, mcnemar(d1, d2)))
    print("         CHARS %s p50=%d  gepo2 p50=%d  (%+.1f%%) | paired median delta=%+d | gepo2 shorter on %d/%d | %s trunc(length)=%d"
          % (tag, med(rc), med(gc), (med(gc) - med(rc)) / med(rc) * 100,
             med(delta), shorter, len(common), tag, ntr))

print("\nPARTIAL -- not a result.")
