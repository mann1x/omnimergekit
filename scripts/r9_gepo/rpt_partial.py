"""Paired mid-run readout: repeat draw vs banked draw 1, on the COMMON problems only.

PROVISIONAL -- the run is unfinished. The only legitimate mid-run comparison is
draw2's completed problems against draw1 ON THOSE SAME task_ids. Comparing a
partial rate to draw1's FULL 0.7532 would be a subset-vs-full artefact, since the
LCB list is not difficulty-ordered and the remaining tail is not exchangeable.

Reports McNemar on the paired flips, which is the actual quantity of interest:
how many problems change verdict between two independent draws of the SAME model
on the SAME basis. That is the noise floor the gepo2-vs-armJ comparison
(13 discordant, -9.33pp, p=0.1185) has to be judged against.
"""
import json
import re
from math import comb

BANKED = ("/srv/ml/eval_results/qwen_suite/lcb_v6_77q/"
          "qwena3bgepo2_q6k/lcb_result.samples.jsonl")
LOG = "/mnt/sdc/ml/brevity/gepo/lcb_repeat.log"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


d1 = {}
for line in open(BANKED, errors="ignore"):
    if line.strip():
        r = json.loads(line)
        d1[r.get("task_id")] = (bool(r.get("passed")), len(r.get("completion") or ""))

pat = re.compile(r"\[(\d+)/77\]\s+(\S+)\s+(PASS|FAIL)\s+([\d.]+)s\s+chars=(\d+)")
d2 = {}
for line in open(LOG, errors="ignore"):
    m = pat.search(line)
    if m:
        d2[m.group(2)] = (m.group(3) == "PASS", int(m.group(5)))

common = [t for t in d2 if t in d1]
p1 = sum(d1[t][0] for t in common)
p2 = sum(d2[t][0] for t in common)
b = sum(1 for t in common if d1[t][0] and not d2[t][0])   # draw1-only wins
c = sum(1 for t in common if d2[t][0] and not d1[t][0])   # draw2-only wins
n = len(common)
print(f"PROVISIONAL -- {n}/77 problems done in draw 2\n")
print(f"  paired on the SAME {n} problems:")
print(f"    draw1 {p1}/{n} = {p1/n:.4f}")
print(f"    draw2 {p2}/{n} = {p2/n:.4f}   delta {100*(p2-p1)/n:+.2f}pp")
print(f"  discordant: draw1-only={b}  draw2-only={c}  "
      f"total flips={b+c}  McNemar p={mcnemar(b,c):.4f}")
ch1 = sorted(d1[t][1] for t in common)
ch2 = sorted(d2[t][1] for t in common)
print(f"  chars p50: draw1 {ch1[len(ch1)//2]}  draw2 {ch2[len(ch2)//2]}")
print(f"\n  reference: gepo2-vs-armJ (different models) was 13 flips, "
      f"p=0.1185 on 70 both-untruncated")
