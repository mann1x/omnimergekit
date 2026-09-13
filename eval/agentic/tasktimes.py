"""Per-task execution time from the inspect .eval logs.

inspect records TWO clocks per sample:
  total_time   = wall clock for the sample, start to finish (includes queueing
                 behind other samples' model calls, tool execution, retries)
  working_time = time actually spent doing work (model calls + tools), with
                 retry/backoff waits excluded
Both are already-reported metrics -- nothing is being re-derived here.
"""
import sys, statistics as st
from inspect_ai.log import read_eval_log

def load(p):
    log = read_eval_log(p)
    out = {}
    for s in (log.samples or []):
        tot = getattr(s, "total_time", None)
        wrk = getattr(s, "working_time", None)
        out[str(s.id)] = (tot, wrk)
    return out, log.status

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def desc(name, vals):
    if not vals:
        print("  %-22s (none)" % name); return
    print("  %-22s n=%3d  p50=%8.1f  p90=%8.1f  max=%8.1f  sum=%10.0f s (%.2f h)"
          % (name, len(vals), q(vals, .5), q(vals, .9), max(vals), sum(vals), sum(vals) / 3600))

A, sa = load(sys.argv[1])
B, sb = load(sys.argv[2])
elapsed = float(sys.argv[3])
print("status: armA=%s (%d samples)   armB=%s (%d samples)" % (sa, len(A), sb, len(B)))
print("\n=== per-task TOTAL time (wall, seconds) ===")
desc("armA all done", [a for a, _ in A.values() if a is not None])
desc("armB all done", [a for a, _ in B.values() if a is not None])
shared = sorted(set(A) & set(B))
print("\n=== per-task TOTAL time, SHARED %d tasks ===" % len(shared))
desc("armA shared", [A[k][0] for k in shared if A[k][0] is not None])
desc("armB shared", [B[k][0] for k in shared if B[k][0] is not None])
r = [A[k][0] / B[k][0] for k in shared if A[k][0] and B[k][0]]
if r: print("  median per-task RATIO armA/armB = %.3f  (n=%d)" % (st.median(r), len(r)))

print("\n=== per-task WORKING time, SHARED ===")
desc("armA shared", [A[k][1] for k in shared if A[k][1] is not None])
desc("armB shared", [B[k][1] for k in shared if B[k][1] is not None])
rw = [A[k][1] / B[k][1] for k in shared if A[k][1] and B[k][1]]
if rw: print("  median per-task RATIO armA/armB = %.3f  (n=%d)" % (st.median(rw), len(rw)))

print("\n=== armB ONLY: the 129 tasks armA has NOT finished ===")
rest = sorted(set(B) - set(A))
desc("armB on those", [B[k][0] for k in rest if B[k][0] is not None])

print("\n=== time accounting against %.0f s elapsed ===" % elapsed)
for nm, D in (("armA", A), ("armB", B)):
    w = [x for _, x in D.values() if x is not None]
    print("  %s: working-time sum %.0f s over %d done -> effective concurrency %.2fx"
          % (nm, sum(w), len(w), sum(w) / elapsed))
