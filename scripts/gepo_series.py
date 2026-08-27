"""Print the full GEPO run2 generation-round series + matched-index run1 comparison."""
import ast
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/ml/brevity/gepo/full_run2.log"

# run1's clipped_ratio, in round order, read from full.log during the 2026-08-25
# sizing exercise. Used as the matched-round-index anchor -- comparing run2's
# early rounds against run1's whole-run MEAN is invalid, because that mean
# already contains run1's own trained-down late rounds.
RUN1_CLIPPED = [0.50, 0.5312, 0.3125, 0.0625, 0.2812, 0.50, 0.4062, 0.3125,
                0.3438, 0.375, 0.1562, 0.50, 0.4375, 0.5312, 0.1562, 0.125]
RUN1_MEANLEN = [5699, 5706, 4629, 4142, 4034, 6342, 4820, 5035,
                5666, 4994, 5103, 7305, 5395, 5289, 4704, 4370]

rows = []
for line in open(LOG, errors="ignore"):
    m = re.search(r"\{.*?'grad_norm'.*?\}", line)
    if not m:
        continue
    try:
        rows.append(ast.literal_eval(m.group(0)))
    except (ValueError, SyntaxError):
        pass

gen = [r for r in rows if "completions/clipped_ratio" in r]
if not gen:
    sys.exit("no generation rounds parsed from %s" % LOG)


def f(r, k):
    return float(r[k])


print("%-5s %9s %9s %9s %9s %9s %9s"
      % ("rnd", "mean_len", "term_len", "clipped", "reward", "grad_nrm", "step_s"))
for i, r in enumerate(gen, 1):
    print("%-5d %9.0f %9.0f %9.4f %9.4f %9.4f %9.0f"
          % (i, f(r, "completions/mean_length"), f(r, "completions/mean_terminated_length"),
             f(r, "completions/clipped_ratio"), f(r, "reward"),
             f(r, "grad_norm"), f(r, "step_time")))

cr = [f(r, "completions/clipped_ratio") for r in gen]
ml = [f(r, "completions/mean_length") for r in gen]
n = len(cr)
mean = lambda xs: sum(xs) / len(xs)

# run1 ran only 16 rounds. Slicing RUN1_* by n silently stops matching once
# n > 16 -- Python returns the short list without complaint, so the row would
# keep printing "1..20" while comparing 20 run2 rounds against 16 run1 ones.
# Clamp to the anchor's length and say so.
k = min(n, len(RUN1_CLIPPED))
print()
print("=== matched round index 1..%d (run1 has only %d rounds; run2 truncated to match) ==="
      % (k, len(RUN1_CLIPPED)))
print("  clipped   run2 %.4f   run1 %.4f   delta %+.4f"
      % (mean(cr[:k]), mean(RUN1_CLIPPED[:k]), mean(cr[:k]) - mean(RUN1_CLIPPED[:k])))
print("  mean_len  run2 %.0f     run1 %.0f     delta %+.0f"
      % (mean(ml[:k]), mean(RUN1_MEANLEN[:k]), mean(ml[:k]) - mean(RUN1_MEANLEN[:k])))
if n > k:
    print("  (run2 rounds %d-%d have NO run1 counterpart: clipped %.4f, mean_len %.0f"
          " -- read against run2's own earlier rounds only)"
          % (k + 1, n, mean(cr[k:]), mean(ml[k:])))
h = n // 2
if h:
    print()
    print("=== within-run trend ===")
    print("  run2 clipped  rounds 1-%d %.4f -> rounds %d-%d %.4f"
          % (h, mean(cr[:h]), h + 1, n, mean(cr[h:])))
    print("  run2 mean_len rounds 1-%d %.0f   -> rounds %d-%d %.0f"
          % (h, mean(ml[:h]), h + 1, n, mean(ml[h:])))
print()
print("NOTE: the within-run trend is confounded with training progress -- GEPO is")
print("      optimising for brevity, so length falling over rounds is the objective")
print("      working, not evidence about the cap. Only the matched-index row above")
print("      speaks to the 8192 -> 12288 change.")
