# suite_gepo2.log's end-of-run table has ONE void row — gsm8k (bug-633)

`suite_gepo2.sh` builds its table from `summary.json .score` for each arm with no
basis check. On `gsm8k_100_boxed` the arms do not share a basis:

| arm | filter its summary selected | score |
|---|---|---|
| qwenhybridp24_q6k (armJ) | `boxed` | 0.9600 |
| qwena3bgepo1_q6k | `flexible-extract` | 0.9700 |
| qwena3bgepo2_q6k | `flexible-extract` | 0.9700 |

So the logged row reads as a **+1pp gepo2 win over armJ. That win is a filter
artifact, not a measurement.** omk's canonical filter selection for this bench
changed between 2026-08-20 (armJ) and 2026-08-24 (gepo1) — a milder cousin of the
bug-604 scorer-fix pattern, where the results tree is silently partitioned by DATE.

The raw `results_*.json` carries BOTH filters for ALL THREE arms, so the matched
basis is recoverable with no re-run:

| basis | armJ | gepo1 | gepo2 |
|---|---|---|---|
| `exact_match,flexible-extract` | 0.9700 | 0.9700 | 0.9700 |
| `exact_match,boxed` | 0.9600 | 0.9700 | 0.9700 |

**gepo1 vs gepo2 is an exact tie on BOTH bases** — that conclusion is basis-independent.
Only the armJ comparison is basis-sensitive, and at most by one problem.

Every other bench in the suite was audited and all arms agree on (metric, filter)
and on `sampler.name = recommended`.

Regenerate the corrected table with `python suite_table.py` (basis-checked; refuses
rather than prints on a mismatch it cannot repair). **Quote that table, not the log's.**
The running script was deliberately NOT edited mid-run.
