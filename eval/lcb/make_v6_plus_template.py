"""Write eval/templates/lcb_v6_plus_73q.yaml as an exact sibling of lcb_v6_77q.

Everything that affects the measurement is copied verbatim from lcb_v6_77q -- generation
(greedy, frozen), backend_args, backend_overrides, scoring, reports, dependencies -- so
the two templates can be pooled into one 150-problem result. Only identity fields change:
name, n, the task-id file, the sqlite prefix, and the documented difficulty.
"""
import sys
from pathlib import Path

T = Path("/srv/ml/repos/omnimergekit/eval/templates")
src = (T / "lcb_v6_77q.yaml").read_text()

HDR = """# LiveCodeBench v6 MEDIUM-73 (lcb_v6_plus_73q) — the COMPANION half of a 150-problem
# measurement. lcb_v6_77q holds every hard problem the v6 window offers (verified: the
# pool at difficulty=hard, min_date=2024-01-01, testtype=functional is exactly 77, all
# already in that set), so this half is drawn from the 132-problem medium pool, sorted by
# task_id and asserted DISJOINT from the 77.
#
# USE: run both templates per cell and pool them. 77 + 73 = 150 problems. Every field
# below that affects the measurement is byte-identical to lcb_v6_77q -- same greedy
# generation block, same backend_args, same backend_overrides, same scorer -- so pooling
# is legitimate. Only name / n / task-id file / cache prefix differ.
#
# Build script: eval/lcb/build_lcb_v6_plus_73q.py (deterministic, re-runnable).
#
# NOTE ON DIFFICULTY: these are medium problems and medium saturates for strong models.
# Read the pooled 150 as the primary number, and keep the 77-hard split available --
# a effect that lives only in the hard half is a real finding, not an artifact.
"""

body = src.split("name: lcb_v6_77q", 1)[1]
out = HDR + "name: lcb_v6_plus_73q" + body
out = out.replace("n: 77", "n: 73")
out = out.replace("task_ids_file: eval/lcb/lcb_v6_77q_taskids.json",
                  "task_ids_file: eval/lcb/lcb_v6_plus_73q_taskids.json")
out = out.replace("difficulty: hard", "difficulty: medium")
out = out.replace("sqlite_prefix: lcb_v6_77q", "sqlite_prefix: lcb_v6_plus_73q")

p = T / "lcb_v6_plus_73q.yaml"
p.write_text(out)
print("wrote %s" % p)

# verify the measurement-bearing blocks match lcb_v6_77q exactly
import yaml
a = yaml.safe_load((T / "lcb_v6_77q.yaml").read_text())
b = yaml.safe_load(p.read_text())
bad = []
for k in ("generation", "backend_args", "backend_overrides", "scoring", "reports",
          "dependencies", "backend", "task"):
    if a.get(k) != b.get(k):
        bad.append(k)
print("measurement blocks identical to lcb_v6_77q: %s" % ("YES" if not bad else "NO -> " + ", ".join(bad)))
print("  n=%s  task_ids_file=%s  prefix=%s"
      % (b.get("n"), (b.get("selection") or {}).get("task_ids_file"),
         (b.get("cache") or {}).get("sqlite_prefix")))
g = b.get("generation") or {}
print("  greedy check: temp=%s top_p=%s top_k=%s do_sample=%s"
      % (g.get("temperature"), g.get("top_p"), g.get("top_k"), g.get("do_sample")))
if bad:
    sys.exit(1)
