"""Build lcb_v6_plus_73q — 73 problems DISJOINT from the existing hard 77, so the two
templates together give one 150-problem measurement.

Selection rule (as specified): take any remaining HARD problems first; fill the rest from
MEDIUM. The existing 77 is believed to be all or nearly all of the hard pool in the v6
window, so most of the 73 will be medium — this script reports the actual split rather
than assuming it.

Disjointness from lcb_v6_77q is enforced, not hoped for: the 77 ids are subtracted before
selection and the result is asserted to have zero intersection.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/srv/ml/repos/omnimergekit")
sys.path.insert(0, "/srv/ml/repos/omnimergekit/eval/lcb")
import lcb_helpers  # noqa: E402

L = Path("/srv/ml/repos/omnimergekit/eval/lcb")
EXISTING = set(json.load(open(L / "lcb_v6_77q_taskids.json")))
WANT = 73
MIN_DATE = "2024-01-01"      # same contamination window as lcb_v6_77q
BIG = 100000                 # "no limit" — load_lcb takes a positional limit

print("existing hard-77 ids: %d" % len(EXISTING))

def pull(diff):
    try:
        rows = lcb_helpers.load_lcb(BIG, difficulty=diff, min_date=MIN_DATE,
                                    testtype="functional")
    except Exception as e:
        print("  load_lcb(%s) failed: %s" % (diff, e))
        return []
    return rows

hard = pull("hard")
med = pull("medium")
print("v6 pool at min_date=%s, testtype=functional:" % MIN_DATE)
print("  hard   : %d  (of which already in the 77: %d)"
      % (len(hard), sum(1 for r in hard if r["task_id"] in EXISTING)))
print("  medium : %d  (of which already in the 77: %d)"
      % (len(med), sum(1 for r in med if r["task_id"] in EXISTING)))

# hard first, then medium; both deterministic (sorted by task_id), both disjoint from the 77
extra_hard = sorted({r["task_id"] for r in hard} - EXISTING)
extra_med = sorted({r["task_id"] for r in med} - EXISTING)
print()
print("available NEW hard  : %d" % len(extra_hard))
print("available NEW medium: %d" % len(extra_med))

picked = extra_hard[:WANT]
if len(picked) < WANT:
    picked += extra_med[:WANT - len(picked)]
picked = sorted(picked)

n_hard = sum(1 for t in picked if t in set(extra_hard))
n_med = len(picked) - n_hard
print()
print("SELECTED %d: %d hard + %d medium" % (len(picked), n_hard, n_med))
if len(picked) < WANT:
    print("WARNING: only %d available, wanted %d" % (len(picked), WANT))

assert not (set(picked) & EXISTING), "FATAL: selection overlaps the existing 77"
print("disjointness from the 77: VERIFIED (intersection = 0)")
print("combined bench size: %d + %d = %d" % (len(EXISTING), len(picked), len(EXISTING) + len(picked)))

out = L / "lcb_v6_plus_73q_taskids.json"
out.write_text(json.dumps(picked, indent=1))
print("wrote %s" % out)
