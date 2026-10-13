"""Derive --classes/--class-weights for the v9 drop map from the map's OWN metadata.

Policy (ported from published STD16 = v8coder_fk16_drop_map.json):
  generic_code            3   the code-specialist signal
  generic_{math,logic,science,creative}  1
  generic_multilingual    0   -> ABSENT in v9 (needs --multilingual-calib, not passed).
                              Under strategy=max a w=0 class can never win the max, so
                              omitting it is equivalent, not an approximation. It does
                              change the breadth_bonus denominator, but v9's class basis
                              (12 classes) differs from STD16's (9) regardless.
  targeted_*              2   STD16 gave its ONE targeted code bench w=2. w is a per-class
                              MULTIPLIER under max, not a mass to divide, so each of the 7
                              agentic code benches gets 2. Splitting 2/7 would drop a
                              language specialist from 2.0 to 0.29 and delete the axis.
Unknown category -> hard fail, rather than silently weighting it.
"""
import json, sys

data = sys.argv[1]
cats = json.load(open(data))["metadata"]["categories"]

GENERIC = {"generic_code": 3, "generic_math": 1, "generic_logic": 1,
           "generic_science": 1, "generic_creative": 1, "generic_multilingual": 0}
classes, weights, unknown = [], [], []
for c in cats:
    if c in GENERIC:
        classes.append(c); weights.append(GENERIC[c])
    elif c.startswith("targeted_"):
        classes.append(c); weights.append(2)
    elif c.startswith("agentic_eog"):
        continue  # emit-position category, never a pruning class
    else:
        unknown.append(c)
if unknown:
    sys.exit("UNKNOWN categories, refusing to guess a weight: %r" % unknown)
if "generic_code" not in classes:
    sys.exit("generic_code absent — this is not a code-variant map")
if not any(c.startswith("targeted_") for c in classes):
    sys.exit("no targeted_* categories — Tier-B produced nothing")

sys.stderr.write("[v9classes] %d classes: %s\n"
                 % (len(classes), " ".join("%s=%d" % (c, w)
                                           for c, w in zip(classes, weights))))
print(" ".join(classes))
print(" ".join(str(w) for w in weights))
