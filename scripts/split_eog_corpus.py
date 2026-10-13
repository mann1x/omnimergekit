"""Split an emit corpus into two disjoint halves for a stability check.

The t106 map rests on 71 model-owned positions -- ~4.4 expert observations per
(layer, expert) cell. "Thin" is an adjective until you measure it: build the map on two
disjoint halves and ask how much their top-k expert sets actually overlap. If the halves
agree, thinness is tolerable; if they do not, the map is noise and no pin drawn from it
means anything. Deterministic interleave, not a shuffle, so it is reproducible.
"""
import sys

src, out_a, out_b = sys.argv[1], sys.argv[2], sys.argv[3]
lines = [x for x in open(src) if x.strip()]
a = lines[0::2]
b = lines[1::2]
open(out_a, "w").writelines(a)
open(out_b, "w").writelines(b)
print("split %s: %d lines -> %d (A, even idx) + %d (B, odd idx)"
      % (src, len(lines), len(a), len(b)))
