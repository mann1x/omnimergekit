"""make_diag_maps.py — the two diagnostic maps between map A and the shipped ml_k8 (owner 2026-10-17: k8, k9, k12 all
loop/stall in Cerebriline, map A does not; all three share 5 swaps vs A).

k8 = A + core5 + ml3, where (per layer, every rescue paired with an eviction in the same layer):
  core5 = rescue 0:9 8:122 15:64 15:106 27:48   / evict 0:23 8:96 15:3 15:38 27:106   (shared by k8, k9, k12)
  ml3   = rescue 23:116 26:16 29:25             / evict 23:48 26:108 29:26            (k8's multilingual-only swaps)
a_core5 = A + core5 only ; a_ml3 = A + ml3 only. Set arithmetic on the two shipped maps, no generator involved.
Gates: every layer drops exactly 30; a_core5 and a_ml3 each differ from A by exactly their swap set; the
union of their swaps reproduces k8 exactly (so the two arms partition k8's change).
usage: make_diag_maps.py <A.json> <ml_k8.json> <out_dir>
"""
import json
import sys
from pathlib import Path

CORE5 = {"rescue": [(0, 9), (8, 122), (15, 64), (15, 106), (27, 48)],
         "evict": [(0, 23), (8, 96), (15, 3), (15, 38), (27, 106)]}
ML3 = {"rescue": [(23, 116), (26, 16), (29, 25)], "evict": [(23, 48), (26, 108), (29, 26)]}


def load(p):
    d = json.load(open(p))
    d = d.get("drop_map", d)
    return {int(k): set(v) for k, v in d.items() if str(k).isdigit()}


def apply(a, sw):
    m = {li: set(s) for li, s in a.items()}
    for li, e in sw["rescue"]:
        assert e in m[li], f"{li}:{e} is not dropped by A"
        m[li].discard(e)
    for li, e in sw["evict"]:
        assert e not in m[li], f"{li}:{e} is already dropped by A"
        m[li].add(e)
    return m


def main():
    a, k8, out = load(sys.argv[1]), load(sys.argv[2]), Path(sys.argv[3])
    maps = {"a_core5": apply(a, CORE5), "a_ml3": apply(a, ML3)}
    both = apply(apply(a, CORE5), ML3)
    assert both == k8, "core5 + ml3 does not reproduce ml_k8"
    for n, m in maps.items():
        assert all(len(m[li]) == 30 for li in range(30)), f"{n}: a layer does not drop 30"
        diff = sum(len(m[li] ^ a[li]) for li in range(30)) // 2
        print(f"{n}: {diff} swaps vs A")
        with open(out / f"{n}.json", "w") as f:
            json.dump({str(li): sorted(m[li]) for li in range(30)}, f, indent=1)
    print("DIAG_MAPS_OK (a_core5 + a_ml3 partition ml_k8's 8 swaps)")


if __name__ == "__main__":
    main()
