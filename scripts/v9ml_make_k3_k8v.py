#!/usr/bin/env python3
"""Build ml_k3 (A + only ml_k8's late-layer swaps L23/L26/L29) and ml_k8v (ml_k8 with L23 e48 kept,
evicting instead the L23 survivor whose MAX rank percentile over all 14 categories is lowest)."""
import copy
import json
import sys

sys.path.insert(0, "/mnt/sdc/harness")
from v9ml_ladder import load_map, pct  # noqa: E402

G = "/mnt/sdc/ml/google"
L = "/mnt/sdc/ml/v9ml/ladder"
A_P = "/mnt/sdc/ml/std12v5/candidates/A_lift120_plus_fk12.json"
P12 = "0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"
raw_k8 = json.load(open(f"{L}/ml_k8.json"))
A, K8 = load_map(A_P), load_map(f"{L}/ml_k8.json")
pins = {tuple(map(int, x.split(":"))) for x in (open(f"{G}/eog_v4_t106_lift_pins.txt").read().strip() + "," + P12).split(",") if x}

resc = {(ly, e) for ly in A for e in A[ly] - K8[ly]}
evic = {(ly, e) for ly in A for e in K8[ly] - A[ly]}
print("ml_k8 vs A rescued:", sorted(resc), "\n             evicted:", sorted(evic))
assert len(resc) == len(evic) == 8


def write(name, drop):
    out = copy.deepcopy(raw_k8)
    tgt = out.get("drop_map", out)
    for ly in range(30):
        assert len(drop[ly]) == len(A[ly]), (name, ly)
        tgt[str(ly)] = sorted(drop[ly])
    json.dump(out, open(f"{L}/{name}.json", "w"), indent=1)
    d = sum(len(A[ly] ^ drop[ly]) for ly in A) // 2
    print(f"wrote {L}/{name}.json  swaps vs A = {d}")


k3 = {ly: set(A[ly]) for ly in A}
for ly in (23, 26, 29):
    r = [e for (l2, e) in resc if l2 == ly]
    v = [e for (l2, e) in evic if l2 == ly]
    assert len(r) == len(v) == 1, (ly, r, v)
    k3[ly] = (k3[ly] - {r[0]}) | {v[0]}
    print(f"ml_k3 L{ly}: rescue e{r[0]}, evict e{v[0]}")
write("ml_k3", k3)

cats = json.load(open(f"{G}/expert_neuron_v9ml_code.json"))["categories"]
P = {c: pct(cats[c]) for c in cats}
ly = 23
kept = [e for e in range(128) if e not in K8[ly]]
cand = []
for e in kept:
    if (ly, e) in pins or (ly, e) in resc:
        continue
    sc = {c: P[c][ly][e] for c in P}
    cand.append((max(sc.values()), e, sc))
cand.sort()
print(f"\nL23 cheapest survivors of ml_k8 (by max rank over {len(P)} categories); e48 =",
      f"{max(P[c][ly][48] for c in P):.2f}")
for m, e, sc in cand[:5]:
    swe = max(v for c, v in sc.items() if c.startswith("targeted_swe"))
    print(f"   e{e:<3d} max={m:.2f} swe_max={swe:.2f} code={sc['generic_code']:.2f} ml={sc['generic_multilingual']:.2f} if={sc['generic_ifeval']:.2f}")
best = cand[0][1]
k8v = {l2: set(K8[l2]) for l2 in K8}
k8v[ly] = (k8v[ly] - {48}) | {best}
print(f"ml_k8v L23: keep e48, evict e{best} instead")
write("ml_k8v", k8v)
