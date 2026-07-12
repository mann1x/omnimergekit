#!/usr/bin/env python3
"""Outlier-vs-broad check on the dropped code/LCB specialists.

A high-wnorm "specialist" can be (a) a genuine broad contributor or (b) a narrow
outlier whose wnorm is driven by one/few neurons spiking on rare tokens. The
selection deliberately discounts (b). For the dropped top-16 experts in
generic_code and targeted_lcb_medium_55, classify each via the neuron-activation
PARTICIPATION RATIO  PR = (Σa)^2 / Σ(a^2)  (effective #contributing neurons;
PR/len in [0,1], low = narrow/spiky, high = broad) and token count tc.

A dropped specialist is BROAD (genuine loss) if its PR/len is >= the 25th
percentile of the KEPT specialists' PR/len in the same class; NARROW (benign
outlier drop) otherwise. Reports counts + the broad losses.
"""
import json
import statistics as st

MAP = "/mnt/sdc/ml/google/expert_neuron_v7_code.json"
MD = "/mnt/sdc/ml/sft_heal/dern11-soft-soft2-it/expert_drop_metadata.json"
TOPN = 16

cats = json.load(open(MAP))["categories"]
md = json.load(open(MD))
drop = {int(k): set(v) for k, v in md["per_layer_drop"].items()}
keepm = {int(k): set(v) for k, v in md["per_layer_keep"].items()}
layers = sorted(drop.keys())


def pr_frac(na):
    s = sum(na)
    s2 = sum(x * x for x in na)
    if s2 <= 0:
        return 0.0
    return (s * s / s2) / len(na)


def specialists(cat):
    """Return per-layer list of (eid, wnorm, tc, pr_frac) for the top-TOPN by wnorm."""
    out = {}
    for L in layers:
        arr = cats[cat][str(L)]
        order = sorted(arr, key=lambda e: e["wnorm"], reverse=True)[:TOPN]
        out[L] = [(e["id"], e["wnorm"], e.get("tc", 0), pr_frac(e["neuron_act"])) for e in order]
    return out


for cat in ("generic_code", "targeted_lcb_medium_55"):
    sp = specialists(cat)
    kept_pr, kept_tc, drop_rows = [], [], []
    for L in layers:
        for (eid, wn, tc, prf) in sp[L]:
            if eid in keepm[L]:
                kept_pr.append(prf)
                kept_tc.append(tc)
            elif eid in drop[L]:
                drop_rows.append((L, eid, wn, tc, prf))
    if not kept_pr:
        print(f"\n### {cat}: no kept specialists?!")
        continue
    kp = sorted(kept_pr)
    p25 = kp[len(kp) // 4]
    p50 = kp[len(kp) // 2]
    kept_tc_med = st.median(kept_tc) if kept_tc else 0
    narrow = [r for r in drop_rows if r[4] < p25]
    broad = [r for r in drop_rows if r[4] >= p25]
    print(f"\n### {cat}  (top-{TOPN} specialists/layer)")
    print(f"  KEPT specialists: n={len(kept_pr)}  PR/len p25={p25:.3f} p50={p50:.3f}  tc_med={kept_tc_med:.0f}")
    print(f"  DROPPED specialists: n={len(drop_rows)}  -> NARROW(outlier)={len(narrow)}  BROAD(loss)={len(broad)}")
    broad.sort(key=lambda r: -r[4])
    print(f"  broad (genuine-loss) dropped specialists [{len(broad)}], PR/len desc:")
    for (L, eid, wn, tc, prf) in broad[:20]:
        print(f"    L{L:02d} e{eid:03d}  wnorm={wn:8.3f}  tc={tc:5d}  PR/len={prf:.3f}")
