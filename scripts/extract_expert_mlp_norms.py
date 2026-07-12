#!/usr/bin/env python3
"""Per-expert MLP norms for A2 vs A2_EAC.
Gemma 4 26B-A4B fused-experts layout per layer:
  experts.down_proj      shape (62, hidden=2816, intermediate=704)
  experts.gate_up_proj   shape (62, 2*intermediate=1408, hidden=2816)  -- gate+up concatenated
  mlp.{down,gate,up}_proj  -- shared dense MLP (NOT per-expert)
  router.{scale, proj.weight, per_expert_scale}
  layer_scalar           -- single scalar
"""
import json, sys, os, statistics, time
from pathlib import Path
import torch
from safetensors.torch import safe_open

A2 = Path("/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it")
A2_EAC = Path("/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it")
DROP_MAP = Path("/srv/ml/repos/omnimergekit/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json")

def collect_expert_norms(p, label):
    idx = p / "model.safetensors.index.json"
    wmap = json.load(open(idx))["weight_map"]
    t0 = time.time()
    out = {}  # layer -> {down_norms: [62 floats], gate_up_norms: [62 floats], layer_scalar: float, shared_mlp_norms: dict}
    # group keys per shard for efficient open
    keys_of_interest = []
    for L in range(30):
        keys_of_interest += [
            f"model.language_model.layers.{L}.experts.down_proj",
            f"model.language_model.layers.{L}.experts.gate_up_proj",
            f"model.language_model.layers.{L}.layer_scalar",
            f"model.language_model.layers.{L}.mlp.down_proj.weight",
            f"model.language_model.layers.{L}.mlp.gate_proj.weight",
            f"model.language_model.layers.{L}.mlp.up_proj.weight",
        ]
    by_shard = {}
    for k in keys_of_interest:
        if k in wmap:
            by_shard.setdefault(wmap[k], []).append(k)
    for shard_name, keys in by_shard.items():
        with safe_open(str(p / shard_name), framework="pt") as f:
            for k in keys:
                parts = k.split(".")
                L = int(parts[3])
                out.setdefault(L, {"shared": {}})
                t = f.get_tensor(k)
                tf = t.float()
                tail = ".".join(parts[4:])
                if tail == "experts.down_proj":
                    # (62, 2816, 704) -> norm per expert
                    norms = tf.reshape(tf.shape[0], -1).norm(dim=1).cpu().tolist()
                    out[L]["expert_down_norms"] = norms
                elif tail == "experts.gate_up_proj":
                    norms = tf.reshape(tf.shape[0], -1).norm(dim=1).cpu().tolist()
                    out[L]["expert_gate_up_norms"] = norms
                elif tail == "layer_scalar":
                    out[L]["layer_scalar"] = float(tf.item())
                elif tail.startswith("mlp."):
                    out[L]["shared"][tail.replace("mlp.","")] = float(tf.norm().item())
    print(f"  [{label}] collected {len(out)} layers in {time.time()-t0:.1f}s", flush=True)
    return out

def main():
    print(f"# Per-expert MLP-norm snapshot — Gemma 4 26B-A4B 62e A2 vs A2_EAC")
    print(f"# Each layer: fused experts.down_proj (62,2816,704) + experts.gate_up_proj (62,1408,2816)")
    print()
    print("Loading A2 (pre-EAC) ...", flush=True)
    a2 = collect_expert_norms(A2, "A2")
    print("Loading A2_EAC (current 'now') ...", flush=True)
    eac = collect_expert_norms(A2_EAC, "A2_EAC")
    print()

    # Drop map: surviving expert IDs per layer
    dm_raw = json.load(open(DROP_MAP))
    # drop_map.json structure: {<layer>: [keep_ids], ...} OR with metadata wrapper
    # From earlier probe: dm_raw[0] was a list directly
    surviving = {}
    for k,v in dm_raw.items():
        try:
            L = int(k)
            if isinstance(v, list):
                surviving[L] = v
            elif isinstance(v, dict):
                for tag in ("keep_ids","kept","surviving","keep","keeps"):
                    if tag in v: surviving[L] = v[tag]; break
        except (ValueError, TypeError): pass

    print("LAYER-LEVEL SUMMARY: did EAC change anything?")
    print(f"{'L':>3} | {'A2 down-mean':>13} {'EAC down-mean':>14} {'Δ down %':>10} | {'A2 gateUp-mean':>15} {'EAC gateUp-mean':>16} {'Δ gateUp %':>11} | {'A2 layer_sc':>12} {'EAC layer_sc':>12} | {'shared mlp Δ':>13}")
    print("-" * 145)
    n_layers_changed = 0
    total_expert_changes = 0
    for L in sorted(a2.keys()):
        a, e = a2[L], eac[L]
        a_dn = statistics.mean(a["expert_down_norms"])
        e_dn = statistics.mean(e["expert_down_norms"])
        a_gu = statistics.mean(a["expert_gate_up_norms"])
        e_gu = statistics.mean(e["expert_gate_up_norms"])
        d_dn_pct = 100.0 * (e_dn - a_dn) / a_dn
        d_gu_pct = 100.0 * (e_gu - a_gu) / a_gu

        # Per-expert diff: count experts where down norm changed by >0.5%
        diffs = [abs((eN - aN) / aN) * 100 for aN, eN in zip(a["expert_down_norms"], e["expert_down_norms"])]
        n_changed = sum(1 for d in diffs if d > 0.5)
        total_expert_changes += n_changed
        if n_changed > 0: n_layers_changed += 1

        # Shared MLP diff (sum of relative)
        shared_diffs = []
        for k in ("down_proj","gate_proj","up_proj"):
            av = a["shared"].get(k); ev = e["shared"].get(k)
            if av and ev: shared_diffs.append(abs(ev-av)/av * 100)
        shared_max = max(shared_diffs) if shared_diffs else 0.0

        print(f"{L:>3} | {a_dn:>13.4f} {e_dn:>14.4f} {d_dn_pct:>+9.3f}% | {a_gu:>15.4f} {e_gu:>16.4f} {d_gu_pct:>+10.3f}% | {a['layer_scalar']:>12.4f} {e['layer_scalar']:>12.4f} | {shared_max:>12.4f}%")

    print()
    print(f"=> Layers where EAC changed any expert >0.5%: {n_layers_changed}/30")
    print(f"=> Total expert-layer cells changed >0.5%: {total_expert_changes}/{30*62}")
    print()

    # Per-expert table for one representative layer (find layer with most change)
    diffs_by_L = []
    for L in sorted(a2.keys()):
        d = sum(abs((eN-aN)/aN)*100 for aN,eN in zip(a2[L]["expert_down_norms"], eac[L]["expert_down_norms"]))
        diffs_by_L.append((d, L))
    diffs_by_L.sort(reverse=True)
    top_L = diffs_by_L[0][1] if diffs_by_L else 0
    print(f"PER-EXPERT TABLE — most-changed layer = L{top_L} (sum-abs-diff={diffs_by_L[0][0]:.2f}%)")
    print(f"{'pos':>3} {'phys_id':>7} | {'A2 down':>10} {'EAC down':>10} {'Δ down %':>10} | {'A2 gateUp':>12} {'EAC gateUp':>12} {'Δ gU %':>9}")
    print("-" * 100)
    a, e = a2[top_L], eac[top_L]
    s = surviving.get(top_L, list(range(62)))
    for pos in range(62):
        ad, ed = a["expert_down_norms"][pos], e["expert_down_norms"][pos]
        ag, eg = a["expert_gate_up_norms"][pos], e["expert_gate_up_norms"][pos]
        ddn = (ed-ad)/ad*100
        dgu = (eg-ag)/ag*100
        phys = s[pos] if pos < len(s) else "?"
        print(f"{pos:>3} {str(phys):>7} | {ad:>10.4f} {ed:>10.4f} {ddn:>+9.3f}% | {ag:>12.4f} {eg:>12.4f} {dgu:>+8.3f}%")

    # save bundle
    out_path = "/srv/ml/logs/expert_mlp_norms_snapshot_20260528.json"
    json.dump({
        "drop_map": str(DROP_MAP),
        "surviving_expert_ids_by_layer": {str(k):v for k,v in surviving.items()},
        "a2_pre_eac": {str(k):v for k,v in a2.items()},
        "a2_eac_current": {str(k):v for k,v in eac.items()},
    }, open(out_path, "w"), indent=2)
    print()
    print(f"=> Full bundle: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
