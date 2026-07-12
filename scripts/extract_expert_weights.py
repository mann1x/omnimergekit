#!/usr/bin/env python3
"""Extract per-expert weights from A2 (pre-EAC), A2_EAC (current "now"), and KD ckpt.
Router tensors per Gemma 4 26B-A4B:
  router.scale            shape (hidden_size=2816,)  - per-hidden-dim input scaling
  router.per_expert_scale shape (n_surviving=62,)    - per-expert mixing scale
  router.proj.weight      shape (62, 2816)           - routing projection
"""
import json, sys, os, statistics
from pathlib import Path
import torch
from safetensors.torch import safe_open

A2 = Path("/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it")
A2_EAC = Path("/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it")
KD_CKPT = Path("/srv/ml/logs/router_kd_iterate_A2_20260528_204645/ckpt/router_step000100.pt")
DROP_MAP = Path("/srv/ml/repos/omnimergekit/scripts/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json")

def _store(out, i_layer, tail, t):
    out.setdefault(i_layer, {})
    tf = t.float()
    if tail == "scale":   # (2816,)
        v = tf.cpu().tolist()
        out[i_layer]["scale_vec_mean"] = float(tf.mean().item())
        out[i_layer]["scale_vec_std"]  = float(tf.std().item())
        out[i_layer]["scale_vec_min"]  = float(tf.min().item())
        out[i_layer]["scale_vec_max"]  = float(tf.max().item())
        out[i_layer]["scale_vec_norm"] = float(tf.norm().item())
        out[i_layer]["scale_vec"]      = v
    elif tail == "per_expert_scale":  # (62,)
        out[i_layer]["per_expert_scale"] = tf.cpu().tolist()
    elif tail == "proj.weight":  # (62, 2816)
        out[i_layer]["proj_weight_shape"] = list(t.shape)
        out[i_layer]["proj_weight_norm"] = float(tf.norm().item())
        if tf.dim() == 2:
            out[i_layer]["proj_weight_per_expert_norm"] = tf.norm(dim=1).cpu().tolist()

def collect_from_safetensors(p):
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        print(f"  WARNING: no index.json in {p.name}; rebuilding")
        wmap = {}
        for s in sorted(p.glob("model-*.safetensors")):
            if str(s).endswith(".pre_eac_calibrate") or str(s).endswith(".pre_per_expert_rescale"):
                continue
            with safe_open(str(s), framework="pt") as f:
                for k in f.keys(): wmap[k] = s.name
    else:
        wmap = json.load(open(idx))["weight_map"]
    out = {}
    router_keys = sorted([k for k in wmap if "router" in k.lower()])
    by_shard = {}
    for k in router_keys: by_shard.setdefault(wmap[k], []).append(k)
    for shard_name, keys in by_shard.items():
        with safe_open(str(p / shard_name), framework="pt") as f:
            for k in keys:
                parts = k.split(".")
                try: i_layer = int(parts[3])
                except (ValueError, IndexError): continue
                tail = ".".join(parts[5:])
                _store(out, i_layer, tail, f.get_tensor(k))
    return out

def collect_from_kd_ckpt(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck["router"]
    out = {}
    for k, t in sd.items():
        parts = k.split(".")
        try: i_layer = int(parts[3])
        except (ValueError, IndexError): continue
        tail = ".".join(parts[5:])
        _store(out, i_layer, tail, t)
    return out, ck.get("step"), ck.get("recipe", {})

def main():
    dm_raw = json.load(open(DROP_MAP))
    surviving = {}
    if isinstance(dm_raw, dict):
        for k,v in dm_raw.items():
            if isinstance(v, dict):
                for tag in ("keep_ids","kept","surviving","keep","keeps"):
                    if tag in v:
                        try: surviving[int(k)] = v[tag]
                        except ValueError: pass
                        break
    print(f"# Expert weights snapshot — A2 publish candidate")
    print(f"# Drop map: {DROP_MAP.name} (layers with surviving expert ids: {len(surviving)})")
    print()
    print("=== A2 (pre-EAC) ===", flush=True)
    a2 = collect_from_safetensors(A2)
    print(f"  layers: {len(a2)}")
    print("=== A2_EAC (current model) ===", flush=True)
    eac = collect_from_safetensors(A2_EAC)
    print(f"  layers: {len(eac)}")
    print("=== KD ckpt (rejected) ===", flush=True)
    kd, step, recipe = collect_from_kd_ckpt(KD_CKPT)
    print(f"  layers: {len(kd)}, step={step}")
    print()

    def m(d,key="per_expert_scale"):
        v = d.get(key)
        if not v: return "?"
        return f"{statistics.mean(v):.4f}"
    def std(d,key="per_expert_scale"):
        v = d.get(key)
        if not v or len(v)<2: return "?"
        return f"{statistics.stdev(v):.4f}"
    def mn(d,key="per_expert_scale"):
        v = d.get(key)
        if not v: return "?"
        return f"{min(v):.4f}"
    def mx(d,key="per_expert_scale"):
        v = d.get(key)
        if not v: return "?"
        return f"{max(v):.4f}"
    def fv(d,key,fmt="7.3f"):
        v = d.get(key)
        if v is None: return "  ?  "
        return f"{v:{fmt}}"

    print("PER-EXPERT-SCALE (the 62-dim vector that amplifies/dampens each surviving expert)")
    print(f"{'layer':>5} | {'PRE  mean':>10} {'PRE  std':>9} {'PRE  min':>9} {'PRE  max':>9} | {'EAC  mean':>10} {'EAC  std':>9} {'EAC  min':>9} {'EAC  max':>9} | {'KD   mean':>10} {'KD   std':>9} {'KD   min':>9} {'KD   max':>9}")
    print("-" * 165)
    for L in sorted(a2.keys()):
        a, e, k = a2[L], eac.get(L, {}), kd.get(L, {})
        print(f"{L:>5} | {m(a):>10} {std(a):>9} {mn(a):>9} {mx(a):>9} | {m(e):>10} {std(e):>9} {mn(e):>9} {mx(e):>9} | {m(k):>10} {std(k):>9} {mn(k):>9} {mx(k):>9}")

    print()
    print("ROUTER PROJECTION (proj.weight) NORM + SCALE-VEC NORM per layer")
    print(f"{'layer':>5} | {'PRE proj-W':>11} {'EAC proj-W':>11} {'KD proj-W':>11} | {'PRE scaleV':>11} {'EAC scaleV':>11} {'KD scaleV':>11}")
    print("-" * 90)
    for L in sorted(a2.keys()):
        a, e, k = a2[L], eac.get(L, {}), kd.get(L, {})
        print(f"{L:>5} | {fv(a,'proj_weight_norm'):>11} {fv(e,'proj_weight_norm'):>11} {fv(k,'proj_weight_norm'):>11} | {fv(a,'scale_vec_norm'):>11} {fv(e,'scale_vec_norm'):>11} {fv(k,'scale_vec_norm'):>11}")

    # Surviving expert IDs summary
    print()
    print("SURVIVING EXPERT IDs (which physical experts the 62 positions map to)")
    if surviving:
        sample_layers = sorted(surviving.keys())[:5] + ["..."] + sorted(surviving.keys())[-5:] if len(surviving) > 10 else sorted(surviving.keys())
        for L in sorted(surviving.keys()):
            ids = surviving[L]
            if isinstance(ids, list):
                print(f"  L{L:>2}: count={len(ids)} ids={ids[:10]}{'...' if len(ids)>10 else ''}")
    else:
        print("  (drop_map keys not parsed — see raw JSON below)")
        for k in list(dm_raw.keys())[:5]: print(f"   {k}: {type(dm_raw[k]).__name__}, sample: {str(dm_raw[k])[:150]}")

    bundle = {
        "drop_map_path": str(DROP_MAP),
        "surviving_expert_ids_by_layer": {str(k): v for k,v in surviving.items()},
        "a2_pre_eac":     {str(k): v for k,v in a2.items()},
        "a2_eac_current": {str(k): v for k,v in eac.items()},
        "kd_ckpt_rejected": {str(k): v for k,v in kd.items()},
        "kd_meta": {"step": step, "recipe": {k:str(v)[:200] for k,v in recipe.items()}}
    }
    out_path = "/srv/ml/logs/expert_weights_snapshot_20260528.json"
    json.dump(bundle, open(out_path, "w"), indent=2)
    print()
    print(f"=> Full bundle JSON: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
