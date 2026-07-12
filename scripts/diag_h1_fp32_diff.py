"""H1: fp32 diff between .pre_eac_calibrate backup (= input to EAC) and current main shard (= what's on disk now)."""
import json, statistics
from pathlib import Path
from safetensors.torch import safe_open
import torch

A2_EAC = Path("/srv/ml/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-eac-it")
wmap = json.load(open(A2_EAC/"model.safetensors.index.json"))["weight_map"]

# Check shard 1 (where layer 0 lives)
shards_to_check = ["model-00001-of-00006.safetensors", "model-00003-of-00006.safetensors"]
for shard_name in shards_to_check:
    print(f"\n=== {shard_name} ===")
    current = A2_EAC / shard_name
    backup = A2_EAC / (shard_name + ".pre_eac_calibrate")
    if not backup.exists():
        print(f"  no .pre_eac_calibrate for this shard")
        continue
    print(f"  current SHA  : {open(current,'rb').read(8).hex()}..."[:40] + f"   size={current.stat().st_size:,}  inode={current.stat().st_ino}")
    print(f"  backup  SHA  : {open(backup,'rb').read(8).hex()}..."[:40] + f"   size={backup.stat().st_size:,}  inode={backup.stat().st_ino}")
    n_keys_diff, n_keys_total, max_abs_delta, max_rel_delta, total_l2 = 0, 0, 0.0, 0.0, 0.0
    delta_by_kind = {}
    with safe_open(str(current), framework="pt") as fc, safe_open(str(backup), framework="pt") as fb:
        keys = sorted(fc.keys())
        for k in keys:
            tc = fc.get_tensor(k).float()
            tb = fb.get_tensor(k).float()
            n_keys_total += 1
            if tc.shape != tb.shape:
                print(f"  !! shape mismatch {k}: cur={tc.shape} bak={tb.shape}"); continue
            d = (tc - tb).abs()
            l2 = float(d.norm().item())
            total_l2 += l2*l2
            if l2 > 0:
                n_keys_diff += 1
                mad = float(d.max().item())
                rel = mad / max(float(tb.abs().max().item()), 1e-9)
                if mad > max_abs_delta: max_abs_delta = mad
                if rel > max_rel_delta: max_rel_delta = rel
                kind = ".".join(k.split(".")[-3:])[:40]
                delta_by_kind.setdefault(kind, []).append((float(d.mean().item()), mad, rel))
    print(f"  keys: {n_keys_diff}/{n_keys_total} differ")
    print(f"  total L2(diff): {total_l2**0.5:.6f}")
    print(f"  max abs delta : {max_abs_delta:.6f}")
    print(f"  max rel delta : {max_rel_delta:.6e}")
    print(f"  tensors changed by kind (sample):")
    for kind, deltas in sorted(delta_by_kind.items())[:10]:
        mean_mad = statistics.mean(d[1] for d in deltas)
        print(f"    {kind:>30}  count={len(deltas):3d}  mean(max_delta)={mean_mad:.6e}")
