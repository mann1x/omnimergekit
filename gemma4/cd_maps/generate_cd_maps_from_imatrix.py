#!/usr/bin/env python3
"""Generate CD tensor-type maps from an imatrix.dat file.

The imatrix contains per-tensor activation statistics (in_sum2/counts) computed
during llama-imatrix calibration. This is the cleanest signal for layer importance
because it:
  1) comes from the target model itself (not a different teacher)
  2) uses the exact calibration data used for quantization
  3) uses llama.cpp's fp32 reduction (no bf16 NaN bug)

Layer importance = sum over all tensors in the layer of (in_sum2/counts).
Ranking: top 1 → highest tier, next 7 → target tier, remaining 22 → lowest tier.

CD tier table:
  CD-Q6_K:    top=Q8_0,  mid=Q6_K,  low=Q5_K
  CD-Q5_K_M:  top=Q6_K,  mid=Q5_K,  low=Q4_K
  CD-Q4_K_M:  top=Q5_K,  mid=Q4_K,  low=Q3_K
  CD-Q3_K_M:  top=Q4_K,  mid=Q3_K,  low=IQ3_S
  CD-Q2_K:    top=Q3_K,  mid=Q2_K,  low=IQ2_S

Output: one tensor_types_CD-<LEVEL>.txt per CD level, written to --out-dir.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Tier definitions — (top_1, mid_7, low_22)
CD_TIERS = {
    "CD-Q6_K":   ("Q8_0", "Q6_K", "Q5_K"),
    "CD-Q5_K_M": ("Q6_K", "Q5_K", "Q4_K"),
    "CD-Q4_K_M": ("Q5_K", "Q4_K", "Q3_K"),
    "CD-Q3_K_M": ("Q4_K", "Q3_K", "IQ3_S"),
    "CD-Q2_K":   ("Q3_K", "Q2_K", "IQ2_S"),
}

# How many layers in each tier (fixed for Gemma 4 26B-A4B: 30 layers)
TOP_N, MID_N = 1, 7  # LOW_N = total - TOP_N - MID_N

# Per-layer tensor roles that get quantized (all block-level tensors)
BLOCK_TENSOR_ROLES = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_up_exps.weight",
    "ffn_down_exps.weight",
]


def load_imatrix_importance(imatrix_path: Path) -> tuple[dict[int, float], int]:
    """Load imatrix.dat and return (per-layer importance dict, num_layers)."""
    sys.path.insert(0, "/opt/llama.cpp/gguf-py")
    from gguf import GGUFReader

    reader = GGUFReader(str(imatrix_path))

    # in_sum2 is shape [N], counts is shape [1]. Importance = mean in_sum2 / counts
    in_sum2 = {}
    counts = {}
    for t in reader.tensors:
        name = t.name
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            # Mean of the squared activations (proxy for L2 magnitude)
            data = t.data
            in_sum2[base] = float(data.sum())
        elif name.endswith(".counts"):
            base = name[: -len(".counts")]
            counts[base] = float(t.data[0])

    # Per-layer aggregate
    layer_re = re.compile(r"^blk\.(\d+)\.")
    per_layer = defaultdict(float)
    max_layer = -1
    for base, s2 in in_sum2.items():
        c = counts.get(base, 1.0)
        if c <= 0:
            continue
        # Normalize by counts to get mean squared activation
        importance = s2 / c
        m = layer_re.match(base)
        if m:
            li = int(m.group(1))
            per_layer[li] += importance
            max_layer = max(max_layer, li)

    num_layers = max_layer + 1
    return dict(per_layer), num_layers


def rank_layers(per_layer: dict[int, float], num_layers: int) -> list[int]:
    """Return layers sorted by importance descending."""
    scores = [(li, per_layer.get(li, 0.0)) for li in range(num_layers)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [li for li, _ in scores]


def assign_tiers(ranking: list[int]) -> dict[int, str]:
    """Return {layer_idx: tier_name} where tier ∈ {'top', 'mid', 'low'}."""
    out = {}
    for rank, li in enumerate(ranking):
        if rank < TOP_N:
            out[li] = "top"
        elif rank < TOP_N + MID_N:
            out[li] = "mid"
        else:
            out[li] = "low"
    return out


def write_tensor_types_file(
    cd_name: str, tiers: dict[int, str], num_layers: int, out_dir: Path
):
    top, mid, low = CD_TIERS[cd_name]
    tier_quant = {"top": top, "mid": mid, "low": low}
    out_path = out_dir / f"tensor_types_{cd_name}.txt"
    lines = []
    # Block tensors per layer
    for li in range(num_layers):
        q = tier_quant[tiers[li]]
        for role in BLOCK_TENSOR_ROLES:
            lines.append(f"blk.{li}.{role}={q}")
        # scale tensors mirror their parent
        lines.append(f"blk.{li}.ffn_down_exps.scale={q}")
        lines.append(f"blk.{li}.ffn_gate_inp.scale={q}")
    # Global: token_embd + output always at Q8_0 (follows existing convention)
    lines.append("token_embd.weight=q8_0")
    lines.append("output.weight=q8_0")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", required=True, help="Path to imatrix.dat")
    ap.add_argument("--out-dir", default=".", help="Output directory for tensor_types_*.txt")
    ap.add_argument("--dry-run", action="store_true", help="Only print ranking, no files")
    args = ap.parse_args()

    imatrix_path = Path(args.imatrix)
    if not imatrix_path.exists():
        print(f"ERROR: {imatrix_path} not found", file=sys.stderr)
        sys.exit(1)

    per_layer, num_layers = load_imatrix_importance(imatrix_path)
    if num_layers == 0:
        print("ERROR: no block-level tensors found in imatrix", file=sys.stderr)
        sys.exit(1)

    ranking = rank_layers(per_layer, num_layers)
    tiers = assign_tiers(ranking)

    # Normalize scores for display
    max_score = max(per_layer.values()) if per_layer else 1.0
    print(f"=== Per-layer importance from imatrix (num_layers={num_layers}) ===")
    print(f"{'rank':>4}  {'layer':>5}  {'tier':>4}  {'score':>15}  {'norm':>6}")
    for rank, li in enumerate(ranking):
        score = per_layer.get(li, 0.0)
        pct = 100 * score / max_score
        print(f"  {rank+1:>3}  L{li:<4}  {tiers[li]:>4}  {score:>15.4e}  {pct:>5.1f}%")

    print()
    print("=== Tier summary ===")
    top_layers = [li for li, t in tiers.items() if t == "top"]
    mid_layers = sorted([li for li, t in tiers.items() if t == "mid"])
    low_layers = sorted([li for li, t in tiers.items() if t == "low"])
    print(f"  top (1): L{top_layers}")
    print(f"  mid (7): L{mid_layers}")
    print(f"  low ({num_layers-TOP_N-MID_N}): L{low_layers}")

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Writing tensor_types files to {out_dir} ===")
    for cd_name in CD_TIERS:
        path = write_tensor_types_file(cd_name, tiers, num_layers, out_dir)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
