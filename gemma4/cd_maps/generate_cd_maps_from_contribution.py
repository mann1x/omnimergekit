#!/usr/bin/env python3
"""Generate CD tensor-type maps from expert_contribution_pod.py output.

Input: JSON with per-topic per-layer `total_moe_norm` — the actual contribution
of each MoE layer to the residual stream, measured by replaying prompts through
the model with expert hooks.

Layer importance = mean(total_moe_norm) across topics.
Ranking: top 1 → highest tier, next 7 → target tier, remaining 22 → lowest tier.

CD tier table (matches April 5 convention):
  CD-Q6_K:    top=Q8_0,  mid=Q6_K,  low=Q5_K
  CD-Q5_K_M:  top=Q6_K,  mid=Q5_K,  low=Q4_K
  CD-Q4_K_M:  top=Q5_K,  mid=Q4_K,  low=Q3_K
  CD-Q3_K_L:  top=Q4_K,  mid=Q3_K,  low=IQ3_S
  CD-Q2_K:    top=Q3_K,  mid=Q2_K,  low=IQ2_S

token_embd and output always at Q8_0.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

CD_TIERS = {
    "CD-Q6_K":   ("Q8_0", "Q6_K", "Q5_K"),
    "CD-Q5_K_M": ("Q6_K", "Q5_K", "Q4_K"),
    "CD-Q4_K_M": ("Q5_K", "Q4_K", "Q3_K"),
    "CD-Q3_K_L": ("Q4_K", "Q3_K", "IQ3_S"),
    "CD-Q2_K":   ("Q3_K", "Q2_K", "IQ2_S"),
}

TOP_N, MID_N = 1, 7  # low gets the rest (30 - 8 = 22 for Gemma 4 26B-A4B)

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


def load_layer_importance(contrib_path: Path) -> tuple[dict[int, float], int]:
    """Return (layer_idx -> mean importance, num_layers)."""
    data = json.loads(contrib_path.read_text())

    # Collect per-topic per-layer total_moe_norm
    scores_by_layer = defaultdict(list)
    max_layer = -1
    for topic, layers in data.items():
        for li_str, d in layers.items():
            li = int(li_str)
            score = d.get("total_moe_norm", 0.0)
            scores_by_layer[li].append(score)
            if li > max_layer:
                max_layer = li

    num_layers = max_layer + 1
    layer_importance = {li: sum(vals) / len(vals) if vals else 0.0
                        for li, vals in scores_by_layer.items()}
    return layer_importance, num_layers


def rank_and_tier(per_layer: dict[int, float], num_layers: int):
    ranking = sorted(range(num_layers), key=lambda li: per_layer.get(li, 0.0), reverse=True)
    tiers = {}
    for rank, li in enumerate(ranking):
        if rank < TOP_N:
            tiers[li] = "top"
        elif rank < TOP_N + MID_N:
            tiers[li] = "mid"
        else:
            tiers[li] = "low"
    return ranking, tiers


def write_tensor_types(cd_name, tiers, num_layers, out_dir):
    top, mid, low = CD_TIERS[cd_name]
    tier_quant = {"top": top, "mid": mid, "low": low}
    out_path = out_dir / f"tensor_types_{cd_name}.txt"
    lines = []
    for li in range(num_layers):
        q = tier_quant[tiers[li]]
        for role in BLOCK_TENSOR_ROLES:
            lines.append(f"blk.{li}.{role}={q}")
        lines.append(f"blk.{li}.ffn_down_exps.scale={q}")
        lines.append(f"blk.{li}.ffn_gate_inp.scale={q}")
    lines.append("token_embd.weight=q8_0")
    lines.append("output.weight=q8_0")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrib", required=True, help="expert_contributions.json from expert_contribution_pod.py")
    ap.add_argument("--out-dir", default=".", help="where to write tensor_types_*.txt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    contrib_path = Path(args.contrib)
    if not contrib_path.exists():
        print(f"ERROR: {contrib_path} not found", file=sys.stderr)
        sys.exit(1)

    per_layer, num_layers = load_layer_importance(contrib_path)
    ranking, tiers = rank_and_tier(per_layer, num_layers)

    max_score = max(per_layer.values()) if per_layer else 1.0
    print(f"=== Per-layer importance (num_layers={num_layers}) ===")
    print(f"{'rank':>4}  {'layer':>5}  {'tier':>4}  {'score':>15}  {'norm%':>6}")
    for rank, li in enumerate(ranking):
        score = per_layer.get(li, 0.0)
        pct = 100 * score / max_score if max_score else 0
        print(f"  {rank+1:>3}  L{li:<4}  {tiers[li]:>4}  {score:>15.4e}  {pct:>5.1f}%")

    top_layers = [li for li, t in tiers.items() if t == "top"]
    mid_layers = sorted([li for li, t in tiers.items() if t == "mid"])
    low_layers = sorted([li for li, t in tiers.items() if t == "low"])
    print()
    print("=== Tier summary ===")
    print(f"  top (1): L{top_layers}")
    print(f"  mid (7): L{mid_layers}")
    print(f"  low ({num_layers-TOP_N-MID_N}): L{low_layers}")

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Writing tensor_types files to {out_dir} ===")
    for cd_name in CD_TIERS:
        p = write_tensor_types(cd_name, tiers, num_layers, out_dir)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
