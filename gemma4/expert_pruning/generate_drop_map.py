#!/usr/bin/env python3
"""
Generate expert drop maps from teacher-force analysis data.

Uses per_question_teacher_force.json (produced by teacher_force_analysis.py) to
rank experts by importance and generate drop maps for expert_drop.py.

The protect_top parameter prevents the N most important experts per layer from
being dropped, even if they'd otherwise be candidates. This is the "p16" variant
(protect_top=16) used in production maps.

Usage:
  # 120e with protect_top=16 (production)
  python generate_drop_map.py --target 120 --protect-top 16 --output teacher_force_120e_p16_drop_map.json

  # 109e with protect_top=16
  python generate_drop_map.py --target 109 --protect-top 16 --output teacher_force_109e_p16_drop_map.json

  # 98e with protect_top=16
  python generate_drop_map.py --target 98 --protect-top 16 --output teacher_force_98e_p16_drop_map.json

  # Plain map without protection
  python generate_drop_map.py --target 98 --protect-top 0 --output teacher_force_98e_drop_map.json

Input:
  scripts/per_question_teacher_force.json — from teacher_force_analysis.py
  Each question → 30 layers → 128 experts with:
    - wnorm: weighted activation norm (importance signal)
    - rnorm: router norm
    - tc: token count (how often the expert was selected)

Scoring:
  score[layer][expert] = sum over all questions of (wnorm * alpha + tc)
  alpha=2.0 weights activation magnitude over selection frequency.

Protection:
  The top protect_top experts by score in each layer are never dropped.
  Drops are chosen from the remaining experts, lowest score first.
"""

import argparse
import json
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Generate expert drop map from teacher-force data")
    ap.add_argument("--data", default="scripts/per_question_teacher_force.json",
                    help="Path to teacher-force analysis JSON")
    ap.add_argument("--target", type=int, required=True,
                    help="Target number of experts per layer (e.g., 98, 109, 120)")
    ap.add_argument("--protect-top", type=int, default=16,
                    help="Number of top experts to protect from dropping (default: 16)")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="Weight for wnorm vs tc in scoring (default: 2.0)")
    ap.add_argument("--output", type=str, required=True,
                    help="Output JSON file path")
    args = ap.parse_args()

    num_experts = 128
    num_layers = 30
    drop_count = num_experts - args.target

    if drop_count <= 0:
        print(f"ERROR: target {args.target} >= {num_experts}, nothing to drop")
        sys.exit(1)
    if args.protect_top >= num_experts - drop_count:
        print(f"ERROR: protect_top {args.protect_top} leaves fewer than {drop_count} candidates to drop")
        sys.exit(1)

    print(f"Loading teacher-force data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)

    questions = data['questions']
    print(f"  {len(questions)} questions, {num_layers} layers, {num_experts} experts")
    print(f"  Target: {args.target}e (drop {drop_count}/layer)")
    print(f"  protect_top={args.protect_top}, alpha={args.alpha}")

    # Aggregate expert importance across all questions
    layer_scores = {}
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        expert_scores = np.zeros(num_experts)

        for q_key, q_data in questions.items():
            if layer_key not in q_data:
                continue
            for expert_data in q_data[layer_key]:
                eid = expert_data['id']
                expert_scores[eid] += expert_data['wnorm'] * args.alpha + expert_data['tc']

        layer_scores[layer_idx] = expert_scores

    # Generate drop map
    drop_map = {}
    for layer_idx in range(num_layers):
        scores = layer_scores[layer_idx]

        # Protect top-N experts
        if args.protect_top > 0:
            protected = set(np.argsort(scores)[-args.protect_top:].tolist())
        else:
            protected = set()

        # From remaining, drop the lowest-scoring
        remaining = [(i, scores[i]) for i in range(num_experts) if i not in protected]
        remaining.sort(key=lambda x: x[1])
        to_drop = sorted([r[0] for r in remaining[:drop_count]])

        drop_map[str(layer_idx)] = to_drop

        if layer_idx < 3 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx:2d}: drop {len(to_drop)}, "
                  f"min_score={remaining[0][1]:.1f}, "
                  f"max_dropped={remaining[drop_count-1][1]:.1f}, "
                  f"min_kept={remaining[drop_count][1]:.1f}")

    # Verify
    for lk, drops in drop_map.items():
        assert len(drops) == drop_count, f"Layer {lk}: expected {drop_count}, got {len(drops)}"

    with open(args.output, 'w') as f:
        json.dump(drop_map, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
