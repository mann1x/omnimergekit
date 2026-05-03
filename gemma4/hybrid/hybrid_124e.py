#!/usr/bin/env python3
"""
Hybrid 124e: Drop only the "noisy" experts from 128e to beat 128e.

Strategy: identify experts in E19 (128e\\109e, the 19 dropped) that are active on
109e's GAIN questions (Q8, Q71) — these "noisy" experts confused 128e, dropping
them gave 109e the gain. Keep everything else from 128e.

Method:
1. For each expert in E19, compute "noise score":
   - Active on Q8 (rank low) AND active on Q71 (rank low)
   - NOT active on Q32 (a regression where dropping hurts)
   noise_score = (rank_Q32) - (rank_Q8 + rank_Q71)/2
   (Higher = more noisy = drop me)
2. Drop top-K noisy per layer (K small, e.g., 4)
3. Result: 128 - K experts per layer (e.g., 124e)

Output: scripts/hybrid_124e_drop_map.json
"""

import json

with open('scripts/per_question_v4.json') as f:
    pq = json.load(f)
with open('google/gemma-4-A4B-109e/expert_drop_metadata.json') as f:
    e109 = json.load(f)

keep_109 = {int(k): set(v) for k, v in e109['per_layer_keep'].items()}
NUM_LAYERS = 30
NUM_EXPERTS = 128
ALL = set(range(NUM_EXPERTS))

# E19 = 128e \ 109e
E19 = {li: sorted(ALL - keep_109[li]) for li in range(NUM_LAYERS)}

def rank(layer_data):
    sorted_e = sorted(layer_data, key=lambda x: x['wnorm'], reverse=True)
    return {e['id']: r for r, e in enumerate(sorted_e)}

ranks = {int(d): {int(li): rank(le) for li, le in q.items()}
         for d, q in pq['questions'].items()}

# Compute noise score for each E19 expert
print("=== E19 noise analysis ===")
print("Score = rank_Q32 - (rank_Q8 + rank_Q71)/2")
print("Higher = more noisy (active on gains, not on regression)\n")

noise_scores = {}  # (li, eid) -> score
for li in range(NUM_LAYERS):
    noise_scores[li] = []
    for eid in E19[li]:
        r_q8 = ranks[8][li].get(eid, 128)
        r_q71 = ranks[71][li].get(eid, 128)
        r_q32 = ranks[32][li].get(eid, 128)
        # gain_activity: low rank = active on gains = "noisy"
        gain_activity = (r_q8 + r_q71) / 2
        critical_activity = r_q32
        # noise_score: positive if gain-active AND non-critical
        score = critical_activity - gain_activity
        noise_scores[li].append((eid, score, r_q8, r_q71, r_q32))
    noise_scores[li].sort(key=lambda x: -x[1])  # highest noise first

# Print top noisy per layer
print(f'{"Layer":>5}  {"Top noise picks (eid, score, r_Q8, r_Q71, r_Q32)":>50}')
for li in range(NUM_LAYERS):
    top = noise_scores[li][:3]
    print(f'  L{li:2d}  ' + ', '.join(f'E{e}({s:+d})' for e, s, _, _, _ in top))

# Build hybrid 124e: drop top-4 noisy per layer (128-4=124)
DROP_K = 4
print(f'\n=== Building hybrid {NUM_EXPERTS - DROP_K}e: drop top-{DROP_K} noisy per layer ===')

hybrid_drop = {}
for li in range(NUM_LAYERS):
    drop = [eid for eid, _, _, _, _ in noise_scores[li][:DROP_K]]
    hybrid_drop[li] = sorted(drop)
    print(f'  L{li:2d}: drop {hybrid_drop[li]}')

# Save
with open('scripts/hybrid_124e_drop_map.json', 'w') as f:
    json.dump({str(li): hybrid_drop[li] for li in range(NUM_LAYERS)}, f, indent=2)
print(f'\nSaved to scripts/hybrid_124e_drop_map.json')

avg_keep = (NUM_EXPERTS - DROP_K)
print(f'Per-layer keep: {avg_keep} (vs 128e=128, 109e=109)')
