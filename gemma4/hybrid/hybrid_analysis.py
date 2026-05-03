#!/usr/bin/env python3
"""
Hybrid expert map analysis.

Uses per-question expert activation data (scripts/per_question_v4.json) and
the existing 109e/96e-old keep maps to find which experts to keep/add for
maximum coverage of the curated 6 questions.

Key sets:
- B = base 96e-old (96 experts/layer)
- E13 = 109e \\ 96e-old (the 13 extras per layer)
- E19 = 128e \\ 109e (the 19 originally dropped per layer)

For each question Q in {17, 13, 27, 32, 8, 71}:
- Get per-expert wnorm from 128e on Q
- Identify top experts (by wnorm) in each layer
- Cross-reference with B/E13/E19

Goal: build a hybrid keep map that adds the right E13/E19 experts to recover
questions that 96e-old loses (Q8, Q32) without breaking Q27.
"""

import json

print("=== Hybrid Expert Analysis ===\n")

# Load data
with open('scripts/per_question_v4.json') as f:
    pq = json.load(f)

with open('google/gemma-4-A4B-109e/expert_drop_metadata.json') as f:
    e109 = json.load(f)
keep_109 = {int(k): set(v) for k, v in e109['per_layer_keep'].items()}

with open('google/gemma-4-A4B-96e-old/expert_drop_metadata.json') as f:
    e96 = json.load(f)
keep_96 = {int(k): set(v) for k, v in e96['per_layer_keep'].items()}

NUM_LAYERS = 30
NUM_EXPERTS = 128

# 6 test questions and their expected behavior
TEST_DOCS = {
    17: '128wrong/109wrong/96wrong',  # both wrong - hardest
    13: '128wrong/109right/96right',  # gain
    27: '128wrong/109wrong/96right',  # 96e-old uniquely right
    32: '128right/109wrong/96wrong',  # regression
    8: '128wrong/109right/96wrong',   # gain only 109e has
    71: '128wrong/109right/96right',  # gain
}

# Per question, per layer: rank of each expert by wnorm
def rank_experts(layer_data):
    """Return dict {expert_id: rank} where rank 0 = highest wnorm."""
    sorted_e = sorted(layer_data, key=lambda x: x['wnorm'], reverse=True)
    return {e['id']: rank for rank, e in enumerate(sorted_e)}

# Build per-question per-layer ranks
ranks = {}
for did_str, q_data in pq['questions'].items():
    did = int(did_str)
    ranks[did] = {}
    for li_str, layer_experts in q_data.items():
        ranks[did][int(li_str)] = rank_experts(layer_experts)

# === Analysis 1: For each layer, what's the rank of dropped experts on each question? ===
# Focus on E13 (109e \ 96e-old) and E19 (128e \ 109e)

# Build E13, E19 for each layer
E13 = {}  # 109e \ 96e-old (the 13 extras dropped from 109 to 96)
E19 = {}  # all_128 \ 109e (the original 19 dropped)
ALL = set(range(NUM_EXPERTS))
for li in range(NUM_LAYERS):
    E13[li] = sorted(keep_109[li] - keep_96[li])
    E19[li] = sorted(ALL - keep_109[li])
    assert len(E13[li]) == 13
    assert len(E19[li]) == 19

# === Analysis 2: For each "interesting" question, find top-K active experts in E13 and E19 ===
# Q8: helps 109e but not 96e → likely E13 contains Q8 helpers
# Q27: 96e-old uniquely right → E13 contains Q27 BLOCKERS (active when wrong)
# Q32: 128e right only → E19 contains Q32 critical experts

print("Per-layer top E13 experts (109e\\96e-old) by question wnorm rank:")
print("Lower rank = more active = more important for that question\n")

def expert_avg_rank(question, expert_set_per_layer):
    """For each layer, find avg rank of experts in the set on this question."""
    total_score = 0
    n = 0
    for li in range(NUM_LAYERS):
        for eid in expert_set_per_layer[li]:
            r = ranks[question][li].get(eid, 999)
            total_score += r
            n += 1
    return total_score / n if n > 0 else 0

print(f'{"Q":>4} {"avg rank E13":>14} {"avg rank E19":>14}')
print('-' * 40)
for did in [8, 27, 32]:
    e13_rank = expert_avg_rank(did, E13)
    e19_rank = expert_avg_rank(did, E19)
    print(f'  Q{did:>2} {e13_rank:>13.1f}  {e19_rank:>13.1f}')

# Lower rank = more important. So if Q8 has lower avg E13 rank than Q27, then E13 experts are MORE
# important for Q8 than Q27, suggesting they're helpers for Q8.

# === Analysis 3: For each E13 expert, score it on Q8 (good) vs Q27 (bad) ===
# An expert that's high-rank (low number) on Q8 and low-rank (high number) on Q27 is a "Q8 helper"
print("\n=== E13 experts: helpers for Q8 vs Q27 blockers ===")
print("Score = rank_on_Q27 - rank_on_Q8 (higher = better Q8 helper, less Q27 blocker)\n")

q8_helpers = []  # (layer, expert_id, score)
q27_blockers = []
for li in range(NUM_LAYERS):
    for eid in E13[li]:
        r_q8 = ranks[8][li].get(eid, 128)
        r_q27 = ranks[27][li].get(eid, 128)
        score = r_q27 - r_q8  # higher = good
        if score > 0:
            q8_helpers.append((li, eid, score, r_q8, r_q27))
        else:
            q27_blockers.append((li, eid, score, r_q8, r_q27))

# Sort by score
q8_helpers.sort(key=lambda x: -x[2])
q27_blockers.sort(key=lambda x: x[2])

print(f'Total E13 experts: {sum(len(E13[li]) for li in range(NUM_LAYERS))}')
print(f'  Q8-helper candidates (score > 0): {len(q8_helpers)}')
print(f'  Q27-blocker candidates (score <= 0): {len(q27_blockers)}')

print('\nTop 15 Q8 helpers (low Q8 rank = active on Q8, high Q27 rank = inactive on Q27):')
for li, eid, score, r8, r27 in q8_helpers[:15]:
    print(f'  L{li:2d} E{eid:3d}: r_Q8={r8:3d}, r_Q27={r27:3d}, score={score:+d}')

print('\nTop 10 Q27 blockers (low Q27 rank = active on Q27, suggesting they pushed wrong direction in 109e):')
for li, eid, score, r8, r27 in q27_blockers[:10]:
    print(f'  L{li:2d} E{eid:3d}: r_Q8={r8:3d}, r_Q27={r27:3d}, score={score:+d}')

# === Analysis 4: For E19, find Q32 critical experts ===
print("\n=== E19 experts: Q32 critical (128e gets it, 109e/96e-old don't) ===")
q32_critical = []
for li in range(NUM_LAYERS):
    for eid in E19[li]:
        r_q32 = ranks[32][li].get(eid, 128)
        q32_critical.append((li, eid, r_q32))
q32_critical.sort(key=lambda x: x[2])

print(f'Total E19 experts: {sum(len(E19[li]) for li in range(NUM_LAYERS))}')
print('\nTop 15 Q32 critical (lowest rank = most active on Q32):')
for li, eid, r in q32_critical[:15]:
    print(f'  L{li:2d} E{eid:3d}: r_Q32={r:3d}')

# === Analysis 5: Build hybrid keep map ===
# Start from 96e-old base (96/layer)
# For each layer, add the top Q8-helper E13 expert (if score > 0)
# Also add the top Q32-critical E19 expert (lowest rank)
# Result: 96 + 2 = 98 experts/layer

print("\n=== Proposed hybrid keep map: 96e-old + 1 Q8-helper E13 + 1 Q32-critical E19 ===")
print("Per layer: 98 experts (still under 109)\n")

hybrid_keep = {}
for li in range(NUM_LAYERS):
    base = set(keep_96[li])

    # Add top Q8-helper from E13 in this layer (always pick best to keep uniform count)
    layer_e13_helpers = [(eid, r27 - r8) for eid in E13[li]
                         for r8 in [ranks[8][li].get(eid, 128)]
                         for r27 in [ranks[27][li].get(eid, 128)]]
    layer_e13_helpers.sort(key=lambda x: -x[1])
    if layer_e13_helpers:
        e13_pick = layer_e13_helpers[0][0]
        base.add(e13_pick)
    else:
        e13_pick = None

    # Add top Q32-critical from E19 in this layer
    layer_e19_q32 = [(eid, ranks[32][li].get(eid, 128)) for eid in E19[li]]
    layer_e19_q32.sort(key=lambda x: x[1])
    if layer_e19_q32:
        e19_pick = layer_e19_q32[0][0]
        base.add(e19_pick)
    else:
        e19_pick = None

    hybrid_keep[li] = sorted(base)
    print(f'  L{li:2d}: +E13={e13_pick}, +E19={e19_pick}, total={len(base)}')

# Save hybrid drop map
hybrid_drop = {li: sorted(set(range(NUM_EXPERTS)) - set(hybrid_keep[li])) for li in range(NUM_LAYERS)}
with open('scripts/hybrid_drop_map.json', 'w') as f:
    json.dump({str(li): hybrid_drop[li] for li in range(NUM_LAYERS)}, f, indent=2)
print(f'\nSaved hybrid drop map to scripts/hybrid_drop_map.json')

avg = sum(len(v) for v in hybrid_keep.values()) / NUM_LAYERS
print(f'Average experts per layer: {avg:.1f}')
