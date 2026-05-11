"""Curated-subset filter for omnimergekit eval smoke tests.

Each function takes a `datasets.Dataset` and returns the subset selected
by a hardcoded index list. Indices are stratified into easy/medium/hard
tiers by a deterministic complexity heuristic (assertion count + canonical
solution length for code; topic for GPQA), then picked evenly across each
tier so the chosen problems span the difficulty range.

The lists are baked-in (not regenerated at runtime) so eval results are
bit-exactly reproducible across lm-eval versions and dataset re-shuffles.

Pick rationale (run once, frozen):
  HumanEval (164 → 20):  5 trivial / 10 medium / 5 complex
                         scored by (n_asserts, sol_lines, prompt_lines)
  MBPP full (500 → 40): 10 trivial / 20 medium / 10 complex
                         scored by (ast_node_count, code_lines)
  GPQA Diamond (198 → 10): 5 chemistry / 4 physics / 1 biology
                            evenly spaced within each topic to span
                            sub-domains (organic chem, quantum,
                            astrophysics, genetics, ...).

After filtering, the upstream `process_docs` (only used by GPQA, which
randomises the answer-choice order) is re-applied so the subset behaves
identically to the full task.
"""
from __future__ import annotations

# ── Curated indices ────────────────────────────────────────────
HUMANEVAL_SMOKE20 = [
    6, 34, 42, 43, 54,                       # 5 trivial
    55, 61, 63, 65, 84, 89, 91, 97, 106, 108, # 10 medium
    110, 120, 123, 145, 163,                 # 5 complex
]

MBPP_SMOKE40 = [
    34, 35, 42, 51, 52, 59, 64, 65, 70, 74,                 # 10 trivial
    77, 89, 113, 125, 127, 157, 170, 217, 224, 228,
    232, 247, 271, 279, 285, 308, 316, 335, 338, 373,       # 20 medium
    380, 382, 389, 399, 419, 437, 460, 463, 467, 473,       # 10 complex
]

GPQA_DIAMOND_SMOKE10 = [
    0,    # Physics — Physics (general)
    1,    # Chemistry — Organic Chemistry
    7,    # Biology — Genetics
    37,   # Chemistry — Organic Chemistry
    55,   # Physics — Quantum Mechanics
    80,   # Chemistry — Organic Chemistry
    112,  # Chemistry — Chemistry (general)
    114,  # Physics — Physics (general)
    142,  # Chemistry — Organic Chemistry
    168,  # Physics — Quantum Mechanics
]

# Extended 20-question subset: superset of SMOKE10 + 10 stratified picks to
# cover sub-domains the SMOKE10 missed (Astrophysics, HEP, Rel-Mech, E&M,
# Inorganic, Molecular Biology). Final domain mix: 11 Phys / 7 Chem / 2 Bio.
GPQA_DIAMOND_SMOKE20 = [
    0,    # Physics — Physics (general)
    1,    # Chemistry — Organic Chemistry
    2,    # Physics — Quantum Mechanics
    6,    # Physics — High-energy particle physics
    7,    # Biology — Genetics
    9,    # Physics — Astrophysics
    12,   # Chemistry — Organic Chemistry
    26,   # Biology — Molecular Biology
    31,   # Physics — High-energy particle physics
    35,   # Physics — Astrophysics
    37,   # Chemistry — Organic Chemistry
    40,   # Physics — Relativistic Mechanics
    50,   # Physics — Electromagnetism and Photonics
    55,   # Physics — Quantum Mechanics
    63,   # Chemistry — Inorganic Chemistry
    80,   # Chemistry — Organic Chemistry
    112,  # Chemistry — Chemistry (general)
    114,  # Physics — Physics (general)
    142,  # Chemistry — Organic Chemistry
    168,  # Physics — Quantum Mechanics
]


def _select(dataset, indices):
    """Return dataset.select(indices) clipped to the dataset length."""
    n = len(dataset)
    keep = sorted({i for i in indices if 0 <= i < n})
    if len(keep) != len(indices):
        missing = [i for i in indices if i >= n or i < 0]
        print(f"[subset_filter] WARN: dropped out-of-range indices {missing} "
              f"(dataset has {n} rows)", flush=True)
    return dataset.select(keep)


def humaneval_smoke20(dataset):
    return _select(dataset, HUMANEVAL_SMOKE20)


def mbpp_smoke40(dataset):
    return _select(dataset, MBPP_SMOKE40)


def gpqa_diamond_smoke10(dataset):
    """Filter first, then re-apply upstream choice randomisation so the
    subset's letter-mapping matches a normal GPQA run."""
    sub = _select(dataset, GPQA_DIAMOND_SMOKE10)
    # Lazy import: only needed for GPQA, may not be present in all envs.
    from lm_eval.tasks.gpqa.cot_zeroshot.utils import process_docs as upstream
    return upstream(sub)


def gpqa_diamond_smoke20(dataset):
    """20-question variant: superset of SMOKE10, broader sub-domain coverage."""
    sub = _select(dataset, GPQA_DIAMOND_SMOKE20)
    from lm_eval.tasks.gpqa.cot_zeroshot.utils import process_docs as upstream
    return upstream(sub)
