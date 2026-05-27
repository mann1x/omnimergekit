"""ifeval_rum3 — the 3 IFEval RUMINATION prompts of the 21q screen.

The 3 IFEval dataset keys (143, 1300, 1477) the 62e v1-coder (fc15_25-p8) fails
by RUMINATION. Only the doc selection changes; the scorers are re-exported from
the stock lm-eval ifeval task so strict/loose prompt+instruction accuracy is
computed identically.
"""
from lm_eval.tasks.ifeval.utils import (  # noqa: F401  (re-exported for the yaml)
    process_results,
    agg_inst_level_acc,
)

_RUM_KEYS = {143, 1300, 1477}


def select_ifeval_rum3(dataset):
    out = dataset.filter(lambda x: x.get("key") in _RUM_KEYS)
    if len(out) != len(_RUM_KEYS):
        raise ValueError(
            f"ifeval_rum3: expected {len(_RUM_KEYS)} docs, got {len(out)} "
            f"(keys present={sorted(set(out['key']))}); the rumination keys "
            f"{sorted(_RUM_KEYS)} must all exist in google/IFEval train split"
        )
    return out
