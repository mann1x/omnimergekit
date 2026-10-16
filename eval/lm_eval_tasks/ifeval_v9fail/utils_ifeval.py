"""ifeval_v9fail10 — the 10 IFEval prompts gemma-4-20B-v9-agentic (map A, Q6_K) fails at greedy.

Produced from the canonical ifeval_100 cell (pod 55947563, 2026-10-15, llama.cpp b9700,
score 0.90). ifeval_100 runs with --limit 100, i.e. IFEval rows 0-99 (omk_eval does not
honour selection.indices), so every key below lies in rows 0-99. A calibration corpus for
a variant tested on this slice must draw ONLY from rows 100-540, or the test is contaminated.

Failure modes in the parent run:
  repetition loops      1237 (Empanadas), 13 (JSON, Prospect Park), 1154 (Punjabi)
  low-resource language 1108 (Kannada), 1154 (Punjabi), 1259
  false refusal         1000 (Raymond III "does not exist")
  format constraints    1040, 1174, 1300, 1377

Only the doc SELECTION changes. process_results / agg_inst_level_acc are re-exported from
the stock lm-eval ifeval task, so the per-doc verdicts are computed by the same code as the
parent run. Selection is by dataset "key", never by row index; the count assert makes a
short subset fatal.

The slice was SELECTED on failure, so a re-run of the same model regresses toward the mean
under batching noise. Always run map A itself on this task as the baseline arm, on the same
host/build/parallelism as the variant; never compare a variant against the parent's 0/10.
"""
from lm_eval.tasks.ifeval.utils import (  # noqa: F401  (re-exported for the yaml)
    agg_inst_level_acc,
    process_results,
)

_V9FAIL10_KEYS = {13, 1000, 1040, 1108, 1154, 1174, 1237, 1259, 1300, 1377}


def select_ifeval_v9fail10(dataset):
    out = dataset.filter(lambda x: x.get("key") in _V9FAIL10_KEYS)
    if len(out) != len(_V9FAIL10_KEYS):
        missing = sorted(_V9FAIL10_KEYS - set(out["key"]))
        raise ValueError(
            f"ifeval_v9fail10: expected {len(_V9FAIL10_KEYS)} docs, got {len(out)} "
            f"(missing keys={missing}); a short subset is NOT a valid comparison"
        )
    return out
