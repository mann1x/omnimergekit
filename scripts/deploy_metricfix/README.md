# Backport of the canonical-score selection fix (`f06095a`) to stale run hosts

`f06095a` fixed `extract_canonical_score`: its last resort was
`next(iter(score_dict))`, an arbitrary key. lm-eval mixes bookkeeping entries into
that dict, so when a template's declared metric did not match what the task
emitted, it returned **`sample_len` — the row COUNT — as the score**. Four Ornith
cells were stamped `score=164.0` whose real `pass@1` was 0.89/0.83/0.90/0.81.

Trigger: `humaneval_full.yaml` declares `pass_at_1` / `build_predictions`
(validated against stock `humaneval`), but the run executed `humaneval_chat`,
which emits `pass@1,extract_chat`. Neither the exact key nor the `pass_at_1,`
prefix matched, so the fallback fired.

## Why a patcher and not a file sync

The run hosts are not clean checkouts:

- the pod's `/workspace/omk` is **not a git clone at all**, it is a plain copy;
- bs2's clone has ~150 locally-modified files, `eval/omk_eval.py` among them.

Overwriting either would destroy run-host work. This applies only the one block.
It is idempotent, refuses unless the old block appears exactly once, `ast.parse`s
before writing, and swaps atomically so a bench launching mid-write can never see
a partial file.

## Use

    python patch_metric_selection.py <path>/omk_eval.py old_block.txt new_block.txt
    python verify_metricfix.py <path>/eval        # must print BOTH PASS

`verify_metricfix.py` checks both directions — that the real `pass@1` comes back,
AND that a bookkeeping-only dict is REFUSED rather than turned into a plausible
number. A one-sided check would pass on the broken code.

`new_block.txt` is a copy of the fixed block as of `f06095a`. Once every host is
synced past that commit this directory is dead weight and should be deleted.
