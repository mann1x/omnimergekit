"""Chat-aware HumanEval filter.

Stock lm-eval `humaneval` / `humaneval_instruct` / `humaneval_plus` set
`generation_kwargs.until = ["\\nclass", "\\ndef", "\\n#", "\\nif", "\\nprint"]`.
That works for completion-mode where the prompt ends mid-`def` and the model
fills in the body. With chat-completions + reasoning parser (Gemma 4 et al.)
the model emits a fresh `\\`\\`\\`python\\nfrom typing import ...\\n\\ndef foo(...):\\n  body`
response. The `\\ndef` until then truncates after the imports, BEFORE the body
ever lands in `resps` — `pass@1` collapses to 0 even when the model is correct.

The override tasks `humaneval_chat` / `humaneval_plus_chat` instead stop at the
closing markdown fence (`until: ["\\n\\`\\`\\`", "</s>"]`) and route resps through
`build_predictions_chat` below, which:

  1. Strips a leading ```python / ``` / ```py fence.
  2. Trims at any surviving closing fence (defensive — until should have caught
     it, but model variants sometimes emit stray triple-backticks mid-output).
  3. If the model emitted only a function BODY (no `def`), prepends `doc["prompt"]`
     so the resulting candidate is a self-contained function (matches the
     `build_predictions` completion-mode contract).
  4. Otherwise, returns the response as-is — the model redeclared the signature
     and the candidate is already self-contained.

The downstream `pass_at_k` metric concatenates `doc_to_target` (test cases) to
each candidate and execs the result. Either shape works as long as the function
named `entry_point` is defined exactly once.
"""

from __future__ import annotations

import re

import evaluate as hf_evaluate

# Initialize the HF code_eval scorer once (mirrors lm-eval's humaneval/utils.py).
_compute = hf_evaluate.load("code_eval")
# Warm-up call — surfaces broken sandbox configs early.
_compute.compute(
    references=["assert add(2, 3)==5"],
    predictions=[["def add(a,b): return a*b"]],
    k=[1],
)


def pass_at_k(references, predictions, k=None):
    """Local copy of lm-eval's humaneval/utils.pass_at_k so the YAML can
    reference it via `!function utils_chat.pass_at_k` instead of a fragile
    relative path into the site-packages tree."""
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    res = _compute.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


_OPEN_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n?", re.IGNORECASE)


def _clean_one(resp: str, prompt: str) -> str:
    r = _OPEN_FENCE.sub("", resp)
    i = r.find("```")
    if i != -1:
        r = r[:i]
    # If the model didn't redeclare the function signature, prepend the prompt
    # so the candidate is a complete function (matches completion-mode contract).
    if "def " not in r:
        r = prompt + r
    return r


def build_predictions_chat(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [_clean_one(r, doc["prompt"]) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
