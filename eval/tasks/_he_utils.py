"""Re-export of lm_eval.tasks.humaneval.utils functions.

lm-eval's !function tag resolves module names by treating them as a path
relative to the YAML directory (so `!function pkg.subpkg.utils.func` looks
for `pkg.subpkg.utils.py`). To reuse upstream HumanEval helpers without
forking them, we re-export from a sibling module that lm-eval can locate
in the include_path.
"""
import re
from lm_eval.tasks.humaneval.utils import (  # noqa: F401
    pass_at_k,
    build_predictions,
    build_predictions_instruct,
)


# Match a fenced code block. Tolerates ```python, ```py, or bare ```.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def _extract_code(resp: str) -> str:
    """Pull the largest ```python``` block from a chat response.

    The model's chat reply for HumanEval re-emits the entire function
    (signature + docstring + body), wrapped in a single ```python ... ```
    fence. Upstream `build_predictions_instruct` is geared to `gen_prefix`
    mode where the model only emits the body — without `gen_prefix` it
    truncates at the prompt's docstring close. This extractor returns
    the largest fenced Python block instead, which exec()'s correctly
    against `check(entry_point)`.
    """
    matches = _FENCE_RE.findall(resp)
    if matches:
        # Prefer the largest block — model sometimes emits a tiny
        # incidental fence before the real solution.
        return max(matches, key=len)
    # No fences: assume raw code.
    return resp


def build_predictions_chat(resps, docs):
    """Filter for chat-completions HumanEval.

    Signature mirrors upstream `build_predictions` / `build_predictions_instruct`:
    takes a list-of-list of model responses (one inner list per doc) plus the
    docs themselves, returns the predictions list-of-list shaped the same way.
    Each prediction is `extracted_code` (which already contains the function
    definition). `check(entry_point)` is appended by the doc_to_target template
    so the final exec'd source is `extracted_code\\n{test}\\ncheck({entry_point})`.
    """
    out = []
    for resp_list, doc in zip(resps, docs):
        out.append([_extract_code(r) for r in resp_list])
    return out
