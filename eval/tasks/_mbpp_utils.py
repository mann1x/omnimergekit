"""Re-export of lm_eval.tasks.mbpp.utils for use by curated subset YAMLs."""
import re
from lm_eval.tasks.mbpp.utils import (  # noqa: F401
    pass_at_1,
    list_fewshot_samples,
    build_predictions,
)


# Match a fenced code block. Prefers ```python / ```py, falls back to bare ```.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def _extract_code(resp: str) -> str:
    """Pull the largest ```python``` block from a chat response.

    Chat-tuned IT models often write a paragraph of natural-language
    reasoning ("To solve this problem, we need to ...") followed by the
    code. Upstream `build_predictions` extracts whatever's between the
    first fence pair, but on prose-heavy responses with no fences it
    returns the prose verbatim and exec() throws SyntaxError → 0%.

    This extractor returns the largest ```python``` block. If no fences
    are present, it falls back to scanning for the first `def ...`
    block to the end of the response — best-effort recovery for prose-
    only responses.
    """
    matches = _FENCE_RE.findall(resp)
    if matches:
        return max(matches, key=len)
    # Fallback: try to find a `def` line and return everything from there.
    m = re.search(r"^[ \t]*def [^\n]+\n", resp, re.MULTILINE)
    if m:
        return resp[m.start():]
    # Last resort — return as-is and let exec fail loudly.
    return resp


def build_predictions_chat(resps, docs):
    """Filter for chat-completions MBPP.

    Same shape as upstream `build_predictions`: list-of-list in, list-of-list
    out, one prediction per response per doc. Each prediction is the largest
    ```python``` block from the model's chat reply.
    """
    out = []
    for resp_list, doc in zip(resps, docs):
        out.append([_extract_code(r) for r in resp_list])
    return out
