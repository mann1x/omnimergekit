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
`build_predictions_chat` below.

Filter design (`_clean_one`):

  Reasoning models emit multi-section output: prompt-echo prefix + reasoning
  bullets + one OR MORE attempted ```python ... ``` blocks. The user's final
  answer is the LAST fenced block. Earlier (2026-05-12) the filter assumed a
  single fenced response and cut at the FIRST stray ``` — which on Gemma 4
  reasoning chopped before any code block and yielded pass@1=0 across all
  variants in the v5fixed sweep even when correct code was present.

  Current strategy (2026-05-16):
    1. Walk for every ```python|py|<no-lang>``` ... ``` block (line-leading
       OR indented — Gemma 4 reasoning uses 4-space indented fences). Allow
       the opener fence to be at line start with arbitrary leading whitespace.
    2. For each candidate block, `ast.parse`-trim: drop trailing lines until
       the block is valid Python. This handles the Gemma 4 rumination shape
       where the model emits "def + body + return\\n* reasoning bullets\\n..."
       inside ONE fence pair (the regex sees one block, but only the prefix
       is code). It also drops the model's BUGGY first attempts which often
       have indentation errors.
    3. Among the parse-valid blocks, take the LAST one that contains `def `.
       That's the model's final answer.
    4. If no valid fenced block — fall back to the legacy "strip leading
       fence, truncate at first stray ```" path (covers clean-fence responses
       and `until="\\n```"`-terminated bodies).
    5. If the candidate still has no `def`, prepend `doc["prompt"]` (matches
       the completion-mode contract — `pass_at_k` exec'd `cand + test`).

The downstream `pass_at_k` metric concatenates `doc_to_target` (test cases) to
each candidate and execs the result. Either shape works as long as the function
named `entry_point` is defined exactly once.

Regression discipline: the bottom of this module ships a tiny test (
`python utils_chat.py`) that mimics the Gemma 4 reasoning output shape and the
clean-fence shape; any change to `_clean_one` that breaks either case fails the
script immediately. CI does not run this — it's a developer-side trip wire.
"""

from __future__ import annotations

import ast
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


# Match every complete fenced block. The opener `[ \t]*```` allows the line-
# leading indentation that Gemma 4 reasoning uses inside bullets; DOTALL so .
# spans newlines; non-greedy `(.*?)` so multi-block responses split correctly;
# the content disallows `\`\`\`` via lookahead so we don't accidentally span
# across a closer into the next block.
_FENCED_BLOCK = re.compile(
    r"[ \t]*```(?:python|py)?[ \t]*\n((?:(?!```).)*?)\n[ \t]*```",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n?", re.IGNORECASE)


def _smart_dedent(block: str) -> str:
    """Like textwrap.dedent but anchored on the FIRST non-empty line's indent.

    `textwrap.dedent` uses the common leading whitespace across ALL lines,
    which becomes the empty string when the model contaminates the block
    with prose at column 0 ("*   Input: ..."). Anchoring on the first
    non-empty line keeps the dedent working in that case — the contamination
    stays at its weird indentation and fails parse-trim later (the desired
    behavior, since we trim it off anyway)."""
    prefix = ""
    for ln in block.split("\n"):
        if ln.strip():
            m = re.match(r"^[ \t]*", ln)
            prefix = m.group() if m else ""
            break
    if not prefix:
        return block
    out = []
    for ln in block.split("\n"):
        out.append(ln[len(prefix):] if ln.startswith(prefix) else ln)
    return "\n".join(out)


def _parse_trim(block: str) -> str | None:
    """Recover the longest leading parse-valid Python prefix containing `def `.

    Walks line-by-line FROM THE FRONT, keeping each line in turn and
    `ast.parse`-ing the cumulative candidate. Records the last version that
    parsed; returns it if it contains `def `. Skips a line if ast.parse
    rejects it (it might be a stray bullet) and keeps trying — this handles
    the "valid function followed by reasoning bullets" rumination shape.

    Also defends against IndentationError / TabError on the model's BUGGY
    first attempts by simply not recording them as best.
    """
    block = _smart_dedent(block)
    lines = block.split("\n")
    best: str | None = None
    cur: list[str] = []
    for ln in lines:
        cur.append(ln)
        candidate = "\n".join(cur).rstrip()
        if not candidate.strip():
            continue
        try:
            tree = ast.parse(candidate)
            # Skip degenerate cases: function defined but body is JUST the
            # docstring with no executable statements — pass_at_k would
            # return None implicitly, fails the test. Require at least one
            # non-docstring statement in any FunctionDef.
            if _has_real_body(tree):
                best = candidate
        except (SyntaxError, IndentationError):
            continue
    if best and "def " in best:
        return best
    return None


def _has_real_body(tree: ast.AST) -> bool:
    """True if any FunctionDef in `tree` has at least one statement that's
    not just a docstring. (A def with body == [Expr(Constant(str))] is a
    no-op shell that won't pass any test.)"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if not body:
                continue
            if len(body) == 1 and isinstance(body[0], ast.Expr) and \
               isinstance(body[0].value, ast.Constant) and \
               isinstance(body[0].value.value, str):
                continue  # docstring-only
            return True
    return False


def _clean_one(resp: str, prompt: str) -> str:
    # Walk every complete fenced block. For each, ast.parse-trim to recover
    # the longest valid code prefix. Take the LAST surviving candidate.
    blocks = _FENCED_BLOCK.findall(resp)
    valid: list[str] = []
    for b in blocks:
        v = _parse_trim(b)
        if v is not None:
            valid.append(v)
    if valid:
        cand = valid[-1]
    else:
        # No parseable fenced block. Fall back: strip leading opening fence,
        # truncate at first stray ``` (legacy clean-fence / `until`-stopped path).
        r = _OPEN_FENCE.sub("", resp)
        i = r.find("```")
        if i != -1:
            r = r[:i]
        # Try parse-trim on the fallback too — handles untrimmed reasoning trail.
        v = _parse_trim(r)
        cand = v if v is not None else r

    # Always prepend `doc["prompt"]`. The prompt carries `from typing import
    # List` (etc.) + the function signature with docstring. The model's `def`
    # in `cand` (if present) redefines the function — Python's last-binding
    # wins, so the body comes from `cand` while the imports come from the
    # prompt. This eliminates a class of pass@1=0 failures where the model
    # emitted only the function (`def foo(...): body`) without re-importing
    # the type aliases used in its own signature.
    return prompt + cand


def build_predictions_chat(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [_clean_one(r, doc["prompt"]) for r in resp]
        for resp, doc in zip(resps, docs)
    ]


# ─── Curated subset selectors ────────────────────────────────────────────────
# `process_docs` lm-eval hooks. Each takes a `datasets.Dataset` and returns
# a filtered copy. Used by smoke shadow tasks that need a non-contiguous
# subset (lm-eval `--limit` only takes first-N, which can't probe specific
# v4-fail indices). The indices below match the matching template YAML
# under eval/templates/. Keep these two in sync if you ever re-roll.

# HE+ curated 30-problem set for v5-coder T21 qualification (2026-05-17):
#   5 v4-fails (128e-PASS) + 25 lowest-128e-chars v4-passes from the 128e-
#   PASS pool. v4 anchor 25/30, 128e 30/30.
_HE_PLUS_CURATED30_INDICES = {
    11, 13, 14, 15, 16, 20, 22, 23, 24, 27,
    28, 30, 34, 35, 42, 45, 47, 48, 49, 52,
    53, 55, 58, 60, 62, 85, 121, 140, 154, 163,
}


def select_he_plus_curated30(dataset):
    """Filter HE+ dataset to the 30 curated indices for the smoke template."""
    return dataset.select(sorted(_HE_PLUS_CURATED30_INDICES))


# HE+ curated 15-problem FAST recipe-screening diagnostic (T125, 2026-05-26):
# built from the 62e-v1-coder EAC router-recovery run as the exemplar recipe.
# Composition + provenance in eval/templates/humanevalplus_15.yaml; indices are
# auto-filled by scripts/build_he15_curated.py (keep in sync with that template).
_HE_PLUS_CURATED15_INDICES = {0, 1, 5, 10, 13, 14, 23, 39, 45, 53, 59, 75, 129, 140, 143}  # AUTOINDICES15 — filled by build_he15_curated.py


def select_he_plus_curated15(dataset):
    """Filter HE+ dataset to the 15 curated fast-diagnostic indices."""
    if not _HE_PLUS_CURATED15_INDICES:
        raise ValueError(
            "humaneval_plus_chat_curated15: indices not yet populated — run "
            "scripts/build_he15_curated.py after the post-EAC HE+ run completes."
        )
    return dataset.select(sorted(_HE_PLUS_CURATED15_INDICES))


# fc15_25 HE+ regression fast-screen: the 15 HumanEval+ problems the 62e
# v1-coder (fc15_25-p8 mean-20 tie-break winner) FAILS — 3 rumination
# (76, 84, 156) + 12 capability. Used to screen recipe-fine-tuning candidates
# on the exact HE+ failures without a full 164-problem run. doc_id == HumanEval/N.
# See scripts/fc15_25_anomaly_testset.json + scripts/fc15_25_rerun_diff.py.
_HE_PLUS_FC15_INDICES = {15, 32, 39, 76, 83, 84, 91, 95, 103, 115, 132, 139, 145, 156, 163}


def select_he_plus_fc15(dataset):
    """Filter HE+ dataset to the 15 fc15_25-p8 failing indices (recipe screen)."""
    if not _HE_PLUS_FC15_INDICES:
        raise ValueError("humaneval_plus_chat_fc15: indices not populated")
    return dataset.select(sorted(_HE_PLUS_FC15_INDICES))


# 21q RUMINATION fast-screen — the 3 HE+ problems the 62e v1-coder fc15_25-p8
# fails by RUMINATION (loop/runaway/ruminate-fail), not capability. These are
# the recipe-addressable HE+ slice of the 21q rumination set (the other 18 are
# IFEval×3 + MPE×15). doc_id == HumanEval/N. See scripts/fc15_25_anomaly_testset.json.
_HE_PLUS_RUM3_INDICES = {76, 84, 156}


def select_he_plus_rum3(dataset):
    """Filter HE+ dataset to the 3 fc15_25-p8 RUMINATION indices (21q screen)."""
    if not _HE_PLUS_RUM3_INDICES:
        raise ValueError("humaneval_plus_chat_rum3: indices not populated")
    return dataset.select(sorted(_HE_PLUS_RUM3_INDICES))


# ─── Regression test ──────────────────────────────────────────────────────────
# `python utils_chat.py` runs these. If you change `_clean_one`, run it.

def _selftest() -> None:
    PROMPT = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ docstring """\n'
    )
    CORRECT_BODY = (
        "def has_close_elements(numbers, threshold):\n"
        "    n = len(numbers)\n"
        "    for i in range(n):\n"
        "        for j in range(i + 1, n):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n"
    )

    # Case 1: clean single fenced response (legacy good case)
    r1 = "```python\n" + CORRECT_BODY + "```"
    out1 = _clean_one(r1, PROMPT)
    assert "return True" in out1 and "def has_close_elements" in out1, (
        "case 1 (clean fence) regressed: " + out1[:200]
    )

    # Case 2: Gemma 4 reasoning shape — prompt-echo + reasoning + buggy first
    # attempt + CORRECT second attempt. Filter MUST take the second attempt.
    r2 = (
        PROMPT
        + "*   Input: list, threshold...\n"
        + "*   Iterate pairs...\n\n"
        + "```python\n"
        + "def has_close_elements(numbers, threshold):\n"
        + "    return False  # WRONG: never returns True\n"
        + "```\n\n"
        + "Wait, that's wrong. Let me fix.\n\n"
        + "```python\n"
        + CORRECT_BODY
        + "```\n"
    )
    out2 = _clean_one(r2, PROMPT)
    assert "return True" in out2, (
        "case 2 (reasoning, last-block) regressed — got first attempt or prefix: "
        + out2[:300]
    )
    # And the buggy first attempt must NOT be in the candidate
    assert "WRONG" not in out2, (
        "case 2 leaked first attempt into candidate: " + out2[:300]
    )

    # Case 3: body-only with no fence and no `def` — should prepend prompt
    r3 = "    return any(abs(a-b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n"
    out3 = _clean_one(r3, PROMPT)
    assert "def has_close_elements" in out3, "case 3 (body-only) missed prompt-prepend"

    # Case 4: indented fences inside reasoning bullets (Gemma 4 actual shape)
    r4 = (
        "Some reasoning.\n\n"
        "    ```python\n"
        "    " + CORRECT_BODY.replace("\n", "\n    ").rstrip()
        + "\n    ```\n"
    )
    out4 = _clean_one(r4, PROMPT)
    assert "return True" in out4, "case 4 (indented fence) missed code: " + out4[:300]

    # Case 5: Gemma 4 rumination shape — code prefix inside the fence followed
    # by reasoning bullets that contaminate the closing-side of the same block.
    # ast.parse-trim must drop the trailing bullets and keep the code.
    r5 = (
        "Reasoning.\n```python\n"
        + CORRECT_BODY
        + "*   Input: A list of numbers...\n"
        + "*   Goal: pairs check.\n"
        + "```\n"
    )
    out5 = _clean_one(r5, PROMPT)
    assert "return True" in out5 and "*   Input" not in out5, (
        "case 5 (parse-trim) leaked reasoning bullets: " + out5[:400]
    )

    # Case 6: buggy first attempt (IndentationError) followed by correct second.
    # parse-trim must reject the buggy one and pick the correct one.
    r6 = (
        "```python\n"
        "def has_close_elements(numbers, threshold):\n"
        "    for i in range(len(numbers)):\n"
        "        for j in range(i+1, len(numbers)):\n"
        "            if abs(numbers[i]-numbers[j]) < threshold:\n"
        "    return False\n"  # IndentationError — return outside if
        "```\n"
        "Wait, that doesn't work.\n"
        "```python\n"
        + CORRECT_BODY
        + "```\n"
    )
    out6 = _clean_one(r6, PROMPT)
    assert "return True" in out6, "case 6 (skip-buggy) missed correct attempt: " + out6[:300]

    print("utils_chat self-test: 6/6 cases ok")


if __name__ == "__main__":
    _selftest()
