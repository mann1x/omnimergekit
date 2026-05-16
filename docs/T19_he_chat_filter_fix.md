# HE chat extract filter fix — RCA + patch

**Date**: 2026-05-16
**File**: [`eval/lm_eval_tasks/humaneval_chat/utils_chat.py`](../eval/lm_eval_tasks/humaneval_chat/utils_chat.py)
**Backup**: `utils_chat.py.bak.20260516_141606` (same dir)
**Impact**: every Gemma 4 NVFP4A16 variant evaluated through `humaneval_chat`
task in the v5fixed sweep (and earlier) reported `pass@1 = 0.0` even when the
model produced correct code. **Not the stock lm-eval scorer** — the bug is in
our `extract_chat` filter (our `humaneval_chat` task is a custom shadow we
built to fix the stock-task chat-mode `until=["\ndef"]` truncation; the
shadow's filter then assumed a cleaner output shape than reasoning models
produce).

## Symptoms

T19 v5fixed sweep first 7 variants all scored HE-1 = 0.0:
```
A2_lp4_uni        HE-1 = 0
A3_top3mean_uni   HE-1 = 0
A4_softmax_t4_uni HE-1 = 0
A5_second_uni     HE-1 = 0
B1_max_genheavy   HE-1 = 0
B2_max_mathcode   HE-1 = 0
B3_max_tgtheavy   HE-1 = 0
```

Manual inspection of `samples_humaneval_chat_*.jsonl` showed all 7
contained the **correct** has_close_elements solution in their `resps`
field — but `filtered_resps` (filter output) had only the prompt-echo +
reasoning bullets, no function body.

## Root cause

The old `_clean_one` filter:

```python
_OPEN_FENCE = re.compile(r"^\s*```(?:python|py)?\s*\n?", re.IGNORECASE)

def _clean_one(resp: str, prompt: str) -> str:
    r = _OPEN_FENCE.sub("", resp)   # strip leading ```python ONLY
    i = r.find("```")                # find FIRST stray ``` anywhere
    if i != -1:
        r = r[:i]                    # truncate everything after
    if "def " not in r:
        r = prompt + r               # prepend prompt only if no def
    return r
```

Designed (2026-05-12) for the **clean case** where the model emits a single
fenced ```python ... ``` response. That's the case for completion-mode
models. With chat-completions + reasoning parser (Gemma 4), the model
emits:

```
[prompt-echo prefix]                  ← echoes the function signature stub
def has_close_elements(...):
    """ docstring """
*   Input: a list...                  ← reasoning bullets
*   Iterate through pairs...
    ```python
    [buggy first attempt]             ← first ``` here
    ```
    Wait, return True needs...
    ```python
    [CORRECT final attempt]
    ```
```

The old filter:
1. `_OPEN_FENCE.sub` — no leading fence to strip; no-op
2. `r.find("```")` — finds the FIRST `\`\`\`python` (opener of the first
   buggy attempt)
3. `r[:i]` — keeps only "[prompt-echo + reasoning bullets BEFORE the first
   `\`\`\``]"
4. `def has_close_elements` IS in that text (from the prompt echo) → no
   prompt re-prepend
5. Filtered candidate = prompt header + reasoning prose, NO callable body
6. `code_eval.compute` exec's "candidate + test cases" → `has_close_elements`
   isn't a real function → tests crash → `pass@1 = 0`

**It is not the official lm-eval HumanEval scorer.** That's
`evaluate.code_eval` and it's fine — it just exec's whatever it's handed.
The bug is **upstream** in the extraction step: our `extract_chat` filter
took the wrong slice. Stock lm-eval `humaneval` is the same shape (it has
its own bug for chat-mode that's the reason we wrote the shadow in the
first place).

## Fix

Rewrote `_clean_one` in [`utils_chat.py`](../eval/lm_eval_tasks/humaneval_chat/utils_chat.py)
with four pieces:

### 1. `_FENCED_BLOCK` regex

Matches every complete `\`\`\`[python|py]?\n ... \n\`\`\`` block — DOTALL +
non-greedy + negative-lookahead so `(.*?)` does NOT span across `\`\`\``.
Also `[ \t]*` before the closer to allow Gemma 4's 4-space-indented bullet
fences. Allows opener at any indent level.

### 2. `_smart_dedent`

`textwrap.dedent` finds the COMMON leading whitespace across all lines —
becomes empty when the block is contaminated with prose at column 0 (e.g.
`*   Input: ...`). The fix anchors on the **first non-empty line's prefix**
and strips THAT from every line starting with it. Prose lines at other
indents stay where they are and break the parser later (desired — they
get trimmed by `_parse_trim`).

### 3. `_parse_trim`

Line-by-line forward `ast.parse`. Build the candidate by adding one line
at a time; if it parses, record as best; if not, skip the line and keep
trying. Returns the longest parse-valid prefix containing `def `, or None.

This handles three rumination shapes:
- "def + body + return\n* reasoning bullets" — stops at the bullets
- "buggy first attempt with IndentationError" — never records as best
- Mixed tab/space indentation — Python tolerates if consistent within block

### 4. `_has_real_body`

Rejects degenerate candidates where the FunctionDef has only a docstring
(no real statements). Required because the regex would otherwise match
`def f(): """docstring"""` blocks (the model emits the prompt 71× in some
rumination cases) — those parse fine but execute as no-ops.

### 5. Always prepend the prompt

The previous behaviour ("prepend only if `def ` not in candidate") missed
the import lines. The candidate's `def` declaration uses `List[float]` but
typing.List is only imported in the prompt's preamble. Now `_clean_one`
ALWAYS prepends `doc["prompt"]`; the candidate's `def` redefines the
function (Python's last-binding wins), so the body comes from the
candidate while the imports come from the prompt.

### 6. In-module regression test

Six cases at the bottom of `utils_chat.py`:
1. Clean fence response (the original good case)
2. Reasoning + multiple attempts → must take last block
3. Body-only no fence → must prepend prompt
4. Indented fence → must extract
5. Code prefix contaminated by trailing bullets → parse-trim recovery
6. Buggy first attempt + correct second → skip the buggy one

Run with `python utils_chat.py` (set `HF_ALLOW_CODE_EVAL=1`). Trip wire —
fails immediately if anyone breaks the filter again.

## Rescore results

Offline rescore via [`eval/rescore_he1_smoke.py`](../eval/rescore_he1_smoke.py)
on the 7 completed-pre-fix samples files:

| Variant | OLD (broken) | NEW (rescored) |
|---|---:|---:|
| A2_lp4_uni | 0.0 | 0.0 |
| A3_top3mean_uni | 0.0 | 0.0 |
| **A4_softmax_t4_uni** | 0.0 | **1.0** |
| A5_second_uni | 0.0 | 0.0 |
| B1_max_genheavy | 0.0 | 0.0 |
| B2_max_mathcode | 0.0 | 0.0 |
| **B3_max_tgtheavy** | 0.0 | **1.0** |

A4 and B3 went 0 → 1.0 (real solutions correctly extracted now). The other
5 are genuine model failures (truncated mid-body, IndentationError in
final attempt, or all code outside fences in reasoning prose). I manually
inspected each remaining 0; none have a complete correct function in any
parse-valid fence.

Live runs from B4 onward used the fixed filter directly; **B4, B5, D3,
D4 all scored HE-1 = 1.0 live**.

## LCB shim was NOT affected

`lcb_llama_server.py` has its own `cleaned` extractor that walks for the
LAST `\`\`\`python ... \`\`\`` block — exactly the strategy the new
`_clean_one` now uses. B3 LCB-1 = 1.0 was correct from the start.

## Files touched

* [`eval/lm_eval_tasks/humaneval_chat/utils_chat.py`](../eval/lm_eval_tasks/humaneval_chat/utils_chat.py)
  — rewrite (kept backup as `*.bak.20260516_141606`)
* [`eval/rescore_he1_smoke.py`](../eval/rescore_he1_smoke.py) — offline
  rescore script (general-purpose; works on any v5fixed sweep variant)

## Cross-references

* [T19_v5fixed_sweep_results.md](T19_v5fixed_sweep_results.md) — full sweep
  table with rescored HE column
* [EVAL_PROTOCOL.md](../eval/EVAL_PROTOCOL.md) — eval-suite contract
