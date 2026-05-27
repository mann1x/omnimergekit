#!/usr/bin/env python3
"""Fix-A: lm-eval reasoning_content fallback for openai_completions.parse_generations.

Reasoning models (Gemma 4 with --reasoning-parser, Qwen3.5 with
--reasoning-format deepseek) can return {"content": "", "reasoning_content":
"<the answer>"} when the answer lands in the stripped reasoning channel. Stock
lm-eval reads only `content` -> scores empty -> wrong. Fix-A makes the parser
content-first with a reasoning_content fallback ONLY when content is empty
(concat-when-both contaminates rule-based scorers like IFEval). RCA:
memory/feedback_vllm_gemma4_silentempty_rca.md.

Why a string-replace and NOT a `.patch`: a context-diff assumes you know the
file's starting state. Fresh pods ship the STOCK lm-eval 0.4.11 form; solidpc
carried an intermediate form; the old `.patch` only matched the latter and
silently "Hunk FAILED" on every fresh pod (2026-05-27 day-burn). A targeted
string-replace anchored on the stock line works regardless of line numbers and
is idempotent via the sentinel.

Idempotent: re-running is a no-op once the sentinel is present.
Usage: fix_a_lm_eval_patch.py /path/to/lm_eval/models/openai_completions.py
"""
import sys
from pathlib import Path

SENTINEL = "refined 2026-05-21"

# Stock lm-eval 0.4.11 LocalChatCompletion.parse_generations body.
OLD = (
    '                for choices in out["choices"]:\n'
    '                    tmp[choices["index"]] = choices["message"]["content"]'
)

NEW = (
    '                for choices in out["choices"]:\n'
    '                    msg = choices["message"]\n'
    '                    content = msg.get("content") or ""\n'
    '                    # Fix-A (2026-05-14, refined 2026-05-21 stack@2): content-first;\n'
    '                    # reasoning_content fallback ONLY when content empty (avoids\n'
    '                    # rule-based-scorer contamination). RCA:\n'
    '                    # feedback_vllm_gemma4_silentempty_rca.md\n'
    '                    if content:\n'
    '                        text = content\n'
    '                    else:\n'
    '                        text = msg.get("reasoning") or msg.get("reasoning_content") or ""\n'
    '                    tmp[choices["index"]] = text'
)


def patch_file(p: Path) -> str:
    src = p.read_text()
    if SENTINEL in src:
        return "skip-already-applied"
    if OLD not in src:
        return ("FAIL-no-match (file is neither stock 0.4.11 nor already "
                "Fix-A'd; inspect parse_generations manually)")
    p.write_text(src.replace(OLD, NEW, 1))
    return "ok-patched"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: fix_a_lm_eval_patch.py <openai_completions.py> ...")
    rc = 0
    for f in sys.argv[1:]:
        r = patch_file(Path(f))
        print(f, "->", r)
        if r.startswith("FAIL"):
            rc = 3
    sys.exit(rc)
