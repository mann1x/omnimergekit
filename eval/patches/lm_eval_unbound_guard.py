#!/usr/bin/env python3
"""lm-eval UnboundLocalError guard for api_models.py.

Upstream lm-eval 0.4.11 references `outputs` inside an `except BaseException`
handler without initializing it first. If `session.post(...)` raises before
`outputs = await response.json()` lands (e.g. tenacity retries exhaust on a
transient TimeoutError), the handler crashes with
`UnboundLocalError: cannot access local variable 'outputs'` — which propagates
as a task-level FAIL even though every completed sample is already in the
sqlite cache. Hit repeatedly on cloud pods (36755693, 2026-05-14) on the
tail-end completions of gsm8k / gpqa.

Fix: initialize `outputs = None` before the try. Idempotent: detects the
inserted marker and skips. Mirrors fix_a_lm_eval_patch.py's interface so the
bootstrap can call both the same way.

    python3 lm_eval_unbound_guard.py <path/to/api_models.py> [more paths...]

See memory/feedback_lm_eval_unbound_outputs_bug.md.
"""
import sys
from pathlib import Path

PATCH_MARKER = "outputs = None  # PATCH"

OLD = (
    '        cache_method = "generate_until" if generate else "loglikelihood"\n'
    "        acquired = await sem.acquire()\n"
    "        try:"
)

NEW = (
    '        cache_method = "generate_until" if generate else "loglikelihood"\n'
    "        outputs = None  # PATCH: avoid UnboundLocalError in except\n"
    "        acquired = await sem.acquire()\n"
    "        try:"
)


def patch_file(p: Path) -> str:
    src = p.read_text()
    if PATCH_MARKER in src:
        return "skip-already-patched"
    if OLD not in src:
        return f"FAIL-no-match in {p} (lm-eval version drift?)"
    p.write_text(src.replace(OLD, NEW, 1))
    return "ok-patched"


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, "→", patch_file(Path(f)))
