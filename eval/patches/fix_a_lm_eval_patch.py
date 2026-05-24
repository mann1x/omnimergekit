#!/usr/bin/env python3
"""Fix A: lm-eval reasoning_content fallback for openai_completions.parse_generations.

When vLLM returns {"content": "", "reasoning_content": "..."} for a Gemma 4
reasoning model that didn't emit a content phase (silent-empty pathology),
falls back to reasoning_content so the eval gets the model's actual output.

Idempotent: safe to run multiple times. Detects existing patch via marker.
"""
import sys, re
from pathlib import Path

PATCH_MARKER = "# Fix-A: reasoning_content fallback"

OLD = '''                for choices in out["choices"]:
                    tmp[choices["index"]] = choices["message"]["content"]'''

NEW = f'''                for choices in out["choices"]:
                    {PATCH_MARKER}
                    msg = choices["message"]
                    c = msg.get("content")
                    if not c:
                        rc = msg.get("reasoning_content")
                        if rc:
                            c = rc
                    tmp[choices["index"]] = c'''

def patch_file(p: Path) -> str:
    src = p.read_text()
    if PATCH_MARKER in src:
        return "skip-already-patched"
    if OLD not in src:
        return f"FAIL-no-match in {p}"
    p.write_text(src.replace(OLD, NEW))
    return "ok-patched"

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, "→", patch_file(Path(f)))
