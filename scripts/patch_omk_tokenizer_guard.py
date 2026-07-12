#!/usr/bin/env python3
# patch_omk_tokenizer_guard.py — durable fix for the GGUF-as-tokenizer footgun.
#
# Adds `_resolve_hf_tokenizer()` to omk_eval.py and calls it at the top of
# dispatch_lm_eval (the lm-eval HF-tokenizer path). When --tokenizer is omitted
# and --model is a .gguf, lm-eval would call AutoTokenizer.from_pretrained(<binary>)
# and crash ~97s in at construction ("not a valid JSON file"), AFTER the server
# booted, leaving a 0-sample / score=null result and (with naive orchestrators)
# a fake done-marker. This guard auto-resolves to a sibling HF tokenizer dir when
# one sits next to the .gguf, else fails fast (sub-second) with the exact fix.
#
# ADDITIVE + IDEMPOTENT: inserts one function + one call, anchored relative to
# `def dispatch_lm_eval(` (NOT a blind string-replace — there are 5 cache_dir
# lines). Re-running is a no-op. Safe on bs2's divergent repo (no clobber).
#
# Usage: python3 patch_omk_tokenizer_guard.py [/path/to/omk_eval.py]
import py_compile
import sys
from pathlib import Path

F = sys.argv[1] if len(sys.argv) > 1 else "/srv/ml/repos/omnimergekit/eval/omk_eval.py"

HELPER = '''def _resolve_hf_tokenizer(tokenizer: str) -> str:
    """Guard the GGUF-as-tokenizer footgun. lm-eval's `tokenizer_backend=
    huggingface` needs a real HF tokenizer (a dir with tokenizer.json /
    tokenizer_config.json, or a hub id). When `--tokenizer` is omitted, main()
    defaults it to `--model`; if `--model` is a `.gguf`, lm-eval calls
    `AutoTokenizer.from_pretrained(<binary>)` and dies ~97s in at construction
    ("not a valid JSON file") - AFTER the server booted, leaving a
    0-sample / score=null result. Fail fast here (sub-second, pre-construction)
    with an actionable message, and auto-resolve to a sibling HF dir when one
    sits next to the .gguf."""
    p = Path(tokenizer)
    if p.is_dir() and ((p / "tokenizer.json").is_file()
                       or (p / "tokenizer_config.json").is_file()):
        return tokenizer
    if p.suffix == ".gguf" or p.is_file():
        sib = p.parent
        if (sib / "tokenizer.json").is_file() or (sib / "tokenizer_config.json").is_file():
            log(f"tokenizer: --tokenizer pointed at a GGUF/file ({p.name}); "
                f"auto-resolved to sibling HF tokenizer dir {sib}")
            return str(sib)
        fatal(21,
              f"this template needs an HF tokenizer but --tokenizer resolved to a "
              f"GGUF/file ({tokenizer}). lm-eval cannot load a tokenizer from a .gguf "
              f"and would crash ~97s in at construction. Pass --tokenizer <HF model "
              f"dir or hub id> (e.g. the bf16 source dir, or google/gemma-4-26B-A4B-it). "
              f"Failing now instead of after server boot.")
    # Not a local path -> assume a HF hub id (org/name); lm-eval validates it.
    return tokenizer


'''

CALL = ('    # Durable guard: feeds `tokenizer` to lm-eval HF backend; a GGUF\n'
        '    # --model defaulted as tokenizer crashes ~97s in. Auto-resolve or fail fast.\n'
        '    tokenizer = _resolve_hf_tokenizer(tokenizer)\n')

MARKER = "def dispatch_lm_eval("
ANCHOR = '    cache_dir = out_dir / "sqlite_cache"\n'


def patch(text: str) -> str:
    if "_resolve_hf_tokenizer" in text:
        print("ALREADY_PATCHED")
        return text
    i = text.find("\n" + MARKER)
    if i == -1:
        sys.exit("FATAL: 'def dispatch_lm_eval(' not found")
    i += 1  # position of 'def'
    text = text[:i] + HELPER + text[i:]
    j = text.find(MARKER)
    k = text.find(ANCHOR, j)
    if k == -1:
        sys.exit("FATAL: cache_dir anchor not found inside dispatch_lm_eval")
    text = text[:k] + CALL + text[k:]
    return text


def selftest() -> None:
    # exec the helper with stubs; prove fail-fast + auto-resolve behavior.
    class _Fatal(Exception):
        pass

    def _fatal(code, msg):
        raise _Fatal(msg)

    ns = {"Path": Path, "log": lambda m: None, "fatal": _fatal}
    exec(HELPER, ns)
    fn = ns["_resolve_hf_tokenizer"]
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as d:
        # case 1: a real HF tokenizer dir -> returned unchanged
        Path(d, "tokenizer.json").write_text("{}")
        assert fn(d) == d, "case1 HF dir"
        # case 2: a .gguf whose dir HAS a tokenizer.json -> auto-resolve to dir
        g = os.path.join(d, "model-Q6_K.gguf")
        Path(g).write_text("GGUF")
        assert fn(g) == d, "case2 gguf auto-resolve"
        # case 3: a .gguf in a dir with NO tokenizer -> fail fast
        with tempfile.TemporaryDirectory() as d2:
            g2 = os.path.join(d2, "x.gguf")
            Path(g2).write_text("GGUF")
            try:
                fn(g2)
            except _Fatal:
                pass
            else:
                raise AssertionError("case3 should have failed fast")
        # case 4: a hub id (not a local path) -> passthrough
        assert fn("google/gemma-4-26B-A4B-it") == "google/gemma-4-26B-A4B-it", "case4 hub id"
    print("SELFTEST_OK (HF-dir / gguf-auto-resolve / gguf-fail-fast / hub-id)")


if __name__ == "__main__":
    selftest()
    src = Path(F).read_text()
    out = patch(src)
    if out != src:
        Path(F).write_text(out)
        py_compile.compile(F, doraise=True)
        print(f"PATCHED_OK {F}")
