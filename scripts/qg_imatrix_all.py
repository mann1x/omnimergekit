#!/usr/bin/env python
"""qg_imatrix_all.py — run quantize_gguf.py with IMATRIX_EXCLUDE emptied so EVERY
_K tier is built WITH the imatrix (STD16 loop-safety; user directive 2026-06-22:
"we may need imatrix for all quants to avoid looping").

Monkeypatch, not a source edit — keeps bs2's divergent quantize_gguf.py untouched.
All CLI args pass through verbatim to quantize_gguf.main()."""
import importlib.util
import sys

QG = "/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py"

spec = importlib.util.spec_from_file_location("quantize_gguf", QG)
qg = importlib.util.module_from_spec(spec)
sys.modules["quantize_gguf"] = qg          # so internal `import quantize_gguf` resolves
spec.loader.exec_module(qg)                 # __name__=="quantize_gguf" -> main() NOT auto-called

_before = sorted(qg.IMATRIX_EXCLUDE)
qg.IMATRIX_EXCLUDE.clear()                  # in-place -> the rule's global lookup sees it empty
print(f">>> IMATRIX_ALL: cleared IMATRIX_EXCLUDE (was {_before}) -> imatrix forced on ALL _K tiers",
      flush=True)

qg.main()
