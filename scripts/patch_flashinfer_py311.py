#!/usr/bin/env python3
"""Make flashinfer 0.6.16.post3 importable on Python 3.11.

THE BUG IS UPSTREAM AND IT IS A HARD BLOCKER FOR TP>1.
`flashinfer/comm/fd_exchange.py` annotates a return type as:

    def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:

`array.array` only became subscriptable in **Python 3.12**. On 3.11 the annotation is
evaluated at def time and raises:

    TypeError: type 'array.array' is not subscriptable

vLLM imports `flashinfer.comm` from `cuda_communicator.py` with **no try/except**, and
only does so when building a TP group -- so this is invisible at
`vllm_tensor_parallel_size=1` and fatal at 2. Removing flashinfer is not an option
either: vllm 0.27.1 declares `flashinfer-python==0.6.16.post3` as a hard dependency and
the import site is unguarded.

THE FIX: add `from __future__ import annotations` to that module. All annotations
become lazy strings, so the offending expression is never evaluated. Annotations are
not read at runtime in this file, so behaviour is unchanged -- this defers a type
expression, it does not alter one.

A pip reinstall/upgrade of flashinfer SILENTLY REVERTS THIS. Re-run after any rebuild
of the omk-grpo env, the same way the lm-eval sampled-cache patch has to be re-applied.
[[feedback_use_cache_is_a_noop_when_do_sample_is_true]]

Idempotent; verifies with ast.parse + a real import before declaring success.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import shutil
import subprocess
import sys

FUTURE = "from __future__ import annotations\n"


def find_target(python_bin: str) -> pathlib.Path:
    out = subprocess.run(
        [python_bin, "-c",
         "import flashinfer, pathlib, sys; "
         "sys.stdout.write(str(pathlib.Path(flashinfer.__file__).parent))"],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"REFUSE: cannot locate flashinfer with {python_bin}: {out.stderr[-400:]}")
    p = pathlib.Path(out.stdout.strip()) / "comm" / "fd_exchange.py"
    if not p.is_file():
        sys.exit(f"REFUSE: expected {p} to exist; flashinfer layout changed.")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/srv/ml/envs/envs/omk-grpo/bin/python")
    a = ap.parse_args()

    target = find_target(a.python)
    src = target.read_text()

    if FUTURE in src:
        print(f"already patched: {target}")
    else:
        # Insert as the first STATEMENT -- after the license comment block and after a
        # module docstring if one exists, because `from __future__` must precede every
        # other statement or Python raises SyntaxError.
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        insert_at = 0
        if tree.body:
            first = tree.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_at = first.end_lineno          # after the docstring
            else:
                insert_at = first.lineno - 1          # before the first real statement
        patched = "".join(lines[:insert_at]) + FUTURE + "".join(lines[insert_at:])
        try:
            ast.parse(patched)
        except SyntaxError as e:
            sys.exit(f"REFUSE: patched file does not parse ({e}); nothing written.")
        bak = target.with_suffix(".py.bak_py311")
        if not bak.exists():
            shutil.copy2(target, bak)
        target.write_text(patched)
        print(f"patched: {target}  (backup {bak.name})")

    # The real gate: does the import that vLLM performs actually work now?
    chk = subprocess.run(
        [a.python, "-c",
         "import flashinfer.comm as c; "
         "from flashinfer.comm.fd_exchange import _fd_ancillary; "
         "print('flashinfer.comm import OK')"],
        capture_output=True, text=True)
    print(chk.stdout.strip() or chk.stderr[-600:])
    if chk.returncode != 0:
        sys.exit("REFUSE: flashinfer.comm still does not import; the patch is not "
                 "sufficient and TP>1 will still fail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
