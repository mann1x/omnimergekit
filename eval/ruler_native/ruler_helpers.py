"""RULER (NVIDIA/RULER, arXiv:2404.06654) helpers for the omk ruler_native backend.

Three things live here:

1. The inline scorer port (`string_match_all`) — verbatim from upstream
   `scripts/eval/synthetic/constants.py:25`. We inline rather than subprocess
   into upstream's `evaluate.py` because that script forces a
   `nemo-toolkit[all]` install that silently downgrades transformers / torch /
   safetensors / modelopt out from under the canonical omk env pins.
   See module-level note in `__init__.py` for the RCA.

2. RULER root discovery (`locate_ruler_root`) — checks env `RULER_ROOT`, then
   `/workspace/RULER` (pod), then `/shared/dev/RULER` (solidpc). The runner
   subprocess-calls `<root>/scripts/data/prepare.py` to generate the staged
   validation.jsonl per (task, max_seq_length, num_samples). No part of the
   upstream repo is vendored or copied — runtime-clone pattern, same as
   `/opt/llama.cpp`.

3. Parser for upstream's `validation.jsonl` schema (`load_validation_jsonl`) —
   one record per (task, sample) with fields {input, outputs, length, index}.
   We feed `input` to the model and score the model's response against
   `outputs` (a list of acceptable substrings — RULER VT, NIAH, CWE, FWE
   variants all share this schema).

License attribution: NVIDIA/RULER is Apache-2.0. The scorer function below is
copied verbatim from upstream commit fda33fc (`scripts/eval/synthetic/constants.py`)
under that license. No other upstream code is included; the runtime clone is
the canonical source for everything else (prepare.py, the synthetic generators,
etc.).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Scorer: verbatim port from upstream eval/synthetic/constants.py ─────────
#
# Apache-2.0 attribution: scripts/eval/synthetic/constants.py from NVIDIA/RULER
# at commit fda33fc. The case-insensitive substring-match semantics + the
# `r.lower() in pred.lower()` per-needle accumulation are EXACTLY the upstream
# behavior. Do not "modernize" this — comparisons against RULER paper numbers
# and other published RULER scores depend on byte-identical scoring math.

def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    """Case-insensitive substring match averaged over needles, averaged over samples.

    For each (pred, ref_list) pair: average over ref_list of
    (1.0 if needle.lower() in pred.lower() else 0.0). Then average over
    all samples and multiply by 100 to get a percentage.

    Verbatim from upstream `eval/synthetic/constants.py:25`. The exact form
    of the comprehension matters — `sum(...) / len(ref)` per-sample, then
    `sum(...) / len(preds) * 100` over samples — because RULER's published
    numbers depend on it. Do NOT factor or simplify.
    """
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    """Verbatim alternate metric from upstream constants.py:20 — `max` per sample.

    Used by a subset of RULER tasks (specifically `qa_*` variants). We expose
    both; the runner picks based on the task name (see `metric_for_task`).
    """
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


# Per-task metric dispatch — also verbatim from upstream
# `scripts/eval/synthetic/constants.py` `TASKS` mapping. The 13 RULER synthetic
# tasks split into two scorer families:
#   string_match_all  — VT, NIAH-*, CWE, FWE (most of the suite)
#   string_match_part — qa_1, qa_2 (the SQuAD/HotpotQA wrappers)
# When a new RULER task lands upstream, add it here BEFORE shipping a template
# that selects it. The runner refuses unknown tasks at startup.

TASK_METRICS: dict[str, str] = {
    # Variable tracking
    "vt": "string_match_all",
    # Common-words extraction / frequent-words extraction
    "cwe": "string_match_all",
    "fwe": "string_match_all",
    # NIAH single needle
    "niah_single_1": "string_match_all",
    "niah_single_2": "string_match_all",
    "niah_single_3": "string_match_all",
    # NIAH multi-key
    "niah_multikey_1": "string_match_all",
    "niah_multikey_2": "string_match_all",
    "niah_multikey_3": "string_match_all",
    # NIAH multi-query / multi-value
    "niah_multiquery": "string_match_all",
    "niah_multivalue": "string_match_all",
    # SQuAD / HotpotQA wrappers
    "qa_1": "string_match_part",
    "qa_2": "string_match_part",
}


def metric_for_task(task: str) -> str:
    """Return the upstream scorer name for a RULER task. KeyError on unknown."""
    if task not in TASK_METRICS:
        raise KeyError(
            f"unknown RULER task: {task!r}. Known: {sorted(TASK_METRICS)}. "
            "If upstream added a new task, register it in ruler_helpers.TASK_METRICS."
        )
    return TASK_METRICS[task]


def score_task(task: str, preds: list[str], refs: list[list[str]]) -> float:
    """Dispatch to the correct scorer for a RULER task."""
    name = metric_for_task(task)
    if name == "string_match_all":
        return string_match_all(preds, refs)
    if name == "string_match_part":
        return string_match_part(preds, refs)
    raise ValueError(f"internal: unknown metric {name!r}")


# ── RULER root discovery + prepare.py subprocess ────────────────────────────

def locate_ruler_root() -> Path:
    """Locate the NVIDIA/RULER clone.

    Search order (first hit wins):
      1. $RULER_ROOT
      2. /workspace/RULER       (pod canonical)
      3. /shared/dev/RULER      (solidpc canonical)

    Raises SystemExit with a fix-it message if none is found.
    """
    env = os.environ.get("RULER_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([Path("/workspace/RULER"), Path("/shared/dev/RULER")])
    for c in candidates:
        if (c / "scripts" / "data" / "prepare.py").is_file():
            return c
    raise SystemExit(
        "ruler_native: cannot locate NVIDIA/RULER. Set $RULER_ROOT, or clone to "
        "/workspace/RULER (pod) or /shared/dev/RULER (solidpc). "
        "git clone https://github.com/NVIDIA/RULER"
    )


def ensure_nltk_data() -> None:
    """RULER's prepare.py needs nltk's punkt + punkt_tab corpora for sentence
    tokenization in cwe/fwe/qa. Idempotent — `nltk.download` is a no-op when
    cached. Failure here is a hard stop (we'd silently produce empty samples).
    """
    try:
        import nltk
    except ImportError as e:
        raise SystemExit(f"ruler_native: nltk not installed: {e}") from e
    for corpus in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{corpus}")
        except LookupError:
            print(f"[ruler] nltk.download({corpus})", flush=True)
            nltk.download(corpus, quiet=True)


def run_prepare(*, ruler_root: Path, task: str, max_seq_length: int,
                num_samples: int, tokenizer_path: str, tokenizer_type: str = "hf",
                save_dir: Path, model_template_type: str = "base",
                random_seed: int = 42) -> Path:
    """Subprocess into upstream `scripts/data/prepare.py` to materialize the
    staged validation.jsonl for one (task, max_seq_length, num_samples).

    Returns the path to validation.jsonl. Idempotent — if the file already
    exists and is non-empty, skip the subprocess (saves ~30s/task).

    PATH semantics — upstream prepare.py LITERALLY shells out to
    `python <child_synthetic_script>` via subprocess (see prepare.py:125 — bare
    'python' string, not sys.executable). The child resolves via PATH and runs
    in whatever environment that resolves to. We force the omk env's bin/ onto
    the front of PATH so the child sees tenacity / nltk / transformers /
    AutoTokenizer the same way prepare.py's parent does. WITHOUT this PATH
    override, the child crashes with `ModuleNotFoundError: No module named
    'tenacity'` and prepare.py's parent **still exits 0** — the only signal
    you get is the missing output file, not the child's traceback. We detect
    that downstream by checking out_path existence.

    Output layout — upstream writes to `<save_dir>/<task>/<subset>.jsonl`
    (prepare.py:111). This function returns that full path, so callers should
    pass `save_dir=<stage>/data` (NOT `<stage>/data/<task>`) to match
    upstream's contract.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Upstream writes <save_dir>/<task>/<subset>.jsonl (prepare.py:111)
    out_path = save_dir / task / "validation.jsonl"
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    prepare_py = ruler_root / "scripts" / "data" / "prepare.py"
    cmd = [
        sys.executable, str(prepare_py),
        "--save_dir", str(save_dir),
        "--benchmark", "synthetic",
        "--task", task,
        "--tokenizer_path", tokenizer_path,
        "--tokenizer_type", tokenizer_type,
        "--max_seq_length", str(max_seq_length),
        "--num_samples", str(num_samples),
        "--model_template_type", model_template_type,
        "--random_seed", str(random_seed),
        "--subset", "validation",
    ]
    # Force omk env's bin/ to the front of PATH so the child python resolves
    # to our interpreter (with tenacity / nltk / transformers loaded). Without
    # this, the child uses the system /usr/bin/python which lacks those deps
    # and prepare.py masks the crash with an exit-0 return.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    print(f"[ruler] prepare: {' '.join(cmd[:4])} ... task={task} "
          f"max_seq_length={max_seq_length} num_samples={num_samples}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ruler_root), check=False, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout.decode("utf-8", "replace"))
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(
            f"ruler_native: prepare.py failed (rc={proc.returncode}) for "
            f"task={task} max_seq_length={max_seq_length}. "
            "Common causes: missing nltk punkt/punkt_tab corpus, tokenizer "
            "path wrong, $RULER_ROOT not pointing at a valid clone."
        )
    if not out_path.is_file():
        # prepare.py masks child-process failures with exit 0 — the only
        # diagnostic is the missing output file PLUS prepare.py's own stdout
        # (which contains the child's traceback). Surface BOTH.
        sys.stderr.write(proc.stdout.decode("utf-8", "replace"))
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(
            f"ruler_native: prepare.py exited 0 but {out_path} does not exist. "
            "Upstream prepare.py masks child-process failures with rc=0; look "
            "above for the child's traceback. Most likely cause: a module dep "
            "missing in the child env (omk env's bin/ should be first on PATH; "
            "see the env= override in run_prepare)."
        )
    return out_path


# ── Parser for the staged validation.jsonl ──────────────────────────────────

def load_validation_jsonl(path: Path) -> list[dict]:
    """Read a RULER staged validation.jsonl produced by `prepare.py`.

    Each line is `{"index": int, "input": str, "outputs": list[str], "length": int}`.
    Some tasks add extra fields (e.g. CWE adds `frequent_words`); we keep
    everything as-is and only consume the four canonical fields. Returns rows
    in file order (deterministic, since prepare.py uses --random_seed).
    """
    rows: list[dict] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if not isinstance(rec, dict):
                raise SystemExit(
                    f"{path}: malformed jsonl row (not a dict): {ln[:120]!r}…"
                )
            for k in ("index", "input", "outputs"):
                if k not in rec:
                    raise SystemExit(
                        f"{path}: row missing required field {k!r}: keys={list(rec)}"
                    )
            if not isinstance(rec["outputs"], list):
                raise SystemExit(
                    f"{path}: row outputs is not a list: {type(rec['outputs']).__name__}"
                )
            rows.append(rec)
    return rows


# ── Cache key helper ────────────────────────────────────────────────────────

def make_cache_key(*, task: str, max_seq_length: int, sample_index: int) -> str:
    """Sqlite cache key for one (task, ctx, sample) cell. Stable across resumes."""
    return f"{task}::{max_seq_length}::{sample_index}"


# ── Small utility: pretty-print a per-task summary block ────────────────────

def format_summary(task: str, max_seq_length: int, n: int, score: float,
                   elapsed_secs: float, *, metric_name: str | None = None) -> str:
    metric_name = metric_name or metric_for_task(task)
    return (
        f"=== RULER {task} @ ctx={max_seq_length}: {score:.2f}% "
        f"(n={n}, metric={metric_name}, elapsed={elapsed_secs:.0f}s) ==="
    )


def find_executable(name: str) -> str | None:
    """Best-effort shutil.which fallback for prepare.py child checks."""
    return shutil.which(name)
