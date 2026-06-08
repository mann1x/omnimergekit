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


# Per-task haystack source — verbatim from upstream `scripts/synthetic.yaml`
# (`type_haystack:` for each `niah*` entry; vt/cwe/fwe/qa carry none). Only the
# `essay` haystack needs an EXTERNAL data file — the Paul Graham Essays corpus,
# fetched once by `scripts/data/synthetic/json/download_paulgraham_essay.py`.
# `noise`/`needle` are self-contained string constants inside niah.py, and the
# `qa_*` tasks pull SQuAD/HotpotQA via `download_qa_dataset.sh`. Keep this in
# lock-step with synthetic.yaml when upstream adds a task (the runner refuses
# unknown tasks at startup). This map is what lets the launcher preflight the
# haystack file BEFORE serving, instead of niah.py raising FileNotFoundError
# deep inside prepare.py — which masks the child crash with rc=0 (2026-06-08
# T87 niah_multikey_1 / PaulGrahamEssays.json trap).
TASK_HAYSTACK: dict[str, str] = {
    "niah_single_1":  "noise",
    "niah_single_2":  "essay",
    "niah_single_3":  "essay",
    "niah_multikey_1": "essay",
    "niah_multikey_2": "needle",
    "niah_multikey_3": "needle",
    "niah_multiquery": "essay",
    "niah_multivalue": "essay",
    "vt":  "noise",
    "cwe": "none",   # common-words extraction: generated word lists, no corpus
    "fwe": "none",   # frequent-words extraction: generated word lists, no corpus
    "qa_1": "qa",    # SQuAD dev-v2.0       (download_qa_dataset.sh → squad.json)
    "qa_2": "qa",    # HotpotQA distractor  (download_qa_dataset.sh → hotpotqa.json)
}

# Python modules upstream's synthetic generators import at runtime (niah.py →
# wonderwords/html2text, prepare.py → tenacity, cwe/fwe/qa → nltk). When any is
# missing, prepare.py's child subprocess crashes while prepare.py ITSELF still
# exits 0 — the staged validation.jsonl never appears and the only signal is a
# downstream FileNotFoundError. Preflighting them converts that into a loud,
# actionable launch-time abort.
RULER_RUNTIME_MODULES: tuple[str, ...] = ("tenacity", "nltk", "wonderwords", "html2text")

# Where upstream keeps downloadable corpora, relative to the RULER clone root.
_JSON_SUBDIR = Path("scripts") / "data" / "synthetic" / "json"
_ESSAY_FILE = "PaulGrahamEssays.json"
_QA_FILES = {"qa_1": "squad.json", "qa_2": "hotpotqa.json"}


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


def haystack_for_task(task: str) -> str:
    """Return the haystack source for a RULER task (noise/needle/essay/qa/none).

    KeyError on an unknown task — same contract as `metric_for_task`, so a
    template that selects a task we haven't registered fails loudly instead of
    silently picking a default haystack.
    """
    if task not in TASK_HAYSTACK:
        raise KeyError(
            f"unknown RULER task: {task!r}. Known: {sorted(TASK_HAYSTACK)}. "
            "If upstream added a task, register it in ruler_helpers.TASK_HAYSTACK "
            "(and TASK_METRICS) — copy its `task:` + `type_haystack:` from "
            "scripts/synthetic.yaml."
        )
    return TASK_HAYSTACK[task]


def required_data_file(ruler_root: Path, task: str) -> "tuple[Path | None, str | None]":
    """The external corpus a task needs, plus the command that fetches it.

    Returns ``(path, fix_command)`` for `essay`/`qa_*` tasks, or ``(None, None)``
    for self-contained tasks (noise/needle haystack, or the cwe/fwe word-list
    generators). The path is where upstream's generators look — niah.py loads
    ``json/PaulGrahamEssays.json``; qa.py loads ``json/squad.json`` /
    ``json/hotpotqa.json``.
    """
    hay = haystack_for_task(task)
    json_dir = ruler_root / _JSON_SUBDIR
    if hay == "essay":
        return json_dir / _ESSAY_FILE, (
            f"cd {json_dir} && python download_paulgraham_essay.py   "
            "(needs: pip install beautifulsoup4 html2text tqdm)")
    if hay == "qa":
        fname = _QA_FILES.get(task)
        if fname:
            return json_dir / fname, f"cd {json_dir} && bash download_qa_dataset.sh"
    return None, None


def ruler_native_readiness(task: str) -> "list[str]":
    """Preflight a ruler_native task end-to-end WITHOUT serving a model.

    Returns a list of human-readable problem strings (empty list = ready),
    checking, in order: the RULER clone is present, the synthetic-generator
    python modules import, the nltk punkt corpora are downloaded, and the
    task's external haystack/qa corpus is on disk. `omk_eval` calls this at
    launch and aborts on any problem, so a chain fails fast at the broken bench
    instead of losing the whole run to prepare.py's exit-0-masked child crash or
    a deep FileNotFoundError (the 2026-06-08 T87 PaulGrahamEssays.json trap).
    """
    import importlib.util

    problems: "list[str]" = []

    # 1. RULER clone present (env $RULER_ROOT / pod / solidpc canonical paths).
    try:
        ruler_root = locate_ruler_root()
    except SystemExit as e:
        return [str(e)]

    # 2. Runtime python modules the generators import (prepare.py masks their
    #    absence with rc=0, so importlib.find_spec is the only reliable signal).
    missing = [m for m in RULER_RUNTIME_MODULES
               if importlib.util.find_spec(m) is None]
    if missing:
        problems.append(
            f"missing python modules for RULER synthetic generators: {missing} — "
            f"fix: pip install {' '.join(missing)}")

    # 3. nltk punkt corpora (cwe/fwe/qa sentence tokenization).
    if importlib.util.find_spec("nltk") is not None:
        try:
            import nltk
            for corpus in ("punkt", "punkt_tab"):
                try:
                    nltk.data.find(f"tokenizers/{corpus}")
                except LookupError:
                    problems.append(
                        f"nltk corpus '{corpus}' not downloaded — fix: "
                        f"python -c \"import nltk; nltk.download('{corpus}')\"")
        except Exception as e:  # pragma: no cover - defensive
            problems.append(f"nltk present but unusable: {e}")

    # 4. External haystack/qa corpus for this specific task.
    try:
        data_path, fix = required_data_file(ruler_root, task)
    except KeyError as e:
        return [str(e)]
    if data_path is not None and not (
            data_path.is_file() and data_path.stat().st_size > 0):
        problems.append(
            f"RULER corpus file missing: {data_path} (task '{task}' uses the "
            f"'{haystack_for_task(task)}' haystack) — fix: {fix}")

    return problems


def ensure_haystack_data(ruler_root: Path, task: str) -> None:
    """Runner-side defense-in-depth: hard-stop BEFORE run_prepare when the task's
    external corpus is absent.

    niah.py loads ``json/PaulGrahamEssays.json`` the moment it builds an `essay`
    haystack and raises FileNotFoundError, which prepare.py then masks with
    rc=0 — leaving only a downstream "validation.jsonl does not exist" error.
    Catching it here (called by ruler_runner.py right after ensure_nltk_data)
    surfaces a loud, actionable message even when the runner is invoked directly,
    not via omk_eval's launch-time preflight.
    """
    try:
        data_path, fix = required_data_file(ruler_root, task)
    except KeyError as e:
        raise SystemExit(f"ruler_native: {e}") from e
    if data_path is not None and not (
            data_path.is_file() and data_path.stat().st_size > 0):
        raise SystemExit(
            f"ruler_native: required corpus missing for task '{task}' "
            f"({haystack_for_task(task)} haystack):\n  {data_path}\n  fix: {fix}")


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
