#!/usr/bin/env python3
"""Load + validate an eval template (YAML), resolve to a flat dict, and
emit shell-friendly env vars or a JSON blob the eval suite can consume.

A template is either:
  - Deterministic indices (`selection.type: indices`) — bit-exactly
    reproducible across dataset re-shuffles.
  - Criteria filter (`selection.type: filter`) — used for LCB-medium
    where the dataset's own metadata defines a stable subset
    (difficulty, min_date, testtype).

Usage:
  template_loader.py <name|path> [--emit env|json|python]
    --emit env     KEY=VAL lines (default; for `eval $(...)` in bash)
    --emit json    full resolved dict
    --emit python  short repr for debugging

The loader looks for `<name>.yaml` in `<repo>/eval/templates/` first; if
absent, treats the arg as a path and tries to read it directly.

Validation refuses templates that:
  - omit any required field
  - declare `n` that disagrees with `len(indices)`
  - use an unknown backend or scorer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML not installed. `pip install pyyaml`", file=sys.stderr)
    sys.exit(2)


REPO_TEMPLATES_DIR = Path(__file__).resolve().parent
KNOWN_BACKENDS = {"lm-eval", "lcb_custom", "multipl_e"}
# Selection types:
#   indices  — list[int] honored by the runner (or baked into a shadow task
#              via process_docs for lm-eval which doesn't honor --limit indices)
#   filter   — runner-side filter (currently lcb_custom: difficulty/min_date)
#   explicit — lcb_custom only: pinned list of task_ids passed to the shim via
#              --task-ids. Used by curated smoke subsets.
#   langs    — multipl_e only: list[str] of MultiPL-E languages; n is the
#              per-language problem count (first n of each split).
KNOWN_SELECTION_TYPES = {"indices", "filter", "explicit", "langs"}
REQUIRED_TOP_LEVEL = {"name", "backend", "task", "n", "selection", "generation", "scoring", "cache"}


def resolve_path(arg: str) -> Path:
    """Resolve a template arg to a path. Bundled-name first, then literal path."""
    if arg.endswith(".yaml") or arg.endswith(".yml") or "/" in arg:
        p = Path(arg)
        if not p.exists():
            raise SystemExit(f"template not found at path: {p}")
        return p
    # Bundled name (no extension, no slash) → <REPO>/<name>.yaml
    p = REPO_TEMPLATES_DIR / f"{arg}.yaml"
    if not p.exists():
        raise SystemExit(
            f"template '{arg}' not found at {p}. "
            f"Available: {sorted(t.stem for t in REPO_TEMPLATES_DIR.glob('*.yaml'))}"
        )
    return p


def validate(t: dict[str, Any], path: Path) -> None:
    missing = REQUIRED_TOP_LEVEL - set(t.keys())
    if missing:
        raise SystemExit(f"{path}: missing required keys: {sorted(missing)}")
    if t["backend"] not in KNOWN_BACKENDS:
        raise SystemExit(
            f"{path}: backend={t['backend']!r} not in {sorted(KNOWN_BACKENDS)}"
        )
    sel = t["selection"]
    if sel.get("type") not in KNOWN_SELECTION_TYPES:
        raise SystemExit(
            f"{path}: selection.type must be one of {sorted(KNOWN_SELECTION_TYPES)}"
        )
    if sel["type"] == "indices":
        idx = sel.get("indices")
        if not isinstance(idx, list) or not all(isinstance(i, int) for i in idx):
            raise SystemExit(f"{path}: selection.indices must be a list[int]")
        if t["n"] != len(idx):
            raise SystemExit(
                f"{path}: n={t['n']} disagrees with len(indices)={len(idx)}"
            )
        if len(set(idx)) != len(idx):
            raise SystemExit(f"{path}: selection.indices contains duplicates")
    elif sel["type"] == "filter":
        # filter cardinality is validated at run time against the dataset,
        # not here — the template just declares the expected n.
        if t["backend"] == "lcb_custom":
            for k in ("difficulty", "min_date", "testtype"):
                if k not in sel:
                    raise SystemExit(f"{path}: lcb_custom filter missing {k!r}")
    elif sel["type"] == "explicit":
        # Currently only lcb_custom supports this — explicit task_id list
        # passed to the LCB shim via --task-ids.
        if t["backend"] != "lcb_custom":
            raise SystemExit(
                f"{path}: selection.type=explicit only supported for "
                f"backend=lcb_custom (got {t['backend']!r})"
            )
        tids = sel.get("task_ids")
        if not isinstance(tids, list) or not all(isinstance(x, str) for x in tids):
            raise SystemExit(f"{path}: selection.task_ids must be a list[str]")
        if t["n"] != len(tids):
            raise SystemExit(
                f"{path}: n={t['n']} disagrees with len(task_ids)={len(tids)}"
            )
        if len(set(tids)) != len(tids):
            raise SystemExit(f"{path}: selection.task_ids contains duplicates")
        for k in ("difficulty", "min_date", "testtype"):
            if k not in sel:
                raise SystemExit(f"{path}: lcb_custom explicit missing {k!r}")
    elif sel["type"] == "langs":
        # MultiPL-E: a list of languages; n is the per-language problem count.
        if t["backend"] != "multipl_e":
            raise SystemExit(
                f"{path}: selection.type=langs only supported for "
                f"backend=multipl_e (got {t['backend']!r})"
            )
        langs = sel.get("langs")
        if not isinstance(langs, list) or not langs or not all(
            isinstance(x, str) for x in langs
        ):
            raise SystemExit(f"{path}: selection.langs must be a non-empty list[str]")
        if not isinstance(t["n"], int) or t["n"] <= 0:
            raise SystemExit(f"{path}: n must be a positive int (per-language count)")


def load(arg: str) -> dict[str, Any]:
    path = resolve_path(arg)
    with path.open() as f:
        t = yaml.safe_load(f)
    if not isinstance(t, dict):
        raise SystemExit(f"{path}: top-level YAML must be a mapping")
    t["__source__"] = str(path)
    validate(t, path)
    return t


def emit_env(t: dict[str, Any]) -> str:
    """Flatten the template for shell consumption (eval $(...))."""
    out = []
    out.append(f"TEMPLATE_NAME={t['name']}")
    out.append(f"TEMPLATE_BACKEND={t['backend']}")
    out.append(f"TEMPLATE_TASK={t['task']}")
    out.append(f"TEMPLATE_N={t['n']}")
    out.append(f"TEMPLATE_SELECTION_TYPE={t['selection']['type']}")
    if t["selection"]["type"] == "indices":
        # Indices file written next to the template so we don't blast envs.
        idx_str = ",".join(str(i) for i in t["selection"]["indices"])
        out.append(f"TEMPLATE_INDICES='{idx_str}'")
    else:
        sel = t["selection"]
        out.append(f"TEMPLATE_LCB_DIFFICULTY={sel.get('difficulty','')}")
        out.append(f"TEMPLATE_LCB_MIN_DATE={sel.get('min_date','')}")
        out.append(f"TEMPLATE_LCB_TESTTYPE={sel.get('testtype','')}")
    g = t["generation"]
    out.append(f"TEMPLATE_MAX_GEN_TOKS={g.get('max_gen_toks', 2048)}")
    out.append(f"TEMPLATE_TEMPERATURE={g.get('temperature', 0.0)}")
    out.append(f"TEMPLATE_TOP_P={g.get('top_p', 1.0)}")
    out.append(f"TEMPLATE_TOP_K={g.get('top_k', 0)}")
    out.append(f"TEMPLATE_DO_SAMPLE={'1' if g.get('do_sample', False) else '0'}")
    s = t["scoring"]
    out.append(f"TEMPLATE_METRIC={s.get('metric', '')}")
    out.append(f"TEMPLATE_FILTER={s.get('filter', '')}")
    out.append(f"TEMPLATE_SCORER={s.get('scorer', '')}")
    ba = t.get("backend_args", {})
    out.append(f"TEMPLATE_APPLY_CHAT={'1' if ba.get('apply_chat_template', False) else '0'}")
    out.append(f"TEMPLATE_NUM_FEWSHOT={ba.get('num_fewshot', 0)}")
    out.append(f"TEMPLATE_BATCH_SIZE={ba.get('batch_size', 1)}")
    out.append(f"TEMPLATE_SQLITE_PREFIX={t['cache'].get('sqlite_prefix', t['name'])}")
    out.append(f"TEMPLATE_REPORT_TOKEN_STATS={'1' if t.get('reports',{}).get('token_stats', True) else '0'}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="template name (bundled) or path to .yaml")
    ap.add_argument("--emit", choices=("env", "json", "python"), default="env")
    args = ap.parse_args()
    t = load(args.template)
    if args.emit == "env":
        print(emit_env(t))
    elif args.emit == "json":
        print(json.dumps(t, indent=2, sort_keys=True))
    else:
        print(repr(t))


if __name__ == "__main__":
    main()
