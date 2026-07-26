#!/usr/bin/env python
r"""Keep only the rows of a replay SOURCE pool that render CLEAN through the native
Gemma-4 template. Writes ``<input>.clean.jsonl`` + ``.clean_report.json``.

WHY THIS EXISTS. ``validate_replay_render.py`` is a GATE: it tells you a source is
dirty and (with ``--strict``) refuses to curate it. That is right for a source whose
defects are systematic — a wrong tool-schema shape, a reasoning channel that is
missing everywhere — because those must be fixed in the converter, once, for every
source. It is the wrong tool for an irreducible LONG TAIL of per-row upstream
garbage, where each additional parser special-case buys a handful of rows and adds a
branch that can misfire on data it was never shown.

Measured case that motivated it (2026-07-26): after four converter fixes,
interstellarninja/hermes_reasoning_tool_use renders 51,002/51,004 free of foreign
tokens with 69 rows (0.135%) still carrying a ``{<key>:<|"|>{`` blob, spread across
at least six distinct upstream tool-response envelope shapes
(``{content,name}``, ``{content,name,tool_call_id}``, ``{name,result}``,
``{content,name,status}``, ``{arguments,name,result}``, plus non-JSON bodies). A tier
draws 1.6k-3.5k rows from a 51k pool, so dropping 69 costs nothing and chasing them
costs a branch each.

PRE-FILTER, DON'T TRUNCATE, DON'T PATCH. Rows are dropped whole, never edited, so the
kept set is exactly what the trainer would have rendered. Run this ONCE per pool and
point the mix YAML at the ``.clean.jsonl``; the audit then passes on the real thing
instead of passing on a subset you sampled by luck.

    python filter_render_clean.py pool.jsonl --format hermes --apply

Sibling of ``purge_nonanswering_pool.py`` (which drops rows on CONTENT grounds — a
target that withholds an answer). This one drops on RENDER grounds only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_normalize as rn  # noqa: E402

# Kept byte-identical in intent to validate_replay_render's checks so a pool that
# passes here passes the gate. FOREIGN: a competing chat format's control tokens.
# BLOB: a whole tool payload kept as ONE unstructured string, which the template
# wraps as `{key:<|"|>{...}}` -- the shape that taught v1 to emit `<|"|>` inside its
# own tool-call arguments.
FOREIGN = ["<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>", "<|im_start|>", "<|im_end|>"]
BLOB = re.compile(r"[A-Za-z_][A-Za-z0-9_]*:<\|\"\|>[\[{]")


def verdict(row: dict, fmt: str, tok) -> str | None:
    """Return a defect label, or None when the row renders clean."""
    try:
        msgs, tools = rn.normalize(row, fmt)
    except Exception as e:  # noqa: BLE001 -- an unconvertible row is a dropped row
        return f"convert_error:{type(e).__name__}"
    if not msgs:
        return "no_messages"
    try:
        txt = tok.apply_chat_template(msgs, tools=tools, tokenize=False,
                                      preserve_thinking=True)
    except Exception as e:  # noqa: BLE001 -- e.g. a tool schema the template rejects
        return f"render_error:{type(e).__name__}"
    hits = [f for f in FOREIGN if f in txt]
    if hits:
        return "foreign:" + ",".join(hits)
    if BLOB.search(txt):
        return "blob"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--format", default="hermes",
                    help="replay_normalize format id (hermes / messages / ...)")
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--apply", action="store_true", help="write the .clean.jsonl")
    ap.add_argument("--max-drop-frac", type=float, default=0.05,
                    help="refuse to write if the drop rate exceeds this — a high rate "
                         "means a SYSTEMATIC defect that belongs in the converter, "
                         "not a long tail to be filtered away")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    rc = 0
    for fp in a.inputs:
        rows = [json.loads(x) for x in open(fp) if x.strip()]
        keep, stats = [], {}
        for r in rows:
            v = verdict(r, a.format, tok)
            if v is None:
                keep.append(r)
            else:
                stats[v] = stats.get(v, 0) + 1
        n, k = len(rows), len(keep)
        frac = (n - k) / max(1, n)
        print(f"\n{fp}\n    rows={n}  clean={k}  dropped={n - k} ({frac:.3%})")
        for key in sorted(stats, key=lambda x: -stats[x]):
            print(f"      {key:44s} {stats[key]:6d}")
        if frac > a.max_drop_frac:
            print(f"    REFUSING: drop rate {frac:.2%} > {a.max_drop_frac:.2%}. That is a "
                  f"SYSTEMATIC defect — fix replay_normalize, do not filter it away.",
                  file=sys.stderr)
            rc = 1
            continue
        if a.apply:
            op = fp.rsplit(".jsonl", 1)[0] + ".clean.jsonl"
            with open(op, "w") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(fp.rsplit(".jsonl", 1)[0] + ".clean_report.json", "w") as fh:
                json.dump({"input": fp, "format": a.format, "model": a.model,
                           "rows": n, "kept": k, "dropped": n - k,
                           "drop_frac": frac, "by_defect": stats}, fh, indent=2)
            print(f"    -> wrote {op}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
