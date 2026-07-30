#!/usr/bin/env python
r"""Query the `thinking` table next to the cached answers in an lm-eval sqlite db.

The db written by `--use_cache` holds two things once
eval/patches/patch_lmeval_thinking_table.py is installed:

  unnamed(key, value)   -- lm-eval's response cache; value is a PICKLED answer str,
                           keyed by a hash of the REQUEST (not recoverable here)
  thinking(...)         -- one row per response, keyed by sha256(content)

sha256 of the unpickled answer is the bridge. This tool does that join and
reports the two channels together, which is the thing that was impossible before:
telling "ruminated until the budget was gone" apart from "wrote a long answer".

Subcommands:
  stats    per-channel length distributions + how many responses hit the wall
  rows     per-response table (answer tokens, thinking tokens, degeneracy flags)
  dump     full thinking text for the worst offenders
  --emit-jsonl PATH  rewrite the legacy reasoning_log.jsonl from the table, for
                     scripts/analyze_track_results.py which still reads it

Examples:
  query_thinking.py stats  eval/.../sqlite_cache/aime_100_v24_rank0.db
  query_thinking.py rows   <db> --max-gen-toks 16384 --tokenizer out/e2b-an-v24/merged
  query_thinking.py dump   <db> --n 3
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sqlite3
import sys

MARKERS = ["wait", "alternatively", "let me reconsider", "hmm", "actually,",
           "but wait", "no,", "let me re", "rethink", "on second thought", "let's re"]
_WS = re.compile(r"\s+")


def sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "replace")).hexdigest()


def has_thinking(cx) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name='thinking'"
    return bool(list(cx.execute(q)))


def load(db: str):
    """Return [{content, reasoning, content_chars, reasoning_chars, cached}] joined."""
    cx = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    if not has_thinking(cx):
        sys.exit(f"FATAL: no `thinking` table in {db}.\n"
                 "       This run predates patch_lmeval_thinking_table.py, or the\n"
                 "       patch was not installed in the env that produced it. The\n"
                 "       thinking text is NOT recoverable from this file.")
    # answers actually kept by lm-eval (the ones a score was computed from)
    cached = {}
    try:
        for _k, v in cx.execute("SELECT key, value FROM unnamed"):
            try:
                cached[sha(pickle.loads(v))] = True
            except Exception:
                continue
    except sqlite3.OperationalError:
        pass  # no `unnamed` yet (thinking-only db)
    rows = []
    for s, ci, cc, rc, r, c, ts in cx.execute(
            "SELECT content_sha, choice_idx, content_chars, reasoning_chars,"
            " reasoning, content, ts FROM thinking ORDER BY ts"):
        rows.append(dict(sha=s, choice_idx=ci, content_chars=cc, reasoning_chars=rc,
                         reasoning=r or "", content=c or "", ts=ts,
                         in_cache=s in cached))
    cx.close()
    return rows


def degen(t: str):
    """(max_line_repeat, dup_shingle_frac, rumination_hits) — LOOP vs RUMINATION."""
    lines = [_WS.sub(" ", x).strip().lower() for x in t.split("\n")]
    lines = [x for x in lines if len(x) > 20]
    mlr = max(collections.Counter(lines).values()) if lines else 0
    w = t.split()
    dup = 0.0
    if len(w) > 12:
        sh = [" ".join(w[i:i + 12]) for i in range(len(w) - 12)]
        c = collections.Counter(sh)
        dup = sum(v - 1 for v in c.values() if v > 1) / len(sh)
    low = t.lower()
    return mlr, dup, sum(low.count(m) for m in MARKERS)


def q(v, p):
    return v[min(len(v) - 1, int(len(v) * p))] if v else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "rows", "dump"])
    ap.add_argument("db")
    ap.add_argument("--tokenizer", default=None,
                    help="HF dir for exact token counts; omit to report chars")
    ap.add_argument("--max-gen-toks", type=int, default=0,
                    help="generation ceiling, to count responses at the wall")
    ap.add_argument("--n", type=int, default=5, help="dump: how many to print")
    ap.add_argument("--emit-jsonl", default=None,
                    help="also rewrite the legacy reasoning_log.jsonl here")
    a = ap.parse_args()

    rows = load(a.db)
    print(f"{len(rows)} thinking rows; {sum(r['in_cache'] for r in rows)} join to a "
          f"cached answer")
    if not rows:
        return 0

    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    def ntok(s, chars):
        if tok is None:
            return chars
        return len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0

    for r in rows:
        r["a_tok"] = ntok(r["content"], r["content_chars"])
        r["t_tok"] = ntok(r["reasoning"], r["reasoning_chars"])
        r["a_deg"] = degen(r["content"])
        r["t_deg"] = degen(r["reasoning"])

    if a.emit_jsonl:
        with open(a.emit_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps({"idx": r["choice_idx"],
                                    "content_chars": r["content_chars"],
                                    "reasoning_chars": r["reasoning_chars"]}) + "\n")
        print(f"wrote legacy reasoning_log -> {a.emit_jsonl}")

    unit = "tok" if tok else "chars"
    if a.cmd == "stats":
        for label, key, dk in (("ANSWER  ", "a_tok", "a_deg"),
                               ("THINKING", "t_tok", "t_deg")):
            v = sorted(r[key] for r in rows)
            ml = sorted(r[dk][0] for r in rows)
            du = sorted(r[dk][1] for r in rows)
            ru = sorted(r[dk][2] for r in rows)
            print(f"  {label} {unit} p50={q(v, .5):6d} p90={q(v, .9):6d} max={v[-1]:6d}"
                  f" | max_line_rep p50={q(ml, .5):3d} max={ml[-1]:4d}"
                  f" | dup p50={q(du, .5):.3f} max={du[-1]:.3f}"
                  f" | rum p50={q(ru, .5):3d} max={ru[-1]:4d}")
        if a.max_gen_toks and tok:
            wall = [r for r in rows
                    if r["a_tok"] + r["t_tok"] >= a.max_gen_toks * 0.97]
            print(f"  at the {a.max_gen_toks}-token wall: {len(wall)}/{len(rows)}")
            if wall:
                mt = sum(r["t_tok"] for r in wall) / len(wall)
                ma = sum(r["a_tok"] for r in wall) / len(wall)
                lt = sum(1 for r in wall if r["t_deg"][0] >= 4 or r["t_deg"][1] >= .30)
                la = sum(1 for r in wall if r["a_deg"][0] >= 4 or r["a_deg"][1] >= .30)
                print(f"    mean split  thinking={mt:.0f}  answer={ma:.0f}")
                print(f"    LOOP in THINKING {lt}/{len(wall)}   in ANSWER {la}/{len(wall)}"
                      "   <- which channel actually degenerated")
    elif a.cmd == "rows":
        print(f"  {'ans_' + unit:>9} {'think_' + unit:>10} {'a_lrep':>7} {'a_dup':>6}"
              f" {'t_lrep':>7} {'t_dup':>6} {'t_rum':>6}  cached")
        for r in sorted(rows, key=lambda x: -(x["a_tok"] + x["t_tok"])):
            print(f"  {r['a_tok']:9d} {r['t_tok']:10d} {r['a_deg'][0]:7d}"
                  f" {r['a_deg'][1]:6.3f} {r['t_deg'][0]:7d} {r['t_deg'][1]:6.3f}"
                  f" {r['t_deg'][2]:6d}  {'y' if r['in_cache'] else '-'}")
    else:  # dump
        worst = sorted(rows, key=lambda x: -(x["t_deg"][0] + x["t_deg"][2]))[:a.n]
        for i, r in enumerate(worst, 1):
            print(f"\n{'=' * 78}\n#{i}  thinking={r['t_tok']}{unit} "
                  f"answer={r['a_tok']}{unit}  t_lrep={r['t_deg'][0]} "
                  f"t_rum={r['t_deg'][2]}\n{'-' * 78}")
            print("--- THINKING ---")
            print(r["reasoning"][:4000])
            print("--- ANSWER (tail) ---")
            print(r["content"][-800:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
