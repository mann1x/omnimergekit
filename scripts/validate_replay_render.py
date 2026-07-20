#!/usr/bin/env python
"""Render each mix source through the NATIVE Gemma-4 template — audit, measure, curate.

All three jobs read the SAME mix YAML the trainer uses, so they audit exactly what
will be trained:

  1. FORMAT audit (always): the rendered target contains NO foreign surface tokens
     (``<think>`` / ``<tool_call>`` / ``<tool_response>`` XML), DOES contain the
     native family where expected (``<|channel>`` reasoning, ``<|tool_call>`` calls),
     and carries NO ``{value:<|"|>{`` stringified-JSON blob — the pattern that broke
     v1 tool-calling (model learned to emit ``<|"|>`` inside its own call args).
     Fast: ``--n`` rows/source, streaming.

  2. LENGTH coverage (``--full`` or ``--max-seq-len``): tokenize every row and print
     the max_seq_len coverage table (rows over each threshold + % covered). This is
     what picks max_seq_len — sized so the over-length purge is tiny.

  3. CURATE (``--out-dir``): write a deterministic, length-PURGED (<= --max-seq-len),
     per-source-``max_samples``-sampled, PRE-RENDERED jsonl (``{"text": ...}``) per
     source. The trainer then consumes it as a ``format: prerendered`` data_files
     source — no shuffle+``range`` at load, no re-render, over-length rows already
     dropped (never truncated).

    python validate_replay_render.py --config mix_v1.yaml                       # audit
    python validate_replay_render.py --config mix_v1.yaml --full --max-seq-len 65536
    python validate_replay_render.py --config mix_v1.yaml --max-seq-len 65536 \
        --out-dir /srv/ml/an-finetune/curated
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

import replay_normalize as rn

FOREIGN = ["<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>", "<|im_start|>", "<|im_end|>"]
NATIVE = ["<|channel>", "<channel|>", "<|think|>", "<|tool_call>",
          "<tool_call|>", "<|tool_response>", "<tool_response|>"]
# The v1 tool-call breakage: a WHOLE tool response kept as one unstructured string
# and wrapped by the template in a synthetic `{value:<|"|>...<|"|>}` envelope, whose
# content opens a JSON object/array (`{value:<|"|>{` or `[`). _coerce_response now
# structures such strings; this catches a regression. Kept NARROW on the `value:`
# envelope on purpose: broadening to any key false-positives on legitimate
# structured responses that merely CARRY JSON text in a leaf — e.g. a terminal
# command whose `output:<|"|>{...json...}<|"|>` is real (properly structured,
# model reads it, never emits it). Measured 257 such benign leaves in curated
# hermes; the narrow `value:`-envelope form has zero.
BLOB = re.compile(r'\{value:<\|"\|>\s*[\[{]')
# Corrupt tool-call arg KEYS — the `""task_id""` class. A native call renders
# `call:name{"k": "v"}`, i.e. the args object opens `{"` (one quote, then a key
# char). A double-quoted key (`call:name{""k""`) or backslash-escaped key
# (`call:name{\"k\"`) at the args-OPEN is the corruption. Anchored on `call:NAME{`
# so `{"`/`{\"` INSIDE argument VALUES (a write_file whose `content` is a JSON/code
# file) never match; the `""` lookahead excludes a benign empty-key `{"":...}`.
DQ_KEY = re.compile(r'call:[A-Za-z0-9_]+\{\s*(?:""(?=[A-Za-z_])|\\")')
THRESHOLDS = [8192, 16384, 24576, 32768, 49152, 65536, 98304]


def unwrap_tok(tokenizer):
    """Return the TEXT tokenizer: a Gemma-4 processor's __call__ treats the 1st
    positional arg as *images*, so processor(text) would image-decode + crash."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def src_label(src: dict) -> str:
    return src.get("id") or (src["data_files"] if isinstance(src.get("data_files"), str)
                             else (src.get("data_files") or ["?"])[0])


def full_dataset(src: dict, remap: tuple[str, str] | None):
    """Non-streaming load of an ENTIRE source (Hub id or local data_files)."""
    if src.get("data_files"):
        files = src["data_files"]
        files = [files] if isinstance(files, str) else list(files)
        if remap:
            files = [f.replace(remap[0], remap[1]) for f in files]
        return load_dataset("json", data_files=files, split=src.get("split", "train"))
    return load_dataset(src["id"], src.get("config"), split=src.get("split", "train"))


def stream_rows(src: dict, n: int, remap: tuple[str, str] | None):
    """First ``n`` rows (streaming for Hub) — for the quick format audit."""
    if src.get("data_files"):
        files = src["data_files"]
        files = [files] if isinstance(files, str) else list(files)
        if remap:
            files = [f.replace(remap[0], remap[1]) for f in files]
        ds = load_dataset("json", data_files=files, split=src.get("split", "train"))
        return [ds[i] for i in range(min(n, len(ds)))]
    ds = load_dataset(src["id"], src.get("config"),
                      split=src.get("split", "train"), streaming=True)
    out, it = [], iter(ds)
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


def render_text(ex, fmt, tokenizer):
    if fmt == "prerendered":
        return ex.get("text")
    msgs, tools = rn.normalize(ex, fmt)
    if not msgs:
        return None
    return tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=False,
        enable_thinking=False, preserve_thinking=True)


def audit_source(src, tokenizer, n, show, remap):
    """Quick FORMAT gate on the first ``n`` rows. Returns True if clean."""
    label, fmt = src_label(src), src["format"]
    rows = stream_rows(src, n, remap)
    n_render = n_ok = n_reason = n_tools = n_blob = n_dq = 0
    foreign_hits: dict[str, int] = {}
    first_render = None
    dq_example = None
    for ex in rows:
        txt = render_text(ex, fmt, tokenizer)
        if not txt:
            continue
        n_render += 1
        first_render = first_render or txt
        fh = [t for t in FOREIGN if t in txt]
        for t in fh:
            foreign_hits[t] = foreign_hits.get(t, 0) + 1
        if not fh:
            n_ok += 1
        n_reason += "<|channel>" in txt
        n_tools += "<|tool_call>" in txt
        n_blob += bool(BLOB.search(txt))
        # double-quoted / escaped tool-call arg keys (the tool-call corruption class)
        dq_m = DQ_KEY.search(txt)
        if dq_m:
            n_dq += 1
            dq_example = dq_example or txt[dq_m.start():dq_m.start() + 120]
    clean = bool(n_render) and not foreign_hits and not n_blob and not n_dq
    status = "OK" if clean else "FAIL"
    print(f"\n[{status}] {label}  fmt={fmt}  rendered={n_render}/{len(rows)}  "
          f"clean={n_ok}  native_reason={n_reason}  native_toolcall={n_tools}  "
          f"blob={n_blob}  dq_key={n_dq}", flush=True)
    if foreign_hits:
        print(f"     FOREIGN TOKENS LEAKED: {foreign_hits}", flush=True)
    if n_blob:
        print(f"     BLOB LEAK: {n_blob} rows carry a {{<key>:<|\"|>{{ object blob", flush=True)
    if n_dq:
        print(f"     DQ-KEY LEAK: {n_dq} rows carry double-quoted/escaped tool-call keys "
              f"(e.g. {dq_example!r})", flush=True)
    if show and first_render:
        print("     --- first rendered target (truncated 1200c) ---")
        print("     " + first_render[:1200].replace("\n", "\n     "))
    return clean


def length_pass(src, tokenizer, num_proc, remap, max_seq_len):
    """Tokenize the ENTIRE source; print distribution + coverage table + purge count."""
    label, fmt = src_label(src), src["format"]
    ds = full_dataset(src, remap)
    count_tok = unwrap_tok(tokenizer)

    def fn(ex):
        txt = render_text(ex, fmt, tokenizer)
        return {"_ntok": len(count_tok(txt, add_special_tokens=False)["input_ids"]) if txt else 0}

    ds = ds.map(fn, num_proc=num_proc, desc=f"toklen {label}")
    lens = sorted(t for t in ds["_ntok"] if t > 0)
    n = len(lens)
    if not n:
        print(f"\n[LEN] {label}: no usable rows", flush=True)
        return
    def q(x):
        return lens[min(int(x * n), n - 1)]

    def over(th):
        return sum(1 for t in lens if t > th)

    print(f"\n[LEN] {label}  fmt={fmt}  usable={n}", flush=True)
    print(f"      tokens: p50={q(.5)} p90={q(.9)} p95={q(.95)} p99={q(.99)} "
          f"p99.9={q(.999)} max={lens[-1]}", flush=True)
    for th in THRESHOLDS:
        o = over(th)
        print(f"      rows > {th:6d}: {o:6d} ({100*o/n:5.2f}%)  "
              f"-> covered if max_seq_len={th}: {100*(n-o)/n:5.2f}%", flush=True)
    if max_seq_len:
        o = over(max_seq_len)
        print(f"      => at max_seq_len={max_seq_len}: PURGE {o} ({100*o/n:.2f}%), "
              f"retain {n-o} rows all <= {max_seq_len} tok", flush=True)


def curate_source(src, tokenizer, num_proc, remap, max_seq_len, seed, out_dir):
    """Write a deterministic, sampled + length-purged, PRE-RENDERED jsonl for one source."""
    label, fmt = src_label(src), src["format"]
    ds = full_dataset(src, remap)
    nmax = src.get("max_samples")
    if nmax and nmax < len(ds):
        ds = ds.shuffle(seed=seed).select(range(nmax))  # match trainer's select-then-purge
    count_tok = unwrap_tok(tokenizer)

    def fn(ex):
        txt = render_text(ex, fmt, tokenizer)
        ntok = len(count_tok(txt, add_special_tokens=False)["input_ids"]) if txt else 0
        return {"text": txt, "_ntok": ntok}

    ds = ds.map(fn, num_proc=num_proc, remove_columns=ds.column_names, desc=f"render {label}")
    before = len(ds)
    ds = ds.filter(lambda r: bool(r["text"]) and len(r["text"]) > 20
                   and 0 < r["_ntok"] <= max_seq_len)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    if src.get("config"):
        slug += f"_{src['config']}"
    out_path = Path(out_dir) / f"{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in ds:
            fh.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"\n[CURATE] {label} -> {out_path}", flush=True)
    print(f"         wrote {len(ds)} rows (from {before} sampled; purged "
          f"{before-len(ds)} empty/over-{max_seq_len}). Train via: "
          f"{{data_files: [{out_path}], format: prerendered}}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="mix YAML (same one the trainer reads)")
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=4, help="rows/source for the format audit")
    ap.add_argument("--show", type=int, default=1, help="print full render for first K rows/source")
    ap.add_argument("--map", help="remap local data_files prefix FROM=TO")
    ap.add_argument("--full", action="store_true", help="tokenize every row: length coverage table")
    ap.add_argument("--max-seq-len", type=int, default=0,
                    help="report the over-length purge at this seq len (implies --full)")
    ap.add_argument("--out-dir", help="materialize curated prerendered jsonl per source "
                    "(requires --max-seq-len)")
    ap.add_argument("--seed", type=int, default=3407, help="sampling seed (match trainer)")
    ap.add_argument("--num-proc", type=int, default=12)
    args = ap.parse_args()
    remap = tuple(args.map.split("=", 1)) if args.map else None
    if args.out_dir and not args.max_seq_len:
        ap.error("--out-dir requires --max-seq-len")

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f">>> native chat_template len: {len(tok.chat_template or '')}", flush=True)
    cfg = yaml.safe_load(open(args.config))

    all_ok = True
    for src in cfg["sources"]:
        all_ok &= audit_source(src, tok, args.n, args.show, remap)
        if args.full or args.max_seq_len:
            length_pass(src, tok, args.num_proc, remap, args.max_seq_len)
        if args.out_dir:
            curate_source(src, tok, args.num_proc, remap, args.max_seq_len,
                          args.seed, args.out_dir)

    print("\n>>> " + ("ALL SOURCES CLEAN" if all_ok else "FORMAT LEAK DETECTED"), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
