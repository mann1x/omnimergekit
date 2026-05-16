#!/usr/bin/env python3
"""build_diff_corpus.py — assemble a task-specific calibration corpus for
the T18 differential router calibration (see router_eac_calibrate.py
--corpus-file).

Reads samples_*.jsonl from eval_results_vllm_suite/128e/<bench>/... for
the benches with clean schema (HumanEval, gsm8k, ifeval, math500). For each
sample:
  filter (a): correct  →  pass@1==1.0 or exact_match==1.0
  filter (b): NOT ruminative → response length in bottom-N% per bench
                              (configurable; default 70%)

For each surviving (prompt, response) pair, applies the chat template
of `google/gemma-4-26B-A4B-it` to produce a chat-formatted text string,
then concatenates with a `<corpus_sep>` newline separator into one
output text file. router_eac_calibrate.py --corpus-file consumes it as
plain text (re-tokenized by the same tokenizer at capture time).

The rumination filter is the critical feature: 128e can ALSO ruminate
on hard problems and still arrive at the correct answer. Those
trajectories are noisy router signals. Filtering to the shortest-by-bench
keeps only the trajectories where 128e routed STRAIGHT to the answer —
"this is what 'correct routing' looks like on this task".

Schemas handled:
  humaneval_full       — `doc.prompt` (function sig+docstring), `pass@1`
  gsm8k / gsm8k_100    — `doc.question`, `exact_match`
  ifeval / ifeval_100  — `doc.prompt`, `prompt_level_strict_acc`
  math500 / math500_100 — `doc.problem`, `exact_match`

Usage:
  python scripts/build_diff_corpus.py \\
      --root eval_results_vllm_suite/128e \\
      --tokenizer google/gemma-4-26B-A4B-it \\
      --out logs/diff_corpus.txt \\
      --rumination-percentile 70 \\
      --max-examples 0

Output: one text file + sidecar JSON with per-bench stats.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


# ─── per-bench adapters: (bench_dir_glob, schema_kind) ───────────────────────

BENCH_CONFIG = {
    # bench_dir_pattern : (correctness_field, doc_prompt_field_priority)
    # correctness_field: name of the metric key that is 1.0 when correct
    # doc_prompt_field_priority: ordered list of doc.* keys to try as prompt
    "humaneval_full":     ("pass@1",                  ["prompt"]),
    "humanevalplus_full": ("pass@1",                  ["prompt"]),
    "gsm8k_100":          ("exact_match",             ["question"]),
    "gsm8k":              ("exact_match",             ["question"]),
    "math500_100":        ("exact_match",             ["problem", "question"]),
    "math500":            ("exact_match",             ["problem", "question"]),
    "ifeval_100":         ("prompt_level_strict_acc", ["prompt"]),
    "ifeval_full":        ("prompt_level_strict_acc", ["prompt"]),
}


def iter_samples(samples_path: Path) -> Iterator[dict[str, Any]]:
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def latest_samples_file(bench_dir: Path) -> Path | None:
    """Find the most-recent samples_*.jsonl under bench_dir (recursive)."""
    candidates = sorted(bench_dir.rglob("samples_*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def extract_prompt(sample: dict, doc_priority: list[str]) -> str | None:
    doc = sample.get("doc", {}) or {}
    for field in doc_priority:
        v = doc.get(field)
        if v:
            return str(v)
    # Fall back to args.gen_args_0.arg_0 if available (often empty under
    # apply_chat_template=true, but try anyway)
    arg0 = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
    return arg0 or None


def extract_response(sample: dict) -> str | None:
    rsp = sample.get("resps", [[""]])
    if not rsp or not rsp[0]:
        return None
    return rsp[0][0] or None


def extract_correctness(sample: dict, field: str) -> float | None:
    v = sample.get(field)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="eval_results_vllm_suite/128e",
                    help="Root dir holding per-bench eval result folders")
    ap.add_argument("--tokenizer", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", required=True,
                    help="Output text file (single concatenated corpus)")
    ap.add_argument("--rumination-percentile", type=int, default=70,
                    help="Keep responses with length ≤ Nth percentile per bench")
    ap.add_argument("--min-response-chars", type=int, default=50,
                    help="Drop responses shorter than this (likely silent-empty)")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="Optional cap total examples (0 = no cap)")
    ap.add_argument("--sep", default="<|im_end|>\n\n",
                    help="Separator between examples in the output text")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"FAIL: {root} does not exist")
        return 1

    # Lazy tokenizer load (only if we have anything to format)
    from transformers import AutoTokenizer
    print(f"loading tokenizer {args.tokenizer}...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")
    stats = defaultdict(lambda: dict(total=0, correct=0, kept=0,
                                     len_p50=0, len_p70=0, len_p90=0,
                                     tokens=0))
    grand_total_examples = 0
    grand_total_tokens = 0

    for bench_dir_name, (corr_field, prompt_priority) in BENCH_CONFIG.items():
        bench_dir = root / bench_dir_name
        if not bench_dir.exists():
            print(f"  {bench_dir_name}: dir not present, skip")
            continue
        sf = latest_samples_file(bench_dir)
        if sf is None:
            print(f"  {bench_dir_name}: no samples_*.jsonl, skip")
            continue
        print(f"  {bench_dir_name}: reading {sf.name}")

        # First pass: collect lengths of correct responses
        correct: list[dict] = []
        for s in iter_samples(sf):
            c = extract_correctness(s, corr_field)
            if c is None or c < 1.0:
                continue
            rsp = extract_response(s)
            if not rsp or len(rsp) < args.min_response_chars:
                continue
            prompt = extract_prompt(s, prompt_priority)
            if not prompt:
                continue
            correct.append({"prompt": prompt, "response": rsp,
                            "len": len(rsp)})
        stats[bench_dir_name]["total"] = sum(
            1 for _ in iter_samples(sf))
        stats[bench_dir_name]["correct"] = len(correct)
        if not correct:
            print("    no correct + non-empty samples; skip")
            continue

        # Compute response-length percentiles
        lens = sorted(s["len"] for s in correct)
        n = len(lens)
        def pct(p): return lens[max(0, min(n - 1, int(p / 100.0 * n)))]
        cutoff = pct(args.rumination_percentile)
        stats[bench_dir_name]["len_p50"] = pct(50)
        stats[bench_dir_name]["len_p70"] = pct(70)
        stats[bench_dir_name]["len_p90"] = pct(90)
        kept = [s for s in correct if s["len"] <= cutoff]
        stats[bench_dir_name]["kept"] = len(kept)
        print(f"    {len(correct)} correct → length cutoff {cutoff} "
              f"({args.rumination_percentile}th pct) → {len(kept)} non-rumi")

        # Render with chat template and emit
        local_tokens = 0
        for ex in kept:
            chat = [
                {"role": "user",      "content": ex["prompt"]},
                {"role": "assistant", "content": ex["response"]},
            ]
            try:
                text = tok.apply_chat_template(chat, tokenize=False,
                                               add_generation_prompt=False)
            except Exception:
                # Some templates need slightly different shape; just concat
                text = f"{ex['prompt']}\n\n{ex['response']}"
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            out_f.write(text)
            out_f.write(args.sep)
            local_tokens += ntok
            grand_total_examples += 1
            grand_total_tokens += ntok
            if args.max_examples and grand_total_examples >= args.max_examples:
                break
        stats[bench_dir_name]["tokens"] = local_tokens
        print(f"    tokens written: {local_tokens:,}")
        if args.max_examples and grand_total_examples >= args.max_examples:
            print("    --max-examples reached, stop")
            break

    out_f.close()
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(sidecar, "w") as f:
        json.dump({
            "args": vars(args),
            "stats": dict(stats),
            "grand_total_examples": grand_total_examples,
            "grand_total_tokens": grand_total_tokens,
        }, f, indent=2)

    print("\nDONE")
    print(f"  examples: {grand_total_examples}")
    print(f"  tokens:   {grand_total_tokens:,}")
    print(f"  corpus:   {out_path}")
    print(f"  meta:     {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
