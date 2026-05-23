#!/usr/bin/env python3
"""
parse_llama_server_throughput.py — extract tokens/sec aggregate from llama-server log.

llama-server logs per-completion timing in this shape (varies slightly by version):

    print_timings: prompt eval time =   X ms /   N tokens (M ms per token, W tok/s)
    print_timings:        eval time =   Y ms /   M tokens (K ms per token, T tok/s)
    print_timings:       total time = X+Y ms /  N+M tokens

We aggregate the DECODE side (`eval time`, NOT `prompt eval time`):
    total_decode_tokens = sum(M_i)
    total_decode_seconds = sum(Y_i) / 1000
    avg_tokens_per_sec = total_decode_tokens / total_decode_seconds

Emits JSON with the aggregate + sample counts + count of timing lines found.
"""
import argparse
import json
import re
import sys
from pathlib import Path


EVAL_RE = re.compile(
    r"\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token\s*,\s*([\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
    re.IGNORECASE,
)


def parse(log_path: Path) -> dict:
    """Walk the log and collect every decode-side timing. Returns a dict with
    aggregate decode throughput + per-completion stats."""
    decode_times_ms = []
    decode_tokens = []
    decode_rates = []
    prompt_eval_times_ms = []
    prompt_eval_tokens = []
    n_lines = 0
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            n_lines += 1
            # Decode-side. We want lines that are NOT "prompt eval time" — the
            # regex requires the bare "eval time" (with word boundary before).
            # Some llama-server versions also tag this as "generation time" — try both.
            m = EVAL_RE.search(ln)
            if m and "prompt" not in ln.lower().split("eval time")[0][-12:]:
                ms = float(m.group(1))
                tok = int(m.group(2))
                rate = float(m.group(3))
                decode_times_ms.append(ms)
                decode_tokens.append(tok)
                decode_rates.append(rate)
                continue
            m2 = PROMPT_EVAL_RE.search(ln)
            if m2:
                prompt_eval_times_ms.append(float(m2.group(1)))
                prompt_eval_tokens.append(int(m2.group(2)))
    if decode_times_ms:
        total_ms = sum(decode_times_ms)
        total_tok = sum(decode_tokens)
        agg_tok_s = (total_tok / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        # Average of per-completion rates (different from aggregate when concurrent)
        mean_per_completion_rate = sum(decode_rates) / len(decode_rates)
    else:
        agg_tok_s = 0.0
        mean_per_completion_rate = 0.0
        total_tok = 0
        total_ms = 0.0
    return {
        "log_path": str(log_path),
        "log_lines_scanned": n_lines,
        "n_completions_timed": len(decode_times_ms),
        "total_decode_tokens": total_tok,
        "total_decode_seconds": total_ms / 1000.0 if total_ms else 0.0,
        "aggregate_decode_tokens_per_sec": round(agg_tok_s, 2),
        "mean_per_completion_rate": round(mean_per_completion_rate, 2),
        "prompt_eval": {
            "n_lines": len(prompt_eval_times_ms),
            "total_tokens": sum(prompt_eval_tokens),
            "total_seconds": sum(prompt_eval_times_ms) / 1000.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="path to llama-server stderr/stdout log")
    ap.add_argument("--output", default=None, help="JSON output path (default: <log>.throughput.json)")
    args = ap.parse_args()

    p = Path(args.log_path)
    if not p.exists():
        print(f"ERROR: log not found: {p}", file=sys.stderr)
        sys.exit(1)
    stats = parse(p)
    out_path = Path(args.output) if args.output else p.with_suffix(p.suffix + ".throughput.json")
    out_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
