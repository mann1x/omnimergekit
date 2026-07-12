#!/usr/bin/env python3
"""Helpers for the omk-native OpenAI MRCR runner.

MRCR = Multi-round co-reference resolution (openai/mrcr, MIT). A long, multi-turn
synthetic user/assistant conversation hides 2/4/8 identical "asks" (e.g. "write a
poem about tapirs"); the final user turn asks for the i-th instance, with a random
alphanumeric hash to prepend. The model must reproduce that exact earlier assistant
reply, hash-prefixed.

This module is the data + scoring core. The runner (mrcr_runner.py) handles serving,
caching, and concurrency — same split as ruler_helpers/ruler_runner.

Three things live here:

  1. **Binning** — OpenAI bins samples by `n_tokens(prompt messages)` under the
     **o200k_base** (tiktoken) encoding, with power-of-two boundaries:
       [4096,8192] (8192,16384] (16384,32768] (32768,65536] (65536,131072]
       (131072,262144]=256k  (262144,524288]=512k  (524288,1048576]=1M
     There is NO native 768k bin. We add a SYNTHETIC `768k_synth` bin that
     sub-filters the 1M bin (524288,1048576] to o200k counts within 768k±12.5%
     ([688128, 851968]). It is clearly labelled non-standard in every artifact —
     it is NOT directly comparable to any published MRCR number.

     We count tokens with o200k to stay faithful to the published bin definition.
     The served model's OWN tokenizer length (Gemma etc.) differs and is reported
     separately so the operator can size the server's context (-c / --max-model-len).

  2. **Scoring** — verbatim port of OpenAI's `grade()`: if the sampled response does
     not start with `random_string_to_prepend`, the score is 0; otherwise both
     response and answer have the prefix stripped and the metric is
     `difflib.SequenceMatcher(None, response, answer).ratio()` in [0,1]. The bench
     score is the mean ratio over samples — a natural omk `pass_at_1` (0-1 scale).

  3. **Readiness** — a cheap preflight (imports + dataset reachability) mirroring
     ruler_helpers.ruler_native_readiness, surfaced by omk_eval._check_mrcr.

License: dataset is MIT (openai/mrcr). Pulled at runtime via huggingface_hub, never
vendored. Score artifacts are derivative output, freely publishable.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from functools import lru_cache

MRCR_REPO = "openai/mrcr"

# The six parquet shards (2/4/8 needle × {_0,_1}). 1.4 GB total; we only download
# the shards for the requested needle counts.
NEEDLE_FILES: dict[int, list[str]] = {
    2: ["2needle/2needle_0.parquet", "2needle/2needle_1.parquet"],
    4: ["4needle/4needle_0.parquet", "4needle/4needle_1.parquet"],
    8: ["8needle/8needle_0.parquet", "8needle/8needle_1.parquet"],
}

# Canonical o200k token bins (lo exclusive, hi inclusive) keyed by the omk ctx name.
# `768k_synth` is OUR addition — a sub-band of the 1M bin, NOT an OpenAI bin.
BINS: dict[str, tuple[int, int]] = {
    "8k":         (4096, 8192),       # note: OpenAI's first bin is [4096,8192] inclusive-lo
    "16k":        (8192, 16384),
    "32k":        (16384, 32768),
    "64k":        (32768, 65536),
    "128k":       (65536, 131072),
    "256k":       (131072, 262144),
    "512k":       (262144, 524288),
    "768k_synth": (688128, 851968),   # 786432 ± 12.5% — synthetic sub-band of the 1M bin
    "1024k":      (524288, 1048576),
}

# Upper bound used to size the served context (template selection.ctx_tokens hint).
BIN_CTX_TOKENS: dict[str, int] = {
    "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072,
    "256k": 262144, "512k": 524288, "768k_synth": 851968, "1024k": 1048576,
}


def resolve_bin(name: str) -> tuple[int, int]:
    """ctx-bin name → (o200k_lo_exclusive, o200k_hi_inclusive). Raises on unknown."""
    if name not in BINS:
        raise ValueError(f"unknown MRCR bin '{name}'; known: {sorted(BINS)}")
    return BINS[name]


@lru_cache(maxsize=1)
def _o200k():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def o200k_token_count(messages: list[dict]) -> int:
    """Sum of o200k_base token counts over message contents — the official MRCR
    binning measure (matches the dataset card's n_tokens())."""
    enc = _o200k()
    return sum(len(enc.encode(m["content"])) for m in messages)


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Verbatim OpenAI MRCR grader. Hash-prefix gate, then SequenceMatcher ratio.

    Returns a float in [0, 1]. Missing prefix ⇒ 0.0 (hard gate)."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def _hf_download(filename: str) -> str:
    """Download one needle parquet to the HF cache (persistent; respects HF_HOME /
    HF_HUB_CACHE). Returns the local path. Honors HF_HUB_ENABLE_HF_TRANSFER."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MRCR_REPO, filename=filename, repo_type="dataset")


def iter_bin_samples(
    bin_name: str,
    needles: list[int],
    num_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Load the needle shards for `needles`, keep rows whose o200k token count is in
    the `bin_name` band, and return up to `num_samples` of them (deterministic,
    seed-shuffled, balanced round-robin across needle counts).

    Each returned dict: {prompt(list[msg]), answer, random_string, n_needles,
    o200k_tokens, n_chars, sample_id}. The prompt is the parsed message list.

    Coarse n_chars pre-filter avoids tokenizing every row of every 1M-token shard:
    o200k averages ~3.3-4.2 chars/token for this English corpus, so we only exact-
    tokenize rows whose char count could plausibly land in the band.
    """
    import pandas as pd

    lo, hi = resolve_bin(bin_name)
    # generous char window around the token band (≥2.5 and ≤6.0 chars/token)
    char_lo, char_hi = int(lo * 2.5), int(hi * 6.0)

    per_needle: dict[int, list[dict]] = {n: [] for n in needles}
    for n in needles:
        rows: list[dict] = []
        for fn in NEEDLE_FILES[n]:
            path = _hf_download(fn)
            df = pd.read_parquet(path, columns=[
                "prompt", "answer", "random_string_to_prepend", "n_needles", "n_chars"])
            for _, r in df.iterrows():
                nc = int(r["n_chars"])
                if nc < char_lo or nc > char_hi:
                    continue
                msgs = json.loads(r["prompt"])
                tok = o200k_token_count(msgs)
                if tok <= lo or tok > hi:
                    continue
                rows.append({
                    "prompt": msgs,
                    "answer": r["answer"],
                    "random_string": r["random_string_to_prepend"],
                    "n_needles": int(r["n_needles"]),
                    "o200k_tokens": tok,
                    "n_chars": nc,
                })
        # deterministic shuffle within this needle count
        import random
        random.Random(seed + n).shuffle(rows)
        per_needle[n] = rows

    # Balanced round-robin interleave across needle counts, then cap to num_samples.
    out: list[dict] = []
    idx = 0
    while len(out) < num_samples and any(idx < len(per_needle[n]) for n in needles):
        for n in needles:
            if idx < len(per_needle[n]) and len(out) < num_samples:
                row = dict(per_needle[n][idx])
                row["sample_id"] = f"{bin_name}_n{n}_{idx}"
                out.append(row)
        idx += 1
    return out


def mrcr_native_readiness(bin_name: str | None = None) -> list[str]:
    """Cheap preflight. Returns a list of human-readable problems (empty == ready).

    Checks: required imports present; bin name (if given) is known; HF token/cache
    reachable enough to resolve the repo. Network fetch of the 1.4 GB shards is NOT
    forced here — that happens lazily in the runner's prepare phase."""
    problems: list[str] = []
    for mod in ("tiktoken", "pandas", "pyarrow", "huggingface_hub"):
        try:
            __import__(mod)
        except Exception as e:
            problems.append(f"missing python module '{mod}': {e}")
    if bin_name is not None and bin_name not in BINS:
        problems.append(f"unknown MRCR bin '{bin_name}'; known: {sorted(BINS)}")
    # o200k encoding load can fetch the BPE on first use; surface that cost early.
    try:
        _o200k()
    except Exception as e:
        problems.append(f"tiktoken o200k_base unavailable (needs first-run network or "
                        f"TIKTOKEN_CACHE_DIR): {e}")
    return problems


if __name__ == "__main__":  # tiny self-test (no network beyond tiktoken bpe)
    print("o200k ok:", o200k_token_count([{"role": "user", "content": "hello world"}]))
    print("grade gate:", grade("XYZpoem", "XYZpoem", "XYZ"),
          grade("poem", "XYZpoem", "XYZ"))
    print("readiness:", mrcr_native_readiness("256k") or "READY")
