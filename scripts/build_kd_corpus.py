#!/usr/bin/env python3
"""T192: build a chat-formatted KD corpus for E-ExpertKD (router_kd.py --corpus-file).

router_kd.build_batches tokenizes the `text` field RAW (no chat template), but the
multilingual/constrained loops happen in CHAT mode on the assistant completion. So we
pre-apply the chat template to each {prompt, completion} pair from the bucketed
antiloop corpus and emit {"text": <full chat string>}. KD then matches the teacher's
next-token distribution across the assistant answer — exactly the loop region.

Two balance profiles (KD-corpus-balance drives routing — the load-bearing lesson):
  ml_heavy  : multilingual dominant (recover the diffuse capability)
  balanced  : even across buckets   (recover without regressing code/IF)
"""
import argparse
import json
import random
from collections import defaultdict

from transformers import AutoTokenizer

PROFILES = {
    # bucket -> cap (None = take all available)
    "ml_heavy": {"multilingual": None, "constrained": 60, "code": 40, "retain": 60, "openended": 20},
    "balanced": {"multilingual": 120, "constrained": 110, "code": 110, "retain": 110, "openended": 50},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl")
    ap.add_argument("--tokenizer", default="/srv/ml/models/base/gemma-4-26B-A4B-it")
    ap.add_argument("--profile", choices=list(PROFILES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    by_bucket = defaultdict(list)
    for line in open(args.src):
        r = json.loads(line)
        p, c = (r.get("prompt") or "").strip(), (r.get("completion") or "").strip()
        if p and c:
            by_bucket[r.get("bucket", "?")].append((p, c))

    rng = random.Random(args.seed)
    caps = PROFILES[args.profile]
    rows, counts = [], {}
    for bucket, cap in caps.items():
        pool = by_bucket.get(bucket, [])
        rng.shuffle(pool)
        take = pool if cap is None else pool[:cap]
        counts[bucket] = len(take)
        for p, c in take:
            msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            rows.append({"text": text, "bucket": bucket})
    rng.shuffle(rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[kd-corpus:{args.profile}] wrote {len(rows)} rows -> {args.out}")
    print("  by bucket:", counts)


if __name__ == "__main__":
    main()
