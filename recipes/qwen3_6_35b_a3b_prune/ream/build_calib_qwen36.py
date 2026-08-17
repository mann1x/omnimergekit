#!/usr/bin/env python
"""Build REAM calibration batches tokenized with the Qwen3.6 tokenizer.

WHY THIS EXISTS. Samsung ships pre-tokenized batches in ream/data/*.pt, but they were
built with the **Qwen3** tokenizer (vocab 151,936). Qwen3.6-35B-A3B has vocab 248,044.
Feeding their ids to our model would run the whole profiling pass on garbage tokens and
produce a saliency vector that looks fine and means nothing. So we regenerate.

Their originals are never touched: we write with sfx='qwen36', they use 'qwen3'.

Their own runner (data/calibration_data.py __main__) loops over ['c4','math','code'] and
dies on c4 because C4_LOCATION=/datasets/c4 does not exist here. That is fine to skip --
at REAM's default --mix_ratio 0.0,0.3,0.7 the c4 share is 0.0, and Merger drops any
dataset whose n_samples==0 (ream/merger.py:116) before it ever opens the file. So math+code
is the complete default recipe. We refuse loudly if asked for c4 rather than silently
emitting a file the merger would misread.

Also supports --from-jsonl to build a batch from OUR targeted router-calib corpus
(results/router_calib_corpus_*.jsonl, field "text", already chat-templated), for the
"same calibrating dataset" variant asked for in the HF discussion.
"""
import argparse
import os
import sys

import torch


def build_from_jsonl(tokenizer, path, batch_size, seq_len, seed):
    """Tokenize our own jsonl corpus into the merger's batch format.

    Mirrors create_batch()'s output contract: dict of stacked input_ids/attention_mask,
    right-padded to seq_len. We do NOT apply create_batch's short-sequence filter -- our
    rows are curated and dropping the short ones would silently re-weight the bench mix.
    """
    import json

    texts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    if not texts:
        raise SystemExit(f"refusing: no rows with a 'text' field in {path}")

    import numpy as np

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(texts))[:batch_size]
    texts = [texts[i] for i in order]

    enc = tokenizer([t for t in texts], return_tensors="pt",
                    padding="max_length", truncation=True, max_length=seq_len)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    print(f"jsonl rows available={len(order)} used={out['input_ids'].shape[0]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/ml/models/Qwen3.6-35B-A3B")
    ap.add_argument("--ream-dir", default="/shared/dev/ream",
                    help="clone of SamsungSAILMontreal/ream (for data/calibration_data.py)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the .pt (default: <ream-dir>/data)")
    ap.add_argument("--datasets", default="math,code",
                    help="comma list from {math,code}; c4 is refused (see module docstring)")
    ap.add_argument("--from-jsonl", default=None,
                    help="build a single batch from our own corpus instead")
    ap.add_argument("--name", default=None, help="dataset name to use for --from-jsonl")
    ap.add_argument("--batch-size", type=int, default=3072)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--sfx", default="qwen36")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.sfx == "qwen3":
        raise SystemExit("refusing: sfx='qwen3' would overwrite Samsung's shipped batches")

    sys.path.insert(0, args.ream_dir)
    from data.calibration_data import create_batch, print_seq_stats  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    out_dir = args.out_dir or os.path.join(args.ream_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer {tok.__class__.__name__} vocab_size={tok.vocab_size} len={len(tok)} "
          f"pad={tok.pad_token_id} eos={tok.eos_token_id}", flush=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        print(f"pad_token_id was None -> set to eos {tok.pad_token_id}", flush=True)

    jobs = []
    if args.from_jsonl:
        if not args.name:
            raise SystemExit("--from-jsonl requires --name (it becomes the dataset key)")
        jobs.append((args.name, args.from_jsonl))
    else:
        for d in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            if d == "c4":
                raise SystemExit(
                    "refusing c4: C4_LOCATION=/datasets/c4 does not exist on this host, and "
                    "the default mix_ratio gives c4 a 0.0 share anyway (merger.py:116 skips it)")
            if d not in ("math", "code"):
                raise SystemExit(f"refusing unknown dataset {d!r}")
            jobs.append((d, None))

    for name, jsonl in jobs:
        batch_file = os.path.join(
            out_dir, f"{name}_b{args.batch_size}_seq{args.seq_len}_{args.sfx}_seed{args.seed}.pt")
        if os.path.exists(batch_file):
            print(f"{batch_file} already exists, skipping", flush=True)
            continue
        print(f"\n=== building {batch_file}", flush=True)
        if jsonl:
            batch = build_from_jsonl(tok, jsonl, args.batch_size, args.seq_len, args.seed)
            print_seq_stats(batch, tok)
        else:
            batch = create_batch(tok, name, args.batch_size, args.seq_len, seed=args.seed)
        tmp = batch_file + ".tmp"
        torch.save(batch, tmp)
        os.replace(tmp, batch_file)
        print(f">>> CALIB_WROTE {batch_file} {tuple(batch['input_ids'].shape)}", flush=True)

    print(">>> CALIB_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
