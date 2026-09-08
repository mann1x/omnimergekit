"""Build the REAM calibration batch for Gemma-4 from the AtomicChat corpus.

REAM's own data/calibration_data.py only knows c4/math/code and streams them
from HF. We already have a Gemma-4-tokenizer-matched corpus on disk (the AC
build), so this tokenizes that instead -- same output contract, same short-
sequence filter, plus a provenance sidecar.

CALIBRATION IS A BASIS: every merge arm (B'/C'/E'/K') must consume the SAME
.pt file. The sidecar records the corpus sha256 and the batch sha256 so a later
table can be checked rather than assumed.
"""
import argparse, hashlib, json, os, random, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="AC calib_train.txt")
    ap.add_argument("--tokenizer", required=True, help="model dir with the Gemma-4 tokenizer")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="ac")
    ap.add_argument("--batch-size", type=int, default=3072)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--sfx", default="gemma4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-seq-len", type=int, default=128,
                    help="drop sequences with fewer real tokens than this")
    ap.add_argument("--separator", default="\n\n")
    ap.add_argument("--mode", choices=["docs", "contiguous"], default="docs",
                    help="docs: one sequence per document (pads short ones). "
                         "contiguous: tokenize the whole corpus and slice into "
                         "seq_len blocks -- zero padding, and the same way "
                         "llama-imatrix consumes this file. NOTE the AC corpus "
                         "declares document_separator '\\n\\n' but that yields "
                         "49802 splits against a manifest count of 3148, so its "
                         "document boundaries are NOT recoverable; prefer "
                         "contiguous for this corpus.")
    a = ap.parse_args()

    out = os.path.join(a.out_dir,
                       f"{a.name}_b{a.batch_size}_seq{a.seq_len}_{a.sfx}_seed{a.seed}.pt")
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(out):
        sys.exit(f"REFUSING to overwrite an existing basis: {out}")

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"tokenizer: pad={tok.pad_token_id} eos={tok.eos_token_id}")

    raw = open(a.corpus, encoding="utf-8").read()

    if a.mode == "contiguous":
        # Tokenize in slabs to bound peak memory, concatenate, then slice.
        SLAB = 1 << 20
        ids_all = []
        for off in range(0, len(raw), SLAB):
            ids_all.extend(tok(raw[off:off + SLAB], add_special_tokens=False)["input_ids"])
        n_blocks = len(ids_all) // a.seq_len
        print(f"corpus: {len(raw)} bytes -> {len(ids_all)} tokens -> {n_blocks} full blocks")
        if n_blocks < a.batch_size:
            sys.exit(f"REFUSING: only {n_blocks} full blocks, need {a.batch_size}")
        blocks = [ids_all[i * a.seq_len:(i + 1) * a.seq_len] for i in range(n_blocks)]
        random.Random(a.seed).shuffle(blocks)
        blocks = blocks[:a.batch_size]
        batch = {"input_ids": torch.tensor(blocks, dtype=torch.long),
                 "attention_mask": torch.ones(a.batch_size, a.seq_len, dtype=torch.long)}
        seen, dropped = n_blocks, 0
        real = batch["attention_mask"].sum(1).float()
        torch.save(batch, out)
        _finish(a, out, seen, dropped, real)
        return

    docs = [d for d in raw.split(a.separator) if d.strip()]
    print(f"corpus: {len(raw)} bytes -> {len(docs)} documents")
    random.Random(a.seed).shuffle(docs)

    keep_ids, keep_mask, seen, dropped = [], [], 0, 0
    for d in docs:
        seen += 1
        t = tok(d, return_tensors="pt", truncation=True, max_length=a.seq_len)
        ids, msk = t["input_ids"], t["attention_mask"]
        if ids.shape[1] < a.seq_len:
            ids = F.pad(ids, (0, a.seq_len - ids.shape[1]), value=tok.pad_token_id)
            msk = F.pad(msk, (0, a.seq_len - msk.shape[1]), value=0)
        # upstream filter: skip sequences whose real-token count is too small
        if (ids[0] == tok.pad_token_id).sum() > (a.seq_len - a.min_seq_len):
            dropped += 1
            continue
        keep_ids.append(ids[0])
        keep_mask.append(msk[0])
        if len(keep_ids) >= a.batch_size:
            break

    if len(keep_ids) < a.batch_size:
        sys.exit(f"REFUSING: corpus yielded only {len(keep_ids)} usable sequences, "
                 f"need {a.batch_size}. A short basis is not a cheaper basis, it is a "
                 f"different one.")

    batch = {"input_ids": torch.stack(keep_ids), "attention_mask": torch.stack(keep_mask)}
    real = batch["attention_mask"].sum(1).float()
    torch.save(batch, out)
    _finish(a, out, seen, dropped, real)


def _finish(a, out, seen, dropped, real):
    meta = {
        "output": out,
        "output_sha256": sha256(out),
        "corpus": a.corpus,
        "corpus_sha256": sha256(a.corpus),
        "tokenizer_dir": a.tokenizer,
        "batch_size": a.batch_size, "seq_len": a.seq_len, "seed": a.seed,
        "min_seq_len": a.min_seq_len,
        "mode": a.mode,
        "documents_seen": seen, "documents_dropped_short": dropped,
        "real_tokens_total": int(real.sum().item()),
        "real_tokens_mean": round(real.mean().item(), 1),
        "real_tokens_min": int(real.min().item()),
        "pad_fraction": round(1 - real.mean().item() / a.seq_len, 4),
    }
    with open(out + ".basis.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("\nCALIB_BASIS_OK")


if __name__ == "__main__":
    main()
