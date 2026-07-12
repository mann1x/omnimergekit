#!/usr/bin/env python3
"""eac_prepare_corpus.py — build the EAC calibration corpus = WikiText-2 + the
domain router-calib corpus.

router_eac_calibrate.py reads --corpus-file as ONE text blob, tokenizes it, and
keeps enc[:n_seq*seq_len]. A naive `cat wiki calib` (WikiText-2 ~2.4M tok) would
fill the window and TRUNCATE the calib corpus away. Two modes:

DEFAULT (balanced-window, the chosen tradeoff): emit a window-sized (n_seq*seq_len
= 128*2048 = 262,144 tok) sample, split 50/50 between calib and WikiText-2 and
emitted as ALTERNATING seq-len blocks (64 calib + 64 wiki sequences). The
calibrator consumes the whole file with no meaningful truncation. Fast
(~v5coder-scale capture).

--full-corpus (opt-in, cover-all): full WikiText-2 + full calib, n_seq set to
consume everything (~tens of hours of paired-26B capture single-GPU). Rejected
as too long for now; kept available.

Writes <out> and <out>.meta.json {mode,total_tokens,seq_len,n_seq,calib_tokens,
wiki_tokens}. The runner reads n_seq from the meta. No GPU.

Usage:
  eac_prepare_corpus.py --seq-len 2048                  # balanced 256k (default)
  eac_prepare_corpus.py --seq-len 2048 --full-corpus    # cover-all (~tens of hours)
"""
import argparse
import json
import sys
from pathlib import Path

BM = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=f"{BM}/google/gemma-4-26B-A4B-it",
                    help="Gemma tokenizer dir (must match the calibrator's)")
    ap.add_argument("--calib-corpus", default=f"{BM}/scripts/router_calib_corpus.txt")
    ap.add_argument("--wikitext-config", default="wikitext-2-raw-v1")
    ap.add_argument("--out", default=f"{BM}/scripts/eac_corpus_wiki2_plus_calib.txt")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="Must match the EAC run's --seq-len (drives implied n_seq)")
    ap.add_argument("--n-seq", type=int, default=128,
                    help="n_seq for the default balanced-window mode "
                         "(window = n-seq*seq-len; 128x2048 = 262144 tok)")
    ap.add_argument("--full-corpus", action="store_true",
                    help="Cover-all mode (opt-in): emit the FULL WikiText-2 + calib "
                         "and set n_seq to consume it all (~tens of hours capture). "
                         "Default = the balanced 256k window (the chosen tradeoff).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    calib_text = Path(args.calib_corpus).read_text()
    calib_ids = tok(calib_text, add_special_tokens=False)["input_ids"]
    ds = load_dataset("Salesforce/wikitext", args.wikitext_config, split="train")
    wiki_text = "\n".join(s for s in ds["text"] if s.strip())
    wiki_ids = tok(wiki_text, add_special_tokens=False)["input_ids"]
    print(f"[corpus] calib={len(calib_ids):,} tok   wikitext-2={len(wiki_ids):,} tok")

    if not args.full_corpus:
        # default: balanced 256k window — 50/50, alternating seq-len blocks
        # (64 calib + 64 wiki sequences), consumed whole by n_seq=--n-seq.
        window = args.n_seq * args.seq_len
        half = window // 2
        def take(ids, n):
            out = list(ids[:n])
            while len(out) < n:
                out.extend(ids)
            return out[:n]
        c = take(calib_ids, half)
        w = take(wiki_ids, half)
        pieces, ci, wi, turn = [], 0, 0, True
        while ci < len(c) or wi < len(w):
            if turn and ci < len(c):
                pieces.append(c[ci:ci + args.seq_len])
                ci += args.seq_len
            elif not turn and wi < len(w):
                pieces.append(w[wi:wi + args.seq_len])
                wi += args.seq_len
            turn = not turn
        out_text = "\n".join(tok.decode(p, skip_special_tokens=False) for p in pieces)
        total = sum(len(p) for p in pieces)
        n_seq = args.n_seq
        mode = "balanced-window"
    else:
        # cover-all: full calib + full wiki, concatenated; n_seq covers everything
        out_text = calib_text + "\n" + wiki_text
        total = len(calib_ids) + len(wiki_ids)
        n_seq = total // args.seq_len
        mode = "cover-all"

    Path(args.out).write_text(out_text)
    meta = {"mode": mode, "total_tokens": total, "seq_len": args.seq_len,
            "n_seq": n_seq, "calib_tokens": len(calib_ids),
            "wiki_tokens": len(wiki_ids), "out": args.out}
    Path(args.out + ".meta.json").write_text(json.dumps(meta, indent=2))

    cap_tok = n_seq * args.seq_len
    print(f"[corpus] mode={mode}  total={total:,} tok")
    print(f"[corpus] => implied --n-seq {n_seq}  (covers {cap_tok:,} tok = "
          f"{100*cap_tok/max(total,1):.1f}% of corpus; remainder {total-cap_tok} tok < seq_len)")
    print(f"[corpus] wrote {args.out} ({len(out_text)/1e6:.1f} MB) + .meta.json")
    if mode == "cover-all":
        cap_tok_total = n_seq * args.seq_len
        print(f"[corpus] CAPTURE COST: {cap_tok_total:,} tok x 2 models (teacher+variant) "
              f"— budget tens of hours single-GPU with offload. Confirm before launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
