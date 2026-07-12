#!/usr/bin/env python3
"""eog_corpus_terminator_scan.py — T202.5 decisive corpus pre-check.

Two jobs, both CPU/tokenizer-only:
 1. Resolve what the Gemma 4 26B-A4B eog ids [1, 106, 50] actually ARE in this
    tokenizer, and which token closes an assistant turn (Gemma 4 uses <|turn>
    style chat markers, NOT <start_of_turn>/<end_of_turn>). Decode the ids and
    round-trip candidate terminator strings so we measure the RIGHT token.
 2. Scan each corpus: tokenize every line the way the producer does and count
    emit positions (ids[t+1] == TERM) for each of the three eog ids, per corpus.
    A terminator with ~0 positions cannot be mapped from that corpus.

Usage: eog_corpus_terminator_scan.py <tokenizer_dir> <corpus1.jsonl> [corpus2 ...]
"""
import json
import sys

from transformers import AutoTokenizer

TOK_DIR = sys.argv[1]
CORPORA = sys.argv[2:]
EOG_IDS = [1, 106, 50]

tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
print("tokenizer:", TOK_DIR)
print("vocab_size:", tok.vocab_size, " len:", len(tok))
print()

# --- 1. resolve token identities --------------------------------------------
print("=== decode eog + neighbor ids ===")
for tid in [0, 1, 2, 3, 49, 50, 51, 104, 105, 106, 107, 108]:
    try:
        s = tok.convert_ids_to_tokens(tid)
    except Exception as e:
        s = "<err %s>" % e
    print("  id %-4d -> %r" % (tid, s))
print()

print("=== round-trip candidate terminator strings ===")
for name in ["<eos>", "<end_of_turn>", "<|turn>", "<|end_turn>",
             "<|tool_response>", "<start_of_turn>", "<end_of_image>"]:
    ids = tok.encode(name, add_special_tokens=False)
    print("  %-18s -> ids %s" % (name, ids))
print()

print("=== generation_config eos / special tokens ===")
print("  eos_token:", getattr(tok, "eos_token", None),
      "eos_token_id:", getattr(tok, "eos_token_id", None))
print("  bos_token:", getattr(tok, "bos_token", None),
      "bos_token_id:", getattr(tok, "bos_token_id", None))
print("  pad_token:", getattr(tok, "pad_token", None),
      "pad_token_id:", getattr(tok, "pad_token_id", None))
print()

# --- 2. scan corpora for emit positions -------------------------------------
print("=== emit-position counts per corpus (next-token == TERM) ===")
hdr = "%-46s %8s %10s %10s %10s %10s" % (
    "corpus", "lines", "tokens", "term=1", "term=106", "term=50")
print(hdr)
for path in CORPORA:
    nlines = ntok = 0
    cnt = {t: 0 for t in EOG_IDS}
    lines_with = {t: 0 for t in EOG_IDS}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                txt = json.loads(line)["text"]
            except Exception:
                continue
            ids = tok.encode(txt, add_special_tokens=False)
            nlines += 1
            ntok += len(ids)
            seen = {t: False for t in EOG_IDS}
            for t in range(len(ids) - 1):
                nxt = ids[t + 1]
                if nxt in cnt:
                    cnt[nxt] += 1
                    seen[nxt] = True
            for t in EOG_IDS:
                if seen[t]:
                    lines_with[t] += 1
    base = path.rsplit("/", 1)[-1]
    print("%-46s %8d %10d %10d %10d %10d" % (
        base, nlines, ntok, cnt[1], cnt[106], cnt[50]))
    print("%-46s %8s %10s %10d %10d %10d" % (
        "  (lines containing >=1)", "", "", lines_with[1],
        lines_with[106], lines_with[50]))
print()
print("[done]")
