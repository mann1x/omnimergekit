#!/usr/bin/env python3
"""Decompose ARC / IFEval failures into genuine-miss vs loop-driven.

Usage: bench_fail_audit.py <mode:arc|ifeval> <samples.jsonl> [dump_n]

ARC    : single-filter MC. wrong = exact_match!=1. Split into genuine
         wrong-letter (a clear final A-D extracted, but != target) vs
         loop/no-answer (degenerate repetition or no extractable letter),
         plus a permissive re-extraction upper bound.
IFEval : prompt_level_strict_acc is the score. For each FAILED prompt show
         which instruction types failed and whether the response looped;
         aggregate failed-instruction-type counts and loop vs genuine split.
"""
import json
import re
import sys
from collections import Counter

mode = sys.argv[1]
path = sys.argv[2]
dump_n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

recs = []
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))


def ngram_loop(text, n=5, thresh=8):
    """True if any word n-gram repeats >= thresh times (degenerate loop)."""
    words = text.split()
    if len(words) < n * thresh:
        return False, None, 0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    g, c = grams.most_common(1)[0]
    return (c >= thresh), " ".join(g), c


LETTER_PATS = [
    re.compile(r"answer is[:\s]*\(?([A-D])\)?", re.I),
    re.compile(r"answer[:\s]*\(?([A-D])\)?\b", re.I),
    re.compile(r"\*\*\(?([A-D])\)?\*\*"),
    re.compile(r"\(([A-D])\)"),
    re.compile(r"\b([A-D])\b(?=[^A-Za-z]*$)"),
]


def extract_letter(text):
    best = None
    for pat in LETTER_PATS:
        for m in pat.finditer(text):
            best = m.group(1).upper()
        if best:
            return best
    return None


def full_resp(r):
    fr = r.get("filtered_resps")
    if isinstance(fr, list) and fr and isinstance(fr[0], str):
        return fr[0]
    rp = r.get("resps")
    while isinstance(rp, list) and rp:
        rp = rp[0]
    return rp if isinstance(rp, str) else str(rp)


if mode == "arc":
    n = len(recs)
    correct = sum(1 for r in recs if r.get("exact_match") == 1)
    print(f"=== ARC n={n} correct={correct} score={correct/n:.4f} ===")
    wrong = [r for r in recs if r.get("exact_match") != 1]
    genuine, loops, no_letter = [], [], []
    recoverable = 0
    for r in wrong:
        txt = full_resp(r)
        tgt = str(r.get("target")).strip()[:1]
        looped, gram, cnt = ngram_loop(txt)
        letter = extract_letter(txt)
        if looped:
            loops.append((r, gram, cnt, letter, tgt))
        elif letter in set("ABCD"):
            genuine.append((r, letter, tgt))
            if letter == tgt:
                recoverable += 1  # extractor disagreement w/ task metric
        else:
            no_letter.append((r, txt, tgt))
    print(f"wrong={len(wrong)}  genuine-wrong-letter={len(genuine)}  "
          f"loops(ngram>=8)={len(loops)}  no-extractable-letter={len(no_letter)}")
    print(f"permissive-extractor agrees-with-gold on {recoverable} wrong "
          f"(metric/extractor disagreement, upper-bound +{recoverable/n*100:.2f}pp)")
    print(f"\n--- up to {dump_n} LOOP samples (degenerate repetition) ---")
    for r, gram, cnt, letter, tgt in loops[:dump_n]:
        print(f"[doc {r.get('doc_id')}] target={tgt} extracted={letter} "
              f"loop_gram x{cnt}: {gram!r}")
    print(f"\n--- up to {dump_n} NO-LETTER (non-loop) tails ---")
    for r, txt, tgt in no_letter[:dump_n]:
        print(f"[doc {r.get('doc_id')}] target={tgt} tail=...{txt[-200:].strip()[-180:]}")
    print(f"\n--- up to {min(4,dump_n)} GENUINE wrong-letter ---")
    for r, letter, tgt in genuine[:min(4, dump_n)]:
        print(f"[doc {r.get('doc_id')}] target={tgt} picked={letter}")

elif mode == "ifeval":
    n = len(recs)
    passed = sum(1 for r in recs if r.get("prompt_level_strict_acc"))
    print(f"=== IFEval n={n} prompt_strict_pass={passed} "
          f"score={passed/n:.4f} ===")
    failed = [r for r in recs if not r.get("prompt_level_strict_acc")]
    failtype = Counter()
    loops, genuine = [], []
    for r in failed:
        txt = full_resp(r)
        ids = r.get("doc", {}).get("instruction_id_list", [])
        flags = r.get("inst_level_strict_acc", [])
        failed_ids = [i for i, ok in zip(ids, flags) if not ok]
        for fid in failed_ids:
            failtype[fid] += 1
        looped, gram, cnt = ngram_loop(txt)
        tok = len(txt) // 4
        rec = (r, failed_ids, tok, looped, gram, cnt)
        (loops if looped else genuine).append(rec)
    print(f"failed_prompts={len(failed)}  loop-driven={len(loops)}  "
          f"genuine-instruction-miss={len(genuine)}")
    print(f"failed-instruction-type counts: {dict(failtype.most_common())}")
    print(f"\n--- failed prompts (up to {dump_n}) ---")
    for r, fids, tok, looped, gram, cnt in (loops + genuine)[:dump_n]:
        tag = f"LOOP x{cnt}" if looped else "genuine"
        print(f"[doc {r.get('doc_id')}] {tag} ~{tok}tok  failed={fids}")
else:
    print(f"unknown mode {mode!r}; use arc|ifeval")
    sys.exit(2)
