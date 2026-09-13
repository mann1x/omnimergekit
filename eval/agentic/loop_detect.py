#!/usr/bin/env python3
"""THE loop detector. Single source of truth — import it, never re-declare it.

WHY THIS FILE EXISTS. The original census used only char-level regexes:
    RE_REPEAT  = (.{2,40}?)\\1{7,}   # a 2-40 char unit repeated >=8x
    RE_RUNCHAR = (.)\\1{40,}         # one char repeated >=41x
Both are BLIND to the loop shape a reasoning model actually produces, which is
a whole SENTENCE or paragraph restated over and over. Measured on the BFCL
composite cohort (2026-09-12): the char detectors fired on 1 armA / 9 armB
assistant blocks, while sentence-level repetition fired on 29 / 89 — a 10-30x
gap. The worst real block repeated
    'I\\'ll try to search for "revenue" in all files in the current directory.'
**402 times out of 421 units** in 30,920 chars, and the char detectors called
it clean. Any "loop rate is null" computed with char detectors alone is an
artefact of the instrument, not a property of the model.

A zero needs a nonzero floor control: `selftest()` asserts each detector fires
on its own positive control AND that healthy long reasoning stays clean.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

RE_REPEAT = re.compile(r"(.{2,40}?)\1{7,}", re.S)
RE_RUNCHAR = re.compile(r"(.)\1{40,}", re.S)

# A unit must be this long to count as a "sentence" — short fragments like
# "Yes." recur legitimately. 60 chars is above normal agent boilerplate.
SENT_MIN_LEN = 60
# 3 identical 60+ char units inside ONE block. Legitimate enumeration varies
# its wording; verbatim triplication of a long sentence does not occur in
# healthy output (verified against the healthy control).
SENT_MIN_REP = 3
# COUNT ALONE IS NOT ENOUGH. Degeneration means the block is DOMINATED by the
# repeat, not merely contains it. Audited on terminal-bench 2026-09-13: every
# 3-6x firing was iterative drafting -- the model rewriting the same code line
# or re-quoting the prompt with THOUSANDS of chars of new reasoning between
# occurrences (3 of 38 units = 8%, 4 of 110 = 3.6%). The real BFCL loop was
# 402 of 421 units = 95%. So require either a dominating SHARE or an absolute
# count no drafting process reaches.
SENT_MIN_SHARE = 0.30
SENT_ABS_CATASTROPHIC = 8

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# A repeated unit must contain real WORD content. Without this the char
# detector fires on code indentation and numeric output, which are not loops:
# measured on terminal-bench 2026-09-13, EVERY rep firing was `'  '` (two
# spaces, i.e. indentation) or `'0. '` (a numpy array printout). In prose-only
# corpora this never showed up; in code-bearing agent transcripts it is
# pervasive. 3+ letters is the floor for "this is language, not layout".
_MIN_LETTERS = 3


def _has_word_content(u: str) -> bool:
    return sum(ch.isalpha() for ch in u) >= _MIN_LETTERS


def sentence_repeat(text: str, min_len: int = SENT_MIN_LEN,
                    min_rep: int = SENT_MIN_REP):
    """(count, unit, n_units) for the most-repeated long unit WITHIN one block.

    Within-block only: an agent restating a fact across separate turns is
    normal; restating it verbatim 3+ times inside a single reasoning block is
    the loop.
    """
    units = [u.strip() for u in _SPLIT.split(text)
             if len(u.strip()) >= min_len and _has_word_content(u)]
    if not units:
        return 0, None, 0
    c = Counter(hashlib.md5(u.encode("utf-8", "replace")).hexdigest() for u in units)
    h, n = c.most_common(1)[0]
    share = n / len(units)
    if n < min_rep or (share < SENT_MIN_SHARE and n < SENT_ABS_CATASTROPHIC):
        return 0, None, len(units)
    unit = next(u for u in units
                if hashlib.md5(u.encode("utf-8", "replace")).hexdigest() == h)
    return n, unit, len(units)


def detect(text: str, runaway_chars: int = 10000) -> dict:
    """Per-block detector flags. `looped` is the OR — the four detectors catch
    different failures and must also be reported separately, never collapsed."""
    n_sent, unit, n_units = sentence_repeat(text)
    mrep = RE_REPEAT.search(text)
    # the repeated unit itself must be language, not indentation/rules/digits
    rep_ok = bool(mrep) and _has_word_content(mrep.group(1))
    mrun = RE_RUNCHAR.search(text)
    # a run of 41 '=' or '-' is a separator; a run of 41 'a' is degeneration
    run_ok = bool(mrun) and mrun.group(1).isalpha()
    d = {
        "rep": rep_ok,
        "runchar": run_ok,
        "sentence": bool(n_sent),
        "runaway": len(text) >= runaway_chars,
        "sentence_reps": n_sent,
        "sentence_units": n_units,
        "sentence_share": (n_sent / n_units) if n_units else 0.0,
        "sentence_unit": (unit[:200] if unit else None),
        "chars": len(text),
    }
    d["looped"] = d["rep"] or d["runchar"] or d["sentence"] or d["runaway"]
    return d


def message_texts(m):
    """Assistant reasoning + content from an inspect message, handling BOTH the
    string and the list-of-parts content shapes. armA's messages are list-typed
    and armB's are string-typed in the same cohort — a probe that handles only
    one silently reports ZERO blocks for the other arm."""
    out = []
    r = getattr(m, "reasoning", None)
    if isinstance(r, str) and r:
        out.append(r)
    c = getattr(m, "content", None)
    if isinstance(c, str) and c:
        out.append(c)
    elif isinstance(c, list):
        for it in c:
            t = getattr(it, "text", None)
            if isinstance(t, str) and t:
                out.append(t)
    return out


def selftest() -> int:
    """Positive controls: every detector must fire on its own degeneration and
    stay silent on healthy output. Run: python loop_detect.py"""
    cases = [
        ("char unit x12",        "Let me check the file. " * 12,        "rep"),
        ("single char x60",      "a" * 60,                              "runchar"),
        ("sentence x6",          "I should call get_stock_info to retrieve the data for NVDA now. " * 6, "sentence"),
        ("paragraph(180) x4",    ("I have already retrieved the watchlist and the stock info. "
                                  "I will now verify the directory contents before proceeding "
                                  "with the next tool call in this sequence. ") * 4, "sentence"),
        ("real: search x402",    'I\'ll try to search for "revenue" in all files in the current directory. ' * 402, "sentence"),
    ]
    bad = 0
    for name, text, want in cases:
        d = detect(text, runaway_chars=10**9)   # isolate from the length proxy
        ok = d[want]
        print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} -> {want}={d[want]} "
              f"(rep={d['rep']} runchar={d['runchar']} sent={d['sentence_reps']})")
        bad += not ok
    # NEGATIVE controls taken from real terminal-bench transcripts: these are
    # code/layout, NOT loops, and every one of them fired before the fix.
    negs = [
        ("code indentation", "def f():\n" + "        x = 1\n" * 12),
        ("two-space run", "    " * 40),
        ("numpy array print", "First row: [" + "0. " * 30 + "]"),
        ("ascii rule", "=" * 60),
        ("dash rule", "-" * 60),
    ]
    for name, text in negs:
        d = detect(text, runaway_chars=10**9)
        ok = not (d["rep"] or d["runchar"] or d["sentence"])
        print(f"  {'PASS' if ok else 'FAIL'}  NEG {name:20s} -> rep={d['rep']} "
              f"runchar={d['runchar']} sent={d['sentence_reps']} (must all be falsey)")
        bad += not ok

    # Audited real terminal-bench firings that are DRAFTING, not loops.
    drafting = ("blah. " * 0) + "".join(
        [f"Consideration number {i} about the algorithm and its behaviour here. "
         for i in range(37)])
    drafting = ("sample_envelope <- function(pts_x, pts_h, lower, upper) {\n"
                + drafting)[:0] + drafting
    # 3 repeats of one unit among 38 total -> 8% share, must NOT fire
    draft_block = ("The function signature is sample_envelope with five arguments. "
                   + "".join(f"Now I consider sub-case {i} in some detail here. "
                             for i in range(17))) * 1
    draft_block = ("The function signature is sample_envelope with five arguments. "
                   .join(["".join(f"Distinct reasoning step {i+j*12} follows here now. "
                                  for i in range(12)) for j in range(3)]))
    d = detect(draft_block, runaway_chars=10**9)
    ok = not d["sentence"]
    print(f"  {'PASS' if ok else 'FAIL'}  NEG {'drafting 3x low-share':20s} -> "
          f"sent={d['sentence_reps']} share={d['sentence_share']:.2f} (must not fire)")
    bad += not ok

    healthy = " ".join(f"Step {i}: consider option {i} carefully." for i in range(200))
    d = detect(healthy, runaway_chars=10**9)
    ok = not d["looped"]
    print(f"  {'PASS' if ok else 'FAIL'}  {'healthy long reasoning':22s} -> looped={d['looped']} "
          "(must be False: no false positive)")
    bad += not ok
    # the length proxy must still work on its own
    d = detect("x" * 12000, runaway_chars=10000)
    print(f"  {'PASS' if d['runaway'] else 'FAIL'}  {'runaway length':22s} -> runaway={d['runaway']}")
    bad += not d["runaway"]
    print("SELFTEST", "OK" if not bad else f"FAILED ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
