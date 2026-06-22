"""Channel-aware degenerate-loop detection for the agentic replay harness.

A model that "loops" in an agentic coding session fails in one of two ways, so
the detector runs two oracles over each generated turn:

  * ANSWER channel -- long runaway cycles that grow to the token cap. Detected by
    `detect_loop`: the tail (~220 words) has a distinct-5-gram ratio < 0.30, OR a
    single 5-gram shingle repeats >= 12x across the whole answer. Both thresholds
    are deliberately conservative, so a hit is almost certainly pathological and
    NOT a legitimately repetitive answer (a numbered list, "repeat N times", ...).

  * THINKING channel -- short 1-3 sentence cycles repeated verbatim until the
    reasoning budget is exhausted, e.g.
        "Actually, I'll fix the meta tag. Wait, I'll also fix initialscale."
    repeated. These sit below `detect_loop`'s thresholds early on, so a dedicated
    short-cycle detector finds the smallest 1..MAX_UNIT-sentence block that
    repeats >= MIN_REPEAT times (normalised: lowercased, whitespace-collapsed).

`detect_turn_loop(content, reasoning)` returns per-channel verdicts plus the
repeating unit, so a harness can both COUNT loops and SHOW what they look like.

This module is self-contained (stdlib only) so the harness can be vendored and
run anywhere.
"""
from __future__ import annotations

import re
from collections import Counter

# ---- answer-channel oracle (long runaway cycles) ---------------------------

LOOP_MIN_CHARS = 600       # below this an answer is too short to be a runaway
LOOP_MIN_WORDS = 60
LOOP_TAIL_WORDS = 220      # window for the tail distinct-5-gram test
LOOP_TAIL_RATIO = 0.30     # tail distinct-5-gram ratio below this == cycle
LOOP_SHINGLE_REPEAT = 12   # any 5-gram repeated >= this many times == cycle


def detect_loop(t: str) -> bool:
    """High-precision degenerate-loop detector for the ANSWER channel.

    Returns True only for runaway cycles, NOT for legitimately repetitive answers.
    Two phase-independent signals over whitespace tokens:
      (1) the TAIL (last ~220 words) has a distinct-5-gram ratio < 0.30 -- the
          classic "stuck in a cycle until the token cap" signature; and
      (2) any 5-word shingle repeats >= 12x across the whole answer -- a count
          far beyond any plausible "repeat N times" instruction.
    """
    t = t or ""
    if len(t) < LOOP_MIN_CHARS:
        return False
    w = t.split()
    if len(w) < LOOP_MIN_WORDS:
        return False
    tail = w[-LOOP_TAIL_WORDS:]
    if len(tail) >= LOOP_MIN_WORDS:
        sh = [" ".join(tail[i:i + 5]) for i in range(len(tail) - 4)]
        if sh and len(set(sh)) / len(sh) < LOOP_TAIL_RATIO:
            return True
    sh_all = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
    if sh_all:
        if Counter(sh_all).most_common(1)[0][1] >= LOOP_SHINGLE_REPEAT:
            return True
    return False


# ---- thinking-channel oracle (short verbatim cycles) -----------------------

MAX_UNIT = 3        # a repeating block is 1..3 consecutive sentences
MIN_REPEAT = 4      # >= 4 verbatim repeats of the block == loop
MIN_UNIT_CHARS = 12  # ignore trivially short units ("ok.", "yes")

_SENT = re.compile(r"[^.!?\n]*[.!?\n]+|\S[^.!?\n]*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sentences(text: str):
    out = []
    for m in _SENT.finditer(text or ""):
        s = m.group().strip()
        if s:
            out.append(s)
    return out


def detect_short_cycle(text: str):
    """Return (is_loop, info) for the THINKING channel: the max consecutive
    verbatim repeat of any 1..MAX_UNIT-sentence block (normalised). Conservative:
    needs MIN_REPEAT repeats and a non-trivial unit so legit short restatements
    don't trip it."""
    sents = [_norm(s) for s in _sentences(text)]
    sents = [s for s in sents if s]
    n = len(sents)
    best = None
    for u in range(1, MAX_UNIT + 1):
        i = 0
        while i + u <= n:
            unit = sents[i:i + u]
            if sum(len(x) for x in unit) < MIN_UNIT_CHARS:
                i += 1
                continue
            reps = 1
            j = i + u
            while j + u <= n and sents[j:j + u] == unit:
                reps += 1
                j += u
            if reps >= MIN_REPEAT and (best is None or reps > best["repeats"]):
                best = {"unit": " ".join(unit), "unit_sentences": u,
                        "repeats": reps, "span": [i, j]}
            i += 1
    return (best is not None), (best or {})


def detect_channel(text: str):
    """Combine the answer-channel and short-cycle oracles on one channel's text.
    Returns (is_loop, info)."""
    long_loop = detect_loop(text or "")
    short_loop, info = detect_short_cycle(text or "")
    info = dict(info)
    info["long_loop"] = bool(long_loop)
    info["short_loop"] = bool(short_loop)
    return (long_loop or short_loop), info


def detect_turn_loop(content: str, reasoning: str):
    """Top-level: a turn loops if EITHER channel loops. Thinking-channel loops are
    the primary chat-template-reinjection signal; the answer channel covers the
    relocated case (a reasoning budget pushes the loop into the answer)."""
    ans_loop, ans = detect_channel(content or "")
    think_loop, think = detect_channel(reasoning or "")
    return {
        "is_loop": bool(ans_loop or think_loop),
        "answer_loop": bool(ans_loop),
        "thinking_loop": bool(think_loop),
        "answer": ans,
        "thinking": think,
        "answer_len": len(content or ""),
        "thinking_len": len(reasoning or ""),
    }


if __name__ == "__main__":
    looped = ("Actually, I'll write it now. (If this fails, I'll try to fix it.) "
              * 6)
    clean = "Here is the plan. First create index.html. Then main.js. Done."
    print("looped ->", detect_turn_loop("", looped)["is_loop"],
          detect_short_cycle(looped)[1].get("repeats"))
    print("clean  ->", detect_turn_loop("", clean)["is_loop"])
