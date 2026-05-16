"""Chat-aware ARC-Challenge scorer. Stock arc_challenge_chat uses
`until=['\\n\\n', '.']` which truncates chat-mode reasoning at the first
period or paragraph break — the model never emits a letter. Result on
128e NVFP4A16 with thinking off: 0% over 1172 items.

This shadow version:
  - Removes `until` truncation (lets the model finish its reply)
  - Extracts the answer letter via a chat-aware regex:
      1. "The best answer is X" / "Answer: X" / "answer is X"
      2. Last \\boxed{X}
      3. Last standalone letter A/B/C/D in the response

Returns exact_match: 1 if extracted letter matches the gold letter
(case-insensitive). Gold is the `answerKey` mapped to A-D (numeric "1"-"4"
→ "A"-"D" handled by doc_to_target in the YAML).
"""
from __future__ import annotations
import re
from typing import Dict, List

_PHRASE = re.compile(
    r"(?:best\s+answer\s+is|answer\s+is|the\s+answer\s+is|final\s+answer\s*:?|answer\s*:)"
    r"\s*\*{0,2}\s*\(?\s*([A-D])\b",
    re.IGNORECASE,
)
_BOXED  = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
_LETTER = re.compile(r"\b([A-D])\b")


def _extract(resp: str) -> str | None:
    if not resp:
        return None
    # 1. natural-language phrase
    p = _PHRASE.findall(resp)
    if p:
        return p[-1].upper()
    # 2. \boxed{X}
    bx = _BOXED.findall(resp)
    if bx:
        return bx[-1].upper()
    # 3. Last standalone A/B/C/D in the tail
    ints = _LETTER.findall(resp[-400:])
    if ints:
        return ints[-1].upper()
    return None


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    resp = results[0] if results else ""
    # gold: prefer answerKey mapped through "1234"→"ABCD"; fall back to direct
    ak = str(doc.get("answerKey", "")).strip()
    if ak in "1234":
        gold = "ABCD"[int(ak) - 1]
    elif ak.upper() in "ABCD":
        gold = ak.upper()
    else:
        gold = ak.upper()
    got = _extract(resp)
    return {"exact_match": 1 if (got and got == gold) else 0}
