"""Chat-aware AIME-24 scorer. Stock aime24.utils.process_results extracts
the first-$ to last-$ LaTeX blob and rejects plain "= 33" chat-mode
endings. This shadow version uses the same heuristic as the offline
rescorer at backup_models/scripts/rescore_aime_chat.py (validated on
128e NVFP4A16 → 22/30 = 73.33% vs stock 0/30).

Order:
  1. last \\boxed{N}
  2. "Final answer/Answer is/Answer: N"
  3. last-line "is N" / "= N" / "equals N"
  4. last "= N" anywhere in the response
  5. last integer in the last 400 chars
"""
from __future__ import annotations
import re
from typing import Dict, List

_BOXED  = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_FA     = re.compile(r"(?:final\s*answer|answer\s*is|answer\s*:)\s*\*{0,2}\s*\$?\\?boxed?\{?([\-+]?\d+)", re.IGNORECASE)
_IS_END = re.compile(r"\b(?:is|equals|=)\s*\$?\s*([\-+]?\d+)\s*\$?\.?\s*$", re.IGNORECASE | re.MULTILINE)
_EQ_ANY = re.compile(r"=\s*\*{0,2}\s*\$?\s*([\-+]?\d+)\b")
_INT    = re.compile(r"[\-+]?\d+")


def _extract(resp: str) -> str | None:
    bx = _BOXED.findall(resp)
    if bx:
        cand = re.sub(r"[^\d\-+]", "", bx[-1])
        if cand:
            return cand
    fa = _FA.findall(resp)
    if fa:
        return fa[-1]
    last_line = resp.rstrip().rsplit("\n", 1)[-1]
    m = _IS_END.search(last_line)
    if m:
        return m.group(1)
    eq = _EQ_ANY.findall(resp)
    if eq:
        return eq[-1]
    ints = _INT.findall(resp[-400:])
    if ints:
        return ints[-1]
    return None


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    resp = results[0] if results else ""
    answer_key = next((k for k in doc.keys() if k.lower() == "answer"), "Answer")
    target = str(doc[answer_key]).strip()
    try:
        target_norm = str(int(target))
    except Exception:
        target_norm = target
    got = _extract(resp or "")
    return {"exact_match": 1 if got == target_norm else 0}
