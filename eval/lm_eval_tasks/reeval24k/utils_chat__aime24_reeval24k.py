"""Chat-aware AIME-24 scorer. Stock aime24.utils.process_results extracts
the first-$ to last-$ LaTeX blob and rejects plain "= 33" chat-mode
endings. This shadow version uses the same heuristic as the offline
rescorer at backup_models/scripts/rescore_aime_chat.py (validated on
128e NVFP4A16 → 22/30 = 73.33% vs stock 0/30).

Order:
  1. last \\boxed{N}
  2. line-anchored "Answer: N" / "**Final Answer:** N"
  3. "Final answer/Answer is/Answer: \\boxed{N}"
  3. last-line "is N" / "= N" / "equals N"
  4. last "= N" anywhere in the response
  5. last integer in the last 400 chars
"""
from __future__ import annotations
import re
from typing import Dict, List

_BOXED  = re.compile(r"\\boxed\s*\{([^{}]+)\}")
# Tier 2 (added 2026-07-26): an answer LINE. Line-anchored so it cannot fire
# inside prose such as "if the answer is 6, then ..." -- that regression is why
# _FA below is NOT loosened to accept a bare number anywhere. Kept byte-aligned
# with eval/lm_eval_tasks/aime24_chat/utils_chat.py (see its docstring for the
# 1623-doc measurement: line-anchored +12/-0, bare-number-anywhere +17/-2).
_ANSLN  = re.compile(r"(?:^|\n)\s*\**\s*(?:final\s*)?answer\s*:?\s*\**\s*\$?\s*([\-+]?\d+)", re.IGNORECASE)
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
    aln = _ANSLN.findall(resp)
    if aln:
        return aln[-1]
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
