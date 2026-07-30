"""Chat-aware AIME-24 scorer. Stock aime24.utils.process_results extracts
the first-$ to last-$ LaTeX blob and rejects plain "= 33" chat-mode
endings. This shadow version uses the same heuristic as the offline
rescorer at backup_models/scripts/rescore_aime_chat.py (validated on
128e NVFP4A16 → 22/30 = 73.33% vs stock 0/30).

NOTE: aime100_chat/utils_chat.py is a SYMLINK to this file — aime_30/aime24_chat
and aime_100 share one scorer. Any change here re-scores both.

Order:
  1. last \\boxed{N}
  2. a line-anchored answer line: "Answer: N" / "**Final Answer:** N"
  3. "Final answer/Answer is/Answer: \\boxed{N}"
  4. last-line "is N" / "= N" / "equals N"
  5. last "= N" anywhere in the response   (salvage — manufactures an answer)
  6. last integer in the last 400 chars    (salvage — manufactures an answer)

Tier 2 added 2026-07-26. Tier 3's `_FA` is written `\\$?\\\\?boxed?\\{?`, i.e. the
literal "boxe" plus an OPTIONAL "d" — so "boxe" is REQUIRED, and a plain
`Answer: 405` never matched it, falling through to the tier-5 salvage ("last
`= N` anywhere"), which returns a digit from an unrelated equation. The old
docstring claimed tier 2 handled the bare form; it did not.

The obvious repair — making `\\boxed{` optional inside `_FA` so a bare number
matches anywhere after the answer phrase — was MEASURED AND REJECTED: over all
59 AIME samples files / 1623 docs on disk it recovers 17 but LOSES 2, because
mid-reasoning speculation ("Wait, if the answer is 6, then … is impossible")
matches and, being the last match, wins. The line-anchored tier below recovers
12 and loses 0. A positional "last marker anywhere wins" rewrite was also
measured: bit-identical to this cascade on all 1623 docs, so not worth the
complexity. Moves: aime_100 base 0.3500→0.3600, v24 0.2000→0.2100,
v25 0.1800→0.1900; aime_30 v4 20→21, v5-coder 16→17, qwen35-9b 17→21.
"""
from __future__ import annotations
import re
from typing import Dict, List

_BOXED  = re.compile(r"\\boxed\s*\{([^{}]+)\}")
# Tier 2: an answer LINE. Anchored to line start so it cannot fire inside prose
# such as "if the answer is 6, then ..." — that is exactly the regression that
# disqualified the bare-number generalisation of _FA (see module docstring).
_ANSLN  = re.compile(r"(?:^|\n)\s*\**\s*(?:final\s*)?answer\s*:?\s*\**\s*\$?\s*([\-+]?\d+)", re.IGNORECASE)
# Tier 3: answer phrase followed by a \boxed{...} form. `boxe` is REQUIRED here
# BY DESIGN — the bare-number case is tier 2's job. Do not loosen this.
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
