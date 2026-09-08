#!/usr/bin/env python3
"""Truncation census for the b604 MPE cells -- the measurement bug-609 suppressed.

Same detector as omk_eval's cap-check: tokenize the RAW generation and count rows
sitting within TOL of the max_tokens ceiling. The banked run recorded
verdict=CAPPED capped_pct=4.67 at max_gen_toks=1024; the b604 run recorded
status='unverified' because the tokenizer would not load (no sentencepiece).
With sentencepiece installed the same number is now computable offline from the
persisted raws -- no GPU, no regeneration.
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("/srv/ml/models/Qwen3.6-35B-A3B")
CAP, TOL = 1024, 16
ROOT = Path("/srv/ml/eval_results_b604/qwen_suite/multipl_e_100")
CELLS = ["qwencodermpe_t10_q6k", "qwen256e_q6k", "qwencodermpe_q6k", "qwenhybridp24_q6k"]

h = f"{'cell':<24} {'lang':<5} {'n':>4} {'p50':>6} {'p95':>6} {'max':>6} {'at_cap':>7} {'pct':>6}"
print(h); print("-" * len(h))
for cell in CELLS:
    tot = capped = 0
    for lang in ("rs", "java", "js"):
        d = ROOT / cell / "generations" / f"humaneval-{lang}"
        if not d.is_dir():
            continue
        lens = []
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            raw = (j.get("raw_completions") or [""])[0] or ""
            lens.append(len(TOK(raw, add_special_tokens=False).input_ids))
        lens.sort()
        n = len(lens)
        c = sum(1 for x in lens if x >= CAP - TOL)
        tot += n; capped += c
        print(f"{cell:<24} {lang:<5} {n:>4} {lens[n//2]:>6} {lens[int(n*.95)]:>6} "
              f"{lens[-1]:>6} {c:>7} {100*c/n:>5.1f}%")
    v = "CAPPED" if capped else "CLEAN"
    print(f"{cell:<24} {'ALL':<5} {tot:>4} {'':>6} {'':>6} {'':>6} {capped:>7} "
          f"{100*capped/tot:>5.2f}%  -> {v}")
    print("-" * len(h))
