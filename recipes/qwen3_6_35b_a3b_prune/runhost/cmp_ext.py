#!/usr/bin/env python3
"""bug-604 blast radius with GENERATION HELD FIXED.

Same raw replies, same problems, same docker evaluator. The ONLY difference
between the two trees is chat_to_body (pre-fix 3221739^ vs current). Any pass@1
gap is the extractor fix and nothing else.
"""
import json
from pathlib import Path
NEW = Path("/srv/ml/eval_results_b604/qwen_suite/multipl_e_100")
OLD = Path("/srv/ml/eval_results_b604_oldext/qwen_suite/multipl_e_100")
BANK = Path("/srv/ml/eval_results/qwen_suite/multipl_e_100")
LANGS = ["rs", "java", "js"]

def verd(cell_root, cell, lang):
    d = cell_root / cell / "results" / f"humaneval-{lang}"
    out = {}
    if not d.is_dir(): return out
    for f in d.glob("*.results.json"):
        if f.name == "_summary.json": continue
        try: rs = json.loads(f.read_text()).get("results") or []
        except Exception: continue
        if rs: out[f.name[:-13]] = rs[0].get("status") == "OK"
    return out

CELLS = ["qwencodermpe_t10_q6k","qwen256e_q6k","qwencodermpe_q6k","qwenhybridp24_q6k"]
h = (f"{'cell':<24} {'lang':<5} {'oldext':>7} {'newext':>7} {'Δ_b604':>7} "
     f"{'f>p':>4} {'p>f':>4} | {'banked':>7} {'Δ_regen':>8}")
print(h); print("-"*len(h))
agg = {}
for cell in CELLS:
    for lang in LANGS:
        o, n, b = verd(OLD,cell,lang), verd(NEW,cell,lang), verd(BANK,cell,lang)
        k = sorted(set(o) & set(n))
        if not k: continue
        po, pn = sum(o[x] for x in k)/len(k), sum(n[x] for x in k)/len(k)
        fp = sum(1 for x in k if n[x] and not o[x]); pf = sum(1 for x in k if o[x] and not n[x])
        kb = sorted(set(b) & set(n))
        pb = sum(b[x] for x in kb)/len(kb) if kb else float("nan")
        print(f"{cell:<24} {lang:<5} {po:>7.3f} {pn:>7.3f} {100*(pn-po):>+7.1f} "
              f"{fp:>4} {pf:>4} | {pb:>7.3f} {100*(po-pb):>+8.1f}")
        a = agg.setdefault(cell, [0.,0.,0.,0])
        a[0]+=po; a[1]+=pn; a[2]+=pb; a[3]+=1
print("-"*len(h))
print(f"{'MACRO (mean over 3 langs)':<30} {'oldext':>7} {'newext':>7} {'Δ_b604':>7}   "
      f"{'banked':>7} {'Δ_regen':>8}")
for cell,(so,sn,sb,c) in agg.items():
    print(f"{cell:<30} {so/c:>7.4f} {sn/c:>7.4f} {100*(sn-so)/c:>+7.2f}   "
          f"{sb/c:>7.4f} {100*(so-sb)/c:>+8.2f}")
