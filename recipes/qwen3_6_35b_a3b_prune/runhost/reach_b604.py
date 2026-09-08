#!/usr/bin/env python3
"""Two follow-ups to the paired stratification.

(1) REACH. Delta-same == 0 is only evidence the extractor does nothing if the
    extractor actually RAN differently on that stratum. MPE stores only the
    post-extraction program (bug-607), so old-vs-new `results[0].program` on a
    byte-identical completion IS the extraction diff. If program-changed == 0,
    the SAME stratum is depleted of the population bug-604 touches (short
    completions are both more likely to be reproducible AND less likely to be
    body-only), and the paired test is uninformative rather than negative.

(2) MECHANISM. If the DIFF stratum is systematically (not symmetrically)
    better, the new generations should differ in shape. Census completion
    length + a truncation proxy old vs new.
"""
import json, statistics as st
from pathlib import Path

OLD = Path("/srv/ml/eval_results"); NEW = Path("/srv/ml/eval_results_b604")
LANGS = ["rs", "java", "js"]

def load(cell, lang):
    g, r = {}, {}
    d = cell / "generations" / f"humaneval-{lang}"
    if d.is_dir():
        for f in d.glob("*.json"):
            try: c = (json.loads(f.read_text()).get("completions") or [None])[0]
            except Exception: c = None
            g[f.stem] = c
    d = cell / "results" / f"humaneval-{lang}"
    if d.is_dir():
        for f in d.glob("*.results.json"):
            if f.name == "_summary.json": continue
            try: rs = json.loads(f.read_text()).get("results") or []
            except Exception: continue
            if rs: r[f.name[:-13]] = (rs[0].get("status"), rs[0].get("program") or "")
    return g, r

def run(bench, cells):
    print(f"\n### {bench}")
    h = (f"{'cell':<34} {'lang':<5} | {'SAME':>4} {'prog≠':>5} {'v≠':>3} | "
         f"{'len_old':>8} {'len_new':>8} {'Δmed':>7} | {'trunc_o':>7} {'trunc_n':>7}")
    print(h); print("-"*len(h))
    agg = {}
    for cell in cells:
        oc, nc = OLD/bench/cell, NEW/bench/cell
        if not nc.is_dir(): continue
        for lang in LANGS:
            og, orr = load(oc, lang); ng, nr = load(nc, lang)
            keys = sorted(set(og) & set(ng) & set(orr) & set(nr))
            if not keys: continue
            same = [k for k in keys if og[k] is not None and og[k] == ng[k]]
            prog_ne = sum(1 for k in same if orr[k][1] != nr[k][1])
            v_ne    = sum(1 for k in same if orr[k][0] != nr[k][0])
            lo = [len(og[k] or "") for k in keys]; ln = [len(ng[k] or "") for k in keys]
            # truncation proxy: reply ends without a closing fence and mid-token
            def trunc(cs): return sum(1 for k in keys if (cs[k] or "").count("```") % 2 == 1)
            to, tn = trunc(og), trunc(ng)
            print(f"{cell:<34} {lang:<5} | {len(same):>4} {prog_ne:>5} {v_ne:>3} | "
                  f"{st.median(lo):>8.0f} {st.median(ln):>8.0f} "
                  f"{st.median(ln)-st.median(lo):>+7.0f} | {to:>7} {tn:>7}")
            a = agg.setdefault(lang, [0,0,0,0,0])
            for i,v in enumerate((len(same), prog_ne, v_ne, to, tn)): a[i]+=v
    print("-"*len(h))
    for lang,(s,p,v,to,tn) in agg.items():
        print(f"{'POOLED':<34} {lang:<5} | {s:>4} {p:>5} {v:>3} | "
              f"{'':>8} {'':>8} {'':>7} | {to:>7} {tn:>7}")

run("qwen_suite/multipl_e_100",
    ["qwencodermpe_t10_q6k","qwen256e_q6k","qwencodermpe_q6k","qwenhybridp24_q6k"])
nb = NEW/"ream_arms/multipl_e_100"
if nb.is_dir():
    run("ream_arms/multipl_e_100", sorted(p.name for p in nb.iterdir() if p.is_dir()))
