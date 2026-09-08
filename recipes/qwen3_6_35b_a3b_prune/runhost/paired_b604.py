#!/usr/bin/env python3
"""Paired attribution of the MPE b604 re-run.

The re-run regenerated as well as re-extracted, so the raw old->new score delta
confounds two effects. This isolates them by PAIRING on the problem and
partitioning by whether the COMPLETION is byte-identical:

  stratum SAME : old completion == new completion.  Generation is held fixed by
                 construction, so ANY verdict change here is caused by the
                 EXTRACTOR alone. This is the true bug-604 blast radius.
  stratum DIFF : the model produced different text. Verdict changes here are
                 uninterpretable -- extractor and generation both moved.

A delta may only be attributed to bug-604 if it lives in the SAME stratum.
"""
import json, sys
from pathlib import Path

OLD = Path("/srv/ml/eval_results")
NEW = Path("/srv/ml/eval_results_b604")
LANGS = ["rs", "java", "js"]


def comps(cell, lang):
    d = cell / "generations" / f"humaneval-{lang}"
    out = {}
    if not d.is_dir():
        return out
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        c = j.get("completions") or []
        out[f.stem] = c[0] if c else None
    return out


def verdicts(cell, lang):
    d = cell / "results" / f"humaneval-{lang}"
    out = {}
    if not d.is_dir():
        return out
    for f in d.glob("*.results.json"):
        if f.name == "_summary.json":
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        rs = j.get("results") or []
        if not rs:
            continue
        out[f.name[:-len(".results.json")]] = (rs[0].get("status") == "OK")
    return out


def main(bench_rel, cells):
    print(f"### {bench_rel}")
    hdr = (f"{'cell':<34} {'lang':<5} {'n':>3} | {'SAME':>4} {'f>p':>4} {'p>f':>4} "
           f"{'Δsame':>6} | {'DIFF':>4} {'f>p':>4} {'p>f':>4} {'Δdiff':>6} | {'Δtot':>6}")
    print(hdr); print("-" * len(hdr))
    tot = {}
    for cell in cells:
        o_cell, n_cell = OLD / bench_rel / cell, NEW / bench_rel / cell
        if not n_cell.is_dir():
            continue
        for lang in LANGS:
            oc, nc = comps(o_cell, lang), comps(n_cell, lang)
            ov, nv = verdicts(o_cell, lang), verdicts(n_cell, lang)
            keys = sorted(set(oc) & set(nc) & set(ov) & set(nv))
            if not keys:
                continue
            s_n = s_fp = s_pf = d_n = d_fp = d_pf = 0
            for k in keys:
                same = oc[k] is not None and oc[k] == nc[k]
                a, b = ov[k], nv[k]
                if same:
                    s_n += 1
                    if b and not a: s_fp += 1
                    elif a and not b: s_pf += 1
                else:
                    d_n += 1
                    if b and not a: d_fp += 1
                    elif a and not b: d_pf += 1
            n = len(keys)
            ds, dd = 100.0*(s_fp-s_pf)/n, 100.0*(d_fp-d_pf)/n
            print(f"{cell:<34} {lang:<5} {n:>3} | {s_n:>4} {s_fp:>4} {s_pf:>4} {ds:>+6.1f} | "
                  f"{d_n:>4} {d_fp:>4} {d_pf:>4} {dd:>+6.1f} | {ds+dd:>+6.1f}")
            t = tot.setdefault(lang, [0,0,0,0,0,0,0])
            for i, v in enumerate((n, s_n, s_fp, s_pf, d_n, d_fp, d_pf)):
                t[i] += v
    print("-" * len(hdr))
    for lang, (n, s_n, s_fp, s_pf, d_n, d_fp, d_pf) in tot.items():
        print(f"{'POOLED':<34} {lang:<5} {n:>3} | {s_n:>4} {s_fp:>4} {s_pf:>4} "
              f"{100.0*(s_fp-s_pf)/n:>+6.1f} | {d_n:>4} {d_fp:>4} {d_pf:>4} "
              f"{100.0*(d_fp-d_pf)/n:>+6.1f} |")
    print()


if __name__ == "__main__":
    main("qwen_suite/multipl_e_100",
         ["qwencodermpe_t10_q6k", "qwen256e_q6k", "qwencodermpe_q6k", "qwenhybridp24_q6k"])
    nb = NEW / "ream_arms/multipl_e_100"
    if nb.is_dir():
        main("ream_arms/multipl_e_100", sorted(p.name for p in nb.iterdir() if p.is_dir()))
