#!/usr/bin/env python3
"""Aggregate the tool-eval-bench 64k cohort into a reportable table.

Guards the two comparability traps:
  1. MIXED DENOMINATORS — a cell with an infrastructure timeout is graded on
     <176 points. Raw Total Points across mixed denominators is NOT a column.
     We report raw, plus `adj` = points/176 (excluded scenario scored 0), and
     flag every affected cell.
  2. Never average across seeds unless every model has the same seed set.
"""
import re, os, glob, statistics, sys

def results_dir():
    d = os.environ.get("OMK_TB_OUT")
    if not d:
        sys.exit("OMK_TB_OUT must point at the results directory")
    return d
T95 = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}

def parse(cell):
    md = sorted(glob.glob(os.path.join(cell,"**","*.md"), recursive=True))
    if not md: return None
    t = open(md[-1], errors="replace").read()
    m = re.search(r"\*\*Total Points\*\*:\s*(\d+)\s*/\s*(\d+)", t)
    if not m: return None
    pts, mx = int(m.group(1)), int(m.group(2))
    exc = re.search(r"(\d+)\s+scenario\(s\) excluded", t)
    which = re.findall(r"5xx\)[^`]*`([^`]+)`", t)
    return dict(pts=pts, mx=mx, excluded=int(exc.group(1)) if exc else 0,
                which=which[0] if which else "")

def load_cells(W):
    """{model: {seed: {pts, mx, excluded, which}}} for every parsable cell."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(W, "*-s*"))):
        name, seed = os.path.basename(d).rsplit("-s", 1)
        r = parse(d)
        if r:
            cells.setdefault(name, {})[int(seed)] = r
    return cells


def balanced(cells):
    """Seeds every model shares, plus the ragged extras that must NOT be pooled."""
    if not cells:
        return set(), {}
    sets = {n: set(v) for n, v in cells.items()}
    common = set.intersection(*sets.values())
    ragged = {m: sorted(s - common) for m, s in sets.items() if s - common}
    return common, ragged


def summarize(cells, common):
    """[(mean, name, sd, half_ci, n_excluded, mixed_denominator)] for balanced seeds."""
    out = []
    for name, byseed in cells.items():
        raw = [byseed[s]["pts"] for s in sorted(common)]
        ex = sum(byseed[s]["excluded"] for s in sorted(common))
        mixed = len({byseed[s]["mx"] for s in sorted(common)}) > 1
        n = len(raw)
        mean = statistics.mean(raw)
        if n >= 2:
            sd = statistics.stdev(raw)
            half = T95.get(n, 2.0) * sd / (n ** 0.5)
        else:
            sd = half = 0.0
        out.append((mean, name, sd, half, ex, mixed))
    return sorted(out, reverse=True)


W = results_dir()
cells = load_cells(W)

if not cells: print("no cells yet"); sys.exit()
common, ragged = balanced(cells)
n = len(common)
print(f"BALANCED across all {len(cells)} models at n={n}  seeds={sorted(common)}")
if ragged: print(f"  (extra unbalanced cells, NOT pooled: {ragged})")

anyexc = False
print(f"\n  {'model':<18}{'mean raw':>10}{'95% CI':>18}{'mean adj/176':>14}{'excl':>6}")
print("  " + "-"*68)
rows=[]
for name, byseed in cells.items():
    raw = [byseed[s]["pts"] for s in sorted(common)]
    adj = [byseed[s]["pts"]/176*176 for s in sorted(common)]   # points are absolute
    ex  = sum(byseed[s]["excluded"] for s in sorted(common))
    mx  = {byseed[s]["mx"] for s in sorted(common)}
    anyexc |= ex > 0
    mean = statistics.mean(raw)
    if n >= 2:
        sd = statistics.stdev(raw); half = T95.get(n,2.0)*sd/(n**0.5)
        ci = f"[{mean-half:6.1f},{mean+half:6.1f}]"
    else:
        ci = "     (n=1)      "
    rows.append((mean, name, ci, ex, mx))
for mean, name, ci, ex, mx in sorted(rows, reverse=True):
    flag = f"  {'MIXED' if len(mx)>1 else ''}"
    print(f"  {name:<18}{mean:>10.1f}{ci:>18}{mean/176*100:>13.1f}%{ex:>6}{flag}")
if anyexc:
    print("\n  ** Cells with excluded scenarios are graded on <176. Raw points are NOT")
    print("     directly comparable there; the adj column divides by the full 176.")
    for name, byseed in cells.items():
        for s in sorted(common):
            c = byseed[s]
            if c["excluded"]:
                print(f"     {name} s{s}: {c['pts']}/{c['mx']} excluded={c['excluded']} ({c['which']})")
