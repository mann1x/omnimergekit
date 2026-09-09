#!/usr/bin/env python3
"""Render the tool-eval-bench cohort as a dot-and-whisker chart.

Reads the same cells as summarize_cohort.py, so the denominator and balance
guards are shared rather than reimplemented.

    OMK_TB_OUT=/path/to/results python eval/toolbench/plot_cohort.py [-o out.png]

Only the BALANCED seed set is plotted. Cells graded on <176 (a scenario dropped
to an infrastructure timeout) are marked, because their raw points are not
directly comparable to a full cell.
"""
import os, re, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("OMK_TB_OUT", os.environ.get("OMK_TB_OUT", ""))
import summarize_cohort as sc

# Published r/LocalLLaMA values. DIFFERENT BASIS (256k ctx, harness v2.6.0,
# no MTP, other quants) -- shown only as a ghost tick for ordering context.
# NEVER pooled with our numbers.
THREAD = {
    "qwen3.8-27b": 152.6,
    "ornith-1.5-35b": 144.2,
    "qwen3.6-27b": 134.8,
    "qwen3.6-35b-a3b": 131.5,
}
# The harness these numbers come from. Taken from the clone's own `git remote`
# (both /shared/dev/tool-eval-bench and the opencoti clone agree), so the credit
# on the chart names the actual tool that produced the cells.
BENCH_URL = "github.com/SeraphimSerapis/tool-eval-bench"

FAMILY = {  # colour by provenance
    "omnimerge-v4": "ours", "omnimerge-v6": "ours",
    "a3b-coder": "ours", "a3b-coderx": "ours",
    # Ornith-1.5-27B-A3B Coder/CoderX are in-house builds (ManniX-ITA), added
    # 2026-09-09. NOTE ornith-1.5-35b is deliberately NOT here: that row is
    # bartowski's quant of a third-party model, carried as an external anchor.
    "ornith-27b-coder": "ours", "ornith-27b-coderx": "ours",
}


def served_quant(W, model):
    """The quant tier actually SERVED for `model`, read from its own server log.

    Never inferred from parameter count: the chart used to carry a blanket
    "27B Q4_K_M / 35B IQ4_XS" subtitle, which silently mislabels any cell whose
    file differs (qwen3.8-27b is UD-Q4_K_M, not plain Q4_K_M). Returns "" when the
    log is missing or names no gguf, so an unknown quant renders as no label
    rather than a guessed one.
    """
    import glob as _glob
    path = os.path.join(W, f"{model}.server.log")
    if not os.path.exists(path):
        cand = _glob.glob(os.path.join(W, f"{model}.server*.log"))
        if not cand:
            return ""
        path = cand[0]
    try:
        blob = open(path, errors="replace").read()
    except OSError:
        return ""
    names = re.findall(r"[A-Za-z0-9._/-]+\.gguf", blob)
    for nm in names:
        m = re.search(r"((?:UD-)?(?:IQ|Q)\d+(?:_[A-Z0-9]+)*)\.gguf$", nm.split("/")[-1])
        if m:
            return m.group(1)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="toolbench_scores.png")
    ap.add_argument("--title", default="Tool-calling benchmark — hardmode (88 scenarios, 176 pts)")
    a = ap.parse_args()

    W = sc.results_dir()
    cells = sc.load_cells(W)
    if not cells:
        sys.exit("no parsable cells in " + W)
    common, ragged = sc.balanced(cells)
    if not common:
        sys.exit("no balanced seed set yet")
    rows = sc.summarize(cells, common)
    n = len(common)

    # A model the driver has attempted but that has no balanced cell yet would
    # otherwise disappear from the chart with no explanation. Recover the intended
    # roster from the driver log and report the gap.
    roster = set()
    dl = os.path.join(W, "driver.log")
    if os.path.exists(dl):
        import re as _re
        roster = set(_re.findall(r"== ([A-Za-z0-9._-]+) \(", open(dl, errors="replace").read()))
    pending = sorted(roster - set(cells)) if roster else []

    C = {"ours": "#c2410c", "base": "#334155", "excl": "#b45309"}
    names = [r[1] for r in rows][::-1]          # best at top
    means = [r[0] for r in rows][::-1]
    halfs = [r[3] for r in rows][::-1]
    excl = [r[4] for r in rows][::-1]
    cols = [C["ours"] if FAMILY.get(m) == "ours" else C["base"] for m in names]

    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(names) + 2.5), dpi=160)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    y = range(len(names))

    for i, (m, mu, h, c) in enumerate(zip(names, means, halfs, cols)):
        if h > 0:
            ax.plot([mu - h, mu + h], [i, i], color=c, lw=2.2, alpha=.45,
                    solid_capstyle="round", zorder=2)
            for e in (mu - h, mu + h):
                ax.plot([e, e], [i - .13, i + .13], color=c, lw=1.6, alpha=.55, zorder=2)
        ax.plot(mu, i, "o", ms=8.5, color=c, zorder=4,
                markeredgecolor="white", markeredgewidth=1.4)
        t = THREAD.get(m)
        if t is not None:
            ax.plot(t, i, "|", ms=13, color="#94a3b8", mew=2.0, zorder=3)

    # Anchor each value at the RIGHT END of everything drawn on its row (the upper
    # CI cap, or the r/LocalLLaMA ghost tick when that sits further right), not at
    # a fixed offset from the mean: with a fixed offset a wide interval slides its
    # own cap underneath the text, which is why "157.5" and "150.0" used to be
    # struck through by their whisker.
    for i, (m, mu, h, e) in enumerate(zip(names, means, halfs, excl)):
        lab = f"{mu:.1f}" + (f"  ±{h:.1f}" if h > 0 else "")
        if e: lab += "  *"
        anchor = mu + h
        t = THREAD.get(m)
        if t is not None:
            anchor = max(anchor, t)
        ax.annotate(lab, (anchor, i), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color="#0f172a",
                    fontfamily="DejaVu Sans Mono")

    # Model name with its SERVED quant beneath it (smaller, grey). Drawn as two
    # texts rather than one tick label because a tick label cannot carry two
    # sizes/colours. get_yaxis_transform: x in axes fraction, y in data coords.
    quants = [served_quant(W, m) for m in names]
    ax.set_yticks(list(y)); ax.set_yticklabels([])
    _tr = ax.get_yaxis_transform()
    for _i, (_m, _q) in enumerate(zip(names, quants)):
        ax.text(-0.012, _i + (0.17 if _q else 0.0), _m, transform=_tr,
                ha="right", va="center", fontsize=10.5, color="#0f172a")
        if _q:
            ax.text(-0.012, _i - 0.19, _q, transform=_tr, ha="right", va="center",
                    fontsize=8.3, color="#94a3b8")
    ax.set_xlabel("Total Points (max 176)", fontsize=10.5, color="#334155")
    lo = min(min(means) - max(halfs) - 8, min(THREAD.values()) - 8)
    ax.set_xlim(lo, 176)
    ax.axvline(176, color="#cbd5e1", lw=1, ls=":")
    ax.grid(axis="x", color="#e2e8f0", lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", length=0)

    ax.set_title(a.title, fontsize=13.5, fontweight="bold",
                 color="#0f172a", loc="left", pad=26)
    sub = (f"n={n} paired seeds {sorted(common)} · 64k ctx · context-pressure 0.25 · "
           f"greedy-off (temp 0.6/top-p 0.95/top-k 20)")
    ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 12),
                textcoords="offset points", fontsize=8.8, color="#64748b")

    handles = [
        Line2D([], [], marker="o", ls="", color=C["ours"], ms=8, label="ours (Omnimerge / A3B / Ornith-27B)"),
        Line2D([], [], marker="o", ls="", color=C["base"], ms=8, label="vendor / third-party base"),
        Line2D([], [], marker="|", ls="", color="#94a3b8", ms=11, mew=2,
               label="r/LocalLLaMA published (256k, v2.6.0 — NOT comparable)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.6)

    notes = []
    if any(excl): notes.append("* scenario(s) dropped on a 120s request timeout (infrastructure, not model failure); "
                               "plotted value counts them as 0, so it is a LOWER BOUND")
    if n < 2:     notes.append("n=1 — no confidence interval yet")
    if ragged:    notes.append("in-flight cells excluded to keep the cohort balanced")
    if pending:   notes.append("awaiting cells (re-running, not yet plotted): " + ", ".join(pending))
    if notes:
        fig.text(.012, .012, "  ·  ".join(notes), fontsize=8.2, color="#94a3b8")
    # Benchmark credit, bottom-right. Same baseline as the notes on the left.
    fig.text(.988, .012, BENCH_URL, fontsize=8.2, color="#94a3b8", ha="right")

    fig.tight_layout()
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print(f"wrote {a.out}  ({len(names)} models, n={n}, seeds={sorted(common)})")


if __name__ == "__main__":
    main()
