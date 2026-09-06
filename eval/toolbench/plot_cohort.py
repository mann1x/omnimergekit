#!/usr/bin/env python3
"""Render the tool-eval-bench cohort as a dot-and-whisker chart.

Reads the same cells as summarize_cohort.py, so the denominator and balance
guards are shared rather than reimplemented.

    OMK_TB_OUT=/path/to/results python eval/toolbench/plot_cohort.py [-o out.png]

Only the BALANCED seed set is plotted. Cells graded on <176 (a scenario dropped
to an infrastructure timeout) are marked, because their raw points are not
directly comparable to a full cell.
"""
import os, sys, argparse
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
FAMILY = {  # colour by provenance
    "omnimerge-v4": "ours", "omnimerge-v6": "ours",
    "a3b-coder": "ours", "a3b-coderx": "ours",
}


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

    for i, (mu, h, e) in enumerate(zip(means, halfs, excl)):
        lab = f"{mu:.1f}" + (f"  ±{h:.1f}" if h > 0 else "")
        if e: lab += "  *"
        ax.annotate(lab, (mu, i), xytext=(11, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color="#0f172a",
                    fontfamily="DejaVu Sans Mono")

    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10.5)
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
           f"greedy-off (temp 0.6/top-p 0.95/top-k 20) · 27B Q4_K_M / 35B IQ4_XS")
    ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 12),
                textcoords="offset points", fontsize=8.8, color="#64748b")

    handles = [
        Line2D([], [], marker="o", ls="", color=C["ours"], ms=8, label="ours (Omnimerge / A3B)"),
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

    fig.tight_layout()
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print(f"wrote {a.out}  ({len(names)} models, n={n}, seeds={sorted(common)})")


if __name__ == "__main__":
    main()
