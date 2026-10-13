#!/usr/bin/env python3
"""rank_agentic_eog_lift.py — rank terminator experts by LIFT, not by raw mass.

WHY THIS EXISTS. `rank_agentic_eog_pins.py` ranks "highest emit-position RMS wnorm".
That is raw mass, and it is confounded: an expert that is hot on EVERY token tops the
emit-position table simply by being hot, not by having anything to do with stopping. The
same defect was found and fixed in the Qwen3.6 emit-map on 2026-08-20, where raw-mass
ranking produced a mechanism hypothesis that did not survive its cross-check; the fix was
to build a background arm and score

    lift(layer, expert) = emit-position routing share / background routing share

Both arms come from the SAME corpus and the SAME role filter, differing only in
--emit-mode, so the denominator is matched by construction.

A lift of 1.0 means "this expert fires at terminators exactly as often as it fires
anywhere else" — i.e. it carries no terminator-specific signal at all, however hot it is.
"""
from __future__ import annotations

import argparse
import json


def load(path, key):
    d = json.load(open(path))
    cats = d.get("categories", {})
    if key not in cats:
        raise SystemExit("%s has no category %r (has: %s)"
                         % (path, key, list(cats)))
    return cats[key], d.get("metadata", {})


def shares(cat, field):
    """{layer: [share per expert]} — routing share within each layer."""
    out = {}
    for layer, rows in cat.items():
        vals = [float(r.get(field, 0.0)) for r in rows]
        tot = sum(vals)
        out[layer] = [v / tot if tot > 0 else 0.0 for v in vals]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eog-map", required=True, help="map built with --emit-mode emit")
    ap.add_argument("--bg-map", required=True, help="map built with --emit-mode background")
    ap.add_argument("--field", default="tc",
                    help="tc (selection count) or wnorm. tc is the honest one for a "
                         "'which experts are used' question; wnorm mixes in magnitude.")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--min-tc", type=int, default=8,
                    help="Floor on EMIT-arm tc. Without it an expert selected twice with "
                         "near-zero background gets an enormous, meaningless lift.")
    ap.add_argument("--out", default=None, help="write a --force-keep string here")
    args = ap.parse_args()

    emit, m_e = load(args.eog_map, "agentic_eog")
    bg, m_b = load(args.bg_map, "agentic_eog_bg")
    se, sb = shares(emit, args.field), shares(bg, args.field)

    for lab, m in (("emit", m_e), ("bg  ", m_b)):
        print("[%s] token=%r role=%s mode=%s stats=%s"
              % (lab, m.get("agentic_eog_emit_token"), m.get("agentic_eog_emit_role"),
                 m.get("agentic_eog_emit_mode"), m.get("agentic_eog_stats")))
    if (m_e.get("agentic_eog_emit_token_id") != m_b.get("agentic_eog_emit_token_id")
            or m_e.get("agentic_eog_corpus") != m_b.get("agentic_eog_corpus")):
        raise SystemExit("REFUSING: the two arms differ in terminator or corpus — a lift "
                         "computed across mismatched arms is meaningless.")

    pins, rows_out = [], []
    agree = 0
    layers = sorted(emit, key=lambda x: int(x) if str(x).isdigit() else x)
    for layer in layers:
        rows = emit[layer]
        cand = []
        for e in range(len(rows)):
            tc = float(rows[e].get("tc", 0.0))
            if tc < args.min_tc:
                continue
            b = sb[layer][e]
            # An expert with zero background but real emit mass is maximally specific;
            # rank it above everything with a finite denominator rather than dividing
            # by an epsilon and inventing a number.
            lift = float("inf") if b == 0 else se[layer][e] / b
            cand.append((lift, e, tc))
        cand.sort(key=lambda x: (-x[0], -x[2]))
        top_lift = [e for _, e, _ in cand[:args.topk]]
        raw = sorted(range(len(rows)), key=lambda e: -float(rows[e].get(args.field, 0.0)))
        top_raw = raw[:args.topk]
        agree += len(set(top_lift) & set(top_raw))
        rows_out.append((layer, top_lift, top_raw,
                         [("inf" if c[0] == float("inf") else round(c[0], 2))
                          for c in cand[:args.topk]]))
        pins += ["%s:%d" % (layer, e) for e in top_lift]

    print("\n%-6s %-26s %-26s %s" % ("layer", "top-%d by LIFT" % args.topk,
                                     "top-%d by RAW %s" % (args.topk, args.field), "lift"))
    for layer, tl, tr, lv in rows_out:
        print("%-6s %-26s %-26s %s" % (layer, ",".join(map(str, tl)),
                                       ",".join(map(str, tr)), lv))

    denom = max(args.topk * len(layers), 1)
    print("\nRAW-vs-LIFT agreement: %d/%d = %.1f%% of the top-%d picks are the same expert."
          % (agree, denom, 100.0 * agree / denom, args.topk))
    print("Low agreement means the raw-mass ranker (rank_agentic_eog_pins.py) was mostly "
          "selecting globally-hot experts, not terminator-specific ones.")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(",".join(pins) + "\n")
        print("\nwrote %d pins -> %s" % (len(pins), args.out))


if __name__ == "__main__":
    main()
