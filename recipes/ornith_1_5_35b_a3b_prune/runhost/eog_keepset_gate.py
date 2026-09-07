#!/usr/bin/env python
"""EOG keepset gate — refuse an arm whose keep set is depleted of terminator experts.

WHY THIS EXISTS
  A pruned MoE can lose the experts that carry routing mass at end-of-generation
  positions. When it does, the model fails to terminate cleanly: it overruns, emits a
  malformed thinking channel, and the server's chat parser rejects the response. The
  eval then records an EMPTY completion which scores as a plain wrong answer, so the
  damage is invisible in the score and looks like a quality regression.

  This was a real Ornith incident (2026-09-07): CoderX returned 20/77 empty completions
  via llama-server HTTP 500 (`common_chat_peg_parse`), while Coder on the identical
  binary, template and flags returned 0. See README.md.

WHAT IT MEASURES
  lift[L][e] = (share of routing weight e receives at EOG-predicting positions)
             / (share it receives at all other positions)
  Laplace-smoothed, identical to eog_arm_correlation.py so the two are commensurable.
  A lift > 1 means the expert is preferentially recruited when the model is about to
  stop. Ranked WITHIN a layer (rank01 in [0,1]) so a globally-hot expert cannot top the
  list merely by being hot everywhere -- that confound is what made the raw T202
  emit-mass ranking unusable.

WHAT IT GATES ON
  mean rank01 of EOG-lift over each arm's keep set, versus a reference arm. An arm that
  sheds terminator experts scores lower. --max-drop sets how much regression is
  tolerated before the gate REFUSES (exit 20).

  NOTE ON PRIOR EVIDENCE: the same cross-check on Qwen3.6-35B-A3B (2026-08-20) found NO
  EOG-lift signal separating p12 from p24. This gate is therefore a CHECK, not a
  believed mechanism -- it is cheap, it is model-specific, and a negative result is a
  legitimate and useful outcome. Do not read a passing gate as proof of termination
  health, only as absence of this particular defect.
"""
import argparse, json, sys
import numpy as np


def load_map(path):
    m = json.load(open(path))
    for k in ("n_experts", "n_layers", "emit_weight", "bg_weight"):
        if k not in m:
            sys.exit("eog map %s missing key %r" % (path, k))
    return m


def lift_rank01(m, alpha=1.0):
    E = int(m["n_experts"])
    ew = np.asarray(m["emit_weight"], dtype=float)
    bw = np.asarray(m["bg_weight"], dtype=float)
    es = (ew + alpha) / (ew.sum(1, keepdims=True) + alpha * E)
    bs = (bw + alpha) / (bw.sum(1, keepdims=True) + alpha * E)
    lift = es / bs
    rank01 = np.argsort(np.argsort(lift, axis=1), axis=1) / (E - 1.0)
    return lift, rank01


def keepsets(drop_map_path, E):
    d = json.load(open(drop_map_path))
    out = {}
    allE = set(range(E))
    for k, v in d.items():
        if not str(k).isdigit():
            continue            # 'mtp' and any other non-layer key
        out[int(k)] = sorted(allE - set(int(x) for x in v))
    return out


def score(rank01, ks):
    per_layer = {}
    for L, keep in ks.items():
        if L >= rank01.shape[0]:
            continue
        per_layer[L] = float(rank01[L, keep].mean())
    mean = float(np.mean(list(per_layer.values()))) if per_layer else float("nan")
    return mean, per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eog-map", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=path/to/drop_map.json (repeatable)")
    ap.add_argument("--reference", default=None,
                    help="arm name to compare against; default = first --arm")
    ap.add_argument("--max-drop", type=float, default=0.02,
                    help="refuse if an arm's mean EOG-lift rank falls more than this "
                         "below the reference (absolute, rank01 units). Default 0.02.")
    ap.add_argument("--top-evicted", type=int, default=8,
                    help="how many highest-lift evicted experts to list per worst layer")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 20 when the gate refuses (otherwise report only)")
    a = ap.parse_args()

    m = load_map(a.eog_map)
    E = int(m["n_experts"])
    lift, rank01 = lift_rank01(m)
    print("EOG map: n_experts=%d n_layers=%d emit_pos=%s bg_pos=%s"
          % (E, int(m["n_layers"]), m.get("n_emit_positions", "?"),
             m.get("n_bg_positions", "?")))

    arms = {}
    for spec in a.arm:
        if "=" not in spec:
            sys.exit("--arm expects NAME=path, got %r" % spec)
        name, path = spec.split("=", 1)
        arms[name] = keepsets(path, E)

    ref = a.reference or list(arms)[0]
    if ref not in arms:
        sys.exit("--reference %r is not one of %s" % (ref, list(arms)))

    results = {}
    for name, ks in arms.items():
        mean, per_layer = score(rank01, ks)
        results[name] = {"mean_eog_lift_rank": mean, "per_layer": per_layer}

    ref_mean = results[ref]["mean_eog_lift_rank"]
    print("\nmean EOG-lift rank01 over keep set (higher = retains more terminator mass)")
    print("  %-28s %8s   %s" % ("arm", "mean", "delta vs " + ref))
    worst = None
    for name in arms:
        mean = results[name]["mean_eog_lift_rank"]
        delta = mean - ref_mean
        results[name]["delta_vs_reference"] = delta
        flag = ""
        if name != ref and delta < -a.max_drop:
            flag = "  <-- REGRESSION"
            if worst is None or delta < results[worst]["delta_vs_reference"]:
                worst = name
        print("  %-28s %8.4f   %+8.4f%s" % (name, mean, delta, flag))

    # Where does the regression live, and which experts did it shed?
    if worst is not None:
        wk = arms[worst]
        rk = arms[ref]
        rows = []
        for L in sorted(set(wk) & set(rk)):
            if L >= rank01.shape[0]:
                continue
            rows.append((results[worst]["per_layer"][L] - results[ref]["per_layer"][L], L))
        rows.sort()
        print("\nworst layers for %s (delta vs %s):" % (worst, ref))
        for delta, L in rows[:5]:
            evicted = sorted(set(rk[L]) - set(wk[L]), key=lambda e: -lift[L, e])
            top = [(int(e), round(float(lift[L, e]), 3)) for e in evicted[:a.top_evicted]]
            print("  L%-3d %+7.4f  highest-lift experts evicted: %s" % (L, delta, top))

    verdict = "PASS" if worst is None else "REFUSE"
    print("\nEOG_KEEPSET_GATE %s (max_drop=%.3f, reference=%s)" % (verdict, a.max_drop, ref))

    if a.json_out:
        json.dump({"reference": ref, "max_drop": a.max_drop, "verdict": verdict,
                   "arms": {k: {"mean_eog_lift_rank": v["mean_eog_lift_rank"],
                                "delta_vs_reference": v.get("delta_vs_reference", 0.0)}
                            for k, v in results.items()}},
                  open(a.json_out, "w"), indent=2)
        print("wrote %s" % a.json_out)

    if worst is not None and a.fail_on_regression:
        sys.exit(20)


if __name__ == "__main__":
    main()
