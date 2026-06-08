#!/usr/bin/env python3
"""imatrix_expert_coverage.py — audit per-expert routing coverage of an imatrix.

Canonical pre-quantization gate for any pruned/merged MoE. i-quants (IQ2_*/IQ3_*)
assign a fixed lattice codebook to each weight block, weighted by the imatrix's
per-expert importance statistics. If the calibration corpus under-routes some
experts **in a given layer**, those experts' per-layer importance is near-zero,
the codebook assignment is unconditioned, and the resulting quant degenerates
(stop-token collapse → rumination) at inference. Crucially this is a *per-layer*
problem: an expert can be globally well-covered yet starved in specific layers
(the corpus routes it elsewhere). k-quants are immune (per-block affine), so this
only bites the i-quant tiers — exactly where it's least expected.

This tool reads the per-expert `.counts` tensors an imatrix records for MoE expert
blocks and reports the routing distribution + how many (expert, layer) pairs are
starved, so you can decide whether to rebuild the calibration corpus before
committing to i-quant tiers. Arch-agnostic: discovers expert dimension and layer
set from the GGUF itself.

Exit status: 0 if no layer exceeds --max-starved-frac starved experts; 2 if the
imatrix is starved (gate fail); 1 on error / no per-expert counts found.

Origin: 2026-06-06 RCA of Gemma-4 26B-A4B 98e i-quant rumination — middle layers
(16-22) starved ~25-30/98 experts under a code-heavy corpus while the aggregate
looked clean. See docs/quantization.md.
"""
import argparse
import json
import sys

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("FATAL: `gguf` not importable — run under the omnimergekit env "
             "(/srv/ml/envs/envs/omnimergekit/bin/python or equivalent).")


def _counts_vec(tensor):
    """Flatten an imatrix count tensor to a 1-D per-expert vector.

    Handles the shapes current llama.cpp emits: dense tensors carry a scalar
    count ([1]); MoE expert tensors carry one count per expert ([n_expert] or
    [n_expert, 1]). Returns None for the scalar/dense case (no per-expert info).
    """
    v = np.asarray(tensor.data).reshape(-1).astype(np.float64)
    return v if v.size > 1 else None


def audit(path, max_starved_frac=0.20, starve_layer_frac=0.20):
    r = GGUFReader(path)
    # Per-expert count tensors live on MoE expert blocks: name contains "exps"
    # and ends with ".counts". Group by tensor (each is one layer's expert set).
    count_tensors = [t for t in r.tensors
                     if "exps" in t.name and t.name.lower().endswith("counts")
                     and _counts_vec(t) is not None]
    if not count_tensors:
        return None  # not a MoE imatrix, or counts not recorded by this builder

    per_layer = {t.name: _counts_vec(t) for t in count_tensors}
    n_expert = max(len(v) for v in per_layer.values())
    # aggregate across layers (only same-width vectors contribute to the sum)
    agg = np.zeros(n_expert, dtype=np.float64)
    for v in per_layer.values():
        if len(v) == n_expert:
            agg += v

    a = np.sort(agg)
    mean = a.mean() or 1.0

    def pct(p):
        return float(a[min(len(a) - 1, int(p * len(a)))])

    layer_rows = []
    for name, v in per_layer.items():
        m = v.mean() or 1.0
        starved = int((v < starve_layer_frac * m).sum())
        zero = int((v == 0).sum())
        layer_rows.append({"tensor": name, "n_expert": int(len(v)),
                           "starved": starved, "zero": zero,
                           "min": float(v.min()), "mean": float(m)})
    worst_frac = max(row["starved"] / row["n_expert"] for row in layer_rows)
    return {
        "imatrix": path,
        "n_expert": int(n_expert),
        "layer_count_tensors": len(per_layer),
        "total_routed": float(agg.sum()),
        "aggregate": {"min": float(a[0]), "p10": pct(.1), "p50": pct(.5),
                      "p90": pct(.9), "max": float(a[-1]), "mean": float(mean)},
        "aggregate_starved": {
            "never_routed": int((agg == 0).sum()),
            "lt_5pct_mean": int((agg <= 0.05 * mean).sum()),
            "lt_20pct_mean": int((agg <= 0.20 * mean).sum()),
        },
        "worst_layer_starved_frac": worst_frac,
        "layers": sorted(layer_rows, key=lambda x: -x["starved"]),
        "gate_pass": worst_frac <= max_starved_frac,
        "max_starved_frac": max_starved_frac,
        "starve_layer_frac": starve_layer_frac,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imatrix", help="path to imatrix.dat (GGUF imatrix)")
    ap.add_argument("--max-starved-frac", type=float, default=0.20,
                    help="gate fails if any layer has more than this fraction of "
                         "experts starved (default 0.20)")
    ap.add_argument("--starve-layer-frac", type=float, default=0.20,
                    help="an expert is 'starved' in a layer if its count is below "
                         "this fraction of that layer's mean (default 0.20)")
    ap.add_argument("--top", type=int, default=8,
                    help="show this many worst layers (default 8)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()

    res = audit(args.imatrix, args.max_starved_frac, args.starve_layer_frac)
    if res is None:
        print(f"[imatrix-coverage] no per-expert .counts tensors in {args.imatrix} "
              "— not a MoE imatrix, or counts not recorded by this llama.cpp build.",
              file=sys.stderr)
        sys.exit(1)

    if args.json:
        res_out = dict(res)
        res_out["layers"] = res["layers"][:args.top]
        print(json.dumps(res_out, indent=2))
        sys.exit(0 if res["gate_pass"] else 2)

    ag = res["aggregate"]
    print(f"imatrix: {res['imatrix']}")
    print(f"experts/layer={res['n_expert']}  expert-count tensors={res['layer_count_tensors']}  "
          f"total routed={res['total_routed']:,.0f}")
    print(f"aggregate per-expert count: min={ag['min']:,.0f} p10={ag['p10']:,.0f} "
          f"p50={ag['p50']:,.0f} p90={ag['p90']:,.0f} max={ag['max']:,.0f}")
    asv = res["aggregate_starved"]
    print(f"aggregate starved: never_routed={asv['never_routed']} "
          f"<5%mean={asv['lt_5pct_mean']} <20%mean={asv['lt_20pct_mean']}  (all-layers pooled)")
    print(f"\nworst {args.top} layers (experts < {res['starve_layer_frac']:.0%} of layer mean):")
    for row in res["layers"][:args.top]:
        print(f"  {row['tensor']:50s} starved={row['starved']:3d}/{row['n_expert']} "
              f"zero={row['zero']:3d} min={row['min']:,.0f} mean={row['mean']:,.0f}")
    verdict = "PASS" if res["gate_pass"] else "FAIL"
    print(f"\nGATE {verdict}: worst-layer starved fraction "
          f"{res['worst_layer_starved_frac']:.1%} vs threshold {res['max_starved_frac']:.0%}")
    if not res["gate_pass"]:
        print("  → rebuild the calibration corpus to route every expert per layer "
              "before building i-quant (IQ2_*/IQ3_*) tiers, or restrict to k-quants.")
    sys.exit(0 if res["gate_pass"] else 2)


if __name__ == "__main__":
    main()
