#!/usr/bin/env python3
"""Generate CD-IQ2_NL tensor-type maps for the STD16 (v7-coder force-keep) cohort.

Layer ranking combines BOTH signals the user asked for ("contribdynamic AND imatrix"):
  --rank-source blend  [default] : z-score average of the imatrix activation importance
                                   AND the main-map per-layer importance (top98_mean_per_layer)
  --rank-source main             : main-map per-layer importance only
  --rank-source imatrix          : raw imatrix activation magnitude only
(both readers come from generate_cd_maps: load_imatrix_importance / load_layer_importance_file.)

Tiers (INVERTED vs the stock generator so the BULK is 3-bit, per the request):
  - top-1 most-important layer  -> IQ4_NL  (the IQ4_NL base the user anchors on)
  - bulk  (next BULK_N layers)  -> IQ3_S   (best VALID per-tensor 3-bit i-quant codebook;
                                            IQ3_M / IQ3_XS are file-mixes, NOT per-tensor)
  - tail  (least-important)     -> <TAIL>  (IQ2_S best 2-bit i-quant, or Q2_K safe sibling)

Anti-rumination protection (#562/#563 — low-bit i-quant on attn AND ffn_down is the
confirmed v7 loop trigger):
  - attn_v / attn_k / attn_q / attn_output : IQ4_NL on EVERY layer
  - router ffn_gate_inp (.weight + .scale)  : IQ4_NL on EVERY layer (routing-collapse guard)
  - ffn_down (.weight + _exps.weight + _exps.scale) : FLOORED at IQ3_S, never the 2-bit tail
  - ONLY ffn_gate / ffn_up / ffn_gate_up_exps reach the 2-bit tail (where the param mass is)
  - token_embd + output : Q8_0
Emits one tensor_types_*.txt per tail codebook + a provenance .meta.json.
"""
import sys
import json
import argparse
import datetime
import statistics
from pathlib import Path

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from generate_cd_maps import (  # noqa: E402
    load_imatrix_importance,
    load_layer_importance_file,
    rank_layers,
    sha256_of,
)

GATE_UP_ROLES = ["ffn_gate.weight", "ffn_up.weight", "ffn_gate_up_exps.weight"]  # -> tier quant
DOWN_ROLES = ["ffn_down.weight", "ffn_down_exps.weight"]                          # -> IQ3_S floor
ATTN_ROLES = ["attn_v.weight", "attn_k.weight", "attn_q.weight", "attn_output.weight"]
PROTECT_Q = "IQ4_NL"   # attn + router
DOWN_FLOOR = "IQ3_S"   # ffn_down never below this
TOP_Q, BULK_Q = "IQ4_NL", "IQ3_S"


def _z(d):
    vals = list(d.values())
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}


def emit(ranking, num_layers, top_n, bulk_n, tail_q, out_path):
    tier = {}
    for r, li in enumerate(ranking):
        tier[li] = "top" if r < top_n else ("bulk" if r < top_n + bulk_n else "tail")
    lines = []
    for li in range(num_layers):
        t = tier[li]
        gq = TOP_Q if t == "top" else (BULK_Q if t == "bulk" else tail_q)
        dq = TOP_Q if t == "top" else DOWN_FLOOR          # ffn_down floored at IQ3_S
        for role in GATE_UP_ROLES:
            lines.append(f"blk.{li}.{role}={gq}")
        for role in DOWN_ROLES:
            lines.append(f"blk.{li}.{role}={dq}")
        for role in ATTN_ROLES:
            lines.append(f"blk.{li}.{role}={PROTECT_Q}")
        lines.append(f"blk.{li}.ffn_gate_inp.weight={PROTECT_Q}")   # router protect
        lines.append(f"blk.{li}.ffn_gate_inp.scale={PROTECT_Q}")
        lines.append(f"blk.{li}.ffn_down_exps.scale={dq}")
    lines.append("token_embd.weight=q8_0")
    lines.append("output.weight=q8_0")
    out_path.write_text("\n".join(lines) + "\n")
    return tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", required=True,
                    help="imatrix.dat (activation importance + needed at build time)")
    ap.add_argument("--layer-importance", required=True,
                    help="main-map per-layer JSON (top98_mean_per_layer / floor_per_layer)")
    ap.add_argument("--rank-source", choices=["blend", "main", "imatrix"], default="blend",
                    help="blend = z-avg(main, imatrix) [default]; main / imatrix select one")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--bulk-n", type=int, default=21)   # tail = num_layers - top - bulk (=8 @ 30L)
    args = ap.parse_args()

    imat_per_layer, num_layers = load_imatrix_importance(Path(args.imatrix))
    main_per_layer = load_layer_importance_file(Path(args.layer_importance))
    n_layers = max(num_layers, max(main_per_layer) + 1)

    if args.rank_source == "imatrix":
        per_layer = imat_per_layer
    elif args.rank_source == "main":
        per_layer = main_per_layer
    else:
        zi, zm = _z(imat_per_layer), _z(main_per_layer)
        per_layer = {li: zi.get(li, 0.0) + zm.get(li, 0.0) for li in range(n_layers)}
    ranking = rank_layers(per_layer, n_layers)

    tail_n = n_layers - args.top_n - args.bulk_n
    if tail_n < 1:
        print(f"ERROR: tail_n={tail_n} (top {args.top_n} + bulk {args.bulk_n} >= {n_layers})",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imsha = sha256_of(Path(args.imatrix))
    lisha = sha256_of(Path(args.layer_importance))
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    for name, tail_q in (("CD-IQ2_NL", "IQ2_S"), ("CD-IQ2_NL-q2k", "Q2_K")):
        p = out_dir / f"tensor_types_{name}.txt"
        tier = emit(ranking, n_layers, args.top_n, args.bulk_n, tail_q, p)
        tail_layers = sorted(li for li, t in tier.items() if t == "tail")
        meta = {
            "generated_from_imatrix_sha256": imsha,
            "importance_source": f"rank-source={args.rank_source}",
            "importance_file_name": Path(args.layer_importance).name,
            "importance_file_sha256": lisha,
            "generated_at": ts,
            "tier_profile": name,
            "recipe": {"top": TOP_Q, "bulk": BULK_Q, "tail": tail_q,
                       "ffn_down_floor": DOWN_FLOOR, "attn": PROTECT_Q, "router": PROTECT_Q},
            "split": {"top_n": args.top_n, "bulk_n": args.bulk_n, "tail_n": tail_n},
            "num_layers": n_layers, "tail_layers": tail_layers,
        }
        (out_dir / f"tensor_types_{name}.txt.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  wrote {p.name:34s} tail={tail_q:6s} tail_layers={tail_layers}")
    print(f"rank-source={args.rank_source}  ranking most->least important: {ranking}")
    print("CD_IQ2NL_MAPS_DONE")


if __name__ == "__main__":
    main()
