#!/usr/bin/env python3
"""Generate CD-mix tensor-type maps.

CD-mix recipes ride on top of llama-quantize's file-base FTYPE heuristic
instead of overriding every tensor. The override file only lists body-tier
tensors (attn_q/k/output, ffn_gate/up, ffn_gate_up_exps). Load-bearing
tensors (attn_v, ffn_down, ffn_down_exps, token_embd, router) fall through
to the file-base heuristic, which protects them per llama-quant.cpp.

Motivation — v5-coder 2026-05-19 RCA:
  Plain Q4_K_M scored 92.07% HE+ at ~13.2 GB. CD-Q4_K_M scored ~88% at
  ~12.0 GB. The file-mix heuristic for Q4_K_M promotes ffn_down* to Q5_0/
  Q8_0 alternating + attn_v to Q6_K alternating; CD-Q4_K_M demoted both to
  Q3_K on 22 layers. CD-mix preserves the heuristic's protection of those
  tensors while CD's per-layer ranking drives the body.

Two families share the same heuristic-skip rules but differ at LOW tier:

  _L family   — LOW dropped one codebook below MID. Cheapest body, banks
                on the heuristic to protect attn_v + ffn_down + token_embd.
                The "safety-first" naming aligns with llama.cpp Q*_K_L
                semantics: heavier than _M because heuristic refuses to
                downgrade load-bearing tensors.

  _h     — LOW kept in same codebook class as MID (K-class for Q*,
                second-best IQ for IQ*). Matches the old full-override
                CD-* recipes' body protection while still riding the
                heuristic for attn_v + ffn_down + token_embd. Targets
                the 92-93% quality band at slightly heavier disk.

All entries use per-tensor-assignable ggml_types ONLY (no file-mix names
like IQ2_M / IQ3_M / IQ3_XS in the override file — those silently fall
through llama-quantize's parser).

Recipe family:

  CD-Q6_K_L:        file-base Q6_K     TOP=Q6_K    MID=Q5_K   LOW=IQ4_NL
  CD-Q5_K_M_L:      file-base Q5_K_M   TOP=Q6_K    MID=Q5_K   LOW=IQ4_XS
  CD-Q4_K_M_L:      file-base Q4_K_M   TOP=Q5_K    MID=Q4_K   LOW=IQ4_XS
  CD-IQ3_M_L:       file-base IQ3_M    TOP=Q5_K    MID=IQ4_XS LOW=IQ3_S
  CD-IQ3_XS_L:      file-base IQ3_XS   TOP=IQ4_XS  MID=IQ3_S  LOW=IQ3_XXS
  CD-IQ2_M_L:       file-base IQ2_M    TOP=IQ3_S   MID=IQ2_S  LOW=IQ2_XS
  CD-IQ2_XS_L:      file-base IQ2_XS   TOP=IQ3_S   MID=IQ2_XS LOW=IQ2_XXS

  CD-Q6_K_h:   file-base Q6_K     TOP=Q6_K    MID=Q5_K   LOW=Q5_K
  CD-Q5_K_M_h: file-base Q5_K_M   TOP=Q6_K    MID=Q5_K   LOW=Q4_K
  CD-Q4_K_M_h: file-base Q4_K_M   TOP=Q5_K    MID=Q4_K   LOW=Q4_K
  CD-IQ3_M_h:  file-base IQ3_M    TOP=Q5_K    MID=IQ4_XS LOW=IQ4_XS
  CD-IQ3_XS_h: file-base IQ3_XS   TOP=IQ4_XS  MID=IQ3_S  LOW=IQ3_S
  CD-IQ2_M_h:  file-base IQ2_M    TOP=IQ3_S   MID=IQ2_S  LOW=IQ2_S
  CD-IQ2_XS_h: file-base IQ2_XS   TOP=IQ3_S   MID=IQ2_XS LOW=IQ2_XS

Output: one tensor_types_<NAME>.txt per recipe in --out-dir. The companion
file_base.json maps each recipe name → its required `-q <FTYPE>` arg for
llama-quantize.

Legacy `_mix` suffix: scripts/old recipes that still pass --recipes
CD-Q*_mix get auto-mapped to the equivalent _L name with a stderr warning.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


META_SUFFIX = ".meta.json"

# Recipe → (TOP_1, MID_7, LOW_22) — all PER-TENSOR ggml_types only.
# These names show up in --tensor-type-file lines, so they MUST be valid
# parse_ggml_type() inputs. File-mix FTYPEs (IQ2_M, IQ3_M, IQ3_XS, Q3_K_M,
# etc.) are NOT valid here — they only work as `-q <ftype>` to llama-quantize.
CD_TIERS_MIX = {
    # _L family — heuristic-protected attn_v/ffn_down/token_embd,
    # body LOW dropped one codebook below MID for max disk savings.
    "CD-Q6_K_L":         ("Q6_K",   "Q5_K",   "IQ4_NL"),
    # CD-Q5_K_M_L deprecated 2026-05-19 for v5-coder: scored 90.85% vs old
    # CD-Q5_K_M's 92.68% at HEAVIER disk (13 vs 12.1 GB). Hybrid covers this
    # band; legacy alias kept for back-compat but excluded from defaults.
    "CD-Q5_K_M_L":       ("Q6_K",   "Q5_K",   "IQ4_XS"),
    "CD-Q4_K_M_L":       ("Q5_K",   "Q4_K",   "IQ4_XS"),
    "CD-IQ3_M_L":        ("Q5_K",   "IQ4_XS", "IQ3_S"),
    "CD-IQ3_XS_L":       ("IQ4_XS", "IQ3_S",  "IQ3_XXS"),
    "CD-IQ2_M_L":        ("IQ3_S",  "IQ2_S",  "IQ2_XS"),
    "CD-IQ2_XS_L":       ("IQ3_S",  "IQ2_XS", "IQ2_XXS"),
    # _h family — heuristic-skip same as _L, but LOW kept at the same
    # codebook class as MID. Matches the old full-override CD body
    # protection. Heavier than _L, targets the 92-93% HE+ quality band.
    "CD-Q6_K_h":    ("Q6_K",   "Q5_K",   "Q5_K"),
    "CD-Q5_K_M_h":  ("Q6_K",   "Q5_K",   "Q4_K"),
    "CD-Q4_K_M_h":  ("Q5_K",   "Q4_K",   "Q4_K"),
    "CD-IQ3_M_h":   ("Q5_K",   "IQ4_XS", "IQ4_XS"),
    "CD-IQ3_XS_h":  ("IQ4_XS", "IQ3_S",  "IQ3_S"),
    "CD-IQ2_M_h":   ("IQ3_S",  "IQ2_S",  "IQ2_S"),
    "CD-IQ2_XS_h":  ("IQ3_S",  "IQ2_XS", "IQ2_XS"),
}

# Each recipe needs llama-quantize -q <FTYPE> at quant time. The file-base
# heuristic is what protects attn_v + ffn_down* + token_embd + router.
FILE_BASE_FTYPE = {
    "CD-Q6_K_L":         "Q6_K",
    "CD-Q5_K_M_L":       "Q5_K_M",
    "CD-Q4_K_M_L":       "Q4_K_M",
    "CD-IQ3_M_L":        "IQ3_M",
    "CD-IQ3_XS_L":       "IQ3_XS",
    "CD-IQ2_M_L":        "IQ2_M",
    "CD-IQ2_XS_L":       "IQ2_XS",
    "CD-Q6_K_h":    "Q6_K",
    "CD-Q5_K_M_h":  "Q5_K_M",
    "CD-Q4_K_M_h":  "Q4_K_M",
    "CD-IQ3_M_h":   "IQ3_M",
    "CD-IQ3_XS_h":  "IQ3_XS",
    "CD-IQ2_M_h":   "IQ2_M",
    "CD-IQ2_XS_h":  "IQ2_XS",
}

# Legacy aliases (2026-05-19 rename): old _mix suffix → _L. Emits a stderr
# warning at resolve time so callers know to update.
LEGACY_ALIASES = {
    "CD-Q6_K_mix":    "CD-Q6_K_L",
    "CD-Q5_K_M_mix":  "CD-Q5_K_M_L",
    "CD-Q4_K_M_mix":  "CD-Q4_K_M_L",
    "CD-IQ3_M_mix":   "CD-IQ3_M_L",
    "CD-IQ3_XS_mix":  "CD-IQ3_XS_L",
    "CD-IQ2_M_mix":   "CD-IQ2_M_L",
    "CD-IQ2_XS_mix":  "CD-IQ2_XS_L",
}

# Tier sizes (Gemma 4 26B-A4B: 30 layers, fixed for now)
TOP_N, MID_N = 1, 7

# Body-only tensor roles that CD-mix overrides. Compare to the legacy
# BLOCK_TENSOR_ROLES in generate_cd_maps.py — these omit ffn_down*,
# ffn_gate_inp (router stays F32), and attn_v.weight (heuristic-protected).
# attn_k.weight IS included: in IQ3_M/IQ2_M file-mixes the heuristic does
# NOT promote attn_k, so CD's tier matters; in Q4_K_M+ file-mixes attn_k
# defaults to Q4_K which matches MID anyway.
BODY_TENSOR_ROLES_MIX = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_gate_up_exps.weight",
]

# Skipped tensors (NOT in override → fall through to file-base heuristic).
# Documented here for the .meta.json sidecar and for code readers.
HEURISTIC_PROTECTED_TENSORS = [
    "attn_v.weight",        # heuristic: Q4_K (IQ3_M) / Q5_K / Q6_K (Q4_K_M alt)
    "ffn_down.weight",      # heuristic: alternates Q5_0/Q8_0 (Q4_K_M)
    "ffn_down_exps.weight", # same
    "ffn_down_exps.scale",  # F32 anyway
    "ffn_gate_inp.weight",  # F32 (router)
    "ffn_gate_inp.scale",   # F32
    "token_embd.weight",    # heuristic: Q6_K
    "output.weight",        # tied in Gemma 4; heuristic Q6_K otherwise
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_imatrix_importance(imatrix_path: Path) -> tuple[dict[int, float], int]:
    sys.path.insert(0, "/opt/llama.cpp/gguf-py")
    from gguf import GGUFReader

    reader = GGUFReader(str(imatrix_path))
    in_sum2 = {}
    counts = {}
    for t in reader.tensors:
        name = t.name
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            in_sum2[base] = float(t.data.sum())
        elif name.endswith(".counts"):
            base = name[: -len(".counts")]
            counts[base] = float(t.data[0])

    layer_re = re.compile(r"^blk\.(\d+)\.")
    per_layer = defaultdict(float)
    max_layer = -1
    for base, s2 in in_sum2.items():
        c = counts.get(base, 1.0)
        if c <= 0:
            continue
        importance = s2 / c
        m = layer_re.match(base)
        if m:
            li = int(m.group(1))
            per_layer[li] += importance
            max_layer = max(max_layer, li)
    return dict(per_layer), max_layer + 1


def load_layer_importance_file(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "top98_mean_per_layer" in data:
        raw = data["top98_mean_per_layer"]
    elif isinstance(data, dict) and "floor_per_layer" in data and not any(
        isinstance(v, (int, float)) for v in data.values()
    ):
        raw = data["floor_per_layer"]
        print(f"  NOTE: using floor_per_layer from {path.name}", file=sys.stderr)
    elif isinstance(data, dict) and all(
        isinstance(v, (int, float)) for v in data.values()
    ):
        raw = data
    else:
        raise ValueError(f"{path}: unrecognized shape")
    return {int(k): float(v) for k, v in raw.items()}


def rank_layers(per_layer: dict[int, float], num_layers: int) -> list[int]:
    scores = [(li, per_layer.get(li, 0.0)) for li in range(num_layers)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [li for li, _ in scores]


def assign_tiers(ranking: list[int]) -> dict[int, str]:
    out = {}
    for rank, li in enumerate(ranking):
        if rank < TOP_N:
            out[li] = "top"
        elif rank < TOP_N + MID_N:
            out[li] = "mid"
        else:
            out[li] = "low"
    return out


def write_tensor_types_file(
    cd_name: str, tiers: dict[int, str], num_layers: int, out_dir: Path,
    imatrix_sha: str = "",
    importance_source: str = "imatrix",
    importance_file_sha: str | None = None,
    importance_file_name: str | None = None,
):
    top, mid, low = CD_TIERS_MIX[cd_name]
    tier_quant = {"top": top, "mid": mid, "low": low}
    out_path = out_dir / f"tensor_types_{cd_name}.txt"
    lines = []
    for li in range(num_layers):
        q = tier_quant[tiers[li]]
        for role in BODY_TENSOR_ROLES_MIX:
            lines.append(f"blk.{li}.{role}={q}")
    out_path.write_text("\n".join(lines) + "\n")
    if imatrix_sha:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        meta_path = out_path.with_name(out_path.name + META_SUFFIX)
        meta_path.write_text(json.dumps({
            "generated_from_imatrix_sha256": imatrix_sha,
            "importance_source": importance_source,
            "importance_file_sha256": importance_file_sha,
            "importance_file_name": importance_file_name,
            "generated_at": ts,
            "tier_profile": cd_name,
            "tiers": {"top": top, "mid": mid, "low": low},
            "file_base_ftype": FILE_BASE_FTYPE[cd_name],
            "body_tensor_roles": BODY_TENSOR_ROLES_MIX,
            "heuristic_protected_tensors": HEURISTIC_PROTECTED_TENSORS,
            "num_layers": num_layers,
            "mode": "mix",
        }, indent=2) + "\n")
    return out_path


def write_file_base_index(out_dir: Path, recipes: list[str]):
    """Emit file_base.json: recipe_name → -q FTYPE arg for llama-quantize."""
    idx = {name: FILE_BASE_FTYPE[name] for name in recipes}
    (out_dir / "file_base.json").write_text(
        json.dumps(idx, indent=2) + "\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", required=True,
                    help="Path to imatrix.dat (required — low-tier IQ*/Q2_K block "
                         "layouts need it at quantize time)")
    ap.add_argument("--layer-importance",
                    help="Optional layer-importance JSON; overrides imatrix ranking")
    ap.add_argument("--out-dir", default=".",
                    help="Output directory for tensor_types_*_mix.txt + file_base.json")
    ap.add_argument("--recipes", nargs="*", default=None,
                    help="Subset of recipe names to emit (default: all in CD_TIERS_MIX)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    imatrix_path = Path(args.imatrix)
    if not imatrix_path.exists():
        print(f"ERROR: {imatrix_path} not found", file=sys.stderr)
        sys.exit(1)

    imatrix_per_layer, num_layers = load_imatrix_importance(imatrix_path)
    if num_layers == 0:
        print("ERROR: no block-level tensors found in imatrix", file=sys.stderr)
        sys.exit(1)

    importance_file_sha = None
    importance_file_name = None
    if args.layer_importance:
        lif_path = Path(args.layer_importance)
        if not lif_path.exists():
            print(f"ERROR: --layer-importance {lif_path} not found", file=sys.stderr)
            sys.exit(1)
        per_layer = load_layer_importance_file(lif_path)
        importance_file_sha = sha256_of(lif_path)
        importance_file_name = lif_path.name
        importance_source = "layer-importance-file"
        if max(per_layer) >= num_layers:
            print(f"  WARNING: --layer-importance max layer {max(per_layer)} "
                  f"vs imatrix num_layers={num_layers}", file=sys.stderr)
        print(f"=== Per-layer importance from {lif_path.name} "
              f"(sha256={importance_file_sha[:12]}) ===")
    else:
        per_layer = imatrix_per_layer
        importance_source = "imatrix"
        print(f"=== Per-layer importance from imatrix (num_layers={num_layers}) ===")

    ranking = rank_layers(per_layer, num_layers)
    tiers = assign_tiers(ranking)

    max_score = max(per_layer.values()) if per_layer else 1.0
    print(f"{'rank':>4}  {'layer':>5}  {'tier':>4}  {'score':>15}  {'norm':>6}")
    for rank, li in enumerate(ranking):
        score = per_layer.get(li, 0.0)
        pct = 100 * score / max_score
        print(f"  {rank+1:>3}  L{li:<4}  {tiers[li]:>4}  {score:>15.4e}  {pct:>5.1f}%")

    print()
    print("=== Tier summary ===")
    top_layers = [li for li, t in tiers.items() if t == "top"]
    mid_layers = sorted([li for li, t in tiers.items() if t == "mid"])
    low_layers = sorted([li for li, t in tiers.items() if t == "low"])
    print(f"  top (1):  L{top_layers}")
    print(f"  mid (7):  L{mid_layers}")
    print(f"  low ({num_layers-TOP_N-MID_N}): L{low_layers}")

    # Recipes excluded from the default --recipes list (still valid if
    # explicitly named). CD-Q5_K_M_L is heavier + worse than old CD-Q5_K_M
    # on v5-coder (90.85% / 13 GB vs 92.68% / 12.1 GB) but may behave
    # differently on other model families — keep as opt-in.
    DEPRECATED_DEFAULTS = {"CD-Q5_K_M_L"}
    DEFAULT_RECIPES = [r for r in CD_TIERS_MIX if r not in DEPRECATED_DEFAULTS]

    raw_recipes = args.recipes or DEFAULT_RECIPES
    recipes = []
    for r in raw_recipes:
        if r in LEGACY_ALIASES:
            new = LEGACY_ALIASES[r]
            print(f"  NOTE: legacy alias '{r}' → '{new}'", file=sys.stderr)
            recipes.append(new)
        else:
            recipes.append(r)
    bad = [r for r in recipes if r not in CD_TIERS_MIX]
    if bad:
        print(f"ERROR: unknown recipes: {bad}", file=sys.stderr)
        print(f"       valid: {sorted(CD_TIERS_MIX)}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Recipes to emit ({len(recipes)}) ===")
    for r in recipes:
        top, mid, low = CD_TIERS_MIX[r]
        print(f"  {r:<18}  file-base={FILE_BASE_FTYPE[r]:<8}  "
              f"TOP={top:<8} MID={mid:<8} LOW={low:<8}")
    print()
    print(f"Body-only override roles: {BODY_TENSOR_ROLES_MIX}")
    print(f"Heuristic-protected (NOT overridden): {HEURISTIC_PROTECTED_TENSORS}")

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imatrix_sha = sha256_of(imatrix_path)
    print(f"\n=== Writing to {out_dir} ===")
    print(f"  imatrix sha256: {imatrix_sha[:12]}...")
    for cd_name in recipes:
        path = write_tensor_types_file(
            cd_name, tiers, num_layers, out_dir,
            imatrix_sha=imatrix_sha,
            importance_source=importance_source,
            importance_file_sha=importance_file_sha,
            importance_file_name=importance_file_name,
        )
        print(f"  wrote {path}")
    write_file_base_index(out_dir, recipes)
    print(f"  wrote {out_dir/'file_base.json'}")


if __name__ == "__main__":
    main()
