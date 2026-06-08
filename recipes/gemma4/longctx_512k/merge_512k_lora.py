#!/usr/bin/env python
# merge_512k_lora.py — SURGICAL merge of the 512k LoRA adapter onto the YaRN base.
#
# WHAT / WHY SURGICAL (not peft merge_and_unload):
#   The adapter touches ONLY q/k/o_proj on the 5 full-attention layers
#   [5,11,17,23,29] — 15 small matrices. Everything else (vision tower,
#   embeddings, MoE experts, the 25 sliding layers, the proportional_yarn
#   config, tokenizer, preprocessor) is BYTE-IDENTICAL to the YaRN base. So we
#   hardlink the entire base dir and rewrite ONLY the 2-3 safetensors shards
#   that hold the adapted weights, applying the exact peft LoRA math
#       W' = W + scaling · (B @ A),   scaling = lora_alpha / (sqrt(r) if rslora else r)
#   This:
#     * preserves the multimodal structure → the merged dir serves through vLLM
#       EXACTLY like the validated base RULER reference (ruler_ref_base_a4b.sh);
#     * avoids AutoModelForCausalLM silently dropping the vision tower on a
#       Gemma4 multimodal checkpoint (a text-only save would mismatch the config);
#     * is fast (~touch a few shards) and disk-cheap (hardlink the rest);
#     * keeps the canonical proportional_yarn config intact (faithful to training).
#   The cheap proportional_yarn→vLLM-"yarn" SERVE translation is a SEPARATE step
#   (make_serve_config.py) so a failed anchor gate costs seconds, not a re-merge.
#
# GROUND TRUTH for the LoRA math is adapter_config.json (r, lora_alpha,
# use_rslora). We READ it — never assume. Every base weight key is verified
# present in the base index before any write; any miss is FATAL (catches a
# target_modules prefix mismatch before corrupting a shard).
#
# Usage:
#   merge_512k_lora.py --base /srv/ml/longctx/yarn_cfg_98e \
#                      --adapter /srv/ml/longctx/ckpt_98e_ddp32k_fa2/ckpt-000952 \
#                      --out  /srv/ml/longctx/gemma-4-26B-A4B-it-512k [--dry-run]
#
# bf16 throughout; the rank-r update is accumulated in fp32 then cast back to the
# base dtype (bf16). Idempotent: writes a .merge_done marker; re-runs that see a
# complete marker for the same (adapter, base) no-op unless --force.

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
PEFT_PREFIX = "base_model.model."


def fatal(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def base_key_for(lora_module: str) -> str:
    """peft stores '<PEFT_PREFIX><module>.lora_{A,B}.weight'. The merged base
    tensor lives at '<module>.weight'. Strip the wrapper prefix if present."""
    m = lora_module
    if m.startswith(PEFT_PREFIX):
        m = m[len(PEFT_PREFIX):]
    return m + ".weight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="YaRN base dir (hardlinked from the 128e base; has model.safetensors.index.json)")
    ap.add_argument("--adapter", required=True, help="LoRA ckpt dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="merged output dir (created; weights hardlinked except rewritten shards)")
    ap.add_argument("--force", action="store_true", help="rebuild even if a matching .merge_done marker exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (scaling, targets, affected shards) and exit")
    args = ap.parse_args()

    base = Path(args.base)
    adapter = Path(args.adapter)
    out = Path(args.out)

    # --- preflight (FATAL-loud) ----------------------------------------------
    for p in (base, adapter):
        if not p.is_dir():
            fatal(f"missing dir {p}")
    acfg_path = adapter / "adapter_config.json"
    aw_path = adapter / "adapter_model.safetensors"
    idx_path = base / "model.safetensors.index.json"
    for p in (acfg_path, aw_path, idx_path):
        if not p.is_file():
            fatal(f"missing {p}")

    acfg = json.loads(acfg_path.read_text())
    r = int(acfg["r"])
    lora_alpha = float(acfg["lora_alpha"])
    use_rslora = bool(acfg.get("use_rslora", False))
    fan_in_fan_out = bool(acfg.get("fan_in_fan_out", False))
    if fan_in_fan_out:
        fatal("adapter has fan_in_fan_out=True — surgical merge assumes nn.Linear "
              "(B@A, no transpose). Refusing to guess; use peft merge_and_unload.")
    scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)

    # --- collect (module -> base_key) from the adapter weights ---------------
    with safe_open(str(aw_path), framework="pt") as f:
        akeys = list(f.keys())
    modules = {}  # lora_module -> {"A": key, "B": key}
    for k in akeys:
        if k.endswith(LORA_A_SUFFIX):
            mod = k[:-len(LORA_A_SUFFIX)]
            modules.setdefault(mod, {})["A"] = k
        elif k.endswith(LORA_B_SUFFIX):
            mod = k[:-len(LORA_B_SUFFIX)]
            modules.setdefault(mod, {})["B"] = k
        # ignore anything else (e.g. no modules_to_save expected for this adapter)
    if not modules:
        fatal("no lora_A/lora_B pairs found in adapter — nothing to merge")
    for mod, ab in modules.items():
        if "A" not in ab or "B" not in ab:
            fatal(f"module {mod} missing {'A' if 'A' not in ab else 'B'} half")

    # --- map each target's base weight to its shard --------------------------
    weight_map = json.loads(idx_path.read_text())["weight_map"]
    target_base_key = {}   # lora_module -> base_key
    shard_targets = {}     # shard filename -> [lora_module, ...]
    for mod in modules:
        bk = base_key_for(mod)
        if bk not in weight_map:
            fatal(f"base weight key {bk!r} (from adapter module {mod!r}) not in "
                  f"{idx_path.name} — target_modules prefix mismatch. Aborting "
                  f"before any write.")
        target_base_key[mod] = bk
        shard_targets.setdefault(weight_map[bk], []).append(mod)

    print("=== surgical LoRA merge plan ===")
    print(f"  base       : {base}")
    print(f"  adapter    : {adapter}")
    print(f"  out        : {out}")
    print(f"  r={r}  lora_alpha={lora_alpha}  use_rslora={use_rslora}  "
          f"scaling={scaling:.6g}")
    print(f"  targets    : {len(modules)} modules across {len(shard_targets)} shard(s)")
    for shard, mods in sorted(shard_targets.items()):
        print(f"     {shard}: {len(mods)} target(s)")

    # idempotency marker
    marker = out / ".merge_done"
    want = {"adapter": str(adapter), "base": str(base), "scaling": scaling,
            "n_targets": len(modules)}
    if marker.is_file() and not args.force:
        try:
            have = json.loads(marker.read_text())
        except Exception:
            have = {}
        if have.get("adapter") == want["adapter"] and have.get("base") == want["base"] \
           and int(have.get("n_targets", -1)) == want["n_targets"]:
            print(f"[idempotent] {marker} already records this exact merge — skipping. "
                  f"Use --force to rebuild.")
            return

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    # --- materialize out dir: hardlink everything except rewritten shards -----
    out.mkdir(parents=True, exist_ok=True)
    rewrite = set(shard_targets.keys())
    n_linked = 0
    skipped_dirs = []
    for f in base.iterdir():
        if f.is_dir():
            skipped_dirs.append(f.name)
            continue
        if f.name in rewrite:
            continue  # these get freshly written below
        tgt = out / f.name
        if tgt.exists():
            continue
        try:
            tgt.hardlink_to(f)
        except OSError:
            shutil.copy2(f, tgt)
        n_linked += 1
    print(f"  hardlinked {n_linked} base file(s); will write {len(rewrite)} shard(s)")
    if skipped_dirs:
        print(f"  skipped subdir(s): {', '.join(skipped_dirs)}")

    # --- load adapter tensors once -------------------------------------------
    adapter_sd = load_file(str(aw_path))

    # --- rewrite each affected shard -----------------------------------------
    for shard, mods in sorted(shard_targets.items()):
        src_shard = base / shard
        tensors = load_file(str(src_shard))
        for mod in mods:
            bk = target_base_key[mod]
            W = tensors[bk]
            A = adapter_sd[modules[mod]["A"]]   # [r, in]
            B = adapter_sd[modules[mod]["B"]]   # [out, r]
            delta = scaling * (B.float() @ A.float())   # [out, in]
            if delta.shape != tuple(W.shape):
                fatal(f"{mod}: delta shape {tuple(delta.shape)} != base {tuple(W.shape)}")
            tensors[bk] = (W.float() + delta).to(W.dtype)
            print(f"     merged {bk}  (|delta|_2={delta.norm().item():.4g}, "
                  f"|W|_2={W.float().norm().item():.4g})")
        # preserve safetensors metadata (format) on the rewritten shard
        with safe_open(str(src_shard), framework="pt") as f:
            meta = f.metadata() or {}
        save_file(tensors, str(out / shard), metadata=meta)
        print(f"  wrote {out / shard}")

    # --- provenance marker ----------------------------------------------------
    prov = dict(want)
    try:
        prov["adapter_meta"] = json.loads((adapter / "meta.json").read_text())
    except Exception:
        pass
    marker.write_text(json.dumps(prov, indent=2))
    print(f"[done] surgical merge complete → {out}")
    print(f"  marker: {marker}")
    print("  NOTE: config.json is hardlinked from the YaRN base (rope_type="
          "'proportional_yarn'). Run make_serve_config.py to derive the vLLM "
          "'yarn' serve dir.")


if __name__ == "__main__":
    main()
