#!/usr/bin/env python3
"""
Drop least-contributing experts from Qwen3.6-35B-A3B (packed MoE, hybrid linear-attn,
+ shared expert + native MTP head). Adapted from scripts/expert_drop.py (Gemma-4 26B-A4B),
which already handles the same packed `experts.{gate_up,down}_proj` layout.

Deltas vs the Gemma version (verified against the 256e base tensor headers):
  main experts : model.language_model.layers.N.mlp.experts.gate_up_proj [256,1024,2048] -> [K,...]
                 model.language_model.layers.N.mlp.experts.down_proj    [256,2048, 512] -> [K,...]
  router (gate): model.language_model.layers.N.mlp.gate.weight          [256,2048]      -> [K,2048]
                 (Qwen uses mlp.gate.weight; NO router.proj / per_expert_scale)
  MTP head     : mtp.layers.0.mlp.experts.{gate_up,down}_proj + mtp.layers.0.mlp.gate.weight
                 sliced too (config num_experts is shared) via drop_map["mtp"] (default = layer 0's set)
  PASS THROUGH : shared_expert.*, shared_expert_gate, linear_attn.*, mtp.fc/norm/pre_fc_*,
                 vision tower, embeddings, lm_head, norms  (keep vision + nextn verbatim)

Drop map JSON: {"0":[dropped ids], "1":[...], ..., "mtp":[...]}  (all main layers drop the same count).

Usage:
  python expert_drop_qwen35b.py --source-dir /srv/ml/models/Qwen3.6-35B-A3B \
      --drop-map recipes/qwen3_6_35b_a3b_prune/results/drop_map_184e.json \
      --output-dir /srv/ml/models/Qwen3.6-35B-A3B-184e [--dry-run]
"""
import argparse, json, re, shutil
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

RE_EXPERT     = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
RE_GATE       = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
RE_MTP_EXPERT = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
RE_MTP_GATE   = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.gate\.weight$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--drop-map", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate slicing + report shapes/param counts; write nothing.")
    return ap.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    with open(args.drop_map) as f:
        raw = json.load(f)
    drop_map = {int(k): v for k, v in raw.items() if k != "mtp"}
    mtp_drop = raw.get("mtp", drop_map[0])

    with open(source_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts_orig = text_cfg["num_experts"]
    drop_count = len(drop_map[0])
    target_experts = num_experts_orig - drop_count

    print(f"Source:   {source_dir}")
    print(f"Experts:  {num_experts_orig} -> {target_experts}  (drop {drop_count}/layer, "
          f"{100*drop_count/num_experts_orig:.1f}%)")
    print(f"Layers:   {num_layers}  + MTP head (drop {len(mtp_drop)})")

    keep_map = {}
    for li in range(num_layers):
        drop_set = set(drop_map[li])
        keep_map[li] = sorted(set(range(num_experts_orig)) - drop_set)
        assert len(keep_map[li]) == target_experts, \
            f"Layer {li}: expected {target_experts} keep, got {len(keep_map[li])}"
    mtp_keep = sorted(set(range(num_experts_orig)) - set(mtp_drop))
    assert len(mtp_keep) == target_experts, f"MTP: expected {target_experts}, got {len(mtp_keep)}"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = source_dir.parent / f"Qwen3.6-35B-A3B-{target_experts}e{args.suffix}"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:   {output_dir}{'  [DRY-RUN]' if args.dry_run else ''}")

    with open(source_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    def slice_expert(key, tensor):
        m = RE_EXPERT.match(key)
        if m:   return tensor[keep_map[int(m.group(1))]], "expert"
        m = RE_GATE.match(key)
        if m:   return tensor[keep_map[int(m.group(1))]], "router"
        if RE_MTP_EXPERT.match(key): return tensor[mtp_keep], "mtp-expert"
        if RE_MTP_GATE.match(key):   return tensor[mtp_keep], "mtp-router"
        return tensor, None

    # dry-run: validate a representative set, report shapes + projected size
    if args.dry_run:
        probe = ["model.language_model.layers.0.mlp.experts.gate_up_proj",
                 "model.language_model.layers.0.mlp.experts.down_proj",
                 "model.language_model.layers.0.mlp.gate.weight",
                 "mtp.layers.0.mlp.experts.gate_up_proj",
                 "mtp.layers.0.mlp.gate.weight"]
        opened = {}
        for k in probe:
            sh = weight_map[k]
            opened.setdefault(sh, safe_open(str(source_dir / sh), framework="pt", device="cpu"))
            t = opened[sh].get_tensor(k)
            nt, kind = slice_expert(k, t)
            print(f"  [{kind}] {k}\n      {tuple(t.shape)} -> {tuple(nt.shape)}")
        # projected total params (all tensors)
        counts = defaultdict(int)
        tot_new = 0
        for sh, keys in shard_keys.items():
            sf = safe_open(str(source_dir / sh), framework="pt", device="cpu")
            for k in keys:
                t = sf.get_tensor(k)
                nt, kind = slice_expert(k, t)
                counts[kind or "keep"] += 1
                tot_new += nt.numel()
        print(f"  tensor classes: {dict(counts)}")
        print(f"  projected total params: {tot_new/1e9:.2f} B  (target ~26 B)")
        print("  DRY-RUN OK — no files written.")
        return

    # real run: stream shards, slice, re-shard at 5GB
    new_weight_map, current_shard, current_size, shard_idx = {}, {}, 0, 1
    max_shard = int(5 * 1024**3); total_size = 0; n_sliced = 0
    for shard_name in tqdm(sorted(shard_keys), desc="Processing"):
        sf = safe_open(str(source_dir / shard_name), framework="pt", device="cpu")
        for key in shard_keys[shard_name]:
            tensor, kind = slice_expert(key, sf.get_tensor(key))
            if kind: n_sliced += 1
            tsz = tensor.numel() * tensor.element_size()
            if current_size + tsz > max_shard and current_shard:
                nm = f"model-{shard_idx:05d}.safetensors"
                save_file(current_shard, str(output_dir / nm))
                for k in current_shard: new_weight_map[k] = nm
                shard_idx += 1; current_shard = {}; current_size = 0
            current_shard[key] = tensor; current_size += tsz; total_size += tsz
    if current_shard:
        nm = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(output_dir / nm))
        for k in current_shard: new_weight_map[k] = nm
    for oi in range(1, shard_idx + 1):
        old = output_dir / f"model-{oi:05d}.safetensors"
        new = output_dir / f"model-{oi:05d}-of-{shard_idx:05d}.safetensors"
        old.rename(new)
        for k, v in list(new_weight_map.items()):
            if v == f"model-{oi:05d}.safetensors": new_weight_map[k] = new.name

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, f, indent=2)
    new_config = json.loads(json.dumps(config))
    new_config["text_config"]["num_experts"] = target_experts
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)
    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "processor_config.json", "special_tokens_map.json",
               "preprocessor_config.json", "video_preprocessor_config.json", "vocab.json", "merges.txt"]:
        src = source_dir / fn
        if src.exists(): shutil.copy2(src, output_dir / fn)
    print(f"Done: {n_sliced} tensors sliced, total {total_size/1e9:.1f} GB bf16 -> {output_dir}")


if __name__ == "__main__":
    main()
