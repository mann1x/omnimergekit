#!/bin/bash
# Pod setup + parallel GPQA Diamond run on a vast.ai 2x3090 instance.
#
# This script is fully self-contained — does not depend on any files
# from solidpc. It:
#   1. Installs build deps + builds llama.cpp from source with CUDA
#   2. Installs Python deps (lm_eval, huggingface_hub, transformers)
#   3. Downloads 128e from HF, runs expert_drop to create 120e v3,
#      quantizes to Q6_K, cleans up intermediates
#   4. Downloads 109e from HF, quantizes to Q6_K, cleans up
#   5. Starts 2 llama-server instances (one per GPU)
#   6. Runs 2 lm_eval gpqa_diamond_cot_zeroshot in parallel with the
#      locked methodology (temp=1.0/top_p=0.95/top_k=64/seed=42 via
#      --gen_kwargs override and llama-server CLI defaults)
#   7. Saves results to ~/eval_results/{120e_v3,109e}/
#
# Disk peak: ~106 GB. Fits in 150 GB with margin.
# Wall time: ~50 min setup + ~11h GPQA = ~12h total.

set -euo pipefail

WORK="/workspace"
mkdir -p "$WORK" && cd "$WORK"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in env before running this script}"
export HUGGINGFACE_TOKEN="$HF_TOKEN"
export HF_HOME="$WORK/.hf_cache"
mkdir -p "$HF_HOME"

echo "===== pod setup start: $(date) ====="
echo "  workdir: $WORK"
echo "  HF cache: $HF_HOME"

#==============================================================
# 1. Build deps + llama.cpp
#==============================================================
echo
echo "===== installing build deps ====="
apt-get update -qq && apt-get install -y -qq cmake git build-essential curl wget pigz

echo
echo "===== cloning llama.cpp ====="
if [[ ! -d /opt/llama.cpp ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp
fi
cd /opt/llama.cpp
mkdir -p build && cd build
echo
echo "===== building llama.cpp with CUDA ====="
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build . --config Release -j$(nproc) --target llama-server llama-quantize 2>&1 | tail -5
echo "  llama-server built"

#==============================================================
# 2. Python deps
#==============================================================
echo
echo "===== installing Python deps ====="
pip install --quiet --upgrade pip
pip install --quiet huggingface_hub transformers safetensors tqdm requests datasets
# Use lm-eval main branch for latest gpqa task config
pip install --quiet "lm-eval[api] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"

cd "$WORK"

#==============================================================
# 3. HF login
#==============================================================
huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true

#==============================================================
# 4. Embed expert_drop.py and the v3 drop map
#==============================================================
mkdir -p scripts
cat > scripts/hybrid_120e_drop_map.json <<'DROPMAP_EOF'
{
  "0": [
    17,
    30,
    37,
    40,
    42,
    57,
    66,
    88
  ],
  "1": [
    3,
    17,
    31,
    43,
    45,
    58,
    61,
    121
  ],
  "2": [
    11,
    27,
    31,
    42,
    72,
    81,
    90,
    119
  ],
  "3": [
    7,
    32,
    47,
    48,
    87,
    89,
    95,
    108
  ],
  "4": [
    0,
    4,
    6,
    11,
    14,
    16,
    21,
    95
  ],
  "5": [
    3,
    12,
    23,
    33,
    67,
    74,
    77,
    85
  ],
  "6": [
    15,
    17,
    31,
    34,
    36,
    40,
    67,
    75
  ],
  "7": [
    4,
    6,
    15,
    36,
    38,
    52,
    95,
    110
  ],
  "8": [
    11,
    22,
    25,
    35,
    40,
    55,
    59,
    89
  ],
  "9": [
    9,
    12,
    60,
    65,
    91,
    97,
    110,
    113
  ],
  "10": [
    5,
    7,
    11,
    13,
    48,
    68,
    101,
    126
  ],
  "11": [
    19,
    28,
    29,
    44,
    47,
    62,
    116,
    119
  ],
  "12": [
    23,
    31,
    34,
    62,
    65,
    89,
    110,
    111
  ],
  "13": [
    3,
    5,
    7,
    17,
    20,
    23,
    51,
    127
  ],
  "14": [
    11,
    41,
    61,
    84,
    88,
    92,
    104,
    120
  ],
  "15": [
    3,
    8,
    10,
    26,
    50,
    92,
    102,
    107
  ],
  "16": [
    1,
    9,
    19,
    34,
    60,
    87,
    98,
    112
  ],
  "17": [
    6,
    14,
    16,
    25,
    31,
    36,
    58,
    100
  ],
  "18": [
    11,
    26,
    63,
    65,
    76,
    79,
    113,
    117
  ],
  "19": [
    0,
    4,
    8,
    13,
    16,
    17,
    95,
    116
  ],
  "20": [
    0,
    13,
    31,
    37,
    76,
    86,
    107,
    125
  ],
  "21": [
    0,
    3,
    11,
    29,
    59,
    71,
    78,
    125
  ],
  "22": [
    1,
    3,
    7,
    12,
    15,
    68,
    125,
    127
  ],
  "23": [
    6,
    14,
    15,
    16,
    35,
    83,
    94,
    96
  ],
  "24": [
    15,
    41,
    61,
    73,
    80,
    81,
    91,
    118
  ],
  "25": [
    24,
    38,
    69,
    71,
    73,
    90,
    96,
    122
  ],
  "26": [
    13,
    27,
    30,
    41,
    54,
    62,
    74,
    80
  ],
  "27": [
    1,
    32,
    37,
    65,
    96,
    116,
    118,
    122
  ],
  "28": [
    26,
    37,
    54,
    55,
    66,
    97,
    101,
    105
  ],
  "29": [
    16,
    23,
    24,
    31,
    41,
    50,
    73,
    120
  ]
}
DROPMAP_EOF

cat > scripts/expert_drop.py <<'EXPERTDROP_EOF'
#!/usr/bin/env python3
"""
Drop least-contributing experts from Gemma 4 26B-A4B based on contribution analysis.
Uses per-layer drop maps to remove experts and remap router weights.

For each layer:
  - gate_up_proj: [128, 1408, 2816] → [N, 1408, 2816]  (keep only N)
  - down_proj: [128, 2816, 704] → [N, 2816, 704]
  - router.proj.weight: [128, 2816] → [N, 2816]
  - router.per_expert_scale: [128] → [N]

Usage:
  python expert_drop.py                                      # legacy default: 109e from eval_results/expert_drop_map_109.json (cwd=128e dir)
  python expert_drop.py --drop-map scripts/hybrid_120e_drop_map.json --suffix -hybrid
  python expert_drop.py --source-dir google/gemma-4-26B-A4B-it --drop-map scripts/hybrid_120e_drop_map.json --suffix -hybrid
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=str, default=".",
                    help="path to base model dir (default: cwd)")
    ap.add_argument("--drop-map", type=str, default=None,
                    help="path to drop map JSON (default: <source>/eval_results/expert_drop_map_109.json)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="output dir (default: <source>.parent/gemma-4-A4B-{N}e{suffix})")
    ap.add_argument("--suffix", type=str, default="",
                    help="suffix for default output name, e.g. -hybrid")
    return ap.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source_dir).resolve()
    drop_map_file = Path(args.drop_map) if args.drop_map else (source_dir / "eval_results" / "expert_drop_map_109.json")
    drop_map_file = drop_map_file.resolve()

    with open(drop_map_file) as f:
        drop_map = {int(k): v for k, v in json.load(f).items()}

    with open(source_dir / "config.json") as f:
        config = json.load(f)

    text_cfg = config["text_config"]
    num_layers = text_cfg["num_hidden_layers"]
    num_experts_orig = text_cfg["num_experts"]
    target_experts = num_experts_orig - len(drop_map[0])  # all layers drop same count

    print(f"Source:    {source_dir}")
    print(f"Drop map:  {drop_map_file}")
    print(f"Experts:   {num_experts_orig} → {target_experts}")
    print(f"Layers:    {num_layers}")

    # Build per-layer keep indices (sorted for consistent ordering)
    keep_map = {}
    for li in range(num_layers):
        drop_set = set(drop_map[li])
        keep_map[li] = sorted(set(range(num_experts_orig)) - drop_set)
        assert len(keep_map[li]) == target_experts, f"Layer {li}: expected {target_experts} keep, got {len(keep_map[li])}"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = source_dir.parent / f"gemma-4-A4B-{target_experts}e{args.suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output:    {output_dir}")

    # Load index
    with open(source_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Open all shard files
    shard_files = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_files[shard_name] = safe_open(str(source_dir / shard_name), framework="pt", device="cpu")

    # Group keys by shard
    shard_keys = defaultdict(list)
    for key, shard in weight_map.items():
        shard_keys[shard].append(key)

    # Process
    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    max_shard_bytes = int(5 * 1024**3)  # 5GB shards
    total_size = 0
    n_expert_tensors = 0
    n_router_tensors = 0

    for shard_name in tqdm(sorted(shard_keys.keys()), desc="Processing"):
        sf = shard_files[shard_name]
        for key in shard_keys[shard_name]:
            tensor = sf.get_tensor(key)

            # Expert stacked weights: gate_up_proj or down_proj
            m_expert = re.match(
                r"model\.language_model\.layers\.(\d+)\.experts\.(gate_up_proj|down_proj)",
                key
            )
            if m_expert:
                layer_idx = int(m_expert.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128,...] → [109,...]
                n_expert_tensors += 1

            # Router proj.weight
            m_router_proj = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.proj\.weight",
                key
            )
            if m_router_proj:
                layer_idx = int(m_router_proj.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128, hidden] → [109, hidden]
                n_router_tensors += 1

            # Router per_expert_scale
            m_router_scale = re.match(
                r"model\.language_model\.layers\.(\d+)\.router\.per_expert_scale",
                key
            )
            if m_router_scale:
                layer_idx = int(m_router_scale.group(1))
                keep_ids = keep_map[layer_idx]
                tensor = tensor[keep_ids]  # [128] → [109]
                n_router_tensors += 1

            # Write to shard
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > max_shard_bytes and current_shard:
                sf_name = f"model-{shard_idx:05d}.safetensors"
                save_file(current_shard, str(output_dir / sf_name))
                for k in current_shard:
                    new_weight_map[k] = sf_name
                shard_idx += 1
                current_shard = {}
                current_size = 0

            current_shard[key] = tensor
            current_size += tensor_size
            total_size += tensor_size

    # Final shard
    if current_shard:
        sf_name = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(output_dir / sf_name))
        for k in current_shard:
            new_weight_map[k] = sf_name

    # Rename shards with total count
    for old_idx in range(1, shard_idx + 1):
        old_name = output_dir / f"model-{old_idx:05d}.safetensors"
        new_name = output_dir / f"model-{old_idx:05d}-of-{shard_idx:05d}.safetensors"
        old_name.rename(new_name)
        for k, v in new_weight_map.items():
            if v == f"model-{old_idx:05d}.safetensors":
                new_weight_map[k] = new_name.name

    # Close source files
    for sf in shard_files.values():
        del sf

    # Write index
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Update config
    new_config = json.loads(json.dumps(config))
    new_config["text_config"]["num_experts"] = target_experts
    with open(output_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy non-weight files
    for fn in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json", "processor_config.json",
               "special_tokens_map.json", "preprocessor_config.json"]:
        src = source_dir / fn
        if src.exists():
            shutil.copy2(src, output_dir / fn)

    # Save metadata
    meta = {
        "base_model": "google/gemma-4-26B-A4B-it",
        "method": "expert_drop_by_contribution",
        "drop_map_file": str(drop_map_file),
        "original_experts": num_experts_orig,
        "target_experts": target_experts,
        "per_layer_keep": {str(li): ids for li, ids in keep_map.items()},
        "per_layer_drop": {str(li): ids for li, ids in drop_map.items()},
    }
    with open(output_dir / "expert_drop_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Expert tensors pruned: {n_expert_tensors}")
    print(f"  Router tensors pruned: {n_router_tensors}")
    print(f"  Total size: {total_size / 1024**3:.1f} GB ({total_size / 2 / 1e9:.1f}B params @ bf16)")
    print(f"  Shards: {shard_idx}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
EXPERTDROP_EOF

#==============================================================
# 5a. Download 109e Q6_K GGUF directly (no surgery needed —
#     ManniX-ITA already published the quantization)
#==============================================================
echo
echo "===== downloading 109e Q6_K GGUF directly from HF ====="
huggingface-cli download ManniX-ITA/gemma-4-A4B-109e-it-GGUF \
    gemma-4-A4B-109e-it-Q6_K.gguf \
    --local-dir . \
    --max-workers 4 2>&1 | tail -5
mv -f gemma-4-A4B-109e-it-Q6_K.gguf gemma-4-A4B-109e-Q6_K.gguf || true
ls -lh gemma-4-A4B-109e-Q6_K.gguf
df -h "$WORK" | tail -1

#==============================================================
# 5b. Download 128e + build 120e v3 + quantize
#     (only 120e v3 needs the surgery path — 109e is published as GGUF)
#==============================================================
echo
echo "===== downloading 128e from HF (~50 GB) ====="
huggingface-cli download google/gemma-4-26B-A4B-it \
    --local-dir gemma-4-26B-A4B-it \
    --max-workers 8 2>&1 | tail -5

echo
echo "===== running expert_drop for 120e v3 ====="
python3 scripts/expert_drop.py \
    --source-dir gemma-4-26B-A4B-it \
    --drop-map scripts/hybrid_120e_drop_map.json \
    --suffix=-hybrid

# Free disk: don't need 128e HF anymore
rm -rf gemma-4-26B-A4B-it
df -h "$WORK" | tail -1

echo
echo "===== converting 120e v3 to F16 GGUF ====="
python3 /opt/llama.cpp/convert_hf_to_gguf.py gemma-4-A4B-120e-hybrid \
    --outfile gemma-4-A4B-120e-hybrid-F16.gguf --outtype f16 2>&1 | tail -3

echo
echo "===== quantizing 120e v3 to Q6_K ====="
/opt/llama.cpp/build/bin/llama-quantize \
    gemma-4-A4B-120e-hybrid-F16.gguf \
    gemma-4-A4B-120e-hybrid-Q6_K.gguf Q6_K 2>&1 | tail -3

# Free disk: F16 + HF dir
rm gemma-4-A4B-120e-hybrid-F16.gguf
rm -rf gemma-4-A4B-120e-hybrid
df -h "$WORK" | tail -1

ls -lh *.gguf

#==============================================================
# 7. Start 2 llama-server instances (one per GPU)
#==============================================================
echo
echo "===== starting 2 llama-server instances ====="

LLAMA="/opt/llama.cpp/build/bin/llama-server"
COMMON_ARGS="-c 32768 -t 16 -ngl 99 --no-warmup --reasoning-format deepseek --reasoning-budget 16384 --temp 1.0 --top-p 0.95 --top-k 64 --seed 42"

CUDA_VISIBLE_DEVICES=0 $LLAMA -m gemma-4-A4B-120e-hybrid-Q6_K.gguf --port 8099 $COMMON_ARGS \
    >llama_120e_v3.log 2>&1 &
SPID0=$!
disown $SPID0 || true

CUDA_VISIBLE_DEVICES=1 $LLAMA -m gemma-4-A4B-109e-Q6_K.gguf --port 8100 $COMMON_ARGS \
    >llama_109e.log 2>&1 &
SPID1=$!
disown $SPID1 || true

# Wait for both servers
echo -n "  waiting for both servers..."
for i in $(seq 1 240); do
    H0=$(curl -fsS http://localhost:8099/health 2>/dev/null | grep -c ok || echo 0)
    H1=$(curl -fsS http://localhost:8100/health 2>/dev/null | grep -c ok || echo 0)
    if [[ "$H0" -ge 1 && "$H1" -ge 1 ]]; then
        echo " ready (pid0=$SPID0 pid1=$SPID1)"
        break
    fi
    if ! kill -0 $SPID0 2>/dev/null; then
        echo
        echo "ERROR: server 0 (120e v3) died — see llama_120e_v3.log"
        tail -20 llama_120e_v3.log
        exit 1
    fi
    if ! kill -0 $SPID1 2>/dev/null; then
        echo
        echo "ERROR: server 1 (109e) died — see llama_109e.log"
        tail -20 llama_109e.log
        exit 1
    fi
    echo -n "."
    sleep 1
done

#==============================================================
# 8. Run 2 lm_eval in parallel
#==============================================================
echo
echo "===== launching 2 lm_eval gpqa_diamond_cot_zeroshot in parallel ====="
mkdir -p eval_results

# 120e v3 -> server on port 8099 (GPU 0)
lm_eval \
    --model local-chat-completions \
    --model_args "model=120e_v3,base_url=http://localhost:8099/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=google/gemma-4-26B-A4B-it,max_gen_toks=24576" \
    --tasks gpqa_diamond_cot_zeroshot \
    --apply_chat_template \
    --batch_size 1 \
    --gen_kwargs "temperature=1.0,top_p=0.95,max_gen_toks=24576" \
    --log_samples \
    --output_path eval_results/120e_v3 \
    >lm_eval_120e_v3.log 2>&1 &
LMPID0=$!

# 109e -> server on port 8100 (GPU 1)
lm_eval \
    --model local-chat-completions \
    --model_args "model=109e,base_url=http://localhost:8100/v1/chat/completions,num_concurrent=1,tokenizer_backend=huggingface,tokenizer=google/gemma-4-26B-A4B-it,max_gen_toks=24576" \
    --tasks gpqa_diamond_cot_zeroshot \
    --apply_chat_template \
    --batch_size 1 \
    --gen_kwargs "temperature=1.0,top_p=0.95,max_gen_toks=24576" \
    --log_samples \
    --output_path eval_results/109e \
    >lm_eval_109e.log 2>&1 &
LMPID1=$!

echo "  120e_v3 lm_eval pid: $LMPID0 (log: lm_eval_120e_v3.log)"
echo "  109e    lm_eval pid: $LMPID1 (log: lm_eval_109e.log)"
echo
echo "===== both evals running. ETA ~11h. =====
  Monitor progress:
    tail -f $WORK/lm_eval_120e_v3.log
    tail -f $WORK/lm_eval_109e.log
  Check llama-server logs:
    tail -f $WORK/llama_120e_v3.log
    tail -f $WORK/llama_109e.log
  Results saved at:
    $WORK/eval_results/120e_v3/
    $WORK/eval_results/109e/
"

# Wait for both to complete (so script blocks until done)
wait $LMPID0
wait $LMPID1

echo
echo "===== both evals complete: $(date) ====="

# Stop servers cleanly
kill $SPID0 $SPID1 2>/dev/null || true

# Print final results
echo
for d in eval_results/120e_v3 eval_results/109e; do
    echo "--- $d ---"
    find "$d" -name "results_*.json" -exec cat {} \; 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for k, v in d.get('results', {}).items():
        print(f'  {k}:')
        for mk, mv in v.items():
            if isinstance(mv, (int, float, str)):
                print(f'    {mk}: {mv}')
except Exception as e:
    print(f'  parse error: {e}')
"
done

echo "===== pod setup + parallel GPQA done: $(date) ====="
