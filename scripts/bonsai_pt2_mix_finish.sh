#!/usr/bin/env bash
set -uo pipefail
S=/mnt/sdc/opencoti-scratch/v6-ternary; mkdir -p /srv/ml/pt2-export-tmp
PT2_DIR=$S/pt2_combined PT2_TAG=pt2mix PT2_TMP=/srv/ml/pt2-export-tmp bash /srv/ml/scripts/bonsai_pt2_export.sh || exit 1
rmdir /srv/ml/pt2-export-tmp 2>/dev/null
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68; D=/mnt/sdc/opencoti-models/sdc-ml/qwen38-base; W=$S/trace_eval
ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-perplexity" "$@" -f "$W/heldout_traces.txt" -c 2048 --chunks 16 -b 2048 -ub 512 -ngl 99; }
ppl -m "$D/Qwen3.8-27B-BF16.gguf" --kl-divergence-base "$W/base.kld" > "$W/base2.log" 2>&1 || { echo "FATAL base logits"; exit 1; }
echo "base again: $(grep -E 'Final estimate' "$W/base2.log")"
for N in pt2 pt2mix; do ppl -m "$D/Qwen3.8-27B-$N-PQ2_0.gguf" --kl-divergence-base "$W/base.kld" --kl-divergence > "$W/kld2-$N.log" 2>&1
  echo "--- traces: $N ---"; grep -E "^(Mean  *PPL\(Q\) |Mean  *KLD|Same top p)" "$W/kld2-$N.log" | sed 's/^/     /'; done
rm -f "$W/base.kld"; df -h / /mnt/sdc | tail -2; echo "=== MIX_FINISH_DONE ==="
