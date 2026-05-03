#!/bin/bash
# After 98e-hybrid GPQA finishes on solidpc:
# 1. Build 109e v2 from the new teacher-force drop map
# 2. Convert to F16 GGUF
# 3. Quantize to Q6_K
# 4. Run full GPQA Diamond eval via eval_gpqa_v3.sh
set -uo pipefail

REPO="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
cd "$REPO"

LOG="eval_results/server_logs/post_98e_109e_v2.log"

eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda activate lightseek

echo "===== post-98e: build + eval 109e v2 =====" | tee -a "$LOG"
echo "  start: $(date)" | tee -a "$LOG"

# Wait for 98e to finish
echo "  waiting for 98e to finish..." | tee -a "$LOG"
while true; do
    if ! pgrep -f "eval_gpqa_v3.*98e" >/dev/null 2>&1 && ! pgrep -f "llama-server.*98e" >/dev/null 2>&1; then
        echo "  98e processes gone at $(date)" | tee -a "$LOG"
        break
    fi
    sleep 60
done
sleep 10

# Kill any remaining llama-server
pkill -9 -f llama-server 2>/dev/null || true
sleep 5

echo | tee -a "$LOG"
echo "===== building 109e v2 from teacher-force map =====" | tee -a "$LOG"
python3 scripts/expert_drop.py \
    --source-dir google/gemma-4-26B-A4B-it \
    --drop-map scripts/teacher_force_109e_p16_drop_map.json \
    --suffix=-v2 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "===== converting to F16 =====" | tee -a "$LOG"
python3 /opt/llama.cpp/convert_hf_to_gguf.py google/gemma-4-A4B-109e-v2 \
    --outfile google/gemma-4-A4B-109e-v2-F16.gguf --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

echo "===== quantizing to Q6_K =====" | tee -a "$LOG"
/opt/llama.cpp/build/bin/llama-quantize \
    google/gemma-4-A4B-109e-v2-F16.gguf \
    google/gemma-4-A4B-109e-v2-Q6_K.gguf Q6_K 2>&1 | tail -3 | tee -a "$LOG"

rm google/gemma-4-A4B-109e-v2-F16.gguf
ls -lh google/gemma-4-A4B-109e-v2-Q6_K.gguf | tee -a "$LOG"

echo | tee -a "$LOG"
echo "===== running full GPQA Diamond on 109e v2 =====" | tee -a "$LOG"
./scripts/eval_gpqa_v3.sh 109e_v2_Q6K google/gemma-4-A4B-109e-v2-Q6_K.gguf 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "===== 109e v2 done: $(date) =====" | tee -a "$LOG"
