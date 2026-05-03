#!/bin/bash
set -uo pipefail

REPO="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
cd "$REPO"

eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda activate lightseek

LOG="eval_results/server_logs/solidpc_120e_v4_98e_v2.log"

echo "===== solidpc: build 120e v4 + eval 120e v4 + eval 98e v2 =====" | tee "$LOG"
echo "  start: $(date)" | tee -a "$LOG"

# Step 1: Build 120e v4
echo | tee -a "$LOG"
echo "===== building 120e v4 from teacher-force p16 map =====" | tee -a "$LOG"
python3 scripts/expert_drop.py \
    --source-dir google/gemma-4-26B-A4B-it \
    --drop-map scripts/teacher_force_120e_p16_drop_map.json \
    --suffix="-v4" 2>&1 | tee -a "$LOG"

echo "===== converting to F16 =====" | tee -a "$LOG"
python3 /opt/llama.cpp/convert_hf_to_gguf.py google/gemma-4-A4B-120e-v4 \
    --outfile google/gemma-4-A4B-120e-v4-F16.gguf --outtype f16 2>&1 | tail -3 | tee -a "$LOG"

echo "===== quantizing to Q6_K =====" | tee -a "$LOG"
/opt/llama.cpp/build/bin/llama-quantize \
    google/gemma-4-A4B-120e-v4-F16.gguf \
    google/gemma-4-A4B-120e-v4-Q6_K.gguf Q6_K 2>&1 | tail -3 | tee -a "$LOG"

rm google/gemma-4-A4B-120e-v4-F16.gguf
rm -rf google/gemma-4-A4B-120e-v4/
ls -lh google/gemma-4-A4B-120e-v4-Q6_K.gguf | tee -a "$LOG"

# Step 2: Eval 120e v4
echo | tee -a "$LOG"
echo "===== full GPQA Diamond: 120e v4 =====" | tee -a "$LOG"
./scripts/eval_gpqa_v3.sh 120e_v4_Q6K google/gemma-4-A4B-120e-v4-Q6_K.gguf 2>&1 | tee -a "$LOG"

# Kill server before next eval
pkill -9 llama-server 2>/dev/null || true
sleep 5

# Step 3: Eval 98e v2 (GGUF already exists)
echo | tee -a "$LOG"
echo "===== full GPQA Diamond: 98e v2 =====" | tee -a "$LOG"
./scripts/eval_gpqa_v3.sh 98e_v2_Q6K google/gemma-4-A4B-98e-v2-Q6_K.gguf 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "===== ALL DONE: $(date) =====" | tee -a "$LOG"
