#!/bin/bash
# Post-128e chain: triggered after the solidpc 128e Q6_K full GPQA finishes.
# 1. Extract truncated doc_ids from full_gpqa_128e_Q6K_def.json
# 2. Run patch_truncated_gpqa.sh with --dry-multiplier 0.5 on those docs
# 3. Run eval_full_gpqa.sh for 98e-hybrid
# 4. Move final files into eval_results/gpqa_full/
#
# Starts watching as soon as launched, checks every 60s.
set -uo pipefail

REPO="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
cd "$REPO"

CHAIN_LOG="eval_results/post_128e_chain.log"

eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda activate lightseek

echo "===== post-128e chain start: $(date) =====" | tee -a "$CHAIN_LOG"

# Wait for 128e to finish (in_progress: false)
INPROG=eval_results/full_gpqa_128e_Q6K_def.json
echo "  waiting for $INPROG to be in_progress=false..." | tee -a "$CHAIN_LOG"
while true; do
    if [[ -f "$INPROG" ]]; then
        STATE=$(python3 -c "import json; d=json.load(open('$INPROG')); print(d.get('in_progress', True))" 2>/dev/null)
        if [[ "$STATE" == "False" ]]; then
            echo "  128e finished at $(date)" | tee -a "$CHAIN_LOG"
            break
        fi
    fi
    sleep 60
done

sleep 10

# Make sure no llama-server is holding the GPU
if pgrep -f llama-server >/dev/null; then
    echo "  killing leftover llama-server..." | tee -a "$CHAIN_LOG"
    pkill -f llama-server || true
    sleep 5
fi

# Extract truncated doc_ids
TRUNC=$(python3 -c "
import json
d = json.load(open('$INPROG'))
truncs = sorted([r['doc_id'] for r in d['results'] if r.get('truncated', False)])
print(','.join(str(x) for x in truncs))
")
echo "  truncated docs: $TRUNC" | tee -a "$CHAIN_LOG"

#==============================================================
# Step 1: patch run on 128e with --dry-multiplier 0.5
#==============================================================
if [[ -n "$TRUNC" ]]; then
    echo
    echo "===== launching patch run on 128e: docs=$TRUNC =====" | tee -a "$CHAIN_LOG"
    ./scripts/patch_truncated_gpqa.sh 128e_Q6K_def google/gemma-4-26B-A4B-it-Q6_K.gguf "$TRUNC" 2>&1 | tee -a "$CHAIN_LOG"
    sleep 10
    if pgrep -f llama-server >/dev/null; then
        pkill -f llama-server || true
        sleep 5
    fi
else
    echo "  no truncated docs to patch on 128e — skipping patch step" | tee -a "$CHAIN_LOG"
fi

# Move 128e final + server log to gpqa_full subfolder for organization
mkdir -p eval_results/gpqa_full
mv -f eval_results/full_gpqa_128e_Q6K_def.json eval_results/gpqa_full/ 2>/dev/null || true
mv -f eval_results/full_gpqa_128e_Q6K_def_server.log eval_results/server_logs/ 2>/dev/null || true

#==============================================================
# Step 2: full GPQA Diamond on 98e-hybrid
#==============================================================
echo
echo "===== launching full GPQA Diamond on 98e-hybrid =====" | tee -a "$CHAIN_LOG"
./scripts/eval_full_gpqa.sh 98e_hybrid_Q6K google/gemma-4-A4B-98e-hybrid-Q6_K.gguf 2>&1 | tee -a "$CHAIN_LOG"

# Move result + server log
mv -f eval_results/full_gpqa_98e_hybrid_Q6K.json eval_results/gpqa_full/ 2>/dev/null || true
mv -f eval_results/full_gpqa_98e_hybrid_Q6K_server.log eval_results/server_logs/ 2>/dev/null || true

#==============================================================
# Step 3: also patch 98e-hybrid if it has truncations
#==============================================================
EIGHT_RESULT=eval_results/gpqa_full/full_gpqa_98e_hybrid_Q6K.json
if [[ -f "$EIGHT_RESULT" ]]; then
    EIGHT_TRUNC=$(python3 -c "
import json
d = json.load(open('$EIGHT_RESULT'))
truncs = sorted([r['doc_id'] for r in d['results'] if r.get('truncated', False)])
print(','.join(str(x) for x in truncs))
")
    if [[ -n "$EIGHT_TRUNC" ]]; then
        echo
        echo "===== launching patch run on 98e-hybrid: docs=$EIGHT_TRUNC =====" | tee -a "$CHAIN_LOG"
        sleep 10
        if pgrep -f llama-server >/dev/null; then
            pkill -f llama-server || true
            sleep 5
        fi
        ./scripts/patch_truncated_gpqa.sh 98e_hybrid_Q6K google/gemma-4-A4B-98e-hybrid-Q6_K.gguf "$EIGHT_TRUNC" 2>&1 | tee -a "$CHAIN_LOG"
    else
        echo "  no truncated docs on 98e-hybrid — skipping patch" | tee -a "$CHAIN_LOG"
    fi
fi

echo
echo "===== post-128e chain done: $(date) =====" | tee -a "$CHAIN_LOG"
