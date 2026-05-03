#!/bin/bash
# Watcher: poll vast.ai pod for completion of 109e and 120e v3 GPQA runs.
# When in_progress=false on either, scp the JSON down to solidpc's
# eval_results/gpqa_full/. Continues watching until both are retrieved.
#
# Once both are retrieved, also pulls the run logs for posterity.
set -uo pipefail

REPO="/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"
cd "$REPO"

LOG="eval_results/pod_retrieve.log"
mkdir -p eval_results/gpqa_full eval_results/server_logs

SSH="ssh -o StrictHostKeyChecking=no -p 40736 root@185.113.120.195"

echo "===== pod retrieve watcher start: $(date) =====" | tee -a "$LOG"

retrieved_109e=0
retrieved_120e=0

check_one() {
    local label="$1"
    local remote_json="$2"
    local local_json="$3"
    # Pull current state via SSH and check in_progress field
    local state
    state=$($SSH "python3 -c 'import json; d=json.load(open(\"$remote_json\")); print(d.get(\"in_progress\", True))'" 2>/dev/null)
    if [[ "$state" == "False" ]]; then
        echo "  $label finished, downloading..." | tee -a "$LOG"
        scp -o StrictHostKeyChecking=no -P 40736 "root@185.113.120.195:$remote_json" "$local_json" 2>&1 | tee -a "$LOG"
        return 0
    fi
    return 1
}

while [[ $retrieved_109e -eq 0 || $retrieved_120e -eq 0 ]]; do
    if [[ $retrieved_109e -eq 0 ]]; then
        if check_one "109e" /workspace/eval_results/full_gpqa_109e_Q6K.json eval_results/gpqa_full/full_gpqa_109e_Q6K.json; then
            retrieved_109e=1
            scp -o StrictHostKeyChecking=no -P 40736 "root@185.113.120.195:/workspace/full_gpqa_109e_run.log" eval_results/server_logs/full_gpqa_109e_Q6K_run.log 2>&1 | tee -a "$LOG"
            scp -o StrictHostKeyChecking=no -P 40736 "root@185.113.120.195:/workspace/llama_109e.log" eval_results/server_logs/full_gpqa_109e_Q6K_server.log 2>&1 | tee -a "$LOG"
        fi
    fi
    if [[ $retrieved_120e -eq 0 ]]; then
        if check_one "120e_v3" /workspace/eval_results/full_gpqa_120e_v3_Q6K.json eval_results/gpqa_full/full_gpqa_120e_v3_Q6K.json; then
            retrieved_120e=1
            scp -o StrictHostKeyChecking=no -P 40736 "root@185.113.120.195:/workspace/watch_120e_v3.log" eval_results/server_logs/full_gpqa_120e_v3_Q6K_run.log 2>&1 | tee -a "$LOG"
            scp -o StrictHostKeyChecking=no -P 40736 "root@185.113.120.195:/workspace/llama_120e_v3.log" eval_results/server_logs/full_gpqa_120e_v3_Q6K_server.log 2>&1 | tee -a "$LOG"
        fi
    fi
    if [[ $retrieved_109e -eq 0 || $retrieved_120e -eq 0 ]]; then
        sleep 300
    fi
done

echo "===== both pod results retrieved: $(date) =====" | tee -a "$LOG"
echo "  files:"
ls -lh eval_results/gpqa_full/full_gpqa_109e_Q6K.json eval_results/gpqa_full/full_gpqa_120e_v3_Q6K.json | tee -a "$LOG"
