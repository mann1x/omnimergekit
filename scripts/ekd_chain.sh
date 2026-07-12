#!/usr/bin/env bash
# T192 E-ExpertKD first chain: finite-loss preflight (mid experts, 2 steps, no-save) ->
# first smoke (mid experts, ml-heavy corpus, 400 steps, lr 1e-4) on GPU0 single-card
# (teacher-4bit + student-bf16 + 8bit-Adam + grad-ckpt ~ 73GB).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SCR=/srv/ml/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
WORK=/srv/ml/redist_work
ML=/mnt/sdc/ml/corpora/kd_corpus_ml_heavy.jsonl
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[ekd-chain $(date -u +%H:%M:%S)] PREFLIGHT (mid experts, 2 steps, no-save)"
"$PY" "$SCR/router_kd.py" \
  --base-dir "$TEACHER" --variant-dir "$STUDENT" --out-dir "$WORK/ekd_preflight" \
  --train-tensors experts --train-layers mid --student-load bf16 --teacher-load 4bit \
  --teacher-device '{"":0}' --student-device '{"":0}' --gpu-mem-gib 85 --optim paged_adamw8bit --grad-checkpointing \
  --corpus-file "$ML" --epochs 100 --max-samples 100000 \
  --tau 1.0 --lr 1e-4 --max-steps 2 --batch-size 1 --grad-accum 8 --max-seq-len 1024 \
  --no-canary --no-save --log-every 1
RC=$?
if [ "$RC" != "0" ]; then echo "[ekd-chain] PREFLIGHT_FAIL rc=$RC"; exit 1; fi
echo "[ekd-chain $(date -u +%H:%M:%S)] PREFLIGHT_OK -> first smoke (mid experts ml_heavy lr1e-4 400 steps)"
bash "$SCR/redist_expert_kd_run.sh" mid_mlheavy_lr1e4 "$ML" experts mid 400 1e-4 0 4bit 1024
echo "[ekd-chain $(date -u +%H:%M:%S)] CHAIN_DONE"
