#!/bin/bash
# run_t141_master.sh — master orchestrator for T141: 2×2 shared×PES ablation
# on GPU 0, EAC-MoE on GPU 1 in parallel, then Router-KD on both GPUs.
#
# DESIGNED FOR THE PERMANENT linode-blackswan-2 SERVER (NOT solidpc).
# Run THIS script ON solidpc — it SSHes to the pod and orchestrates there.
#
# Phases:
#   0. Wait gate     — block until pod has B1 + B2 bf16 dirs + 128e teacher complete
#   1. Build A1, A2  — bf16 dirs via cp -al + invert shared (and PES for A2)
#                      ablation script handles this transparently inside its build_cell
#   2. Launch in parallel:
#        GPU 0: run_shared_x_pes_ablation.sh --run        (builds + quants + evals all 4 cells)
#        GPU 1: run_eac_baseline.sh --target baseline --run --with-eval (EAC + post-eval)
#   3. Wait both done
#   4. Launch run_router_kd.sh (uses BOTH GPUs)
#   5. Summary — print result table, ship to solidpc via rsync
#
# Usage:
#   run_t141_master.sh                    # dry-run plan
#   run_t141_master.sh --skip-wait        # don't wait for rsync (assume weights ready)
#   run_t141_master.sh --run              # execute
#   run_t141_master.sh --run --skip-kd    # skip Router-KD phase (do only ablation+EAC)
#
# Logs land under /srv/ml/logs/t141/ on the pod and are rsynced back to solidpc on completion.
#
# IMPORTANT: this script does NOT directly run GPU work — it dispatches via SSH so the
# actual heavy lifting is on the pod. solidpc just orchestrates and rsync-collects results.

set -uo pipefail

POD=${POD:-linode-blackswan-2}
DO_RUN=0
SKIP_WAIT=0
SKIP_KD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --pod)        POD=$2; shift 2;;
    --run)        DO_RUN=1; shift;;
    --skip-wait)  SKIP_WAIT=1; shift;;
    --skip-kd)    SKIP_KD=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

BM_POD=/srv/ml                         # canonical root on the pod
B1=$BM_POD/google/gemma-4-A4B-62e-fc15_25-p8-it
B2=$BM_POD/google/gemma-4-A4B-62e-fc15_25-p8-pes1_20-it
T128=$BM_POD/google/gemma-4-26B-A4B-it
LOG_DIR=$BM_POD/logs/t141
RES=$BM_POD/eval_results_t141_shared_x_pes

echo "=================== T141 MASTER ORCHESTRATOR ==================="
echo "  pod              : $POD"
echo "  B1 bf16          : $B1"
echo "  B2 bf16          : $B2"
echo "  128e teacher     : $T128"
echo "  ablation script  : run_shared_x_pes_ablation.sh"
echo "  EAC script       : run_eac_baseline.sh --target baseline"
echo "  Router-KD script : run_router_kd.sh  (skip-kd=$SKIP_KD)"
echo "  log dir on pod   : $LOG_DIR"
echo "  results on pod   : $RES"
echo "  with-run         : $DO_RUN"
echo "  skip-wait        : $SKIP_WAIT"
echo "================================================================"

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] nothing executed. Re-run with --run."
  exit 0
fi

# ---------- 0. Wait gate ----------
if [ "$SKIP_WAIT" -ne 1 ]; then
  echo "[t141-master] phase 0 — wait for B1 + B2 + 128e on pod"
  while true; do
    ready=$(ssh -o BatchMode=yes "$POD" "bash -c '
      b1=\$(test -f $B1/model.safetensors.index.json && echo Y || echo N)
      b2=\$(test -f $B2/model.safetensors.index.json && echo Y || echo N)
      t=\$(test -f $T128/model.safetensors.index.json && echo Y || echo N)
      echo \$b1\$b2\$t
    '" 2>/dev/null)
    if [ "$ready" = "YYY" ]; then
      echo "[t141-master] all 3 weight trees present — proceeding"
      break
    fi
    echo "  $(date -u +%H:%M:%SZ) — weights state B1=$ready[0:1] B2=$ready[1:2] T128=$ready[2:3]"
    sleep 120
  done
fi

# ---------- 1+2. Launch ablation (GPU 0) + EAC (GPU 1) in parallel ----------
echo "[t141-master] phase 1+2 — launching ablation on GPU 0 + EAC on GPU 1"
TS=$(date +%Y%m%d_%H%M%S)
ssh -o BatchMode=yes "$POD" "bash -s" <<REMOTE_EOF
set -uo pipefail
mkdir -p $LOG_DIR
export PATH=/root/anaconda3/envs/omnimergekit/bin:/srv/ml/tools/llama.cpp/build/bin:\$PATH

# --- GPU 0: 2×2 ablation ---
nohup env CUDA_VISIBLE_DEVICES=0 \\
    bash $BM_POD/scripts/run_shared_x_pes_ablation.sh --run \\
  > $LOG_DIR/ablation_${TS}.log 2>&1 &
ABL_PID=\$!
disown
echo "  ablation launched as PID \$ABL_PID (GPU 0) — log $LOG_DIR/ablation_${TS}.log"

# --- GPU 1: EAC on baseline ---
nohup env CUDA_VISIBLE_DEVICES=1 \\
    bash $BM_POD/scripts/run_eac_baseline.sh --target baseline --run --with-eval \\
  > $LOG_DIR/eac_${TS}.log 2>&1 &
EAC_PID=\$!
disown
echo "  EAC      launched as PID \$EAC_PID (GPU 1) — log $LOG_DIR/eac_${TS}.log"

# record pids for the watcher
echo "\$ABL_PID" > $LOG_DIR/ablation.pid
echo "\$EAC_PID" > $LOG_DIR/eac.pid
REMOTE_EOF

# ---------- 3. Wait both ----------
echo "[t141-master] phase 3 — waiting for both jobs to finish"
while true; do
  state=$(ssh -o BatchMode=yes "$POD" "bash -c '
    abl_pid=\$(cat $LOG_DIR/ablation.pid 2>/dev/null)
    eac_pid=\$(cat $LOG_DIR/eac.pid 2>/dev/null)
    abl_run=N; eac_run=N
    kill -0 \$abl_pid 2>/dev/null && abl_run=Y
    kill -0 \$eac_pid 2>/dev/null && eac_run=Y
    echo \"abl=\$abl_run eac=\$eac_run\"
  '" 2>/dev/null)
  echo "  $(date -u +%H:%M:%SZ) $state"
  case "$state" in
    "abl=N eac=N"*) echo "[t141-master] both jobs done"; break;;
  esac
  sleep 300  # 5 min poll
done

# Snapshot scores at this point (read summary.json, never raw exact_match)
ssh -o BatchMode=yes "$POD" "bash -c '
for cell in A1 A2 B1 B2; do
  for tpl in humanevalplus_full multipl_e_100; do
    s=$RES/\$tpl/t141-62e-\$cell/summary.json
    if [ -f \"\$s\" ]; then
      score=\$(/root/anaconda3/envs/omnimergekit/bin/python -c \"import json;print(round(json.load(open(\\\"\$s\\\")).get(\\\"score\\\",0)*100,2))\")
      echo \"  \$cell \$tpl -> \$score%\"
    fi
  done
done
'" 2>&1 | tee $BM_POD/../../../srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/logs/t141_master_${TS}.log

# ---------- 4. Router-KD ----------
if [ "$SKIP_KD" -ne 1 ]; then
  echo "[t141-master] phase 4 — Router-KD on both GPUs (3-4h wall-clock per memory)"
  ssh -o BatchMode=yes "$POD" "bash -s" <<REMOTE_EOF
set -uo pipefail
nohup bash $BM_POD/scripts/run_router_kd.sh --run \\
  > $LOG_DIR/router_kd_${TS}.log 2>&1 &
KD_PID=\$!
disown
echo "\$KD_PID" > $LOG_DIR/router_kd.pid
echo "  Router-KD launched as PID \$KD_PID — log $LOG_DIR/router_kd_${TS}.log"
REMOTE_EOF

  # Wait
  while true; do
    state=$(ssh -o BatchMode=yes "$POD" "bash -c '
      pid=\$(cat $LOG_DIR/router_kd.pid 2>/dev/null)
      kill -0 \$pid 2>/dev/null && echo running || echo done
    '" 2>/dev/null)
    echo "  $(date -u +%H:%M:%SZ) Router-KD: $state"
    [ "$state" = "done" ] && break
    sleep 600  # 10 min poll
  done
fi

# ---------- 5. rsync results back to solidpc ----------
echo "[t141-master] phase 5 — rsync results back to solidpc"
SOLIDPC_BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
mkdir -p $SOLIDPC_BM/eval_results_t141_shared_x_pes $SOLIDPC_BM/logs/t141
rsync -ah --info=stats2 "$POD:$RES/" "$SOLIDPC_BM/eval_results_t141_shared_x_pes/"
rsync -ah --info=stats2 "$POD:$LOG_DIR/" "$SOLIDPC_BM/logs/t141/"
echo "[t141-master] DONE — results in $SOLIDPC_BM/eval_results_t141_shared_x_pes/"
