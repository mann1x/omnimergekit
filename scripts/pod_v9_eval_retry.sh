#!/usr/bin/env bash
# v9-agentic eval RETRY runner — same omk_eval call as pod_v9_eval_chain.sh run_q, plus a GPU wait.
#
# Why: pod_v9_eval_chain.sh starts each cell the instant the previous one exits. The previous
# llama-server has not released its VRAM yet, gpu_planner sees free < need with OMK_GPU_WAIT_S=0
# and refuses (FATAL exit=8). 2026-10-14 20:26Z: HE+ ended, then multipl_e_100 / arc_challenge_full /
# lcb_medium_55_v4 / lcb_medium_100_v4 all died within 1s (GPU1 free 15440 < need 17680 MiB).
#
# usage: pod_v9_eval_retry.sh <gpu> <first_port> <wait_pid|-> <bench>...
#   wait_pid: a PID that must exit first (e.g. the chain's live run_q for that GPU), or "-".
#   A bench whose summary.json already exists is skipped, so this never re-runs a finished cell.
set -uo pipefail
G=$1; P=$2; WP=$3; shift 3
W=/workspace
STEM=gemma-4-20B-v9-agentic
Q6=$W/eval_gguf/q6k/$STEM-Q6_K.gguf
TOK=$W/$STEM
RES=$W/eval_results
SN=v9agentic-q6k
PY=$W/miniconda/envs/omnimergekit/bin/python
OMK=$W/omnimergekit/eval/omk_eval.py
LOG=$W/logs/eval_chain.log
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export MPE_MODE=native MPE_HARNESS=$W/MultiPL-E HF_ALLOW_CODE_EVAL=1
export OMK_GPU_WAIT_S="${OMK_GPU_WAIT_S:-900}"
L(){ echo "[$(date -u +%FT%TZ)] RETRY $*" | tee -a "$LOG"; }

if [ "$WP" != "-" ]; then
  L "GPU$G: waiting for PID $WP to exit before retrying $*"
  while kill -0 "$WP" 2>/dev/null; do sleep 60; done
  L "GPU$G: PID $WP exited"
fi
for b in "$@"; do
  sj=$RES/$b/$SN/summary.json
  if [ -f "$sj" ]; then L "GPU$G $b: summary exists, skip"; continue; fi
  if [ "$b" = multipl_e_100 ] && ! grep -q "MPE_SETUP_RC=0" $W/logs/mpe_setup.log 2>/dev/null; then
    L "GPU$G $b: MPE toolchain setup not OK, cell skipped"; continue
  fi
  [ -f "$W/logs/eval/$b.log" ] && mv "$W/logs/eval/$b.log" "$W/logs/eval/$b.rc8_$(date -u +%H%M%S).log"
  par=2; [ "$b" = lcb_v6_77q ] && par=auto
  L "GPU$G $b: start (port $P, --parallel $par, OMK_GPU_WAIT_S=$OMK_GPU_WAIT_S)"
  ( cd $W/omnimergekit && CUDA_VISIBLE_DEVICES=$G "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" \
      --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
      > "$W/logs/eval/$b.log" 2>&1 ); rc=$?
  if [ -f "$sj" ]; then
    L "GPU$G $b: rc=$rc RESULT $("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")"
  else
    L "GPU$G $b: rc=$rc NO summary.json (see logs/eval/$b.log)"
  fi
  P=$((P+1)); sleep 30   # let this cell's llama-server release VRAM before the next one
done
L "GPU$G retry queue done"
