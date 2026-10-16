#!/usr/bin/env bash
# v9-ml ml_k8 suite: ONE GPU queue. Same omk_eval call as pod_v9_eval_retry.sh (llama backend,
# --parallel 2 / auto for lcb_v6_77q, OMK_GPU_WAIT_S=900, 30 s VRAM gap) PLUS the model's own
# sampler: --sampler-profile auto -> eval/models/gemma-4-v9-agentic.yaml greedy_pen
# (temp 0 + repeat_penalty 1.06 / repeat_last_n 96 / presence_penalty 0.025). Owner 2026-10-15:
# cells without these penalties are useless.
# LIVE gate per cell: the llama-server on the cell's port must carry all three penalty flags, or
# the cell is killed and the queue stops. Post gate: summary.json sampler.name must be greedy_pen.
# usage: pod_v9ml_k8_q.sh <gpu> <first_port> <bench>...
set -uo pipefail
G=$1; P=$2; shift 2
W=/workspace; S=$W/v9ml_k8_suite; RES=$S/results; LD=$S/logs/eval
Q6=$W/gguf/gemma-4-20B-v9ml-ml_k8/gemma-4-20B-v9ml-ml_k8-Q6_K.gguf
TOK=$W/v9ml_k8/gemma-4-20B-v9ml-ml_k8; SN=v9ml-k8-imat-q6k
PY=$W/miniconda/envs/omnimergekit/bin/python
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export MPE_MODE=native MPE_HARNESS=$W/MultiPL-E HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900
say(){ echo "[q GPU$G $(date -u +%FT%TZ)] $*"; }
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
mkdir -p $RES $LD
for b in "$@"; do
  sj=$RES/$b/$SN/summary.json
  [ -f "$sj" ] && { say "$b: summary exists, skip"; continue; }
  if [ "$b" = multipl_e_100 ] && ! grep -q "MPE_SETUP_RC=0" $W/logs/mpe_setup.log 2>/dev/null; then
    say "Q_FAIL $b: MPE toolchain setup not OK"; exit 1; fi
  par=2; [ "$b" = lcb_v6_77q ] && par=auto
  say "$b: start (port $P, --parallel $par, --sampler-profile auto)"
  ( cd $W/omnimergekit && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model "$Q6" --tokenizer "$TOK" \
      --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
      --sampler-profile auto > "$LD/$b.log" 2>&1 ) & cp=$!
  live=none
  for _ in $(seq 1 180); do
    spid=$(ss -ltnp 2>/dev/null | grep ":$P " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
    if [ -n "$spid" ]; then
      cl=$(tr "\0" " " < /proc/$spid/cmdline)
      if grep -q -- "--repeat-penalty 1.06 " <<<"$cl" && grep -q -- "--repeat-last-n 96 " <<<"$cl" \
         && grep -q -- "--presence-penalty 0.025 " <<<"$cl"; then live=ok; else live=bad; fi
      break
    fi
    kill -0 $cp 2>/dev/null || break
    sleep 10
  done
  case $live in
    ok)   say "$b: LIVE sampler OK (llama-server pid $spid carries rep 1.06 / last 96 / presence 0.025)";;
    bad)  say "Q_FAIL $b: LIVE llama-server pid $spid is MISSING the penalty flags: $cl"; killtree $cp; exit 1;;
    none) if kill -0 $cp 2>/dev/null; then say "Q_FAIL $b: no llama-server on port $P after 30 min"; killtree $cp; exit 1; fi;;
  esac
  wait $cp; rc=$?
  if [ -f "$sj" ]; then
    r=$("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")
    say "$b: rc=$rc RESULT $r"
    [ "${r##* }" = greedy_pen ] || { say "Q_FAIL $b: summary sampler is not greedy_pen"; exit 1; }
    [ $live = ok ] || say "WARN $b: finished before the live gate saw its server"
  else
    say "$b: rc=$rc NO summary.json (see $LD/$b.log)"
  fi
  P=$((P+1)); sleep 30
done
say "QUEUE_DONE"
