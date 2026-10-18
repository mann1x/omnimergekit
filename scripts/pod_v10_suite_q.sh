#!/usr/bin/env bash
# pod_v10_suite_q.sh — ON THE POD: ONE GPU queue of the v9-agentic card suite for one v10-recipe map's Q6_K.
# Same omk_eval call as pod_v9ml_k8_q.sh (the published suite): llama backend (eval tree /workspace/llama.cpp,
# b9700 + PEG-degrade patch), --parallel 2 / auto for lcb_v6_77q, OMK_GPU_WAIT_S=900, 30 s VRAM gap,
# --sampler-profile auto -> greedy_pen (temp 0 + rep 1.06 / last 96 / presence 0.025).
# Owner 2026-10-17: "run the 10-bench evals on both while I test".
# LIVE gate per cell: the llama-server must carry all three penalty flags, or the queue stops (systemic).
# Post gate per cell: rc 0, score not None, sampler greedy_pen — a failed cell is logged CELL_FAIL and the queue
# moves on (one bad bench must not cost the rest of the night). A void summary is moved aside, never deleted.
# usage: pod_v10_suite_q.sh <map> <gpu> <first_port> <bench>...
set -uo pipefail
n=$1; G=$2; P=$3; shift 3
W=/workspace; S=$W/v10/suite; RES=$S/results; LD=$S/logs/eval
NAME=gemma-4-20B-v9-agentic-$n; Q6=$W/v10/gguf/$n/$NAME-Q6_K.gguf; TOK=$W/v10/$n/$NAME; SN=v9-agentic-$n-q6k
PY=$W/miniconda/envs/omnimergekit/bin/python
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export MPE_MODE=native MPE_HARNESS=$W/MultiPL-E HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900
say(){ echo "[q $n GPU$G $(date -u +%FT%TZ)] $*"; }
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
mkdir -p $RES $LD
[ -s "$Q6" ] || { say "Q_FAIL no $Q6"; exit 1; }
fails=0
for b in "$@"; do
  sj=$RES/$b/$SN/summary.json
  if [ -f "$sj" ]; then   # skip only a SCORED cell; a void summary (score null) is moved aside, never deleted
    if "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$sj')).get('score') is not None else 1)"; then
      say "$b: scored summary exists, skip"; continue; fi
    mv "$sj" "$sj.void_$(date -u +%Y%m%dT%H%M%SZ)"; say "$b: void summary moved aside, re-running (sqlite cache reused)"
  fi
  if [ "$b" = multipl_e_100 ] && ! grep -q "MPE_SETUP_RC=0" $W/logs/mpe_setup.log 2>/dev/null; then
    say "CELL_FAIL $b: MPE toolchain setup not OK"; fails=$((fails+1)); continue; fi
  par=2; [ "$b" = lcb_v6_77q ] && par=auto
  say "$b: start (port $P, --parallel $par, --sampler-profile auto)"
  ( cd $W/omnimergekit && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model "$Q6" --tokenizer "$TOK" \
      --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
      --sampler-profile auto > "$LD/$n.$b.log" 2>&1 ) & cp=$!
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
  r="None None None"
  [ -f "$sj" ] && r=$("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")
  say "$b: rc=$rc RESULT $r"
  if [ "$rc" = 0 ] && [ "${r%% *}" != None ] && [ "${r##* }" = greedy_pen ]; then
    [ $live = ok ] || say "WARN $b: finished before the live gate saw its server"
    say "CELL_OK $b"
  else
    say "CELL_FAIL $b: rc=$rc result=$r (see $LD/$n.$b.log)"; fails=$((fails+1))
  fi
  P=$((P+1)); sleep 30
done
say "QUEUE_DONE fails=$fails"
