#!/usr/bin/env bash
# pod_v9_quant_mpe.sh — ON THE POD: MultiPL-E-100 (or BENCH=<template>) on published v9-agentic GGUF
# tiers, ONE GPU queue. Owner 2026-10-16 follow-up: "run IFEval on the same tiers" -> BENCH=ifeval_100.
# Owner 2026-10-16: "can we use MPE to check the quants quality between each other" — the suite's MPE
# cell call verbatim (pod_v9ml_k8_q.sh: llama backend, --parallel 2, --sampler-profile auto =
# greedy_pen = temp 0 + repeat_penalty 1.06 / repeat_last_n 96 / presence_penalty 0.025), so every
# tier is on the same basis as the published Q6_K cell (0.90). Only the model and served name change.
# Tier "Q6_K_repeat" re-runs the evaluated Q6_K file itself: the same-file repeat is the noise band.
# Each downloaded tier waits for its <gguf>.ok marker from pod_v9_quant_fetch.py.
# LIVE gate per cell: the llama-server must carry all three penalty flags, else the cell is killed.
# usage: [BENCH=ifeval_100] pod_v9_quant_mpe.sh <gpu> <first_port> <tier>...   (default multipl_e_100)
set -uo pipefail
G=$1; P=$2; shift 2
B=${BENCH:-multipl_e_100}
W=/workspace; S=$W/v9qmpe; RES=$S/results; LD=$S/logs; GD=$S/gguf
TOK=$W/v9pub/gemma-4-20B-v9-agentic
Q6=$W/gguf/gemma-4-20B-v9ml-ml_k8/gemma-4-20B-v9ml-ml_k8-Q6_K.gguf; Q6SHA=5957b63a9b0733cf
PY=$W/miniconda/envs/omnimergekit/bin/python
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export MPE_MODE=native MPE_HARNESS=$W/MultiPL-E HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900
say(){ echo "[mpe GPU$G $(date -u +%FT%TZ)] $*"; }
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
mkdir -p $RES $LD
if [ "$B" = multipl_e_100 ]; then grep -q "MPE_SETUP_RC=0" $W/logs/mpe_setup.log 2>/dev/null || { say "Q_FAIL MPE toolchain setup not OK"; exit 1; }; fi
[ -f $TOK/tokenizer.json ] || { say "Q_FAIL no tokenizer at $TOK"; exit 1; }
for t in "$@"; do
  if [ "$t" = Q6_K_repeat ]; then
    M=$Q6
    [ "$(sha256sum $M | cut -c1-16)" = $Q6SHA ] || { say "Q_FAIL $t: Q6_K sha"; exit 1; }
  else
    M=$GD/gemma-4-20B-v9-agentic-$t.gguf
    for _ in $(seq 1 240); do [ -f "$M.ok" ] && break; grep -q "FETCH_FAIL $t" $LD/fetch.log 2>/dev/null && break; sleep 15; done
    [ -f "$M.ok" ] || { say "Q_FAIL $t: no fetched GGUF ($M.ok)"; exit 1; }
  fi
  SN=v9-agentic-$t; sj=$RES/$B/$SN/summary.json
  if [ -f "$sj" ]; then   # skip only a SCORED cell; a void summary (score null) is moved aside, never deleted
    if "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$sj')).get('score') is not None else 1)"; then
      say "$t: scored summary exists, skip"; continue; fi
    mv "$sj" "$sj.void_$(date -u +%Y%m%dT%H%M%SZ)"; say "$t: void summary moved aside, re-running (sqlite cache reused)"
  fi
  say "$t: $B start (port $P, --parallel 2, --sampler-profile auto) $(du -h "$M" | cut -f1)"
  ( cd $W/omnimergekit && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model "$M" --tokenizer "$TOK" \
      --template "$B" --backend llama --parallel 2 --port "$P" --served-name "$SN" --results-dir "$RES" \
      --sampler-profile auto > "$LD/$B.$t.log" 2>&1 ) & cp=$!
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
    ok)   say "$t: LIVE sampler OK (llama-server pid $spid)";;
    bad)  say "Q_FAIL $t: LIVE llama-server pid $spid is MISSING the penalty flags: $cl"; killtree $cp; exit 1;;
    none) if kill -0 $cp 2>/dev/null; then say "Q_FAIL $t: no llama-server on port $P after 30 min"; killtree $cp; exit 1; fi;;
  esac
  wait $cp; rc=$?
  if [ -f "$sj" ]; then
    r=$("$PY" -c "import json;d=json.load(open('$sj'));s=d.get('scores') or {};print(d.get('score'),s.get('rs_pass_at_1'),s.get('java_pass_at_1'),s.get('js_pass_at_1'),(d.get('sampler') or {}).get('name'))")
    say "$t: rc=$rc RESULT $r"
    [ "${r##* }" = greedy_pen ] || { say "Q_FAIL $t: summary sampler is not greedy_pen"; exit 1; }
    { [ "$rc" = 0 ] && [ "${r%% *}" != None ]; } || { say "Q_FAIL $t: rc=$rc score=${r%% *} (void cell, see $LD/$B.$t.log)"; exit 1; }
  else
    say "Q_FAIL $t: rc=$rc NO summary.json (see $LD/$B.$t.log)"; exit 1
  fi
  P=$((P+1)); sleep 30
done
say "QUEUE_DONE"
