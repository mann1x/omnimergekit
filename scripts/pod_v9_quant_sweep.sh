#!/usr/bin/env bash
# pod_v9_quant_sweep.sh — ON THE POD: IFEval-100 (or BENCH=<template>) over published v9-agentic GGUF tiers,
# one GPU, rolling disk. Owner 2026-10-17: "evaluate all the others. put the ifeval scores in the GGUF HF
# model card and the ollama model card".
# Per tier: pod_v9_quant_fetch.py (aria2c + LFS sha256 gate + .ok marker) -> pod_v9_quant_mpe.sh <one tier>
# (greedy_pen, --parallel 2, live penalty-flag gate, rc/score gate) -> delete the downloaded GGUF copy (the
# HF repo is the canonical copy; results, logs and sqlite caches are kept). A failed tier is recorded and the
# sweep moves on; the GGUF of a failed tier is KEPT for the rerun.
# usage: [BENCH=ifeval_100] pod_v9_quant_sweep.sh <gpu> <first_port> <tier>...
set -uo pipefail
G=$1; P=$2; shift 2
export BENCH=${BENCH:-ifeval_100}
W=/workspace; S=$W/v9qmpe; LD=$S/logs; GD=$S/gguf
PY=$W/miniconda/envs/omnimergekit/bin/python
say(){ echo "[sweep GPU$G $(date -u +%FT%TZ)] $*"; }
mkdir -p "$LD" "$GD"
for t in "$@"; do
  f=$GD/gemma-4-20B-v9-agentic-$t.gguf
  if ! "$PY" $W/omnimergekit/scripts/pod_v9_quant_fetch.py "$GD" "$t" >> "$LD/fetch_sweep_gpu$G.log" 2>&1; then
    say "CELL_FAIL $t: fetch failed (see $LD/fetch_sweep_gpu$G.log)"; continue
  fi
  avail=$(df --output=avail -BG $W | tail -n1 | tr -dc 0-9)
  say "$t: fetched $(du -h "$f" | cut -f1), disk free ${avail}G"
  out=$(bash $W/omnimergekit/scripts/pod_v9_quant_mpe.sh "$G" "$P" "$t" 2>&1); echo "$out"
  P=$((P+1))
  if grep -q "RESULT" <<<"$out" && grep -q "QUEUE_DONE" <<<"$out" && ! grep -q "Q_FAIL" <<<"$out"; then
    rm -f "$f" "$f.ok"; say "CELL_OK $t (download copy removed)"
  else
    say "CELL_FAIL $t (GGUF kept for rerun)"
  fi
done
say "SWEEP_DONE"
