#!/usr/bin/env bash
# std16_9bench_rerun.sh — run the 8 canonical 9-bench benches that STD16 (the chosen,
# no-DERN release: gemma-4-A4B-98e-v7-coder-it, fk16 + shared a=1.2) is MISSING.
# STD16's Q6 already has greedy GPQA 54.04 / HE+ 92.68 / MPE 89.67 (the t223_fk tradecheck);
# the full 9-bench that ran earlier was on the DERN v8 (fkbroad-soft2), a DIFFERENT model.
# This fills the 8 gaps on STD16's OWN publish GGUF, GREEDY (the cross-variant anchor),
# llama.cpp b9700, ctx 32768 / reasoning-budget 8192 (proven STD16 tradecheck config).
# 2-GPU claim pool, resumable (omk --use_cache + --log_samples internally).
set -uo pipefail
BM=/srv/ml
PY="$BM/envs/envs/omnimergekit/bin/python"
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
OMK="$BM/repos/omnimergekit/eval/omk_eval.py"
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
GGUF="$GG/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf"
TOK=/mnt/sdc/ml/t223_fk/STD16-bf16          # valid HF tokenizer dir (mandatory: GGUF has no sibling)
WORK=/mnt/sdc/ml/std16_gate/9bench
mkdir -p "$WORK/logs" "$WORK/locks" "$WORK/done" "$WORK/results"
LOG="$WORK/run_9bench.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
magic(){ head -c4 "$1" 2>/dev/null; }

# the 8 missing benches (long-pole first so the claim pool balances); GREEDY, no --sampler.
BENCHES=(arc_challenge_full lcb_medium_100_v4 lcb_medium_55_v4 ifeval_100 humaneval_full math500_100 gsm8k_100 aime_30)
# optional smoke/subset: pass bench names as args
[ "$#" -gt 0 ] && BENCHES=("$@")

[ -f "$TOK/tokenizer.json" ] || { echo "FATAL tokenizer dir invalid: $TOK"; exit 1; }
[ "$(magic "$GGUF")" = GGUF ] || { echo "FATAL bad gguf magic: $GGUF"; exit 1; }
echo "==================== STD16 9-bench (8 missing) start $(ts) UTC  benches=${BENCHES[*]} ===================="

run_one(){ # bench gpu port
  local B="$1"
  local G="$2"
  local P="$3"
  local TD="$WORK/results"
  [ -f "$WORK/done/${B}.done" ] && { echo "[$(ts)] $B already done"; return 0; }
  echo "[$(ts)] [GPU$G:$P] $B start (greedy, tok=$TOK)"
  CUDA_VISIBLE_DEVICES=$G "$PY" "$OMK" --model "$GGUF" --tokenizer "$TOK" --template "$B" \
    --backend llama --quant gguf --port "$P" --results-dir "$TD" --served-name STD16 \
    > "$WORK/logs/${B}.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    touch "$WORK/done/${B}.done"
  else
    echo "[$(ts)] $B rc=$rc (NO done-marker; resumable from sqlite cache on re-launch)"
  fi
  echo "[$(ts)] [GPU$G:$P] $B end rc=$rc  score=$(grep -oE 'score=[0-9.]+|score=None' "$WORK/logs/${B}.log" | tail -1)"
}
worker(){ # gpu port
  local G="$1"
  local P="$2"
  local B
  for B in "${BENCHES[@]}"; do
    mkdir "$WORK/locks/$B.lock" 2>/dev/null || continue
    run_one "$B" "$G" "$P"
  done
}
rm -rf "$WORK/locks"/*.lock 2>/dev/null
NW="${NWORKERS:-4}"          # 4 workers = 2 per GPU (Blackwell 97GB: 2x 17.8GB model + KV fits)
for i in $(seq 0 $((NW - 1))); do
  worker "$((i % 2))" "$((8290 + i))" &
done
wait
echo "[$(ts)] ==================== STD16 9-bench DONE $(ts) ===================="
echo "STD16_9BENCH_DONE"
