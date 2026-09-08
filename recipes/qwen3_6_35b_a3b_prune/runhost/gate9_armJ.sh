#!/usr/bin/env bash
# CoderX publish gate 1/2: canonical no-regression benches, armJ (ours + REAP floor p=24)
# vs pub184e (the published cut), on ONE pinned geometry per bench.
#
# WHY A FRESH pub184e ROW. qwen_suite already has pub184e cells, but they were run on a
# different host/geometry and one of them (lcb_v6_77q) is the confirmed bug-597 casualty.
# Re-running the comparator here costs 2x but makes the CoderX table self-contained and
# matched. Arms run back-to-back per bench so the pair is temporally adjacent.
#
# GEOMETRY. llama_ctx is the TOTAL, divided by --parallel. We derive the pin from each
# template's own max_gen_toks: per_slot = max_gen_toks + 16384 headroom, total = per_slot*PAR.
# T172.4: per-slot ctx MUST exceed prompt + max_gen_toks or generation collapses silently.
# The readback gate is FATAL -- a cell that did not get its geometry is aborted, not banked.
set -u
export CUDA_VISIBLE_DEVICES=1          # bs2 GPU1 is ours; GPU0 is NOT.
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
command -v lm-eval >/dev/null || { echo "REFUSING: lm-eval not on PATH"; exit 1; }

OMK=/srv/ml/repos/omnimergekit; OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
T=$OMK/eval/templates; WORK=/mnt/sdc/ream-work; GG=$WORK/gguf
RES=/srv/ml/eval_results/ream_arms; PORT=8099; PAR=2
PUB_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
PUB_T=/srv/ml/models/Qwen3.6-35B-A3B-184e-coder-lcbmpe
say() { echo "[gate9 $(date -u +%H:%M:%S)Z] $*"; }

ARMS=( "armJ|$GG/armJ_imat/armJ-Q6_K.gguf|$WORK/armJ"
       "pub184e|$PUB_G|$PUB_T" )
BENCHES=(gpqa_diamond_full aime_30 gsm8k_100_boxed math500_100 arc_challenge_full ifeval_100)

ran=0; failed=0; aborted=0
for B in "${BENCHES[@]}"; do
  MGT=$(grep -oE "^ *max_gen_toks: *[0-9]+" "$T/$B.yaml" | grep -oE "[0-9]+$" | head -1)
  [ -n "$MGT" ] || { say "SKIP $B: cannot read max_gen_toks"; failed=$((failed+1)); continue; }
  SLOT=$(( MGT + 16384 )); TOTAL=$(( SLOT * PAR ))
  say "===== BENCH $B  max_gen_toks=$MGT -> per_slot=$SLOT total=$TOTAL par=$PAR"
  for spec in "${ARMS[@]}"; do
    arm=${spec%%|*}; rest=${spec#*|}; g=${rest%%|*}; tok=${rest#*|}
    name="gate9_${arm}"
    [ -s "$g" ] || { say "SKIP $B/$arm: gguf missing $g"; failed=$((failed+1)); continue; }
    [ -f "$RES/$B/$name/summary.json" ] && { say "SKIP $B/$name (exists)"; continue; }
    say "---- $B / $name"
    "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$B" --quant q6_k \
        --model "$g" --tokenizer "$tok" --served-name "$name" --port "$PORT" \
        --results-dir "$RES" --parallel "$PAR" --metadata backend_args.llama_ctx=$TOTAL
    say "<<<< END $B/$name rc=$?"
    L="$RES/$B/$name/server.log"
    got=$(grep -aoE "new slot, n_ctx = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    slots=$(grep -aoE "n_slots = [0-9]+" "$L" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    say "GEOMETRY $B/$name per_slot=${got:-unknown} slots=${slots:-unknown} (want $SLOT / $PAR)"
    if [ "${got:-0}" != "$SLOT" ] || [ "${slots:-0}" != "$PAR" ]; then
      say "GATE9_ABORT $B/$name: geometry not honoured -- cell NOT comparable"
      aborted=$((aborted+1)); continue; fi
    s="$RES/$B/$name/summary.json"
    if [ -f "$s" ]; then ran=$((ran+1))
      say "SCORE $B/$name = $("$OMKPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('score'),d.get('metric'),d.get('filter'))" "$s")"
    else say "FAIL $B/$name: no summary.json"; failed=$((failed+1)); fi
  done
done
say "=== GATE9_DONE ran=$ran failed=$failed aborted=$aborted ==="
