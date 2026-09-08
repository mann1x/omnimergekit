#!/usr/bin/env bash
# LCB-v6-77q @ 24k thinking / 48k total -- the CAP-VALIDATION pass over every arm.
#
# WHY
# ---
# At 12k/32k the truncation rate tracked the arm, not the bench: armE 6/77 length-terminated
# vs armB 55/77 (71.4%, MEDIAN completion 32,166 tok against a 32,768 ceiling). A ceiling that
# binds 9x harder on one arm than another is part of what is being measured. armB's 0.4675 is
# truncation-taxed and cannot be compared to armE's 0.6494 at the same wall. This pass moves
# the wall out (max_gen_toks 32768->49152, thinking 12288->24576) on the SAME 77 problems and
# asks whether the ordering survives.
#
# This is a SECOND BASIS, not a replacement. The 12k/32k cells stay; the new template has its
# own name (so its own results dir) and its own sqlite prefix (so it cannot resume the old
# truncated generations -- the LCB cache keys on task_id ONLY, max_tokens is not in the key,
# and sharing a prefix would silently re-serve the 32k completions and fake a null result).
# Never merge rows from the two ceilings into one table.
#
# GEOMETRY GATE. Per-slot ctx must exceed max_gen_toks or slots saturate and generations
# collapse silently (T172.4). 524288/8 = 65536 > 49152. armD runs FIRST and is checked on
# BOTH axes before the chain commits hours to armB:
#   (a) server.log must show `new slot, n_ctx = 65536`  -- geometry actually applied
#   (b) score must be >= FLOOR                          -- raising a cap must not lower a
#       clean column; armD read 0.6104 at 32k with only 6/77 truncated, so a collapse here
#       means the 512k KV or the 24k reasoning budget broke something, not that the model did.
# Either check failing aborts the whole chain rather than producing 8 more unusable columns.
#
# TRUNCATION READOUT: token_stats.finish_reasons.length, NOT OMK_CAP_CHECK. That sentinel
# compares re-tokenized length against max_gen_toks, undershoots the server's own count, and
# reported verdict=CLEAN capped=0/77 on the 71%-truncated armB column (bug-592).
#
# GPU1 only. Exported, not polled.
set -uo pipefail

WORK=/mnt/sdc/ream-work
GG=$WORK/gguf
OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
LOG=$WORK/chain_lcb48k.log
PORT=${PORT:-8099}
TMPL=lcb_v6_77q_48k
FLOOR=${FLOOR:-0.45}     # armD read 0.6104 at 32k; well below that = the geometry broke
WANT_CTX=65536

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"

exec >>"$LOG" 2>&1
say() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }
echo "=== chain_lcb48k start $(date -u +%F' '%T) ==="
[ -s "$TMPL_FIX" ] || { echo "ABORT: fixed chat template missing"; exit 2; }
[ -s "$OMK/eval/templates/$TMPL.yaml" ] || { echo "ABORT: $TMPL.yaml not on this host"; exit 2; }

BASE_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-256e-GGUF/Qwen3.6-35B-A3B-Q6_K.gguf
BASE_T=/srv/ml/models/Qwen3.6-35B-A3B
PUB_G=/srv/ml/models/gguf/Qwen3.6-35B-A3B-184e-coder-lcbmpe-GGUF/Qwen3.6-35B-A3B-184e-coder-lcbmpe-Q6_K.gguf
PUB_T=/srv/ml/models/Qwen3.6-35B-A3B-184e-coder-lcbmpe

# ---- wait for the hybrid chain to release GPU1 --------------------------------------------
# Live process first, then sentinel. If chain_hybrid never ran (hybrids not wanted), the
# process check falls through immediately and the two hybrid columns simply get SKIPped below
# for want of a GGUF -- the rest of the matrix still runs.
waited=0
while pgrep -f "chain_hybrid.sh" >/dev/null 2>&1; do
  sleep 120; waited=$((waited+120))
  [ $((waited % 1800)) -eq 0 ] && say "hybrid chain still running (${waited}s)"
  if [ $waited -ge 43200 ]; then say "ABORT: hybrid chain still running after 12h"; exit 3; fi
done
say "hybrid chain not running (waited ${waited}s)"
for _ in $(seq 1 30); do
  free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 60000 ] && break
  say "GPU1 only ${free}MiB free, waiting"; sleep 60
done

# name | gguf-or-dir | tokenizer
# Order = decisiveness, with armD first as the geometry+sanity gate (fast, clean, and its 32k
# value is known). armB second: it is the column this whole pass exists to disambiguate.
COLS=(
  "lcb48k_armD_ourssal_nomerge|$GG/armD_ourssal_nomerge_imat|$WORK/armD_ourssal_nomerge"
  "lcb48k_armB_reapsal_merge|$GG/armB_imat|$WORK/armB"
  "lcb48k_armE_reapsal_nomerge|$GG/armE_imat|$WORK/armE"
  "lcb48k_base256e|$BASE_G|$BASE_T"
  "lcb48k_armG_merge_gs2|$GG/armG_imat|$WORK/armG"
  "lcb48k_armF_rnorm_nomerge|$GG/armF_rnorm_nomerge_imat|$WORK/armF_rnorm_nomerge"
  "lcb48k_armI_hybrid_p12|$GG/armI_imat|$WORK/armI"
  "lcb48k_armJ_hybrid_p24|$GG/armJ_imat|$WORK/armJ"
  "lcb48k_pub184e|$PUB_G|$PUB_T"
)

resolve() { local p=$1; if [ -f "$p" ]; then echo "$p"; else ls "$p"/*-Q6_K.gguf 2>/dev/null | head -1; fi; }
score_of() { "$OMKPY" -c "
import json,sys
try: print(json.load(open(sys.argv[1])).get('score'))
except Exception: print('')" "$1" 2>/dev/null; }
trunc_of() { "$OMKPY" -c "
import json,sys
try:
    t=json.load(open(sys.argv[1]))['token_stats']
    c,f=t['completion_tokens'],t['finish_reasons']
    print(f\"len={f.get('length',0)}/{t['n']} stop={f.get('stop',0)} p50={c['p50']} max={c['max']}\")
except Exception as e: print(f'(no token_stats: {e})')" "$1" 2>/dev/null; }

ran=0; failed=0; first=1
for col in "${COLS[@]}"; do
  IFS='|' read -r name src tok <<<"$col"
  g=$(resolve "$src")
  if [ -z "$g" ] || [ ! -s "$g" ]; then say "SKIP $name: no Q6_K under $src"; failed=$((failed+1)); continue; fi
  s=$RES/$TMPL/$name/summary.json
  if [ -f "$s" ]; then say "SKIP $name (summary exists)"; first=0; continue; fi

  say ">>>> START $name (greedy, 24k think / 48k gen, parallel=8)"
  t0=$SECONDS
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$TMPL" --quant q6_k \
      --model "$g" --tokenizer "$tok" --served-name "$name" --port "$PORT" \
      --results-dir "$RES" --parallel 8
  say "<<<< END $name rc=$? in $(( (SECONDS-t0)/60 ))m"

  if [ -f "$s" ]; then
    ran=$((ran+1))
    v=$(score_of "$s")
    say "SCORE $name = $v   TRUNC $(trunc_of "$s")"
  else
    failed=$((failed+1)); say "FAIL $name: no summary.json"
    v=""
  fi

  if [ "$first" -eq 1 ]; then
    # (a) geometry actually applied?
    slog=$RES/$TMPL/$name/server.log
    ctx=$(grep -a -o "new slot, n_ctx = [0-9]*" "$slog" 2>/dev/null | head -1 | tr -dc 0-9)
    say "GEOMETRY per-slot n_ctx=${ctx:-unknown} (want $WANT_CTX)"
    if [ "${ctx:-0}" -ne "$WANT_CTX" ]; then
      say "ABORT: per-slot ctx is ${ctx:-unknown}, not $WANT_CTX. At 49152 max_gen_toks that"
      say "       is the T172.4 saturation trap. Fix llama_ctx/parallel before re-running."
      echo ">>> LCB48K_ABORT_GEOMETRY ctx=${ctx:-unknown}"
      exit 4
    fi
    # (b) did raising the cap break a column that was clean at 32k?
    bad=$("$OMKPY" -c "
import sys
try: sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
except Exception: sys.exit(0)" "$v" "$FLOOR"; echo $?)
    if [ "$bad" = "0" ]; then
      say "ABORT: $name scored '$v' < floor $FLOOR. It read 0.6104 at 32k with only 6/77"
      say "       truncated, so raising the ceiling should not have moved it. Something in"
      say "       the 512k KV or the 24576 reasoning budget is broken -- not the model."
      echo ">>> LCB48K_ABORT_SANITY score=$v"
      exit 5
    fi
    say "gate OK (ctx=$ctx, score=$v >= $FLOOR) -- continuing with the remaining columns"
    first=0
  fi
done

say "=== chain_lcb48k done ran=$ran failed=$failed ==="
"$OMKPY" "$WORK/lcb_readout.py" $(for c in "${COLS[@]}"; do echo "$RES/$TMPL/${c%%|*}"; done) 2>&1 || true
echo ">>> LCB48K_CHAIN_DONE ran=$ran failed=$failed"
