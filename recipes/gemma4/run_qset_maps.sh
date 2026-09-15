#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_qset_maps.sh — build one competence map per q-set, sequentially on GPU0.
#
# PURPOSE (owner's convergence plan): two INDEPENDENT 1-trace-per-language maps.
# If they rank experts the same, 1q/language suffices; if they move marginally,
# 2q; if they diverge, keep adding sets until the movement flattens. Comparing
# two independent 1q maps is a sampling-stability test, which is a stronger
# statement than comparing A against A+B.
#
# TIER-A IS SKIPPED ON PURPOSE. The generic_* categories are identical across
# q-sets, so including them would cost hours and add nothing to an A-vs-B
# comparison. Consequence to remember: make_drop_map's --floor-count needs the
# generic categories, so these maps are for COMPARISON only. The production map
# runs Tier-A once and reuses it via --load-tier-a-from.
#
# THE TEMPLATE IS LOAD-BEARING. reasoning_content is 69-90% of these traces and
# reaches the forward pass only under the GENERATION-TIME template with
# preserve_thinking=True. The stock model-dir template drops it SILENTLY -- the
# run would look fine and profile ~30% of the signal. Hence --chat-template-file
# is passed explicitly and gated below.
# ---------------------------------------------------------------------------
set -uo pipefail

PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
REPO=/srv/ml/repos/omnimergekit
MODEL=/mnt/sdc/v7rework/base/gemma-4-26B-A4B-it
TPL=/mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja
TB=/mnt/sdc/agentbench/tierb
# Tier-A source. --skip-tier-a HARD-FAILS without it (by design: better than
# discovering an incomplete map after an hour of Tier-B). This file is the May-15
# Gemma-4 v5-code map -- the only one using the `generic_*` naming the loader
# requires; the Apr-08 expert_neuron_v5.json uses bare category names and is both
# older and unusable here. The loader imports ONLY generic_* categories, so its
# stale targeted_humaneval / _lcb_medium_55 cannot contaminate the new map.
TIERA=$TB/tier_a_source_gemma4.json
LOG=$TB/qset_maps.log
SETS="${*:-A B}"

say(){ echo "[qmap $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

for F in "$PY" "$REPO/scripts/expert_neuron_analysis_v5_targeted.py" "$TPL" "$TIERA"; do
  [ -e "$F" ] || { say "FATAL: missing $F"; exit 1; }
done
[ -d "$MODEL" ] || { say "FATAL: model dir missing: $MODEL"; exit 1; }

# The template must be the one that preserves thinking; a stock template has no
# reasoning support and would silently discard most of the corpus.
if ! grep -q "preserve_thinking" "$TPL"; then
  say "FATAL: $TPL has no preserve_thinking support — it would drop reasoning SILENTLY."
  say "       Refusing rather than profiling 30% of the intended signal."
  exit 1
fi
say "template gate OK: preserve_thinking present in $(basename $TPL)"

U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
say "GPU0 in use: ${U} MiB"
[ "${U:-999999}" -lt 2000 ] || { say "FATAL: GPU0 is not free"; exit 1; }

for S in $SETS; do
  IN=$TB/qset_${S}.json
  OUT=$TB/map_qset${S}.json
  [ -f "$IN" ] || { say "SKIP set $S — $IN missing"; continue; }
  N=$($PY -c "import json;print(len(json.load(open('$IN'))['traces']))")
  say "=== SET $S — $N traces -> $OUT ==="
  CUDA_VISIBLE_DEVICES=0 $PY "$REPO/scripts/expert_neuron_analysis_v5_targeted.py" \
      --variant code \
      --load-tier-a-from "$TIERA" \
      --tier-b-json "$IN" \
      --out "$OUT" \
      --model "$MODEL" \
      --chat-template-file "$TPL" \
      --skip-tier-a \
      --gpu-budget-gib 80 \
      --device cuda --dtype bfloat16 \
      >> "$LOG" 2>&1
  RC=$?
  say "SET $S rc=$RC  out=$([ -f "$OUT" ] && du -h "$OUT" | cut -f1 || echo MISSING)"
  [ "$RC" -eq 0 ] || { say "ABORT: set $S failed — not starting the next set"; exit 1; }
done
say "QSET_MAPS_DONE"
