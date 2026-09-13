#!/usr/bin/env bash
# COMPOSITE, UNBOUNDED THINKING — the budget-censoring control.
#
# WHY. The 8192-budget composite cell is where the significant looping result
# lives, and on that cell the budget BITES: 2.19% (armA) / 3.28% (armB) of
# generations pile up at 8192 and fall off a cliff above it. Two problems:
#   1. CENSORING, and it is ASYMMETRIC — armB is cut ~50% more often, which
#      gives a nascent loop less room to reach the share threshold and can
#      inflate the armA-vs-armB gap in the direction we reported.
#   2. The cap may itself CAUSE degeneration: an interrupted think is
#      force-closed with </think>, and the LIVE template then RE-INJECTS that
#      truncated (possibly already-looping) block into the next turn, seeding
#      more of it. If so, cap and template INTERACT and the 8192 cell cannot
#      separate them.
# This arm re-runs the identical cohort with the budget removed, giving a 2x2:
#   {live, fixed} x {budget 8192, unbounded}.
#
# ONLY THE BUDGET CHANGES. --reasoning-format auto is used as agreed; in
# llama.cpp every branch tests `!= COMMON_REASONING_FORMAT_NONE`, so auto and
# deepseek are behaviourally identical here and the budget stays the single
# effective variable. Everything else (ctx/slot, np, max_tokens, greedy,
# templates, GGUF, engine build) is byte-identical to the 8192 run.
set -uo pipefail
L=/mnt/sdc/harness/logs
TP=/mnt/sdc/v7rework/template_probe
GGUF=/srv/ml/models/gguf/google-a4b-128e/google_gemma-4-26B-A4B-it-Q4_K_M-eos106.gguf
LC=/srv/ml/tools/llama.cpp/build-pegfix/bin
VENV=/mnt/sdc/harness/venv
REPO=/shared/dev/omnimergekit
NP=8; CTX=$(( 262144 * NP )); MAXTOK=32768; THINK=-1        # <-- the variable
NEED_MIB=${NEED_MIB:-45000}
export OMK_COHORT="gemma4-128e-template-ab-256k-UNBOUNDED-2026-09-13"
export OMK_DECLARED_VARIABLE="reasoning-budget -1 (vs 8192); chat_template as labelled"
exec >> "$L/comp_unbounded.log" 2>&1
echo "=================================================================="
echo ">>> COMPOSITE UNBOUNDED start $(date -u +%FT%TZ)  THINK=$THINK"

WAIT_FOR="${WAIT_FOR:-}"
if [ -n "$WAIT_FOR" ]; then
  ok=0
  for i in $(seq 1 960); do
    all=1
    for spec in $WAIT_FOR; do
      k=${spec%%:*}; f=${spec#*:}
      n=$(grep -ac "$k" "$f" 2>/dev/null); n=${n:-0}
      [ "$n" -ge 1 ] || all=0
    done
    [ "$all" -eq 1 ] && { ok=1; break; }
    sleep 30
  done
  [ "$ok" -eq 1 ] || { echo "COMP_UNBOUNDED_ABORT chain wait TIMED OUT on: $WAIT_FOR"; exit 2; }
  echo ">>> chain markers seen $(date -u +%FT%TZ)"
fi

# never start on a busy box; never kill anyone else
ok=0
for i in $(seq 1 240); do
  NS=$(ps -eo args | grep -c "[l]lama-server"); NI=$(ps -eo args | grep -c "[i]nspect eval")
  F0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -dc "0-9")
  F1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -dc "0-9")
  : "${F0:=0}"; : "${F1:=0}"
  if [ "$NS" -eq 0 ] && [ "$NI" -eq 0 ] && [ "$F0" -ge "$NEED_MIB" ] && [ "$F1" -ge "$NEED_MIB" ]; then ok=1; break; fi
  echo "    waiting: llama-server=$NS inspect=$NI gpu0=${F0} gpu1=${F1}"
  sleep 30
done
[ "$ok" -eq 1 ] || { echo "COMP_UNBOUNDED_ABORT box never idle $(date -u +%FT%TZ)"; exit 1; }
echo ">>> isolation gate PASSED (gpu0=$F0 gpu1=$F1 MiB free)"

for f in "$GGUF" "$TP/google_live_chat_template.jinja" "$TP/v7coder_chat_template.jinja" \
         "$LC/llama-server" "$VENV/bin/inspect" "$REPO/eval/agentic/write_stack_bfcl.sh"; do
  [ -e "$f" ] || { echo "COMP_UNBOUNDED_ABORT missing input: $f"; exit 1; }
done
echo ">>> input preflight PASSED"

serve() { # gpu port tpl tag
  CUDA_VISIBLE_DEVICES="$1" LD_LIBRARY_PATH="$LC:${LD_LIBRARY_PATH:-}" \
  nohup "$LC/llama-server" -m "$GGUF" --host 127.0.0.1 --port "$2" \
    -c "$CTX" -np "$NP" -ngl 99 --no-warmup -t 16 \
    --jinja --chat-template-file "$3" \
    --reasoning-format auto --reasoning-budget "$THINK" \
    > "$L/server_unb_$4.log" 2>&1 &
  disown
  for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$2/health" 2>/dev/null)" = "200" ] && return 0
    sleep 5
  done
  echo "COMP_UNBOUNDED_ABORT server :$2 never healthy"; tail -8 "$L/server_unb_$4.log"; return 1
}
serve 0 8501 "$TP/google_live_chat_template.jinja" A || exit 1
serve 1 8502 "$TP/v7coder_chat_template.jinja"     B || exit 1
echo ">>> both servers healthy $(date -u +%FT%TZ)"

run() { # tag port tpl
  export LLAMACPP_BASE_URL="http://127.0.0.1:$2/v1" LLAMACPP_API_KEY=none
  export HF_HOME=/mnt/sdc/harness/hf
  D="$L/bfcl_$1"; mkdir -p "$D"
  bash "$REPO/eval/agentic/write_stack_bfcl.sh" "$1" "$D" "$3" "$GGUF" "$LC" "$VENV" \
       "$CTX" "$NP" "$MAXTOK" "$THINK" multi_turn_composite 200 \
    || { echo "COMP_UNBOUNDED_ABORT stack writer failed for $1 (no provenance -> no run)"; return 1; }
  "$VENV/bin/inspect" eval inspect_evals/bfcl -T categories=multi_turn_composite \
    --model "openai-api/llamacpp/$1" --limit 200 \
    --max-connections "$NP" --max-tokens "$MAXTOK" --temperature 0.0 \
    --log-dir "$D" --no-fail-on-error > "$L/run_$1.log" 2>&1
  echo "BFCL_$1_DONE $(date -u +%FT%TZ)" >> "$L/run_$1.log"
}
run armA_comp_unb 8501 "$TP/google_live_chat_template.jinja" &
PA=$!
run armB_comp_unb 8502 "$TP/v7coder_chat_template.jinja" &
PB=$!
wait "$PA"; wait "$PB"
echo "COMP_UNBOUNDED_DONE $(date -u +%FT%TZ)"
