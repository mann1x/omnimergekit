#!/usr/bin/env bash
# COMPOSITE, BUDGET 8192 *WITH* THE CAP MESSAGE — the third cell.
#
# WHY. The budget does not stop rumination, it relocates it into the message
# channel, and the switch is nearly transparent to the model: it keeps going as
# before, just outside the thinking block. What makes the cap actionable is
# TELLING the model it has hit it. llama.cpp already supports this in the build
# we serve: --reasoning-budget-message injects the text before the forced
# end-of-thinking tag (common/reasoning-budget.cpp forces `message + end_tag`
# token by token), so no patch and no rebuild are needed.
#
# The message is the one our ollama builds ship as think_budget_message, VERBATIM
# (including its missing space after "needed." — matching production matters more
# than tidiness).
#
# BASIS. One variable against the existing capped armB cell (bfcl_armB_comp):
# same GGUF, same libllama, same template, same -c/-np/max_tokens/greedy, same
# --reasoning-format deepseek, same --reasoning-budget 8192. ONLY the message is
# added.
#
# SCOPE: the 29 tasks that actually HIT the cap in the no-message run, by id, via
# --sample-id. Those are the only tasks that carry any information about a message
# that fires on a cap hit; the other 171 never reach the budget and would spend
# ~85% of the GPU time producing identical trajectories. The restriction is stated,
# the ids are frozen in capmsg_ids.csv, and the comparison is paired within task.
set -uo pipefail
L=/mnt/sdc/harness/logs
TP=/mnt/sdc/v7rework/template_probe
GGUF=/srv/ml/models/gguf/google-a4b-128e/google_gemma-4-26B-A4B-it-Q4_K_M-eos106.gguf
LC=/srv/ml/tools/llama.cpp/build-pegfix/bin
VENV=/mnt/sdc/harness/venv
REPO=/srv/ml/repos/omnimergekit
NP=8; CTX=$(( 262144 * NP )); MAXTOK=32768; THINK=8192; FMT=deepseek
GPU=1; PORT=8503; TAG=armB_capmsg29
IDS_FILE=/mnt/sdc/harness/capmsg_ids.csv
NEED_MIB=${NEED_MIB:-45000}
MSG="I have used my thinking budget. I must stop analysing now and act on what I have: make the tool call, or give a short final answer if no call is needed.I'll be more terse and concise now and maybe I need to consider a different approach."
export OMK_COHORT="gemma4-128e-template-ab-256k-CAPMSG-2026-09-13"
export OMK_DECLARED_VARIABLE="reasoning-budget-message present (vs absent); budget 8192 both"
exec >> "$L/comp_capmsg.log" 2>&1
echo "=================================================================="
echo ">>> CAP-MESSAGE CELL start $(date -u +%FT%TZ) THINK=$THINK GPU=$GPU PORT=$PORT"

for f in "$GGUF" "$TP/v7coder_chat_template.jinja" "$LC/llama-server" "$VENV/bin/inspect" \
         "$REPO/eval/agentic/write_stack_bfcl.sh" "$IDS_FILE"; do
  [ -e "$f" ] || { echo "CAPMSG_ABORT missing input: $f"; exit 1; }
done
# NOT `... | grep -q ...`: grep -q exits on first match, the writer takes SIGPIPE,
# and under `set -o pipefail` the pipeline reports failure even though the flag IS
# present. Capture first, match after.
HELP=$("$LC/llama-server" --help 2>&1 || true)
case "$HELP" in
  *--reasoning-budget-message*) ;;
  *) echo "CAPMSG_ABORT this llama-server has no --reasoning-budget-message"; exit 1 ;;
esac
echo ">>> input preflight PASSED (binary supports --reasoning-budget-message)"

# gate on THIS GPU only: armA is still running the unbounded cell on GPU0.
ok=0
for i in $(seq 1 240); do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -dc "0-9"); : "${F:=0}"
  [ "$F" -ge "$NEED_MIB" ] && { ok=1; break; }
  echo "    waiting: gpu$GPU free=${F}MiB"; sleep 30
done
[ "$ok" -eq 1 ] || { echo "CAPMSG_ABORT gpu$GPU never had ${NEED_MIB}MiB free"; exit 1; }
echo ">>> VRAM gate PASSED (gpu$GPU free=${F}MiB)"

CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$LC:${LD_LIBRARY_PATH:-}" \
nohup "$LC/llama-server" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -np "$NP" -ngl 99 --no-warmup -t 16 \
  --jinja --chat-template-file "$TP/v7coder_chat_template.jinja" \
  --reasoning-format "$FMT" --reasoning-budget "$THINK" \
  --reasoning-budget-message "$MSG" \
  > "$L/server_capmsg.log" 2>&1 &
disown
ok=0
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { ok=1; break; }
  sleep 5
done
[ "$ok" -eq 1 ] || { echo "CAPMSG_ABORT server never healthy"; tail -8 "$L/server_capmsg.log"; exit 1; }
echo ">>> server healthy $(date -u +%FT%TZ)"

export LLAMACPP_BASE_URL="http://127.0.0.1:$PORT/v1" LLAMACPP_API_KEY=none HF_HOME=/mnt/sdc/harness/hf
D="$L/bfcl_$TAG"; mkdir -p "$D"
bash "$REPO/eval/agentic/write_stack_bfcl.sh" "$TAG" "$D" "$TP/v7coder_chat_template.jinja" \
     "$GGUF" "$LC" "$VENV" "$CTX" "$NP" "$MAXTOK" "$THINK" multi_turn_composite 29 "$FMT" \
  || { echo "CAPMSG_ABORT stack writer failed (no provenance -> no run)"; exit 1; }
printf "\n## cap message (verbatim, as passed to --reasoning-budget-message)\n%s\n" "$MSG" >> "$D/STACK.txt"

"$VENV/bin/inspect" eval inspect_evals/bfcl -T categories=multi_turn_composite \
  --model "openai-api/llamacpp/$TAG" --sample-id "$(cat $IDS_FILE)" \
  --max-connections "$NP" --max-tokens "$MAXTOK" --temperature 0.0 \
  --log-dir "$D" --no-fail-on-error > "$L/run_$TAG.log" 2>&1
echo "CAPMSG_DONE $(date -u +%FT%TZ)"
