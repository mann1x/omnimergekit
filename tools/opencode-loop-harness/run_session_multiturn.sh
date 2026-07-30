#!/usr/bin/env bash
# run_session_multiturn.sh — drive ONE multi-turn ADVERSARIAL opencode session: build a
# task, then send escalating "still broken / still not fullscreen" follow-ups to rebuild the
# deep frustrating context that produces the field loop. Captures every turn through one
# per-session wire_proxy, then compacts. Faithful reproduction of the rid-37 loop origin.
#
#   MODEL_PORT=8101 MODEL_LABEL=std16-field TASK_ID=snake-adversarial PROXY_PORT=8095 \
#   PER_TURN_TIMEOUT=300 bash run_session_multiturn.sh
set -uo pipefail
# Host paths are env-overridable; the defaults are the bs2 reference rig.
ROOT="${OPENCODE_CAPTURE_ROOT:-/mnt/sdc/ml/opencode_capture}"
PY="${OMK_PYTHON:-/srv/ml/envs/envs/omnimergekit/bin/python}"
OPENCODE="${OPENCODE_BIN:-/root/.opencode/bin/opencode}"
OCLOG="${OPENCODE_LOG:-/root/.local/share/opencode/log/opencode.log}"
SESSROOT="$ROOT/sessions"

MODEL_PORT="${MODEL_PORT:?need MODEL_PORT}"
MODEL_LABEL="${MODEL_LABEL:?need MODEL_LABEL}"
MODEL_NAME="${MODEL_NAME:-std16-q6}"   # opencode model id sent to the upstream (llama.cpp: std16-q6; ollama: the tag)
TASK_ID="${TASK_ID:-snake-adversarial}"
PROXY_PORT="${PROXY_PORT:-8095}"
PER_TURN_TIMEOUT="${PER_TURN_TIMEOUT:-1800}"   # 1800s: a dense 31B doing real tool work needs far more than 600s per agentic
                                              # turn. 600s under-delivered turns (only 11/82 std16-field
                                              # sessions ever reached turn 8), which is the comparability
                                              # axis that matters. See README "budget vs verdict".

INIT="${INIT:-Build a polished terminal Snake game in Python in a single file snake.py using the curses library: full-screen responsive board, arrow-key control, a growing snake, food, a live score, speed that increases as you eat, a game-over screen, and high-score persistence to high_scores.json. After writing it, run it to check for syntax and runtime errors and fix any issues so it imports and initializes cleanly.}"

# Escalating, unsatisfiable visual complaints (the rid-37 frustration engine). Each demands a
# fix to something the model cannot verify headlessly, so it keeps trying across many turns.
FOLLOWUPS=(
"It still isn't fullscreen — the board only fills part of the terminal and there's a big empty margin on the right and bottom. Make it truly fill the entire screen."
"No, still broken. Now the snake flickers badly on every move and the score line overlaps the top border. Fix the flicker and the layout."
"Still not right — when I resize the terminal it crashes, and in fullscreen the food is drawn outside the play area. Make resize work and keep everything inside the border."
"It's STILL not fullscreen and the colors don't show at all. The whole thing must fill the screen edge to edge with colored graphics. Fix it properly this time."
"Nope, still wrong. The borders are doubled, the snake leaves a trail that never clears, and it's still not using the full height. Fix all of it."
"This is still completely broken — not fullscreen, trail not clearing, flicker, and it freezes after a few seconds. Make it actually work, fully fullscreen, smooth, no artifacts."
"Still not fullscreen. I don't understand why you can't make the board fill the whole terminal. Do whatever it takes — make it fill the entire screen now."
)

# FOLLOWUPS_FILE (optional): one escalating adversarial follow-up per line -> overrides the
# default snake array so this same frustration engine drives ANY task family (TUI/web/refactor/
# data-pipeline/API/algo/systems). Blank lines and #comments are skipped. Backward-compatible:
# unset FOLLOWUPS_FILE keeps the snake array verbatim. Combine with INIT= per task.
if [ -n "${FOLLOWUPS_FILE:-}" ] && [ -r "${FOLLOWUPS_FILE:-}" ]; then
    FOLLOWUPS=()
    while IFS= read -r _fl; do
        case "$_fl" in ""|'#'*) continue;; esac
        FOLLOWUPS+=("$_fl")
    done < "$FOLLOWUPS_FILE"
    echo "[mt] loaded ${#FOLLOWUPS[@]} followups from $FOLLOWUPS_FILE"
fi

ts=$(date +%Y%m%d-%H%M%S)
SID="${ts}_${MODEL_LABEL}_${TASK_ID}"
SDIR="$SESSROOT/$SID"
mkdir -p "$SDIR/root" "$SDIR/wirelog"

cat > "$SDIR/root/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": { "local-llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local model under test",
      "options": { "baseURL": "http://127.0.0.1:${PROXY_PORT}/v1", "apiKey": "local" },
      "models": { "${MODEL_NAME}": { "name": "model under test", "tools": true, "reasoning": true } } } },
  "model": "local-llama/${MODEL_NAME}"
}
JSON

curl -s -m8 "http://127.0.0.1:${MODEL_PORT}/props" > "$SDIR/server_props.json" 2>/dev/null || echo '{}' > "$SDIR/server_props.json"

# one proxy for the WHOLE multi-turn session -> all turns land in one wire log
pkill -f "wire_proxy.py --listen 127.0.0.1:${PROXY_PORT} " 2>/dev/null || true
sleep 1
setsid "$PY" "$ROOT/wire_proxy.py" --listen "127.0.0.1:${PROXY_PORT}" \
    --upstream "127.0.0.1:${MODEL_PORT}" --logdir "$SDIR/wirelog" --rawdir "$SDIR/wirelog/raw" \
    > "$SDIR/proxy.log" 2>&1 < /dev/null &
disown
for _ in $(seq 1 20); do curl -s -m2 "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1 && break; sleep 0.5; done

reap() { pkill -9 -f "$SDIR/root" 2>/dev/null || true; sleep 1; }  # SCOPED to this session's --dir -> concurrency-safe (two batches won't kill each other's opencode)

OC_SID=""
ANY_TIMEOUT=0
turn_no=0
run_turn() {  # $1 = message
    local msg="$1"; reap; turn_no=$((turn_no+1))
    local args=(run --dir "$SDIR/root" --model "local-llama/$MODEL_NAME" --format json
                --dangerously-skip-permissions --log-level INFO)
    [ -n "$OC_SID" ] && args=(run --session "$OC_SID" --dir "$SDIR/root" --model "local-llama/$MODEL_NAME"
                              --format json --dangerously-skip-permissions --log-level INFO)
    echo "--- TURN $turn_no (sid=${OC_SID:-NEW}) ---" >> "$SDIR/opencode.log"
    HOME=/root timeout --signal=TERM --kill-after=20 "$PER_TURN_TIMEOUT" "$OPENCODE" "${args[@]}" \
        "$msg" >> "$SDIR/opencode.log" 2>&1
    local rc=$?
    [ $rc -eq 124 ] || [ $rc -eq 137 ] && ANY_TIMEOUT=1
    if [ -z "$OC_SID" ]; then
        OC_SID=$(grep "directory=$SDIR/root" "$OCLOG" 2>/dev/null | grep -oE "ses_[A-Za-z0-9]+" | tail -1)
    fi
    return $rc
}

echo "[mt] $SID upstream=:$MODEL_PORT proxy=:$PROXY_PORT per_turn=${PER_TURN_TIMEOUT}s followups=${#FOLLOWUPS[@]}"
start=$(date +%s)
run_turn "$INIT"; rc=$?
echo "[mt] turn 1 (init) rc=$rc sid=$OC_SID"
MAXF="${N_FOLLOWUPS:-${#FOLLOWUPS[@]}}"
for f in "${FOLLOWUPS[@]:0:$MAXF}"; do
    fexp=$(printf '%b' "$f")           # decode — etc.
    run_turn "$fexp"; rc=$?
    echo "[mt] turn $turn_no rc=$rc"
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        echo "[mt] turn $turn_no exceeded PER_TURN_TIMEOUT=${PER_TURN_TIMEOUT}s (rc=$rc) -> stopping early. NOTE: a wall-kill is NOT loop evidence on its own (a slow dense model does this while working); only DEGENERATE/TOOL_LOOP in the compacted verdict are."; break
    fi
done
end=$(date +%s)
reap
pkill -f "wire_proxy.py --listen 127.0.0.1:${PROXY_PORT} " 2>/dev/null || true

# rc passed to compactor: 137 if any turn timed out (=> TIMEOUT verdict unless a turn degenerated)
crc=0; [ $ANY_TIMEOUT -eq 1 ] && crc=137
"$PY" "$ROOT/compact_session.py" --session "$SDIR" \
    --model-label "$MODEL_LABEL" --model-port "$MODEL_PORT" \
    --task-id "$TASK_ID" --rc "$crc" --wall "$((end-start))" --timeout "$PER_TURN_TIMEOUT" \
    --task-prompt "[multi-turn adversarial] $INIT"

echo "[mt] DONE $SID turns=$turn_no wall=$((end-start))s any_timeout=$ANY_TIMEOUT -> $SDIR/summary.md"
