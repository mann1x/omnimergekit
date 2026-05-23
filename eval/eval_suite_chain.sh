#!/usr/bin/env bash
# eval_suite_chain.sh — run the full 9-bench vLLM suite on 128e then 98e v4,
# strictly sequentially (one vLLM at a time on the single 3090). Each variant
# kills its own vLLM before the next variant's chain starts.
#
# Smoke caches (only 5 prompts per template) survive — lm-eval reuses them
# transparently and runs the remaining prompts fresh.
#
# Signal-file integration (signals/README.md):
#   The inner eval_suite_vllm.sh consumes chain.signal mid-eval and acts on
#   pause/resume/stop/stop-now/skip/rewind. When the inner suite exits, THIS
#   outer loop ALSO checks chain.signal between variants and honors stop /
#   stop-now / skip — all three mean "do not start the next variant".
#   Defense-in-depth: we also scan recent (last 60s) consumed-marker files
#   for stop/stop-now/skip so that a signal already consumed by the inner
#   suite still halts the outer loop.
#
# Logs to logs/eval_suite_chain_<TS>.log; per-variant suites log to their own
# files. Run detached via setsid; the parent terminal can exit.
set -uo pipefail
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
cd "$WS"
TS=$(date +%Y%m%d_%H%M%S)
CHAIN_LOG="$WS/logs/eval_suite_chain_${TS}.log"
SIGNAL_FILE="$WS/signals/chain.signal"
SIGNAL_DIR="$WS/signals"

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$CHAIN_LOG"; }

# Returns the action keyword on stdout (empty string if no actionable signal).
# Checks both the live chain.signal file (consuming it) AND any recently
# (60s) consumed marker files. Honored actions: stop, stop-now, skip.
check_chain_signal(){
    local action=""
    if [[ -f "$SIGNAL_FILE" ]]; then
        action=$(head -n1 "$SIGNAL_FILE" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$action" ]]; then
            local ts; ts=$(date +%s)
            mv "$SIGNAL_FILE" "${SIGNAL_FILE}.consumed.${ts}.${action}" 2>/dev/null || true
        fi
    fi
    if [[ -z "$action" ]]; then
        # Scan consumed markers from the last 60s for stop/stop-now/skip
        action=$(find "$SIGNAL_DIR" -maxdepth 1 -name 'chain.signal.consumed.*' \
                 -newermt "-60 seconds" 2>/dev/null \
                 | grep -oE '\.(stop-now|stop|skip)$' \
                 | head -n1 | sed -E 's/^\.//' || true)
    fi
    case "$action" in
        stop|stop-now|skip) echo "$action" ;;
        "") echo "" ;;
        *)
            # Redirect log to stderr — we must keep stdout clean so the
            # caller's $(check_chain_signal) only captures the action.
            log "[chain-signal] ignored at variant boundary: $action (mid-eval only)" >&2
            echo "" ;;
    esac
}

log "===== eval_suite_chain.sh ====="
log "  TS=$TS"
log "  variants: 128e v4 (sequential)"
log "  chain_log=$CHAIN_LOG"
log "  signal:    $SIGNAL_FILE (honors stop/stop-now/skip between variants)"

for V in 128e v4; do
    # Pre-variant check — catches signals posted while a previous variant
    # finished but before this one started.
    act=$(check_chain_signal)
    if [[ -n "$act" ]]; then
        log "[chain-signal] '$act' received before variant=$V — skipping all remaining variants"
        break
    fi

    log "--- variant=$V START ---"
    T0=$(date +%s)
    bash scripts/eval_suite_vllm.sh --variant "$V" \
        >>"$CHAIN_LOG" 2>&1
    rc=$?
    wall=$(($(date +%s)-T0))
    log "--- variant=$V END rc=$rc wall=${wall}s ---"
    if [[ $rc -ne 0 ]]; then
        log "variant $V exited non-zero — continuing to next variant anyway"
    fi

    # Post-variant check — most common path. Inner suite may have already
    # consumed `stop`/`stop-now`/`skip` mid-eval; we still want to honor
    # the user's intent at the outer loop.
    act=$(check_chain_signal)
    if [[ -n "$act" ]]; then
        log "[chain-signal] '$act' detected after variant=$V — skipping all remaining variants"
        break
    fi
done

log "===== eval_suite_chain.sh — ALL VARIANTS DONE ====="
for V in 128e v4; do
    SUM="$WS/eval_results_vllm_suite/$V/SUMMARY.md"
    if [[ -f "$SUM" ]]; then
        log "==== $V SUMMARY ===="
        cat "$SUM" | tee -a "$CHAIN_LOG"
    fi
done
