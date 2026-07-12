#!/usr/bin/env bash
# eval_suite_vllm.sh — thin driver over omk_eval.
#
# Launches vLLM once per variant, then loops over templates calling
# `omk_eval --no-server` for each. All bench logic, scoring, validation,
# token stats, and SQLite cache live in omk_eval + its YAML templates.
# This script just enumerates which templates to run and in what order.
#
# Per protocol (`feedback_omk_eval_is_canonical`): we do NOT call lm_eval
# or any bench runner directly here — only omk_eval. New benches are added
# as templates under /shared/dev/omnimergekit/eval/templates/, not here.
#
# Usage:
#   scripts/eval_suite_vllm.sh --variant {v3|v4|128e} [--smoke] [--only csv] [--skip csv]
#
#   --smoke   Reduce every template to its 5-row smoke subset via TEMPLATE_LIMIT=5
#             (omk_eval honors it). For pre-flight GO/NO-GO verification.
#   --only    CSV of template names; runs only those.
#   --skip    CSV of template names; skips those.
#
# Templates ordered for this suite (10 benches; FP-grouping reorders at runtime):
#   gpqa_diamond_full · gsm8k_100 · math500_100 · aime_30
#   arc_challenge_full · ifeval_100 · humaneval_full · humanevalplus_full · lcb_medium_55
#   lcb_v6_77q  (all-HARD 77 @ 2024+, ALWAYS 32k max gen; vLLM sizes max-model-len per-request)
#
# 2026-05-12: ifeval_full (541q, ~8.5h) swapped for ifeval_100 (stride-5 → 100q,
# ~95min). LCB-55 added back as a fresh re-run.
#
set -uo pipefail

# ── args ────────────────────────────────────────────────────────────────
VARIANT=""
SMOKE=0
ONLY=""
SKIP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2;;
        --smoke)   SMOKE=1; shift;;
        --only)    ONLY="$2"; shift 2;;
        --skip)    SKIP="$2"; shift 2;;
        -h|--help) sed -n '2,28p' "$0"; exit 0;;
        *) echo "unknown arg: $1"; exit 2;;
    esac
done
[[ "$VARIANT" =~ ^(v3|v4|v5|v5coder|v6coder|128e)$ ]] || { echo "ERR: --variant must be v3, v4, v5, v5coder, v6coder, or 128e"; exit 2; }

# ── paths ───────────────────────────────────────────────────────────────
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
OMK=/shared/dev/omnimergekit
PORT=8195
VLLM_PY=/root/anaconda3/envs/vllm/bin/python
OMK_PY=/root/anaconda3/envs/omnimergekit/bin/python
# omk_eval respects LM_EVAL_BIN — the omnimergekit env ships lm-eval but its bin
# isn't on root's default PATH, so spell it out absolutely.
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
TOKENIZER=$WS/google/gemma-4-26B-A4B-it
TS=$(date +%Y%m%d_%H%M%S)
LOGS=$WS/logs

case "$VARIANT" in
    v3)   MODEL_DIR=$WS/google/Gemma-4-A4B-98e-v3-NVFP4A16;  SERVED_NAME=98e_v3_nvfp4a16  ;;
    v4)   MODEL_DIR=$WS/google/Gemma-4-A4B-98e-v4-NVFP4A16;  SERVED_NAME=98e_v4_nvfp4a16  ;;
    v5)   MODEL_DIR=$WS/google/gemma-4-A4B-98e-v5-NVFP4A16;  SERVED_NAME=98e_v5_nvfp4a16  ;;
    v5coder) MODEL_DIR=$WS/google/gemma-4-A4B-98e-v5-coder-NVFP4A16;  SERVED_NAME=98e_v5_coder_nvfp4a16  ;;
    v6coder) MODEL_DIR=$WS/google/gemma-4-A4B-98e-v6-coder-NVFP4A16;  SERVED_NAME=98e_v6_coder_nvfp4a16  ;;
    128e) MODEL_DIR=$WS/google/Gemma-4-26B-A4B-it-NVFP4A16;  SERVED_NAME=128e_nvfp4a16    ;;
esac
[[ -f "$MODEL_DIR/config.json" ]] || { echo "ERR: $MODEL_DIR not built"; exit 2; }

RESULTS_DIR=$WS/eval_results_vllm_suite/$VARIANT
SUITE_LOG=$LOGS/eval_suite_${VARIANT}_${TS}.log
SUMMARY=$RESULTS_DIR/SUMMARY.md
mkdir -p "$RESULTS_DIR" "$LOGS"

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

# ── template order ──────────────────────────────────────────────────────
# Per-variant LCB template swap: pruned variants (v3, v4, and any future
# 98e/109e) need parser=gemma4 + enable_thinking=true + thinking_token_
# budget=12288 to suppress rumination loops on hard LCB problems (e.g.
# lcb/leetcode/3566 — see lcb_medium_55_v4.yaml header for the comparison
# data). 128e keeps the canonical no-parser recipe that scores 50/55. The
# v4 template's sqlite_prefix differs (lcb_med_55_v4) so the two recipes
# never share a cache key.
case "$VARIANT" in
    v3|v4|v5|v5coder|v6coder) LCB_TPL="lcb_medium_55_v4" ;;
    128e)     LCB_TPL="lcb_medium_55" ;;
    *)        LCB_TPL="lcb_medium_55" ;;
esac
TEMPLATES=(
    gpqa_diamond_full
    gsm8k_100
    math500_100
    aime_30
    arc_challenge_full
    ifeval_100
    humaneval_full
    humanevalplus_full
    "$LCB_TPL"
    lcb_v6_77q          # all-HARD 77 (2024+); template pins 32k max gen — discriminating hard-tier LCB
)

# selection filter
selected=()
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra arr <<<"$ONLY"
    for k in "${arr[@]}"; do selected+=("$k"); done
else
    selected=("${TEMPLATES[@]}")
fi
if [[ -n "$SKIP" ]]; then
    IFS=',' read -ra skip_arr <<<"$SKIP"
    filtered=()
    for k in "${selected[@]}"; do
        skip_it=0
        for s in "${skip_arr[@]}"; do [[ "$k" == "$s" ]] && skip_it=1; done
        [[ $skip_it -eq 0 ]] && filtered+=("$k")
    done
    selected=("${filtered[@]}")
fi

log "===== eval_suite_vllm.sh ====="
log "  variant:   $VARIANT  (served-name=$SERVED_NAME)"
log "  model:     $MODEL_DIR"
log "  smoke:     $SMOKE"
log "  templates: ${selected[*]}"
log "  results:   $RESULTS_DIR"

# ── vLLM lifecycle (shared across the whole suite) ──────────────────────
SLOG="$LOGS/vllm_suite_${VARIANT}_${TS}.log"
VLLM_PID=""

vllm_kill() {
    pkill -KILL -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
    # EngineCore is named "VLLM::EngineCore" (uppercase, set by prctl) — case-insensitive grep.
    for pid in $(pgrep -f -i "VLLM::EngineCore|vllm.*EngineCore" 2>/dev/null); do
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 5
}

vllm_start() {
    # Parameterized by the current vLLM-config fingerprint:
    #   $1 = reasoning_parser (may be empty string for "no parser")
    #   $2 = default_chat_template_kwargs JSON (may be empty)
    local parser="${1:-}"
    local ctk_json="${2:-}"
    vllm_kill
    log "[vllm] launching shared server — parser='${parser:-<none>}' ctk='${ctk_json:-<none>}' slog=$SLOG"
    local args=(
        --model "$MODEL_DIR" --served-model-name "$SERVED_NAME"
        --port "$PORT" --gpu-memory-utilization 0.92
        --max-model-len 32768 --max-num-batched-tokens 4096
        --max-num-seqs 4
        --dtype bfloat16 --trust-remote-code
    )
    # NOTE: 0.92 (not 0.90) is the stack@2 floor on 3090 + 128e Gemma 4
    # 26B-A4B NVFP4A16. At 0.90 vLLM v0.21.1 reports "Available KV cache
    # memory: 1.19 GiB" which is below the 1.6 GiB needed for 32k context.
    # 0.92 gives 1.65 GiB — comfortable. Validated by stack@2 canary
    # (2026-05-21 17:19 CEST). Do NOT lower below 0.92 unless max-model-len
    # is also reduced.
    [[ -n "$parser"   ]] && args+=( --reasoning-parser "$parser" )
    [[ -n "$ctk_json" ]] && args+=( --default-chat-template-kwargs "$ctk_json" )
    LD_PRELOAD=/root/anaconda3/envs/vllm/lib/libstdc++.so.6 \
        "$VLLM_PY" -m vllm.entrypoints.openai.api_server "${args[@]}" \
        > "$SLOG" 2>&1 &
    VLLM_PID=$!; disown
    for i in $(seq 1 360); do
        if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            log "[vllm] ready after ${i}x2s (pid=$VLLM_PID)"
            return 0
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            log "[vllm] died — last 50 lines of slog:"
            tail -50 "$SLOG" | sed 's/^/    /' | tee -a "$SUITE_LOG"
            return 1
        fi
        sleep 2
    done
    log "[vllm] startup timeout"
    return 1
}

trap 'log "[trap] cleaning up"; vllm_kill' EXIT INT TERM

# ── Signal-file watcher (graceful chain control) ────────────────────────
# Single file at $SIGNAL_FILE; first line is the action keyword. Once
# consumed, the file is moved to chain.signal.consumed.<ts>.<action> so
# the same signal doesn't re-fire. Actions:
#   pause       — block before next template until 'resume' or 'stop'
#   resume      — release a pause (also a no-op if nothing is paused)
#   stop        — finish current eval gracefully, then exit chain
#   stop-now    — kill current eval immediately, exit chain
#   skip        — kill current eval, mark SKIP, advance to next template
#   rewind      — kill current eval, clear its cache+samples, re-run
#   repeat      — alias for rewind
# Touch: `echo skip > $WS/signals/chain.signal` from anywhere on the host.
SIGNAL_DIR=$WS/signals
SIGNAL_FILE=$SIGNAL_DIR/chain.signal
mkdir -p "$SIGNAL_DIR"

read_signal_action() {
    [[ -f "$SIGNAL_FILE" ]] || { echo ""; return; }
    head -1 "$SIGNAL_FILE" 2>/dev/null | tr -d ' \r\n\t' | tr '[:upper:]' '[:lower:]'
}

consume_signal() {
    local action="$1"
    local ts; ts=$(date +%s)
    mv "$SIGNAL_FILE" "$SIGNAL_FILE.consumed.${ts}.${action}" 2>/dev/null || true
    log "[signal] consumed: $action"
}

check_pause_at_boundary() {
    # Called BETWEEN templates. Blocks while signal=pause; honors stop.
    while [[ -f "$SIGNAL_FILE" ]]; do
        local act; act=$(read_signal_action)
        case "$act" in
            pause)
                log "[signal] paused — touch $SIGNAL_FILE with 'resume' / 'stop' / 'skip' to continue"
                sleep 10
                ;;
            resume)
                consume_signal resume
                return 0
                ;;
            stop|stop-now)
                consume_signal "$act"
                log "[signal] stop while paused — exiting chain"
                vllm_kill
                exit 0
                ;;
            *)
                # other actions (skip/rewind) handled in the inner loop
                return 0
                ;;
        esac
    done
}

# Background watcher: polls $SIGNAL_FILE while the omk_eval pid is alive.
# On skip/rewind/repeat/stop/stop-now it writes the action to $STATUS_FILE,
# consumes the signal, and SIGTERMs the omk_eval process tree.
watch_signal_during_eval() {
    local pid="$1" status_file="$2"
    while kill -0 "$pid" 2>/dev/null; do
        if [[ -f "$SIGNAL_FILE" ]]; then
            local act; act=$(read_signal_action)
            case "$act" in
                skip|rewind|repeat|stop|stop-now)
                    echo "$act" > "$status_file"
                    consume_signal "$act"
                    log "[signal] $act received mid-eval — terminating omk_eval pid=$pid"
                    pkill -TERM -P "$pid" 2>/dev/null
                    kill -TERM "$pid" 2>/dev/null
                    sleep 3
                    pkill -KILL -P "$pid" 2>/dev/null
                    kill -KILL "$pid" 2>/dev/null
                    return
                    ;;
                pause)
                    # Mid-eval pause is treated as a stop signal — we can't
                    # truly suspend lm-eval mid-request. Use rewind+pause
                    # by issuing 'skip' first if you want pause-then-resume
                    # behavior for the same template.
                    log "[signal] pause received mid-eval — treating as 'skip then pause' (touch signal again to resume)"
                    echo "skip-then-pause" > "$status_file"
                    consume_signal pause
                    pkill -TERM -P "$pid" 2>/dev/null
                    kill -TERM "$pid" 2>/dev/null
                    return
                    ;;
            esac
        fi
        sleep 5
    done
}

# ── compute vLLM-config fingerprint per template + group-sort ───────────
# Each template's `backend_args.vllm_*` keys define server-startup flags
# (parser, chat-template kwargs). Templates with the same fingerprint can
# share one vLLM boot; the chain only reloads when the fingerprint changes.
# We stable-sort selected[] by fingerprint to maximize clustering, so
# at most one reload happens per distinct fingerprint.
FP_FILE=$(mktemp)
"$OMK_PY" - "$OMK/eval/templates" "${selected[@]}" <<'PY' > "$FP_FILE"
import json, sys, yaml, hashlib
from pathlib import Path
tpl_dir = Path(sys.argv[1])
for name in sys.argv[2:]:
    cfg = yaml.safe_load((tpl_dir / f"{name}.yaml").read_text())
    ba = cfg.get("backend_args", {}) or {}
    parser = ba.get("vllm_reasoning_parser", "") or ""
    ctk = ba.get("vllm_default_chat_template_kwargs", "") or ""
    if isinstance(ctk, dict):
        ctk = json.dumps(ctk, sort_keys=True, separators=(",", ":"))
    fp = hashlib.sha1(f"{parser}|{ctk}".encode()).hexdigest()[:10]
    # tab-separated: fp \t name \t parser \t ctk_json
    print(f"{fp}\t{name}\t{parser}\t{ctk}")
PY

# Stable-sort by fingerprint (group same-config templates), preserving
# input order within groups.
mapfile -t fp_rows < <(awk 'BEGIN{FS=OFS="\t"} {print NR, $0}' "$FP_FILE" | sort -t$'\t' -k2,2 -k1,1n | cut -f2-)

# Re-emit `selected` in grouped order and remember each template's flags.
declare -A FP_PARSER FP_CTK FP_OF
selected=()
for row in "${fp_rows[@]}"; do
    fp=$(  echo "$row" | cut -f1)
    name=$(echo "$row" | cut -f2)
    parser=$(echo "$row" | cut -f3)
    ctk=$(echo "$row" | cut -f4)
    selected+=("$name")
    FP_OF[$name]=$fp
    FP_PARSER[$fp]=$parser
    FP_CTK[$fp]=$ctk
done
log "  fp-grouped order: ${selected[*]}"
log "  distinct fingerprints: $(printf '%s\n' "${selected[@]}" | while read n; do echo "${FP_OF[$n]}"; done | sort -u | wc -l)"

# ── fire ────────────────────────────────────────────────────────────────
CURRENT_FP=""
boot_for_fp() {
    local fp=$1
    if [[ "$fp" == "$CURRENT_FP" ]]; then
        return 0  # already up with the right config
    fi
    log "[fp] config change ${CURRENT_FP:-<none>} → ${fp} — restarting vLLM"
    vllm_start "${FP_PARSER[$fp]}" "${FP_CTK[$fp]}" || { log "abort: vllm failed to start for fp=$fp"; exit 1; }
    CURRENT_FP=$fp
}
# Boot once for the first template's fingerprint.
boot_for_fp "${FP_OF[${selected[0]}]}"

# warmup
curl -sS "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"temperature\":0}" \
    >/dev/null 2>&1 || true

# fresh summary
{
    echo "# vLLM suite — $VARIANT ($SERVED_NAME)"
    echo
    echo "Started: $(date -Iseconds) · smoke=$SMOKE · LCB-55 reused from prior run."
    echo
    echo "| template | rc | score | n_samples | empty | p10 chars | length-cap% | warnings |"
    echo "|---|---:|---:|---:|---:|---:|---:|---|"
} > "$SUMMARY"

# smoke mode → pass --limit 5 through to omk_eval (it now supports the flag).
EXTRA_ARGS=()
if [[ $SMOKE -eq 1 ]]; then
    EXTRA_ARGS+=(--limit 5)
fi

OVERALL_RC=0
for tpl in "${selected[@]}"; do
    # Signal-file boundary check: honor pause/stop/skip BEFORE we boot vLLM
    # or invoke omk_eval. Blocks for pause; consumes resume; honors stop/skip.
    check_pause_at_boundary
    if [[ -f "$SIGNAL_FILE" ]]; then
        boundary_act=$(read_signal_action)
        case "$boundary_act" in
            stop|stop-now)
                consume_signal "$boundary_act"
                log "[signal] $boundary_act at boundary — exiting chain"
                break
                ;;
            skip)
                consume_signal skip
                log "[signal] skip at boundary — skipping $tpl"
                echo "| $tpl | - | - | - | - | - | - | skipped via signal |" >> "$SUMMARY"
                continue
                ;;
        esac
    fi

    # Reload vLLM only if this template's fingerprint differs from current.
    boot_for_fp "${FP_OF[$tpl]}"

    # Inner loop supports rewind/repeat (re-run same template after cache clear).
    rewind_count=0
    MAX_REWINDS=3
    while true; do
        log "=== template=$tpl (fp=${FP_OF[$tpl]})${rewind_count:+ rewind=$rewind_count} ==="
        T0=$(date +%s)
        blog=$LOGS/eval_${VARIANT}_${tpl}_${TS}${rewind_count:+_r$rewind_count}.log
        STATUS_FILE=$(mktemp)
        : > "$STATUS_FILE"

        # NB: --no-server because we manage the lifecycle here for the whole
        # suite. Errexit is OFF (`set -uo pipefail`, NO -e) — we read rc
        # explicitly. omk_eval runs in background so the signal watcher can
        # TERM it on skip/rewind/stop.
        "$OMK_PY" "$OMK/eval/omk_eval.py" \
            --model "$MODEL_DIR" \
            --template "$tpl" \
            --backend vllm \
            --port "$PORT" \
            --results-dir "$RESULTS_DIR" \
            --tokenizer "$TOKENIZER" \
            --served-name "$SERVED_NAME" \
            --no-server \
            "${EXTRA_ARGS[@]}" \
            > "$blog" 2>&1 &
        OMK_PID=$!

        watch_signal_during_eval "$OMK_PID" "$STATUS_FILE" &
        WATCHER_PID=$!

        wait "$OMK_PID" 2>/dev/null
        rc=$?
        kill "$WATCHER_PID" 2>/dev/null
        wait "$WATCHER_PID" 2>/dev/null

        wall=$(($(date +%s)-T0))
        signal_action=""
        [[ -s "$STATUS_FILE" ]] && signal_action=$(cat "$STATUS_FILE")
        rm -f "$STATUS_FILE"

        # Dispatch on signal action FIRST (overrides rc since we killed it).
        case "$signal_action" in
            stop|stop-now)
                log "[$tpl] STOPPED via signal at ${wall}s (rc=$rc)"
                echo "| $tpl | killed | - | - | - | - | - | stopped via signal |" >> "$SUMMARY"
                vllm_kill
                trap - EXIT INT TERM
                log "=== suite stopped via signal — partial $SUMMARY ==="
                exit 0
                ;;
            skip)
                log "[$tpl] SKIPPED via signal at ${wall}s (rc=$rc)"
                echo "| $tpl | killed | - | - | - | - | - | skipped via signal |" >> "$SUMMARY"
                status="SKIP"
                break  # exit inner while, advance to next template
                ;;
            rewind|repeat)
                rewind_count=$((rewind_count+1))
                if (( rewind_count > MAX_REWINDS )); then
                    log "[$tpl] rewind limit ($MAX_REWINDS) reached — stopping rewind, accepting last rc=$rc"
                    signal_action=""
                else
                    log "[$tpl] REWIND via signal at ${wall}s — clearing cache+samples and re-running"
                    # Clear sqlite cache + samples for this template only
                    find "$RESULTS_DIR" -path "*${tpl}*sqlite_cache*" -delete 2>/dev/null || true
                    find "$RESULTS_DIR" -path "*${tpl}*samples_*.jsonl" -delete 2>/dev/null || true
                    find "$RESULTS_DIR" -path "*${tpl}*results_*.json" -delete 2>/dev/null || true
                    find "$RESULTS_DIR" -path "*${tpl}*lcb_result*" -delete 2>/dev/null || true
                    # Check for pause-after-rewind by re-reading signal at boundary
                    check_pause_at_boundary
                    continue  # re-run same template
                fi
                ;;
            skip-then-pause)
                log "[$tpl] paused after kill — block until resume/stop"
                # Synthesize a pause file so check_pause_at_boundary blocks
                echo "pause" > "$SIGNAL_FILE"
                check_pause_at_boundary
                status="SKIP"
                break
                ;;
        esac

        # No signal-driven action — normal completion path.
        case "$rc" in
            0)  status="OK" ;;
            40) status="WARN" ;;
            50) status="SMOKE_FLOOR_FAIL"; OVERALL_RC=$rc ;;   # score <= floor in smoke
            60) status="LCB_EARLY_ABORT"; OVERALL_RC=$rc ;;     # infra signature (all empty/length)
            *)  status="FAIL"; OVERALL_RC=$rc ;;
        esac
        break  # exit inner while (no rewind)
    done

    log "[$tpl] wall=${wall}s rc=$rc status=$status log=$blog"

    # Pull summary.json for the row (omk_eval writes it next to samples)
    summary_json=$(find "$RESULTS_DIR" -name summary.json -path "*${SERVED_NAME}*" -newer "$SUITE_LOG" 2>/dev/null | head -1)
    if [[ -z "$summary_json" ]]; then
        # fall back: search by sqlite_prefix derived from template name
        summary_json=$(find "$RESULTS_DIR" -name summary.json 2>/dev/null | xargs -I{} grep -l "\"template\": \"$tpl\"" {} 2>/dev/null | head -1)
    fi
    if [[ -n "$summary_json" && -f "$summary_json" ]]; then
        "$OMK_PY" - "$summary_json" "$tpl" "$rc" <<'PY' >> "$SUMMARY"
import json, sys
s = json.load(open(sys.argv[1]))
tpl, rc = sys.argv[2], sys.argv[3]
ts = s.get('token_stats', {}) or {}
cc = ts.get('completion_chars', {}) or {}
warns = s.get('sanity_warnings') or []
n = ts.get('n_samples', '?')
empty = ts.get('empty_completions', '?')
p10 = cc.get('p10', '?')
lcap = ts.get('length_cap_pct', '?')
score = s.get('score')
score_txt = f"{score*100:.2f}%" if isinstance(score, (int, float)) else '-'
warn_txt = ', '.join(warns) if warns else 'none'
print(f"| {tpl} | {rc} | {score_txt} | {n} | {empty} | {p10} | {lcap} | {warn_txt} |")
PY
    else
        echo "| $tpl | $rc | - | ? | ? | ? | ? | no summary.json found |" >> "$SUMMARY"
    fi

    # In smoke mode, halt on FAIL, SMOKE_FLOOR_FAIL, or LCB_EARLY_ABORT.
    # Warnings still continue — they signal but don't stop.
    case "$status" in
        FAIL|SMOKE_FLOOR_FAIL|LCB_EARLY_ABORT)
            if [[ $SMOKE -eq 1 ]]; then
                log "SMOKE $status on [$tpl] — halting chain. Inspect $blog."
                echo >> "$SUMMARY"
                echo "**HALTED at $tpl ($status).**" >> "$SUMMARY"
                break
            fi
            # Non-smoke: surface but continue — the chain can still produce
            # useful results for other benches, and the operator can rewind
            # the broken one via the signal file.
            ;;
    esac
done

vllm_kill
trap - EXIT INT TERM

log "=== suite done — overall_rc=$OVERALL_RC — see $SUMMARY ==="
cat "$SUMMARY" | tee -a "$SUITE_LOG"
exit $OVERALL_RC
