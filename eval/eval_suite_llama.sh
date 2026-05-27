#!/usr/bin/env bash
# eval_suite_llama.sh — thin llama.cpp-backend driver over omk_eval.
#
# 9-bench canonical cohort with llama-server + Q6_K GGUF. Companion to
# eval_suite_vllm.sh (which uses vLLM + NVFP4A16). Per the 2026-05-22
# blacklist (Gemma 4 MoE + vLLM rumination cliff, see project_t91_h3_*),
# the llama.cpp + Q6_K path is the canonical validation backend for the
# Gemma 4 MoE cohort until a stable vLLM ships.
#
# Per protocol (`feedback_omk_eval_is_canonical`): we do NOT call lm_eval
# or any bench runner directly here — only omk_eval. New benches are added
# as templates under /shared/dev/omnimergekit/eval/templates/, not here.
#
# omk_eval handles llama-server lifecycle: starts/kills per template.
# Adds ~10s/template (Q6_K load) — acceptable; bench durations dwarf it.
#
# Usage:
#   scripts/eval_suite_llama.sh --variant <name> --gguf <path> [--limit N] [--only csv] [--skip csv] [--port N]
#
#   --variant   Short label used in results dir + served-name (e.g. v6coder, v5coder).
#   --gguf      Absolute path to the Q6_K (or other) GGUF.
#   --limit N   --limit per template (smoke). 0 (default) = full.
#   --only csv  Run only these templates.
#   --skip csv  Skip these templates.
#   --port      llama-server port (default 8099). Pick non-default if 8099 is busy.
#
# Templates (9 benches, in-order):
#   gpqa_diamond_full · gsm8k_100 · math500_100 · aime_30
#   arc_challenge_full · ifeval_100 · humaneval_full · humanevalplus_full · lcb_medium_55_v4
#
set -uo pipefail

# ── args ────────────────────────────────────────────────────────────────
VARIANT=""
GGUF=""
LIMIT=0
ONLY=""
SKIP=""
PORT=8099
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2;;
        --gguf)    GGUF="$2"; shift 2;;
        --limit)   LIMIT="$2"; shift 2;;
        --only)    ONLY="$2"; shift 2;;
        --skip)    SKIP="$2"; shift 2;;
        --port)    PORT="$2"; shift 2;;
        -h|--help) sed -n '2,32p' "$0"; exit 0;;
        *) echo "unknown arg: $1"; exit 2;;
    esac
done
[[ -n "$VARIANT" ]] || { echo "ERR: --variant required"; exit 2; }
[[ -n "$GGUF"    ]] || { echo "ERR: --gguf required"; exit 2; }
[[ -f "$GGUF"    ]] || { echo "ERR: $GGUF not found"; exit 2; }

# ── paths / env ─────────────────────────────────────────────────────────
WS=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
OMK=/shared/dev/omnimergekit
OMK_PY=/root/anaconda3/envs/omnimergekit/bin/python
# omk_eval respects LM_EVAL_BIN — the omnimergekit env ships lm-eval but its bin
# isn't on root's default PATH, so spell it out absolutely.
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
TOKENIZER=$WS/google/gemma-4-26B-A4B-it
SERVED_NAME="${VARIANT}_q6k"
TS=$(date +%Y%m%d_%H%M%S)
LOGS=$WS/logs
RESULTS_DIR=$WS/eval_results_llama_suite/$VARIANT
SUITE_LOG=$LOGS/eval_suite_llama_${VARIANT}_${TS}.log
SUMMARY=$RESULTS_DIR/SUMMARY.md
mkdir -p "$RESULTS_DIR" "$LOGS"

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$SUITE_LOG"; }

# ── template order ──────────────────────────────────────────────────────
# Pruned variants (v3/v4/v5/v5coder/v6coder) use lcb_medium_55_v4 (parser +
# thinking budget recipe) to suppress rumination on hard LCB problems.
# 128e is dense and uses lcb_medium_55. For llama.cpp the parser flags are
# emitted by omk_eval/llama_bench_defaults; we just pick the right template.
case "$VARIANT" in
    v3|v4|v5|v5coder|v6coder) LCB_TPL="lcb_medium_55_v4" ;;
    *)                         LCB_TPL="lcb_medium_55" ;;
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

log "===== eval_suite_llama.sh ====="
log "  variant:     $VARIANT  (served-name=$SERVED_NAME)"
log "  gguf:        $GGUF ($(stat -c %s "$GGUF" | numfmt --to=iec))"
log "  port:        $PORT"
log "  limit:       $LIMIT (0=full)"
log "  templates:   ${selected[*]}"
log "  results:     $RESULTS_DIR"
log "  log:         $SUITE_LOG"

# ── signal file (graceful chain control) ────────────────────────────────
SIGNAL_DIR=$WS/signals
SIGNAL_FILE=$SIGNAL_DIR/chain.signal
mkdir -p "$SIGNAL_DIR"

check_signal() {
    [[ -f "$SIGNAL_FILE" ]] || return 0
    local action
    action=$(head -1 "$SIGNAL_FILE" 2>/dev/null | tr -d '[:space:]')
    [[ -z "$action" ]] && return 0
    local consumed="$SIGNAL_DIR/chain.signal.consumed.$(date +%Y%m%d_%H%M%S).$action"
    mv "$SIGNAL_FILE" "$consumed" 2>/dev/null || true
    case "$action" in
        stop|stop-now)
            log "[signal] $action — exiting chain"
            exit 0
            ;;
        skip)
            log "[signal] skip — skipping template"
            return 1
            ;;
        pause)
            log "[signal] pause — touch $SIGNAL_FILE with 'resume' or 'stop' to continue"
            while [[ ! -f "$SIGNAL_FILE" ]]; do sleep 5; done
            check_signal
            ;;
        resume) ;;
        *) log "[signal] unknown action '$action' — ignored" ;;
    esac
    return 0
}

# ── trap: kill any orphan llama-server on this port at exit ─────────────
trap '
    log "[trap] cleaning up llama-server on port '$PORT'"
    pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true
' EXIT INT TERM

# ── main loop ───────────────────────────────────────────────────────────
declare -A SCORES
declare -A DURS
START_TS=$(date +%s)
log "===== suite start ====="
for t in "${selected[@]}"; do
    log "----- $t -----"
    check_signal || continue
    out="$RESULTS_DIR/$t"
    mkdir -p "$out"
    # llama-server lifecycle is handled inside omk_eval (--backend llama).
    # Each template re-spawns the server (~10s overhead per bench).
    cmd=(
        "$OMK_PY" "$OMK/eval/omk_eval.py"
        --backend llama
        --template "$t"
        --quant q6_k
        --model "$GGUF"
        --tokenizer "$TOKENIZER"
        --served-name "$SERVED_NAME"
        --port "$PORT"
        --results-dir "$RESULTS_DIR"
    )
    [[ "$LIMIT" -gt 0 ]] && cmd+=(--limit "$LIMIT")
    log "[$t] cmd: ${cmd[*]}"
    bench_log="$LOGS/eval_suite_llama_${VARIANT}_${t}_${TS}.log"
    bstart=$(date +%s)
    # ── per-template START marker (suite log + bench log) ───────────────────
    log "[$t] >>> TEMPLATE START  ts=$(date -Iseconds)  epoch=${bstart}"
    echo "[$(date -Iseconds)] >>> OMK_TEMPLATE_START template=$t variant=$VARIANT port=$PORT epoch=${bstart}" > "$bench_log"
    # Timestamp EVERY raw line from omk_eval/lm-eval/llama-server via the
    # ts_prefix.py helper (python is everywhere; awk strftime is a gawk-only
    # extension older mawk lacks). pipefail + PIPESTATUS[0] keeps the real
    # omk_eval exit code through the filter. Protocol: no untimestamped log line.
    set -o pipefail
    ( "${cmd[@]}" ) 2>&1 \
        | python3 -u "$OMK/eval/ts_prefix.py" \
        >> "$bench_log"
    rc=${PIPESTATUS[0]}
    set +o pipefail
    if [[ $rc -eq 7 ]]; then
        log "[$t] FATAL: omk_eval hf-token pre-flight (exit 7) — '$t' uses a GATED dataset and no HF_TOKEN is set."
        log "      export HF_TOKEN in the suite env (or: hf auth login), then re-run. ABORTING suite."
        exit 7
    fi
    bend=$(date +%s); bdur=$((bend - bstart))
    # ── per-template FINISH marker (suite log + bench log) ──────────────────
    echo "[$(date -Iseconds)] <<< OMK_TEMPLATE_FINISH template=$t rc=$rc dur_s=${bdur}" >> "$bench_log"
    if [[ $rc -eq 0 ]]; then
        log "[$t] <<< TEMPLATE FINISH ts=$(date -Iseconds)  dur_s=${bdur}  rc=0"
    else
        log "[$t] <<< TEMPLATE FINISH ts=$(date -Iseconds)  dur_s=${bdur}  rc=$rc — see $bench_log"
    fi
    # ── Score extraction — canonical summary.json FIRST ─────────────────────
    # omk_eval.py writes summary.json at <results>/<template>/<served>/summary.json
    # with the already-correct headline `score` (it picks flexible-extract /
    # math_verify / pass@1,extract_chat / pass_at_1 over strict-match/none, and
    # covers both the lm-eval and the custom LCB runner). That is the single
    # source of truth — read it, do NOT re-derive from raw results_*.json.
    # Origin: 2026-05-23 — re-parsing raw json picked strict-match (GPQA 1.52%)
    # / exact_match,none (math500 41%) when summary.json already had the real
    # 72.73% / 94%. SUMMARY.md must never disagree with summary.json again.
    summ="$out/$SERVED_NAME/summary.json"
    score=""
    if [[ -f "$summ" ]]; then
        score=$(python3 -c "
import json,sys
d=json.load(open('$summ'))
s=d.get('score')
if s is None: print('NO_SCORE'); sys.exit()
m=d.get('metric'); f=d.get('filter')
tag=(f'{m},{f}' if m and f else (m or 'score'))
print(f'{s*100:.2f}%  ({tag})' if isinstance(s,(int,float)) and s<=1.0 else f'{s}  ({tag})')
" 2>/dev/null || echo "PARSE_ERR")
    fi
    # Fallback only if summary.json is missing/unreadable: LCB json, then raw
    # lm-eval results (note the DOUBLED served-name dir lm-eval makes from
    # --output_path). Fallback metric order keeps flexible-extract/math_verify
    # ahead of strict-match/none so the bug can't recur even here.
    if [[ -z "$score" || "$score" == "NO_SCORE" || "$score" == "PARSE_ERR" ]]; then
        lcbj="$out/$SERVED_NAME/lcb_result.json"
        rj=$(ls -t "$out/$SERVED_NAME/lm_eval_out/$SERVED_NAME/results_"*.json \
                   "$out/$SERVED_NAME/lm_eval_out/results_"*.json 2>/dev/null | head -1)
        if [[ -f "$lcbj" ]]; then
            score=$(python3 -c "
import json;d=json.load(open('$lcbj'))
v=d.get('pass_at_1') or d.get('all',{}).get('pass_at_1')
print(f'{v*100:.2f}%  (pass_at_1)' if v is not None else 'NO_SCORE')" 2>/dev/null || echo "PARSE_ERR")
        elif [[ -n "$rj" && -f "$rj" ]]; then
            score=$(python3 -c "
import json,sys
d=json.load(open('$rj')); r=d.get('results',{})
task=next(iter(r),None)
if task is None: print('NO_TASK'); sys.exit()
m=r[task]
for pref in ('exact_match,flexible-extract','math_verify,none','pass@1,extract_chat','pass@1,create_test','prompt_level_strict_acc,none','acc_norm,none','acc,none','exact_match,strict-match','exact_match,none'):
    if pref in m and isinstance(m[pref],(int,float)): print(f'{m[pref]*100:.2f}%  ({pref})'); sys.exit()
for k,v in m.items():
    if isinstance(v,(int,float)) and 'stderr' not in k: print(f'{v*100:.2f}%  ({k})'); sys.exit()
print('NO_METRIC')
" 2>/dev/null || echo "PARSE_ERR")
        else
            score="NO_RESULT"
        fi
    fi
    SCORES["$t"]="$score"
    DURS["$t"]="$bdur"
    log "[$t] score: $score"
    # Persist this template's wall-clock duration into summary.json so future
    # dual-GPU splits balance on real per-bench runtime. omk_eval writes its own
    # (more precise) duration_s; only fill in if absent (e.g. LCB custom runner).
    if [[ -f "$summ" ]]; then
        BDUR="$bdur" python3 -c "
import json,os
p='$summ'; d=json.load(open(p))
if 'duration_s' not in d or not d.get('duration_s'):
    d['duration_s']=int(os.environ['BDUR']); d['duration_s_source']='suite'
    json.dump(d,open(p,'w'),indent=1)
" 2>/dev/null || true
    fi
    # Inter-bench cooldown: kill any straggling llama-server before next template.
    pkill -KILL -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 2
done

# ── summary ─────────────────────────────────────────────────────────────
END_TS=$(date +%s)
DUR=$((END_TS - START_TS))
log "===== suite done — ${DUR}s ====="
{
    echo "# Suite summary — $VARIANT — $TS"
    echo ""
    echo "GGUF: \`$GGUF\`"
    echo "Served-name: \`$SERVED_NAME\`"
    echo "Backend: llama.cpp Q6_K"
    echo "Duration: ${DUR}s"
    echo ""
    echo "| Template | Score | Duration (s) |"
    echo "|----------|-------|--------------|"
    for t in "${selected[@]}"; do
        echo "| $t | ${SCORES[$t]:-MISSING} | ${DURS[$t]:-?} |"
    done
} | tee "$SUMMARY" | sed 's/^/  /' | tee -a "$SUITE_LOG"

log "SUMMARY written to $SUMMARY"
