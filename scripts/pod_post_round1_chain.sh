#!/bin/bash
# pod_post_round1_chain.sh — auto-chain runner for pod 36755693 et al.
#
# CANONICAL LOCATION: omnimergekit/scripts/pod_post_round1_chain.sh
#
# Pipeline:
#   1. Wait for round-1 reeval24k to finish (looks for "ALL REEVAL DONE"
#      in tmux session "reeval", or for both lm-eval clients to be gone).
#   2. Round 2: re-run pod_reeval_failures.sh with sqlite caches intact —
#      fills tasks that died with UnboundLocal (now patched) or missing
#      deps (langdetect installed) or hit transient retry exhaust. Cache
#      makes completed work free.
#   3. 31B build: pod_quant_31b.sh — downloads 31B + 31b-he1, builds both
#      NVFP4A16 quants via omnimergekit/scripts/quantize_any.py, pushes
#      PRIVATE to HF, purges BF16 sources.
#   4. 31B eval: pod_eval_31b.sh — full 9-template chain on both variants
#      in parallel (one per GPU), ~7-10 h wall.
#
# Detach-safe: runs under its own tmux session "chain". Each phase logs to
# /workspace/logs/chain_<phase>_<timestamp>.log. Phase results are written
# to /workspace/chain_status.json so you can check progress without ssh-
# scraping tmux.
#
# Failure semantics: each phase failure is logged but DOES NOT abort the
# chain (we always want the 31B run to start, even if reeval24k round-2
# has hiccups; 31B is a fresh dataset, independent results).
set -uo pipefail   # NOTE: no -e — we want phase failures recorded, not fatal

OMK=/workspace/omnimergekit
LOGS=/workspace/logs
STATUS=/workspace/chain_status.json
mkdir -p "$LOGS"

log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }

# Initialize status file
echo '{"phase": "waiting_round1", "started": "'"$(date -Iseconds)"'", "phases": {}}' > "$STATUS"
jq_update() {
    # key and val arrive already JSON-encoded ('"phase"' '"x"' etc.) so we
    # pass them through env (no embedded quoting hell) and let python
    # json.loads them. The old version embedded $key/$val directly inside
    # a single-quoted Python heredoc → never resolved → silent no-op.
    local key="$1" val="$2"
    STATUS_PATH="$STATUS" JKEY="$key" JVAL="$val" python3 - <<'PY' 2>/dev/null || true
import json, os
p = os.environ["STATUS_PATH"]
k = json.loads(os.environ["JKEY"])
v = json.loads(os.environ["JVAL"])
d = json.load(open(p))
# Top-level keys we keep flat; per-phase RCs go under "phases".
if k in ("phase", "started", "finished"):
    d[k] = v
else:
    d.setdefault("phases", {})[k] = v
json.dump(d, open(p, "w"), indent=2)
PY
}

# ── Phase 0: wait for round-1 to finish ────────────────────────────────────
log "Phase 0: waiting for round-1 reeval24k to complete"
while true; do
    # Done signal: "ALL REEVAL DONE" in tmux buffer, OR no lm-eval children
    # running for 60+ seconds (in case tmux buffer rotated out).
    if tmux capture-pane -t reeval -p 2>/dev/null | grep -q "ALL REEVAL DONE"; then
        log "round-1 signaled DONE in tmux"
        break
    fi
    # NB: `[x]y` regex trick prevents the pgrep search from self-matching the
    # bash wrapper that contains the search string literally (the `bash -lc
    # "..."` cmdline contains the pgrep pattern). Without the brackets the
    # check is permanently false. Hit on 2026-05-14 — chain hung in Phase 0
    # for 2 h after round-1 actually finished.
    if ! pgrep -f "[l]m-eval --model local-chat" >/dev/null 2>&1 && \
       ! pgrep -f "[v]llm.entrypoints" >/dev/null 2>&1; then
        # Belt-and-suspenders: 2 minute confirmation pause to rule out a
        # transient gap between templates
        sleep 120
        if ! pgrep -af "lm-eval --model local-chat" >/dev/null 2>&1 && \
           ! pgrep -af "vllm.entrypoints" >/dev/null 2>&1; then
            log "round-1 quiescent (no vllm/lm-eval processes for 2 min)"
            break
        fi
    fi
    sleep 60
done

# ── Phase 1: round-2 reeval24k (resume from sqlite caches) ─────────────────
log "Phase 1: round-2 reeval24k — re-running pod_reeval_failures.sh with cache preserved"
PHASE_LOG="$LOGS/chain_round2_$(date +%Y%m%d_%H%M%S).log"
{
    echo "=== round-2 start $(date -Iseconds) ==="
    # NOTE: cache dirs under /workspace/eval_results_reeval24k/<variant>/
    # <template>/<variant>/sqlite_cache/ are preserved; lm-eval will skip
    # samples already cached and re-run only missing IDs. Deps (langdetect,
    # immutabledict, nltk) are installed by Phase 0 of pod_setup_eval_envs.sh
    # (or hand-installed at hotfix time). api_models.py UnboundLocal patch is
    # already in place.
    PYTHONDONTWRITEBYTECODE=1 \
    bash /workspace/scripts/pod_reeval_failures.sh
    echo "=== round-2 end $(date -Iseconds) ==="
} >> "$PHASE_LOG" 2>&1
RC_R2=$?
jq_update '"phase"' '"round2_done"'
jq_update '"round2"' "$RC_R2"
log "Phase 1 done: exit=$RC_R2 log=$PHASE_LOG"

# ── Phase 2: 31B quant ─────────────────────────────────────────────────────
log "Phase 2: 31B NVFP4A16 quant (base + he1)"
PHASE_LOG="$LOGS/chain_quant31b_$(date +%Y%m%d_%H%M%S).log"
{
    echo "=== 31B quant start $(date -Iseconds) ==="
    bash /workspace/scripts/pod_quant_31b.sh
    echo "=== 31B quant end $(date -Iseconds) ==="
} >> "$PHASE_LOG" 2>&1
RC_Q=$?
# Sanity gate: pod_quant_31b.sh can exit 0 even when the conda env is missing
# (`conda activate modelopt` prints to stderr but doesn't propagate failure
# without `set -e` in the calling shell). Verify the expected outputs exist
# before claiming Phase 2 success. See memory/feedback_pod_modelopt_env_missing.md.
if [ $RC_Q -eq 0 ]; then
    for D in /workspace/models/Gemma-4-31B-it-NVFP4A16 /workspace/models/gemma-4-31b-he1-it-NVFP4A16; do
        if [ ! -f "$D/hf_quant_config.json" ]; then
            log "Phase 2 SANITY FAIL: $D missing hf_quant_config.json (env or script error swallowed)"
            RC_Q=2
        fi
    done
fi
jq_update '"phase"' '"quant31b_done"'
jq_update '"quant31b"' "$RC_Q"
log "Phase 2 done: exit=$RC_Q log=$PHASE_LOG"

# ── Phase 3: 31B eval (full 9-template chain × 2 variants, parallel GPU) ──
# Only attempt if Phase 2 produced the NVFP4A16 dirs.
if [ $RC_Q -ne 0 ]; then
    log "Phase 3 SKIPPED: Phase 2 failed (rc=$RC_Q), nothing to eval"
    RC_E=255
else
    log "Phase 3: 31B eval (parallel 2-GPU)"
    PHASE_LOG="$LOGS/chain_eval31b_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== 31B eval start $(date -Iseconds) ==="
        PYTHONDONTWRITEBYTECODE=1 \
        bash /workspace/scripts/pod_eval_31b.sh
        echo "=== 31B eval end $(date -Iseconds) ==="
    } >> "$PHASE_LOG" 2>&1
    RC_E=$?
fi
jq_update '"phase"' '"eval31b_done"'
jq_update '"eval31b"' "$RC_E"
log "Phase 3 done: exit=$RC_E log=$PHASE_LOG"

# ── Wrap-up ────────────────────────────────────────────────────────────────
log "ALL CHAIN PHASES COMPLETE: round2=$RC_R2 quant31b=$RC_Q eval31b=$RC_E"
jq_update '"phase"' '"complete"'
jq_update '"finished"' '"\"'"$(date -Iseconds)"'\""'
