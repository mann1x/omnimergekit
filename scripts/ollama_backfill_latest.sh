#!/usr/bin/env bash
# Backfill the `:latest` tag on an already-published ollama target by pulling
# an existing tier and re-tagging it. Idempotent — skips models that already
# have `:latest` on ollama.com.
#
# Usage:
#   bash ollama_backfill_latest.sh <ollama_target> [latest_tier]
#
# Defaults: latest_tier=Q4_K_M.
# Returns 0 on success or skip, non-zero on hard failure.
#
# Note: this is a ONE-SHOT script for fixing the historical 7 published
# models that were pushed before the chain emitted `:latest`. For future
# builds, the `--latest-tier <T>` / `--no-latest` flags in
# `ollama_push_gemma4.sh`, `ollama_push_generic.sh`, and
# `quantize_gguf.py --ollama` cover this automatically.

set -uo pipefail
OL_TARGET="${1:?ollama target (e.g. mannix/gemma4-98e-v5-coder)}"
LATEST_TIER="${2:-Q4_K_M}"

WORKDIR=/workspace/ollama_backfill
LOG_DIR=/workspace/logs
mkdir -p "$WORKDIR" "$LOG_DIR"
LOG="$LOG_DIR/ollama_backfill_latest_${OL_TARGET//\//_}.log"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== backfill :latest for $OL_TARGET (source=:$LATEST_TIER) ==="

# Pre-flight: ollama daemon up
if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "FATAL: ollama daemon not responding on 127.0.0.1:11434"
    exit 2
fi

# Idempotency: if :latest already exists on ollama.com, skip.
# We use the tags-page scrape (same approach as the push scripts).
EXISTING=$(curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_.-]+" | sed "s|.*:||" | sort -u || true)
if echo "$EXISTING" | grep -q "^latest$"; then
    log "$OL_TARGET: :latest already on ollama.com — skipping"
    exit 0
fi

# Confirm latest-tier exists on ollama.com (or is privately known to exist).
# For private models the scrape returns empty; in that case we trust the caller.
if [[ -n "$EXISTING" ]] && ! echo "$EXISTING" | grep -q "^${LATEST_TIER}$"; then
    log "FATAL: ${OL_TARGET}:${LATEST_TIER} not visible in public tags; available: $(echo $EXISTING | tr '\n' ' ')"
    log "       (model may be private; pass tier explicitly as arg 2)"
    exit 3
fi

# Pull source tier
log "step 1/4: pull ${OL_TARGET}:${LATEST_TIER}"
PULL_OUT=$(mktemp)
if ! ollama pull "${OL_TARGET}:${LATEST_TIER}" >"$PULL_OUT" 2>&1; then
    log "FATAL: pull failed"
    log "$(tail -5 "$PULL_OUT")"
    rm -f "$PULL_OUT"
    exit 4
fi
rm -f "$PULL_OUT"
log "  pull OK"

# Tag as :latest locally
log "step 2/4: ollama cp ${OL_TARGET}:${LATEST_TIER} ${OL_TARGET}:latest"
if ! ollama cp "${OL_TARGET}:${LATEST_TIER}" "${OL_TARGET}:latest" >>"$LOG" 2>&1; then
    log "FATAL: cp failed"
    exit 5
fi
log "  cp OK"

# Push :latest (manifest only — blobs already on ollama.com from initial push)
log "step 3/4: push ${OL_TARGET}:latest"
PUSH_OUT=$(mktemp)
ollama push "${OL_TARGET}:latest" >"$PUSH_OUT" 2>&1
PUSH_RC=$?
cat "$PUSH_OUT" >>"$LOG"
HAS_SUCCESS=$(grep -cE "^success$|^success " "$PUSH_OUT" 2>/dev/null || echo 0)
HAS_URL=$(grep -cE "You can find your model at" "$PUSH_OUT" 2>/dev/null || echo 0)

if [[ $PUSH_RC -ne 0 ]]; then
    log "FATAL: push exited rc=$PUSH_RC"
    log "  last 5 lines: $(tail -5 "$PUSH_OUT" | tr '\n' ' | ')"
    rm -f "$PUSH_OUT"
    exit 6
fi
if [[ "$HAS_SUCCESS" -lt 1 ]] || [[ "$HAS_URL" -lt 1 ]]; then
    log "WARN: push rc=0 but no positive markers — may have failed silently"
    log "  last 5 lines: $(tail -5 "$PUSH_OUT" | tr '\n' ' | ')"
    rm -f "$PUSH_OUT"
    exit 7
fi
rm -f "$PUSH_OUT"
log "  push OK"

# Cleanup: remove local manifests, sweep orphan blobs.
# `ollama rm` removes the manifest file but the content-addressed blobs in
# .ollama/models/blobs/ stay forever unless we sweep — same gotcha as the
# main push scripts.
log "step 4/4: cleanup local manifests + orphan blobs"
ollama rm "${OL_TARGET}:${LATEST_TIER}" >>"$LOG" 2>&1 || true
ollama rm "${OL_TARGET}:latest" >>"$LOG" 2>&1 || true

swept_count=0
for od in /usr/share/ollama/.ollama/models /root/.ollama/models; do
    [ -d "$od/manifests" ] || continue
    [ -d "$od/blobs" ] || continue
    # Build the set of blob digests still referenced by any manifest.
    in_use=$(grep -rhEo 'sha256:[0-9a-f]{64}' "$od/manifests" 2>/dev/null | sort -u | sed 's/:/-/')
    for blob in "$od"/blobs/sha256-*; do
        [ -f "$blob" ] || continue
        bn=$(basename "$blob")
        # Strip -partial-* and trailing-anchor noise; compare base hash.
        bn_base=$(echo "$bn" | cut -d- -f1-2)
        if ! echo "$in_use" | grep -q "^${bn_base}$"; then
            rm -f "$blob" && swept_count=$((swept_count+1))
        fi
    done
done
log "  swept $swept_count orphan blob(s)"

log "$OL_TARGET: DONE — :latest now points to (was :$LATEST_TIER)"
exit 0
