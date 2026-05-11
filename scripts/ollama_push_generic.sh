#!/usr/bin/env bash
# Generic HF GGUF → ollama.com pusher.
# Unlike ollama_push_98e.sh (hardcoded Gemma 4 tools template), this version
# lets ollama auto-detect chat template from the GGUF metadata. Right call for
# Qwen3.5/3.6 omnimerge models, where the GGUF carries Qwen's native chatml
# template and we don't need a custom override.
#
# Usage:
#   bash ollama_push_generic.sh <hf_repo> <ollama_target> [include_pattern]
# e.g.
#   bash ollama_push_generic.sh ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF mannix/omnimerge-v4

set -euo pipefail
HF_REPO="${1:?hf repo (e.g. ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF)}"
OL_TARGET="${2:?ollama target (e.g. mannix/omnimerge-v4)}"
INCLUDE="${3:-}"

WORKDIR=/workspace/ollama_push
LOG_DIR=/workspace/logs
mkdir -p "$WORKDIR" "$LOG_DIR"

LOG="$LOG_DIR/ollama_push_${OL_TARGET//\//_}.log"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Per-tag retry config (see also ollama_push_98e.sh)
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_SLEEP=${RETRY_SLEEP:-5}

# Helper: aggressively flush any cached state for a failing tag so the next
# attempt redownloads from HF instead of resuming a half-broken cache.
flush_tag_cache() {
    local TAG="$1"
    # Drop any half-formed manifest referring to this tag
    ollama rm "hf.co/${HF_REPO}:${TAG}"   >>"$LOG" 2>&1 || true
    ollama rm "${OL_TARGET}:${TAG}"        >>"$LOG" 2>&1 || true
    # Wipe -partial-* residue from prior `ollama create` attempts.
    # No tag-specific way to identify which partial belongs to which create,
    # so we wipe all partials (safe — they only exist while creates are
    # in-flight, and any concurrent create on this daemon is its own problem).
    rm -f /root/.ollama/models/blobs/sha256-*-partial* 2>/dev/null || true
}

# Helper: run the create + push for one tag with retry. Returns 0 on success.
push_one_tag() {
    local TAG="$1"
    local LABEL="${2:-}"   # e.g. "[3/13]" or "[retry]"
    local MF="$WORKDIR/Modelfile.${OL_TARGET//\//_}.${TAG}"
    cat > "$MF" <<EOF
FROM hf.co/${HF_REPO}:${TAG}
EOF

    local attempt
    local create_ok=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        log "$LABEL $TAG: create attempt $attempt/$MAX_ATTEMPTS"
        if ollama create "${OL_TARGET}:${TAG}" -f "$MF" >>"$LOG" 2>&1; then
            create_ok=1
            break
        fi
        log "$LABEL $TAG:   create attempt $attempt failed; flushing cache"
        flush_tag_cache "$TAG"
        if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
            sleep "$RETRY_SLEEP"
        fi
    done
    if [[ $create_ok -eq 0 ]]; then
        log "$LABEL $TAG:   FINAL CREATE FAIL after $MAX_ATTEMPTS attempts"
        return 1
    fi

    log "$LABEL $TAG: pushing ..."
    if ! ollama push "${OL_TARGET}:${TAG}" >>"$LOG" 2>&1; then
        log "$LABEL $TAG:   PUSH FAILED"
        return 2
    fi

    # --- capture blob digests before removing refs (belt + suspenders against leak) ---
    local BLOBS
    BLOBS=$(
        for MF_PATH in \
            "/root/.ollama/models/manifests/registry.ollama.ai/${OL_TARGET}/${TAG}" \
            "/root/.ollama/models/manifests/hf.co/${HF_REPO}/${TAG}"; do
            [ -f "$MF_PATH" ] && grep -oE 'sha256:[0-9a-f]{64}' "$MF_PATH"
        done | sort -u
    )
    # Remove BOTH refs so ollama GCs the blob (hf.co ref keeps it pinned otherwise)
    ollama rm "${OL_TARGET}:${TAG}"      >>"$LOG" 2>&1 || true
    ollama rm "hf.co/${HF_REPO}:${TAG}"  >>"$LOG" 2>&1 || true
    # Force-purge captured blobs + any -partial-* residue
    local B
    for B in $BLOBS; do
        local FN="/root/.ollama/models/blobs/${B/sha256:/sha256-}"
        rm -f "$FN" "$FN-partial"* 2>/dev/null || true
    done

    df -h /workspace | tail -1 | tee -a "$LOG" >/dev/null
    log "$LABEL $TAG: DONE"
    # Brief pause to relieve HF/ollama-daemon contention before next iteration
    sleep "$RETRY_SLEEP"
    return 0
}

log "Discovering tags in $HF_REPO ..."
TAGS=$(curl -fsSL -H "Authorization: Bearer ${HF_TOKEN:?HF_TOKEN env required}" \
    "https://huggingface.co/api/models/$HF_REPO/tree/main" \
    | python3 -c "
import json, sys, re
d = json.load(sys.stdin)
out = []
for x in d:
    p = x.get('path','')
    if x.get('type') != 'file' or not p.endswith('.gguf'):
        continue
    base = p.rsplit('.gguf', 1)[0]
    parts = base.split('-')
    quant_pat = re.compile(r'^(F16|Q\d+(_[KS01ML]+)?(_[KSML01]+)?|IQ\d+(_[A-Z]+)?|CD-Q\d+(_[KS]+)?(_[KSML01]+)?)\$')
    for n in (3, 2, 1):
        if n > len(parts): continue
        candidate = '-'.join(parts[-n:])
        if quant_pat.match(candidate):
            out.append(candidate); break
print('\\n'.join(sorted(set(out))))
")

if [[ -n "$INCLUDE" ]]; then
    TAGS=$(echo "$TAGS" | grep -E "$INCLUDE" || true)
fi
TAGS=$(echo "$TAGS" | grep -vE "^F16$" || true)

log "Tags to push (after filter):"
echo "$TAGS" | sed 's/^/  /' | tee -a "$LOG"
N_TAGS=$(echo "$TAGS" | grep -c . || true)
log "  total: $N_TAGS"

EXISTING=$(curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_-]+" | sed "s|.*:||" | sort -u || true)
if [[ -n "$EXISTING" ]]; then
    log "Already on ollama.com:"
    echo "$EXISTING" | sed 's/^/  /' | tee -a "$LOG"
fi

TODO=$(comm -23 <(echo "$TAGS" | sort -u) <(echo "$EXISTING" | sort -u) || true)
N_TODO=$(echo "$TODO" | grep -c . || true)
log "TODO: $N_TODO tag(s)"

# Main pass — failures go to FAILED_TAGS for the retry pass at the end
FAILED_TAGS=()
i=0
for TAG in $TODO; do
    i=$((i+1))
    if ! push_one_tag "$TAG" "[$i/$N_TODO]"; then
        FAILED_TAGS+=("$TAG")
        # Make sure we don't carry over a half-broken cache to the next tag
        flush_tag_cache "$TAG"
    fi
done

# Retry pass: redownload from HF for each previously-failed tag
if [[ ${#FAILED_TAGS[@]} -gt 0 ]]; then
    log "=== RETRY PASS: ${#FAILED_TAGS[@]} previously-failed tag(s): ${FAILED_TAGS[*]} ==="
    STILL_FAILED=()
    j=0
    for TAG in "${FAILED_TAGS[@]}"; do
        j=$((j+1))
        # Force a fresh download by flushing any cached partials/manifests first
        flush_tag_cache "$TAG"
        if ! push_one_tag "$TAG" "[retry $j/${#FAILED_TAGS[@]}]"; then
            STILL_FAILED+=("$TAG")
            flush_tag_cache "$TAG"
        fi
    done
    if [[ ${#STILL_FAILED[@]} -gt 0 ]]; then
        log "=== AFTER RETRY: still failed: ${STILL_FAILED[*]} ==="
        log "    (re-run this script later — these will be re-attempted as the TODO set is recomputed.)"
    else
        log "=== AFTER RETRY: all previously-failed tags pushed successfully. ==="
    fi
fi

log "All done for $OL_TARGET. Tags published:"
curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_-]+" | sort -u | tee -a "$LOG"
