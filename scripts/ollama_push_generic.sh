#!/usr/bin/env bash
# Generic HF GGUF → ollama.com pusher.
# Unlike ollama_push_98e.sh (hardcoded Gemma 4 tools template), this version
# lets ollama auto-detect chat template from the GGUF metadata. Right call for
# Qwen3.5/3.6 omnimerge models, where the GGUF carries Qwen's native chatml
# template and we don't need a custom override.
#
# Strategy (revised 2026-05-11): pre-download GGUF via `hf download` to a local
# path, then `ollama create FROM /local/path.gguf`. Bypasses ollama's HF-pull
# machinery (whose per-create deadline is hardcoded short and tends to fail on
# 12-25 GB blobs even when raw HF bandwidth is fine). Trade-off: we use ~25 GB
# scratch space per tag, deleted immediately after the push.
#
# Usage:
#   bash ollama_push_generic.sh <hf_repo> <ollama_target> <hf_token> [include_pattern]
#   # legacy: HF_TOKEN env var also accepted if <hf_token> arg is omitted with "-"
# e.g.
#   bash ollama_push_generic.sh ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF mannix/omnimerge-v4 "$(cat ~/.cache/huggingface/token)"

set -euo pipefail
HF_REPO="${1:?hf repo (e.g. ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF)}"
OL_TARGET="${2:?ollama target (e.g. mannix/omnimerge-v4)}"
HF_TOKEN_ARG="${3:--}"
INCLUDE="${4:-}"

# Token: command-line arg takes precedence over env
if [[ "$HF_TOKEN_ARG" != "-" && -n "$HF_TOKEN_ARG" ]]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi
: "${HF_TOKEN:?HF_TOKEN required as arg-3 or env (read token for HF GGUF repo)}"

WORKDIR=/workspace/ollama_push
LOG_DIR=/workspace/logs
SCRATCH=/workspace/ollama_push/scratch     # holds the one-at-a-time downloaded GGUF
mkdir -p "$WORKDIR" "$LOG_DIR" "$SCRATCH"

LOG="$LOG_DIR/ollama_push_${OL_TARGET//\//_}.log"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_SLEEP=${RETRY_SLEEP:-5}

# Wipe a tag's local artifacts so the next attempt is a clean redownload.
flush_tag_cache() {
    local TAG="$1" GGUF="${2:-}"
    ollama rm "${OL_TARGET}:${TAG}"   >>"$LOG" 2>&1 || true
    # Old strategy carried a hf.co ref; we no longer create one, but rm
    # any leftover from a prior script version just in case.
    ollama rm "hf.co/${HF_REPO}:${TAG}" >>"$LOG" 2>&1 || true
    rm -f /root/.ollama/models/blobs/sha256-*-partial* 2>/dev/null || true
    [ -n "$GGUF" ] && rm -f "$GGUF" 2>/dev/null || true
}

# Download one GGUF from HF to local scratch dir, return path on stdout.
# Returns non-zero on failure. Retries internally with hf download's own
# resume logic.
hf_download_one() {
    local FILENAME="$1"
    local DEST="$SCRATCH/$FILENAME"
    rm -f "$DEST" 2>/dev/null || true
    if HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$HF_REPO" "$FILENAME" \
            --local-dir "$SCRATCH" >>"$LOG" 2>&1; then
        if [ -s "$DEST" ]; then
            echo "$DEST"
            return 0
        fi
    fi
    return 1
}

# Run pre-download + create + push for one tag with retry. Returns 0 on success.
push_one_tag() {
    local TAG="$1" FILENAME="$2"
    local LABEL="${3:-}"
    local MF="$WORKDIR/Modelfile.${OL_TARGET//\//_}.${TAG}"
    local GGUF=""

    local attempt
    local create_ok=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        log "$LABEL $TAG: download + create attempt $attempt/$MAX_ATTEMPTS"

        # 1. Download GGUF from HF
        if ! GGUF=$(hf_download_one "$FILENAME"); then
            log "$LABEL $TAG:   download attempt $attempt failed"
            flush_tag_cache "$TAG" "$SCRATCH/$FILENAME"
            [[ $attempt -lt $MAX_ATTEMPTS ]] && sleep "$RETRY_SLEEP"
            continue
        fi
        log "$LABEL $TAG:   downloaded $(du -h "$GGUF" | cut -f1) to $GGUF"

        # 2. Modelfile referencing the local file
        cat > "$MF" <<EOF
FROM $GGUF
EOF

        # 3. ollama create from local file (no HF pull)
        if ollama create "${OL_TARGET}:${TAG}" -f "$MF" >>"$LOG" 2>&1; then
            create_ok=1
            break
        fi
        log "$LABEL $TAG:   create attempt $attempt failed; flushing cache"
        flush_tag_cache "$TAG" "$GGUF"
        if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
            sleep "$RETRY_SLEEP"
        fi
    done
    if [[ $create_ok -eq 0 ]]; then
        log "$LABEL $TAG:   FINAL FAIL after $MAX_ATTEMPTS attempts"
        rm -f "$SCRATCH/$FILENAME" 2>/dev/null || true
        return 1
    fi

    log "$LABEL $TAG: pushing ..."
    if ! ollama push "${OL_TARGET}:${TAG}" >>"$LOG" 2>&1; then
        log "$LABEL $TAG:   PUSH FAILED"
        rm -f "$GGUF" 2>/dev/null || true
        return 2
    fi

    # --- capture blob digests before removing refs (belt + suspenders against leak) ---
    local BLOBS
    BLOBS=$(
        for MF_PATH in \
            "/root/.ollama/models/manifests/registry.ollama.ai/${OL_TARGET}/${TAG}"; do
            [ -f "$MF_PATH" ] && grep -oE 'sha256:[0-9a-f]{64}' "$MF_PATH"
        done | sort -u
    )
    # Remove mannix ref (we never created hf.co/ ref with this strategy)
    ollama rm "${OL_TARGET}:${TAG}" >>"$LOG" 2>&1 || true
    # Force-purge captured blobs + any -partial-* residue
    local B
    for B in $BLOBS; do
        local FN="/root/.ollama/models/blobs/${B/sha256:/sha256-}"
        rm -f "$FN" "$FN-partial"* 2>/dev/null || true
    done
    # Delete the local source GGUF — we're done with it
    rm -f "$GGUF" 2>/dev/null || true

    df -h /workspace | tail -1 | tee -a "$LOG" >/dev/null
    log "$LABEL $TAG: DONE"
    sleep "$RETRY_SLEEP"
    return 0
}

log "Discovering tags in $HF_REPO ..."
# We need both the tag (Q4_K_M) AND the filename to download.
DISCOVERY=$(curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
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
            out.append((candidate, p)); break
seen = set()
for tag, path in out:
    if tag in seen: continue
    seen.add(tag)
    print(f'{tag}\t{path}')
")

# DISCOVERY is "TAG<TAB>FILENAME" lines; filter F16 + include pattern
DISCOVERY_FILTERED=$(echo "$DISCOVERY" | awk -F'\t' '$1 != "F16"')
if [[ -n "$INCLUDE" ]]; then
    DISCOVERY_FILTERED=$(echo "$DISCOVERY_FILTERED" | awk -F'\t' -v inc="$INCLUDE" '$1 ~ inc')
fi

ALL_TAGS=$(echo "$DISCOVERY_FILTERED" | awk -F'\t' '{print $1}' | sort -u)
log "Tags discovered (after filter):"
echo "$ALL_TAGS" | sed 's/^/  /' | tee -a "$LOG"
N_TAGS=$(echo "$ALL_TAGS" | grep -c . || true)
log "  total: $N_TAGS"

EXISTING=$(curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_-]+" | sed "s|.*:||" | sort -u || true)
if [[ -n "$EXISTING" ]]; then
    log "Already on ollama.com:"
    echo "$EXISTING" | sed 's/^/  /' | tee -a "$LOG"
fi

TODO=$(comm -23 <(echo "$ALL_TAGS" | sort -u) <(echo "$EXISTING" | sort -u) || true)
N_TODO=$(echo "$TODO" | grep -c . || true)
log "TODO: $N_TODO tag(s)"

# Lookup filename for each TODO tag from DISCOVERY_FILTERED
get_filename() {
    echo "$DISCOVERY_FILTERED" | awk -F'\t' -v t="$1" '$1==t {print $2; exit}'
}

# Main pass
FAILED_TAGS=()
i=0
for TAG in $TODO; do
    i=$((i+1))
    FILENAME=$(get_filename "$TAG")
    if [[ -z "$FILENAME" ]]; then
        log "[$i/$N_TODO] $TAG:   ERROR no filename mapping; skipping"
        FAILED_TAGS+=("$TAG")
        continue
    fi
    if ! push_one_tag "$TAG" "$FILENAME" "[$i/$N_TODO]"; then
        FAILED_TAGS+=("$TAG")
        flush_tag_cache "$TAG" "$SCRATCH/$FILENAME"
    fi
done

# Retry pass
if [[ ${#FAILED_TAGS[@]} -gt 0 ]]; then
    log "=== RETRY PASS: ${#FAILED_TAGS[@]} previously-failed tag(s): ${FAILED_TAGS[*]} ==="
    STILL_FAILED=()
    j=0
    for TAG in "${FAILED_TAGS[@]}"; do
        j=$((j+1))
        FILENAME=$(get_filename "$TAG")
        flush_tag_cache "$TAG" "$SCRATCH/$FILENAME"
        if ! push_one_tag "$TAG" "$FILENAME" "[retry $j/${#FAILED_TAGS[@]}]"; then
            STILL_FAILED+=("$TAG")
            flush_tag_cache "$TAG" "$SCRATCH/$FILENAME"
        fi
    done
    if [[ ${#STILL_FAILED[@]} -gt 0 ]]; then
        log "=== AFTER RETRY: still failed: ${STILL_FAILED[*]} ==="
        log "    (re-run this script later — these will be re-attempted via TODO recomputation.)"
    else
        log "=== AFTER RETRY: all previously-failed tags pushed successfully. ==="
    fi
fi

# Final scratch cleanup
rm -rf "$SCRATCH" 2>/dev/null || true

log "All done for $OL_TARGET. Tags published:"
curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_-]+" | sort -u | tee -a "$LOG"
