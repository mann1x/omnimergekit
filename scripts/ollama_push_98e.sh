#!/usr/bin/env bash
# Push all GGUF quants from a HF repo to ollama.com under <ollama-user>/<ollama-repo>:<TAG>
# Reuses the existing template/parameters from mannix/gemma4-98e:CD-Q6_K
# (tools template + 2nd-turn workaround preserved).
#
# Strategy (revised 2026-05-11): pre-download GGUF via `hf download` to a local
# path, then `ollama create FROM /local/path.gguf`. Bypasses ollama's HF-pull
# machinery (whose per-create deadline is hardcoded short and tends to fail on
# 12-25 GB blobs). One quant on disk at a time → ~25 GB scratch budget.
#
# Usage:
#   bash ollama_push_98e.sh <hf_repo> <ollama_target> <hf_token> [include_pattern]
#   # legacy: HF_TOKEN env var also accepted if <hf_token> is "-"
# e.g.
#   bash ollama_push_98e.sh ManniX-ITA/gemma-4-A4B-98e-v4-it-GGUF mannix/gemma4-98e-v4 "$(cat ~/.cache/huggingface/token)"

set -euo pipefail
HF_REPO="${1:?hf repo (e.g. ManniX-ITA/gemma-4-A4B-98e-v3-it-GGUF)}"
OL_TARGET="${2:?ollama target (e.g. mannix/gemma4-98e)}"
HF_TOKEN_ARG="${3:--}"
INCLUDE="${4:-}"

if [[ "$HF_TOKEN_ARG" != "-" && -n "$HF_TOKEN_ARG" ]]; then
    export HF_TOKEN="$HF_TOKEN_ARG"
fi
: "${HF_TOKEN:?HF_TOKEN required as arg-3 or env (read token for HF GGUF repo)}"

WORKDIR=/workspace/ollama_push
LOG_DIR=/workspace/logs
SCRATCH=/workspace/ollama_push/scratch
mkdir -p "$WORKDIR" "$LOG_DIR" "$SCRATCH"

LOG="$LOG_DIR/ollama_push_${OL_TARGET//\//_}.log"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_SLEEP=${RETRY_SLEEP:-5}

# === The template + parameters from existing CD-Q6_K push (verbatim) ===
TEMPLATE_FILE="$WORKDIR/template.tmpl"
cat > "$TEMPLATE_FILE" <<'TMPL_EOF'
{{- if or .System .Tools }}<bos><|turn>system
{{ if .System }}{{ .System }}
{{ end }}{{- if .Tools }}
You may call one or more functions to assist the user. Available functions:
{{- range .Tools }}
<|tool>declaration:{{ .Function.Name }}{{ .Function.Parameters }}<tool|>
{{- end }}

To call a function, emit:
<|tool_call>call:FUNCTION_NAME{arg_name:<|"|>string_value<|"|>,other_arg:number}<tool_call|>

All string values MUST be wrapped in <|"|>...<|"|> delimiters. Numbers and booleans are raw.
{{- end }}<turn|>
{{ end -}}
{{- range $i, $msg := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq $msg.Role "system" }}
{{- if gt $i 0 }}<|turn>system
{{ $msg.Content }}<turn|>
{{ end }}
{{- else if eq $msg.Role "user" }}<|turn>user
{{ $msg.Content }}<turn|>
{{ else if eq $msg.Role "assistant" }}<|turn>model
{{- if $msg.Content }}
{{ $msg.Content }}
{{- end }}
{{- range $msg.ToolCalls }}
<|tool_call>call:{{ .Function.Name }}{{ .Function.Arguments }}<tool_call|>
{{- end }}{{ if not $last }}<turn|>
{{ end }}
{{- else if eq $msg.Role "tool" }}<|turn>user
<|tool_response>response:{{ $msg.Name }}{ {{ $msg.Content }} }<tool_response|><turn|>
{{ end }}
{{- if and (ne $msg.Role "assistant") $last }}<|turn>model
{{ end }}
{{- end }}
TMPL_EOF

# Modelfile builder for one tag — FROM local file path + Gemma 4 chat config
write_modelfile() {
    local TAG="$1" MF="$2" GGUF="$3"
    cat > "$MF" <<EOF
FROM $GGUF
TEMPLATE """$(cat "$TEMPLATE_FILE")"""
RENDERER gemma4
PARSER gemma4
PARAMETER repeat_penalty 1.15
PARAMETER stop <turn|>
PARAMETER stop <|tool_response>
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_ctx 256000
PARAMETER repeat_last_n 256
EOF
}

flush_tag_cache() {
    local TAG="$1" GGUF="${2:-}"
    ollama rm "${OL_TARGET}:${TAG}"   >>"$LOG" 2>&1 || true
    ollama rm "hf.co/${HF_REPO}:${TAG}" >>"$LOG" 2>&1 || true
    rm -f /root/.ollama/models/blobs/sha256-*-partial* 2>/dev/null || true
    [ -n "$GGUF" ] && rm -f "$GGUF" 2>/dev/null || true
}

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

push_one_tag() {
    local TAG="$1" FILENAME="$2"
    local LABEL="${3:-}"
    local MF="$WORKDIR/Modelfile.${TAG}"
    local GGUF=""

    local attempt
    local create_ok=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        log "$LABEL $TAG: download + create attempt $attempt/$MAX_ATTEMPTS"

        if ! GGUF=$(hf_download_one "$FILENAME"); then
            log "$LABEL $TAG:   download attempt $attempt failed"
            flush_tag_cache "$TAG" "$SCRATCH/$FILENAME"
            [[ $attempt -lt $MAX_ATTEMPTS ]] && sleep "$RETRY_SLEEP"
            continue
        fi
        log "$LABEL $TAG:   downloaded $(du -h "$GGUF" | cut -f1) to $GGUF"

        write_modelfile "$TAG" "$MF" "$GGUF"

        if ollama create "${OL_TARGET}:${TAG}" -f "$MF" >>"$LOG" 2>&1; then
            create_ok=1
            break
        fi
        log "$LABEL $TAG:   create attempt $attempt failed; flushing cache"
        flush_tag_cache "$TAG" "$GGUF"
        [[ $attempt -lt $MAX_ATTEMPTS ]] && sleep "$RETRY_SLEEP"
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

    local BLOBS
    BLOBS=$(
        for MF_PATH in \
            "/root/.ollama/models/manifests/registry.ollama.ai/${OL_TARGET}/${TAG}"; do
            [ -f "$MF_PATH" ] && grep -oE 'sha256:[0-9a-f]{64}' "$MF_PATH"
        done | sort -u
    )
    ollama rm "${OL_TARGET}:${TAG}" >>"$LOG" 2>&1 || true
    local B
    for B in $BLOBS; do
        local FN="/root/.ollama/models/blobs/${B/sha256:/sha256-}"
        rm -f "$FN" "$FN-partial"* 2>/dev/null || true
    done
    rm -f "$GGUF" 2>/dev/null || true

    df -h /workspace | tail -1 | tee -a "$LOG" >/dev/null
    log "$LABEL $TAG: DONE"
    sleep "$RETRY_SLEEP"
    return 0
}

log "Discovering tags in $HF_REPO ..."
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
    # Pattern for 98e gemma 4 GGUFs: *-it-<TAG>.gguf  (e.g. CD-Q6_K)
    m = re.match(r'.+?-it-(.+)\\.gguf\$', p)
    if m:
        out.append((m.group(1), p))
seen = set()
for tag, path in out:
    if tag in seen: continue
    seen.add(tag)
    print(f'{tag}\t{path}')
")

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

get_filename() {
    echo "$DISCOVERY_FILTERED" | awk -F'\t' -v t="$1" '$1==t {print $2; exit}'
}

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

rm -rf "$SCRATCH" 2>/dev/null || true

log "All done for $OL_TARGET. Tags published:"
curl -sSL "https://ollama.com/${OL_TARGET}/tags" 2>/dev/null \
    | grep -oE "${OL_TARGET}:[A-Za-z0-9_-]+" | sort -u | tee -a "$LOG"
