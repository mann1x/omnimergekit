#!/usr/bin/env bash
# Push all GGUF quants from a HF repo to ollama.com under <ollama-user>/<ollama-repo>:<TAG>
# Reuses the existing template/parameters from mannix/gemma4-98e:CD-Q6_K
# (tools template + 2nd-turn workaround preserved).
#
# Strategy: one tag at a time → ollama create from local pulled GGUF → push →
# delete local copy + free blob to keep disk under control.
#
# Usage:
#   bash ollama_push_98e.sh <hf_repo> <ollama_target> [include_pattern]
# e.g.
#   bash ollama_push_98e.sh ManniX-ITA/gemma-4-A4B-98e-v3-it-GGUF mannix/gemma4-98e
#   bash ollama_push_98e.sh ManniX-ITA/gemma-4-A4B-98e-v4-it-GGUF mannix/gemma4-98e-v4

set -euo pipefail
HF_REPO="${1:?hf repo (e.g. ManniX-ITA/gemma-4-A4B-98e-v3-it-GGUF)}"
OL_TARGET="${2:?ollama target (e.g. mannix/gemma4-98e)}"
INCLUDE="${3:-}"   # optional grep pattern to filter tags

WORKDIR=/workspace/ollama_push
LOG_DIR=/workspace/logs
mkdir -p "$WORKDIR" "$LOG_DIR"

LOG="$LOG_DIR/ollama_push_${OL_TARGET//\//_}.log"
ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Per-tag retry config
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_SLEEP=${RETRY_SLEEP:-5}

# === The template + parameters from existing CD-Q6_K push (verbatim) ===
# Tools template includes the 2nd-turn fix.
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

# Build a Modelfile for one tag (Gemma 4 template + params)
write_modelfile() {
    local TAG="$1" MF="$2"
    cat > "$MF" <<EOF
FROM hf.co/${HF_REPO}:${TAG}
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

# Flush any cached state for a failing tag so the next attempt redownloads.
flush_tag_cache() {
    local TAG="$1"
    ollama rm "hf.co/${HF_REPO}:${TAG}"  >>"$LOG" 2>&1 || true
    ollama rm "${OL_TARGET}:${TAG}"       >>"$LOG" 2>&1 || true
    rm -f /root/.ollama/models/blobs/sha256-*-partial* 2>/dev/null || true
}

# Run the create + push for one tag with retry. Returns 0 on success.
push_one_tag() {
    local TAG="$1"
    local LABEL="${2:-}"
    local MF="$WORKDIR/Modelfile.${TAG}"
    write_modelfile "$TAG" "$MF"

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
    ollama rm "${OL_TARGET}:${TAG}"      >>"$LOG" 2>&1 || true
    ollama rm "hf.co/${HF_REPO}:${TAG}"  >>"$LOG" 2>&1 || true
    local B
    for B in $BLOBS; do
        local FN="/root/.ollama/models/blobs/${B/sha256:/sha256-}"
        rm -f "$FN" "$FN-partial"* 2>/dev/null || true
    done

    df -h /workspace | tail -1 | tee -a "$LOG" >/dev/null
    log "$LABEL $TAG: DONE"
    sleep "$RETRY_SLEEP"
    return 0
}

# Discover all .gguf files in the HF repo
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
    m = re.match(r'.+?-it-(.+)\\.gguf\$', p)
    if m:
        out.append((m.group(1), p))
print('\\n'.join(t for t, _ in out))
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

# Main pass — failures collected into FAILED_TAGS for retry pass
FAILED_TAGS=()
i=0
for TAG in $TODO; do
    i=$((i+1))
    if ! push_one_tag "$TAG" "[$i/$N_TODO]"; then
        FAILED_TAGS+=("$TAG")
        flush_tag_cache "$TAG"
    fi
done

# Retry pass: redownload from HF each previously-failed tag
if [[ ${#FAILED_TAGS[@]} -gt 0 ]]; then
    log "=== RETRY PASS: ${#FAILED_TAGS[@]} previously-failed tag(s): ${FAILED_TAGS[*]} ==="
    STILL_FAILED=()
    j=0
    for TAG in "${FAILED_TAGS[@]}"; do
        j=$((j+1))
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
