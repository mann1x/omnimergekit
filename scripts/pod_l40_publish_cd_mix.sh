#!/bin/bash
# Pod L40 — publish 4 v5-coder CD-mix scored recipes to HF.
#
# Targets HF repo: ManniX-ITA/gemma-4-A4B-98e-v5-coder-it-GGUF
# Sequence:
#   1. Upload existing CD-IQ3_M_mix GGUF (already on disk) — fast win
#   2. Download F16 from HF (~50 GB)
#   3. Generate maps for missing recipes
#   4. For each: quantize → upload → delete to free disk
#
# Ollama push: NOT done here (needs ed25519 key on solidpc). After all
# HF uploads complete, solidpc runs omnimergekit/scripts/ollama_push_98e.sh
# to fetch from HF and push to mannix/gemma4-98e-v5-coder.
#
# Designed for: 137 GB free disk, 1 GGUF at a time, fail-safe on each.
set -uo pipefail

WORK=/workspace/cd_mix_publish
F16_DIR=/workspace/v5coder_bf16
LOGS=/workspace/logs
mkdir -p "$WORK" "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOGS/cd_mix_publish_$TS.log"
exec > >(tee -a "$LOG") 2>&1

LLAMA_QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK_PY=/workspace/miniconda/envs/omnimergekit/bin/python
export PATH=/workspace/miniconda/envs/omnimergekit/bin:$PATH
: "${HF_TOKEN:?HF_TOKEN required — export before launch}"

log() { echo "[$(date -Iseconds)] $*"; }

TOKENIZER="ManniX-ITA/gemma-4-A4B-98e-v5-coder-it"
F16_REPO="ManniX-ITA/gemma-4-A4B-98e-v5-coder-it-GGUF"
F16_NAME="gemma-4-A4B-98e-v5-coder-it-F16.gguf"
F16="$F16_DIR/$F16_NAME"
GGUF_REPO="ManniX-ITA/gemma-4-A4B-98e-v5-coder-it-GGUF"
IMATRIX=/workspace/v5coder_imatrix.dat
LAYER_IMP=/workspace/v5coder_layer_importance_v3_imix.json
GEN_MIX=/workspace/generate_cd_maps_mix.py

# Recipe ordering: smallest-first uploads while large download proceeds.
# All names use the new _L suffix (renamed 2026-05-19 from _mix). The
# legacy CD-IQ3_M_mix GGUF on disk is re-uploaded under its _L name; the
# stale _mix file on HF will be deleted in a separate step.
RECIPES_PHASE1=(
  "CD-IQ3_M_L:IQ3_M:/workspace/canary_v_prime/gemma-4-A4B-98e-v5-coder-it-CD-IQ3_M_mix.gguf"
)
# phase 2 = build-from-F16 (recipe:file_base — GGUF path derived)
RECIPES_PHASE2=(
  "CD-Q4_K_M_L:Q4_K_M"
  "CD-IQ3_XS_L:IQ3_XS"
  "CD-Q6_K_L:Q6_K"
)

log "##########################################################"
log "Pod L40 CD-mix publish — 4 recipes → HF GGUF repo"
log "##########################################################"

# Sanity
for f in "$IMATRIX" "$LLAMA_QUANT"; do
    [ -e "$f" ] || { log "ERR missing $f"; exit 1; }
done
which hf >/dev/null 2>&1 || { log "ERR hf CLI missing"; exit 1; }
hf auth whoami 2>&1 | head -1

# Phase 1: Upload existing CD-IQ3_M_mix GGUF (no rebuild)
for entry in "${RECIPES_PHASE1[@]}"; do
    IFS=':' read -r recipe ftype gguf_path <<< "$entry"
    log "######### Phase 1: upload $recipe (pre-built) #########"
    if [ ! -f "$gguf_path" ]; then
        log "  WARN $gguf_path missing — skipping phase-1 upload"
        continue
    fi
    log "  source GGUF: $gguf_path ($(du -h "$gguf_path" | awk '{print $1}'))"
    target_name="gemma-4-A4B-98e-v5-coder-it-${recipe}.gguf"
    log "  uploading to $GGUF_REPO/$target_name ..."
    hf upload "$GGUF_REPO" "$gguf_path" "$target_name" \
        --commit-message "Add $recipe (HE+ 90.24% / 10.2 GB; mix-recipe, body-only CD + heuristic protect)" \
      && log "  ✓ uploaded $recipe" \
      || log "  ERR upload failed for $recipe"
done

# Phase 2: pull F16 if not present
if [ ! -f "$F16" ]; then
    mkdir -p "$F16_DIR"
    log "Phase 2 prep: downloading F16 from $F16_REPO ..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    hf download "$F16_REPO" "$F16_NAME" --local-dir "$F16_DIR"
    log "  F16 download done: $(du -h "$F16" | awk '{print $1}')"
fi
[ -f "$F16" ] || { log "ERR F16 still missing — aborting phase 2"; exit 1; }

# Generate maps for phase 2 recipes
log "Phase 2: generating maps for: ${RECIPES_PHASE2[*]} ..."
[ -e "$GEN_MIX" ] || { log "ERR $GEN_MIX missing — pull omnimergekit"; exit 1; }
recipe_names=()
for entry in "${RECIPES_PHASE2[@]}"; do
    IFS=':' read -r r _ <<< "$entry"
    recipe_names+=("$r")
done
"$OMK_PY" "$GEN_MIX" \
    --imatrix "$IMATRIX" \
    --layer-importance "$LAYER_IMP" \
    --out-dir "$WORK" \
    --recipes "${recipe_names[@]}"

# Phase 2: build + upload each. Track failures so F16 is preserved when any
# recipe fails to create or upload — saves re-downloading 38 GB on retry.
# Override with --keep-f16 to always preserve, or --force-purge to force
# delete even on failure.
KEEP_F16_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --keep-f16)    KEEP_F16_FLAG="keep" ;;
        --force-purge) KEEP_F16_FLAG="purge" ;;
    esac
done

FAILED_RECIPES=()
SUCCEEDED_RECIPES=()

for entry in "${RECIPES_PHASE2[@]}"; do
    IFS=':' read -r recipe file_base <<< "$entry"
    out_gguf="$WORK/gemma-4-A4B-98e-v5-coder-it-${recipe}.gguf"
    map_file="$WORK/tensor_types_${recipe}.txt"

    log "######### Phase 2: build $recipe (file-base=$file_base) #########"
    log "  df: $(df -h /workspace | tail -1 | awk '{print $4}') avail"

    if [ ! -f "$map_file" ]; then
        log "  ERR map $map_file missing — skipping"
        FAILED_RECIPES+=("$recipe (map missing)")
        continue
    fi

    if [ -f "$out_gguf" ] && [ "$(stat -c%s "$out_gguf")" -gt 1000000000 ]; then
        log "  GGUF exists ($(du -h "$out_gguf" | awk '{print $1}')) — skipping quantize"
    else
        rm -f "$out_gguf"
        log "  quantizing..."
        "$LLAMA_QUANT" --imatrix "$IMATRIX" --tensor-type-file "$map_file" \
            "$F16" "$out_gguf" "$file_base"
        q_rc=$?
        log "  quantize rc=$q_rc"
        if [ ! -f "$out_gguf" ]; then
            log "  ERR quantize failed for $recipe — skipping"
            FAILED_RECIPES+=("$recipe (quantize failed)")
            continue
        fi
    fi
    log "  GGUF: $(du -h "$out_gguf" | awk '{print $1}')"

    target_name="gemma-4-A4B-98e-v5-coder-it-${recipe}.gguf"
    log "  uploading to $GGUF_REPO/$target_name ..."
    if hf upload "$GGUF_REPO" "$out_gguf" "$target_name" \
        --commit-message "Add $recipe (CD-mix recipe; body-only override + file-base heuristic protect)"; then
        log "  ✓ uploaded $recipe"
        SUCCEEDED_RECIPES+=("$recipe")
        rm -f "$out_gguf"
        log "  freed disk: $(df -h /workspace | tail -1 | awk '{print $4}') avail"
    else
        log "  ERR upload failed for $recipe — keeping GGUF for retry"
        FAILED_RECIPES+=("$recipe (upload failed; GGUF kept)")
    fi
done

# F16 cleanup decision matrix:
#   --force-purge → always delete (legacy behaviour)
#   --keep-f16    → always keep
#   default       → keep IF any failure (saves re-download), purge if all OK
if [ "$KEEP_F16_FLAG" = "purge" ]; then
    log "F16 cleanup: --force-purge requested; deleting $F16"
    rm -f "$F16"
elif [ "$KEEP_F16_FLAG" = "keep" ]; then
    log "F16 cleanup: --keep-f16 requested; preserved at $F16"
elif [ ${#FAILED_RECIPES[@]} -gt 0 ]; then
    log "F16 cleanup: ${#FAILED_RECIPES[@]} recipe(s) failed — keeping F16 for retry:"
    for f in "${FAILED_RECIPES[@]}"; do log "    - $f"; done
    log "    F16 preserved at $F16 ($(du -h "$F16" 2>/dev/null | awk '{print $1}'))"
    log "    Re-run this script after fixing the failure — completed recipes will be skipped"
else
    log "F16 cleanup: all ${#SUCCEEDED_RECIPES[@]} recipe(s) succeeded — deleting $F16"
    rm -f "$F16"
fi

log "##########################################################"
log "L40 publish chain DONE  (ok=${#SUCCEEDED_RECIPES[@]}, fail=${#FAILED_RECIPES[@]})"
log "##########################################################"
if [ ${#SUCCEEDED_RECIPES[@]} -gt 0 ]; then
    log "Uploaded: ${SUCCEEDED_RECIPES[*]}"
fi
if [ ${#FAILED_RECIPES[@]} -gt 0 ]; then
    log "Failures: ${FAILED_RECIPES[*]}"
fi
log "Next step: ollama push from L40 via omnimergekit/scripts/ollama_push_98e.sh"
