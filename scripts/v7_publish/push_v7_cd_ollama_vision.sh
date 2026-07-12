#!/usr/bin/env bash
# Push the 3 new CD tiers (CD-Q2_K, CD-Q3_K_L, CD-qat-Q4_K_M) to ollama for both
# v7 models, as PLAIN + vision-<tier>. Force-overwrites (CD-Q3_K_L's ollama tag
# is the stale old-map version). Uses the Gemma-4 Modelfile template (RENDERER/
# PARSER gemma4 + chat template) copied from an existing tier's Modelfile, and
# pairs mmproj-gemma4.gguf for the vision variant. Bounded: 3 tiers x 2 models.
set -uo pipefail
export HF_TOKEN="$(cat /root/.cache/huggingface/token 2>/dev/null)"
: "${HF_TOKEN:?HF_TOKEN must be readable}"
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
MMPROJ=/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf
BLOBS=/usr/share/ollama/.ollama/models/blobs
MANIFESTS=/usr/share/ollama/.ollama/models/manifests
SCRATCH=/mnt/sdc/ml/ollama_cd_scratch
WORK=/mnt/sdc/ml/ollama_cd_work
mkdir -p "$SCRATCH" "$WORK"
LOG(){ echo "[cdoll $(date -u +%H:%M:%S)] $*"; }
[ -f "$MMPROJ" ] || { LOG "FATAL mmproj missing $MMPROJ"; exit 1; }
command -v "$HF" >/dev/null || { LOG "FATAL hf missing $HF"; exit 1; }
MMSHA="sha256-$(sha256sum "$MMPROJ" | cut -d' ' -f1)"
LOG "mmproj protected blob $MMSHA"

TIERS="CD-Q2_K CD-Q3_K_L CD-qat-Q4_K_M"
# ollama_target | hf_gguf_repo | stem | template_modelfile (any existing gemma4 tier)
ROWS=(
"mannix/gemma4-98e-v7-coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/_ollama_push/Modelfile.Q4_K_M"
"mannix/gemma4-98e-v7-coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/_ollama_push/Modelfile.Q4_K_M"
)

reg200(){ [ "$(curl -s -o /dev/null -w '%{http_code}' "https://registry.ollama.ai/v2/$1/manifests/$2" 2>/dev/null)" = 200 ]; }

gc_blobs(){
  local ref; ref=$(mktemp)
  grep -rhoE "sha256[:-][0-9a-f]{64}" "$MANIFESTS" 2>/dev/null | sed 's/sha256:/sha256-/' | sort -u > "$ref"
  local b bn
  for b in "$BLOBS"/sha256-*; do
    [ -e "$b" ] || continue; bn=$(basename "$b")
    [ "$bn" = "$MMSHA" ] && continue
    grep -qx "$bn" "$ref" || rm -f "$b"
  done
  rm -f "$ref"
}

rc=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r TARGET HFREPO STEM TMPL <<<"$row"
  LOG "########## $TARGET ($HFREPO) ##########"
  [ -f "$TMPL" ] || { LOG "FATAL template modelfile missing $TMPL"; rc=1; continue; }
  TBODY=$(grep -v '^FROM ' "$TMPL")   # gemma4 RENDERER/PARSER/TEMPLATE, no FROM
  for T in $TIERS; do
    GG="$SCRATCH/${STEM}-${T}.gguf"
    LOG ">>> [$TARGET] $T"
    rm -f "$GG"
    if ! "$HF" download "$HFREPO" "${STEM}-${T}.gguf" --local-dir "$SCRATCH" >/dev/null 2>&1; then
      LOG "  FATAL hf download ${STEM}-${T}.gguf"; rc=1; continue
    fi
    [ -s "$GG" ] || { LOG "  FATAL downloaded file empty $GG"; rc=1; continue; }
    LOG "  downloaded $(du -h "$GG"|cut -f1)"
    # ---- plain tag (force overwrite) ----
    { echo "FROM $GG"; echo "$TBODY"; } > "$WORK/mf.$T"
    if ! ollama create "$TARGET:$T" -f "$WORK/mf.$T" >/dev/null 2>&1; then LOG "  FATAL create plain $T"; rc=1; rm -f "$GG"; continue; fi
    ollama push "$TARGET:$T" >/dev/null 2>&1 || true
    if reg200 "$TARGET" "$T"; then LOG "  PLAIN pushed OK $TARGET:$T (registry 200)"; else LOG "  WARN plain $T registry!=200"; rc=1; fi
    # ---- vision tag (pair mmproj) ----
    VT="vision-$T"
    { echo "FROM $GG"; echo "$TBODY"; echo; echo "FROM $MMPROJ"; } > "$WORK/mfv.$T"
    if ! ollama create "$TARGET:$VT" -f "$WORK/mfv.$T" >/dev/null 2>&1; then LOG "  FATAL create vision $VT"; rc=1; ollama rm "$TARGET:$T" >/dev/null 2>&1; rm -f "$GG"; continue; fi
    vcap=0
    for ck in 1 2 3; do
      ollama show "$TARGET:$VT" 2>/dev/null | grep -qi vision && { vcap=1; break; }
      ollama create "$TARGET:$VT" -f "$WORK/mfv.$T" >/dev/null 2>&1; sleep 3
    done
    if [ "$vcap" = 1 ]; then
      ollama push "$TARGET:$VT" >/dev/null 2>&1 || true
      if reg200 "$TARGET" "$VT"; then LOG "  VISION pushed OK $TARGET:$VT (registry 200)"; else LOG "  WARN vision $VT registry!=200"; rc=1; fi
    else
      LOG "  WARN $VT no vision cap after 3 tries — NOT pushing"; rc=1
    fi
    ollama rm "$TARGET:$T" "$TARGET:$VT" >/dev/null 2>&1
    gc_blobs
    rm -f "$GG" "$WORK/mf.$T" "$WORK/mfv.$T"
    df -h /usr/share/ollama 2>/dev/null | awk 'NR==2{print "[cdoll] store used="$5" avail="$4}'
    LOG "  DONE $T (plain + vision)"
  done
done
LOG "###### V7_CD_OLLAMA_DONE rc=$rc ######"
exit $rc
