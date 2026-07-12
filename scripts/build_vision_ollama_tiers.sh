#!/usr/bin/env bash
# T185: build vision-<tier> ollama variants for mannix/gemma4-98e-v6-coder by
# pairing each existing text quant with mmproj-gemma4.gguf (Gemma4 vision tower).
#
# Per tier: pull text quant from the ollama registry -> derive its Modelfile ->
# append `FROM mmproj` -> create vision-<tier> -> push -> rm both tags ->
# purge orphaned blob files (KEEP the shared mmproj blob). Disk-hygienic so the
# 8-21 GB quants don't accumulate in /root/.ollama/models/blobs.
#
# Idempotent: a tier whose vision tag is already on ollama.com is skipped.
# Usage: build_vision_ollama_tiers.sh <tier> [<tier> ...]
set -uo pipefail

REPO="mannix/gemma4-98e-v6-coder"
MMPROJ="/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf"
BLOBS="/root/.ollama/models/blobs"
MANIFESTS="/root/.ollama/models/manifests"
WORK="/mnt/sdc/ml/gguf/v6coder/vision_work"
mkdir -p "$WORK"
LOG(){ echo "[vis $(date -u +%H:%M:%S)] $*"; }

[ -f "$MMPROJ" ] || { LOG "FATAL: mmproj missing $MMPROJ"; exit 1; }
MMSHA="sha256-$(sha256sum "$MMPROJ" | cut -d' ' -f1)"
LOG "mmproj blob protected: $MMSHA"

gc_blobs(){
  # delete blob files not referenced by ANY current manifest, except the mmproj blob
  local ref; ref=$(mktemp)
  grep -rhoE "sha256[:-][0-9a-f]{64}" "$MANIFESTS" 2>/dev/null | sed 's/sha256:/sha256-/' | sort -u > "$ref"
  local n=0
  for b in "$BLOBS"/sha256-*; do
    [ -e "$b" ] || continue
    local bn; bn=$(basename "$b")
    [ "$bn" = "$MMSHA" ] && continue
    if ! grep -qx "$bn" "$ref"; then
      local sz; sz=$(du -h "$b" 2>/dev/null | cut -f1)
      rm -f "$b" && { LOG "  purged orphan blob $bn ($sz)"; n=$((n+1)); }
    fi
  done
  rm -f "$ref"
  LOG "  gc: purged $n orphan blob(s)"
}

for T in "$@"; do
  VT="vision-$T"
  LOG "===== tier $T -> $REPO:$VT ====="
  if curl -s "https://ollama.com/$REPO/tags" 2>/dev/null | grep -q ":$VT\b"; then
    LOG "  $VT already on registry — skip"; continue
  fi
  LOG "  pull $REPO:$T"
  ollama pull "$REPO:$T" >/dev/null 2>&1 || { LOG "  PULL FAILED — skip"; continue; }
  ollama show "$REPO:$T" --modelfile > "$WORK/mf_$T" 2>/dev/null || { LOG "  modelfile FAILED — skip"; continue; }
  # vision modelfile = original (template/params/FROM base-blob) + projector
  { cat "$WORK/mf_$T"; echo; echo "FROM $MMPROJ"; } > "$WORK/mfv_$T"
  LOG "  create $REPO:$VT"
  ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1 || { LOG "  CREATE FAILED — skip"; continue; }
  # confirm vision capability before pushing; ollama create occasionally
  # returns before the projector layer is queryable -> recreate+recheck up to 3x.
  vcap=0
  for _ck in 1 2 3; do
    if ollama show "$REPO:$VT" 2>/dev/null | grep -qi vision; then vcap=1; break; fi
    LOG "  vision cap not visible (check $_ck/3) - recreate+retry"
    ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1
    sleep 3
  done
  if [ "$vcap" = 0 ]; then
    LOG "  WARN: $VT no vision capability after 3 tries - NOT pushing"; ollama rm "$REPO:$VT" >/dev/null 2>&1; continue
  fi
  LOG "  push $REPO:$VT"
  if ! ollama push "$REPO:$VT" 2>&1 | tail -2; then
    LOG "  PUSH FAILED — leaving tags for inspection"; continue
  fi
  ollama rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1
  gc_blobs
  df -h /root | awk 'NR==2{print "[vis] /root used="$5" avail="$4}'
  rm -f "$WORK/mf_$T" "$WORK/mfv_$T"
  LOG "  DONE $VT"
done
LOG "BATCH DONE"
