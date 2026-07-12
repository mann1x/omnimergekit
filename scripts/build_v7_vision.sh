#!/usr/bin/env bash
# Build vision-<tier> ollama variants for the v7-coder cohort (both models) by
# pairing each published text quant with mmproj-gemma4.gguf (the Gemma 4 SigLIP
# vision tower + projector — untouched by expert pruning, shared across all 98e
# prunes). Mirrors T185's v6-coder builder, with:
#   - corrected daemon store paths (/usr/share/ollama/.ollama, not /root)
#   - both v7 repos, tier set derived from each GGUF dir's Modelfiles
#   - SELF-GATES on the clean re-publish completion marker (waits until every
#     base tier is on ollama before building vision-<tier>).
# Per tier: pull text quant -> append `FROM mmproj` to its Modelfile ->
# create vision-<tier> -> verify vision cap (retry 3x) -> push -> rm both tags
# -> gc orphan blobs (protect the shared mmproj blob).
# Idempotent: a tier whose vision tag is already on ollama.com is skipped.
# Launch:  setsid nohup bash build_v7_vision.sh >LOG 2>&1 </dev/null &
set -uo pipefail

MMPROJ="/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf"
BLOBS="/usr/share/ollama/.ollama/models/blobs"
MANIFESTS="/usr/share/ollama/.ollama/models/manifests"
WORK="/mnt/sdc/ml/gguf/v7_vision_work"
REPUB_LOG="/srv/ml/logs/republish_v7_clean.log"
GATE="REPUBLISH_ALL_DONE"
mkdir -p "$WORK"
LOG(){ echo "[v7vis $(date -u +%H:%M:%S)] $*"; }

[ -f "$MMPROJ" ] || { LOG "FATAL: mmproj missing $MMPROJ"; exit 1; }
MMSHA="sha256-$(sha256sum "$MMPROJ" | cut -d' ' -f1)"
LOG "mmproj blob protected: $MMSHA"

# rows:  <ollama_repo>|<gguf_output_dir>
ROWS=(
  "mannix/gemma4-98e-v7-coder|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF"
  "mannix/gemma4-98e-v7-coderx|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF"
)

gc_blobs(){
  local ref; ref=$(mktemp)
  grep -rhoE "sha256[:-][0-9a-f]{64}" "$MANIFESTS" 2>/dev/null | sed 's/sha256:/sha256-/' | sort -u > "$ref"
  local n=0 b bn sz
  for b in "$BLOBS"/sha256-*; do
    [ -e "$b" ] || continue
    bn=$(basename "$b")
    [ "$bn" = "$MMSHA" ] && continue
    if ! grep -qx "$bn" "$ref"; then
      sz=$(du -h "$b" 2>/dev/null | cut -f1)
      rm -f "$b" && { LOG "  purged orphan blob $bn ($sz)"; n=$((n+1)); }
    fi
  done
  rm -f "$ref"; LOG "  gc: purged $n orphan blob(s)"
}

build_one(){
  local REPO="$1" GD="$2" T VT
  # tiers = Modelfile.* basenames minus F16 (F16 isn't a deployed ollama tier)
  local TIERS; TIERS=$(ls "$GD"/_ollama_push/Modelfile.* 2>/dev/null | sed 's#.*/Modelfile.##' | grep -vx 'F16' | sort)
  [ -n "$TIERS" ] || { LOG "WARN: no Modelfiles in $GD/_ollama_push — skip $REPO"; return 0; }
  LOG "===== $REPO : $(echo "$TIERS" | wc -w) tiers ====="
  for T in $TIERS; do
    VT="vision-$T"
    LOG "  -- tier $T -> $REPO:$VT"
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "https://registry.ollama.ai/v2/$REPO/manifests/$VT" 2>/dev/null)" = 200 ]; then
      LOG "     $VT already on registry — skip + reclaim local"
      ollama rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1; gc_blobs
      continue
    fi
    ollama pull "$REPO:$T" >/dev/null 2>&1 || { LOG "     PULL FAILED — skip"; continue; }
    ollama show "$REPO:$T" --modelfile > "$WORK/mf_$T" 2>/dev/null || { LOG "     modelfile FAILED — skip"; continue; }
    { cat "$WORK/mf_$T"; echo; echo "FROM $MMPROJ"; } > "$WORK/mfv_$T"
    ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1 || { LOG "     CREATE FAILED — skip"; continue; }
    local vcap=0 ck
    for ck in 1 2 3; do
      if ollama show "$REPO:$VT" 2>/dev/null | grep -qi vision; then vcap=1; break; fi
      LOG "     vision cap not visible (check $ck/3) — recreate+retry"
      ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1; sleep 3
    done
    [ "$vcap" = 1 ] || { LOG "     WARN: $VT no vision cap after 3 tries — NOT pushing"; ollama rm "$REPO:$VT" >/dev/null 2>&1; continue; }
    ollama push "$REPO:$VT" >/dev/null 2>&1 || true
    # authoritative: registry manifest HTTP 200 (last-line grep was a false-negative trap)
    pushok=0
    for pk in 1 2 3; do
      [ "$(curl -s -o /dev/null -w '%{http_code}' "https://registry.ollama.ai/v2/$REPO/manifests/$VT" 2>/dev/null)" = 200 ] && { pushok=1; break; }
      sleep 5
    done
    if [ "$pushok" = 1 ]; then
      LOG "     pushed OK $VT (registry 200)"
    else
      LOG "     PUSH FAILED $VT (registry != 200) — rm local, skip"; ollama rm "$REPO:$VT" >/dev/null 2>&1; continue
    fi
    ollama rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1
    gc_blobs
    df -h /usr/share/ollama 2>/dev/null | awk 'NR==2{print "[v7vis] store used="$5" avail="$4}'
    rm -f "$WORK/mf_$T" "$WORK/mfv_$T"
    LOG "     DONE $VT"
  done
  LOG "===== DONE $REPO ====="
}

if [ "${WAIT_GATE:-1}" = 1 ]; then
  LOG "gate: waiting for '$GATE' in $REPUB_LOG (base tiers must be on ollama first)"
  while ! grep -q "$GATE" "$REPUB_LOG" 2>/dev/null; do sleep 60; done
  LOG "gate released — re-publish complete"
fi

for row in "${ROWS[@]}"; do
  IFS='|' read -r repo gd <<<"$row"
  build_one "$repo" "$gd"
done
LOG "###### V7_VISION_ALL_DONE ######"
