#!/usr/bin/env bash
# republish_coderx_ollama.sh — republish CODERX (code4/lcb3 = CX16c4l3, the re-release that
# REPLACES the looping fs2440 build) to ollama with the DEPLOY sampler + per-tier loop-gated
# temperature taken from the CODERX-SPECIFIC 48-seed sweep (coderx_loop_sweep/out/*.json),
# NOT std16's. 13 loop-clean tiers: text + vision-<tier> + latest. Per tier: create+push text,
# build vision (inherit text params + mmproj), then rm both + gc orphan blobs (protect mmproj)
# so disk stays bounded to ~1 tier at a time. F16 is intentionally NOT pushed.
# Launch:  setsid nohup bash republish_coderx_ollama.sh >LAUNCH 2>&1 </dev/null &
set -uo pipefail
OL=/usr/local/bin/ollama
REPO=mannix/gemma4-98e-v7-coderx
GG=/mnt/sdc/ml/cx_std16/gguf_coderx
STEM=CX16c4l3-bf16
MMPROJ=/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf
STORE=/usr/share/ollama/.ollama/models
BLOBS="$STORE/blobs"
MANIFESTS="$STORE/manifests"
WORK=/mnt/sdc/ml/cx_std16/ollama_republish
mkdir -p "$WORK"
LOG="$WORK/republish.log"
exec >>"$LOG" 2>&1
L(){ echo "[republish-coderx $(date -u +%H:%M:%S)] $*"; }

# per-tier temp = CODERX 48-seed loop gate (coderx_loop_sweep). 0.8 for the tiers that loop at 0.9.
# Q3_K_M loops 1/48 at BOTH 0.8 and 0.9 (no clean temp in sweep) -> ship 0.8 (conservative), flagged.
declare -A TEMP=(
  [Q8_0]=0.9 [Q6_K_L]=0.9 [Q6_K]=0.9 [Q5_K_L]=0.9 [Q5_K_M]=0.9 [Q4_K_L]=0.9 [Q4_K_M]=0.9 [IQ4_NL]=0.9
  [Q4_K_S]=0.8 [IQ4_XS]=0.8 [Q3_K_L]=0.8 [Q3_K_M]=0.8 [CD-Q2_K]=0.8
)
TIERS=(Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)
LATEST_TIER=Q4_K_M

[ -f "$MMPROJ" ] || { L "FATAL: mmproj missing $MMPROJ"; exit 1; }
[ -d "$BLOBS" ] || { L "FATAL: ollama store $BLOBS not found"; exit 1; }
MMSHA="sha256-$(sha256sum "$MMPROJ" | cut -d' ' -f1)"
L "==================== CODERX ollama republish start ($($OL --version 2>/dev/null|head -1)) ===================="
L "mmproj blob protected: $MMSHA"

gc_blobs(){
  local ref; ref=$(mktemp)
  grep -rhoE "sha256[:-][0-9a-f]{64}" "$MANIFESTS" 2>/dev/null | sed "s/sha256:/sha256-/" | sort -u > "$ref"
  local n=0 b bn
  for b in "$BLOBS"/sha256-*; do
    [ -e "$b" ] || continue; bn=$(basename "$b")
    [ "$bn" = "$MMSHA" ] && continue
    grep -qx "$bn" "$ref" || { rm -f "$b" && n=$((n+1)); }
  done
  rm -f "$ref" "$BLOBS"/sha256-*-partial* 2>/dev/null; L "  gc: purged $n orphan blob(s)"
}
mf_params(){ local T="$1"; cat <<EOF
PARAMETER temperature ${TEMP[$T]}
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER min_p 0.05
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
EOF
}
reg200(){ [ "$(curl -s -o /dev/null -w '%{http_code}' "https://registry.ollama.ai/v2/$REPO/manifests/$1" 2>/dev/null)" = 200 ]; }

TEXT_OK=(); TEXT_FAIL=(); VIS_OK=(); VIS_FAIL=(); LATEST_OK=no
for T in "${TIERS[@]}"; do
  G="$GG/$STEM-$T.gguf"
  [ -f "$G" ] || { L "[$T] MISSING gguf — skip"; TEXT_FAIL+=("$T"); continue; }
  MF="$WORK/Modelfile.$T"; { echo "FROM $G"; mf_params "$T"; } > "$MF"
  # ---- text ----
  ok=0
  for a in 1 2 3; do
    $OL create "$REPO:$T" -f "$MF" >/dev/null 2>&1 && $OL push "$REPO:$T" >/dev/null 2>&1 && reg200 "$T" && { ok=1; break; }
    L "  [$T] text attempt $a failed; retry"; sleep 8
  done
  if [ "$ok" = 1 ]; then L "[$T temp=${TEMP[$T]}] text pushed OK"; TEXT_OK+=("$T"); else L "[$T] text FAILED"; TEXT_FAIL+=("$T"); $OL rm "$REPO:$T" >/dev/null 2>&1; gc_blobs; continue; fi
  # ---- latest (from LATEST_TIER) ----
  if [ "$T" = "$LATEST_TIER" ]; then
    $OL create "$REPO:latest" -f "$MF" >/dev/null 2>&1 && $OL push "$REPO:latest" >/dev/null 2>&1 && reg200 latest \
      && { L "  latest -> $LATEST_TIER pushed OK"; LATEST_OK=yes; } || L "  latest push FAILED"
    $OL rm "$REPO:latest" >/dev/null 2>&1
  fi
  # ---- vision-<tier> (inherit text params + mmproj) ----
  VT="vision-$T"
  if $OL show "$REPO:$T" --modelfile > "$WORK/mf_$T" 2>/dev/null; then
    { cat "$WORK/mf_$T"; echo; echo "FROM $MMPROJ"; } > "$WORK/mfv_$T"
    vcap=0
    for ck in 1 2 3; do
      $OL create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1
      $OL show "$REPO:$VT" 2>/dev/null | grep -qi vision && { vcap=1; break; }
      sleep 3
    done
    if [ "$vcap" = 1 ]; then
      $OL push "$REPO:$VT" >/dev/null 2>&1
      pk=0; for x in 1 2 3; do reg200 "$VT" && { pk=1; break; }; sleep 5; done
      if [ "$pk" = 1 ]; then L "  [$VT] vision pushed OK"; VIS_OK+=("$T"); else L "  [$VT] push FAILED"; VIS_FAIL+=("$T"); fi
    else L "  [$VT] no vision cap after 3 tries — skip"; VIS_FAIL+=("$T"); fi
  else L "  [$VT] show modelfile failed — skip"; VIS_FAIL+=("$T"); fi
  # ---- cleanup this tier ----
  $OL rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1
  gc_blobs
done

L "==================== CODERX ollama republish DONE ===================="
L "text   OK (${#TEXT_OK[@]}/13): ${TEXT_OK[*]:-none}"
L "text   FAIL: ${TEXT_FAIL[*]:-none}"
L "vision OK (${#VIS_OK[@]}/13): ${VIS_OK[*]:-none}"
L "vision FAIL: ${VIS_FAIL[*]:-none}"
L "latest: $LATEST_OK"
echo "CODERX_OLLAMA_REPUBLISH_FIN"
