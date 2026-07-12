#!/usr/bin/env bash
set -uo pipefail
OL=/usr/local/bin/ollama; REPO=mannix/gemma4-98e-v7-coder; T=Q5_K_M; VT=vision-Q5_K_M
MMPROJ=/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf; W=/mnt/sdc/ml/std16_gate/ollama_republish
exec >>"$W/retry_vision.log" 2>&1; L(){ echo "[$(date -u +%T)] $*"; }
L "pull $REPO:$T"; $OL pull "$REPO:$T" >/dev/null 2>&1 || { L "pull FAIL"; exit 1; }
$OL show "$REPO:$T" --modelfile > "$W/mf_$T" 2>/dev/null || { L "show FAIL"; exit 1; }
{ cat "$W/mf_$T"; echo; echo "FROM $MMPROJ"; } > "$W/mfv_$T"
vc=0; for c in 1 2 3; do $OL create "$REPO:$VT" -f "$W/mfv_$T" >/dev/null 2>&1; $OL show "$REPO:$VT" 2>/dev/null | grep -qi vision && { vc=1; break; }; sleep 3; done
[ "$vc" = 1 ] || { L "no vision cap"; exit 1; }
$OL push "$REPO:$VT" >/dev/null 2>&1
for p in 1 2 3; do [ "$(curl -s -o /dev/null -w "%{http_code}" "https://registry.ollama.ai/v2/$REPO/manifests/$VT")" = 200 ] && { L "$VT pushed OK (200)"; $OL rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1; echo VISION_Q5KM_DONE; exit 0; }; sleep 5; done
L "$VT push FAIL"
