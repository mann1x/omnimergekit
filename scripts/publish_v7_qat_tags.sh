#!/usr/bin/env bash
# publish_v7_qat_tags.sh — ollama tag work for the v7 QAT-Q4_0 tier:
#   1) create+push :qat        (qat GGUF retagged; drops the -Q4_0 suffix)
#   2) create+push :vision-qat (qat GGUF + shared SigLIP mmproj; vision-cap verified)
# for BOTH v7-coder + v7-coderx. Verifies via registry manifests HTTP 200.
# NOTE: cannot delete the old :qat-Q4_0 (ollama 0.30.4 has no remote rm) — flagged for web.
set -uo pipefail
MMPROJ=/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf
WORK=/mnt/sdc/ml/gguf/v7_qat_vision_work; mkdir -p "$WORK"
LOG=/srv/ml/logs/publish_v7_qat_tags_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[qattag $(date -u +%H:%M:%S)] $*"; }
[ -f "$MMPROJ" ] || { L "FATAL no mmproj $MMPROJ"; exit 1; }
man200(){ [ "$(curl -s -o /dev/null -w "%{http_code}" "https://registry.ollama.ai/v2/$1/manifests/$2")" = 200 ]; }

ROWS=(
"mannix/gemma4-98e-v7-coder|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/_ollama_push"
"mannix/gemma4-98e-v7-coderx|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/_ollama_push"
)
for r in "${ROWS[@]}"; do
  IFS="|" read -r repo od <<<"$r"
  mf="$od/Modelfile.qat-Q4_0"
  [ -f "$mf" ] || { L "FATAL missing $mf"; exit 1; }

  # ---- :qat ----
  if man200 "$repo" qat; then L "$repo:qat already on registry — skip"; else
    L ">>> create+push $repo:qat"
    ollama create "$repo:qat" -f "$mf" || { L "FATAL create $repo:qat"; exit 1; }
    ollama push "$repo:qat"           || { L "FATAL push $repo:qat";   exit 1; }
  fi
  ok=0; for k in 1 2 3; do man200 "$repo" qat && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] && L "  [ok] $repo:qat (registry 200)" || { L "FATAL $repo:qat not on registry"; exit 1; }

  # ---- :vision-qat (qat text Modelfile + FROM mmproj) ----
  vmf="$od/Modelfile.vision-qat"
  { cat "$mf"; echo; echo "FROM $MMPROJ"; } > "$vmf"
  if man200 "$repo" vision-qat; then L "$repo:vision-qat already on registry — skip"; else
    L ">>> create $repo:vision-qat"
    ollama create "$repo:vision-qat" -f "$vmf" || { L "FATAL create $repo:vision-qat"; exit 1; }
    vok=0; for ck in 1 2 3; do
      ollama show "$repo:vision-qat" 2>/dev/null | grep -qi vision && { vok=1; break; }
      L "  vision cap not visible (check $ck/3) — recreate+retry"; ollama create "$repo:vision-qat" -f "$vmf" >/dev/null 2>&1; sleep 3
    done
    [ "$vok" = 1 ] || { L "FATAL $repo:vision-qat no vision cap after 3 tries"; exit 1; }
    L ">>> push $repo:vision-qat"
    ollama push "$repo:vision-qat" || { L "FATAL push $repo:vision-qat"; exit 1; }
  fi
  ok=0; for k in 1 2 3; do man200 "$repo" vision-qat && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] && L "  [ok] $repo:vision-qat (registry 200)" || { L "FATAL $repo:vision-qat not on registry"; exit 1; }

  ollama rm "$repo:qat" "$repo:vision-qat" >/dev/null 2>&1
done
L "===== verify final tag set ====="
for repo in mannix/gemma4-98e-v7-coder mannix/gemma4-98e-v7-coderx; do
  for t in qat vision-qat qat-Q4_0; do
    printf "  %s:%s -> %s\n" "$repo" "$t" "$(curl -s -o /dev/null -w "%{http_code}" "https://registry.ollama.ai/v2/$repo/manifests/$t")"
  done
done
L "###### V7_QAT_TAGS_DONE $(date -u) ######"
