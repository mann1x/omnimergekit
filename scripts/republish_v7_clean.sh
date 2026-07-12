#!/usr/bin/env bash
# Clean re-publish of the v7-coder cohort GGUF + ollama with CLEAN FILENAMES.
# Root cause this fixes:
#   1. ollama pushes silently failed (bs2 was not authenticated) — now fixed
#      (registered hd5o key installed); pushes work.
#   2. GGUF *filenames* carried the internal codename (g15f2440 / fs2440). The
#      user wants clean published filenames; codename in gguf *metadata* is OK.
# Mechanism: set config _name_or_path to the clean repo id so quantize_gguf's
#   model_name (-> output filename) is clean, reuse the local imatrix.dat (no
#   GPU), rebuild all tiers from F16, HF-upload clean names + ollama push, then
#   delete the old codenamed files from the HF GGUF repo.
# CPU-ONLY (CUDA_VISIBLE_DEVICES="") so it never disturbs the GPU evals.
# Launch:  source ~/.bashrc; setsid nohup bash republish_v7_clean.sh >LOG 2>&1 </dev/null &
set -uo pipefail
GOOGLE=/mnt/sdc/ml/google
OMK=/srv/ml/repos/omnimergekit
PY=/srv/ml/envs/envs/omnimergekit/bin/python
QGGUF=$OMK/scripts/quantize_gguf.py
export CUDA_VISIBLE_DEVICES=""
export HF_XET_HIGH_PERFORMANCE=1
: "${HF_TOKEN:?HF_TOKEN must be exported (source ~/.bashrc before launch)}"

L(){ echo "[republish $(date -u +%H:%M:%S)] $*"; }

# rows:  <local_bf16_basename>  <public_suffix>
ROWS=(
  "gemma-4-A4B-98e-v7-coder-g15f2440-it|v7-coder"
  "gemma-4-A4B-98e-v7-coder-fs2440-it|v7-coderx"
)

republish_one(){
  local BF16BASE="$1" PUB="$2"
  local BF16="$GOOGLE/$BF16BASE"
  local GD="$BF16-GGUF"
  local CLEAN="gemma-4-A4B-98e-${PUB}-it"
  local REPO_IT="ManniX-ITA/${CLEAN}"
  local REPO_GGUF="ManniX-ITA/${CLEAN}-GGUF"
  local OLLAMA="mannix/gemma4-98e-${PUB}"

  L "================= REPUBLISH $PUB  (clean_name=$CLEAN) ================="
  [ -d "$BF16" ] || { L "FATAL: $BF16 missing"; return 1; }
  [ -f "$GD/imatrix.dat" ] || { L "FATAL: $GD/imatrix.dat missing (would force GPU recompute)"; return 1; }

  # 1. clean model_name via _name_or_path (idempotent)
  "$PY" - "$BF16" "$REPO_IT" <<'PY'
import json,sys
d,repo=sys.argv[1],sys.argv[2]; p=d+"/config.json"
c=json.load(open(p)); c["_name_or_path"]=repo; json.dump(c,open(p,"w"),indent=2)
print("  _name_or_path =",repo)
PY

  # 2. rename any codenamed F16 -> clean F16 name (reuse; metadata-codename is fine) [idempotent]
  local oldF16 newF16; newF16="$GD/${CLEAN}-F16.gguf"
  oldF16=$(ls "$GD/"*-F16.gguf 2>/dev/null | grep -v "/${CLEAN}-F16.gguf$" | head -1)
  if [ -n "$oldF16" ]; then mv -v "$oldF16" "$newF16"; fi

  # 3. remove orphan codenamed tier ggufs + sha sidecars (keep clean F16 + imatrix)
  find "$GD" -maxdepth 1 -name "*2440*-*.gguf" -delete -print 2>/dev/null || true
  find "$GD" -maxdepth 1 -name "*2440*.gguf.sha256" -delete -print 2>/dev/null || true

  # 4. quantize_gguf: reuse F16 + imatrix, build CLEAN-named tiers, HF upload + ollama push (+ :latest=Q4_K_M)
  L "[$PUB] quantize_gguf full tier sweep -> $REPO_GGUF + ollama $OLLAMA"
  "$PY" "$QGGUF" --model "$BF16" --output-dir "$GD" \
      --repo "$REPO_GGUF" --base-model-id "$REPO_IT" \
      --ollama-target "$OLLAMA" --ollama-template gemma4-a4b \
      --hf-token "$HF_TOKEN" \
      || { L "FATAL: quantize_gguf failed for $PUB"; return 1; }
  L "[$PUB] clean GGUF + ollama OK"

  # 5. purge codenamed files from the HF GGUF repo (batch commit)
  "$PY" - "$REPO_GGUF" <<'PY'
import sys
from huggingface_hub import HfApi, list_repo_files, CommitOperationDelete
repo=sys.argv[1]; api=HfApi()
dead=[f for f in list_repo_files(repo) if "2440" in f and (f.endswith(".gguf") or f.endswith(".gguf.sha256"))]
if dead:
    api.create_commit(repo_id=repo, repo_type="model",
        operations=[CommitOperationDelete(path_in_repo=f) for f in dead],
        commit_message="purge codenamed GGUF filenames (clean names re-uploaded)")
    print("  purged %d codenamed files from %s:"%(len(dead),repo))
    for f in dead: print("   -",f)
else:
    print("  no codenamed files left on",repo)
PY
  L "================= DONE $PUB ================="
}

L "###### v7 cohort CLEAN re-publish START (CPU-only) ######"
RC=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r base pub <<<"$row"
  if [ -n "${ONLY_MODEL:-}" ] && [ "$pub" != "$ONLY_MODEL" ]; then L "skip $pub (ONLY_MODEL=$ONLY_MODEL)"; continue; fi
  if ! republish_one "$base" "$pub"; then L "###### ABORT at $pub ######"; RC=1; break; fi
done
L "###### v7 cohort CLEAN re-publish END rc=$RC  REPUBLISH_ALL_DONE ######"
exit $RC
