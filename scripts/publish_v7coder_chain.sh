#!/usr/bin/env bash
# Publish chain for the v7-coder cohort (BOTH models), public, full chain.
#   v7-coder  = g15f2440 (fs2440 recipe + targeted_gpqa 1.5x, floor [24,40]) — science-augmented coder
#   v7-coderx = fs2440   (generic_code 3x + targeted_lcb 2x, floor [24,40])   — code-maximal sibling
#
# Per model, sequentially (GPU0 for imatrix + NVFP4A16):
#   1. preflight (bf16 dir, .shared_applied, disk)
#   2. synth preprocessor_config.json (Gemma 4 vLLM requirement) if missing
#   3. bf16 safetensors  -> HF  ManniX-ITA/gemma-4-A4B-98e-<pub>-it          (public)
#   4. GGUF chain via quantize_gguf.py (imatrix + 29 default tiers + F16)
#        -> HF  ...-it-GGUF (public)  + ollama mannix/gemma4-98e-<pub> (public)
#        imatrix.dat uploaded to GGUF repo (mandatory in quantize_gguf.py:2185)
#   5. NVFP4A16 via quantize_any.py (modelopt env) -> HF ...-NVFP4A16 (public)
#   6. verify: imatrix.dat in -GGUF dir, NVFP4A16 dir has weights, markers
#
# Launch:  source ~/.bashrc; setsid nohup bash publish_v7coder_chain.sh >LOG 2>&1 </dev/null &
set -uo pipefail

GOOGLE=/mnt/sdc/ml/google
OMK=/srv/ml/repos/omnimergekit
PY_OMK=/srv/ml/envs/envs/omnimergekit/bin/python
PY_MODELOPT=/srv/ml/envs/envs/modelopt/bin/python
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
QGGUF=$OMK/scripts/quantize_gguf.py
QANY=$OMK/scripts/quantize_any.py
GPU=0
export CUDA_VISIBLE_DEVICES=$GPU
export HF_XET_HIGH_PERFORMANCE=1

: "${HF_TOKEN:?HF_TOKEN must be exported (source ~/.bashrc before launch)}"

L(){ echo "[publish $(date -u +%H:%M:%S)] $*"; }

# config rows:  <local_bf16_basename>  <public_suffix>
ROWS=(
  "gemma-4-A4B-98e-v7-coder-g15f2440-it|v7-coder"
  "gemma-4-A4B-98e-v7-coder-fs2440-it|v7-coderx"
)

synth_preproc(){
  local d="$1"
  [ -f "$d/preprocessor_config.json" ] && return 0
  L "synth preprocessor_config.json in $d"
  "$PY_OMK" - "$d" <<'PY'
import json, sys, os
d = sys.argv[1]
pc_path = os.path.join(d, "processor_config.json")
out = os.path.join(d, "preprocessor_config.json")
fe = {}
if os.path.exists(pc_path):
    pc = json.load(open(pc_path))
    fe = pc.get("feature_extractor") or pc.get("image_processor") or {}
if not fe:
    # Gemma 4 SigLIP-style default
    fe = {
        "image_processor_type": "Gemma4ImageProcessor",
        "processor_class": "Gemma4Processor",
        "do_convert_rgb": True, "do_normalize": True, "do_rescale": True,
        "do_resize": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "image_seq_length": 256, "resample": 2, "rescale_factor": 0.00392156862745098,
        "size": {"height": 896, "width": 896},
    }
json.dump(fe, open(out, "w"), indent=2)
print("wrote", out)
PY
}

publish_one(){
  local BF16BASE="$1" PUB="$2"
  local BF16="$GOOGLE/$BF16BASE"
  local GGUFDIR="$BF16-GGUF"
  local NVFP4="$GOOGLE/gemma-4-A4B-98e-${PUB}-NVFP4A16"
  local REPO_IT="ManniX-ITA/gemma-4-A4B-98e-${PUB}-it"
  local REPO_GGUF="ManniX-ITA/gemma-4-A4B-98e-${PUB}-it-GGUF"
  local REPO_NVFP4="ManniX-ITA/gemma-4-A4B-98e-${PUB}-NVFP4A16"
  local OLLAMA="mannix/gemma4-98e-${PUB}"
  local MARK="$GOOGLE/PUBLISH_${PUB}_DONE"

  L "================= PUBLISH $PUB  (from $BF16BASE) ================="
  # 1. preflight
  [ -d "$BF16" ] || { L "FATAL: $BF16 missing"; return 1; }
  [ -f "$BF16/.shared_applied" ] || { L "FATAL: $BF16 missing .shared_applied marker"; return 1; }
  [ -f "$BF16/README.md" ] || { L "FATAL: $BF16/README.md (model card) not staged"; return 1; }
  local avail; avail=$(df -BG --output=avail "$GOOGLE" | tail -1 | tr -dc 0-9)
  L "disk avail: ${avail}G"
  [ "${avail:-0}" -ge 120 ] || { L "FATAL: <120G free"; return 1; }

  # 2. synth preprocessor_config.json
  synth_preproc "$BF16"

  # 3. bf16 -> HF -it (public)
  L "[3/5] upload bf16 -> $REPO_IT"
  "$HF" upload "$REPO_IT" "$BF16" . \
      --exclude ".shared_applied" --exclude "*.pre_shared" --exclude "*.gguf" --exclude "*.pt" \
      --commit-message "v7 cohort: ${PUB} bf16 (98e prune + shared a=1.2)" \
      || { L "FATAL: bf16 upload failed for $PUB"; return 1; }
  L "[3/5] bf16 upload OK"

  # 4. GGUF chain (imatrix + tiers + ollama). output-dir pinned to existing -GGUF dir.
  L "[4/5] GGUF chain -> $REPO_GGUF + ollama $OLLAMA"
  "$PY_OMK" "$QGGUF" \
      --model "$BF16" \
      --output-dir "$GGUFDIR" \
      --repo "$REPO_GGUF" \
      --base-model-id "$REPO_IT" \
      --ollama-target "$OLLAMA" \
      --ollama-template gemma4-a4b \
      --hf-token "$HF_TOKEN" \
      || { L "FATAL: GGUF chain failed for $PUB"; return 1; }
  L "[4/5] GGUF chain OK"
  # imatrix preservation gate
  if [ -f "$GGUFDIR/imatrix.dat" ]; then
      L "imatrix.dat preserved: $(ls -la "$GGUFDIR/imatrix.dat" | awk '{print $5}') bytes"
  else
      L "WARN: imatrix.dat NOT in $GGUFDIR after chain (check it reached $REPO_GGUF)"
  fi

  # 5. NVFP4A16 (modelopt env, GPU) — NON-FATAL: a modelopt plugin gap must not
  #    block the reliable bf16+GGUF+ollama publish. On failure, defer via marker.
  L "[5/5] NVFP4A16 quant -> $NVFP4"
  if [ "${SKIP_NVFP4:-0}" = 1 ]; then
      L "[5/5] NVFP4A16 SKIPPED (SKIP_NVFP4=1) — deferred"; touch "$GOOGLE/NEEDS_NVFP4A16_${PUB}"
  elif "$PY_MODELOPT" "$QANY" --src "$BF16" --dst "$NVFP4" --method nvfp4a16; then
      synth_preproc "$NVFP4"
      local nshard; nshard=$(ls "$NVFP4"/*.safetensors 2>/dev/null | wc -l)
      if [ "$nshard" -ge 1 ] && "$HF" upload "$REPO_NVFP4" "$NVFP4" . \
            --exclude ".shared_applied" --exclude "*.pre_shared" \
            --commit-message "v7 cohort: ${PUB} NVFP4A16 (modelopt)"; then
          L "[5/5] NVFP4A16 upload OK"; rm -f "$GOOGLE/NEEDS_NVFP4A16_${PUB}"
      else
          L "WARN: NVFP4A16 upload failed/empty for $PUB — DEFERRED"; touch "$GOOGLE/NEEDS_NVFP4A16_${PUB}"
      fi
  else
      L "WARN: NVFP4A16 quant FAILED for $PUB (modelopt fused-experts plugin) — DEFERRED"; touch "$GOOGLE/NEEDS_NVFP4A16_${PUB}"
  fi

  touch "$MARK"
  L "================= DONE $PUB  (marker $MARK) ================="
  return 0
}

L "###### v7-coder cohort publish chain START (public, full) ######"
RC=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r base pub <<<"$row"
  if [ -n "${ONLY_MODEL:-}" ] && [ "$pub" != "$ONLY_MODEL" ]; then L "skip $pub (ONLY_MODEL=$ONLY_MODEL)"; continue; fi
  if ! publish_one "$base" "$pub"; then
    L "###### ABORT: $pub failed; not proceeding to remaining models ######"
    RC=1
    break
  fi
done
L "###### v7-coder cohort publish chain END rc=$RC ######"
exit $RC
