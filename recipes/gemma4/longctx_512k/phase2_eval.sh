#!/bin/bash
# phase2_eval.sh — post-train eval on blackswan-2.
#
# Per target T:
#   1. Merge final LoRA adapter into base bf16 → <T>-512k-it/
#   2. (auto by patch_yarn_config.py at merge time) ensure config.json has
#      yarn factor=2.0 + max_position_embeddings=524288
#   3. Canonical 9-bench at 32k context (no-quality-loss gate)
#   4. RULER NIAH single-needle at 32k, 64k, 128k, 256k, 512k
#   5. MRCR-v2 8-needle at 256k
#   6. (v6c only) routing_entropy_probe.py — capture per-layer entropy + dominance
#   7. Write VERDICT file based on hard gates: pass | caveat | hard-fail
#
# ### COUNCIL — read brief §5 (gate calibration). The verdict logic at the
# bottom of this script encodes the hard-gate thresholds from plan v2.

set -uo pipefail

BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK_EVAL=/srv/ml/repos/omnimergekit/eval/omk_eval.py
ROUTING_PROBE=$BM/scripts/routing_entropy_probe.py

TARGET=""
DO_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2;;
    --run)    DO_RUN=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

case "$TARGET" in
  31b) RUN_DIR=$BM/runs/longctx_512k/31b
       OUT_DIR=$BM/eval_results_longctx_512k/31b
       MERGED=$BM/models/variants/gemma-4-31B-it-512k
       ANCHORS=$BM/eval_results_longctx_512k_anchors/31b   # rsync'd from solidpc
       IS_MOE=0
       ;;
  v6c) RUN_DIR=$BM/runs/longctx_512k/v6c
       OUT_DIR=$BM/eval_results_longctx_512k/v6c
       MERGED=$BM/models/variants/gemma-4-A4B-98e-v6-coder-it-512k
       ANCHORS=$BM/eval_results_longctx_512k_anchors/v6c
       IS_MOE=1
       ;;
  *) echo "FATAL: --target must be 31b|v6c"; exit 2;;
esac

mkdir -p "$OUT_DIR" "$MERGED"

echo "=== Phase 2 eval: $TARGET ==="
echo "  run_dir   : $RUN_DIR"
echo "  merged    : $MERGED"
echo "  out_dir   : $OUT_DIR"
echo "  anchors   : $ANCHORS"

if [ "$DO_RUN" -ne 1 ]; then
  echo "[dry-run] would: merge → 9-bench@32k → NIAH@{32k…512k} → MRCR@256k → routing-entropy@v6c → VERDICT"
  exit 0
fi

# ---------- 1. Merge LoRA ----------
LAST_CKPT=$(ls -d "$RUN_DIR/ckpts/step-"* 2>/dev/null | sort | tail -1)
if [ -z "$LAST_CKPT" ]; then
  echo "FATAL: no ckpts found under $RUN_DIR/ckpts/"
  exit 3
fi
echo "[1/7] merging LoRA $LAST_CKPT into $MERGED"
# TODO COUNCIL: implement merge step (peft.PeftModel.merge_and_unload + save).
# Sketch:
#   python -c "
#     from transformers import AutoModelForCausalLM
#     from peft import PeftModel
#     base = AutoModelForCausalLM.from_pretrained('$RUN_DIR/yarn_patched_base', dtype='bfloat16')
#     model = PeftModel.from_pretrained(base, '$LAST_CKPT').merge_and_unload()
#     model.save_pretrained('$MERGED', max_shard_size='10GB')
#   "

# ---------- 2. (handled by step 1 — merged model dir inherits yarn config) ----------

# ---------- 3. 9-bench at 32k ----------
echo "[3/7] 9-bench at 32k (no-quality-loss gate)"
# TODO: bash eval_suite_vllm.sh --model $MERGED --output-root $OUT_DIR/9bench_32k --ctx 32768

# ---------- 4. RULER NIAH single-needle at {32k, 64k, 128k, 256k, 512k} ----------
for L in 32k 64k 128k 256k 512k; do
  case "$L" in 32k) CTX=32768;; 64k) CTX=65536;; 128k) CTX=131072;; 256k) CTX=262144;; 512k) CTX=524288;; esac
  echo "[4/7] RULER NIAH @ $L (ctx=$CTX)"
  # KV memory: 31B @ 512k = 80GB bf16. Use fp8 KV.
  KV_DTYPE=""
  if [ "$TARGET" = "31b" ] && { [ "$CTX" -ge 262144 ]; }; then
    KV_DTYPE="--kv-cache-dtype fp8"
  fi
  # TODO: $PY $OMK_EVAL --template ruler_niah_single_$L --model $MERGED \
  #         --backend vllm --ctx $CTX --results-dir $OUT_DIR/niah/$L $KV_DTYPE
done

# ---------- 5. MRCR-v2 8-needle @ 256k ----------
echo "[5/7] MRCR-v2 8-needle @ 256k"
# TODO: $PY $OMK_EVAL --template mrcr_v2_8needle_256k --model $MERGED \
#         --backend vllm --ctx 262144 --results-dir $OUT_DIR/mrcr_256k

# ---------- 6. Routing entropy (v6-coder only) ----------
if [ "$IS_MOE" -eq 1 ]; then
  echo "[6/7] routing entropy probe (v6-coder MoE)"
  # TODO: $PY $ROUTING_PROBE --model $MERGED \
  #         --positions 32k,256k,512k --n-docs 100 \
  #         --out $OUT_DIR/routing_entropy.json
fi

# ---------- 7. Verdict ----------
echo "[7/7] computing verdict from gates"
# Verdict logic (plan v2 §"Decision matrix"):
#   pass    : all hard gates clean
#   caveat  : hard gates clean BUT long-ctx goal missed (NIAH@512k <85% but ≥80%, or NIAH@256k <90% but ≥85%)
#   hard-fail: ANY hard-gate breach
# TODO: $PY -c "compute verdict from $OUT_DIR/* + $ANCHORS/* — write $OUT_DIR/VERDICT"

echo
echo "=== Phase 2 complete ==="
echo "  verdict : $(cat $OUT_DIR/VERDICT 2>/dev/null || echo 'NOT YET COMPUTED')"
echo "  results : $OUT_DIR/"
