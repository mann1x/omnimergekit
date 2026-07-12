#!/bin/bash
# run_router_kd.sh — T18 Step 3: Router-KD (arXiv:2603.02217) pod chain.
#
# ARMED, NOT auto-launched. Per the T18 ladder Router-KD is the escalation that
# runs ONLY if Step 2 (EAC-MoE, scripts/router_eac_calibrate.py) underperforms.
# Pod-only: needs a 2×(≥80 GiB) host (BF16 teacher ~52 GB on GPU0, BF16 student
# ~25 GB on GPU1) — it deliberately will NOT run on a single 24 GB 3090.
#
# Chain: preflight -> router_kd.py (train + AR-canary gate + write-back to a new
# -rkd-it dir) -> shared-α sanity -> F16/Q6_K convert -> eval_suite_llama 3-bench
# + 5-canary -> archive. Drop-map-free (Router-KD distills output logits).
#
# Recipe is the paper's Table 1 (τ=1.0, lr=5e-5, 1 epoch, bs2/ga4, C4 3000×512);
# router_kd.py hard-codes those as defaults — do not pass overrides without a
# reason recorded in the run log.
#
# LOAD MODE: 2-GPU BF16-both with explicit dict device_map ({"":0} teacher,
# {"":1} student). This bypasses the bnb-NF4 + accelerate-1.13 meta-tensor
# crash (council csl-2026-05-28-1948-1245: accelerate 1.13.0 added a
# `len(module.state_dict()) > 0` check in attach_execution_device_hook
# (hooks.py:459) that calls _save_to_state_dict → quant_state.as_dict() →
# .item() on a meta tensor → RuntimeError for any bnb-Params4bit initialized
# via init_empty_weights, which dispatch_model does by default for the auto
# device_map). BF16-both also removes per-step NF4 dequantization overhead —
# council projects ~30-40 min wall-clock instead of the 3-4 h NF4 estimate
# below. Use only on a 2x≥80GiB host (e.g. linode-blackswan-2: 2× 96GB).
#
# WALL-CLOCK (council csl-…-1d54 CONCERN #2 — pre-BF16-both): the paper's "~2h"
# is for SMALLER MoEs. With NF4 student + bf16 teacher, 3000×512 was 3-4h on
# one A100-80GB. With BF16-both on 2 GPUs the estimate drops to ~30-40 min.
# Chain is checkpoint-gated (--checkpoint-dir + --save-every 100), so an
# overrun is resumable; budget the pod for ≥1.5 h to leave headroom for eval.
#
# Usage (defaults target v5-coder):
#   bash scripts/run_router_kd.sh
#   VARIANT=v6coder VARIANT_HF=ManniX-ITA/gemma-4-A4B-98e-v6-coder-it \
#     bash scripts/run_router_kd.sh
set -uo pipefail

WORKDIR="${WORKDIR:-/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models}"
cd "$WORKDIR"

PY="${PY:-/root/anaconda3/envs/omnimergekit/bin/python}"
LLAMA="${LLAMA:-/opt/llama.cpp}"

# --- targets (override via env) ---
VARIANT="${VARIANT:-v5coder}"
BASE_HF="${BASE_HF:-google/gemma-4-26B-A4B-it}"
VARIANT_HF="${VARIANT_HF:-ManniX-ITA/gemma-4-A4B-98e-v5-coder-it}"
BASE_DIR="${BASE_DIR:-google/gemma-4-26B-A4B-it}"
VARIANT_DIR="${VARIANT_DIR:-google/gemma-4-A4B-98e-v5-coder-it}"
# Domain-matched calibration corpus (council csl-…-1d54): defaults to the
# 9-bench corpus from build_router_calib_corpus.py instead of paper-C4, because
# rumination is on instruction-following, not narrative. Set CORPUS_FILE="" to
# revert to faithful C4. CORPUS_PAD=1 tops it up with C4 to the paper token
# regime (the domain corpus alone is ~100k tok → few KD steps).
CORPUS_FILE="${CORPUS_FILE:-scripts/router_calib_corpus.jsonl}"
CORPUS_PAD="${CORPUS_PAD:-1}"
CORPUS_ARGS=""
[ -n "$CORPUS_FILE" ] && CORPUS_ARGS="--corpus-file $CORPUS_FILE"
[ -n "$CORPUS_FILE" ] && [ "$CORPUS_PAD" = "1" ] && CORPUS_ARGS="$CORPUS_ARGS --corpus-pad-c4"
OUT_DIR="${OUT_DIR:-google/gemma-4-A4B-98e-${VARIANT}-rkd-it}"
GGUF_DIR="${OUT_DIR}-GGUF"
F16_GGUF="${GGUF_DIR}/$(basename "$OUT_DIR")-F16.gguf"
Q6K_GGUF="${GGUF_DIR}/$(basename "$OUT_DIR")-Q6_K.gguf"
CKPT_DIR="${CKPT_DIR:-logs/router_kd_${VARIANT}/ckpt}"
LOGS="logs/router_kd_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS" "$GGUF_DIR" "$CKPT_DIR"

echo "[chain $(date -Iseconds)] === Router-KD chain: $VARIANT ==="
echo "  base:    $BASE_DIR   ($BASE_HF)"
echo "  variant: $VARIANT_DIR ($VARIANT_HF)"
echo "  out:     $OUT_DIR"
echo "  gguf:    $Q6K_GGUF"
echo "  logs:    $LOGS/"

free_gpu() {
    command -v ollama >/dev/null 2>&1 || return 0
    ollama ps 2>/dev/null | awk 'NR>1{print $1}' | while read -r m; do
        [ -n "$m" ] && { echo "[chain] ollama stop $m"; ollama stop "$m" 2>/dev/null || true; }
    done
}

# ─── Phase 0: preflight (fail loud) ──────────────────────────────────────────
echo "[chain $(date -Iseconds)] [0] preflight"
[ -d "$LLAMA" ] || ln -sf /workspace/llama.cpp "$LLAMA" 2>/dev/null || true

# 2-GPU BF16-both check: BOTH GPUs need ≥55 GiB free (teacher 52 + buffer on
# GPU0; student 25 + activation+optimizer state buffer on GPU1). Use the
# WORST of the two so a busy second card fails the check.
FREE_MIB_WORST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
FREE_MIB_BEST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
N_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
FREE_GIB_WORST=$(( ${FREE_MIB_WORST:-0} / 1024 ))
FREE_GIB_BEST=$(( ${FREE_MIB_BEST:-0} / 1024 ))
echo "  free VRAM: best=${FREE_GIB_BEST} GiB worst=${FREE_GIB_WORST} GiB n_gpus=${N_GPUS}"
echo "  need: BF16 teacher 52 GiB on GPU0 + BF16 student 25 GiB on GPU1 = ~77 GiB total"
if [ "${N_GPUS:-0}" -lt 2 ]; then
    echo "[chain] FATAL: <2 GPUs visible — BF16-both router_kd needs a 2×≥80 GiB host."
    echo "        (the 4-bit student path that fit on a single A100 hits accelerate-1.13"
    echo "         dispatch_model meta-tensor crash on bnb Params4bit — council csl-…-1245)."
    exit 1
fi
if [ "${FREE_GIB_BEST:-0}" -lt 55 ]; then
    echo "[chain] FATAL: best GPU has <55 GiB free — BF16 teacher (52 GiB) won't fit."
    exit 1
fi
if [ "${FREE_GIB_WORST:-0}" -lt 28 ]; then
    echo "[chain] FATAL: worst GPU has <28 GiB free — BF16 student (25 GiB) won't fit."
    exit 1
fi

# datasets + accelerate are required. bitsandbytes is OPTIONAL in BF16-both
# mode — router_kd._patch_bnb_for_accelerate() gracefully skips when bnb is
# not importable (see scripts/router_kd.py:~133). We still warn so the operator
# knows a future fallback to 4bit is unavailable without `pip install bnb`.
for mod in datasets accelerate; do
    $PY -c "import $mod" 2>/dev/null && echo "  dep $mod OK" || {
        echo "[chain] FATAL: python module '$mod' missing. pip install $mod"; exit 1; }
done
$PY -c "import bitsandbytes" 2>/dev/null && echo "  dep bitsandbytes OK (unused in BF16-both)" \
    || echo "  dep bitsandbytes MISSING — fine for BF16-both, but 4bit fallback unavailable"

# pull bf16 weights if absent
if [ ! -f "$BASE_DIR/model.safetensors.index.json" ]; then
    echo "  pulling base $BASE_HF -> $BASE_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$BASE_HF" --local-dir "$BASE_DIR" || {
        echo "[chain] FATAL: base download failed"; exit 1; }
fi
if [ ! -f "$VARIANT_DIR/model.safetensors.index.json" ]; then
    echo "  pulling variant $VARIANT_HF -> $VARIANT_DIR"
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$VARIANT_HF" --local-dir "$VARIANT_DIR" || {
        echo "[chain] FATAL: variant download failed"; exit 1; }
fi

free_gpu

# ─── Phase 1: Router-KD train + canary gate + write-back ─────────────────────
if [ -f "$OUT_DIR/model.safetensors.index.json" ]; then
    echo "[chain $(date -Iseconds)] [1] $OUT_DIR exists — skip train"
else
    echo "[chain $(date -Iseconds)] [1] router_kd.py train (faithful paper recipe, BF16-both)"
    # BF16-both 2-GPU split per council csl-2026-05-28-1948-1245:
    #   --teacher-load bf16   default but explicit
    #   --student-load bf16   override the 4bit default — bypasses the
    #                         bnb-NF4 + accelerate-1.13 meta-tensor crash
    #   --teacher-device '{"":0}'  dict device_map -> direct placement,
    #                              no auto-walk -> bypasses dispatch_model
    #   --student-device '{"":1}'  student on the second GPU
    # _dev_arg in router_kd.py JSON-parses '{"":N}' into a real dict so bnb
    # placement contracts hold (see router_kd.py:252-258).
    HF_HUB_ENABLE_HF_TRANSFER=1 $PY scripts/router_kd.py \
        --base-dir "$BASE_DIR" \
        --variant-dir "$VARIANT_DIR" \
        --out-dir "$OUT_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --teacher-load bf16 \
        --student-load bf16 \
        --teacher-device '{"":0}' \
        --student-device '{"":1}' \
        --canary-file scripts/ifeval_rumination_canaries.json \
        --canary-gate \
        $CORPUS_ARGS \
        2>&1 | tee "$LOGS/router_kd_train.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 2 ]; then
        echo "[chain] HALT: canary gate FAILED (rumination worsened). Checkpoint kept in $CKPT_DIR."
        echo "        Inspect, then rerun manually with --no-canary to force-save if desired."
        exit 2
    fi
    [ "$rc" -eq 0 ] || { echo "[chain] FATAL: training rc=$rc"; exit 1; }
fi

# ─── Phase 2: shared-α sanity (carried over by write-back, NOT reapplied) ─────
# Router-KD only overwrites router.* tensors; mlp.down_proj (where shared α=1.2
# lives) is copied untouched from the source. So we VERIFY, never reapply.
echo "[chain $(date -Iseconds)] [2] shared-α marker sanity"
if [ -f "$VARIANT_DIR/.shared_applied" ] && [ ! -f "$OUT_DIR/.shared_applied" ]; then
    echo "  WARN: source had .shared_applied but out-dir lost it — copying marker"
    cp -p "$VARIANT_DIR/.shared_applied" "$OUT_DIR/.shared_applied" 2>/dev/null || true
fi
[ -f "$OUT_DIR/.shared_applied" ] && echo "  shared α=1.2 carried over (marker present)" \
    || echo "  NOTE: no .shared_applied marker on source — verify the variant's recipe"

# ─── Phase 3: F16 GGUF convert ───────────────────────────────────────────────
if [ -f "$F16_GGUF" ] || [ -f "$Q6K_GGUF" ]; then
    echo "[chain $(date -Iseconds)] [3] F16/Q6_K present — skip convert"
else
    echo "[chain $(date -Iseconds)] [3] convert_hf_to_gguf --outtype f16"
    $PY "$LLAMA/convert_hf_to_gguf.py" "$OUT_DIR" --outfile "$F16_GGUF" --outtype f16 \
        2>&1 | tee "$LOGS/convert_f16.log" | tail -10
fi

# ─── Phase 4: Q6_K quantize ──────────────────────────────────────────────────
if [ -f "$Q6K_GGUF" ]; then
    echo "[chain $(date -Iseconds)] [4] Q6_K present — skip"
else
    echo "[chain $(date -Iseconds)] [4] llama-quantize -> Q6_K"
    "$LLAMA/build/bin/llama-quantize" "$F16_GGUF" "$Q6K_GGUF" Q6_K \
        2>&1 | tee "$LOGS/quant_q6k.log" | tail -10
    [ -f "$F16_GGUF" ] && [ -f "$Q6K_GGUF" ] && { echo "  removing F16 (free ~50G)"; rm -f "$F16_GGUF"; }
fi

# ─── Phase 5: eval — 3-bench + 5-canary vs v5-coder baseline ─────────────────
free_gpu
echo "[chain $(date -Iseconds)] [5] eval_suite_llama 3-bench (no rumination halt)"
echo "  anchors: v5-coder Q6_K  HE+ 93.29 / ifeval 94 / LCB-55-v4 85.45"
bash scripts/eval_suite_llama.sh \
    --variant "${VARIANT}_rkd" \
    --gguf "$Q6K_GGUF" \
    --only "ifeval_100,humanevalplus_full,lcb_medium_55_v4" \
    2>&1 | tee "$LOGS/eval_suite.log"
ev=${PIPESTATUS[0]}

echo "[chain $(date -Iseconds)] === done (eval rc=$ev) ==="
echo "  router checkpoint (reproducibility artifact): $CKPT_DIR"
echo "  REMINDER: archive $CKPT_DIR + eval samples BEFORE destroying the pod (sacred-results)."
exit "$ev"
