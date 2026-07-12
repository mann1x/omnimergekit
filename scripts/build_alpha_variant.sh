#!/bin/bash
# build_alpha_variant.sh — T172 atomic per-variant builder.
#
# Build a single α-variant: cp -a pristine → out, optionally apply
# router_shared_upweight + router_per_expert_rescale per --shared-alpha and
# --pes-alpha, then convert F16 GGUF and quantize Q6_K with A2 imatrix.
#
# HARDLINK HAZARD (task T172.h): router_shared_upweight.py and PES use
# safetensors.save_file which writes through hardlinks. We use `cp -a` (full
# copy, not `cp -al`) so pristine is never corrupted. ~62 GB per variant.
#
# Idempotent on `.audit_ready` marker (set after Q6_K + imatrix landed).
#
# Usage:
#   build_alpha_variant.sh \
#       --src /mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it \
#       --shared-alpha 1.10 --pes-alpha 1.00 --order shared-first \
#       --out  /mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-shared110-it
#
# Author: claude opus 4.7  2026-05-29
set -uo pipefail

BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
OMK_REPO=$BM/repos/omnimergekit
RECIPE_DIR=$OMK_REPO/recipes/gemma4/v5_moe_sweep
SCR=$OMK_REPO/scripts

SHARED_SCRIPT=$RECIPE_DIR/router_shared_upweight.py
PES_SCRIPT=$BM/scripts/router_per_expert_rescale.py
[ -f "$PES_SCRIPT" ] || PES_SCRIPT=$RECIPE_DIR/router_per_expert_rescale.py

CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
[ -f "$CONVERT" ] || CONVERT=/workspace/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
[ -x "$QUANT" ]   || QUANT=/workspace/llama.cpp/build/bin/llama-quantize

A2_GGUF_DIR=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF
A2_IMATRIX=$A2_GGUF_DIR/imatrix.dat

# --- parse args ---
SRC=""; OUT=""; SHARED_A="1.00"; PES_A="1.00"; ORDER="shared-first"
while [ $# -gt 0 ]; do
    case "$1" in
        --src)          SRC=$2; shift 2 ;;
        --out)          OUT=$2; shift 2 ;;
        --shared-alpha) SHARED_A=$2; shift 2 ;;
        --pes-alpha)    PES_A=$2; shift 2 ;;
        --order)        ORDER=$2; shift 2 ;;
        *)              echo "FATAL: unknown arg $1"; exit 2 ;;
    esac
done

[ -n "$SRC" ] && [ -n "$OUT" ] || { echo "FATAL: --src and --out required"; exit 2; }
[ -f "$SRC/.pristine_built" ] || { echo "FATAL: $SRC missing .pristine_built marker"; exit 2; }
[ -f "$A2_IMATRIX" ] || { echo "FATAL: A2 imatrix missing at $A2_IMATRIX"; exit 2; }
case "$ORDER" in shared-first|pes-first) ;; *) echo "FATAL: --order must be shared-first|pes-first"; exit 2 ;; esac

LOG_DIR=$BM/logs/t172
mkdir -p "$LOG_DIR"
NAME=$(basename "$OUT")
LOG=$LOG_DIR/build_${NAME}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1

GGUF_DIR="${OUT}-GGUF"
F16="$GGUF_DIR/${NAME}-F16.gguf"
Q6="$GGUF_DIR/${NAME}-Q6_K.gguf"

echo "[$(date -Iseconds)] === build_alpha_variant ==="
echo "  src       : $SRC"
echo "  out       : $OUT"
echo "  shared_α  : $SHARED_A"
echo "  pes_α     : $PES_A"
echo "  order     : $ORDER"
echo "  gguf dir  : $GGUF_DIR"
echo "  imatrix   : $A2_IMATRIX (hardlinked)"
echo

# Idempotence
if [ -f "$OUT/.audit_ready" ] && [ -f "$Q6" ]; then
    echo "[$(date -Iseconds)] SKIP — variant already built and Q6 exists"
    cat "$OUT/.audit_ready"
    exit 0
fi

# Disk preflight: need ~62 GB bf16 + ~31 GB F16 transient + ~21 GB Q6
avail_kib=$(df -k "$(dirname "$OUT")" | awk 'NR==2 {print $4}')
avail_gib=$((avail_kib / 1024 / 1024))
echo "  disk free: ${avail_gib} GiB (need ~115)"
if [ "$avail_gib" -lt 120 ]; then
    echo "FATAL: insufficient disk (${avail_gib} GiB free, need ≥120)"
    exit 3
fi

# --- step 1: cp -a pristine -> out (FULL copy, no hardlinks) ---
if [ ! -f "$OUT/model.safetensors.index.json" ]; then
    echo "[$(date -Iseconds)] step 1: cp -a $SRC -> $OUT"
    cp -a "$SRC" "$OUT"
    rm -f "$OUT/.pristine_built"  # this is a copy, not pristine itself
else
    echo "[$(date -Iseconds)] step 1 SKIP — out already populated"
fi

# --- step 2 + 3: apply α in declared order ---
apply_shared() {
    if [ "$SHARED_A" = "1.00" ] || [ "$SHARED_A" = "1.0" ]; then
        echo "  shared_α=1.00 — SKIP"
        return 0
    fi
    if [ -f "$OUT/.shared_applied" ]; then
        echo "  shared_α=$SHARED_A already applied — SKIP"
        return 0
    fi
    echo "  apply shared_α=$SHARED_A"
    "$PY" "$SHARED_SCRIPT" \
        --model-dir "$OUT" \
        --target mlp.down_proj.weight \
        --alpha "$SHARED_A" 2>&1 | tail -8
    [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: shared upweight failed"; exit 4; }
    echo "shared_alpha=$SHARED_A target=mlp.down_proj.weight ts=$(date -Iseconds)" > "$OUT/.shared_applied"
}

apply_pes() {
    if [ "$PES_A" = "1.00" ] || [ "$PES_A" = "1.0" ]; then
        echo "  pes_α=1.00 — SKIP"
        return 0
    fi
    if [ -f "$OUT/.pes_applied" ]; then
        echo "  pes_α=$PES_A already applied — SKIP"
        return 0
    fi
    echo "  apply pes_α=$PES_A"
    "$PY" "$PES_SCRIPT" \
        --model-dir "$OUT" \
        --alpha "$PES_A" 2>&1 | tail -8
    [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: PES rescale failed"; exit 5; }
    echo "pes_alpha=$PES_A ts=$(date -Iseconds)" > "$OUT/.pes_applied"
}

echo "[$(date -Iseconds)] step 2+3: apply α recipe (order=$ORDER)"
if [ "$ORDER" = "shared-first" ]; then
    apply_shared
    apply_pes
else
    apply_pes
    apply_shared
fi

# --- step 4: alpha_recipe marker ---
cat > "$OUT/.alpha_recipe" <<EOF
{
  "shared_alpha": $SHARED_A,
  "pes_alpha":    $PES_A,
  "order":        "$ORDER",
  "src":          "$SRC",
  "built_ts":     "$(date -Iseconds)"
}
EOF

# --- step 5: convert HF -> F16 GGUF ---
mkdir -p "$GGUF_DIR"
if [ ! -f "$F16" ]; then
    echo "[$(date -Iseconds)] step 5: convert HF -> F16 GGUF"
    "$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -10
    [ -f "$F16" ] || { echo "FATAL: F16 GGUF not created"; exit 6; }
    echo "  F16 created: $(du -sh "$F16" | cut -f1)"
else
    echo "[$(date -Iseconds)] step 5 SKIP — F16 already exists"
fi

# --- step 6: quantize F16 -> Q6_K with A2 imatrix ---
if [ ! -f "$Q6" ]; then
    echo "[$(date -Iseconds)] step 6: quantize F16 -> Q6_K (A2 imatrix)"
    "$QUANT" --imatrix "$A2_IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -10
    [ -f "$Q6" ] || { echo "FATAL: Q6_K not created"; exit 7; }
    echo "  Q6_K created: $(du -sh "$Q6" | cut -f1)"
else
    echo "[$(date -Iseconds)] step 6 SKIP — Q6_K already exists"
fi

# --- step 7: hardlink imatrix + seed tokenizer/config into -GGUF dir ---
if [ ! -e "$GGUF_DIR/imatrix.dat" ]; then
    ln "$A2_IMATRIX" "$GGUF_DIR/imatrix.dat" 2>/dev/null || cp "$A2_IMATRIX" "$GGUF_DIR/imatrix.dat"
    echo "  imatrix.dat linked (hardlink → A2 source)"
fi
for f in tokenizer.json tokenizer_config.json chat_template.jinja \
         generation_config.json preprocessor_config.json processor_config.json \
         special_tokens_map.json config.json; do
    if [ -s "$OUT/$f" ] && [ ! -s "$GGUF_DIR/$f" ]; then
        cp "$OUT/$f" "$GGUF_DIR/$f"
    fi
done

# --- step 8: cleanup F16 (Q6 is what we keep) ---
if [ -f "$Q6" ] && [ -f "$F16" ]; then
    rm -f "$F16"
    echo "  removed F16 (Q6_K preserved)"
fi

# --- audit-ready marker ---
cat > "$OUT/.audit_ready" <<EOF
{
  "ts":           "$(date -Iseconds)",
  "name":         "$NAME",
  "q6":           "$Q6",
  "gguf_dir":     "$GGUF_DIR",
  "shared_alpha": $SHARED_A,
  "pes_alpha":    $PES_A,
  "order":        "$ORDER",
  "src":          "$SRC"
}
EOF

echo
echo "[$(date -Iseconds)] === variant build DONE ==="
echo "  bf16: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "  q6:   $Q6   ($(du -sh "$Q6" | cut -f1))"
echo "  audit-ready marker: $OUT/.audit_ready"
