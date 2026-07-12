#!/bin/bash
# build_pristine_62e.sh — T172 Phase 0
# Rebuild the pristine 62e fc15_25-p8 baseline with NO shared upweight and
# NO PES applied — the factorial origin for the T172 α-sweep.
#
# NOT the same as the pod's B1 (`fc15_25-p8-it` with shared α=1.2 baked in
# per recreate_62e_on_pod.sh:43-50) or A2 (`fc15_25-p8-s1_0p1_20-it` whose
# α-baking is opaque). Pristine gets its own `-pristine-it` suffix.
#
# Idempotent on `.pristine_built` marker.
# Designed to run ON bs2 (linode-blackswan-2). Outputs to /mnt/sdc/ml (second
# disk), keeps logs on /srv/ml (sacred retention).
#
# Author: claude opus 4.7  2026-05-29
set -uo pipefail

BM=/srv/ml                                          # eval results + logs (sacred)
BM_WORKING=${BM_WORKING:-/mnt/sdc/ml}               # bf16 + GGUF builds (scratch)
PY=$BM/envs/envs/omnimergekit/bin/python
SCR=$BM/repos/omnimergekit/scripts

BASE=$BM/google/gemma-4-26B-A4B-it
DROP_MAP=$SCR/v6coder_C6v3lcb_62e_fc15_25_p8_drop_map.json
OUT=$BM_WORKING/google/gemma-4-A4B-62e-fc15_25-p8-pristine-it

LOG_DIR=$BM/logs/t172
LOG=$LOG_DIR/build_pristine_62e_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$LOG_DIR" "$BM_WORKING/google"
exec > >(tee "$LOG") 2>&1

echo "[$(date -Iseconds)] === T172 Phase 0: build pristine 62e ==="
echo "  base      : $BASE"
echo "  drop map  : $DROP_MAP"
echo "  out       : $OUT (BM_WORKING=$BM_WORKING)"
echo "  log       : $LOG"
echo

# Idempotence
if [ -f "$OUT/.pristine_built" ]; then
    echo "[$(date -Iseconds)] SKIP — $OUT/.pristine_built already present"
    cat "$OUT/.pristine_built"
    exit 0
fi

# Preconditions
[ -f "$BASE/config.json" ] || { echo "FATAL: base 128e missing at $BASE"; exit 1; }
[ -f "$DROP_MAP" ]         || { echo "FATAL: drop map missing at $DROP_MAP"; exit 1; }
[ -d "$BM_WORKING" ]       || { echo "FATAL: BM_WORKING=$BM_WORKING missing"; exit 1; }

# Disk preflight: need ~62 GB
avail_kib=$(df -k "$BM_WORKING" | awk 'NR==2 {print $4}')
avail_gib=$((avail_kib / 1024 / 1024))
echo "  /mnt/sdc free: ${avail_gib} GiB (need ~62)"
if [ "$avail_gib" -lt 70 ]; then
    echo "FATAL: insufficient disk on $BM_WORKING (${avail_gib} GiB free, need ≥70)"
    exit 1
fi

echo "[$(date -Iseconds)] step 1: expert_drop -> $OUT"
"$PY" "$SCR/expert_drop.py" \
    --source-dir "$BASE" \
    --drop-map   "$DROP_MAP" \
    --output-dir "$OUT" \
    2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "FATAL: expert_drop failed exit=$RC"
    exit $RC
fi

# Sanity: index + shards present
if [ ! -f "$OUT/model.safetensors.index.json" ]; then
    echo "FATAL: $OUT/model.safetensors.index.json not found post-drop"
    exit 2
fi
nshards=$(ls "$OUT"/*.safetensors 2>/dev/null | wc -l)
sz=$(du -sh "$OUT" | cut -f1)
echo "  pristine 62e built: shards=$nshards size=$sz"

# Mandatory: tokenizer / chat template / generation_config / processor_config
# (eval needs these — copy from base if not present)
for f in tokenizer.json tokenizer_config.json chat_template.jinja \
         generation_config.json preprocessor_config.json processor_config.json \
         special_tokens_map.json; do
    if [ ! -s "$OUT/$f" ] && [ -f "$BASE/$f" ]; then
        cp "$BASE/$f" "$OUT/$f"
        echo "  seeded $f from base"
    fi
done

# Verify config.json has correct expert count
expected_experts=62
if [ -f "$OUT/config.json" ]; then
    num_experts=$("$PY" -c "import json; c=json.load(open('$OUT/config.json')); print(c.get('text_config',{}).get('num_local_experts', c.get('num_local_experts','?')))" 2>/dev/null)
    if [ "$num_experts" != "$expected_experts" ]; then
        echo "  WARN: config num_local_experts=$num_experts (expected $expected_experts) — may need patch"
    else
        echo "  config OK: num_local_experts=$num_experts"
    fi
fi

# Marker
cat > "$OUT/.pristine_built" <<EOF
{
  "built_ts": "$(date -Iseconds)",
  "base": "$BASE",
  "drop_map": "$DROP_MAP",
  "drop_map_sha256": "$(sha256sum "$DROP_MAP" | cut -d' ' -f1)",
  "shards": $nshards,
  "size": "$sz",
  "shared_alpha": null,
  "pes_alpha": null,
  "note": "T172 pristine — NO shared upweight, NO PES. Factorial origin for α-sweep."
}
EOF

# --- step 2: also build pristine F16 + Q6_K (A2 imatrix) for sanity audit ---
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
[ -f "$CONVERT" ] || CONVERT=/workspace/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
[ -x "$QUANT" ]   || QUANT=/workspace/llama.cpp/build/bin/llama-quantize
A2_IMATRIX=$BM/models/variants/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/imatrix.dat

GGUF_DIR="${OUT}-GGUF"
NAME=$(basename "$OUT")
F16="$GGUF_DIR/${NAME}-F16.gguf"
Q6="$GGUF_DIR/${NAME}-Q6_K.gguf"
mkdir -p "$GGUF_DIR"

if [ ! -f "$Q6" ]; then
    if [ ! -f "$F16" ]; then
        echo "[$(date -Iseconds)] step 2a: convert HF -> F16 GGUF"
        "$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -10
        [ -f "$F16" ] || { echo "FATAL: F16 GGUF not created"; exit 6; }
    fi
    echo "[$(date -Iseconds)] step 2b: quantize F16 -> Q6_K (A2 imatrix)"
    [ -f "$A2_IMATRIX" ] || { echo "FATAL: A2 imatrix missing at $A2_IMATRIX"; exit 7; }
    "$QUANT" --imatrix "$A2_IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -10
    [ -f "$Q6" ] || { echo "FATAL: Q6_K not created"; exit 8; }
    echo "  Q6_K created: $(du -sh "$Q6" | cut -f1)"
    # link imatrix
    if [ ! -e "$GGUF_DIR/imatrix.dat" ]; then
        ln "$A2_IMATRIX" "$GGUF_DIR/imatrix.dat" 2>/dev/null || cp "$A2_IMATRIX" "$GGUF_DIR/imatrix.dat"
    fi
    # seed tokenizer + configs into GGUF dir for eval
    for f in tokenizer.json tokenizer_config.json chat_template.jinja \
             generation_config.json preprocessor_config.json processor_config.json \
             special_tokens_map.json config.json; do
        if [ -s "$OUT/$f" ] && [ ! -s "$GGUF_DIR/$f" ]; then
            cp "$OUT/$f" "$GGUF_DIR/$f"
        fi
    done
    # purge transient F16
    rm -f "$F16"
    echo "  F16 purged (Q6_K kept)"
fi

echo
echo "[$(date -Iseconds)] === pristine 62e DONE ==="
echo "  bf16: $OUT  ($sz, $nshards shards)"
echo "  q6:   $Q6   ($(du -sh "$Q6" 2>/dev/null | cut -f1))"
echo
echo "Next: full-bench sanity audit (HE+/IF/MPE) — must agree with A2 within ±2pp"
