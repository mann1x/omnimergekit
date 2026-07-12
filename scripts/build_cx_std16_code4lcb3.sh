#!/usr/bin/env bash
# build_cx_std16_code4lcb3.sh — coderx-STD16 (the "more aggressive" sibling).
#
# Same STD16 recipe (fkbroad generate_drop_map_v5fk + agentic_eog force-keep + shared a=1.2,
# NO DERN, NO redistribution fold) but with heavier code weighting: class-weights
# `1 1 4 1 1 0 0 0 3` (generic_code 4x, targeted_lcb_medium_55 3x) vs STD16's `... 3 ... 2`.
# Map already generated: cx_std16_code4lcb3_drop_map.json (force-keep verified 0/46 dropped,
# 51 experts swapped vs STD16 fk16). This builds it -> no-imat-Q4_0 (the discriminating
# loop-exposing tier) -> 48-seed b9700 vendor_minp_rep {t0.9,t0.8} loop gate.
#
# Single arm: we DON'T build a no-force-keep baseline — the pre-check already proves the
# fs2440/coderx selection drops 44/46 loop-protection experts and will loop. The question
# is only whether the force-keep holds 0/48 on this more-concentrated code prune.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
MAP=/srv/ml/scripts/cx_std16_code4lcb3_drop_map.json
AL=/srv/ml/agentic_loop
WORK=/mnt/sdc/ml/cx_std16
NM=CX16c4l3
ts(){ date '+%T %Z'; }
mkdir -p "$WORK" "$AL/logs" "$AL/results"
BF=$WORK/${NM}-bf16 F16=$WORK/${NM}-F16.gguf Q4=$WORK/${NM}-Q4_0.gguf
OUT=$AL/results/cx_std16_${NM}_Q40_minp48.json

echo "==================== CX16 code4/lcb3 BUILD+GATE $(ts) ===================="
for f in "$SRC/config.json" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$MAP" "$GATE" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[preflight $(ts)] disk:"; df -h "$WORK" | tail -1

# 1) build student: expert_drop -> shared a=1.2
if [ ! -f "$BF/.shared_applied" ]; then
  rm -rf "$BF"
  echo "[build $(ts)] expert_drop (map=$(basename "$MAP"))"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
    || { echo "FATAL expert_drop"; exit 1; }
  [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer"; exit 1; }
  echo "[build $(ts)] router_shared_upweight a=1.2 mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
    --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared"; exit 1; }
  touch "$BF/.shared_applied"
else echo "[build] student exists, skip drop/shared"; fi

# 2) convert F16 + quant Q4_0 (NO imatrix)
if [ ! -f "$Q4" ]; then
  echo "[build $(ts)] convert F16"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$BF" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 1; }
  echo "[build $(ts)] llama-quantize Q4_0 (no imat)"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q4" Q4_0 32 || { echo "FATAL quant"; exit 1; }
  magic=$("$PY" -c "import sys;print(open(sys.argv[1],'rb').read(4).decode('latin1'))" "$Q4" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF magic=$magic"; exit 1; }
  rm -f "$F16"
  echo "[build $(ts)] Q4_0 done: $(stat -c%s "$Q4") bytes"
else echo "[build] Q4_0 exists, skip convert/quant"; fi

# 3) 48-seed loop gate (GPU0, b9700, vendor_minp_rep x {t0.9,t0.8})
if [ ! -f "$OUT" ]; then
  echo "[gate $(ts)] 48-seed b9700 -> $OUT"
  MATRIX=matrix_minp_2temp.json bash "$GATE" "$Q4" 0 8410 "$OUT" "cx-std16-c4l3-q40" \
    > "$AL/logs/cx_std16_${NM}_gate.log" 2>&1 || echo "[gate] WARN rc=$?"
else echo "[gate] result exists, skip"; fi

# 4) result table
echo "[result $(ts)]"
"$PY" - <<PYEOF
import json, os
p="$OUT"
if not os.path.exists(p):
    print("NO RESULT FILE"); raise SystemExit
d=json.load(open(p))
print("=== CX16 code4/lcb3 + force-keep loop gate (no-imat-Q4_0, n=48/temp) ===")
for r in d.get("results", []):
    cfg=str(r.get("config","?"))
    print(f"  {cfg}: fails={r.get('fails','?')} loops={r.get('loops','?')}")
PYEOF
echo "==== gate-log FAIL=True lines (newly-induced loopers, want none) ===="
grep -E "FAIL=True" "$AL/logs/cx_std16_${NM}_gate.log" 2>/dev/null | head || echo "  (none)"
echo "###### CX16_C4L3_DONE $(ts) ######"
