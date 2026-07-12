#!/usr/bin/env bash
# build_cx_c35l25.sh — single variant code3.5/lcb2.5 on GPU0. Same STD16 recipe,
# only --class-weights = 1 1 3.5 1 1 0 0 0 2.5. Map already generated
# (cx_std16_code35lcb25_drop_map.json, 40 experts moved vs STD16). Build →
# imat-Q6 (imatrix.dat preserved) → HE+ only, GPU0:8422.
set -uo pipefail
PYB=/root/anaconda3/envs/omnimergekit/bin/python
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
export HF_ALLOW_CODE_EVAL=1
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
TOK=$SRC
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
WORK=/mnt/sdc/ml/cx_altw
RES=/srv/ml/eval_results_cx_altw
MAP=/srv/ml/scripts/cx_std16_code35lcb25_drop_map.json
TAG=cx-c35l25
GPU=0; PORT=8422
BF=$WORK/${TAG}-bf16 F16=$WORK/${TAG}-F16.gguf IMAT=$WORK/${TAG}-imatrix.dat Q6=$WORK/${TAG}-imat-Q6_K.gguf
ts(){ date '+%T %Z'; }
mkdir -p "$WORK" "$RES"
for f in "$SRC/config.json" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" "$MAP" \
         "$CALIB" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$OMK"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

echo "==================== $TAG (code3.5/lcb2.5) BUILD+HE+ $(ts) ===================="
df -h "$WORK" | tail -1
if [ ! -f "$Q6" ]; then
  rm -rf "$BF"
  echo "[$TAG $(ts)] expert_drop"
  "$PYB" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" || { echo FATAL drop; exit 1; }
  [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer"; exit 1; }
  echo "[$TAG $(ts)] router_shared_upweight a=1.2"
  "$PYB" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" --alpha 1.2 --target mlp.down_proj.weight || { echo FATAL shared; exit 1; }
  echo "[$TAG $(ts)] convert F16"
  "$PYB" "$LCPP/convert_hf_to_gguf.py" "$BF" --outfile "$F16" --outtype f16 || { echo FATAL convert; exit 1; }
  echo "[$TAG $(ts)] imatrix GPU$GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" \
    --chunks 128 -ngl 99 > "$WORK/${TAG}_imatrix.log" 2>&1 || { echo FATAL imatrix; tail -20 "$WORK/${TAG}_imatrix.log"; exit 1; }
  echo "[$TAG $(ts)] imat-Q6_K"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 || { echo FATAL quant; exit 1; }
  magic=$("$PYB" -c "import sys;print(open(sys.argv[1],'rb').read(4).decode('latin1'))" "$Q6" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad magic=$magic"; exit 1; }
  rm -rf "$BF" "$F16"
  echo "[$TAG $(ts)] Q6 done $(stat -c%s "$Q6") bytes ; imatrix preserved $IMAT"
else echo "[$TAG] Q6 exists, skip build"; fi

echo "[$TAG $(ts)] HE+ GPU$GPU:$PORT"
CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template humanevalplus_full \
  --backend llama --port "$PORT" --served-name "$TAG" --results-dir "$RES" \
  > "$WORK/${TAG}_heplus.log" 2>&1 || echo "[$TAG] WARN HE+ rc=$?"

echo "[$TAG $(ts)] score:"
"$PYB" - <<PYEOF
import json, glob
for c in glob.glob("$RES/**/summary.json", recursive=True):
    if "/$TAG/" in c and "humaneval" in c.lower():
        try: print("  $TAG HE+ =", round(json.load(open(c)).get("score")*100,2), "%")
        except Exception: pass
PYEOF
echo "###### CX_C35L25_DONE $(ts) ######"
