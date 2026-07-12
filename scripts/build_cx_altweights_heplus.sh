#!/usr/bin/env bash
# build_cx_altweights_heplus.sh — alternate-weight coderx variants, HE+ ONLY.
#
# STD16 recipe held byte-identical (gen_drop_v5_fk + same C6 args + same 46-pin
# agentic_eog --force-keep + shared a=1.2 + imat-Q6), ONLY --class-weights changed:
#   cx-c3l3 = code3/lcb3  (1 1 3 1 1 0 0 0 3)   map: cx_std16_code3lcb3_drop_map.json
#   cx-c3l4 = code3/lcb4  (1 1 3 1 1 0 0 0 4)   map: cx_std16_code3lcb4_drop_map.json
# Anchors (same imat-Q6 recipe): STD16 (code3/lcb2) HE+ 93.29 ; coderx (code4/lcb3) HE+ 93.29.
# Question: does leaning the targeted-LCB weight MOVE HE+ off 93.29?
#
# Gated on v7-coder v6-55 finishing (GPU0 free). Disk-serialized: build-part of each
# variant runs to Q6 + deletes its 52GB bf16/F16 BEFORE the next variant's build starts,
# so peak disk = one bf16+F16 (~104GB) + one Q6 (~17GB). HE+ runs in background per-GPU.
# imatrix.dat PRESERVED next to each Q6 (mandatory).
set -uo pipefail
PYB=/root/anaconda3/envs/omnimergekit/bin/python       # build/convert (cpu)
PYE=/srv/ml/envs/envs/omnimergekit/bin/python          # omk_eval env
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH" # HE+ shells out to lm-eval CLI
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
V6F=/srv/ml/eval_results_lcb_v6/lcb_v6_55/v7coder-q6/summary.json   # gate
ts(){ date '+%T %Z'; }
mkdir -p "$WORK" "$RES"

for f in "$SRC/config.json" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$CALIB" "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$OMK" "$PYE" \
         /srv/ml/scripts/cx_std16_code3lcb3_drop_map.json \
         /srv/ml/scripts/cx_std16_code3lcb4_drop_map.json; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done

wait_gpu_free() {   # $1 = physical gpu index; wait until <2GB used (~40min cap)
  local g=$1 u
  for _ in $(seq 1 80); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -lt 2000 ] && return 0
    sleep 30
  done
}

build_part() {   # $1=tag $2=map $3=gpu  -> produces $WORK/<tag>-imat-Q6_K.gguf, frees heavy files
  local TAG=$1 MAP=$2 GPU=$3
  local BF=$WORK/${TAG}-bf16 F16=$WORK/${TAG}-F16.gguf IMAT=$WORK/${TAG}-imatrix.dat Q6=$WORK/${TAG}-imat-Q6_K.gguf
  [ -f "$Q6" ] && { echo "[$TAG] Q6 exists, skip build"; return 0; }
  echo "[$TAG $(ts)] df:"; df -h "$WORK" | tail -1
  echo "[$TAG $(ts)] expert_drop ($(basename "$MAP"))"
  rm -rf "$BF"
  "$PYB" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" || { echo "FATAL $TAG drop"; return 1; }
  [ -f "$BF/tokenizer.json" ] || { echo "FATAL $TAG no tokenizer"; return 1; }
  echo "[$TAG $(ts)] router_shared_upweight a=1.2"
  "$PYB" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL $TAG shared"; return 1; }
  echo "[$TAG $(ts)] convert F16"
  "$PYB" "$LCPP/convert_hf_to_gguf.py" "$BF" --outfile "$F16" --outtype f16 || { echo "FATAL $TAG convert"; return 1; }
  echo "[$TAG $(ts)] wait GPU$GPU free, then imatrix"
  wait_gpu_free "$GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" \
    --chunks 128 -ngl 99 > "$WORK/${TAG}_imatrix.log" 2>&1 || { echo "FATAL $TAG imatrix"; tail -20 "$WORK/${TAG}_imatrix.log"; return 1; }
  echo "[$TAG $(ts)] imat-Q6_K"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 || { echo "FATAL $TAG quant"; return 1; }
  local magic; magic=$("$PYB" -c "import sys;print(open(sys.argv[1],'rb').read(4).decode('latin1'))" "$Q6" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL $TAG bad magic=$magic"; return 1; }
  rm -rf "$BF" "$F16"     # free disk; KEEP Q6 + imatrix.dat
  echo "[$TAG $(ts)] Q6 done $(stat -c%s "$Q6") bytes ; imatrix preserved $IMAT"
}

heplus() {   # $1=tag $2=gpu $3=port  (background HE+)
  local TAG=$1 GPU=$2 PORT=$3        # NOTE: TAG must bind on its own line before
  local Q6=$WORK/${TAG}-imat-Q6_K.gguf  # ${TAG} is used here (set -u nounset)
  echo "[$TAG $(ts)] HE+ launch GPU$GPU:$PORT"
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template humanevalplus_full \
    --backend llama --port "$PORT" --served-name "$TAG" --results-dir "$RES" \
    > "$WORK/${TAG}_heplus.log" 2>&1 || echo "[$TAG] WARN HE+ rc=$?"
}

echo "==================== CX alt-weights HE+ $(ts) ===================="
echo "[gate $(ts)] waiting v7-coder v6-55 (GPU0 free) ..."
for _ in $(seq 1 50); do [ -f "$V6F" ] && break; sleep 30; done
[ -f "$V6F" ] && echo "[gate $(ts)] v6-55 done; proceeding" || echo "[gate $(ts)] WARN v6-55 summary missing; proceeding"
sleep 12

# A: code3/lcb3 → GPU0 (build to Q6, free heavy) then HE+ on GPU0 in background
build_part cx-c3l3 /srv/ml/scripts/cx_std16_code3lcb3_drop_map.json 0 || echo "[cx-c3l3] build FAILED"
heplus cx-c3l3 0 8420 &
AHE=$!

# B: code3/lcb4 → GPU1 (A's heavy files now deleted → disk-safe). HE+ on GPU1 in background.
build_part cx-c3l4 /srv/ml/scripts/cx_std16_code3lcb4_drop_map.json 1 || echo "[cx-c3l4] build FAILED"
heplus cx-c3l4 1 8421 &
BHE=$!

wait "$AHE" "$BHE"

echo "[scores $(ts)]"
"$PYE" - <<PYEOF
import json, glob
RES="$RES"
anchors={"STD16 code3/lcb2":93.29,"coderx code4/lcb3":93.29}
print("=== HE+ (humanevalplus_full, imat-Q6, greedy) ===")
for tag in ["cx-c3l3","cx-c3l4"]:
    found=None
    for c in glob.glob(f"{RES}/**/summary.json", recursive=True):
        if f"/{tag}/" in c and "humanevalplus" in c:
            try: found=json.load(open(c))
            except Exception: pass
    sc = found.get("score") if found else None
    lbl = {"cx-c3l3":"code3/lcb3","cx-c3l4":"code3/lcb4"}[tag]
    print(f"  {lbl:<12} HE+ = {round(sc*100,2) if isinstance(sc,(int,float)) else 'NO RESULT'}")
print("  --- anchors (same imat-Q6 recipe) ---")
for k,v in anchors.items(): print(f"  {k:<18} HE+ = {v}")
PYEOF
echo "###### CX_ALTW_HEPLUS_DONE $(ts) ######"
