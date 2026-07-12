#!/usr/bin/env bash
# build_cx16_imatq6_eval.sh — CX16 code4/lcb3 imat-Q6_K on GPU1 + HE+/MPE100/LCB55 (greedy).
#
# Reuses the already-built CX16c4l3-bf16 student. Canonical imat recipe (identical to the
# published v8/STD16 imat-Q6): calib_both.txt, llama-imatrix -ngl 99 --chunks 128, then
# llama-quantize --imatrix ... Q6_K 32. imatrix.dat PRESERVED next to the Q6 (mandatory).
# Then omk_eval HE+ -> MPE100 -> LCB55 (greedy template-default), llama backend, GPU1-pinned.
# F16 uses a DISTINCT name so it can't race the GPU0 build's CX16c4l3-F16.gguf.
set -uo pipefail
PYB=/root/anaconda3/envs/omnimergekit/bin/python      # convert (cpu) + magic check
PYE=/srv/ml/envs/envs/omnimergekit/bin/python         # omk_eval env
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it            # tokenizer = original 128e dir
TOK=$SRC
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
WORK=/mnt/sdc/ml/cx_std16
BF=$WORK/CX16c4l3-bf16
F16=$WORK/CX16c4l3-imatF16.gguf
IMAT=$WORK/CX16c4l3-imatrix.dat
Q6=$WORK/CX16c4l3-imat-Q6_K.gguf
RES=/srv/ml/eval_results_cx_std16
NAME=cx16-c4l3-imatq6
GPU=1; PORT=8412
ts(){ date '+%T %Z'; }
mkdir -p "$WORK" "$RES"
echo "==================== CX16 imat-Q6 + benches $(ts) ===================="
for f in "$BF/.shared_applied" "$CALIB" "$LCPP/convert_hf_to_gguf.py" \
         "$LCPP/build/bin/llama-imatrix" "$LCPP/build/bin/llama-quantize" "$OMK" "$PYE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
df -h "$WORK" | tail -1

# 1) F16 (distinct name)
if [ ! -f "$F16" ]; then
  echo "[1 $(ts)] convert F16 -> $F16"
  "$PYB" "$LCPP/convert_hf_to_gguf.py" "$BF" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 3; }
fi

# 2) imatrix (GPU1, calib_both, 128 chunks, ngl 99) — PRESERVED
if [ ! -f "$IMAT" ]; then
  echo "[2 $(ts)] llama-imatrix -ngl 99 --chunks 128 GPU$GPU -> $IMAT"
  CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
    -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
    > "$WORK/cx16_imatrix_build.log" 2>&1 \
    || { echo "FATAL imatrix"; tail -25 "$WORK/cx16_imatrix_build.log"; exit 4; }
fi
echo "[2 $(ts)] imatrix.dat $(stat -c%s "$IMAT" 2>/dev/null) bytes"

# 3) imat-Q6_K, then drop F16
if [ ! -f "$Q6" ]; then
  echo "[3 $(ts)] quantize imat-Q6_K -> $Q6"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant"; exit 5; }
fi
magic=$("$PYB" -c "import sys;print(open(sys.argv[1],'rb').read(4).decode('latin1'))" "$Q6" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF magic=$magic"; exit 5; }
rm -f "$F16"
echo "[3 $(ts)] Q6 done: $(stat -c%s "$Q6") bytes ; imatrix preserved at $IMAT"

# 4) benches: HE+ -> MPE100 -> LCB55 (greedy template-default, GPU1)
for TMPL in humanevalplus_full multipl_e_100 lcb_medium_55_v4; do
  echo "[eval $(ts)] $TMPL"
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$Q6" --tokenizer "$TOK" --template "$TMPL" \
    --backend llama --port "$PORT" --served-name "$NAME" --results-dir "$RES" \
    > "$WORK/cx16_eval_${TMPL}.log" 2>&1 || echo "[eval] WARN $TMPL rc=$?"
done

# 5) scores from summary.json (.score is canonical)
echo "[scores $(ts)]"
"$PYE" - <<PYEOF
import json, glob
RES="$RES"
for tm in ["humanevalplus_full","multipl_e_100","lcb_medium_55_v4"]:
    best=None
    for c in glob.glob(f"{RES}/**/summary.json", recursive=True):
        if f"/{tm}/" in c or tm in c:
            try: d=json.load(open(c))
            except Exception: continue
            best=(c, d)
    if best:
        d=best[1]
        print(f"  {tm}: score={d.get('score')} metric={d.get('metric')} filter={d.get('filter')}")
    else:
        print(f"  {tm}: NO summary.json under {RES}")
PYEOF
echo "###### CX16_IMATQ6_EVAL_DONE $(ts) ######"
