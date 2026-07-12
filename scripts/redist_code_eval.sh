#!/usr/bin/env bash
# T193 code-fold eval: folded bf16 62e -> F16 -> Q6_K (A2 imatrix) -> HE+164 + MPE-100 via llama backend.
# Apples-to-apples vs A2(pes120) Q6_K llama: HE+ 0.8963 / MPE 0.7533.
# Usage: redist_code_eval.sh <method> <gpu>
set -uo pipefail
METHOD="${1:-ream}"; GPU="${2:-1}"
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
IMATRIX=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/imatrix.dat
SRC=/mnt/sdc/ml/google/redist_code_${METHOD}_62e
GG=/mnt/sdc/ml/google/redist_code_${METHOD}_62e-GGUF
F16=$GG/redist_code_${METHOD}-F16.gguf
Q6=$GG/redist_code_${METHOD}-Q6_K.gguf
NAME=redist-code-${METHOD}
RES=$BM/eval_results_redist/code_eval
mkdir -p "$GG" "$RES"
L(){ echo "[codeeval_$METHOD $(date -u +%H:%M:%S)] $*"; }
[ -d "$SRC" ] || { L "FAIL: folded model $SRC missing"; exit 1; }
if [ ! -f "$Q6" ]; then
  L "=== convert bf16 -> F16 GGUF ==="
  "$PY" "$CONVERT" "$SRC" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$F16\").tensors))")
  L "F16 tensors: $n"
  [ "$n" -lt 600 ] && { L "FAIL: too few tensors ($n)"; exit 1; }
  L "=== quantize Q6_K (A2 imatrix) ==="
  "$QUANT" --imatrix "$IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -3
  rm -f "$F16"
fi
ls -la "$Q6"
for tpl in humanevalplus_full multipl_e_100; do
  L ">>> eval $tpl (llama backend)"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$SRC" \
    --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -15
done
L "=== SCORES: REDIST-code-$METHOD vs A2(pes120) Q6_K ==="
"$PY" - "$RES" "$NAME" <<PYEOF
import json, sys
RES, NAME = sys.argv[1], sys.argv[2]
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
he=sc(f"{RES}/humanevalplus_full/{NAME}/summary.json")
mpe=sc(f"{RES}/multipl_e_100/{NAME}/summary.json")
A2HE, A2MPE = 0.8963415, 0.7533333
def fmt(x): return "None" if x is None else f"{x:.4f}"
def dlt(x,a): return "None" if x is None else f"{x-a:+.4f}"
print(f"  HE+164 : {fmt(he)}  (A2 0.8963)  delta {dlt(he,A2HE)}")
print(f"  MPE-100: {fmt(mpe)}  (A2 0.7533)  delta {dlt(mpe,A2MPE)}")
print("  VERDICT:", "RECOVERED/HELD" if (he and mpe and he>=A2HE-0.005 and mpe>=A2MPE-0.005) else "REGRESSED-or-incomplete")
PYEOF
L "CODE_EVAL_DONE $METHOD"
