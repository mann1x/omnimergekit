#!/usr/bin/env bash
# T193 code-fold eval: folded bf16 62e -> F16 -> Q6_K (A2 imatrix) -> HE+164 + MPE-100 via llama.
# Apples-to-apples vs A2(pes120) Q6_K llama: HE+ 0.8963 / MPE 0.7533.
#
# Usage: redist_code_eval.sh [--run] <method> [gpu]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
#   Operates on the model dir produced by redist_code_fold.sh ($REDIST_OUT_BASE/redist_code_<method>_62e).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

METHOD="${1:-ream}"; GPU="${2:-1}"
REDIST_PY="${REDIST_PY:-python}"
OMK_EVAL="${OMK_EVAL:-$REPO_ROOT/eval/omk_eval.py}"
CONVERT_HF_TO_GGUF="${CONVERT_HF_TO_GGUF:-$(command -v convert_hf_to_gguf.py || true)}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$(command -v llama-quantize || true)}"
OUT_BASE="${REDIST_OUT_BASE:-$PWD/redist_models}"
SRC="$OUT_BASE/redist_code_${METHOD}_62e"; GG="$SRC-GGUF"
F16="$GG/redist_code_${METHOD}-F16.gguf"; Q6="$GG/redist_code_${METHOD}-Q6_K.gguf"
NAME="redist-code-${METHOD}"
RES="${REDIST_RESULTS:-$PWD/eval_results_redist}/code_eval"

cat <<PLAN
=== redist_code_eval plan (method=$METHOD) ===
  python        : $REDIST_PY
  omk_eval      : $OMK_EVAL
  convert/quant : ${CONVERT_HF_TO_GGUF:-<convert_hf_to_gguf.py not on PATH>} / ${LLAMA_QUANTIZE:-<llama-quantize not on PATH>}
  src model     : $SRC
  imatrix       : ${REDIST_IMATRIX:-<unset REDIST_IMATRIX>}
  gguf -> q6    : $Q6
  results       : $RES   served=$NAME
  gpu           : $GPU
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_IMATRIX:?set REDIST_IMATRIX (A2 imatrix.dat)}"
[ -d "$SRC" ] || { echo "FAIL: folded model $SRC missing (run redist_code_fold.sh --run first)"; exit 1; }
[ -n "$CONVERT_HF_TO_GGUF" ] || { echo "FAIL: convert_hf_to_gguf.py not found (set CONVERT_HF_TO_GGUF)"; exit 1; }
[ -n "$LLAMA_QUANTIZE" ]     || { echo "FAIL: llama-quantize not found (set LLAMA_QUANTIZE)"; exit 1; }
[ -n "${REDIST_ENV_BIN:-}" ] && export PATH="$REDIST_ENV_BIN:$PATH"
mkdir -p "$GG" "$RES"
export CUDA_VISIBLE_DEVICES="$GPU"
exec > >(tee -a "$RES/redist_code_eval_${METHOD}.log") 2>&1
L(){ echo "[codeeval_$METHOD $(date -u +%H:%M:%S)] $*"; }

if [ ! -f "$Q6" ]; then
  L "=== convert bf16 -> F16 GGUF ==="
  "$REDIST_PY" "$CONVERT_HF_TO_GGUF" "$SRC" --outfile "$F16" --outtype f16 2>&1 | tail -3
  n=$("$REDIST_PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader('$F16').tensors))")
  L "F16 tensors: $n"
  [ "$n" -lt 600 ] && { L "FAIL: too few tensors ($n)"; exit 1; }
  L "=== quantize Q6_K (A2 imatrix) ==="
  "$LLAMA_QUANTIZE" --imatrix "$REDIST_IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -3
  rm -f "$F16"
fi
ls -la "$Q6"
for tpl in humanevalplus_full multipl_e_100; do
  L ">>> eval $tpl (llama backend)"
  "$REDIST_PY" "$OMK_EVAL" --model "$Q6" --tokenizer "$SRC" \
    --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -15
done
L "=== SCORES: REDIST-code-$METHOD vs A2(pes120) Q6_K ==="
"$REDIST_PY" - "$RES" "$NAME" <<'PYEOF'
import json, sys
RES, NAME = sys.argv[1], sys.argv[2]
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
he = sc(f"{RES}/humanevalplus_full/{NAME}/summary.json")
mpe = sc(f"{RES}/multipl_e_100/{NAME}/summary.json")
A2HE, A2MPE = 0.8963415, 0.7533333
def fmt(x): return "None" if x is None else f"{x:.4f}"
def dlt(x, a): return "None" if x is None else f"{x-a:+.4f}"
print(f"  HE+164 : {fmt(he)}  (A2 0.8963)  delta {dlt(he, A2HE)}")
print(f"  MPE-100: {fmt(mpe)}  (A2 0.7533)  delta {dlt(mpe, A2MPE)}")
print("  VERDICT:", "RECOVERED/HELD" if (he and mpe and he >= A2HE - 0.005 and mpe >= A2MPE - 0.005) else "REGRESSED-or-incomplete")
PYEOF
L "CODE_EVAL_DONE $METHOD"
