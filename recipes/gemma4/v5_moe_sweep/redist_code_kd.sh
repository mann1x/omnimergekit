#!/usr/bin/env bash
# T193b trainable code-KD: expert-KD A2 survivors toward 128e teacher on code corpus,
# then loop_screen(regression, bf16 unconfounded) + Q6_K HE+164/MPE-100 vs A2.
#
# Usage: redist_code_kd.sh [--run] [train_tensors] [train_layers] [steps] [lr] [gpu] [seqlen]
#   Dry-run by default (prints the plan, exits 0). Pass --run to execute.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ -f "$SCRIPT_DIR/redist_config.sh" ] && . "$SCRIPT_DIR/redist_config.sh"
DO_RUN=0; ARGS=()
for a in "$@"; do [ "$a" = "--run" ] && DO_RUN=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

TT="${1:-experts+router}"; TL="${2:-mid}"; STEPS="${3:-200}"; LR="${4:-2e-5}"; GPU="${5:-0}"; SEQ="${6:-1024}"
REDIST_PY="${REDIST_PY:-python}"
REDIST_SCRIPTS_DIR="${REDIST_SCRIPTS_DIR:-$REPO_ROOT/scripts}"
OMK_EVAL="${OMK_EVAL:-$REPO_ROOT/eval/omk_eval.py}"
CONVERT_HF_TO_GGUF="${CONVERT_HF_TO_GGUF:-$(command -v convert_hf_to_gguf.py || true)}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-$(command -v llama-quantize || true)}"
RES="${REDIST_RESULTS:-$PWD/eval_results_redist}"; CRES="$RES/code_eval"
OUT_BASE="${REDIST_OUT_BASE:-$PWD/redist_models}"
OUT="$OUT_BASE/redist_code_kd_62e"; GG="$OUT-GGUF"
Q6="$GG/redist_code_kd-Q6_K.gguf"; F16="$GG/redist_code_kd-F16.gguf"

cat <<PLAN
=== redist_code_kd plan (tt=$TT layers=$TL steps=$STEPS lr=$LR seq=$SEQ) ===
  python        : $REDIST_PY
  scripts-dir   : $REDIST_SCRIPTS_DIR
  omk_eval      : $OMK_EVAL
  convert/quant : ${CONVERT_HF_TO_GGUF:-<convert_hf_to_gguf.py not on PATH>} / ${LLAMA_QUANTIZE:-<llama-quantize not on PATH>}
  teacher       : ${REDIST_TEACHER:-<unset REDIST_TEACHER>}
  student       : ${REDIST_STUDENT:-<unset REDIST_STUDENT>}
  code-KD corpus: ${REDIST_KD_CORPUS_CODE:-<unset REDIST_KD_CORPUS_CODE>}
  sample        : ${REDIST_SAMPLE:-<unset REDIST_SAMPLE>}
  imatrix       : ${REDIST_IMATRIX:-<unset REDIST_IMATRIX>}
  out / gguf    : $OUT  ->  $Q6
  gpu           : $GPU
PLAN
[ "$DO_RUN" = 1 ] || { echo "(dry-run: pass --run to execute)"; exit 0; }

: "${REDIST_TEACHER:?set REDIST_TEACHER (128e teacher dir)}"
: "${REDIST_STUDENT:?set REDIST_STUDENT (pruned 62e dir)}"
: "${REDIST_KD_CORPUS_CODE:?set REDIST_KD_CORPUS_CODE (code KD train corpus jsonl)}"
: "${REDIST_SAMPLE:?set REDIST_SAMPLE (loop_screen jsonl)}"
: "${REDIST_IMATRIX:?set REDIST_IMATRIX (A2 imatrix.dat)}"
[ -n "$CONVERT_HF_TO_GGUF" ] || { echo "FAIL: convert_hf_to_gguf.py not found (set CONVERT_HF_TO_GGUF)"; exit 1; }
[ -n "$LLAMA_QUANTIZE" ]     || { echo "FAIL: llama-quantize not found (set LLAMA_QUANTIZE)"; exit 1; }
[ -n "${REDIST_ENV_BIN:-}" ] && export PATH="$REDIST_ENV_BIN:$PATH"
mkdir -p "$GG" "$RES" "$CRES" "$OUT_BASE"
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a "$RES/redist_code_kd.log") 2>&1
L(){ echo "[codekd $(date -u +%H:%M:%S)] $*"; }

L "=== [1/4] expert-KD train (tt=$TT layers=$TL steps=$STEPS lr=$LR seq=$SEQ) ==="
rm -rf "$OUT"
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/router_kd.py" \
  --base-dir "$REDIST_TEACHER" --variant-dir "$REDIST_STUDENT" --out-dir "$OUT" \
  --train-tensors "$TT" --train-layers "$TL" --student-load bf16 --teacher-load 4bit \
  --teacher-device "{\"\":0}" --student-device "{\"\":0}" --gpu-mem-gib 85 \
  --optim paged_adamw8bit --grad-checkpointing \
  --corpus-file "$REDIST_KD_CORPUS_CODE" --epochs 100 --max-samples 100000 \
  --tau 1.0 --lr "$LR" --max-steps "$STEPS" --batch-size 1 --grad-accum 8 \
  --max-seq-len "$SEQ" --no-canary --log-every 10 || { L "KD FAIL"; exit 1; }

L "=== [2/4] loop_screen regression (bf16) ==="
"$REDIST_PY" "$REDIST_SCRIPTS_DIR/loop_screen.py" --model "$OUT" --out "$RES/loop_code_kd.json" \
  --name "code_kd" --sample "$REDIST_SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$REDIST_PY" - "$RES/loop_code_kd.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); bb = d.get("by_bucket", {})
print("[%s] loop_pct=%s loops=%s/%s" % (d["name"], d.get("loop_pct"), d.get("loops"), d.get("n")))
print("  by_bucket: " + "  ".join("%s=%s/%s" % (b, v.get("loops"), v.get("n")) for b, v in sorted(bb.items())))
print("  A2 ANCHOR 15.5% (constr 6 ML 21 oe 2)  | REAM-fold was 19.0% (constr 8 ML 23 oe 5)")
PYEOF

L "=== [3/4] convert -> Q6_K (A2 imatrix) ==="
"$REDIST_PY" "$CONVERT_HF_TO_GGUF" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -3
n=$("$REDIST_PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader('$F16').tensors))")
L "F16 tensors: $n"; [ "$n" -lt 600 ] && { L "FAIL tensors"; exit 1; }
"$LLAMA_QUANTIZE" --imatrix "$REDIST_IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -3; rm -f "$F16"

L "=== [4/4] HE+164 + MPE-100 (llama) ==="
for tpl in humanevalplus_full multipl_e_100; do
  L ">>> $tpl"
  "$REDIST_PY" "$OMK_EVAL" --model "$Q6" --tokenizer "$OUT" --template "$tpl" --backend llama \
    --served-name "redist-code-kd" --results-dir "$CRES" 2>&1 | tail -12
done
"$REDIST_PY" - "$CRES" <<'PYEOF'
import json, sys
CRES = sys.argv[1]
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
he = sc(CRES + "/humanevalplus_full/redist-code-kd/summary.json")
mpe = sc(CRES + "/multipl_e_100/redist-code-kd/summary.json")
A2HE, A2MPE = 0.8963415, 0.7533333
def f(x): return "None" if x is None else "%.4f" % x
def dl(x, a): return "None" if x is None else "%+.4f" % (x - a)
print("CODE-KD vs A2(pes120) Q6_K:")
print("  HE+164 : %s (A2 0.8963) d %s" % (f(he), dl(he, A2HE)))
print("  MPE-100: %s (A2 0.7533) d %s" % (f(mpe), dl(mpe, A2MPE)))
print("  REAM-fold was HE+ 0.8841 / MPE 0.7200")
PYEOF
L "CODE_KD_DONE"
