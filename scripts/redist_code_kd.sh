#!/usr/bin/env bash
# T193b trainable code-KD: expert-KD A2 survivors toward 128e teacher on code corpus,
# then loop_screen(regression, bf16 unconfounded) + Q6_K HE+164/MPE-100 vs A2.
# Usage: redist_code_kd.sh <train_tensors> <train_layers> <steps> <lr> <gpu> [seqlen]
set -uo pipefail
TT="${1:-experts+router}"; TL="${2:-mid}"; STEPS="${3:-200}"; LR="${4:-2e-5}"; GPU="${5:-0}"; SEQ="${6:-1024}"
BM=/srv/ml; PY=$BM/envs/envs/omnimergekit/bin/python; SCR=$BM/scripts
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TEACHER=$BM/models/base/gemma-4-26B-A4B-it
STUDENT=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it
CODE_KD=/mnt/sdc/ml/corpora/kd_corpus_code.jsonl
SAMPLE=/mnt/sdc/ml/corpora/loop_screen_sample.jsonl
IMATRIX=/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it-GGUF/imatrix.dat
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
OUT=/mnt/sdc/ml/google/redist_code_kd_62e
GG=/mnt/sdc/ml/google/redist_code_kd_62e-GGUF
Q6=$GG/redist_code_kd-Q6_K.gguf; F16=$GG/redist_code_kd-F16.gguf
RES=$BM/eval_results_redist; CRES=$RES/code_eval
mkdir -p "$GG" "$RES" "$CRES"
L(){ echo "[codekd $(date -u +%H:%M:%S)] $*"; }

L "=== [1/4] expert-KD train (tt=$TT layers=$TL steps=$STEPS lr=$LR seq=$SEQ) ==="
rm -rf "$OUT"
"$PY" "$SCR/router_kd.py" \
  --base-dir "$TEACHER" --variant-dir "$STUDENT" --out-dir "$OUT" \
  --train-tensors "$TT" --train-layers "$TL" --student-load bf16 --teacher-load 4bit \
  --teacher-device "{\"\":0}" --student-device "{\"\":0}" --gpu-mem-gib 85 \
  --optim paged_adamw8bit --grad-checkpointing \
  --corpus-file "$CODE_KD" --epochs 100 --max-samples 100000 \
  --tau 1.0 --lr "$LR" --max-steps "$STEPS" --batch-size 1 --grad-accum 8 \
  --max-seq-len "$SEQ" --no-canary --log-every 10 || { L "KD FAIL"; exit 1; }

L "=== [2/4] loop_screen regression (bf16) ==="
"$PY" "$SCR/loop_screen.py" --model "$OUT" --out "$RES/loop_code_kd.json" --name "code_kd" \
  --sample "$SAMPLE" --bs 16 --max-new 2048 || { L "SCREEN FAIL"; exit 1; }
"$PY" - "$RES/loop_code_kd.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); bb=d.get("by_bucket",{})
print("[%s] loop_pct=%s loops=%s/%s"%(d["name"],d.get("loop_pct"),d.get("loops"),d.get("n")))
print("  by_bucket: "+"  ".join("%s=%s/%s"%(b,v.get("loops"),v.get("n")) for b,v in sorted(bb.items())))
print("  A2 ANCHOR 15.5% (constr 6 ML 21 oe 2)  | REAM-fold was 19.0% (constr 8 ML 23 oe 5)")
PYEOF

L "=== [3/4] convert -> Q6_K (A2 imatrix) ==="
"$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -3
n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$F16\").tensors))")
L "F16 tensors: $n"; [ "$n" -lt 600 ] && { L "FAIL tensors"; exit 1; }
"$QUANT" --imatrix "$IMATRIX" "$F16" "$Q6" Q6_K 2>&1 | tail -3; rm -f "$F16"

L "=== [4/4] HE+164 + MPE-100 (llama) ==="
for tpl in humanevalplus_full multipl_e_100; do
  L ">>> $tpl"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$OUT" --template "$tpl" --backend llama \
    --served-name "redist-code-kd" --results-dir "$CRES" 2>&1 | tail -12
done
"$PY" - "$CRES" <<'PYEOF'
import json,sys
CRES=sys.argv[1]
def sc(p):
    try: return json.load(open(p)).get("score")
    except: return None
he=sc(CRES+"/humanevalplus_full/redist-code-kd/summary.json")
mpe=sc(CRES+"/multipl_e_100/redist-code-kd/summary.json")
A2HE,A2MPE=0.8963415,0.7533333
def f(x): return "None" if x is None else "%.4f"%x
def dl(x,a): return "None" if x is None else "%+.4f"%(x-a)
print("CODE-KD vs A2(pes120) Q6_K:")
print("  HE+164 : %s (A2 0.8963) d %s"%(f(he),dl(he,A2HE)))
print("  MPE-100: %s (A2 0.7533) d %s"%(f(mpe),dl(mpe,A2MPE)))
print("  REAM-fold was HE+ 0.8841 / MPE 0.7200")
PYEOF
L "CODE_KD_DONE"
