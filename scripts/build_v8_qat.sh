#!/usr/bin/env bash
# build_v8_qat.sh — faithful v8 (fkbroad-soft2) recipe applied to Google's QAT
# q4_0-unquantized base, producing the qat-Q4_0 tier. UNLIKE v7-qat (drop+shared
# only), v8 REQUIRES the DERN Eq.11 soft-top-2 fold — that is the step that buys
# the 0/48 anti-loop guarantee — so this mirrors build_v8_fkbroad_soft2_combo.sh
# exactly, with the student sourced from the QAT base. Q4_0 is no-imatrix (QAT
# calibration is baked in).
#
# HARD GATE: the qat-Q4_0 must pass the SAME 48-seed agentic loop gate
# (vendor_minp_rep {0.9,0.8}) at 0/48 before it is eligible to ship. DERN-on-QAT
# then Q4_0-rounded is novel; if it can't hold 0/48, qat cannot carry v8's
# anti-loop identity and the tier is dropped (reported, not shipped).
#
# Pipeline: expert_drop(QAT base, fkbroad map) -> shared a=1.2 -> redist_prep
#   -> redist_dern_eq11(--assign-topk 2, teacher=128e) -> F16 -> Q4_0(no-imat)
#   -> 48-seed loop gate -> HE+164 / MPE-100 (greedy, b9700).
# Self-scheduling (waits first GPU <2000 MiB), resumable.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
LLAMA_BIN_EVAL=/mnt/sdc/ml/llama.cpp-b9700/build/bin
SFT=/mnt/sdc/ml/sft_heal
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it           # 128e teacher + tokenizer
QAT_BASE=/mnt/sdc/ml/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
DROP=/srv/ml/scripts/v8coder_fkbroad_drop_map.json       # SAME fkbroad map as v8
CORPUS=$SFT/eog_corpus_solar.jsonl
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop
BASE_TOK=/srv/ml/models/base/gemma-4-26B-A4B-it

FKBQ=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-qat-it        # student bf16 (drop+shared, QAT)
KEEPMETA=$SFT/v7coder_fkbroad_qat_keepmeta.json
COMBOQ=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-qat-it # combo bf16 (after DERN), QAT
F16=$SFT/fkbroad-soft2-qat-F16.gguf
Q40=$SFT/gemma-4-A4B-98e-v7-coder-qat-Q4_0.gguf          # the qat-Q4_0 ship tier
GRES=$AL/results/fkbroad-soft2-qat-Q40_minp48.json
NAME=v8-qat-Q4_0
TD=/mnt/sdc/ml/v8_qat/results
mkdir -p "$TD"
PORT=8290
ts(){ date '+%T %Z'; }
echo "==================== build v8 qat-Q4_0 $(ts) ===================="

for f in "$TEACHER/config.json" "$QAT_BASE/config.json" "$DROP" "$CORPUS" \
         "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$SCR/redist_prep_v7coder.py" "$DERN" "$LCPP/convert_hf_to_gguf.py" \
         "$LCPP/build/bin/llama-quantize" "$GATE" "$OMK" \
         "$LLAMA_BIN_EVAL/llama-server"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL redist not patched"; exit 2; }
[ -f "$QAT_BASE/model.safetensors.index.json" ] || { echo "FATAL QAT base incomplete"; exit 2; }
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

# ---- stages 1-3 (CPU) ----
if [ ! -f "$COMBOQ/model.safetensors.index.json" ] && [ ! -f "$COMBOQ/model.safetensors" ]; then
  if [ ! -f "$FKBQ/model.safetensors.index.json" ] && [ ! -f "$FKBQ/model.safetensors" ]; then
    echo "[1 $(ts)] expert_drop(QAT base, fkbroad) -> $FKBQ"
    "$PY" "$SCR/expert_drop.py" --source-dir "$QAT_BASE" --drop-map "$DROP" --output-dir "$FKBQ" \
      || { echo "FATAL expert_drop"; exit 3; }
    [ -f "$FKBQ/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
  else echo "[1] $FKBQ exists, skip"; fi
  if [ ! -f "$FKBQ/.shared_applied" ]; then
    echo "[2 $(ts)] router_shared_upweight --alpha 1.2"
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$FKBQ" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared"; exit 4; }
    rm -f "$FKBQ"/*.pre_shared_upweight
    touch "$FKBQ/.shared_applied"
  else echo "[2] .shared_applied exists, skip"; fi
  if [ ! -f "$KEEPMETA" ]; then
    echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
    "$PY" "$SCR/redist_prep_v7coder.py" "$FKBQ" "$DROP" "$KEEPMETA" \
      || { echo "FATAL keepmeta"; exit 5; }
  fi
else echo "[1-3] $COMBOQ exists, skip student build"; fi

# ---- acquire GPU ----
GPU=""
for i in $(seq 1 240); do
  for g in 0 1; do
    U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc "0-9")
    [ "${U:-99999}" -lt 2000 ] && { GPU=$g; break; }
  done
  [ -n "$GPU" ] && break
  echo "[acquire $(ts)] both GPUs busy, wait 60s ($i/240)"; sleep 60
done
[ -n "$GPU" ] || { echo "FATAL no free GPU"; exit 6; }
echo "[acquire $(ts)] GPU$GPU"

# ---- 4. DERN Eq.11 soft-top-2 (teacher=128e, student=qat-98e) ----
if [ ! -f "$COMBOQ/model.safetensors.index.json" ] && [ ! -f "$COMBOQ/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11 --assign-topk 2 -> $COMBOQ (GPU$GPU)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DERN" --teacher "$TEACHER" --student "$FKBQ" --keep-meta "$KEEPMETA" \
      --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBOQ" --device cuda:0 \
      --assign-topk 2 || { echo "FATAL redist_dern_eq11"; exit 7; }
  echo "[4 $(ts)] DERN done; removing intermediate $FKBQ"
  rm -rf "$FKBQ"
else echo "[4] $COMBOQ exists, skip redist"; fi

# ---- 4b. aux fill (QAT base lacks preprocessor_config.json; copy from 128e teacher) ----
for aux in preprocessor_config.json processor_config.json generation_config.json; do
  if [ ! -f "$COMBOQ/$aux" ] && [ -f "$TEACHER/$aux" ]; then
    cp -n "$TEACHER/$aux" "$COMBOQ/$aux" && echo "[4b $(ts)] copied aux $aux from teacher"
  fi
done

# ---- 5. F16 ----
[ -f "$F16" ] || { echo "[5 $(ts)] convert -> F16";
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBOQ" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 8; }; }

# ---- 6. Q4_0 (NO imatrix — QAT) ----
[ -f "$Q40" ] || { echo "[6 $(ts)] quantize Q4_0 (no-imat) -> $Q40";
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q40" Q4_0 32 \
    || { echo "FATAL quant"; exit 9; }; }
magic=$("$PY" -c "import sys;print(open('$Q40','rb').read(4).decode('latin1'))" 2>/dev/null)
[ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF header"; exit 9; }
echo "[6 $(ts)] qat-Q4_0 $(du -h "$Q40"|cut -f1)"

# ---- 7. HARD 48-seed loop gate ----
echo "[7 $(ts)] 48-seed loop gate $NAME {t0.9,t0.8} GPU$GPU:$PORT (HARD 0/48 bar)"
bash "$GATE" "$Q40" "$GPU" "$PORT" "$GRES" "$NAME" || echo "[7] WARN gate rc=$?"
GVERDICT=$("$PY" -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    loops=[r.get('loops',99) for r in d.get('results',[])]
    print('PASS' if loops and all(l==0 for l in loops) else 'FAIL', loops)
except Exception as e:
    print('FAIL', 'no-results', e)
" "$GRES" 2>/dev/null)
echo "[7 $(ts)] gate verdict: $GVERDICT"

# ---- 8. HE+164 / MPE-100 (greedy, b9700) ----
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
export HF_ALLOW_CODE_EVAL=1
p=8291
for TPL in humanevalplus_full multipl_e_100; do
  echo "[8 $(ts)] $TPL (GPU$GPU:$p)"
  LLAMA_BIN="$LLAMA_BIN_EVAL" CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" \
    --model "$Q40" --template "$TPL" --backend llama --quant gguf \
    --port "$p" --results-dir "$TD" --served-name "$NAME" \
    --tokenizer "$BASE_TOK" --parallel 2 || echo "[8] WARN $TPL rc=$?"
  p=$((p+1))
done

# ---- 9. summary ----
echo "[9 $(ts)] ===== V8 QAT-Q4_0 DONE ====="
echo "gate: $GVERDICT  ($GRES)"
for TPL in humanevalplus_full multipl_e_100; do
  S="$TD/$TPL/$NAME/summary.json"
  [ -f "$S" ] && "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('$TPL','score',d.get('score'))" "$S"
done
echo "combo-qat bf16: $COMBOQ"
echo "qat-Q4_0: $Q40 (F16 kept for CD-qat: $F16)"
echo "###### BUILD_V8_QAT_DONE GPU$GPU $(ts) ######"
