#!/bin/bash
# Validate plain A2 (s1_0p1_20-it = current publish candidate) on the 21q rumination probe.
# Independent of the buggy router_kd.py canary — uses omk_eval + llama-server (production path).
set -uo pipefail
BM=/srv/ml
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
OUT=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it
Q6=$BM/google/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-GGUF/gemma-4-A4B-62e-fc15_25-p8-s1_0p1_20-it-Q6_K.gguf
RES=$BM/eval_results_a2_21q_validation
[ -f "$Q6" ] || { echo "FATAL: Q6_K missing at $Q6"; exit 1; }
if pgrep -f "[l]m-eval|[l]lama-server|[o]mk_eval" >/dev/null 2>&1; then
  echo "FATAL: GPU busy"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; exit 1
fi
TS=$(date +%Y%m%d_%H%M%S)
LOG=$BM/logs/a2_21q_validation_${TS}.log
mkdir -p "$BM/logs" "$RES"
export PATH=/root/anaconda3/envs/omnimergekit/bin:/opt/llama.cpp/build/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 OMK_NO_README=1
echo "[a2-21q] START $(date -Iseconds) Q6_K=$Q6" | tee -a "$LOG"
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
  echo "  [eval $tpl]" | tee -a "$LOG"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$OUT" --template "$tpl" --backend llama \
      --served-name "a2-62e-fc15_25-p8-s1_0p1_20" --results-dir "$RES" 2>&1 | tee -a "$LOG" \
    || echo "  WARN omk_eval $tpl nonzero (continuing)"
done
echo "[a2-21q] DONE $(date -Iseconds)" | tee -a "$LOG"
echo
echo "=== summary ==="
for tpl in humanevalplus_rum3 ifeval_rum3 multipl_e_rum15; do
  s=$RES/$tpl/a2-62e-fc15_25-p8-s1_0p1_20/summary.json
  [ -f "$s" ] && "$PY" -c "import json;d=json.load(open('$s'));print('  $tpl score=',round(d.get('score',0),4))"
done | tee -a "$LOG"
touch "$BM/logs/A2_21Q_${TS}_DONE"
