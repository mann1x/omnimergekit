#!/usr/bin/env bash
# arm_coderx_code_extra.sh — the 2 code benches code4/lcb3 (CX16c4l3-imat-Q6) is
# still missing to fully replace the v7-coderx card column: plain HumanEval-164
# (humaneval_full) + LCB-medium-100 (lcb_medium_100_v4, the pruned-variant template
# fs2440 used). Greedy, --parallel 2, same RES/SN so they co-locate with the rest.
# Gated on GPU0-free: runs on GPU0 as soon as the GPQA cell frees it, using that
# idle while GPU1 still finishes its 5-bench chain. Emits the COMPLETE card column.
set -uo pipefail
CODERX=/mnt/sdc/ml/cx_std16/CX16c4l3-imat-Q6_K.gguf
TOK=/srv/ml/models/base/gemma-4-26B-A4B-it
RES=/srv/ml/eval_results_cx_std16
SN=cx16-c4l3-imatq6
PYE=/srv/ml/envs/envs/omnimergekit/bin/python
PYB=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
TPLDIR=/srv/ml/repos/omnimergekit/eval/templates
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
ts(){ date '+%T %Z'; }
magic_ok(){ [ -f "$1" ] && [ "$(head -c4 "$1" 2>/dev/null)" = "GGUF" ]; }
gpu_free(){ local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 2000 ]; }
wait_gpu(){ local g=$1 i; for i in $(seq 1 540); do gpu_free "$g" && return 0; sleep 20; done; return 0; }

BENCHES=(humaneval_full lcb_medium_100_v4)
echo "==================== coderx (code4/lcb3) code-column extras $(ts) ===================="
magic_ok "$CODERX" || { echo "FATAL: coderx Q6 invalid ($CODERX)"; exit 2; }
for b in "${BENCHES[@]}"; do
  f="$TPLDIR/$b.yaml"; [ -f "$f" ] || { echo "FATAL: missing template $f"; exit 2; }
  if grep -qE "do_sample:[[:space:]]*true" "$f"; then echo "FATAL greedy-drift $b (do_sample:true)"; exit 3; fi
  if ! grep -qE "temperature:[[:space:]]*0\.0" "$f"; then echo "FATAL greedy-drift $b (temp!=0.0)"; exit 3; fi
done
echo "[$(ts)] greedy check PASS for $* ${BENCHES[*]}"

run_bench(){ # bench port
  local B=$1 PORT=$2 rc
  cd /srv/ml/repos/omnimergekit
  echo "[$(ts)] $B: launch GPU0:$PORT (greedy, --parallel 2)"
  CUDA_VISIBLE_DEVICES=0 "$PYE" "$OMK" --model "$CODERX" --tokenizer "$TOK" --template "$B" \
    --backend llama --parallel 2 --port "$PORT" --served-name "$SN" --results-dir "$RES" \
    > "/mnt/sdc/ml/coderx_extra_${B}.log" 2>&1
  rc=$?; echo "[$(ts)] $B exit=$rc"
}

echo "[$(ts)] waiting for GPU0 free (GPQA cell to finish) ..."; wait_gpu 0
run_bench humaneval_full   8452
run_bench lcb_medium_100_v4 8452

echo "================ coderx (code4/lcb3) COMPLETE CARD COLUMN $(ts) ================"
"$PYB" - <<PYEOF
import json,os
def sc(rd,b,sn):
    try: return json.load(open(f"{rd}/{b}/{sn}/summary.json")).get("score")
    except Exception: return None
R="$RES"; SN="$SN"; H="/srv/ml/eval_results_lcb_hard77"; V6="/srv/ml/eval_results_lcb_v6"
rows=[("GPQA-diamond (198)","gpqa_diamond_full",R,SN),("GSM8K-100","gsm8k_100",R,SN),
      ("MATH500-100","math500_100",R,SN),("AIME-30","aime_30",R,SN),
      ("ARC-Challenge","arc_challenge_full",R,SN),("IFEval-100","ifeval_100",R,SN),
      ("HumanEval-164","humaneval_full",R,SN),("HumanEval+-164","humanevalplus_full",R,SN),
      ("MultiPL-E-100","multipl_e_100",R,SN),
      ("LCB-medium-55 (v6)","lcb_v6_55",V6,"cx16-c4l3"),
      ("LCB-medium-100 (v4)","lcb_medium_100_v4",R,SN),
      ("LCB-hard-77 (v6_77q)","lcb_hard_77",H,"coderx-h77")]
print("%-22s %9s"%("bench (coderx code4/lcb3 imat-Q6, greedy)","score"))
for lbl,b,rd,sn in rows:
    o=sc(rd,b,sn)
    print("%-22s %8.2f%%"%(lbl,o*100) if isinstance(o,(int,float)) else "%-22s %9s"%(lbl,"PENDING"))
PYEOF
echo "###### CODERX_COLUMN_COMPLETE $(ts) ######"
