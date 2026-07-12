#!/usr/bin/env bash
# arm_coderx_noncode_9bench.sh — measure the 6 NON-code benches on coderx = code4/lcb3
# (CX16c4l3-imat-Q6) to complete its canonical 9-bench for the HF cards. The 3 code
# benches (HE+ 93.29 / MPE 89.00 / lcb_medium_55 92.73) + lcb_v6_77q hard-77 (85.71)
# + 48-seed loop gate (0/48 loops) are already measured; this fills GPQA / GSM8K /
# MATH500 / AIME / ARC / IFEval on the SAME build, results dir, served-name so all
# benches co-locate for card assembly.
#
# GREEDY (template_default) — the only sampler valid for the cross-variant card table
# (CLAUDE.md doctrine). --parallel 2 => per-slot 16384 >= reasoning_budget 12288, so
# NO SAT_COLLAPSE; greedy scores are parallel-invariant given enough per-slot ctx.
# Gated on GPU-free (hard-77 holds both GPUs until ~21:20 CEST): GPU0 = GPQA (long
# pole), GPU1 = the other 5 sequential.
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
wait_gpu(){ local g=$1 i; for i in $(seq 1 360); do gpu_free "$g" && return 0; sleep 20; done; return 0; }

# MANDATORY pre-launch greedy check (CLAUDE.md): every frozen template must show greedy.
BENCHES=(gpqa_diamond_full gsm8k_100 math500_100 aime_30 arc_challenge_full ifeval_100)
echo "==================== coderx (code4/lcb3) non-code 9-bench $(ts) ===================="
magic_ok "$CODERX" || { echo "FATAL: coderx Q6 invalid ($CODERX)"; exit 2; }
for b in "${BENCHES[@]}"; do
  f="$TPLDIR/$b.yaml"
  [ -f "$f" ] || { echo "FATAL: missing template $f"; exit 2; }
  # Greedy assertion (robust): some greedy templates (ifeval_100) omit do_sample and
  # rely on temperature:0.0 alone. FATAL only on an actual sampled block.
  if grep -qE "do_sample:[[:space:]]*true" "$f"; then
    echo "FATAL greedy-drift in $b — do_sample: true; STOP."; exit 3
  fi
  if ! grep -qE "temperature:[[:space:]]*0\.0" "$f"; then
    echo "FATAL greedy-drift in $b — temperature not 0.0; STOP."; exit 3
  fi
done
echo "[$(ts)] greedy check PASS for all 6 templates"

run_bench(){ # bench gpu port
  local B=$1 GPU=$2 PORT=$3 rc
  echo "[$(ts)] $B: wait GPU$GPU ..."; wait_gpu "$GPU"
  echo "[$(ts)] $B: launch GPU$GPU:$PORT (greedy, --parallel 2)"
  cd /srv/ml/repos/omnimergekit
  CUDA_VISIBLE_DEVICES=$GPU "$PYE" "$OMK" --model "$CODERX" --tokenizer "$TOK" --template "$B" \
    --backend llama --parallel 2 --port "$PORT" --served-name "$SN" --results-dir "$RES" \
    > "/mnt/sdc/ml/coderx_nc_${B}.log" 2>&1
  rc=$?; echo "[$(ts)] $B exit=$rc"
}

# GPU0: GPQA (long pole). GPU1: the other 5 sequential.
( run_bench gpqa_diamond_full   0 8450 ) & W0=$!
( run_bench gsm8k_100           1 8451
  run_bench math500_100         1 8451
  run_bench aime_30             1 8451
  run_bench arc_challenge_full  1 8451
  run_bench ifeval_100          1 8451 ) & W1=$!
wait "$W0" "$W1"

echo "================ coderx (code4/lcb3) FULL 10-bench $(ts) ================"
"$PYB" - <<PYEOF
import json,os
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
RES="$RES"; SN="$SN"
rows=[("GPQA-diamond","gpqa_diamond_full",RES),("GSM8K-100","gsm8k_100",RES),
      ("MATH500-100","math500_100",RES),("AIME-30","aime_30",RES),
      ("ARC-Challenge","arc_challenge_full",RES),("IFEval-100","ifeval_100",RES),
      ("HumanEval+ (164)","humanevalplus_full",RES),("MultiPL-E-100","multipl_e_100",RES),
      ("LCB-medium-55-v4","lcb_medium_55_v4",RES),
      ("lcb_v6_77q (hard-77)","lcb_hard_77","/srv/ml/eval_results_lcb_hard77")]
print("%-22s %9s" % ("bench (coderx code4/lcb3 imat-Q6)","greedy"))
for lbl,b,rd in rows:
    sn = "coderx-h77" if b=="lcb_hard_77" else SN
    o=sc(f"{rd}/{b}/{sn}/summary.json")
    print("%-22s %8.2f%%" % (lbl, o*100) if isinstance(o,(int,float)) else "%-22s %9s" % (lbl,"PENDING"))
PYEOF
echo "###### CODERX_NONCODE_9BENCH_DONE $(ts) ######"
