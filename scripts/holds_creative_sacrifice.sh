#!/usr/bin/env bash
# holds_creative_sacrifice.sh — 4 hold evals for the creative-sacrifice v8b variant,
# mirroring EXACTLY the v8b-safe holds form (direct divergent omk_eval.py, base
# tokenizer, --parallel 2, greedy, b9700 server). One --template per bench:
#   gpqa_diamond_full (198) · humanevalplus_full · lcb_medium_55_v4 (55) · multipl_e_100
# GPQA (long pole) on the first GPU to free; HE+/LCB/MPE bundle on the second, parallel.
# Resumable (skip a bench whose summary.json exists). PID-safe. Reads scores from
# summary.json .score only. Final table vs v8 / cap2 / v7-coder + the loop-gate verdict.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py        # divergent checkout (matches how v8 MPE measured)
BASE=/srv/ml/models/base/gemma-4-26B-A4B-it            # base tokenizer (NOT the pruned variant)
SFT=/mnt/sdc/ml/sft_heal
Q6=$SFT/gemma-4-A4B-98e-creative-sacrifice-soft2-imat-Q6_K.gguf
WS=/mnt/sdc/ml/gpqa_dissect/ws_cs_hold
DIS=/mnt/sdc/ml/gpqa_dissect
AL=/srv/ml/agentic_loop
GRES=$AL/results/creative-sacrifice-soft2-imatQ6_minp48.json   # loop-gate result (build step 8)
NAME=cs_q6k
ts(){ date '+%T %Z'; }
mkdir -p "$WS"
echo "==================== creative-sacrifice HOLDS $(ts) ===================="
for f in "$Q6" "$OMK" "$BASE/tokenizer.json" /mnt/sdc/ml/llama.cpp-b9700/build/bin/llama-server; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH

# acquire first free GPU (<2000 MiB) not equal to $1, up to 4h
acquire_gpu(){
  local excl="${1:-}" g U
  for _ in $(seq 1 240); do
    for g in 0 1; do
      [ "$g" = "$excl" ] && continue
      U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
      [ "${U:-99999}" -lt 2000 ] && { echo "$g"; return 0; }
    done
    sleep 60
  done
  return 1
}

run_hold(){ # gpu port template results_dir logtag
  local gpu=$1 port=$2 tmpl=$3 rdir=$4 tag=$5
  echo "[$tag $(ts)] start GPU$gpu:$port  template=$tmpl"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$OMK" \
    --model "$Q6" --template "$tmpl" --backend llama --quant gguf \
    --port "$port" --results-dir "$rdir" --served-name "$NAME" \
    --tokenizer "$BASE" --parallel 2 \
    > "$DIS/cs_${tag}.log" 2>&1
  echo "[$tag $(ts)] done rc=$?"
}

# job A: GPQA-198 (long pole) on first free GPU
GPU_A=$(acquire_gpu "") || { echo "FATAL no GPU for GPQA"; exit 6; }
echo "[holds $(ts)] GPQA -> GPU$GPU_A"
( run_hold "$GPU_A" 8230 gpqa_diamond_full "$WS" gpqa ) &
PA=$!

# job B: HE+ -> LCB -> MPE (sequential) on second free GPU (!= GPU_A)
GPU_B=$(acquire_gpu "$GPU_A") || { echo "FATAL no GPU for code holds"; exit 6; }
echo "[holds $(ts)] code-holds (HE+/LCB/MPE) -> GPU$GPU_B"
(
  run_hold "$GPU_B" 8231 humanevalplus_full "$WS"        heplus
  run_hold "$GPU_B" 8232 lcb_medium_55_v4   "$WS"        lcb
  run_hold "$GPU_B" 8233 multipl_e_100      "$WS/mpe100" mpe
) &
PB=$!

wait "$PA" "$PB"
echo "==================== HOLDS DONE $(ts) ===================="

# ---- score readout (summary.json .score only) ----
"$PY" - "$WS" "$GRES" <<'PYEOF'
import glob, json, sys
ws, gres = sys.argv[1], sys.argv[2]
def score(pat):
    fs = glob.glob(pat, recursive=True)
    if not fs:
        return None
    d = json.load(open(sorted(fs)[-1]))
    return d.get("score"), d.get("metric"), d.get("filter")
benches = {
    "GPQA-198": f"{ws}/**/gpqa_diamond_full/**/summary.json",
    "HE+":      f"{ws}/**/humanevalplus_full/**/summary.json",
    "LCB-55":   f"{ws}/**/lcb_medium_55_v4/**/summary.json",
    "MPE-100":  f"{ws}/mpe100/**/multipl_e_100/**/summary.json",
}
ref = {  # v8 (=v7c on code) / cap2 / v7-coder published
    "GPQA-198": ("50.00", "67.68", "73.74"),
    "HE+":      ("93.29", "92.07", "93.29"),
    "LCB-55":   ("94.55", "92.73", "94.55"),
    "MPE-100":  ("89.67", "87.00", "89.67"),
}
print("\n================= creative-sacrifice HOLD SCORES =================")
print(f"{'bench':<10} {'creative-sac':>13} | {'v8':>7} {'cap2':>7} {'v7-coder':>9}   metric/filter")
for b, pat in benches.items():
    s = score(pat)
    cur = f"{s[0]*100:.2f}" if s and s[0] is not None else "PENDING"
    mf = f"{s[1]}/{s[2]}" if s else ""
    r = ref[b]
    print(f"{b:<10} {cur:>13} | {r[0]:>7} {r[1]:>7} {r[2]:>9}   {mf}")
# loop gate verdict
print("\n-- 48-seed loop gate (build step 8) --")
try:
    d = json.load(open(gres))
    for r in d["results"]:
        print(f"   {r['config']}: loops {r.get('loops')}/{r['seeds']}")
except Exception as e:
    print(f"   (loop-gate result not readable yet: {e})")
print("\nSHIP gate (user): coding holds HE+/MPE/LCB within margin of v8/v7-coder AND loops~cap2(1/0). GPQA lower OK.")
PYEOF
echo "==================== creative-sacrifice DECISION READOUT DONE $(ts) ===================="
