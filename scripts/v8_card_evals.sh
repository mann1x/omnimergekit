#!/usr/bin/env bash
# v8_card_evals.sh — clean, self-consistent v8 (fkbroad-soft2) re-runs for the
# v7-coder card overwrite. Three contested cells, all on the SAME binary the v8
# 9-bench used (b9700) at --parallel 2 (adequate per-slot ctx → no LCB truncation
# variance), greedy (template default), base tokenizer. One --template per bench:
#   arc_challenge_full (chat shadow, 1172)  ·  lcb_medium_55_v4 (55)  ·  lcb_medium_100_v4 (100)
# ARC on GPU0; LCB-55 -> LCB-100 sequential on GPU1; both lanes in parallel.
# Resumable (skip a bench whose summary.json exists). Reads scores from
# summary.json .score ONLY (GPQA/ARC flexible-extract; LCB pass_at_1).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py          # divergent checkout (matches v8 9-bench)
BASE=/srv/ml/models/base/gemma-4-26B-A4B-it              # base tokenizer (NOT the pruned variant)
SFT=/mnt/sdc/ml/sft_heal
Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf
LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin          # SAME binary the v8 9-bench used
WS=/mnt/sdc/ml/gpqa_dissect/ws_v8_card
NAME=v8_card
ts(){ date '+%T %Z'; }
mkdir -p "$WS"
echo "==================== v8 CARD RE-RUNS $(ts) ===================="
for f in "$Q6" "$OMK" "$BASE/tokenizer.json" "$LLAMA_BIN/llama-server"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
export LLAMA_BIN
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH

run_bench(){ # gpu port template logtag
  local gpu=$1 port=$2 tmpl=$3 tag=$4
  if find "$WS" -path "*/$tmpl/*/summary.json" 2>/dev/null | grep -q .; then
    echo "[$tag $(ts)] summary.json exists -> skip"; return 0; fi
  echo "[$tag $(ts)] start GPU$gpu:$port  template=$tmpl  bin=$LLAMA_BIN"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$OMK" \
    --model "$Q6" --template "$tmpl" --backend llama --quant gguf \
    --port "$port" --results-dir "$WS" --served-name "$NAME" \
    --tokenizer "$BASE" --parallel 2 \
    > "$WS/v8card_${tag}.log" 2>&1
  echo "[$tag $(ts)] done rc=$?"
}

# lane A: ARC-1172 (chat shadow; reasoning off via template override) on GPU0
( run_bench 0 8240 arc_challenge_full arc ) &
PA=$!
# lane B: LCB-55 then LCB-100 (sequential) on GPU1
( run_bench 1 8241 lcb_medium_55_v4  lcb55
  run_bench 1 8242 lcb_medium_100_v4 lcb100 ) &
PB=$!

wait "$PA" "$PB"
echo "==================== v8 CARD RE-RUNS DONE $(ts) ===================="

# ---- score readout (summary.json .score only) ----
"$PY" - "$WS" <<'PYEOF'
import glob, json, sys
ws = sys.argv[1]
def score(pat):
    fs = glob.glob(pat, recursive=True)
    if not fs:
        return None
    d = json.load(open(sorted(fs)[-1]))
    return d.get("score"), d.get("metric"), d.get("filter")
benches = {
    "ARC-1172": f"{ws}/**/arc_challenge_full/**/summary.json",
    "LCB-55":   f"{ws}/**/lcb_medium_55_v4/**/summary.json",
    "LCB-100":  f"{ws}/**/lcb_medium_100_v4/**/summary.json",
}
ref = {  # (v8 9-bench/prior, card v7-coder published)
    "ARC-1172": ("85.32 (9-bench)", "94.80"),
    "LCB-55":   ("87.27 g / 94.55 g / 96.36 dep", "96.36"),
    "LCB-100":  ("(missing)", "97.00"),
}
print("\n================= v8 CARD RE-RUN SCORES =================")
print(f"{'bench':<10} {'v8 fresh':>10} | {'prior':>30} {'card v7c':>9}   metric/filter")
for b, pat in benches.items():
    s = score(pat)
    cur = f"{s[0]*100:.2f}" if s and s[0] is not None else "PENDING"
    mf = f"{s[1]}/{s[2]}" if s else ""
    r = ref[b]
    print(f"{b:<10} {cur:>10} | {r[0]:>30} {r[1]:>9}   {mf}")
print("\nNote: ARC fresh should land ~85 (genuine -9.48 vs old v7-coder = new narrative).")
print("      LCB-55 fresh @ parallel-2 breaks the greedy truncation tie (expect ~94-96).")
PYEOF
echo "==================== v8 CARD READOUT DONE $(ts) ===================="
