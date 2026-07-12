#!/usr/bin/env bash
# orchestrate_minp48.sh — full 48-seed vendor_minp_rep x {0.9,0.8} confirmation, with
# proper reasoning flags, on dern11 at BOTH Q4 (GPU0, exists) and noimat Q6 (GPU1, built
# here from the existing F16). Parallel. Combined table at the end.
set -uo pipefail
SC=/srv/ml/scripts; AL=/srv/ml/agentic_loop; SFT=/mnt/sdc/ml/sft_heal
Q=/mnt/sdc/ml/llama.cpp-latest/build/bin/llama-quantize
PY=/root/anaconda3/envs/omnimergekit/bin/python
ts(){ date '+%T %Z'; }
DERN_Q4=$SFT/gemma-4-A4B-98e-v7-coder-dern11-it-Q4_K_M.gguf
DERN_F16=$SFT/v7coder-dern11-F16.gguf
DERN_Q6=$SFT/gemma-4-A4B-98e-v7-coder-dern11-it-Q6_K.gguf
mkdir -p "$AL/results" "$AL/logs"
echo "==================== orchestrate_minp48 $(ts) ===================="

echo "[$(ts)] GPU0: launch dern11-Q4 minp48 (no build needed)"
bash "$SC/gate_sweep48_minp.sh" "$DERN_Q4" 0 8190 "$AL/results/dern11_q4_minp48.json" dern11-Q4 \
  > "$AL/logs/minp48_q4.log" 2>&1 &
P0=$!

echo "[$(ts)] GPU1: build noimat dern11-Q6 (from F16)"
if [ ! -e "$DERN_Q6" ]; then
  "$Q" "$DERN_F16" "$DERN_Q6" Q6_K 32 > "$SFT/q6_dern11_quant.log" 2>&1 \
    && tail -2 "$SFT/q6_dern11_quant.log" || { echo "FATAL Q6 quant"; tail -15 "$SFT/q6_dern11_quant.log"; }
fi
ft=$("$PY" /mnt/sdc/ml/llama.cpp-latest/gguf-py/gguf/scripts/gguf_dump.py --no-tensors "$DERN_Q6" 2>/dev/null | grep -i general.file_type)
echo "[$(ts)] dern11-Q6: $ft (expect 18)"
echo "[$(ts)] GPU1: launch dern11-Q6 minp48"
bash "$SC/gate_sweep48_minp.sh" "$DERN_Q6" 1 8191 "$AL/results/dern11_q6_minp48.json" dern11-Q6 \
  > "$AL/logs/minp48_q6.log" 2>&1 &
P1=$!

wait $P0 || echo "WARN q4 rc=$?"
wait $P1 || echo "WARN q6 rc=$?"
echo "==================== COMBINED minp48 ($(ts)) ===================="
"$PY" - <<'PYEOF'
import json, os
AL = "/srv/ml/agentic_loop/results"
print("%-12s %-10s %-9s %-7s" % ("variant", "arm", "fails", "rate%"))
for tag, f in [("dern11-Q4", "dern11_q4_minp48.json"), ("dern11-Q6", "dern11_q6_minp48.json")]:
    p = os.path.join(AL, f)
    if not os.path.exists(p): print("%-12s (missing)" % tag); continue
    d = json.load(open(p))
    for r in d["results"]:
        print("%-12s %-10s %d/%-7d %.1f" % (tag, r["config"], r["fails"], r["seeds"], 100 * r["fail_rate"]))
PYEOF
echo "[$(ts)] === ORCHESTRATE_MINP48 DONE ==="