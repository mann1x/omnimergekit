#!/usr/bin/env bash
# run_agentic_eog_map.sh — T202.2/T202.5: build an agentic_eog emit-position
# competence map on the 128e teacher (bs2). Measures which experts predict the
# turn terminator at emit positions only.
#
# Gemma 4 26B-A4B stop tokens: eos_token_id = [1, 106, 50]
#   id 1   = <eos>            (true sequence end; ~0 positions in chat corpora)
#   id 50  = <|tool_response> (tool-call close; T202.2 default)
#   id 106 = <turn|>          (assistant TURN close — the agentic-loop terminator; T202.5)
# An agentic runaway (finish_reason="length") is the model failing to emit 106.
#
# Usage: run_agentic_eog_map.sh [GPU] [CORPUS] [OUT] [EMIT_TOKEN] [EMIT_ID]
set -euo pipefail
GPU="${1:-0}"
CORPUS="${2:-/mnt/sdc/ml/sft_heal/eog_corpus_solar.jsonl}"
OUT="${3:-/mnt/sdc/ml/google/expert_neuron_v7_agentic_eog.json}"
EMIT_TOKEN="${4:-<|tool_response>}"
EMIT_ID="${5:-50}"

PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it

echo "[agentic_eog] start $(date -u +%H:%M:%S)Z gpu=$GPU"
echo "[agentic_eog] teacher=$TEACHER"
echo "[agentic_eog] corpus=$CORPUS"
echo "[agentic_eog] out=$OUT  emit=$EMIT_TOKEN ($EMIT_ID)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCR/expert_neuron_analysis_v5_targeted.py" \
  --variant agentic_eog \
  --model "$TEACHER" \
  --gpu-budget-gib 90 \
  --eog-corpus "$CORPUS" \
  --emit-token "$EMIT_TOKEN" --emit-token-id "$EMIT_ID" \
  --window-tokens 2048 --window-overlap 0 \
  --out "$OUT"
echo "[agentic_eog] DONE $(date -u +%H:%M:%S)Z"

# Post-run sanity: total tc ~ emit_positions * top_k * layers (hundreds of
# thousands here), NOT total_tokens * top_k (millions). Millions => emit mask
# did NOT restrict accumulation — REJECT the map.
"$PY" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cat = d["categories"]["agentic_eog"]
tc = sum(int(r["tc"]) for rows in cat.values() for r in rows)
nz = sum(1 for rows in cat.values() for r in rows if int(r["tc"]) > 0)
L = len(cat); E = len(next(iter(cat.values())))
print(f"[sanity] agentic_eog total tc={tc}  experts_with_tc>0={nz}/{L*E}")
print("[sanity] EXPECT hundreds of thousands (emit_pos*8*30). If MILLIONS -> emit mask broken, REJECT.")
PY
