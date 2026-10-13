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
# EMIT_ROLE (2026-09-16): which terminators to accumulate. The mask used to be a bare
# `ids[t+1]==emit` with no role filter, so it also accumulated PROMPT-SIDE terminators --
# the system turn close and every user turn close, which are context the model conditions
# on, never output it produces. Measured on the manic-v4 delivery corpus: 213 `<turn|>`
# total, only 71 model-owned -> 66.7% of the accumulated positions were contamination.
# `model` is the default and the right choice for a TURN terminator (106). `any` is the
# legacy behaviour and is equivalent only for a terminator the model alone emits, such as
# <tool_call|> (49): 2523 vs 2523 on v4, 573 vs 573 on qset_A.
#
# Usage: run_agentic_eog_map.sh [GPU] [CORPUS] [OUT] [EMIT_TOKEN] [EMIT_ID] [EMIT_ROLE]
set -euo pipefail
GPU="${1:-0}"
CORPUS="${2:-/mnt/sdc/ml/sft_heal/eog_corpus_solar.jsonl}"
OUT="${3:-/mnt/sdc/ml/google/expert_neuron_v7_agentic_eog.json}"
EMIT_TOKEN="${4:-<|tool_response>}"
EMIT_ID="${5:-50}"
EMIT_ROLE="${6:-model}"
# emit | background. background accumulates the COMPLEMENT and is the denominator for
# lift; without it a ranking is raw mass, which a globally hot expert tops by default.
EMIT_MODE="${EMIT_MODE:-${7:-emit}}"

PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
TEACHER=/srv/ml/models/base/gemma-4-26B-A4B-it

echo "[agentic_eog] start $(date -u +%H:%M:%S)Z gpu=$GPU"
echo "[agentic_eog] teacher=$TEACHER"
echo "[agentic_eog] corpus=$CORPUS"
echo "[agentic_eog] out=$OUT  emit=$EMIT_TOKEN ($EMIT_ID)  emit_role=$EMIT_ROLE  emit_mode=$EMIT_MODE"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SCR/expert_neuron_analysis_v5_targeted.py" \
  --variant agentic_eog \
  --model "$TEACHER" \
  --gpu-budget-gib 90 \
  --eog-corpus "$CORPUS" \
  --emit-token "$EMIT_TOKEN" --emit-token-id "$EMIT_ID" \
  --emit-role "$EMIT_ROLE" --emit-mode "$EMIT_MODE" \
  --window-tokens 2048 --window-overlap 0 \
  --out "$OUT"
echo "[agentic_eog] DONE $(date -u +%H:%M:%S)Z"

# Post-run sanity: total tc ~ emit_positions * top_k * layers (hundreds of
# thousands here), NOT total_tokens * top_k (millions). Millions => emit mask
# did NOT restrict accumulation — REJECT the map.
"$PY" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cats = d["categories"]
cat = cats.get("agentic_eog") or cats["agentic_eog_bg"]
tc = sum(int(r["tc"]) for rows in cat.values() for r in rows)
nz = sum(1 for rows in cat.values() for r in rows if int(r["tc"]) > 0)
L = len(cat); E = len(next(iter(cat.values())))
print(f"[sanity] agentic_eog total tc={tc}  experts_with_tc>0={nz}/{L*E}")
print("[sanity] EXPECT hundreds of thousands (emit_pos*8*30). If MILLIONS -> emit mask broken, REJECT.")
m = d.get("metadata", {})
print("[provenance] emit_token_id=%s emit_token=%r emit_role=%s corpus=%s stats=%s"
      % (m.get("agentic_eog_emit_token_id"), m.get("agentic_eog_emit_token"),
         m.get("agentic_eog_emit_role"), m.get("agentic_eog_corpus"),
         m.get("agentic_eog_stats")))
if m.get("agentic_eog_emit_token_id") is None:
    print("[provenance] WARNING: emit axis not recorded -- this map predates the bug-361 fix")
PY
