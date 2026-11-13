#!/usr/bin/env bash
# build_v9_agentic_sft11_maps.sh — v9-agentic refA (map A) steps 1–3, recomputed on the sft11 agentic model.
#
# Owner 2026-11-13: "rebuild from scratch gemma4-98e v9-agentic, create new weights using 26b sft agentic";
# drop map = "Recompute maps on sft11"; "the map was refA" (A_lift120_plus_fk12, no multilingual ladder).
#
# Every argument is map A's (build_v9_agentic_map.sh, run_agentic_eog_map.sh, chain_eog_bg.sh); only two
# things change: the model (sft11 export instead of the vendor base) and the GPU (bs2 GPU1). Inputs are the
# same files by sha256 (qset_FULL 9489785b, eog_corpus_v4_t106 49bff86b, trace template d9f21aac), tokenizer
# identical to base (cc8d3a0c).
#
#   1. competence map   variant code, Tier A 2048 tok thinking off, Tier B qset_FULL, window 2048/0
#   2. t106 emit map    agentic_eog, <turn|> 106, role model           (71 delivery closes)
#   3. t106 background  same corpus, complement positions               (the lift denominator)
#   4. lift ranking     rank_agentic_eog_lift.py --topk 4 -> 120 end-of-turn pins
#
# Usage: build_v9_agentic_sft11_maps.sh [GPU]      outputs in $O, never overwritten (refuses if present)
set -u
GPU=${1:-1}
MODEL=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
S=/srv/ml/repos/omnimergekit/scripts
T=/mnt/sdc/agentbench/tierb
TPL=/mnt/sdc/v7rework/gemma4_a4b_fixed_chat_template.jinja
O=/mnt/sdc/ml/v9a_sft11
REF_LIFT=/srv/ml/repos/omnimergekit/recipes/gemma4/v9_agentic/maps/candidates/eog_v4_t106_lift_pins.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%H:%M:%SZ; }
fail() { echo "V9A_SFT11_FAIL $*"; exit 1; }

mkdir -p "$O"
for f in competence_code.json eog_t106.json eog_t106_bg.json eog_t106_lift_pins.txt; do
  [ -e "$O/$f" ] && fail "$O/$f exists — refusing to overwrite"
done
[ -f "$MODEL/config.json" ] || fail "missing $MODEL"
cd /mnt/sdc/harness || fail "cd /mnt/sdc/harness"
echo "[v9a] start $(ts) gpu=$GPU model=$MODEL"

echo "########## 1/4 competence map $(ts) ##########"
"$PY" "$S/expert_neuron_analysis_v5_targeted.py" \
  --variant code --model "$MODEL" --gpu-budget-gib 90 \
  --tier-a-max-tokens 2048 --no-tier-a-thinking \
  --tier-b-json "$T/qset_FULL.json" \
  --window-tokens 2048 --window-overlap 0 \
  --chat-template-file "$TPL" \
  --out "$O/competence_code.json" || fail "competence map rc=$?"
"$PY" - "$O/competence_code.json" <<'PY' || fail "competence map metadata"
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metadata", {})
print("[v9a] categories:", list(d.get("categories", {})))
for k in ("model", "tier_a_max_tokens", "tier_a_thinking", "tier_b_source", "tier_b_trace_count",
          "tier_b_shape", "tier_b_set_counts", "chat_template_file"):
    print("   %-22s %s" % (k, m.get(k)))
PY
echo "V9A_STEP_OK competence $(ts)"

eog() { # mode out
  echo "########## EOG $1 -> $2 $(ts) ##########"
  "$PY" "$S/expert_neuron_analysis_v5_targeted.py" \
    --variant agentic_eog --model "$MODEL" --gpu-budget-gib 90 \
    --eog-corpus "$T/eog_corpus_v4_t106.jsonl" \
    --emit-token "<turn|>" --emit-token-id 106 --emit-role model --emit-mode "$1" \
    --window-tokens 2048 --window-overlap 0 \
    --out "$2" || fail "eog $1 rc=$?"
  "$PY" - "$2" <<'PY' || fail "eog sanity $2"
import json, sys
d = json.load(open(sys.argv[1])); cats = d["categories"]
cat = cats.get("agentic_eog") or cats["agentic_eog_bg"]
tc = sum(int(r["tc"]) for rows in cat.values() for r in rows)
nz = sum(1 for rows in cat.values() for r in rows if int(r["tc"]) > 0)
L = len(cat); E = len(next(iter(cat.values())))
m = d.get("metadata", {})
print(f"[sanity] total tc={tc}  experts_with_tc>0={nz}/{L*E}  emit_id={m.get('agentic_eog_emit_token_id')} "
      f"role={m.get('agentic_eog_emit_role')} stats={m.get('agentic_eog_stats')}")
PY
}
eog emit "$O/eog_t106.json";       echo "V9A_STEP_OK eog_emit $(ts)"
eog background "$O/eog_t106_bg.json"; echo "V9A_STEP_OK eog_bg $(ts)"

echo "########## 4/4 lift ranking $(ts) ##########"
"$PY" "$S/rank_agentic_eog_lift.py" --eog-map "$O/eog_t106.json" --bg-map "$O/eog_t106_bg.json" \
  --topk 4 --out "$O/eog_t106_lift_pins.txt" || fail "lift rc=$?"
"$PY" - "$O/eog_t106_lift_pins.txt" "$REF_LIFT" <<'PY'
import sys
def pins(p): return {x.strip() for x in open(p).read().replace("\n", ",").split(",") if x.strip()}
new, ref = pins(sys.argv[1]), pins(sys.argv[2])
print(f"[v9a] lift pins sft11={len(new)} refA={len(ref)} shared={len(new & ref)}")
PY
echo "V9A_STEP_OK lift $(ts)"
echo "V9A_SFT11_MAPS_DONE $(ts)"
