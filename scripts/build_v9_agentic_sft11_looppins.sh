#!/usr/bin/env bash
# build_v9_agentic_sft11_looppins.sh — v9-agentic refA steps 4–5 on sft11: loop pins + map A′.
# Runs after build_v9_agentic_sft11_maps.sh (needs competence_code.json + eog_t106_lift_pins.txt).
#
# Owner decisions 2026-11-13: looper = sft11 pruned with the PRE-PIN map (refA generator args + the 120 sft11
# lift pins, no loop pins) + shared α 1.2 (refA's looper was likewise expert_drop + α 1.2, no fold);
# loop-generation sampler = the deployment sampler t0.8 / top_k 64 / top_p 0.95 / min_p 0 / presence 0.025,
# no repeat penalty (refA: t0.9 / min_p 0.05 / rep 1.1). Owner 14:58: "relabel, real loops only" — LOOP = cap hit AND
# detect_loop AND tail 8-gram repeat share >= 0.8, CLEAN = neither; ambiguous ones leave both sets (refA's 8 were
# 6 real + 2 healthy pages cut by the cap). Gate: >= 6 REAL loop sequences, else stop and report.
#
#   0. control   the generator on refA's own data + pins must reproduce A_lift120_plus_fk12 exactly
#   1. pre-pin   map Apre = generator(sft11 competence, force-keep = 120 sft11 lift pins)
#   2. looper    expert_drop(sft11, Apre) -> router_shared_upweight α 1.2 -> official template
#   3. diff      router_diff_agentic: looper generates 9 fixtures x 6 seeds, sft11 128e teacher-forces
#   4. pins      top 12 by specificity among the top 40 by loop mass (reproduces refA's P12 from its diff)
#   5. map A′    generator(sft11 competence, force-keep = 120 lift + 12 loop pins); diff vs refA
#
# Usage: build_v9_agentic_sft11_looppins.sh [GPU]       never overwrites an output
set -u
GPU=${1:-1}
MODEL=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
REPO=/srv/ml/repos/omnimergekit
O=/mnt/sdc/ml/v9a_sft11
I=$O/inputs
LOOPER=$O/looper_Apre_a12
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%H:%M:%SZ; }
fail() { echo "V9A_LP_FAIL $*"; exit 1; }
# Resumable, never overwriting: a step whose output exists AND passed (its V9A_LP_OK is in the log the waiter keeps,
# or its own marker) is skipped; later outputs must not exist yet.
for f in loop_diff.json loop_pins.txt A_sft11.json; do
  [ -e "$O/$f" ] && fail "$O/$f exists — refusing to overwrite"
done
DONE_PREPIN=0; DONE_LOOPER=0
[ -s "$O/Apre.json" ] && DONE_PREPIN=1
[ -f "$LOOPER/.shared_applied" ] && [ -f "$LOOPER/chat_template.jinja" ] && DONE_LOOPER=1
[ "$DONE_PREPIN" = 0 ] && [ -e "$LOOPER" ] && fail "$LOOPER exists without Apre.json — refusing to guess"
[ "$DONE_LOOPER" = 0 ] && [ -e "$LOOPER" ] && fail "$LOOPER exists but is incomplete — move it aside first"
for f in "$O/competence_code.json" "$O/eog_t106_lift_pins.txt" "$I/expert_neuron_base_v7.slim.json" \
         "$I/expert_neuron_v9_agentic_code.slim.json" "$I/A_lift120_plus_fk12.json" "$I/eog_v4_t106_lift_pins.txt" \
         "$I/chat_template.jinja" "$I/router_diff_agentic.py" "$I/v9ml_same_map.py"; do
  [ -f "$f" ] || fail "missing $f"
done
cd "$REPO" || fail "cd $REPO"

gen() { # data force-keep out
  local cw; mapfile -t cw < <("$PY" scripts/v9_classes.py "$1")
  # shellcheck disable=SC2086
  "$PY" recipes/gemma4/generate_drop_map_v5fk.py --data "$1" --target 98 --protect-top 16 --alpha 2.0 \
    --score-mode legacy --strategy max --normalize rank --classes ${cw[0]} --class-weights ${cw[1]} \
    --protect-strategy same --class-protect-floor 0 \
    --v4-floor-data "$I/expert_neuron_base_v7.slim.json" --v4-floor-top 0 \
    --v4-floor-map scripts/v4_layer_floor_map_v7.json \
    --breadth-bonus 0.5 --outlier-wnorm-thresh 10000.0 --outlier-mode median \
    --baseline-drop-map scripts/teacher_force_98e_p16_clean.json --force-keep "$2" \
    --output "$3" > "$3.log" 2>&1
}
P12_REF="0:62,1:114,3:68,21:111,8:0,0:100,7:63,19:116,8:85,18:102,25:122,5:85"

echo "########## 0 control: refA reproduces $(ts) ##########"
mkdir -p "$O/control"
gen "$I/expert_neuron_v9_agentic_code.slim.json" "$(cat "$I/eog_v4_t106_lift_pins.txt"),$P12_REF" \
  "$O/control/A_ref_repro.json" || fail "control generator rc=$?"
"$PY" "$I/v9ml_same_map.py" "$O/control/A_ref_repro.json" "$I/A_lift120_plus_fk12.json" \
  || fail "control: generator on refA data does NOT reproduce map A"
echo "V9A_LP_OK control $(ts)"

echo "########## 1 pre-pin map $(ts) ##########"
LIFT=$(tr '\n' ',' < "$O/eog_t106_lift_pins.txt" | sed 's/,*$//')
if [ "$DONE_PREPIN" = 1 ]; then
  echo "Apre.json exists ($(sha256sum "$O/Apre.json" | cut -c1-16)) — kept"
else
  gen "$O/competence_code.json" "$LIFT" "$O/Apre.json" || fail "pre-pin generator rc=$?"
fi
tail -3 "$O/Apre.json.log"
echo "V9A_LP_OK prepin $(ts)"

echo "########## 2 looper $(ts) ##########"
if [ "$DONE_LOOPER" = 1 ]; then
  echo "$LOOPER exists with .shared_applied + template — kept"
else
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "${free:-0}" -lt 120 ] && fail "/mnt/sdc ${free}G free < 120G"
"$PY" scripts/expert_drop.py --source-dir "$MODEL" --drop-map "$O/Apre.json" --output-dir "$LOOPER" \
  || fail "expert_drop rc=$?"
"$PY" recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir "$LOOPER" \
  --alpha 1.2 --target mlp.down_proj.weight || fail "shared_upweight rc=$?"
touch "$LOOPER/.shared_applied"
cp "$I/chat_template.jinja" "$LOOPER/chat_template.jinja"
fi
"$PY" - "$LOOPER" <<'PY' || fail "looper config check"
import json, sys
c = json.load(open(sys.argv[1] + "/config.json")); t = c.get("text_config", c)
print("[looper] num_experts", t.get("num_experts"), "top_k", t.get("top_k_experts", t.get("num_experts_per_tok")))
assert t.get("num_experts") == 98, "looper is not 98e"
PY
echo "V9A_LP_OK looper $(ts)"

echo "########## 3 loop differential $(ts) ##########"
"$PY" "$I/router_diff_agentic.py" --looper "$LOOPER" --base "$MODEL" --drop-map "$O/Apre.json" \
  --out "$O/loop_diff.json" \
  --fixtures csharp_loop_parity,csharp_loop_parity_nothink,solar_system_v4,kvstore_deep,solar_build_friction,solar_build_start,agentic_filewrite_v1,agentic_plan_todo_v1,threejs_build \
  --seeds 6 --gen-tokens 2048 --prefill-chunk 2048 --teacher-chunk 1024 \
  --temp 0.8 --top-p 0.95 --top-k 64 --min-p 0 --rep 1.0 --presence 0.025 \
  --relabel real --real-tail-share 0.8 \
  || fail "differential rc=$?"
echo "V9A_LP_OK diff $(ts)"

echo "########## 4 loop pins $(ts) ##########"
"$PY" - "$O/loop_diff.json" "$O/loop_pins.txt" <<'PY' || fail "loop gate"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[pins] loop={d['n_loop_seqs']} clean={d['n_clean_seqs']} dropped/tok={d['total_dropped_mass_per_tok']}")
if d["n_loop_seqs"] < 6:
    print("V9A_LP_GATE fewer than 6 loop sequences — not pinning from noise"); sys.exit(3)
c = sorted(d["dropped_loop_expert_candidates"], key=lambda r: -r["specificity"])[:12]
for r in c:
    print(f"   {r['layer']}:{r['expert']} loop={r['loop_mass']} clean={r['clean_mass']} spec={r['specificity']}")
open(sys.argv[2], "w").write(",".join(f"{r['layer']}:{r['expert']}" for r in c) + "\n")
PY
echo "V9A_LP_OK pins $(ts)"

echo "########## 5 map A′ $(ts) ##########"
gen "$O/competence_code.json" "$LIFT,$(cat "$O/loop_pins.txt")" "$O/A_sft11.json" || fail "A′ generator rc=$?"
"$PY" - "$O/A_sft11.json" "$I/A_lift120_plus_fk12.json" "$O/Apre.json" <<'PY'
import json, sys
def load(p):
    d = json.load(open(p)); d = d.get("drop_map", d)
    return {int(k): set(v) for k, v in d.items() if str(k).isdigit()}
new, ref, pre = (load(p) for p in sys.argv[1:4])
for name, other in (("refA", ref), ("pre-pin", pre)):
    same = sum(len(new[l] & other[l]) for l in new); tot = sum(len(new[l]) for l in new)
    print(f"[A'] dropped experts shared with {name}: {same}/{tot} ({100*same/tot:.1f}%)")
PY
echo "V9A_LP_DONE $(ts)"
