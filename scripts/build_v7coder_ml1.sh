#!/usr/bin/env bash
# build_v7coder_ml1.sh — 98e v7-coder with generic_multilingual class-weight=1
# (ml1). IDENTICAL to build_v7coder.sh except --class-weights index5 0->1 and
# output names (-ml1-) + 3rd eval bench (lcb_medium_55_v4). Compare vs ml0
# (HE+ 0.9268 / MPE 0.8967; ml0 LCB run separately).
# Recipe = PUBLISHED v6-coder C6v3lcb knobs, byte-identical, fed the NEW v7
# fixed maps (T176 rebuild): --data expert_neuron_v7_code.json + the v7-rebuild
# per-layer floor v4_layer_floor_map_v7.json. Only the competence map + floor
# change vs v6-coder; every selection knob is held fixed.
#   target=98, protect-top=16, alpha=2.0, strategy=max, normalize=rank,
#   breadth-bonus=0.5, outlier median@1e4, class-weights 1 1 3 1 1 0 0 2
#   MANDATORY post: router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight
# Quant: PLAIN Q6_K (matches build_v6coder_C6v3lcb.sh -> HE+164 0.9329 anchor).
# Eval: HE+164 (humanevalplus_full) + MPE-100 (multipl_e_100), llama backend.
# Anchors (v6-coder C6v3lcb Q6_K): HE+164 0.9329 / MPE-100 0.89.
set -uo pipefail
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR=$BM/scripts
GEN=$SCR/generate_drop_map_v5.py
DATA=/mnt/sdc/ml/google/expert_neuron_v7_code.json
FLOOR=$BM/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
FLOORDATA=/mnt/sdc/ml/google/expert_neuron_base_v7.json
BASELINE=$SCR/teacher_force_98e_p16_clean.json
DROP=$SCR/v7coder_ml1_drop_map.json
SRC_HF=$BM/models/base/gemma-4-26B-A4B-it
OUT=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-ml1-it
GG=${OUT}-GGUF
F16=$GG/gemma-4-A4B-98e-v7-coder-ml1-it-F16.gguf
Q6=$GG/gemma-4-A4B-98e-v7-coder-ml1-it-Q6_K.gguf
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
NAME=v7coder-ml1-q6k
RES=$BM/eval_results_v7coder_ml1
GPU="${1:-0}"
mkdir -p "$GG" "$RES"
L(){ echo "[v7coder $(date -u +%H:%M:%S)] $*"; }

for f in "$GEN" "$DATA" "$FLOOR" "$FLOORDATA" "$BASELINE" "$SCR/expert_drop.py" "$SCR/router_shared_upweight.py" "$SRC_HF" "$CONVERT" "$QUANT" "$OMK"; do
  [ -e "$f" ] || { L "FATAL missing: $f"; exit 1; }
done
L "=== v7-coder build (v6 C6v3lcb knobs + v7 maps) ==="

# --- [1] drop map ---
if [ -f "$DROP" ]; then L "[1] drop map exists, skip"; else
  L "[1] generate_drop_map_v5 (v7 code map + v7 floor, C6v3lcb knobs)"
  "$PY" "$GEN" \
    --data "$DATA" --target 98 \
    --protect-top 16 --alpha 2.0 --strategy max --normalize rank \
    --breadth-bonus 0.5 --v4-floor-map "$FLOOR" --v4-floor-data "$FLOORDATA" \
    --baseline-drop-map "$BASELINE" \
    --outlier-mode median --outlier-wnorm-thresh 1e4 \
    --classes generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55 \
    --class-weights 1 1 3 1 1 1 0 0 2 \
    --output "$DROP" --summary-output "$DROP.summary.json" 2>&1 | tail -12
  [ -f "$DROP" ] || { L "FATAL drop map not written"; exit 1; }
fi
# verify 98e (drop 30/layer)
DPL=$("$PY" -c "import json,statistics as s;d=json.load(open('$DROP'));v=[len(x) for x in d.values() if isinstance(x,list)];print(int(s.mean(v)) if v else -1)")
L "[1] dropped/layer=$DPL (expect 30 -> 98e)"
[ "$DPL" = "30" ] || { L "FATAL not 98e (dropped/layer=$DPL)"; exit 1; }

# --- [2] expert_drop ---
if [ -f "$OUT/model.safetensors.index.json" ]; then L "[2] bf16 exists, skip"; else
  L "[2] expert_drop.py -> $OUT"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC_HF" --drop-map "$DROP" --output-dir "$OUT" 2>&1 | tail -8
  [ -f "$OUT/model.safetensors.index.json" ] || { L "FATAL expert_drop failed"; exit 1; }
fi

# --- [3] MANDATORY shared alpha=1.2 ---
if [ -f "$OUT/.shared_applied" ]; then L "[3] .shared_applied exists, skip"; else
  L "[3] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$SCR/router_shared_upweight.py" --model-dir "$OUT" --alpha 1.2 --target mlp.down_proj.weight 2>&1 | tail -6
  rm -f "$OUT"/*.pre_shared_upweight
  touch "$OUT/.shared_applied"
fi

# --- [4] convert F16 + plain Q6_K ---
if [ -f "$Q6" ]; then L "[4] Q6_K exists, skip"; else
  L "[4] convert bf16 -> F16"
  "$PY" "$CONVERT" "$OUT" --outfile "$F16" --outtype f16 2>&1 | tail -4
  n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader('$F16').tensors))")
  L "[4] F16 tensors=$n"; [ "$n" -lt 600 ] && { L "FATAL too few tensors"; exit 1; }
  L "[4] quantize plain Q6_K"
  "$QUANT" "$F16" "$Q6" Q6_K 2>&1 | tail -4
  rm -f "$F16"
fi
ls -la "$Q6"

# --- [5] eval HE+164 + MPE-100 (llama, GPU$GPU) ---
export CUDA_VISIBLE_DEVICES="$GPU"
for tpl in humanevalplus_full multipl_e_100 lcb_medium_55_v4; do
  L ">>> eval $tpl (llama backend, GPU$GPU)"
  "$PY" "$OMK" --model "$Q6" --tokenizer "$OUT" \
    --template "$tpl" --backend llama \
    --served-name "$NAME" --results-dir "$RES" 2>&1 | tail -15
done

L "=== SCORES: v7-coder vs v6-coder C6v3lcb Q6_K ==="
"$PY" - "$RES" "$NAME" <<PYEOF
import json, sys
RES, NAME = sys.argv[1], sys.argv[2]
def sc(p):
    try: return json.load(open(p)).get("score")
    except Exception: return None
he=sc(f"{RES}/humanevalplus_full/{NAME}/summary.json")
mpe=sc(f"{RES}/multipl_e_100/{NAME}/summary.json")
lcb=sc(f"{RES}/lcb_medium_55_v4/{NAME}/summary.json")
V6HE, V6MPE = 0.9329268, 0.89
def fmt(x): return "None" if x is None else f"{x:.4f}"
def dlt(x,a): return "None" if x is None else f"{x-a:+.4f}"
print(f"  HE+164 : {fmt(he)}  (v6 0.9329)  delta {dlt(he,V6HE)}")
print(f"  MPE-100: {fmt(mpe)}  (v6 0.8900)  delta {dlt(mpe,V6MPE)}")
    print(f"  LCB-55 : {fmt(lcb)}  (ml0 LCB run separately)")
ok = he is not None and mpe is not None
print("  VERDICT:", ("BEATS-v6" if (ok and he>=V6HE and mpe>=V6MPE) else
                      "MIXED" if (ok and (he>=V6HE or mpe>=V6MPE)) else
                      "BELOW-v6" if ok else "INCOMPLETE"))
PYEOF
L "V7CODER_ML1_DONE"
