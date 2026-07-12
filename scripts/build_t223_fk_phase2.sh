#!/usr/bin/env bash
# build_t223_fk_phase2.sh — T223 Phase 2: imat-Q6 capability trade-check for the three
# force-keep winners {STD16, STD27, ADD16} vs the v8 anchor (fkbroad-soft2).
#
# Phase 1 (no-imat-Q4_0 loop gate) proved force-keeping the T223 loop experts collapses the
# loop monotonically (CTRL 31 -> STD16 1 -> STD27 0; ADD16 0). Phase 2 answers the SHIP
# question: at the actual serving tier (imat-Q6), does the loop kill HOLD, and what does
# force-keeping COST in capability (it evicts 16-27 low-agg survivors)? Each variant:
#   [expert_drop(map)+shared a=1.2  | reuse ADD16-combo] -> F16 -> model-specific imatrix
#   (calib_both, 128 chunks, ngl99, PRESERVED) -> imat-Q6 -> {b9700 48-seed loop gate,
#   HE+164, MPE-100, GPQA-198}. SINGLE-VARIABLE vs v8: identical imatrix recipe / shared a /
#   DERN(ADD16 only) / greedy eval. Anchors (v8 fkbroad-soft2 imat-Q6): loop 0/48 0/48 |
#   HE+ 93.29 | MPE 89.33 | GPQA ~50.0 (re-measured here on the v8 Q6 for same-template).
# Resumable (skips existing Q6 / gate result / bench summary). bf16 students dropped post-Q6.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/srv/ml/repos/omnimergekit/eval/omk_eval.py
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it          # 128e teacher + tokenizer
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt       # imatrix calib (same as v8)
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh   # b9700-pinned, matches Phase 1
TMPL=/srv/ml/repos/omnimergekit/eval/templates
AL=/srv/ml/agentic_loop
WORK=/mnt/sdc/ml/t223_fk
MAPDIR=/srv/ml/scripts
V8Q6=$SFT/gemma-4-A4B-98e-v7-coder-fkbroad-soft2-imat-Q6_K.gguf   # v8 anchor Q6 (exists)
ts(){ date '+%T %Z'; }
mkdir -p "$WORK" "$AL/logs"

# name | drop_map | dern(0/1) | combo_dir(reuse if dern)
VARS=(
  "STD16|v8coder_fk16_drop_map.json|0|"
  "STD27|v8coder_fk27_drop_map.json|0|"
  "ADD16|v8coder_fk16_drop_map.json|1|$WORK/ADD16-combo"
)

echo "==================== T223 FK PHASE-2 (imat-Q6 trade-check) $(ts) ===================="
# ── preflight ──
for f in "$SRC/config.json" "$CALIB" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE" "$OMK" \
         "$MAPDIR/v8coder_fk16_drop_map.json" "$MAPDIR/v8coder_fk27_drop_map.json" \
         "$TMPL/gpqa_diamond_full.yaml" "$TMPL/humanevalplus_full.yaml" "$TMPL/multipl_e_100.yaml"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
# greedy-template guard (2026-05-17 GPQA sampler incident): GPQA must be greedy for a fair anchor.
if grep -Eq 'do_sample:\s*true|temperature:\s*0*[1-9]' "$TMPL/gpqa_diamond_full.yaml"; then
  echo "FATAL gpqa_diamond_full.yaml is NOT greedy — fix template before launch"; exit 2; fi
[ -d "$WORK/ADD16-combo" ] || { echo "FATAL ADD16-combo missing (needed for ADD16)"; exit 2; }
[ -f "$V8Q6" ] || echo "[preflight $(ts)] WARN v8 anchor Q6 missing ($V8Q6) — GPQA anchor will skip"
echo "[preflight $(ts)] disk:"; df -h "$WORK" | tail -1

acquire_gpu(){  # echo index of first GPU <8000 MiB, wait up to 4h
  for _ in $(seq 1 240); do
    for g in 0 1; do
      local u; u=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
      [ "${u:-99999}" -lt 8000 ] && { echo "$g"; return 0; }
    done
    sleep 30
  done
  echo "-1"; return 1
}

# ── Phase A: build the 3 imat-Q6 (imatrix on first free GPU; serial) ──
build_q6(){  # name map dern combo
  local NM=$1 MAP=$MAPDIR/$2 DF=$3 COMBO=$4
  local BF=$WORK/${NM}-bf16 F16=$WORK/${NM}-F16.gguf
  local IMAT=$WORK/${NM}-imatrix.dat Q6=$WORK/${NM}-imatq6.gguf
  [ -f "$Q6" ] && { echo "[A $(ts)] $NM imat-Q6 exists, skip"; return 0; }
  local SRCDIR
  if [ "$DF" = "1" ]; then
    [ -d "$COMBO" ] || { echo "FATAL $NM combo $COMBO missing"; return 1; }
    SRCDIR=$COMBO
  else
    if [ ! -f "$BF/.shared_applied" ]; then
      echo "[A $(ts)] $NM expert_drop + shared a=1.2"
      rm -rf "$BF"
      "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
        || { echo "FATAL expert_drop $NM"; return 1; }
      [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer $NM"; return 1; }
      "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
        --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared $NM"; return 1; }
      touch "$BF/.shared_applied"
    else echo "[A $(ts)] $NM bf16 student exists, skip drop/shared"; fi
    SRCDIR=$BF
  fi
  [ -f "$F16" ] || { echo "[A $(ts)] $NM convert F16";
    "$PY" "$LCPP/convert_hf_to_gguf.py" "$SRCDIR" --outfile "$F16" --outtype f16 \
      || { echo "FATAL convert $NM"; return 1; }; }
  if [ ! -f "$IMAT" ]; then
    local G; G=$(acquire_gpu); [ "$G" = "-1" ] && { echo "FATAL no GPU for imatrix $NM"; return 1; }
    echo "[A $(ts)] $NM imatrix calib_both 128 chunks ngl99 (GPU$G) -> $IMAT"
    CUDA_VISIBLE_DEVICES=$G "$LCPP/build/bin/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" \
      --chunks 128 -ngl 99 > "$WORK/${NM}_imatrix.log" 2>&1 \
      || { echo "FATAL imatrix $NM"; tail -20 "$WORK/${NM}_imatrix.log"; return 1; }
  fi
  echo "[A $(ts)] $NM quant imat-Q6_K"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quant $NM"; return 1; }
  local magic; magic=$("$PY" -c "print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF $NM"; return 1; }
  rm -f "$F16"
  [ "$DF" = "0" ] && rm -rf "$BF"   # STD student reproducible from map; ADD16-combo kept
  echo "[A $(ts)] $NM imat-Q6 done: $(stat -c%s "$Q6") bytes (imatrix preserved $IMAT)"
}

echo "[A $(ts)] ===== PHASE A: build 3 imat-Q6 ====="
for c in "${VARS[@]}"; do
  IFS='|' read -r NM MAP DF COMBO <<<"$c"
  build_q6 "$NM" "$MAP" "$DF" "$COMBO" || echo "[A] WARN build $NM failed"
done

# ── Phase B: eval matrix, 2-at-a-time across GPU0/GPU1 ──
# fast jobs first (gate/heplus/mpe), GPQA last (the long pole).
echo "[B $(ts)] ===== PHASE B: loop gate + HE+/MPE/GPQA ====="
run_job(){  # nm kind gpu port
  local NM=$1 K=$2 G=$3 P=$4
  local Q6=$WORK/${NM}-imatq6.gguf
  [ "$NM" = "V8" ] && Q6=$V8Q6
  [ -f "$Q6" ] || { echo "[B] $NM Q6 missing, skip $K"; return 0; }
  if [ "$K" = "gate" ]; then
    local OUT=$AL/results/t223_${NM}_imatQ6_minp48.json
    [ -f "$OUT" ] && { echo "[B] $NM gate exists, skip"; return 0; }
    echo "[B $(ts)] $NM gate GPU$G:$P -> $OUT"
    MATRIX=matrix_minp_2temp.json bash "$GATE" "$Q6" "$G" "$P" "$OUT" "t223-${NM}-q6" \
      > "$AL/logs/t223_${NM}_q6_gate.log" 2>&1 || echo "[B] WARN $NM gate rc=$?"
    return 0
  fi
  local T; case "$K" in heplus) T=humanevalplus_full;; mpe) T=multipl_e_100;; gpqa) T=gpqa_diamond_full;; esac
  local TD=$WORK/tradecheck_${NM} S=$WORK/tradecheck_${NM}/$T/t223-${NM}/summary.json
  [ -f "$S" ] && { echo "[B] $NM $K exists, skip"; return 0; }
  mkdir -p "$TD"
  echo "[B $(ts)] $NM $K ($T) GPU$G:$P"
  CUDA_VISIBLE_DEVICES=$G PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH HF_ALLOW_CODE_EVAL=1 \
    "$PY" "$OMK" --model "$Q6" --template "$T" --backend llama --quant gguf --port "$P" \
      --results-dir "$TD" --served-name "t223-${NM}" --tokenizer "$SRC" --parallel 2 \
      > "$AL/logs/t223_${NM}_${K}.log" 2>&1 || echo "[B] WARN $NM $K rc=$?"
}
JOBS=()
for K in gate heplus mpe; do for NM in STD16 STD27 ADD16; do JOBS+=("$NM:$K"); done; done
for NM in STD16 STD27 ADD16 V8; do JOBS+=("$NM:gpqa"); done   # V8 = same-template GPQA anchor
i=0; port=8420
while [ $i -lt ${#JOBS[@]} ]; do
  IFS=: read -r NM0 K0 <<<"${JOBS[$i]}"; run_job "$NM0" "$K0" 0 "$port" & P0=$!; port=$((port+1))
  P1=""
  if [ $((i+1)) -lt ${#JOBS[@]} ]; then
    IFS=: read -r NM1 K1 <<<"${JOBS[$((i+1))]}"; run_job "$NM1" "$K1" 1 "$port" & P1=$!; port=$((port+1))
  fi
  wait $P0; [ -n "$P1" ] && wait $P1
  i=$((i+2)); [ $port -gt 8470 ] && port=8420
done

# ── Phase C: trade-check table ──
echo "[C $(ts)] ===== PHASE C: results ====="
"$PY" - <<'PYEOF'
import json, os
AL="/srv/ml/agentic_loop/results"; WORK="/mnt/sdc/ml/t223_fk"
V8TD="/srv/ml/agentic_loop/results/fkbroad_soft2_tradecheck"
def gate(nm):
    p=f"{AL}/t223_{nm}_imatQ6_minp48.json"
    if not os.path.exists(p): return "-"
    d=json.load(open(p)); o={}
    for r in d.get("results",[]):
        c=str(r.get("config","")); t="0.9" if "0.9" in c else "0.8" if "0.8" in c else c
        o[t]=f"l{r.get('loops','?')}"
    return f"{o.get('0.9','-')}/{o.get('0.8','-')}"
def sc(nm,t):
    s=f"{WORK}/tradecheck_{nm}/{t}/t223-{nm}/summary.json"
    if not os.path.exists(s): return "-"
    try: return round(float(json.load(open(s)).get("score"))*100,2)
    except Exception: return "?"
def v8sc(t):  # anchor HE+/MPE from existing v8 tradecheck dir (served-name unknown -> glob)
    base=f"{V8TD}/{t}"
    if not os.path.isdir(base): return "-"
    for sn in os.listdir(base):
        s=f"{base}/{sn}/summary.json"
        if os.path.exists(s):
            try: return round(float(json.load(open(s)).get("score"))*100,2)
            except Exception: pass
    return "-"
print(f"{'variant':9} {'loopQ6 0.9/0.8':>14} {'HE+':>7} {'MPE':>7} {'GPQA':>7}")
print("-"*52)
print(f"{'v8(ref)':9} {'0/0':>14} {v8sc('humanevalplus_full'):>7} {v8sc('multipl_e_100'):>7} {sc('V8','gpqa_diamond_full'):>7}")
for nm in ["STD16","STD27","ADD16"]:
    print(f"{nm:9} {gate(nm):>14} {sc(nm,'humanevalplus_full'):>7} {sc(nm,'multipl_e_100'):>7} {sc(nm,'gpqa_diamond_full'):>7}")
PYEOF
echo "###### T223_FK_PHASE2_DONE $(ts) ######"
