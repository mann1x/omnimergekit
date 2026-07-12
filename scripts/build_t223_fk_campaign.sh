#!/usr/bin/env bash
# build_t223_fk_campaign.sh — T223 force-keep campaign (all 4 user options + 2 baselines).
#
# Tests whether structurally protecting the T223 loop-specific dropped experts reduces the
# v8 no-imat-Q4_0 loop fragility (v8 imat-Q6 0/48 but no-imat-Q4_0 9-12/48). 6 cells, a clean
# 2x2 (force-keep x DERN) plus a K-sweep, ALL gated same-binary (b9700) at the discriminating
# no-imat-Q4_0 tier, vendor_minp_rep {t0.9,t0.8} x 48 seeds on solar_build_start:
#
#                 no force-keep        +force-keep
#   no DERN         CTRL                 STD8 / STD16 / STD27   (fkbroad+shared)
#   +DERN soft2     V8 (baseline)        ADD16                  (+redist_dern_eq11 --assign-topk 2)
#
# Each cell: expert_drop(map) -> router_shared_upweight a=1.2 -> [keepmeta+DERN if dern] ->
#            convert F16 -> llama-quantize Q4_0 (NO imatrix) -> 48-seed b9700 loop gate.
# Resumable (skips existing Q4_0 / gate result). bf16 kept for a later imat-Q6 winner pass.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=$SCR/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
CORPUS=$SFT/eog_corpus_solar.jsonl
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh   # b9700-pinned canonical gate
AL=/srv/ml/agentic_loop
WORK=/mnt/sdc/ml/t223_fk
MAPDIR=/srv/ml/scripts
ts(){ date '+%T %Z'; }
mkdir -p "$WORK"

# name|drop_map|dern(0/1)
CELLS=(
  "CTRL|v8coder_fkbroad_drop_map.json|0"
  "STD8|v8coder_fk8_drop_map.json|0"
  "STD16|v8coder_fk16_drop_map.json|0"
  "STD27|v8coder_fk27_drop_map.json|0"
  "ADD16|v8coder_fk16_drop_map.json|1"
  "V8|v8coder_fkbroad_drop_map.json|1"
)

echo "==================== T223 FK CAMPAIGN $(ts) ===================="
# ── preflight ──
for f in "$SRC/config.json" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$SCR/redist_prep_v7coder.py" "$DERN" "$CORPUS" "$GATE" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-quantize" \
         "$MAPDIR/v8coder_fkbroad_drop_map.json" "$MAPDIR/v8coder_fk8_drop_map.json" \
         "$MAPDIR/v8coder_fk16_drop_map.json" "$MAPDIR/v8coder_fk27_drop_map.json"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL DERN not patched"; exit 2; }
echo "[preflight $(ts)] disk:"; df -h "$WORK" | tail -1

build_cell(){  # name map dern
  local NM=$1 MAP=$MAPDIR/$2 DF=$3
  local BF=$WORK/${NM}-bf16 Q4=$WORK/${NM}-Q4_0.gguf F16=$WORK/${NM}-F16.gguf
  if [ -f "$Q4" ]; then echo "[build $(ts)] $NM Q4_0 exists, skip"; return 0; fi
  echo "[build $(ts)] === $NM (map=$2 dern=$DF) ==="
  if [ ! -f "$BF/.shared_applied" ]; then
    rm -rf "$BF"
    "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
      || { echo "FATAL expert_drop $NM"; return 1; }
    [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer $NM"; return 1; }
    "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
      --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared $NM"; return 1; }
    touch "$BF/.shared_applied"
  else echo "[build] $NM bf16 student exists, skip drop/shared"; fi
  local SRCDIR=$BF
  if [ "$DF" = "1" ]; then
    local COMBO=$WORK/${NM}-combo KM=$WORK/${NM}-keepmeta.json
    if [ ! -f "$COMBO/.dern_done" ]; then
      [ -f "$KM" ] || "$PY" "$SCR/redist_prep_v7coder.py" "$BF" "$MAP" "$KM" \
        || { echo "FATAL keepmeta $NM"; return 1; }
      echo "[build $(ts)] $NM DERN soft-top-2 -> $COMBO (GPU0)"
      CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" "$DERN" --teacher "$SRC" --student "$BF" --keep-meta "$KM" \
          --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" \
          --device cuda:0 --assign-topk 2 || { echo "FATAL DERN $NM"; return 1; }
      touch "$COMBO/.dern_done"
    else echo "[build] $NM combo exists, skip DERN"; fi
    SRCDIR=$COMBO
  fi
  echo "[build $(ts)] $NM convert F16 + quant Q4_0 (no imat)"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$SRCDIR" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert $NM"; return 1; }
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q4" Q4_0 32 || { echo "FATAL quant $NM"; return 1; }
  local magic; magic=$("$PY" -c "print(open('$Q4','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL bad GGUF $NM"; return 1; }
  rm -f "$F16"
  echo "[build $(ts)] $NM Q4_0 done: $(stat -c%s "$Q4") bytes"
}

# ── Phase A: build all 6 Q4_0 (serial; DERN serial on GPU0) ──
echo "[A $(ts)] ===== PHASE A: build 6 Q4_0 ====="
for c in "${CELLS[@]}"; do
  IFS='|' read -r NM MAP DF <<<"$c"
  build_cell "$NM" "$MAP" "$DF" || echo "[A] WARN build $NM failed (rc=$?)"
done

# ── Phase B: 48-seed b9700 gates, 2-at-a-time across GPU0/GPU1 ──
echo "[B $(ts)] ===== PHASE B: 6 loop gates (no-imat-Q4_0, b9700) ====="
gate_cell(){  # name gpu port
  local NM=$1 G=$2 P=$3
  local Q4=$WORK/${NM}-Q4_0.gguf OUT=$AL/results/t223_${NM}_Q40_minp48.json
  [ -f "$Q4" ] || { echo "[gate] $NM Q4_0 MISSING, skip"; return 0; }
  [ -f "$OUT" ] && { echo "[gate] $NM result exists, skip"; return 0; }
  echo "[gate $(ts)] $NM GPU$G:$P -> $OUT"
  MATRIX=matrix_minp_2temp.json bash "$GATE" "$Q4" "$G" "$P" "$OUT" "t223-${NM}-q40" \
    > "$AL/logs/t223_${NM}_gate.log" 2>&1 || echo "[gate] WARN $NM rc=$?"
}
NAMES=(CTRL STD8 STD16 STD27 ADD16 V8)
mkdir -p "$AL/logs"
i=0; port=8400
while [ $i -lt ${#NAMES[@]} ]; do
  gate_cell "${NAMES[$i]}" 0 "$port" & P0=$!; port=$((port+1))
  P1=""
  if [ $((i+1)) -lt ${#NAMES[@]} ]; then
    gate_cell "${NAMES[$((i+1))]}" 1 "$port" & P1=$!; port=$((port+1))
  fi
  wait $P0; [ -n "$P1" ] && wait $P1
  i=$((i+2))
done

# ── Phase C: results table ──
echo "[C $(ts)] ===== PHASE C: results ====="
"$PY" - <<'PYEOF'
import json, os
AL="/srv/ml/agentic_loop/results"
order=["CTRL","STD8","STD16","STD27","ADD16","V8"]
print(f"{'cell':8} {'t0.9 fail/loop':>16} {'t0.8 fail/loop':>16}  (n=48)")
for nm in order:
    p=f"{AL}/t223_{nm}_Q40_minp48.json"
    if not os.path.exists(p):
        print(f"{nm:8} {'(no result)':>16}"); continue
    d=json.load(open(p))
    row={}
    for r in d.get("results",[]):
        cfg=str(r.get("config","")); t=("t0.9" if "0.9" in cfg else "t0.8" if "0.8" in cfg else cfg)
        row[t]=f"f{r.get('fails','?')}/l{r.get('loops','?')}"
    print(f"{nm:8} {row.get('t0.9','-'):>16} {row.get('t0.8','-'):>16}")
PYEOF
echo "###### T223_FK_CAMPAIGN_DONE $(ts) ######"
