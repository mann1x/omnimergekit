#!/usr/bin/env bash
# build_eval_mix_cdiq.sh — PROPER i-quant CD tiers for v7-coder via the MIX recipe
# (generate_cd_maps_mix.py), user 2026-06-07: "protect all ffn_down / all experts".
#
# WHY the per-tensor CD-IQ failed: it wrote ffn_down=IQ3_S explicitly, DEFEATING
# llama.cpp's heuristic protection -> 0/5 collapse. The MIX recipe OMITS attn_v,
# ffn_down, ffn_down_exps, token_embd, router from the override file so the file-base
# FTYPE heuristic PROTECTS them. We then DUMP the built tensor types to PROVE the
# critical tensors were bumped above the i-quant base, then smoke + HE+/MPE100.
#
# Recipes: CD-IQ3_XS_h (~9.7 GB ref, v5=92.07%), CD-IQ2_M_h (sub-8 GB new band).
# Vanilla base + 'both' imatrix (mandatory at i-quant). Greedy/canonical.
set -uo pipefail
GPU=${GPU:-0}
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-}
BIN=/opt/llama.cpp/build/bin
GEN=$BM/repos/omnimergekit/scripts/generate_cd_maps_mix.py
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
TPL=$BM/repos/omnimergekit/eval/templates
SMOKE=$BM/scripts/smoke_gguf.sh
VANF16=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf
TOK=/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it
IMAT=/mnt/sdc/ml/cd_fixed_v7/imat_matrix/imat_both.dat
MAPDIR=/srv/ml/scripts/cd_maps_v7_mix/coder
RES=/srv/ml/eval_results_mix_cdiq
WORK=/mnt/sdc/ml/mix_cdiq
mkdir -p "$MAPDIR" "$RES" "$WORK"
RECIPES="CD-IQ3_XS_h CD-IQ2_M_h"
LOG=$WORK/mix_cdiq_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[mixcdiq $(date -u +%H:%M:%S)] $*"; }

L "=== MIX CD-IQ build+verify+eval (GPU$GPU) ==="
for f in "$VANF16" "$TOK" "$IMAT" "$GEN" "$OMK" "$SMOKE" "$BIN/llama-quantize" \
         "$TPL/humanevalplus_full.yaml" "$TPL/multipl_e_100.yaml"; do
  [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }
done
command -v lm-eval >/dev/null || { L "FATAL lm-eval not on PATH"; exit 1; }

# [1] generate the mix maps + file_base.json (body-only overrides; ffn_down* omitted)
L "[1] generate mix maps: $RECIPES"
"$PY" "$GEN" --imatrix "$IMAT" --out-dir "$MAPDIR" --recipes $RECIPES 2>&1 | tail -16
[ -f "$MAPDIR/file_base.json" ] || { L "FATAL no file_base.json"; exit 1; }
# sanity: override file must NOT contain ffn_down (it must fall through to heuristic)
for r in $RECIPES; do
  bad=$(grep -cE 'ffn_down' "$MAPDIR/tensor_types_${r}.txt" || true)
  [ "${bad:-0}" -eq 0 ] || { L "FATAL $r override lists ffn_down ($bad) — would defeat heuristic"; exit 1; }
done
L "[1] OK — no ffn_down in any override (heuristic will protect it)"

gguf_ok(){ local f="$1"; [ -s "$f" ] && [ "$("$PY" -c "print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)" = GGUF ]; }
dump_critical(){
  "$PY" - "$1" <<'PYEOF'
import sys
sys.path.insert(0, "/opt/llama.cpp/gguf-py")
from gguf import GGUFReader
from collections import Counter
r = GGUFReader(sys.argv[1])
roles = {}
WANT = ("attn_v.weight","ffn_down.weight","ffn_down_exps.weight",
        "ffn_gate_exps.weight","ffn_up_exps.weight","ffn_gate_up_exps.weight",
        "attn_q.weight","token_embd.weight")
for t in r.tensors:
    for role in WANT:
        if t.name.endswith(role):
            roles.setdefault(role, Counter())[t.tensor_type.name] += 1
for role in WANT:
    if role in roles:
        print(f"     {role:26s} -> {dict(roles[role])}")
PYEOF
}

sport=8560; eport=8570
build_one(){
  local r="$1" cd="$MAPDIR/tensor_types_${r}.txt"
  local fb; fb=$("$PY" -c "import json;print(json.load(open('$MAPDIR/file_base.json'))['$r'])")
  local served="mix-${r}"
  local out="$WORK/${served}.gguf" lg="$WORK/.q_${served}.log"
  [ -f "$RES/humanevalplus_full/${served}/summary.json" ] && { L "[$served] cached"; return 0; }
  [ -f "$RES/.${served}.collapse" ] && { L "[$served] prior collapse"; return 0; }
  [ -f "$cd" ] || { L "[$served] FATAL map $cd"; return 1; }
  L "[$served] build file-base=$fb override=$(basename "$cd") (overrides=$(wc -l <"$cd"))"
  if ! gguf_ok "$out"; then
    "$BIN/llama-quantize" --imatrix "$IMAT" --tensor-type-file "$cd" "$VANF16" "$out" "$fb" >"$lg" 2>&1
    gguf_ok "$out" || { L "  FATAL build $served"; tail -5 "$lg"; return 1; }
  fi
  L "  built $(du -h "$out"|cut -f1) $(grep -oE '[0-9.]+ BPW' "$lg"|tail -1)"
  L "  >>> CRITICAL-TENSOR PROTECTION AUDIT (ffn_down* + attn_v must be ABOVE i-quant base $fb):"
  dump_critical "$out"
  local res stop; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1); sport=$((sport+1))
  stop=$(echo "$res"|grep -oE '[0-9]+/5 STOP'|head -1); L "  smoke: ${stop:-?}"
  if echo "${stop:-0/5}"|grep -qE '^[0-2]/5'; then
    L "  COLLAPSE $served (smoke ${stop})"; echo "$stop">"$RES/.${served}.collapse"; rm -f "$out"; return 0
  fi
  local b sj
  for b in humanevalplus_full multipl_e_100; do
    sj="$RES/$b/${served}/summary.json"
    if [ ! -f "$sj" ]; then
      L "  eval $served/$b (port $eport)"
      CUDA_VISIBLE_DEVICES=$GPU "$PY" "$OMK" --model "$out" --tokenizer "$TOK" --template "$TPL/$b.yaml" \
        --backend llama --served-name "$served" --results-dir "$RES" --port "$eport" >"$WORK/.eval_${served}_${b}.log" 2>&1
      eport=$((eport+1))
    fi
    [ -f "$sj" ] && L "  RESULT $served/$b = $($PY -c "import json;print(json.load(open('$sj')).get('score'))" 2>/dev/null)"
  done
  L "  kept gguf for publish: $out ($(du -h "$out"|cut -f1))"
}

for r in $RECIPES; do build_one "$r" || L "[$r] FAILED"; done
L "###### MIX_CDIQ_DONE ######"
sc(){ local sv="$1" bn="$2"; local sj="$RES/$bn/${sv}/summary.json"; [ -f "$sj" ] && "$PY" -c "import json;print(round(json.load(open('$sj')).get('score'),4))" 2>/dev/null || ([ -f "$RES/.${sv}.collapse" ] && echo "COLLAPSE($(cat "$RES/.${sv}.collapse"))" || echo "-"); }
L "=== MIX CD-IQ RESULTS (HE+ / MPE-100) ==="
for r in $RECIPES; do
  printf "  %-16s %s\n" "$r" "$(sc mix-$r humanevalplus_full)/$(sc mix-$r multipl_e_100)  size=$(du -h "$WORK/mix-$r.gguf" 2>/dev/null|cut -f1 || echo rm)"
done
L "[done] $(date -u)"
