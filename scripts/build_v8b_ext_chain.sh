#!/usr/bin/env bash
# build_v8b_ext_chain.sh — build + gate the two extended-science v8b variants.
# cap1 (+30) and cap2 (+60): v8b-safe keep-set extended with N best loop-safe
# generic_science per layer into FREE non-pin slots (drop maps from extend_v8b_safe.py).
# Recipe IDENTICAL to v8b-safe (build_v8b_safe_p1.sh), only the drop map + names differ:
#   expert_drop(128e teacher, cap drop map) -> shared a=1.2 down_proj
#   -> redist_prep keepmeta -> DERN Eq.11 --assign-topk 2 -> F16
#   -> calib_both imatrix (128 chunks, ngl99, PRESERVED) -> imat-Q6_K.
# Phase 1: build BOTH Q6 sequentially (disk-bound; rm COMBO bf16 after each Q6).
# Phase 2: gate BOTH in parallel (GPU0=cap1, GPU1=cap2): 48-seed agentic loop gate
#          (b9700, vendor_minp_rep) + GPQA G_gap (eval_suite_llama, greedy).
# Resumable (skip on existing artifacts), PID-kill only, imatrix never deleted.
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=$SCR/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest                 # build binary (matches v8/v8b-safe)
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it        # 128e teacher + tokenizer
CORPUS=$SFT/eog_corpus_solar.jsonl
CALIB=/mnt/sdc/ml/qat_investig/calib_both.txt
SWEEP=/mnt/sdc/ml/gpqa_dissect/v8b_sweep
AL=/srv/ml/agentic_loop
GATE=/srv/ml/scripts/gate_sweep48_minp_p_b9700.sh
# GPQA eval (b9700, greedy, frozen template)
OMK_ROOT=/srv/ml/repos/omnimergekit-canonical
OMK_TOKENIZER=/srv/ml/google/gemma-4-26B-A4B-it
LLAMA_BIN=/mnt/sdc/ml/llama.cpp-b9700/build/bin
SUITE=$OMK_ROOT/eval/eval_suite_llama.sh
DIS=/mnt/sdc/ml/gpqa_dissect
GAP=$DIS/gpqa_gap.json
SCORER=$DIS/gpqa_score_subset.py
ts(){ date '+%T %Z'; }

echo "==================== build+gate v8b-ext cap1+cap2 $(ts) ===================="
for f in "$SRC/config.json" "$CORPUS" "$CALIB" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$SCR/redist_prep_v7coder.py" "$DERN" \
         "$LCPP/convert_hf_to_gguf.py" "$LCPP/build/bin/llama-imatrix" \
         "$LCPP/build/bin/llama-quantize" "$GATE" "$SUITE" "$GAP" "$SCORER" \
         "$LLAMA_BIN/llama-server" \
         "$SWEEP/v8b_ext_free_cap1_drop_map.json" "$SWEEP/v8b_ext_free_cap2_drop_map.json"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
grep -q -- "--assign-topk" "$DERN" || { echo "FATAL redist not patched"; exit 2; }
echo "[preflight $(ts)] disk:"; df -h "$SFT" | tail -1

acquire_gpu(){ # echo first free GPU (<2000 MiB), wait up to 2h
  local g U
  for _ in $(seq 1 120); do
    for g in 0 1; do
      U=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc '0-9')
      [ "${U:-99999}" -lt 2000 ] && { echo "$g"; return 0; }
    done
    sleep 60
  done
  return 1
}

build_one(){ # cap
  local cap=$1
  local DROP=$SWEEP/v8b_ext_free_cap${cap}_drop_map.json
  local BF=$SFT/gemma-4-A4B-98e-v8b-ext-cap${cap}-it
  local COMBO=$SFT/gemma-4-A4B-98e-v8b-ext-cap${cap}-soft2-it
  local KEEPMETA=$SFT/v8b_ext_cap${cap}_keepmeta.json
  local F16=$SFT/v8b-ext-cap${cap}-soft2-F16.gguf
  local IMAT=$SFT/v8b-ext-cap${cap}-soft2-imatrix.dat
  local Q6=$SFT/gemma-4-A4B-98e-v8b-ext-cap${cap}-soft2-imat-Q6_K.gguf
  if [ -f "$Q6" ]; then echo "[cap$cap $(ts)] Q6 exists, skip build"; return 0; fi
  echo "[cap$cap $(ts)] BUILD start  drop=$DROP"
  # 1. expert_drop
  if [ ! -f "$COMBO/model.safetensors" ] && [ ! -f "$COMBO/model.safetensors.index.json" ]; then
    if [ ! -f "$BF/model.safetensors" ] && [ ! -f "$BF/model.safetensors.index.json" ]; then
      echo "[cap$cap.1 $(ts)] expert_drop -> $BF"
      "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$BF" \
        || { echo "FATAL cap$cap expert_drop"; return 3; }
      [ -f "$BF/tokenizer.json" ] || { echo "FATAL cap$cap tokenizer.json"; return 3; }
    fi
    # 2. shared a=1.2
    if [ ! -f "$BF/.shared_applied" ]; then
      echo "[cap$cap.2 $(ts)] router_shared_upweight a=1.2 down_proj"
      "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
        --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL cap$cap shared"; return 4; }
      touch "$BF/.shared_applied"
    fi
    # 3. keep-meta
    [ -f "$KEEPMETA" ] || { echo "[cap$cap.3 $(ts)] redist_prep -> keepmeta";
      "$PY" "$SCR/redist_prep_v7coder.py" "$BF" "$DROP" "$KEEPMETA" \
        || { echo "FATAL cap$cap keepmeta"; return 5; }; }
    # 4. DERN Eq.11 soft-top-2
    local GPU; GPU=$(acquire_gpu) || { echo "FATAL cap$cap no GPU"; return 6; }
    echo "[cap$cap.4 $(ts)] DERN --assign-topk 2 (GPU$GPU) -> $COMBO"
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$DERN" --teacher "$SRC" --student "$BF" --keep-meta "$KEEPMETA" \
        --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
        --assign-topk 2 || { echo "FATAL cap$cap DERN"; return 7; }
    echo "[cap$cap.4 $(ts)] DERN done; rm student $BF"; rm -rf "$BF"
  fi
  # 5. convert F16
  [ -f "$F16" ] || { echo "[cap$cap.5 $(ts)] convert F16";
    "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBO" --outfile "$F16" --outtype f16 \
      || { echo "FATAL cap$cap convert"; return 8; }; }
  # 6. imatrix (PRESERVED)
  if [ ! -f "$IMAT" ]; then
    local GPU; GPU=$(acquire_gpu) || { echo "FATAL cap$cap no GPU imat"; return 9; }
    echo "[cap$cap.6 $(ts)] imatrix calib_both 128 ngl99 (GPU$GPU) -> $IMAT"
    CUDA_VISIBLE_DEVICES=$GPU "$LCPP/build/bin/llama-imatrix" \
      -m "$F16" -f "$CALIB" -o "$IMAT" --chunks 128 -ngl 99 \
      > "$SFT/v8b_ext_cap${cap}_imatrix_build.log" 2>&1 \
      || { echo "FATAL cap$cap imatrix"; tail -25 "$SFT/v8b_ext_cap${cap}_imatrix_build.log"; return 9; }
  fi
  echo "[cap$cap.6 $(ts)] imatrix $(stat -c%s "$IMAT" 2>/dev/null) bytes"
  # 7. imat-Q6
  echo "[cap$cap.7 $(ts)] quantize imat-Q6_K -> $Q6"
  "$LCPP/build/bin/llama-quantize" --imatrix "$IMAT" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL cap$cap quant"; return 10; }
  local magic; magic=$("$PY" -c "import sys;print(open('$Q6','rb').read(4).decode('latin1'))" 2>/dev/null)
  [ "$magic" = "GGUF" ] || { echo "FATAL cap$cap bad GGUF header"; return 10; }
  rm -f "$F16"
  # disk hygiene: drop COMBO bf16 (reproducible from drop map); keep imatrix+Q6
  echo "[cap$cap.7 $(ts)] Q6 ready; rm COMBO bf16 $COMBO"; rm -rf "$COMBO"
  echo "[cap$cap $(ts)] BUILD done: $(ls -la "$Q6" | sed 's/  */ /g')"
}

gate_one(){ # cap gpu gate_port gpqa_port
  local cap=$1 gpu=$2 gport=$3 qport=$4
  local Q6=$SFT/gemma-4-A4B-98e-v8b-ext-cap${cap}-soft2-imat-Q6_K.gguf
  local NAME=v8b-ext-cap${cap}-soft2-imatq6
  local GRES=$AL/results/${NAME}_minp48.json
  local WS=$DIS/ws_extcap${cap}
  echo "[gate.cap$cap $(ts)] 48-seed loop gate GPU$gpu:$gport"
  bash "$GATE" "$Q6" "$gpu" "$gport" "$GRES" "$NAME" > "$DIS/extcap${cap}.gate.log" 2>&1 \
    || echo "[gate.cap$cap] WARN loop-gate rc=$?"
  echo "[gate.cap$cap $(ts)] GPQA G_gap GPU$gpu:$qport"
  OMK_GPUS=$gpu OMK_GPU_WAIT_S=300 OMK_WS=$WS OMK_ROOT=$OMK_ROOT OMK_TOKENIZER=$OMK_TOKENIZER LLAMA_BIN=$LLAMA_BIN \
    bash "$SUITE" --variant "extcap${cap}_q6" --gguf "$Q6" --port "$qport" --only gpqa_diamond_full \
    > "$DIS/extcap${cap}.gpqa.log" 2>&1 || echo "[gate.cap$cap] WARN gpqa rc=$?"
  echo "[gate.cap$cap $(ts)] gate+gpqa done"
}

# ===== Phase 1: build both Q6 sequentially (disk-bound) =====
build_one 1 || { echo "cap1 build FAILED"; exit 11; }
build_one 2 || { echo "cap2 build FAILED"; exit 12; }
echo "==================== Phase 1 builds done $(ts) ===================="
df -h "$SFT" | tail -1

# ===== Phase 2: gate both in parallel (GPU0=cap1, GPU1=cap2) =====
( gate_one 1 0 8201 8202 ) &
P1=$!
( gate_one 2 1 8203 8204 ) &
P2=$!
wait $P1 $P2
echo "==================== Phase 2 gates done $(ts) ===================="

# ===== Phase 3: comparison table =====
S="samples_gpqa_diamond_cot_zeroshot_*.jsonl"
C1="$DIS/ws_extcap1/eval_results_llama_suite/extcap1_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
C2="$DIS/ws_extcap2/eval_results_llama_suite/extcap2_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
V8B="$DIS/ws_v8bsafe/eval_results_llama_suite/v8bsafe_q6/gpqa_diamond_full/*/lm_eval_out/*/$S"
echo "==================== GPQA G_gap RECOVERY (greedy) $(ts) ===================="
"$PY" "$SCORER" "$GAP" \
  "v8b_safe(ref65.15)=$V8B" \
  "extcap1(+30)=$C1" \
  "extcap2(+60)=$C2"
echo "==================== loop-gate results $(ts) ===================="
for cap in 1 2; do
  R=$AL/results/v8b-ext-cap${cap}-soft2-imatq6_minp48.json
  echo "cap$cap:"
  [ -f "$R" ] && "$PY" -c "import json;d=json.load(open('$R'));[print('  ',r['config'],'loops',r.get('loops'),'/',r['seeds']) for r in d['results']]" 2>/dev/null || echo "  (no gate result)"
done
echo "==================== CHAIN DONE $(ts) ===================="
