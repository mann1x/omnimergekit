#!/usr/bin/env bash
# build_v7coder_eogfk_dern11_combo.sh — T206 combo: agentic_eog force-keep (K4) + DERN Eq.11.
#
# User-directed (2026-06-17): "protect only the critical experts for eog together with dern11.
# minimum protection gave us some improvement, we need to check the combo."
#
# SINGLE-VARIABLE vs dern11 (T203): identical pipeline + identical corpus (eog_corpus_solar),
# the ONLY change is the student's keep set — eogfk force-keeps the 4 EOG-critical experts/layer
# (evicting the 4 lowest-aggregate survivors) BEFORE the DERN fold. Everything else matches dern11:
#   expert_drop(eog_fk map) -> router_shared_upweight(alpha=1.2, bf16) ->
#   redist_prep_v7coder(keepmeta from eog_fk drop map) ->
#   redist_dern_eq11(teacher=128e, student=eogfk, freq=seq=eog_corpus_solar) ->
#   convert F16 (llama.cpp-latest) -> quantize Q6_K (NO imatrix, matches dern11-Q6_K-noimat baseline) ->
#   loop gate vendor_minp_rep {0.9, 0.8} (48 seeds x 2 arms).
#
# Compare combo-Q6_K-noimat vs dern11-Q6_K-noimat (t0.9=4/48, t0.8=2/48) and 128e ref (0/48).
# GPU0 throughout (redist fits one 97GB Blackwell; teacher freed before student load).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
DERN=/srv/ml/repos/omnimergekit/scripts/redist_dern_eq11.py
LCPP=/mnt/sdc/ml/llama.cpp-latest
SFT=/mnt/sdc/ml/sft_heal
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
DROP=$SFT/v7coder_eog_fk_drop_map.json
CORPUS=$SFT/eog_corpus_solar.jsonl
GATE=/srv/ml/scripts/gate_sweep48_minp_p.sh
AL=/srv/ml/agentic_loop

EOGFK=$SFT/gemma-4-A4B-98e-v7-coder-eogfk-it            # bf16 student (rebuild)
KEEPMETA=$SFT/v7coder_eogfk_keepmeta.json
COMBO=$SFT/gemma-4-A4B-98e-v7-coder-eogfk-dern11-it     # bf16 combo out
F16=$SFT/v7coder-eogfk-dern11-F16.gguf
Q6=$SFT/gemma-4-A4B-98e-v7-coder-eogfk-dern11-it-Q6_K.gguf
GPU=0

ts(){ date '+%T %Z'; }
echo "==================== build eogfk+dern11 combo $(ts) ===================="
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# preflight
for f in "$SRC/config.json" "$DROP" "$CORPUS" "$SCR/expert_drop.py" "$RECIPE/router_shared_upweight.py" \
         "$SCR/redist_prep_v7coder.py" "$DERN" "$LCPP/convert_hf_to_gguf.py" \
         "$LCPP/build/bin/llama-quantize" "$GATE"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[preflight] disk:"; df -h "$SFT" | tail -1

# ── 1. expert_drop (eogfk keep set) ───────────────────────
if [ ! -f "$EOGFK/model.safetensors.index.json" ]; then
  echo "[1 $(ts)] expert_drop -> $EOGFK"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$EOGFK" \
    || { echo "FATAL expert_drop"; exit 3; }
  [ -f "$EOGFK/tokenizer.json" ] || { echo "FATAL tokenizer.json not copied"; exit 3; }
else
  echo "[1] $EOGFK exists, skip expert_drop"
fi

# ── 2. router_shared_upweight (alpha=1.2, bf16) ───────────
if [ ! -f "$EOGFK/.shared_applied" ]; then
  echo "[2 $(ts)] router_shared_upweight --alpha 1.2 --target mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$EOGFK" \
    --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared_upweight"; exit 4; }
  touch "$EOGFK/.shared_applied"
else
  echo "[2] .shared_applied exists, skip"
fi

# ── 3. keep-meta from eog_fk drop map (schema: keep/drop/num_experts) ──
if [ ! -f "$KEEPMETA" ]; then
  echo "[3 $(ts)] redist_prep_v7coder -> $KEEPMETA"
  "$PY" "$SCR/redist_prep_v7coder.py" "$EOGFK" "$DROP" "$KEEPMETA" \
    || { echo "FATAL keepmeta prep"; exit 5; }
  [ -f "$KEEPMETA" ] || { echo "FATAL keepmeta not written"; exit 5; }
fi

# ── 4. DERN Eq.11 survivor-anchored fold (same corpus as dern11) ──
if [ ! -f "$COMBO/model.safetensors.index.json" ] && [ ! -f "$COMBO/model.safetensors" ]; then
  echo "[4 $(ts)] redist_dern_eq11 (freq=seq=eog_corpus_solar) -> $COMBO"
  "$PY" "$DERN" --teacher "$SRC" --student "$EOGFK" --keep-meta "$KEEPMETA" \
    --freq-corpus "$CORPUS" --seq-corpus "$CORPUS" --out "$COMBO" --device cuda:0 \
    || { echo "FATAL redist_dern_eq11"; exit 6; }
else
  echo "[4] $COMBO exists, skip redist"
fi

# ── 5. convert F16 ────────────────────────────────────────
if [ ! -f "$F16" ]; then
  echo "[5 $(ts)] convert -> $F16"
  "$PY" "$LCPP/convert_hf_to_gguf.py" "$COMBO" --outfile "$F16" --outtype f16 \
    || { echo "FATAL convert"; exit 7; }
  echo "  F16 size: $(du -h "$F16" | cut -f1)"
fi

# ── 6. quantize Q6_K (NO imatrix — matches dern11-Q6_K-noimat single-variable baseline) ──
if [ ! -f "$Q6" ]; then
  echo "[6 $(ts)] quantize Q6_K -> $Q6"
  "$LCPP/build/bin/llama-quantize" "$F16" "$Q6" Q6_K 32 \
    || { echo "FATAL quantize"; exit 8; }
  echo "  Q6 size: $(du -h "$Q6" | cut -f1)"
fi

# ── 7. loop gate vendor_minp_rep {0.9, 0.8} ──────────────
echo "[7 $(ts)] loop gate eogfk-dern11-Q6 {t0.9,t0.8} on GPU$GPU:8190"
bash "$GATE" "$Q6" "$GPU" 8190 "$AL/results/eogfk_dern11_q6_minp48.json" eogfk-dern11-Q6

echo "[$(ts)] === COMBO DONE ==="
echo "combo Q6: $Q6"
ls -la "$Q6" 2>/dev/null
echo "compare vs dern11-Q6_K-noimat (t0.9=4/48 t0.8=2/48), dern11-imat-Q6 (1/48,2/48), 128e ref (0/48)"
