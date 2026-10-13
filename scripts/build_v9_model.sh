#!/usr/bin/env bash
# build_v9_model.sh — build the 98e student on the v9-agentic drop map, then Q6_K + calib_v5
# imatrix. Mirrors the STD16 (published v7-coder) build EXACTLY except for the drop map:
#   expert_drop(map) -> router_shared_upweight alpha=1.2 mlp.down_proj.weight -> F16 -> imatrix -> Q6_K
# (STD16 = no DERN, so no DERN here. Source: build_t223_fk_campaign.sh cell "STD16".)
#
# imatrix basis is DELIBERATELY the defaults: --cal-data defaults to the bundled
# calibration_datav5.txt (plain text, so --parse-special is inert) and chunks stays 128.
# Raising chunks would rebase the imatrix and break comparability with the published tier.
#
# Chat template is dumped FROM THE SHIPPED v7-coder Q6_K (not a local jinja, which is stale:
# shipped 18,051 chars sha 2e4c13d9 vs local 19,177 chars sha d9f21aac).
set -uo pipefail
PY=/root/anaconda3/envs/omnimergekit/bin/python
SCR=/srv/ml/repos/omnimergekit/scripts
RECIPE=/srv/ml/repos/omnimergekit/recipes/gemma4/v5_moe_sweep
WRAP=/srv/ml/scripts/qg_imatrix_all.py
SRC=/srv/ml/models/base/gemma-4-26B-A4B-it
MAP=/mnt/sdc/ml/google/v9_agentic_drop_map.json
WORK=/mnt/sdc/ml/v9build
BF=$WORK/v9-agentic-bf16
GG=$WORK/v9-agentic-GGUF
TMPL=$WORK/tmpl/gemma-4-A4B-98e-v7-coder-it-Q6_K.jinja
TSHA=2e4c13d90aa29be968d48e4eecfa8da22e2656d890149e7924b445865490cbac
ts(){ date -u '+%H:%M:%SZ'; }
mkdir -p "$WORK" "$GG"

echo "===== V9 BUILD START $(ts) ====="
for f in "$SRC/config.json" "$MAP" "$SCR/expert_drop.py" \
         "$RECIPE/router_shared_upweight.py" "$WRAP" "$TMPL"; do
  [ -e "$f" ] || { echo "FATAL missing $f"; exit 2; }
done
echo "[preflight $(ts)] disk:"; df -h "$WORK" | tail -1
echo "[preflight $(ts)] map sanity:"
"$PY" -c "
import json; d=json.load(open('$MAP')); d=d.get('drop_map',d)
ks=[k for k in d if str(k).lstrip('-').isdigit()]
n={len(d[k]) for k in ks}
print('  layers=%d dropped/layer=%s kept=%d' % (len(ks), sorted(n), 128-list(n)[0]))
assert len(ks)==30 and n=={30}, 'UNEXPECTED MAP SHAPE'
" || exit 2

# ---- phase 1: expert drop + shared-FFN upweight ----
if [ ! -f "$BF/.shared_applied" ]; then
  echo "[1 $(ts)] expert_drop -> $BF"
  rm -rf "$BF"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
    || { echo "FATAL expert_drop"; exit 1; }
  [ -f "$BF/tokenizer.json" ] || { echo "FATAL no tokenizer in $BF"; exit 1; }
  echo "[1 $(ts)] router_shared_upweight alpha=1.2 mlp.down_proj.weight"
  "$PY" "$RECIPE/router_shared_upweight.py" --model-dir "$BF" \
    --alpha 1.2 --target mlp.down_proj.weight || { echo "FATAL shared upweight"; exit 1; }
  touch "$BF/.shared_applied"
else
  echo "[1 $(ts)] bf16 student exists, skipping drop+shared"
fi
echo "V9BUILD_BF16_OK $(ts) size=$(du -sh "$BF" | cut -f1)"

# ---- phase 2: F16 -> imatrix(calib_v5) -> Q6_K ----
echo "[2 $(ts)] qg_imatrix_all --only Q6_K (imatrix on GPU0, calib_v5 default, chunks 128)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$WRAP" \
  --model "$BF" --output-dir "$GG" --only Q6_K \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it \
  --no-upload --keep-local --ngl 99 --sanity-check \
  || { echo "FATAL quantize rc=$?"; exit 1; }

echo "[2 $(ts)] artifacts:"; ls -la "$GG"
echo "[2 $(ts)] imatrix preserved:"; ls -la "$GG"/imatrix.dat 2>/dev/null || echo "  WARNING no imatrix.dat"
Q=$(ls "$GG"/*Q6_K*.gguf 2>/dev/null | head -1)
[ -n "$Q" ] || { echo "FATAL no Q6_K produced"; exit 1; }
echo "[2 $(ts)] probing built tier:"
"$PY" "$SCR/gguf_probe_header.py" "$Q" 2>&1 | grep -E 'general.name|eos_token_id|chat_template *=|file_type'
echo "V9BUILD_Q6K=$Q"
echo "###### V9BUILD_DONE $(ts) ######"
