#!/usr/bin/env bash
# build_std12_v5_model.sh — build the 98e student on the STD12 (fk12) drop map, then
# Q6_K + Q4_K_M off ONE calib_v5 imatrix. Mirrors the STD16 (published v7-coder) build
# EXACTLY except for the drop map:
#   expert_drop(map) -> router_shared_upweight alpha=1.2 mlp.down_proj.weight -> F16 -> imatrix -> Q6_K
# (STD16 = no DERN, so no DERN here. Source: build_t223_fk_campaign.sh cell "STD16".)
#
# WHY STD12 EXISTS. v9-agentic (STD16 rung) is the best variant so far but still loops;
# STD27 scored LCB 70/77 yet was REJECTED in real use ("doesn't converge, loops, gets in
# trouble"), so more force-keep pins is not the direction. The open rung is BELOW 16.
# The fk12 map comes from gen_fk12_map.sh, whose selection rule (fkK = top-K by
# loop/clean specificity) was re-derived from expert_neuron_v8_agentic_diff.json and
# proved exact against fk8, fk16 and fk27 before use. Relative to fk8 it rescues exactly
# 5:85, 8:85, 18:102, 25:122 and evicts 5:90, 8:33, 18:106, 25:91 to hold the 30/layer
# budget.
#
# This build makes a calib_v5 imatrix so STD12 is basis-comparable to v9-agentic
# (also calib_v5) and to STD27. imatrix basis is DELIBERATELY the defaults: --cal-data defaults to the bundled
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
MAP=/srv/ml/repos/omnimergekit/scripts/v8coder_fk12_drop_map.json
WORK=/mnt/sdc/ml/std12v5
BF=$WORK/STD12-v5-bf16
GG=$WORK/STD12-v5-GGUF
TMPL=/mnt/sdc/ml/v9build/tmpl/gemma-4-A4B-98e-v7-coder-it-Q6_K.jinja
TSHA=2e4c13d90aa29be968d48e4eecfa8da22e2656d890149e7924b445865490cbac
ts(){ date -u '+%H:%M:%SZ'; }
mkdir -p "$WORK" "$GG"

echo "===== STD12-v5 BUILD START $(ts) ====="
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
echo "STD12V5_BF16_OK $(ts) size=$(du -sh "$BF" | cut -f1)"

# ---- phase 2: F16 -> imatrix(calib_v5) -> Q6_K ----
# ONE imatrix, both tiers: qg_imatrix_all computes it once and applies it to every tier
# in --only, so Q6_K and Q4_K_M share an identical calibration basis. Building them in
# two passes would make two imatrices and two bases.
echo "[2 $(ts)] qg_imatrix_all --only Q6_K,Q4_K_M (imatrix on GPU0, calib_v5 default, chunks 128)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$WRAP" \
  --model "$BF" --output-dir "$GG" --only Q6_K,Q4_K_M \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it \
  --no-upload --keep-local --ngl 99 --sanity-check \
  || { echo "FATAL quantize rc=$?"; exit 1; }

echo "[2 $(ts)] artifacts:"; ls -la "$GG"
echo "[2 $(ts)] imatrix preserved:"; ls -la "$GG"/imatrix.dat 2>/dev/null || echo "  WARNING no imatrix.dat"
# BOTH tiers must exist, and both must be probed — a run that silently produced only
# one tier would otherwise read as success.
RC2=0
for T in Q6_K Q4_K_M; do
  Q=$(ls "$GG"/*-"$T".gguf 2>/dev/null | head -1)
  if [ -z "$Q" ]; then echo "FATAL no $T produced"; RC2=1; continue; fi
  echo "[2 $(ts)] probing $T: $(basename "$Q") $(stat -c%s "$Q") bytes"
  "$PY" "$SCR/gguf_probe_header.py" "$Q" 2>&1 \
    | grep -E 'general.name|eos_token_id|chat_template *=|file_type'
  "$PY" - "$Q" <<'PYEOF'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
hits = {k: v for k, v in r.fields.items() if "imatrix" in k.lower()}
if not hits:
    sys.exit("FATAL: no imatrix KV — this tier was quantized WITHOUT the imatrix")
for k in sorted(hits):
    f = hits[k]
    try:
        v = (bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
             if f.types and f.types[0] == 8 else f.parts[f.data[0]].tolist())
    except Exception:
        v = "<?>"
    print("   %-34s %s" % (k, str(v)[:100]))
PYEOF
  [ $? -eq 0 ] || RC2=1
  sha256sum "$Q" | tee "$Q.sha256"
  echo "STD12V5_${T}=$Q"
done
[ $RC2 -eq 0 ] || { echo "FATAL tier verification failed"; exit 1; }
echo "###### STD12V5_BUILD_DONE $(ts) ######"
