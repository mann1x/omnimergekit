#!/usr/bin/env bash
# build_v9_agentic_ab_solidpc.sh — Q4_K_M test builds of the two AGENTIC-MAP candidates for
# gemma-4-A4B-20B-v9-agentic, on solidpc (RTX 3090), delivered to the opencoti models dir.
#
# WHY. The files previously delivered as "v9-agentic-std12" / "-std27" were built from the
# v7 CODE map ladder (v8coder_fk12 / fk27), not from the agentic competence map. These two
# are the real thing: data = expert_neuron_v9_agentic_code.json, classes and weights from
# v9_classes.py, generator arguments as build_v9_dropmap.sh, and only --force-keep varies:
#   A  the September v9-agentic pins (120 end-of-turn lift pins) + the 12 loop-specific pins
#   B  the 30 broad pins + the 12 loop-specific pins
# The maps were generated on bs2 by v9_std12_candidates.sh, gated on reproducing the
# September map exactly from its own pins.
#
# Per candidate: expert_drop -> router_shared_upweight alpha=1.2 mlp.down_proj.weight ->
# F16 -> calib_v5 imatrix (128 chunks: the quick basis the earlier hand-test builds used) ->
# Q4_K_M with the shipped fixed chat template. bf16 and F16 are kept until the owner picks.
#
# Knobs (defaults reproduce the 2026-10-13 A/B builds exactly):
#   CANDS="A B"        which candidates
#   SHARED_ALPHA=1.2   router_shared_upweight alpha on mlp.down_proj.weight; "none" skips the upweight
#   SUFFIX=""          appended to the build name, e.g. SUFFIX=-noshared -> gemma-4-A4B-20B-v9-agentic-A-noshared
# e.g. the A-without-shared-upweight looping test (owner 2026-10-21):
#   CANDS=A SHARED_ALPHA=none SUFFIX=-noshared bash build_v9_agentic_ab_solidpc.sh
set -uo pipefail
SHARED_ALPHA=${SHARED_ALPHA:-1.2}
SUFFIX=${SUFFIX:-}
[ "$SHARED_ALPHA" = none ] || [ -n "$SUFFIX" ] || [ "$SHARED_ALPHA" = 1.2 ] || { echo "a non-default SHARED_ALPHA needs a SUFFIX"; exit 1; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
OMK=/shared/dev/omnimergekit
BM=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models
W=$BM/google/v9_agentic_ab
SRC=$W/src
TMPL=$BM/scripts/v9_agentic_std12_pod/tmpl/gemma-4-A4B-98e-v7-coder-it-Q6_K.jinja
TSHA=2e4c13d90aa29be968d48e4eecfa8da22e2656d890149e7924b445865490cbac
DEST=/shared/dev/opencoti/.opencoti/models
ts(){ date -u '+%H:%M:%SZ'; }
die(){ echo "[$(ts)] FATAL $*"; exit 1; }
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
for C in ${CANDS:-A B}; do
  case $C in A) MAP=$W/maps/A_lift120_plus_fk12.json;; B) MAP=$W/maps/B_broad30_plus_fk12.json;; *) die "unknown $C";; esac
  NAME=gemma-4-A4B-20B-v9-agentic-$C$SUFFIX
  BF=$W/$NAME; GG=$W/gguf-$C$SUFFIX
  echo "[$(ts)] ===== candidate $C  map $(basename "$MAP") sha $(sha256sum "$MAP" | cut -c1-16) ====="
  "$PY" - "$MAP" <<'PYEOF' || die "map gate"
import json, sys
d = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[1] + ".summary.json"))
m = d.get("drop_map", d); ks = [k for k in m if str(k).isdigit()]
assert len(ks) == 30 and {len(m[k]) for k in ks} == {30}, "map shape"
a = str(s.get("args"))
assert "expert_neuron_v9_agentic_code.json" in a, "NOT the agentic map: " + a[:200]
print("  map gate OK: 30 layers x 30 dropped, data = expert_neuron_v9_agentic_code.json")
PYEOF
  MARK="$BF/.shared_alpha_$SHARED_ALPHA"
  [ "$SHARED_ALPHA" = 1.2 ] && [ -f "$BF/.shared_applied" ] && MARK="$BF/.shared_applied"  # 2026-10-13 builds
  if [ ! -f "$MARK" ]; then
    rm -rf "$BF"
    CUDA_VISIBLE_DEVICES="" "$PY" $OMK/scripts/expert_drop.py --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" > $W/expert_drop_$C$SUFFIX.log 2>&1 || die "expert_drop $C"
    if [ "$SHARED_ALPHA" != none ]; then
      CUDA_VISIBLE_DEVICES="" "$PY" $OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir "$BF" --alpha "$SHARED_ALPHA" --target mlp.down_proj.weight > $W/upweight_$C$SUFFIX.log 2>&1 || die "upweight $C"
      rm -f "$BF"/*.pre_shared_upweight
    fi
    printf '{"candidate": "%s", "shared_alpha": "%s", "target": "mlp.down_proj.weight"}\n' "$C" "$SHARED_ALPHA" > "$BF/shared_upweight.json"
    touch "$MARK"
  fi
  echo "  shared upweight: $(cat "$BF/shared_upweight.json" 2>/dev/null || echo 'alpha 1.2 (2026-10-13 build)')"
  cp "$TMPL" "$BF/chat_template.jinja"
  grep -o '"drop_map_file": "[^"]*"' "$BF/expert_drop_metadata.json"
  echo "[$(ts)] bf16 OK $(du -sh "$BF" | cut -f1)"
  mkdir -p "$GG"
  if ! ls "$GG"/*-Q4_K_M.gguf >/dev/null 2>&1; then
    LLAMA_CPP_HOME=/opt/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py --model "$BF" --output-dir "$GG" --only Q4_K_M \
      --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" --base-model-id google/gemma-4-26B-A4B-it \
      --no-upload --keep-local --sanity-check > $W/quant_$C$SUFFIX.log 2>&1 || die "quantize $C (see quant_$C$SUFFIX.log)"
  fi
  Q=$(ls "$GG"/*-Q4_K_M.gguf | head -1); [ -s "$Q" ] || die "no Q4_K_M for $C"
  "$PY" - "$Q" <<'PYEOF' || die "no imatrix KV in $C"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
h = [k for k in r.fields if "imatrix" in k.lower()]
assert h, "quantized WITHOUT imatrix"
print("  imatrix KV:", h)
PYEOF
  OUT=$DEST/$NAME-Q4_K_M.gguf
  cp "$Q" "$OUT.part" && mv "$OUT.part" "$OUT"
  ( cd "$DEST" && sha256sum "$(basename "$OUT")" | tee "$(basename "$OUT").sha256" )
  [ "$(sha256sum "$Q" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$OUT.sha256")" ] || die "delivery sha $C"
  echo "[$(ts)] DELIVERED_$C $OUT $(stat -c%s "$OUT") B"
done
echo "[$(ts)] ###### V9_AGENTIC_AB_DONE ######"
