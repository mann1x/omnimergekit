#!/usr/bin/env bash
# build_v9_agentic_sft11_quick.sh — quick test build of a v9-agentic drop map on the sft11 weights (bm-893).
#
# Owner 2026-11-13 20:0x: A_sft11 (refA recipe re-derived on sft11) "struggles a lot converging" — test other maps.
# First: refA verbatim (A_lift120_plus_fk12.json). A base map applies to sft11 unchanged: sft11 is a fine-tune of
# the same 128e, expert numbering is identical. Owner chose "quick build": same weights pipeline as the A_sft11
# build (build_v9_agentic_sft11_build.sh: expert_drop -> router_shared_upweight α 1.2 -> official template + sft11
# generation/preprocessor config), then a QUICK imatrix (omk calibration_datav5.txt bd925aea, all chunks, as the
# v10 quick builds) and Q4_K_M only. Not the AC imatrix: a quick test basis, not a publish basis.
#
# Usage: build_v9_agentic_sft11_quick.sh <drop_map.json> <tag> [GPU] [ALPHA]
#   ALPHA = router_shared_upweight α (default 1.2, the recipe's); "none" skips the upweight (owner 2026-11-14:
#   "regenerate ... without the shared alpha 1.2" — the bm-880 A-noshared idea on sft11).
#   -> /mnt/sdc/ml/v9a_sft11/<tag>/gemma-4-20B-v9-agentic-sft11-<tag>{,-gguf}     never overwrites
set -uo pipefail
MAP=${1:?drop map}
TAG=${2:?tag}
GPU=${3:-1}
ALPHA=${4:-1.2}
SRC=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/srv/ml/repos/omnimergekit                      # the tool tree the A_sft11 build used (same expert_drop/quantize)
LC=/mnt/sdc/ml/llama.cpp-b12386   # 2026-11-14 owner: switch to latest; b9700 CUDA IQ3_S matmul is wrong on sm_120 (bm-892)
O=/mnt/sdc/ml/v9a_sft11/$TAG
NAME=gemma-4-20B-v9-agentic-sft11-$TAG
BF=$O/$NAME
GG=$O/gguf
TMPL=/mnt/sdc/ml/v9a_sft11/inputs/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/v9a_sft11/omk/scripts/calibration_datav5.txt; CSHA=bd925aea03bec5dd
export CUDA_DEVICE_ORDER=PCI_BUS_ID
say() { echo "[v9a-quick $(date -u +%H:%M:%SZ)] $*"; }
die() { say "V9A_QUICK_FAIL $*"; exit 1; }

[ -f "$MAP" ] || die "missing $MAP"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ "$(sha256sum "$CALIB" | cut -c1-16)" = "$CSHA" ] || die "calib sha"
[ -e "$O" ] && die "$O exists — refusing to overwrite"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "${free:-0}" -lt 200 ] && die "/mnt/sdc ${free}G free < 200G (the r16k run's checkpoints share this disk)"
mkdir -p "$O/logs" "$GG"
cp "$MAP" "$O/drop_map.json"; sha256sum "$O/drop_map.json" | tee "$O/drop_map.json.sha256"
"$PY" - "$O/drop_map.json" <<'PY' || die "map check"
import json, sys
d = json.load(open(sys.argv[1])); d = d.get("drop_map", d)
m = {int(k): v for k, v in d.items() if str(k).isdigit()}
n = [len(v) for v in m.values()]
print("[map] layers", len(m), "drops/layer", sorted(set(n)), "total", sum(n))
assert len(m) == 30 and set(n) == {30}
PY

say "expert_drop sft11 -> $BF (CPU)"
CUDA_VISIBLE_DEVICES="" "$PY" "$OMK/scripts/expert_drop.py" --source-dir "$SRC" --drop-map "$O/drop_map.json" \
  --output-dir "$BF" > "$O/logs/expert_drop.log" 2>&1 || { tail -20 "$O/logs/expert_drop.log"; die "expert_drop"; }
if [ "$ALPHA" = none ]; then
  say "router_shared_upweight SKIPPED (ALPHA=none): shared expert as dropped, unscaled"
  echo "none" > "$O/shared_alpha.txt"
else
  CUDA_VISIBLE_DEVICES="" "$PY" "$OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py" --model-dir "$BF" \
    --alpha "$ALPHA" --target mlp.down_proj.weight > "$O/logs/upweight.log" 2>&1 || { tail -20 "$O/logs/upweight.log"; die "upweight"; }
  rm -f "$BF"/*.pre_shared_upweight
  echo "$ALPHA" > "$O/shared_alpha.txt"
fi
cp "$TMPL" "$BF/chat_template.jinja"
cp "$SRC/generation_config.json" "$SRC/preprocessor_config.json" "$BF/"
"$PY" - "$BF" <<'PY' || die "config check"
import json, sys
t = json.load(open(sys.argv[1] + "/config.json"))["text_config"]
print("[build] num_experts", t["num_experts"], "top_k", t["top_k_experts"])
assert t["num_experts"] == 98
PY
touch "$BF/BUILD_OK"; du -sh "$BF"
say "V9A_QUICK_STEP_OK weights"

say "quick imatrix (calibration_datav5, all chunks) on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --imatrix-only \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > "$O/logs/imatrix.out" 2>&1 || { tail -30 "$O/logs/imatrix.out"; die "imatrix"; }
grep -q IMATRIX_ONLY_DONE "$O/logs/imatrix.out" || die "no IMATRIX_ONLY_DONE marker"
"$PY" - "$GG/imatrix.dat" <<'PY' || die "imatrix check"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["imatrix.chunk_count"]; print("[imatrix] chunk_count=%d entries=%d" % (f.parts[f.data[0]].tolist()[0], len(r.tensors)))
PY
sha256sum "$GG/imatrix.dat" | tee "$GG/imatrix.dat.sha256"
say "V9A_QUICK_STEP_OK imatrix"

say "Q4_K_M with the quick imatrix on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --only Q4_K_M \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --ngl 99 --no-upload --keep-local \
  > "$O/logs/quant.out" 2>&1 || { tail -30 "$O/logs/quant.out"; die "quant"; }
g=$(ls "$GG"/*Q4_K_M.gguf 2>/dev/null | head -1)
[ -s "$g" ] || die "no Q4_K_M GGUF"
sha256sum "$g" | tee "$g.sha256"
"$PY" - "$g" <<'PY' || die "GGUF KV check"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
im, ex, eos = get("quantize.imatrix.file"), get("gemma4.expert_count"), get("tokenizer.ggml.eos_token_id")
print(f"   imatrix.file={im} chunks={get('quantize.imatrix.chunks_count')} experts={ex} eos={eos}")
assert im and ex == 98 and eos == 106
PY
ls -la "$GG"
say "V9A_QUICK_DONE $g"
