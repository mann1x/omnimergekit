#!/usr/bin/env bash
# build_v9_agentic_sft11_build.sh — v9-agentic refA steps 6–7 on sft11: weights, AC imatrix, Q6_K + Q4_K_M.
# Runs after build_v9_agentic_sft11_looppins.sh (needs A_sft11.json).
#
# Same build as the published v9-agentic (pod_v9_agentic_publish.sh stages build/imatrix/quants); only the source
# model (sft11) and the drop map (A′) change:
#   expert_drop(sft11, A′) -> router_shared_upweight α 1.2 mlp.down_proj.weight -> official template (ae53464b)
#   -> AC imatrix: AtomicChat gemma-4-20b-v9-agentic calib_train.txt (sha 33fef70c), ALL chunks, --parse-special
#      (quantize_gguf.py --imatrix-only, the v9 publish stage verbatim) -> imatrix.dat kept next to the quants
#   -> Q6_K, Q4_K_M with that imatrix (quantize_gguf.py --only, --no-upload --keep-local)
# generation_config.json = sft11's (the 26B agentic's); the recommended deployment sampler lives in the cards.
#
# Usage: build_v9_agentic_sft11_build.sh [GPU]     refuses to overwrite a model dir, imatrix or GGUF
set -uo pipefail
GPU=${1:-1}
SRC=/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
OMK=/srv/ml/repos/omnimergekit
LC=/mnt/sdc/ml/llama.cpp-b9700
O=/mnt/sdc/ml/v9a_sft11
MAP=$O/A_sft11.json
NAME=gemma-4-20B-v9-agentic-sft11
BF=$O/$NAME
GG=$O/gguf/$NAME
TMPL=$O/inputs/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
CALIB=/mnt/sdc/ml/calib/ac_gemma4_v9official/calib_train.txt; CSHA=33fef70c1f8d5965
export CUDA_DEVICE_ORDER=PCI_BUS_ID
say() { echo "[v9a-build $(date -u +%H:%M:%SZ)] $*"; }
die() { say "V9A_BUILD_FAIL $*"; exit 1; }

[ -f "$MAP" ] || die "missing $MAP"
[ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
[ "$(sha256sum "$CALIB" | cut -c1-16)" = "$CSHA" ] || die "calib sha"
[ -e "$BF" ] && die "$BF exists — refusing to overwrite"
[ -e "$GG/imatrix.dat" ] && die "$GG/imatrix.dat exists — refusing to clobber an imatrix"
free=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "${free:-0}" -lt 150 ] && die "/mnt/sdc ${free}G free < 150G"
mkdir -p "$O/logs" "$GG"
sha256sum "$MAP" | tee "$O/A_sft11.json.sha256"

say "expert_drop sft11 -> $BF (CPU)"
CUDA_VISIBLE_DEVICES="" "$PY" "$OMK/scripts/expert_drop.py" --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
  > "$O/logs/expert_drop.log" 2>&1 || { tail -20 "$O/logs/expert_drop.log"; die "expert_drop"; }
CUDA_VISIBLE_DEVICES="" "$PY" "$OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py" --model-dir "$BF" \
  --alpha 1.2 --target mlp.down_proj.weight > "$O/logs/upweight.log" 2>&1 || { tail -20 "$O/logs/upweight.log"; die "upweight"; }
rm -f "$BF"/*.pre_shared_upweight
( cd "$BF" && sha256sum model-*.safetensors > SHA256SUMS )
cp "$TMPL" "$BF/chat_template.jinja"
cp "$SRC/generation_config.json" "$SRC/preprocessor_config.json" "$BF/"
[ "$(sha256sum "$BF/chat_template.jinja" | cut -d' ' -f1)" = "$TSHA" ] || die "template copy"
"$PY" - "$BF" <<'PY' || die "config check"
import json, sys
t = json.load(open(sys.argv[1] + "/config.json"))["text_config"]
print("[build] num_experts", t["num_experts"], "top_k", t["top_k_experts"])
assert t["num_experts"] == 98
PY
touch "$BF/BUILD_OK"; du -sh "$BF"
say "V9A_BUILD_STEP_OK weights"

say "AC imatrix (full corpus, --parse-special) on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --imatrix-only \
  --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-save-frequency 1000 --imatrix-timeout 172800 \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
  > "$O/logs/imatrix.out" 2>&1 || { tail -30 "$O/logs/imatrix.out"; die "imatrix"; }
grep -q IMATRIX_ONLY_DONE "$O/logs/imatrix.out" || die "no IMATRIX_ONLY_DONE marker"
"$PY" - "$GG/imatrix.dat" <<'PY' || die "imatrix chunk check"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields["imatrix.chunk_count"]; n = f.parts[f.data[0]].tolist()[0]
print("[imatrix] chunk_count=%d entries=%d" % (n, len(r.tensors)))
assert n >= 9500, "imatrix did not consume the full corpus"
PY
sha256sum "$GG/imatrix.dat" | tee "$GG/imatrix.dat.sha256"
say "V9A_BUILD_STEP_OK imatrix"

say "quants Q6_K,Q4_K_M with the AC imatrix on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU LLAMA_CPP_HOME=$LC "$PY" "$OMK/scripts/quantize_gguf.py" \
  --model "$BF" --output-dir "$GG" --only Q6_K,Q4_K_M \
  --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
  --base-model-id google/gemma-4-26B-A4B-it --ngl 99 --no-upload --keep-local \
  > "$O/logs/quants.out" 2>&1 || { tail -30 "$O/logs/quants.out"; die "quants"; }
n=0
for g in "$GG"/*Q6_K.gguf "$GG"/*Q4_K_M.gguf; do
  [ -s "$g" ] || continue; n=$((n + 1))
  sha256sum "$g" | tee "$g.sha256"
  "$PY" - "$g" <<'PY' || die "no imatrix KV in $g"
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
ks = [k for k in r.fields if "imatrix" in k.lower()]
print("   %s imatrix KVs: %s" % (sys.argv[1].rsplit("/", 1)[-1], ks))
assert ks
PY
done
[ "$n" -eq 2 ] || die "expected 2 GGUFs (Q6_K, Q4_K_M), found $n"
ls -la "$GG"
say "V9A_BUILD_DONE"
