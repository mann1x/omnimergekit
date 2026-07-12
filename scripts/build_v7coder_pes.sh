#!/usr/bin/env bash
# build_v7coder_pes.sh — 98e analog of the 62e T141 2x2: PER-EXPERT-SCALE (PES, router.per_expert_scale)
# vs the shared-alpha=1.2 that ml0 uses. All from ml0 drop map (v7coder_C6v3lcb_drop_map.json),
# fresh expert_drop = NO shared. Variants:
#   floor  = no upweight   (A1-equiv baseline)
#   pes120 = PES alpha 1.20 (A2-equiv = 62e champion mechanism)
#   pes130 = PES alpha 1.30
# Anchor = current ml0 (shared-1.2): HE+ 0.9268 / MPE 0.8967 / LCB (running).
# Eval HE+/MPE on GPU0/port 8195 (GPU1/8196 busy w/ ml0 LCB). LCB gated on HE+/MPE.
set -uo pipefail
BM=/srv/ml; PY=$BM/envs/envs/omnimergekit/bin/python
export PATH="$BM/envs/envs/omnimergekit/bin:${PATH:-}"
export LD_LIBRARY_PATH="$BM/envs/envs/omnimergekit/lib:${LD_LIBRARY_PATH:-}"
SCR=$BM/scripts; G=/mnt/sdc/ml/google
DROP=$SCR/v7coder_C6v3lcb_drop_map.json
SRC=$BM/models/base/gemma-4-26B-A4B-it
PES=$SCR/router_per_expert_rescale.py
CONVERT=$BM/tools/llama.cpp/convert_hf_to_gguf.py
QUANT=/opt/llama.cpp/build/bin/llama-quantize
OMK=$BM/repos/omnimergekit/eval/omk_eval.py
FLOOR=$G/gemma-4-A4B-98e-v7-coder-floor-it
RES=$BM/eval_results_v7coder_pes; PORT=8195
mkdir -p "$RES"
L(){ echo "[pes $(date -u +%H:%M:%S)] $*"; }
for f in "$DROP" "$SRC" "$SCR/expert_drop.py" "$PES" "$CONVERT" "$QUANT" "$OMK"; do [ -e "$f" ] || { L "FATAL missing $f"; exit 1; }; done

quant_dir(){ local d="$1" q="$2" f16="${2%.gguf}-F16.gguf"
  [ -f "$q" ] && { L "  Q6_K exists $q"; return 0; }
  "$PY" "$CONVERT" "$d" --outfile "$f16" --outtype f16 2>&1 | tail -2
  local n=$("$PY" -c "from gguf.gguf_reader import GGUFReader; print(len(GGUFReader(\"$f16\").tensors))")
  L "  F16 tensors=$n"; [ "$n" -lt 600 ] && { L "FATAL tensors"; return 1; }
  "$QUANT" "$f16" "$q" Q6_K 2>&1 | tail -2; rm -f "$f16"; }

# [1] floor bf16 (fresh expert_drop, NO upweight)
if [ -f "$FLOOR/model.safetensors.index.json" ]; then L "[1] floor exists skip"; else
  L "[1] expert_drop -> floor (no upweight)"
  "$PY" "$SCR/expert_drop.py" --source-dir "$SRC" --drop-map "$DROP" --output-dir "$FLOOR" 2>&1 | tail -6
  [ -f "$FLOOR/model.safetensors.index.json" ] || { L "FATAL expert_drop"; exit 1; }
fi
# [2] floor Q6_K
mkdir -p "$FLOOR-GGUF"
quant_dir "$FLOOR" "$FLOOR-GGUF/floor-Q6_K.gguf" || exit 1
# [3] PES variants (in-place on floor + restore)
for alpha in 1.20 1.30; do
  tag=$(echo "$alpha" | tr -d .)
  Q6=$G/gemma-4-A4B-98e-v7-coder-pes$tag-it-GGUF/pes$tag-Q6_K.gguf
  mkdir -p "$(dirname "$Q6")"
  [ -f "$Q6" ] && { L "[3] pes$tag exists skip"; continue; }
  L "[3] PES alpha=$alpha in-place on floor"
  out=$("$PY" "$PES" --model-dir "$FLOOR" --alpha "$alpha" --target router.per_expert_scale 2>&1); echo "$out" | tail -3
  echo "$out" | grep -qE "scaled [1-9]" || { L "FATAL PES scaled 0 tensors"; "$PY" "$PES" --model-dir "$FLOOR" --restore 2>&1|tail -1; exit 1; }
  quant_dir "$FLOOR" "$Q6" || { "$PY" "$PES" --model-dir "$FLOOR" --restore; exit 1; }
  L "[3] restore floor (undo PES)"
  "$PY" "$PES" --model-dir "$FLOOR" --restore 2>&1 | tail -2
done
# [4] eval HE+/MPE (GPU0/8195); tokenizer=FLOOR for all (identical)
export CUDA_VISIBLE_DEVICES=0
for v in "floor:$FLOOR-GGUF/floor-Q6_K.gguf" \
         "pes120:$G/gemma-4-A4B-98e-v7-coder-pes120-it-GGUF/pes120-Q6_K.gguf" \
         "pes130:$G/gemma-4-A4B-98e-v7-coder-pes130-it-GGUF/pes130-Q6_K.gguf"; do
  name=${v%%:*}; q6=${v#*:}
  for tpl in humanevalplus_full multipl_e_100; do
    L ">>> eval $name $tpl (GPU0 port $PORT)"
    "$PY" "$OMK" --model "$q6" --tokenizer "$FLOOR" --template "$tpl" --backend llama --port $PORT \
      --served-name "v7coder-$name-q6k" --results-dir "$RES" 2>&1 | tail -10
  done
done
# [5] table
L "=== PES sweep vs ml0(shared-1.2): HE+0.9268 MPE0.8967 ==="
"$PY" - "$RES" <<PYEOF
import json,sys
R=sys.argv[1]
def sc(n,t):
  try: return json.load(open(f"{R}/{t}/v7coder-{n}-q6k/summary.json")).get("score")
  except: return None
print(f"  {\"variant\":10} {\"HE+164\":>8} {\"MPE100\":>8}")
print(f"  {\"ml0(shr1.2)\":10} {0.9268:>8.4f} {0.8967:>8.4f}  <- anchor")
for n in (\"floor\",\"pes120\",\"pes130\"):
  he=sc(n,\"humanevalplus_full\"); mpe=sc(n,\"multipl_e_100\")
  f=lambda x:\"  None \" if x is None else f\"{x:.4f}\"
  print(f\"  {n:10} {f(he):>8} {f(mpe):>8}\")
PYEOF
L "PES_SWEEP_DONE"
