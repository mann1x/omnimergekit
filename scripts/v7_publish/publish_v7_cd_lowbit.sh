#!/usr/bin/env bash
# publish_v7_cd_lowbit.sh — build + HF-upload the 3 new low-bit CD tiers for BOTH
# v7 models (user go 2026-06-07). Built with llama-quantize + the EXACT evaluated
# maps so the published bytes are bit-identical to what was HE+/MPE-scored.
#
#   CD-Q2_K        vanilla base + vanilla 'both' imat + CD-Q2_K   map  -> Q2_K
#   CD-Q3_K_L      vanilla base + vanilla 'both' imat + CD-Q3_K_L map  -> Q3_K_L  (replaces old 10.01GB map)
#   CD-qat-Q4_K_M  QAT base     + QAT 'both' imat     + CD-Q4_K_M map  -> Q4_K_M
#
# Per tier: build -> gguf magic check -> smoke gate (GPU0, >=3/5) -> sha256 ->
# hf upload .gguf + .sha256 -> rm local (bound disk). HF card README + ollama are
# done in follow-up steps (cards from solidpc; ollama via ollama_push_generic.sh +
# build_v7_vision.sh once these land on HF). Greedy/canonical. No secrets inline.
set -uo pipefail
GPU=${GPU:-0}
BM=/srv/ml
PY=$BM/envs/envs/omnimergekit/bin/python
export PATH=$BM/envs/envs/omnimergekit/bin:${PATH:-}
BIN=/opt/llama.cpp/build/bin
SMOKE=$BM/scripts/smoke_gguf.sh
export HF_TOKEN="$(cat /root/.cache/huggingface/token 2>/dev/null)"
: "${HF_TOKEN:?HF_TOKEN must be readable at /root/.cache/huggingface/token}"
export HF_XET_HIGH_PERFORMANCE=1
WORK=/mnt/sdc/ml/cd_lowbit_publish
mkdir -p "$WORK"
LOG=$WORK/publish_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[pub $(date -u +%H:%M:%S)] $*"; }

# model | stem | repo | VANF16 | IMAT_VAN | QATF16 | IMAT_QAT | MAPDIR
MODELS=(
"coder|gemma-4-A4B-98e-v7-coder-it|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/gemma-4-A4B-98e-v7-coder-it-F16.gguf|/mnt/sdc/ml/cd_fixed_v7/imat_matrix/imat_both.dat|/mnt/sdc/ml/qat_investig/v7coder-qat-F16.gguf|/mnt/sdc/ml/qat_investig/imat_qat_both.dat|$BM/scripts/cd_maps_v7_fixed/coder"
"coderx|gemma-4-A4B-98e-v7-coderx-it|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/gemma-4-A4B-98e-v7-coderx-it-F16.gguf|/mnt/sdc/ml/qat_investig/imat_van_coderx_both.dat|/mnt/sdc/ml/qat_investig/v7coderx-qat-F16.gguf|/mnt/sdc/ml/qat_investig/imat_qat_coderx_both.dat|$BM/scripts/cd_maps_v7_fixed/coderx"
)
# tiername | filebase | base(van|qat) | mapfile-basename
TIERS=(
"CD-Q2_K|Q2_K|van|tensor_types_CD-Q2_K.txt"
"CD-Q3_K_L|Q3_K_L|van|tensor_types_CD-Q3_K_L.txt"
"CD-qat-Q4_K_M|Q4_K_M|qat|tensor_types_CD-Q4_K_M.txt"
)

gguf_ok(){ local f="$1"; [ -s "$f" ] && [ "$("$PY" -c "print(open('$f','rb').read(4).decode('latin1'))" 2>/dev/null)" = GGUF ]; }

L "=== preflight ==="
for row in "${MODELS[@]}"; do IFS='|' read -r m stem repo van imv qat imq md <<<"$row"
  for f in "$van" "$imv" "$qat" "$imq" "$md/tensor_types_CD-Q2_K.txt" "$md/tensor_types_CD-Q3_K_L.txt" "$md/tensor_types_CD-Q4_K_M.txt"; do
    [ -e "$f" ] || { L "FATAL missing [$m] $f"; exit 1; }
  done
done
for f in "$BIN/llama-quantize" "$SMOKE"; do [ -x "$f" ] || { L "FATAL missing $f"; exit 1; }; done
command -v hf >/dev/null || { L "FATAL hf CLI not on PATH"; exit 1; }
L "preflight OK"

sport=8600
build_upload(){
  local stem="$1" repo="$2" tiername="$3" filebase="$4" f16="$5" imat="$6" map="$7"
  local out="$WORK/${stem}-${tiername}.gguf" lg="$WORK/.q_${stem}-${tiername}.log"
  L ">>> [$stem] $tiername  (filebase=$filebase, map=$(basename "$map"))"
  if ! gguf_ok "$out"; then
    "$BIN/llama-quantize" --imatrix "$imat" --tensor-type-file "$map" "$f16" "$out" "$filebase" >"$lg" 2>&1
    gguf_ok "$out" || { L "  FATAL build $stem/$tiername"; tail -5 "$lg"; return 1; }
  fi
  L "  built $(du -h "$out"|cut -f1)  $(grep -oE '[0-9.]+ BPW' "$lg"|tail -1)  ($(stat -c%s "$out") bytes = $(awk "BEGIN{printf \"%.2f\", $(stat -c%s "$out")/1e9}") GB)"
  # smoke gate (GPU0)
  local res stop; res=$(bash "$SMOKE" "$out" "$sport" "$GPU" 2048 2>&1); sport=$((sport+1))
  stop=$(echo "$res"|grep -oE '[0-9]+/5 STOP'|head -1); L "  smoke: ${stop:-?}"
  if echo "${stop:-0/5}"|grep -qE '^[0-2]/5'; then
    L "  FATAL COLLAPSE $stem/$tiername (smoke ${stop}) — NOT uploading"; return 1
  fi
  # sha256 sidecar (standard 'hash  name' form, repo-relative name)
  ( cd "$WORK" && sha256sum "${stem}-${tiername}.gguf" > "${stem}-${tiername}.gguf.sha256" )
  L "  sha256 $(awk '{print $1}' "$out.sha256")"
  # upload gguf + sha256
  L "  hf upload .gguf ..."
  hf upload "$repo" "$out" "${stem}-${tiername}.gguf" >/dev/null 2>>"$lg" || { L "  FATAL upload gguf"; tail -3 "$lg"; return 1; }
  hf upload "$repo" "$out.sha256" "${stem}-${tiername}.gguf.sha256" >/dev/null 2>>"$lg" || { L "  FATAL upload sha256"; return 1; }
  L "  UPLOADED $repo :: ${stem}-${tiername}.gguf"
  rm -f "$out"   # bound disk; keep .sha256 + log
}

rc=0
for row in "${MODELS[@]}"; do
  IFS='|' read -r m stem repo van imv qat imq md <<<"$row"
  L "########## MODEL $m ($repo) ##########"
  for trow in "${TIERS[@]}"; do
    IFS='|' read -r tiername filebase basekind mapbn <<<"$trow"
    if [ "$basekind" = qat ]; then f16="$qat"; imat="$imq"; else f16="$van"; imat="$imv"; fi
    build_upload "$stem" "$repo" "$tiername" "$filebase" "$f16" "$imat" "$md/$mapbn" || { L "[$m/$tiername] FAILED"; rc=1; }
  done
done

L "########## VERIFY on HF ##########"
for row in "${MODELS[@]}"; do
  IFS='|' read -r m stem repo rest <<<"$row"
  L "--- $repo ---"
  curl -sS "https://huggingface.co/api/models/$repo/tree/main?recursive=1" 2>/dev/null \
  | "$PY" -c "
import json,sys
d=json.load(sys.stdin)
want=['CD-Q2_K','CD-Q3_K_L','CD-qat-Q4_K_M']
for w in want:
    hit=[x for x in d if x.get('path','').endswith(w+'.gguf')]
    if hit:
        sz=((hit[0].get('lfs') or {}).get('size')) or hit[0].get('size') or 0
        print(f'   OK  {w:14s} {sz/1e9:.2f} GB')
    else:
        print(f'   MISSING {w}')
"
done
L "###### PUBLISH_CD_LOWBIT_DONE rc=$rc ######"
exit $rc
