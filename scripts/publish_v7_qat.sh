#!/usr/bin/env bash
# publish_v7_qat.sh — publish v7-coder + v7-coderx NOSHARED QAT-Q4_0 GGUFs to
# HF GGUF repos (clean name ...-it-qat-Q4_0.gguf) + ollama :qat-Q4_0 tag.
# Recipe: NOSHARED (drop-only). Clean published name carries noshared bytes.
# imatrix: NONE (Q4_0 imatrix-free; QAT calibration baked in) — documented, not lost.
set -uo pipefail
export PATH="/srv/ml/envs/envs/omnimergekit/bin:$PATH"
export HF_XET_HIGH_PERFORMANCE=1
PY=/srv/ml/envs/envs/omnimergekit/bin/python
PUB=/mnt/sdc/ml/eval_gguf/qat/publish
LOG=/srv/ml/logs/publish_v7_qat_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
L(){ echo "[pub $(date -u +%H:%M:%S)] $*"; }

ROWS=(
"coder|ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF|gemma-4-A4B-98e-v7-coder-it-qat-Q4_0.gguf|mannix/gemma4-98e-v7-coder|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-g15f2440-it-GGUF/_ollama_push/Modelfile.qat-Q4_0"
"coderx|ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF|gemma-4-A4B-98e-v7-coderx-it-qat-Q4_0.gguf|mannix/gemma4-98e-v7-coderx|/mnt/sdc/ml/google/gemma-4-A4B-98e-v7-coder-fs2440-it-GGUF/_ollama_push/Modelfile.qat-Q4_0"
)

L "=== PREFLIGHT ==="
hf auth whoami >/dev/null 2>&1 || { L "FATAL hf not logged in"; exit 1; }
for r in "${ROWS[@]}"; do
  IFS="|" read -r m repo gg om mf <<<"$r"; f="$PUB/$gg"
  [ -f "$f" ] || { L "FATAL missing gguf $f"; exit 1; }
  [ -f "$mf" ] || { L "FATAL missing modelfile $mf"; exit 1; }
  $PY -c "import sys;assert open(\"$f\",\"rb\").read(4)==b\"GGUF\"" || { L "FATAL bad header $f"; exit 1; }
  code=$(curl -s -o /tmp/olp_$m.html -w "%{http_code}" "https://ollama.com/$om")
  [ "$code" = "200" ] || { L "FATAL ollama model $om not on ollama.com (http $code)"; exit 1; }
  grep -q ":qat-Q4_0" /tmp/olp_$m.html && L "WARN $om already has :qat-Q4_0 (will overwrite)"
  L "ok $m  gguf=$(stat -c%s "$f"|numfmt --to=iec)  ollama=$om (page 200)"
done

L "=== HF UPLOADS ==="
for r in "${ROWS[@]}"; do
  IFS="|" read -r m repo gg om mf <<<"$r"; f="$PUB/$gg"
  L ">>> hf upload $m -> $repo : $gg"
  hf upload "$repo" "$f" "$gg" --commit-message "Add QAT-Q4_0 (noshared drop-only; Google QAT-aware base; imatrix-free)" \
     || { L "FATAL hf upload failed $m"; exit 1; }
  sz=$(curl -s "https://huggingface.co/api/models/$repo/tree/main" | $PY -c "import sys,json;d=json.load(sys.stdin);print(next((p.get(\"size\",0) for p in d if p[\"path\"]==\"$gg\"),0))" 2>/dev/null)
  L "  remote present: $gg size=$sz"
  [ "${sz:-0}" -gt 1000000000 ] || { L "FATAL remote file missing/too small ($sz)"; exit 1; }
done

L "=== OLLAMA CREATE + PUSH (:qat-Q4_0) ==="
for r in "${ROWS[@]}"; do
  IFS="|" read -r m repo gg om mf <<<"$r"
  L ">>> ollama create $om:qat-Q4_0 -f $(basename "$mf")"
  ollama create "$om:qat-Q4_0" -f "$mf" || { L "FATAL ollama create failed $m"; exit 1; }
  L ">>> ollama push $om:qat-Q4_0"
  ollama push "$om:qat-Q4_0" || { L "FATAL ollama push failed $m"; exit 1; }
  sleep 5
  curl -s "https://ollama.com/$om" | grep -q ":qat-Q4_0" \
    && L "  [ok] $om:qat-Q4_0 visible on ollama.com" \
    || L "  [WARN] qat-Q4_0 not yet visible on ollama.com (propagation lag)"
done

L "###### PUBLISH_V7_QAT_DONE $(date -u) ######"
