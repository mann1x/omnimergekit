#!/usr/bin/env bash
# std16_ollama_smoke.sh — builds 2 STD16 smoke models on ollama (Q6_K@0.9, Q8_0@0.8, deploy
# sampler) and runs the loop-transfer + think-level-budget probe. NOT all 13 tiers — just the 2
# that decide whether the b9700 loop gate transfers to ollama 0.30.10. Cleans up after.
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
OL=/usr/local/bin/ollama
GG=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
WORK=/srv/ml/std16_gate
mkdir -p "$WORK"
LOG="$WORK/ollama_smoke.log"
exec >>"$LOG" 2>&1
ts(){ date -u +%T; }
echo "==================== STD16 ollama smoke start $(ts) UTC ($($OL --version 2>/dev/null | head -1)) ===================="

# storage preflight: ollama create COPIES each GGUF into the blob store (~17.8 + 21.2 = 39 GB)
OLDIR="${OLLAMA_MODELS:-/root/.ollama/models}"
avail=$(df -BG --output=avail "$OLDIR" 2>/dev/null | tail -1 | tr -dc 0-9)
echo "[$(ts)] ollama store $OLDIR avail=${avail:-?}G"
[ "${avail:-0}" -ge 60 ] || { echo "[$(ts)] FATAL: <60G free in ollama store — abort (would fill disk)"; exit 1; }

mk(){ # tier temp tag
  local T="$1" TEMP="$2" TAG="$3" MF="$WORK/Modelfile.$3"
  [ -f "$GG/gemma-4-A4B-98e-v7-coder-it-$T.gguf" ] || { echo "[$(ts)] FATAL: missing $T gguf"; exit 1; }
  cat > "$MF" <<EOF
FROM $GG/gemma-4-A4B-98e-v7-coder-it-$T.gguf
PARAMETER temperature $TEMP
PARAMETER top_p 0.95
PARAMETER top_k 64
PARAMETER min_p 0.05
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
EOF
  echo "[$(ts)] ollama create $TAG (from $T, temp $TEMP)"
  $OL create "$TAG" -f "$MF" 2>&1 | tail -2
}
mk Q6_K 0.9 std16-smoke-q6
mk Q8_0 0.8 std16-smoke-q8
echo "[$(ts)] models:"; $OL list 2>/dev/null | grep std16-smoke || true

echo "[$(ts)] running smoke harness..."
$PY "$WORK/std16_ollama_smoke.py" std16-smoke-q6 0.9 std16-smoke-q8 0.8

echo "[$(ts)] cleanup smoke models + blobs"
# capture digests BEFORE rm so we can force-purge orphan blobs (ollama has no prune)
for TAG in std16-smoke-q6 std16-smoke-q8; do
  MAN="$OLDIR/manifests/registry.ollama.ai/library/$TAG/latest"
  digs=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['config']['digest'].split(':')[1]);[print(l['digest'].split(':')[1]) for l in d['layers']]" "$MAN" 2>/dev/null || true)
  $OL rm "$TAG" 2>/dev/null || true
  for h in $digs; do rm -f "$OLDIR/blobs/sha256-$h" "$OLDIR/blobs/sha256-$h-partial"* 2>/dev/null || true; done
done
echo "[$(ts)] ==================== STD16 ollama smoke DONE $(ts) ===================="
echo "STD16_OLLAMA_SMOKE_FIN"
