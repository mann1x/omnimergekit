#!/usr/bin/env bash
# Pull a3b-gepo3-Q4_K_M from bs2 and register it in Ollama next to gepo1_tb / gepo2_tb.
#
# THE SERVING CONFIG IS CLONED, NOT RETYPED. `ollama show gepo2_tb --modelfile` is the
# source: for a harness A/B, gepo3_tb must differ from gepo2_tb in WEIGHTS ONLY. Hand-
# typing 9 PARAMETER lines is how a temperature or a think_budget silently drifts and
# puts a serving difference underneath every behavioural delta. gepo1_tb and gepo2_tb
# were verified to already carry identical configs, so cloning either is equivalent.
#
# NOTE (carried, deliberate): gepo1_tb is Q6_K while gepo2/gepo3 are Q4_K_M. That was an
# explicit call -- gepo2-vs-gepo3 is a matched comparison; any gepo1 delta additionally
# carries a quantization confound and must be discounted by hand.
#
# INTEGRITY: the sha256 written at build time on bs2 is re-verified AFTER transfer on
# solidpc. A 16GB copy that silently truncates produces a GGUF that still has the magic
# bytes and still loads.
set -uo pipefail

REMOTE=bs2
RQ4=/mnt/sdc/ml/brevity/gepo/gguf_gepo3/a3b-gepo3-Q4_K_M.gguf
STAGE=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/gepo3
LQ4=$STAGE/a3b-gepo3-Q4_K_M.gguf
TAG=gepo3_tb:Q4_K_M
SRC_TAG=gepo2_tb:Q4_K_M
LOG=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/scripts/bundle_gepo3.log

say(){ echo "[$(date -u '+%F %T UTC')] $*" | tee -a "$LOG"; }
mkdir -p "$STAGE" "$(dirname "$LOG")"

say "=== bundle_gepo3: waiting for the Q4_K_M on $REMOTE ==="
DEADLINE=$(( $(date +%s) + 5*3600 ))
while :; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { say "ABORT: deadline waiting for $RQ4"; exit 1; }
  if ssh -n "$REMOTE" "[ -s '$RQ4' ] && [ -s '$RQ4.sha256' ]" 2>/dev/null; then
    # the sha256 sidecar is written AFTER the file is complete, so its presence is the
    # completion predicate -- file size alone is true mid-write too
    say "Q4_K_M present on $REMOTE"; break
  fi
  if ! ssh -n "$REMOTE" 'pgrep -f "[b]uild_a3b_gepo3.sh" >/dev/null' 2>/dev/null; then
    sleep 60
    ssh -n "$REMOTE" "[ -s '$RQ4.sha256' ]" 2>/dev/null || {
      say "REFUSE: build gone and no Q4_K_M sha256 -- build died or Q4 tier failed"
      ssh -n "$REMOTE" 'tail -n 15 /mnt/sdc/ml/brevity/gepo/build_a3b_gepo3.log' | tee -a "$LOG"
      exit 2; }
  fi
  sleep 60
done

REMOTE_SHA=$(ssh -n "$REMOTE" "cut -d' ' -f1 '$RQ4.sha256'")
RSIZE=$(ssh -n "$REMOTE" "stat -c %s '$RQ4'")
say "remote: $(numfmt --to=iec "$RSIZE")  sha256=${REMOTE_SHA:0:16}..."

say "--- transfer ---"
rsync -a --partial --inplace --info=progress2 "$REMOTE:$RQ4" "$LQ4" 2>&1 | tail -2 | tee -a "$LOG"
[ -s "$LQ4" ] || { say "FATAL: transfer produced no file"; exit 3; }

say "--- verifying sha256 after transfer ---"
LOCAL_SHA=$(sha256sum "$LQ4" | cut -d' ' -f1)
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
  say "FATAL: sha256 MISMATCH after transfer"
  say "  remote=$REMOTE_SHA"
  say "  local =$LOCAL_SHA"
  exit 4
fi
say "SHA256_OK ($(stat -c %s "$LQ4" | numfmt --to=iec))"

say "--- cloning serving config from $SRC_TAG ---"
MF=$STAGE/Modelfile
ollama show "$SRC_TAG" --modelfile 2>/dev/null \
  | grep -v '^# Modelfile generated' \
  | grep -v '^# To build a new Modelfile' \
  | grep -v '^# FROM ' \
  | sed "s|^FROM .*|FROM $LQ4|" > "$MF"
grep -c '^PARAMETER' "$MF" | sed 's/^/  PARAMETER lines: /' | tee -a "$LOG"
grep -qE '^FROM '"$LQ4"'$' "$MF" || { say "FATAL: FROM line not rewritten in $MF"; exit 5; }
grep -qE '^RENDERER qwen3\.5$' "$MF" || { say "FATAL: RENDERER lost from cloned Modelfile"; exit 5; }

say "--- ollama create $TAG ---"
ollama create "$TAG" -f "$MF" 2>&1 | tail -5 | tee -a "$LOG"

say "--- verify: config must match $SRC_TAG except FROM ---"
norm(){ ollama show "$1" --modelfile 2>/dev/null | grep -vE '^#|^FROM ' | sort; }
if diff <(norm "$SRC_TAG") <(norm "$TAG") > "$STAGE/config_diff.txt"; then
  say "CONFIG_IDENTICAL_OK -- gepo3_tb differs from gepo2_tb in weights only"
else
  say "!!! CONFIG DIFFERS from $SRC_TAG -- harness A/B would be confounded:"
  cat "$STAGE/config_diff.txt" | tee -a "$LOG"
fi

# SMOKE VIA THE API, NOT `ollama run`. `ollama run` writes an ANSI progress spinner to
# stdout, so capturing it yields escape codes and proves only that the binary ran -- the
# 2026-08-27 version of this check "passed" while capturing zero generated text. The API
# returns parseable JSON. num_predict must be generous: this is a THINKING model, and a
# tight budget truncates inside the reasoning channel and returns EMPTY content with
# done_reason=length, which reads as a broken model when it is a broken test.
say "--- smoke: real generation via /api/chat ---"
SMOKE=$(curl -sf http://127.0.0.1:11434/api/chat -d "{\"model\":\"$TAG\",\"stream\":false,\"options\":{\"num_predict\":900,\"temperature\":0},\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python one-liner that reverses a string. Code only.\"}]}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin); m=d.get('message',{})
c=(m.get('content') or '').strip()
print('done_reason=%s eval_count=%s thinking_len=%d content=%r'
      % (d.get('done_reason'), d.get('eval_count'), len(m.get('thinking') or ''), c[:160]))
raise SystemExit(0 if c else 1)
") || { say "FATAL: smoke produced EMPTY content -- $SMOKE"; exit 6; }
say "SMOKE_OK $SMOKE"

say "=== bundle ==="
ollama list | awk 'NR==1 || /gepo/' | tee -a "$LOG"
say "BUNDLE_GEPO3_DONE"
