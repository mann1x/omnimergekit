#!/usr/bin/env bash
# Rebuild the WHOLE Qwen3.8-27B-Omnimerge-v6 GGUF ladder on the AtomicChat imatrix.
#
# WHY: a census of every published tier's own GGUF KV (2026-09-09) showed the ladder is
# split by the stale Gemma-scoped IMATRIX_EXCLUDE:
#     Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M -> NO quantize.imatrix.* at all (plain)
#     Q3_K_* IQ* Q2_K                              -> imatrix, calibration_datav5.txt, 128 chunks
# So seven K tiers shipped uncalibrated, and the rest were calibrated on 128 chunks
# (~1.3% of a 5M-token corpus). This rebuild puts every _K/IQ tier on the AtomicChat
# basis at 9,703 chunks.
#
# The engine is the canonical omk quantize_gguf.py -- never a hand-rolled llama-quantize
# loop. bs2's clone was synced to origin/main 05347da first, because its stale copy still
# carried IMATRIX_EXCLUDE={Q4/Q5/Q6 K band} and would have rebuilt the SAME defect.
#
# --no-sanity-check is MANDATORY for this family: the built-in check serves with
# --reasoning-format deepseek --reasoning-budget 0, which strands the answer of a
# thinking-in-content reasoner, false-FAILs every tier, and SKIPS ALL UPLOADS.
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)Z] $*"; }

W=/mnt/sdc/ml/omnimerge-v6
KEEP=$W/superseded_calib_v5
REPO=ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF
MODEL=ManniX-ITA/Qwen3.8-27B-Omnimerge-v6
PY=/srv/ml/envs/envs/omnimergekit/bin/python3.11
ENGINE=/srv/ml/repos/omnimergekit/scripts/quantize_gguf.py
WAIT_PID="${WAIT_PID:-}"

# Q8_0 is deliberately ABSENT: it is imatrix-free by the rule (no "_K"), so it is
# already correct and a rebuild would re-upload 29 GB of bit-identical file.
# F16 is absent for the same reason -- it is not a quant.
TIERS="Q6_K_L,Q6_K,Q5_K_L,Q5_K_M,Q4_K_L,Q4_K_M,Q3_K_XL,Q3_K_L,Q3_K_M,Q3_K_S,IQ4_NL,IQ4_XS,IQ3_M,IQ3_XS,IQ3_XXS,Q2_K,IQ2_M,IQ2_S"

eval "$(grep -m1 '^export HF_TOKEN=' ~/.bashrc)"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

exec 9>/var/lock/v6_ac_rebuild.lock
flock -n 9 || { say "already running - refusing"; exit 1; }

# ── 0. wait for bs2 to be free ───────────────────────────────────────────────
if [ -n "$WAIT_PID" ]; then
  say "waiting for pid $WAIT_PID (ornith ollama publish) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "pid $WAIT_PID gone -- bs2 free"
fi

# ── 1. preflight: assert the inputs are what we think, before any 6-hour run ──
F16=$W/Qwen3.8-27B-Omnimerge-v6-F16.gguf
IMAT=$W/imatrix.dat
[ -s "$F16" ]  || { say "ABORT: no F16 at $F16"; exit 2; }
[ -s "$IMAT" ] || { say "ABORT: no imatrix at $IMAT"; exit 3; }
[ "$(head -c4 "$IMAT")" = GGUF ] || { say "ABORT: imatrix is not GGUF-format"; exit 4; }
IMAT_SHA=$(sha256sum "$IMAT" | cut -d" " -f1)
say "imatrix $(stat -c%s "$IMAT") B sha256=$IMAT_SHA"
[ "$IMAT_SHA" = 50f44d8f0944866aa3d2b0de108cb9d31324cb7aaab0a6ab2250d04cd5e29cdc ] \
  || { say "ABORT: imatrix sha256 is not the AtomicChat one measured on 2026-09-09"; exit 5; }

# The engine must be the SYNCED one. A stale IMATRIX_EXCLUDE rebuilds the exact defect
# this run exists to fix, so gate on the constant, not on a git hash.
grep -q "^IMATRIX_EXCLUDE: set\[str\] = set()" "$ENGINE" \
  || { say "ABORT: $ENGINE still has a non-empty IMATRIX_EXCLUDE -- would rebuild the defect"; exit 6; }
say "engine OK: IMATRIX_EXCLUDE is empty"

FREE=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc '0-9')
[ "$FREE" -ge 80 ] || { say "ABORT: need 80G on /mnt/sdc, have ${FREE}G"; exit 7; }
say "disk ${FREE}G free"

# ── 2. install the already-built AC Q6_K under its published name ────────────
# It was evaluated as -Q6_K-AC (9-bench + the greedy GPQA pair), so it must ship
# unchanged rather than be rebuilt. The superseded plain Q6_K is KEPT, never deleted.
mkdir -p "$KEEP"
AC=$W/Qwen3.8-27B-Omnimerge-v6-Q6_K-AC.gguf
PUB=$W/Qwen3.8-27B-Omnimerge-v6-Q6_K.gguf
if [ -s "$AC" ]; then
  if [ -s "$PUB" ] && ! cmp -s "$AC" "$PUB"; then
    mv "$PUB" "$KEEP/Qwen3.8-27B-Omnimerge-v6-Q6_K.plain-calib_v5.gguf"
    say "kept superseded plain Q6_K -> $KEEP/"
  fi
  [ -s "$PUB" ] || { cp -a "$AC" "$PUB"; say "installed AC Q6_K as $(basename "$PUB")"; }
  rm -f "$PUB.sha256"
fi

# ── 3. rebuild + upload ──────────────────────────────────────────────────────
say "launching quantize_gguf.py for: $TIERS"
"$PY" "$ENGINE" \
  --model "$MODEL" \
  --repo "$REPO" \
  --output-dir "$W" \
  --only "$TIERS" \
  --cal-data "$W/calib_train.txt" \
  --imatrix-chunks -1 \
  --no-sanity-check \
  --threads 32 \
  --ngl 99
rc=$?
say "quantize_gguf rc=$rc"
say "V6_AC_REBUILD_DONE rc=$rc"
exit $rc
