#!/usr/bin/env bash
# smoke_std16_imatrix_all.sh — de-risk the imatrix-everywhere build BEFORE the full cohort.
# Sets up the clean-named STD16 cohort dir (reuse F16/Q6/imatrix via hardlink) and builds
# ONE crossover-noimat tier (Q5_K_M) through qg_imatrix_all.py. Decisive check: did Q5_K_M
# get built WITH the imatrix? CPU-only (--ngl 0), --no-upload — never touches HF or the GPUs.
set -uo pipefail
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
PY=/srv/ml/envs/envs/omnimergekit/bin/python
SRC=/mnt/sdc/ml/t223_fk
WORK=/mnt/sdc/ml/std16_cohort
GGUFDIR=$WORK/gemma-4-A4B-98e-v7-coder-it-GGUF
WRAP=/srv/ml/scripts/qg_imatrix_all.py
MODEL=$WORK/gemma-4-A4B-98e-v7-coder-it
export OMK_NO_README=1
ts(){ date '+%T %Z'; }

echo "[smoke $(ts)] setup clean-named cohort dir"
mkdir -p "$GGUFDIR"
ln -sfn "$SRC/STD16-bf16" "$MODEL"                                        # clean-named model -> bf16 student
ln -f "$SRC/STD16-imatrix.dat"  "$GGUFDIR/imatrix.dat"                        2>/dev/null || cp -f "$SRC/STD16-imatrix.dat" "$GGUFDIR/imatrix.dat"
ln -f "$SRC/STD16-F16.gguf"     "$GGUFDIR/gemma-4-A4B-98e-v7-coder-it-F16.gguf" 2>/dev/null || true
ln -f "$SRC/STD16-imatq6.gguf"  "$GGUFDIR/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf" 2>/dev/null || true
ls -la "$GGUFDIR/"
[ -e "$MODEL/config.json" ] || { echo "FATAL: model symlink broken ($MODEL)"; exit 2; }

echo "[smoke $(ts)] build Q5_K_M (crossover would NOIMAT this) via imatrix-all wrapper"
"$PY" "$WRAP" \
    --model "$MODEL" \
    --output-dir "$GGUFDIR" \
    --only Q5_K_M \
    --base-model-id ManniX-ITA/gemma-4-A4B-98e-v7-coder-it \
    --no-upload --keep-local \
    --ngl 0 \
    --sanity-check > "$WORK/smoke_q5km.log" 2>&1
rc=$?
echo "[smoke $(ts)] quantize rc=$rc"
echo "--- imatrix / sanity signal lines ---"
grep -iE "IMATRIX_ALL|Using pre-staged imatrix|imatrix|--imatrix|Q5_K_M|sanity|PASS|FAIL|capital" "$WORK/smoke_q5km.log" | head -40

OUT="$GGUFDIR/gemma-4-A4B-98e-v7-coder-it-Q5_K_M.gguf"
echo "[smoke $(ts)] === verdict ==="
if [ "$rc" -eq 0 ] && [ -f "$OUT" ]; then
  magic=$("$PY" -c "import sys;sys.stdout.write(open('$OUT','rb').read(4).decode('latin1'))" 2>/dev/null)
  used=$(grep -ciE "imatrix" "$WORK/smoke_q5km.log")
  echo "Q5_K_M: $(stat -c%s "$OUT") bytes, magic='$magic', imatrix-log-mentions=$used"
  [ "$magic" = "GGUF" ] && echo "STD16_SMOKE_PASS" || echo "STD16_SMOKE_FAIL (bad magic)"
else
  echo "STD16_SMOKE_FAIL (rc=$rc, out present=$([ -f "$OUT" ] && echo yes || echo no))"
  tail -15 "$WORK/smoke_q5km.log"
fi
