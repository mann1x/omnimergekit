#!/usr/bin/env bash
# PPL control: base Jprime-p3 vs GRPO-trained checkpoint-278, SAME quant tier
# (Q6_K), SAME AC imatrix corpus, SAME chunk count, SAME settings.
#
# WHY: the trained arm's full 9438-chunk imatrix run reported PPL 218.1752. The
# base's imatrix_run.log was not retained, so that number has NO control and is
# uninterpretable on its own. A PPL needs its realized floor.
# Both arms are re-run here at --chunks 300 so they cover the IDENTICAL corpus
# prefix; the absolute value will differ from 218 (different chunk count), but
# the BASE-vs-TRAINED delta is the measurement.
#
# Writes imatrix output to SCRATCH paths. The preserved imatrix.dat files next to
# the quants are mandatory archival and are never touched.
set -uo pipefail
BIN=/opt/llama.cpp/build/bin/llama-imatrix
CAL=/mnt/sdc/ornith-pod/workspace_root/calib-ac/builds/gemma-4-26b-a4b/calib_train.txt
SCRATCH=/mnt/sdc/v7rework/ppl_control; mkdir -p "$SCRATCH"
CHUNKS=300

BASE=/mnt/sdc/v7rework/quants_ac/Jprime-p3-bf16/Jprime-p3-bf16-Q6_K.gguf
TRAINED=/mnt/sdc/v7rework/quants_ac/Jprime-grpo-eff-278-bf16/Jprime-grpo-eff-278-bf16-Q6_K.gguf
for f in "$BIN" "$CAL" "$BASE" "$TRAINED"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

run() {  # name gguf gpu
  CUDA_VISIBLE_DEVICES=$3 "$BIN" -m "$2" -f "$CAL" \
      -o "$SCRATCH/$1.imatrix.scratch" -ngl 99 --chunks "$CHUNKS" --parse-special \
      > "$SCRATCH/$1.log" 2>&1
  echo "$1 rc=$?"
}
echo ">>> $(date -u +%FT%TZ) launching both on separate GPUs, $CHUNKS chunks each"
run base    "$BASE"    0 &
run trained "$TRAINED" 1 &
wait

echo; echo "=== RESULT ==="
for n in base trained; do
  printf "  %-9s %s\n" "$n" "$(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$SCRATCH/$n.log" | tail -1)"
done
echo ">>> PPL_CONTROL_DONE"
