#!/usr/bin/env bash
# Is a TRL checkpoint dir COMPLETE? Exit 0 = complete, 1 = not.
#
# Extracted from chain_gepo4_2ck.sh so the gate can be POLARITY-TESTED. Inline, it sat
# behind the live-trainer refusal and could never be exercised -- a gate nothing ever
# fires is indistinguishable from a gate that always passes.
# [[feedback_a_check_gold_fails_is_a_broken_check]]
#
# A complete save is 12 files; the two large ones must be at full size, because the
# failure this guards against is a kill landing MID-SAVE, which yields a short or
# missing adapter/optimizer rather than a missing directory.
set -uo pipefail
d=${1:?usage: gate_ck_complete.sh <checkpoint-dir>}
[ -d "$d" ] || { echo "INCOMPLETE $d: not a directory"; exit 1; }
n=$(ls -1 "$d" 2>/dev/null | wc -l)
a=$(stat -c %s "$d/adapter_model.safetensors" 2>/dev/null || echo 0)
o=$(stat -c %s "$d/optimizer.pt" 2>/dev/null || echo 0)
[ "$n" -eq 12 ]          || { echo "INCOMPLETE $d: files=$n want 12"; exit 1; }
[ "$a" -ge 169000000 ]   || { echo "INCOMPLETE $d: adapter=$a want >=169000000"; exit 1; }
[ "$o" -ge 339000000 ]   || { echo "INCOMPLETE $d: optimizer=$o want >=339000000"; exit 1; }
echo "COMPLETE $d (files=$n adapter=$a optimizer=$o)"
