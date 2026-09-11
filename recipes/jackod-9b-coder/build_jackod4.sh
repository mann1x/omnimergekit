#!/bin/bash
# JackOD4-9B-Coder, 2026-09-11 (user weights): Delta 0.55 / Qwopus 0.30 / Ornith 0.15.
# Literal fractions -- --base == --task-base == Qwen3.5-9B, the shared ancestor.
# Same shape as build_two_new.sh ARM 2 so the arm is comparable: identical method,
# density, darex-q, seed, skip-patterns, and the SAME patch_jackod.py root-eos fix
# (Qwen3.5-9B and DeltaCoder-applied carry no root eos_token_id -> GGUF control-token
# leak class). Root eos + tokenizer + chat template come from Qwopus, as JackOD serves.
set -uo pipefail
export PATH=/root/anaconda3/envs/omnimergekit/bin:$PATH
M=/mnt/sdc/oxopus/models
D=/mnt/sdc/oxopus/DeltaCoder-9B-applied
OMK=/srv/ml/repos/omnimergekit/omnimergekit.py
COMMON="--method omnimerge_v2 --density 0.53 --darex-q 0.75 --seed 42 --no-auto-mlp-skip --skip-patterns visual.,mtp."
O=/mnt/sdc/oxopus/JackOD4-9B-Coder

for d in "$M/Qwen3.5-9B" "$M/Qwopus3.5-9B-Coder" "$M/Ornith-1.5-9B" "$D"; do
  [ -d "$d" ] || { echo "ABORT: missing input $d"; exit 2; }
done
echo "sdc free before: $(df -h /mnt/sdc | awk 'NR==2{print $4}')"

if [ -d "$O" ]; then echo "SKIP $O exists"; else
echo ">>> $(date -u +%H:%M:%S) JackOD4-9B-Coder  (0.55 Delta / 0.30 Qwopus / 0.15 Ornith)"
python $OMK \
  --base      "$M/Qwen3.5-9B" \
  --task-base "$M/Qwen3.5-9B" \
  --source    "$D" \
  --source    "$M/Qwopus3.5-9B-Coder" \
  --source    "$M/Ornith-1.5-9B" \
  --weights   0.55,0.30,0.15 \
  $COMMON --output "$O"
rc=$?; echo ">>> JackOD4 merge rc=$rc"
[ $rc -ne 0 ] && { echo "JACKOD4_MERGE_FAIL rc=$rc"; exit $rc; }
python /mnt/sdc/oxopus/patch_jackod.py "$M/Qwopus3.5-9B-Coder" "$O"
python /mnt/sdc/oxopus/census_jackod.py "$M/Qwen3.5-9B" "$O"
echo "JACKOD4_BUILD_DONE $(du -sh "$O"|cut -f1)"
fi
echo "sdc free after: $(df -h /mnt/sdc | awk 'NR==2{print $4}')"
echo "JACKOD4_DONE $(date -u +%FT%TZ)"
