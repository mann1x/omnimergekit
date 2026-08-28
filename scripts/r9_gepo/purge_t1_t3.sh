#!/usr/bin/env bash
# Tier 1 (F16 build intermediates) + Tier 3 (links=1 REAM bf16 arms) purge on bs2.
#
# WHAT IS DELIBERATELY *NOT* IN HERE
# ----------------------------------
#   armJ/                     -- the SHIPPED CoderX weights, and hardlinked 3 ways
#                                (armJ / coderx_st_upload / publish/Qwen3.6-27B-A3B-CoderX
#                                are ONE inode, links=3). Untouched.
#   every *-Q6_K.gguf         -- the quants the F16s were built to produce
#   every imatrix.dat         -- mandatory archival; a quant whose imatrix is lost
#                                cannot be reproduced or audited
#   mmproj-*-F16.gguf         -- a real vision projector, NOT an intermediate
#   maps/ noimat_controls/ publish/ data/
#
# EVERY DELETE IS AN EXPLICIT LITERAL PATH. No globs, no find -delete, no recursion
# into anything that was not named by the inventory.
set -uo pipefail

ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*"; }

RW=/mnt/sdc/ream-work

# ---- Tier 1: F16 intermediates whose Q6_K already exists ---------------------
declare -A T1_GUARD=(
  ["$RW/gguf/armG_imat/armG-F16.gguf"]="$RW/gguf/armG_imat/armG-Q6_K.gguf"
  ["$RW/gguf/armH_imat/armH-F16.gguf"]="$RW/gguf/armH_imat/armH-Q6_K.gguf"
  ["$RW/gguf/armI_imat/armI-F16.gguf"]="$RW/gguf/armI_imat/armI-Q6_K.gguf"
  ["$RW/gguf/armJ_imat/armJ-F16.gguf"]="$RW/gguf/armJ_imat/armJ-Q6_K.gguf"
  ["/mnt/sdc/ml/brevity/gepo/gguf_gepo1/a3b-gepo1-F16.gguf"]="/mnt/sdc/ml/brevity/gepo/gguf_gepo1/a3b-gepo1-Q6_K.gguf"
  ["/mnt/sdc/ml/eval_gguf/128e-F16-g0709.gguf"]="/mnt/sdc/ml/eval_gguf/128e-Q6_K.gguf"
)

# ---- Tier 1b: the CoderX F16, which needs a DIFFERENT guard -----------------
# gguf_coderx/ holds NO local .gguf quants -- only the F16 and 20 *.sha256
# receipts. Every CoderX tier was built there, published to HF + ollama
# (#841 R6.imat / #842 R6.ollama), and reaped locally, leaving the checksums as
# the record. So "is the local Q6_K present" is the WRONG question for this file;
# it would refuse forever. The correct evidence that this F16 has already yielded
# its outputs is: imatrix.dat preserved AND the tier receipts present. Anything
# less and we keep it.
CX_F16=$RW/gguf_coderx/Qwen3.6-27B-A3B-CoderX-F16.gguf
CX_IMAT=$RW/gguf_coderx/imatrix.dat

# ---- Tier 3: bf16 arms, links=1 only. armJ is NOT here. ----------------------
T3_ARMS=(armB armC armE armG armH armI armD_ourssal_nomerge armF_rnorm_nomerge)

say "=== PREFLIGHT ==="
FAIL=0

# Tier 1: refuse to drop an F16 unless its Q6_K exists and carries GGUF magic.
for f in "${!T1_GUARD[@]}"; do
  q="${T1_GUARD[$f]}"
  if [ ! -s "$f" ]; then say "  SKIP (absent): $f"; continue; fi
  if [ ! -s "$q" ] || [ "$(head -c4 "$q")" != "GGUF" ]; then
    say "  REFUSE: $f -- its quant $q is missing or not a GGUF"; FAIL=1; continue
  fi
  say "  T1 ok: $(basename "$f") ($(stat -c %s "$f" | numfmt --to=iec)) <- quant present"
done

# Tier 1b: CoderX F16 -- evidence is the imatrix + the published-tier receipts.
if [ -s "$CX_F16" ]; then
  NREC=$(ls "$RW"/gguf_coderx/*.gguf.sha256 2>/dev/null | wc -l)
  if [ ! -s "$CX_IMAT" ]; then
    say "  REFUSE: $CX_F16 -- imatrix.dat missing from gguf_coderx"; FAIL=1
  elif [ "$NREC" -lt 10 ]; then
    say "  REFUSE: $CX_F16 -- only $NREC tier receipts, expected the full published ladder"; FAIL=1
  else
    say "  T1b ok: CoderX-F16 ($(stat -c %s "$CX_F16" | numfmt --to=iec)) <- $NREC tier receipts + imatrix.dat"
  fi
fi

# Tier 3: refuse to drop an arm unless (a) it is NOT armJ, (b) links==1 on its
# first shard, (c) its Q6_K exists. (b) is the one that matters: a links>1 arm
# shares inodes with something else and deleting it frees nothing while
# potentially orphaning the sibling's expectations.
for a in "${T3_ARMS[@]}"; do
  d=$RW/$a
  [ -d "$d" ] || { say "  SKIP (absent): $d"; continue; }
  case "$a" in armJ|*armJ*) say "  REFUSE: $a is the shipped CoderX -- never in this set"; FAIL=1; continue;; esac
  s=$(ls "$d"/model-00001-of-*.safetensors 2>/dev/null | head -1)
  [ -n "$s" ] || { say "  REFUSE: $d has no shard 1"; FAIL=1; continue; }
  n=$(stat -c %h "$s")
  if [ "$n" != "1" ]; then
    say "  REFUSE: $a shard1 has links=$n (shared inode) -- not a free-standing copy"; FAIL=1; continue
  fi
  q=$(ls "$RW/gguf/${a}_imat/"*-Q6_K.gguf "$RW/gguf/${a}_noimat/"*-Q6_K.gguf 2>/dev/null | head -1)
  if [ -z "$q" ] || [ "$(head -c4 "$q")" != "GGUF" ]; then
    say "  REFUSE: $a -- no valid Q6_K under gguf/${a}_imat"; FAIL=1; continue
  fi
  say "  T3 ok: $a ($(du -sh "$d" | cut -f1), links=1) <- $(basename "$q")"
done

# imatrix must not be inside anything we are about to remove.
for a in "${T3_ARMS[@]}"; do
  if ls "$RW/$a"/imatrix.dat >/dev/null 2>&1; then
    say "  REFUSE: $RW/$a contains imatrix.dat -- archival rule"; FAIL=1
  fi
done

[ "$FAIL" = 0 ] || { say "PREFLIGHT FAILED -- nothing deleted"; exit 2; }
say "PREFLIGHT_OK"

BEFORE=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
say "/mnt/sdc free BEFORE: ${BEFORE}G"

say "=== TIER 1: F16 intermediates ==="
for f in "${!T1_GUARD[@]}"; do
  [ -s "$f" ] || continue
  sz=$(stat -c %s "$f" | numfmt --to=iec)
  rm -f -- "$f" && say "  removed $f ($sz)" || say "  FAILED $f"
done
if [ -s "$CX_F16" ]; then
  sz=$(stat -c %s "$CX_F16" | numfmt --to=iec)
  rm -f -- "$CX_F16" && say "  removed $CX_F16 ($sz)" || say "  FAILED $CX_F16"
fi

say "=== TIER 3: bf16 arms ==="
for a in "${T3_ARMS[@]}"; do
  d=$RW/$a
  [ -d "$d" ] || continue
  sz=$(du -sh "$d" | cut -f1)
  rm -rf -- "$d" && say "  removed $d ($sz)" || say "  FAILED $d"
done

AFTER=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
say "/mnt/sdc free AFTER: ${AFTER}G  (reclaimed $((AFTER-BEFORE))G)"

say "=== POST-CHECK: everything that had to survive ==="
echo "--- Q6_K quants still present ---"
ls -la "$RW"/gguf/*/*-Q6_K.gguf "$RW"/gguf_coderx/*.gguf 2>/dev/null | awk '{printf "  %s  %.0fG\n",$9,$5/1e9}'
echo "--- imatrix.dat still present ---"
find "$RW" /mnt/sdc/ml/brevity -xdev -name imatrix.dat -printf "  %p  %s\n" 2>/dev/null | numfmt --field=2 --to=iec
echo "--- armJ + its hardlink siblings ---"
for p in "$RW/armJ" "$RW/coderx_st_upload" "$RW/publish/Qwen3.6-27B-A3B-CoderX"; do
  [ -d "$p" ] && printf "  %s  %s  shard1 links=%s\n" "$p" "$(du -sh "$p"|cut -f1)" \
    "$(stat -c %h "$p"/model-00001-of-*.safetensors 2>/dev/null | head -1)"
done
echo "--- GEPO adapter ---"
ls -la /mnt/sdc/ml/brevity/gepo/run1/adapter_model.safetensors 2>/dev/null
echo "###### PURGE_T1_T3_DONE $(ts) ######"
