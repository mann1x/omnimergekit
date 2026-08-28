#!/usr/bin/env bash
# Second pass: drop the 10 REAM arm Q6_K quants + noimat_controls (~218 G).
#
# THIS ONE IS NOT REVERSIBLE THE WAY TIER 1 WAS.
# Tier 1 deleted F16s whose quants survived; tier 3 deleted bf16 arms whose quants
# survived. This pass deletes those quants, and the bf16 they'd be rebuilt from is
# already gone. Re-deriving any arm now means a full REAM merge from the 256e base.
# That is the accepted cost -- R2/R4 are closed and their RESULTS (scores, summary
# JSON) live outside this tree. Authorized 2026-08-24.
#
# EXPLICITLY PRESERVED, and gated after the fact:
#   */imatrix.dat        -- archival rule; 15-20 min of GPU each to recompute, and a
#                           quant whose imatrix is lost cannot be reproduced or audited
#   armJ + coderx_st_upload + publish/Qwen3.6-27B-A3B-CoderX  (one inode, links=3)
#   maps/ mmproj/ gguf_coderx/ data/ ollama_*/
set -uo pipefail

ts(){ date -u '+%F %T UTC'; }
say(){ echo "[$(ts)] $*"; }
RW=/mnt/sdc/ream-work

say "=== PREFLIGHT ==="
mapfile -t Q6 < <(ls "$RW"/gguf/*/*-Q6_K.gguf 2>/dev/null)
[ "${#Q6[@]}" -gt 0 ] || { say "no arm Q6_K found -- nothing to do"; exit 0; }

# Count the imatrix set BEFORE, so the post-check is a comparison and not a vibe.
IMAT_BEFORE=$(find "$RW" -xdev -name imatrix.dat 2>/dev/null | wc -l)
say "imatrix.dat present before: $IMAT_BEFORE"

FAIL=0
for q in "${Q6[@]}"; do
  case "$q" in
    */imatrix.dat) say "  REFUSE: $q is an imatrix"; FAIL=1;;
    "$RW"/gguf/*) say "  ok: $q ($(stat -c %s "$q" | numfmt --to=iec))";;
    *) say "  REFUSE: $q is outside $RW/gguf"; FAIL=1;;
  esac
done
[ -d "$RW/noimat_controls" ] && say "  ok: noimat_controls ($(du -sh "$RW/noimat_controls" | cut -f1))"
if ls "$RW/noimat_controls"/imatrix.dat >/dev/null 2>&1; then
  say "  REFUSE: noimat_controls holds an imatrix.dat"; FAIL=1
fi
[ "$FAIL" = 0 ] || { say "PREFLIGHT FAILED -- nothing deleted"; exit 2; }
say "PREFLIGHT_OK (${#Q6[@]} quants + noimat_controls)"

BEFORE=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
say "/mnt/sdc free BEFORE: ${BEFORE}G"

say "=== deleting arm Q6_K ==="
for q in "${Q6[@]}"; do
  sz=$(stat -c %s "$q" | numfmt --to=iec)
  rm -f -- "$q" "$q.sha256" && say "  removed $q ($sz)" || say "  FAILED $q"
done

say "=== deleting noimat_controls ==="
if [ -d "$RW/noimat_controls" ]; then
  sz=$(du -sh "$RW/noimat_controls" | cut -f1)
  rm -rf -- "$RW/noimat_controls" && say "  removed $RW/noimat_controls ($sz)" || say "  FAILED"
fi

AFTER=$(df --output=avail -BG /mnt/sdc | tail -1 | tr -dc 0-9)
say "/mnt/sdc free AFTER: ${AFTER}G  (reclaimed $((AFTER-BEFORE))G)"

say "=== POST-CHECK ==="
IMAT_AFTER=$(find "$RW" -xdev -name imatrix.dat 2>/dev/null | wc -l)
say "imatrix.dat present after: $IMAT_AFTER (was $IMAT_BEFORE)"
if [ "$IMAT_AFTER" != "$IMAT_BEFORE" ]; then
  say "!!! IMATRIX COUNT CHANGED -- archival rule breached, investigate immediately"
fi
find "$RW" -xdev -name imatrix.dat -printf "  %p  %s\n" 2>/dev/null | numfmt --field=2 --to=iec
echo "--- armJ + hardlink siblings ---"
for p in "$RW/armJ" "$RW/coderx_st_upload" "$RW/publish/Qwen3.6-27B-A3B-CoderX"; do
  [ -d "$p" ] && printf "  %s  %s  shard1 links=%s\n" "$p" "$(du -sh "$p"|cut -f1)" \
    "$(stat -c %h "$p"/model-00001-of-*.safetensors 2>/dev/null | head -1)"
done
echo "--- ream-work now ---"
du -x -h -d1 "$RW" 2>/dev/null | sort -hr | head -12
echo "###### PURGE_ARMQ6K_DONE $(ts) ######"
