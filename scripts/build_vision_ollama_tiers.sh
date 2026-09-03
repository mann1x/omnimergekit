#!/usr/bin/env bash
# T185: build vision-<tier> ollama variants for mannix/gemma4-98e-v6-coder by
# pairing each existing text quant with mmproj-gemma4.gguf (Gemma4 vision tower).
#
# Per tier: pull text quant from the ollama registry -> derive its Modelfile ->
# append `FROM mmproj` -> create vision-<tier> -> push -> rm both tags ->
# purge orphaned blob files (KEEP the shared mmproj blob). Disk-hygienic so the
# 8-21 GB quants don't accumulate in /root/.ollama/models/blobs.
#
# Idempotent: a tier whose vision tag is already on ollama.com is skipped.
# Usage: build_vision_ollama_tiers.sh <tier> [<tier> ...]
#
# Model-agnostic since 2026-08-05: REPO / MMPROJ / WORK are env overrides so the
# same script drives other families (e.g. Qwen3.6 OmniMerge v4, whose projector is
# qwen3vl_merger rather than Gemma4's). The T185 Gemma4 values remain the defaults
# so existing invocations are byte-for-byte unchanged. Override, never fork:
#   REPO=mannix/omnimerge-v4-mtp \
#   MMPROJ=/mnt/sdc/ml/v4vision/mmproj-Qwen3.6-27B-Omnimerge-v4-F16.gguf \
#   WORK=/mnt/sdc/ml/v4vision/vision_work \
#   build_vision_ollama_tiers.sh q4_K_M q6_K q8_0
set -uo pipefail

REPO="${REPO:-mannix/gemma4-98e-v6-coder}"
MMPROJ="${MMPROJ:-/mnt/sdc/ml/gguf/v6coder/mmproj-gemma4.gguf}"
WORK="${WORK:-/mnt/sdc/ml/gguf/v6coder/vision_work}"
# Settle time between daemon operations. ollama create/rm prune and rewrite blobs
# in the background; the next operation can start before the previous one's
# bookkeeping has landed. A pause between steps costs ~1 min/tier and removes a
# whole class of race. Override with SETTLE=0 to run flat out.
SETTLE="${SETTLE:-10}"
# Minimum ollama version floor to stamp on the vision tags (see the REQUIRES note
# at modelfile assembly). Empty = omit the directive.
REQUIRES="${REQUIRES:-}"
mkdir -p "$WORK"
LOG(){ echo "[vis $(date -u +%H:%M:%S)] $*"; }

# Resolve the store the DAEMON actually uses -- never assume /root/.ollama. `ollama` runs under
# its own systemd User= (User=ollama on bs2 -> /usr/share/ollama/.ollama/models), so a hardcoded
# /root path GCs a store nothing writes to: the real library grows unbounded while the log
# cheerfully reports purges. Order: explicit env > OLLAMA_MODELS > the unit's User= home > /root.
resolve_store(){
  [ -n "${OLLAMA_MODELS:-}" ] && { echo "$OLLAMA_MODELS"; return; }
  local u home
  u=$(systemctl show ollama -p User --value 2>/dev/null)
  if [ -n "$u" ]; then
    home=$(getent passwd "$u" | cut -d: -f6)
    [ -n "$home" ] && [ -d "$home/.ollama/models" ] && { echo "$home/.ollama/models"; return; }
  fi
  echo "/root/.ollama/models"
}
STORE="$(resolve_store)"
BLOBS="${BLOBS:-$STORE/blobs}"
MANIFESTS="${MANIFESTS:-$STORE/manifests}"
STRIPPER="$WORK/_strip_template.py"
mkdir -p "$WORK"
cat > "$STRIPPER" <<'PYSTRIP'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
out, i = [], 0
while i < len(lines):
    if re.match(r"\s*TEMPLATE\b", lines[i], re.I):
        parts = lines[i].split(None, 1)
        rest = parts[1] if len(parts) > 1 else ""
        if rest.startswith('"""') and rest.count('"""') < 2:
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
        i += 1
        continue
    out.append(lines[i]); i += 1
print("\n".join(out))
PYSTRIP
LOG "ollama store: $STORE"

[ -f "$MMPROJ" ] || { LOG "FATAL: mmproj missing $MMPROJ"; exit 1; }
MMSHA="sha256-$(sha256sum "$MMPROJ" | cut -d' ' -f1)"
LOG "mmproj blob protected: $MMSHA"

gc_blobs(){
  # delete blob files not referenced by ANY current manifest, except the mmproj blob
  local ref; ref=$(mktemp)
  grep -rhoE "sha256[:-][0-9a-f]{64}" "$MANIFESTS" 2>/dev/null | sed 's/sha256:/sha256-/' | sort -u > "$ref"
  # An EMPTY reference set is a bug signal (wrong/missing MANIFESTS dir), never a licence to
  # delete the whole library -- without this guard a mis-resolved store makes every blob look
  # orphaned and gc wipes the user's entire model collection in one pass.
  if [ ! -s "$ref" ]; then
    LOG "  gc: REFUSING -- 0 manifest refs found under $MANIFESTS (wrong store?); nothing purged"
    rm -f "$ref"; return 0
  fi
  local n=0
  for b in "$BLOBS"/sha256-*; do
    [ -e "$b" ] || continue
    local bn; bn=$(basename "$b")
    [ "$bn" = "$MMSHA" ] && continue
    if ! grep -qx "$bn" "$ref"; then
      local sz; sz=$(du -h "$b" 2>/dev/null | cut -f1)
      rm -f "$b" && { LOG "  purged orphan blob $bn ($sz)"; n=$((n+1)); }
    fi
  done
  rm -f "$ref"
  LOG "  gc: purged $n orphan blob(s)"
}

for T in "$@"; do
  VT="vision-$T"
  LOG "===== tier $T -> $REPO:$VT ====="
  # Capture FIRST, then match via herestring (no pipe at all) -- same SIGPIPE trap as the vision-capability check
  # below: `curl | grep -q` under pipefail makes grep exit on match, SIGPIPE curl
  # (exit 23), and the pipeline inherits 23, so a tag that IS published reads as
  # absent. This skip had therefore never fired; every run redid published tiers.
  tags=$(curl -s "https://ollama.com/$REPO/tags" 2>/dev/null || true)
  if grep -q ":$VT\b" <<<"$tags"; then
    LOG "  $VT already on registry — skip"; continue
  fi
  LOG "  pull $REPO:$T"
  sleep "$SETTLE"
  ollama pull "$REPO:$T" >/dev/null 2>&1 || { LOG "  PULL FAILED — skip"; continue; }
  ollama show "$REPO:$T" --modelfile > "$WORK/mf_$T" 2>/dev/null || { LOG "  modelfile FAILED — skip"; continue; }
  # vision modelfile = original (params/renderer/parser/FROM base-blob) + projector,
  # with any TEMPLATE directive STRIPPED. `ollama show --modelfile` renders a
  # GGUF-derived chat template as a literal TEMPLATE for tags carrying no template
  # layer, and on complex Jinja that fallback is the degenerate `{{ .Prompt }}`.
  # Baking it into the vision tag creates a real template layer that then SHADOWS
  # the renderer. That is exactly how mannix/omnimerge-v4:vision-Q4_K_M ended up
  # with a 13-byte `{{ .Prompt }}` layer while its own text tag has none.
  # Vendor tags (library/qwen3.6:27b, library/qwen3.8:27b) ship no template layer.
  python3 "$STRIPPER" "$WORK/mf_$T" > "$WORK/mf_$T.stripped"
  if [ -s "$WORK/mf_$T.stripped" ]; then
    cmp -s "$WORK/mf_$T" "$WORK/mf_$T.stripped" || \
      LOG "  stripped a derived TEMPLATE from $T (would have shadowed the renderer)"
    mv "$WORK/mf_$T.stripped" "$WORK/mf_$T"
  else
    LOG "  WARNING: template strip empty for $T; keeping original"
    rm -f "$WORK/mf_$T.stripped"
  fi
  # Re-assert REQUIRES. `ollama show --modelfile` round-trips RENDERER, PARSER and
  # PARAMETER but NOT REQUIRES, so a vision tag derived from a text tag silently
  # loses the version floor -- and REQUIRES is precisely the guard that makes an
  # older ollama REFUSE the model instead of loading it and degrading at render
  # time. Without this the floor has to be reattached by a second identity pass.
  { cat "$WORK/mf_$T"
    echo
    [ -n "$REQUIRES" ] && echo "REQUIRES $REQUIRES"
    echo "FROM $MMPROJ"; } > "$WORK/mfv_$T"
  # Pre-stage the projector blob under its final digest. `ollama rm` prunes it
  # after every tier, so each create otherwise re-copies 927 MB into a temp blob
  # and renames it -- and ANY concurrent prune (this script's own rm, or another
  # tool GCing the same store) deletes that temp mid-write. The rename then fails
  # with "no such file or directory", the projector layer never lands, and the
  # vision capability check correctly refuses to push. Staging the blob makes
  # create say "using existing layer" and removes the write entirely.
  if [ ! -f "$BLOBS/$MMSHA" ]; then
    LOG "  staging mmproj blob $MMSHA"
    cp -f "$MMPROJ" "$BLOBS/.stage_$MMSHA" && mv -f "$BLOBS/.stage_$MMSHA" "$BLOBS/$MMSHA" || {
      LOG "  mmproj stage FAILED — skip"; rm -f "$BLOBS/.stage_$MMSHA"; continue; }
    OWNER=$(stat -c "%u:%g" "$BLOBS" 2>/dev/null)
    [ -n "$OWNER" ] && chown "$OWNER" "$BLOBS/$MMSHA" 2>/dev/null
  fi
  LOG "  create $REPO:$VT"
  if ! ollama create "$REPO:$VT" -f "$WORK/mfv_$T" > "$WORK/create_$T.log" 2>&1; then
    LOG "  CREATE FAILED — skip"
    sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$WORK/create_$T.log" | grep -viE "gathering|copying file" | tail -3 | while read -r l; do LOG "    | $l"; done
    ollama rm "$REPO:$T" >/dev/null 2>&1
    continue
  fi
  # confirm vision capability before pushing; ollama create occasionally
  # returns before the projector layer is queryable -> recreate+recheck up to 3x.
  sleep "$SETTLE"
  vcap=0
  for _ck in 1 2 3; do
    # Capture FIRST, match via herestring. Never `ollama show | grep -q` under pipefail:
    # grep -q exits the moment it matches, which SIGPIPEs ollama show (exit
    # 141), and pipefail makes the pipeline inherit 141 -- so a model that DOES
    # have vision reports "not visible". It is a race with how fast ollama show
    # flushes, which is why it passed 6 of 19 tiers on 2026-09-03 and failed 13.
    caps=$(ollama show "$REPO:$VT" 2>/dev/null || true)
    if grep -qi vision <<<"$caps"; then vcap=1; break; fi
    LOG "  vision cap not visible (check $_ck/3) - recreate+retry"
    ollama create "$REPO:$VT" -f "$WORK/mfv_$T" > "$WORK/create_$T.log" 2>&1
    sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$WORK/create_$T.log" | grep -viE "gathering|copying file|using existing|writing manifest|^success" | tail -2 | while read -r l; do [ -n "$l" ] && LOG "    | $l"; done
    sleep "$SETTLE"
  done
  if [ "$vcap" = 0 ]; then
    LOG "  WARN: $VT no vision capability after 3 tries - NOT pushing"
    LOG "    manifest layers: $(python3 -c "
import json,sys
try:
    m=json.load(open(sys.argv[1]))
    print(', '.join(l['"'"'mediaType'"'"'].split('"'"'.'"'"')[-1] for l in m['"'"'layers'"'"']))
except Exception as e:
    print('NO MANIFEST', e)
" "$MANIFESTS/registry.ollama.ai/${REPO}/$VT" 2>&1)"
    # Remove the pulled TEXT tag too. Removing only the vision tag leaves the
    # 10-23 GB text blob referenced, so gc reports "purged 0" while the store
    # grows by a full tier per failure -- 195 GB across 13 failures on 2026-09-03.
    sleep "$SETTLE"
    ollama rm "$REPO:$VT" "$REPO:$T" >/dev/null 2>&1
    sleep "$SETTLE"
    gc_blobs
    continue
  fi
  sleep "$SETTLE"
  LOG "  push $REPO:$VT"
  if ! ollama push "$REPO:$VT" 2>&1 | tail -2; then
    LOG "  PUSH FAILED — leaving tags for inspection (disk NOT reclaimed)"; continue
  fi
  sleep "$SETTLE"
  ollama rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1
  sleep "$SETTLE"
  gc_blobs
  df -h /root | awk 'NR==2{print "[vis] /root used="$5" avail="$4}'
  rm -f "$WORK/mf_$T" "$WORK/mfv_$T"
  LOG "  DONE $VT"
done
LOG "BATCH DONE"
