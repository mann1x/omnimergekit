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
  if curl -s "https://ollama.com/$REPO/tags" 2>/dev/null | grep -q ":$VT\b"; then
    LOG "  $VT already on registry — skip"; continue
  fi
  LOG "  pull $REPO:$T"
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
  { cat "$WORK/mf_$T"; echo; echo "FROM $MMPROJ"; } > "$WORK/mfv_$T"
  LOG "  create $REPO:$VT"
  ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1 || { LOG "  CREATE FAILED — skip"; continue; }
  # confirm vision capability before pushing; ollama create occasionally
  # returns before the projector layer is queryable -> recreate+recheck up to 3x.
  vcap=0
  for _ck in 1 2 3; do
    if ollama show "$REPO:$VT" 2>/dev/null | grep -qi vision; then vcap=1; break; fi
    LOG "  vision cap not visible (check $_ck/3) - recreate+retry"
    ollama create "$REPO:$VT" -f "$WORK/mfv_$T" >/dev/null 2>&1
    sleep 3
  done
  if [ "$vcap" = 0 ]; then
    LOG "  WARN: $VT no vision capability after 3 tries - NOT pushing"; ollama rm "$REPO:$VT" >/dev/null 2>&1; continue
  fi
  LOG "  push $REPO:$VT"
  if ! ollama push "$REPO:$VT" 2>&1 | tail -2; then
    LOG "  PUSH FAILED — leaving tags for inspection"; continue
  fi
  ollama rm "$REPO:$T" "$REPO:$VT" >/dev/null 2>&1
  gc_blobs
  df -h /root | awk 'NR==2{print "[vis] /root used="$5" avail="$4}'
  rm -f "$WORK/mf_$T" "$WORK/mfv_$T"
  LOG "  DONE $VT"
done
LOG "BATCH DONE"
