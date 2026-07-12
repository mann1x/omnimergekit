#!/usr/bin/env bash
# Re-embed the FIXED chat template (18051) into the PUBLISHED v7-coder GGUF tiers
# and re-upload via XET. Source = LOCAL std16_cohort tiers, each sha256-GATED
# against the LIVE published LFS oid first, so we only ever patch+upload bytes
# that are byte-identical to what's published (no silent weight swap). Because
# the tensor payload is unchanged (only shifted by the template-size delta), Xet
# content-defined chunking transfers ONLY the changed metadata chunks.
# CPU-only. Patches to a tmp; the local std16_cohort originals are left intact.
set -uo pipefail
export HF_XET_HIGH_PERFORMANCE=1
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
PY=/srv/ml/envs/envs/omnimergekit/bin/python
REPO=ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF
CLEAN=gemma-4-A4B-98e-v7-coder-it
SRCDIR=/mnt/sdc/ml/std16_cohort/gemma-4-A4B-98e-v7-coder-it-GGUF
FIX=/mnt/sdc/ml/cx_std16/gguf_sidecars/chat_template.fixed.jinja
FIXMD5=6c545a0ddb73a5c7dfd6b54286eadcd0
WORK=/mnt/sdc/ml/v7coder_reembed; mkdir -p "$WORK"
TIERS=(F16 Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)
L(){ echo "[v7reembed $(date '+%T %Z')] $*"; }

[ "$(md5sum "$FIX"|awk '{print $1}')" = "$FIXMD5" ] || { L "FATAL fixed template wrong"; exit 2; }
"$HF" auth whoami 2>/dev/null | grep -qi ManniX-ITA || { L "FATAL not authed"; exit 2; }
L "################ v7-coder published-tier metadata re-embed (LOCAL src, xet) ################"

# live published LFS oids (sha256) keyed by filename
declare -A PUB
while read -r name oid; do PUB["$name"]="$oid"; done < <("$PY" - <<'PY' 2>/dev/null
from huggingface_hub import HfApi
a=HfApi()
for t in a.list_repo_tree("ManniX-ITA/gemma-4-A4B-98e-v7-coder-it-GGUF", recursive=False):
    sha=getattr(getattr(t,"lfs",None),"sha256",None)
    if sha and t.path.endswith(".gguf"): print(t.path, sha)
PY
)
L "fetched ${#PUB[@]} published oids"

embed_len(){ "$PY" - "$1" <<'PY' 2>/dev/null
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1]); n=0
for fl in r.fields.values():
    if fl.name=="tokenizer.chat_template": n=sum(len(bytes(fl.parts[p])) for p in fl.data)
print(n)
PY
}

ok=0; fail=0; skip=0
for T in "${TIERS[@]}"; do
  name="$CLEAN-$T.gguf"; src="$SRCDIR/$name"; tmp="$WORK/$name.fixed"
  [ -f "$src" ] || { L "$T: local src MISSING — SKIP"; skip=$((skip+1)); continue; }
  pub="${PUB[$name]:-}"; [ -n "$pub" ] || { L "$T: no published oid — SKIP"; skip=$((skip+1)); continue; }
  loc=$(sha256sum "$src" | awk '{print $1}')
  if [ "$loc" != "$pub" ]; then L "$T: local sha != published — SKIP (won't swap weights) [$loc vs $pub]"; skip=$((skip+1)); continue; fi
  cur=$(embed_len "$src")
  if [ "$cur" = "18051" ]; then L "$T: already fixed on HF — skip"; ok=$((ok+1)); continue; fi
  L "$T: sha-matched published, embedded=$cur -> patch"
  rm -f "$tmp"
  "$PY" -m gguf.scripts.gguf_new_metadata --chat-template-file "$FIX" "$src" "$tmp" >>/mnt/sdc/ml/v7coder_reembed.patch.log 2>&1 \
     || { L "FATAL patch $T"; rm -f "$tmp"; fail=$((fail+1)); break; }
  nl=$(embed_len "$tmp"); [ "$nl" = "18051" ] || { L "FATAL $T post-patch len=$nl"; rm -f "$tmp"; fail=$((fail+1)); break; }
  L "$T: xet upload (only metadata chunks should transfer)"
  "$HF" upload "$REPO" "$tmp" "$name" >>/mnt/sdc/ml/v7coder_reembed.up.log 2>&1 \
     || { L "FATAL upload $T"; rm -f "$tmp"; fail=$((fail+1)); break; }
  rm -f "$tmp"; L "$T OK (re-embedded + xet-uploaded)"; ok=$((ok+1))
done
L "###### V7CODER_REEMBED_DONE ok=$ok fail=$fail skip=$skip $(date '+%T %Z') ######"
