#!/usr/bin/env bash
# coderx_reembed_template.sh — re-embed the FIXED chat template into every local
# code4/lcb3 GGUF (13 tiers + F16) BEFORE publish. The tiers were converted from a
# bf16 that still had the upstream template, so they bake in the 17466-byte upstream
# template; the published cohort embeds the 18051-byte fixed template in tier metadata
# (NOT just a sidecar). This rewrites each GGUF's tokenizer.chat_template in place
# (full rewrite via gguf_new_metadata, all tensors + other KV preserved).
# CPU-only, GPU-free. Sequential (transient = one file's size; F16 ~40G).
set -uo pipefail
PY=/srv/ml/envs/envs/omnimergekit/bin/python
GD=/mnt/sdc/ml/cx_std16/gguf_coderx
FIX=/mnt/sdc/ml/cx_std16/gguf_sidecars/chat_template.fixed.jinja
FIXMD5=6c545a0ddb73a5c7dfd6b54286eadcd0
TIERS=(F16 Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)
L(){ echo "[reembed $(date '+%T %Z')] $*"; }

[ -f "$FIX" ] && [ "$(md5sum "$FIX"|awk '{print $1}')" = "$FIXMD5" ] || { L "FATAL fixed template missing/wrong"; exit 2; }
free=$(df -P "$GD" | awk 'NR==2{print $4}'); [ "$free" -gt 45000000 ] || { L "FATAL disk<45G ($free K)"; exit 2; }
L "################ re-embed fixed template into ${#TIERS[@]} GGUFs ################"

ok=0; fail=0
for T in "${TIERS[@]}"; do
  f="$GD/CX16c4l3-bf16-$T.gguf"; tmp="$f.reembed.tmp"
  [ -f "$f" ] || { L "MISS $T (skip)"; fail=$((fail+1)); continue; }
  cur=$("$PY" - "$f" <<'PYC' 2>/dev/null
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
for fl in r.fields.values():
    if fl.name=="tokenizer.chat_template":
        print(sum(len(bytes(fl.parts[p])) for p in fl.data)); break
else: print(0)
PYC
)
  if [ "$cur" = "18051" ]; then L "$T already fixed (len=18051) — skip"; ok=$((ok+1)); continue; fi
  L "$T: embedded_len=$cur -> rewriting with fixed template"
  rm -f "$tmp"
  if ! "$PY" -m gguf.scripts.gguf_new_metadata --chat-template-file "$FIX" "$f" "$tmp" >>/mnt/sdc/ml/coderx_reembed.uplog 2>&1; then
    L "FATAL gguf_new_metadata failed on $T"; tail -8 /mnt/sdc/ml/coderx_reembed.uplog; rm -f "$tmp"; fail=$((fail+1)); break
  fi
  # post-verify the tmp before clobbering the original
  newlen=$("$PY" - "$tmp" <<'PYV' 2>/dev/null
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1]); n=0
for fl in r.fields.values():
    if fl.name=="tokenizer.chat_template": n=sum(len(bytes(fl.parts[p])) for p in fl.data)
print(n)
PYV
)
  if [ "$newlen" != "18051" ]; then L "FATAL $T post-rewrite len=$newlen (expected 18051)"; rm -f "$tmp"; fail=$((fail+1)); break; fi
  mv -f "$tmp" "$f"; L "$T OK (now len=18051)"; ok=$((ok+1))
done
L "###### CODERX_REEMBED_DONE ok=$ok fail=$fail $(date '+%T %Z') ######"
