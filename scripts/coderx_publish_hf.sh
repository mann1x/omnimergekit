#!/usr/bin/env bash
# coderx_publish_hf.sh — publish the v7-coderx code4/lcb3 (CX16c4l3) re-release to HF.
# Replaces the live fs2440 weights across bf16 + GGUF + NVFP4A16 repos + 3 cards.
# UPLOAD-ONLY (no rebuild): all artifacts already built + chat-template-fixed.
#   - imatrix.dat uploaded FIRST (mandatory archival rule).
#   - 13 validated tiers re-keyed to clean published names (replace in place).
#   - 14 obsolete fs2440 tiers (+ stale .sha256) deleted LAST (after new content lands).
#   - GGUF template fix shipped as sidecars (chat_template.fixed.jinja + unittest),
#     matching the already-correct v7-coder sibling (tier metadata stays upstream).
# GPU-free, CPU-only. Fail-fast on any upload error.
# Launch:  source ~/.bashrc; setsid nohup bash coderx_publish_hf.sh >LOG 2>&1 </dev/null &
set -uo pipefail
export HF_XET_HIGH_PERFORMANCE=1
HF=/srv/ml/envs/envs/omnimergekit/bin/hf
PY=/srv/ml/envs/envs/omnimergekit/bin/python

IT_REPO=ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it
GGUF_REPO=ManniX-ITA/gemma-4-A4B-98e-v7-coderx-it-GGUF
NV_REPO=ManniX-ITA/gemma-4-A4B-98e-v7-coderx-NVFP4A16
CLEAN=gemma-4-A4B-98e-v7-coderx-it

BF=/mnt/sdc/ml/cx_std16/CX16c4l3-bf16
GD=/mnt/sdc/ml/cx_std16/gguf_coderx
NV=/mnt/sdc/ml/cx_std16/CX16c4l3-NVFP4A16
STAGE=/mnt/sdc/ml/cx_std16/gguf_sidecars
CARDS=/mnt/sdc/ml/cx_std16/cards
UPLOG=/mnt/sdc/ml/coderx_publish_hf.uplog
: > "$UPLOG"

KEEP=(Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M CD-Q2_K)
# 14 obsolete tiers to purge from the GGUF repo (fs2440 ladder minus the validated 13 + F16)
DELETE=(CD-Q3_K_L CD-Q4_K_M CD-Q5_K_M CD-Q6_K CD-qat-Q4_K_M IQ2_XS IQ3_M Q2_K_L Q3_K_S Q3_K_XL Q4_0 Q4_1 Q5_K_S qat-Q4_0)

L(){ echo "[publish $(date '+%T %Z')] $*"; }
up(){  # up <repo> <local_path> <path_in_repo>
  L "upload -> ${1##*/} :: $3"
  if ! "$HF" upload "$1" "$2" "$3" >>"$UPLOG" 2>&1; then
    L "FATAL upload failed: $1 :: $3"; tail -8 "$UPLOG"; exit 7
  fi
}
shafile(){  # shafile <src_path> <published_name>  -> writes <src_dir>/<published_name>.sha256
  local src="$1" name="$2" h; h=$(sha256sum "$src" | awk '{print $1}')
  printf '%s  %s\n' "$h" "$name" > "$(dirname "$src")/$name.sha256"
  echo "$(dirname "$src")/$name.sha256"
}

L "################ coderx code4/lcb3 HF publish (UPLOAD-ONLY) ################"

# ---- preflight ----
who=$("$HF" auth whoami 2>/dev/null | grep -i "ManniX-ITA" | head -1)
[ -n "$who" ] || { L "FATAL: hf not authenticated as ManniX-ITA"; exit 2; }
L "auth OK ($who)"
for f in "$BF/config.json" "$BF/model-00009-of-00009.safetensors" "$GD/imatrix.dat" \
         "$GD/CX16c4l3-bf16-F16.gguf" "$NV/model-00002-of-00002.safetensors" \
         "$STAGE/chat_template.fixed.jinja" "$STAGE/template_loop_unittest.py" \
         "$CARDS/bf16_README.md" "$CARDS/gguf_README.md" "$CARDS/nvfp4_README.md"; do
  [ -f "$f" ] || { L "FATAL missing artifact: $f"; exit 3; }
done
for T in "${KEEP[@]}"; do [ -f "$GD/CX16c4l3-bf16-$T.gguf" ] || { L "FATAL missing tier $T"; exit 3; }; done
# template-fix gate: bf16/nvfp4 inline + sidecar must all be the fixed 18051-byte template
FIXMD5=6c545a0ddb73a5c7dfd6b54286eadcd0
for f in "$BF/chat_template.jinja" "$NV/chat_template.jinja" "$STAGE/chat_template.fixed.jinja"; do
  m=$(md5sum "$f" | awk '{print $1}'); [ "$m" = "$FIXMD5" ] || { L "FATAL template not fixed: $f ($m)"; exit 4; }
done
# GGUF tiers + F16 must carry the FIXED template EMBEDDED (re-embed must have run first)
for T in F16 "${KEEP[@]}"; do
  el=$("$PY" - "$GD/CX16c4l3-bf16-$T.gguf" <<'PYE' 2>/dev/null
import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1]); n=0
for fl in r.fields.values():
    if fl.name=="tokenizer.chat_template": n=sum(len(bytes(fl.parts[p])) for p in fl.data)
print(n)
PYE
)
  [ "$el" = "18051" ] || { L "FATAL tier $T embedded template not fixed (len=$el) — run coderx_reembed_template.sh first"; exit 4; }
done
L "preflight OK (artifacts present, bf16/nvfp4/sidecar + all GGUF tiers carry fixed template)"

# =========================== GGUF repo ===========================
# 1. imatrix FIRST (mandatory archival rule)
up "$GGUF_REPO" "$GD/imatrix.dat" "imatrix.dat"
# 2. 13 validated tiers (re-keyed clean) + fresh sha256
for T in "${KEEP[@]}"; do
  src="$GD/CX16c4l3-bf16-$T.gguf"; dst="$CLEAN-$T.gguf"
  s=$(shafile "$src" "$dst")
  up "$GGUF_REPO" "$src" "$dst"
  up "$GGUF_REPO" "$s"   "$dst.sha256"
done
# 3. F16 + sha256
src="$GD/CX16c4l3-bf16-F16.gguf"; dst="$CLEAN-F16.gguf"; s=$(shafile "$src" "$dst")
up "$GGUF_REPO" "$src" "$dst"; up "$GGUF_REPO" "$s" "$dst.sha256"
# 4. template-fix sidecars (cohort parity) + card
up "$GGUF_REPO" "$STAGE/chat_template.fixed.jinja" "chat_template.fixed.jinja"
up "$GGUF_REPO" "$STAGE/template_loop_unittest.py" "template_loop_unittest.py"
up "$GGUF_REPO" "$CARDS/gguf_README.md" "README.md"
L "GGUF repo: 13 tiers + F16 + imatrix + sidecars + card uploaded"

# =========================== bf16 repo ===========================
"$PY" - "$BF/config.json" "$IT_REPO" <<'PYC'
import json,sys; p,r=sys.argv[1],sys.argv[2]
c=json.load(open(p)); c["_name_or_path"]=r; json.dump(c,open(p,"w"),indent=2)
print("  bf16 _name_or_path =",r)
PYC
L "bf16: upload weights (exclude backups/markers/internal metadata)"
if ! "$HF" upload "$IT_REPO" "$BF" . \
      --exclude "*.pre_shared_upweight" --exclude ".shared_applied" \
      --exclude "*.bak" --exclude "expert_drop_metadata.json" >>"$UPLOG" 2>&1; then
  L "FATAL bf16 folder upload failed"; tail -8 "$UPLOG"; exit 7
fi
up "$IT_REPO" "$CARDS/bf16_README.md" "README.md"
L "bf16 repo: weights + card uploaded"

# =========================== NVFP4A16 repo ===========================
L "NVFP4A16: upload weights (exclude template backup)"
if ! "$HF" upload "$NV_REPO" "$NV" . --exclude "*.bak" >>"$UPLOG" 2>&1; then
  L "FATAL nvfp4a16 folder upload failed"; tail -8 "$UPLOG"; exit 7
fi
up "$NV_REPO" "$CARDS/nvfp4_README.md" "README.md"
L "NVFP4A16 repo: weights + card uploaded"

# =========================== purge obsolete GGUF tiers (LAST) ===========================
"$PY" - "$GGUF_REPO" "$CLEAN" <<'PYD'
import sys
from huggingface_hub import HfApi, list_repo_files, CommitOperationDelete
repo, clean = sys.argv[1], sys.argv[2]
dead_tiers = ["CD-Q3_K_L","CD-Q4_K_M","CD-Q5_K_M","CD-Q6_K","CD-qat-Q4_K_M",
              "IQ2_XS","IQ3_M","Q2_K_L","Q3_K_S","Q3_K_XL","Q4_0","Q4_1","Q5_K_S","qat-Q4_0"]
want=set()
for t in dead_tiers:
    want.add(f"{clean}-{t}.gguf"); want.add(f"{clean}-{t}.gguf.sha256")
fs=set(list_repo_files(repo))
dead=sorted(f for f in fs if f in want)
api=HfApi()
if dead:
    api.create_commit(repo_id=repo, repo_type="model",
        operations=[CommitOperationDelete(path_in_repo=f) for f in dead],
        commit_message="drop 14 non-validated tiers (code4/lcb3 re-release keeps the validated 13 + F16)")
    print("  purged %d obsolete files:" % len(dead)); [print("   -",f) for f in dead]
else:
    print("  no obsolete tiers present (already clean)")
PYD

# =========================== verify ===========================
"$PY" - "$GGUF_REPO" "$IT_REPO" "$NV_REPO" "$CLEAN" <<'PYV'
import sys
from huggingface_hub import list_repo_files
gguf, it, nv, clean = sys.argv[1:5]
keep=["Q8_0","Q6_K_L","Q6_K","Q5_K_L","Q5_K_M","Q4_K_L","Q4_K_M","Q4_K_S","IQ4_NL","IQ4_XS","Q3_K_L","Q3_K_M","CD-Q2_K"]
fs=set(list_repo_files(gguf))
tiers=sorted(f for f in fs if f.endswith(".gguf") and f!=f"{clean}-F16.gguf" and "mmproj" not in f)
print("=== VERIFY GGUF repo ===")
print("  .gguf tiers (excl F16/mmproj): %d  -> %s" % (len(tiers), [t.replace(clean+'-','').replace('.gguf','') for t in tiers]))
miss=[t for t in keep if f"{clean}-{t}.gguf" not in fs]
print("  missing validated tiers:", miss or "none")
for must in [f"{clean}-F16.gguf","imatrix.dat","mmproj-gemma4.gguf","chat_template.fixed.jinja","template_loop_unittest.py","README.md"]:
    print("   ", "OK " if must in fs else "MISS", must)
print("=== VERIFY bf16 repo ===")
fb=set(list_repo_files(it)); sh=[f for f in fb if f.endswith(".safetensors")]
print("  safetensors:", len(sh), "| README:", "README.md" in fb, "| chat_template.jinja:", "chat_template.jinja" in fb)
print("=== VERIFY NVFP4A16 repo ===")
fn=set(list_repo_files(nv)); sn=[f for f in fn if f.endswith(".safetensors")]
print("  safetensors:", len(sn), "| README:", "README.md" in fn, "| preprocessor_config:", "preprocessor_config.json" in fn)
PYV
L "###### CODERX_PUBLISH_HF_DONE $(date '+%T %Z') ######"
