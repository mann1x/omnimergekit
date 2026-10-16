#!/usr/bin/env bash
# pod_v9ml_publish.sh — ON THE POD: re-publish gemma-4-20B-v9-agentic from the v9-ml ml_k8 build,
# REPLACING the map-A weights and quants in the existing (private) repos. Owner 2026-10-16: "start
# publishing, replace all v9-agentic quants on HF and Ollama, use the pod" + "replace bf16 too" +
# "keep private". Derived from pod_v9_agentic_publish.sh (same template, imatrix recipe, tier set).
#
#   stage prep      verify the ml_k8 bf16 (SHA256SUMS, template sha, tokenizer/processor files == the
#                   map-A build), then MOVE it to $W/v9pub/gemma-4-20B-v9-agentic: the dir basename
#                   is every GGUF's filename stem and general.name. Map-A dirs are not touched.
#   stage snapshot  both HF repos' file list + LFS sha BEFORE anything is overwritten (audit trail)
#   stage upload    bf16 -> ManniX-ITA/gemma-4-20B-v9-agentic (stays PRIVATE), hub sha256 == local,
#                   hub files absent locally are LISTED as stale (never deleted here)
#   stage imatrix   the ml_k8 AtomicChat imatrix (9590 chunks, --parse-special), sha-gated, pre-staged
#   stage mmproj    vision projector from the bf16
#   stage quants    every published tier (Q6_K first, Q4_K_M second) -> the GGUF repo, --no-kld (A4B
#                   has no working PPL/KLD), sanity on GPU0 only; then every hub GGUF must equal THIS
#                   run's sidecar sha -> REPLACED_OK, else the leftovers are listed as STALE
# HF token: ~/.cache/huggingface/token (0600), never in argv, never echoed.
set -uo pipefail
W=/workspace
OMK=$W/omnimergekit
NAME=gemma-4-20B-v9-agentic
REPO=ManniX-ITA/$NAME
GREPO=ManniX-ITA/$NAME-GGUF
SRCBF=$W/v9ml_k8/gemma-4-20B-v9ml-ml_k8
ABF=$W/$NAME                                  # map-A bf16, reference for the metadata gate only
P=$W/v9pub
BF=$P/$NAME
GG=$P/gguf/$NAME
IMAT_SRC=$W/gguf/gemma-4-20B-v9ml-ml_k8/imatrix.dat
ISHA=820aab4a06031984dec21d1d85ffe6e8b30bb2aa6eec87d1c0e603ddb9a2cdf7
MSHA=84f2bd961772adbd                         # ml_k8 drop map (first 16)
TMPL=$W/stage/tmpl/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
PY=$W/miniconda/envs/omnimergekit/bin/python
TF=$HOME/.cache/huggingface/token
QLOG=$W/logs/v9pub_quants.out
# The tier set currently on the GGUF repo (docs/cards/v9_agentic/tiers.json, 39 quants + F16):
# pod_v9_agentic_publish.sh's list minus CD-IQ2_M_L / CD-IQ2_XS_L (withdrawn, bug-784).
TIERS="${TIERS:-Q6_K,Q4_K_M,Q8_0,Q6_K_L,Q5_K_L,Q5_K_M,Q5_K_S,Q4_K_L,Q4_K_S,IQ4_NL,IQ4_XS,Q3_K_XL,IQ3_M,Q3_K_L,Q3_K_M,Q3_K_S,IQ3_XS,IQ3_XXS,Q2_K_L,IQ2_M,IQ2_S,IQ2_XS,CD-Q6_K,CD-Q5_K_M,CD-Q4_K_M,CD-IQ4_K_M,CD-Q3_K_L,CD-Q6_K_L,CD-Q5_K_M_L,CD-Q4_K_M_L,CD-IQ3_M_L,CD-IQ3_XS_L,CD-Q6_K_h,CD-Q5_K_M_h,CD-Q4_K_M_h,CD-IQ3_M_h,CD-IQ3_XS_h,CD-IQ2_M_h,CD-IQ2_XS_h}"
STAGES="${STAGES:-prep snapshot upload imatrix mmproj quants}"
ts(){ date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say(){ echo "[$(ts)] $*"; }
die(){ echo "[$(ts)] FATAL $*"; exit 1; }
has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
mkdir -p $W/logs $P/gguf
[ -s "$TF" ] || die "no HF token file"
export HF_TOKEN; HF_TOKEN="$(tr -d '\n' < "$TF")"; : "${HF_TOKEN:?HF_TOKEN must be exported}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
[ "$(echo "$TIERS" | tr ',' '\n' | wc -l)" -eq 39 ] || [ -n "${ALLOW_TIER_COUNT:-}" ] || die "expected 39 tiers"

if has prep; then
  say "stage prep"
  if [ ! -d "$BF" ]; then
    [ -d "$SRCBF" ] || die "no $SRCBF"
    ( cd "$SRCBF" && sha256sum -c --quiet SHA256SUMS ) || die "ml_k8 shards fail their SHA256SUMS"
    [ "$(sha256sum "$SRCBF/chat_template.jinja" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
    for f in tokenizer.json tokenizer_config.json processor_config.json preprocessor_config.json generation_config.json chat_template.jinja; do
      cmp -s "$SRCBF/$f" "$ABF/$f" || die "$f differs from the map-A build"
    done
    for f in config.json model.safetensors.index.json expert_drop_metadata.json; do
      cmp -s "$SRCBF/$f" "$ABF/$f" && say "  $f == map A" || say "  $f DIFFERS from map A (expected for the drop metadata)"
    done
    mv "$SRCBF" "$BF" || die "mv"
    printf '%s\n' "source  $SRCBF (moved $(ts))" "map     ml_k8 $MSHA (expert_drop -> router_shared_upweight a1.2)" \
      "imatrix $ISHA (AtomicChat full corpus, 9590 chunks, --parse-special)" "template $TSHA (google official)" > "$P/PROVENANCE.txt"
  fi
  ( cd "$BF" && sha256sum -c --quiet SHA256SUMS ) || die "shards fail SHA256SUMS at $BF"
  say "PREP_OK $BF"
fi

if has snapshot; then
  say "stage snapshot (HF state before replacement)"
  "$PY" - "$REPO" "$GREPO" "$P" <<'PYEOF' || die "snapshot"
import json, sys
from huggingface_hub import HfApi
api = HfApi()
for repo in sys.argv[1:3]:
    i = api.model_info(repo, files_metadata=True)
    snap = {"repo": repo, "sha": i.sha, "private": i.private,
            "files": {s.rfilename: {"size": s.size, "lfs": s.lfs.sha256 if s.lfs else None} for s in i.siblings}}
    out = f"{sys.argv[3]}/hf_before_{repo.split('/')[1]}.json"
    json.dump(snap, open(out, "w"), indent=1)
    print(repo, "commit", i.sha[:12], "private", i.private, "files", len(snap["files"]), "->", out)
PYEOF
  say "SNAPSHOT_OK"
fi

if has upload; then
  say "stage upload -> $REPO (stays private)"
  "$PY" - "$REPO" "$BF" > $W/logs/v9pub_upload.log 2>&1 <<'PYEOF' || die "upload (see logs/v9pub_upload.log)"
import sys
from huggingface_hub import HfApi
repo, d = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=d, repo_type="model",
                  ignore_patterns=["*.pre_shared_upweight", ".shared_applied", "README.md", ".cache/*", "BUILD_OK"],
                  commit_message="v9-agentic: replace map-A weights with the v9-ml ml_k8 build (multilingual/IFEval-targeted map, shared-FFN alpha 1.2, official chat template)")
i = api.model_info(repo, files_metadata=True)
print("private", i.private)
for s in i.siblings:
    print("  %-44s %s %s" % (s.rfilename, s.size, (s.lfs.sha256[:16] if s.lfs else "")))
PYEOF
  tail -25 $W/logs/v9pub_upload.log
  "$PY" - "$REPO" "$BF" <<'PYEOF' || die "hub sha256 != local"
import os, sys
from huggingface_hub import HfApi
repo, d = sys.argv[1], sys.argv[2]
info = HfApi().model_info(repo, files_metadata=True)
hub = {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}
loc = dict(reversed(l.split()) for l in open(d + "/SHA256SUMS"))
bad = [f for f in loc if hub.get(f) != loc[f]]
local = set(os.listdir(d))
keep = {"README.md", ".gitattributes", "toolbench_scores.png"}
stale = sorted(s.rfilename for s in info.siblings if s.rfilename not in local and s.rfilename not in keep)
print("shards", len(loc), "mismatch", bad, "| hub-only (not deleted):", stale or "none")
sys.exit(1 if bad or not loc or not info.private else 0)
PYEOF
  say "UPLOAD_OK"
fi

if has imatrix; then
  say "stage imatrix (pre-staged ml_k8 AC imatrix)"
  mkdir -p "$GG"
  if [ ! -f "$GG/imatrix.dat" ]; then cp "$IMAT_SRC" "$GG/imatrix.dat" || die "cp imatrix"; fi
  [ "$(sha256sum "$GG/imatrix.dat" | cut -d' ' -f1)" = "$ISHA" ] || die "imatrix sha"
  say "IMATRIX_OK $(cut -c1-16 <<<"$ISHA")"
fi

if has mmproj; then
  say "stage mmproj"
  MM=$GG/mmproj-$NAME-F16.gguf
  if [ ! -s "$MM" ]; then
    "$PY" $W/llama.cpp/convert_hf_to_gguf.py "$BF" --mmproj --outtype f16 --outfile "$MM" \
      > $W/logs/v9pub_mmproj.log 2>&1 || { tail -20 $W/logs/v9pub_mmproj.log; die "mmproj convert"; }
  fi
  ls -la "$MM"; sha256sum "$MM" | tee "$MM.sha256"
  say "MMPROJ_OK"
fi

if has quants; then
  say "stage quants -> $GREPO (private) tiers: $TIERS"
  [ -s "$GG/imatrix.dat" ] || die "no imatrix"
  CUDA_VISIBLE_DEVICES=0 LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
    --model "$BF" --output-dir "$GG" --repo "$GREPO" --only "$TIERS" --no-kld \
    --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
    --base-model-id google/gemma-4-26B-A4B-it --ngl 99 \
    > $QLOG 2>&1 || die "quants (see $QLOG)"
  "$PY" -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$GG/mmproj-$NAME-F16.gguf', path_in_repo='mmproj-$NAME-F16.gguf', repo_id='$GREPO',
                    commit_message='mmproj (Gemma 4 vision tower, F16) for the v9-ml ml_k8 build')" || die "mmproj upload"
  "$PY" - "$GREPO" "$GG" "$NAME" "$TIERS" "$ISHA" <<'PYEOF' || die "replacement verification (see above)"
import hashlib, os, sys
from huggingface_hub import HfApi
repo, gg, name, tiers, isha = sys.argv[1:6]
info = HfApi().model_info(repo, files_metadata=True)
hub = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
ok, bad = [], []
for t in tiers.split(","):
    f = f"{name}-{t}.gguf"; side = os.path.join(gg, f + ".sha256")
    want = open(side).read().split()[0] if os.path.exists(side) else None
    (ok if want and hub.get(f) == want else bad).append(f"{t}:{(hub.get(f) or '-')[:12]}/{(want or '-')[:12]}")
f16 = os.path.join(gg, f"{name}-F16.gguf")
extra = {"imatrix.dat": isha, f"mmproj-{name}-F16.gguf": sha(os.path.join(gg, f"mmproj-{name}-F16.gguf"))}
if os.path.exists(f16): extra[f"{name}-F16.gguf"] = sha(f16)
for f, w in extra.items():
    (ok if hub.get(f) == w else bad).append(f"{f}:{(hub.get(f) or '-')[:12]}/{w[:12]}")
verified = {x.split(":")[0] for x in ok}
known = {f"{name}-{t}.gguf" for t in tiers.split(",")} | set(extra)
stale = sorted(f for f in hub if f.endswith((".gguf", ".dat")) and f not in known)
print("verified", len(ok), "| mismatched", bad or "none", "| hub GGUFs outside this run (STALE, not deleted):", stale or "none")
print("private", info.private)
sys.exit(1 if bad or not info.private else 0)
PYEOF
  say "REPLACED_OK"
fi
say "###### POD_V9ML_PUBLISH_DONE stages: $STAGES ######"
