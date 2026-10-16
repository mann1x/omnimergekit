#!/usr/bin/env bash
# pod_v9_agentic_publish.sh — ON THE POD: build and publish gemma-4-20B-v9-agentic, which is
# candidate A of the v9-agentic A/B (map A_lift120_plus_fk12, the agentic map) shipped with
# Google's OFFICIAL chat template, the one that re-injects prior reasoning.
#
#   stage env      apt + llama.cpp pinned at the stack lock (b9700) + conda env (pod_eval_bootstrap)
#   stage base     google/gemma-4-26B-A4B-it weight shards via aria2c, sha256 checked against HF's LFS record
#   stage build    expert_drop(A map) -> router_shared_upweight alpha=1.2 mlp.down_proj.weight, then every
#                  shard must hash to the A build delivered on solidpc (stage/A_SHA256SUMS.solidpc)
#   stage render   AtomicChat corpus re-rendered through the official template (recipe gemma-4-20b-v9-agentic)
#   stage upload   bf16 -> ManniX-ITA/gemma-4-20B-v9-agentic (created PRIVATE), hub sha256 == local
#   stage imatrix  F16 + AtomicChat imatrix: full corpus, --parse-special; ngl sized by quantize_gguf
#                  (the 40 GB F16 does not fit one 24 GB card, so a fixed -ngl 99 would fail)
#   stage mmproj   vision projector from the bf16 (the tower is untouched by the expert drop)
#   stage quants   every tier, Q6_K first, uploaded to ManniX-ITA/gemma-4-20B-v9-agentic-GGUF (PRIVATE)
#
# The model dir is named after the repo: its basename becomes every GGUF's filename stem and
# general.name. HF token: read from ~/.cache/huggingface/token (0600). Never in argv, never echoed.
set -uo pipefail
W=/workspace
OMK=$W/omnimergekit
STAGE=$W/stage                       # tmpl/ maps/ a_meta/ A_SHA256SUMS.solidpc, landed by the controller
CC=/shared/dev/calib-corpora         # same absolute path as on solidpc: the recipe names it
NAME=gemma-4-20B-v9-agentic
REPO=ManniX-ITA/$NAME
GREPO=ManniX-ITA/$NAME-GGUF
SRC=$W/base/gemma-4-26B-A4B-it
BF=$W/$NAME
GG=$W/gguf/$NAME
MAP=$STAGE/maps/A_lift120_plus_fk12.json
MSHA=24c7b6cb8c707152                # first 16 of the map sha, as build_ab.log recorded it
TMPL=$STAGE/tmpl/chat_template.jinja
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4   # google main @ 35b4173
RECIPE=$CC/recipes/gemma-4-20b-v9-agentic.yaml
CALIB=$CC/builds/gemma-4-20b-v9-agentic/calib_train.txt
LCOMMIT=9724f664e803e70eb8d046a3fac411122ad42ff7      # eval/stack.lock.yaml llama_cpp b9700
PY=$W/miniconda/envs/omnimergekit/bin/python
TF=$HOME/.cache/huggingface/token
# Q6_K first (the eval tier), then Q4_K_M (the :latest tier), then the rest of the default sweep,
# then both CD mix families. CD-Q2_K / CD-IQ3_K_M / CD-IQ2_K stay out: known to collapse or broken.
TIERS="${TIERS:-Q6_K,Q4_K_M,Q8_0,Q6_K_L,Q5_K_L,Q5_K_M,Q5_K_S,Q4_K_L,Q4_K_S,IQ4_NL,IQ4_XS,Q3_K_XL,IQ3_M,Q3_K_L,Q3_K_M,Q3_K_S,IQ3_XS,IQ3_XXS,Q2_K_L,IQ2_M,IQ2_S,IQ2_XS,CD-Q6_K,CD-Q5_K_M,CD-Q4_K_M,CD-IQ4_K_M,CD-Q3_K_L,CD-Q6_K_L,CD-Q5_K_M_L,CD-Q4_K_M_L,CD-IQ3_M_L,CD-IQ3_XS_L,CD-IQ2_M_L,CD-IQ2_XS_L,CD-Q6_K_h,CD-Q5_K_M_h,CD-Q4_K_M_h,CD-IQ3_M_h,CD-IQ3_XS_h,CD-IQ2_M_h,CD-IQ2_XS_h}"
STAGES="${STAGES:-env base build render upload imatrix mmproj quants}"
ts(){ date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say(){ echo "[$(ts)] $*"; }
die(){ echo "[$(ts)] FATAL $*"; exit 1; }
has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
mkdir -p $W/logs $W/base $W/gguf
[ -s "$TF" ] || die "no HF token file"
export HF_TOKEN; HF_TOKEN="$(tr -d '\n' < "$TF")"; : "${HF_TOKEN:?HF_TOKEN must be exported}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1

if has env; then
  say "stage env"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq aria2 cmake build-essential git tmux rsync zstd curl ca-certificates > $W/logs/apt.log 2>&1 || die "apt"
  if [ ! -d $W/llama.cpp/.git ]; then
    git clone -q https://github.com/ggml-org/llama.cpp $W/llama.cpp || die "llama clone"
  fi
  ( cd $W/llama.cpp && git checkout -q $LCOMMIT ) || die "llama checkout $LCOMMIT"
  bash $OMK/eval/pod_eval_bootstrap.sh --install-miniconda --no-repo --cuda-arch "${CUDA_ARCH:-89}" --deps eval-full \
    > $W/logs/bootstrap.out 2>&1 || die "bootstrap (see logs/bootstrap.out)"
  [ -x $W/llama.cpp/build/bin/llama-imatrix ] || die "no llama-imatrix"
  [ -x "$PY" ] || die "no conda env"
  say "llama.cpp $(cd $W/llama.cpp && git rev-parse --short HEAD) $($W/llama.cpp/build/bin/llama-server --version 2>&1 | head -1)"
  "$PY" -c "import torch, transformers, safetensors, gguf, yaml, huggingface_hub as h; print('env', torch.__version__, transformers.__version__, h.__version__)" || die "env imports"
  say "ENV_OK"
fi

if has base; then
  say "stage base"
  mkdir -p "$SRC"
  "$PY" - "$SRC" > $W/logs/base_lfs.txt <<'PYEOF' || die "LFS metadata"
import sys
from huggingface_hub import HfApi, hf_hub_download
repo = "google/gemma-4-26B-A4B-it"
i = HfApi().model_info(repo, files_metadata=True)
print("revision", i.sha)
for s in i.siblings:
    if s.rfilename.endswith(".safetensors"):
        print(s.lfs.sha256, s.size, s.rfilename)
    elif "/" not in s.rfilename and not s.rfilename.endswith((".md", ".gitattributes")):
        hf_hub_download(repo, s.rfilename, revision=i.sha, local_dir=sys.argv[1])
PYEOF
  cat $W/logs/base_lfs.txt
  IN=$(mktemp $W/.aria.XXXXXX); chmod 600 "$IN"
  grep safetensors $W/logs/base_lfs.txt | while read -r _ _ f; do
    printf '%s\n  out=%s\n  header=Authorization: Bearer %s\n' \
      "https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f" "$f" "$HF_TOKEN" >> "$IN"
  done
  aria2c -x 16 -s 16 -k 64M -c --file-allocation=none --console-log-level=warn --summary-interval=120 \
    -d "$SRC" -i "$IN" > $W/logs/aria_base.log 2>&1; RC=$?; rm -f "$IN"
  [ $RC -eq 0 ] || die "aria2c rc=$RC"
  grep safetensors $W/logs/base_lfs.txt | while read -r sha size f; do
    got=$(sha256sum "$SRC/$f" | cut -d' ' -f1)
    [ "$got" = "$sha" ] && echo "  sha256 OK $f" || { echo "  sha256 MISMATCH $f"; exit 1; }
  done || die "base sha256"
  say "BASE_OK"
fi

if has build; then
  say "stage build"
  [ "$(sha256sum "$MAP" | cut -c1-16)" = "$MSHA" ] || die "map sha"
  [ "$(sha256sum "$TMPL" | cut -d' ' -f1)" = "$TSHA" ] || die "template sha"
  if [ ! -f "$BF/.shared_applied" ]; then
    rm -rf "$BF"
    CUDA_VISIBLE_DEVICES="" "$PY" $OMK/scripts/expert_drop.py --source-dir "$SRC" --drop-map "$MAP" --output-dir "$BF" \
      > $W/logs/expert_drop.log 2>&1 || die "expert_drop"
    CUDA_VISIBLE_DEVICES="" "$PY" $OMK/recipes/gemma4/v5_moe_sweep/router_shared_upweight.py --model-dir "$BF" \
      --alpha 1.2 --target mlp.down_proj.weight > $W/logs/upweight.log 2>&1 || die "upweight"
    touch "$BF/.shared_applied"
  fi
  rm -f "$BF"/*.pre_shared_upweight                  # the script's own undo copies; regenerable, never uploaded
  ( cd "$BF" && sha256sum model-*.safetensors > SHA256SUMS && cat SHA256SUMS )
  # The rebuild must be the A that was delivered and judged, byte for byte.
  diff <(sort -k2 "$BF/SHA256SUMS") <(sort -k2 "$STAGE/A_SHA256SUMS.solidpc") > $W/logs/a_sha_diff.txt \
    || { cat $W/logs/a_sha_diff.txt; die "rebuilt shards differ from the solidpc A build"; }
  # Google's repo ships no preprocessor_config.json; vLLM needs one for Gemma 4. It is the
  # feature_extractor block of processor_config.json, synthesised the same way the solidpc A was.
  if [ ! -f "$BF/preprocessor_config.json" ]; then
    "$PY" -c "import json,sys; d=sys.argv[1]; fe=json.load(open(d+'/processor_config.json'))['feature_extractor']; json.dump(fe, open(d+'/preprocessor_config.json','w'), indent=2)" "$BF" \
      || die "preprocessor_config synth"
  fi
  "$PY" -c "import json,sys; a,b=sys.argv[1:]; sys.exit(0 if json.load(open(a))==json.load(open(b)) else 1)" \
    "$BF/preprocessor_config.json" "$STAGE/a_meta/preprocessor_config.json" || die "preprocessor_config differs from the solidpc A build"
  for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json processor_config.json; do
    cmp -s "$BF/$f" "$STAGE/a_meta/$f" || die "$f differs from the solidpc A build"
  done
  cp "$TMPL" "$BF/chat_template.jinja"
  cp "$STAGE/a_meta/generation_config.json" "$BF/generation_config.json"
  [ "$(sha256sum "$BF/chat_template.jinja" | cut -d' ' -f1)" = "$TSHA" ] || die "template copy"
  ls -la "$BF"; du -sh "$BF"
  say "BUILD_OK shards == solidpc A, template official $TSHA"
fi

if has render; then
  say "stage render (AtomicChat corpus through the official template)"
  [ "$(sha256sum "$CC/_render/gemma-4-20b-v9-agentic/chat_template.jinja" | cut -d' ' -f1)" = "$TSHA" ] || die "render template sha"
  if [ ! -s "$CALIB" ]; then
    ( cd $CC && FOUNDRY_MODEL_DIR=$CC/_render/gemma-4-20b-v9-agentic "$PY" tools/build.py --recipe "$RECIPE" ) \
      > $W/logs/render.log 2>&1 || { tail -30 $W/logs/render.log; die "corpus build"; }
  fi
  tail -30 $W/logs/render.log
  for m in '<|im_start|>' '<think>' '<|DSM'; do
    [ "$(grep -aoF "$m" "$CALIB" | wc -l)" -eq 0 ] || die "foreign markup $m in corpus"
  done
  NT=$(grep -aoF '<|turn>' "$CALIB" | wc -l); NC=$(grep -aoF '<|channel>' "$CALIB" | wc -l)
  [ "$NT" -gt 1000 ] || die "corpus has no Gemma-4 turn markup ($NT)"
  say "corpus $(stat -c%s "$CALIB") B sha $(sha256sum "$CALIB" | cut -c1-16) <|turn>=$NT <|channel>=$NC <|tool_call>=$(grep -aoF '<|tool_call>' "$CALIB" | wc -l)"
  say "RENDER_OK"
fi

if has upload; then
  say "stage upload -> $REPO (private)"
  "$PY" - "$REPO" "$BF" > $W/logs/upload.log 2>&1 <<'PYEOF' || die "upload (see logs/upload.log)"
import sys
from huggingface_hub import HfApi
repo, d = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=d, repo_type="model",
                  ignore_patterns=["*.pre_shared_upweight", ".shared_applied", "README.md", ".cache/*"],
                  commit_message="gemma-4-20B-v9-agentic: bf16 weights (agentic map A, shared-FFN alpha 1.2, official chat template)")
i = api.model_info(repo, files_metadata=True)
print("private", i.private)
for s in i.siblings:
    print("  %-44s %s %s" % (s.rfilename, s.size, (s.lfs.sha256[:16] if s.lfs else "")))
PYEOF
  tail -25 $W/logs/upload.log
  "$PY" - "$REPO" "$BF" <<'PYEOF' || die "hub sha256 != local"
import sys
from huggingface_hub import HfApi
repo, d = sys.argv[1], sys.argv[2]
hub = {s.rfilename: s.lfs.sha256 for s in HfApi().model_info(repo, files_metadata=True).siblings if s.lfs}
loc = dict(reversed(l.split()) for l in open(d + "/SHA256SUMS"))
bad = [f for f in loc if hub.get(f) != loc[f]]
print("shards", len(loc), "mismatch", bad)
sys.exit(1 if bad or not loc else 0)
PYEOF
  say "UPLOAD_OK"
fi

if has imatrix; then
  say "stage imatrix (AtomicChat, full corpus, --parse-special)"
  mkdir -p "$GG"
  [ -f "$GG/imatrix.dat" ] && die "$GG/imatrix.dat exists — refusing to clobber an imatrix"
  LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
    --model "$BF" --output-dir "$GG" --imatrix-only \
    --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-save-frequency 1000 --imatrix-timeout 172800 \
    --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local \
    > $W/logs/imatrix.out 2>&1 || die "imatrix (see logs/imatrix.out)"
  ls -la "$GG"; sha256sum "$GG"/imatrix*.dat 2>/dev/null
  say "IMATRIX_OK"
fi

if has mmproj; then
  say "stage mmproj"
  MM=$GG/mmproj-$NAME-F16.gguf
  if [ ! -s "$MM" ]; then
    "$PY" $W/llama.cpp/convert_hf_to_gguf.py "$BF" --mmproj --outtype f16 --outfile "$MM" \
      > $W/logs/mmproj.log 2>&1 || { tail -20 $W/logs/mmproj.log; die "mmproj convert"; }
  fi
  ls -la "$MM"; sha256sum "$MM"
  say "MMPROJ_OK"
fi

if has quants; then
  say "stage quants -> $GREPO (private) tiers: $TIERS"
  [ -s "$GG/imatrix.dat" ] || die "no imatrix"
  "$PY" -c "from huggingface_hub import HfApi; HfApi().create_repo('$GREPO', repo_type='model', private=True, exist_ok=True)" || die "create $GREPO"
  LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
    --model "$BF" --output-dir "$GG" --repo "$GREPO" --only "$TIERS" \
    --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
    --base-model-id google/gemma-4-26B-A4B-it --ngl 99 \
    > $W/logs/quants.out 2>&1 || die "quants (see logs/quants.out)"
  "$PY" -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='$GG/mmproj-$NAME-F16.gguf', path_in_repo='mmproj-$NAME-F16.gguf', repo_id='$GREPO',
                commit_message='mmproj (Gemma 4 vision tower, F16)')" || die "mmproj upload"
  say "QUANTS_OK"
fi
say "###### POD_V9_PUBLISH_DONE stages: $STAGES ######"
