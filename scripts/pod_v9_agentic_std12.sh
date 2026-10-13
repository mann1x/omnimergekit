#!/usr/bin/env bash
# pod_v9_agentic_std12.sh — ON THE POD: build gemma-4-A4B-20B-v9-agentic (the STD12 / fk12
# rung) from the vendor base, publish the bf16 to HF, then build its AtomicChat imatrix.
#
#   stage env      apt + llama.cpp pinned at the stack lock (b9700) + conda env (pod_eval_bootstrap)
#   stage base     google/gemma-4-26B-A4B-it weight shards via aria2c, sha256 checked against HF's LFS record
#   stage build    expert_drop(fk12 map) -> router_shared_upweight alpha=1.2 mlp.down_proj.weight
#                  (byte-for-byte the phase 1 of build_std12_v5_model.sh; no DERN, no floor clamp)
#   stage upload   bf16 -> ManniX-ITA/gemma-4-A4B-20B-v9-agentic (created PRIVATE)
#   stage imatrix  F16 + AtomicChat imatrix: full corpus, --parse-special, both 3090s
#
# The model dir is named after the repo: its basename becomes every GGUF's filename stem and
# general.name. The chat template is the fixed one the delivered STD12 quants carry (sha below),
# not the vendor file in the base dir, which re-injects prior-turn reasoning.
# HF token: read from ~/.cache/huggingface/token (0600). Never in argv, never echoed.
set -uo pipefail
W=/workspace
OMK=$W/omnimergekit
STAGE=$W/stage                       # base_cfg/ tmpl/ calib/ landed by the controller
NAME=gemma-4-A4B-20B-v9-agentic
REPO=ManniX-ITA/$NAME
SRC=$W/base/gemma-4-26B-A4B-it
BF=$W/$NAME
GG=$W/gguf/$NAME
MAP=$OMK/scripts/v8coder_fk12_drop_map.json
TMPL=$STAGE/tmpl/gemma-4-A4B-98e-v7-coder-it-Q6_K.jinja
TSHA=2e4c13d90aa29be968d48e4eecfa8da22e2656d890149e7924b445865490cbac
CALIB=$STAGE/calib/calib_train.txt
CSHA=bf51ee2a575b78700573daf54c0ae1a4cd82843a53b8436b31a653d195edb013
LCOMMIT=9724f664e803e70eb8d046a3fac411122ad42ff7      # eval/stack.lock.yaml llama_cpp b9700
PY=$W/miniconda/envs/omnimergekit/bin/python
TF=$HOME/.cache/huggingface/token
STAGES="${STAGES:-env base build upload imatrix}"
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
  bash $OMK/eval/pod_eval_bootstrap.sh --install-miniconda --no-repo --cuda-arch 86 --deps eval-full \
    > $W/logs/bootstrap.out 2>&1 || die "bootstrap (see logs/bootstrap.out)"
  [ -x $W/llama.cpp/build/bin/llama-imatrix ] || die "no llama-imatrix"
  [ -x "$PY" ] || die "no conda env"
  say "llama.cpp $(cd $W/llama.cpp && git rev-parse --short HEAD) $($W/llama.cpp/build/bin/llama-server --version 2>&1 | head -1)"
  "$PY" -c "import torch, transformers, safetensors, gguf, huggingface_hub as h; print('env', torch.__version__, transformers.__version__, h.__version__)" || die "env imports"
  say "ENV_OK"
fi

if has base; then
  say "stage base"
  mkdir -p "$SRC" && cp -n $STAGE/base_cfg/* $STAGE/base_cfg/.gitattributes "$SRC"/ 2>/dev/null
  "$PY" - > $W/logs/base_lfs.txt <<'PYEOF' || die "LFS metadata"
from huggingface_hub import HfApi
i = HfApi().model_info("google/gemma-4-26B-A4B-it", files_metadata=True)
print("revision", i.sha)
for s in i.siblings:
    if s.rfilename.endswith(".safetensors"):
        print(s.lfs.sha256, s.size, s.rfilename)
PYEOF
  cat $W/logs/base_lfs.txt
  IN=$(mktemp $W/.aria.XXXXXX); chmod 600 "$IN"
  for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
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
  cp "$TMPL" "$BF/chat_template.jinja"
  [ "$(sha256sum "$BF/chat_template.jinja" | cut -d' ' -f1)" = "$TSHA" ] || die "template copy"
  cp $STAGE/generation_config.json "$BF/generation_config.json"
  ( cd "$BF" && sha256sum model-*.safetensors > SHA256SUMS && cat SHA256SUMS )
  ls -la "$BF"; du -sh "$BF"
  say "BUILD_OK"
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
                  commit_message="gemma-4-A4B-20B-v9-agentic: bf16 weights (STD12 / fk12 drop map, shared-FFN alpha 1.2)")
i = api.model_info(repo, files_metadata=True)
print("private", i.private)
for s in i.siblings:
    print("  %-44s %s %s" % (s.rfilename, s.size, (s.lfs.sha256[:16] if s.lfs else "")))
PYEOF
  tail -25 $W/logs/upload.log
  # every shard on the hub must be the shard on disk
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
  [ "$(sha256sum "$CALIB" | cut -d' ' -f1)" = "$CSHA" ] || die "calib sha"
  for m in '<|im_start|>' '<think>' '<|DSM'; do
    [ "$(grep -aoF "$m" "$CALIB" | wc -l)" -eq 0 ] || die "foreign markup $m in corpus"
  done
  say "corpus <|turn>=$(grep -aoF '<|turn>' "$CALIB" | wc -l) <|tool_call>=$(grep -aoF '<|tool_call>' "$CALIB" | wc -l)"
  mkdir -p "$GG"
  [ -f "$GG/imatrix.dat" ] && die "$GG/imatrix.dat exists — refusing to clobber an imatrix"
  LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
    --model "$BF" --output-dir "$GG" --imatrix-only \
    --cal-data "$CALIB" --imatrix-chunks -1 --imatrix-save-frequency 1000 --imatrix-timeout 172800 \
    --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local --ngl 99 \
    > $W/logs/imatrix.out 2>&1 || die "imatrix (see logs/imatrix.out)"
  ls -la "$GG"; sha256sum "$GG"/imatrix*.dat 2>/dev/null
  say "IMATRIX_OK"
fi
say "###### POD_V9_STD12_DONE stages: $STAGES ######"
