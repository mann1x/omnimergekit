#!/usr/bin/env bash
# Republish the FULL Qwen3.8-27B-Omnimerge-v6 ladder to ollama.com.
#
# WHY: the whole HF ladder was re-quantized on the AtomicChat imatrix
# (imatrix.dat sha256 50f44d8f…, confirmed identical on HF), and the 5 low-bit
# tiers were rebuilt again today to restore the blk.64 MTP Q4_K floor. NOTHING
# was ever pushed to ollama -- neither rebuild script has an ollama step -- so
# every published tag currently serves pre-AC weights. All 19 tiers are stale.
#
# Uses the CANONICAL scripts/ollama_publish_tiers.sh: HF -> text + vision-<tier>
# in one pass, ONE tier resident at a time, resumable via per-tier .done markers.
# "Override, never fork": every v6-specific value below is an env override.
#
# PARAMS are taken from the PUBLISHED manifest of mannix/omnimerge-v6:Q6_K, not
# from the script's CoderX defaults:
#     draft_num_predict 4  temperature 1  top_p 0.95  top_k 20
#     min_p 0  presence_penalty 0  repeat_penalty 1        :latest -> Q4_K_M
# The one DEVIATION is num_ctx: the published tags carry none (ollama default),
# and emit_params always writes one. 32768 is what every other publish in this
# program ships. This is a stated change, not a silent one.
#
# RENDERER/PARSER read from the PUBLISHED CONFIG BLOB, not guessed from the family.
# The shipped v6 combination is RENDERER qwen3.8 + PARSER **qwen3.5** -- a deliberate
# mismatch. The first attempt set both to qwen3.8 because I read only the params blob;
# renderer/parser/requires live in the CONFIG blob. 5 tiers (10 tags) went out wrong
# and are re-pushed here by deleting their .done markers.
#   old Q6_K control : renderer qwen3.8  parser qwen3.5  requires 0.32.12
#   first attempt    : renderer qwen3.8  parser qwen3.8  requires ABSENT
# `requires` is emitted by ollama itself, not settable here -- verify it comes back on
# the first re-pushed tier and STOP if it does not.
# requires one, because ollama returns HTTP 200 for options it silently drops.
# qwen3.8 is the correct family for this base and bs2 runs ollama 0.33.2
# (>= 0.32.12, the floor for that renderer) with it present in the binary.
set -euo pipefail
eval "$(grep -m1 '^export HF_TOKEN=' ~/.bashrc)"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

export PY=/srv/ml/envs/envs/omnimergekit/bin/python
export HF_REPO=ManniX-ITA/Qwen3.8-27B-Omnimerge-v6-MTP-GGUF
export OL_BASE=mannix/omnimerge-v6
export GGUF_STEM=Qwen3.8-27B-Omnimerge-v6
export MMPROJ=/mnt/sdc/omnimerge_v6/mmproj/mmproj-Qwen3.8-27B-Omnimerge-v6-F16.gguf
export WORK=/mnt/sdc/omnimerge_v6/ollama_pub
export STAGE=/mnt/sdc/omnimerge_v6/ollama_stage
export LATEST_TIER=Q4_K_M

export OL_RENDERER=qwen3.8
export OL_PARSER=qwen3.5
export OL_NUM_CTX=32768
export OL_TEMPERATURE=1
export OL_TOP_P=0.95
export OL_TOP_K=20
export OL_MIN_P=0
export OL_PRESENCE_PENALTY=0
export OL_REPEAT_PENALTY=1
export DRAFT_N=4

# Smallest tier first: a config or gate error then surfaces on a 9.6 GB download
# instead of a 29 GB one. The store lives on /mnt/sdc (daemon has
# OLLAMA_MODELS=/mnt/sdc/ollama/models), so the root-fs 200G floor is untouched.
export TIERS="IQ2_S IQ2_M Q2_K IQ3_XXS IQ3_XS Q3_K_S IQ3_M Q3_K_M Q3_K_L Q3_K_XL IQ4_XS IQ4_NL Q4_K_M Q4_K_L Q5_K_M Q5_K_L Q6_K Q6_K_L Q8_0"

cd /srv/ml/repos/omnimergekit
bash scripts/ollama_publish_tiers.sh
echo "V6_OLLAMA_REPUBLISH_DONE"
