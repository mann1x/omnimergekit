#!/usr/bin/env bash
# Evaluate the ATOMICCHAT-imatrix Q6_K of Qwen3.8-27B-Omnimerge-v6 on bs2 GPU1.
#
# Purpose: measure what the AtomicChat calibration corpus buys over the imatrix
# the published tiers were built with. That is only answerable if the two cells
# differ in NOTHING BUT the imatrix, so this run pins the published cohort's
# basis exactly: same templates, same sampler profile, same quant tier (Q6_K),
# same engine. Comparator cells (sampler=recommended):
#     gpqa_diamond_full 0.7677 · humaneval_full_think 0.9817 · ifeval_100 0.9600
#     lcb_v6_77q 0.8831 · multipl_e_100 0.8867
#
# Bench order is cheapest-first so the comparison starts landing early; LCB is
# last because it is the multi-hour cell.
set -uo pipefail
ts(){ date -u +%H:%M:%S; }
say(){ echo "[$(ts)Z] $*"; }

GGUF=/mnt/sdc/ml/omnimerge-v6/Qwen3.8-27B-Omnimerge-v6-Q6_K-AC.gguf
TOK=/mnt/sdc/ml/omnimerge-v6/tokenizer
RES=/srv/ml/eval_results/v6_ac
NAME=qwenomnimergev6_q6k_AC
PORT=8097
PROFILE=/srv/ml/repos/omnimergekit/eval/models/qwen3_6.yaml

exec 9>/var/lock/v6_ac_eval.lock
flock -n 9 || { say "already running - refusing"; exit 1; }

[ -s "$GGUF" ] || { say "ABORT: no AC quant at $GGUF"; exit 2; }
[ -f "$TOK/tokenizer.json" ] || { say "ABORT: no tokenizer at $TOK"; exit 3; }
[ -f "$PROFILE" ] || { say "ABORT: no sampler profile"; exit 4; }

# GPU1 only -- GPU0 is not ours to take.
export CUDA_VISIBLE_DEVICES=1
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -dc '0-9')
[ "${USED:-0}" -lt 2000 ] || { say "ABORT: GPU1 busy (${USED} MiB used)"; exit 5; }
say "GPU1 free (${USED} MiB). quant=$(stat -c%s "$GGUF") B"

# Prove the served file really carries the AtomicChat imatrix before spending GPU hours.
/srv/ml/envs/envs/omnimergekit/bin/python3.11 - "$GGUF" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
def s(k):
    f = r.fields.get(k)
    if not f: return None
    p = f.parts[f.data[0]]
    try: return bytes(p).decode()
    except Exception: return int(p[0])
ds, ch = s("quantize.imatrix.dataset"), s("quantize.imatrix.chunks_count")
print("  imatrix dataset=%s chunks=%s" % (ds, ch))
assert ds and "calib_train" in str(ds), "served quant is NOT AtomicChat-calibrated"
assert int(ch) > 9000, "chunk count %s is not the full AC run" % ch
print("  AC imatrix CONFIRMED in the served artifact")
PY
[ $? -eq 0 ] || { say "ABORT: imatrix provenance gate failed"; exit 6; }

export LLAMA_BIN=/opt/llama.cpp/build/bin
export OMK_PYTHON=/srv/ml/envs/envs/omnimergekit/bin/python3.11
# omk_eval shells out to the `lm-eval` CONSOLE SCRIPT, not the module, so the
# env's bin must be on PATH. Without this every template dies instantly with
# FileNotFoundError: 'lm-eval' and the suite burns through all five in seconds.
export PATH=/srv/ml/envs/envs/omnimergekit/bin:$PATH
command -v lm-eval >/dev/null || { say "ABORT: lm-eval not on PATH"; exit 7; }
say "lm-eval: $(command -v lm-eval)"
export HF_ALLOW_CODE_EVAL=1
mkdir -p "$RES"
cd /srv/ml/repos/omnimergekit

for T in ifeval_100 humaneval_full_think gpqa_diamond_full multipl_e_100 lcb_v6_77q; do
  say ">>> $T"
  "$OMK_PYTHON" eval/omk_eval.py --backend llama --template "$T" --quant q6_k \
      --model "$GGUF" --tokenizer "$TOK" --served-name "$NAME" --port "$PORT" \
      --results-dir "$RES" --sampler-profile "$PROFILE" --sampler recommended 2>&1 | tail -6
  say "<<< $T rc=${PIPESTATUS[0]}"
done
say "V6_AC_EVAL_DONE"
