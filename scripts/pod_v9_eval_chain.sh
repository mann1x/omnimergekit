#!/usr/bin/env bash
# pod_v9_eval_chain.sh — ON THE POD: the v7-coder eval suite on gemma-4-20B-v9-agentic Q6_K.
# Q6_K · llama.cpp b9700 · greedy (frozen templates, no --sampler) · q8_0 KV (omk default).
# Cells = the v7-coder card: GPQA-198, AIME-30, MATH500-100, GSM8K-100, ARC full, IFEval-100,
# HumanEval, HumanEval+, LCB-medium-55 v4, LCB-medium-100 v4, MultiPL-E-100 (+ LCB-hard-77).
# Fetches the Q6_K from the HF GGUF repo itself (aria2c, verified against the LFS sha), independent
# of the ollama publisher, then runs two GPU queues. A cell with a summary.json is skipped (resumable).
set -uo pipefail
W=/workspace
STEM=gemma-4-20B-v9-agentic
REPO=ManniX-ITA/$STEM-GGUF
Q6=$W/eval_gguf/q6k/$STEM-Q6_K.gguf   # own copy: the evals never wait on ollama publishing
TOK=$W/$STEM
RES=$W/eval_results
SN=v9agentic-q6k
PY=$W/miniconda/envs/omnimergekit/bin/python
OMK=$W/omnimergekit/eval/omk_eval.py
TPL=$W/omnimergekit/eval/templates
LOG=$W/logs/eval_chain.log
export PATH="$W/miniconda/envs/omnimergekit/bin:$PATH" LM_EVAL_BIN="$W/miniconda/envs/omnimergekit/bin/lm_eval"
export MPE_MODE=native MPE_HARNESS=$W/MultiPL-E HF_ALLOW_CODE_EVAL=1
mkdir -p "$RES" "$W/logs/eval"
L(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

Q0=(gpqa_diamond_full aime_30 math500_100 gsm8k_100 ifeval_100 lcb_v6_77q)
Q1=(humaneval_full humanevalplus_full multipl_e_100 arc_challenge_full lcb_medium_55_v4 lcb_medium_100_v4)

L "===== v9-agentic Q6_K eval chain START ====="
for b in "${Q0[@]}" "${Q1[@]}"; do
  f=$TPL/$b.yaml
  [ -f "$f" ] || { L "FATAL missing template $f"; exit 2; }
  grep -qE "do_sample:[[:space:]]*true" "$f" && { L "FATAL greedy-drift in $b (do_sample true)"; exit 3; }
  grep -qE "temperature:[[:space:]]*0\.0" "$f" || { L "FATAL greedy-drift in $b (temperature not 0.0)"; exit 3; }
done
L "greedy check PASS for ${#Q0[@]}+${#Q1[@]} templates"

L "waiting for $STEM-Q6_K.gguf on $REPO"
want=""
until [ -n "$want" ]; do
  want=$("$PY" -c "
from huggingface_hub import HfApi; from pathlib import Path
t=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
for s in HfApi(token=t).model_info('$REPO', files_metadata=True).siblings:
    if s.rfilename=='$STEM-Q6_K.gguf' and s.lfs: print(s.lfs.sha256)" 2>/dev/null)
  [ -n "$want" ] || sleep 60
done
mkdir -p "$(dirname "$Q6")"
if [ ! -s "$Q6" ] || [ "$(sha256sum "$Q6" | cut -d' ' -f1)" != "$want" ]; then
  IN=$(mktemp -p "$(dirname "$Q6")" .aria.XXXX); chmod 600 "$IN"
  printf '%s\n  out=%s\n  header=Authorization: Bearer %s\n' "https://huggingface.co/$REPO/resolve/main/$STEM-Q6_K.gguf" \
    "$(basename "$Q6")" "$(cat ~/.cache/huggingface/token)" > "$IN"
  aria2c -x 16 -s 16 --console-log-level=warn --summary-interval=0 --allow-overwrite=true -d "$(dirname "$Q6")" -i "$IN" \
    > $W/logs/eval/q6k_download.log 2>&1; rc=$?; rm -f "$IN"
  [ $rc -eq 0 ] || { L "FATAL Q6_K download rc=$rc"; exit 5; }
fi
[ "$(head -c4 "$Q6")" = "GGUF" ] || { L "FATAL $Q6 is not a GGUF"; exit 5; }
got=$(sha256sum "$Q6" | cut -d' ' -f1)
[ -n "$want" ] && [ "$want" = "$got" ] || { L "FATAL Q6_K sha $got != HF $want"; exit 5; }
L "Q6_K verified: $(stat -c %s "$Q6") B sha ${got:0:16} == HF"

run_q(){ # gpu port bench...
  local g=$1 p=$2; shift 2; local b sj rc
  for b in "$@"; do
    sj=$RES/$b/$SN/summary.json
    if [ -f "$sj" ]; then L "GPU$g $b: summary exists, skip"; continue; fi
    if [ "$b" = multipl_e_100 ]; then
      until grep -q "MPE_SETUP_RC=" $W/logs/mpe_setup.log 2>/dev/null; do sleep 30; done
      grep -q "MPE_SETUP_RC=0" $W/logs/mpe_setup.log || { L "GPU$g $b: MPE toolchain setup FAILED, cell skipped"; continue; }
    fi
    local par=2; [ "$b" = lcb_v6_77q ] && par=auto   # v7 arms ran --parallel 2; h77 sizes itself (256k ctx)
    L "GPU$g $b: start (port $p, --parallel $par)"
    ( cd $W/omnimergekit && CUDA_VISIBLE_DEVICES=$g "$PY" "$OMK" --model "$Q6" --tokenizer "$TOK" \
        --template "$b" --backend llama --parallel "$par" --port "$p" --served-name "$SN" --results-dir "$RES" \
        > "$W/logs/eval/$b.log" 2>&1 ); rc=$?
    if [ -f "$sj" ]; then
      L "GPU$g $b: rc=$rc RESULT $("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")"
    else
      L "GPU$g $b: rc=$rc NO summary.json (see logs/eval/$b.log)"
    fi
    p=$((p+1))
  done
  L "GPU$g queue done"
}
run_q 0 8450 "${Q0[@]}" & P0=$!
run_q 1 8470 "${Q1[@]}" & P1=$!
wait $P0 $P1
L "###### EVAL_CHAIN_DONE ######"
