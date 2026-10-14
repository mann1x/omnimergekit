#!/bin/bash
# Qwen3.5-4B ternary students on HumanEval with the invocation RECORDED in the saved MicroCoder-era results
# (4b_phase1/eval_results/humaneval_qwen3.5-4b-base): lm-eval `humaneval`, local-completions, num_concurrent=2,
# model_args max_gen_toks=2048, gen_kwargs temperature=0.0,top_p=1.0 (task cap 1024 and its stop list apply).
# Difference from those runs: served bf16 by vLLM, not Q6_K by llama.cpp; the base model runs first as the control.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; BASE=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; H=$W/he3; L=$H/he_chain.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; VPY=/root/anaconda3/envs/vllm/bin/python; PORT=8151
mkdir -p $H; ts() { date -u +%H:%M:%SZ; }
export HF_ALLOW_CODE_EVAL=1 PYTHONPATH=/srv/ml/pylibs/fla
for r in r1_mix25 r1_ctrl; do
  if [ ! -f $W/$r/export/bonsai_export.json ]; then
    echo "[$(ts)] export $r" >> $L
    $PY /srv/ml/scripts/bonsai4b_export.py --base $BASE --latent $W/$r/latent.pt --out $W/$r/export > $W/$r/export.log 2>&1 || { echo "[$(ts)] FATAL export $r" >> $L; exit 1; }
  fi
  $PY /srv/ml/scripts/bonsai4b_qat.py --base $BASE --eval-only $W/$r/export --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --device cuda:1 > $W/$r/export_gate.out 2>&1
  echo "[$(ts)] gate $r $(grep -E '^EVAL' $W/$r/export_gate.out | cut -c1-140)" >> $L
done
unset PYTHONPATH
he() { # name, model dir
  local name=$1 dir=$2
  CUDA_VISIBLE_DEVICES=1 $VPY -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT --model $dir \
    --served-model-name $name --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.30 --max-num-seqs 16 --max-num-batched-tokens 8192 > $H/vllm.$name.log 2>&1 &
  local srv=$! n=0
  until curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "\"$name\""; do
    kill -0 $srv 2>/dev/null || { echo "[$(ts)] FATAL server $name died" >> $L; return 1; }
    n=$((n+1)); [ $n -gt 180 ] && { echo "[$(ts)] FATAL server $name not ready" >> $L; kill $srv; return 1; }; sleep 5
  done
  echo "[$(ts)] serving $name" >> $L
  /root/anaconda3/envs/omnimergekit/bin/lm_eval --model local-completions \
    --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$BASE,max_gen_toks=2048" \
    --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$name --log_samples --output_path $H/out_$name > $H/lm_eval.$name.log 2>&1
  echo "[$(ts)] $name rc=$? $(grep -E 'humaneval' $H/lm_eval.$name.log | tail -1 | cut -c1-200)" >> $L
  kill $srv; wait $srv 2>/dev/null
}
mkdir -p $H/cache
he qwen3.5-4b-base $BASE
he ternary-r1-mix25 $W/r1_mix25/export
he ternary-r1-ctrl $W/r1_ctrl/export
echo "[$(ts)] === HE_CHAIN_DONE ===" >> $L
