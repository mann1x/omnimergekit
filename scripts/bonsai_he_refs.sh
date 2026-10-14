#!/bin/bash
# HumanEval yardstick for published ternary models, same invocation as bonsai4b_he_chain3.sh (recorded MicroCoder-era
# run: lm-eval `humaneval`, local-completions, greedy, num_concurrent=2), served bf16 by vLLM. Each ternary is paired
# with its own full-precision base as the control.
set -u
H=/mnt/sdc/opencoti-scratch/bonsai4b/he_refs; L=$H/he_chain.log; M=/mnt/sdc/opencoti-models
VPY=/root/anaconda3/envs/vllm/bin/python; PORT=8151
mkdir -p $H/cache; ts() { date -u +%H:%M:%SZ; }
export HF_ALLOW_CODE_EVAL=1
he() { # name, model dir, tokenizer dir
  local name=$1 dir=$2 tk=$3
  CUDA_VISIBLE_DEVICES=1 $VPY -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT --model $dir --tokenizer $tk \
    --served-model-name $name --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.25 --max-num-seqs 16 > $H/vllm.$name.log 2>&1 &
  local srv=$! n=0
  until curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "\"$name\""; do
    kill -0 $srv 2>/dev/null || { echo "[$(ts)] FATAL server $name died: $(grep -E 'Error' $H/vllm.$name.log | tail -1 | cut -c1-240)" >> $L; return 1; }
    n=$((n+1)); [ $n -gt 180 ] && { echo "[$(ts)] FATAL server $name not ready" >> $L; kill $srv; return 1; }; sleep 5
  done
  /root/anaconda3/envs/omnimergekit/bin/lm_eval --model local-completions \
    --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$tk,max_gen_toks=2048" \
    --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$name --log_samples --output_path $H/out_$name > $H/lm_eval.$name.log 2>&1
  echo "[$(ts)] $name rc=$? $(grep -E 'humaneval' $H/lm_eval.$name.log | tail -1 | cut -c1-200)" >> $L
  kill $srv; wait $srv 2>/dev/null
}
he qwen3-4b-base        $M/bonsai4b/Qwen3-4B        $M/bonsai4b/Qwen3-4B
he prism-ternary-4b     $M/bonsai4b/prism-unpacked  $M/bonsai4b/Qwen3-4B
he qwen3-1.7b-base      $M/bonsai17/Qwen3-1.7B      $M/bonsai17/Qwen3-1.7B
he prism-ternary-1.7b   $M/bonsai17/prism-unpacked  $M/bonsai17/Qwen3-1.7B
he alllwang-ternary-1.7b $M/bonsai17/alllwang       $M/bonsai17/Qwen3-1.7B
echo "[$(ts)] === HE_REFS_DONE ===" >> $L
