#!/bin/bash
# HumanEval on the 27B target: PrismML Ternary Bonsai 2, our no-gradient PT2 trace-mix ternary, and base Qwen3.8-27B.
# Same lm-eval invocation as bonsai4b_he_chain3.sh. Ternary GGUFs on the PrismML engine, base bf16 on vLLM.
set -u
H=/mnt/sdc/opencoti-scratch/v6-ternary/he27; L=$H/he_chain.log; TK=/mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
mkdir -p $H/cache; ts() { date -u +%H:%M:%SZ; }
export HF_ALLOW_CODE_EVAL=1
ev() { # name, port
  /root/anaconda3/envs/omnimergekit/bin/lm_eval --model local-completions \
    --model_args "model=$1,base_url=http://127.0.0.1:$2/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$TK,max_gen_toks=2048" \
    --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$1 --log_samples --output_path $H/out_$1 > $H/lm_eval.$1.log 2>&1
  echo "[$(ts)] $1 rc=$? $(grep -E 'humaneval' $H/lm_eval.$1.log | tail -1 | cut -c1-200)" >> $L
}
for A in "bonsai2-27b:/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf" "ours-pt2mix-27b:/mnt/sdc/opencoti-models/sdc-ml/qwen38-base/Qwen3.8-27B-pt2mix-PQ2_0.gguf"; do
  name=${A%%:*}; gg=${A#*:}
  LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-server" -m "$gg" --alias $name --host 127.0.0.1 --port 8131 -ngl 99 -c 16384 -np 2 --no-warmup > $H/server.$name.log 2>&1 &
  SP=$!; i=0
  until curl -sf http://127.0.0.1:8131/health 2>/dev/null | grep -q '"ok"'; do i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "[$(ts)] FATAL server $name died" >> $L; continue 2; }; [ $i -gt 240 ] && { kill $SP; continue 2; }; sleep 5; done
  ev $name 8131
  kill $SP; wait $SP 2>/dev/null
done
CUDA_VISIBLE_DEVICES=1 /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8132 --model $TK \
  --served-model-name qwen3.8-27b-base --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.66 --max-num-seqs 8 --max-num-batched-tokens 8192 > $H/vllm.base.log 2>&1 &
SP=$!; i=0
until curl -sf http://127.0.0.1:8132/v1/models 2>/dev/null | grep -q 'qwen3.8-27b-base'; do i=$((i+1)); kill -0 $SP 2>/dev/null || { echo "[$(ts)] FATAL vllm base died" >> $L; exit 1; }; [ $i -gt 240 ] && { kill $SP; exit 1; }; sleep 5; done
ev qwen3.8-27b-base 8132
kill $SP; wait $SP 2>/dev/null
echo "[$(ts)] === HE27_DONE ===" >> $L
