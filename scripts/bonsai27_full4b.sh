#!/bin/bash
# Full-length Qwen3.5-4B run of bonsai27_qat.py (same stream / schedule / eval rows as the AdamW run r1_mix25), then
# dense export and HumanEval with the recorded MicroCoder-era invocation. AdamW reference: wiki KL 0.800, HumanEval 10.37%.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; C=$W/corpus_v1; B=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B; K=$W/cache_r1mix; L=$W/full27.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; ts() { date -u +%H:%M:%SZ; }
NAME=${NAME:?NAME must be set}; LR=${LR:?LR must be set}
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/srv/ml/pylibs/fla:/srv/ml/scripts HF_ALLOW_CODE_EVAL=1
if [ "$($PY -c "import json;print(json.load(open('$K/cache.json'))['n_seq'])")" -lt 16000 ]; then
  $PY /srv/ml/scripts/bonsai27_qat.py --phase cache --device cuda:0 --base $B --cache $K --n-seq 16000 \
    --train-parquet /mnt/sdc/opencoti-models/bonsai17/data/fineweb-edu-10BT-000.parquet --mix-frac 0.25 \
    --mix-text $C/train_code.jsonl=0.4,$C/train_tech.jsonl=0.2,$C/train_general.jsonl=0.25,$C/agent_train.jsonl=0.15 \
    --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $C/eval_code.jsonl,$C/eval_tech.jsonl,$C/eval_general.jsonl,$C/agent_eval.jsonl \
    > $K.extend.out 2>&1 || { echo "[$(ts)] FATAL cache extend" >> $L; exit 1; }
  echo "[$(ts)] cache extended to 16000" >> $L
fi
mkdir -p $W/$NAME
$PY /srv/ml/scripts/bonsai27_qat.py --phase train --device cuda:0 --base $B --cache $K --out $W/$NAME --steps 2000 --lr $LR \
  --eval-every 250 --save-every 1000 --eval-teacher online > $W/$NAME/run.out 2>&1
echo "[$(ts)] $NAME train rc=$? $(grep -E '^EVAL' $W/$NAME/run.out | tail -1 | cut -c1-300)" >> $L
$PY /srv/ml/scripts/bonsai27_export.py --base $B --latent $W/$NAME/latent.pt --out $W/$NAME/export > $W/$NAME/export.log 2>&1 || { echo "[$(ts)] FATAL export" >> $L; exit 1; }
H=$W/he_$NAME; mkdir -p $H/cache
/root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8151 --model $W/$NAME/export \
  --served-model-name $NAME --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.30 --max-num-seqs 16 --max-num-batched-tokens 8192 > $H/vllm.log 2>&1 &
SRV=$!; n=0
until curl -sf http://127.0.0.1:8151/v1/models 2>/dev/null | grep -q "\"$NAME\""; do
  kill -0 $SRV 2>/dev/null || { echo "[$(ts)] FATAL server died" >> $L; exit 1; }
  n=$((n+1)); [ $n -gt 180 ] && { kill $SRV; echo "[$(ts)] FATAL server not ready" >> $L; exit 1; }; sleep 5
done
/root/anaconda3/envs/omnimergekit/bin/lm_eval --model local-completions \
  --model_args "model=$NAME,base_url=http://127.0.0.1:8151/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$B,max_gen_toks=2048" \
  --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
  --use_cache $H/cache/$NAME --log_samples --output_path $H/out_$NAME > $H/lm_eval.log 2>&1
echo "[$(ts)] $NAME HumanEval rc=$? $(grep -E 'humaneval' $H/lm_eval.log | tail -1 | cut -c1-120)" >> $L
kill $SRV; wait $SRV 2>/dev/null
echo "[$(ts)] === FULL27_DONE $NAME ===" >> $L
