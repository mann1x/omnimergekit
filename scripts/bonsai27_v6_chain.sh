#!/bin/bash
# Omnimerge v6 (Qwen3.8-27B) ternary recovery, end to end on GPU1 alone:
#   split the generated corpus by family -> teacher cache (top-64 over the exact stream) -> train in the PrismML basis ->
#   dense export + folded tensors -> HumanEval of the ternary and of v6 bf16 (recorded MicroCoder-era invocation).
# Waits for the corpus generation to finish. Each stage is skipped when its output exists.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai27; C=$W/corpus_v6; B=/mnt/sdc/opencoti-models/hf/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6
PG=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; K=$W/cache_v6; R=$W/${RUN:-r1_v6}; L=$W/v6_chain.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; ts() { date -u +%H:%M:%SZ; }
LR=${LR:-2e-5}; STEPS=${STEPS:-2000}; BS=${BS:-8}
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/srv/ml/pylibs/fla:/srv/ml/scripts HF_ALLOW_CODE_EVAL=1
until grep -qE "CORPUS27_DONE" <(tail -n +5 $C/corpus.log); do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 500 ]; do sleep 10; done
if [ ! -s $C/train_code.jsonl ]; then
  $PY - $C <<'PYEOF' >> $L 2>&1 || { echo "[$(ts)] FATAL split" >> $L; exit 1; }
import collections, json, sys
c = sys.argv[1]
for st in ("train", "eval"):
    by = collections.defaultdict(list)
    for line in open(f"{c}/gen_{st}.jsonl"):
        by[json.loads(line)["fam"]].append(line)
    for fam, rows in by.items():
        rows.sort(key=lambda x: json.loads(x)["id"])
        open(f"{c}/{st}_{fam}.jsonl", "w").writelines(rows)
        print(f"split {st}_{fam}: {len(rows)} conversations, {sum(json.loads(x)['gen_tokens'] for x in rows)} generated tokens")
PYEOF
fi
MIX=$C/train_code.jsonl=0.4,$C/train_tech.jsonl=0.2,$C/train_general.jsonl=0.25,$C/agent_train.jsonl=0.15
if [ ! -f $K/cache.json ]; then
  echo "[$(ts)] teacher cache start" >> $L
  $PY /srv/ml/scripts/bonsai27_qat.py --phase cache --device cuda:0 --base $B --cache $K --n-seq $((STEPS * BS)) \
    --train-parquet /mnt/sdc/opencoti-models/bonsai17/data/fineweb-edu-10BT-000.parquet --mix-frac 0.25 --mix-text $MIX \
    --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $C/eval_code.jsonl,$C/eval_tech.jsonl,$C/eval_general.jsonl,$C/agent_eval.jsonl \
    > $K.out 2>&1 || { echo "[$(ts)] FATAL cache: $(tail -1 $K.out | cut -c1-200)" >> $L; exit 1; }
  echo "[$(ts)] teacher cache done: $(grep -E '^eval:' $K.out | cut -d';' -f2- | tr '\n' ' ' | cut -c1-300)" >> $L
fi
if [ ! -f $R/latent.pt ] || ! grep -q QAT27_DONE $R/run.out 2>/dev/null; then
  mkdir -p $R
  echo "[$(ts)] train start lr $LR steps $STEPS bs $BS" >> $L
  $PY /srv/ml/scripts/bonsai27_qat.py --phase train --device cuda:0 --base $B --cache $K --out $R --basis prism --prism-gguf $PG \
    --steps $STEPS --bs $BS --lr $LR --fp-lr $LR --eval-every 250 --save-every 500 > $R/run.out 2>&1
  echo "[$(ts)] train rc=$? $(grep -E '^EVAL' $R/run.out | tail -1 | cut -c1-330)" >> $L
  grep -q QAT27_DONE $R/run.out || { echo "[$(ts)] FATAL train: $(grep -E 'Error' $R/run.out | tail -1 | cut -c1-200)" >> $L; exit 1; }
fi
if [ ! -f $R/export/bonsai_export.json ]; then
  $PY /srv/ml/scripts/bonsai27_export.py --base $B --latent $R/latent.pt --out $R/export --prism-gguf $PG --qdir $R/q > $R/export.log 2>&1 \
    || { echo "[$(ts)] FATAL export: $(tail -1 $R/export.log | cut -c1-200)" >> $L; exit 1; }
  echo "[$(ts)] export done" >> $L
fi
H=$W/he; mkdir -p $H/cache
he() { # name, model dir
  [ -n "$(ls $H/out_$1/*/results_*.json 2>/dev/null)" ] && return 0
  /root/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8151 --model $2 --served-model-name $1 \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.66 --max-num-seqs 8 --max-num-batched-tokens 8192 > $H/vllm.$1.log 2>&1 &
  local srv=$! n=0
  until curl -sf http://127.0.0.1:8151/v1/models 2>/dev/null | grep -q "\"$1\""; do
    kill -0 $srv 2>/dev/null || { echo "[$(ts)] FATAL server $1 died: $(grep -E 'Error' $H/vllm.$1.log | tail -1 | cut -c1-200)" >> $L; return 1; }
    n=$((n+1)); [ $n -gt 360 ] && { kill $srv; echo "[$(ts)] FATAL server $1 not ready" >> $L; return 1; }; sleep 5
  done
  /root/anaconda3/envs/omnimergekit/bin/lm_eval --model local-completions \
    --model_args "model=$1,base_url=http://127.0.0.1:8151/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=$B,max_gen_toks=2048" \
    --tasks humaneval --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
    --use_cache $H/cache/$1 --log_samples --output_path $H/out_$1 > $H/lm_eval.$1.log 2>&1
  echo "[$(ts)] $1 HumanEval rc=$? $(grep -E 'humaneval' $H/lm_eval.$1.log | tail -1 | cut -c1-120)" >> $L
  kill $srv; wait $srv 2>/dev/null
}
he ternary-v6-${RUN:-r1_v6} $R/export
he omnimerge-v6-bf16 $B
echo "[$(ts)] === V6_CHAIN_DONE ===" >> $L
