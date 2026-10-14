#!/bin/bash
# Qwen3.5-4B: the f_lr2e-5 reference run PLUS the Qwen3.8 AtomicChat calibration corpus (~5M tokens) spread
# evenly through the stream. Same reference rows, schedule shape, lr, eval rows and HumanEval invocation as
# bonsai27_full4b.sh; the run is longer by exactly the added rows (every row seen once, as in the reference).
# Reference f_lr2e-5: wiki KL 0.779 (online), HumanEval 17.07%.
set -u
W=/mnt/sdc/opencoti-scratch/bonsai4b; B=/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B
K0=$W/cache_r1mix; K=$W/cache_r1mix_ac; AC=/mnt/sdc/ml/omnimerge-v6/calib_train.txt; L=$W/full27_ac.log
PY=/root/anaconda3/envs/omnimergekit/bin/python; ts() { date -u +%H:%M:%SZ; }
NAME=${NAME:-f_lr2e-5_ac}; LR=${LR:-2e-5}
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/srv/ml/pylibs/fla:/srv/ml/scripts HF_ALLOW_CODE_EVAL=1
echo "[$(ts)] start $NAME; AC sha $(sha256sum $AC | cut -c1-16); $(df -h /mnt/sdc | tail -1)" >> $L
if ! grep -q AC_CACHE_DONE $K/cache.log 2>/dev/null; then
  $PY /srv/ml/scripts/bonsai27_ac_cache.py --base $B --ref-cache $K0 --text $AC --out $K > $K.out 2>&1 \
    || { echo "[$(ts)] FATAL ac cache: $(tail -2 $K.out | cut -c1-200)" >> $L; exit 1; }
fi
grep -E 'GATE|rows of|AC_CACHE_DONE' $K/cache.log | sed 's/^/   /' >> $L
N=$($PY -c "import json;print(json.load(open('$K/cache.json'))['n_seq'])"); STEPS=$((N / 8))
echo "[$(ts)] cache $N rows -> $STEPS steps (reference 2000)" >> $L
mkdir -p $W/$NAME
$PY /srv/ml/scripts/bonsai27_qat.py --phase train --device cuda:0 --base $B --cache $K --out $W/$NAME --steps $STEPS --lr $LR \
  --eval-every 250 --save-every 1000 --eval-teacher online > $W/$NAME/run.out 2>&1
echo "[$(ts)] $NAME train rc=$? $(grep -E '^EVAL' $W/$NAME/run.out | tail -1 | cut -c1-300)" >> $L
grep -q '"tag": "step'$STEPS'"' $W/$NAME/run.out || { echo "[$(ts)] FATAL train did not reach step $STEPS: $(grep -E 'Error|error' $W/$NAME/run.out | tail -1 | cut -c1-200)" >> $L; exit 1; }
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
echo "[$(ts)] === FULL27_AC_DONE $NAME ===" >> $L
