W=/mnt/sdc/opencoti-scratch/bonsai17; M=/mnt/sdc/opencoti-models/bonsai17
for arm in top86 gptq; do
  /root/anaconda3/envs/omnimergekit/bin/python -u /srv/ml/scripts/bonsai17_qat.v3.py --base $M/Qwen3-1.7B --eval-text /srv/ml/ppl-corpus/wiki.test.mid.raw --eval-text-b $W/traces/heldout_traces.txt --init $arm --train-parquet $M/data/fineweb-edu-10BT-000.parquet --load $W/r2_${arm}_8k/latent.pt --steps 0 --out $W/r4_eval_${arm}_8k > $W/r4_eval_${arm}_8k.out 2>&1
  echo "[$(date -u +%H:%M:%SZ)] ours ${arm} 8k (rerun): $(grep -E 'EVAL|Error' $W/r4_eval_${arm}_8k.out | tail -1)" >> $W/r4_support.log
done
echo "=== R4_SUPPORT_RERUN_DONE ===" >> $W/r4_support.log
