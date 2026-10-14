#!/usr/bin/env bash
# after bonsai_pt2_full.py finishes: export -> pack -> gates -> KLD vs base on the prism fork
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
L=${PT2_DIR:?PT2_DIR must be set}; TAG=${PT2_TAG:?PT2_TAG must be set}
i=0; until grep -qE 'PT2_FULL_DONE|Traceback|Error' $L/run.log; do i=$((i+1)); [ $i -ge 960 ] && { echo "FATAL run not done in 4h"; exit 1; }; sleep 15; done
grep -q PT2_FULL_DONE $L/run.log || { echo "FATAL the ternarization run failed"; grep -E "Traceback|Error" -A3 $L/run.log | tail -8; exit 1; }
echo "[$(ts)] ternarization done: $(grep -cE 'fitted|loaded' $L/run.log) groups"
PY=/root/anaconda3/envs/omnimergekit/bin/python; export PYTHONPATH=/srv/ml/dca-cleanroom/llama.cpp/gguf-py
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
QUANT=/srv/ml/repos/omnimergekit/tools/opencoti-quantize/bin/opencoti-quantize
META=/srv/ml/scripts/gguf_read_prism_meta.py
DIR=/mnt/sdc/opencoti-models/sdc-ml/qwen38-base; REFD=/mnt/sdc/opencoti-models/bonsai2
GG=$DIR/Qwen3.8-27B-BF16.gguf; F=${PT2_TMP:-$DIR}/Qwen3.8-27B-$TAG-folded-BF16.gguf
BASE=/mnt/sdc/opencoti-scratch/v6-ternary/base/qwen38-bf16-c2048-prism.kld
TEXT=/srv/ml/ppl-corpus/wiki.test.mid.raw
ppl(){ LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-perplexity" "$@" -f "$TEXT" -c 2048 --chunks 8 -b 2048 -ub 512 -ngl 99; }
show(){ grep -E "^(Mean  *PPL\(Q\) |Mean  *KLD|Same top p)" "$1" | sed 's/^/     /'; }
"$PY" /srv/ml/scripts/bonsai_hadamard_fold.py --src "$GG" --ref "$REFD/Ternary-Bonsai-2-27B-PQ2_0.gguf" --override-dir $L/q --out "$F" --device cuda:1 > $L/export.log 2>&1
grep -q FOLD_OK $L/export.log || { echo "FATAL export"; tail -8 $L/export.log; exit 1; }
echo "[$(ts)] exported $(stat -c%s "$F") B"
ppl -m "$F" --kl-divergence-base "$BASE" --kl-divergence > $L/kld-bf16.log 2>&1 || tail -5 $L/kld-bf16.log
echo "[$(ts)] pre-pack (bf16 holding exact +-d):"; show $L/kld-bf16.log
for T in PQ2_0 PTQ1_0; do
  Q=$DIR/Qwen3.8-27B-$TAG-$T.gguf
  /bin/sh "$QUANT" --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 --token-embedding-type "$T" --output-tensor-type "$T" "$F" "$Q" "$T" 32 > $L/quantize-$T.log 2>&1 || { echo "FATAL quantize $T"; tail -6 $L/quantize-$T.log; exit 1; }
  sha256sum "$Q" > "$Q.sha256"; echo "[$(ts)] $T $(stat -c%s "$Q") B sha $(cut -c1-16 "$Q.sha256")"
  "$PY" "$META" "$Q" --tensors > $L/ours-$T.tensors.txt 2>&1; "$PY" "$META" "$REFD/Ternary-Bonsai-2-27B-$T.gguf" --tensors > $L/ref-$T.tensors.txt 2>&1
  diff <(tail -n +2 $L/ref-$T.tensors.txt) <(tail -n +2 $L/ours-$T.tensors.txt) > $L/census-$T.diff && echo "     census IDENTICAL to PrismML $T" || echo "     CENSUS DIFFERS: $(grep -c '^>' $L/census-$T.diff) rows"
  ppl -m "$Q" --kl-divergence-base "$BASE" --kl-divergence > $L/kld-$T.log 2>&1 || tail -5 $L/kld-$T.log
  show $L/kld-$T.log
done
rm -f "$F"; echo "[$(ts)] intermediate bf16 removed"; df -h /mnt/sdc | tail -1
echo "[$(ts)] === PT2_EXPORT_DONE ==="
