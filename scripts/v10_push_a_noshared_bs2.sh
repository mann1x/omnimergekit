#!/usr/bin/env bash
# v10_push_a_noshared_bs2.sh — after v10_quant_solidpc.sh: a_noshared Q4_K_M -> bs2 xollama (shared ollama store + key)
# -> 4 private tags in mannix/gemma-4-20B-v9-agentic via the canonical pod_v10_ollama_test_tags.py (a263e65):
# a_noshared-Q4_K_M, vision-, mtp- (drafter 5963a2ed, n=2), mtp-vision- (mmproj 1985002f). Each tag: create ->
# local check -> 3-question generation check -> push -> registry read-back digest == created -> local tag removed.
set -uo pipefail
V=/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/google/v9_agentic_ab/v10_recipe
Q=$V/gguf/a_noshared/gemma-4-20B-v9-agentic-a_noshared-Q4_K_M.gguf; S=/mnt/sdc/ollama/xollama-home/stage
say(){ echo "[push $(date -u +%FT%TZ)] $*"; }; die(){ say "FATAL $*"; exit 1; }
until grep -q -e "a_noshared: QUANT_OK" -e FATAL $V/run_v10_quant_128.log; do sleep 30; done
grep -q "a_noshared: QUANT_OK" $V/run_v10_quant_128.log || die "quant failed: $(grep FATAL $V/run_v10_quant_128.log)"
PYTHONPATH=/shared/dev/llama.cpp-1268eda/gguf-py /root/anaconda3/envs/omnimergekit/bin/python -c "
from gguf import GGUFReader; f=GGUFReader('$Q').fields
c=f['quantize.imatrix.chunks_count'].contents(); d=f['quantize.imatrix.dataset'].contents()
print('KV chunks', c, 'dataset', d); assert c==128 and d.endswith('calibration_datav5.txt')" || die "imatrix KV is not the quick v5 basis"
LS=$(sha256sum $Q | cut -d' ' -f1); say "Q4_K_M sha $LS -> bs2"
rsync -a $Q bs2:$S/ || die "rsync"
RS=$(ssh bs2 "sha256sum $S/$(basename $Q)" | cut -d' ' -f1); [ "$LS" = "$RS" ] || die "sha mismatch on bs2"
ssh bs2 "cd $S && OLLAMA_MODELS=/mnt/sdc/ollama/models python3 -I $S/pod_v10_ollama_test_tags.py --gguf $S/$(basename $Q) --prefix a_noshared \
  --params $S/ollama_params.json --mmproj $S/mmproj/mmproj-gemma-4-20B-v9-agentic-F16.gguf \
  --drafter $S/drafter/gemma-4-26B-A4B-it-assistant.Q8_0.gguf --ref-modelfile $S/Modelfile.Q4_K_M \
  --scratch $S/scratch --report $S/ollama_test_tags.tsv" > $V/push_a_noshared.log 2>&1 || die "test tags (see push_a_noshared.log)"
grep -q "TEST_TAGS_DONE a_noshared-Q4_K_M" $V/push_a_noshared.log || die "no TEST_TAGS_DONE"
say "PUSHED $(grep -c ' PUSHED ' $V/push_a_noshared.log) tags; PUSH_DONE"
