#!/usr/bin/env bash
# pod_v10_lcb77.sh — ON THE POD: Q6_K builds for LCB-77 on the untested v9-agentic maps (owner 2026-10-18: "run
# LCBv77 on the pod GPUS for the latest versions that we didn't test"; Cerebriline bugs explained the instability).
# Basis = the release LCB-77 cell (k8 Q6_K 93.51): Q6_K, greedy_pen, q8 KV, eval tree b9700+PEG (pod_v10_suite_q.sh).
# Each Q6_K: quantize_gguf.py on llama.cpp b11707 with the map's PRE-STAGED quick-v5 imatrix (sha gated before/after),
# sanity PASS + quantize.imatrix.file KV gate. Builds use GPU1 only, so GPU0 can evaluate a_ref meanwhile.
# Disk (owner OK 10:30 CEST): after a_ref's Q6_K, its bf16 shards (map A, rebuildable byte-exact) and its Q4_K_M
# (sha-verified copy on solidpc) are removed. a_ml3/a_core5: base -> bf16 -> rmbase -> Q6_K -> shards removed,
# one map at a time (peak: Q6_K 98G with the bf16 on disk). Then UPLOAD_GO: solidpc uploads the hub_k12/hub_k9 Q6_K
# (the exact suite files), sha-checks them here and writes READY_hub_*. Each READY_<map> starts that map's LCB-77.
set -uo pipefail
W=/workspace; V=$W/v10; S=$W/omnimergekit/scripts; PY=$W/miniconda/envs/omnimergekit/bin/python
LC=$W/llama.cpp-latest; LCREV=1268eda; CAL=$W/omnimergekit/scripts/calibration_datav5.txt
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
M=$V/lcb77; L=$M/logs; mkdir -p $L
declare -A ISHA=([a_ref]=6e8ba9658e1d2304 [a_ml3]=db8c78dcc76a17d0 [a_core5]=8bcb74565d3ac63b)
say(){ echo "[lcb77 $(date -u +%FT%TZ)] $*"; }
die(){ say "LCB77_FAIL $*"; touch $M/FAIL; exit 1; }
free(){ df --output=avail -BG $W | tail -n1 | tr -dc 0-9; }
need(){ local f; f=$(free); [ "$f" -ge "$1" ] || die "disk ${f}G < $1G before $2"; say "disk ${f}G free (need $1G) before $2"; }
qb(){ bash $S/pod_v10_quick_build.sh "$@" || die "quick_build $*"; }
q6k(){
  local n=$1 NAME=gemma-4-20B-v9-agentic-$1; local BF=$V/$n/$NAME GG=$V/gguf/$n; local Q6=$GG/$NAME-Q6_K.gguf
  [ -f $BF/BUILD_OK ] || die "$n: no bf16"
  [ "$(sha256sum $GG/imatrix.dat | cut -c1-16)" = "${ISHA[$n]}" ] || die "$n: imatrix sha before quantize"
  need 100 "Q6_K $n"
  CUDA_VISIBLE_DEVICES=1 LLAMA_CPP_HOME=$LC "$PY" $W/omnimergekit/scripts/quantize_gguf.py --model $BF --output-dir $GG \
    --only Q6_K --cal-data $CAL --no-kld --ngl 99 \
    --chat-template-file $BF/chat_template.jinja --chat-template-sha256 $TSHA \
    --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local > $L/$n.quant_q6k.log 2>&1 \
    || die "$n quantize (see $L/$n.quant_q6k.log)"
  [ -s "$Q6" ] || die "no $Q6"
  grep -q "Using pre-staged imatrix.dat" $L/$n.quant_q6k.log || die "$n: the pre-staged imatrix was not used"
  [ "$(sha256sum $GG/imatrix.dat | cut -c1-16)" = "${ISHA[$n]}" ] || die "$n: imatrix.dat changed"
  grep -qE "^\s+Q6_K: sanity PASS" $L/$n.quant_q6k.log || die "$n: Q6_K sanity not PASS"
  local kv; kv=$(PYTHONPATH=$LC/gguf-py "$PY" -c "
from gguf import GGUFReader
f=GGUFReader('$Q6').fields
g=lambda k: f[k].contents() if k in f else None
print(g('general.file_type'), g('quantize.imatrix.file'), g('quantize.imatrix.chunks_count'))")
  grep -q "imatrix.dat" <<<"$kv" || die "$n: Q6_K carries no quantize.imatrix.file ($kv)"
  rm -f $GG/$NAME-F16.gguf
  [ -s "$Q6.sha256" ] || sha256sum "$Q6" > "$Q6.sha256"
  say "$n: Q6_K OK $(du -h $Q6 | cut -f1) sha $(cut -c1-16 $Q6.sha256) KV $kv"
  # tokenizer/config stay (pod_v10_suite_q.sh --tokenizer); shards go, BUILD_OK -> SHARDS_REMOVED
  rm -f $BF/model-*.safetensors; mv $BF/BUILD_OK $BF/SHARDS_REMOVED
  touch $M/READY_$n; say "READY_$n (bf16 shards removed, tokenizer/config kept)"
}
[ "$(git -C $LC rev-parse --short=7 HEAD)" = $LCREV ] || die "llama.cpp-latest is not $LCREV"
bash $S/pod_v10_suite.sh gate > $L/gate.log 2>&1 && grep -q GATE_OK $L/gate.log || die "greedy template gate (see $L/gate.log)"
say "greedy template gate OK"
q6k a_ref
q4=$V/gguf/a_ref/gemma-4-20B-v9-agentic-a_ref-Q4_K_M.gguf
[ "$(sha256sum $q4 | cut -c1-16)" = 723f9539e62120dd ] || die "a_ref Q4_K_M is not the archived 723f9539 — not removing"
rm -f $q4; say "a_ref Q4_K_M removed (archived on solidpc, 723f9539)"
for n in a_ml3 a_core5; do
  need 95 "base for $n"; qb base
  qb bf16 $n
  qb rmbase
  q6k $n
done
need 40 "hub Q6_K upload"; touch $M/UPLOAD_GO; say "UPLOAD_GO (solidpc uploads hub_k12/hub_k9 Q6_K)"
say "BUILDS_DONE"
