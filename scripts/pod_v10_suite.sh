#!/usr/bin/env bash
# pod_v10_suite.sh — ON THE POD: the v9-agentic card suite (the 12 benches of pod_v9ml_k8_suite.sh) on the
# v10-recipe maps hub_k12 / hub_k9. Owner 2026-10-17: "run the 10-bench evals on both while I test".
# The published numbers are Q6_K, so each map gets a Q6_K: quantize_gguf.py on llama.cpp-latest (b11707, owner:
# "make sure the quantize is from the latest llama.cpp") with the map's PRE-STAGED v5 imatrix.dat (not
# recomputed; sha checked before/after) -> sanity PASS + quantize.imatrix.file KV gate -> F16 removed.
# Then the greedy-template gate, and ONE MAP PER GPU: hub_k12 on GPU0, hub_k9 on GPU1, same bench order.
# Stages (2026-10-17 split after the hub_k9 Q6_K hit ENOSPC): build <map> | gate | queue <map> <gpu> <port>.
# Disk: quantize_gguf.py writes the F16 (~40 GB) AND an EOG-normalised copy of it (~40 GB) before the Q6_K
# (~18 GB) — peak ~98 GB; an F16 already on disk is reused, so the floor drops by 40 GB.
set -uo pipefail
W=/workspace; S=$W/v10/suite; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit
LC=$W/llama.cpp-latest; LCREV=1268eda; CAL=$OMK/scripts/calibration_datav5.txt
TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
declare -A ISHA=([hub_k12]=b053238f6ae4cbf8 [hub_k9]=b95d61e527bfab6e)
B=(ifeval_100 humaneval_full humanevalplus_full multipl_e_100 gsm8k_100_qwen math500_100 aime_30
   arc_challenge_full lcb_medium_55_v4 lcb_medium_100_v4 lcb_v6_77q gpqa_diamond_full)
say(){ echo "[suite $(date -u +%FT%TZ)] $*"; }
die(){ say "SUITE_FAIL $*"; exit 1; }
mkdir -p $S/logs
case ${1:?build <map> | gate | queue <map> <gpu> <port>} in
build)
  n=${2:?map}; [ -n "${ISHA[$n]:-}" ] || die "unknown map $n"
  [ "$(git -C $LC rev-parse --short=7 HEAD)" = $LCREV ] || die "llama.cpp-latest is not $LCREV"
  NAME=gemma-4-20B-v9-agentic-$n; BF=$W/v10/$n/$NAME; GG=$W/v10/gguf/$n; Q6=$GG/$NAME-Q6_K.gguf
  [ -f $BF/BUILD_OK ] || die "$n: no bf16"
  [ "$(sha256sum $GG/imatrix.dat | cut -c1-16)" = "${ISHA[$n]}" ] || die "$n: imatrix sha before quantize"
  if [ ! -s "$Q6" ]; then
    need=100; [ -s $GG/$NAME-F16.gguf ] && need=60
    f=$(df --output=avail -BG $W | tail -n1 | tr -dc 0-9); [ "$f" -ge $need ] || die "disk ${f}G < ${need}G before $n Q6_K"
    say "$n: Q6_K with the pre-staged imatrix (${ISHA[$n]}) on llama.cpp $LCREV (disk ${f}G, need ${need}G)"
    CUDA_VISIBLE_DEVICES=0,1 LLAMA_CPP_HOME=$LC "$PY" $OMK/scripts/quantize_gguf.py --model $BF --output-dir $GG \
      --only Q6_K --cal-data $CAL --no-kld --ngl 99 \
      --chat-template-file $BF/chat_template.jinja --chat-template-sha256 $TSHA \
      --base-model-id google/gemma-4-26B-A4B-it --no-upload --keep-local > $S/logs/$n.quant_q6k.log 2>&1 \
      || die "$n quantize (see $S/logs/$n.quant_q6k.log)"
  fi
  [ -s "$Q6" ] || die "no $Q6"
  grep -q "Using pre-staged imatrix.dat" $S/logs/$n.quant_q6k.log || die "$n: the pre-staged imatrix was not used"
  [ "$(sha256sum $GG/imatrix.dat | cut -c1-16)" = "${ISHA[$n]}" ] || die "$n: imatrix.dat changed"
  grep -qE "^\s+Q6_K: sanity PASS" $S/logs/$n.quant_q6k.log || die "$n: Q6_K sanity not PASS"
  kv=$(PYTHONPATH=$LC/gguf-py "$PY" -c "
from gguf import GGUFReader
f=GGUFReader('$Q6').fields
g=lambda k: f[k].contents() if k in f else None
print(g('general.file_type'), g('quantize.imatrix.file'), g('quantize.imatrix.chunks_count'))")
  grep -q "imatrix.dat" <<<"$kv" || die "$n: Q6_K carries no quantize.imatrix.file ($kv)"
  rm -f $GG/$NAME-F16.gguf
  [ -s "$Q6.sha256" ] || sha256sum "$Q6" > "$Q6.sha256"
  say "$n: Q6_K OK $(du -h $Q6 | cut -f1) sha $(cut -c1-16 $Q6.sha256) KV $kv";;
gate)
  "$PY" - $OMK/eval/templates "${B[@]}" <<'PY' || die "a template is not frozen greedy"
import sys, yaml
bad = []
for t in sys.argv[2:]:
    g = (yaml.safe_load(open(f"{sys.argv[1]}/{t}.yaml")) or {}).get("generation") or {}
    if g.get("do_sample") or float(g.get("temperature", 0)) != 0.0:
        bad.append(t)
print("greedy gate:", "PASS" if not bad else f"FAIL {bad}", f"({len(sys.argv) - 2} templates)")
sys.exit(1 if bad else 0)
PY
  say "GATE_OK greedy templates";;
queue)
  n=${2:?map}; g=${3:?gpu}; p=${4:?port}
  bash $OMK/scripts/pod_v10_suite_q.sh "$n" "$g" "$p" "${B[@]}" > $S/logs/q_$n.log 2>&1
  rc=$?; say "queue $n exited rc=$rc | $(grep -h QUEUE_DONE $S/logs/q_$n.log)"
  [ $rc = 0 ] && grep -q "QUEUE_DONE fails=0" $S/logs/q_$n.log && say "SUITE_DONE $n" || say "SUITE_FAIL $n queue rc=$rc";;
*) die "unknown stage $1";;
esac
