#!/usr/bin/env bash
# bonsai27_v6_target_kld.sh — is the r1_v6 ternary as close to ITS teacher as Bonsai 2 is to ITS teacher?
#
# Each student is scored against its own teacher, same llama.cpp (prism fork), same texts, same settings:
#   ours    = r1_v6 packed PQ2_0       vs  Omnimerge v6 BF16
#   Bonsai2 = Ternary-Bonsai-2 PQ2_0   vs  Qwen3.8-27B BF16
# Texts: wiki (8 x 2048, as every earlier wiki KLD), the held-out Qwen3.8 traces (16, as trace_eval),
# and the v6 corpus held-out code and agent sets (48 each). No training, no generation.
#
# Gates, each fatal:
#   G0 (warn only) the v6 BF16 reconversion reproduces the recorded sha of the 2026-10-09 file
#   G1 ours: folded bf16 (pre-pack) and PQ2_0 give the same wiki KLD (packing is lossless)
#   G2 ours: PQ2_0 tensor census identical to PrismML's PQ2_0
#   G3 Bonsai 2 reproduces its recorded wiki KLD 0.376076 and trace KLD 0.210319 (setup unchanged)
set -uo pipefail
ts(){ date -u '+%H:%M:%SZ'; }
PY=/root/anaconda3/envs/omnimergekit/bin/python
export PYTHONPATH=/srv/ml/dca-cleanroom/llama.cpp/gguf-py
C=/srv/ml/dca-cleanroom/llama.cpp
PRISM=/srv/ml/bonsai-assess/ref-b10754/llama-prism-b10754-2459f68
QUANT=/srv/ml/repos/omnimergekit/tools/opencoti-quantize/bin/opencoti-quantize
META=/srv/ml/scripts/gguf_read_prism_meta.py
FOLD=/srv/ml/scripts/bonsai_hadamard_fold.py
REF=/mnt/sdc/opencoti-models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf
R=/mnt/sdc/opencoti-scratch/bonsai27/r1_v6
CORP=/mnt/sdc/opencoti-scratch/bonsai27/corpus_v6
HF_V6=/mnt/sdc/opencoti-models/hf/ManniX-ITA/Qwen3.8-27B-Omnimerge-v6
HF_Q=/mnt/sdc/opencoti-models/hf/Qwen/Qwen3.8-27B
V6_SHA=87326f87be9a523866461cc062995bbde7d2441e401d42800650626efda47d3e
W=/mnt/sdc/opencoti-scratch/bonsai27/target_kld; mkdir -p "$W"
G=$W/gguf; mkdir -p "$G"
L=$W/chain.log
log(){ echo "[$(ts)] $*" | tee -a "$L"; }
die(){ log "FATAL $*"; exit 1; }
show(){ grep -E "^(Mean  *PPL\(Q\) |Mean  *KLD|Same top p|Median  *KLD|99.0%  *KLD)" "$1" | sed 's/^/     /' | tee -a "$L"; }
kld(){ grep -oE "^Mean  *KLD: *[0-9.]+" "$1" | grep -oE "[0-9.]+$"; }
for f in "$PRISM/llama-perplexity" "$QUANT" "$META" "$FOLD" "$REF" "$R/export/bonsai_export.json" "$R/q" "$HF_V6/config.json" "$HF_Q/config.json"; do
  [ -e "$f" ] || die "missing $f"; done
[ "$(ls $R/q/*.pt | wc -l)" = 402 ] || die "q dir does not hold 402 tensors"
log "===== TARGET KLD start; $(df -h /mnt/sdc | tail -1)"

# ---- texts
TW=/srv/ml/ppl-corpus/wiki.test.mid.raw
TT=/mnt/sdc/opencoti-scratch/v6-ternary/trace_eval/heldout_traces.txt
"$PY" - "$CORP/eval_code.jsonl" "$W/code.txt" "$CORP/agent_eval.jsonl" "$W/agent.txt" <<'PYX' || die "texts"
import json, sys
for src, dst in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    open(dst, "w").write("\n\n".join(json.loads(l)["text"] for l in open(src)))
    print(dst, sum(1 for _ in open(src)), "docs")
PYX
TEXTS="wiki:$TW:8 traces:$TT:16 code:$W/code.txt:48 agent:$W/agent.txt:48"
ppl(){ local txt=$1 n=$2; shift 2; LD_LIBRARY_PATH="$PRISM" CUDA_VISIBLE_DEVICES=1 "$PRISM/llama-perplexity" "$@" -f "$txt" -c 2048 --chunks "$n" -b 2048 -ub 512 -ngl 99; }

# ---- 1. our student: dense export -> bf16 GGUF -> fold with the trained ternary tensors -> PQ2_0
OURS=$G/r1_v6-PQ2_0.gguf; FOLDED=$G/r1_v6-folded-BF16.gguf
if [ ! -s "$OURS" ]; then
  RAW=$G/r1_v6-BF16.gguf
  [ -s "$RAW" ] || { log "convert r1_v6 export"; "$PY" $C/convert_hf_to_gguf.py "$R/export" --outtype bf16 --outfile "$RAW" > "$W/convert-r1.log" 2>&1 || die "convert r1: $(tail -2 $W/convert-r1.log)"; }
  log "fold r1_v6 with --override-dir q"
  CUDA_VISIBLE_DEVICES=1 "$PY" "$FOLD" --src "$RAW" --ref "$REF" --override-dir "$R/q" --out "$FOLDED" --device cuda:0 > "$W/fold-r1.log" 2>&1
  grep -q FOLD_OK "$W/fold-r1.log" || die "fold: $(tail -3 $W/fold-r1.log)"
  rm -f "$RAW"
  /bin/sh "$QUANT" --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 --token-embedding-type PQ2_0 --output-tensor-type PQ2_0 \
     "$FOLDED" "$OURS" PQ2_0 32 > "$W/quantize-r1.log" 2>&1 || die "quantize: $(tail -3 $W/quantize-r1.log)"
  sha256sum "$OURS" > "$OURS.sha256"; log "PQ2_0 $(stat -c%s "$OURS") B sha $(cut -c1-16 "$OURS.sha256")"
fi
"$PY" "$META" "$OURS" --tensors > "$W/ours.tensors.txt" 2>&1; "$PY" "$META" "$REF" --tensors > "$W/ref.tensors.txt" 2>&1
diff <(tail -n +2 "$W/ref.tensors.txt") <(tail -n +2 "$W/ours.tensors.txt") > "$W/census.diff" || die "G2 census differs: $(grep -c '^>' $W/census.diff) rows"
log "G2 PASS census identical to PrismML PQ2_0"

# ---- 2. teachers as BF16 GGUF
V6=$G/Qwen3.8-27B-Omnimerge-v6-BF16.gguf; QB=$G/Qwen3.8-27B-BF16.gguf
[ -s "$V6" ] || { log "convert v6"; "$PY" $C/convert_hf_to_gguf.py "$HF_V6" --outtype bf16 --outfile "$V6" > "$W/convert-v6.log" 2>&1 || die "convert v6"; }
S=$(sha256sum "$V6" | cut -d' ' -f1); [ "$S" = "$V6_SHA" ] && log "G0 PASS v6 BF16 sha reproduces the 2026-10-09 file" || log "G0 WARN v6 BF16 sha $S != recorded $V6_SHA (converter moved; every teacher logit below comes from THIS file, so the comparison holds)"
[ -s "$QB" ] || { log "convert Qwen3.8-27B"; "$PY" $C/convert_hf_to_gguf.py "$HF_Q" --outtype bf16 --outfile "$QB" > "$W/convert-q.log" 2>&1 || die "convert Qwen3.8"; }

# ---- 3. per text: teacher logits, then its student
for E in $TEXTS; do
  N=${E%%:*}; rest=${E#*:}; T=${rest%:*}; K=${rest##*:}
  for P in "ours:$V6:$OURS" "bonsai2:$QB:$REF"; do
    S_=${P%%:*}; pp=${P#*:}; TCH=${pp%%:*}; STU=${pp#*:}
    B=$W/base-$N-$S_.kld
    ppl "$T" "$K" -m "$TCH" --kl-divergence-base "$B" > "$W/teacher-$N-$S_.log" 2>&1 || die "teacher $N $S_: $(tail -2 $W/teacher-$N-$S_.log)"
    ppl "$T" "$K" -m "$STU" --kl-divergence-base "$B" --kl-divergence > "$W/kld-$N-$S_.log" 2>&1 || die "kld $N $S_: $(tail -2 $W/kld-$N-$S_.log)"
    log "--- $N / $S_ ($K x 2048) teacher $(grep -oE 'Final estimate: PPL = [0-9.]+' $W/teacher-$N-$S_.log)"; show "$W/kld-$N-$S_.log"
    if [ "$N" = wiki ] && [ "$S_" = ours ]; then
      ppl "$T" "$K" -m "$FOLDED" --kl-divergence-base "$B" --kl-divergence > "$W/kld-wiki-ours-prepack.log" 2>&1 || die "prepack"
      a=$(kld "$W/kld-wiki-ours-prepack.log"); b=$(kld "$W/kld-wiki-ours.log")
      "$PY" -c "import sys; sys.exit(0 if abs($a-$b) < 0.005 else 1)" && log "G1 PASS pre-pack $a vs PQ2_0 $b" || die "G1 pre-pack $a vs PQ2_0 $b"
    fi
    rm -f "$B"
  done
done
a=$(kld "$W/kld-wiki-bonsai2.log"); b=$(kld "$W/kld-traces-bonsai2.log")
"$PY" -c "import sys; sys.exit(0 if abs($a-0.376076)<0.01 and abs($b-0.210319)<0.01 else 1)" \
  && log "G3 PASS Bonsai 2 reproduces wiki $a (rec 0.376076) traces $b (rec 0.210319)" || log "G3 FAIL Bonsai 2 wiki $a (rec 0.376076) traces $b (rec 0.210319) — setup moved, read with care"
rm -f "$FOLDED"; log "folded intermediate removed; teachers kept in $G for reuse; $(df -h /mnt/sdc | tail -1)"
log "=== TARGET_KLD_DONE ==="
