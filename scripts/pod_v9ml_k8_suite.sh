#!/usr/bin/env bash
# v9-ml ml_k8: after its AC imatrix lands -> quantize Q6_K with it (local, no upload) -> verify the
# GGUF carries the imatrix and passed sanity -> greedy-template gate -> v9-agentic's card suite on
# both GPUs with the model's own sampler (pod_v9ml_k8_q.sh, --sampler-profile auto = greedy_pen).
# Owner 2026-10-15: "when done quantize ml_k8 Q6_K and run the full suite".
set -uo pipefail
W=/workspace; S=$W/v9ml_k8_suite; PY=$W/miniconda/envs/omnimergekit/bin/python; OMK=$W/omnimergekit
NAME=gemma-4-20B-v9ml-ml_k8; BF=$W/v9ml_k8/$NAME; GG=$W/gguf/$NAME; Q6=$GG/$NAME-Q6_K.gguf
TMPL=$W/stage/tmpl/chat_template.jinja; TSHA=ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4
Q0=(gpqa_diamond_full aime_30 math500_100 gsm8k_100_qwen ifeval_100 arc_challenge_full)
Q1=(humaneval_full humanevalplus_full multipl_e_100 lcb_medium_55_v4 lcb_medium_100_v4 lcb_v6_77q)
say(){ echo "[suite $(date -u +%FT%TZ)] $*"; }
die(){ say "SUITE_FAIL $*"; exit 1; }
mkdir -p $S/logs
say "waiting for K8IMAT_OK"
until grep -q K8IMAT_OK $W/v9ml_k8_imatrix.log; do
  grep -q K8IMAT_FAIL $W/v9ml_k8_imatrix.log && die "imatrix job failed"
  tmux has-session -t v9imk8 2>/dev/null || grep -q K8IMAT_OK $W/v9ml_k8_imatrix.log || die "imatrix session gone without K8IMAT_OK"
  sleep 60
done
[ -s $GG/imatrix.dat ] || die "no imatrix.dat"
say "imatrix sha $(sha256sum $GG/imatrix.dat | cut -c1-16); quantize Q6_K"
if [ ! -s "$Q6" ]; then
  CUDA_VISIBLE_DEVICES=0,1 LLAMA_CPP_HOME=$W/llama.cpp "$PY" $OMK/scripts/quantize_gguf.py \
    --model "$BF" --output-dir "$GG" --only Q6_K --no-upload --keep-local \
    --chat-template-file "$TMPL" --chat-template-sha256 "$TSHA" \
    --base-model-id google/gemma-4-26B-A4B-it --ngl 99 > $S/logs/quant_q6k.out 2>&1 || die "quantize (see logs/quant_q6k.out)"
fi
[ -s "$Q6" ] || die "no $Q6"
grep -qE "^\s+Q6_K: sanity PASS" $S/logs/quant_q6k.out || die "Q6_K sanity not PASS (see logs/quant_q6k.out)"
"$PY" - "$Q6" <<'PY' || die "Q6_K GGUF does not record the imatrix"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def s(k):
    v = f.get(k)
    return None if v is None else bytes(v.parts[v.data[0]]).decode("utf-8", "replace")
im, ft = s("quantize.imatrix.file"), f.get("general.file_type")
print("quantize.imatrix.file =", im, "| general.file_type =", None if ft is None else int(ft.parts[ft.data[0]][0]))
sys.exit(0 if im and im.endswith("imatrix.dat") else 1)
PY
sha256sum "$Q6" > "$Q6.sha256"; say "Q6_K OK $(du -h $Q6 | cut -f1) sha $(cut -c1-16 $Q6.sha256)"
"$PY" - $OMK/eval/templates "${Q0[@]}" "${Q1[@]}" <<'PY' || die "a template is not frozen greedy"
import sys, yaml
bad = []
for t in sys.argv[2:]:
    g = (yaml.safe_load(open(f"{sys.argv[1]}/{t}.yaml")) or {}).get("generation") or {}
    if g.get("do_sample") or float(g.get("temperature", 0)) != 0.0:
        bad.append(t)
print("greedy gate:", "PASS" if not bad else f"FAIL {bad}", f"({len(sys.argv) - 2} templates)")
sys.exit(1 if bad else 0)
PY
say "launch queues GPU0: ${Q0[*]} | GPU1: ${Q1[*]}"
bash $W/v9ml_k8_q.sh 0 8800 "${Q0[@]}" > $S/logs/q_gpu0.log 2>&1 & p0=$!
sleep 20
bash $W/v9ml_k8_q.sh 1 8820 "${Q1[@]}" > $S/logs/q_gpu1.log 2>&1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "queues exited rc0=$r0 rc1=$r1"
[ $r0 = 0 ] && [ $r1 = 0 ] && say "SUITE_DONE" || say "SUITE_FAIL queue rc0=$r0 rc1=$r1"
