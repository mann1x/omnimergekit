#!/usr/bin/env bash
# eval_v9a_q6k_ab_bs2.sh — paired A/B of the two v9-agentic (sft12) Q6_K twins on bs2 GPU1 (owner 2026-11-17 13:42 CET:
# "I'm not convinced about kfac quality gains ... run evals on the q6k ac I-matrix, I want to compare against the
# published kfac version. Ifeval, aime, lcbv6 77q, mpe, he and heplus").
#   arm acimat = ptq.gguf  (AC-imatrix PTQ, llama-quantize)        arm kfac = kfac.gguf (the published K-FAC re-rounding)
# Both arms run HERE, at the same time, one llama-server queue each on the same 96 GB GPU, same PEG-fixed llama build
# (b9393), same omnimergekit eval tree (88092f6: eval/ identical to solidpc HEAD 5535483), same templates, same sampler
# (greedy_pres: greedy + presence 0.025, repeat 1.0) — so the only difference between the two columns is the quant.
# The published pod cells (2x 3090, pegfix b12386) are a DIFFERENT basis and are shown next to the pair for reference only.
# Gates as eval_v9a_q6k_full_pod.sh: recorded sha, imatrix KV (>= 9500 chunks, eos 106), frozen-greedy templates,
# PEG-fixed library, LIVE llama-server flags per cell, summary sampler name. Resumable (scored rc-0 summary skipped,
# incomplete one moved aside). MPE scores in the nuprl docker image (MPE_MODE=docker).
# usage: eval_v9a_q6k_ab_bs2.sh            sentinels V9AB_EVAL_DONE / V9AB_EVAL_FAIL
#   repeat draw (owner 2026-11-17 22:50 CET "Settle the lcb gap"): BENCHES="lcb_hard_77" RES=$O/eval_q6k_ab_rep2/results
#   LD=$O/eval_q6k_ab_rep2/logs — a NEW results dir, so the sqlite cache (<results>/<bench>/<served>/sqlite_cache) is fresh
#   and the pair is a genuine second draw of the same files on the same basis; the first run is never touched.
#   Finding 2026-11-18: that repeat reproduced all 154 generations byte for byte (greedy at a fixed --parallel and request
#   order is deterministic on bs2/b9393), so run-to-run noise on this basis is nil and repeats are not independent draws.
#   PAR=4 / PAR=1 change the batch composition instead (eval_q6k_ab_par4/, eval_q6k_ab_par1/).
set -uo pipefail
O=${O:-/mnt/sdc/ml/v9a_sft11/mlk8-noa-sft12}
OMK=${OMK:-/mnt/sdc/ml/v9a_sft11/omk}
G=${G:-$O/gguf/Q6_K}
TOK=${TOK:-$O/gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12}
RES=${RES:-$O/eval_q6k_ab/results}; LD=${LD:-$O/eval_q6k_ab/logs}
ENV=${ENV:-/srv/ml/envs/envs/omnimergekit}
PY=$ENV/bin/python3.11
PROFILE=gemma-4-v9-agentic-sft11
WANT=greedy_pres
GPU=${GPU:-1}
read -r -a BENCHES <<< "${BENCHES:-ifeval_100 humaneval_full humanevalplus_full multipl_e_100 aime_30 lcb_hard_77}"
export LLAMA_BIN=${LLAMA_BIN:-/opt/llama.cpp/build-pegfix/bin}
export PATH="$ENV/bin:$PATH" LM_EVAL_BIN="$ENV/bin/lm_eval" OMK_PYTHON="$PY"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900 MPE_MODE=docker CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMK_GPU_UTIL_THRESH=1.01   # the other arm's server shares this GPU: lift only the planner's util test
say(){ echo "[v9ab $(date -u +%FT%TZ)] $*"; }
die(){ say "V9AB_EVAL_FAIL $*"; exit 1; }
trap 'say "V9AB_EVAL_FAIL terminated"; trap - TERM INT; kill 0' TERM INT
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
sn(){ echo "v9a-sft12-q6k-$1-bs2"; }
queue() { # arm gguf first_port bench...
  local arm=$1 gguf=$2 P=$3; shift 3
  local SN b sj par cp spid cl live rc r
  SN=$(sn $arm)
  for b in "$@"; do
    sj=$RES/$b/$SN/summary.json
    if [ -f "$sj" ]; then
      if "$PY" -c "import json,sys;d=json.load(open('$sj'));sys.exit(0 if d.get('rc')==0 and d.get('score') is not None else 1)"; then
        say "[$arm] $b: scored summary exists, skip"; continue
      fi
      mv "$RES/$b/$SN" "$RES/$b/$SN.incomplete.$(date -u +%H%M%S)"; say "[$arm] $b: incomplete summary moved aside, rerun"
    fi
    par=${PAR:-2}   # PAR=4 / PAR=1 = the batch-composition perturbation draws (greedy is bit-reproducible at a fixed slot count)
    say "[$arm] $b: start (GPU$GPU port $P --parallel $par --sampler-profile $PROFILE)"
    ( cd "$OMK" && CUDA_VISIBLE_DEVICES=$GPU "$PY" eval/omk_eval.py --model "$gguf" --tokenizer "$TOK" \
        --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
        --sampler-profile "$PROFILE" > "$LD/$SN.$b.log" 2>&1 ) & cp=$!
    live=none; spid=""; cl=""
    for _ in $(seq 1 180); do
      spid=$(ss -ltnp 2>/dev/null | grep ":$P " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
      if [ -n "$spid" ]; then
        cl=$(tr "\0" " " < /proc/$spid/cmdline)
        if grep -q -- "--presence-penalty 0.025 " <<<"$cl" \
           && ! grep -qE -- "--repeat-penalty (0*[02-9]|1\.0*[1-9])" <<<"$cl"; then live=ok; else live=bad; fi
        break
      fi
      kill -0 $cp 2>/dev/null || break
      sleep 10
    done
    case $live in
      ok)   say "[$arm] $b: LIVE sampler OK (llama-server pid $spid: presence 0.025, no repeat penalty)";;
      bad)  say "[$arm] Q_FAIL $b: LIVE llama-server pid $spid has the wrong sampler flags: $cl"; killtree $cp; return 1;;
      none) if kill -0 $cp 2>/dev/null; then say "[$arm] Q_FAIL $b: no llama-server on port $P after 30 min"; killtree $cp; return 1; fi;;
    esac
    wait $cp; rc=$?
    if [ -f "$sj" ]; then
      r=$("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")
      say "[$arm] $b: rc=$rc RESULT $r"
      [ "${r##* }" = "$WANT" ] || { say "[$arm] Q_FAIL $b: summary sampler is not $WANT"; return 1; }
      [ $live = ok ] || say "[$arm] WARN $b: finished before the live gate saw its server"
    else
      say "[$arm] $b: rc=$rc NO summary.json (see $LD/$SN.$b.log)"
    fi
    P=$((P+1)); sleep 30
  done
  say "[$arm] QUEUE_DONE"
}
table() {
  "$PY" - "$RES" "$(sn acimat)" "$(sn kfac)" "${BENCHES[@]}" <<'PY'
import json, os, sys
res, a, k, *benches = sys.argv[1:]
pod = {"ifeval_100": 0.9400, "humaneval_full": 0.9817, "humanevalplus_full": 0.9085, "multipl_e_100": 0.8933,
       "aime_30": 0.7667, "lcb_hard_77": None}   # published K-FAC cells, pod 2x3090 pegfix b12386 (reference basis only)
def get(b, s):
    p = f"{res}/{b}/{s}/summary.json"
    if not os.path.exists(p): return None, ""
    d = json.load(open(p)); return (d.get("score") if d.get("rc") == 0 else None), d.get("metric", "")
print(f"{'bench':22s} {'acimat(bs2)':>12s} {'kfac(bs2)':>10s} {'delta':>8s} {'kfac(pod,pub)':>14s}  metric")
for b in benches:
    va, m = get(b, a); vk, _ = get(b, k)
    f = lambda v: "-" if v is None else f"{v:.4f}"
    d = "-" if va is None or vk is None else f"{vk - va:+.4f}"
    print(f"{b:22s} {f(va):>12s} {f(vk):>10s} {d:>8s} {f(pod.get(b)):>14s}  {m}")
PY
}
mkdir -p "$RES" "$LD"
for f in "$G/ptq.gguf" "$G/ptq.gguf.sha256" "$G/kfac.gguf" "$G/kfac.gguf.sha256" "$TOK/config.json" "$TOK/tokenizer.json" \
         "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml" "$OMK/eval/lcb/lcb_hard_77_taskids.json"; do
  [ -e "$f" ] || die "missing $f"
done
(cd "$G" && sha256sum -c ptq.gguf.sha256 kfac.gguf.sha256 --quiet) || die "sha mismatch in $G"
for f in "$G/ptq.gguf" "$G/kfac.gguf"; do
"$PY" - "$f" <<'PY' || die "GGUF does not record the full AC imatrix / eos 106: $f"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
im, ch, eos = get("quantize.imatrix.file"), get("quantize.imatrix.chunks_count"), get("tokenizer.ggml.eos_token_id")
print(f"{sys.argv[1]}: imatrix.file={im} chunks={ch} eos={eos} file_type={get('general.file_type')}")
sys.exit(0 if im and ch and ch >= 9500 and eos == 106 else 1)
PY
done
"$PY" - "$OMK/eval/templates" "${BENCHES[@]}" <<'PY' || die "a template is not frozen greedy"
import sys, yaml
bad = []
for t in sys.argv[2:]:
    g = (yaml.safe_load(open(f"{sys.argv[1]}/{t}.yaml")) or {}).get("generation") or {}
    if g.get("do_sample") or float(g.get("temperature", 0)) != 0.0:
        bad.append(t)
print("greedy gate:", "PASS" if not bad else f"FAIL {bad}", f"({len(sys.argv) - 2} templates)")
sys.exit(1 if bad else 0)
PY
docker image inspect ghcr.io/nuprl/multipl-e-evaluation:latest > /dev/null 2>&1 || die "MPE docker image missing"
lib=$(ls "$LLAMA_BIN"/libllama-common.so.0.0.* 2>/dev/null | head -1)
[ -x "$LLAMA_BIN/llama-server" ] && [ -n "$lib" ] && grep -q LLAMA_CHAT_PEG_STRICT "$lib" \
  || die "$LLAMA_BIN is not the PEG-fixed build (no LLAMA_CHAT_PEG_STRICT in ${lib:-libllama-common})"
used=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 2000 ] || die "GPU$GPU busy ($used MiB used)"
say "pair on bs2 GPU$GPU | results $RES | omk $(git -C "$OMK" rev-parse --short HEAD 2>/dev/null) | llama $LLAMA_BIN pegfix $(git -C /opt/llama.cpp describe --tags --always 2>/dev/null) | acimat $(cut -c1-16 "$G/ptq.gguf.sha256") | kfac $(cut -c1-16 "$G/kfac.gguf.sha256") | benches ${BENCHES[*]}"
queue acimat "$G/ptq.gguf"  8930 "${BENCHES[@]}" > "$LD/queue_acimat.log" 2>&1 & p0=$!
sleep 90
queue kfac   "$G/kfac.gguf" 8950 "${BENCHES[@]}" > "$LD/queue_kfac.log" 2>&1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "queues exited acimat=$r0 kfac=$r1"
table | tee "$RES/q6k_ab.table.txt"
[ $r0 = 0 ] && [ $r1 = 0 ] && say "V9AB_EVAL_DONE" || say "V9AB_EVAL_FAIL queue rc acimat=$r0 kfac=$r1"
