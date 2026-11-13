#!/usr/bin/env bash
# eval_v9_agentic_sft11_q6k.sh — bm-893 step 8: the v9-agentic Q6_K suite on gemma-4-20B-v9-agentic-sft11 (bs2).
#
# Owner 2026-11-13: "q6_k, q4_k_m and then run the evals again", eval tiers "Q6_K only"; sampler = the 26B agentic's
# (presence 0.025, min_p 0, no repetition penalty). Eval basis = profile eval/models/gemma-4-v9-agentic-sft11.yaml
# `greedy_pres` (greedy + presence 0.025, rep 1.0). NOT comparable with the v9 `greedy_pen` rows (other cohort).
#
# Two llama-server queues on ONE GPU (Q6_K 17.8 GB each; 96 GB card), llama backend, --parallel 2.
# Gates: (1) every template's frozen generation block is greedy; (2) LIVE per cell: the cell's llama-server carries
#   --presence-penalty 0.025 and no repeat penalty other than 1.0, else the cell is killed and its queue stops;
#   (3) post: summary.json sampler.name == greedy_pres. A cell whose summary.json exists is skipped (resumable).
# MPE scores in the nuprl Docker image (MPE_MODE=docker, image present on bs2).
#
# usage: eval_v9_agentic_sft11_q6k.sh [GPU] [OMK_DIR]     sentinels V9A_EVAL_DONE / V9A_EVAL_FAIL
set -uo pipefail
G=${1:-1}
OMK=${2:-/mnt/sdc/ml/v9a_sft11/omk}
O=/mnt/sdc/ml/v9a_sft11
NAME=gemma-4-20B-v9-agentic-sft11
Q6=$O/gguf/$NAME/$NAME-Q6_K.gguf
TOK=$O/$NAME
SN=v9a-sft11-imat-q6k
PROFILE=gemma-4-v9-agentic-sft11
WANT=greedy_pres
RES=$O/eval/results; LD=$O/eval/logs
ENV=/srv/ml/envs/envs/omnimergekit
PY=$ENV/bin/python3.11
Q0=(gpqa_diamond_full aime_30 math500_100 gsm8k_100_qwen ifeval_100 arc_challenge_full)
Q1=(humaneval_full humanevalplus_full multipl_e_100 lcb_medium_55_v4 lcb_medium_100_v4)
export PATH="$ENV/bin:$PATH" LM_EVAL_BIN="$ENV/bin/lm_eval" OMK_PYTHON="$PY"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900 MPE_MODE=docker CUDA_DEVICE_ORDER=PCI_BUS_ID
# Two queues share ONE GPU on purpose: the gpu_planner's busy-GPU test (util < 0.15) would refuse the second server
# (exit 8 after OMK_GPU_WAIT_S). Lift only the util test; its VRAM-free test (~17.7 GB per Q6_K server) still applies.
export OMK_GPU_UTIL_THRESH=1.01
# llama-server with the PEG degrade-not-throw fix (patches/llamacpp_peg_parse_degrade_not_throw.patch): the
# unpatched build answers a full parse failure with HTTP 500 and lm-eval stores an EMPTY completion (a scored fail).
# bs2 /opt/llama.cpp/build-pegfix = the same source rev as /opt/llama.cpp/build (d6be315) + the patch.
export LLAMA_BIN=${LLAMA_BIN:-/opt/llama.cpp/build-pegfix/bin}
say(){ echo "[v9a-eval $(date -u +%FT%TZ)] $*"; }
die(){ say "V9A_EVAL_FAIL $*"; exit 1; }
# setsid makes this script the process-group leader: a TERM takes the queues, omk_eval and llama-servers with it.
trap 'say "V9A_EVAL_FAIL terminated"; trap - TERM INT; kill 0' TERM INT
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }

queue() { # tag first_port bench...
  local tag=$1 P=$2; shift 2
  local b sj par cp spid cl live rc r
  for b in "$@"; do
    sj=$RES/$b/$SN/summary.json
    # omk_eval writes a summary.json even for a killed cell (rc -15, score null): only a scored rc-0 summary is done.
    # An incomplete one is moved aside (never deleted) and the cell reruns, resuming from its SQLite cache.
    if [ -f "$sj" ]; then
      if "$PY" -c "import json,sys;d=json.load(open('$sj'));sys.exit(0 if d.get('rc')==0 and d.get('score') is not None else 1)"; then
        say "[$tag] $b: scored summary exists, skip"; continue
      fi
      mv "$sj" "${sj%.json}.incomplete_$(date -u +%Y%m%dT%H%M%SZ).json"
      say "[$tag] $b: incomplete summary moved aside, rerun"
    fi
    par=2
    say "[$tag] $b: start (GPU$G port $P --parallel $par --sampler-profile $PROFILE)"
    ( cd "$OMK" && CUDA_VISIBLE_DEVICES=$G "$PY" eval/omk_eval.py --model "$Q6" --tokenizer "$TOK" \
        --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
        --sampler-profile "$PROFILE" > "$LD/$b.log" 2>&1 ) & cp=$!
    live=none; spid=""
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
      ok)   say "[$tag] $b: LIVE sampler OK (llama-server pid $spid: presence 0.025, no repeat penalty)";;
      bad)  say "[$tag] Q_FAIL $b: LIVE llama-server pid $spid has the wrong sampler flags: $cl"; killtree $cp; return 1;;
      none) if kill -0 $cp 2>/dev/null; then say "[$tag] Q_FAIL $b: no llama-server on port $P after 30 min"; killtree $cp; return 1; fi;;
    esac
    wait $cp; rc=$?
    if [ -f "$sj" ]; then
      r=$("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'))")
      say "[$tag] $b: rc=$rc RESULT $r"
      [ "${r##* }" = "$WANT" ] || { say "[$tag] Q_FAIL $b: summary sampler is not $WANT"; return 1; }
      [ $live = ok ] || say "[$tag] WARN $b: finished before the live gate saw its server"
    else
      say "[$tag] $b: rc=$rc NO summary.json (see $LD/$b.log)"
    fi
    P=$((P+1)); sleep 30
  done
  say "[$tag] QUEUE_DONE"
}

mkdir -p "$RES" "$LD"
for f in "$Q6" "$TOK/config.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml"; do
  [ -e "$f" ] || die "missing $f"
done
"$PY" - "$Q6" <<'PY' || die "Q6_K GGUF does not record the full AC imatrix"
import sys
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
def get(k):
    v = f.get(k)
    if v is None: return None
    p = v.parts[v.data[0]]
    return bytes(p).decode() if v.types[0].name == "STRING" else int(p[0])
im, ch, eos = get("quantize.imatrix.file"), get("quantize.imatrix.chunks_count"), get("tokenizer.ggml.eos_token_id")
print(f"imatrix.file={im} chunks={ch} eos={eos} file_type={get('general.file_type')}")
sys.exit(0 if im and ch and ch >= 9500 and eos == 106 else 1)
PY
"$PY" - "$OMK/eval/templates" "${Q0[@]}" "${Q1[@]}" <<'PY' || die "a template is not frozen greedy"
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
# Probe the shared library, not the ~17 KB thin llama-server binary.
lib=$(ls "$LLAMA_BIN"/libllama-common.so.0.0.* 2>/dev/null | head -1)
[ -x "$LLAMA_BIN/llama-server" ] && [ -n "$lib" ] && grep -q LLAMA_CHAT_PEG_STRICT "$lib" \
  || die "$LLAMA_BIN is not the PEG-fixed build (no LLAMA_CHAT_PEG_STRICT in ${lib:-libllama-common})"
say "omk $(git -C "$OMK" rev-parse --short HEAD 2>/dev/null) | llama $LLAMA_BIN (pegfix) | Q6 sha $(cut -c1-16 "$Q6.sha256") | queues A: ${Q0[*]} | B: ${Q1[*]}"
queue A 8930 "${Q0[@]}" > "$LD/queue_A.log" 2>&1 & p0=$!
sleep 60
queue B 8950 "${Q1[@]}" > "$LD/queue_B.log" 2>&1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "queues exited rcA=$r0 rcB=$r1"
[ $r0 = 0 ] && [ $r1 = 0 ] && say "V9A_EVAL_DONE" || say "V9A_EVAL_FAIL queue rcA=$r0 rcB=$r1"
