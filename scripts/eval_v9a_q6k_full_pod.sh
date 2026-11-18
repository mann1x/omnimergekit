#!/usr/bin/env bash
# eval_v9a_q6k_full_pod.sh — the v9-agentic (sft12) Q6_K canonical suite on a 2x RTX 3090 pod, ONE model, two queues
# (one llama-server per GPU: Q6_K is 17 GB on a 24 GB card). Which Q6_K (AC-imatrix PTQ or K-FAC re-rounded) is decided
# upstream by the 44-row top-256 KL twin on bs2 (owner 2026-11-16 19:09 CET: "I want to know which one is better without
# running the full evals, like with the other twins... then run the full evals on the one better").
#
# Sampler = eval/models/gemma-4-v9-agentic-sft11.yaml greedy_pres (greedy + presence 0.025, repeat 1.0). Gates as
# eval_v9_agentic_sft11_q6k.sh: frozen-greedy templates, LIVE llama-server flags per cell, summary sampler name, imatrix
# KV (>= 9500 chunks, eos 106), PEG-fixed llama-server LIBRARY. MPE runs only when the nuprl docker image is usable here.
# Resumable: a scored rc-0 summary.json is skipped, an incomplete one is moved aside (never deleted) and the cell reruns
# from its SQLite cache.
# usage: eval_v9a_q6k_full_pod.sh <acimat|kfac>      sentinels V9F_EVAL_DONE / V9F_EVAL_FAIL
set -uo pipefail
ARM=${1:?acimat|kfac}
case $ARM in acimat) FILE=ptq;; kfac) FILE=kfac;; *) echo "arm must be acimat or kfac"; exit 2;; esac
OMK=${OMK:-/workspace/omnimergekit}
G=${G:-/workspace/v9agentic/gguf/Q6_K}
GGUF=$G/$FILE.gguf
TOK=${TOK:-/workspace/v9agentic/bf16}
RES=${RES:-/workspace/v9agentic/eval/results}; LD=${LD:-/workspace/v9agentic/eval/logs}
ENV=${ENV:-/workspace/miniconda/envs/omnimergekit}
PY=$ENV/bin/python
SN=v9a-sft12-q6k-$ARM
PROFILE=gemma-4-v9-agentic-sft11
WANT=greedy_pres
export LLAMA_BIN=${LLAMA_BIN:-/opt/llama.cpp/build-pegfix/bin}
export PATH="$ENV/bin:$PATH" LM_EVAL_BIN="$ENV/bin/lm_eval" OMK_PYTHON="$PY"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMK_GPU_UTIL_THRESH=1.01   # the other GPU is busy with the other queue: lift only the planner's util test
Q0=(gpqa_diamond_full aime_30 math500_100 gsm8k_100_qwen ifeval_100 arc_challenge_full)
Q1=(humaneval_full humanevalplus_full lcb_medium_55_v4 lcb_medium_100_v4)
say(){ echo "[v9f-eval $(date -u +%FT%TZ)] $*"; }
die(){ say "V9F_EVAL_FAIL $*"; exit 1; }
trap 'say "V9F_EVAL_FAIL terminated"; trap - TERM INT; kill 0' TERM INT
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
queue() { # tag gpu first_port bench...
  local tag=$1 gpu=$2 P=$3; shift 3
  local b sj par cp spid cl live rc r
  for b in "$@"; do
    sj=$RES/$b/$SN/summary.json
    if [ -f "$sj" ]; then
      if "$PY" -c "import json,sys;d=json.load(open('$sj'));sys.exit(0 if d.get('rc')==0 and d.get('score') is not None else 1)"; then
        say "[$tag] $b: scored summary exists, skip"; continue
      fi
      mv "$RES/$b/$SN" "$RES/$b/$SN.incomplete.$(date -u +%H%M%S)"; say "[$tag] $b: incomplete summary moved aside, rerun"
    fi
    par=2
    say "[$tag] $b: start (GPU$gpu port $P --parallel $par --sampler-profile $PROFILE)"
    ( cd "$OMK" && CUDA_VISIBLE_DEVICES=$gpu "$PY" eval/omk_eval.py --model "$GGUF" --tokenizer "$TOK" \
        --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
        --sampler-profile "$PROFILE" > "$LD/$SN.$b.log" 2>&1 ) & cp=$!
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
      say "[$tag] $b: rc=$rc NO summary.json (see $LD/$SN.$b.log)"
    fi
    P=$((P+1)); sleep 30
  done
  say "[$tag] QUEUE_DONE"
}
table() {
  "$PY" - "$RES" "$SN" "${Q0[@]}" "${Q1[@]}" <<'PY'
import json, os, sys
res, s, *benches = sys.argv[1:]
for b in benches:
    p = f"{res}/{b}/{s}/summary.json"
    if not os.path.exists(p): print(f"{b:24s} -"); continue
    d = json.load(open(p)); v = d.get("score") if d.get("rc") == 0 else None
    print(f"{b:24s} {'-' if v is None else f'{v:.4f}'}  {d.get('metric','')}")
PY
}
mkdir -p "$RES" "$LD"
for f in "$GGUF" "$GGUF.sha256" "$TOK/config.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml"; do
  [ -e "$f" ] || die "missing $f"
done
(cd "$G" && sha256sum -c "$FILE.gguf.sha256" --quiet) || die "sha mismatch $GGUF"
"$PY" - "$GGUF" <<'PY' || die "GGUF does not record the full AC imatrix / eos 106"
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
"$PY" - "$OMK/eval/templates" "${Q0[@]}" "${Q1[@]}" multipl_e_100 <<'PY' || die "a template is not frozen greedy"
import sys, yaml
bad = []
for t in sys.argv[2:]:
    g = (yaml.safe_load(open(f"{sys.argv[1]}/{t}.yaml")) or {}).get("generation") or {}
    if g.get("do_sample") or float(g.get("temperature", 0)) != 0.0:
        bad.append(t)
print("greedy gate:", "PASS" if not bad else f"FAIL {bad}", f"({len(sys.argv) - 2} templates)")
sys.exit(1 if bad else 0)
PY
MPE_HARNESS=${MPE_HARNESS:-/workspace/MultiPL-E}
if docker image inspect ghcr.io/nuprl/multipl-e-evaluation:latest > /dev/null 2>&1; then
  export MPE_MODE=docker; Q1+=(multipl_e_100); say "MPE docker image present: multipl_e_100 appended to queue B"
elif command -v rustc >/dev/null && command -v javac >/dev/null && node --version 2>/dev/null | grep -qE "^v(1[6-9]|[2-9][0-9])" \
     && [ -f "$MPE_HARNESS/evaluation/src/main.py" ] && [ -f /usr/multiple/javatuples-1.2.jar ]; then
  # eval/multipl_e/install_mpe_toolchains_native.sh provisioned this pod: UNSANDBOXED model code, throwaway pod only
  export MPE_MODE=native MPE_HARNESS; Q1+=(multipl_e_100)
  say "MPE NATIVE (no docker): $(rustc --version | cut -d' ' -f2) / $(javac -version 2>&1) / node $(node --version), harness $(git -C "$MPE_HARNESS" rev-parse --short HEAD 2>/dev/null); multipl_e_100 appended to queue B"
else
  say "MPE SKIPPED: neither the nuprl docker image nor the native toolchains (install_mpe_toolchains_native.sh) are present (run multipl_e_100 for $SN elsewhere)"
fi
lib=$(ls "$LLAMA_BIN"/libllama-common.so.[0-9]* 2>/dev/null | head -1)
[ -x "$LLAMA_BIN/llama-server" ] && [ -n "$lib" ] && grep -q LLAMA_CHAT_PEG_STRICT "$lib" \
  || die "$LLAMA_BIN is not the PEG-fixed build (no LLAMA_CHAT_PEG_STRICT in ${lib:-libllama-common})"
say "arm $ARM -> $GGUF (sha $(cut -c1-16 "$GGUF.sha256")) served $SN | llama $LLAMA_BIN pegfix $(cd /workspace/llama.cpp && git describe --tags --always 2>/dev/null) | A(GPU0): ${Q0[*]} | B(GPU1): ${Q1[*]}"
queue A 0 8930 "${Q0[@]}" > "$LD/$SN.queue_A.log" 2>&1 & p0=$!
sleep 60
queue B 1 8950 "${Q1[@]}" > "$LD/$SN.queue_B.log" 2>&1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "queues exited rcA=$r0 rcB=$r1"
table | tee "$RES/$SN.table.txt"
[ $r0 = 0 ] && [ $r1 = 0 ] && say "V9F_EVAL_DONE $SN" || say "V9F_EVAL_FAIL queue rcA=$r0 rcB=$r1"
