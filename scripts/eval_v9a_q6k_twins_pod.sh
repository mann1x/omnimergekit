#!/usr/bin/env bash
# eval_v9a_q6k_twins_pod.sh — head-to-head of the two 20B v9-agentic (sft12) Q6_K twins on a 2x RTX 3090 pod
# (owner 2026-11-16 18:40 CET: "we need to know which q6_k to use for the eval. use the pod to compare k-fac q6_k and
# ac i-matrix q6_k").
#
#   GPU0  ptq.gguf   llama-quantize b12386 + AtomicChat imatrix (9590 chunks), attention Q8_0   served v9a-sft12-q6k-acimat
#   GPU1  kfac.gguf  the same file re-rounded by omk_qad.kfac_round --stems all (bs2 factors)   served v9a-sft12-q6k-kfac
#
# One llama-server per GPU (Q6_K is 17 GB on a 24 GB card), --parallel 2, llama backend, the sft11 eval sampler
# greedy_pres (greedy + presence 0.025, repeat 1.0; eval/models/gemma-4-v9-agentic-sft11.yaml). Same gates as
# eval_v9_agentic_sft11_q6k.sh: frozen-greedy templates, LIVE llama-server flags per cell, summary sampler name,
# imatrix KV on both GGUFs, PEG-fixed llama-server LIBRARY. Fast benches first so the comparison fills in early; MPE is
# appended only when the nuprl docker image is usable here. Resumable: a scored rc-0 summary.json is skipped, an
# incomplete one is moved aside (never deleted) and the cell reruns from its SQLite cache.
# usage: eval_v9a_q6k_twins_pod.sh [BENCH ...]       sentinels V9Q_EVAL_DONE / V9Q_EVAL_FAIL; table -> $RES/q6k_twins_table.txt
set -uo pipefail
OMK=${OMK:-/workspace/omnimergekit}
G=${G:-/workspace/v9agentic/gguf/Q6_K}
TOK=${TOK:-/workspace/v9agentic/bf16}
RES=${RES:-/workspace/v9agentic/eval/results}; LD=${LD:-/workspace/v9agentic/eval/logs}
ENV=${ENV:-/workspace/miniconda/envs/omnimergekit}
PY=$ENV/bin/python
PROFILE=gemma-4-v9-agentic-sft11
WANT=greedy_pres
export LLAMA_BIN=${LLAMA_BIN:-/opt/llama.cpp/build-pegfix/bin}
export PATH="$ENV/bin:$PATH" LM_EVAL_BIN="$ENV/bin/lm_eval" OMK_PYTHON="$PY"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900 CUDA_DEVICE_ORDER=PCI_BUS_ID
# one server per GPU, but the other GPU is busy with the twin: lift only the planner's util test (VRAM test still applies)
export OMK_GPU_UTIL_THRESH=1.01
declare -A ARM=([acimat]=ptq [kfac]=kfac) GPU=([acimat]=0 [kfac]=1) PORT=([acimat]=8930 [kfac]=8950)
if [ $# -gt 0 ]; then BENCH=("$@"); else
  BENCH=(ifeval_100 gsm8k_100_qwen math500_100 arc_challenge_full aime_30 humaneval_full humanevalplus_full lcb_medium_55_v4 lcb_medium_100_v4 gpqa_diamond_full)
fi
say(){ echo "[v9q-eval $(date -u +%FT%TZ)] $*"; }
die(){ say "V9Q_EVAL_FAIL $*"; exit 1; }
trap 'say "V9Q_EVAL_FAIL terminated"; trap - TERM INT; kill 0' TERM INT
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
sn(){ echo "v9a-sft12-q6k-$1"; }
queue() { # arm bench...
  local arm=$1; shift
  local gpu=${GPU[$arm]} P=${PORT[$arm]} gguf=$G/${ARM[$arm]}.gguf SN; SN=$(sn $arm)
  local b sj par cp spid cl live rc r
  for b in "$@"; do
    sj=$RES/$b/$SN/summary.json
    if [ -f "$sj" ]; then
      if "$PY" -c "import json,sys;d=json.load(open('$sj'));sys.exit(0 if d.get('rc')==0 and d.get('score') is not None else 1)"; then
        say "[$arm] $b: scored summary exists, skip"; continue
      fi
      mv "$RES/$b/$SN" "$RES/$b/$SN.incomplete.$(date -u +%H%M%S)"; say "[$arm] $b: incomplete summary moved aside, rerun"
    fi
    par=2
    say "[$arm] $b: start (GPU$gpu port $P --parallel $par --sampler-profile $PROFILE)"
    ( cd "$OMK" && CUDA_VISIBLE_DEVICES=$gpu "$PY" eval/omk_eval.py --model "$gguf" --tokenizer "$TOK" \
        --template "$b" --backend llama --parallel "$par" --port "$P" --served-name "$SN" --results-dir "$RES" \
        --sampler-profile "$PROFILE" > "$LD/$arm.$b.log" 2>&1 ) & cp=$!
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
      say "[$arm] $b: rc=$rc NO summary.json (see $LD/$arm.$b.log)"
    fi
    P=$((P+1)); sleep 30
  done
  say "[$arm] QUEUE_DONE"
}
table() {
  "$PY" - "$RES" "$(sn acimat)" "$(sn kfac)" "${BENCH[@]}" <<'PY'
import json, os, sys
res, a, k, *benches = sys.argv[1:]
def sc(b, s):
    p = f"{res}/{b}/{s}/summary.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d.get("score") if d.get("rc") == 0 else None
rows = [("bench", "AC imatrix", "K-FAC", "delta")]
for b in benches:
    x, y = sc(b, a), sc(b, k)
    f = lambda v: "-" if v is None else f"{v:.4f}"
    rows.append((b, f(x), f(y), "-" if None in (x, y) else f"{y - x:+.4f}"))
w = [max(len(r[i]) for r in rows) for i in range(4)]
print("\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows))
PY
}
mkdir -p "$RES" "$LD"
for f in "$G/ptq.gguf" "$G/kfac.gguf" "$TOK/config.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml"; do
  [ -e "$f" ] || die "missing $f"
done
for f in "$G/ptq.gguf" "$G/kfac.gguf"; do
  (cd "$G" && sha256sum -c "$(basename $f).sha256" --quiet) || die "sha mismatch $f"
  "$PY" - "$f" <<'PY' || die "$f does not record the full AC imatrix / eos 106"
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
"$PY" - "$OMK/eval/templates" "${BENCH[@]}" <<'PY' || die "a template is not frozen greedy"
import sys, yaml
bad = []
for t in sys.argv[2:]:
    g = (yaml.safe_load(open(f"{sys.argv[1]}/{t}.yaml")) or {}).get("generation") or {}
    if g.get("do_sample") or float(g.get("temperature", 0)) != 0.0:
        bad.append(t)
print("greedy gate:", "PASS" if not bad else f"FAIL {bad}", f"({len(sys.argv) - 2} templates)")
sys.exit(1 if bad else 0)
PY
if docker image inspect ghcr.io/nuprl/multipl-e-evaluation:latest > /dev/null 2>&1; then
  export MPE_MODE=docker; BENCH+=(multipl_e_100); say "MPE docker image present: multipl_e_100 appended"
else
  say "MPE SKIPPED: nuprl docker image not usable on this pod (not a failure of either arm)"
fi
# Probe the shared library, not the thin llama-server binary.
lib=$(ls "$LLAMA_BIN"/libllama-common.so.[0-9]* 2>/dev/null | head -1)
[ -x "$LLAMA_BIN/llama-server" ] && [ -n "$lib" ] && grep -q LLAMA_CHAT_PEG_STRICT "$lib" \
  || die "$LLAMA_BIN is not the PEG-fixed build (no LLAMA_CHAT_PEG_STRICT in ${lib:-libllama-common})"
say "llama $LLAMA_BIN (pegfix, $(cd /workspace/llama.cpp && git describe --tags --always 2>/dev/null)) | ptq sha $(cut -c1-16 "$G/ptq.gguf.sha256") | kfac sha $(cut -c1-16 "$G/kfac.gguf.sha256") | benches: ${BENCH[*]}"
queue acimat "${BENCH[@]}" > "$LD/queue_acimat.log" 2>&1 & p0=$!
sleep 60
queue kfac "${BENCH[@]}" > "$LD/queue_kfac.log" 2>&1 & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
say "queues exited acimat=$r0 kfac=$r1"
table | tee "$RES/q6k_twins_table.txt"
[ $r0 = 0 ] && [ $r1 = 0 ] && say "V9Q_EVAL_DONE" || say "V9Q_EVAL_FAIL queue rc acimat=$r0 kfac=$r1"
