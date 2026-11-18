#!/usr/bin/env bash
# eval_v9a_q6k_lcbhard_pod.sh — the one cell the v9-agentic (sft12) K-FAC Q6_K canonical suite still lacks: lcb_hard_77,
# on the pod that ran the other eleven (eval_v9a_q6k_full_pod.sh, 2026-11-16/17), same model file, served name, sampler
# profile and gates, so the result lands in the same results tree and table. One queue on GPU0 (--parallel 2).
#   sampler  eval/models/gemma-4-v9-agentic-sft11.yaml greedy_pres (greedy + presence 0.025, repeat 1.0) — the suite's
#   gates    sha of the GGUF, imatrix KV (>= 9500 chunks, eos 106), frozen-greedy template, PEG-fixed llama-server
#            library, LIVE llama-server flags, summary sampler name
# usage: eval_v9a_q6k_lcbhard_pod.sh <acimat|kfac>      sentinels V9H_EVAL_DONE / V9H_EVAL_FAIL
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
BENCH=${BENCH:-lcb_hard_77}
GPU=${GPU:-0}; PORT=${PORT:-8970}
export LLAMA_BIN=${LLAMA_BIN:-/opt/llama.cpp/build-pegfix/bin}
export PATH="$ENV/bin:$PATH" LM_EVAL_BIN="$ENV/bin/lm_eval" OMK_PYTHON="$PY"
export HF_ALLOW_CODE_EVAL=1 OMK_GPU_WAIT_S=900 CUDA_DEVICE_ORDER=PCI_BUS_ID
say(){ echo "[v9h-eval $(date -u +%FT%TZ)] $*"; }
die(){ say "V9H_EVAL_FAIL $*"; exit 1; }
trap 'say "V9H_EVAL_FAIL terminated"; trap - TERM INT; kill 0' TERM INT
tree(){ for c in $(ps -o pid= --ppid "$1"); do tree "$c"; echo "$c"; done; }
killtree(){ local k; k="$(tree "$1") $1"; kill $k 2>/dev/null; sleep 5; kill -9 $k 2>/dev/null; }
mkdir -p "$RES" "$LD"
for f in "$GGUF" "$GGUF.sha256" "$TOK/config.json" "$OMK/eval/omk_eval.py" "$OMK/eval/models/$PROFILE.yaml" "$OMK/eval/templates/$BENCH.yaml"; do
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
"$PY" - "$OMK/eval/templates" "$BENCH" <<'PY' || die "template is not frozen greedy"
import sys, yaml
g = (yaml.safe_load(open(f"{sys.argv[1]}/{sys.argv[2]}.yaml")) or {}).get("generation") or {}
bad = bool(g.get("do_sample")) or float(g.get("temperature", 0)) != 0.0
print("greedy gate:", "FAIL" if bad else "PASS", sys.argv[2], g)
sys.exit(1 if bad else 0)
PY
lib=$(ls "$LLAMA_BIN"/libllama-common.so.[0-9]* 2>/dev/null | head -1)
[ -x "$LLAMA_BIN/llama-server" ] && [ -n "$lib" ] && grep -q LLAMA_CHAT_PEG_STRICT "$lib" \
  || die "$LLAMA_BIN is not the PEG-fixed build (no LLAMA_CHAT_PEG_STRICT in ${lib:-libllama-common})"
[ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] || die "GPU$GPU is busy"
say "arm $ARM -> $GGUF (sha $(cut -c1-16 "$GGUF.sha256")) served $SN | llama $LLAMA_BIN pegfix $(cd /workspace/llama.cpp && git describe --tags --always 2>/dev/null) | bench $BENCH on GPU$GPU port $PORT"
sj=$RES/$BENCH/$SN/summary.json
if [ -f "$sj" ]; then
  if "$PY" -c "import json,sys;d=json.load(open('$sj'));sys.exit(0 if d.get('rc')==0 and d.get('score') is not None else 1)"; then
    say "$BENCH: scored summary exists already: $(cat "$sj" | head -c 300)"; say "V9H_EVAL_DONE $SN (pre-existing)"; exit 0
  fi
  mv "$RES/$BENCH/$SN" "$RES/$BENCH/$SN.incomplete.$(date -u +%H%M%S)"; say "$BENCH: incomplete summary moved aside, rerun"
fi
say "$BENCH: start (GPU$GPU port $PORT --parallel 2 --sampler-profile $PROFILE)"
( cd "$OMK" && CUDA_VISIBLE_DEVICES=$GPU "$PY" eval/omk_eval.py --model "$GGUF" --tokenizer "$TOK" \
    --template "$BENCH" --backend llama --parallel 2 --port "$PORT" --served-name "$SN" --results-dir "$RES" \
    --sampler-profile "$PROFILE" > "$LD/$SN.$BENCH.log" 2>&1 ) & cp=$!
live=none; spid=""; cl=""
for _ in $(seq 1 180); do
  spid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
  if [ -n "$spid" ]; then
    cl=$(tr "\0" " " < /proc/$spid/cmdline)
    if grep -q -- "--presence-penalty 0.025 " <<<"$cl" && ! grep -qE -- "--repeat-penalty (0*[02-9]|1\.0*[1-9])" <<<"$cl"; then live=ok; else live=bad; fi
    break
  fi
  kill -0 $cp 2>/dev/null || break
  sleep 10
done
case $live in
  ok)   say "$BENCH: LIVE sampler OK (llama-server pid $spid: presence 0.025, no repeat penalty)";;
  bad)  say "Q_FAIL $BENCH: LIVE llama-server pid $spid has the wrong sampler flags: $cl"; killtree $cp; die "live sampler";;
  none) if kill -0 $cp 2>/dev/null; then say "Q_FAIL $BENCH: no llama-server on port $PORT after 30 min"; killtree $cp; die "no server"; fi;;
esac
wait $cp; rc=$?
[ -f "$sj" ] || die "$BENCH: rc=$rc NO summary.json (see $LD/$SN.$BENCH.log: $(tail -n 2 "$LD/$SN.$BENCH.log" | tr '\n' ' ' | cut -c1-200))"
r=$("$PY" -c "import json;d=json.load(open('$sj'));print(d.get('score'),d.get('metric'),(d.get('sampler') or {}).get('name'),'rc',d.get('rc'))")
say "$BENCH: rc=$rc RESULT $r"
[ "$(cut -d' ' -f3 <<<"$r")" = "$WANT" ] || die "summary sampler is not $WANT"
# samples sanity: count, empties, fences (the lm-eval rule)
"$PY" - "$RES/$BENCH/$SN" <<'PY'
import glob, json, sys
fs = glob.glob(sys.argv[1] + "/**/samples_*.jsonl", recursive=True)
n = empty = fence = 0
for f in fs:
    for line in open(f, errors="replace"):
        try: d = json.loads(line)
        except Exception: continue
        n += 1
        r = d.get("resps") or d.get("filtered_resps") or []
        t = r[0][0] if r and isinstance(r[0], list) else (r[0] if r else "")
        t = t if isinstance(t, str) else json.dumps(t)
        if len(t.strip()) < 5: empty += 1
        if "```" in t: fence += 1
print(f"samples: files={len(fs)} rows={n} empty={empty} fenced={fence}")
PY
say "V9H_EVAL_DONE $SN $BENCH"
