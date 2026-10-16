#!/usr/bin/env bash
# pod_v9pub_toolbench.sh — ON THE POD, GPU1: tool-eval-bench cohort64k arm for the RE-PUBLISHED
# gemma-4-20B-v9-agentic (v9-ml ml_k8 build). Same basis as cohort64k/V9-ARM-BASIS-2026-10-14 except
# the host (pod RTX 3090 instead of solidpc RTX 3090):
#   model   the published Q4_K_M (kept by pod_v9_xollama_publish.py after its tags are on ollama.com),
#           sha == this run's quantize_gguf sidecar
#   drafter NONE (owner 2026-10-16: "no need for the drafter"; speculation never changes a score)
#   server  STOCK llama.cpp b9700 (9724f664e) llama-server via bin/llama-server-cohort, which only drops
#           the llamafile-only `--server` argv. Owner 2026-10-16: "run it on the pod with stock llama.cpp"
#           -- the opencoti c13-r2 CUDA-13 lib cannot use this pod's driver 560 (bug-790) and runs on CPU.
#   sampler greedy + the model's penalties: temp 0 / top_k 0 / top_p 1 / min_p 0 / rep 1.06 / last 96 /
#           presence 0.025 — forwarded per request (= omk greedy_pen, the basis of the card's evals)
#   harness tool-eval-bench 2.6.0 (992a6978), --hardmode --weight-by-difficulty, ctx 65536, pressure 0.25,
#           timeout 600, seeds 42..46
set -uo pipefail
W=/workspace/v9pub; T=$W/tb; NAME=gemma-4-20B-v9-agentic
Q=$W/eval_gguf/$NAME-Q4_K_M.gguf
SIDE=$W/gguf/$NAME/$NAME-Q4_K_M.gguf.sha256
LS=/workspace/llama.cpp/build/bin/llama-server
WRAP=$T/bin/llama-server-cohort
OUT=$T/cohort64k_v9ml
LOG=$T/tb.log
say(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOG; }
die(){ say "TB_FAIL $*"; exit 1; }
mkdir -p $OUT $T/models
say "waiting for the published Q4_K_M at $Q"
until [ -s "$Q" ]; do
  grep -q "FATAL" /workspace/logs/v9pub_xollama.out 2>/dev/null && die "ollama publisher FATAL before Q4_K_M"
  tmux has-session -t v9xpub 2>/dev/null || [ -s "$Q" ] || die "publisher session gone without the Q4_K_M"
  sleep 120
done
[ "$(sha256sum "$Q" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$SIDE")" ] || die "Q4_K_M sha != this run's sidecar"
[ -x "$LS" ] || die "no stock llama-server"
printf '#!/usr/bin/env bash\n# run_cohort.sh speaks llamafile argv; stock llama-server has no --server flag.\n[ "${1:-}" = "--server" ] && shift\nexec %s "$@"\n' "$LS" > "$WRAP"; chmod +x "$WRAP"
# capture first: `--version | grep -m1` under pipefail SIGPIPEs the server (rc 141) and reads as a failure
LSO=$("$LS" --version 2>&1) || die "llama-server --version rc=$?"
LSV=$(grep -m1 -oE "version: [0-9]+ \([0-9a-f]+\)" <<<"$LSO") || die "llama-server --version: no version line"
[ "$LSV" = "version: 9700 (9724f664e)" ] || die "llama-server is $LSV, expected b9700"
ln -sf "$Q" $T/models/$NAME-Q4_K_M.gguf
export PATH=$HOME/.local/bin:$PATH
TEB=$(command -v tool-eval-bench) || die "no tool-eval-bench"
"$(dirname "$(readlink -f "$TEB")")/python" -c "import json,glob,os,tool_eval_bench as t;d=glob.glob(os.path.dirname(t.__file__)+'/../tool_eval_bench*.dist-info/direct_url.json');c=json.load(open(d[0]))['vcs_info']['commit_id'];print('harness',c);assert c.startswith('992a6978')" | tee -a $LOG || die "harness commit"
printf '%s\n' "v9-agentic (re-published from v9-ml ml_k8) arm, cohort64k basis, $(date -u +%F)" \
  "model    ManniX-ITA/$NAME-GGUF $NAME-Q4_K_M.gguf sha256 $(cut -d' ' -f1 "$SIDE")" \
  "drafter  none (owner: no drafter; speculation does not change scores)" \
  "sampler  greedy + penalties: temp 0 / top_k 0 / top_p 1 / min_p 0 / repeat_penalty 1.06 / repeat_last_n 96 / presence_penalty 0.025 (forwarded per request)" \
  "server   STOCK llama.cpp $LSV llama-server (the cohort rows ran opencoti-llamafile c7 / c13-r2; c13-r2 cannot use this pod's driver 560)" \
  "host     vast pod 55947563 RTX 3090 GPU1 (the 2026-10-14 v9 arm ran on solidpc RTX 3090)" \
  "harness  tool-eval-bench 2.6.0 (992a6978), --hardmode --weight-by-difficulty, ctx 65536, pressure 0.25, timeout 600, seeds 42..46" > $OUT/RUN_BASIS.txt
say "running 5 seeds on GPU1"
CUDA_VISIBLE_DEVICES=1 OMK_TB_BIN=$WRAP OMK_TB_MODELS=$T/models OMK_TB_OUT=$OUT OMK_TB_PORT=8266 \
OMK_TB_PNG=$OUT/toolbench_scores_arm.png \
OMK_TB_MODELS_LIST="v9-agentic|$NAME-Q4_K_M.gguf||temp=0,top_p=1,top_k=0,min_p=0,presence_penalty=0.025,repeat_penalty=1.06,repeat_last_n=96" \
  bash /workspace/omnimergekit/eval/toolbench/run_cohort.sh >> $T/run_cohort.out 2>&1 || die "run_cohort rc=$?"
n=$(grep -rlE "\*\*Total Points\*\*" $OUT/v9-agentic-s4[2-6] 2>/dev/null | wc -l)
[ "$n" -eq 5 ] || die "only $n/5 seed reports"
[ -s $OUT/SKIPPED.txt ] && say "SKIPPED.txt: $(tr '\n' ';' < $OUT/SKIPPED.txt)"
say "TB_DONE 5/5 seeds"
