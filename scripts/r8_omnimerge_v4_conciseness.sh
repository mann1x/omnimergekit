#!/usr/bin/env bash
# R8 (#846) — fill the OmniMerge v4 cell in the qwen_suite conciseness cohort.
#
# The question: is A3B-Coder or OmniMerge v4 the more concise coder? v4 has never been
# run on any bench that Coder/CoderX share, so the comparison does not exist yet.
#
# BASIS — copied from the banked cohort, not invented. Every arm already in
# /srv/ml/eval_results/qwen_suite was run at:
#     quant Q6_K · backend llama · sampler profile qwen3_6, name `recommended`
#     (temp 0.6 / top_p 0.95 / top_k 20 / do_sample true) · tokenizer qwen36-35b-a3b-tok
# Any deviation makes the new row incomparable, so this script asserts the resolved
# sampler AFTER each bench rather than trusting the flag.
# [[feedback_sampler_is_a_cohort_fact_read_it]] [[feedback_provenance_ask_the_service_not_the_flag]]
#
# QUANT: the NON-MTP repo. The cohort arms are plain Q6_K; the -MTP-GGUF Q6_K carries an
# extra nextn block and would be a different serve geometry. Quant is part of the basis,
# which is also why the Q4_K_M already on disk is NOT usable here.
#
# GPU1 ONLY. GPU0 is not ours to take without asking.
set -uo pipefail

REPO=/srv/ml/repos/omnimergekit
GGUF_DIR=/mnt/sdc/ml/gguf
GGUF="$GGUF_DIR/Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf"
HF_REPO=ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF
GGUF_FILE=Qwen3.6-27B-Omnimerge-v4-Q6_K.gguf
TOK=/mnt/sdc/ml/google/qwen36-35b-a3b-tok
RES=/srv/ml/eval_results/qwen_suite
SN=qwenomnimergev4_q6k
PY=/root/anaconda3/envs/omnimergekit/bin/python
# omk_eval shells out to a BARE `lm-eval` for the lm-eval-backed templates (HumanEval,
# IFEval, GPQA...). Setting PY alone is not enough — without the env's bin dir on PATH
# those benches die with FileNotFoundError: 'lm-eval' while the native-backend ones
# (LCB, MultiPL-E) sail through, so the failure looks bench-specific rather than
# environmental. bug-623.
export PATH="$(dirname "$PY"):$PATH"
PORT=8471
# ifeval + gpqa added 2026-08-22: they are the two cells that decide v4 against CoderX
# for an AGENTIC deployment (instruction-following and the non-code floor), and v4 has
# never been run on either. Existing cells are skipped, so this is additive.
BENCHES=(lcb_v6_77q humaneval_full_think multipl_e_100 ifeval_100 gpqa_diamond_full)
LOGDIR=/mnt/sdc/ml/r8_logs
mkdir -p "$LOGDIR"

ts(){ date '+%F %T %Z'; }
say(){ echo "[$(ts)] $*"; }

# ---------------------------------------------------------------- preflight
say "=== R8 preflight ==="
FREE=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc '0-9')
say "/mnt/sdc free: ${FREE}G"
[ "$FREE" -ge 40 ] || { say "FATAL: need >=40G free for a 22GB pull + headroom"; exit 2; }

USED1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
say "GPU1 used: ${USED1} MiB"
[ "$USED1" -lt 2000 ] || { say "FATAL: GPU1 busy (${USED1} MiB) — refusing to contend"; exit 2; }

for b in "${BENCHES[@]}"; do
  [ -f "$REPO/eval/templates/$b.yaml" ] || { say "FATAL: missing template $b"; exit 2; }
done
say "templates present: ${BENCHES[*]}"

# The cohort rows we are joining. If these are absent the comparison has no anchor.
for arm in qwen184e_q6k qwenhybridp24_q6k qwen256e_q6k; do
  [ -f "$RES/lcb_v6_77q/$arm/summary.json" ] || { say "FATAL: anchor $arm missing"; exit 2; }
done
say "cohort anchors present (184e Coder / hybridp24 CoderX / 256e base)"

# ---------------------------------------------------------------- fetch
if [ -f "$GGUF" ] && [ "$(head -c4 "$GGUF")" = "GGUF" ]; then
  say "GGUF already present: $(stat -c %s "$GGUF" | numfmt --to=iec)"
else
  say "downloading $GGUF_FILE from $HF_REPO ..."
  # `hf`, not `huggingface-cli` and not `python -m huggingface_hub.commands.*`:
  # hub >= 1.x dropped the commands submodule entirely (bs2 runs 1.24.0).
  # [[feedback_hf_cli_renamed]]
  command -v hf >/dev/null || { say "FATAL: hf CLI not found"; exit 3; }
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$HF_REPO" "$GGUF_FILE" \
      --local-dir "$GGUF_DIR" > "$LOGDIR/download.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { say "FATAL: download exit=$rc (see $LOGDIR/download.log)"; exit 3; }
  [ "$(head -c4 "$GGUF")" = "GGUF" ] || { say "FATAL: bad magic after download"; exit 3; }
  say "downloaded: $(stat -c %s "$GGUF" | numfmt --to=iec)"
fi

# ---------------------------------------------------------------- run
cd "$REPO" || exit 2
for b in "${BENCHES[@]}"; do
  if [ -f "$RES/$b/$SN/summary.json" ]; then
    say "$b: cell already exists, skipping"
    continue
  fi
  say "$b: launch on GPU1:$PORT (sampler-profile qwen3_6 / recommended)"
  CUDA_VISIBLE_DEVICES=1 "$PY" "$REPO/eval/omk_eval.py" \
      --backend llama --template "$b" --quant q6_k \
      --model "$GGUF" --tokenizer "$TOK" \
      --sampler-profile qwen3_6 --sampler recommended \
      --served-name "$SN" --results-dir "$RES" \
      --port "$PORT" --parallel 2 \
      > "$LOGDIR/${b}.log" 2>&1
  rc=$?
  say "$b: exit=$rc"

  # PROVENANCE GATE — ask the artifact, not the flag. A row whose sampler silently
  # resolved to greedy (or whose quant is not q6_k) is not in this cohort and must
  # not be tabulated next to it.
  s="$RES/$b/$SN/summary.json"
  if [ -f "$s" ]; then
    "$PY" - "$s" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
sam = d.get("sampler") or {}
res = sam.get("resolved") or {}
bad = []
if sam.get("name") != "recommended":
    bad.append(f"sampler.name={sam.get('name')!r} want 'recommended'")
if res.get("temperature") != 0.6:
    bad.append(f"temperature={res.get('temperature')} want 0.6")
if not res.get("do_sample"):
    bad.append(f"do_sample={res.get('do_sample')} want True")
if str(d.get("quant", "")).lower() != "q6_k":
    bad.append(f"quant={d.get('quant')!r} want 'q6_k'")
if bad:
    print("BASIS_MISMATCH: " + " | ".join(bad))
    sys.exit(1)
print(f"BASIS_OK score={d.get('score')} sampler={sam.get('name')} "
      f"temp={res.get('temperature')} quant={d.get('quant')}")
PYEOF
    [ $? -eq 0 ] || say "!!! $b basis mismatch — DO NOT tabulate this row"
  else
    say "!!! $b produced no summary.json"
  fi
done

say "=== R8 conciseness report (v4 row joins the cohort) ==="
"$PY" "$REPO/eval/conciseness_report.py" "$RES/lcb_v6_77q/*" \
      --json-out "$LOGDIR/conciseness_lcb.json"
echo "###### R8_DONE $(ts) ######"
