#!/usr/bin/env bash
# a3b-gepo1 (armJ + R9 GEPO brevity adapter) on the EXACT banked armJ basis.
#
# WHY EACH SETTING IS WHAT IT IS
# ------------------------------
# Every knob below is copied from the driver that produced the armJ cell this run is
# compared against. Nothing here is a fresh choice:
#
#   multipl_e_100  <- ream-work/eval_ream_arms.sh
#       --parallel 4  AND  --metadata backend_args.llama_ctx=49152
#                          --metadata backend_args.llama_content_headroom=8192
#       The template's own frozen block says llama_ctx: 16384 / llama_parallel: 4
#       (4096 per slot). eval_ream_arms.sh OVERRIDES that to 49152 (12288 per slot),
#       so armJ's banked cell is a 12288/slot column. Dropping the override would
#       silently run 4096/slot -- a different basis under the same template name.
#
#   lcb_v6_77q_48k <- ream-work/chain_lcb48k.sh
#       --parallel 8, NO metadata override (the template carries llama_ctx: 524288).
#       524288/8 = 65536 per slot > the 49152 gen cap. GEOMETRY IS GATED: if per-slot
#       n_ctx is not exactly 65536 the slots saturate and generations collapse
#       SILENTLY (T172.4). Abort rather than bank a collapsed column.
#
# SAMPLER: greedy for both, from the frozen templates. No --sampler flag. Verified
# out of summary.json.sampler.name AFTER the fact -- a greedy cohort and a sampled
# cohort must never be tabulated together.
#
# TOKENIZER: the merged gepo dir. tokenizer.json and chat_template.jinja are
# byte-identical to armJ's; the only tokenizer_config.json delta is `local_files_only`,
# a save_pretrained artifact with no tokenization effect.
#
# GPU1 ONLY. GPU0 is not mine.
set -uo pipefail

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
GGUF=/mnt/sdc/ml/brevity/gepo/gguf_gepo1/a3b-gepo1-Q6_K.gguf
TOK=/mnt/sdc/ml/brevity/gepo/a3b-gepo1
LOG=/mnt/sdc/ml/brevity/gepo/eval_gepo1.log

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"
# LLAMA_ARG_SPEC_TYPE deliberately UNSET -- no speculative decoding, same as both anchors.

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }

echo "=== eval_gepo1 start $(ts) ==="

# ---- preflight ---------------------------------------------------------------
FAIL=0
for p in "$GGUF" "$TMPL_FIX" "$TOK/tokenizer.json" "$OMK/eval/omk_eval.py" \
         "$OMK/eval/templates/multipl_e_100.yaml" "$OMK/eval/templates/lcb_v6_77q_48k.yaml"; do
  [ -s "$p" ] || { say "MISSING: $p"; FAIL=1; }
done
[ "$(head -c4 "$GGUF")" = "GGUF" ] || { say "not a GGUF: $GGUF"; FAIL=1; }

# The frozen greedy block must still be greedy in BOTH templates. Drift here means a
# sampled run masquerading as the greedy cohort -- stop before spending hours.
for t in multipl_e_100 lcb_v6_77q_48k; do
  "$OMKPY" - "$OMK/eval/templates/$t.yaml" "$t" <<'PY' || FAIL=1
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); g = d.get("generation", {}) or {}
want = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "do_sample": False}
bad = {k: g.get(k) for k, v in want.items() if g.get(k) != v}
print(f"[greedy-gate] {sys.argv[2]}: " + ("OK" if not bad else f"DRIFT {bad}"))
raise SystemExit(1 if bad else 0)
PY
done

free=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
say "GPU1 free=${free}MiB"
[ "$free" -ge 60000 ] || { say "GPU1 has only ${free}MiB free"; FAIL=1; }
[ "$FAIL" = 0 ] || { say "PREFLIGHT FAILED -- nothing launched"; exit 2; }
say "PREFLIGHT_OK"

# ---- readout helpers ---------------------------------------------------------
readout(){ "$OMKPY" - "$1" "$2" <<'PY'
import json, sys
name = sys.argv[2]
try: d = json.load(open(sys.argv[1]))
except Exception as e: print(f"[readout] {name}: NO SUMMARY ({e})"); raise SystemExit(0)
s = d.get("sampler") or {}
t = d.get("token_stats") or {}
c = t.get("completion_tokens") or {}
f = t.get("finish_reasons") or {}
print(f"[readout] {name}: score={d.get('score')} metric={d.get('metric')} "
      f"filter={d.get('filter')} quant={d.get('quant')} sampler={s.get('name')} "
      f"dur={round((d.get('duration_s') or 0)/3600,2)}h")
print(f"[readout] {name}: n={t.get('n')} completion p50={c.get('p50')} p90={c.get('p90')} "
      f"max={c.get('max')} sum={c.get('sum')} finish={f} empty={t.get('empty_completions')}")
if s.get("name") != "template_default":
    print(f"[readout] {name}: !!! SAMPLER IS {s.get('name')} -- NOT the greedy cohort, do not tabulate")
PY
}

run_one(){  # name template port parallel extra_args...
  local name=$1 tmpl=$2 port=$3 par=$4; shift 4
  local s=$RES/$tmpl/$name/summary.json
  if [ -f "$s" ]; then say "SKIP $name (summary exists)"; readout "$s" "$name"; return 0; fi
  say ">>>> START $name  tmpl=$tmpl parallel=$par port=$port $*"
  local t0=$SECONDS
  "$OMKPY" "$OMK/eval/omk_eval.py" --backend llama --template "$tmpl" --quant q6_k \
      --model "$GGUF" --tokenizer "$TOK" --served-name "$name" --port "$port" \
      --results-dir "$RES" --parallel "$par" "$@"
  local rc=$?
  say "<<<< END $name rc=$rc in $(( (SECONDS-t0)/60 ))m"
  [ -f "$s" ] || { say "FAIL $name: no summary.json"; return 1; }
  readout "$s" "$name"
  return 0
}

# ---- column 1: MultiPL-E 100 (fast; armJ = 0.71) -----------------------------
run_one a3b_gepo1 multipl_e_100 8097 4 \
    --metadata backend_args.llama_ctx=49152 \
    --metadata backend_args.llama_content_headroom=8192
MPE_RC=$?

# ---- column 2: LCB v6 77q @ 48k (long; armJ = 0.7273) ------------------------
run_one lcb48k_a3b_gepo1 lcb_v6_77q_48k 8099 8
LCB_RC=$?

# GEOMETRY GATE, after the fact. Per-slot ctx must be 65536; anything else means the
# T172.4 saturation trap and the column is not trustworthy at 49152 max_gen_toks.
SLOG=$RES/lcb_v6_77q_48k/lcb48k_a3b_gepo1/server.log
if [ -f "$SLOG" ]; then
  CTX=$(grep -a -o "new slot, n_ctx = [0-9]*" "$SLOG" 2>/dev/null | head -1 | tr -dc 0-9)
  say "LCB GEOMETRY per-slot n_ctx=${CTX:-unknown} (want 65536)"
  [ "${CTX:-0}" -eq 65536 ] || say "!!! LCB GEOMETRY WRONG (${CTX:-unknown}) -- column NOT comparable"
fi
MSLOG=$RES/multipl_e_100/a3b_gepo1/server.log
if [ -f "$MSLOG" ]; then
  MCTX=$(grep -a -o "new slot, n_ctx = [0-9]*" "$MSLOG" 2>/dev/null | head -1 | tr -dc 0-9)
  say "MPE GEOMETRY per-slot n_ctx=${MCTX:-unknown} (want 12288 = 49152/4)"
fi

# ---- the comparison table ----------------------------------------------------
say "=== a3b-gepo1 vs armJ (both greedy / template_default / q6_k) ==="
"$OMKPY" - <<'PY'
import json, os
R = "/srv/ml/eval_results/ream_arms"
rows = [("MPE-100",  f"{R}/multipl_e_100/hybrid_p24_ourssal_reapfloor/summary.json",
                     f"{R}/multipl_e_100/a3b_gepo1/summary.json"),
        ("LCB-48k",  f"{R}/lcb_v6_77q_48k/lcb48k_armJ_hybrid_p24/summary.json",
                     f"{R}/lcb_v6_77q_48k/lcb48k_a3b_gepo1/summary.json")]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

print(f"{'bench':<9} {'arm':<10} {'score':>8} {'p50 tok':>9} {'p90 tok':>9} "
      f"{'sum tok':>10} {'trunc':>8} {'empty':>6} {'sampler':>17}")
print("-" * 92)
for bench, pa, pb in rows:
    for tag, p in (("armJ", pa), ("gepo1", pb)):
        d = load(p)
        if d is None:
            print(f"{bench:<9} {tag:<10} {'(missing)':>8}"); continue
        t = d.get("token_stats") or {}; c = t.get("completion_tokens") or {}
        f = t.get("finish_reasons") or {}
        n = t.get("n") or 0
        tr = f.get("length", 0)
        print(f"{bench:<9} {tag:<10} {d.get('score'):>8} {c.get('p50'):>9} {c.get('p90'):>9} "
              f"{c.get('sum'):>10} {f'{tr}/{n}':>8} {t.get('empty_completions'):>6} "
              f"{(d.get('sampler') or {}).get('name'):>17}")
    a, b = load(pa), load(pb)
    if a and b:
        sa, sb = a.get("score"), b.get("score")
        ca = (a.get("token_stats") or {}).get("completion_tokens") or {}
        cb = (b.get("token_stats") or {}).get("completion_tokens") or {}
        try:
            dscore = (sb - sa) * 100
            dtok = (cb.get("sum") - ca.get("sum")) / ca.get("sum") * 100
            dp50 = (cb.get("p50") - ca.get("p50")) / max(ca.get("p50"), 1) * 100
            print(f"{'':<9} {'DELTA':<10} {dscore:>+7.2f}pp {dp50:>+8.1f}% "
                  f"{'':>9} {dtok:>+9.1f}%   <- brevity readout")
        except Exception: pass
    print()
PY

echo "###### EVAL_GEPO1_DONE $(ts) rc_mpe=$MPE_RC rc_lcb=$LCB_RC ######"
