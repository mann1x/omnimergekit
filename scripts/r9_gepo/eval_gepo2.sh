#!/usr/bin/env bash
# a3b-gepo2 (armJ + R9 run2 GEPO brevity adapter) on the EXACT banked armJ/gepo1 basis.
#
# WHY Q6_K AND NOT THE Q4_K_M WE ALSO BUILT
# -----------------------------------------
# Every banked cell this is compared against is Q6_K: armJ
# (hybrid_p24_ourssal_reapfloor / lcb48k_armJ_hybrid_p24) and gepo1 (a3b_gepo1 /
# lcb48k_a3b_gepo1). A Q4_K_M column would have no arm to sit next to -- it would
# need armJ AND gepo1 re-run at Q4_K_M to mean anything, which is a different and
# much larger job. Q4_K_M is built as the DEPLOYMENT tier; Q6_K is the MEASUREMENT
# tier. Do not tabulate one against the other.
#
# WHY EACH SETTING IS WHAT IT IS -- unchanged from eval_gepo1.sh, which took them
# from the driver that produced the armJ cell:
#
#   multipl_e_100  <- ream-work/eval_ream_arms.sh
#       --parallel 4  AND  --metadata backend_args.llama_ctx=49152
#                          --metadata backend_args.llama_content_headroom=8192
#       The template's own frozen block says llama_ctx: 16384 / llama_parallel: 4
#       (4096 per slot). eval_ream_arms.sh OVERRIDES that to 49152 (12288 per slot),
#       so the banked cells are a 12288/slot column. Dropping the override would
#       silently run 4096/slot -- a different basis under the same template name.
#
#   lcb_v6_77q_48k <- ream-work/chain_lcb48k.sh
#       --parallel 8, NO metadata override (the template carries llama_ctx: 524288).
#       524288/8 = 65536 per slot > the 49152 gen cap. GEOMETRY IS GATED: if per-slot
#       n_ctx is not exactly 65536 the slots saturate and generations collapse
#       SILENTLY (T172.4).
#
# SAMPLER: greedy for both, from the frozen templates. No --sampler flag. Verified
# out of summary.json.sampler.name AFTER the fact.
#
# READ THE BREVITY DELTA WITH THE CAP CROSS-TAB IN MIND: on MPE-100 a capped
# completion never compiles, so capped => fail, and a score delta between a
# brevity-tuned arm and its base can be a LENGTH delta wearing a capability
# delta's clothes (2026-08-24: gepo1's entire +2.67 pp over armJ was 8 fewer caps;
# on the both-uncapped set it was +0.00). The trunc column below is what makes
# that visible -- do not report the score without it.
#
# GPU1 ONLY. GPU0 is not mine.
set -uo pipefail

OMK=/srv/ml/repos/omnimergekit
OMKPY=/root/anaconda3/envs/omnimergekit/bin/python
RES=/srv/ml/eval_results/ream_arms
TMPL_FIX=/srv/ml/models/qwen36_chat_template_fixed.jinja
GGUF=/mnt/sdc/ml/brevity/gepo/gguf_gepo2/a3b-gepo2-Q6_K.gguf
TOK=/mnt/sdc/ml/brevity/gepo/a3b-gepo2
LOG=/mnt/sdc/ml/brevity/gepo/eval_gepo2.log

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export LM_EVAL_BIN=/root/anaconda3/envs/omnimergekit/bin/lm-eval
export LLAMA_EXTRA="--jinja --chat-template-file $TMPL_FIX"
# LLAMA_ARG_SPEC_TYPE deliberately UNSET -- no speculative decoding, same as the anchors.

exec >>"$LOG" 2>&1
ts(){ date -u '+%F %T UTC'; }
say(){ echo ">>> [$(ts)] $*"; }

echo "=== eval_gepo2 start $(ts) ==="

# ---- preflight ---------------------------------------------------------------
FAIL=0
for p in "$GGUF" "$TMPL_FIX" "$TOK/tokenizer.json" "$OMK/eval/omk_eval.py" \
         "$OMK/eval/templates/multipl_e_100.yaml" "$OMK/eval/templates/lcb_v6_77q_48k.yaml"; do
  [ -s "$p" ] || { say "MISSING: $p"; FAIL=1; }
done
[ "$(head -c4 "$GGUF")" = "GGUF" ] || { say "not a GGUF: $GGUF"; FAIL=1; }

# The GGUF must carry an imatrix, same as the two arms it is tabulated against. A
# no-imat column against two imat columns is a recipe difference, not a model delta.
"$OMKPY" - "$GGUF" <<'PY' || FAIL=1
import sys
from gguf import GGUFReader
kv = [k for k in GGUFReader(sys.argv[1]).fields if k.startswith("quantize.imatrix")]
print("[imat-gate] " + ("OK " + str(len(kv)) + " KV" if kv else "IMATRIX MISSING"))
raise SystemExit(0 if kv else 1)
PY

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

# ---- column 1: MultiPL-E 100 (fast; armJ = 0.7000, gepo1 = 0.7267) -----------
run_one a3b_gepo2 multipl_e_100 8097 4 \
    --metadata backend_args.llama_ctx=49152 \
    --metadata backend_args.llama_content_headroom=8192
MPE_RC=$?

# ---- column 2: LCB v6 77q @ 48k (long; armJ = 0.7273) ------------------------
run_one lcb48k_a3b_gepo2 lcb_v6_77q_48k 8099 8
LCB_RC=$?

# GEOMETRY GATE, after the fact. Per-slot ctx must be 65536; anything else means the
# T172.4 saturation trap and the column is not trustworthy at 49152 max_gen_toks.
SLOG=$RES/lcb_v6_77q_48k/lcb48k_a3b_gepo2/server.log
if [ -f "$SLOG" ]; then
  CTX=$(grep -a -o "new slot, n_ctx = [0-9]*" "$SLOG" 2>/dev/null | head -1 | tr -dc 0-9)
  say "LCB GEOMETRY per-slot n_ctx=${CTX:-unknown} (want 65536)"
  [ "${CTX:-0}" -eq 65536 ] || say "!!! LCB GEOMETRY WRONG (${CTX:-unknown}) -- column NOT comparable"
fi
MSLOG=$RES/multipl_e_100/a3b_gepo2/server.log
if [ -f "$MSLOG" ]; then
  MCTX=$(grep -a -o "new slot, n_ctx = [0-9]*" "$MSLOG" 2>/dev/null | head -1 | tr -dc 0-9)
  say "MPE GEOMETRY per-slot n_ctx=${MCTX:-unknown} (want 12288 = 49152/4)"
fi

# ---- the comparison table (THREE arms, one basis) ----------------------------
say "=== armJ vs a3b-gepo1 vs a3b-gepo2 (all greedy / template_default / imat-q6_k) ==="
"$OMKPY" - <<'PY'
import json
R = "/srv/ml/eval_results/ream_arms"
BENCH = [
    ("MPE-100", "multipl_e_100", [
        ("armJ",  "hybrid_p24_ourssal_reapfloor"),
        ("gepo1", "a3b_gepo1"),
        ("gepo2", "a3b_gepo2")]),
    ("LCB-48k", "lcb_v6_77q_48k", [
        ("armJ",  "lcb48k_armJ_hybrid_p24"),
        ("gepo1", "lcb48k_a3b_gepo1"),
        ("gepo2", "lcb48k_a3b_gepo2")]),
]

def load(tmpl, name):
    try: return json.load(open(f"{R}/{tmpl}/{name}/summary.json"))
    except Exception: return None

print(f"{'bench':<9} {'arm':<7} {'score':>8} {'p50 tok':>9} {'p90 tok':>9} "
      f"{'sum tok':>10} {'trunc':>8} {'empty':>6} {'sampler':>17}")
print("-" * 90)
for bench, tmpl, arms in BENCH:
    loaded = {}
    for tag, name in arms:
        d = load(tmpl, name)
        loaded[tag] = d
        if d is None:
            print(f"{bench:<9} {tag:<7} {'(missing)':>8}"); continue
        t = d.get("token_stats") or {}; c = t.get("completion_tokens") or {}
        f = t.get("finish_reasons") or {}
        n = t.get("n") or 0
        tr = f.get("length", 0)
        print(f"{bench:<9} {tag:<7} {d.get('score'):>8} {c.get('p50'):>9} {c.get('p90'):>9} "
              f"{c.get('sum'):>10} {f'{tr}/{n}':>8} {t.get('empty_completions'):>6} "
              f"{(d.get('sampler') or {}).get('name'):>17}")
    # Both deltas are against armJ, the shared base. gepo2-vs-gepo1 is also printed
    # because that is the run1-vs-run2 question, but it is a delta between two
    # tuned arms, not against the base -- read it as such.
    base = loaded.get("armJ")
    for tag in ("gepo1", "gepo2"):
        d = loaded.get(tag)
        if not (base and d): continue
        cb = (base.get("token_stats") or {}).get("completion_tokens") or {}
        cd = (d.get("token_stats") or {}).get("completion_tokens") or {}
        try:
            dscore = (d["score"] - base["score"]) * 100
            dtok = (cd["sum"] - cb["sum"]) / cb["sum"] * 100
            dp50 = (cd["p50"] - cb["p50"]) / max(cb["p50"], 1) * 100
            print(f"{'':<9} {tag+'-armJ':<7} {dscore:>+7.2f}pp {dp50:>+8.1f}% {'':>9} "
                  f"{dtok:>+9.1f}%   <- brevity readout")
        except Exception: pass
    g1, g2 = loaded.get("gepo1"), loaded.get("gepo2")
    if g1 and g2:
        c1 = (g1.get("token_stats") or {}).get("completion_tokens") or {}
        c2 = (g2.get("token_stats") or {}).get("completion_tokens") or {}
        try:
            print(f"{'':<9} {'g2-g1':<7} {(g2['score']-g1['score'])*100:>+7.2f}pp "
                  f"{(c2['p50']-c1['p50'])/max(c1['p50'],1)*100:>+8.1f}% {'':>9} "
                  f"{(c2['sum']-c1['sum'])/c1['sum']*100:>+9.1f}%   <- run2 vs run1")
        except Exception: pass
    print()
print("NOTE: on MPE-100 a truncated completion cannot compile, so capped => fail.")
print("      A score delta here is only a CAPABILITY delta on the problems uncapped")
print("      in BOTH arms -- read the trunc column before quoting the score.")
PY

echo "###### EVAL_GEPO2_DONE $(ts) rc_mpe=$MPE_RC rc_lcb=$LCB_RC ######"
