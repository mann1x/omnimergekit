#!/usr/bin/env bash
# PILOT: one CoderX tier end-to-end through ollama, verified at every step, BEFORE
# generating ~38 tags. Nothing is pushed here -- create + verify only.
#
# WHY A PILOT: ollama SILENTLY IGNORES unknown options (a bogus option returns HTTP 200
# with a normal completion). A misspelled or unsupported option baked into 38 tags is an
# invisible no-op: the tags look MTP-enabled and are not. And RENDERER/PARSER live in a
# 462-byte config blob that is invisible from `ollama list` and from the GGUF. So every
# claim below is checked against what ollama ACTUALLY STORED / ACTUALLY LOGGED, never
# against what the Modelfile said. [[feedback_provenance_ask_the_service_not_the_flag]]
#
# Params copied from the SHIPPED sibling tag mannix/qwen3.6-27b-a3b-coder:Q4_K_M
# (read via `ollama show --modelfile`), plus draft_num_predict 3.
# n=3 is the user's decision 2026-08-21: Blackwell peak (+33%); n=8 measured -23% there.
set -uo pipefail
BASE=mannix/qwen3.6-27b-a3b-coderx
TIER=Q4_K_M
GGUF=/mnt/sdc/ream-work/gguf_coderx/Qwen3.6-27B-A3B-CoderX-Q4_K_M.gguf
MMPROJ=/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf
WORK=/mnt/sdc/ream-work/ollama_pilot
mkdir -p "$WORK"
say(){ echo "[pilot $(date -u +%H:%M:%SZ)] $*"; }
fail(){ say "GATE FAIL: $*"; exit 1; }

[ -f "$GGUF" ]   || fail "no GGUF at $GGUF"
[ -f "$MMPROJ" ] || fail "no mmproj at $MMPROJ"

# 200G floor on bs2 root fs -- the ollama blob store lives there.
rootfree=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "${rootfree:-0}" -ge 215 ] || fail "root fs ${rootfree}G free, need >=215G to stay above the 200G floor"
say "root fs ${rootfree}G free"

TXT="${BASE}:pilot-${TIER}"
VIS="${BASE}:pilot-vision-${TIER}"

# ---- 1. text tag -----------------------------------------------------------
# TEMPLATE {{ .Prompt }} + explicit RENDERER/PARSER is exactly what the shipped sibling
# carries: the Go renderer supersedes, the passthrough template never renders.
cat > "$WORK/Modelfile.txt" <<EOF
FROM $GGUF
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER num_ctx 32768
PARAMETER temperature 1
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0
PARAMETER presence_penalty 1.5
PARAMETER repeat_penalty 1
PARAMETER draft_num_predict 3
EOF
say "creating $TXT"
ollama create "$TXT" -f "$WORK/Modelfile.txt" > "$WORK/create_txt.log" 2>&1 \
    || { tail -20 "$WORK/create_txt.log"; fail "create $TXT"; }

# ---- 2. verify what ollama STORED, not what the Modelfile said -------------
MF=$(ollama show --modelfile "$TXT" 2>/dev/null)
echo "$MF" | grep -q '^RENDERER qwen3.5' || fail "RENDERER not stored on $TXT"
echo "$MF" | grep -q '^PARSER qwen3.5'   || fail "PARSER not stored on $TXT"
echo "$MF" | grep -q '^PARAMETER draft_num_predict 3' \
    || fail "draft_num_predict NOT stored -- ollama dropped it silently"
say "  stored: RENDERER+PARSER qwen3.5, draft_num_predict 3  OK"

# ---- 3. does draft-mtp ACTUALLY ENGAGE? ------------------------------------
# The only trustworthy evidence is the runner's own log line. A completion that merely
# succeeds proves nothing -- an ignored option also completes normally.
say "generating to force a runner load..."
curl -s -m 600 http://localhost:11434/api/generate \
     -d "{\"model\":\"$TXT\",\"prompt\":\"Write a Python function that reverses a linked list.\",\"stream\":false,\"options\":{\"num_predict\":200}}" \
     > "$WORK/gen.json" 2>&1
/root/anaconda3/envs/omnimergekit/bin/python - "$WORK/gen.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
r = d.get("response", "")
print(f"  response {len(r)} chars, eval_count={d.get('eval_count')}, "
      f"tok/s={d.get('eval_count',0)/max(d.get('eval_duration',1)/1e9,1e-9):.1f}")
print("  head:", r[:120].replace("\n", " "))
PY
say "  runner log evidence:"
journalctl -u ollama --since "3 min ago" --no-pager 2>/dev/null \
  | grep -aiE "draft-mtp|MTP draft context|speculative implementation|no implementations specified" \
  | tail -5 | sed 's/^/    /'

# ---- 4. vision variant ------------------------------------------------------
# Append the projector to the SAME manifest so weights/params/license carry over by digest.
ollama show --modelfile "$TXT" 2>/dev/null | grep -v '^#' > "$WORK/Modelfile.vis"
echo "FROM $MMPROJ" >> "$WORK/Modelfile.vis"
say "creating $VIS"
ollama create "$VIS" -f "$WORK/Modelfile.vis" > "$WORK/create_vis.log" 2>&1 \
    || { tail -20 "$WORK/create_vis.log"; fail "create $VIS"; }
ollama show "$VIS" 2>/dev/null | grep -qi vision \
    || fail "$VIS does not advertise the vision capability"
say "  $VIS advertises vision  OK"
ollama show --modelfile "$VIS" 2>/dev/null | grep -q '^PARAMETER draft_num_predict 3' \
    || fail "vision variant lost draft_num_predict"
say "  vision variant kept draft_num_predict 3  OK"

say "PILOT_OK -- tags $TXT / $VIS created locally, NOT pushed"
