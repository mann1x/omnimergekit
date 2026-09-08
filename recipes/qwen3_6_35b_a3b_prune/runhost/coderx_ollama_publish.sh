#!/usr/bin/env bash
# CoderX ollama publish: per tier, a text tag and a vision-<tier> tag, both carrying MTP
# via `PARAMETER draft_num_predict 3`.
#
# RUNS ONLY AFTER coderx_ollama_pilot.sh PRINTS PILOT_OK. The pilot proves ollama actually
# STORES draft_num_predict and RENDERER/PARSER and that draft-mtp engages; without that,
# this loop would mint ~38 tags that look MTP-enabled and silently are not (ollama returns
# HTTP 200 for unknown options). [[feedback_provenance_ask_the_service_not_the_flag]]
#
# Sampler set is copied verbatim from the SHIPPED sibling tag
# mannix/qwen3.6-27b-a3b-coder:Q4_K_M, read back with `ollama show --modelfile`.
# n=3: user decision 2026-08-21. Blackwell peak +33%; n=8 measured -23% THERE while n=8
# won on a 3090 — the optimum is hardware-dependent, n=3 is near-peak on both.
#
# DISK: the campaign already freed every tier locally, so each tier is re-fetched from the
# (private) HF repo, used, and deleted. One tier resident at a time; blob store GC'd per tier.
# Since 2026-08-21 OLLAMA_MODELS=/mnt/sdc/ollama/models, so the blob copy lands on /mnt/sdc
# alongside the staged download -- that fs is what gets checked before every tier, not just
# once. Root is asserted too (>=200G) but is no longer on this loop's write path.
#
# RESUMABLE: a .done marker per tier; re-running skips finished tiers.
set -uo pipefail
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
PY=/root/anaconda3/envs/omnimergekit/bin/python
REPO=ManniX-ITA/Qwen3.6-27B-A3B-CoderX-MTP-GGUF
BASE=mannix/qwen3.6-27b-a3b-coderx
MMPROJ=/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf
WORK=/mnt/sdc/ream-work/ollama_pub
STAGE=/mnt/sdc/ream-work/ollama_stage
# Resolve the GC helper next to THIS script, not under $WORK (which is the ollama_pub
# subdir) and not relative to cwd -- the loop is launched with nohup from varying places.
GCPY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ollama_gc_orphans.py"
LATEST_TIER=Q4_K_M
mkdir -p "$WORK" "$STAGE"

TIERS="Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q5_K_S Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_XL Q3_K_L Q3_K_M Q3_K_S IQ3_M Q2_K_L IQ2_M IQ2_XS"

say(){ echo "[pub $(date -u +%H:%M:%SZ)] $*" | tee -a "$WORK/publish.log"; }
[ -f "$MMPROJ" ] || { say "REFUSE: no mmproj at $MMPROJ"; exit 1; }
# Refuse loudly rather than silently skipping the per-tier reclaim: without it the loop
# stalls on the /mnt/sdc floor ~10 tiers in, mid-campaign.
[ -f "$GCPY" ] || { say "REFUSE: no ollama_gc_orphans.py at $GCPY"; exit 1; }

# `ollama push` PRINTS an auth error and EXITS 0 (2026-05-18: 31 tags reported DONE,
# zero uploaded, ~150 GB egress wasted). The OUTPUT is the success signal, not $?.
# Failure markers must be PHRASES. A bare "401" matched inside a sha256 layer digest and
# aborted a SUCCESSFUL push (2026-08-21) -- a substring that occurs in normal output cannot
# be a failure oracle. The authoritative signal is the POSITIVE one ollama prints on upload.
AUTHMARK='need to be signed in|not authenticated|unauthorized|sign in to|push failed|forbidden|access denied|HTTP 401|status 401'
OKMARK='You can find your model at'
push_checked(){   # $1 = tag, $2 = logfile
  local tmp="$2.last"
  ollama push "$1" >"$tmp" 2>&1
  local rc=$?
  # strip ANSI/CR spinner noise before matching, then append to the durable log
  sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r/\n/g' "$tmp" | grep -vE '^\s*$' >> "$2"
  local clean; clean=$(sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r/\n/g' "$tmp")
  rm -f "$tmp"
  if grep -qiE "$AUTHMARK" <<<"$clean"; then
    say "AUTH FAILURE on $1 -- aborting the whole loop (same key fails every tier)"
    grep -iE "$AUTHMARK" <<<"$clean" | tail -3
    exit 3
  fi
  grep -qF "$OKMARK" <<<"$clean" || { say "$1: no upload confirmation in push output"; return 1; }
  return $rc
}

emit_params(){   # shared by text and vision so they can never drift apart
  cat <<'EOF'
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
}

for T in $TIERS; do
  [ -f "$WORK/$T.done" ] && { say "$T already done, skipping"; continue; }

  # 2026-08-21: OLLAMA_MODELS moved to /mnt/sdc/ollama/models, so BOTH the staged download
  # and the blob copy land on /mnt/sdc -- root is no longer on the write path. The floor that
  # matters is now sdc's, not root's. Root is still asserted (cheap) because a root at <200G
  # means something ELSE is filling it and this campaign should not add load.
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -lt 200 ]; then say "REFUSE at $T: root fs ${free}G is under the 200G floor (not this loop's doing -- investigate)"; exit 1; fi
  # coarse pre-fetch check; the exact per-tier one runs after staging (see 'need=' below).
  # A gate-failed tier KEEPS its staged GGUF (so a retry costs nothing), so the stage dir can
  # accumulate -- 100G covers the largest tier's download plus its blob copy.
  sfree=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ "${sfree:-0}" -lt 100 ]; then
    say "REFUSE at $T: /mnt/sdc ${sfree}G free; clear $STAGE of gate-failed tiers first"; exit 1
  fi

  G="$STAGE/Qwen3.6-27B-A3B-CoderX-$T.gguf"
  if [ -s "$G" ]; then
    # a prior aborted tier left this staged; re-fetching would burn tens of GB for nothing
    say "=== $T (/mnt/sdc ${sfree}G free) — reusing staged $(du -h "$G" | cut -f1)"
  else
  say "=== $T (/mnt/sdc ${sfree}G free) — fetching"
  "$PY" - "$T" "$G" <<'PYEOF' || { say "$T: download FAILED"; continue; }
import sys, os, shutil
from huggingface_hub import hf_hub_download
tier, dest = sys.argv[1], sys.argv[2]
p = hf_hub_download("ManniX-ITA/Qwen3.6-27B-A3B-CoderX-MTP-GGUF",
                    f"Qwen3.6-27B-A3B-CoderX-{tier}.gguf",
                    local_dir=os.path.dirname(dest))
if os.path.abspath(p) != os.path.abspath(dest):
    shutil.move(p, dest)
print("fetched", dest, os.path.getsize(dest))
PYEOF
  fi
  [ -f "$G" ] || { say "$T: no file after download"; continue; }

  # `ollama create` COPIES the GGUF into the blob store (now /mnt/sdc/ollama/models), so the
  # staged file and its blob copy coexist on the SAME fs until the push finishes. The
  # requirement is therefore not a fixed threshold: /mnt/sdc must survive a copy of THIS tier
  # (+mmproj +margin). A flat check passes just above the line and then lands below it on a
  # 28G Q8_0 -- breaching the floor it was meant to protect.
  need=$(( $(stat -c %s "$G") / 1000000000 + 3 ))
  sdcfree=$(df -BG --output=avail /mnt/sdc | tail -1 | tr -dc 0-9)
  if [ $(( sdcfree - need )) -lt 40 ]; then
    say "REFUSE at $T: /mnt/sdc ${sdcfree}G, blob copy needs ~${need}G -> would leave $(( sdcfree - need ))G, under the 40G floor"
    exit 1
  fi
  say "$T: /mnt/sdc ${sdcfree}G, blob copy ~${need}G -> $(( sdcfree - need ))G after (floor 40G) OK"

  TXT="$BASE:$T"
  VIS="$BASE:vision-$T"

  { echo "FROM $G"; emit_params; } > "$WORK/Modelfile.$T"
  ollama create "$TXT" -f "$WORK/Modelfile.$T" >"$WORK/create.$T.log" 2>&1 \
      || { say "$T: create text FAILED"; tail -5 "$WORK/create.$T.log"; continue; }   # keep $G for retry

  # Gate on what ollama STORED, never on the Modelfile we just wrote.
  MF=$(ollama show --modelfile "$TXT" 2>&1)
  ok=1
  echo "$MF" | grep -q '^RENDERER qwen3.5'                 || { say "$T: RENDERER missing"; ok=0; }
  echo "$MF" | grep -q '^PARSER qwen3.5'                   || { say "$T: PARSER missing"; ok=0; }
  echo "$MF" | grep -q '^PARAMETER draft_num_predict 3'    || { say "$T: draft_num_predict DROPPED"; ok=0; }
  [ "$ok" = 1 ] || { say "$T: GATE FAIL, not pushing"; continue; }   # keep $G for retry

  { ollama show --modelfile "$TXT" | grep -v '^#'; echo "FROM $MMPROJ"; } > "$WORK/Modelfile.vis.$T"
  ollama create "$VIS" -f "$WORK/Modelfile.vis.$T" >"$WORK/create.vis.$T.log" 2>&1 \
      || { say "$T: create vision FAILED"; ok=0; }
  if [ "$ok" = 1 ]; then
    # Retry: `ollama show` can come back empty while the daemon is still settling after a
    # large create/push (that is what failed Q8_0 at 10:11Z on 2026-08-21 even though the
    # same construction passes standalone). NEVER 2>/dev/null here -- swallowing stderr is
    # what made that failure undiagnosable.
    vshow=""; vmf=""
    for try in 1 2 3 4 5; do
      vshow=$(ollama show "$VIS" 2>&1)
      vmf=$(ollama show --modelfile "$VIS" 2>&1)
      grep -qi vision <<<"$vshow" && grep -q '^PARAMETER draft_num_predict 3' <<<"$vmf" && break
      say "$T: vision probe attempt $try inconclusive, retrying"
      sleep 10
    done
    grep -qi vision <<<"$vshow" || { say "$T: vision tag lacks vision cap; ollama show said:"; sed 's/^/      /' <<<"$vshow" | head -20; ok=0; }
    grep -q '^PARAMETER draft_num_predict 3' <<<"$vmf" || { say "$T: vision tag lost draft_num_predict"; ok=0; }
  fi
  [ "$ok" = 1 ] || { say "$T: VISION GATE FAIL, pushing neither"; ollama rm "$TXT" "$VIS" >/dev/null 2>&1; continue; }   # keep $G for retry

  say "$T: gates OK — pushing $TXT and $VIS"
  push_checked "$TXT" "$WORK/push.$T.log" || { say "$T: push text FAILED"; ok=0; }
  push_checked "$VIS" "$WORK/push.$T.log" || { say "$T: push vision FAILED"; ok=0; }

  if [ "$T" = "$LATEST_TIER" ] && [ "$ok" = 1 ]; then
    ollama cp "$TXT" "$BASE:latest" >/dev/null 2>&1 \
      && push_checked "$BASE:latest" "$WORK/push.latest.log" \
      && say "  :latest -> $T pushed"
    ollama rm "$BASE:latest" >/dev/null 2>&1
  fi

  ollama rm "$TXT" "$VIS" >/dev/null 2>&1
  # `ollama rm` drops the MANIFEST but NOT the blobs -- the daemon only prunes unreferenced
  # blobs at startup. Without this, every finished tier's full weight stays in the store and
  # the loop eats the same volume it stages downloads on: 113G -> 207G over 10 tiers on
  # 2026-08-21, until the /mnt/sdc floor (correctly) refused to continue. Reclaim per tier.
  "$PY" "$GCPY" --apply 2>&1 | grep -E "OLLAMA_GC_OK|REFUSE" | tee -a "$WORK/publish.log"
  rm -f "$G"
  [ "$ok" = 1 ] && touch "$WORK/$T.done" && say "$T DONE"
done

say "remaining local tags:"; ollama list | grep -i coderx | tee -a "$WORK/publish.log"
echo "CODERX_OLLAMA_PUBLISH_DONE"
