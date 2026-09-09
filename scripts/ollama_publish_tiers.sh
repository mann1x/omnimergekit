#!/usr/bin/env bash
# ollama_publish_tiers.sh — publish a GGUF tier sweep from Hugging Face to ollama.com as
# BOTH a text tag `:<tier>` and a vision tag `:vision-<tier>`, in one pass, disk-bounded.
#
# Differs from the neighbours, so pick the right tool:
#   ollama_push_generic.sh        HF -> ollama, text only, one target per invocation
#   build_vision_ollama_tiers.sh  pulls an ALREADY-PUBLISHED text tag and adds the projector
#   republish_*_ollama.sh         re-publish from a LOCAL gguf dir with per-tier sampler temps
#   THIS                          HF -> text + vision-<tier> together, one tier resident at a
#                                 time, gated on what ollama STORED (not on what we wrote)
#
# "Override, never fork" (the build_vision_ollama_tiers.sh convention): every model-specific
# value is an env override and the defaults are the Qwen3.6-27B-A3B-CoderX release that this
# script was written for, so that campaign's invocation stays byte-for-byte reproducible.
#
#   HF_REPO=ManniX-ITA/Foo-GGUF OL_BASE=mannix/foo GGUF_STEM=Foo \
#   MMPROJ=/path/mmproj.gguf TIERS="Q8_0 Q4_K_M" \
#   bash ollama_publish_tiers.sh
#
# HF_TOKEN must be exported (private repos). It is read with `${HF_TOKEN:?}` and NEVER given a
# `${HF_TOKEN:-<literal>}` convenience default -- that is exactly how a live token leaked on
# 2026-05-23. See docs/SECURITY.md.
#
# RESUMABLE: one `.done` marker per tier under $WORK; re-running skips finished tiers and
# reuses an already-staged GGUF, so a failed tier costs no re-download.
set -uo pipefail
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

PY="${PY:-/root/anaconda3/envs/omnimergekit/bin/python}"
HF_REPO="${HF_REPO:-ManniX-ITA/Qwen3.6-27B-A3B-CoderX-MTP-GGUF}"
OL_BASE="${OL_BASE:-mannix/qwen3.6-27b-a3b-coderx}"
GGUF_STEM="${GGUF_STEM:-Qwen3.6-27B-A3B-CoderX}"
MMPROJ="${MMPROJ:-/mnt/sdc/ream-work/mmproj/mmproj-Qwen3.6-27B-A3B-Coder-F16.gguf}"
WORK="${WORK:-/mnt/sdc/ream-work/ollama_pub}"
STAGE="${STAGE:-/mnt/sdc/ream-work/ollama_stage}"
LATEST_TIER="${LATEST_TIER:-Q4_K_M}"
STAGE_FLOOR_G="${STAGE_FLOOR_G:-40}"      # keep this much free after a tier's blob copy
PREFETCH_FLOOR_G="${PREFETCH_FLOOR_G:-100}"
TIERS="${TIERS:-Q8_0 Q6_K_L Q6_K Q5_K_L Q5_K_M Q5_K_S Q4_K_L Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_XL Q3_K_L Q3_K_M Q3_K_S IQ3_M Q2_K_L IQ2_M IQ2_XS}"

# Sampler + runtime params baked into every tag. Defaults = the shipped Qwen3.6 coder set.
# Minimum ollama version, emitted as the Modelfile `REQUIRES` directive
# (parser/parser.go:136,699 -> server/create.go:68 `config.Requires = r.Requires`).
# Without it a client on an older ollama pulls the tag, loads it, and dies at render
# with `unknown renderer "qwen3.8"` instead of getting a clean version refusal. The
# official library/qwen3.8:27b declares 0.32.12 and so did our own pre-2026-09-09
# v6 tags; every tag this script published lacked it purely because we never set it.
OL_REQUIRES="${OL_REQUIRES:-0.32.12}"
OL_RENDERER="${OL_RENDERER:-qwen3.5}"
OL_PARSER="${OL_PARSER:-qwen3.5}"
OL_NUM_CTX="${OL_NUM_CTX:-32768}"
OL_TEMPERATURE="${OL_TEMPERATURE:-1}"
OL_TOP_P="${OL_TOP_P:-0.95}"
OL_TOP_K="${OL_TOP_K:-20}"
OL_MIN_P="${OL_MIN_P:-0}"
OL_PRESENCE_PENALTY="${OL_PRESENCE_PENALTY:-1.5}"
OL_REPEAT_PENALTY="${OL_REPEAT_PENALTY:-1}"
# MTP self-speculative decoding. ollama maps draft_num_predict to
# `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling`.
# n=3 is a deliberate default: on Blackwell (5080) 189.65 -> 251.94 tok/s (+33%) at n=3 but
# 145.37 (-23%, WORSE than not speculating) at n=8, while a 3090 peaks at n=8. The optimum is
# hardware-dependent and n=3 is near-peak on both. Set DRAFT_N=0 to omit the parameter for a
# model with no MTP block.
DRAFT_N="${DRAFT_N:-3}"

# Resolve the GC helper next to THIS script, never relative to cwd -- the loop is normally
# launched detached from an arbitrary directory.
GCPY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ollama_gc_orphans.py"

mkdir -p "$WORK" "$STAGE"
say(){ echo "[pub $(date -u +%H:%M:%SZ)] $*" | tee -a "$WORK/publish.log"; }

if [ "$MMPROJ" = none ]; then
  say "MMPROJ=none -- TEXT-ONLY campaign, no vision-<tier> tags will be built"
else
  [ -f "$MMPROJ" ] || { say "REFUSE: no mmproj at $MMPROJ"; exit 1; }
fi
# Refuse loudly rather than silently skipping the per-tier reclaim: `ollama rm` frees no
# blobs, so without the GC the store grows by a full tier each pass and the loop stalls on
# the disk floor mid-campaign (measured: 113G -> 207G over 10 tiers, 2026-08-21).
[ -f "$GCPY" ] || { say "REFUSE: no ollama_gc_orphans.py next to this script ($GCPY)"; exit 1; }

# Daemon-resolved store, for the no-TEMPLATE gate below. Resolved the way the DAEMON
# sees it (OLLAMA_MODELS -> unit Environment -> daemon user's home), never guessed.
OL_STORE="$("$PY" "$GCPY" --print-store 2>/dev/null | tail -1)"
[ -d "$OL_STORE/manifests" ] || { say "REFUSE: cannot resolve the ollama store (got '$OL_STORE'); the no-TEMPLATE gate cannot run and this script will not publish blind."; exit 1; }
say "store: $OL_STORE"

# HARD GATE. `TEMPLATE {{ .Prompt }}` was introduced in ca77bfe and shipped on every
# tag this script published until 2026-09-09, including via the VISION path, which
# rebuilt its Modelfile from `ollama show --modelfile` -- and ollama SYNTHESISES a
# TEMPLATE line into that output even when the source Modelfile had none. A comment
# alone did not stop it coming back, so this refuses to push instead.
#
# Gates on the STORED MANIFEST, not on the Modelfile we wrote and not on
# `ollama show` (which synthesises the very line under test).
SANITIZE_PY='
import sys
has_tmpl, req = sys.argv[1] == "1", sys.argv[2]
out, saw_req = [], False
for ln in sys.stdin.read().splitlines():
    st = ln.strip()
    if st.startswith("TEMPLATE ") and not has_tmpl:
        continue
    if st.startswith("REQUIRES "):
        saw_req = True
    out.append(ln)
if req and not saw_req:
    idx = next((i for i, l in enumerate(out) if l.strip().startswith("FROM ")), -1)
    out.insert(idx + 1, "REQUIRES " + req)
sys.stdout.write("\n".join(out) + "\n")
'

sanitize_shown_modelfile(){
  # `ollama show --modelfile <tag>` is NOT a faithful round-trip of what was stored.
  # Measured 2026-09-09 on mannix/omnimerge-v6:IQ2_S:
  #   TEMPLATE  stored: NO template layer     shown: `TEMPLATE {{ .Prompt }}` INVENTED
  #   REQUIRES  stored: requires "0.32.12"    shown: OMITTED entirely
  # Anything rebuilt from that output therefore GAINS a template the model never had
  # and LOSES its version floor -- which is exactly how the vision tags acquired a
  # template after it had been removed from the text path.
  #
  # This repairs the shown Modelfile against the STORED BLOBS, which are the truth:
  #   * drop every TEMPLATE line when the manifest has NO template layer
  #     (keep it when there IS one -- then the template is real and must survive)
  #   * re-add REQUIRES from the config blob when the config carries one and the
  #     shown output dropped it
  # $1 = full tag, stdin = `ollama show --modelfile` output, stdout = repaired.
  local full="$1" name tg mf cd cfg has_tmpl req
  name="${full%%:*}"; tg="${full##*:}"
  mf="$OL_STORE/manifests/registry.ollama.ai/$name/$tg"
  if [ ! -f "$mf" ]; then say "  REFUSE: cannot sanitize $full -- no stored manifest"; return 1; fi
  has_tmpl=0
  grep -q "vnd\.ollama\.image\.template" "$mf" && has_tmpl=1
  cd="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["config"]["digest"].replace(":","-"))' "$mf")" || return 1
  cfg="$OL_STORE/blobs/$cd"
  req="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("requires") or "")' "$cfg" 2>/dev/null)"
  # NOTE: `$PY - <<HEREDOC` would feed the PROGRAM on stdin and leave nothing for the
  # script to read -- caught by the truth-table test, which returned only the REQUIRES
  # line. Pass the code with -c so stdin stays the piped Modelfile.
  "$PY" -c "$SANITIZE_PY" "$has_tmpl" "$req"
}

assert_stored_identity(){
  # Gate on the STORED ARTIFACTS -- the manifest and the config/params blobs the
  # daemon actually wrote -- never on `ollama show --modelfile`. Measured 2026-09-09
  # on IQ2_S, `ollama show` is BOTH lossy and inventive:
  #     requires        stored 0.32.12          shown: OMITTED   -> false "missing"
  #     template layer  stored: ABSENT          shown: TEMPLATE {{ .Prompt }} INVENTED
  # so it would hide a real REQUIRES drop and report a TEMPLATE that does not exist.
  # ollama also returns success for Modelfile directives it silently ignores, which is
  # why every field is verified after the fact rather than trusted from the write.
  local full="$1" name tg mf cd cfg
  name="${full%%:*}"; tg="${full##*:}"
  mf="$OL_STORE/manifests/registry.ollama.ai/$name/$tg"
  if [ ! -f "$mf" ]; then say "  REFUSE: no stored manifest at $mf"; return 1; fi
  if grep -q "vnd\.ollama\.image\.template" "$mf"; then
    say "  REFUSE: $full carries a TEMPLATE layer. The RENDERER is the chat format; a"
    say "          passthrough TEMPLATE beside it is redundant at best and overrides it"
    say "          at worst. Never rebuild a Modelfile from \`ollama show --modelfile\`,"
    say "          which synthesises one. NOT PUSHING."
    return 1
  fi
  cd="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["config"]["digest"].replace(":","-"))' "$mf")" || {
    say "  REFUSE: cannot read config digest from $mf"; return 1; }
  cfg="$OL_STORE/blobs/$cd"
  [ -f "$cfg" ] || { say "  REFUSE: config blob $cfg missing"; return 1; }
  "$PY" - "$cfg" "$OL_RENDERER" "$OL_PARSER" "$OL_REQUIRES" <<'PYGATE'
import json, sys
c = json.load(open(sys.argv[1]))
want = {"renderer": sys.argv[2], "parser": sys.argv[3], "requires": sys.argv[4]}
bad = [f"{k}={c.get(k)!r} want {v!r}" for k, v in want.items() if v and c.get(k) != v]
if bad:
    print("  REFUSE: stored config mismatch -- " + "; ".join(bad))
    sys.exit(1)
PYGATE
}

# `ollama push` PRINTS an auth error and EXITS 0 (2026-05-18: 31 tags reported DONE, zero
# uploaded, ~150 GB egress wasted). The OUTPUT is the success signal, not $?.
# Failure markers must be PHRASES. A bare "401" matched "401 MB" in the progress counter and
# aborted a SUCCESSFUL 28 GB push (2026-08-21) -- a substring that occurs in NORMAL output
# cannot be a failure oracle. The authoritative signal is the POSITIVE one.
AUTHMARK='need to be signed in|not authenticated|unauthorized|sign in to|push failed|forbidden|access denied|HTTP 401|status 401'
OKMARK='You can find your model at'
push_checked(){   # $1 = tag, $2 = logfile
  local tmp="$2.last" rc clean
  ollama push "$1" >"$tmp" 2>&1
  rc=$?
  # strip ANSI/CR spinner noise before matching, then append to the durable log
  sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r/\n/g' "$tmp" | grep -vE '^[[:space:]]*$' >> "$2"
  clean=$(sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r/\n/g' "$tmp")
  rm -f "$tmp"
  if grep -qiE "$AUTHMARK" <<<"$clean"; then
    say "AUTH FAILURE on $1 -- aborting the whole loop (the same key fails every tier)"
    grep -iE "$AUTHMARK" <<<"$clean" | tail -3
    exit 3
  fi
  grep -qF "$OKMARK" <<<"$clean" || { say "$1: no upload confirmation in push output"; return 1; }
  return $rc
}

emit_params(){   # shared by text and vision so the two can never drift apart
  # DO NOT ADD A `TEMPLATE` LINE HERE, and do not rebuild a Modelfile from
  # `ollama show --modelfile` -- it SYNTHESISES one. `TEMPLATE {{ .Prompt }}` was
  # added in ca77bfe (2026-08-21) on the mistaken belief that `ollama create` needs a
  # template when a RENDERER is set. It does not. The RENDERER *is* the chat format;
  # a passthrough TEMPLATE beside it is redundant at best and overrides it at worst.
  # It shipped on every tag this script published until 2026-09-09, and on the vision
  # tags it survived the first removal because that path went through `ollama show`.
  # assert_no_template() now refuses to push a tag whose STORED manifest has one.
  #
  # NOTE: an earlier note here claimed the TEMPLATE line suppressed `requires`. That
  # was WRONG and is retracted -- `requires` was absent simply because we never
  # emitted the directive. Both facts are independent; see OL_REQUIRES above.
  [ -n "$OL_REQUIRES" ] && echo "REQUIRES $OL_REQUIRES"
  echo "RENDERER $OL_RENDERER"
  echo "PARSER $OL_PARSER"
  echo "PARAMETER num_ctx $OL_NUM_CTX"
  echo "PARAMETER temperature $OL_TEMPERATURE"
  echo "PARAMETER top_p $OL_TOP_P"
  echo "PARAMETER top_k $OL_TOP_K"
  echo "PARAMETER min_p $OL_MIN_P"
  echo "PARAMETER presence_penalty $OL_PRESENCE_PENALTY"
  echo "PARAMETER repeat_penalty $OL_REPEAT_PENALTY"
  [ "$DRAFT_N" != 0 ] && echo "PARAMETER draft_num_predict $DRAFT_N"
}

for T in $TIERS; do
  [ -f "$WORK/$T.done" ] && { say "$T already done, skipping"; continue; }

  # Both the staged download AND `ollama create`'s blob copy land on the store's filesystem
  # when OLLAMA_MODELS points there, so that is the fs to gate on -- not the root fs.
  sfree=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc 0-9)
  if [ "${sfree:-0}" -lt "$PREFETCH_FLOOR_G" ]; then
    say "REFUSE at $T: $STAGE has ${sfree}G free (floor ${PREFETCH_FLOOR_G}G); clear gate-failed tiers first"
    exit 1
  fi

  G="$STAGE/$GGUF_STEM-$T.gguf"
  if [ -s "$G" ]; then
    say "=== $T (${sfree}G free) — reusing staged $(du -h "$G" | cut -f1)"
  else
    say "=== $T (${sfree}G free) — fetching"
    HF_REPO="$HF_REPO" GGUF_STEM="$GGUF_STEM" "$PY" - "$T" "$G" <<'PYEOF' || { say "$T: download FAILED"; continue; }
import os, shutil, sys
from huggingface_hub import hf_hub_download
tier, dest = sys.argv[1], sys.argv[2]
p = hf_hub_download(os.environ["HF_REPO"],
                    f"{os.environ['GGUF_STEM']}-{tier}.gguf",
                    local_dir=os.path.dirname(dest))
if os.path.abspath(p) != os.path.abspath(dest):
    shutil.move(p, dest)
print("fetched", dest, os.path.getsize(dest))
PYEOF
  fi
  [ -f "$G" ] || { say "$T: no file after download"; continue; }

  # `ollama create` COPIES the GGUF into the blob store, so the staged file and its copy
  # coexist. The requirement is not a fixed threshold: the fs must survive a copy of THIS
  # tier. A flat check passes just above the line and then lands below it on a 28 GB tier.
  need=$(( $(stat -c %s "$G") / 1000000000 + 3 ))
  sfree=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc 0-9)
  if [ $(( sfree - need )) -lt "$STAGE_FLOOR_G" ]; then
    say "REFUSE at $T: ${sfree}G free, blob copy needs ~${need}G -> would leave $(( sfree - need ))G, under the ${STAGE_FLOOR_G}G floor"
    exit 1
  fi
  say "$T: ${sfree}G free, blob copy ~${need}G -> $(( sfree - need ))G after (floor ${STAGE_FLOOR_G}G) OK"

  TXT="$OL_BASE:$T"
  VIS="$OL_BASE:vision-$T"

  { echo "FROM $G"; emit_params; } > "$WORK/Modelfile.$T"
  ollama create "$TXT" -f "$WORK/Modelfile.$T" >"$WORK/create.$T.log" 2>&1 \
      || { say "$T: create text FAILED"; tail -5 "$WORK/create.$T.log"; continue; }   # keep $G

  # Gate on what ollama STORED, never on the Modelfile we just wrote: ollama returns HTTP 200
  # for options it does not recognise, so an unsupported PARAMETER is silently dropped.
  ok=1
  # RENDERER/PARSER/REQUIRES are verified from the STORED CONFIG by
  # assert_stored_identity below, not from `ollama show` -- see the note there.
  # Repaired against the stored blobs first: `ollama show` invents a TEMPLATE and
  # drops REQUIRES, so an unrepaired read gates on fiction in both directions.
  MF=$(ollama show --modelfile "$TXT" 2>&1 | sanitize_shown_modelfile "$TXT")
  if [ "$DRAFT_N" != 0 ]; then
    grep -q "^PARAMETER draft_num_predict $DRAFT_N" <<<"$MF" \
      || { say "$T: draft_num_predict DROPPED"; ok=0; }
  fi
  [ "$ok" = 1 ] || { say "$T: GATE FAIL, not pushing"; continue; }   # keep $G for retry

  if [ "$MMPROJ" != none ]; then
    # Build the vision Modelfile from emit_params DIRECTLY -- never by round-tripping
    # through `ollama show --modelfile`, which SYNTHESISES a `TEMPLATE {{ .Prompt }}`
    # line that was never in the text Modelfile. Measured 2026-09-09 on IQ2_S: the text
    # tag came out with template NONE and the vision tag with '{{ .Prompt }}', so the
    # two tags silently diverged on the exact axis emit_params exists to keep identical.
    { echo "FROM $G"; echo "FROM $MMPROJ"; emit_params; } > "$WORK/Modelfile.vis.$T"
    ollama create "$VIS" -f "$WORK/Modelfile.vis.$T" >"$WORK/create.vis.$T.log" 2>&1 \
        || { say "$T: create vision FAILED"; ok=0; }
    if [ "$ok" = 1 ]; then
      # `ollama show` can come back EMPTY while the daemon is still settling after a large
      # create/push, which reads identically to a tag that genuinely lacks the capability.
      # Retry, and NEVER 2>/dev/null -- swallowing stderr is what made that undiagnosable.
      vshow=""; vmf=""
      for try in 1 2 3 4 5; do
        vshow=$(ollama show "$VIS" 2>&1)
        vmf=$(ollama show --modelfile "$VIS" 2>&1 | sanitize_shown_modelfile "$VIS")
        grep -qi vision <<<"$vshow" && { [ "$DRAFT_N" = 0 ] || grep -q "^PARAMETER draft_num_predict $DRAFT_N" <<<"$vmf"; } && break
        say "$T: vision probe attempt $try inconclusive, retrying"
        sleep 10
      done
      grep -qi vision <<<"$vshow" || { say "$T: vision tag lacks vision cap; ollama show said:"; sed 's/^/      /' <<<"$vshow" | head -20; ok=0; }
      if [ "$DRAFT_N" != 0 ]; then
        grep -q "^PARAMETER draft_num_predict $DRAFT_N" <<<"$vmf" \
          || { say "$T: vision tag lost draft_num_predict"; ok=0; }
      fi
    fi
    [ "$ok" = 1 ] || { say "$T: VISION GATE FAIL, pushing neither"; ollama rm "$TXT" "$VIS" >/dev/null 2>&1; continue; }
  fi


  assert_stored_identity "$TXT" || { say "$T: IDENTITY GATE FAIL"; ollama rm "$TXT" "$VIS" >/dev/null 2>&1; continue; }
  if [ "$MMPROJ" != none ]; then
    assert_stored_identity "$VIS" || { say "$T: IDENTITY GATE FAIL (vision)"; ollama rm "$TXT" "$VIS" >/dev/null 2>&1; continue; }
  fi

  if [ "$MMPROJ" = none ]; then say "$T: gates OK — pushing $TXT (text-only)"
  else say "$T: gates OK — pushing $TXT and $VIS"; fi
  push_checked "$TXT" "$WORK/push.$T.log" || { say "$T: push text FAILED"; ok=0; }
  [ "$MMPROJ" = none ] || push_checked "$VIS" "$WORK/push.$T.log" || { say "$T: push vision FAILED"; ok=0; }

  if [ "$T" = "$LATEST_TIER" ] && [ "$ok" = 1 ]; then
    ollama cp "$TXT" "$OL_BASE:latest" >/dev/null 2>&1 \
      && push_checked "$OL_BASE:latest" "$WORK/push.latest.log" \
      && say "  :latest -> $T pushed"
    ollama rm "$OL_BASE:latest" >/dev/null 2>&1
  fi

  ollama rm "$TXT" "$VIS" >/dev/null 2>&1
  # `ollama rm` drops the MANIFEST but NOT the blobs; the daemon only prunes at startup.
  "$PY" "$GCPY" --apply 2>&1 | grep -E "OLLAMA_GC_OK|REFUSE" | tee -a "$WORK/publish.log"
  rm -f "$G"
  [ "$ok" = 1 ] && touch "$WORK/$T.done" && say "$T DONE"
done

say "remaining local tags for $OL_BASE:"; ollama list | grep -F "$OL_BASE" | tee -a "$WORK/publish.log"
echo "OLLAMA_PUBLISH_TIERS_DONE"
