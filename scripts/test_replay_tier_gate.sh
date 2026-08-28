#!/usr/bin/env bash
# Polarity tests for gate_replay_tiers() in r9_gepo_hf.sh.
#
# WHY THIS FILE EXISTS. This gate has now been wrong three times, each time in a way
# that looked like a correct verdict:
#   bug-641  it read ONE sampled per-rank line as a population census and aborted
#            claiming "only 1 tier scored" when two had scored and a third never printed
#   bug-644  after tiers gained a /T|/N suffix, the observed-side regex r"(\w+):n=" could
#            not span the slash and captured the bare letter "T" as the tier name
#   bug-651  --limit-per-tier 1 meant a zero tier was backed by ONE problem, so a hard
#            GPQA item read as a dead tier and aborted a 41h run after a 25-min smoke
#
# Every one of those was found by running the gate against a real log, never by reading
# it. The gate's whole job is to be trusted unattended before a 41h run, so its
# polarities belong in a file that runs, not in a shell that closes.
#
# A gate is only proven by making it FIRE. Passing on a good log shows nothing on its
# own -- a gate that returns 0 unconditionally passes that test too.
# [[feedback_a_check_gold_fails_is_a_broken_check]]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/root/anaconda3/envs/omnimergekit/bin/python}
export PY
SP=$(mktemp -d "${TMPDIR:-/tmp}/tiergate.XXXXXX")
trap 'rm -rf "$SP"' EXIT
sed -n '/^gate_replay_tiers()/,/^}/p' scripts/r9_gepo_hf.sh > "$SP/fn.sh"
[ -s "$SP/fn.sh" ] || { echo "REFUSE: could not extract gate_replay_tiers from the launcher"; exit 2; }
# shellcheck disable=SC1090
. "$SP/fn.sh"

CEN='[t] POOL_CENSUS {"lcb_exec/T": 128, "mbpp_exec/N": 371, "mbpp_exec/T": 100, "mc_letter/N": 250} -> 849 problems'
GR='[gepo] GROUPS_OK 1 group(s)/rank x world=2 = 2 group(s)/step (G=4)'
fails=0

mklog(){ # mklog <file> <n-per-tier> <mc_nz>
  { echo "$CEN"; echo "$GR"
    echo ">>> V2_REWARD rank=0 | TIERS lcb_exec/T:n=$2,mean=0.3,nz=$2 mbpp_exec/T:n=$2,mean=0.5,nz=$2"
    echo ">>> V2_REWARD rank=1 | TIERS mbpp_exec/N:n=$2,mean=1.0,nz=$2 mc_letter/N:n=$2,mean=0.0,nz=$3"
  } > "$1"
}

want(){ # want <name> <logfile> <expect-rc> <expect-substring>
  local out rc
  out=$(gate_replay_tiers "$2" 2>&1); rc=$?
  if [ "$rc" != "$3" ] || ! grep -qF "$4" <<<"$out"; then
    echo "  [FAIL] $1  (rc=$rc want=$3)"; echo "$out" | tail -2 | sed 's/^/         /'
    fails=$((fails+1))
  else
    echo "  [PASS] $1"
  fi
}

echo "=== a zero tier must be judged against the SAMPLE SIZE behind it ==="
mklog "$SP/thin.log"  4  0
want "1 zero tier on ONE problem -> refuses as UNPROVABLE, not as dead" \
     "$SP/thin.log" 1 "backed by only ONE problem"
mklog "$SP/dead3.log" 12 0
want "2 zero tier on THREE problems -> refuses as DEAD" \
     "$SP/dead3.log" 1 "produced ZERO non-zero rewards across"
mklog "$SP/ok3.log"   12 6
want "3 all four tiers alive on three problems -> passes" \
     "$SP/ok3.log" 0 "REPLAY_TIER_ALIVE_OK"

echo "=== a tier that is MIXED IN but never scores is not the same as one scoring zero ==="
grep -v 'mc_letter/N:n=12,mean=0.0,nz=0' "$SP/dead3.log" > "$SP/missing.log"
want "4 tier in the census but absent from every TIERS line -> refuses as NEVER SCORED" \
     "$SP/missing.log" 1 "never scored"

echo "=== the observed-side parser must span the /T|/N suffix (bug-644) ==="
want "5 suffixed tier names survive parsing (not captured as bare 'T'/'N')" \
     "$SP/ok3.log" 0 "mbpp_exec/T(n=12"

echo "=== a legacy bare-kind census must still match suffixed observations ==="
{ echo "[t] REPLAY_MIX lcb=2 replay=4 {'mc_letter': 2, 'mbpp_exec': 2} -> 6 problems"; echo "$GR"
  echo '>>> V2_REWARD rank=0 | TIERS lcb_exec/T:n=12,mean=0.6,nz=9 mbpp_exec/N:n=12,mean=0.5,nz=8 mc_letter/N:n=12,mean=0.3,nz=4'
} > "$SP/legacy.log"
want "6 REPLAY_MIX (bare kinds) vs suffixed observations -> no phantom miss" \
     "$SP/legacy.log" 0 "REPLAY_TIER_ALIVE_OK"

echo "=== no evidence at all must never read as success ==="
{ echo "$CEN"; echo "$GR"; echo "nothing here"; } > "$SP/none.log"
want "7 no V2_REWARD line at all -> refuses" \
     "$SP/none.log" 1 "no V2_REWARD tier line"

echo
if [ "$fails" = 0 ]; then echo "TIER_GATE_OK"; else echo "TIER_GATE_FAIL ($fails failing)"; fi
exit $((fails ? 1 : 0))
