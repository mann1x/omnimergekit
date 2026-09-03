#!/usr/bin/env bash
# Polarity battery for gate_entropy_fires, extracted from r9_gepo_hf.sh so the test
# runs the SHIPPED function rather than a copy. A gate is only trustworthy once its
# GOLD-FAILS direction has been seen to fail. [[feedback_a_check_gold_fails_is_a_broken_check]]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/root/anaconda3/envs/omnimergekit/bin/python}
export PY
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

# lift the function body verbatim out of the harness
sed -n '/^gate_entropy_fires(){/,/^}/p' scripts/r9_gepo_hf.sh > "$T/fn.sh"
[ -s "$T/fn.sh" ] || { echo "FATAL: could not extract gate_entropy_fires"; exit 1; }
# shellcheck disable=SC1090
. "$T/fn.sh"

P=0; F=0
chk(){ if [ "$2" = "$3" ]; then echo "  ok   $1"; P=$((P+1));
       else echo "  FAIL $1 (want rc=$2 got rc=$3)"; F=$((F+1)); fi; }

SEED='>>> GEPO_ENTROPY_SEEDED n_groups=50 mu=1.2 sigma=0.3 H_low=1.14 H_high=1.29'
LIVE='>>> GEPO_ENTROPY H_mean=1.2000 H_std=0.3000 lo=1.1400 hi=1.2900 groups=2 pos_att=0.125 neg_att=0.000 any=0.125 lcb_exec/T:1/1@H1.050 mbpp_exec/N:0/1@H1.310'
DEAD='>>> GEPO_ENTROPY H_mean=1.2000 H_std=0.3000 lo=1.1400 hi=1.2900 groups=2 pos_att=0.000 neg_att=0.000 any=0.000 lcb_exec/T:0/1@H1.200'

printf '%s\n%s\n' "$SEED" "$LIVE" > "$T/ok.log"
gate_entropy_fires "$T/ok.log" >"$T/o1" 2>&1; chk "A  fired -> PASS" 0 $?
grep -q 'tiers_seen=lcb_exec/T,mbpp_exec/N' "$T/o1" \
  && { echo "  ok   A2 per-tier labels parsed off the log line"; P=$((P+1)); } \
  || { echo "  FAIL A2 tiers_seen not parsed: $(cat "$T/o1")"; F=$((F+1)); }

printf '%s\n%s\n%s\n' "$SEED" "$DEAD" "$DEAD" > "$T/dead.log"
gate_entropy_fires "$T/dead.log" >/dev/null 2>&1; chk "B  seeded but any=0 everywhere -> FAIL" 1 $?

printf '%s\n' "$DEAD" > "$T/noseed.log"
gate_entropy_fires "$T/noseed.log" >/dev/null 2>&1; chk "C  never seeded -> FAIL" 1 $?

printf '%s\n' "$SEED" > "$T/nosteps.log"
gate_entropy_fires "$T/nosteps.log" >/dev/null 2>&1; chk "D  seeded, zero step lines -> FAIL" 1 $?

echo "irrelevant training chatter" > "$T/empty.log"
gate_entropy_fires "$T/empty.log" >/dev/null 2>&1; chk "E  a run4-shaped log (no entropy at all) -> FAIL" 1 $?

printf '%s\n%s\n%s\n' "$SEED" "$DEAD" "$LIVE" > "$T/mixed.log"
gate_entropy_fires "$T/mixed.log" >/dev/null 2>&1; chk "F  one live step among dead ones -> PASS" 0 $?

echo "ENTROPY_SMOKE_GATE pass=$P fail=$F"
[ "$F" -eq 0 ]
