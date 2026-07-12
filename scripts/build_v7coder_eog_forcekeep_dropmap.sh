#!/usr/bin/env bash
# build_v7coder_eog_forcekeep_dropmap.sh — T202.4 drop-map stage (CPU only).
#
# 1. Reproduce the gated C6v3lcb drop-map from captured args, NO force-keep.
#    Assert per-layer set-equality vs the published drop. This proves the
#    canonical (force-keep-capable) generator's SCORING is identical to the
#    generator that built the published v7-coder — the single-variable control.
# 2. Select the targeted dropped-terminator pins (tc>=100, emit-RMS>=0.25).
# 3. Regenerate WITH --force-keep <pins>. Verify pins now kept + budget held +
#    exactly the pinned experts moved into keep (clean single-variable delta).
set -euo pipefail

PY=/root/anaconda3/envs/omnimergekit/bin/python
GEN=/srv/ml/scripts/gen_drop_v5_fk.py
SEL=/srv/ml/repos/omnimergekit/scripts/select_dropped_terminator_pins.py
GATED=/srv/ml/repos/omnimergekit/scripts/v7coder_C6v3lcb_drop_map.json   # == published drop
EOG=/mnt/sdc/ml/google/expert_neuron_v7_agentic_eog.json
WORK=/mnt/sdc/ml/sft_heal
REPRO=$WORK/repro_C6v3lcb_noFK.json
PINS=$WORK/v7coder_eog_targeted_pins.txt
OUT=$WORK/v7coder_eog_fk_drop_map.json

# Exact C6v3lcb generate args (from v7coder_C6v3lcb_drop_map.json.summary.json).
C6_ARGS=(
  --data /mnt/sdc/ml/google/expert_neuron_v7_code.json
  --target 98
  --protect-top 16
  --alpha 2.0
  --score-mode legacy
  --strategy max
  --normalize rank
  --classes generic_math generic_logic generic_code generic_science generic_creative generic_multilingual targeted_humaneval targeted_humanevalplus targeted_lcb_medium_55
  --class-weights 1 1 3 1 1 0 0 0 2
  --protect-strategy same
  --class-protect-floor 0
  --v4-floor-data /mnt/sdc/ml/google/expert_neuron_base_v7.json
  --v4-floor-top 0
  --v4-floor-map /srv/ml/repos/omnimergekit/scripts/v4_layer_floor_map_v7.json
  --breadth-bonus 0.5
  --outlier-wnorm-thresh 10000.0
  --outlier-mode median
  --baseline-drop-map /srv/ml/scripts/teacher_force_98e_p16_clean.json
)

echo "=== [1/3] REPRODUCTION CONTROL (no force-keep) ==="
"$PY" "$GEN" "${C6_ARGS[@]}" --output "$REPRO"

"$PY" - "$REPRO" "$GATED" <<'PY'
import json, sys
def norm(p):
    d = json.load(open(p))
    return {int(k): set(int(x) for x in v) for k, v in d.items()}
a, b = norm(sys.argv[1]), norm(sys.argv[2])
if a != b:
    print("FATAL: reproduction != gated C6v3lcb — scoring diverged. ABORT.")
    for L in sorted(set(a) | set(b)):
        if a.get(L, set()) != b.get(L, set()):
            print(f"  L{L} repro-only {sorted(a.get(L,set())-b.get(L,set()))} "
                  f"gated-only {sorted(b.get(L,set())-a.get(L,set()))}")
    sys.exit(3)
print(f"OK: reproduction == gated C6v3lcb ({len(a)} layers, "
      f"{ {len(v) for v in a.values()} } dropped/layer). Scoring identical.")
PY

echo
echo "=== [2/3] SELECT targeted dropped-terminator pins ==="
"$PY" "$SEL" --eog-map "$EOG" --drop-map "$GATED" \
  --min-tc 100 --min-rms 0.25 --out "$PINS"
FK=$(cat "$PINS")
echo "force-keep string: $FK"

echo
echo "=== [3/3] REGENERATE with --force-keep (single-variable delta) ==="
"$PY" "$GEN" "${C6_ARGS[@]}" --force-keep "$FK" --output "$OUT"

"$PY" - "$OUT" "$GATED" "$PINS" <<'PY'
import json, sys
def norm(p):
    d = json.load(open(p))
    return {int(k): set(int(x) for x in v) for k, v in d.items()}
new, gated = norm(sys.argv[1]), norm(sys.argv[2])
pins = [t.split(":") for t in open(sys.argv[3]).read().strip().split(",")]
pins = [(int(l), int(e)) for l, e in pins]
# budget held?
bad = {L: len(v) for L, v in new.items() if len(v) != 30}
assert not bad, f"BUDGET BROKEN: {bad}"
# every pin now KEPT (not in dropped)?
still_dropped = [(L, e) for L, e in pins if e in new.get(L, set())]
assert not still_dropped, f"PINS STILL DROPPED: {still_dropped}"
# what moved: per layer, experts that were dropped in gated but kept now = pins;
# experts kept in gated but dropped now = evicted survivors.
print(f"pins applied: {len(pins)} across layers {sorted(set(l for l,_ in pins))}")
n_into_keep = n_evicted = 0
for L in range(30):
    into_keep = gated[L] - new[L]   # dropped before, kept now
    evicted = new[L] - gated[L]     # kept before, dropped now
    if into_keep or evicted:
        n_into_keep += len(into_keep); n_evicted += len(evicted)
        print(f"  L{L}: +keep {sorted(into_keep)}  -evict {sorted(evicted)}")
print(f"TOTAL moved into keep: {n_into_keep}  evicted: {n_evicted}  "
      f"(must be equal & == #pins={len(pins)})")
assert n_into_keep == n_evicted == len(pins), "DELTA MISMATCH — not a clean single-variable change"
print("\nOK: clean single-variable delta. Drop-map ready:", sys.argv[1])
PY
echo
echo "[done] drop-map at $OUT"
