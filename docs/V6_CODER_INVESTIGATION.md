# v6-coder regression — investigation & recipe history

**Status:** investigation complete 2026-05-23. v6-coder regressed on code vs v5-coder
(HE −2.44pp, HE+ −0.61pp) while gaining on math (AIME +10pp). Root cause is in the
recipe (selection), not the data (v3.json is verified clean per producer council
review). New recipe C11 staged for evaluation.

## History — recipe lineage

The v5-coder → v6-coder lineage chained 11 named recipes (C1..C11) over 2026-05-15
to 2026-05-23. Each tried a different mechanism to bias expert selection toward
code-axis classes while not destroying the v4 generalist backbone.

| recipe | base data | key knobs | observed | notes |
|--------|-----------|-----------|----------|-------|
| C1 (max_codetb) | v5_code | class_weights 3× code, strategy=max | LCB pass@1 dropped; 26× longer responses, 11/12 LCB syntax fails | 503/2940 swaps vs v4; lost mid-rank generalists ("output-discipline" carriers) |
| C2 (v4floor80) | v5_code | + v4-floor protect rank ≥80 | similar to C1; 255 swaps; tokens still explode | swap zone v4_rank 29–47 = right above floor cliff |
| C4 (v4floor90) | v5_code | floor=90, 8 swap slots/layer | improved on C1/C2 but still below v4 | |
| C5 (v4floor95) | v5_code | floor=95 | further constrained | superseded by C6 |
| **C6 (v4floor perlayer)** | v5_code_fixed | per-layer v4 floor (75-95, mean 83), breadth-bonus 0.5, normalize=rank, strategy=max, class_weights [1,1,3,1,1,2,2,2] | **v5-coder ships this** | HE 99.39%, HE+ 93.29%, AIME 53.33% |
| C6 + α=1.2 shared upweight | v5_code_fixed | + mlp.down_proj.weight ×1.2 | v5-coder published recipe | router KD step |
| C7 (v4-class-min) | v5_code | generalist-protected swap | not adopted | |
| C8 (layer-profile) / C9 / C10 (gini) | v5_code variants | per-layer profiling, monotonic descent, gini-based floor | exploratory | |
| **C6 / v3map (= v6-coder)** | v5_code_v3 | **same C6 recipe on v3 data** | HE 96.95%, HE+ 92.68%, AIME 63.33% | code regression, math gain |
| **C11 (proposed)** | v5_code_v3 | protect_top=22, breadth=0.3, class_weights [1,1,3,1,1,**5,5,5**], protect_class_min code:10 | **staged for build** | 241/2940 swaps vs v5-coder (8.2%) |

## Investigation — what we found

### Producer data files

| file | sane | NaN | huge (>1e9) | inf | notes |
|---|---:|---:|---:|---:|---|
| `expert_neuron_v5_code_fixed.json` | 39.6% | **60.4%** | 0% | 11 | sparse, NaN-riddled. v5-coder was built on this. |
| `expert_neuron_v5_code_v3.json` | **98.5%** | 0% | 1.5% | 0 | council-verified rebuild. v6-coder built on this. |

**`fixed.json` was the broken file** (60% NaN cells). v5-coder still produced strong
results because `v4_layer_floor_map.json` (per-layer pre-computed floor) carries
~80% of the kept-expert backbone regardless of live class signal — NaN entries
fell through under `--normalize rank`.

**`v3.json` is the correct rebuilt data.** Council-verified after re-execution of
tier A and tier B from scratch. The 1.5% "huge" values (1e9 to 4e18) are legitimate
outliers on highly-specialized experts (e.g. `targeted_humaneval` cells with `tc=67`
co-occurring with `wnorm=8.4e17`) — under `--normalize rank` they collapse to rank-1
and don't drive ranking through sheer magnitude.

### Why C6 on richer data underperforms

The C6 recipe was tuned against a **sparse NaN-riddled fixed.json**. When applied to
**dense v3.json** the same knobs select differently:

1. **`--strategy max` over 8 class ranks** — fixed.json had 1–2 valid ranks per
   expert (rest NaN). v3.json has all 8 valid. Single-class specialists now win the
   max more aggressively against multi-class generalists. The "output-discipline"
   generalists v4-coder kept get demoted.

2. **`--breadth-bonus 0.5`** — calibrated against fixed.json where most experts had
   1–2 non-NaN classes. On v3.json every expert has 8 valid ranks, so breadth no
   longer discriminates; the bonus contributes a constant offset.

3. **Rank-17 to rank-30 marginal zone churn** — top-16 (`--protect-top 16`) and
   v4-floor (rank ≥75) are stable across both maps. The 7.6% swaps (223 experts)
   happen in the marginal zone (rank 17–74) where the recipe selects differently.

**The TOP-scoring targeted experts in early layers (blocks 0–10) are all retained
by both maps.** v6-coder did NOT miss those. The losses are concentrated in
rank 17–30 LCB-medium specialists with scores in the 5k–52k range, and in
generic-math/science experts that v6-coder happened to pick up because their v3 rank
was tied or marginally higher than dropped code experts.

### Empirical signal

Of 223 swaps (v5-coder kept, v6-coder dropped):

| best-class on v3 | v6-coder dropped (LOST) | v6-coder kept (GAINED) |
|---|---:|---:|
| `none` (all-zero in v3) | 43.5% | 53.4% |
| generic_math | 27.4% | 17.5% |
| generic_science | 9.0% | 9.4% |
| generic_creative | 5.8% | 4.5% |
| generic_logic | 4.9% | 2.7% |
| **generic_code** | 4.5% | 7.2% |
| **targeted_lcb_medium_55** | 3.6% | 3.1% |
| **targeted_humaneval** | 0.9% | 1.8% |
| **targeted_humanevalplus** | 0.4% | 0.4% |

By count, v6-coder net-gained code-axis (+7 swaps). But the TOP-7 lost experts were
all `targeted_lcb_medium_55` with scores 5k–52k — high enough to dominate per-block
code competence. Dropping them stung disproportionately even though math/none
losses outnumbered code losses.

## New recipe — C11

Built and staged at
`scripts/v6coder_C11_targeted5x_p22_drop_map.json` for canary build.

```bash
python3 scripts/generate_drop_map_v5.py \
  --data scripts/expert_neuron_v5_code_v3.json \
  --target 98 \
  --protect-top 22 \
  --alpha 2.0 \
  --strategy max --normalize rank \
  --class-weights 1 1 3 1 1 5 5 5 \
  --protect-class-min code:10 \
  --v4-floor-map scripts/v4_layer_floor_map.json \
  --breadth-bonus 0.3 \
  --baseline-drop-map scripts/v5coder_C6_v4floor_perlayer_breadth50_drop_map.json \
  --output scripts/v6coder_C11_targeted5x_p22_drop_map.json
```

**Changes vs C6:**

| knob | C6 | C11 | rationale |
|---|---|---|---|
| `--protect-top` | 16 | **22** | adds 6 unconditional slots/layer, captures rank-17 to rank-22 targeted experts |
| `--class-weights` (targeted ×3) | `[1,1,3,1,1,2,2,2]` | `[1,1,3,1,1,5,5,5]` | targeted classes from 2× to **5×** — pulls high-targeted-score experts up the aggregate ranking |
| `--protect-class-min` | (unset) | `code:10` | guarantees ≥10 code-class (v4-coded) experts per layer |
| `--breadth-bonus` | 0.5 | **0.3** | reduces generalist reward → specialists win more |

C11 vs v5-coder: 241 swaps (8.2%) — comparable scale to v6-coder's 7.6% but
shifted dramatically toward targeted-axis.

### Net keep-count change vs v5-coder

| class | v5-coder kept | C11 kept | Δ |
|---|---:|---:|---:|
| generic_math | 12 | 13 | +1 |
| generic_logic | 13 | 14 | +1 |
| **generic_code** | 47 | 41 | **−6** |
| generic_science | 95 | 88 | −7 |
| generic_creative | 112 | 103 | −9 |
| **targeted_humaneval** | 481 | 507 | **+26** |
| **targeted_humanevalplus** | 242 | 278 | **+36** |
| **targeted_lcb_medium_55** | 1938 | 1896 | **−42** |

**Tradeoff in C11**: gains 62 HE/HE+ experts, loses 42 lcb_medium_55 + 6 generic_code.
Net code-axis +14 (2722 vs 2708). Net targeted-axis +20 (2681 vs 2661). LCB-medium
count drops, but the LCB experts that C11 keeps are higher-scored on average
(C11 selected the top-quality specialists; v5-coder kept more mid-quality lcb).

**Risk assessment**: C11 should improve HE/HE+ vs v6-coder by 1-3pp. LCB outcome
ambiguous (count down, quality up). Math/science marginally weaker. AIME unclear.

### Per-layer swap distribution

Layers 0–8 are nearly stable (1–4 swaps each — the early-layer experts the user
emphasized are essentially intact). Bulk of churn lands in mid-late layers (9–29
average 9–14 swaps/layer):

```
blk. 0:  1  █
blk. 1:  1  █
blk. 2-8: 2-4 each  (early layers preserved)
blk. 9: 12  ████████████
blk.15: 14  ██████████████  ← peak churn
blk.29: 13  █████████████
```

## Way forward — recipes to sweep

Build all 5 maps in parallel (~3 min total), build GGUFs in sequence, smoke each on
HE+ / LCB-30 / AIME-30 (~30 min × 5 = 2.5 h on solidpc 3090).

| variant | protect_top | class_weights | protect_class_min | breadth_bonus | hypothesis |
|---|---|---|---|---|---|
| **C11** (staged) | 22 | `1 1 3 1 1 5 5 5` | code:10 | 0.3 | targeted strong + class floor |
| C12 (light) | 20 | `1 1 3 1 1 4 4 4` | code:8 | 0.4 | moderate variant |
| C13 (heavy) | 24 | `1 1 4 1 1 6 6 6` | code:12 | 0.2 | aggressive targeted |
| C14 (mean-norm) | 20 | `1 1 3 1 1 4 4 4` | code:10 | 0.3 | **`--normalize mean`** vs rank — addresses the "rank-max on dense data" failure pattern documented in memory 2026-05-16 |
| C15 (lcb-only) | 22 | `1 1 3 1 1 1 1 8` | code:10 | 0.3 | super-bias LCB to recover the −42 count |

**Decision gate**: variant with HE-20 ≥ v5-coder's HE-20 baseline AND LCB-30
within 2pp of v5-coder ships as v6-coder-v2.

## Mandatory guard rails for future recipes

Drop into `scripts/generate_drop_map_v5.py` data loader:

```python
# Reject pathologically-large input scores (producer fp32→bf16 cast bugs)
max_seen = 0.0
for cls in data["categories"]:
    for li in data["categories"][cls]:
        for e in data["categories"][cls][li]:
            v = abs(e.get("wnorm", 0) * alpha + e.get("tc", 0))
            if math.isnan(v) or math.isinf(v):
                continue  # rank-normalize handles NaN; flag inf elsewhere
            max_seen = max(max_seen, v)
if max_seen > 1e20:
    sys.exit(f"FATAL: pathological scores detected (max={max_seen:.2e}). "
             f"This may indicate producer fp32 hot-path corruption. "
             f"If verified intentional, pass --allow-extreme-scores.")
```

Also: when running a recipe, log overlap-with-baseline at end of generation
(already done in C6/C11) but ALSO log per-class keep-count delta vs baseline —
this is the most informative diagnostic when the score doesn't match expectation.

## References

- `expert_neuron_v5_code_v3.json` — current canonical Tier-A/B map (council-verified)
- `expert_neuron_v5_code_fixed.json` — legacy, 60% NaN, kept only for v5-coder
  build reproduction
- `v4_layer_floor_map.json` — per-layer v4-baseline floor (rank ≥75–95)
- `teacher_force_98e_p16_clean.json` — v3 baseline (single-class TF pooled)
- `scripts/generate_drop_map_v5.py` — multi-class C-family generator
- `memory/reference_moe_router_recovery_methods.md` — rumination-fix ladder
- Task IDs: T17.3 (initial v5 mapping), T19 (sweep), T22 (C2 with floor), T25 (C6),
  T33 (v5 publish), T77b (v6-coder rebuild), this investigation 2026-05-23.
