# Qwen3.6-35B-A3B → ~26B expert-prune (v7-coder methodology)

Canonical home for the Qwen3.6-35B-A3B MoE expert-prune. **All scripts, logs, evals live here.**
Goal: 256e → **184e ≈ 26B** (28% expert drop), **keep vision, keep+slice native MTP head**, then eval
vs the 256e base with the same loop/quality gates used for v7-coder.

## Base model (Qwen/Qwen3.6-35B-A3B, 71.9 GB bf16)
- `qwen3_5_moe_text` / `Qwen3_5MoeForConditionalGeneration` — **multimodal** (vision tower kept).
- 40 layers, hidden 2048, **hybrid LINEAR attention** (`linear_attn.*`, Mamba/GatedDeltaNet — near-zero KV).
- MoE: **256 routed experts, top-8**, `moe_intermediate_size 512` (fine-grained), **+ shared expert** (always-on) + `shared_expert_gate`.
- **Native MTP head** (`mtp_num_hidden_layers 1`): full MoE decoder layer `mtp.layers.0.*` (own 256 experts) + `mtp.fc` + norms.
- Packed expert tensors: `experts.gate_up_proj [256,…]`, `experts.down_proj [256,…]`; router `mlp.gate.weight [256,2048]` (no per_expert_scale).

## Pruning ladder
routed experts = 32.2 B = 92% of text params; fixed floor ≈ 2.8 B (attn + shared experts + embed/head + router + ~1 B vision).
`total(N/layer) ≈ 2.8 + 0.126·N GB` → **26 B ⇒ keep 184 experts/layer (drop 72, 28%)**. Active stays ~3 B (top-8, N≫8).
`moe_intermediate_size 512 % 32 = 0` ⇒ no Q4_K/Q8_0 GGUF F16-fallback gotcha.

## Why favorable vs Gemma-4/v7-coder
- **Same packed layout** — `expert_drop.py` already slices `experts.gate_up_proj`/`down_proj` + router; adaptation is only `.mlp.` prefix + `gate.weight` router name + slice the MTP layer.
- **256 fine-grained experts** (2× Gemma) + **shared expert** (capability floor) ⇒ more redundancy, gentler drop.
- Linear attn ⇒ expert drop is orthogonal to the attention arch (the MicroCoder-HE hybrid-heal failure does NOT apply — we don't touch attention).

## Phased plan (all artifacts under this dir)
- **P0 setup** — this dir + STATE. Weights downloading to `/srv/ml/models/Qwen3.6-35B-A3B`.
- **P1 tooling** — `expert_drop_qwen35b.py` (adapt expert_drop.py: `.mlp.experts`, `mlp.gate.weight`, slice `mtp.layers.0.mlp.experts`, config 256→184, keep shared/linear_attn/vision/mtp.fc). Competence profiler (adapt `gate_competence_map.py`/`expert_neuron_analysis_v5`: hook `mlp.gate` router logits + packed-expert contribution on calib corpus → per-layer drop map, bottom 72).
- **P2 drop** — run competence → drop map → `expert_drop_qwen35b` → 184e model (vision + MTP retained/sliced). Router recovery: renorm → optional shared-α / EAC (shared-expert-aware — new).
- **P3 eval** — GGUF F16 + Q6_K/Q5_K_M (MTP retained, `--spec-type draft-mtp`) → omk_eval suite (GPQA/HE+/MBPP/LCB + loop/rumination canary) vs 256e base. Logs+evals under this dir.

## Open decisions / risks (empirical)
- 28% is aggressive (v7-coder's 23% flirted with IFEval rumination) — run a drop-ladder + loop canary; 256-expert granularity may be more forgiving.
- Router recovery on a **shared-expert** MoE is new to our toolbox (shared expert changes routed-importance baseline).
- MTP-head expert slicing vs retain-256 — start by slicing to 184 for config consistency; measure draft acceptance on the pruned model.
- Both bs2 GPUs currently busy (v7-coder SWE-bench GPU0, `.ape` GPU1) → P2/P3 GPU work waits for a free GPU; P1 tooling is CPU-side.

## P1 status — drop mechanism VALIDATED (2026-07-12)
`expert_drop_qwen35b.py` written + dry-run validated against the real 256e weights
(`/srv/ml/envs/envs/omnimergekit/bin/python`, `--dry-run`, no write):
- Sliced correctly: **80 expert tensors** (40 layers × gate_up+down) + **40 routers** (`mlp.gate.weight`)
  + **MTP head** (2 expert + 1 router). 256→184 on every one; shapes exact
  (`experts.gate_up_proj (256,1024,2048)→(184,…)`, `down_proj (256,2048,512)→(184,…)`, `gate (256,2048)→(184,2048)`).
- Passed through untouched: **922 tensors** (vision tower, shared_expert, shared_expert_gate, linear_attn,
  mtp.fc/norms, embed/lm_head, norms).
- Projected total: **26.66 B params** (from 35 B; ~8.3 B removed) — on target.
- Drop-map format: `{"0":[ids],…,"39":[ids],"mtp":[ids]}`; MTP defaults to layer-0 set if `"mtp"` absent.
- Placeholder map used for the smoke = drop experts 184–255/layer; REAL map comes from P1 competence profiler (GPU-gated).

**Remaining P1:** competence profiler (adapt `gate_competence_map.py` — hook `mlp.gate` router logits +
packed-expert contribution on calib corpus → rank experts, bottom-72/layer → real drop map). GPU-gated
(waits for GPU0 free after v7-coder SWE-bench). The drop→model step is CPU/IO once the map exists.

### P1 bridge also validated — `make_drop_map.py` (competence map → drop map)
`make_drop_map.py` ranks experts/layer by importance (`--score tc|wnorm|wnorm_tc`,
`--agg sum|max|mean`) aggregated across categories, drops the lowest N/layer, emits the
`{"0":[…],…,"mtp":[…]}` map. `--mtp-strategy global|layer0|none` (default global = bottom-72
by importance summed over all 40 layers). Full bridge smoke-tested on CPU: synthetic
competence map (40L×256E, 3 cats) → make_drop_map (drop 72, keep 184) →
expert_drop_qwen35b `--dry-run` → 26.66 B, all shapes exact. **Pipeline now:**
`competence producer` ✅ → competence-map JSON → `make_drop_map.py` ✅ →
`expert_drop_qwen35b.py` ✅ → 184e model.

### P1 COMPLETE — competence producer generalized (2026-07-13)
Rather than fork a Qwen-only profiler, the canonical Gemma producer
**`gemma4/neuron_analysis/expert_neuron_analysis_v5_targeted.py` was made arch-generic**.
The transformers-5.5 fused-experts refactor gives Gemma 4 and Qwen3.5-MoE the *same*
`Experts.forward(hidden_states, top_k_index, top_k_weights)`, so the existing per-expert
hook (tc / wnorm / neuron_act) works for both — the only deltas are the attach point
(`layer.experts` for Gemma vs `layer.mlp.experts` for Qwen, auto-detected), `mm_token_type_ids`
(passed only when the model's forward accepts it), config via `get_text_config()`, and a
`--model` / `--device cuda:N` full-GPU load. New flags (Gemma tier-a/tier-b path byte-unchanged):
- `--model <dir>` — profile any packed-MoE arch; `--device cuda:0` loads the whole bf16 model
  on a big-VRAM GPU (Blackwell 96 GB fits 67 GB, no CPU spill).
- `--corpus <jsonl>` — **Tier-C** routing-frequency mode (no pass-traces, no generation);
  makes `--variant`/`--tier-b-json` optional; emits `corpus_<cat>` categories in the same map.
- `--probe` — attach hooks + one forward, verify tc accumulates, exit.

**Probe-verified on Qwen (2026-07-13, bs2 GPU0):** loads in 11 s, 40 expert-hooks attached,
`4160 = 13 tok × 8 top_k × 40 layers` top-k selections (exact), 80/256 experts used in L0.

**Run (P2):**
```
python gemma4/neuron_analysis/expert_neuron_analysis_v5_targeted.py \
    --model /srv/ml/models/Qwen3.6-35B-A3B --device cuda:0 \
    --corpus <qwen-calib>.jsonl --corpus-cat-field bench \
    --out recipes/qwen3_6_35b_a3b_prune/results/competence_qwen35b.json
```
then `make_drop_map.py --competence-map … --drop-count 72 --score tc` → `expert_drop_qwen35b.py`.

⚠️ **Corpus caveat:** `scripts/router_calib_corpus_9bench_balanced.jsonl` is **Gemma-templated**
(`<bos><|turn>…`), so it's not directly right for Qwen routing. For P2 build a Qwen-templated
calib corpus (or feed raw domain text with `--corpus-apply-template` off, accepting minor
template-token noise for a first drop-ladder). Producer + bridge + dropper are all ready + tested.
