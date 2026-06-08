# Gemma 4 → 512k context extension recipe (T87)

**Status:** skeletons / pre-council-review. Do NOT launch any phase that
writes weights or burns >1h of compute without council green-light.
**Plan:** `backup_models/docs/plans/gemma4_512k_plan_v2.md`
**Council brief:** `backup_models/docs/plans/gemma4_512k_council_brief.md`

## Cascade

1. **31B-it** (dense, 10 full-attn layers) — first target. Validates
   YaRN-2.0 + LoRA recipe on a known-good non-MoE arch.
2. **98e-v6-coder-it** (MoE 98 experts × top-8, 5 full-attn layers) —
   second target. Cascade-gated on 31B-it Phase 2 outcome.

YaRN factor=2.0 on `rope_parameters.full_attention` only. LoRA r=16/α=32 on
Q/K/V/O of full-attn layers only. ~250-300M tokens PG-19 + math + RedPajama-V2.

## Files

| File | Type | Status | Purpose |
|------|------|--------|---------|
| `README.md` | doc | done | this file |
| `run_longctx_512k.sh` | orchestrator | skeleton | top-level cascade driver: phase0 → phase1 → phase2 per target, dry-run by default |
| `phase0_anchor.sh` | runner | skeleton | solidpc anchor measurements: 9-bench + RULER NIAH @{32k,64k,128k,256k} + MRCR-v2 @128k |
| `phase1_train.sh` | runner | skeleton | blackswan-2 training launcher: GPU 0 = trainer, GPU 1 = probe watcher |
| `phase1_train_yarn_lora.py` | training | skeleton | LoRA continued pretrain on YaRN-patched config, packed 256k seqs |
| `phase1_probe_watcher.py` | training-side | skeleton | watcher: on each new ckpt, merge LoRA → quick NIAH-256k probe; abort signal if 3× <80% |
| `phase2_eval.sh` | runner | skeleton | post-train eval: 9-bench@32k + RULER NIAH@{32k…512k} + MRCR@256k + (v6-coder) routing-entropy probe |
| `patch_yarn_config.py` | helper | concrete | patches a model's `config.json` to apply YaRN factor on `rope_parameters.full_attention` only |
| `pack_pg19_math_rpv2.py` | data | skeleton | packs PG-19 + proof-pile-2 + RedPajama-V2 into jsonl of 256k/512k chunks |
| `routing_entropy_probe.py` | post-train | skeleton | v6-coder-only: capture router logits at 32k/256k/512k positions, compute per-layer entropy + per-expert dominance |

## Templates required (under `omnimergekit/eval/templates/`)

TODO (pre-launch): add these. None exist yet.

- `ruler_niah_single_{32k,64k,128k,256k,512k}.yaml`
- `mrcr_v2_8needle_{128k,256k}.yaml`

These should use RULER from the `lm-evaluation-harness` upstream
(`Project-Numina/ruler` / NVIDIA RULER repository — TBD on which packaging
gives us a clean lm-eval task). Council may have opinions on which RULER
fork is canonical for 2026.

## Runbook (DO NOT execute until council approves)

```bash
# 1. Phase 0 anchors on solidpc — no blackswan-2 burn yet
bash phase0_anchor.sh --target 31b   # ~3-4h on solidpc 3090
bash phase0_anchor.sh --target v6c   # ~2-3h (some anchors cached)

# 2. Phase 1 training on blackswan-2 (31B first)
ssh linode-blackswan-2 'tmux new -d -s longctx_31b \
    "bash /srv/ml/scripts/phase1_train.sh --target 31b --run"'
# ~10-14h. GPU 0 trains; GPU 1 runs periodic NIAH probe.

# 3. Phase 2 eval (31B) — runs automatically after train
ssh linode-blackswan-2 'tail -f /srv/ml/runs/longctx_512k/31b/orchestrator.log'

# 4. Cascade gate — if 31B passes, repeat 2+3 for v6-coder.
#    If 31B fails, write up RCA, halt.
```

## Knobs / TODO markers

Every script has a clearly-marked `### COUNCIL` section at the top listing
the parameters the council should validate before launch. Do not silently
change these — they're the variables the council's "approve" is conditioned
on.

## Convention reminders

- All Python uses `/root/anaconda3/envs/omnimergekit/bin/python` on
  solidpc, `/srv/ml/envs/envs/omnimergekit/bin/python` on blackswan-2.
- All scripts dry-run by default; pass `--run` to execute.
- Logs land in `/srv/ml/runs/longctx_512k/<target>/` on blackswan-2,
  `backup_models/runs/longctx_512k/<target>/` on solidpc.
- Phase 0 results are the *canonical anchors* for the no-quality-loss gate
  — archive them under `eval_results_longctx_512k_anchors/` per
  [[feedback-eval-results-are-sacred]].

## After execution

- Update `gemma4_512k_plan_v2.md` "Status" header with outcome per phase.
- Append per-target outcome memory entries:
  `project_gemma4_31b_512k_outcome.md` and
  `project_gemma4_v6coder_512k_outcome.md`.
- Add a stack@3 entry to `STACK_HISTORY` if the training stack diverges
  from blackswan-2 baseline.
