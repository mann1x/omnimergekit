# Replay a published prune

`scripts/replay_prune.sh` reproduces a previously-validated prune on any
host using cached Phase-1 importance + Phase-0 canary baseline. This is the
fast path to re-creating a published variant on a cloud pod (or migrating
production runs between hosts).

## What it skips

The recipe (`recipes/gemma4_31b/prune_local_heal.py`) has four heavy phases.
Replay reuses the two most expensive caches:

| Phase | Without cache | With cache (replay) |
|---|---|---|
| 0 — canary baseline (3 prompts × 50 gen tokens through unpruned model under accelerate offload) | ~17 min | **skipped** via `--canary-baseline-cache` |
| 1 — nf4 Michel α-grad per-layer per-head importance | ~30 min | **skipped** via `--imp-cache` |
| 0' — Phase 0 calib forward (capture `o_proj` outputs on calib tokens) | 30-60 min | still runs (activation tensors too big to cache) |
| 2 — ridge-regularized lstsq heal on kept-cols of `o_proj` (full 60 layers) | 3-7 h | still runs |

Net savings on 31B: ~45-50 min off a 5-9 h pipeline.

## Required cache artifacts

A single source directory with these files (all small — total ~45 KB):

| File | Size | Required? | Produced by |
|---|---|---|---|
| `gemma4_31b_imp_full_nf4.pt` | 26 KB | **yes** | Phase 1 of original run |
| `gemma4_31b_canary_baseline_n50.pt` | 5 KB | optional (recipe recaptures if missing) | Phase 0 of original run |
| `prune_manifest.json` | 14 KB | optional (used for head-selection cross-check) | written next to original output weights |

The script accepts both **local paths** and **`host:path` rsync syntax**, so
the source dir can live on solidPC, another pod, or anywhere reachable by
`rsync -e ssh`.

## Usage

```bash
# Replay he125 on a pod, uploading to ManniX-ITA/gemma-4-31b-he1-it:
bash scripts/replay_prune.sh \
    google/gemma-4-31B-it \
    ManniX-ITA/gemma-4-31b-he1-it \
    0.125 \
    solidpc:/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/4b_phase1 \
    "$(cat ~/.cache/huggingface/token)"

# Or from a local cache dir (useful if you rsync once and replay many times):
bash scripts/replay_prune.sh \
    google/gemma-4-31B-it \
    ManniX-ITA/gemma-4-31b-he1-it \
    0.125 \
    /workspace/he1_cache \
    "$TOKEN"

# Test path: keep the result local, skip the HF upload:
SKIP_UPLOAD=1 bash scripts/replay_prune.sh ... "$TOKEN"

# Override heal strategy or pass any other prune_local_heal flag:
bash scripts/replay_prune.sh ... "$TOKEN" --heal lstsq --ridge 5e-3
```

## Env knobs

| Var | Default | Effect |
|---|---|---|
| `WORKDIR` | `/workspace/replay_prune` | staging dir (caches + base + pruned output) |
| `RECIPE` | auto-detect | path to `prune_local_heal.py` (tries `/workspace/omnimergekit/...` then `/shared/dev/omnimergekit/...`) |
| `GPU_MEM` | `8GiB` | accelerate offload max for cuda:0 |
| `CPU_MEM` | `200GiB` | accelerate offload CPU budget |
| `CHUNK_TOKENS` | `384` | Phase 0 chunk size |
| `RIDGE` | `1e-2` | lstsq ridge term |
| `SKIP_UPLOAD` | unset | when `1`, keep output local, don't upload |

## Pipeline

1. **Fetch caches** (rsync or local cp) — ~30 s.
2. **Download base** from HF via `hf download --exclude '*.gguf' --exclude 'imatrix.dat'` — ~5-30 min depending on pod bandwidth.
3. **Run `prune_local_heal.py`** with `--imp-cache` (+ `--canary-baseline-cache` if present) and any extra args you pass after the token.
4. **Canary check** is run by the recipe itself; if it fails the output lands at `${OUT}.broken/` and the script exits 2 without uploading.
5. **Cross-check head selection** against the source `prune_manifest.json` (warning printed if it diverges — expected to be bit-identical).
6. **Upload** the BF16 safetensors + `prune_manifest.json` + tokenizer files to `<target_hf_id>` via `hf upload` (unless `SKIP_UPLOAD=1`).
7. **Cleanup** the staged base dir (keeps the cache dir and the pruned output for quantization).

## Why bit-identical replay matters

`prune_manifest.json` records the per-layer dropped-head indices. With
`--imp-cache` the importance tensor is identical, head ranking is identical
(stable `argsort`), and `--prune-frac` controls top-k cutoff — so the head
selection is fully deterministic and matches the source manifest entry-for-entry.

Phase 0' captured `o_proj` outputs and Phase 2 lstsq fit do depend on the
calibration corpus, GPU determinism settings, and floating-point reduction
order. With the same `scripts/calibration_datav5.txt` and CUDA/lstsq config,
the resulting weights are reproducible to within numerical noise; **a fresh
quant ladder built on the replayed model is identical for practical
purposes** to one built on the original.

## Origin

Introduced 2026-05-11 after the first cross-host replay of the 31B he125
recipe (solidPC original → pod 36480025 replay → `ManniX-ITA/gemma-4-31b-he1-it`
publish). Replaces the ad-hoc "rsync the .pt files and pass two flags to
the recipe" recipe with a single audited command.
