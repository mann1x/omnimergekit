# Training Run Release Protocol

Every **real training run** (NOT smoke runs) MUST be published as a GitHub **release
artifact** — a complete, reproducible, auditable record captured at launch and
finalized on completion. Applies to `omnimergekit` and downstream project repos
(e.g. `an-finetune`); publish to the repo that owns the run's tooling.

## Principle

A release is the single source of truth for reproducing and auditing a run: recipe +
exact data + full toolchain + environment + smoke evidence at launch; final metrics +
eval + weight pointers at completion. Same "archive before it can be lost" spirit as
imatrix / eval-result preservation, extended to the whole training run.

## Lifecycle — LIVING release (one tag per run)

### At launch (at/just before training start)

Create a release tagged `<variant>-<YYYYMMDD>` (private repos OK). The release
**description** mirrors the doc summary. Attach the following, **COMPRESSED**:

1. **`DOC_<variant>.txt`** — full run documentation:
   - **Recipe**: all hyperparameters — LoRA (rank / alpha / target modules), epochs,
     sequence length, response-masking on/off, optimizer / lr / schedule / warmup /
     weight-decay, and any architecture-specific unfreeze (e.g. Gemma-4 PLE modules).
   - **Datasets**: each source, row counts, mix weights, format.
   - **Script paths** (repo-relative or absolute): dataset creation, curation,
     validation, sanitization; runner; training script; docker image + tag.
   - **Environment**: host + GPU (model, VRAM), conda/venv, key library versions.
   - **Smoke runs**: every smoke and its result.
2. **`datasets.tar.gz`** — the FULL curated data actually trained on (every source,
   exact bytes) **plus the mix config**. Not a manifest.
3. **`scripts.tar.gz`** — every script used (creation / curation / validation /
   sanitization, runner, training script, docker build files).
4. **`smokes.txt`** — all smoke runs + results. (Smoke runs get no release of their
   own; their evidence lives inside the real run's release.)

### On completion

Update the **same** release/tag and append:

- **`final_metrics.txt`** — final training metrics (loss, steps, wall-clock).
- **`eval_results.tar.gz`** — eval summaries (e.g. crucible / a2a-t / netconfig /
  `omk_eval` `summary.json`).
- **`checkpoint_manifest.txt`** — sha256 + path + **HF repo** of each checkpoint /
  merged model / GGUF.

## Model-weights policy

- **Full/merged weights and GGUF do NOT go in the release** — they go to
  **HuggingFace**. Record the HF repo id + sha256 in `checkpoint_manifest.txt`.
- **LoRA adapters DO go in the release** (small; compressed).
- The release is for **small** artifacts (scripts, datasets, adapters, docs). GitHub
  caps release assets at **2 GB/file**; if a small artifact legitimately exceeds 2 GB,
  split into `<2 GB` parts. Big artifacts (weights / GGUF) go to HF — never split into
  a release.

## Canonical builder — `scripts/build_training_release.sh`

Do NOT hand-assemble releases per run (that is how asset layout drifts). The
`an-finetune` repo ships one parametrized builder that produces the exact asset
set above from a per-variant fact block:

```
scripts/build_training_release.sh <variant>            # dry-run: stage + gitleaks, no publish
scripts/build_training_release.sh <variant> --publish  # gh release create (or upload --clobber if the tag exists)
```

- **Per-variant case block** (in the script) pins: `TAG` (`<variant>-<YYYYMMDD>`),
  `TITLE`, the bs2 `out/` dir, adapter subdir (`lora` for SFT, `adapter` for
  ORPO/GRPO), F16 GGUF path, train log, curated-data dir, and `KIND`
  (`sft`/`orpo`/`grpo`) which selects how `datasets.tar.gz` is built.
- **Pulls from bs2** (compute host): LoRA adapter, per-model eval summaries
  (`eval/v9_gate` + `omk/**/<v>/summary.json`), final train metrics, and the
  sha256 manifest (weights are hashed remotely — never transferred; they stay on
  bs2 / go to HF, never into the release).
- **Pulls from this repo**: the toolchain snapshot (`scripts.tar.gz` = code-only
  `docker/ simpo/ eval/ netconfig/ a2areason/`), the mix/config, and the two
  hand-written prose files it REQUIRES per run:
  `docs/releases/BODY_<v>.md` (the release notes / markdown body) and
  `docs/releases/DOC_<v>.txt` (the full run documentation asset).
- **gitleaks-scans** the staged assets before any publish; aborts on a finding.
- Dry-run copies staged assets to `docs/releases/staged_<v>/` for inspection.

The mechanical 80% (tarballs, metrics, manifest, scan, `gh release create`) is
identical across runs; only `BODY_<v>.md` + `DOC_<v>.txt` are authored per run.

## Constraints

- **Secrets**: never archive a file containing a token/key. The gitleaks gate applies;
  treat private repos as fully auditable. See [`SECURITY.md`](SECURITY.md).
- **Authorship**: follow each repo's commit-author convention.

## Launch checklist

- [ ] release tagged `<variant>-<YYYYMMDD>`
- [ ] `DOC_<variant>.txt`: recipe + datasets + script paths + env (host/GPU) + smokes
- [ ] `datasets.tar.gz` (full curated data + mix config)
- [ ] `scripts.tar.gz` (all pipeline scripts)
- [ ] `smokes.txt`
- [ ] LoRA adapter attached; weights/GGUF → HF (referenced by `checkpoint_manifest.txt`)
- [ ] gitleaks clean
