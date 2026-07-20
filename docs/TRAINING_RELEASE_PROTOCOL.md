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
