# tool-eval-bench cohort runner

Runs [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) hardmode
across a set of GGUF models on one GPU, with a fixed basis, and aggregates the
results with the comparability guards the raw reports do not give you.

## Why this exists

`tool-eval-bench` reports a **Total Points** score (hardmode = 88 scenarios × 2 pts
= 176 max). Comparing that number across models is only valid when every cell shares
a basis. These scripts pin the basis, record it, and refuse to pool cells that do not
share it.

## Layout

| file | role |
|---|---|
| `run_cohort.sh` | the driver: boots a server per (seed, model), runs the bench, tears down |
| `fetch_models.sh` | byte-size-verified download of the external GGUFs |
| `summarize_cohort.py` | aggregation + 95% CI, with denominator/balance guards |
| `plot_cohort.py` | dot-and-whisker PNG (shares the loader, so it shares the guards) |

## Usage

```bash
export OMK_TB_BIN=/path/to/opencoti-llamafile-0.10.5-c7-x86_64.llamafile
export OMK_TB_MODELS=/path/to/gguf/dir
export OMK_TB_OUT=/path/to/results/dir

bash eval/toolbench/fetch_models.sh
bash eval/toolbench/run_cohort.sh
python eval/toolbench/summarize_cohort.py
python eval/toolbench/plot_cohort.py -o toolbench_scores.png
```

The chart plots only the balanced seed set, draws 95% CI whiskers once n>=2, marks
any cell graded on <176 with `*`, and shows the published r/LocalLLaMA values as a
grey tick — as a *separate, labelled* series, never merged into ours, because those
were measured on a different basis (256 K ctx, harness v2.6.0, no MTP, other quants).

`plot_cohort.py` needs matplotlib (`requirements-eval.txt`). Install it with
numpy/pillow constrained — an unconstrained resolve can pull a newer numpy that
breaks the pinned `torch 2.10.0+cu128` build.

Overridable: `OMK_TB_SEEDS` (default `42 43 44 45 46`), `OMK_TB_CTX` (65536),
`OMK_TB_PRESSURE` (0.25), `OMK_TB_PORT` (8265).

## The basis (change any of these and every prior number is rebased)

- harness `--hardmode --weight-by-difficulty --backend llamacpp`
- `--context-size 65536 --context-pressure 0.25` → ~10 K filler tokens
- server: temp 0.6 / top-p 0.95 / top-k 20 / min-p 0 / presence 0 / repeat 1,
  `-ngl 99 --ubatch-size 2048 --fit-target 256 -ctk q8_0 -ctv q8_0 --parallel 1`
- seeds are **paired** across models, so comparisons are paired
- **pin the harness commit.** Between v2.6.0 and v2.6.0-45 there are ~45 commits that
  are almost entirely scorer fixes crediting previously-penalised behaviour. A newer
  scorer returns systematically higher scores and **voids every earlier cell**.

## Traps this tooling guards (each cost real time to find)

1. **Mixed denominators — set `--timeout` high enough or the metric silently changes.**
   A scenario lost to a request timeout is *dropped from scoring, not zeroed*, so that
   cell is graded on `<176` and its Total Points is not comparable to anything. The
   default `--timeout 120` is **too low for a dense model without an MTP head**:
   measured 2026-09-06, `qwen3.6-27b` (dense 27B, ~22.8 t/s decode) needs ~131 s for a
   3 k-token answer, reached **335.5 s** on its slowest turn, and had **8 turns over
   120 s** — losing 1 scenario at one seed and 2 at the next (174, then 172).
   No other model in the cohort exceeded 120 s more than once (max 127.9 s), so raising
   the timeout is a **no-op for them** and does not rebase their cells. This runner
   therefore defaults to `--timeout 600` (`OMK_TB_TIMEOUT`).
   The exclusion is **not random** — it lands on the slowest models — and it also leaks
   infrastructure variance into that row's CI. `summarize_cohort.py` still flags any cell
   graded on `<176`; treat a flagged cell as a defect to re-run, not a number to caveat.
2. **Speculation is a property of the file, not the repo tag or the name.**
   `bartowski/Ornith-1.5-35B-A3B-IQ4_XS` carries a NextN head despite no `mtp` tag. Worse,
   **unsloth ships Qwen3.6-27B in two repos under the SAME filename** — only
   `unsloth/Qwen3.6-27B-MTP-GGUF` has the head (`Q4_K_M`, +289 MB, 866 tensors vs 851). Pick
   the wrong one and the cell just runs slower, with nothing in the output saying its serve
   geometry differed. The driver reads `nextn`/`mtp` tensors out of the GGUF header and never
   assumes; save the two builds under **distinct local filenames**.

   Pass `--require-mtp` (opt-in, default OFF) to make a cell with no speculation an ERROR
   rather than a silent downgrade. Default is off so a legitimately head-less model can still
   be benchmarked.

### Speculation: self-MTP vs an external drafter

Model entries are `name|gguf[|drafter_gguf]`:

| form | serves as | for |
|---|---|---|
| `name\|model.gguf` + NextN head in file | `--spec-type draft-mtp` | Qwen3.5/3.6/3.8 family |
| `name\|model.gguf\|drafter.gguf` | `--spec-type draft-assistant --mtp-head <drafter>` | **Gemma-4** (`google/gemma-4-26B-A4B-it-assistant`) |
| `name\|model.gguf`, no head | no speculation (or abort under `--require-mtp`) | — |

Gemma-4's drafter is a *separate* model loaded **into** the target (EAGLE-style: it shares
the target's KV cache and reads its residual stream), which is why it needs `--mtp-head`
rather than `-md/--spec-draft-model`. Draft depth is `OMK_TB_DRAFT_N` (default 3).
A named drafter that is missing on disk is a loud failure, never a silent fallback.

Speculative decoding is **distribution-preserving**, so it changes speed, not scores — but
a cohort should still hold it constant, because it changes the serve geometry the numbers
were produced under.
3. **`cmd | grep -q PAT` under `set -o pipefail` returns non-zero ON MATCH** — `grep -q`
   exits at the first hit and SIGPIPEs the producer (141). A readiness gate written that
   way fails while the server is healthy, and silently skips every model. Capture first,
   then match a herestring.
4. **Runtime scales with ACTIVE params, not the name.** A dense 27B costs ~2.7× an
   A3B (3 B active) per rep — 46 min vs 17 min — even though both are called "27B".
   Classify by counting `*_exps` tensors before estimating wall clock.
5. **Never price a run from its first N scenarios.** Hardmode is ordered easy→hard;
   the first ~26 gave 11 min/rep against a true 17.6 min/rep (+60%).
6. **Prefill dominates.** The context-pressure filler is built once per run but is
   deliberately shuffled and nonce-prefixed *to defeat prefix caching*. At 128 K/0.5,
   ttft was ~90% of each scenario. `--cache-reuse` would recover it but works by KV
   shifting, which rewrites position encodings — **not usable on a scored run**.

## Seed-major ordering

The driver iterates seeds on the outside and models on the inside, so a **balanced
cohort exists at every seed pass**. An interrupted run yields a complete table at
n=k rather than some models finished and others untouched. The cost is one extra
server boot per (seed, model) — ~10 s each.

`summarize_cohort.py` intersects the seed sets across models and reports only the
balanced subset, listing any ragged extra cells separately rather than pooling them.
