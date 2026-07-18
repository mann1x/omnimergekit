# MTP / speculative-decoding bench + sweeps

Tooling to measure **MTP (Multi-Token-Prediction) speculative-decoding** throughput and
draft-acceptance for Gemma-4 (and other `draft-assistant` / `draft-mtp`) models served by
an **opencoti llamafile** build. This is the harness that answers "did MTP engage, what's
the acceptance, and what `--spec-draft-n-max` / drafter-quant should we serve with?"

> Requires an **opencoti llamafile** build (`--spec-type draft-assistant` / `draft-mtp`).
> Upstream Mozilla llamafile and stock llama.cpp do **not** have the Gemma-4 MTP path.

## Files

| file | what |
|---|---|
| `mtp-bench.py`          | the bench engine — 9 fixed prompts (8 coding/topic categories + a long code-review), each sent to `/v1/chat/completions`; reads the `timings` block and reports per-prompt + **aggregate** `accept_rate` and `tok/s`. Based on am17an's llama.cpp mtp-bench (PR #23398). |
| `mtp_nmax_sweep.sh`     | sweep `--spec-draft-n-max` (draft tokens per round) over a fixed target+drafter. Finds the tps-optimal n. |
| `accept_quant_sweep.sh` | sweep the **drafter GGUF quant** (F16/Q8/Q5/Q4…) over a fixed target+n_max. Bounds "does drafter quant move acceptance / is it worth imatrix-ing the drafter?" |

All three are **env-driven, nothing host-hardcoded** (public-repo safe). One server at a
time, reaped by `PORT` (`fuser -k`).

## The spec-decoding knobs (opencoti llamafile)

| flag | meaning |
|---|---|
| `--spec-type draft-assistant` | Gemma-4 external-assistant MTP (drafter is a separate `gemma4_assistant` GGUF). |
| `--spec-type draft-mtp`       | Qwen-style NextN self-spec (drafter layers baked into the target GGUF; no separate file). |
| `--mtp-head <assistant.gguf>` | the drafter file, loaded into the target (Gemma `draft-assistant` path). |
| `-ngld / --spec-draft-ngl N`  | drafter GPU layers (use 99). |
| `--spec-draft-n-max N`        | **draft tokens per round (default 3).** The knob `mtp_nmax_sweep.sh` sweeps. |
| `--draft-block-size N`        | MTP draft block size B (drafts B-1/round); alternative to n-max on some paths. |

Boot log confirms MTP engaged:
```
load_model: MTP assistant '…assistant.Q8_0.gguf' loaded into target
MTP assistant dual-context draft created (ctx_other=target, shared KV)
common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'  n_max=…
```
Per-request acceptance shows in the server log as
`draft acceptance = 0.xxxx ( accepted / generated )`.

## Run it (bs2 example)

```bash
BIN=/srv/ml/opencoti-llamafile/llamafile \
TGT=/srv/ml/an-finetune/gguf/e2b-an-synth-v0-1ep-F16.gguf \
DFT=/srv/ml/models/nextn/drafters-adapted/E2B/gemma-4-E2B-it-assistant.Q8_0.gguf \
BENCH=/shared/dev/omnimergekit/eval/mtp/mtp-bench.py \
PY=/root/anaconda3/envs/omnimergekit/bin/python \
GPU=1 PORT=8263 NMAXES="1 2 3 4 5" OUT=./mtp_nmax_out \
  bash mtp_nmax_sweep.sh
cat ./mtp_nmax_out/run.log
```

Drafter-quant sweep (needs `$DDIR/<DRAFTER_STEM>.<Q>.gguf` files):
```bash
BIN=… TGT=… BENCH=… PY=… GPU=1 \
DDIR=/srv/ml/models/nextn/drafters-adapted/E2B DRAFTER_STEM=gemma-4-E2B-it-assistant \
QUANTS="F16 Q8_0 Q5_K_M Q4_K_M" NMAX=2 OUT=./mtp_quant_out \
  bash accept_quant_sweep.sh
```

Single point (no sweep): boot the server yourself, then
`python mtp-bench.py --url http://127.0.0.1:PORT --out run.json`.
Diff two runs: `python mtp-bench.py --diff a.json b.json`.

Gemma-4 drafters live per-model in the HF account (`ManniX-ITA`, one repo per model);
on bs2 they're mirrored under `/srv/ml/models/nextn/drafters-adapted/<MODEL>/`.

## Worked result — Gemma-4 E2B n_max (2026-07-18, bs2 GPU1)

Tuned E2B F16 target + `gemma-4-E2B-it-assistant.Q8_0` drafter, 9-prompt bench:

| n_max | accept_rate | tok/s_avg | draft/accepted |
|---|---|---|---|
| 1 | 0.747 | 302.1 | 971/725 |
| **2** | 0.538 | **325.6** | 1635/879 |
| 3 (default) | 0.412 | 302.0 | 2274/936 |
| 4 | 0.329 | 285.6 | 2926/962 |
| 5 | 0.263 | 279.5 | 3656/963 |

**`--spec-draft-n-max 2` is optimal for E2B: +7.8% tok/s over the default 3.** Accepted
tokens plateau after n=2 (879→936→962→963) while drafted tokens keep growing — wasted
drafter forwards. Position-1 acceptance is a healthy 0.75; deep drafts decay fast, so
shallow n wins. General rule: **small drafter / high-entropy workload → n=2, not 3.**
(MTP is exact — n_max only affects speed, never the tokens produced.)
