# MTP speculative-decoding bench & sweeps

**Where the tooling lives:** [`eval/mtp/`](../eval/mtp/) — see
[`eval/mtp/README.md`](../eval/mtp/README.md) for the full reference (flags, invocation,
drafter locations). This page is the research-log summary so the harness is never "lost"
again.

## What it is

`eval/mtp/mtp-bench.py` is a 9-prompt MTP bench (8 coding/topic categories + a long
code-review) that reports **per-prompt and aggregate draft-acceptance + decode tok/s** for
a Gemma-4 (or Qwen NextN) model served with MTP under an **opencoti llamafile** build
(`--spec-type draft-assistant` / `draft-mtp`; not present in upstream llamafile / llama.cpp).
Based on am17an's llama.cpp mtp-bench (PR #23398). Two sweep drivers sit on top:

- `mtp_nmax_sweep.sh` — sweeps `--spec-draft-n-max` (draft tokens/round) → tps-optimal n.
- `accept_quant_sweep.sh` — sweeps the drafter GGUF quant → does quant move acceptance.

## Key findings

### Gemma-4 E2B: serve with `--spec-draft-n-max 2` (2026-07-18, bs2)

Tuned E2B F16 + `gemma-4-E2B-it-assistant.Q8_0` drafter, 9-prompt bench:

| n_max | accept_rate | tok/s_avg |
|---|---|---|
| 1 | 0.747 | 302.1 |
| **2** | 0.538 | **325.6** |
| 3 (default) | 0.412 | 302.0 |
| 4 | 0.329 | 285.6 |
| 5 | 0.263 | 279.5 |

**n=2 is optimal (+7.8% vs the default n=3).** Accepted tokens plateau after n=2 while
drafted tokens keep growing (wasted drafter forwards); per-token acceptance decays with
draft depth (0.75 at position 1 → 0.26 at n=5). Position-1 acceptance (0.75) is healthy —
the E2B drafter is fine shallow, poor deep. **Rule of thumb: small drafter / high-entropy
workload → n=2, not the default 3.** MTP is exact, so n_max is a pure throughput knob
(never changes the tokens produced).

Context: E2B acceptance is inherently lower than larger backbones (A4B ~0.66–0.88, 31B
~0.46–0.79 at native ctx) — least drafter capacity + the same-family drafter. Config-
generation workloads (high-entropy IPs/hex/ports) push acceptance down further (~0.40 on
the AN netconfig eval) vs ~0.54 on this mixed bench at n=2.
