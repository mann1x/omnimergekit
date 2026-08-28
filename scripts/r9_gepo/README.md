# R9 / GEPO — run-host tooling and analysis

Everything here was written on **bs2** (the run host) during the R9 GEPO brevity
program and lived only there until 2026-08-28. It is committed so the analysis behind
the published numbers is auditable, and so the next run does not re-derive it.

The **program itself** lives in the parent directory — `gepo_brevity.py` (trainer),
`gepo_reward_v2.py` (reward), `r9_gepo_hf.sh` (launcher), `r9_gepo_run{2,3,4}.sh`
(arms), `gate_replay_artifact.py` (run-host pool gate), `build_gepo_replay_pool.py` /
`build_lcb_rl_pool.py` (pools). This directory is the surrounding apparatus.

## What the program measured

run1–run3 all failed, and the reason is measured, not guessed. The v1 reward
`r = (1 − 0.7·min(ntok/12288,1))` gave length only **11.3 % of within-group
variance** — and GRPO normalises advantage by that std, so ~89 % of the gradient said
"be correct". Worse, the share **fell to 0.8 %** as lengths converged: the brevity
signal died exactly as the model improved. run3 additionally put the 40 MoE routers in
LoRA scope; the mechanism worked (40/40 took gradient, 40/40 merged) and the outcome
was still negative — suite mean 0.8882, the lowest of the four arms.

run4 replaces the reward (group-relative, scale-free, 86.5 % variance share) and adds
a λ=0 capability-replay tier.

## Layout

| Group | Files | What they do |
|---|---|---|
| **Build / merge** | `build_a3b_gepo{1,2,3}.sh`, `merge_gepo1.py`, `gate_router_merged.py` | Merge a GEPO adapter into armJ and gate the result. `gate_router_merged.py` checks **four polarities** — routers moved, run2 scope moved, routed experts identical, MTP router identical. Its 2026-08-27 bug (bug-639) was reading runtime module paths `model.layers.N.*` against checkpoint keys `model.language_model.layers.N.*`, reporting 40/40 "missing" on a *successful* merge. It now discovers keys by suffix and refuses if the count ≠ 40. |
| **Eval drivers** | `suite_gepo{1,2,3}.sh`, `eval_gepo{1,2}.sh`, `gpqa_gepo1.sh`, `chain_gepo3.sh`, `chain_run3.sh`, `lcb_repeat.sh`, `mpe_armj_b604.sh` | Per-arm suite runs and chains. `chain_*.sh` wait on an **observed** condition (process gone AND both GPUs actually free), never a sleep. `lcb_repeat.sh` is the same-model repeat draw that established the LCB noise floor. |
| **Paired stats** | `gpqa_paired.py`, `mpe_paired.py`, `lcb_pairs.py`, `lcb_flip.py`, `paired_cell.py`, `rpt_final.py`, `rpt_partial.py` | McNemar exact on paired pass/fail. `lcb_flip.py` + `lcb_repeat.sh` produced the measured **paired SE = √20/77 = 5.81 pp**, which retired the LCB −7.79 pp "regression" as draw noise. |
| **Length / cap forensics** | `lcb_captab.py`, `lcb_captab_qs.py`, `mpe_captab.py`, `mpe_captab_qs.py`, `mpe_cap_xtab.py`, `lcb_partial.py`, `reason_len.py`, `reason_len2.py`, `reason_dist.py`, `traj.py` | Cap-hit × pass cross-tabs and length distributions. These exist because **cap asymmetry turns a bench into a length meter**: an MPE +2.67 pp collapsed to +0.00 on the both-uncapped pair. |
| **Failure modes** | `lcb_failmode.py`, `lcb_pattern.py`, `lcb_xtab.py`, `gpqa_rumination.py`, `gpqa_xtab.py`, `degrade.py`, `suite_audit.py`, `suite_table.py`, `table_gepo_arms.py` | Why a cell moved, not just that it did. `TABLE_CORRECTION.md` records a table that had to be withdrawn. |
| **Scope / adapter audit** | `lora_audit.py`, `scope_verify.py`, `caps_read.py` | Which modules actually took gradient. A `lora_B` still exactly zero proves no gradient reached that module — the check that made the run3 router verdict trustworthy. |
| **Probes / housekeeping** | `gepo_watch.py`, `hf_probe.py`, `probe_serve.py`, `probe_sqlite.py`, `tp_smoke.py`, `gpqa_n.py`, `purge_armq6k.sh`, `purge_t1_t3.sh`, `bundle_gepo3.sh` | Liveness and readiness probes, cache spot-checks, disk purges. `bundle_gepo3.sh` builds the Ollama bundle (`gepo3_tb:Q4_K_M`, `gepo2_tb:Q4_K_M`, `gepo1_tb:Q6_K`) and **exits 6 on an empty completion** — an earlier version captured ANSI spinner escapes from `ollama run` and proved nothing. |

## Two traps these scripts encode

- **`pgrep -f` self-matches on bs2.** A pattern that appears anywhere else on the
  invoking command line matches the invoker (exit 255). Patterns are bracketed
  (`[l]cb_repeat`), and the filename must not also appear in the same command.
- **A stale orphan silently disables a liveness gate.** An Aug-23 orphan made a
  `pgrep`-based death branch permanently dead code. Gate on a marker or artifact
  first; treat process-absence as secondary, and match a captured PID, not a name.

## Logs

Run logs stay on bs2 under `/mnt/sdc/ml/brevity/gepo/` (some are 180–270 KB). They are
not committed; the scripts and the tables derived from them are.
