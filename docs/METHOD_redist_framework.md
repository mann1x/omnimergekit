# REDIST — driveable capability-redistribution for pruned Gemma-4 MoE

`scripts/redist.py` + its sibling engines and `recipes/gemma4/v5_moe_sweep/redist_*.sh`
runners implement a **driveable** framework for redistributing the *function* of the
experts a prune dropped INTO the surviving experts of a pruned MoE (e.g. the 62e "A2"
of the Gemma-4 26B-A4B line). The *same machine, different driver corpus* yields a
multilingual-/code-/science-recovered variant.

> **Result of the 62e study:** redistribution into a *fixed* (router, survivor) set
> does **not** recover a capability the prune destroyed — neither closed-form fold
> nor trainable KD, on diffuse-multilingual or localized-code. Capability recovery at
> a fixed budget is a **selection** problem. See
> [`experiments/gemma4_62e_redist.md`](../experiments/gemma4_62e_redist.md) for the full
> negative. This toolkit is preserved as reusable, reproducible machinery for that
> conclusion and for re-driving the pipeline on other capabilities/budgets.

## The 5 stages (`redist.py <stage>`)

| stage | what it does |
|---|---|
| `localize` | which `(layer, expert)` cells carry the target capability, and is it LOCALIZED (few experts → closed-form fold works) or DIFFUSE (spread thin → only trainable survivors can absorb it). Loop-failing caps use `router_diff_bucket.py`; fluent-failing caps (code/science) use `redist_localize_divergence.py`. |
| `capture` | teacher-force the driver corpus through the 128e teacher; record the tensor substrate (router softmax, per-expert SwiGLU gate/up/intermediate, MoE block output). |
| `redistribute` | the pluggable step: a `RedistMethod` consumes capture + keep-map and emits new survivor weights / a recovered variant. |
| `gate` | AR-FIRST validation via `loop_screen.py` (generation canary + held-out divergence). **NEVER** reconstruction-MSE-first (off-manifold rule). |
| `build` | materialize the recovered variant (`expert_drop.py` backend if the keep-set changed) + apply the method delta + full `loop_screen`. |

`redist.py run` chains localize+capture+redistribute; `gate`/`build` stay explicit
(they need the built model dir).

## The 5 methods (`--method`)

| method | trainable | summary |
|---|---|---|
| `hcsmoe` | no | output-space agglomerative cluster-merge (corrected DERN) |
| `mergemoe` | no | per-survivor functional least-squares `T = Q·P⁺` |
| `ream` | no | router-weighted expert activation merging (REAP successor) |
| `expert_kd` | yes | KD into TRAINABLE survivor experts (the only DIFFUSE-capable method) |
| `shared_mlp_kd` | yes | KD into Gemma-4's always-on parallel dense `mlp.*` via LoRA (sidesteps routing) |

`python scripts/redist.py methods` prints the live registry + drivers.

## Tools

| file | role |
|---|---|
| `scripts/redist.py` | the 5-stage CLI + method registry |
| `scripts/redist_localize_divergence.py` | fluent-failure (teacher-correct vs student-failing) localization |
| `scripts/redist_rank_probe.py` | E-RankProbe single-layer trainable capacity test |
| `scripts/loop_screen.py` | 200-prompt greedy AR loop canary (the gate); imports `audit_full_bench.detect_loop` |
| `scripts/router_diff_bucket.py` | loop-differential localization (teacher vs student dropped-mass) |
| `scripts/router_kd.py` | the KD trainer (`expert_kd` / `shared_mlp_kd` backends) |
| `recipes/gemma4/v5_moe_sweep/redist_*.sh` | host-independent runners (below) |

Repo deps reused as-is: `gemma4/expert_pruning/{expert_drop,generate_drop_map_v5}.py`,
`recipes/gemma4/v5_moe_sweep/router_shared_upweight.py`, `scripts/audit_full_bench.py`.

## Host independence

### `_resolve_dep` — where `redist.py` finds its sibling scripts

`redist.py` calls a few scripts as subprocesses. They are resolved host-independently,
probing in order: **(1)** `--scripts-dir/<name>` (operator override), **(2)** the dir
holding `redist.py` (co-located: `loop_screen.py`, `router_diff_bucket.py`,
`redist_localize_divergence.py`), **(3)** a repo-relative map (`expert_drop.py`,
`generate_drop_map_v5.py` → `gemma4/expert_pruning/`; `router_shared_upweight.py` →
`recipes/gemma4/v5_moe_sweep/`), **(4)** `PATH`. If none resolve it raises listing every
probed path.

### `REDIST_*` environment

Engines read these as defaults (CLI flags override; absent value → loud fail at use,
never a crash deep in a forward pass). Runners source `redist_config.sh` (see below).

| var | required? | meaning |
|---|---|---|
| `REDIST_TEACHER` | yes (most stages) | 128e teacher model dir |
| `REDIST_STUDENT` | yes (most stages) | pruned 62e (A2) student dir |
| `REDIST_KEEP_META` | yes (localize/redistribute) | A2 survivor keep-metadata json |
| `REDIST_SAMPLE` | yes (gate/screen) | 200-prompt loop_screen jsonl |
| `REDIST_CALIB_MULTILINGUAL` | rankprobe runner | multilingual capacity-probe corpus |
| `REDIST_CALIB_CODE` | code-fold runner | code closed-form capture corpus |
| `REDIST_KD_CORPUS_CODE` | code-KD runner | code trainable-KD train corpus |
| `REDIST_IMATRIX` | quant runners | A2 `imatrix.dat` for Q6_K (preserve it!) |
| `REDIST_PY` | no (def `python`) | interpreter for engine subprocesses |
| `REDIST_SCRIPTS_DIR` | no (def `<repo>/scripts`) | dep-script dir |
| `OMK_EVAL` | no (def `<repo>/eval/omk_eval.py`) | eval driver |
| `CONVERT_HF_TO_GGUF` / `LLAMA_QUANTIZE` | no (def `$(command -v …)`) | GGUF toolchain |
| `REDIST_ENV_BIN` | no | prepended to `PATH` (omk-env tools for `omk_eval`) |
| `REDIST_WORK` / `REDIST_RESULTS` / `REDIST_OUT_BASE` | no (CWD-relative defaults) | captures / summaries / built models. **Never /tmp.** |

## Runners — dry-run first

The `recipes/gemma4/v5_moe_sweep/redist_*.sh` runners share a preamble: locate the repo
from `BASH_SOURCE`, source `redist_config.sh` if present, set the toolchain with safe
defaults. **They print their plan and exit 0 by default** — pass `--run` to execute. The
host/user inputs are guarded with fail-loud `${VAR:?}` (per `docs/SECURITY.md`: `:?` for
required, `:-` for infra defaults, **no secrets ever**).

| runner | pipeline |
|---|---|
| `redist_run.sh <driver> <method> <corpus>` | capture → fit/emit → loop_screen (closed-form) |
| `redist_smoke_closedform.sh <method> [cap.pt]` | fit → emit → shape/finite/generate smoke |
| `redist_rankprobe_run.sh [gpu] [layers]` | capture(expert_kd) → E-RankProbe |
| `redist_expert_kd_run.sh <name> <corpus> …` | KD-train survivors → loop_screen |
| `redist_code_fold.sh <method>` | code capture → fold → loop_screen (KEEP model) |
| `redist_code_kd.sh [tt] [layers] …` | code-KD → loop_screen → Q6_K → HE+164/MPE-100 |
| `redist_code_eval.sh <method>` | folded bf16 → Q6_K → HE+164/MPE-100 |

## Quickstart

```bash
cd recipes/gemma4/v5_moe_sweep
cp redist_config.sh.example redist_config.sh   # edit paths; stays local/uncommitted
bash redist_run.sh code ream /path/to/code_calib.jsonl        # dry-run: prints the plan
bash redist_run.sh code ream /path/to/code_calib.jsonl --run  # execute
```

Off the runners, drive `redist.py` directly:

```bash
export REDIST_TEACHER=… REDIST_STUDENT=… REDIST_KEEP_META=… REDIST_SAMPLE=…
python scripts/redist.py methods
python scripts/redist.py run --driver code --method ream --corpus code_calib.jsonl
```
