# Security — secrets handling


## Gated eval content (GPQA, frozen LCB) — a second class of "never commit"

This repo is public. Alongside credentials there is a second class of content that must
never reach it: **gated benchmark items in plaintext**. GPQA is gated precisely so its
questions stay out of crawls and future pretraining sets, and it ships a per-row canary
for that purpose. Publishing the items also contaminates the benchmarks this project
scores its own model cards on.

### What went wrong

A tier census on **2026-09-08** correctly withheld
`recipes/qwen3_6_35b_a3b_prune/results/router_calib_corpus_ornith.jsonl` (80
`gpqa_diamond` rows). An audit on **2026-09-09** found
`router_calib_corpus_qwen.jsonl` — **md5-identical, the same 80 Diamond rows** —
already tracked and public, along with nine other files:

| findings | file |
|---|---|
| 396 | `eval_results_vllm_suite/.../samples_gpqa_diamond_cot_zeroshot_*.jsonl` |
| 209 | `scripts/router_calib_corpus.jsonl` |
| 100 | `scripts/router_calib_corpus_9bench_balanced.jsonl` |
| 100 | `scripts/router_calib_corpus_ifeval_heavy.jsonl` |
| 95 | `eval/efficiency/grpo_pool_v3_presmoke.jsonl` |
| 80 x4 | `recipes/.../router_calib_corpus_{qwen,coder_qwen,coder_lcbmpe_qwen,coder_lcbmpeife_qwen}.jsonl` |
| 11 | `scripts/router_calib_corpus_protect.jsonl` |

**The withhold was keyed on the FILENAME, and the same bytes under another name walked
straight past it.**

### The gate

Two content-keyed gitleaks rules in `.gitleaks.toml`, enforced by the same pre-commit
hook as the secret scan:

- **`gated-eval-gpqa-canary`** — matches GPQA's own per-row canary (`gpqa:xxxx:<uuid>`).
  Catches raw dataset rows and lm-eval sample dumps.
- **`gated-eval-content-in-data-file`** — matches a row tagged with a gated/frozen eval
  split (`"bench"`/`"tier"`/`"source"`/... = `gpqa*`, `lcb_v6_77q`, `lcb_v6_55`,
  `lcb_hard_77`, `lcb_medium_100`, `lcb_calib`, `mbpp_full_test`), restricted to data
  extensions. A rendered prompt need not carry the canary, so the row tag is the
  detector.

Verified in **both** directions before landing. Catches: 80 findings on
`router_calib_corpus_qwen.jsonl`, 146 on `grpo_pool_v4_budgeted.jsonl`, 396 on the
lm-eval samples — counts matching the known row counts exactly. Does **not** fire on
`build_grpo_pool_v4.py`, `competence_qwen35b.json` (aggregate stats, no question text),
the drop-map summaries, or `docs/METHOD_grpo_efficiency.md`.

A first draft matched the LCB **difficulty** labels (`lcb_v6_hard` / `lcb_v6_medium`)
and flagged `eval/lcb/lcb_rl_pool.jsonl` and `eval/replay/gepo_mixed_pool.open.jsonl`,
both legitimate and already gated by task id. Narrowed. **A gate that cries wolf on
correct artifacts is a gate someone switches off.**

### Rules

1. **Key the withhold on CONTENT, never on path, program, or filename prefix.**
2. `.gitignore` entries are convenience only. **Never add a path to silence the gitleaks
   gate** — census the content instead.
3. `git rm --cached` stops future exposure; it does **not** remove history. Removing
   history needs the `git filter-repo` + force-push runbook below, and it cannot recall
   forks or caches.


## Never hardcode credentials

This is a **public** repository. No token, API key, password, or other secret may
ever be committed — not in code, not in a comment, not as a "convenience default".

The specific antipattern that bit us: a shell `${VAR:-<literal>}` fallback.

```bash
# ✖ NEVER — bakes a live secret into the file so it "just works" without the env var
export HF_TOKEN="${HF_TOKEN:-hf_REALTOKEN...}"

# ✓ ALWAYS — fail loud if the operator forgot to export it
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported in the environment}"
```

`${VAR:?msg}` aborts the script with `msg` when the variable is unset or empty.
That is the correct "make it obvious you forgot" behavior — not a hidden default.

## The pre-commit gate

A [gitleaks](https://github.com/gitleaks/gitleaks) hook scans the **staged diff**
before every commit and blocks any secret. Wire it up once per clone:

```bash
bash scripts/install-git-hooks.sh      # installs .git/hooks/pre-commit
# requires the gitleaks binary on PATH:
#   https://github.com/gitleaks/gitleaks/releases → /usr/local/bin/gitleaks
```

Rules live in [`.gitleaks.toml`](../.gitleaks.toml) (built-in ruleset + an
allowlist for redaction markers and env-var indirection). Contributors who use
the [pre-commit framework](https://pre-commit.com) can instead
`pip install pre-commit && pre-commit install` to pick up
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml).

If gitleaks is not installed the hook **warns loudly and does not silently pass** —
install the binary to restore the gate.

## If a secret is committed anyway

1. **Revoke it immediately** at the provider (HF: Settings → Access Tokens → revoke).
   A leaked token is compromised the instant it hits a public repo; rotation is the
   only real fix. Scrubbing history is cleanup, not remediation.
2. Redact at `HEAD`, then scrub all history:
   ```bash
   printf 'THE_SECRET==>***REMOVED***\n' > /tmp/repl.txt
   git filter-repo --replace-text /tmp/repl.txt --force
   git remote add origin <url>      # filter-repo strips the remote
   git push --force origin main
   ```
   Take a `git bundle create backup.bundle --all` first.
3. Note that GitHub may keep old commits reachable by direct SHA until its GC runs,
   and forks/caches are unaffected — which is exactly why step 1 (revoke) is what
   actually protects you.

> Historical incident: 2026-05-23 — a live HF token leaked via a `${HF_TOKEN:-...}`
> fallback in `scripts/pod_quant_31b.sh`. Token was revoked, history scrubbed, and
> this gate added so the class of mistake is now un-committable.

## Dependency advisories — accepted risk on the pinned eval stack

`requirements.txt` and `.requirements_eval_filtered.txt` pin `torch==2.10.0` and
`transformers==5.5.0`. These pins are **provenance**, not merely a lockfile: they
record the stack every published eval score was produced on. Bumping them makes
future scores incomparable to every result already released, so a dependency
advisory is not on its own sufficient reason to move them. Decide explicitly,
record the decision here, and re-pin the whole cohort together — never one
package silently.

Reviewed 2026-09-03 (4 open Dependabot alerts, 2 high + 2 low, duplicated across
the two manifests):

| advisory | package | status |
|---|---|---|
| `torch.jit.script` memory corruption (`<= 2.12.1`, fixed 2.13.0) | torch 2.10.0 | **NOT EXPOSED** |
| `save_pretrained` path traversal via chat-template names (`< 5.10.0`, fixed 5.10.0) | transformers 5.5.0 | **ACCEPTED** |

**torch — not exposed.** `git grep -nE "torch\.jit|jit\.script|jit\.trace|TorchScript"`
over the whole tracked tree returns zero hits. The vulnerable entry point is never
called. Re-run that grep before assuming this still holds; introducing any
TorchScript use makes the advisory live and this row must be revisited.

**transformers — accepted, with a real surface.** The flaw allows a crafted chat
template *name* to write outside the target directory during `save_pretrained`,
and this repo has ~52 `save_pretrained` call sites and routinely loads
third-party checkpoints from the Hub. It is accepted because re-pinning would
invalidate the published v4/v6/v7 eval bases, not because the risk is nil.
Mitigation until the pin moves: treat an untrusted checkpoint's `config.json` /
`chat_template.jinja` as hostile input — prefer vendor and own-account repos, and
check `save_pretrained` wrote only inside the intended output dir when merging a
checkpoint from an unfamiliar author.

Revisit when a cohort boundary opens (no eval in flight, no run mid-campaign):
that is the only cheap moment to move the pin.
