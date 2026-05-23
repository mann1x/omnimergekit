# Security — secrets handling

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
