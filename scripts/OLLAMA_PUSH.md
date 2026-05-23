# Hugging Face → Ollama.com push runners

End-to-end pipeline for re-publishing every GGUF tag from a Hugging Face GGUF
repo to a corresponding [ollama.com](https://ollama.com) model namespace, one
quant at a time, resumable, with strict disk discipline.

Two scripts (different chat-template handling — pick by base architecture):

| Script | Use it for | Chat template |
|---|---|---|
| `ollama_push_gemma4.sh` | Gemma 4 models (any variant: 26B-A4B v3/v4/v5/v5-coder/v5-logic, 31B-it/he1, …) | **custom** Gemma 4 tools template baked in (the 2nd-tool-call turn fix). Mirrors the published `mannix/gemma4-98e` recipe. |
| `ollama_push_generic.sh` | Anything else (Qwen2/3, Llama, Mistral, etc.) | none — ollama auto-detects from the GGUF metadata's `tokenizer.chat_template`. |

Both produce identical published artifacts otherwise; the GGUF bytes uploaded
to ollama.com are byte-identical to the HF blobs.

## What they do

For each quant tag `T` (e.g. `Q4_K_M`, `IQ3_XXS`, `CD-Q6_K`) found in the source
HF repo:

1. **`hf download $HF_REPO $FILENAME.gguf --local-dir /workspace/.../scratch`** —
   pre-download the GGUF directly from HF, with retry/resume via `hf_transfer`.
2. **`ollama create $OL_TARGET:T -f Modelfile`** — Modelfile uses `FROM
   /workspace/.../scratch/$FILENAME.gguf`. **Local path, no hf.co/ pull.**
   ollama just imports the file's bytes into its blob store and creates a
   manifest. No network involved at this step.
3. `ollama push $OL_TARGET:T` — upload the manifest + blob to ollama.com.
4. `ollama rm $OL_TARGET:T` — drop the local manifest. Since we never created
   a `hf.co/...` ref, there's no second ref to clean.
5. Force-`rm` the captured blob digests + any `-partial-*` residue (belt +
   suspenders — ollama's GC is unreliable on busy daemons).
6. `rm -f` the scratch GGUF file from step 1. Disk usage returns to baseline.
7. `sleep 5` between iterations (relieves daemon contention on rapid loops).

The scripts are resumable. They list existing tags on `ollama.com/$OL_TARGET/tags`
at startup and skip anything already published, so a re-run after a crash or a
manual `pkill` just picks up the gaps.

### Why step 1 was added (2026-05-11)

Earlier versions used `FROM hf.co/$HF_REPO:T` directly, letting ollama do the
pull internally. Two problems showed up over a 60-tag run:

1. **ollama's per-create deadline is hardcoded short** — it consistently
   reported `Error: context deadline exceeded` on ~50% of 12-25 GB blobs even
   when raw HF bandwidth was healthy and disk had room. The retry-with-flush
   logic at the script level didn't help because each retry hit the same
   deadline.
2. **Failed `ollama create` from `hf.co/` left a `hf.co/$HF_REPO:T` ref
   behind** that pinned the partial blob and required a second `ollama rm`.
   Forgetting one of the two `ollama rm` calls leaked the blob — see the
   199 GB blob-leak post-mortem below.

Switching to **pre-download via `hf download` + `FROM /local/path`** sidesteps
both problems: HF bandwidth is whatever it is, ollama's deadline never starts,
and only one ref ever exists.

Trade-off: ~25 GB peak scratch space per quant. Each is deleted right after
push, so steady-state disk is unchanged.

## Where they run

- **vast.ai / RunPod / local box with enough bandwidth.** A 27B-A4B BF16 quant
  ladder (~25 tags) at 200–400 MB/s sustained each way takes ~90–120 min when
  run in parallel with a sibling push.
- They require `ollama serve` running locally and an account signed in via
  `ollama signin` (or the cached `~/.ollama/id_ed25519*` keypair).

## Usage

Both scripts take the HF token as a **command-line argument** (`$3`). The
old `HF_TOKEN` env var still works if you pass `-` as `$3`, but the new
calling convention is preferred so the token doesn't end up in `ps aux`
of a parent shell.

```bash
TOKEN=$(cat ~/.cache/huggingface/token)

# Gemma 4 (custom tools template + 2nd-turn workaround):
bash ollama_push_gemma4.sh \
    ManniX-ITA/gemma-4-A4B-98e-v4-it-GGUF \
    mannix/gemma4-98e-v4 \
    "$TOKEN"

# Generic (Qwen3.6, ollama auto-detects chatml template):
bash ollama_push_generic.sh \
    ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-GGUF \
    mannix/omnimerge-v4 \
    "$TOKEN"

# Optional 4th arg: include-pattern regex (only push matching tags)
bash ollama_push_generic.sh src target "$TOKEN" '^(Q4_K_M|Q6_K|IQ4_XS)$'
```

## Parallel pushes

You can run both scripts at the same time against different HF/Ollama
namespaces. They share `ollama serve` and `/root/.ollama/models/blobs/` but the
ollama daemon serializes blob operations internally (content-addressed store),
and each iteration's `ollama create / push / rm / rm` sequence is independent
per tag.

**Do not** run two parallel pushes targeting the **same** ollama target —
overlapping manifests can step on each other. Use distinct `$OL_TARGET` names.

## Disk discipline (critical)

The non-obvious failure mode that bit us hard on the v3 push:

```
ollama create FROM hf.co/X:T  →  creates BOTH `hf.co/X:T` AND `mannix/Y:T` refs
ollama push   mannix/Y:T      →  uploads blob, both refs still local
ollama rm     mannix/Y:T      →  blob NOT GC'd because hf.co/X:T still pins it
```

After 18 quants the local blob store hit 199 GB and filled the pod's overlay
to 100 %, which then failed a `touch` of a downstream sentinel and stalled the
entire orchestrator for 7 hours unnoticed.

**The fix** (already in these scripts) is a three-layer purge every iteration:

```bash
# 1. Capture the blob digests this tag references, BEFORE removing the refs
BLOBS=$(for MF_PATH in \
    "/root/.ollama/models/manifests/registry.ollama.ai/${OL_TARGET}/${TAG}" \
    "/root/.ollama/models/manifests/hf.co/${HF_REPO}/${TAG}"; do
    [ -f "$MF_PATH" ] && grep -oE 'sha256:[0-9a-f]{64}' "$MF_PATH"
done | sort -u)

# 2. Remove BOTH refs — the hf.co/ ref keeps the blob pinned otherwise
ollama rm "${OL_TARGET}:${TAG}"      || true
ollama rm "hf.co/${HF_REPO}:${TAG}"  || true

# 3. Belt + suspenders: force-purge the captured blobs and any -partial-* residue.
# By here both refs are gone, so these blobs are guaranteed unreferenced.
for B in $BLOBS; do
    FN="/root/.ollama/models/blobs/${B/sha256:/sha256-}"
    rm -f "$FN" "$FN-partial"* 2>/dev/null || true
done

# 4. Brief pause to relieve HF/ollama-daemon contention before the next iter
sleep 5
```

Why all four steps:

- **Step 1** must come before step 2, because once `ollama rm` deletes the
  manifest there's no way to discover which blobs it referenced. We learned
  this the hard way on the 2026-05-11 omnimerge push.
- **Step 3** catches two cases the ollama daemon misses: (a) `-partial-N`
  blobs from a `CREATE FAILED` iteration (the daemon never tracks them as
  refs, so it never GCs them), and (b) the rare race where `ollama rm`
  returns success but leaves the blob on disk because something else briefly
  held an open FD. On a long push these accumulate to 10s of GB.
- **Step 4** isn't strictly about disk — it's about ollama's internal
  HF-pull timeout (`context deadline exceeded`). When parallel pushes
  compete for HF bandwidth + the local daemon, a tight back-to-back
  iteration can saturate the connection pool; a 5 s break drops the
  failure rate to ~zero in practice.

If your disk creeps up during a push, it usually means an older copy of
either script is running. Verify with `du -sh /root/.ollama` — it should
stay flat (< 25 GB) for a 27B model push because at any given moment only
one quant's blob is live.

### Don't run both pushes in parallel against the same daemon

On 2026-05-11 we ran the Gemma 4 push script (then named `ollama_push_98e.sh`, now `ollama_push_gemma4.sh`) (Gemma 4 v4, 31 tags) and
`ollama_push_generic.sh` (Qwen3.6 omnimerge, 25 tags) in parallel against
one `ollama serve`. The Gemma push consistently completed each iteration;
the Qwen push failed ~half of them with `context deadline exceeded` on
`ollama create`. Failure rate correlated with disk pressure: the leak from
each Qwen failure left a `-partial` blob behind, eventually filling the
250 GB overlay and bricking every subsequent iteration. The post-mortem
fix is the four-layer purge above **plus** running the two pushes
**serially** (omnimerge → v4, or v4 → omnimerge — sequence doesn't
matter, just don't overlap them on the same daemon).

## Discovery / tag derivation

Both scripts derive the tag from the GGUF filename via a regex against the
HF repo's `tree/main` API:

- `ollama_push_gemma4.sh` expects `*-it-<TAG>.gguf` (the Gemma 4 naming).
  E.g. `gemma-4-A4B-98e-v4-it-CD-Q6_K.gguf` → tag `CD-Q6_K`.
- `ollama_push_generic.sh` walks the trailing dash-separated components
  and matches against a quant-tag pattern (`F16`, `Q\d+(_[KS01ML]+)?`, `IQ\d+_*`,
  `CD-Q\d+_*`).
  E.g. `Qwen3.6-27B-Omnimerge-v4-Q4_K_M.gguf` → tag `Q4_K_M`.

If your filenames don't match either convention, add a custom regex inline
or rename on HF first. Both scripts log the full discovered tag list at
startup so you can sanity-check before pushes start eating bandwidth.

## Modelfile contents

**Gemma 4 (`ollama_push_gemma4.sh`):** ships a tools-aware chat template that
fixes a 2nd-turn-call regression in the upstream Gemma 4 ollama template, plus
the v4 sampling defaults (`temperature 0.6 top_p 0.95 num_ctx 256000
repeat_penalty 1.15 stop <turn|>`).

**Generic (`ollama_push_generic.sh`):** no TEMPLATE / PARAMETER overrides. The
GGUF's embedded chat template is what ollama will use. This is correct for
Qwen3.5/3.6 (chatml) and most modern model lines whose authors set the chat
template metadata properly.

## Required environment

| Var | Purpose |
|---|---|
| `HF_TOKEN` | Read token for the source HF GGUF repo (write token also works). |
| (ollama account) | `ollama signin` must have been run on this host previously, or the keypair at `~/.ollama/id_ed25519*` must be the one registered with ollama.com. |

## Troubleshooting

- **`CREATE FAILED for X; skipping`** — the most common cause is a transient
  download stall from HF (ollama doesn't retry inside `create`). The script
  logs the failure and continues to the next tag; a subsequent run skips
  already-published tags and retries the failures. If a tag fails twice in a
  row, check the GGUF on HF: maybe it's malformed or hf.co is throttling.
- **Disk creeping toward 100 %** — see "Disk discipline" above. Confirm with
  `du -sh /root/.ollama`; if > 25 GB, an old un-patched copy of the script
  is in flight, or your `ollama rm hf.co/...:TAG` line is missing.
- **`No space left on device` on `touch`** — same root cause as above; the
  touch failure is the first symptom because sentinels are tiny.
- **`Unauthorized` from ollama push** — the running `ollama serve` is not
  signed in to your account, or it's signed in to a different one. Run
  `ollama signin` and verify the public key at `~/.ollama/id_ed25519.pub`
  matches the one in your ollama.com account settings.

## Related

- `mlx_convert.sh` / `MLX_CONVERT.md` — sibling pipeline that publishes MLX
  variants of the same source models to HF (no ollama).
- `quantize_gguf.py` — the upstream step that *produced* the GGUF quants in
  the HF source repo. The ollama push scripts assume that pipeline has
  already landed all the desired tags.

## Version history

- **2026-05-11 (first cut)** — initial public version. Pulled out of
  `pod2_chain_no_destroy.sh` into reusable single-purpose scripts. Added
  double-`ollama rm` after the v3 199 GB blob-leak incident. Verified on
  `mannix/gemma4-98e-v4` (31 tags) and `mannix/omnimerge-v4` (25 tags).
- **2026-05-11 (hardened)** — three-layer purge (capture blobs → dual
  `ollama rm` → force-`rm` captured blobs + `-partial-*` residue) +
  per-iteration `sleep 5` to relieve daemon contention. Added explicit
  warning against running both pushes in parallel against one daemon
  (the second push starves on the daemon's HF connection pool and
  every `CREATE FAILED` leaks a `-partial` blob, eventually filling
  disk). Recipe is now: run them **serially**.
- **2026-05-19 (hardened + renamed)** — two new safeguards landed
  after the v5-it 13-tier push ballooned `/root/.ollama/models` to
  128 GB on pod 36949547:
  - **Orphan blob sweep.** `ollama rm <tag>` removes the manifest but
    leaves the content-addressed blobs in
    `$OLLAMA_MODELS/blobs/sha256-*` if any *other* manifest still
    references them — and for content-deduped blobs they often
    survive long after every referring manifest is gone. The Gemma
    script now scans both `/usr/share/ollama/.ollama/models` and
    `/root/.ollama/models` after each tag and `rm`s blob files with
    no surviving manifest reference. Logs `swept N blob(s), M MB
    freed` per iteration.
  - **Private-model marker file.** The default "already pushed?"
    check (`curl ollama.com/<target>/tags` HTML scrape) returns
    empty for **private** models on ollama.com, even when tags
    *are* pushed. Default-private namespaces (e.g.
    `mannix/gemma4-98e-v5` while still in evaluation) therefore
    looked like fresh repos, and the script would happily re-push
    every tier on a re-run. The script now unions the HTML scrape
    with a local marker file
    `$WORKDIR/pushed_<target>.txt` written after each verified
    push. Pre-populate it manually with the set of known-pushed
    tiers if running across pods. **To make a model truly public**,
    mannix must toggle visibility in the ollama.com web UI — the
    script can't do that.
  - **Rename.** The script was renamed from `ollama_push_98e.sh` to
    `ollama_push_gemma4.sh` because it has always been
    architecture-specific (custom Gemma 4 tools template) not
    98e-specific — it's the canonical pusher for every Gemma 4
    variant (98e v3/v4/v5/v5-coder, 31B-it, 31B-he1, …). A frozen
    historical copy lives at
    `scripts/archived/ollama_push_98e.sh.frozen-2026-05-19` (write-
    protected) for reference. **Use the renamed script for all new
    runs. There is no per-model variant of this script and there
    never should be — one script, all Gemma 4 push targets.**
