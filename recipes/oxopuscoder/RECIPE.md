# OxOpusCoder-9B — merge recipe

Structured on **Omnimerge v6** (`omnimerge_v2`, density 0.53, DAREx q 0.75, seed 42,
explicit `--task-base`), retargeted to the Qwen3.5-9B family.

**REVISED 2026-09-09: base is Qwopus, not OxCoder.** See §1 — the flip fixes three
defects at once and removes a graft step.

| role | model | params | tensors | `mtp.*` |
|---|---|---|---|---|
| **base** | `Jackrong/Qwopus3.5-9B-Coder` | 9653.1 M | 775 | **15** |
| **source** | `OrionLLM/OxCoder-9B` | 9409.8 M | 760 | 0 |
| **task-base** | `Qwen/Qwen3.5-9B` | 9653.1 M | 775 | 15 |

```
Qwen/Qwen3.5-9B  (common ancestor)
  ├── OrionLLM/OxCoder-9B            <- MTP dropped; eos override missing
  └── Jackrong/Qwopus3.5-9B-v3.5
        └── Jackrong/Qwopus3.5-9B-Coder
```

## 1. Why Qwopus is the base

An earlier draft used OxCoder as base and grafted Qwopus's MTP head in. That was worse
on every axis. **OxCoder-9B carries two internal inconsistencies**, and using it as base
propagates both into the output:

| defect | OxCoder | consequence as base | fixed by the flip |
|---|---|---|---|
| `config` declares `mtp_num_hidden_layers: 1`, ships **0** `mtp.*` | yes | output has no MTP, or a randomly-initialised head | base ships all 15 |
| root `eos_token_id` override absent — only `text_config` 248044 = `<\|endoftext\|>` | yes | **model never stops on `<\|im_end\|>`** → control-token leak in chat | base sets root 248046 |
| `tokenizer_config.json` 1,100 B with **no `added_tokens_decoder`** | yes | GGUF conversion falls back to a *different* chat template | base's is complete (33 entries) |

And the MTP head stays on the trunk it was **trained on**. Grafting Qwopus's head onto an
OxCoder-dominant trunk would have cost draft-acceptance rate for nothing.

The name still reads correctly: OxCoder's coding delta applied onto the Opus-distilled
Qwopus trunk.

## 2. Compatibility — checked, not assumed

Configs structurally identical: `vocab_size` **248320** all three, `hidden_size` 4096,
32 layers, `head_dim` 256, `intermediate_size` 12288, identical `layer_types` (SSM/linear
hybrid, `full_attention` every 4th), identical `vision_config` (depth 27, hidden 1152),
`tie_word_embeddings: false`.

**Vocab proof (the v6 check, repeated here because the files are NOT byte-identical):**

| | `tokenizer.json` | `added_tokens_decoder` | 248044 | 248046 |
|---|---|---|---|---|
| Qwen/Qwen3.5-9B | 12,807,982 B | 33 | `<\|endoftext\|>` | `<\|im_end\|>` |
| Qwopus-Coder | 19,989,343 B | 33 | `<\|endoftext\|>` | `<\|im_end\|>` |
| OxCoder | 19,989,343 B | — (stripped) | — | — |

The 7.2 MB gap to the task-base is unsloth's **merges serialisation** (`"Ġ Ġ"` vs
`["Ġ","Ġ"]`), not vocab content — both descendants carry `unsloth_fixed: true`. **Ids
are identical**, so `embed_tokens` / `lm_head` rows align and the delta is valid. This is
the same situation v6 proved for 3.6→3.8; it must be re-proven per merge, never assumed.

## 3. The merge

```bash
python omnimergekit.py \
  --base      /models/Qwopus3.5-9B-Coder \
  --task-base /models/Qwen3.5-9B \
  --source    /models/OxCoder-9B \
  --weights   0.40 \
  --output    /models/OxOpusCoder-9B \
  --method    omnimerge_v2 \
  --density   0.53 \
  --darex-q   0.75 \
  --seed      42 \
  --no-auto-mlp-skip \
  --skip-patterns visual.,mtp.
```

`out = Qwopus-Coder + 0.40 · sparsify(OxCoder − Qwen3.5-9B)`

- **`--task-base` is mandatory.** Without it the delta is `OxCoder − Qwopus`, which
  subtracts Qwopus's own fine-tune instead of isolating OxCoder's.
- **`--no-auto-mlp-skip` — MLP is merged.** This is where coding knowledge lives; see §5.
- **`visual.` skipped** — both towers descend unchanged from Qwen3.5-9B. Qwopus's 333
  visual tensors pass through.
- **`mtp.` skipped** — the source has no `mtp.*` to form a delta against. Skipping is
  explicit intent, not a no-op: it is how v6 preserved its base head verbatim.

## 4. Tokenizer — no fix needed any more

With Qwopus as base this resolves itself: the output ships Qwopus's complete
`tokenizer_config.json` (33-entry `added_tokens_decoder`), its `chat_template.jinja`, and
its **correct root `eos_token_id: 248046`**. OxCoder's stripped config never reaches the
output. Nothing to repair.

**Still pin `eos_token_id` explicitly** in `config.json` *and* `generation_config.json`
before quantising. Qwopus sets root 248046 but `generation_config` inheritance across a
merge is not guaranteed, and `text_config` still says 248044 — an implicit eos is how
control tokens leak.

The two parents ship materially different chat templates (OxCoder 16,289 B vs Qwopus
8,862 B); they are not interchangeable and OxCoder's pairs with its broken eos. **Keep
the base's.**

## 5. MLP is merged — this is the main risk, and it is gated

`omnimergekit` auto-skips `mlp.{gate,up,down}_proj` by default. That policy comes from
**Qwen3.6-27B**, whose think-emission policy flips across a decision boundary under
1–2 % rel-L2 perturbation of `mlp.gate_proj` — while **Qwen3.5-27B was robust** at the
same magnitude (published Omnimerge-v2, 0.2 % leak). These are `qwen3_5`, so the
precedent favours merging MLP.

But note what changes: v6's weight 0.40 was applied with MLP **skipped**. The same 0.40
with MLP live is a materially larger perturbation. Treat it as a new operating point.

**MANDATORY gate before any eval or quant.** Raw `/v1/completions` (not chat), ~10 MBPP
items, count `<think>` rate and *unclosed* `</think>`:

```
healthy   ~0 % leak, every <think> closed          (Qwen3.5-27B-Omnimerge-v2: 0.2 %)
BROKEN    80 % open / 88 % of opens unclosed       (v3a / v3b / v4, all withdrawn)
```

If it breaks, fall back in this order: `--weights 0.30` → drop `--no-auto-mlp-skip`.

## 6. Verify before quantising

```bash
python scripts/verify_merge_artifact.py \
  --merged /models/OxOpusCoder-9B \
  --ref-a  /models/Qwopus3.5-9B-Coder \
  --ref-b  /models/Qwen3.5-9B
```

Assert, from the artifact and not from exit codes:
- **775 tensors** — a 760 result means `mtp.*` was dropped, the failure this recipe exists to avoid
- `mtp.*` **bit-identical** to Qwopus; `visual.*` bit-identical to Qwopus
- `||out − Qwopus||` small, `||out − Qwen3.5-9B||` large on a sampled `q_proj` (the v6
  check that caught an inverted delta)
- `config.json` root `eos_token_id == 248046`

Then: think-policy gate (§5) → MTP draft-acceptance measurement → 9-bench suite against
**both** parents. MTP affects speed only; a degraded head costs tok/s silently, so
measure acceptance rather than assuming the head survived intact.

## 7. Ablation ladder

| arm | change | question |
|---|---|---|
| **A** | as above | does merging MLP at 0.40 hold think-policy? |
| **B** | `--weights 0.60` | is 0.40 too timid once MLP is live? |
| **C** | `--task-base Qwopus3.5-9B-v3.5` | transfer only OxCoder's delta relative to a nearer ancestor |

## 8. Not decided here

Output repo name, quant tiers, publication. Nothing in this recipe downloads or uploads.
