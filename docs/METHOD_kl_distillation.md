# KL Distillation — Findings, Recipes, and Research

**Last updated:** 2026-05-04. Source experiments: pod vast.ai 35822024,
DS-Coder-1.3B-Instruct student, full HE-164 + MBPP-378.

> This is a working distillation playbook for OmniMergeKit recipes. It
> covers what we tried, what worked, what to avoid, and a recommended
> recipe template for "small student + larger teacher, code task".
> Cross-tokenizer mechanics are factored out to the standalone library
> [`mann1x/cross-tokenizer-distill`](https://github.com/mann1x/cross-tokenizer-distill);
> this file focuses on **what to put in the loss**, not the projection plumbing.

## TL;DR — recipe to start with

For a `~1B–7B student` with a `same-vocab teacher 3-7×` larger and a small
(<2 K examples) task-style corpus:

```
Loss:        forward KL on student-sampled positions   (DistillSpec / on-policy FKL)
            T = 1.0                                    (no temperature softening)
            T² rescale: ON                             (Hinton convention)
LoRA:        rank 16, alpha 32, all-linear targets
LR:          5e-5, cosine, 2 epochs, warmup 8 steps
Batch:       2 × grad-accum 8 (eff. 16)
Sampling:    student samples max 128 new tokens, temperature 1.0
Eval gate:   full HE-164 (smoke n=20 has SE ~11pp — not trustworthy)
Reject if:   HE pass@1 < (SFT same-recipe + 2 pp)
             — that means distillation isn't even regularising over SFT
```

This recipe was the best of M3/M4/M5 below; it costs the same as SFT and gains
3.7–4.3 pp HE over SFT at the same recipe.

## What we tried

All runs use the same student (DS-Coder-1.3B-Instruct), the same 374-example
HE-style corpus (MBPP train as docstring prompts), the same LoRA rank-16
recipe, lr 5e-5, 2 epochs.

| Run | Method | Loss | Teacher (vocab) | HE-164 | Δ vs base (59.8) |
|---|---|---|---|---|---|
| BASE | no FT | — | — | **59.8** | — |
| SFT | SFT (mbpp-train) | CE on labels | — | 51.8 | **−8.0** |
| M3 | GKD | generalized JSD (β=0.5) | DS-Coder-6.7B (32 K, same vocab) | 55.5 | −4.3 |
| M4 | MiniLLM | reverse KL on student-sampled positions | DS-Coder-6.7B | 54.3 | −5.5 |
| M5 | DistillSpec | forward KL on student-sampled positions | DS-Coder-6.7B | 56.1 | −3.7 |
| M6 | CTD on-policy FKL (cross-vocab) | FKL via VocabMapper | Qwen2.5-Coder-7B (152 K) | **38.4** | **−21.4 (FAIL)** |

(M6 results land in [`cross-tokenizer-distill/docs/RESULTS.md`](https://github.com/mann1x/cross-tokenizer-distill/blob/main/docs/RESULTS.md).)

### Findings

1. **Distillation regularises vs SFT.** Every KL variant beats SFT on HE by
   3.7–4.3 pp at identical recipe and corpus. This was the biggest single
   finding: the teacher's distributional pull keeps the small student from
   overfitting the 374-example training set.
2. **Forward KL > generalized JSD > reverse KL** for this class
   (small student, larger same-family teacher, code task). The mode-seeking
   tendency of reverse KL (MiniLLM) collapses too fast on a 1.3B student;
   forward KL keeps the distribution wider and generalises better.
3. **MBPP is preserved by every distill recipe** (61.1 % = base) but
   *regresses* under SFT (−1.6 pp). Distillation acts as a true regulariser,
   not just a quality boost.
4. **Smoke sets are not trustworthy for ranking.** With HE-20 the standard
   error is ~11 pp; smaller-than-7-pp differences at 50 % accuracy disappear
   into noise. Use smoke only as a "did training crash" gate. Always rank on
   full HE-164.
5. **Cross-vocab CTD with `byte_anchor` + `multi_token=distribute` on a 5×
   vocab gap is destructive at this projection coverage** (M6: −21.4 pp HE
   vs base, far below the −4.3 pp same-vocab GKD reached). Training was
   healthy (loss 2.57 → 0.92, 73 % positions aligned) but 80.9 % multi-token
   teacher mass smeared probability across many low-confidence student
   tokens. For the v6U decision rule (`C ≤ A → fall back`), this fires:
   prefer DS-Coder-V2-236B same-vocab teacher over a cross-vocab Qwen3-Coder
   teacher with this projection configuration. Future CTD work to revisit:
   `student_offset` alignment (full coverage, 1.5-2× compute) and hybrid
   KL+SFT gated on per-position projection mass.
6. **Same-vocab distill, same recipe, this corpus → cannot beat base on HE.**
   Teacher signal regularises but does not lift the student over its
   no-FT prior. To break that ceiling we need one of: (a) cross-vocab
   stronger teacher (M6), (b) more capacity (M7: rank 64 + 4 ep), (c) more
   data (M8: mixed corpus). M3-M5 are already a successful regularising
   distill — they're just not enough to *beat* base by themselves.

## Loss objectives — implementation notes

Reference implementation:
[`ctd/on_policy_loss.py`](https://github.com/mann1x/cross-tokenizer-distill/blob/main/ctd/on_policy_loss.py).

Each loss is computed *only* at student-sampled continuation positions
(prompt positions are masked out). Inputs: `student_logits`,
`teacher_logits`, both `[B, L, V]`; `mask` is `[B, L]` 1.0 for positions
where the loss applies. `T` is the temperature for both distributions
(default 1.0; Hinton T² rescale baked in).

### Forward KL (FKL, DistillSpec)

```
KL(P_T || P_S) = Σ p_T (log p_T − log p_S)
```

- Mode-covering: pulls the student to assign mass everywhere the teacher does.
- This is what we recommend by default. Best of M3/M4/M5 on HE.

### Reverse KL (RKL, MiniLLM)

```
KL(P_S || P_T) = Σ p_S (log p_S − log p_T)
```

- Mode-seeking: encourages the student to concentrate on the teacher's
  *peak* mode. On a small student with a large vocab this collapses
  probability mass too aggressively.
- M4 was the **worst of the three** here: −5.5 pp HE vs −3.7 pp for FKL.
- Use only when the teacher is much narrower than you want the student to
  end up — not the typical small-student case.

### Generalized JSD (GKD)

```
M = β·P_T + (1−β)·P_S
JSD = β·KL(P_T||M) + (1−β)·KL(P_S||M)
```

- Symmetric blend; β=0.5 is the standard GKD choice.
- M3 with β=0.5 sat between FKL and RKL (−4.3 pp) — "average of the two
  effects" is roughly what you observe.
- Worth tuning if FKL gives you over-broad outputs; we did not see
  meaningful wins from β-sweeps at this scale.

### Hybrid (α·FKL + (1−α)·RKL)

- Convenient knob if you want to dial mode-covering vs mode-seeking
  independently of GKD's symmetric mixture.
- We did not test in M3-M5 (no time); included in `ctd/on_policy_loss.py` for
  ablation use.

## On-policy vs off-policy

**On-policy** = student samples its own continuations, both models forward
on (prompt + student-sampled continuation), KL computed at those positions.

**Off-policy** = teacher cache built once on a fixed corpus, student trained
on the same fixed corpus — no sampling at training time.

In our same-vocab tests on this corpus:
- M3 (off-policy GKD on cached teacher) and M5 (on-policy FKL with live
  teacher forward) finished within ~1 pp of each other on HE.
- On-policy is preferred when the student is expected to *behave differently*
  from the corpus distribution at deployment time (e.g. RLHF-trained student
  vs base-text corpus). For a corpus that already matches deployment
  distribution, off-policy is cheaper and equivalent.

For Mythic-RDT v6U-style work where the student is being *modified*
during distillation (recurrent wrapper), on-policy is the safer default
because the student's policy drifts as recurrence takes effect.

## Cross-tokenizer specifics (when teacher and student vocabs differ)

When the strongest available teacher uses a different tokenizer than the
student (e.g. Qwen → DS-Coder, Llama → Mistral), use the
[CTD library](https://github.com/mann1x/cross-tokenizer-distill):

1. Build a `VocabMapper` once (string-level mapping with multi-token
   "distribute" strategy is the practical default).
2. At training time, decode the student's sampled continuation, re-tokenize
   for the teacher, build a per-example byte-anchor alignment, project the
   teacher's top-K logits to student vocab via the cached mapper, and
   compute KL only at aligned student positions.
3. Expect ~30-50 % of student positions to be dropped under `byte_anchor`
   alignment for divergent tokenizer pairs (e.g. our Qwen2.5-Coder-7B →
   DS-Coder-1.3B run kept ~64 % of positions). For full-coverage,
   `student_offset` mode re-encodes the suffix string at every position
   (~1.5-2× compute).

The mapper coverage report tells you *before* spending compute whether the
projection is going to work for a given pair. From CTD's
[`docs/RESULTS.md`](https://github.com/mann1x/cross-tokenizer-distill/blob/main/docs/RESULTS.md):

| Teacher → Student | Single-token | Multi-token | Dropped |
|---|---|---|---|
| Qwen2.5-Coder-7B (152 K) → DS-Coder-1.3B (32 K) | 19.1 % | 80.9 % | 0.0 % |

If "Dropped" is meaningfully > 10 %, the loss is going to be dominated by
the projected portion only — re-evaluate before spending the GPU budget.

## Anti-patterns to avoid

| Don't | Why |
|---|---|
| Skip `T²` rescale on logits when using temperature softening | Hinton's gradient-magnitude correction; without it the loss scale changes with T and your LR sweep gets tied to T |
| Trust HE-20 / MBPP-20 smokes for recipe ranking | n=20, SE ~11 pp at 50 % accuracy. You will pick wrong recipes |
| Use reverse KL with a tiny student and a much larger teacher | Mode-seeking collapses; we lost 1.2 pp HE this way (M4) |
| Run distill **without** an SFT baseline at the same recipe | You can't tell if your KL is helping or just running |
| Ignore prompt-positions in the loss mask | The student's prompt distribution is not the teacher's training distribution; including those positions adds noise |
| Use `--mbpp-limit 0` thinking it means "skip MBPP" | It means "all of MBPP" — use `--skip-mbpp` |
| Run any LoRA distill without `target_modules="all-linear"` for code tasks | Selective targets (q/v only) under-fit on code: every linear in the block carries syntax mass |

## Open questions / next experiments

These are next on the validation pod (results will land in CTD's
`docs/RESULTS.md` and reflected back here):

1. **M6 — cross-vocab on-policy CTD (in flight).** Does the CTD projection
   recover same-vocab quality? Decision gate for using a Qwen3-Coder
   teacher on Mythic-RDT v6U.
2. **M7 — capacity test.** Same M3 recipe with rank=64 and 4 epochs. Tests
   whether the "no recipe beats base" ceiling is a capacity issue.
3. **M8 — mixed corpus.** MBPP-train + ~1500 synthetic HE-style prompts
   from the base model. Tests whether the corpus is the bottleneck — at
   374 examples the student may be running out of distribution coverage
   before the teacher signal can lift it.

## See also

- [`mann1x/cross-tokenizer-distill`](https://github.com/mann1x/cross-tokenizer-distill)
  — the projection / alignment library these experiments depend on.
- `docs/METHOD_omnimerge_v2.md` — model-merge recipes that pair well with
  the distill recipes here (merge first, distill second is our
  recommended order for OmniMergeKit recipes).
- `docs/METHOD_competence_pipeline.md` — competence-map analysis used to
  pick the *teacher* in the first place.
