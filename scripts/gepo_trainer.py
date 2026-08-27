#!/usr/bin/env python3
"""GEPOTrainer — Group Expectation Policy Optimization (arXiv 2508.17850) over TRL's GRPOTrainer.

Port of an-finetune's `simpo/gepo_trainer.py` from trl 1.8.0 to trl 1.9.0, for a model
that the original could not have run: Qwen3.5-MoE 27B, 248k vocab, 16k completions.

WHAT GEPO IS
------------
GEPO is a GSPO-family method: the importance weight is per-SEQUENCE (length-normalized),
and GEPO replaces GSPO's per-sample denominator q_i with the GROUP EXPECTATION over the
G rollouts of one prompt:

    Ê_q[q] = Σ_j q_j^2 / Σ_j q_j          (paper Eq. 2; group = the G completions of a prompt)

Working with length-normalized sequence log-probs
    lp_i = (Σ_t logπ_θ,t · mask) / |y_i|   (differentiable numerator)
    lq_i = (Σ_t logπ_old,t · mask) / |y_i| (detached sampler)
so q_i = exp(lq_i) is the geometric-mean per-token probability, and the group expectation
is taken in log-space to avoid underflow:

    log Ê_q = logsumexp(2·lq_group) − logsumexp(lq_group)

GEPO log-importance-weight = lp_i − log Ê_q. The denominator is detached; gradient flows
only through lp_i. This mirrors GEPO Listing 1:
    coeff = learner_seq_p / (hat_q · sampler_seq_p).sum()

Only the sequence-branch importance weight changes. Advantage, reward, KL and clipping
structure are GRPO's. GEPO's "no explicit PPO clip" is realized by loosening
epsilon_low/epsilon_high (the launcher does this), so loss_type="grpo" collapses to −coeff·A.

THE ONE DELIBERATE DEVIATION FROM THE 1.8.0 ORIGINAL
----------------------------------------------------
The original computed the group reduction INSIDE `_compute_loss`, by reshaping the
micro-batch to (B//G, G). That requires every micro-batch to contain whole prompt-groups,
i.e. per_device_train_batch_size % num_generations == 0. On E2B at bsz 8 that was free.

Here it is impossible. The logits tensor for ONE sequence at 16k tokens is
16384 × 248320 × 2 B = 8.1 GiB, so this model admits a micro-batch of 1–2 sequences, never
8. Forcing whole groups into a micro-batch would mean capping completions to a few
thousand tokens -- but CoderX's CORRECT LiveCodeBench solutions run 3k-11k tokens, so a cap
truncates most rollouts, every truncated rollout fails execution, the whole group scores 0,
the advantage is uniformly 0, and the gradient dies. That is the documented
`grpo_completion_cap_kills_gradient` failure, and it would look like "GEPO did nothing"
rather than like a broken cap.

So log Ê_q is instead computed ONCE over the full generation group in
`_generate_and_score_completions` and carried per-sequence in `inputs["gepo_log_Eq"]`.
This is EXACT, not an approximation: log Ê_q is a detached constant per prompt-group, so
evaluating it over the full group and indexing it per micro-batch is identical to
evaluating it inside a micro-batch that happened to hold the whole group -- and it is
strictly better, because it stays correct when the micro-batch does NOT hold the group.

It does require `old_per_token_logps` to be materialized. TRL only skips that when
num_iterations == 1 AND steps_per_generation <= gradient_accumulation_steps AND vLLM is
off (grpo_trainer.py:41-47). We always run with vLLM, which forces it for the vLLM
importance-sampling correction, and `_require_old_logps()` below refuses loudly rather
than silently degrading if that ever stops being true.

THE COPY, AND WHY IT IS GATED
-----------------------------
`_compute_loss` is copied VERBATIM from trl 1.9.0's GRPOTrainer with only the
`importance_sampling_level == "sequence"` branch changed. Pin-and-copy is the transparent
way to add a loss variant TRL does not ship, but a copy silently rots when TRL changes the
surrounding code. So `assert_only_sequence_branch_differs()` re-extracts upstream's source
at import time and proves the diff is confined to that branch. A TRL bump that touches
anything else in `_compute_loss` fails the import instead of training on a stale copy.
"""
from __future__ import annotations

import difflib
import inspect
import re
import textwrap

import torch
import trl
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import nanmax, nanmin

_EXPECTED_TRL = "1.9.0"

# The exact upstream sequence-branch BODY this port replaces. Kept as data so the
# import-time gate can prove the ONLY thing we changed is these lines.
#
# The `elif self.importance_sampling_level == "sequence":` header is deliberately NOT here:
# our version keeps it verbatim, so the differ classifies it as unchanged and it never
# appears in the removed set. Including it made the gate refuse its own correct copy.
_UPSTREAM_SEQ_BRANCH_BODY = """\
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
"""


class GEPOTrainer(GRPOTrainer):
    """GRPOTrainer with the GEPO group-expectation sequence importance weight.

    Requires importance_sampling_level == "sequence". `self.gepo` gates the group-expectation
    denominator; with gepo=False this class is behaviourally identical to stock GSPO, which
    makes it usable as its own control arm.
    """

    gepo = True

    # ---------------------------------------------------------------- group expectation
    def _require_old_logps(self, inputs) -> torch.Tensor:
        old = inputs.get("old_per_token_logps")
        if old is None:
            raise ValueError(
                "GEPO needs old_per_token_logps, and TRL did not compute it. That happens "
                "when num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps "
                "and vLLM is off. Without it the sampler log-probs do not exist as a separate "
                "tensor and log(E_q) would silently collapse to the policy's own logps, which "
                "is GSPO, not GEPO. Enable vLLM (or raise num_iterations) rather than letting "
                "this pass."
            )
        return old

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if not getattr(self, "gepo", False):
            return out
        if self.importance_sampling_level != "sequence":
            raise ValueError(
                f"GEPO requires importance_sampling_level='sequence', got "
                f"{self.importance_sampling_level!r}. The group-expectation denominator is only "
                "defined for the sequence-level weight."
            )

        old = self._require_old_logps(out)
        mask = out["completion_mask"]
        if "tool_mask" in out:
            mask = mask * out["tool_mask"]
        seqlen = mask.sum(-1).clamp(min=1.0)
        lq = (old * mask).sum(-1) / seqlen                      # (B,) detached sampler seq logp

        G = int(self.num_generations)
        B = lq.shape[0]
        if B % G != 0:
            raise ValueError(
                f"GEPO group reduction needs the GENERATION batch {B} divisible by "
                f"num_generations {G}. This is the full generation batch, not a micro-batch, "
                "so it should always hold; if it does not, the generation path changed."
            )
        # log E_q[q] = log( Sum_j q_j^2 / Sum_j q_j ), q_j = exp(lq_j), computed in log-space.
        lq_g = lq.detach().reshape(B // G, G)
        log_Eq = torch.logsumexp(2.0 * lq_g, dim=1) - torch.logsumexp(lq_g, dim=1)   # (P,)
        out["gepo_log_Eq"] = log_Eq.repeat_interleave(G)                              # (B,)
        return out

    # ---------------------------------------------------------------- copied loss
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            # === GEPO group-expectation denominator (the ONLY edit vs upstream) ===
            # GSPO: lp - lq.  GEPO: lp - log E_q[q], with log E_q precomputed over the full
            # prompt-group in _generate_and_score_completions (see the module docstring for
            # why it cannot be reduced here). Detached denominator; grad flows via lp only.
            seqlen = mask.sum(-1).clamp(min=1.0)
            lp = (per_token_logps * mask).sum(-1) / seqlen
            if getattr(self, "gepo", False):
                log_Eq = inputs["gepo_log_Eq"].detach().to(lp.device)
                log_importance_weights = (lp - log_Eq).unsqueeze(-1)
            else:
                lq = (old_per_token_logps * mask).sum(-1) / seqlen
                log_importance_weights = (lp - lq).unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
            policy_loss = loss.detach()
        elif self.loss_type == "luspo":
            # Unless importance_sampling_level="token" (not recommended here), per_token_loss is expected to be (B, 1)
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            policy_loss = loss.detach()
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Entropy bonus: add entropy regularization to encourage exploration. _entropy_bonus_enabled is set
        # whenever a non-zero static coef is set OR adaptive mode is enabled (adaptive stays enabled even when
        # entropy_coef has been decremented to entropy_coef_min so it can recover once entropy drops again).
        if self._entropy_bonus_enabled:
            # When top_entropy_quantile < 1.0, entropy_mask restricts policy gradients to high-entropy
            # tokens. Use the same effective mask for the entropy bonus so it acts on the same tokens.
            effective_mask = mask if entropy_mask is None else mask * entropy_mask
            # Entropy bonus = mean per-token entropy H (the documented objective L = L_policy - coef * H), so
            # H does not depend on how each loss type normalizes its policy term. The term is computed so that
            # it accumulates to H over the optimizer step for every loss type and matches world_entropy below.
            # The only wrinkle is the normalizer: most loss types divide by the gradient accumulation step
            # count, but cispo/dapo/vespo divide by a global token count.
            if self.loss_type in ["cispo", "dapo", "vespo"]:
                # normalizer is a global token count, so summing the entropies (instead of averaging them
                # again) makes the term accumulate over the optimizer step to the global mean per-token
                # entropy, like the other loss types.
                entropy_loss = (entropies * effective_mask).sum() / normalizer
            else:
                # Mean per-token entropy of active tokens, scaled for gradient accumulation.
                entropy_loss = (entropies * effective_mask).sum() / effective_mask.sum().clamp(min=1.0) / normalizer

            # Apply the coefficient and gating from the end of the previous optimizer step, so that every
            # micro-batch in the current accumulation window applies the same entropy bonus. The adaptive
            # update below only takes effect on the next step.
            if self.use_adaptive_entropy:
                apply_coef = self.entropy_coef if self._last_world_entropy <= self.args.entropy_target else 0.0
            else:
                apply_coef = self.entropy_coef

            loss = loss - apply_coef * entropy_loss

            self._metrics[mode]["policy_loss"].append(self.accelerator.gather(policy_loss).nanmean().item())

            # Adaptive update. Gated on train mode so evaluation cannot mutate the entropy controller state.
            if self.use_adaptive_entropy and mode == "train":
                # Accumulate the entropy sum and active-token count of every micro-batch into a running window
                # buffer, so the controller measures the exact window-global entropy rather than just the last
                # micro-batch (which would be a 1 / gradient_accumulation_steps subsample).
                stats = torch.stack([(entropies * effective_mask).sum(), effective_mask.sum()]).detach()
                if self._entropy_window_stats is None:
                    self._entropy_window_stats = stats
                else:
                    self._entropy_window_stats = self._entropy_window_stats + stats
                # At the optimizer-step boundary, reduce the window totals across ranks (sum and token count
                # jointly, for a true global mean unbiased when ranks have different completion lengths),
                # update the coefficient for the next step, then reset the window buffer.
                if self.accelerator.sync_gradients:
                    window_stats = self.accelerator.reduce(self._entropy_window_stats, reduction="sum")
                    world_entropy = (window_stats[0] / window_stats[1].clamp(min=1.0)).item()
                    if world_entropy <= self.args.entropy_target:
                        self.entropy_coef = min(
                            self.entropy_coef + self.args.entropy_coef_delta, self.args.entropy_coef_max
                        )
                    else:
                        self.entropy_coef = max(
                            self.entropy_coef - self.args.entropy_coef_delta, self.args.entropy_coef_min
                        )
                    self._last_world_entropy = world_entropy
                    self._entropy_window_stats = None

            # Log entropy_coef on train optimizer-step boundaries (constant for static control; updated just
            # above for adaptive control). sync_gradients is always True in eval (no accumulation context).
            if mode == "train" and self.accelerator.sync_gradients:
                self._metrics[mode]["entropy_coef"].append(self.entropy_coef)

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        # Log the metrics
        def masked_seq_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": already one value per sequence
                return x.squeeze(1)
            return (x * mask).sum(-1) / mask.sum(-1)

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(global_masked_mean(per_token_kl))

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
            gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            self._metrics[mode]["cispo_clip_ratio"].append(global_masked_mean(is_cispo_clipped.float()))
        elif self.loss_type == "vespo":
            self._metrics[mode]["vespo/phi_seq_mean"].append(global_masked_mean(phi_seq))

        return loss


# ---------------------------------------------------------------------- import-time gate
def _norm(src: str) -> list[str]:
    """Dedent + drop blank lines so the diff is about code, not indentation churn."""
    return [ln.rstrip() for ln in textwrap.dedent(src).splitlines() if ln.strip()]


def assert_only_sequence_branch_differs() -> str:
    """Prove our copied `_compute_loss` differs from upstream ONLY in the sequence branch.

    A verbatim copy is the honest way to add a loss TRL does not ship, but it rots silently:
    a TRL bump can change normalization, metrics, or the KL term and the copy would keep
    training on last version's math while reporting success. Re-extracting upstream at import
    and diffing turns that from an invisible drift into a failed import.

    Returns a human-readable summary of what differs.
    """
    if trl.__version__ != _EXPECTED_TRL:
        raise RuntimeError(
            f"GEPOTrainer._compute_loss was copied from trl {_EXPECTED_TRL}; got {trl.__version__}. "
            "Re-extract GRPOTrainer._compute_loss, re-apply the sequence-branch edit, and bump "
            "_EXPECTED_TRL. Do NOT just relax this check."
        )

    up = _norm(inspect.getsource(GRPOTrainer._compute_loss))
    ours = _norm(inspect.getsource(GEPOTrainer._compute_loss))

    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=up, b=ours, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        removed.extend(up[i1:i2])
        added.extend(ours[j1:j2])

    # Everything REMOVED must be the upstream sequence branch, nothing else.
    # Compare on content, not on leading whitespace: textwrap.dedent strips 4 columns from
    # the extracted method source (the `def` indent) but 12 from the standalone literal, so
    # the same two lines would never compare equal with their indentation attached.
    expected_removed = [ln.strip() for ln in _norm(_UPSTREAM_SEQ_BRANCH_BODY)]
    removed = [ln.strip() for ln in removed]
    if removed != expected_removed:
        raise RuntimeError(
            "REFUSE: the copied _compute_loss differs from trl "
            f"{trl.__version__} OUTSIDE the sequence branch.\n"
            f"  removed lines that were not the known branch:\n"
            + "\n".join(f"    - {ln}" for ln in removed[:20])
        )

    # Everything ADDED must live inside the sequence branch we rewrote: it must mention the
    # GEPO denominator and must not touch loss normalization, KL, or metrics.
    joined = "\n".join(added)
    if "gepo_log_Eq" not in joined:
        raise RuntimeError("REFUSE: the replacement branch does not reference gepo_log_Eq.")
    forbidden = [r"\bloss\s*=", r"_metrics\[", r"per_token_kl", r"normalizer"]
    for pat in forbidden:
        if re.search(pat, joined):
            raise RuntimeError(
                f"REFUSE: the replacement branch touches {pat!r}; GEPO must only change the "
                "importance weight, not the loss aggregation."
            )
    return (
        f"GEPO_LOSS_GATE_OK trl={trl.__version__} "
        f"removed={len(removed)} added={len(added)} (sequence branch only)"
    )


if __name__ == "__main__":
    print(assert_only_sequence_branch_differs())
