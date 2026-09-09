#!/usr/bin/env python3
"""Efficiency reward: the an-finetune GRPO shape, with a PER-GROUP length budget.

Knowledge doc (READ FIRST, keep updated): docs/METHOD_grpo_efficiency.md

WHY THIS EXISTS SEPARATELY FROM gepo_reward_v2
----------------------------------------------
`gepo_reward_v2` scores length GROUP-RELATIVELY and is deliberately SCALE-FREE:

    s_i = clip((med - len_i)/scale, -1, +1)      r_i = BASE + ALPHA*lam*s_i

It never reads `meta.length_budget`. Writing a budget onto a pool row and then running
v2 produces a field that no consumer ever reads -- the pool looks configured, the run
looks healthy, and the budget measured nothing.
[[feedback_a_computed_field_that_never_reaches_a_table_is_unmeasured]]

The method that actually worked on an-finetune (E2B: mean_length 779.8 -> 683.9,
-12.3% in one epoch, no score damage) is the ABSOLUTE-BUDGET shape:

    r_i = (1 - lam * min(ntok_i / budget, 1)) if correct else 0

`simpo/train_grpo_e2b.py:100`. Those two rewards are different mechanisms, not two
spellings of one:

  * v2 (relative) always has a gradient as long as the group's lengths differ at all,
    and by construction cannot say "this length is fine, stop pushing".
  * this one (absolute) creates a THRESHOLD. Rollouts under budget sit on the sloped
    part of the penalty and are differentiated; rollouts over budget all clip to
    lenpen = 1.0 and are NOT. Where the budget sits relative to the group's length
    distribution IS the whole design.

AN's operating point is the thing to copy, and it is not the script's default:
budget 512 against a mean length of 780 -- the budget sits at ~0.66x the median, BELOW
it, so a healthy minority of rollouts land on the sloped part each group and the
gradient says "get under the bar", while the ones far over are not asked to compete
with each other on length.

THE TWO WAYS A BUDGET SILENTLY DELETES ITS OWN SIGNAL
-----------------------------------------------------
Both end with a length term that has ZERO within-group variance, which under
`scale_rewards='group'` cancels EXACTLY and contributes no gradient at all:

  budget WAY BELOW the distribution -> every rollout clips to lenpen = 1.0
      r_i = (1 - lam) for every passer. Constant. Cancels.
  budget WAY ABOVE the distribution -> every lenpen is a tiny, near-identical number
      r_i ~ 1 for every passer. Near-constant. Signal drowned by the pass/fail term.

Neither shows up in the loss curve. Both show up in `frac_under_budget` and
`length_share`, which this module logs per tier for exactly that reason. A tier whose
`frac_under_budget` is ~0.00 or ~1.00 is not being taught anything about length,
however carefully its lambda was chosen.

BECAUSE THAT PLACEMENT IS PER-TIER, THE BUDGET IS PER-ROW.
A single global budget cannot sit at 0.66x the median of two tiers whose medians
differ by 8x (efficiency ~1,315 tok thinking; gpqa no-think ~1,595; lcb_exec thinking
~6,000+). One number would put one tier on the slope and pin the others at a constant.
So `meta.length_budget` is REQUIRED on every row with lambda > 0, and its absence is a
hard refusal rather than a fallback -- a fallback is how a tier ends up measured
against someone else's operating point.

TRUNCATION IS NOT A LENGTH MEASUREMENT.
A rollout at the completion cap is CENSORED: its true length is unknown and >= cap.
It is counted in `clipped` and reported, never treated as a short answer. Run the
trainer with `mask_truncated_completions=True` so censored rollouts do not enter the
advantage at all; this module additionally refuses to credit one as a passer.
"""
from __future__ import annotations

import os
import re
import statistics as st
import subprocess
import sys
from typing import Any, Callable

# DDP rank, so a per-rank tier view is never mistaken for the population.
# [[feedback_zero_assertions_need_a_population_witness]]
RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))

# Reward for a correct, zero-length answer. Kept at 1.0 so the reward reads exactly as
# the an-finetune formula and a failure is 0.0 with no offset to reason about.
R_CORRECT = 1.0
LAMBDA_MAX = 1.0

MC_RE = re.compile(r"correct answer is[^A-Da-d]*\(?([A-Da-d])\)?")

# The format bonus is deliberately TINY (an-finetune used the same shape as a separate
# reward function). It exists to keep the answer marker present while length pressure
# pushes down, not to compete with correctness. Anything large enough to matter against
# a 1.0 correctness term would be a second objective, not a nudge.
FORMAT_BONUS = 0.05


# WHERE TO PUT THE BUDGET. Not "0.66x the median" -- that is AN's *realized ratio*,
# not a rule, and it silently degenerates. For a tight distribution (say lengths
# 3000-6000, min/median = 0.67) a budget at 0.66x the median falls BELOW THE MINIMUM,
# every rollout clips to lenpen = 1.0, and the length term is constant -> no gradient.
# The property that actually has to hold is about the LEFT TAIL, so state it directly:
# put the budget at a low quantile of the group's own measured passing lengths, which
# makes `frac_under_budget` come out at ~q by construction.
BUDGET_QUANTILE = 0.35


def budget_from_lengths(lengths, q: float = BUDGET_QUANTILE) -> int:
    """Budget for a tier, from that tier's MEASURED passing lengths.

    Uses the empirical quantile (no interpolation beyond the data) so the budget is
    always a length some rollout actually achieved -- a budget below every observed
    length is not a target, it is an off switch.
    """
    xs = sorted(int(x) for x in lengths if x and x > 0)
    if not xs:
        raise ValueError("REFUSE: cannot derive a budget from zero measured lengths. "
                         "Run the smoke first; do not guess a budget.")
    if not 0.0 < q < 1.0:
        raise ValueError(f"REFUSE: budget quantile must be in (0,1), got {q!r}")
    idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def _compiles(src: str) -> bool:
    try:
        compile(src, "<cand>", "exec")
        return True
    except Exception:
        return False


def extract_code(text: str) -> str:
    """Strip prose/markdown down to something that compiles, else return as-is."""
    cand = text or ""
    fence = re.search(r"```(?:python)?\n(.*?)```", cand, re.DOTALL)
    if fence and _compiles(fence.group(1)):
        return fence.group(1)
    if _compiles(cand):
        return cand
    m = re.search(r"^[ \t]*(?:import |from |class |def |@)", cand, re.MULTILINE)
    if m and _compiles(cand[m.start():]):
        return cand[m.start():]
    i = cand.find("def ")
    return cand[i:] if i >= 0 else cand


# --------------------------------------------------------------------- verifiers
def verify_mc_letter(text: str, gold: str) -> bool:
    """Last stated answer wins: reasoning models restate and revise before committing."""
    hits = MC_RE.findall(text or "")
    if not hits:
        return False
    return hits[-1].upper() == (gold or "").strip().upper()


def verify_mbpp_exec(text: str, meta: dict, timeout: float = 10.0) -> bool:
    """Run the candidate against MBPP's asserts in a separate interpreter."""
    tests = meta.get("tests") or []
    if not tests:
        return False
    src = "\n".join([extract_code(text), meta.get("test_setup_code") or "", *tests])
    try:
        p = subprocess.run([sys.executable, "-c", src], timeout=timeout,
                           capture_output=True)
        return p.returncode == 0
    except Exception:
        return False


def has_answer_marker(text: str, kind: str) -> bool:
    """Cheap format check, per kind. Used only for the tiny format bonus."""
    if kind == "mc_letter":
        return bool(MC_RE.search(text or ""))
    return "```" in (text or "") or "def " in (text or "")


# --------------------------------------------------------------------- the reward
def make_efficiency_reward(tokenizer,
                           lcb_verifier: Callable[[str, dict], Any] | None,
                           max_completion: int,
                           log_every: int = 0,
                           allow_unset_budget: bool = False,
                           dump_rollouts: str | None = None):
    """Build the reward callable TRL will invoke.

    `max_completion` is the trainer's completion cap and is used ONLY to identify
    censored rollouts. It is never used as a length budget: the budget is per row.
    """
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    if allow_unset_budget:
        # MEASURE-ONLY MODE. There is a genuine ordering constraint here: the budget is
        # a QUANTILE of that tier's measured passing lengths, so it cannot exist before
        # the measurement, and the measurement run therefore cannot apply a length
        # term. In this mode a lambda>0 row with no budget scores pure correctness and
        # its passing lengths are recorded. It is NOT a relaxation of the guard -- the
        # refusal stays hard for any run that is actually training a driver tier.
        print(">>> grpo_reward_efficiency: MEASURE-ONLY (allow_unset_budget=True). "
              "Rows with lambda>0 and no budget score correctness only; their passing "
              "lengths are recorded so budgets can be derived. NEVER use for a real "
              "run -- every driver tier would train as pure correctness.", flush=True)
    print(">>> grpo_reward_efficiency: ABSOLUTE per-row budget "
          f"(r = 1 - lambda*min(ntok/meta.length_budget, 1) if correct else 0), "
          f"R_CORRECT={R_CORRECT} FORMAT_BONUS={FORMAT_BONUS} "
          f"max_completion={max_completion}", flush=True)
    state: dict[str, Any] = {"n": 0, "byk": {}}

    def ntoks(text: str, cid) -> int:
        if cid is not None:
            try:
                return len(cid)
            except TypeError:
                pass
        return len(tk(text, add_special_tokens=False).input_ids)

    def verify(kind: str, text: str, meta: dict, gold: str) -> bool:
        if kind == "lcb_exec":
            if lcb_verifier is None:
                raise ValueError(
                    "REFUSE: pool carries lcb_exec rows but no lcb_verifier was "
                    "supplied. Those rows would score 0.0 for every rollout: no "
                    "within-group spread, no gradient, and a tier that drags the "
                    "policy while looking like a normal training tier.")
            ok, _ = lcb_verifier(text, meta)
            return bool(ok)
        if kind == "mc_letter":
            return verify_mc_letter(text, gold)
        if kind == "mbpp_exec":
            return verify_mbpp_exec(text, meta)
        # NEVER fall through to 0.0 -- see the module docstring.
        raise ValueError(
            f"REFUSE: no verifier registered for reward_kind={kind!r}.")

    def group_bounds(prompts: list) -> list[tuple[int, int]]:
        """TRL emits (B*G,) with a prompt's G generations contiguous. Derive the runs
        rather than trusting a passed-in G: a wrong G silently mixes two problems into
        one group."""
        out, s = [], 0
        for i in range(1, len(prompts) + 1):
            if i == len(prompts) or prompts[i] != prompts[s]:
                out.append((s, i))
                s = i
        return out

    def reward(completions, prompts=None, gold=None, meta=None,
               completion_ids=None, **kwargs) -> list[float]:
        n = len(completions)
        golds = list(gold or [""] * n)
        metas = list(meta or [{}] * n)
        cids = list(completion_ids or [None] * n)
        texts = [c if isinstance(c, str) else (c[-1].get("content", "") if c else "")
                 for c in completions]

        rewards = [0.0] * n
        # Per-index length bookkeeping, kept so the GROUP loop below can classify each
        # group and so --dump-rollouts can emit an exact per-rollout record. Without
        # these, `rewards` alone cannot tell a group tied at the clamp from a group that
        # genuinely agrees.
        # A caller that passes mismatched lengths gets a NAMED refusal, not an
        # IndexError five frames down in the group census. The census indexes ok_i by
        # bounds derived from `prompts`, so prompts longer than completions reads off
        # the end -- which is exactly how test 2 silently never ran.
        if prompts is not None and len(list(prompts)) != n:
            raise ValueError(
                f"REFUSE: {len(list(prompts))} prompts for {n} completions. The group "
                "census derives bounds from prompts and indexes per-completion arrays "
                "with them; a mismatch cannot be interpreted.")
        nt_i: list[int] = [0] * n
        ok_i: list[bool] = [False] * n
        under_i: list[bool | None] = [None] * n
        pen_i: list[float] = [0.0] * n      # the CLAMPED lenpen actually applied
        raw_i: list[float] = [0.0] * n      # nt/budget, UNCLAMPED
        for i in range(n):
            m = metas[i] or {}
            kind = m.get("reward_kind")
            lam_raw = m.get("length_lambda", 0.0)
            lam = min(max(float(lam_raw if lam_raw is not None else 0.0), 0.0),
                      LAMBDA_MAX)
            nt = ntoks(texts[i], cids[i])
            censored = nt >= max_completion

            ok = (not censored) and verify(kind, texts[i], m, golds[i])

            if not ok:
                r = 0.0
            elif lam <= 0.0:
                # Replay tier: pure correctness. No length term at all.
                r = R_CORRECT
            else:
                budget = m.get("length_budget")
                if (budget is None or float(budget) <= 0) and allow_unset_budget:
                    r = R_CORRECT          # measure-only: no length term yet
                elif budget is None or float(budget) <= 0:
                    raise ValueError(
                        f"REFUSE: row {m.get('question_sha256') or kind!r} carries "
                        f"length_lambda={lam} but length_budget={budget!r}. An absolute "
                        "budget reward has no meaning without a budget, and a global "
                        "fallback would measure this tier against another tier's "
                        "operating point. Set meta.length_budget per row.")
                else:
                    raw = nt / float(budget)
                    lenpen = min(raw, 1.0)
                    r = R_CORRECT - lam * lenpen
                    pen_i[i], raw_i[i] = lenpen, raw
                    under_i[i] = nt < float(budget)
            if ok and has_answer_marker(texts[i], kind):
                r += FORMAT_BONUS
            rewards[i] = r
            nt_i[i], ok_i[i] = nt, bool(ok)

            b = state["byk"].setdefault(
                f"{kind}/{'T' if m.get('think') else 'N'}",
                {"n": 0, "pass": 0, "tok": 0, "clip": 0, "under": 0,
                 "lam": lam, "budget": m.get("length_budget"), "r": 0.0,
                 # PASSING, non-censored lengths only. A budget derived from all
                 # rollouts would be pulled down by failures (often short, degenerate)
                 # and up by censored ones (whose true length is unknown), so it would
                 # not describe the distribution the length term actually acts on.
                 "pass_lens": []})
            b["n"] += 1
            b["tok"] += nt
            b["r"] += r
            if censored:
                b["clip"] += 1
            if ok:
                b["pass"] += 1
                b["pass_lens"].append(nt)
                if lam > 0 and m.get("length_budget"):
                    if nt < float(m["length_budget"]):
                        b["under"] += 1

        # ---- the diagnostic that killed v1 -------------------------------------
        # Within a group, GRPO normalises advantage. What matters is not the reward
        # scale but the SHARE of within-group variance the length term owns. v1 owned
        # 11.3% at its operating point and that share SHRANK as the model converged.
        # Computed here per group and averaged, so a run can be stopped on it rather
        # than discovered in the eval.  [[feedback_gate_on_the_component_not_the_aggregate_indicator]]
        if prompts is not None:
            shares, tot_stds = [], []
            # ---- GROUP CLASSIFICATION (test A) -------------------------------------
            # `lenpen = min(nt/budget, 1.0)` ties every passer at or above budget at the
            # SAME penalty, so a group whose passers are ALL above budget carries no
            # length contrast at all. That -- not "65% of passers" -- is the clamp's real
            # cost, and it is measurable here rather than assumed. Groups with <2 passers
            # are dead too, but for an unrelated reason (pass rate), so they are counted
            # SEPARATELY: merging the two would blame the clamp for a pass-rate problem.
            # PER TIER, never pooled. The pool is 60/40 driver/replay over tiers with
            # DIFFERENT jobs: the efficiency driver (manic-arm-contrast think,
            # lcb_v6_easy think) and the BREVITY replay (gpqa_main_minus_diamond
            # no-think). A pooled census reports the mean of three different jobs and
            # cannot see the minority tier that carries the objective going dead -- on
            # 2026-09-09 it read "clamp is the minor failure mode, 12.5%" when the
            # clamp was in fact concentrated on the replay.
            # [[feedback_a_pool_wide_metric_cannot_see_a_minority_tier_objective]]
            gcls = state.setdefault("gcls", {})
            rows = []
            for s, e in group_bounds(list(prompts)):
                grp = rewards[s:e]
                if len(grp) < 2:
                    continue
                passers = [x for x in grp if x > 0.0]
                pidx = [j for j in range(s, e) if ok_i[j]]
                n_under = sum(1 for j in pidx if under_i[j] is True)
                m0 = metas[s] or {}
                cls = gcls.setdefault(
                    f"{m0.get('reward_kind')}/{'T' if m0.get('think') else 'N'}",
                    {"lt2pass": 0, "allover": 0, "graded": 0})
                if len(pidx) < 2:
                    cls["lt2pass"] += 1
                elif n_under == 0:
                    cls["allover"] += 1
                else:
                    cls["graded"] += 1
                tot = st.pstdev(grp)
                tot_stds.append(tot)
                if len(passers) >= 2 and tot > 0:
                    shares.append(st.pstdev(passers) / tot)
                # ---- ROLLOUT DUMP (test B) ----------------------------------------
                # One row per rollout, carrying everything needed to RE-SCORE the same
                # rollouts under a different penalty shape offline: the unclamped ratio
                # raw=nt/budget and the clamped pen actually applied. The counterfactual
                # reward is then exactly r + lam*pen - lam*f(raw) for any f -- no
                # regeneration, no GPU, no second training run.
                if dump_rollouts:
                    gid = state.get("gid", 0)
                    state["gid"] = gid + 1
                    for j in range(s, e):
                        mj = metas[j] or {}
                        rows.append({
                            "gid": gid, "rank": RANK,
                            "tier": f"{mj.get('reward_kind')}/"
                                    f"{'T' if mj.get('think') else 'N'}",
                            "ntok": nt_i[j], "ok": ok_i[j], "r": rewards[j],
                            "lam": float(mj.get("length_lambda") or 0.0),
                            "budget": mj.get("length_budget"),
                            "pen": pen_i[j], "raw": raw_i[j],
                            "censored": nt_i[j] >= max_completion,
                        })
            if rows:
                import json as _json
                with open(f"{dump_rollouts}.rank{RANK}.jsonl", "a") as fh:
                    for rec in rows:
                        fh.write(_json.dumps(rec) + "\n")
            state.setdefault("share", []).extend(shares)
            state.setdefault("tot", []).extend(tot_stds)

        state["n"] += n
        if log_every and state["n"] >= log_every:
            state["n"] = 0
            sh = state.get("share") or []
            prev = state.setdefault("prev", {})
            print(f">>> [rank {RANK}] efficiency-reward tiers:", flush=True)
            for k, b in sorted(state["byk"].items()):
                frac_under = (b["under"] / b["pass"]) if b["pass"] else float("nan")
                print(f"      {k:16s} n={b['n']:5d} pass={b['pass'] / max(b['n'],1):.3f} "
                      f"tok={b['tok'] / max(b['n'],1):7.0f} clip={b['clip'] / max(b['n'],1):.3f} "
                      f"lam={b['lam']:.2f} budget={b['budget']} "
                      f"frac_under_budget={frac_under:.3f} "
                      f"mean_r={b['r'] / max(b['n'],1):.3f}", flush=True)
                # ---- SINCE THE LAST PRINT ---------------------------------------
                # The row above is CUMULATIVE: byk is never cleared, only the print
                # trigger state["n"] is. A running mean moves a shrinking fraction of
                # the way the underlying quantity moves, so reading those rows as a
                # per-step trend UNDERSTATES every change -- and "is length falling?"
                # is the one question this log exists to answer. Print the interval
                # next to the cumulative so nobody has to difference it offline.
                # [[feedback_a_pool_wide_metric_cannot_see_a_minority_tier_objective]]
                p = prev.get(k)
                if p and b["n"] > p["n"]:
                    dn = b["n"] - p["n"]
                    dp = b["pass"] - p["pass"]
                    dfu = ((b["under"] - p["under"]) / dp) if dp > 0 else float("nan")
                    print(f"      {'':16s}   since={dn:5d} pass={dp / dn:.3f} "
                          f"tok={(b['tok'] - p['tok']) / dn:7.0f} "
                          f"clip={(b['clip'] - p['clip']) / dn:.3f} "
                          f"frac_under_budget={dfu:.3f} "
                          f"mean_r={(b['r'] - p['r']) / dn:.3f}", flush=True)
                prev[k] = {q: b[q] for q in ("n", "pass", "tok", "clip", "r", "under")}
            if sh:
                mark = state.get("share_mark", 0)
                fresh = sh[mark:]
                state["share_mark"] = len(sh)
                iv = (f" interval={sum(fresh) / len(fresh):.3f} over {len(fresh)}"
                      if fresh else "")
                print(f"      length_share (within-group, among passers): "
                      f"mean={sum(sh)/len(sh):.3f} n_groups={len(sh)}{iv}  "
                      f"[v1 died at 0.113 and falling]", flush=True)
            gc = state.get("gcls") or {}
            gprev = state.setdefault("gcls_prev", {})
            for tk in sorted(gc):
                g = gc[tk]
                tot_g = sum(g.values())
                if not tot_g:
                    continue
                # Cumulative AND interval: gcls is never cleared, so the cumulative
                # share is a running mean that lags a real change for many steps.
                # [[feedback_a_pool_wide_metric_cannot_see_a_minority_tier_objective]]
                pv = gprev.get(tk, {"lt2pass": 0, "allover": 0, "graded": 0})
                dn = sum(g[q] - pv[q] for q in g)
                iv = ""
                if dn:
                    iv = ("  | since: graded=%.3f CLAMP-DEAD=%.3f dead=%.3f over %d"
                          % ((g["graded"] - pv["graded"]) / dn,
                             (g["allover"] - pv["allover"]) / dn,
                             (g["lt2pass"] - pv["lt2pass"]) / dn, dn))
                print(f"      group classes {tk:<14} graded={g['graded']} "
                      f"({g['graded'] / tot_g:.3f})  "
                      f"CLAMP-DEAD(all passers over budget)={g['allover']} "
                      f"({g['allover'] / tot_g:.3f})  "
                      f"dead(<2 passers, NOT the clamp)={g['lt2pass']} "
                      f"({g['lt2pass'] / tot_g:.3f}){iv}", flush=True)
                gprev[tk] = dict(g)
        return rewards

    reward.__name__ = "efficiency_budget_reward"
    reward._state = state  # exposed so the smoke can read tiers without re-parsing logs
    return reward
