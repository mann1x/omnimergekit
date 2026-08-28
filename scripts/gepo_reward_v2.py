#!/usr/bin/env python3
"""R9 run4 reward: group-relative brevity + multi-domain replay, with a hard dispatch.

WHY v1 COULD NOT WORK (measured, not suspected)
-----------------------------------------------
v1 was  r = (1 - lam * min(ntok/budget, 1))  if tests pass  else 0,  lam=0.7,
budget=12288. At the operating point run1-run3 actually reached (~4800-6000 tokens):

    pass @ 6000 tok -> 0.658      FAIL -> 0.000
    a 25% length cut (6000 -> 4500) buys +0.085
    => correctness outweighs a 25% length cut by 7.7x

GRPO/GEPO normalises advantage WITHIN a group, so what matters is not the reward
scale but the share of within-group VARIANCE each term owns. For a realistic group
of 8 with 5 passers, v1's numbers are [0,0,0, .704 .670 .653 .624 .601]:

    total group std        0.3160
    std among passers      0.0356   <- the ENTIRE length signal
    length share           11.3%

So ~89% of the gradient said "be correct" and ~11% said "be short". Worse, that 11%
SHRINKS as the model converges in length, so the signal dies exactly when you still
want it. The measured outcome matches: run2 moved mean length 4% across a full epoch,
and its reward rose mostly because fewer rollouts were being zeroed by truncation --
cliff-avoidance, not concision. run3 (adding the router) ended -12.99pp vs the
untrained base on LCB, the only pairwise delta in the cohort that cleared the measured
5.81pp paired SE (13/3 discordant, p=0.0213).

WHAT v2 CHANGES
---------------
Length is scored GROUP-RELATIVELY, among PASSING rollouts only, on a scale-free
statistic:

    P    = passing, non-truncated rollouts in the group
    med  = median(len_i for i in P)
    mad  = median(|len_i - med|)            # robust; a single outlier cannot set the scale
    scale= max(mad, MAD_FLOOR_FRAC*med, 1)  # floor: converged lengths => small signal, by design
    s_i  = clip((med - len_i)/scale, -1, +1)          # +1 shortest, -1 longest
    r_i  = BASE + ALPHA*s_i     (i in P)  |  0.0  (otherwise)

BASE=0.6, ALPHA=0.4 -> passers span [0.2, 1.0], failures 0.0. Two invariants this
preserves, both asserted in the selftest:
  * ORDERING: the worst passer (0.2) still beats the best failure (0.0). Correctness
    is never worth less than brevity -- the v1 property worth keeping.
  * SCALE-FREE: because s_i is normalised by the group's own spread, the length signal
    does NOT decay as the model gets shorter. It only decays when the group's lengths
    genuinely converge, via the MAD floor, which is the correct time to stop pushing.

TRUNCATION IS NOT A LENGTH MEASUREMENT. A rollout at the cap is CENSORED: its true
length is unknown and >= cap. It still scores 0.0 (it is unusable output), but it is
excluded from med/mad so it cannot drag the scale, and clipped_ratio is logged
separately so "fewer truncations" can never again be mistaken for "shorter answers".

REPLAY: lambda=0 MEANS NO LENGTH TERM AT ALL
--------------------------------------------
Replay rows (built by build_gepo_replay_pool.py) carry meta.length_lambda = 0.0 and
meta.reward_kind in {mc_letter, mbpp_exec}. They score pure correctness, 1.0/0.0:
replay exists to HOLD capability, and putting length pressure on the reasoning that
GEPO was already eroding is precisely backwards.

THE DISPATCH REFUSES. An unregistered reward_kind raises. It must never fall through
to 0.0 -- an all-zero tier has no within-group spread, contributes no gradient, and
drags the policy while looking like a normal training tier. That failure is invisible
in the loss curve, which is why it is a hard error here.
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import subprocess
import sys
from typing import Any

# DDP rank, so a per-rank tier view is never mistaken for the population.
RANK = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))

BASE = 0.6            # reward of a median-length passing rollout
ALPHA = 0.4           # half-range of the length term among passers
MAD_FLOOR_FRAC = 0.02  # scale floor as a fraction of the group median
MC_RE = re.compile(r"correct answer is[^A-Da-d]*\(?([A-Da-d])\)?")
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


# --------------------------------------------------------------------- extraction
def completion_text(comp: Any) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        last = comp[-1]
        if isinstance(last, dict):
            return last.get("content") or ""
        return str(last)
    if isinstance(comp, dict):
        return comp.get("content") or ""
    return str(comp or "")


def _compiles(src: str) -> bool:
    try:
        compile(src, "<cand>", "exec")
        return True
    except Exception:
        return False


def extract_code(text: str) -> str:
    """Longest fenced block; for UNFENCED text, the largest prefix-safe code region.

    The fenced path is byte-identical to eval/tasks/_mbpp_utils._extract_code, and that
    path dominates for chat models, so training-time extraction matches how MBPP is
    actually SCORED. No banked eval cell is affected by anything below.

    The UNFENCED fallback is where the two diverge, deliberately. Upstream slices from
    the first `def ` to the end, which silently destroys everything above it:
        mbpp/774  drops `import re` AND a module-level `regex = ...` global
        mbpp/712  drops `import itertools`
        mbpp/927  slices into `class Node:` and leaves an orphaned `__init__`
    Measured consequence: 6 of 25 MBPP reference solutions fail their OWN tests under
    the naive slice. As a bench extractor that costs a few points; as a REWARD it is
    fatal -- those rows score 0.0 for every rollout no matter what the model writes, so
    the group has no spread, contributes no gradient, and quietly drags the policy.
    A reward that rejects its own gold answer is not measuring the model.

    So: use the whole text when it already compiles, else slice from the first line that
    can legally start a module (import/from/class/def/decorator), else fall back to the
    upstream behaviour."""
    blocks = FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    cand = text or ""
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
    """Run the candidate against MBPP's asserts in a separate interpreter.

    subprocess (not exec in-process): a rollout can define __del__, spin a thread, or
    corrupt globals, and the trainer must survive every rollout it scores."""
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


# --------------------------------------------------------------------- the reward
def make_gepo_reward_v2(tokenizer, lcb_verifier, max_completion: int,
                        base: float = BASE, alpha: float = ALPHA,
                        log_every: int = 0):
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    state = {"n": 0, "pass": 0, "tok": 0, "clip": 0, "byk": {}}

    def ntoks(text: str, cid) -> int:
        if cid is not None:
            try:
                return len(cid)
            except TypeError:
                pass
        return len(tk(text, add_special_tokens=False).input_ids)

    def verify(kind: str, text: str, meta: dict, gold: str) -> bool:
        if kind == "lcb_exec":
            ok, _ = lcb_verifier(text, meta)
            return bool(ok)
        if kind == "mc_letter":
            return verify_mc_letter(text, gold)
        if kind == "mbpp_exec":
            return verify_mbpp_exec(text, meta)
        # NEVER fall through to 0.0 -- see the module docstring.
        raise ValueError(
            f"REFUSE: no verifier registered for reward_kind={kind!r}. A pool row whose "
            "reward_kind is unwired would score 0.0 for every rollout: no within-group "
            "spread, no gradient, and it would drag the policy while looking healthy.")

    def group_bounds(prompts: list) -> list[tuple[int, int]]:
        """TRL emits (B*G,) with a prompt's G generations contiguous. Derive the runs
        rather than trusting a passed-in G: a wrong G silently mixes two problems into
        one group, and the length median would then be computed across DIFFERENT tasks."""
        out, s = [], 0
        for i in range(1, len(prompts) + 1):
            if i == len(prompts) or prompts[i] != prompts[s]:
                out.append((s, i))
                s = i
        return out

    def gepo_reward_v2(completions, prompts=None, completion_ids=None, meta=None,
                       gold=None, log_metric=None, **kwargs):
        n = len(completions)
        metas = meta if meta is not None else [None] * n
        golds = gold if gold is not None else [""] * n
        cids = completion_ids if completion_ids is not None else [None] * n
        prm = prompts if prompts is not None else list(range(n))

        rewards = [0.0] * n
        ok_f = [False] * n
        len_f = [0] * n
        clip_f = [False] * n
        lam_f = [1.0] * n
        kind_f = ["lcb_exec"] * n
        tier_f = ["lcb_exec/T"] * n

        for i, comp in enumerate(completions):
            m = metas[i]
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except Exception:
                    m = None
            m = m or {}
            kind_f[i] = m.get("reward_kind", "lcb_exec")
            # TIER label for bookkeeping only -- the DISPATCH still keys on the bare
            # reward_kind. The mixed pool carries the same reward_kind in both thinking
            # modes (mbpp_exec appears as 371 no-think and 100 thinking rows), and those
            # are different failure surfaces: the thinking slice has to fit its chain of
            # thought AND its code inside the same completion cap, so it can die on its
            # own while the no-think slice scores healthily. Aggregated under one key
            # that death is invisible to the launcher's tier gate.
            tier_f[i] = kind_f[i] + (
                "/T" if m.get("think", kind_f[i] == "lcb_exec") else "/N")
            lam_f[i] = float(m.get("length_lambda", 1.0))
            text = completion_text(comp)
            len_f[i] = ntoks(text, cids[i])
            clip_f[i] = len_f[i] >= max_completion
            try:
                ok_f[i] = verify(kind_f[i], text, m, golds[i])
            except ValueError:
                raise
            except Exception as exc:  # a broken rollout must not kill the run
                print(f">>> v2 reward exception (scored 0.0): "
                      f"{type(exc).__name__}: {exc}", flush=True)
                ok_f[i] = False

        for s, e in group_bounds(list(prm)):
            idx = list(range(s, e))
            P = [i for i in idx if ok_f[i]]
            # lambda=0 anywhere in the group => correctness-only tier (replay).
            if P and max(lam_f[i] for i in P) <= 0.0:
                for i in P:
                    rewards[i] = 1.0
                continue
            # censored rollouts carry no length information -- see docstring
            Pl = [i for i in P if not clip_f[i]]
            if not P:
                continue
            if len(Pl) < 2:
                for i in P:
                    rewards[i] = base
                continue
            lens = [len_f[i] for i in Pl]
            med = st.median(lens)
            mad = st.median([abs(x - med) for x in lens])
            scale = max(mad, MAD_FLOOR_FRAC * med, 1.0)
            for i in P:
                if clip_f[i]:
                    # STRICTLY below the uncapped band, not merely at its floor. Setting
                    # this to base-alpha ties a truncated rollout with the longest
                    # COMPLETE one, and the model then gets no gradient separating "ran
                    # into the cap" from "was simply the longest valid answer" -- the
                    # exact conflation that made v1's reward rise on fewer truncations
                    # while lengths barely moved. Still strictly above a failure: a
                    # truncated-but-passing rollout beats a wrong one.
                    rewards[i] = (base - alpha) / 2.0
                    continue
                s_i = max(-1.0, min(1.0, (med - len_f[i]) / scale))
                rewards[i] = base + alpha * s_i

        state["n"] += n
        state["pass"] += sum(ok_f)
        state["tok"] += sum(len_f)
        state["clip"] += sum(clip_f)
        # PER-TIER accounting, not just counts. A replay tier that scores 0.0 for every
        # rollout has no within-group spread, contributes no gradient, and drags the
        # policy -- and the total grad_norm stays non-zero from the LCB tier alone, so
        # no aggregate gate can see it. `nz` (rollouts with reward > 0) is the number
        # the launcher's REPLAY_TIER_ALIVE gate reads back.
        new_kind = False
        for k, r_i in zip(tier_f, rewards):
            if k not in state["byk"]:
                new_kind = True
            d = state["byk"].setdefault(k, {"n": 0, "sum": 0.0, "nz": 0})
            d["n"] += 1
            d["sum"] += r_i
            d["nz"] += int(r_i > 0)

        # Live per-component diagnostics. run2's failure was only diagnosable in
        # hindsight because nothing separated "fewer truncations" from "shorter".
        if callable(log_metric):
            try:
                passers = [len_f[i] for i in range(n) if ok_f[i] and not clip_f[i]]
                log_metric("v2/pass_rate", sum(ok_f) / max(n, 1))
                log_metric("v2/clipped_ratio", sum(clip_f) / max(n, 1))
                log_metric("v2/len_p50_passing",
                           float(st.median(passers)) if passers else 0.0)
                log_metric("v2/reward_std", float(st.pstdev(rewards)) if n > 1 else 0.0)
                log_metric("v2/replay_frac",
                           sum(1 for k in kind_f if k != "lcb_exec") / max(n, 1))
            except Exception:
                pass

        # ALWAYS emit on the first sighting of a new reward_kind, not only on the
        # interval. 2026-08-28 (bug-641): run4's smoke aborted claiming "only 1 tier
        # scored" when two had scored and a third was never logged at all. Two
        # compounding reasons, both of which this line fixes:
        #   * `state` is PER-RANK and DDP world=2, so one rank's cumulative view is
        #     not the population -- rank0 had seen mbpp_exec, rank1 lcb_exec.
        #   * `log_every` SAMPLES. mc_letter scored but never crossed the boundary,
        #     so no line ever mentioned it.
        # The launcher's gate reads these lines as its evidence, so a tier that never
        # prints is indistinguishable from a tier that is dead. Emitting on first
        # sighting makes every tier that ever scored appear at least once, per rank.
        # [[feedback_one_log_line_is_not_a_cross_arm_reading]]
        interval = log_every and state["n"] // max(log_every, 1) != \
            (state["n"] - n) // max(log_every, 1)
        if interval or new_kind:
            tiers = " ".join(
                f"{k}:n={d['n']},mean={d['sum']/max(d['n'],1):.3f},nz={d['nz']}"
                for k, d in sorted(state["byk"].items()))
            print(f">>> V2_REWARD rank={RANK} n={state['n']} "
                  f"pass={state['pass']/state['n']:.3f} "
                  f"mean_tok={state['tok']/state['n']:.0f} "
                  f"clipped={state['clip']/state['n']:.3f} | TIERS {tiers}",
                  flush=True)
        return rewards

    gepo_reward_v2.verifier = lcb_verifier
    return gepo_reward_v2
