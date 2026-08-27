#!/usr/bin/env python3
"""Correctness-gated LENGTH reward over LiveCodeBench, for GRPO/GSPO/GEPO training.

    reward = (1 - lambda * min(ntok / budget, 1))  if the tests pass  else 0.0

The gate is the point. An ungated length reward teaches the model to emit nothing;
gating on execution means the ONLY way to score is to stay correct and get shorter.
This is the code analogue of an-finetune's simpo/train_grpo_e2b.py, whose verifier was
`grade_numeric` against a numeric gold -- useless for code, which needs execution.

REWARD == SCORER. The verifier is eval/lcb/lcb_helpers.{clean_lcb_completion,
score_lcb_problem}, i.e. literally the functions that produce the published
lcb_v6_77q number. A reward that scores differently from the bench would optimise
something the bench cannot see. [[feedback_eval_methodology]]

ntok COUNTS THE THINKING CHANNEL. The complaint being fixed is that CoderX emits a
p50 of 15,793 tokens where its sibling emits 2,157; almost all of that is reasoning.
Penalising only the visible answer would leave the actual defect unrewarded. So the
penalty is on everything the model emitted, while correctness is judged on the
extracted code.

PROCESS SAFETY. score_lcb_problem forks a child per call. Forking from a training
process that already holds a CUDA context is a known deadlock source, so scoring is
delegated to a persistent worker started at construction time -- construct the reward
BEFORE the policy is loaded onto the GPU. The worker is also where the timeout lives:
a rollout that emits an infinite loop must cost 10s, not the run.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _worker_main(conn):
    """Runs in a child started BEFORE CUDA init; only this child forks per-problem."""
    from eval.lcb.lcb_helpers import clean_lcb_completion, score_lcb_problem
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        text, starter, tests, method, timeout = msg
        try:
            code = clean_lcb_completion(text, starter)
            ok, reason = score_lcb_problem(code, tests, method, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- a bad rollout must never kill the run
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        conn.send((bool(ok), str(reason)[:200]))


class LcbVerifier:
    """Persistent out-of-process LCB execution verifier."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        ctx = mp.get_context("fork")
        self._parent, child = ctx.Pipe()
        # daemon=False is LOAD-BEARING, not a style choice: score_lcb_problem forks its
        # own sandbox child per problem, and a daemonic process may not have children
        # ("AssertionError: daemonic processes are not allowed to have children"). With
        # daemon=True the verifier returns False for EVERY rollout -- every reward
        # collapses to 0.0, the gate looks like it is working, and the run learns
        # nothing. Cost of the non-daemon choice: close() must be called (atexit below).
        self._proc = ctx.Process(target=_worker_main, args=(child,), daemon=False)
        self._proc.start()
        child.close()
        import atexit
        atexit.register(self.close)

    def __call__(self, text: str, meta: dict) -> tuple[bool, str]:
        try:
            self._parent.send((text, meta.get("starter_code", ""), meta["tests"],
                               meta["method_name"], self.timeout))
            # Generous: the child's own per-problem timeout should fire first. This
            # outer wait only catches a wedged worker.
            if not self._parent.poll(self.timeout * 3 + 30):
                return False, "verifier worker unresponsive"
            return self._parent.recv()
        except Exception as exc:  # noqa: BLE001
            return False, f"verifier transport: {type(exc).__name__}: {exc}"

    def close(self):
        try:
            self._parent.send(None)
            self._proc.join(5)
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.terminate()


def _completion_text(c) -> str:
    # TRL conversational -> [{"role":"assistant","content":...}]; else a plain str
    if isinstance(c, list):
        return c[0].get("content", "") if c else ""
    return c or ""


def make_lcb_brevity_reward(tokenizer, budget: int = 4096, lam: float = 0.7,
                            timeout: float = 10.0, verifier: LcbVerifier | None = None,
                            log_every: int = 0):
    """Build the TRL reward callable.

    budget: token count at which the length penalty saturates. 4096 is ~2x the A3B-Coder
            LCB p50 (2,157) and ~1/4 of CoderX's (15,793): reachable without forcing the
            model to abandon reasoning it actually needs.
    lam:    penalty weight. 0.7 matches the an-finetune GEPO recipe, so a correct-but-
            maximally-verbose rollout still scores 0.30 -- strictly better than a wrong
            one at 0.00. Correctness must never be worth less than brevity.
    """
    v = verifier or LcbVerifier(timeout=timeout)
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    state = {"n": 0, "passed": 0, "toks": 0}

    def reward_lcb_correctness_gated_length(completions, meta=None, **kwargs):
        metas = meta if meta is not None else [None] * len(completions)
        out = []
        for comp, m in zip(completions, metas):
            try:
                text = _completion_text(comp)
                if isinstance(m, str):          # datasets may round-trip dicts as JSON
                    m = json.loads(m)
                if not m or "tests" not in m:
                    out.append(0.0)
                    continue
                ntok = len(tk(text, add_special_tokens=False).input_ids)
                ok, _reason = v(text, m)
                r = (1.0 - lam * min(ntok / budget, 1.0)) if ok else 0.0
                out.append(float(r))
                state["n"] += 1
                state["passed"] += int(ok)
                state["toks"] += ntok
                if log_every and state["n"] % log_every == 0:
                    print(f">>> LCB_REWARD n={state['n']} "
                          f"pass={state['passed'] / state['n']:.3f} "
                          f"mean_tok={state['toks'] / state['n']:.0f}", flush=True)
            except Exception as exc:  # noqa: BLE001 -- crash-proof by contract
                print(f">>> reward exception (scored 0.0): {type(exc).__name__}: {exc}",
                      flush=True)
                out.append(0.0)
        return out

    reward_lcb_correctness_gated_length.verifier = v
    return reward_lcb_correctness_gated_length


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    """Gold-anchored: a PASSING short answer, a PASSING long one, and a WRONG one.

    A reward whose only check is 'it returned a float' is not checked at all -- these
    three assert the ORDERING the training actually depends on.
    """
    pool = REPO / "eval" / "lcb" / "lcb_rl_pool.jsonl"
    if not pool.exists():
        print(f"REFUSE: {pool} missing -- run scripts/build_lcb_rl_pool.py first")
        return 2
    rows = [json.loads(ln) for ln in pool.open()]

    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return type("E", (), {"input_ids": s.split()})()

    v = LcbVerifier(timeout=10.0)
    fails = []

    # --- GOLD 1: a synthetic problem with a solution we KNOW is right ----------------
    # A selftest that only ever feeds the verifier WRONG code cannot tell "the tests
    # bind" from "the verifier is broken and returns False for everything" -- which is
    # exactly the daemon=True bug this arm was added to catch.
    gold = {
        "starter_code": "class Solution:\n    def addOne(self, x: int) -> int:\n",
        "method_name": "addOne",
        "tests": [{"input": "1", "output": "2", "testtype": "functional"},
                  {"input": "5", "output": "6", "testtype": "functional"}],
        "difficulty": "synthetic", "n_tests": 2,
    }
    right = ("```python\nclass Solution:\n    def addOne(self, x: int) -> int:\n"
             "        return x + 1\n```")
    off_by_one = ("```python\nclass Solution:\n    def addOne(self, x: int) -> int:\n"
                  "        return x\n```")
    ok_right, why_r = v(right, gold)
    ok_wrong_syn, why_w = v(off_by_one, gold)
    print(f"  gold  correct  (x+1) -> passed={ok_right}   ({why_r[:60]})")
    print(f"  gold  wrong    (x)   -> passed={ok_wrong_syn}   ({why_w[:60]})")
    if not ok_right:
        fails.append(f"VERIFIER BROKEN: known-correct code failed -- {why_r}")
    if ok_wrong_syn:
        fails.append("VERIFIER BLIND: off-by-one code passed -- tests are not binding")

    # --- GOLD 2: a real pool problem, stubbed -> must fail ---------------------------
    row = rows[0]
    m = row["meta"]
    print(f"  pool problem: {row['id']} ({m['difficulty']}, method={m['method_name']}, "
          f"{m['n_tests']} tests)")
    stub = f"```python\n{m['starter_code']}\n        return None\n```"
    ok_stub, why_s = v(stub, m)
    print(f"  pool  stub (None)    -> passed={ok_stub}   ({why_s[:60]})")
    if ok_stub:
        fails.append("a `return None` stub PASSED a real pool problem")

    # --- the reward ORDERING, end to end through the real verifier -------------------
    fn = make_lcb_brevity_reward(_Tok(), budget=100, lam=0.7, verifier=v)
    pad = "\n# " + " ".join(["x"] * 400)          # 400 filler tokens -> penalty saturates
    r_wrong = fn([off_by_one], meta=[gold])[0]
    r_short = fn([right], meta=[gold])[0]
    r_long = fn([right + pad], meta=[gold])[0]

    print(f"  reward wrong           = {r_wrong:.4f}  (want 0.0000)")
    print(f"  reward correct+short   = {r_short:.4f}  (want > 0.85, short answer)")
    print(f"  reward correct+verbose = {r_long:.4f}  (want 0.3000, the saturated floor)")

    if abs(r_wrong) > 1e-9:
        fails.append(f"wrong answer scored {r_wrong}, must be 0.0")
    if r_short < 0.85:
        fails.append(f"short correct scored {r_short}, want > 0.85")
    if abs(r_long - 0.30) > 1e-6:
        fails.append(f"verbose correct scored {r_long}, want 0.30 (1 - lam)")
    if not r_long < r_short:
        fails.append("verbose is not penalised relative to short")
    if not r_wrong < r_long:
        fails.append("a WRONG answer outranks a correct-but-verbose one -- gate inverted")

    v.close()
    if fails:
        print("\nSELFTEST FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nLCB_REWARD_SELFTEST_OK  (gate binds, ordering wrong < verbose < short)")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else
             print("usage: lcb_brevity_reward.py --selftest") or 0)
