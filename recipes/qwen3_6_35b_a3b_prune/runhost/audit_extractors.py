#!/usr/bin/env python3
"""Is bug-604 (extractor destroys a scoreable answer) present on benches other than MPE?

bug-604 was: multipl_e chat_to_body's fallback deleted a line from inside the model's code,
turning a possibly-correct answer into a guaranteed compile failure that was then banked as a
model failure. This audits every OTHER bench for the same DEFECT CLASS, empirically, from the
stored artifacts -- not by reading the extractor and reasoning about it.

The measurement has a CONTROL in each case, which is what makes it decisive: these benches
stored the RAW model reply alongside the extracted artifact (MPE did not -- that is why MPE
needed a regeneration). So for every failed item we can ask the counterfactual directly:

    did the extractor turn a parseable answer into an unparseable one?

  HE / HE+ (lm-eval)  resps[0] = raw reply, filtered_resps[0] = what was scored.
                      DAMAGE := filtered does NOT ast.parse, BUT some fenced block in the
                      raw reply DOES ast.parse and defines a function.
  LCB                 completion = raw, cleaned = what was scored, passed = verdict.
                      DAMAGE := cleaned does NOT ast.parse, BUT raw contains a fenced block
                      that DOES ast.parse and mentions `class Solution`.
  arc/aime/gsm8k/math500  extraction yields a short answer string.
                      DAMAGE := filtered is empty/None while the raw reply is substantial
                      (>200 chars) -- i.e. the model said something and the filter kept none
                      of it. (Weaker than the code benches: a filter legitimately returns
                      nothing when the model never states an answer. Reported as an UPPER
                      BOUND, never as a defect count.)

Nothing is executed. ast.parse only.
"""
import ast, json, os, re, sys, glob
from collections import defaultdict

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def parses(code: str) -> bool:
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, IndentationError, TabError, ValueError, MemoryError, RecursionError):
        return False


def has_def(tree_src: str) -> bool:
    try:
        t = ast.parse(tree_src)
    except Exception:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for n in ast.walk(t))


def raw_has_good_block(raw: str, need: str | None = None) -> bool:
    """Control: does the RAW reply contain a fenced block that parses (and mentions `need`)?"""
    for b in FENCE.findall(raw or ""):
        if need and need not in b:
            continue
        if parses(b) and has_def(b):
            return True
    return False


def first(x):
    while isinstance(x, list) and x:
        x = x[0]
    return x if isinstance(x, str) else ""


# ── HE / HE+ via lm-eval samples ────────────────────────────────────────────
def audit_lm_eval_code(cell: str):
    files = glob.glob(os.path.join(cell, "lm_eval_out", "**", "samples_*.jsonl"), recursive=True)
    n = fail = dead = damage = unscored = 0
    ex = []
    for f in files:
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            filt = first(d.get("filtered_resps"))
            raw = first(d.get("resps"))
            # The score key is BENCH-DEPENDENT: humaneval* uses "pass@1", mbpp uses
            # "pass_at_1". Reading only one silently marks every item a failure and
            # manufactures a 100%-fail bench (measured: mbpp read as 1000/1000 fail
            # against a banked 0.576). Take whichever key is present; if NONE is,
            # that is a broken read -- refuse rather than count it as a failure.
            score = None
            for k in ("pass@1", "pass_at_1", "acc", "exact_match"):
                if k in d:
                    score = d[k]
                    break
            if score is None:
                unscored += 1
                continue
            if isinstance(score, (int, float)) and score > 0:
                continue
            fail += 1
            if not parses(filt):
                dead += 1
                if raw_has_good_block(raw):
                    damage += 1
                    if len(ex) < 3:
                        ex.append(str(d.get("doc_id")))
    return dict(n=n, fail=fail, dead=dead, damage=damage, unscored=unscored, ex=ex)


# ── LCB ─────────────────────────────────────────────────────────────────────
def audit_lcb(cell: str):
    f = os.path.join(cell, "lcb_result.samples.jsonl")
    if not os.path.exists(f):
        return None
    n = fail = dead = damage = 0
    ex = []
    for line in open(f, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        n += 1
        if d.get("passed"):
            continue
        fail += 1
        cleaned, raw = d.get("cleaned") or "", d.get("completion") or ""
        if not parses(cleaned):
            dead += 1
            if raw_has_good_block(raw, need="class Solution"):
                damage += 1
                if len(ex) < 3:
                    ex.append(str(d.get("task_id")))
    return dict(n=n, fail=fail, dead=dead, damage=damage, ex=ex)


# ── short-answer benches ────────────────────────────────────────────────────
def audit_lm_eval_answer(cell: str):
    files = glob.glob(os.path.join(cell, "lm_eval_out", "**", "samples_*.jsonl"), recursive=True)
    n = empty_filt = empty_filt_long_raw = 0
    for f in files:
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            filt = first(d.get("filtered_resps"))
            raw = first(d.get("resps"))
            if filt is None or not str(filt).strip():
                empty_filt += 1
                if raw and len(raw) > 200:
                    empty_filt_long_raw += 1
    return dict(n=n, empty=empty_filt, empty_long_raw=empty_filt_long_raw)


CODE_BENCH = ("humaneval", "mbpp")
ANSWER_BENCH = ("arc", "aime", "gsm8k", "math500", "gpqa", "ifeval")

def main(roots):
    agg = defaultdict(lambda: dict(n=0, fail=0, dead=0, damage=0, cells=0))
    print(f"{'bench':<28} {'cells':>5} {'items':>7} {'fail':>6} {'unparseable':>11} {'EXTRACTOR-DAMAGE':>17}")
    print("-" * 82)
    for root in roots:
        for bench_dir in sorted(glob.glob(os.path.join(root, "*"))):
            bench = os.path.basename(bench_dir)
            if bench == "multipl_e_100":
                continue  # already handled: bug-604, regeneration scheduled
            kind = ("code" if any(k in bench for k in CODE_BENCH)
                    else "lcb" if bench.startswith("lcb")
                    else "answer" if any(k in bench for k in ANSWER_BENCH) else None)
            if kind is None:
                continue
            for cell in sorted(glob.glob(os.path.join(bench_dir, "*"))):
                if not os.path.isdir(cell):
                    continue
                r = (audit_lcb(cell) if kind == "lcb"
                     else audit_lm_eval_code(cell) if kind == "code"
                     else audit_lm_eval_answer(cell))
                if not r or not r.get("n"):
                    continue
                key = f"{os.path.basename(root)}/{bench}"
                a = agg[key]
                a["cells"] += 1
                a["n"] += r["n"]
                if kind == "answer":
                    a["fail"] += r["empty"]
                    a["damage"] += r["empty_long_raw"]
                    a["kind"] = "answer"
                else:
                    a["fail"] += r["fail"]; a["dead"] += r["dead"]; a["damage"] += r["damage"]
                    a["unscored"] = a.get("unscored", 0) + r.get("unscored", 0)
                    a["kind"] = kind
                    a.setdefault("ex", []).extend(r.get("ex", [])[:1])
    for key in sorted(agg):
        a = agg[key]
        if a.get("kind") == "answer":
            print(f"{key:<28} {a['cells']:>5} {a['n']:>7} {a['fail']:>6} {'-':>11} "
                  f"{a['damage']:>17}   (empty-filter w/ long raw = UPPER BOUND)")
        else:
            print(f"{key:<28} {a['cells']:>5} {a['n']:>7} {a['fail']:>6} {a['dead']:>11} "
                  f"{a['damage']:>17}   {'UNSCORED=' + str(a['unscored']) + ' ' if a.get('unscored') else ''}{','.join((a.get('ex') or [])[:3])}")
    print("-" * 82)
    print("EXTRACTOR-DAMAGE = the scored artifact does not parse, but the RAW reply contained a")
    print("parseable fenced definition. That is the bug-604 signature. 0 = extractor exonerated")
    print("on this bench; every unparseable scored artifact traces to the model, not our code.")


if __name__ == "__main__":
    main(sys.argv[1:])
