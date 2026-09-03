#!/usr/bin/env python
"""Polarity battery for validate_entropy_args + the argparse defaults it guards.

Lifted out of gepo_brevity.py by AST so it cannot drift from the real function, and so
it runs without importing torch/trl (the launcher's module-level imports need a GPU box).
Every arm has BOTH directions: a refusal that never refuses is indistinguishable from
one that always passes. [[feedback_a_check_gold_fails_is_a_broken_check]]
"""
import ast
import re
import sys
import types

SRC = open("scripts/gepo_brevity.py").read()
tree = ast.parse(SRC)
want = {"validate_entropy_args"}
nodes = [n for n in tree.body
         if (isinstance(n, ast.FunctionDef) and n.name in want)
         or (isinstance(n, ast.ClassDef) and n.name == "EntropyRefused")]
assert len(nodes) == 2, [getattr(n, "name", n) for n in nodes]
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<gb>", "exec"), ns)
validate, Refused = ns["validate_entropy_args"], ns["EntropyRefused"]

P = F = 0


def chk(label, cond):
    global P, F
    print(("  ok   " if cond else "  FAIL ") + label)
    P, F = (P + 1, F) if cond else (P, F + 1)


def A(**kw):
    d = dict(gepo_entropy=True, no_gepo=False, gepo_alpha_low=0.5, gepo_alpha_high=0.2)
    d.update(kw)
    return types.SimpleNamespace(**d)


def refuses(a):
    try:
        validate(a)
        return None
    except Refused as e:
        return str(e)


chk("A  paper defaults PASS", refuses(A()) is None)
chk("B  --gepo-entropy --no-gepo REFUSED",
    (m := refuses(A(no_gepo=True))) and "no-gepo" in m)
chk("B2 --no-gepo alone (entropy off) passes",
    refuses(A(gepo_entropy=False, no_gepo=True)) is None)
chk("C  alpha_high > alpha_low REFUSED (length collapse)",
    (m := refuses(A(gepo_alpha_high=0.9))) and "LENGTH-COLLAPSE" in m)
chk("C2 alpha_high == alpha_low REFUSED (boundary)",
    refuses(A(gepo_alpha_high=0.5)) is not None)
chk("C3 alpha_high just below alpha_low PASSES (boundary, other side)",
    refuses(A(gepo_alpha_high=0.4999)) is None)
chk("D  entropy OFF is a strict no-op even with a bad alpha pair",
    refuses(A(gepo_entropy=False, gepo_alpha_high=0.9, no_gepo=True)) is None)

# --- the flags must actually EXIST on the parser with the paper's values -------
# A validator that guards defaults nobody passes is theatre; read them off argparse.
defaults = dict(re.findall(
    r'ap\.add_argument\("(--gepo-(?:alpha|beta|gamma|entropy)[a-z-]*)",[^)]*?default=([0-9.]+)',
    SRC, re.S))
want_d = {"--gepo-alpha-low": "0.5", "--gepo-alpha-high": "0.2",
          "--gepo-beta-low": "0.2", "--gepo-beta-high": "0.3", "--gepo-gamma": "0.01",
          "--gepo-entropy-warmup": "10", "--gepo-entropy-silent-limit": "25"}
for k, v in want_d.items():
    chk(f"E  argparse default {k}={v}", defaults.get(k) == v)
chk("E2 --gepo-entropy is a store_true FLAG (not a value)",
    'ap.add_argument("--gepo-entropy", action="store_true"' in SRC)

# --- the validator is WIRED, and fires BEFORE the model is loaded --------------
# NOT SRC.index("validate_entropy_args(args)") -- that substring also matches the `def`
# line, which sits before main() and made this arm AND the three below vacuously true.
m_call = re.search(r"(?<!def )\bvalidate_entropy_args\(args\)", SRC)
call = m_call.start() if m_call else len(SRC)
chk("F  validator is CALLED, not merely defined",
    m_call is not None and call > SRC.index("def main("))
for marker in ("AutoModelForCausalLM.from_pretrained", "GEPOTrainer(", "Dataset.from_list"):
    i = SRC.find(marker)
    chk(f"F2 fires before {marker.split('(')[0].split('.')[0]}", i == -1 or call < i)

# --- the HARNESS and the LAUNCHER must agree on every flag name -----------------
# r9_gepo_hf.sh builds the flag list; gepo_brevity.py parses it. A typo on either
# side is an argparse "unrecognized arguments" death 40 minutes into a smoke, or --
# worse, a flag silently absent from the run. Diff the two lists here instead.
HF = open("scripts/r9_gepo_hf.sh").read()
emitted = set(re.findall(r"(--gepo-[a-z-]+)", HF))
declared = set(re.findall(r'ap\.add_argument\("(--gepo-[a-z-]+)"', SRC))
chk(f"G  harness emits {len(emitted)} --gepo-* flags, all declared by argparse",
    bool(emitted) and emitted <= declared)
if not emitted <= declared:
    print("     UNDECLARED: " + ", ".join(sorted(emitted - declared)))
chk("G2 harness passes --gepo-entropy itself (not just its parameters)",
    "--gepo-entropy" in emitted)
chk("G3 smoke override lowers warmup AND disables the silent-abort",
    "--gepo-entropy-warmup 1" in HF and "--gepo-entropy-silent-limit 0" in HF)
chk("G4 the smoke gate is CALLED, not merely defined",
    HF.count("gate_entropy_fires") >= 2)

print(f"GEPO_ENTROPY_ARGS_GATE pass={P} fail={F}")
sys.exit(1 if F else 0)
