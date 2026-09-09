"""Render gate for T195.D with a POSITIVE CONTROL.

Arm under test must NOT re-inject prior reasoning_content.
Control arm (stock upstream google template) MUST re-inject it -- otherwise the
gate cannot detect a leak and a PASS on the test arm is meaningless.
"""
import sys
from transformers import AutoTokenizer

CANARY = "SECRET_PRIOR_CHAIN_OF_THOUGHT_MARKER"
msgs = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "ok", "reasoning_content": CANARY},
    {"role": "user", "content": "second question"},
    {"role": "assistant", "content": "working on it", "reasoning_content": CANARY},
]

def render(arm, think):
    tok = AutoTokenizer.from_pretrained(arm, trust_remote_code=True)
    s = tok.apply_chat_template(msgs, tokenize=False,
                                add_generation_prompt=True, enable_thinking=think)
    ntok = len(tok(s, add_special_tokens=False)["input_ids"])
    return s, ntok

def report(label, arm, expect_leak):
    ok = True
    for think in (True, False):
        s, ntok = render(arm, think)
        leaked = CANARY in s
        good = (leaked == expect_leak)
        ok &= good
        print("  %-28s think=%-5s %5d ch /%5d tok  reinject=%-4s  %s"
              % (label, think, len(s), ntok, "LEAK" if leaked else "none",
                 "ok" if good else "*** UNEXPECTED ***"))
    return ok

test_arms = sys.argv[1:-1]
control = sys.argv[-1]
print("=== POSITIVE CONTROL (stock google template -- MUST leak) ===")
ctl_ok = report("CONTROL stock-google", control, expect_leak=True)
print()
print("=== ARMS UNDER TEST (T195.D -- must NOT leak) ===")
arms_ok = all(report(a.rstrip("/").split("/")[-1], a, expect_leak=False) for a in test_arms)
print()
s, _ = render(test_arms[0], True)
print("=== rendered prompt, first test arm, think=True ===")
print("\n".join("    " + ln for ln in s.splitlines()))
print()
if not ctl_ok:
    print("GATE: INVALID -- control did not leak, so the gate cannot detect a leak.")
    sys.exit(2)
print("GATE:", "PASS" if arms_ok else "FAIL")
sys.exit(0 if arms_ok else 1)
