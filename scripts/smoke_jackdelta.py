"""Did the DeltaCoder LoRA change BEHAVIOUR, or only bits?

71.67% of parameters changed, but the delta is ~17x smaller than a finished
fine-tune's. Changed bits are not changed behaviour: measure greedy agreement
and next-token divergence on the SAME prompts, same sampler, both models
resident at once so nothing about the environment differs between them.
"""
import json, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "/mnt/sdc/oxopus/models/Qwopus3.5-9B-Coder"
MERGED = "/mnt/sdc/oxopus/JackDeltaCoder-9B"
DEV = "cuda:1"
PROMPTS = [
 "Write a Python function that returns the n-th Fibonacci number iteratively.",
 "Implement binary search over a sorted list in Python. Return -1 if absent.",
 "Write a Rust function that reverses a string in place without allocating.",
 "Given a list of dicts with a 'ts' key, write Python to group them by day.",
 "Write a SQL query returning the top 3 customers by total order value.",
 "Explain in two sentences why a race condition can occur with check-then-act.",
]
tok = AutoTokenizer.from_pretrained(BASE)
print("loading both models onto", DEV, flush=True)
m1 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV).eval()
m2 = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.bfloat16).to(DEV).eval()
print("loaded", flush=True)

same_txt = 0
kls, tops = [], []
for i, p in enumerate(PROMPTS):
    msgs = [{"role": "user", "content": p}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = (enc["input_ids"] if not torch.is_tensor(enc) else enc).to(DEV)
    with torch.no_grad():
        l1 = m1(ids).logits[0, -1].float()
        l2 = m2(ids).logits[0, -1].float()
    p1 = torch.log_softmax(l1, -1)
    p2 = torch.log_softmax(l2, -1)
    kl = float((p1.exp() * (p1 - p2)).sum())
    kls.append(kl)
    tops.append(int(l1.argmax()) == int(l2.argmax()))
    with torch.no_grad():
        g1 = m1.generate(ids, max_new_tokens=96, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        g2 = m2.generate(ids, max_new_tokens=96, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    t1 = tok.decode(g1[0, ids.shape[1]:], skip_special_tokens=True)
    t2 = tok.decode(g2[0, ids.shape[1]:], skip_special_tokens=True)
    eq = t1 == t2
    same_txt += eq
    # first token index where they diverge
    a, b = g1[0, ids.shape[1]:].tolist(), g2[0, ids.shape[1]:].tolist()
    div = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"\n[{i}] {p[:58]}")
    print(f"    next-token KL={kl:.6f}  top1_same={tops[-1]}  "
          f"greedy_identical={eq}  first_divergence_tok={div}")
    if not eq:
        print(f"    base   : {t1[:150]!r}")
        print(f"    merged : {t2[:150]!r}")
n = len(PROMPTS)
print("\n==== SUMMARY ====")
print(f"  prompts                    : {n}")
print(f"  identical greedy output    : {same_txt}/{n}")
print(f"  identical top-1 next token : {sum(tops)}/{n}")
print(f"  next-token KL  min/med/max : {min(kls):.6f} / {sorted(kls)[n//2]:.6f} / {max(kls):.6f}")
print("  VERDICT:", "BEHAVIOURALLY INERT — merge changed bits, not outputs"
      if same_txt == n and max(kls) < 1e-4 else
      "BEHAVIOUR CHANGED — the merge is doing something")
