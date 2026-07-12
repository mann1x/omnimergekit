#!/usr/bin/env python3
"""T174.4 Stage-A loop canary: before/after on ONE model load via adapter
toggling. For each prompt, greedy-generate with the LoRA adapter DISABLED
(= base A2) and ENABLED, and report length + detect_loop() for both. In-domain
seeds (3 trained loopers) are a sanity check; the HELD-OUT prompts (never
trained) are the real generalization test — the adapter must terminate those too,
not just memorize the seeds.
"""
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

A2 = "/mnt/sdc/ml/google/gemma-4-A4B-62e-fc15_25-p8-pes120-it"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdc/ml/google/a2-antiloop-lora/epoch2"
CORPUS = "/mnt/sdc/ml/corpora/antiloop_sft_corpus.jsonl"
MAX_NEW = 3072

# held-out loop-prone prompts (same archetypes, NOT in training)
HELDOUT = [
    "Write a short story about a robot learning to paint. Your entire response "
    "must be in English and in all capital letters. The letter E should appear "
    "at least 50 times.",
    "Write a poem about a dragon visiting Tokyo, in the Japanese language only. "
    "No other language is allowed.",
    "Describe your perfect weekend where every single sentence starts with the "
    "letter S.",
    "بنویس یک شعر کوتاه درباره ماه و دریا به زبان فارسی.",  # Persian: short poem moon+sea
    "Write a motivational note to a student. Do not use any commas, and do not "
    "use the letter a anywhere in your response.",
]


def seeds():
    out = []
    for ln in open(CORPUS):
        r = json.loads(ln)
        if r.get("bucket") == "seeds":
            out.append(r["prompt"])
    return out


def gen(model, tok, prompt):
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=False)
    enc = tok(txt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(A2, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        A2, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    prompts = [("seed", p) for p in seeds()] + [("held-out", p) for p in HELDOUT]
    base_loops = adpt_loops = 0
    print("adapter: %s\n" % ADAPTER)
    for kind, p in prompts:
        with model.disable_adapter():
            b = gen(model, tok, p)
        a = gen(model, tok, p)
        bl, al = detect_loop(b), detect_loop(a)
        base_loops += bl
        adpt_loops += al
        print("[%-8s] BASE  len=%-6d loop=%s | ADAPTER len=%-6d loop=%s | %s" % (
            kind, len(b), bl, len(a), al, p[:55].replace("\n", " ")))
    n = len(prompts)
    print("\nSUMMARY: base loops=%d/%d  adapter loops=%d/%d" % (
        base_loops, n, adpt_loops, n))
    print("STAGE_A_PASS" if adpt_loops == 0 else "STAGE_A_PARTIAL adapter_loops=%d" % adpt_loops)


if __name__ == "__main__":
    main()
