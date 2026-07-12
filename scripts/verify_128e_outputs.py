"""T176.7 false-pass guard: confirm 128e control 0% loops is REAL generation,
not an empty/truncated-output artifact (detect_loop returns False for len<600).
Re-generates the first N multilingual + a few constrained prompts at the IDENTICAL
loop_screen.py settings (greedy, rep_pen 1.0, max_new 2048, chat template, eager)
and prints per-prompt char/word length + tail + detect_loop. If lengths are
substantive and varied, the 0% is a true negative."""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop
M="/srv/ml/models/base/gemma-4-26B-A4B-it"
S="/mnt/sdc/ml/corpora/loop_screen_sample.jsonl"
rows=[json.loads(x) for x in open(S)]
ml=[r for r in rows if r.get("bucket")=="multilingual"][:8]
con=[r for r in rows if r.get("bucket")=="constrained"][:2]
pick=ml+con
print("verifying %d prompts (8 multilingual + 2 constrained)"%len(pick), flush=True)
tok=AutoTokenizer.from_pretrained(M, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True,
    low_cpu_mem_usage=True, attn_implementation="eager", device_map={"":0}).eval()
for r in pick:
    p=r["prompt"]
    txt=tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, tokenize=False)
    enc=tok([txt], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o=model.generate(**enc, max_new_tokens=2048, do_sample=False, repetition_penalty=1.0,
                         use_cache=True, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    out=tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print("\n=== bucket=%s chars=%d words=%d loop=%s ==="%(r["bucket"], len(out), len(out.split()), detect_loop(out)), flush=True)
    print("PROMPT:", p[:110].replace("\n"," "), flush=True)
    print("TAIL  :", repr(out[-180:]), flush=True)
