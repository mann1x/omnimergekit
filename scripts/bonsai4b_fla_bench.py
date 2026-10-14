import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as M
p = "/mnt/sdc/opencoti-models/bonsai4b/Qwen3.5-4B"; dev = "cuda:1"
print("fla kernel:", M.chunk_gated_delta_rule is not None, "conv fast:", M.causal_conv1d_fn is not None)
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to(dev)
ids = tok(open("/srv/ml/ppl-corpus/wiki.test.mid.raw").read()[:60000], return_tensors="pt").input_ids[0][:4096].view(4, 1024).to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(2): lo = m(ids).logits
    torch.cuda.synchronize(); t = time.time()
    for _ in range(5): lo = m(ids).logits
    torch.cuda.synchronize(); print("forward 4x1024: %.3fs" % ((time.time() - t) / 5))
m.gradient_checkpointing_enable(); m.train()
for q in m.parameters(): q.requires_grad_(True)
torch.cuda.synchronize(); t = time.time()
for k in range(6):
    if k == 3:
        torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = m(ids[:2]).logits.float().logsumexp(-1).mean()
    l.backward(); m.zero_grad()
torch.cuda.synchronize(); print("fwd+bwd 2x1024: %.3fs" % ((time.time() - t) / 3))
torch.save(lo[0, :64].float().cpu(), sys.argv[1])
