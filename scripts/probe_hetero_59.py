import transformers, torch, re
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
P = "/mnt/sdc/v7rework/arms/Jprime-p3-bf16"
print("transformers", transformers.__version__)
cfg = AutoConfig.from_pretrained(P, trust_remote_code=True)
tc = getattr(cfg, "text_config", cfg)
print("global head_dim reads as:", getattr(tc, "head_dim", None))
plc = getattr(tc, "per_layer_config", None)
if plc:
    hd = [getattr(l, "head_dim", None) for l in plc]
    kv = [getattr(l, "num_key_value_heads", None) for l in plc]
    print("per-layer head_dim distinct:", sorted(set(map(str, hd))),
          "counts:", {v: hd.count(v) for v in set(hd)})
    print("per-layer n_kv distinct:", sorted(set(map(str, kv))))
else:
    print("NO per_layer_config -> heterogeneity is NOT represented at this version")
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
# Prove the BUILT model is heterogeneous, not just the config: k_proj out_features
# differ where n_kv_heads differ.
outs = {}
for n, mod in m.named_modules():
    mm = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj", n)
    if mm and isinstance(mod, torch.nn.Linear):
        outs[int(mm.group(1))] = mod.out_features
d = {}
for layer, of in outs.items():
    d.setdefault(of, []).append(layer)
print("BUILT k_proj out_features ->", {k: sorted(v)[:8] for k, v in d.items()})
print("distinct k_proj widths:", sorted(d))
lin = sum(1 for _, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)
          and re.fullmatch(r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)", _))
print("LoRA scope module count:", lin, "(expect 205)")
