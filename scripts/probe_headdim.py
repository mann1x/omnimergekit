import json
from transformers import AutoConfig
for name, P in [("Jprime-p3", "/mnt/sdc/v7rework/arms/Jprime-p3-bf16"),
                ("vendor base", "/mnt/sdc/v7rework/base/gemma-4-26B-A4B-it")]:
    try:
        cfg = AutoConfig.from_pretrained(P, trust_remote_code=True)
    except Exception as e:
        print(name, "config load failed:", e); continue
    tc = getattr(cfg, "text_config", cfg)
    plc = getattr(tc, "per_layer_config", None)
    print("###", name)
    if plc is None:
        print("   no per_layer_config")
    else:
        hd = [getattr(l, "head_dim", None) for l in plc]
        kv = [getattr(l, "num_key_value_heads", None) for l in plc]
        ah = [getattr(l, "num_attention_heads", None) for l in plc]
        print("   layers:", len(plc))
        print("   head_dim  distinct:", sorted(set(map(str, hd))))
        print("   n_kv_heads distinct:", sorted(set(map(str, kv))), " counts:",
              {v: kv.count(v) for v in set(kv)})
        print("   n_attn_heads distinct:", sorted(set(map(str, ah))))
    raw = json.load(open(P + "/config.json"))
    t = raw.get("text_config", raw)
    print("   config.json head_dim:", t.get("head_dim"),
          "| allow flag present:", "allow_global_per_layer_attribute_access" in t)
