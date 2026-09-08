import sys
from gguf import GGUFReader
KEYS = ["qwen35moe.expert_used_count", "qwen35moe.expert_count",
        "qwen35moe.block_count", "qwen35moe.nextn_predict_layers"]
for p in sys.argv[1:]:
    r = GGUFReader(p)
    def g(k):
        f = r.fields.get(k)
        try:
            return f.parts[f.data[0]].tolist()[0]
        except Exception:
            return None
    print(p.split("/")[-1])
    print("    tensors                = %d" % len(r.tensors))
    for k in KEYS:
        print("    %-22s = %s" % (k.split(".")[-1], g(k)))
