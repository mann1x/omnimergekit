"""Two GGUFs hold the same model: identical tensors (name, type, shape, bytes) and KV, except the listed keys."""
import sys

sys.path.insert(0, "/shared/dev/llama.cpp-1268eda/gguf-py")
from gguf import GGUFReader  # noqa: E402

a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
skip = {"quantize.imatrix.file"}
ka = {k: f.contents() for k, f in a.fields.items() if not k.startswith("GGUF.")}
kb = {k: f.contents() for k, f in b.fields.items() if not k.startswith("GGUF.")}
kv_diff = sorted(k for k in set(ka) | set(kb) if k not in skip and ka.get(k) != kb.get(k))
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
t_diff = [n for n in sorted(set(ta) | set(tb))
          if n not in ta or n not in tb or ta[n].tensor_type != tb[n].tensor_type
          or list(ta[n].shape) != list(tb[n].shape) or ta[n].data.tobytes() != tb[n].data.tobytes()]
print(f"tensors {len(ta)} vs {len(tb)}: differing {len(t_diff)} {t_diff[:5]}")
print(f"kv {len(ka)} vs {len(kb)}: differing (excl {sorted(skip)}) {len(kv_diff)} {kv_diff[:8]}")
for k in sorted(skip):
    print(f"  {k}: {ka.get(k)!r} vs {kb.get(k)!r}")
print("GGUF_SAME_MODEL" if not t_diff and not kv_diff else "GGUF_DIFFER")
