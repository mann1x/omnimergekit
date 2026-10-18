"""Parse a REMOTE GGUF header over HTTP Range, no full download."""
import struct, sys, json, urllib.request

URL = ("https://huggingface.co/prithivMLmods/OxCoder-9B-GGUF/"
       "resolve/main/OxCoder-9B.BF16.gguf")

def fetch(url, start, end):
    r = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(r, timeout=60) as h:
        return h.read()

class Buf:
    def __init__(self, url):
        self.url = url; self.buf = b""; self.base = 0
    def need(self, upto):
        while len(self.buf) < upto:
            nxt = max(upto, len(self.buf) + (1 << 20))
            self.buf += fetch(self.url, len(self.buf), nxt - 1)
    def rd(self, off, n):
        self.need(off + n); return self.buf[off:off + n]

GT = {0:"UINT8",1:"INT8",2:"UINT16",3:"INT16",4:"UINT32",5:"INT32",6:"FLOAT32",
      7:"BOOL",8:"STRING",9:"ARRAY",10:"UINT64",11:"INT64",12:"FLOAT64"}
FMT = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),
       6:("<f",4),7:("<?",1),10:("<Q",8),11:("<q",8),12:("<d",8)}

def rstr(b, o):
    n = struct.unpack("<Q", b.rd(o, 8))[0]
    return b.rd(o + 8, n).decode("utf-8", "replace"), o + 8 + n

def rval(b, o, t):
    if t == 8: return rstr(b, o)
    if t == 9:
        et = struct.unpack("<I", b.rd(o, 4))[0]
        n = struct.unpack("<Q", b.rd(o + 4, 8))[0]
        o += 12; out = []
        for _ in range(min(n, 12)):
            v, o = rval(b, o, et)
            out.append(v)
        if n > 12:   # skip the rest
            for _ in range(n - 12):
                v, o = rval(b, o, et)
        return (f"[{GT.get(et,et)} x{n}]", out), o
    f, sz = FMT[t]
    return struct.unpack(f, b.rd(o, sz))[0], o + sz

b = Buf(URL)
magic = b.rd(0, 4)
ver, = struct.unpack("<I", b.rd(4, 4))
ntensor, = struct.unpack("<Q", b.rd(8, 8))
nkv, = struct.unpack("<Q", b.rd(16, 8))
print(f"magic={magic} version={ver} tensors={ntensor} kv={nkv}")
o = 24
kvs = {}
for _ in range(nkv):
    k, o = rstr(b, o)
    t, = struct.unpack("<I", b.rd(o, 4)); o += 4
    v, o = rval(b, o, t)
    kvs[k] = v
print("\n=== key metadata ===")
for k in sorted(kvs):
    if any(s in k for s in ("architecture","block_count","name","basename",
                            "file_type","quantization","size_label","version",
                            "embedding_length","context_length","finetune")):
        v = kvs[k]
        print(f"  {k:46s} {str(v)[:70]}")
# tensor infos
TS = {0:("F32",4,1),1:("F16",2,1),30:("BF16",2,1)}
infos = []
for _ in range(ntensor):
    name, o = rstr(b, o)
    nd, = struct.unpack("<I", b.rd(o, 4)); o += 4
    dims = list(struct.unpack("<" + "Q"*nd, b.rd(o, 8*nd))); o += 8*nd
    tt, = struct.unpack("<I", b.rd(o, 4)); o += 4
    off, = struct.unpack("<Q", b.rd(o, 8)); o += 8
    infos.append({"name": name, "dims": dims, "type": tt, "offset": off})
align = kvs.get("general.alignment", 32)
data_start = (o + align - 1) // align * align
print(f"\nheader_end={o} alignment={align} data_start={data_start}")
types = {}
for i in infos: types[i["type"]] = types.get(i["type"], 0) + 1
print("tensor types:", {TS.get(k,(str(k),0,0))[0]: v for k, v in types.items()})
print("\n=== sample tensor names ===")
for i in infos[:6] + [x for x in infos if "norm" in x["name"]][:6]:
    print("  %-44s dims=%-20s type=%s off=%d" % (
        i["name"], i["dims"], TS.get(i["type"],(i["type"],))[0], i["offset"]))
json.dump({"data_start": data_start, "infos": infos, "url": URL},
          open("/mnt/sdc/oxopus/gguf_header.json","w"))
print("\nwrote /mnt/sdc/oxopus/gguf_header.json  (%d tensors)" % len(infos))
print("bytes fetched for header: %.2f MB" % (len(b.buf)/1e6))
