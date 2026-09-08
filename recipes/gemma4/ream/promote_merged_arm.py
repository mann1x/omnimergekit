"""Census a merged Gemma-4 arm and promote it to the arms/ dir.

Our port merges IN PLACE inside Gemma4ForConditionalGeneration, so the saved
model already carries the vision tower -- unlike the Qwen path, where REAM saves
a bare trunk and visual.* must be grafted back. The census is still mandatory:
it is the plan's fold-arm gate, and it must be proven per arm, not assumed.
"""
import argparse, collections, glob, json, os, struct, sys, shutil


def census(d):
    dims, vis, layers = collections.Counter(), 0, set()
    for s in sorted(glob.glob(os.path.join(d, "model-*.safetensors"))):
        with open(s, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            if ".experts.down_proj" in k:
                dims[v["shape"][0]] += 1
                layers.add(k.split(".layers.")[1].split(".")[0])
            if k.startswith("visual.") or "vision_tower" in k:
                vis += 1
    return dims, vis, layers


ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True)
ap.add_argument("--dest", required=True)
ap.add_argument("--expect-experts", type=int, required=True)
ap.add_argument("--expect-layers", type=int, required=True)
ap.add_argument("--expect-vision", type=int, required=True)
ap.add_argument("--move", action="store_true")
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.merged, "config.json")))
tc = cfg.get("text_config", cfg)
dims, vis, layers = census(a.merged)
print(f"  config.text_config.num_experts = {tc.get('num_experts')}")
print(f"  expert-dim histogram = {dict(dims)}  layers = {len(layers)}  vision = {vis}")

bad = []
if tc.get("num_experts") != a.expect_experts:
    bad.append(f"config num_experts {tc.get('num_experts')} != {a.expect_experts}")
if list(dims) != [a.expect_experts]:
    bad.append(f"expert dims {dict(dims)} != all {a.expect_experts}")
if len(layers) != a.expect_layers:
    bad.append(f"layers {len(layers)} != {a.expect_layers}")
if vis != a.expect_vision:
    bad.append(f"vision census {vis} != {a.expect_vision}")
if bad:
    print("CENSUS_FAIL:", "; ".join(bad))
    sys.exit(1)

if os.path.exists(a.dest):
    print(f"REFUSING: destination exists: {a.dest}")
    sys.exit(2)
os.makedirs(os.path.dirname(a.dest), exist_ok=True)
if a.move:
    shutil.move(a.merged, a.dest)
    print(f"  moved -> {a.dest}")
print("CENSUS_OK")
