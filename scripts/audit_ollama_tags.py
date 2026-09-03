#!/usr/bin/env python3
"""Audit every published ollama tag of a namespace against the REGISTRY.

Reads the manifest + config blob for each tag and reports the layer composition
and identity fields. Verifying the artifact where it is consumed -- a log line
saying "REPUBLISHED ok" is not evidence that the published tag is correct.
"""
import argparse
import json
import re
import subprocess
import sys

REG = "https://registry.ollama.ai/v2"
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def curl(url: str, accept: str | None = None) -> str:
    cmd = ["curl", "-sL", url]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout


def tags(ns: str) -> list[str]:
    model = ns.split("/")[-1]
    html = curl(f"https://ollama.com/{ns}/tags")
    return sorted(set(re.findall(rf"{re.escape(model)}:([A-Za-z0-9_.-]+)", html)))


def audit(ns: str, tag: str) -> dict:
    man = curl(f"{REG}/{ns}/manifests/{tag}", ACCEPT)
    try:
        m = json.loads(man)
    except Exception:
        return {"tag": tag, "error": "no manifest"}
    layers = {ly["mediaType"].split(".")[-1]: ly["size"] for ly in m.get("layers", [])}
    cfg = {}
    try:
        cfg = json.loads(curl(f"{REG}/{ns}/blobs/{m['config']['digest']}"))
    except Exception:
        pass
    params = {}
    for ly in m.get("layers", []):
        if ly["mediaType"].endswith("params"):
            try:
                params = json.loads(curl(f"{REG}/{ns}/blobs/{ly['digest']}"))
            except Exception:
                pass
    return {"tag": tag, "layers": layers, "renderer": cfg.get("renderer"),
            "parser": cfg.get("parser"), "requires": cfg.get("requires"),
            "params": params}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("namespaces", nargs="+")
    ap.add_argument("--expect-draft", type=int)
    a = ap.parse_args()
    bad = 0
    for ns in a.namespaces:
        tl = tags(ns)
        print(f"\n=== {ns}  ({len(tl)} tags) ===")
        seen = {}
        for t in tl:
            r = audit(ns, t)
            if "error" in r:
                print(f"  !! {t}: {r['error']}"); bad += 1; continue
            probs = []
            if "model" not in r["layers"]:
                probs.append("NO model layer")
            if t.startswith("vision-") and "projector" not in r["layers"]:
                probs.append("VISION TAG WITHOUT PROJECTOR")
            if "template" in r["layers"]:
                probs.append(f"template layer present ({r['layers']['template']}B)")
            if not r["renderer"]:
                probs.append("no renderer")
            if not r["parser"]:
                probs.append("no parser")
            if not r["requires"]:
                probs.append("no requires")
            if not r["params"]:
                probs.append("no params")
            elif a.expect_draft is not None and r["params"].get("draft_num_predict") != a.expect_draft:
                probs.append(f"draft_num_predict={r['params'].get('draft_num_predict')}")
            key = (r["renderer"], r["parser"], r["requires"],
                   tuple(sorted(r["layers"])), json.dumps(r["params"], sort_keys=True))
            seen.setdefault(key, []).append(t)
            if probs:
                print(f"  !! {t}: {'; '.join(probs)}"); bad += 1
        print(f"  distinct configurations: {len(seen)}")
        for key, ts in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            rend, pars, req, lay, prm = key
            print(f"    [{len(ts):>2}] layers={'+'.join(lay)} renderer={rend} "
                  f"parser={pars} requires={req}")
            print(f"         params={prm}")
            print(f"         e.g. {', '.join(ts[:4])}{' …' if len(ts) > 4 else ''}")
    print(f"\n>>> anomalies: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
