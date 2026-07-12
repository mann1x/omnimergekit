#!/usr/bin/env python3
"""Bake anti-loop sampler params into ALL ollama tags of the v7/v6 coder models.
Per-tag: pull -> create(+PARAMS) -> verify(params added, template/renderer/parser preserved)
-> push -> rm (bound disk). Resumable via .done markers. CSS-noise tags filtered out.
Origin: HF v7-coder discussion #1 loop complaint RCA (2026-06-08)."""
import os, re, subprocess, sys, time, urllib.request

NS = "mannix"
MODELS = ["gemma4-98e-v7-coder", "gemma4-98e-v7-coderx", "gemma4-98e-v6-coder"]
PARAMS = [("temperature","0.6"), ("min_p","0.05"), ("repeat_penalty","1.1"), ("num_ctx","8192")]
WORK = "/srv/ml/rca_v7loop"
DONE = os.path.join(WORK, "bake_done"); os.makedirs(DONE, exist_ok=True)
TAG_RE = re.compile(r"^(CD-|mtp-)?(qat-)?(F16|Q[0-9][A-Za-z0-9_]*|IQ[0-9][A-Za-z0-9_]*)$")

def real_tag(tag):
    t = tag[7:] if tag.startswith("vision-") else tag
    return t in ("qat","latest") or bool(TAG_RE.match(t))

def get_tags(model):
    url = "https://ollama.com/%s/%s/tags" % (NS, model)
    html = urllib.request.urlopen(url, timeout=40).read().decode("utf-8","ignore")
    raw = set(re.findall(r"%s:([A-Za-z0-9_.-]+)" % re.escape(model), html))
    return sorted(t for t in raw if real_tag(t))

def run(cmd, to=3600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=to)

def renderer_parser(mf):
    return sorted(l for l in mf.splitlines() if l.startswith(("RENDERER","PARSER")))

def process(model, tag):
    full = "%s/%s:%s" % (NS, model, tag)
    marker = os.path.join(DONE, "%s__%s.done" % (model, tag.replace("/","_")))
    if os.path.exists(marker):
        return "DONE-CACHED"
    r = run(["ollama","pull",full])
    if r.returncode != 0:
        blob = (r.stderr + r.stdout).lower()
        if "not found" in blob or "404" in blob or "file does not exist" in blob:
            open(marker,"w").write("skip404\n"); return "SKIP404"
        return "FAILPULL: " + (r.stderr or r.stdout).strip()[:120]
    before = run(["ollama","show",full,"--modelfile"], to=120).stdout
    mf = os.path.join(WORK, "mf_bake_%s_%s.txt" % (model, tag.replace("/","_")))
    with open(mf,"w") as f:
        f.write("FROM %s\n" % full)
        for k,v in PARAMS: f.write("PARAMETER %s %s\n" % (k,v))
    r = run(["ollama","create",full,"-f",mf], to=1800)
    if r.returncode != 0:
        return "FAILCREATE: " + (r.stderr or r.stdout).strip()[:120]
    after = run(["ollama","show",full,"--modelfile"], to=120).stdout
    if not all(("PARAMETER %s %s" % (k,v)) in after for k,v in PARAMS):
        return "FAILVERIFY-params"
    if renderer_parser(before) != renderer_parser(after):
        return "FAILVERIFY-renderer(before=%s after=%s)" % (renderer_parser(before), renderer_parser(after))
    if after.count("TEMPLATE") < before.count("TEMPLATE") or before.count("TEMPLATE")==0 and "RENDERER" not in after:
        return "FAILVERIFY-template"
    r = run(["ollama","push",full])
    if r.returncode != 0:
        return "FAILPUSH: " + (r.stderr or r.stdout).strip()[:120]
    run(["ollama","rm",full], to=300)
    try: os.remove(mf)
    except Exception: pass
    open(marker,"w").write("ok\n")
    return "OK"

def main():
    t0 = time.time(); summary = {}
    for model in MODELS:
        try: tags = get_tags(model)
        except Exception as e:
            print("[%s] FAILED to list tags: %s" % (model, e), flush=True); continue
        print("=== %s : %d real tags ===" % (model, len(tags)), flush=True)
        print("    " + " ".join(tags), flush=True)
        for i, tag in enumerate(tags, 1):
            st = process(model, tag)
            key = st.split(":")[0]
            summary[key] = summary.get(key,0)+1
            print("[%s %3d/%d] %-22s -> %s" % (model.replace("gemma4-98e-",""), i, len(tags), tag, st), flush=True)
    dt = int(time.time()-t0)
    print("=== BAKE SUMMARY (%dm%02ds) ===" % (dt//60, dt%60), flush=True)
    for k in sorted(summary): print("   %-16s %d" % (k, summary[k]), flush=True)
    print("BAKE-COMPLETE", flush=True)

if __name__ == "__main__":
    main()
