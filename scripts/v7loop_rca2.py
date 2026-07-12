#!/usr/bin/env python
# v7loop_rca2.py — round 2: quant-specificity (Q6_K) + PROPER budget test + sampler fix.
# Round 1 proved: real intermittent thinking-channel repetition collapse on ambiguous
# prompts (flexcenter reliably loops; palindrome control never does). Round-1 "B" was a
# non-test (max_tokens==budget so budget never fired). Here: budget 2048 << max_tokens 8192.
import argparse, json, os, signal, subprocess, time, urllib.request
from collections import Counter
from pathlib import Path

LLAMA="/opt/llama.cpp/build/bin/llama-server"
Q4KM="/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf"
Q6K ="/mnt/sdc/ml/quant_sweep_gguf_dl/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf"
HOST,PORT="127.0.0.1",8200
OUT=Path("/srv/ml/rca_v7loop")
COMMON=["--jinja","-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0","--fit","off",
        "-c","49152","--host",HOST,"--port",str(PORT),"--no-warmup"]

CONFIGS={
 "C_q6k_bad":      (Q6K, [], {"temperature":0.6},
                    "Q6_K under user bad config (QUANT-SPECIFICITY control)"),
 "G_q4km_budget2k":(Q4KM, ["--reasoning-format","deepseek","--reasoning-budget","2048"], {"temperature":0.6},
                    "budget 2048 << max_tokens 8192 (PROPER budget test)"),
 "K_q4km_antirep": (Q4KM, [], {"temperature":0.6,"min_p":0.05,"repeat_penalty":1.1},
                    "anti-repetition sampling (min_p 0.05 + repeat_penalty 1.1)"),
 "L_q4km_dry":     (Q4KM, [], {"temperature":0.6,"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2},
                    "DRY sampler (the targeted loop-breaker)"),
}
PROMPTS=[
 {"id":"viewport","text":
  "I'm using Next.js 14 (App Router). My viewport meta tag isn't taking effect on "
  "mobile - the page renders zoomed out. Here's my app/layout.tsx:\n\n"
  "export default function RootLayout({ children }: { children: React.ReactNode }) {\n"
  "  return (\n    <html lang=\"en\">\n      <head>\n"
  "        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
  "      </head>\n      <body>{children}</body>\n    </html>\n  );\n}\n\nFix it."},
 {"id":"flexcenter","text":
  "This flexbox isn't centering my modal vertically inside its container. Why, and "
  "what's the fix?\n\n.container { display: flex; justify-content: center; }\n"
  ".modal { width: 400px; }\n"},
 {"id":"palindrome_ctrl","text":
  "Write a Python function `def is_palindrome(s: str) -> bool` that returns True if the "
  "string is a palindrome, ignoring case and non-alphanumeric characters. Add a couple "
  "of test assertions."},
]

def launch(model,extra,logpath):
    env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="1"
    return subprocess.Popen([LLAMA,"-m",model]+COMMON+extra,
        stdout=open(logpath,"wb"),stderr=subprocess.STDOUT,env=env,preexec_fn=os.setsid)
def wait_health(t=300):
    t0=time.time()
    while time.time()-t0<t:
        try:
            if urllib.request.urlopen("http://%s:%d/health"%(HOST,PORT),timeout=5).status==200: return True
        except Exception: pass
        time.sleep(2)
    return False
def kill(p):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try: os.killpg(os.getpgid(p.pid),sig)
        except Exception: pass
        try: p.wait(timeout=20); return
        except Exception: continue
def chat(prompt,ov,max_tokens,seed):
    body={"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"seed":seed,"stream":False}
    body.update(ov)
    req=urllib.request.Request("http://%s:%d/v1/chat/completions"%(HOST,PORT),
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    js=json.loads(urllib.request.urlopen(req,timeout=1800).read().decode())
    js["_elapsed_s"]=round(time.time()-t0,1); return js
def analyze(js):
    ch=js["choices"][0]; m=ch.get("message",{})
    ct=m.get("content") or ""; rc=m.get("reasoning_content") or ""
    fr=ch.get("finish_reason"); full=(rc+"\n"+ct).strip(); w=full.split()
    lines=[l.strip() for l in full.splitlines() if l.strip()]
    mlr=max(Counter(lines).values()) if lines else 0
    g=Counter(tuple(w[i:i+8]) for i in range(len(w)-8)) if len(w)>8 else Counter()
    mg=max(g.values()) if g else 0
    return dict(finish=fr,words=len(w),think_words=len(rc.split()),content_words=len(ct.split()),
                max_line_rep=mlr,max_8gram=mg,looped=(fr=="length" and (mlr>=4 or mg>=6)),
                elapsed_s=js.get("_elapsed_s"))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--configs",default=",".join(CONFIGS))
    ap.add_argument("--seeds",type=int,default=2)
    ap.add_argument("--max-tokens",type=int,default=8192)
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); summ=[]
    for cn in a.configs.split(","):
        cn=cn.strip()
        if cn not in CONFIGS: print("!! unknown",cn); continue
        model,extra,ov,note=CONFIGS[cn]
        if not Path(model).is_file(): print("!! SKIP missing model",cn,model); continue
        seeds=[0] if ov.get("temperature",0.6)==0.0 else list(range(a.seeds))
        print("\n===== CONFIG %s =====\n  %s\n  model=%s extra=%s req=%s"%(cn,note,Path(model).name,extra,ov),flush=True)
        p=launch(model,extra,OUT/("server2_%s.log"%cn))
        try:
            if not wait_health(): print("!! health fail",cn); kill(p); continue
            print("  server up.",flush=True)
            for pr in PROMPTS:
                for sd in seeds:
                    try: js=chat(pr["text"],ov,a.max_tokens,sd)
                    except Exception as e: print("  REQ ERR",pr["id"],sd,e,flush=True); continue
                    an=analyze(js); summ.append(dict(config=cn,prompt=pr["id"],seed=sd,**an))
                    (OUT/("resp2_%s_%s_s%d.json"%(cn,pr["id"],sd))).write_text(json.dumps(js,indent=2))
                    print("  [%s] %-16s s%d finish=%-6s words=%5d think=%5d ans=%4d lrep=%d 8gram=%d %ss"%(
                        "LOOP" if an["looped"] else "ok  ",pr["id"],sd,an["finish"],an["words"],
                        an["think_words"],an["content_words"],an["max_line_rep"],an["max_8gram"],an["elapsed_s"]),flush=True)
        finally:
            kill(p); time.sleep(3)
    (OUT/"summary2.json").write_text(json.dumps(summ,indent=2))
    print("\n--- loop rate per config ---")
    by={}
    for r in summ: by.setdefault(r["config"],[]).append(r["looped"])
    for c,v in by.items(): print("  %-18s %d/%d looped"%(c,sum(v),len(v)))
if __name__=="__main__": main()
