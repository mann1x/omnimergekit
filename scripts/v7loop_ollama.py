#!/usr/bin/env python
# v7loop_ollama.py — does v7-coder Q4_K_M loop when served by OLLAMA, across ctx sizes
# and sampler settings? ollama defaults: num_ctx 4096, repeat_penalty 1.1, min_p 0, temp ~0.8.
import json, urllib.request
from collections import Counter

URL="http://localhost:11434/api/chat"
MODEL="v7loop-test"
PROMPTS=[
 {"id":"flexcenter","text":"This flexbox isn't centering my modal vertically inside its container. Why, and what's the fix?\n\n.container { display: flex; justify-content: center; }\n.modal { width: 400px; }\n"},
 {"id":"viewport","text":"I'm using Next.js 14 (App Router). My viewport meta tag isn't taking effect on mobile - the page renders zoomed out. Here's my app/layout.tsx:\n\nexport default function RootLayout({ children }) {\n  return (<html lang=\"en\"><head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" /></head><body>{children}</body></html>);\n}\n\nFix it."},
]
# name -> options (ollama). think=True always (reasoning model).
CONFIGS={
 "O1_default_4k":  {"num_ctx":4096, "temperature":0.6, "num_predict":8192},                                  # ollama defaults (rp=1.1, no min_p)
 "O2_norp_4k":     {"num_ctx":4096, "temperature":0.6, "num_predict":8192, "repeat_penalty":1.0},            # disable rp (mirror llama.cpp default)
 "O3_fixed_4k":    {"num_ctx":4096, "temperature":0.6, "num_predict":8192, "min_p":0.05, "repeat_penalty":1.1},
 "O4_fixed_8k":    {"num_ctx":8192, "temperature":0.6, "num_predict":8192, "min_p":0.05, "repeat_penalty":1.1},
 "O5_fixed_32k":   {"num_ctx":32768,"temperature":0.6, "num_predict":8192, "min_p":0.05, "repeat_penalty":1.1},
}
def chat(prompt, opts, seed):
    o=dict(opts); o["seed"]=seed
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"think":True,"stream":False,"options":o}
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=1200).read().decode())
def analyze(js):
    m=js.get("message",{}); ct=m.get("content") or ""; th=m.get("thinking") or ""
    fin=js.get("done_reason"); full=(th+"\n"+ct).strip(); w=full.split()
    lines=[l.strip() for l in full.splitlines() if l.strip()]
    mlr=max(Counter(lines).values()) if lines else 0
    g=Counter(tuple(w[i:i+8]) for i in range(len(w)-8)) if len(w)>8 else Counter()
    mg=max(g.values()) if g else 0
    return dict(done=fin,words=len(w),think=len(th.split()),ans=len(ct.split()),
                lrep=mlr,g8=mg,loop=(fin=="length" and (mlr>=4 or mg>=6)))
summ=[]
for cn,opts in CONFIGS.items():
    print("\n===== %s  %s ====="%(cn,opts),flush=True)
    for pr in PROMPTS:
        for sd in (0,1):
            try: js=chat(pr["text"],opts,sd)
            except Exception as e: print("  ERR",pr["id"],sd,e,flush=True); continue
            a=analyze(js); summ.append(dict(cfg=cn,prompt=pr["id"],seed=sd,**a))
            print("  [%s] %-11s s%d done=%-6s words=%5d think=%5d ans=%4d lrep=%d g8=%d"%(
                "LOOP" if a["loop"] else "ok  ",pr["id"],sd,str(a["done"]),a["words"],a["think"],a["ans"],a["lrep"],a["g8"]),flush=True)
json.dump(summ,open("/srv/ml/rca_v7loop/ollama_summary.json","w"),indent=2)
print("\n--- ollama loop rate per config ---")
by={}
for r in summ: by.setdefault(r["cfg"],[]).append(r["loop"])
for c,v in by.items(): print("  %-16s %d/%d looped"%(c,sum(v),len(v)))
