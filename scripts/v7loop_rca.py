#!/usr/bin/env python
# v7loop_rca.py — reproduce + isolate the v7-coder Q4_K_M looping complaint (HF discussion #1).
#
# DPS-900 ran:  llama-server -hf ...:Q4_K_M --jinja -ngl 99 -fa on -ctk q8_0 -ctv q8_0
#               --fit off --cache-prompt -c 49152 --temp 0.6 --tools all
# and got degenerate loops ("Actually, I'll try to fix the meta tag. / Wait, ...").
# The omitted flags vs the model card's "How to Use": --reasoning-format deepseek
# --reasoning-budget 8192. Default --reasoning-budget is -1 (UNBOUNDED thinking) and
# --reasoning auto (active for this channel-format template). Hypothesis: unbounded
# rumination, not a Q4_K_M defect. This harness proves/refutes by isolating the variable.
#
# Each config launches its own llama-server on GPU1:8200, runs the prompts, kills it.
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from collections import Counter
from pathlib import Path

LLAMA = "/opt/llama.cpp/build/bin/llama-server"
Q4KM  = "/mnt/sdc/ml/quant_sweep_gguf/gemma-4-A4B-98e-v7-coder-it-Q4_K_M.gguf"
Q6K   = "/mnt/sdc/ml/quant_sweep_gguf_dl/gemma-4-A4B-98e-v7-coder-it-Q6_K.gguf"
HOST, PORT = "127.0.0.1", 8200
OUT = Path("/srv/ml/rca_v7loop")

# user's exact non-sampling server args (plain-chat faithful; --tools omitted = web-GUI case)
COMMON = ["--jinja","-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0",
          "--fit","off","-c","49152","--host",HOST,"--port",str(PORT),"--no-warmup"]

# name -> (model, extra_server_args, request_overrides, note)
CONFIGS = {
  "A_user_q4km":   (Q4KM, [], {"temperature":0.6},
                    "user's exact config: reasoning=auto, budget=-1 (UNBOUNDED)"),
  "B_q4km_budget": (Q4KM, ["--reasoning-format","deepseek","--reasoning-budget","8192"], {"temperature":0.6},
                    "card recipe: thinking capped at 8192"),
  "C_user_q6k":    (Q6K,  [], {"temperature":0.6},
                    "Q6_K under user's bad config (quant-specificity control)"),
  "D_q4km_greedy": (Q4KM, [], {"temperature":0.0,"top_p":1.0,"top_k":1},
                    "Q4_K_M user config but GREEDY (sampler control)"),
  "E_q4km_budgreedy": (Q4KM, ["--reasoning-format","deepseek","--reasoning-budget","8192"], {"temperature":0.0,"top_p":1.0,"top_k":1},
                    "Q4_K_M + budget + greedy (card recipe at canonical sampler)"),
}

PROMPTS = [
  {"id":"viewport", "text":
   "I'm using Next.js 14 (App Router). My viewport meta tag isn't taking effect on "
   "mobile - the page renders zoomed out. Here's my app/layout.tsx:\n\n"
   "export default function RootLayout({ children }: { children: React.ReactNode }) {\n"
   "  return (\n"
   "    <html lang=\"en\">\n"
   "      <head>\n"
   "        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
   "      </head>\n"
   "      <body>{children}</body>\n"
   "    </html>\n"
   "  );\n}\n\nFix it."},
  {"id":"flexcenter", "text":
   "This flexbox isn't centering my modal vertically inside its container. Why, and "
   "what's the fix?\n\n.container { display: flex; justify-content: center; }\n"
   ".modal { width: 400px; }\n"},
  {"id":"palindrome_ctrl", "text":
   "Write a Python function `def is_palindrome(s: str) -> bool` that returns True if "
   "the string is a palindrome, ignoring case and non-alphanumeric characters. Add a "
   "couple of test assertions."},
]

def launch(model, extra, logpath):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "1"
    args = [LLAMA, "-m", model] + COMMON + extra
    f = open(logpath, "wb")
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    return p

def wait_health(timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=5)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def kill(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try: os.killpg(os.getpgid(p.pid), sig)
        except Exception: pass
        try:
            p.wait(timeout=20); return
        except Exception:
            continue

def chat(prompt, overrides, max_tokens=8192, seed=0):
    body = {"messages":[{"role":"user","content":prompt}],
            "max_tokens":max_tokens, "seed":seed, "stream":False}
    body.update(overrides)
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions",
                                 data=data, headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        js = json.loads(r.read().decode())
    js["_elapsed_s"] = round(time.time()-t0, 1)
    return js

def analyze(js):
    ch = js["choices"][0]
    msg = ch.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = ch.get("finish_reason")
    full = (reasoning + "\n" + content).strip()
    words = full.split()
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    max_line_rep = max(Counter(lines).values()) if lines else 0
    grams = Counter(tuple(words[i:i+8]) for i in range(len(words)-8)) if len(words) > 8 else Counter()
    max_gram = max(grams.values()) if grams else 0
    looped = (finish == "length") and (max_line_rep >= 4 or max_gram >= 6)
    # also: did it terminate naturally with a short, sane answer?
    return dict(finish=finish, words=len(words), think_words=len(reasoning.split()),
                content_words=len(content.split()), max_line_rep=max_line_rep,
                max_8gram=max_gram, looped=looped, elapsed_s=js.get("_elapsed_s"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="A_user_q4km,B_q4km_budget")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for cname in args.configs.split(","):
        cname = cname.strip()
        if cname not in CONFIGS:
            print(f"!! unknown config {cname}"); continue
        model, extra, ov, note = CONFIGS[cname]
        if not Path(model).is_file():
            print(f"!! SKIP {cname}: model missing {model}"); continue
        greedy = ov.get("temperature", 0.6) == 0.0
        seeds = [0] if greedy else list(range(args.seeds))
        logp = OUT / f"server_{cname}.log"
        print(f"\n===== CONFIG {cname} =====\n  {note}\n  model={Path(model).name}\n  extra={extra}\n  req={ov}", flush=True)
        p = launch(model, extra, logp)
        try:
            if not wait_health():
                print(f"!! {cname}: server failed health (see {logp})"); kill(p); continue
            print("  server up.", flush=True)
            for pr in PROMPTS:
                for sd in seeds:
                    try:
                        js = chat(pr["text"], ov, max_tokens=args.max_tokens, seed=sd)
                    except Exception as e:
                        print(f"  [{pr['id']} seed={sd}] REQUEST ERROR: {e}", flush=True); continue
                    a = analyze(js)
                    rec = dict(config=cname, prompt=pr["id"], seed=sd, **a)
                    summary.append(rec)
                    (OUT / f"resp_{cname}_{pr['id']}_s{sd}.json").write_text(json.dumps(js, indent=2))
                    flag = "LOOP" if a["looped"] else "ok  "
                    print(f"  [{flag}] {pr['id']:16s} seed={sd} finish={a['finish']:6s} "
                          f"words={a['words']:5d} (think={a['think_words']} ans={a['content_words']}) "
                          f"line_rep={a['max_line_rep']} 8gram={a['max_8gram']} {a['elapsed_s']}s", flush=True)
        finally:
            kill(p)
            time.sleep(3)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    # final table
    print("\n\n===== SUMMARY =====")
    print(f"{'config':18s} {'prompt':16s} {'seed':4s} {'finish':7s} {'words':6s} {'lrep':5s} {'8gr':4s} loop")
    for r in summary:
        print(f"{r['config']:18s} {r['prompt']:16s} {r['seed']:<4d} {r['finish']:7s} "
              f"{r['words']:<6d} {r['max_line_rep']:<5d} {r['max_8gram']:<4d} {'LOOP' if r['looped'] else ''}")
    # per-config loop rate
    print("\n--- loop rate per config ---")
    by = {}
    for r in summary:
        by.setdefault(r["config"], []).append(r["looped"])
    for c, v in by.items():
        print(f"  {c:18s} {sum(v)}/{len(v)} looped")

if __name__ == "__main__":
    main()
