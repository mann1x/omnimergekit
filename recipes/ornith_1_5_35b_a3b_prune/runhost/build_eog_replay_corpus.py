#!/usr/bin/env python
"""Generate a VALID EOG replay corpus: real serving-format generations that terminate.

WHY THE OLD CORPUS WAS INVALID (Ornith, 2026-09-07)
  router_calib_corpus_ornith_full.jsonl replays assistant turns as

      <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<CoT as ordinary content><|im_end|>

  i.e. THINKING-DISABLED format with an EMPTY think block. The model at serve time emits
  <think>[13k tokens of reasoning]</think><answer><|im_end|>. So the emit-map measured
  termination in a regime where no thinking ever happened -- structurally unlike the
  regime under investigation. Two further defects: 51% of its EOG positions were
  USER-turn closes (prompt tokens the model never generates), and 80 ifeval docs carried
  an unclosed <think>.

WHAT THIS BUILDS
  For each prompt, the BASE model's own generation, captured raw via --reasoning-format
  none so the thinking channel survives verbatim, and kept ONLY when the model
  terminated by itself (finish_reason == stop). A truncated generation has no terminator
  and would poison the very statistic we are measuring.

  Output doc:  <|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{raw}<|im_end|>\\n

  The trailing <|im_end|> is the ONLY EOG in the assistant turn and it is a REAL model
  stop after a REAL chain of thought. Pair with build_eog_emit_map.py
  --assistant-stops-only so user-turn closes are excluded from the emit mask.
"""
import argparse, json, re, sys, time, urllib.request

OPEN = "<" + "think" + ">"
CLOSE = "</" + "think" + ">"


def post(base, path, payload, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def user_turns(path):
    """Pull the user turn out of each doc of the old corpus."""
    out = []
    pat = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = pat.search(r["text"])
        if m:
            out.append((r.get("bench", "?"), m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-corpus", required=True)
    ap.add_argument("--base-url", default="http://localhost:8096")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cat", type=int, default=20,
                    help="prompts per category (long-CoT cats get --per-cat-long)")
    ap.add_argument("--per-cat-long", type=int, default=60,
                    help="prompts for targeted_lcb / targeted_mpe -- the failure regime")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--http-timeout", type=int, default=1800)
    a = ap.parse_args()

    LONG = {"targeted_lcb", "targeted_mpe"}
    by_cat = {}
    for cat, p in user_turns(a.old_corpus):
        by_cat.setdefault(cat, []).append(p)

    picked = []
    for cat, ps in sorted(by_cat.items()):
        n = a.per_cat_long if cat in LONG else a.per_cat
        picked += [(cat, p) for p in ps[:n]]
    print("prompts selected: %d across %d categories" % (len(picked), len(by_cat)), flush=True)

    kept = dropped_trunc = dropped_nothink = failed = 0
    t0 = time.time()
    with open(a.out, "w") as fh:
        for i, (cat, prompt) in enumerate(picked):
            rendered = ("<|im_start|>user\n" + prompt +
                        "<|im_end|>\n<|im_start|>assistant\n")
            try:
                r = post(a.base_url, "/completion", {
                    "prompt": rendered, "n_predict": a.max_tokens,
                    "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
                    "cache_prompt": False, "stream": False,
                }, a.http_timeout)
            except Exception as e:
                failed += 1
                print("  [%d/%d] %s REQUEST FAILED: %s" % (i + 1, len(picked), cat, e), flush=True)
                continue
            raw = r.get("content", "")
            # ONLY natural terminations: a truncated generation has no terminator.
            if r.get("stopped_limit") or not r.get("stopped_eos"):
                dropped_trunc += 1
            elif OPEN not in raw or CLOSE not in raw:
                dropped_nothink += 1      # must exercise the thinking channel
            else:
                fh.write(json.dumps({
                    "bench": cat,
                    "text": ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                             "<|im_start|>assistant\n" + raw + "<|im_end|>\n"),
                }) + "\n")
                fh.flush()
                kept += 1
            if (i + 1) % 10 == 0:
                print("  [%4d/%d] kept=%d trunc=%d nothink=%d failed=%d  %.0fs"
                      % (i + 1, len(picked), kept, dropped_trunc, dropped_nothink,
                         failed, time.time() - t0), flush=True)

    print("\nkept=%d dropped_truncated=%d dropped_nothink=%d request_failed=%d"
          % (kept, dropped_trunc, dropped_nothink, failed))
    if kept == 0:
        sys.exit("REFUSING: corpus is empty -- every generation was dropped")
    print(">>> EOG_REPLAY_CORPUS_OK docs=%d -> %s" % (kept, a.out))


if __name__ == "__main__":
    main()
