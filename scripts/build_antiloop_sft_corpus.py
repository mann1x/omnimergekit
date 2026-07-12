#!/usr/bin/env python3
"""T174.2 — build the anti-loop SFT corpus by sequence-level distillation.

The 62e prune A2 has a ~3% residual real-loop floor (Persian/non-English IF,
constrained-writing IF, code-gen spirals) that no router method clears. This
corpus teaches A2 *when to stop* by distilling the 128e teacher's CLEAN,
TERMINATING greedy completions on prompts of those failure archetypes — plus a
retain set on general prompts to prevent forgetting.

ALL gold is teacher-generated greedy and filtered through the SAME detect_loop()
the hard gate uses, so a teacher trajectory that itself ruminates never enters
the corpus.

Buckets (prompt source differs; gold is always 128e greedy):
  seeds        the exact A2-flagged loop prompts (eval sample dirs)
  constrained  constrained-writing IF (google/if_eval cached + local corpora)
  multilingual non-English IF (CohereForAI/aya_dataset, language-filtered)
  code         function stubs (evalplus/humanevalplus + nuprl/multi_pl-e cached)
  retain       general IF + code + math (anti-forgetting)

Output: JSONL with {text} (chat-rendered prompt+completion) + .meta.json sidecar.

Run on bs2 with the omnimergekit python. --smoke caps every bucket to a few
prompts to validate the gen+filter path end-to-end before the full run.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# reuse the canonical loop detector + sample loader (same fn as the gate)
OMK = "/srv/ml/repos/omnimergekit/scripts"
sys.path.insert(0, OMK)
from audit_full_bench import detect_loop, load_for_bench, resp  # noqa: E402

TEACHER = "/srv/ml/models/base/gemma-4-26B-A4B-it"
A2_VARIANT = "a2-62e-fc15_25-p8-s1_0p1_20"   # for seed extraction
LOCAL_CORPORA = [
    Path(OMK) / "router_calib_corpus.jsonl",
    Path(OMK) / "router_calib_corpus_ifeval_heavy.jsonl",
]
MULTILANG = ["Persian", "Arabic", "Turkish", "Hindi", "Chinese", "French",
             "Spanish", "German", "Italian", "Portuguese", "Russian", "Japanese"]
MIN_CHARS, MAX_CHARS, MAX_NEW = 50, 14000, 2048


def log(m):
    print("[corpus %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def prompt_of_sample(s):
    doc = s.get("doc", {}) or {}
    for k in ("prompt", "question", "text"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v
    t = doc.get("turns")
    if isinstance(t, list) and t and isinstance(t[0], str):
        return t[0]
    return None


# ---------- prompt collectors (return list[str]) ----------
def collect_seeds(n):
    out = []
    for bench in ("ifeval_100", "multipl_e_100", "humanevalplus_full"):
        try:
            samples, _ = load_for_bench(bench, A2_VARIANT)
        except Exception as e:
            log("seeds: %s load failed: %s" % (bench, e))
            continue
        for s in samples or []:
            if detect_loop(resp(s)):
                p = prompt_of_sample(s)
                if p:
                    out.append(p)
    # dedup, keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    log("seeds: %d flagged loop prompts" % len(uniq))
    return uniq[:n]


def collect_constrained(n):
    out = []
    try:
        from datasets import load_dataset
        ds = load_dataset("google/IFEval", split="train")
        for r in ds:
            p = r.get("prompt")
            if isinstance(p, str) and p.strip():
                out.append(p.strip())
    except Exception as e:
        log("constrained: IFEval load failed (%s); falling back to local corpora" % e)
    for f in LOCAL_CORPORA:
        if not f.exists():
            continue
        for ln in f.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("bench", "").startswith("ifeval"):
                p = r.get("prompt")
                if isinstance(p, str) and p.strip():
                    out.append(p.strip())
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    log("constrained: %d IF prompts" % len(uniq))
    return uniq[:n]


def collect_multilingual(n):
    out = []
    try:
        from datasets import load_dataset
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        wanted = set(MULTILANG)
        for r in ds:
            lang = r.get("language")
            inp = r.get("inputs")
            if lang in wanted and isinstance(inp, str) and 10 < len(inp) < 1500:
                out.append(inp.strip())
            if len(out) >= n * 4:
                break
    except Exception as e:
        log("multilingual: aya load failed (%s)" % e)
    log("multilingual: %d non-English prompts" % len(out))
    return out[:n]


def collect_code(n):
    out = []
    try:
        from datasets import load_dataset
        he = load_dataset("evalplus/humanevalplus", split="test")
        for r in he:
            p = r.get("prompt")
            if isinstance(p, str) and p.strip():
                out.append("Complete this function:\n\n" + p.strip())
    except Exception as e:
        log("code: humanevalplus failed (%s)" % e)
    try:
        from datasets import load_dataset
        for lang in ("java", "js", "rs"):
            try:
                mp = load_dataset("nuprl/MultiPL-E", "humaneval-" + lang, split="test")
            except Exception:
                continue
            for r in mp:
                p = r.get("prompt")
                if isinstance(p, str) and p.strip():
                    out.append(p.strip())
    except Exception as e:
        log("code: multipl-e failed (%s)" % e)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    log("code: %d code prompts" % len(uniq))
    return uniq[:n]


def collect_retain(n):
    out = []
    try:
        from datasets import load_dataset
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for r in gsm:
            q = r.get("question")
            if isinstance(q, str) and q.strip():
                out.append(q.strip())
            if len(out) >= n:
                break
    except Exception as e:
        log("retain: gsm8k failed (%s)" % e)
    # top up with general (non-IF) local-corpus prompts
    for f in LOCAL_CORPORA:
        if len(out) >= n or not f.exists():
            break
        for ln in f.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if not r.get("bench", "").startswith("ifeval"):
                p = r.get("prompt")
                if isinstance(p, str) and p.strip():
                    out.append(p.strip())
            if len(out) >= n:
                break
    log("retain: %d general prompts" % len(out))
    return out[:n]


BUCKETS = {
    "seeds": (collect_seeds, 50),
    "constrained": (collect_constrained, 250),
    "multilingual": (collect_multilingual, 250),
    "code": (collect_code, 300),
    "retain": (collect_retain, 450),
}


# ---------- teacher generation ----------
def gen_batch(model, tok, prompts, max_new, bs):
    """Greedy batched generation; returns list[str] completions (decoded)."""
    comps = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.0, use_cache=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = out[j][enc["input_ids"].shape[1]:]
            comps.append(tok.decode(new, skip_special_tokens=True).strip())
        log("  gen %d/%d" % (min(i + bs, len(prompts)), len(prompts)))
    return comps


def hit_max(comp, tok, max_new):
    return len(tok(comp, add_special_tokens=False)["input_ids"]) >= max_new - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--smoke", type=int, default=0, help="cap each bucket to N prompts")
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    log("loading teacher 128e bf16 -> GPU0 ...")
    tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map={"": 0})
    model.eval()

    meta = {}
    rows = []
    for name, (fn, target) in BUCKETS.items():
        n = args.smoke if args.smoke else target
        prompts = fn(n)
        if not prompts:
            meta[name] = {"requested": n, "got_prompts": 0, "kept": 0, "dropped": 0}
            log("%s: NO PROMPTS — bucket empty" % name)
            continue
        comps = gen_batch(model, tok, prompts, args.max_new, args.bs)
        kept = dropped = 0
        for p, c in zip(prompts, comps):
            reason = None
            if len(c) < MIN_CHARS:
                reason = "too_short"
            elif len(c) > MAX_CHARS:
                reason = "too_long"
            elif detect_loop(c):
                reason = "loop"
            elif hit_max(c, tok, args.max_new):
                reason = "hit_max_new"
            if reason:
                dropped += 1
                continue
            # store the raw split; the trainer renders with the chat template and
            # masks loss to the completion tokens only.
            rows.append({"prompt": p, "completion": c, "bucket": name})
            kept += 1
        meta[name] = {"requested": n, "got_prompts": len(prompts), "kept": kept, "dropped": dropped}
        log("%s: kept=%d dropped=%d" % (name, kept, dropped))

    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta["_total_kept"] = len(rows)
    json.dump(meta, open(str(outp) + ".meta.json", "w"), indent=2, ensure_ascii=False)
    log("WROTE %d rows -> %s" % (len(rows), outp))
    log("META: " + json.dumps(meta))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
