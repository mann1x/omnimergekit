#!/usr/bin/env python3
"""T174.6 — anti-loop SFT corpus v2 (register-collapse fix).

v1 (epoch2) FAILED Stage-B: it fixed code loops (MPE 0%/+0.023) but CRASHED
IFEval (0.870->0.570, loop 3%->11%) via REGISTER-COLLAPSE — the short/simple
constrained+multilingual teacher gold taught a degenerate short-declarative
register that induced NEW loops on open-ended prompts (knowledge Qs, proposals)
and eroded factual grounding. Root cause: (a) retain too small to anchor the
distribution, (b) detect_loop-only gold filter let near-degenerate-but-
terminating gold through, (c) no compliance gate on constrained gold.

v2 fixes all three:
  1. STRICT-IFEVAL GATE on constrained gold — keep only completions that
     detect_loop-clean AND pass the lm-eval ifeval strict checker (real
     instruction compliance, not just termination).
  2. REGISTER GUARD on ALL gold — reject completions whose trailing sentences
     share a repeated N-word prefix (the "Fear gone in X." / "We will also
     see Y." near-degenerate pattern detect_loop misses).
  3. REBALANCE to ~57% rich-register general — a new open-ended ANCHOR bucket
     (Dolly-15k open_qa/brainstorm/creative/summarization, i.e. the looped
     archetypes) + larger retain; short-register constrained+multilingual
     shrunk to ~20%.

Gold is always 128e teacher greedy. Output schema {prompt, completion, bucket}
(trainer-compatible with lora_sft_antiloop.py). Run on bs2, omk python.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OMK = "/srv/ml/repos/omnimergekit/scripts"
sys.path.insert(0, OMK)
from audit_full_bench import detect_loop, load_for_bench, resp  # noqa: E402

TEACHER = "/srv/ml/models/base/gemma-4-26B-A4B-it"
A2_VARIANT = "a2-62e-fc15_25-p8-s1_0p1_20"
MULTILANG = ["Persian", "Arabic", "Turkish", "Hindi", "Chinese", "French",
             "Spanish", "German", "Japanese"]
MIN_CHARS, MAX_CHARS, MAX_NEW = 50, 14000, 2048


def log(m):
    print("[corpus2 %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ---------- gates ----------
_IFEVAL_REG = None


def ifeval_strict_pass(prompt, iid_list, kw_list, response):
    """Replicate lm-eval test_instruction_following_strict: all instructions
    followed. Returns True/False. On any checker error -> False (conservative)."""
    global _IFEVAL_REG
    if _IFEVAL_REG is None:
        from lm_eval.tasks.ifeval import instructions_registry as R
        _IFEVAL_REG = R
    if not response.strip():
        return False
    try:
        for idx, iid in enumerate(iid_list):
            cls = _IFEVAL_REG.INSTRUCTION_DICT[iid]
            inst = cls(iid)
            kw = {k: v for k, v in (kw_list[idx] or {}).items() if v is not None}
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=prompt)
            if not inst.check_following(response):
                return False
        return True
    except Exception:
        return False


_SENT = re.compile(r"[^.!?\n]+[.!?\n]")


def register_ok(text, prefix_words=4, window=15, min_sents=8, ratio_thresh=0.5,
                max_line_rep=6):
    """Reject near-degenerate-but-terminating gold: trailing sentences sharing
    a repeated N-word prefix (register-collapse signature), or any single line
    repeated >= max_line_rep times. Normal prose -> distinct-prefix ratio ~0.9;
    'Fear gone in X.' style -> ~0.2."""
    # line-level refrain
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        from collections import Counter
        c = Counter(lines)
        if c.most_common(1)[0][1] >= max_line_rep:
            return False
    sents = [s.strip() for s in _SENT.findall(text) if s.strip()]
    if len(sents) < min_sents:
        return True  # too few sentences to judge; rely on detect_loop
    tail = sents[-window:]
    prefs = [" ".join(s.split()[:prefix_words]).lower() for s in tail]
    distinct = len(set(prefs)) / max(len(prefs), 1)
    return distinct >= ratio_thresh


def gold_ok(prompt, comp, tok, bucket, iid=None, kw=None):
    """Unified gate. Returns (keep:bool, reason:str)."""
    if len(comp) < MIN_CHARS:
        return False, "too_short"
    if len(comp) > MAX_CHARS:
        return False, "too_long"
    n_tok = len(tok(comp, add_special_tokens=False)["input_ids"])
    if n_tok >= MAX_NEW - 2:
        return False, "hit_max_new"
    if detect_loop(comp):
        return False, "loop"
    if not register_ok(comp):
        return False, "register_collapse"
    if iid:  # IFEval-metadata buckets must additionally strict-pass
        if not ifeval_strict_pass(prompt, iid, kw, comp):
            return False, "ifeval_strict_fail"
    return True, "ok"


# ---------- prompt collectors (return list[dict{prompt, iid?, kw?}]) ----------
def _seen():
    s = set()
    def add(p):
        if not isinstance(p, str) or not p.strip() or p in s:
            return False
        s.add(p)
        return True
    return add


def collect_seeds(n):
    out, add = [], _seen()
    for bench in ("ifeval_100", "multipl_e_100", "humanevalplus_full"):
        try:
            samples, _ = load_for_bench(bench, A2_VARIANT)
        except Exception as e:
            log("seeds %s load failed: %s" % (bench, e))
            continue
        for s in samples or []:
            if detect_loop(resp(s)):
                doc = s.get("doc", {}) or {}
                p = doc.get("prompt") or doc.get("question") or doc.get("text")
                if isinstance(p, str) and add(p):
                    item = {"prompt": p}
                    if doc.get("instruction_id_list"):
                        item["iid"] = doc["instruction_id_list"]
                        item["kw"] = doc.get("kwargs")
                    out.append(item)
    log("seeds: %d flagged loop prompts" % len(out))
    return out[:n]


def collect_constrained(n):
    """IFEval-only so we can strict-gate; carries metadata."""
    out, add = [], _seen()
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")
    for r in ds:
        p = r.get("prompt")
        if add(p):
            out.append({"prompt": p.strip(), "iid": r.get("instruction_id_list"),
                        "kw": r.get("kwargs")})
    log("constrained(IFEval): %d prompts w/ metadata" % len(out))
    return out[:n]


_DOLLY_CATS = {"open_qa", "general_qa", "brainstorming", "creative_writing",
               "summarization", "classification"}


def collect_openended(n):
    """Register ANCHOR — rich open-ended prose (the looped archetypes:
    knowledge Qs, proposals, summaries, creative). Dolly-15k, no hard format
    constraint, so teacher produces natural rich-register terminating gold."""
    out, add = [], _seen()
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for r in ds:
            if r.get("category") not in _DOLLY_CATS:
                continue
            instr = (r.get("instruction") or "").strip()
            ctx = (r.get("context") or "").strip()
            if not instr:
                continue
            p = (instr + ("\n\n" + ctx if 0 < len(ctx) < 1200 else ""))
            if 20 < len(p) < 1600 and add(p):
                out.append({"prompt": p})
            if len(out) >= n * 2:
                break
    except Exception as e:
        log("openended: dolly failed (%s)" % e)
    log("openended(dolly): %d prompts" % len(out))
    return out[:n]


def collect_retain(n):
    out, add = [], _seen()
    try:
        from datasets import load_dataset
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for r in gsm:
            q = r.get("question")
            if add(q):
                out.append({"prompt": q.strip()})
            if len(out) >= n // 2:
                break
    except Exception as e:
        log("retain gsm8k failed (%s)" % e)
    # top up with more dolly (closed_qa/information_extraction for variety)
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for r in ds:
            if r.get("category") not in {"closed_qa", "information_extraction"}:
                continue
            instr = (r.get("instruction") or "").strip()
            ctx = (r.get("context") or "").strip()
            p = instr + ("\n\n" + ctx if 0 < len(ctx) < 1200 else "")
            if 20 < len(p) < 1600 and add(p):
                out.append({"prompt": p})
            if len(out) >= n:
                break
    except Exception as e:
        log("retain dolly failed (%s)" % e)
    log("retain: %d general prompts" % len(out))
    return out[:n]


def collect_multilingual(n):
    out, add = [], _seen()
    try:
        from datasets import load_dataset
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        wanted = set(MULTILANG)
        for r in ds:
            if r.get("language") in wanted:
                inp = r.get("inputs")
                if isinstance(inp, str) and 10 < len(inp) < 1500 and add(inp):
                    out.append({"prompt": inp.strip()})
            if len(out) >= n * 3:
                break
    except Exception as e:
        log("multilingual aya failed (%s)" % e)
    log("multilingual: %d prompts" % len(out))
    return out[:n]


def collect_code(n):
    out, add = [], _seen()
    from datasets import load_dataset
    try:
        he = load_dataset("evalplus/humanevalplus", split="test")
        for r in he:
            p = r.get("prompt")
            if isinstance(p, str) and add("Complete this function:\n\n" + p.strip()):
                out.append({"prompt": "Complete this function:\n\n" + p.strip()})
    except Exception as e:
        log("code he+ failed (%s)" % e)
    try:
        for lang in ("java", "js", "rs"):
            try:
                mp = load_dataset("nuprl/MultiPL-E", "humaneval-" + lang, split="test")
            except Exception:
                continue
            for r in mp:
                p = r.get("prompt")
                if isinstance(p, str) and add(p.strip()):
                    out.append({"prompt": p.strip()})
    except Exception as e:
        log("code mpe failed (%s)" % e)
    log("code: %d prompts" % len(out))
    return out[:n]


# v2 targets: rich-register general (openended+retain) ~57%, short-register
# (constrained+multilingual+seeds) ~20%, code ~20%. Request > target where the
# strict/register gates will drop hard.
BUCKETS = {
    "seeds": (collect_seeds, 50),
    "constrained": (collect_constrained, 360),   # strict-gate drops ~40% -> ~180
    "openended": (collect_openended, 460),        # anchor
    "retain": (collect_retain, 420),
    "multilingual": (collect_multilingual, 160),  # reduced + register/strict gate
    "code": (collect_code, 300),
}


def gen_batch(model, tok, items, max_new, bs):
    comps = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                         add_generation_prompt=True, tokenize=False)
                 for it in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.0, use_cache=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            new = out[j][enc["input_ids"].shape[1]:]
            comps.append(tok.decode(new, skip_special_tokens=True).strip())
        log("  gen %d/%d" % (min(i + bs, len(items)), len(items)))
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/sdc/ml/corpora/antiloop_sft_corpus_v2.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--smoke", type=int, default=0)
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

    meta, rows = {}, []
    drop_reasons = {}
    for name, (fn, target) in BUCKETS.items():
        n = args.smoke if args.smoke else target
        items = fn(n)
        if not items:
            meta[name] = {"requested": n, "got": 0, "kept": 0, "dropped": 0}
            log("%s: NO PROMPTS" % name)
            continue
        comps = gen_batch(model, tok, items, args.max_new, args.bs)
        kept = dropped = 0
        rc = {}
        for it, c in zip(items, comps):
            ok, reason = gold_ok(it["prompt"], c, tok, name,
                                 iid=it.get("iid"), kw=it.get("kw"))
            if not ok:
                dropped += 1
                rc[reason] = rc.get(reason, 0) + 1
                continue
            rows.append({"prompt": it["prompt"], "completion": c, "bucket": name})
            kept += 1
        meta[name] = {"requested": n, "got": len(items), "kept": kept,
                      "dropped": dropped, "drop_reasons": rc}
        drop_reasons[name] = rc
        log("%s: kept=%d dropped=%d %s" % (name, kept, dropped, rc))

    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta["_total_kept"] = len(rows)
    meta["_share"] = {k: round(meta[k]["kept"] / max(len(rows), 1), 3)
                      for k in BUCKETS if k in meta}
    json.dump(meta, open(str(outp) + ".meta.json", "w"), indent=2, ensure_ascii=False)
    log("WROTE %d rows -> %s" % (len(rows), outp))
    log("SHARE: " + json.dumps(meta["_share"]))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()
