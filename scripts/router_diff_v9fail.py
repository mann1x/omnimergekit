#!/usr/bin/env python3
"""v9-ml — router differential on gemma-4-20B-v9-agentic's formatting / multilingual failures.

The T183/T189 method (format8 found 8 format experts this way), pointed at v9 map A:
  Phase 1  v9 (bf16) answers the HELD-OUT calibration prompts greedily on the answer channel.
           Each answer is judged: IFEval prompts by the stock lm-eval IFEval scorer
           (prompt-level strict); aya native-language prompts by target-script share and
           detect_loop.
  Phase 2  every sequence is teacher-forced through the full 128e; the router softmax over 128
           experts on the COMPLETION tokens is accumulated onto the experts map A DROPS.
           fail-token mass vs pass-token mass -> which dropped experts the 128e leans on
           exactly where v9 gets it wrong.

CONTAMINATION RULE. The test probe (ifeval_v9fail10) and ifeval_100 are IFEval rows 0-99.
Candidates are ranked from the SELECTION stream only (held-out rows 100-540 + aya). v9's real
Q6_K failures on the test keys are teacher-forced as a separate DIAGNOSTIC stream: reported
next to the ranking as an agreement check, never used to rank.
"""
import argparse
import json
import sqlite3  # noqa: F401  (must precede torch on solidpc; harmless on bs2)
import sys
import time
import unicodedata

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from audit_full_bench import detect_loop  # noqa: E402

# aya language -> Unicode script name prefix (unicodedata.name). Latin-script languages are
# judged on loops only: a script share cannot tell Swahili from English.
SCRIPT = {
    "Kannada": "KANNADA", "Panjabi": "GURMUKHI", "Marathi": "DEVANAGARI", "Hindi": "DEVANAGARI",
    "Nepali": "DEVANAGARI", "Gujarati": "GUJARATI", "Bengali": "BENGALI", "Tamil": "TAMIL",
    "Telugu": "TELUGU", "Malayalam": "MALAYALAM", "Sinhala": "SINHALA", "Urdu": "ARABIC",
    "Iranian Persian": "ARABIC", "Standard Arabic": "ARABIC", "Korean": "HANGUL",
    "Japanese": ("HIRAGANA", "KATAKANA", "CJK"), "Simplified Chinese": "CJK", "Thai": "THAI",
    "Burmese": "MYANMAR", "Amharic": "ETHIOPIC", "Russian": "CYRILLIC",
}
MIN_SCRIPT_SHARE = 0.80


def log(m):
    print("[diff %s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def script_share(text, want):
    want = want if isinstance(want, tuple) else (want,)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hit = sum(1 for c in letters if unicodedata.name(c, "").startswith(want))
    return hit / len(letters)


def judge(row, resp, ifeval_docs, process_results):
    if detect_loop(resp):
        return False, "loop"
    src = row["src"]
    if src.startswith("ifeval:"):
        doc = ifeval_docs[int(src.split(":")[1])]
        ok = bool(process_results(doc, [resp])["prompt_level_strict_acc"])
        return ok, "ifeval_strict" if not ok else "ok"
    want = SCRIPT.get(row.get("lang") or "")
    if want is None:
        return True, "ok_loop_only"
    share = script_share(resp, want)
    return (share >= MIN_SCRIPT_SHARE), ("ok" if share >= MIN_SCRIPT_SHARE else f"script {share:.2f}")


def render(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", required=True, help="v9 bf16 dir (generates Phase 1)")
    ap.add_argument("--base", required=True, help="128e bf16 dir (teacher-forced Phase 2)")
    ap.add_argument("--drop-map", required=True, help="map A drop map json")
    ap.add_argument("--calib", action="append", required=True, metavar="CLASS=JSONL",
                    help="held-out prompts per class, e.g. multilingual=... ifeval=...")
    ap.add_argument("--diag", default=None,
                    help="JSONL {key, prompt, response}: v9's real failing answers on the TEST keys")
    ap.add_argument("--gen-tokens", type=int, default=2048)
    ap.add_argument("--gens-out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dm = json.load(open(a.drop_map))
    dm = dm.get("drop_map", dm)
    dropped = {int(k): sorted(v) for k, v in dm.items() if str(k).isdigit()}
    L = len(dropped)

    from datasets import load_dataset
    from lm_eval.tasks.ifeval.utils import process_results
    ifeval_docs = {d["key"]: d for d in load_dataset("google/IFEval", split="train")}
    test_keys = {d["key"] for d in load_dataset("google/IFEval", split="train").select(range(100))}

    work = []
    for spec in a.calib:
        cls, path = spec.split("=", 1)
        for x in open(path):
            if x.strip():
                r = json.loads(x)
                if r["src"].startswith("ifeval:") and int(r["src"].split(":")[1]) in test_keys:
                    raise SystemExit(f"CONTAMINATION: {r['src']} is a test key")
                work.append({"stream": "selection", "cls": cls, **r})
    log(f"selection prompts: {len(work)}")

    tok = AutoTokenizer.from_pretrained(a.pruned)
    seqs = []
    done = {}
    try:
        for line in open(a.gens_out):
            g = json.loads(line)
            done[(g["cls"], g["prompt"])] = g
        log(f"resuming: {len(done)} generations already saved")
    except FileNotFoundError:
        pass

    todo = [w for w in work if (w["cls"], w["prompt"]) not in done]
    if todo:
        log(f"PHASE 1: load v9 -> cuda:0, generate {len(todo)} greedy answers (answer channel)")
        m = AutoModelForCausalLM.from_pretrained(a.pruned, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 device_map={"": 0}).eval()
        with open(a.gens_out, "a") as gf:
            for i, w in enumerate(todo):
                enc = tok(render(tok, w["prompt"]), return_tensors="pt", add_special_tokens=False).to("cuda:0")
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = m.generate(**enc, max_new_tokens=a.gen_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
                comp = out[0][plen:].tolist()
                resp = tok.decode(comp, skip_special_tokens=True)
                ok, why = judge(w, resp, ifeval_docs, process_results)
                g = {"cls": w["cls"], "src": w["src"], "lang": w.get("lang"), "prompt": w["prompt"],
                     "ids": out[0].tolist(), "plen": plen, "ok": ok, "why": why,
                     "n_comp": len(comp), "resp": resp}
                gf.write(json.dumps(g, ensure_ascii=False) + "\n")
                gf.flush()
                done[(w["cls"], w["prompt"])] = g
                log(f"  [{i + 1}/{len(todo)}] {w['cls']:12} {w['src']:14} {'PASS' if ok else 'FAIL'} "
                    f"{why:14} {len(comp)} tok")
        del m
        torch.cuda.empty_cache()
    for w in work:
        g = done[(w["cls"], w["prompt"])]
        seqs.append({"stream": "selection", "cls": g["cls"], "ok": g["ok"], "plen": g["plen"], "ids": g["ids"]})

    if a.diag:
        for x in open(a.diag):
            r = json.loads(x)
            assert r["key"] in test_keys, r["key"]
            p_ids = tok(render(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            c_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
            seqs.append({"stream": "diagnostic", "cls": "test_fail", "ok": False,
                         "plen": len(p_ids), "ids": p_ids + c_ids})
        log(f"diagnostic sequences: {sum(s['stream'] == 'diagnostic' for s in seqs)}")

    for cls in sorted({s["cls"] for s in seqs if s["stream"] == "selection"}):
        ss = [s for s in seqs if s["stream"] == "selection" and s["cls"] == cls]
        log(f"Phase 1 verdicts {cls}: fail {sum(not s['ok'] for s in ss)} / {len(ss)}")

    log("PHASE 2: load 128e -> cuda:0, teacher-force + capture routing")
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                device_map={"": 0}).eval()
    E = base.config.text_config.num_experts
    routers = sorted([(int(n.split("layers.")[1].split(".")[0]), mod) for n, mod in base.named_modules()
                      if n.endswith(".router")], key=lambda t: t[0])
    if len(routers) != L:
        raise SystemExit(f"router count {len(routers)} != map layers {L}")
    cap = {}

    def mk(li):
        def hook(_m, _i, out):
            cap[li] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        return hook
    handles = [routers[li][1].register_forward_hook(mk(li)) for li in range(L)]
    dmask = np.zeros((L, E), dtype=bool)
    for li, ds in dropped.items():
        dmask[li, ds] = True

    acc, ntok = {}, {}
    softmax_fixed = False
    for s in seqs:
        ids = torch.tensor([s["ids"]], device="cuda:0")
        cap.clear()
        with torch.no_grad():
            base(input_ids=ids, attention_mask=torch.ones_like(ids),
                 mm_token_type_ids=torch.zeros_like(ids), use_cache=False)
        key = (s["stream"], s["cls"], "pass" if s["ok"] else "fail")
        acc.setdefault(key, np.zeros((L, E)))
        n = 0
        for li in range(L):
            p = cap[li]
            p = p[0] if p.dim() == 3 else p
            if p.shape[-1] != E:
                raise SystemExit(f"router output last dim {p.shape[-1]} != {E}")
            p = p[s["plen"]:]
            rs = p.sum(-1)
            if not torch.allclose(rs, torch.ones_like(rs), atol=1e-2):
                p = torch.softmax(p, dim=-1)
                softmax_fixed = True
            acc[key][li] += (p.sum(0).numpy() * dmask[li])
            n = p.shape[0]
        ntok[key] = ntok.get(key, 0) + n
    for h in handles:
        h.remove()
    if softmax_fixed:
        log("NOTE: router hook returned unnormalised scores; softmax applied")

    def per_tok(stream, cls, verdict):
        k = (stream, cls, verdict)
        return acc[k] / max(ntok[k], 1) if k in acc else np.zeros((L, E))

    sel_classes = sorted({s["cls"] for s in seqs if s["stream"] == "selection"})
    fail = {c: per_tok("selection", c, "fail") for c in sel_classes}
    passm = {c: per_tok("selection", c, "pass") for c in sel_classes}
    diag = per_tok("diagnostic", "test_fail", "fail")

    def ranks(mat):
        v = [(mat[li, e], li, e) for li in range(L) for e in dropped[li]]
        v.sort(key=lambda t: (-t[0], t[1], t[2]))
        return {(li, e): i for i, (_, li, e) in enumerate(v)}

    rk = {c: ranks(fail[c]) for c in sel_classes}
    rows = []
    for li in range(L):
        for e in dropped[li]:
            r = {"layer": li, "expert": e, "diag_fail_mass": round(float(diag[li, e]), 5)}
            for c in sel_classes:
                r[f"{c}_fail"] = round(float(fail[c][li, e]), 5)
                r[f"{c}_pass"] = round(float(passm[c][li, e]), 5)
                r[f"{c}_spec"] = round(float(fail[c][li, e] / (passm[c][li, e] + 1e-4)), 2)
                r[f"{c}_rank"] = rk[c][(li, e)]
            r["shared_rank"] = max(r[f"{c}_rank"] for c in sel_classes)
            rows.append(r)
    shared = sorted(rows, key=lambda r: (r["shared_rank"], r["layer"], r["expert"]))
    diag_top = sorted(rows, key=lambda r: -r["diag_fail_mass"])[:32]
    dset = {(r["layer"], r["expert"]) for r in diag_top}
    agree = {k: len({(r["layer"], r["expert"]) for r in shared[:k]} & dset) for k in (8, 16, 32)}

    out = {"method": "router differential, 128e teacher-forced over v9 map A outputs",
           "drop_map": a.drop_map, "pruned": a.pruned, "base": a.base,
           "n_tok": {"/".join(k): v for k, v in ntok.items()},
           "verdicts": {c: {"fail": sum(1 for s in seqs if s["stream"] == "selection" and s["cls"] == c and not s["ok"]),
                            "n": sum(1 for s in seqs if s["stream"] == "selection" and s["cls"] == c)}
                        for c in sel_classes},
           "dropped_mass_per_tok": {"/".join(k): round(float((acc[k] / max(ntok[k], 1)).sum()), 4) for k in acc},
           "shared_top64": shared[:64],
           "diag_top32": diag_top,
           "shared_vs_diag_overlap": agree,
           "softmax_applied": softmax_fixed}
    json.dump(out, open(a.out, "w"), indent=1)
    log("=" * 64)
    for k, v in out["dropped_mass_per_tok"].items():
        log(f"  dropped mass/tok {k:40} {v:.4f}  (n_tok={ntok[tuple(k.split('/'))]})")
    log("  shared ranking (selection stream), top 16:")
    for r in shared[:16]:
        log("    L%-2d e%-3d " % (r["layer"], r["expert"])
            + "  ".join(f"{c}: fail={r[c + '_fail']:.4f} spec={r[c + '_spec']:.1f}x" for c in sel_classes)
            + f"  diag={r['diag_fail_mass']:.4f}")
    log(f"  agreement with the diagnostic (test-failure) top-32: {agree}")
    log(f"DIFF_DONE -> {a.out}")


if __name__ == "__main__":
    main()
