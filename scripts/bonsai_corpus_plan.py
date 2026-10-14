#!/usr/bin/env python
"""Build the generation plan for a teacher replay corpus (complete conversations, thinking on and off).

Families:
  agent      recorded multi-turn tool conversations of the chat corpus, taken VERBATIM (not generated)
  tools      single-turn corpus prompts that carry a tool list (the teacher ends at its first tool call)
  tech       single-turn corpus prompts without tools
  code       rows of a coding-instruction JSONL (field `instruction`, optional `lang`)
  general    rows of a general-instruction JSONL (`instruction` [+ `context`])
Each generated prompt is used in exactly one mode (think / nothink) and is either train or eval, never both.
Held-out eval prompts and held-out agent conversations are fixed here, before anything is generated.

Writes <out>/plan_train.jsonl, plan_eval.jsonl (id, fam, src, mode, sampler, head), agent_train.jsonl, agent_eval.jsonl
(text, src) and plan.stats.json.
"""
import argparse
import collections
import hashlib
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--general", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-prompt-chars", type=int, default=5000)
ap.add_argument("--n", default="tech:think=800,tech:nothink=0,code:think=0,code:nothink=0,general:think=0,general:nothink=0,tools:think=0",
                help="train prompts per family:mode (0 = all that remain after the other mode and eval took theirs)")
ap.add_argument("--n-eval", type=int, default=48, help="held-out prompts per family:mode")
ap.add_argument("--n-eval-agent", type=int, default=40)
ap.add_argument("--samplers", default="tech:think=think_vendor,code:think=think_vendor,general:think=think_vendor,tools:think=think_vendor",
                help="sampler per family:mode; nothink defaults to nothink_vendor")
ap.add_argument("--code-lang-cap", type=float, default=0.4, help="largest share one language may take of the code family")
ap.add_argument("--efforts", default="", help="Qwen3.8 template only: shares of the reasoning-effort setting over thinking rows, e.g. "
                "xhigh=0.4,medium=0.4,low=0.2. The effort sentence is written into (or removed from) the system block as the template does. Empty = heads untouched")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
SYS, USER, ASST, END = "<|im_start|>system", "<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_end|>"
N = {k: int(v) for k, v in (x.split("=") for x in a.n.split(","))}
SAMPLER = {k: v for k, v in (x.split("=") for x in a.samplers.split(","))}

agent, pools = collections.defaultdict(list), collections.defaultdict(dict)
for seg in open(a.corpus).read().split(SYS)[1:]:
    j = seg.rfind(END)
    i = seg.find(ASST)
    if i < 0 or j < i:
        continue
    conv, head = SYS + seg[: j + len(END)] + "\n", SYS + seg[:i]
    if conv.count(ASST) > 1:
        if conv not in agent[head]:               # several recorded conversations may share one opening
            agent[head].append(conv)
    elif i < a.max_prompt_chars:
        pools["tools" if "<tools>" in head else "tech"].setdefault(head, "corpus")
for fam in ("tools", "tech"):                     # an opening that has a recorded conversation is not also generated
    for h in agent:
        pools[fam].pop(h, None)
for line in open(a.general):
    r = json.loads(line)
    q, ctx = r["instruction"].strip(), r.get("context", "").strip()
    if ctx:
        q = f"{q}\n\n{ctx}"
    if 20 <= len(q) < a.max_prompt_chars:
        pools["general"].setdefault(f"{USER}{q}{END}\n", "dolly-ctx" if ctx else "dolly")
by_lang = collections.defaultdict(list)
for line in open(a.code):
    r = json.loads(line)
    q = r["instruction"].strip()
    if 20 <= len(q) < a.max_prompt_chars:
        by_lang[r.get("lang", "?")].append(f"{USER}{q}{END}\n")

def samp(fam, mode, head):                        # "a|b" splits a family between samplers, by prompt hash
    opts = SAMPLER.get(f"{fam}:{mode}", "nothink_vendor").split("|")
    return opts[int(hashlib.sha1(head.encode()).hexdigest(), 16) % len(opts)]


rid = lambda fam, mode, head: f"{fam}-{mode}-{hashlib.sha1(head.encode()).hexdigest()[:16]}"   # noqa: E731
train, evalp, stats = [], [], {}
for fam in ("tools", "tech", "code", "general"):
    modes = [m for m in ("think", "nothink") if f"{fam}:{m}" in N]
    want = sum(N[f"{fam}:{m}"] for m in modes) + a.n_eval * len(modes)
    if fam == "code":                             # natural language mix, no language above code_lang_cap of the family
        allc = sorted({(h, f"codefeedback-{lang}") for lang, hs in by_lang.items() for h in hs})
        rng.shuffle(allc)
        items, seen, cnt = [], set(), collections.Counter()
        for h, src in allc:
            if h not in seen and cnt[src] < a.code_lang_cap * want and len(items) < want:
                seen.add(h)
                cnt[src] += 1
                items.append((h, src))
    else:
        items = sorted(pools[fam].items())
    rng.shuffle(items)
    k = 0
    for m in modes:                               # eval first, so the held-out set never depends on the train counts
        for h, s in items[k: k + a.n_eval]:
            evalp.append(dict(id=rid(fam, m, h), fam=fam, src=s, mode=m, sampler=samp(fam, m, h), head=h))
        k += a.n_eval
    rest = items[k:]
    k = 0
    for m in sorted(modes, key=lambda m: N[f"{fam}:{m}"] == 0):   # fixed counts first, "all that remain" last
        n = N[f"{fam}:{m}"] or len(rest) - k
        for h, s in rest[k: k + n]:
            train.append(dict(id=rid(fam, m, h), fam=fam, src=s, mode=m, sampler=samp(fam, m, h), head=h))
        k += n
    stats[fam] = dict(pool=len(items))
EFF = {"xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible "
                "alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
       "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
       "medium": ""}


def with_effort(head, effort):
    """The head as the Qwen3.8 template renders it for this effort ("" = thinking off: no effort sentence at all)."""
    for sent in (EFF["xhigh"], EFF["low"]):       # drop whatever the source recorded
        head = head.replace(f"{SYS}\n{sent}{END}\n", "").replace(f"{SYS}\n{sent}\n\n", f"{SYS}\n")
    assert "Reasoning effort is set to" not in head, head[:300]
    sent = EFF.get(effort, "")
    if not sent:
        return head
    if head.startswith(SYS + "\n"):
        return f"{SYS}\n{sent}\n\n" + head[len(SYS) + 1:]
    return f"{SYS}\n{sent}{END}\n" + head


if a.efforts:
    sh = [(k, float(v)) for k, v in (x.split("=") for x in a.efforts.split(","))]
    assert all(k in EFF for k, _ in sh) and abs(sum(v for _, v in sh) - 1) < 1e-6, a.efforts
    for r in train + evalp:
        r["effort"] = ""
        if r["mode"] == "think":
            u, acc = int(hashlib.sha1(("effort" + r["id"]).encode()).hexdigest(), 16) % 10000 / 10000, 0.0
            for k, v in sh:
                acc += v
                if u < acc:
                    break
            r["effort"] = k
        r["head"] = with_effort(r["head"], r["effort"])
ag = sorted(agent.items())
rng.shuffle(ag)
rng.shuffle(train)
os.makedirs(a.out, exist_ok=True)
for name, rows in (("plan_train", train), ("plan_eval", evalp)):
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats[name] = dict(collections.Counter(f"{r['fam']}:{r['mode']}:{r['sampler']}" for r in rows))
n_ev, ag_eval = 0, []
while n_ev < a.n_eval_agent:                      # hold out whole openings, so no eval conversation shares one with train
    ag_eval.append(ag.pop())
    n_ev += len(ag_eval[-1][1])
for name, rows in (("agent_eval", ag_eval), ("agent_train", ag)):
    cs = [c for _h, v in rows for c in v]
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for c in cs:
            f.write(json.dumps(dict(text=c, src="agent-verbatim", turns=c.count(ASST), chars=len(c))) + "\n")
    stats[name] = dict(convs=len(cs), openings=len(rows), chars=sum(map(len, cs)), assistant_turns=sum(c.count(ASST) for c in cs))
ids = [r["id"] for r in train + evalp]
heads_t, heads_e = {r["head"] for r in train}, {r["head"] for r in evalp} | {h for h, _ in ag_eval}
assert len(ids) == len(set(ids)), "duplicate plan ids"
assert not heads_t & heads_e, "a held-out prompt is also a train prompt"
assert not heads_e & {h for h, _ in ag}, "a held-out prompt opens a train agent conversation"
stats["code_langs_train"] = dict(collections.Counter(r["src"] for r in train if r["fam"] == "code").most_common(12))
json.dump(dict(args=vars(a), stats=stats), open(f"{a.out}/plan.stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
print("PLAN_DONE")
