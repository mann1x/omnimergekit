"""Non-verbatim degenerate-behaviour oracles for the agentic replay harness.

`detect.py` is a high-precision *verbatim* loop detector: it fires only when a
1-3 sentence block repeats **exactly** >= 4x, or a 5-gram repeats >= 12x. Field
reports (HF discussions on v7-coder / Qwen3.6-omnimerge) showed that the loops
users actually hit are mostly NOT verbatim, so the verbatim gate reported a clean
run while the model was visibly degenerate. The three missed families:

  A. PARAPHRASE / discourse-marker cycles -- the same intent restated with
     rotating leading markers, so no fixed block is identical:
         "I'll write it. Wait, I'll write it. Actually, I'll just do it.
          Let's go. (I will provide the content directly.) I'll write it. ..."
     Signature: after stripping leading discourse markers + trailing
     parentheticals, a tiny set of sentence "cores" dominates a long trace.

  B. TEMPLATED ENUMERATION -- structurally identical lines that differ only by a
     number / identifier / URL / version, so every n-gram is distinct:
         ".wrapfont2 { font-family: serif; } .wrapfont3 { font-family: ... }"
         "use three@15.0.1 ... use three.js/4.13.0 ... use three@4.18.0 ..."
     Signature: collapse each line to a char-class skeleton (letters->a,
     digits->#, urls->U); a tiny set of skeletons dominates.

  C. OVER-THINKING / non-termination -- the dominant *current* complaint:
     "2,000 lines of reasoning for a 95-line script", "larger loops, not quick
     or direct", reasoning that runs to the budget / context cap without
     committing to an answer. Not a loop at all by repetition measure; a
     length / non-commitment signal.

These are reported as SEPARATE verdicts (`paraphrase_loop`, `template_loop`,
`overthinking`) so they never silently change the meaning of the verbatim
`detect.py` gate. A caller chooses which oracles count toward "fail".

Stdlib only, so the harness stays vendorable.
"""
from __future__ import annotations

import re
from collections import Counter

from .detect import _norm, _sentences

# ===========================================================================
# Oracle A -- paraphrase / discourse-marker cycle
# ===========================================================================

A_MIN_SENTS = 10          # too few sentences to judge a cycle
A_STEM_WORDS = 3          # compare sentence "stems" (first N words of the core)
A_MIN_STEM_CHARS = 8      # ignore trivial stems ("go", "ok then")
A_TOP_REPEAT = 4          # the dominant stem recurs >= this many times
A_TOP_SHARE = 0.22        # ... and covers >= this fraction of all sentences

# leading discourse markers stripped before comparing cores. The loop is the
# SAME core ("i'll write it") wrapped in rotating markers ("wait,", "actually,").
_MARKERS = (
    r"wait", r"actually", r"okay", r"ok", r"so", r"hmm+", r"alright",
    r"right", r"well", r"now", r"then", r"also", r"but", r"and", r"no",
    r"yes", r"hold on", r"one more thing", r"self[- ]correction",
    r"let'?s", r"first", r"second", r"finally", r"oh", r"hmm",
)
_LEAD = re.compile(r"^(?:[(\[\"'*\s]*(?:%s)\b[\s,.:;)\-]*)+" % "|".join(_MARKERS))
_BRACKETS = re.compile(r"[()\[\]{}]")
_URLRE = re.compile(r"https?://\S+|www\.\S+")
_VERRE = re.compile(r"\b\d+(?:\.\d+){1,}\b")          # 15.0.1, 4.13.0
_NUMRE = re.compile(r"\d+")


def _core(sentence: str) -> str:
    """Reduce a sentence to its comparable core: drop bracket characters,
    normalise numbers, then strip leading discourse markers. URLs/versions are
    normalised to 'u' by the caller BEFORE sentence-splitting (they contain
    dots, so splitting first would shred them). So 'Wait, I'll write it now' and
    'Actually, I'll write it' both reduce to the stem 'i'll write it'."""
    s = _BRACKETS.sub(" ", sentence)
    s = _NUMRE.sub("#", s)
    s = _norm(s)
    prev = None
    while prev != s:                 # strip stacked leading markers
        prev = s
        s = _LEAD.sub("", s).strip()
    s = re.sub(r"[\s,.:;!?\-]+$", "", s)
    return s


def detect_paraphrase_cycle(text: str):
    """Return (is_loop, info). Fires when one sentence-stem dominates a long
    trace (interleaved paraphrase) -- the case the verbatim block-equality
    detector cannot see, because the loop has rotating markers / trailing
    clauses so no fixed block is identical. The stem (first few words of the
    marker-stripped core) merges 'I'll write it' / 'I'll write it now' /
    'Wait, I'll write it', which is exactly the observed cycle unit."""
    pre = _URLRE.sub(" u ", text or "")    # collapse before split (urls have dots)
    pre = _VERRE.sub(" u ", pre)
    cores = [c for c in (_core(s) for s in _sentences(pre)) if c]
    n = len(cores)
    if n < A_MIN_SENTS:
        return False, {}
    stems = [" ".join(c.split()[:A_STEM_WORDS]) for c in cores]
    cand = Counter(s for s in stems if len(s) >= A_MIN_STEM_CHARS)
    if not cand:
        return False, {}
    top, topn = cand.most_common(1)[0]
    share = topn / n
    is_loop = topn >= A_TOP_REPEAT and share >= A_TOP_SHARE
    return is_loop, {
        "kind": "paraphrase",
        "top_core": top,
        "top_repeats": topn,
        "n_sentences": n,
        "share": round(share, 3),
        "distinct_stems": len(cand),
    }


# ===========================================================================
# Oracle B -- templated enumeration (structure repeats, tokens vary)
# ===========================================================================

B_MIN_LINES = 8
B_DISTINCT_RATIO = 0.34
B_TOP_REPEAT = 6
B_MIN_SKEL_CHARS = 6

_LETTERS = re.compile(r"[A-Za-z_]+")


def _skeleton(line: str) -> str:
    """Char-class skeleton: urls->U, version/number runs->#, letter runs->a,
    whitespace collapsed, punctuation kept. Two lines that differ only by their
    identifiers / numbers / values collapse to the same skeleton."""
    s = _URLRE.sub("U", line)
    s = _VERRE.sub("#", s)
    s = _NUMRE.sub("#", s)
    s = _LETTERS.sub("a", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_template_cycle(text: str):
    """Return (is_loop, info). Fires when structurally-identical lines (same
    skeleton) dominate -- the '.wrapfontN { ... }' / cycling-CDN-version family
    where every literal n-gram is distinct."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    skels = [k for k in (_skeleton(ln) for ln in lines) if len(k) >= B_MIN_SKEL_CHARS]
    n = len(skels)
    if n < B_MIN_LINES:
        return False, {}
    cnt = Counter(skels)
    top, topn = cnt.most_common(1)[0]
    distinct_ratio = len(cnt) / n
    is_loop = topn >= B_TOP_REPEAT or distinct_ratio < B_DISTINCT_RATIO
    return is_loop, {
        "kind": "template",
        "top_skeleton": top,
        "top_repeats": topn,
        "n_lines": n,
        "distinct_skeletons": len(cnt),
        "distinct_ratio": round(distinct_ratio, 3),
    }


# ===========================================================================
# Oracle C -- over-thinking / non-termination
# ===========================================================================

C_RUMINATE_THINK_CHARS = 8000   # long reasoning ...
C_EMPTY_ANS_CHARS = 200         # ... with no real answer/tool == not committing
C_RATIO = 20                    # think/answer ratio ...
C_RATIO_MIN_THINK = 4000        # ... above this absolute reasoning size
C_BUDGET_FRAC = 0.95            # completion_tokens >= frac*budget == hit the cap


def detect_overthinking(content, reasoning, finish_reason=None,
                        completion_tokens=None, budget_tokens=None,
                        has_tool=False):
    """Return (is_overthinking, info). A non-repetition failure: the model
    ruminates without committing, hits the length/budget cap, or burns a wildly
    disproportionate amount of reasoning relative to what it produced."""
    alen = len(content or "")
    tlen = len(reasoning or "")
    reasons = []
    if finish_reason == "length":
        reasons.append("finish=length")
    if budget_tokens and completion_tokens and \
            completion_tokens >= C_BUDGET_FRAC * budget_tokens:
        reasons.append("hit_budget(%s/%s)" % (completion_tokens, budget_tokens))
    if tlen >= C_RUMINATE_THINK_CHARS and alen < C_EMPTY_ANS_CHARS and not has_tool:
        reasons.append("ruminate_no_commit(think=%d ans=%d)" % (tlen, alen))
    if tlen >= C_RATIO_MIN_THINK and tlen > C_RATIO * max(alen, 1):
        reasons.append("think>>answer(%dx)" % (tlen // max(alen, 1)))
    return (len(reasons) > 0), {"kind": "overthinking", "reasons": reasons,
                                "think_len": tlen, "answer_len": alen}


# ===========================================================================
# combined assessment
# ===========================================================================

def assess_extra(content, reasoning, finish_reason=None,
                 completion_tokens=None, budget_tokens=None, has_tool=False):
    """Run the three new oracles over a turn and return a flat verdict dict.
    Paraphrase + template oracles run on BOTH channels (loops appear in
    thinking *and* answer); over-thinking is inherently cross-channel."""
    ans = content or ""
    rea = reasoning or ""
    p_a, pi_a = detect_paraphrase_cycle(rea)
    p_b, pi_b = detect_paraphrase_cycle(ans)
    t_a, ti_a = detect_template_cycle(ans)
    t_b, ti_b = detect_template_cycle(rea)
    over, oi = detect_overthinking(ans, rea, finish_reason, completion_tokens,
                                   budget_tokens, has_tool)
    paraphrase = p_a or p_b
    template = t_a or t_b
    return {
        "paraphrase_loop": bool(paraphrase),
        "template_loop": bool(template),
        "overthinking": bool(over),
        "soft_fail": bool(paraphrase or template or over),
        "paraphrase_info": pi_a if p_a else pi_b,
        "template_info": ti_a if t_a else ti_b,
        "overthinking_info": oi,
    }


def soft_label(v):
    """Compact human tag for the new oracles, mirroring detect.loop_label."""
    parts = []
    if v.get("paraphrase_loop"):
        i = v.get("paraphrase_info") or {}
        parts.append("PARA(x%s:%r)" % (i.get("top_repeats"),
                                       (i.get("top_core") or "")[:34]))
    if v.get("template_loop"):
        i = v.get("template_info") or {}
        parts.append("TMPL(x%s:%r)" % (i.get("top_repeats"),
                                       (i.get("top_skeleton") or "")[:34]))
    if v.get("overthinking"):
        i = v.get("overthinking_info") or {}
        parts.append("OVERTHINK(%s)" % ",".join(i.get("reasons") or []))
    return " ".join(parts)


# ===========================================================================
# self-test -- built from the EXACT strings users posted in the HF threads
# ===========================================================================

if __name__ == "__main__":
    # A: paraphrase cycle (DPS-900, gemma v7-coder #1) -- NOT verbatim-blockable
    para = (
        "Let's go. (I will provide the complete code in one write call.) "
        "Actually, I'll write it now. (If this fails, I'll try to fix it.) "
        "One more thing: I'll add some simple lighting and shaders. "
        "Okay, let's do this. (I'll provide the full content in one write call.) "
        "Actually, I'll write it now. (Wait, I'll use a CDN for Three.js: "
        "https://cdnjs.cloudflare.com/ajax/libs/three.js/4.18.0/three.min.js) "
        "Let's go. (I will provide the complete code in one write call.) "
        "Actually, I'll write it now. (If this fails, I'll try to fix it.) "
        "Wait, I'll write it. I'll write it. Wait, I'll write it. "
        "Actually, I'll just do it. Wait, I'll write it. I'll write it."
    )
    # B: templated enumeration (DPS-900) -- every line distinct by a token
    tmpl = "\n".join(".wrapfont%d { font-family: %s; }" % (i, v) for i, v in
                     enumerate(["serif", "sans-serif", "monospace", "italic",
                                "bold", "underline", "uppercase", "lowercase",
                                "mixedcase", "titlecase", "camelcase",
                                "snakecase", "kebabcase", "PascalCase",
                                "SCREAMING_SNAKE_CASE"], start=2))
    # C: over-thinking -- long reasoning, no committed answer
    over = "I should check this. " * 600   # ~12k chars of rumination

    # clean controls that must NOT trip the oracles
    clean_ans = ("Here is the plan. First create index.html with the canvas. "
                 "Then add main.js to set up the renderer, camera and scene. "
                 "Finally wire the animation loop and the time-speed slider.")
    clean_code = "\n".join([
        "function init() { renderer = new THREE.WebGLRenderer(); }",
        "const camera = new THREE.PerspectiveCamera(75, w/h, 0.1, 1000);",
        "scene.add(sun); planets.forEach(p => scene.add(p.mesh));",
        "function animate(t) { requestAnimationFrame(animate); render(); }",
        "slider.addEventListener('input', e => speed = e.target.value);",
        "const sun = makeSphere(5, 0xffff00); sun.position.set(0,0,0);",
        "function makeSphere(r, c) { return new THREE.Mesh(geo, mat); }",
        "document.body.appendChild(renderer.domElement); resize();",
    ])

    def show(tag, content, reasoning, **kw):
        v = assess_extra(content, reasoning, **kw)
        print("%-9s soft=%-5s  %s" % (tag, v["soft_fail"], soft_label(v)))
        return v

    pa = show("PARA", "", para)
    tm = show("TMPL", tmpl, "")
    ov = show("OVER", "", over)
    c1 = show("clean1", clean_ans, "")
    c2 = show("clean2", clean_code, "")
    c3 = show("clean3", "", clean_ans)

    ok = (pa["paraphrase_loop"] and tm["template_loop"] and ov["overthinking"]
          and not c1["soft_fail"] and not c2["soft_fail"] and not c3["soft_fail"])
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
