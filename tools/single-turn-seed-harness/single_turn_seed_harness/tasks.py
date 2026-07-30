"""Prompt-set loader for the single-turn seed sweep.

Reproduces the June "single-turn HumanEval+ / MultiPL-E seed sweep" prompt set:

  * he   -- HumanEval+ : all 164 problems from `evalplus/humanevalplus`
            (falls back to `openai_humaneval` if evalplus is unavailable).
  * cpp  -- MultiPL-E C++       : `nuprl/MultiPL-E`, config `humaneval-cpp`.
  * js   -- MultiPL-E JavaScript: `nuprl/MultiPL-E`, config `humaneval-js`.

Each problem is turned into a SINGLE-TURN chat request: one `user` message that
hands the model the function stub (signature + docstring / leading comment) and
asks it to complete the function. Thinking is left enabled server-side (the
Gemma-4 chat template + `--reasoning-budget`), matching the field condition the
June test isolated -- the model reasons first, then emits the completion, and
the re-injection / verbatim-repetition failure lives in that reasoning->answer
phase.

Every loader is best-effort and offline-friendly: pass a smaller `limit` to trim
the set, or point HF_DATASETS_CACHE at a warm cache.
"""
from __future__ import annotations

_LANG_LABEL = {"python": "Python", "cpp": "C++", "js": "JavaScript"}
_LANG_FENCE = {"python": "python", "cpp": "cpp", "js": "javascript"}


def _instruction(lang: str, prompt: str) -> str:
    """Wrap a bare function stub in a single-turn completion instruction.

    Deliberately minimal: the point of the June test is to exercise the model's
    natural think->complete behaviour on an ordinary code-completion ask, not to
    prompt-engineer the loop away."""
    label = _LANG_LABEL.get(lang, lang)
    fence = _LANG_FENCE.get(lang, lang)
    return (
        "Complete the following %s function. Provide the full implementation, "
        "returning the completed function in a single code block.\n\n"
        "```%s\n%s\n```" % (label, fence, prompt.rstrip())
    )


def _load_humanevalplus(limit=None):
    from datasets import load_dataset
    try:
        ds = load_dataset("evalplus/humanevalplus", split="test")
        source = "humanevalplus"
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")
        source = "openai_humaneval"
    items = []
    for row in ds:
        items.append({
            "task_id": row.get("task_id") or ("HumanEval/%d" % len(items)),
            "source": source,
            "lang": "python",
            "prompt": row["prompt"],
        })
        if limit and len(items) >= limit:
            break
    return items


def _load_multipl_e(config, lang, limit=None):
    from datasets import load_dataset
    ds = load_dataset("nuprl/MultiPL-E", config, split="test")
    items = []
    for row in ds:
        items.append({
            "task_id": row.get("name") or ("%s/%d" % (config, len(items))),
            "source": config,
            "lang": lang,
            "prompt": row["prompt"],
        })
        if limit and len(items) >= limit:
            break
    return items


# group -> (loader, kwargs, default per-group cap). None cap = take everything.
_GROUPS = {
    "he":  (_load_humanevalplus, {}, None),
    "cpp": (_load_multipl_e, {"config": "humaneval-cpp", "lang": "cpp"}, 50),
    "js":  (_load_multipl_e, {"config": "humaneval-js", "lang": "js"}, 50),
}
_ALIASES = {"all": ["he", "cpp", "js"], "humaneval": ["he"],
            "humanevalplus": ["he"], "multipl-e": ["cpp", "js"], "mpe": ["cpp", "js"]}


def parse_task_spec(spec):
    """'all' | 'he,cpp' | 'he:40,cpp:20' -> ordered [(group, limit_or_None), ...]."""
    out = []
    for tok in (spec or "all").split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, cap = tok.partition(":")
        name = name.strip().lower()
        cap = int(cap) if cap.strip() else None
        for g in _ALIASES.get(name, [name]):
            out.append((g, cap))
    return out


def load_tasks(spec, he_limit=None, mpe_limit=None):
    """Return a flat list of problem dicts for the requested groups.

    spec       : task selector string (see parse_task_spec).
    he_limit   : override the HumanEval+ cap (default = all 164).
    mpe_limit  : override the per-language MultiPL-E cap (default = 50 each).
    """
    items = []
    for group, cap in parse_task_spec(spec):
        if group not in _GROUPS:
            raise ValueError("unknown task group %r (known: %s)"
                             % (group, ", ".join(_GROUPS)))
        loader, kwargs, default_cap = _GROUPS[group]
        limit = cap
        if limit is None:
            limit = he_limit if group == "he" else (
                mpe_limit if mpe_limit is not None else default_cap)
        items.extend(loader(limit=limit, **kwargs))
    for it in items:
        it["message"] = _instruction(it["lang"], it["prompt"])
    return items
