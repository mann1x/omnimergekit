#!/usr/bin/env python
"""Normalize REPLAY datasets to Gemma-4 NATIVE message schema.

Why this exists
---------------
The v1 anti-forgetting mix pulls general reasoning + agentic tool-use replay
from public datasets. Those datasets carry FOREIGN surface formats in the
assistant content:

  * Qwen3.5 / hermes reasoning is wrapped in literal ``<think>...</think>`` tags
  * hermes tool-calls are ``<tool_call>{json}</tool_call>`` XML, tool results are
    ``<tool_response>{json}</tool_response>`` XML

`apply_chat_template` passes assistant *content* through verbatim, so training on
the raw text teaches Gemma-4 to emit those foreign tokens as prose — a competing
surface format against its NATIVE family (`<|channel>...<channel|>` for reasoning,
`<|tool_call>...<tool_call|>` for calls). At serve time we parse the native
channel (`--reasoning-format deepseek`), so a model that learned `<think>` prose
would leak reasoning into the answer and tool intent into plain text — corrupting
the exact competences the fine-tune must preserve.

This module converts each source into the schema the Gemma-4 tool-enabled chat
template consumes (verified against the template's rendering logic):
  * reasoning  -> assistant message ``reasoning_content`` field
                 (template emits ``<|channel>thought\n{text}\n<channel|>``)
  * tool call  -> OpenAI ``tool_calls: [{id, type, function:{name, arguments}}]``
                 (template emits ``<|tool_call>call:{name}{...}<tool_call|>``)
  * tool result-> ``role:"tool"`` message with ``tool_call_id``
                 (template forward-scans + resolves the name; needs ``tools=``)

Every converter returns ``(messages, tools)`` where ``tools`` is the per-row
function schema (only hermes has one; ``None`` otherwise). The trainer renders
with the NATIVE template + ``preserve_thinking=True`` so the training target is
byte-consistent with what the served GGUF emits.

Canonical home: ``omnimergekit/scripts/replay_normalize.py``. Project training
images (e.g. an-finetune) vendor a copy alongside their trainer; keep them in
sync with this file. Offline tests: ``scripts/replay_normalize_test.py``.
"""
from __future__ import annotations

import ast
import json
import re

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

# Foreign chat-control tokens that must NEVER survive into a Gemma-4 native
# training target: ChatML `<|im_start|>{role}\n` / `<|im_end|>` headers (a
# competing chat format), and any STRAY `<think>`/`</think>` left after
# split_think() consumed the paired blocks (unbalanced tags in the source).
# Training on these teaches the model to emit them as prose against its native
# `<|channel>`/`<|tool_call>` family. Paired think + tool_call/tool_response XML
# are handled by the converters above; this only removes the leftovers.
_CHATML_RE = re.compile(r"<\|im_start\|>[^\n]*\n?|<\|im_end\|>")
_STRAY_THINK_RE = re.compile(r"</?think>")
# Same argument for UNPAIRED tool XML. `_TOOLCALL_RE`/`_TOOLRESP_RE` only consume
# BALANCED blocks, so a source turn with a missing closing tag (or a `<tool_call>`
# mentioned inside prose/reasoning) leaves the opener in the residual content and
# it renders verbatim into the target. Measured on the probe: 9/400 rows of
# interstellarninja/hermes_reasoning_tool_use and 2/400 of the SHIPPING
# jofthomas_fc_thinking shard leak exactly this way.
_STRAY_TOOLXML_RE = re.compile(r"</?tool_call>|</?tool_response>")


def _strip_foreign(text: str) -> str:
    """Remove ChatML headers + stray think/tool tokens from a content string."""
    if not text:
        return text
    text = _CHATML_RE.sub("", text)
    text = _STRAY_THINK_RE.sub("", text)
    text = _STRAY_TOOLXML_RE.sub("", text)
    return text


def split_think(text: str) -> tuple[str | None, str]:
    """Pull a leading ``<think>...</think>`` block out of ``text``.

    Returns ``(reasoning_or_None, content_without_think)``. If no think block is
    present the reasoning is ``None`` and the content is returned stripped.
    """
    text = text or ""
    m = _THINK_RE.search(text)
    if not m:
        return None, text.strip()
    reasoning = m.group(1).strip() or None
    content = _THINK_RE.sub("", text).strip()
    return reasoning, content


def _assistant(content: str, reasoning: str | None = None,
               tool_calls: list | None = None) -> dict:
    a: dict = {"role": "assistant", "content": _strip_foreign(content or "")}
    if reasoning:
        reasoning = _strip_foreign(reasoning)
        if reasoning:
            a["reasoning_content"] = reasoning
    if tool_calls:
        a["tool_calls"] = tool_calls
    return a


def convert_instruction(ex: dict) -> tuple[list[dict] | None, None]:
    """Qwen3.5-reasoning et al.: instruction/input + output(=``<think>``+answer)."""
    instr = ex.get("instruction") or ex.get("input") or ex.get("question")
    out = ex.get("output") or ex.get("answer") or ex.get("response")
    if not (instr and out):
        return None, None
    reasoning, content = split_think(out)
    if not content:  # pure-reasoning row with no answer body is unusable
        return None, None
    return [{"role": "user", "content": _strip_foreign(instr.strip())},
            _assistant(content, reasoning)], None


def convert_messages(ex: dict) -> tuple[list[dict] | None, None]:
    """OpenAI ``messages`` sources (TraceInversion, and the local synthetic sets).

    * assistant ``thinking``/``reasoning`` field  -> ``reasoning_content``
    * a stray ``<think>`` still inside content     -> split out (safety net)
    * plain messages (no thinking, no tags)        -> passed through unchanged
    """
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None, None
    out: list[dict] = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            reasoning = (m.get("thinking") or m.get("reasoning")
                         or m.get("reasoning_content"))
            r2, content = split_think(content)
            reasoning = (reasoning or r2)
            reasoning = reasoning.strip() if isinstance(reasoning, str) else None
            out.append(_assistant(content, reasoning or None))
        elif role in ("user", "system"):
            content = _strip_foreign(content)
            if content.strip():
                out.append({"role": role, "content": content})
    # need at least one user + one assistant
    if not any(x["role"] == "assistant" for x in out):
        return None, None
    return out, None


_MAX_RESP_CHARS = 2000  # cap on any single string leaf in a tool response


def _truncate_leaves(obj, cap: int = _MAX_RESP_CHARS, _depth: int = 0):
    """Recursively cap every string leaf so a 100k tool dump can't dominate the
    training sequence. Structure (dict/list) is preserved.

    A string leaf that is ITSELF a serialized dict/list is expanded in place. The
    common upstream shape is a JSON envelope whose payload is a nested *string*:
    ``{"name": "f", "content": "{'trending': [...]}"}``. The outer parse succeeds,
    so ``_coerce_response`` never sees the inner repr, and the template then blobs
    the leaf as ``{content:<|"|>{'trending':...<|"|>}``. Expanding here fixes every
    nesting depth at once instead of special-casing ``content``.
    """
    if isinstance(obj, str):
        if _depth < 4 and obj[:1] in "{[":
            # Same escalation as _coerce_response, deliberately shared: a nested
            # payload is as likely to be a repr, or truncated, as a top-level one.
            inner = _parse_literal(obj)
            if inner is None:  # `is None`, not falsy: {} and [] are valid payloads
                inner = _repair_truncated_literal(obj)
            if inner is not None:
                return _truncate_leaves(inner, cap, _depth + 1)
        return obj if len(obj) <= cap else obj[:cap] + "…(truncated)"
    if isinstance(obj, list):
        return [_truncate_leaves(x, cap, _depth) for x in obj]
    if isinstance(obj, dict):
        return {k: _truncate_leaves(v, cap, _depth) for k, v in obj.items()}
    return obj


_CLOSER = {"{": "}", "[": "]", "(": ")"}


def _scan_literal(s: str):
    """Walk a possibly-truncated JSON **or Python-repr** literal.

    Tracks BOTH quote characters, because a Python repr quotes with ``'`` and may
    contain ``"`` inside a string (and vice versa) — a JSON-only scanner mistakes
    those for openers and mis-balances the stack.

    Returns ``(stack, quote, esc, sep_by_depth)``: the unclosed openers, the open
    quote char (or None), whether the text ends mid-escape, and, per nesting depth,
    the index of the last element separator seen OUTSIDE any string. The last one is
    what lets the caller drop a trailing half-written element and retry.
    """
    stack: list[str] = []
    quote: str | None = None
    esc = False
    sep_by_depth: dict[int, int] = {}
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if quote is not None:
            if ch == "\\":
                esc = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "{[(":
            stack.append(ch)
        elif ch in "}])" and stack:
            stack.pop()
        elif ch == "," and stack:
            sep_by_depth[len(stack)] = i
    return stack, quote, esc, sep_by_depth


def _complete_literal(s: str) -> str:
    """Close a truncated literal's open string and containers, in order."""
    stack, quote, esc, _ = _scan_literal(s)
    if esc:            # cut mid-escape -> drop the dangling backslash
        s = s[:-1]
    if quote is not None:
        s += quote
    for op in reversed(stack):
        s += _CLOSER[op]
    return s


def _jsonish(obj, _depth: int = 0) -> bool:
    """True when ``obj`` is built only from JSON-expressible types.

    This is a REPAIR CORRECTNESS check, not a style check. Closing a truncated repr
    can produce a syntactically valid but semantically wrong object: ``[{'id': 1},
    {'id'`` closes to ``[{'id': 1}, {'id'}]``, where the trailing dict KEY has become
    a ``set``. literal_eval accepts it, the template cannot render it, and json.dumps
    raises. Rejecting non-JSON types here makes the caller trim and retry instead of
    silently inventing a set (or tuple) that was never in the data.
    """
    if _depth > 12:
        return False
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return True
    if isinstance(obj, list):
        return all(_jsonish(x, _depth + 1) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, (str, bool, int, float)) or k is None
                   for k in obj) and all(_jsonish(v, _depth + 1) for v in obj.values())
    return False  # set, tuple, bytes, complex, ...


def _parse_literal(s: str):
    """Parse a complete literal as JSON, else as a Python repr. dict/list only."""
    try:
        v = json.loads(s, strict=False)
    except (json.JSONDecodeError, ValueError):
        v = _literal_eval(s)
    if not isinstance(v, (dict, list)) or not _jsonish(v):
        return None
    return v


def _literal_eval(s: str):
    """``ast.literal_eval`` with every failure mode swallowed. NEVER ``eval`` —
    the input is untrusted corpus text, so only literals may be constructed."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _repair_truncated_literal(s: str, cap: int = _MAX_RESP_CHARS, tries: int = 6):
    """Best-effort repair of a JSON **or Python-repr** literal cut mid-value.

    Sources truncate tool responses at a fixed char budget, so the payload ends
    inside a string, a list, or a key with no value yet. Closing the open string and
    containers fixes the first two. The third needs a retry: we drop back to the last
    element separator at the DEEPEST still-open depth and close again, which salvages
    the largest valid prefix instead of discarding the row.

    Extended from JSON-only on 2026-07-26: the 4/400 residual blob rows on
    interstellarninja/hermes_reasoning_tool_use were truncated Python *reprs*
    (``{'season': 2020, 'results': [{'position': 1, 'constru``), which
    ``_parse_py_repr`` rejects outright and the JSON-only repair could not close
    because it mis-tracked the single quotes. Returns the parsed dict/list, or
    ``None`` if it still won't parse.
    """
    s = s[:cap]
    for _ in range(max(1, tries)):
        v = _parse_literal(_complete_literal(s))
        if v is not None:
            return v
        _, quote, _, sep_by_depth = _scan_literal(s)
        if not sep_by_depth:
            return None
        # Deepest depth first: that is where the truncation happened.
        cut = sep_by_depth[max(sep_by_depth)]
        if quote is not None and cut > s.rfind(quote):
            # The separator we found sits inside the unterminated string; that
            # string is the truncation, so closing it was already the right move
            # and trimming further would not help.
            return None
        s = s[:cut]
    return None


def _parse_py_repr(s: str):
    """Parse a PYTHON-repr dict/list (single-quoted) into a real object.

    Several tool-use corpora emit ``<tool_response>{'result': 42}</tool_response>``
    — a Python repr, not JSON. ``json.loads`` rejects the single quotes, so the
    body used to fall through as a plain string and the template then wrapped it as
    ``{value:<|"|>{'result': 42}<|"|>}``: the blob leak, on 22% of the
    interstellarninja/hermes_reasoning_tool_use pool (85/400 probed rows carried a
    NOT-JSON tool_response). Structuring it here is what lets the template render
    ``response:name{result:42}`` instead.

    Handles a COMPLETE repr only; a truncated one goes to
    ``_repair_truncated_literal``, which closes the open quote/containers first.
    """
    if s[:1] not in "{[":
        return None
    v = _literal_eval(s[:_MAX_RESP_CHARS * 4])
    if not isinstance(v, (dict, list)) or not _jsonish(v):
        return None
    return v


def _coerce_response(body):
    """Normalize a tool-response body to a bounded, template-safe value.

    A JSON *object/array* stays STRUCTURED so the Gemma-4 template renders it
    natively as ``response:name{k:v,...}`` (only leaf strings get the ``<|"|>``
    quote). Anything that *looks* like an object/array but won't parse is escalated
    through three fallbacks — trailing-hint, Python repr, truncation repair — so it
    can NEVER render as a ``{value:<|"|>{...blob...<|"|>}``, the pattern that teaches
    the model to emit ``<|"|>`` inside its own tool-call arguments. A genuine
    plain-text result stays a (capped) string.
    """
    if isinstance(body, str):
        s = body.strip()
        parsed = None
        try:
            parsed = json.loads(s, strict=False)
        except (json.JSONDecodeError, ValueError):
            if s[:1] in "{[":
                # Three upstream shapes fail a full parse, cheapest first:
                #   (a) a valid JSON object followed by a trailing NL hint
                #       (``{...}\n\n[Hint: ...]``) -- ``raw_decode`` grabs the leading
                #       object and we keep the tail as a bounded ``_note``;
                #   (b) a COMPLETE Python repr (``{'result': 42}``);
                #   (c) either syntax, TRUNCATED mid-value -- repair.
                try:
                    obj, end = json.JSONDecoder(strict=False).raw_decode(s)
                    parsed = obj
                    tail = s[end:].strip()
                    if isinstance(obj, dict) and tail:
                        obj.setdefault("_note", tail)
                except (json.JSONDecodeError, ValueError):
                    parsed = _parse_py_repr(s)
                if parsed is None:
                    parsed = _repair_truncated_literal(s)
        if isinstance(parsed, (dict, list)):
            body = parsed
        else:  # plain text (or a scalar) — keep as a bounded string
            return s if len(s) <= _MAX_RESP_CHARS else s[:_MAX_RESP_CHARS] + "…(truncated)"
    return _truncate_leaves(body)


def _coerce_tool_schema(t):
    """Normalise ONE tool entry to the OpenAI ``{type, function:{...}}`` envelope.

    The Gemma-4 native template indexes ``tool["function"]["name"]`` and raises
    ``UndefinedError: 'dict object' has no attribute 'function'`` on the FLAT
    xLAM/BFCL shape ``{name, description, parameters}``. Our two legacy FC pools
    (jofthomas_fc_thinking, nous_glaive_fc) happen to ship the wrapped shape, so
    the flat case never surfaced until a flat-shaped pool was probed.

    Also normalises two things the flat sources do inconsistently:
      * ``parameters.type: "dict"``  -> ``"object"``  (xLAM writes the Python type
        name; leaving it teaches the model a non-JSON-Schema type keyword)
      * a SIBLING ``required: [...]`` next to ``parameters`` -> folded into
        ``parameters.required``, which is where the schema actually declares it.
        ``required: null`` (also present in the wild) is dropped.
    """
    if not isinstance(t, dict):
        return t
    if isinstance(t.get("function"), dict):
        return t
    if not t.get("name"):
        return t
    fn = {k: v for k, v in t.items() if k not in ("required", "type")}
    params = fn.get("parameters")
    if isinstance(params, dict):
        params = dict(params)
        # `type` must be a JSON-Schema type NAME. Some rows put a whole (repr'd)
        # property map there -- `parameters:{type:{'DESCRIPTION': ..., 'TYPE': 'STR'}}`
        # -- which the template then renders as a `type:<|"|>{...}` blob. Any
        # non-string, and xLAM's Python-type spelling, normalise to "object".
        if not isinstance(params.get("type"), str) or params.get("type") == "dict":
            params["type"] = "object"
        req = t.get("required")
        if isinstance(req, list) and req and not params.get("required"):
            params["required"] = req
        fn["parameters"] = params
    return {"type": "function", "function": fn}


def _parse_tools(raw) -> list | None:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):  # single tool, not wrapped in a list
        raw = [raw]
    if isinstance(raw, list):
        return [_coerce_tool_schema(t) for t in raw] or None
    return None


def convert_hermes(ex: dict) -> tuple[list[dict] | None, list | None]:
    """hermes-agent-reasoning-traces (sharegpt + ``tools``) -> native agentic.

    system boilerplate (``<tools>`` XML + instructions) is DROPPED: the native
    template rebuilds the system+tools prefix from the ``tools`` schema. ``gpt``
    turns are split into reasoning + tool_calls + residual content; the following
    ``tool`` turn's ``<tool_response>`` blocks are parsed to STRUCTURED objects and
    embedded as ``tool_responses:[{name,response}]`` on the assistant turn that
    issued the calls (the Gemma-native path), positionally paired to the calls so
    the template resolves each function name and renders the response as
    ``response:name{k:v,...}`` — NOT a ``<|"|>``-wrapped stringified-JSON blob.
    """
    conv = ex.get("conversations") or ex.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None, None
    tools = _parse_tools(ex.get("tools"))
    msgs: list[dict] = []
    call_ctr = 0
    last_calls: list[dict] = []  # tool_call dicts of the most recent assistant turn
    last_assistant: dict | None = None  # the assistant msg that issued last_calls

    for t in conv:
        frm = t.get("from") or t.get("role")
        val = t.get("value") if "value" in t else t.get("content")
        val = val or ""
        if frm in ("system",):
            continue  # native template rebuilds sys+tools from `tools`
        if frm in ("human", "user"):
            uc = _strip_foreign(val.strip())
            if uc.strip():
                msgs.append({"role": "user", "content": uc})
            last_calls = []
            last_assistant = None
        elif frm in ("gpt", "assistant"):
            reasoning, rest = split_think(val)
            calls: list[dict] = []

            def _grab(m):
                nonlocal call_ctr
                try:
                    obj = json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    return ""
                cid = f"call_{call_ctr}"
                call_ctr += 1
                args = obj.get("arguments", {})
                # Keep arguments as a MAPPING so the Gemma-4 template renders the
                # canonical tool-call DSL `{key:val}` (BARE keys, chat_template
                # `arguments is mapping` branch). Passing a JSON *string* forces the
                # template's pre-serialized-string branch, which strips the braces and
                # emits quoted-key JSON `{"key": val}` — the form the serve-time parser
                # mis-binds into keys-with-literal-quotes (`args["\"task_id\""]` ->
                # KeyError at dispatch; v9 a2a proto=0 RCA 2026-07-22). A string that is
                # itself a JSON object is parsed back to a dict; a non-JSON string is
                # left as-is (template string-branch, non-fatal).
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
                    if isinstance(parsed, dict):
                        args = parsed
                calls.append({
                    "id": cid, "type": "function",
                    "function": {"name": obj.get("name"), "arguments": args},
                })
                return ""

            content = _TOOLCALL_RE.sub(_grab, rest).strip()
            am = _assistant(content, reasoning, calls or None)
            msgs.append(am)
            last_calls = calls
            last_assistant = am
        elif frm in ("tool", "observation", "tool_response", "function"):
            blocks = _TOOLRESP_RE.findall(val) or [val.strip()]
            responses: list[dict] = []
            for i, rb in enumerate(blocks):
                body = rb
                resp_id = None
                try:
                    o = json.loads(rb, strict=False)
                    # A tool_response may be a bare ARRAY, not an `{tool_call_id,
                    # name, content}` envelope. Without this isinstance guard the
                    # converter dies with `AttributeError: 'list' object has no
                    # attribute 'get'`. Latent since the envelope was introduced;
                    # first hit on the 51k-row hermes_reasoning_tool_use audit
                    # (2026-07-26) because a 400-row probe contains no such row.
                    if isinstance(o, dict):
                        resp_id = o.get("tool_call_id")
                        body = o.get("content", o)
                    else:
                        body = o
                except json.JSONDecodeError:
                    pass
                # Bound + structure the response: objects/arrays stay native
                # (`response:name{k:v}`), giant leaves are capped, and a truncated
                # JSON *string* is repaired into a dict instead of being wrapped as a
                # `{value:<|"|>{...blob...<|"|>}` (which teaches the model to emit
                # `<|"|>` inside its own tool-call args — the v1-e1 breakage).
                body = _coerce_response(body)
                if i < len(last_calls):
                    cid = resp_id or last_calls[i]["id"]
                    last_calls[i]["id"] = cid  # keep call id == response id
                    tname = last_calls[i]["function"]["name"] or "unknown"
                else:
                    tname = "unknown"
                responses.append({"name": tname, "response": body})
            # Gemma-native path: embed the structured responses on the assistant turn
            # that issued the calls (the template's `tool_responses` branch), instead
            # of separate ``role:"tool"`` messages (which the template forward-scans as
            # opaque `<|"|>` string blobs).
            if last_assistant is not None and responses:
                last_assistant.setdefault("tool_responses", []).extend(responses)
            last_calls = []
            last_assistant = None

    if not any(m["role"] == "assistant" for m in msgs):
        return None, None
    return msgs, tools


# --- dispatch ---------------------------------------------------------------
# `format` string in the mix YAML -> converter. All converters return
# (messages, tools); tools is None except for agentic sources.
CONVERTERS = {
    "messages": convert_messages,
    "instruction": convert_instruction,
    "hermes": convert_hermes,
}


def normalize(ex: dict, fmt: str) -> tuple[list[dict] | None, list | None]:
    conv = CONVERTERS.get(fmt)
    if conv is None:
        raise ValueError(f"unknown replay format: {fmt}")
    return conv(ex)


# --- channel-presence gate --------------------------------------------------
# This module maps a source's reasoning INTO `reasoning_content` so the template
# emits `<|channel>thought ... <channel|>`. What it never did was require that
# reasoning EXISTS: a source with no reasoning field passes through clean, renders
# with NO thought channel, and silently trains "answer with no thinking".
#
# 2026-07-26: that is exactly how `nous_glaive_fc` entered the v26 mix --
# `<|turn>model` in 1640/1640 rows, `<|channel>thought` in 0/1640, i.e. ~12.5% of
# the mix teaching the OPPOSITE of what the other 87.5% teaches. Contradictory
# supervision about whether to open a thinking block at all, invisible to every
# format and repetition check (the surface tokens are all native and correct).
#
# The gate is deliberately opt-OUT, not opt-in: a source that genuinely carries no
# reasoning must SAY SO in the mix YAML (`expect_reasoning: false`). Silence now
# fails loudly instead of shipping.
CHANNEL_OPEN = "<|channel>"
CHANNEL_MIN_FRAC = 0.98


def channel_frac(rendered: "list[str] | tuple[str, ...]") -> tuple[int, int]:
    """(rows_with_thought_channel, rows_rendered) over already-rendered targets."""
    n = 0
    hit = 0
    for txt in rendered:
        if not txt:
            continue
        n += 1
        hit += CHANNEL_OPEN in txt
    return hit, n


def check_channel_presence(n_with_channel: int, n_rendered: int, source: str = "?",
                           expect_reasoning: bool = True,
                           min_frac: float = CHANNEL_MIN_FRAC) -> str | None:
    """Return an error message if the thought channel is missing, else None.

    `expect_reasoning=False` inverts the check: the source is DECLARED
    reasoning-free, so finding a thought channel is the anomaly worth reporting.
    Either way the outcome is explicit rather than an unread counter.
    """
    if not n_rendered:
        return None
    frac = n_with_channel / n_rendered
    if expect_reasoning:
        if frac < min_frac:
            return (f"MISSING THOUGHT CHANNEL: {source} renders {CHANNEL_OPEN} in "
                    f"{n_with_channel}/{n_rendered} rows ({frac:.1%} < "
                    f"{min_frac:.0%}). This trains 'answer with no thinking' and "
                    f"CONTRADICTS the thinking-channel sources in the same mix. "
                    f"Fix the source to carry reasoning, or declare it in the mix "
                    f"YAML with `expect_reasoning: false`.")
        return None
    if n_with_channel:
        return (f"UNEXPECTED THOUGHT CHANNEL: {source} is declared "
                f"`expect_reasoning: false` but {n_with_channel}/{n_rendered} rows "
                f"render {CHANNEL_OPEN}. Drop the declaration or fix the source.")
    return None


def assert_channel_presence(n_with_channel: int, n_rendered: int, source: str = "?",
                            expect_reasoning: bool = True,
                            min_frac: float = CHANNEL_MIN_FRAC) -> None:
    """Raising form of :func:`check_channel_presence`, for build scripts."""
    msg = check_channel_presence(n_with_channel, n_rendered, source,
                                 expect_reasoning, min_frac)
    if msg:
        raise ValueError(msg)
