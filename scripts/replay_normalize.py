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

import json
import re

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOLRESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


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
    a: dict = {"role": "assistant", "content": content or ""}
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
    return [{"role": "user", "content": instr.strip()},
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
            if content.strip():
                out.append({"role": role, "content": content})
    # need at least one user + one assistant
    if not any(x["role"] == "assistant" for x in out):
        return None, None
    return out, None


_MAX_RESP_CHARS = 2000  # cap on any single string leaf in a tool response


def _truncate_leaves(obj, cap: int = _MAX_RESP_CHARS):
    """Recursively cap every string leaf so a 100k tool dump can't dominate the
    training sequence. Structure (dict/list) is preserved."""
    if isinstance(obj, str):
        return obj if len(obj) <= cap else obj[:cap] + "…(truncated)"
    if isinstance(obj, list):
        return [_truncate_leaves(x, cap) for x in obj]
    if isinstance(obj, dict):
        return {k: _truncate_leaves(v, cap) for k, v in obj.items()}
    return obj


def _repair_truncated_json(s: str, cap: int = _MAX_RESP_CHARS):
    """Best-effort repair of JSON the source truncated mid-value.

    Several hermes ``search_files`` responses carry a JSON *string* whose value
    was cut at a fixed char budget, so ``json.loads`` fails (unterminated
    string/array, or a raw control char at the cut). We cap the prefix, then
    close any still-open string/array/object by walking the bracket+quote stack.
    Returns the parsed dict/list, or ``None`` if it still won't parse.
    """
    s = s[:cap]
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if esc:            # cut mid-escape -> drop the dangling backslash
        s = s[:-1]
    if in_str:
        s += '"'
    for op in reversed(stack):
        s += "}" if op == "{" else "]"
    try:
        return json.loads(s, strict=False)
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_response(body):
    """Normalize a tool-response body to a bounded, template-safe value.

    A JSON *object/array* stays STRUCTURED so the Gemma-4 template renders it
    natively as ``response:name{k:v,...}`` (only leaf strings get the ``<|"|>``
    quote). A string that *looks* like JSON but won't parse (dataset truncated it
    mid-value) is ``strict=False``-parsed and, failing that, repaired into valid
    JSON — so it can NEVER render as a ``{value:<|"|>{...blob...<|"|>}``, the
    pattern that teaches the model to emit ``<|"|>`` inside its own tool-call
    arguments. A genuine plain-text result stays a (capped) string.
    """
    if isinstance(body, str):
        s = body.strip()
        parsed = None
        try:
            parsed = json.loads(s, strict=False)
        except (json.JSONDecodeError, ValueError):
            if s[:1] in "{[":
                # Two upstream shapes fail a full parse: (a) a valid JSON object
                # followed by a trailing NL hint (``{...}\n\n[Hint: ...]``) —
                # ``raw_decode`` grabs the leading object and we keep the tail as a
                # bounded ``_note``; (b) JSON the source truncated mid-value — repair.
                try:
                    obj, end = json.JSONDecoder(strict=False).raw_decode(s)
                    parsed = obj
                    tail = s[end:].strip()
                    if isinstance(obj, dict) and tail:
                        obj.setdefault("_note", tail)
                except (json.JSONDecodeError, ValueError):
                    parsed = _repair_truncated_json(s)
        if isinstance(parsed, (dict, list)):
            body = parsed
        else:  # plain text (or a scalar) — keep as a bounded string
            return s if len(s) <= _MAX_RESP_CHARS else s[:_MAX_RESP_CHARS] + "…(truncated)"
    return _truncate_leaves(body)


def _parse_tools(raw) -> list | None:
    if isinstance(raw, list):
        return raw or None
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            return v or None
        except json.JSONDecodeError:
            return None
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
            if val.strip():
                msgs.append({"role": "user", "content": val.strip()})
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
                calls.append({
                    "id": cid, "type": "function",
                    "function": {"name": obj.get("name"),
                                 "arguments": args if isinstance(args, str)
                                 else json.dumps(args, ensure_ascii=False)},
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
                    resp_id = o.get("tool_call_id")
                    body = o.get("content", o)
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
