#!/usr/bin/env python
"""Offline unit tests for replay_normalize (no network, no transformers).

Covers the three converters against the real source schemas:
  * instruction (Qwen3.5-reasoning): <think>..</think> split
  * messages (TraceInversion / local synthetic): thinking field + plain passthrough
  * hermes (agentic sharegpt): <think> + <tool_call> XML + <tool_response> XML,
    call/response id pairing, system-boilerplate drop, tools passthrough

Run: python scripts/replay_normalize_test.py   (or: pytest scripts/replay_normalize_test.py)
"""
import json
import sys

import replay_normalize as rn


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_split_think():
    r, c = rn.split_think("<think>\nreason here\n</think>\n\nfinal answer")
    _assert(r == "reason here", f"reasoning: {r!r}")
    _assert(c == "final answer", f"content: {c!r}")
    r, c = rn.split_think("no tags here")
    _assert(r is None and c == "no tags here", "no-tag passthrough")


def test_instruction_qwen():
    ex = {"input": "What is 2+2?",
          "output": "<think>\nadd them\n</think>\nThe answer is 4."}
    msgs, tools = rn.normalize(ex, "instruction")
    _assert(tools is None, "instruction has no tools")
    _assert(msgs[0] == {"role": "user", "content": "What is 2+2?"}, msgs[0])
    a = msgs[1]
    _assert(a["role"] == "assistant", "asst role")
    _assert(a["reasoning_content"] == "add them", a.get("reasoning_content"))
    _assert(a["content"] == "The answer is 4.", a["content"])
    _assert("tool_calls" not in a, "no tool_calls")
    # pure-reasoning (no answer body) is unusable -> dropped
    m2, _ = rn.normalize({"input": "q", "output": "<think>only</think>"}, "instruction")
    _assert(m2 is None, "pure-reasoning row dropped")


def test_messages_thinking_field():
    # TraceInversion: clean content + separate `thinking` field
    ex = {"messages": [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A clean answer.", "thinking": "my reasoning"},
    ]}
    msgs, tools = rn.normalize(ex, "messages")
    _assert(tools is None, "no tools")
    a = msgs[1]
    _assert(a["reasoning_content"] == "my reasoning", a.get("reasoning_content"))
    _assert(a["content"] == "A clean answer.", a["content"])


def test_messages_plain_passthrough():
    # local synthetic: plain messages, no thinking, no tags -> unchanged, no reasoning
    ex = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "cfg please"},
        {"role": "assistant", "content": "1. do x\n2. do y"},
    ]}
    msgs, _ = rn.normalize(ex, "messages")
    _assert([m["role"] for m in msgs] == ["system", "user", "assistant"], "roles")
    _assert("reasoning_content" not in msgs[2], "no spurious reasoning")
    _assert(msgs[2]["content"] == "1. do x\n2. do y", "content preserved")


def test_messages_stray_think_in_content():
    # safety net: <think> still inside content is split out
    ex = {"messages": [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<think>hidden</think>visible"},
    ]}
    msgs, _ = rn.normalize(ex, "messages")
    _assert(msgs[1]["reasoning_content"] == "hidden", "stray think extracted")
    _assert(msgs[1]["content"] == "visible", "content cleaned")


def test_hermes_call_response_pairing():
    conv = [
        {"from": "system", "value": "You are a function calling AI model ... <tools>...</tools>"},
        {"from": "human", "value": "test the regex"},
        {"from": "gpt", "value": "<think>let me run code</think>\n"
                                 '<tool_call>\n{"name": "execute_code", "arguments": {"code": "print(1)"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"tool_call_id": "functions.execute_code:0", '
                                  '"name": "execute_code", "content": {"output": "1"}}\n</tool_response>'},
        {"from": "gpt", "value": "<think>done</think>The pattern works."},
    ]
    fn = {"name": "execute_code", "description": "run",
          "parameters": {"type": "object", "properties": {}}}
    tools_schema = [{"type": "function", "function": fn}]
    ex = {"conversations": conv, "tools": json.dumps(tools_schema)}
    msgs, tools = rn.normalize(ex, "hermes")

    # system boilerplate dropped; tools schema parsed through
    _assert(all(m["role"] != "system" for m in msgs), "system dropped")
    _assert(isinstance(tools, list) and tools[0]["function"]["name"] == "execute_code", "tools passthrough")

    roles = [m["role"] for m in msgs]
    # responses embed on the assistant turn (Gemma-native) -> no separate role:tool msg
    _assert(roles == ["user", "assistant", "assistant"], f"roles: {roles}")
    _assert(all(m["role"] != "tool" for m in msgs), "no separate role:tool messages")

    call_asst = msgs[1]
    _assert(call_asst["reasoning_content"] == "let me run code", "reasoning on call turn")
    tc = call_asst["tool_calls"][0]
    _assert(tc["function"]["name"] == "execute_code", "call name")
    _assert(json.loads(tc["function"]["arguments"]) == {"code": "print(1)"}, "call args")
    # no foreign XML left in any content field
    _assert("<tool_call>" not in call_asst["content"], "no tool_call XML in content")

    # tool response embedded as a STRUCTURED object (renders response:name{k:v}, not
    # a <|"|>-wrapped stringified blob that corrupts tool-call generation)
    trs = call_asst["tool_responses"]
    _assert(len(trs) == 1, f"one tool_response embedded: {trs}")
    _assert(trs[0]["name"] == "execute_code", "response name resolved")
    _assert(trs[0]["response"] == {"output": "1"}, f"response kept structured: {trs[0]['response']!r}")
    _assert(not isinstance(trs[0]["response"], str), "response is a dict, not a string blob")
    # call id re-id'd to the response's own tool_call_id so the template resolves the name
    _assert(tc["id"] == "functions.execute_code:0", f"call id repaired: {tc['id']}")

    final = msgs[2]
    _assert(final["content"] == "The pattern works.", final["content"])
    _assert("tool_calls" not in final, "final has no calls")


def test_hermes_multi_call_positional():
    conv = [
        {"from": "human", "value": "search two things"},
        {"from": "gpt", "value": "<think>parallel</think>"
                                 '<tool_call>{"name": "s", "arguments": {"q": "a"}}</tool_call>'
                                 '<tool_call>{"name": "s", "arguments": {"q": "b"}}</tool_call>'},
        {"from": "tool", "value": '<tool_response>{"tool_call_id": "functions.s:1", "name": "s", "content": "ra"}</tool_response>'
                                  '<tool_response>{"tool_call_id": "functions.s:2", "name": "s", "content": "rb"}</tool_response>'},
    ]
    msgs, _ = rn.normalize({"conversations": conv, "tools": None}, "hermes")
    calls = msgs[1]["tool_calls"]
    _assert(len(calls) == 2, "two calls parsed")
    trs = msgs[1]["tool_responses"]
    _assert(len(trs) == 2, f"two responses embedded: {trs}")
    # positional pairing: response order matches call order, ids matched to responses
    _assert(calls[0]["id"] == "functions.s:1", "pair 0")
    _assert(calls[1]["id"] == "functions.s:2", "pair 1")
    # plain-text results stay strings (correctly quoted); only JSON objects go structured
    _assert(trs[0]["response"] == "ra" and trs[1]["response"] == "rb", f"string results: {trs}")


def test_hermes_truncated_json_response_repaired():
    # the real hermes failure: a search_files response whose `content` JSON string
    # was truncated by the source at a fixed char budget (unterminated array/string).
    # It must be repaired into a STRUCTURED dict — never left as a string that the
    # template wraps as a `{value:<|"|>{...blob...<|"|>}`.
    huge = '{"total_count": 1, "files": ["' + "./p/" + "x" * 5000  # no closing "]}
    conv = [
        {"from": "human", "value": "find files"},
        {"from": "gpt", "value": '<tool_call>{"name": "search_files", "arguments": {"q": "*"}}</tool_call>'},
        {"from": "tool", "value": '<tool_response>{"tool_call_id": "functions.search_files:0", '
                                  '"name": "search_files", "content": ' + json.dumps(huge) + '}</tool_response>'},
    ]
    msgs, _ = rn.normalize({"conversations": conv, "tools": None}, "hermes")
    trs = msgs[1]["tool_responses"]
    _assert(len(trs) == 1, f"one response: {trs}")
    resp = trs[0]["response"]
    _assert(isinstance(resp, dict), f"truncated JSON repaired to a dict, not a string: {type(resp).__name__}")
    _assert("total_count" in resp or "files" in resp, f"structure recovered: {resp!r}")
    # every string leaf is capped (no 5000-char blob dominates the sequence)
    def _max_leaf(o):
        if isinstance(o, str):
            return len(o)
        if isinstance(o, list):
            return max((_max_leaf(x) for x in o), default=0)
        if isinstance(o, dict):
            return max((_max_leaf(v) for v in o.values()), default=0)
        return 0
    _assert(_max_leaf(resp) <= rn._MAX_RESP_CHARS + 16, f"leaves capped: {_max_leaf(resp)}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f">>> {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
