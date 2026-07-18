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
    _assert(roles == ["user", "assistant", "tool", "assistant"], f"roles: {roles}")

    call_asst = msgs[1]
    _assert(call_asst["reasoning_content"] == "let me run code", "reasoning on call turn")
    tc = call_asst["tool_calls"][0]
    _assert(tc["function"]["name"] == "execute_code", "call name")
    _assert(json.loads(tc["function"]["arguments"]) == {"code": "print(1)"}, "call args")
    # no foreign XML left in any content field
    _assert("<tool_call>" not in call_asst["content"], "no tool_call XML in content")

    tool_msg = msgs[2]
    # call id re-id'd to the response's own tool_call_id so the template resolves the name
    _assert(tool_msg["tool_call_id"] == "functions.execute_code:0", tool_msg["tool_call_id"])
    _assert(tc["id"] == "functions.execute_code:0", f"call id repaired: {tc['id']}")
    _assert("<tool_response>" not in tool_msg["content"], "no tool_response XML")

    final = msgs[3]
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
    tools_msgs = [m for m in msgs if m["role"] == "tool"]
    _assert(len(tools_msgs) == 2, "two responses")
    # positional pairing: response order matches call order, ids matched
    _assert(calls[0]["id"] == tools_msgs[0]["tool_call_id"] == "functions.s:1", "pair 0")
    _assert(calls[1]["id"] == tools_msgs[1]["tool_call_id"] == "functions.s:2", "pair 1")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f">>> {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
