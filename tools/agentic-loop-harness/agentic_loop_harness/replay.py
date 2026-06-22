"""Static multi-turn agentic-loop replay.

Replays a FROZEN agentic conversation (a fixture captured from a real coding
session) against an OpenAI-compatible `/v1/chat/completions` endpoint across a
sampler-config matrix x N seeds, and reports the per-config LOOP RATE using the
channel-aware detector in `detect.py`.

Why a single resident server suffices: a llama.cpp / vLLM `/v1/chat/completions`
accepts the full sampler param set PER REQUEST (temperature, top_p, top_k, min_p,
repeat_penalty, dry_*, seed), so every config in the matrix is swept against one
loaded model with no restarts. The two axes that ARE server-side -- the chat
template and the reasoning budget -- are swept by relaunching the server
(handled by cli.py), because those are exactly the knobs under test.

Fixture format (fixtures/*.json):
  { "name": "...", "messages": [...], "tools": [...],
    "base_params": {"max_tokens": 32000} }

Backend-agnostic: nothing here is llama.cpp-specific. Point `server` at any
OpenAI-compatible base URL.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.request

from .detect import detect_turn_loop


def _fmt_tc(tc_map):
    out = []
    for i in sorted(tc_map):
        t = tc_map[i]
        out.append("[tool_call name=%s args=%s]"
                   % (t.get("name") or "", "".join(t.get("args", []))))
    return "\n".join(out)


def loop_label(v, runaway, finish):
    """Human-readable tag for WHY a run is a fail, distinguishing a short-cycle
    micro-loop (we can name the repeating unit) from a long verbatim-repetition
    loop from a pure runaway. Lets a report show whether a config/template
    eliminates a loop vs merely shortens it."""
    parts = []
    for chan in ("thinking", "answer"):
        if not v.get("%s_loop" % chan):
            continue
        info = v.get(chan) or {}
        if info.get("repeats"):
            parts.append("%s-shortx%s:%r"
                         % (chan[:5].upper(), info["repeats"],
                            (info.get("unit") or "")[:40]))
        elif info.get("long_loop"):
            parts.append("%s-longloop" % chan[:5].upper())
        else:
            parts.append("%s-loop" % chan[:5].upper())
    if runaway:
        parts.append("RUNAWAY(finish=%s)" % finish)
    return " ".join(parts)


def chat(server, messages, tools, params, timeout, stream=True):
    """POST /v1/chat/completions. stream=True reassembles the SSE stream
    (content + reasoning_content + tool-call arguments), because real agentic
    clients (OpenCode etc.) drive the model in STREAMING mode and the degenerate
    runaway is a streamed-generation phenomenon -- a faithful repro must stream."""
    url = server.rstrip("/") + "/v1/chat/completions"
    body = {"messages": messages, "stream": bool(stream)}
    if tools:
        body["tools"] = tools
    body.update(params)
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            obj = json.loads(r.read())
        ch = (obj.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        tc_map = {}
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tc_map[tc.get("index", i)] = {"name": fn.get("name"),
                                          "args": [fn.get("arguments") or ""]}
        return {"content": msg.get("content") or "",
                "tool_text": _fmt_tc(tc_map),
                "reasoning_content": msg.get("reasoning_content")
                or msg.get("reasoning") or "",
                "finish_reason": ch.get("finish_reason"),
                "completion_tokens": (obj.get("usage") or {}).get("completion_tokens"),
                "n_chunks": 1, "latency_s": round(time.time() - t0, 1)}
    content, reasoning, tc_map, finish, n, ctok = [], [], {}, None, 0, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            n += 1
            if obj.get("usage"):
                ctok = obj["usage"].get("completion_tokens", ctok)
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if d.get("reasoning"):
                    reasoning.append(d["reasoning"])
                for tc in (d.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    slot = tc_map.setdefault(idx, {"name": None, "args": []})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {"content": "".join(content), "tool_text": _fmt_tc(tc_map),
            "reasoning_content": "".join(reasoning), "finish_reason": finish,
            "completion_tokens": ctok, "n_chunks": n,
            "latency_s": round(time.time() - t0, 1)}


def replay_fixture(server, fixture, matrix, seeds, max_tokens=None,
                   timeout=1800.0, stream=True, log=print, concurrency=1):
    """Replay one fixture across every (config x seed). Returns the per-config
    results list (the same shape written to the result JSON).

    server      : OpenAI-compatible base URL (e.g. http://127.0.0.1:8080)
    fixture     : loaded fixture dict (messages/tools/base_params)
    matrix      : list of {"name": str, "params": {...}} sampler configs
    seeds       : explicit list of integer seeds
    concurrency : how many seeds to run at once. Must be <= the server's slot
                  count (llama-server --parallel); each slot needs its own ctx
                  window (prompt + max_tokens), so size ctx = concurrency x window.
    """
    messages, tools = fixture["messages"], fixture.get("tools")
    base = dict(fixture.get("base_params") or {})
    if max_tokens:
        base["max_tokens"] = max_tokens
    base.setdefault("max_tokens", 32000)
    # OpenCode-faithful request shape. tool_choice="auto" is LOAD-BEARING: without
    # it the server constrains the model to an immediate tool call (no thinking
    # phase) and the rumination/runaway never reproduces. With it the model thinks
    # first, and that thinking->tool phase is where the failure lives.
    base.setdefault("tool_choice", "auto")
    base.setdefault("stream_options", {"include_usage": True})

    nseeds = len(seeds)
    results = []
    loglock = threading.Lock()

    def emit(m):
        with loglock:
            log(m)

    emit("fixture=%s  n_messages=%d  n_tools=%d  seeds=%d  configs=%d  concurrency=%d"
         % (fixture.get("name"), len(messages), len(tools or []), nseeds,
            len(matrix), concurrency))
    for cfg in matrix:
        name, sp = cfg["name"], dict(cfg.get("params") or {})

        def run_one(sd, name=name, sp=sp):
            params = dict(base)
            params.update(sp)
            params["seed"] = sd
            try:
                out = chat(server, messages, tools, params, timeout, stream=stream)
            except Exception as e:
                emit("  [%s seed=%d] ERROR %s" % (name, sd, e))
                return {"seed": sd, "error": str(e)}
            ans = out["content"]
            if out.get("tool_text"):
                ans = (ans + "\n" + out["tool_text"]) if ans else out["tool_text"]
            v = detect_turn_loop(ans, out["reasoning_content"])
            runaway = out["finish_reason"] not in ("stop", "tool_calls")
            is_fail = bool(v["is_loop"] or runaway)
            emit("  [%s seed=%d] FAIL=%s (loop=%s runaway=%s) think=%d ans=%d finish=%s %s"
                 % (name, sd, is_fail, v["is_loop"], runaway, v["thinking_len"],
                    len(ans), out["finish_reason"],
                    loop_label(v, runaway, out["finish_reason"])))
            return {"seed": sd, "is_fail": is_fail, "is_loop": v["is_loop"],
                    "runaway": runaway, "thinking_loop": v["thinking_loop"],
                    "answer_loop": v["answer_loop"], "think_len": v["thinking_len"],
                    "ans_len": len(ans), "tool_len": len(out.get("tool_text") or ""),
                    "finish": out["finish_reason"],
                    "completion_tokens": out["completion_tokens"]}

        if concurrency > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                runs = list(ex.map(run_one, seeds))
        else:
            runs = [run_one(sd) for sd in seeds]

        loops = sum(int(r.get("is_loop", False)) for r in runs)
        runaways = sum(int(r.get("runaway", False)) for r in runs)
        fails = sum(int(r.get("is_fail", False)) for r in runs)
        rate = fails / max(1, nseeds)
        results.append({"config": name, "params": sp, "fails": fails,
                        "loops": loops, "runaways": runaways, "seeds": nseeds,
                        "fail_rate": rate, "loop_rate": loops / max(1, nseeds),
                        "runaway_rate": runaways / max(1, nseeds), "runs": runs})
        emit("  ==> %-28s fail_rate=%d/%d = %.0f%%  (loop=%d runaway=%d)"
             % (name, fails, nseeds, 100 * rate, loops, runaways))
    return results
