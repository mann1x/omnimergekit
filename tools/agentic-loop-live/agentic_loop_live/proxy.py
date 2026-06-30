#!/usr/bin/env python3
"""proxy.py — transparent logging reverse proxy between the agent and the model server.

Sits between the agentic client (opencode) and the model server (ollama / llamafile /
llama.cpp llama-server / any OpenAI-compatible endpoint). Forwards every request
byte-for-byte and streams the response back unchanged, while logging the FULL request
(messages, tools, sampler) and the FULL response (reassembled content + reasoning +
tool_calls, even when streamed via SSE) to a per-run JSONL file.

This is the ground-truth capture: it sees exactly what the agent sends the model and
exactly what the model emits, independent of server internals. When the model loops,
the loop is in this log verbatim.

Stdlib only. Run with any python3, or via:  python -m agentic_loop_live proxy ...

  python -m agentic_loop_live proxy --listen 0.0.0.0:8090 --upstream 127.0.0.1:8101 --logdir ./wire
"""
import argparse
import json
import os
import sys
import time
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "127.0.0.1:8101"
LOGFILE = None
RAWDIR = None
_loglock = threading.Lock()
_req_counter = [0]


def _log(rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["ts"] = round(time.time(), 3)
    line = json.dumps(rec, ensure_ascii=False)
    with _loglock:
        with open(LOGFILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _summarize_messages(body):
    """Compact view of the request for the log: roles, lengths, tool defs, sampler."""
    msgs = body.get("messages", [])
    out = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):  # multimodal / parts
            txt = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        else:
            txt = content or ""
        entry = {"role": role, "chars": len(txt), "text": txt}
        if m.get("tool_calls"):
            entry["tool_calls"] = [
                {
                    "name": tc.get("function", {}).get("name"),
                    "args": tc.get("function", {}).get("arguments"),
                }
                for tc in m["tool_calls"]
            ]
        if m.get("tool_call_id"):
            entry["tool_call_id"] = m.get("tool_call_id")
        out.append(entry)
    tools = [t.get("function", {}).get("name") for t in body.get("tools", [])]
    sampler = {
        k: body.get(k)
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "stream",
        )
        if k in body
    }
    return {
        "model": body.get("model"),
        "n_messages": len(msgs),
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "sampler": sampler,
        "messages": out,
    }


def _parse_sse_stream(raw_lines):
    """Reassemble an OpenAI streaming chat-completion into (content, reasoning, tool_calls,
    finish_reason, usage) from accumulated raw 'data:' lines."""
    content_parts = []
    reasoning_parts = []
    tool_acc = {}  # index -> {"name":..., "arguments":...}
    finish_reason = None
    usage = None
    for ln in raw_lines:
        ln = ln.strip()
        if not ln.startswith("data:"):
            continue
        payload = ln[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            # thinking field name varies by backend: vLLM/lm-eval = reasoning_content,
            # ollama native gemma4 renderer = reasoning, some = thinking. Capture all,
            # else a multi-thousand-token rumination runaway logs as r=0 (mis-classified
            # context-bound instead of THINK_EXPLODE).
            for rk in ("reasoning_content", "reasoning", "thinking"):
                if delta.get(rk):
                    reasoning_parts.append(delta[rk])
                    break
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = tool_acc.setdefault(idx, {"name": None, "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def _parse_json_response(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        return {"raw": raw[:2000]}
    out = {"usage": obj.get("usage"), "finish_reason": None, "content": "",
           "reasoning_content": "", "tool_calls": []}
    for ch in obj.get("choices", []):
        out["finish_reason"] = ch.get("finish_reason")
        msg = ch.get("message") or {}
        out["content"] = msg.get("content") or ""
        out["reasoning_content"] = (msg.get("reasoning_content") or msg.get("reasoning")
                                    or msg.get("thinking") or "")
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function") or {}
            out["tool_calls"].append(
                {"name": fn.get("name"), "arguments": fn.get("arguments")}
            )
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence default stderr spam
        pass

    def _health(self):
        # Local readiness endpoint so the orchestrator can probe the proxy itself,
        # independent of whether the upstream server implements /health.
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _proxy(self, method):
        if self.path.rstrip("/").endswith("/health"):
            return self._health()
        length = int(self.headers.get("Content-Length", 0) or 0)
        body_bytes = self.rfile.read(length) if length else b""

        _req_counter[0] += 1
        rid = _req_counter[0]
        req_body = None
        if body_bytes:
            try:
                req_body = json.loads(body_bytes)
            except Exception:
                req_body = None

        url = "http://%s%s" % (UPSTREAM, self.path)
        # forward headers, but force identity encoding so we log plaintext
        fwd_headers = {}
        for k, v in self.headers.items():
            if k.lower() in ("host", "accept-encoding", "connection", "content-length"):
                continue
            fwd_headers[k] = v
        fwd_headers["Accept-Encoding"] = "identity"

        is_chat = self.path.rstrip("/").endswith("/chat/completions") and req_body is not None

        if is_chat:
            _log({
                "dir": "request", "rid": rid, "path": self.path,
                "req": _summarize_messages(req_body),
            })
            # also dump the RAW request body (full tool schemas + messages) so the
            # exact wire payload is replayable later as a fixture.
            if RAWDIR:
                try:
                    with open(os.path.join(RAWDIR, "req-%05d.json" % rid), "w",
                              encoding="utf-8") as fh:
                        json.dump(req_body, fh, ensure_ascii=False)
                except Exception:
                    pass

        req = urllib.request.Request(url, data=body_bytes, headers=fwd_headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            if is_chat:
                _log({"dir": "response", "rid": rid, "http": e.code,
                      "error": err[:2000].decode("utf-8", "replace")})
            return
        except Exception as e:
            # upstream down / connection refused — record so the classifier can flag
            # SERVER_DOWN rather than mis-label the turn a model TIMEOUT.
            msg = ("Connection refused: %s" % e).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            if is_chat:
                _log({"dir": "response", "rid": rid, "http": 502,
                      "error": msg.decode("utf-8", "replace")})
            return

        ctype = resp.headers.get("Content-Type", "")
        status = resp.getcode()

        if "text/event-stream" in ctype:
            # streaming: relay raw bytes line-by-line, accumulate for the log
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            raw_lines = []
            t0 = time.time()
            try:
                for raw in resp:
                    try:
                        self.wfile.write(raw)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    try:
                        raw_lines.append(raw.decode("utf-8", "replace"))
                    except Exception:
                        pass
            finally:
                if is_chat:
                    parsed = _parse_sse_stream(raw_lines)
                    parsed["dir"] = "response"
                    parsed["rid"] = rid
                    parsed["stream"] = True
                    parsed["gen_secs"] = round(time.time() - t0, 2)
                    _log(parsed)
            return

        # buffered (non-stream)
        raw = resp.read()
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass
        if is_chat:
            parsed = _parse_json_response(raw.decode("utf-8", "replace"))
            parsed["dir"] = "response"
            parsed["rid"] = rid
            parsed["stream"] = False
            _log(parsed)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


def run_proxy(listen, upstream, logdir, rawdir=None):
    """Programmatic entry: start the proxy server (blocking). Returns on KeyboardInterrupt."""
    global UPSTREAM, LOGFILE, RAWDIR
    UPSTREAM = upstream
    os.makedirs(logdir, exist_ok=True)
    RAWDIR = rawdir
    if RAWDIR:
        os.makedirs(RAWDIR, exist_ok=True)
    LOGFILE = os.path.join(logdir, "session-%s.jsonl" % time.strftime("%Y%m%d-%H%M%S"))
    host, port = listen.split(":")
    latest = os.path.join(logdir, "latest.jsonl")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(LOGFILE), latest)
    except Exception:
        pass
    print("wire_proxy: %s -> %s   log=%s" % (listen, upstream, LOGFILE), flush=True)
    srv = ThreadingHTTPServer((host, int(port)), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentic_loop_live proxy")
    ap.add_argument("--listen", default="127.0.0.1:8090")
    ap.add_argument("--upstream", default="127.0.0.1:8101")
    ap.add_argument("--logdir", default="./wire")
    ap.add_argument("--rawdir", default=None,
                    help="dump full raw chat request bodies here (replayable fixtures)")
    args = ap.parse_args(argv)
    run_proxy(args.listen, args.upstream, args.logdir, args.rawdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
