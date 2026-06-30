#!/usr/bin/env python3
"""backends.py — start/stop the model server under test, host-agnostically.

Supported backends (set `backend.kind`):
  - llamacpp  : stock upstream llama.cpp `llama-server`   (sampler via CLI flags)
  - llamafile : a Mozilla/opencoti llamafile binary       (sampler via CLI flags)
  - ollama    : an `ollama serve` instance + a Modelfile   (sampler via PARAMETER)
  - external  : a server you already started elsewhere; we only probe + record it

The server MUST expose an OpenAI-compatible `/v1/chat/completions`. The harness sends
NO sampler params in requests (matching real agent clients like opencode), so the
SERVER-SIDE default sampler is load-bearing and is configured here per backend.

Stdlib only.
"""
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request

SAMPLER_KEYS = ("temperature", "top_k", "top_p", "min_p", "repeat_penalty")
# CLI flag name per sampler key for llama.cpp / llamafile
_LLAMA_FLAG = {
    "temperature": "--temp", "top_k": "--top-k", "top_p": "--top-p",
    "min_p": "--min-p", "repeat_penalty": "--repeat-penalty",
}


def _http_get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.getcode(), r.read().decode("utf-8", "replace")
    except Exception:
        return None, None


class Backend:
    """Handle for a running model server. `.props()` returns a llama.cpp-/props-shaped
    dict (default_generation_settings.params + n_ctx) for provenance, synthesized for
    backends that don't expose /props."""

    def __init__(self, kind, host, port, model_name, proc=None, sampler=None, ctx=None,
                 ollama_host_env=None):
        self.kind = kind
        self.host = host
        self.port = int(port)
        self.model_name = model_name
        self.proc = proc
        self.sampler = sampler or {}
        self.ctx = ctx
        self.ollama_host_env = ollama_host_env  # "127.0.0.1:11434" for `ollama stop`

    @property
    def base_url(self):
        return "http://%s:%d" % (self.host, self.port)

    def wait_ready(self, timeout_s=360):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("backend %s died during boot (rc=%s)" % (self.kind, self.proc.returncode))
            code, body = _http_get(self.base_url + "/v1/models", timeout=3)
            if code == 200:
                return True
            time.sleep(2)
        raise TimeoutError("backend %s did not become ready in %ss" % (self.kind, timeout_s))

    def props(self):
        """Return a llama.cpp-/props-shaped dict for server_props.json provenance."""
        if self.kind in ("llamacpp", "llamafile"):
            code, body = _http_get(self.base_url + "/props", timeout=5)
            if code == 200 and body:
                try:
                    return json.loads(body)
                except Exception:
                    pass
        # synth for ollama / external (use the sampler/ctx we configured)
        return {"default_generation_settings": {
            "params": {k: self.sampler.get(k) for k in SAMPLER_KEYS},
            "n_ctx": self.ctx}}

    def stop(self):
        if self.kind == "ollama" and self.model_name:
            env = dict(os.environ)
            if self.ollama_host_env:
                env["OLLAMA_HOST"] = self.ollama_host_env
            try:
                subprocess.run(["ollama", "stop", self.model_name], env=env,
                               timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except Exception:
                pass


def _popen(cmd, gpu=None, log_path=None):
    env = dict(os.environ)
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _sampler_flags(sampler):
    flags = []
    for k in SAMPLER_KEYS:
        v = sampler.get(k)
        if v is None:
            continue
        flags += [_LLAMA_FLAG[k], str(v)]
    return flags


def start_backend(cfg, log_path=None):
    """Start (or attach to) the server described by cfg.backend. Returns a Backend."""
    b = cfg["backend"]
    kind = b["kind"]
    host = b.get("host", "127.0.0.1")
    port = int(b["port"])
    name = b.get("model_name") or "model-under-test"
    ctx = int(b.get("ctx", 32768))
    sampler = {k: b["sampler"][k] for k in SAMPLER_KEYS if b.get("sampler", {}).get(k) is not None}
    gpu = b.get("gpu")
    extra = b.get("extra_args") or []
    if isinstance(extra, str):
        extra = shlex.split(extra)

    if kind == "external":
        be = Backend(kind, host, port, name, proc=None, sampler=sampler, ctx=ctx)
        be.wait_ready(b.get("boot_timeout_s", 60))
        return be

    if kind == "llamacpp":
        cmd = [b["bin"], "-m", b["model"], "-c", str(ctx), "-ngl", str(b.get("ngl", 99)),
               "--host", host, "--port", str(port), "-a", name, "--jinja"]
        if b.get("flash_attn", True):
            cmd += ["-fa", "on"]
        cmd += _sampler_flags(sampler) + extra
        proc = _popen(cmd, gpu, log_path)
        be = Backend(kind, host, port, name, proc=proc, sampler=sampler, ctx=ctx)
        be.wait_ready(b.get("boot_timeout_s", 360))
        return be

    if kind == "llamafile":
        cmd = [b["bin"], "--server", "-m", b["model"], "-c", str(ctx),
               "-ngl", str(b.get("ngl", 99)), "--host", host, "--port", str(port), "-a", name]
        if b.get("flash_attn", True):
            cmd += ["--flash-attn", "on"]
        # safe defaults for very large contexts on opencoti/Mozilla llamafile builds
        cmd += ["--cache-ram", str(b.get("cache_ram", 0)),
                "--ctx-checkpoints", str(b.get("ctx_checkpoints", 0))]
        cmd += _sampler_flags(sampler) + extra
        proc = _popen(cmd, gpu, log_path)
        be = Backend(kind, host, port, name, proc=proc, sampler=sampler, ctx=ctx)
        be.wait_ready(b.get("boot_timeout_s", 360))
        return be

    if kind == "ollama":
        # Build a Modelfile that pins ctx + the server-side default sampler, then
        # `ollama create`. ollama must already be serving on host:port (or set
        # backend.ollama_serve: true to spawn one bound to that port).
        ollama_bin = b.get("bin", "ollama")
        host_env = "%s:%d" % (host, port)
        env = dict(os.environ); env["OLLAMA_HOST"] = host_env
        proc = None
        if b.get("ollama_serve"):
            proc = _popen([ollama_bin, "serve"], gpu, log_path)
            time.sleep(3)
        mf_lines = ["FROM %s" % b["model"], "PARAMETER num_ctx %d" % ctx]
        for k in SAMPLER_KEYS:
            v = sampler.get(k)
            if v is not None:
                pk = "num_ctx" if k == "num_ctx" else k
                mf_lines.append("PARAMETER %s %s" % (pk, v))
        for s in (b.get("stop") or []):
            mf_lines.append('PARAMETER stop "%s"' % s)
        mf_path = os.path.join(cfg["_workdir"], "Modelfile.%s" % name)
        with open(mf_path, "w") as fh:
            fh.write("\n".join(mf_lines) + "\n")
        subprocess.run([ollama_bin, "create", name, "-f", mf_path], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=600)
        be = Backend(kind, host, port, name, proc=proc, sampler=sampler, ctx=ctx,
                     ollama_host_env=host_env)
        be.wait_ready(b.get("boot_timeout_s", 360))
        return be

    raise ValueError("unknown backend.kind: %r (use llamacpp|llamafile|ollama|external)" % kind)
