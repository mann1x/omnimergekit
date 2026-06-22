"""llama.cpp server lifecycle for the agentic-loop harness.

This is the ONE backend-specific module. It launches a `llama-server` for a given
GGUF + (optional) chat-template override + Gemma-4 reasoning flags, waits for it
to become healthy, and tears it down cleanly by the PID we own (never a blanket
pkill/fuser on the port). `replay.py` talks to whatever it launches purely over
the OpenAI-compatible HTTP API, so it stays backend-agnostic.

For `backend: endpoint` in the profile, nothing here runs -- the harness points
at an already-running OpenAI-compatible URL (vLLM, a remote llama-server, a
gateway, ...) and only `replay.py` is exercised.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.request


def resolve_llama_server_bin(spec):
    """Resolve the llama-server binary. 'auto' looks at $LLAMA_SERVER_BIN, then a
    `llama-server` on PATH. An explicit path is used verbatim."""
    if spec and spec != "auto":
        if not os.path.isfile(spec):
            raise FileNotFoundError("llama_server_bin not found: %s" % spec)
        return spec
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("llama-server")
    if found:
        return found
    raise FileNotFoundError(
        "could not find llama-server (set server.llama_server_bin in the profile, "
        "or export LLAMA_SERVER_BIN, or put llama-server on PATH; "
        "or run install.sh to build the pinned binary)")


def build_cmd(bin_path, gguf, port, cfg, chat_template=None):
    """Translate the profile's `server` block into a llama-server argv. Only the
    knobs the harness needs are first-class; anything else goes through
    `extra_args`."""
    cmd = [bin_path, "-m", gguf, "--port", str(port), "--alias",
           cfg.get("alias", "agentic-loop"), "--no-warmup"]
    # chat template: a .jinja override (the knob under test) or the GGUF's embedded
    # template via --jinja with no file.
    cmd.append("--jinja")
    if chat_template:
        if not os.path.isfile(chat_template):
            raise FileNotFoundError("chat_template not found: %s" % chat_template)
        cmd += ["--chat-template-file", chat_template]
    cmd += ["-ngl", str(cfg.get("n_gpu_layers", 99))]
    cmd += ["-c", str(cfg.get("ctx_size", 131072))]
    cmd += ["--parallel", str(cfg.get("parallel", 1))]
    if cfg.get("flash_attn", True):
        cmd += ["-fa", "on"]
    if cfg.get("cache_type_k"):
        cmd += ["-ctk", str(cfg["cache_type_k"])]
    if cfg.get("cache_type_v"):
        cmd += ["-ctv", str(cfg["cache_type_v"])]
    # Gemma-4 reasoning flags. Null/empty disables (a non-reasoning template).
    if cfg.get("reasoning_format"):
        cmd += ["--reasoning-format", str(cfg["reasoning_format"])]
    if cfg.get("reasoning_budget") is not None:
        cmd += ["--reasoning-budget", str(cfg["reasoning_budget"])]
    cmd += [str(x) for x in (cfg.get("extra_args") or [])]
    return cmd


class LlamaServer:
    """Owns one llama-server process. Use as a context manager so it is always
    torn down (by PID), even on exception."""

    def __init__(self, bin_path, gguf, port, cfg, chat_template=None,
                 gpu=None, log_path=None, log=print):
        self.cmd = build_cmd(bin_path, gguf, port, cfg, chat_template)
        self.port = port
        self.gpu = gpu
        self.log_path = log_path
        self.log = log
        self.proc = None
        self.base_url = "http://127.0.0.1:%d" % port

    def _healthy(self):
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(self.base_url + path, timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                continue
        return False

    def start(self, boot_timeout=600):
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        logf = open(self.log_path, "wb") if self.log_path else subprocess.DEVNULL
        self._logf = logf
        self.log("launching: %s" % " ".join(self.cmd))
        if self.gpu is not None:
            self.log("  CUDA_VISIBLE_DEVICES=%s" % self.gpu)
        # new session so we can signal the whole group and never touch unrelated pids
        self.proc = subprocess.Popen(self.cmd, stdout=logf, stderr=logf, env=env,
                                     start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < boot_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during boot (rc=%s); see %s"
                                   % (self.proc.returncode, self.log_path))
            if self._healthy():
                self.log("server healthy on %s after %.0fs"
                         % (self.base_url, time.time() - t0))
                return self.base_url
            time.sleep(2)
        self.stop()
        raise TimeoutError("llama-server did not become healthy in %ds" % boot_timeout)

    def stop(self):
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                p.terminate()
            try:
                p.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
            self.log("server (pid=%s) stopped" % p.pid)
        if getattr(self, "_logf", None) not in (None, subprocess.DEVNULL):
            try:
                self._logf.close()
            except Exception:
                pass
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
