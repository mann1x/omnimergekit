#!/usr/bin/env python3
"""omk_eval — omnimergekit unified eval runner.

One tool runs any benchmark template against any model on any backend.
Owns server lifecycle (vllm or llama.cpp), template resolution, eval
dispatch (lm-eval or the validated lcb_helpers shim), token-stats logging,
and resumable SQLite caching.

Backends:
  vllm        — `python -m vllm.entrypoints.openai.api_server ...`
                Prefers unquantized (BF16/FP16) for models that fit VRAM;
                falls back to NVFP4A16 / AWQ / GPTQ when it doesn't.
  llama       — `llama-server` over `--port N`. For Q-quants (Q6_K etc.).

Templates: see `templates/README.md`. Pass `--template <name|path>`.
Token stats: always logged as part of the protocol (prompt + completion
tokens, finish_reason distribution, p10/p50/p90 generation lengths).

Usage:
  omk_eval.py \\
    --model <path-or-hf-id> \\
    --template <name|path> \\
    --backend vllm|llama \\
    [--quant auto|bf16|fp16|nvfp4a16|awq|gptq|q6_k|q4_k_m|...] \\
    [--port 8195] \\
    [--results-dir eval_results] \\
    [--max-vram-gb 24] \\
    [--remote user@host] \\
    [--no-server]                # use an already-running server on --port

Outputs:
  <results-dir>/<bench>/<model-tag>/
    summary.json         # headline score + token stats
    samples.jsonl        # per-question dump (cleaned, passed, reason, tokens)
    sqlite_cache.sqlite  # lm-eval cache (resumable)
    server.log           # backend stdout/stderr

Documented exit codes:
  0   eval completed
  10  template not found / invalid
  20  backend failed to start
  30  eval crashed mid-run; sqlite cache preserved for resume
  40  post-run sanity check failed (empty samples, fence drift, ...)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "eval" / "templates"
LCB_DIR = REPO_ROOT / "eval" / "lcb"


def log(msg: str) -> None:
    print(f"[omk_eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fatal(code: int, msg: str) -> "None":
    print(f"[omk_eval FATAL exit={code}] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ── Quant detection ───────────────────────────────────────────────────────


def detect_native_quant(model_path: str) -> str:
    """Read the model's config.json (if local) to detect its quant format.
    Returns one of: 'bf16', 'fp16', 'nvfp4a16', 'awq', 'gptq', 'gguf', 'unknown'."""
    p = Path(model_path)
    if p.is_file() and p.suffix == ".gguf":
        return "gguf"
    cfg = p / "config.json"
    if not cfg.is_file():
        return "unknown"
    try:
        d = json.loads(cfg.read_text())
    except Exception:
        return "unknown"
    qc = d.get("quantization_config", {})
    qm = qc.get("quant_method", "")
    if qm == "modelopt":
        return "nvfp4a16"
    if qm == "awq":
        return "awq"
    if qm in ("gptq", "gptqmodel"):
        return "gptq"
    # No quant_config — assume native dtype
    td = d.get("torch_dtype", "")
    if "bfloat16" in td:
        return "bf16"
    if "float16" in td:
        return "fp16"
    return "unknown"


# ── Backend launchers ─────────────────────────────────────────────────────


_LIVE_SERVERS: list["ServerHandle"] = []


def _atexit_kill_all() -> None:
    """Make sure no server outlives the interpreter, even on crash."""
    for s in list(_LIVE_SERVERS):
        try:
            s.kill()
        except Exception:
            pass


import atexit as _atexit  # noqa: E402 (late import to keep top tidy)
import signal as _signal  # noqa: E402
_atexit.register(_atexit_kill_all)


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        log(f"caught signal {signum} — killing servers")
        _atexit_kill_all()
        # Re-raise default behavior
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    for sig in (_signal.SIGINT, _signal.SIGTERM, _signal.SIGHUP):
        try:
            _signal.signal(sig, _handler)
        except Exception:
            pass


_install_signal_handlers()


def kill_port(port: int, label: str = "") -> None:
    """Kill any process holding `port` before we try to bind it.

    We've hit this in prod: vllm api_server crashed but the EngineCore
    child kept GPU and port. New launch on the same port fails or, worse,
    succeeds but the engine is the old corrupt one.

    Uses `fuser` first (cleanest), falls back to `lsof`, both via
    SIGTERM then SIGKILL with a small grace.
    """
    import shutil
    try:
        subprocess.run(["fuser", "-k", "-TERM", f"{port}/tcp"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       timeout=5, check=False)
        time.sleep(2)
        subprocess.run(["fuser", "-k", "-KILL", f"{port}/tcp"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       timeout=5, check=False)
    except FileNotFoundError:
        if shutil.which("lsof"):
            try:
                out = subprocess.run(["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
                                     capture_output=True, text=True, timeout=5)
                for pid in out.stdout.split():
                    try:
                        os.kill(int(pid), _signal.SIGTERM)
                    except Exception:
                        pass
            except Exception:
                pass
    # Also nuke orphan vllm/llama-server processes matching the port — this
    # catches EngineCore children whose parent died.
    try:
        out = subprocess.run(["pgrep", "-f", f"--port {port}|--port={port}"],
                             capture_output=True, text=True, timeout=5)
        for pid in out.stdout.split():
            try:
                os.killpg(os.getpgid(int(pid)), _signal.SIGTERM)
            except Exception:
                try:
                    os.kill(int(pid), _signal.SIGTERM)
                except Exception:
                    pass
        time.sleep(2)
        for pid in out.stdout.split():
            try:
                os.kill(int(pid), _signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass
    if label:
        log(f"cleared port {port} ({label})")


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int
    base_url: str
    log_path: Path
    backend: str
    extra: dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        """Kill the server AND its process group so EngineCore / llama-server
        children don't outlive the parent. Two waves: SIGTERM with 10s
        grace, then SIGKILL. Finally a port sweep in case anything orphaned.
        """
        # Drop from registry first so atexit doesn't loop
        try:
            _LIVE_SERVERS.remove(self)
        except ValueError:
            pass
        if self.alive():
            log(f"stopping {self.backend} pid={self.proc.pid} (pgid={os.getpgid(self.proc.pid)})")
            try:
                os.killpg(os.getpgid(self.proc.pid), _signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), _signal.SIGKILL)
                except Exception:
                    self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        # Final port sweep — catches orphaned EngineCore that survived the pgid kill
        kill_port(self.port, label=f"post-kill {self.backend}")


def launch_vllm(model: str, port: int, quant: str, log_path: Path,
                served_name: str, max_model_len: int = 32768,
                gpu_mem_util: float = 0.92,
                enforce_eager: bool = False,
                max_num_batched_tokens: int = 4096,
                reasoning_parser: str | None = None,
                default_chat_template_kwargs: dict | str | None = None,
                extra: list[str] | None = None) -> ServerHandle:
    """Launch vllm OpenAI-compatible api server. Quant 'bf16'/'fp16' →
    unquantized; 'nvfp4a16'/'awq'/'gptq' → loaded as-is from config; 'auto'
    lets vllm infer.

    CUDA-graph capture is ENABLED by default (enforce_eager=False) — the
    LCB-55 v4 NVFP4A16 run on 3090 sustained ~86 tok/s with graphs vs
    ~22 tok/s with `--enforce-eager`, a ~4× throughput win and no
    correctness regression on the 90.91% baseline. Set enforce_eager=True
    only when debugging a vLLM crash or when graph capture fails on a
    weird model/quant combo. See `feedback_backend_decision.md` and the
    LCB-55 results table for context.
    """
    cmd = [
        os.environ.get("VLLM_PYTHON", "/root/anaconda3/envs/vllm/bin/python"),
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served_name,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--trust-remote-code",
    ]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if default_chat_template_kwargs:
        # Accept dict (preferred — let json.dumps quote correctly) or pre-formatted
        # JSON string. Server-side default-on for kwargs like
        # `{"enable_thinking": true}` so per-request bodies don't have to repeat it.
        if isinstance(default_chat_template_kwargs, dict):
            ctk_json = json.dumps(default_chat_template_kwargs)
        else:
            ctk_json = str(default_chat_template_kwargs)
        cmd += ["--default-chat-template-kwargs", ctk_json]
    if quant in ("bf16", "auto"):
        cmd += ["--dtype", "bfloat16"]
    elif quant == "fp16":
        cmd += ["--dtype", "float16"]
    # nvfp4a16/awq/gptq: rely on config.json's quantization_config
    if extra:
        cmd += extra
    env = dict(os.environ)
    env["LD_PRELOAD"] = env.get(
        "LD_PRELOAD",
        "/root/anaconda3/envs/vllm/lib/libstdc++.so.6",
    )
    # Pre-flight: clear any orphan/zombie on the port (EngineCore survivors)
    kill_port(port, label="pre-vllm")
    log(f"vllm cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env, preexec_fn=os.setpgrp)
    h = ServerHandle(proc=proc, port=port,
                     base_url=f"http://localhost:{port}/v1",
                     log_path=log_path, backend="vllm")
    _LIVE_SERVERS.append(h)
    return h


def llama_bench_defaults(task: str) -> list[str]:
    """Per-bench mandatory llama-server flags for Gemma 4 family.

    Coding (HE/MBPP/LCB): suppress reasoning traces so the scorer sees
    answer-only output.

    Reasoning (GPQA/AIME/MMLU-Pro): deepseek-format separation + 8192-
    token reasoning budget. The budget is load-bearing per the project
    rule documented in CLAUDE.md — without it Gemma 4 enters re-read
    loops on hard questions and times out.

    These are applied automatically when the template's `task` matches
    a known family; a template can still override via
    `backend_args.llama_extra`.
    """
    t = (task or "").lower()
    if any(s in t for s in ("humaneval", "mbpp", "livecodebench", "lcb")):
        return ["--jinja", "--reasoning", "off"]
    if any(s in t for s in ("gpqa", "aime", "mmlu_pro", "mmlu-pro")):
        return ["--reasoning-format", "deepseek",
                "--reasoning-budget", "8192"]
    return []


def launch_llama(gguf: str, port: int, log_path: Path,
                 ctx: int = 32768, ngl: int = 99, parallel: int = 2,
                 extra: list[str] | None = None) -> ServerHandle:
    """Launch llama-server. For Q-quants (Q4_K_M, Q6_K, ...).

    `extra` is appended after the mandatory args; pass bench-typed flags
    via `llama_bench_defaults(template['task'])` from the caller, or set
    per-template `backend_args.llama_extra: [--flag, value, ...]`.
    """
    bin_path = os.environ.get("LLAMA_BIN",
                              "/opt/llama.cpp/build/bin") + "/llama-server"
    cmd = [
        bin_path,
        "-m", gguf, "--port", str(port),
        "-c", str(ctx), "-ngl", str(ngl), "--parallel", str(parallel),
        "--no-warmup",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    ]
    if extra:
        cmd += extra
    kill_port(port, label="pre-llama")
    log(f"llama cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=f, stderr=f, preexec_fn=os.setpgrp)
    h = ServerHandle(proc=proc, port=port,
                     base_url=f"http://localhost:{port}/v1",
                     log_path=log_path, backend="llama")
    _LIVE_SERVERS.append(h)
    return h


def wait_ready(server: ServerHandle, served_name: str = "",
               timeout: int = 720, warmup: bool = True) -> None:
    """Block until /v1/models responds AND a warmup chat-completion succeeds.

    vLLM's /v1/models can 200 before the engine is actually warm — the very
    first inference then hangs until the client's ReadTimeout. We send a
    tiny warmup request (4 tokens) before declaring ready. Adds ~3-10s,
    saves a 10-minute timeout on the first real eval task.

    Lost the first task of LCB-55 NVFP4A16 128e on 2026-05-12 to this.
    Documented in EVAL_PROTOCOL.md §v2.5.1.
    """
    import urllib.request
    log(f"waiting up to {timeout}s for {server.backend} on {server.port}")
    t0 = time.time()
    ready = False
    while time.time() - t0 < timeout:
        if not server.alive():
            fatal(20, f"{server.backend} died during startup; tail of log:\n"
                  + Path(server.log_path).read_text().splitlines()[-30:].__str__())
        try:
            req = urllib.request.Request(f"http://localhost:{server.port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    log(f"{server.backend} /v1/models ready after {int(time.time()-t0)}s")
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
    if not ready:
        fatal(20, f"{server.backend} not ready after {timeout}s")
    if not warmup:
        return
    # Warmup: tiny chat-completion. Blocks until response or timeout.
    body = json.dumps({
        "model": served_name or "default",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{server.port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    log("warmup chat-completion ...")
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
            txt = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            log(f"warmup ok after {int(time.time()-t1)}s, response: {txt!r}")
    except Exception as e:
        fatal(20, f"warmup chat-completion failed after {int(time.time()-t1)}s: {e}")


# ── Eval dispatch ─────────────────────────────────────────────────────────


def dispatch_lm_eval(template: dict, model_tag: str, base_url: str,
                     out_dir: Path, tokenizer: str,
                     limit: int | None = None) -> int:
    """Run lm-eval against a chat-completions endpoint. Always uses
    `--use_cache <sqlite>` per project rule, always `--log_samples` for
    post-run sanity checks.

    `limit`: if set, passes `--limit N` to lm-eval (used by smoke runs).

    The function injects `thinking_token_budget=<int>` into gen_kwargs when
    the template's `generation` block carries one. vLLM's payload builder
    in lm-eval (`LocalChatCompletion._create_payload`) forwards any unknown
    gen_kwargs straight into the request body, so the budget reaches the
    server end-to-end. Activation of channel-format reasoning is done at
    the server level (via `backend_args.vllm_chat_template` +
    `backend_args.vllm_reasoning_parser`), so no per-request
    `chat_template_kwargs` plumbing is required."""
    cache_dir = out_dir / "sqlite_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = template["cache"]["sqlite_prefix"]
    g = template["generation"]
    ba = template.get("backend_args", {})
    # num_concurrent is the count of in-flight HTTP requests against vLLM
    # (lm-eval thread pool). It is independent of `batch_size` (which is the
    # per-request lm-eval task batch; we always run that at 1). Default to 2
    # for ~2× wall-clock speedup on reasoning tasks where the model dominates
    # latency — vLLM batches the two requests internally. See
    # memory/feedback_gpqa_parallel_slots.md.
    nconc = int(ba.get("num_concurrent", 2))
    # `max_retries` is lm-eval api_models.AsyncTemplateAPI's per-request retry
    # cap (tenacity stop_after_attempt). Upstream default is 3 — too low for
    # cloud-pod runs where two vLLM servers + two lm-eval clients on the
    # same host hit loopback TCP jitter often enough to exhaust 3 retries on
    # ~5-50 requests per long task. Bumped to 8 (deterministic exponential
    # backoff still bounded; worst-case extra latency per question is small
    # vs the cost of losing a tail-end completion to retry exhaust).
    # `request_timeout` is the per-request HTTP timeout in seconds; default
    # in api_models is 86400, but the actual ceiling we hit is set via the
    # `--http-timeout` CLI flag below (and per-template
    # `generation.http_timeout`). See memory/feedback_lm_eval_retry_tuning.md
    # and memory/feedback_lm_eval_unbound_outputs_bug.md.
    max_retries = int(ba.get("max_retries", 8))
    # `timeout` is lm-eval api_models.AsyncTemplateAPI's per-request aiohttp
    # ClientTimeout.total (seconds). Upstream default is 86400 (24h) so this
    # almost never bites for normal benches, BUT thinking-budgeted Gemma 4
    # at ~22 tok/s on a 3090 with thinking_token_budget=24576 + answer
    # phase ~5k tokens can take ~25 min/request — and we want a hard ceiling
    # so a stuck request doesn't burn an hour silently. Default 1800s
    # (30 min) — overridable per template via backend_args.request_timeout
    # or generation.http_timeout (LCB style).
    request_timeout = int(ba.get("request_timeout",
                                 g.get("http_timeout", 1800)))
    model_args = ",".join([
        f"model={model_tag}",
        f"base_url={base_url}/chat/completions",
        f"num_concurrent={nconc}",
        f"max_retries={max_retries}",
        f"timeout={request_timeout}",
        "tokenizer_backend=huggingface",
        f"tokenizer={tokenizer}",
        f"max_gen_toks={g.get('max_gen_toks', 2048)}",
    ])
    # Build --gen_kwargs from the template's generation block plus any
    # extras (e.g. thinking_token_budget). Scalar values pass straight
    # through simple_parse_args_string; dict values do not, so any
    # client-side templating must go through `vllm_chat_template`.
    gen_kw_parts: list[str] = []
    # `thinking_token_budget` is only valid when vLLM was started with a
    # reasoning parser — without one, vLLM rejects the param with HTTP 400.
    # Skip it (and any other parser-coupled kwargs) when the template runs
    # against a vanilla vLLM config.
    has_parser = bool(ba.get("vllm_reasoning_parser") or "")
    for k in ("temperature", "top_p", "top_k", "max_gen_toks",
              "thinking_token_budget"):
        if k in g:
            if k == "thinking_token_budget" and not has_parser:
                continue
            gen_kw_parts.append(f"{k}={g[k]}")
    cmd = [
        os.environ.get("LM_EVAL_BIN", "lm-eval"),
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", template["task"],
        "--batch_size", str(ba.get("batch_size", 1)),
        "--use_cache", str(cache_dir / f"{cache_prefix}_{model_tag}"),
        "--log_samples",
        "--output_path", str(out_dir / "lm_eval_out"),
    ]
    if gen_kw_parts:
        cmd += ["--gen_kwargs", ",".join(gen_kw_parts)]
    if ba.get("apply_chat_template", False):
        cmd += ["--apply_chat_template"]
    if ba.get("num_fewshot", 0):
        cmd += ["--num_fewshot", str(ba["num_fewshot"])]
    if ba.get("confirm_run_unsafe_code", False):
        cmd += ["--confirm_run_unsafe_code"]
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    # Optional task-override directory: lm-eval discovers any task YAMLs under
    # this path and they shadow built-in tasks of the same name. Used for our
    # chat-aware HumanEval/HumanEval+ overrides under eval/lm_eval_tasks/.
    if ba.get("lm_eval_include_path"):
        inc = ba["lm_eval_include_path"]
        # Resolve relative to the omk_eval.py directory so templates can write
        # short paths like "lm_eval_tasks/humaneval_chat".
        if not os.path.isabs(inc):
            inc = str((Path(__file__).parent / inc).resolve())
        cmd += ["--include_path", inc]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    log(f"lm-eval: {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.call(cmd)


def dispatch_lcb(template: dict, model_tag: str, base_url: str,
                 out_dir: Path) -> int:
    """Run the validated lcb_helpers shim against a chat-completions endpoint."""
    g = template["generation"]
    sel = template["selection"]
    ba = template.get("backend_args", {}) or {}
    cmd = [
        os.environ.get("OMK_PYTHON", "/root/anaconda3/envs/omnimergekit/bin/python"),
        str(LCB_DIR / "lcb_llama_server.py"),
        "--name", model_tag,
        "--base-url", base_url.replace("/v1", ""),
        "--max-tokens", str(g.get("max_gen_toks", 16384)),
        "--http-timeout", str(g.get("http_timeout", 900.0)),
        "--difficulty", sel.get("difficulty", "medium"),
        "--min-date", sel.get("min_date", "2024-10-01"),
        # For smoke runs (n<=10) honor template `n` exactly; for full runs pad
        # by 50 so the shim's post-filter pool has enough candidates to yield n.
        # Bug 2026-05-16: lcb_medium_1_smoke (n=1) was emitting --limit 51
        # → 51 problems evaluated instead of 1. Sweep wall-time blew up ~10x.
        "--limit", str(int(template["n"]) if int(template["n"]) <= 10
                       else int(template["n"]) + 50),
        "--output", str(out_dir / "lcb_result.json"),
    ]
    # MANDATORY for Gemma 4 thinking-on: forward thinking_token_budget +
    # enable_thinking so vLLM clips thinking and force-emits an answer.
    # Without these, ~75% of problems hit max_tokens during thinking and
    # the parser drops both content+reasoning → empty FAIL.
    tb = g.get("thinking_token_budget")
    if tb is not None and int(tb) > 0:
        cmd += ["--thinking-budget", str(int(tb))]
    ctk = ba.get("vllm_default_chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        cmd += ["--enable-thinking", "true" if ctk["enable_thinking"] else "false"]
    log(f"lcb shim: {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.call(cmd)


# ── Post-run sanity + token stats ────────────────────────────────────────


# Thinking-token / reasoning-trace patterns we want to count separately
# from "answer" tokens. The matchers cover the three forms we see in the
# wild: <think>…</think> (DeepSeek-R1, Qwen-R1), <|channel|thought…</think>
# (OpenAI o1-style), and `## Reasoning\n…\n## Final Answer` Markdown.
_REASONING_PATTERNS = [
    (r"<think>(.*?)</think>", "think_tag"),
    (r"<\|channel\|>thought(.*?)</think>", "channel_thought"),
    (r"## Reasoning(.*?)(?:## Final Answer|## Answer|$)", "markdown_reasoning"),
]


def estimate_thinking_chars(completion: str) -> tuple[int, str | None]:
    """Returns (char count inside reasoning blocks, marker kind)."""
    import re as _re
    for pat, kind in _REASONING_PATTERNS:
        m = _re.search(pat, completion, _re.DOTALL | _re.IGNORECASE)
        if m:
            return len(m.group(1)), kind
    return 0, None


def compute_token_stats(samples_path: Path) -> dict:
    """Aggregate prompt/completion/thinking tokens + finish_reasons from a
    samples.jsonl file. Per protocol v2 §2.4, this block is mandatory in
    every run's summary.json — including thinking-token tracking.

    Thinking tokens are estimated from completion text (no separate field
    in the chat-completions response) using ~4 chars/token, which is a
    coarse but stable estimate for Latin scripts. For exact counts, the
    tokenizer would have to be reloaded — not worth the cost when the goal
    is to detect "model spent most of its budget thinking, not answering"."""
    import statistics
    if not samples_path.exists():
        return {"error": f"no samples at {samples_path}"}
    # Use split('\n'), NOT splitlines(): JSON-escaped strings can contain unicode
    # line separators (U+2028/U+2029) which splitlines() splits but json.loads
    # rejects, producing spurious half-lines. Always feed actual `\n` boundaries.
    samples = [json.loads(line) for line in samples_path.read_text().split("\n") if line.strip()]
    if not samples:
        return {"error": "samples file empty"}
    # lm-eval emits one row per (doc, filter). Collapse to unique doc_ids so the
    # n / empty / p10 stats reflect what the user actually requested.
    seen_docs: set = set()
    uniq_samples = []
    for s in samples:
        did = s.get("doc_id")
        if did in seen_docs:
            continue
        seen_docs.add(did)
        uniq_samples.append(s)
    samples = uniq_samples

    def _completion_text(s: dict) -> str:
        # Direct field (set by our own adapters).
        if isinstance(s.get("completion"), str) and s["completion"]:
            return s["completion"]
        # lm-eval API path: resps = [[text]] or filtered_resps = [text]
        for key in ("resps", "filtered_resps"):
            v = s.get(key)
            while isinstance(v, list) and v:
                v = v[0]
            if isinstance(v, str):
                return v
        return ""

    pt = [s.get("prompt_tokens") or 0 for s in samples]
    ct = [s.get("completion_tokens") or 0 for s in samples]
    cl = [len(_completion_text(s)) for s in samples]
    fr: dict[str, int] = {}
    thinking_chars = []
    thinking_kinds: dict[str, int] = {}
    for s in samples:
        k = str(s.get("finish_reason"))
        fr[k] = fr.get(k, 0) + 1
        tk_chars, kind = estimate_thinking_chars(_completion_text(s))
        thinking_chars.append(tk_chars)
        if kind:
            thinking_kinds[kind] = thinking_kinds.get(kind, 0) + 1
    empty = sum(1 for s in samples if not _completion_text(s).strip())
    # ~4 chars/token estimate
    thinking_tokens_est = [c // 4 for c in thinking_chars]
    return {
        "n": len(samples),
        "prompt_tokens": {
            "sum": sum(pt),
            "p10": sorted(pt)[len(pt) // 10] if pt else 0,
            "p50": int(statistics.median(pt)) if pt else 0,
            "p90": sorted(pt)[len(pt) * 9 // 10] if pt else 0,
            "max": max(pt) if pt else 0,
        },
        "completion_tokens": {
            "sum": sum(ct),
            "p10": sorted(ct)[len(ct) // 10] if ct else 0,
            "p50": int(statistics.median(ct)) if ct else 0,
            "p90": sorted(ct)[len(ct) * 9 // 10] if ct else 0,
            "max": max(ct) if ct else 0,
        },
        "thinking_tokens_est": {
            "method": "len(reasoning_block) // 4",
            "sum": sum(thinking_tokens_est),
            "p50": int(statistics.median(thinking_tokens_est)) if thinking_tokens_est else 0,
            "p90": sorted(thinking_tokens_est)[len(thinking_tokens_est) * 9 // 10] if thinking_tokens_est else 0,
            "max": max(thinking_tokens_est) if thinking_tokens_est else 0,
            "kinds": thinking_kinds,
            "ratio_of_completion": (
                round(sum(thinking_tokens_est) / max(sum(ct), 1), 3)
            ),
        },
        "completion_chars": {
            "p10": sorted(cl)[len(cl) // 10] if cl else 0,
            "p50": int(statistics.median(cl)) if cl else 0,
            "p90": sorted(cl)[len(cl) * 9 // 10] if cl else 0,
            "max": max(cl) if cl else 0,
        },
        "finish_reasons": fr,
        "empty_completions": empty,
    }


def sanity_check(stats: dict, expected_n: int, sanity_cfg: dict | None = None) -> list[str]:
    """Post-run gating. Returns a list of warnings; non-empty = fail.

    `sanity_cfg` is the optional `sanity:` block from the template. Supported keys:
      - `min_p10_chars` (int, default 60) — minimum p10 completion length in chars.
        Lower this for short-answer benches (MCQ, IFEval one-liners, ARC letters).

    Defensive against `stats == {"error": "..."}` (compute_token_stats returns
    that when the samples file is missing or empty — typically because lm-eval
    crashed before producing any samples). In that case we surface the error
    as a single warning rather than KeyError'ing on a missing 'n'/'completion_chars'."""
    cfg = sanity_cfg or {}
    min_p10 = int(cfg.get("min_p10_chars", 60))
    warns: list[str] = []
    if "error" in stats:
        warns.append(f"stats error: {stats['error']}")
        return warns
    if stats.get("n") != expected_n:
        warns.append(f"sample count {stats.get('n')} != expected {expected_n}")
    if stats.get("empty_completions", 0) > max(expected_n // 20, 0):
        warns.append(f"too many empty completions: {stats.get('empty_completions')}")
    p10 = (stats.get("completion_chars") or {}).get("p10", 0)
    if p10 < min_p10:
        warns.append(f"p10 completion length {p10} < {min_p10} chars")
    return warns


# ── Score extraction ─────────────────────────────────────────────────────


def extract_canonical_score(template: dict, out_dir: Path) -> tuple[float | None, dict]:
    """Pull the canonical headline score for a finished bench, plus the
    full metric dict for the row.

    Strategy:
      - lm-eval backend: load the latest results_*.json under
        out_dir/**/lm_eval_out/**/results_*.json. The task name comes from
        template['task']; pick the metric specified by
        template['scoring']['metric'] (default "exact_match") and the
        filter under template['scoring']['filter'] (default "flexible-extract"
        or "none"). Fall back to the first numeric metric on that task.
      - lcb_custom backend: load out_dir/lcb_result.json and return
        `pass_at_1` as the score.

    Returns (score, score_dict). score may be None if no result file was
    written (eval crashed early).
    """
    backend = template.get("backend", "lm-eval")
    scoring = template.get("scoring") or {}
    if backend == "lcb_custom":
        rj = out_dir / "lcb_result.json"
        if not rj.exists():
            return None, {}
        try:
            d = json.loads(rj.read_text())
        except Exception as e:  # pragma: no cover
            return None, {"error": f"lcb_result.json parse: {e}"}
        score = d.get("pass_at_1")
        score_dict = {
            "pass_at_1": d.get("pass_at_1"),
            "n_pass": d.get("n_pass"),
            "n": d.get("n"),
        }
        return (float(score) if score is not None else None), score_dict

    # lm-eval path
    results = list(out_dir.glob("**/lm_eval_out/**/results_*.json"))
    if not results:
        return None, {}
    results.sort()
    rj = results[-1]
    try:
        d = json.loads(rj.read_text())
    except Exception as e:  # pragma: no cover
        return None, {"error": f"results.json parse: {e}"}
    task_results = (d.get("results") or {})
    # Pick the task block: prefer template['task'] exactly; else any single task.
    want_task = template.get("task", "")
    if want_task in task_results:
        block = task_results[want_task]
    elif len(task_results) == 1:
        block = next(iter(task_results.values()))
    else:
        # Try fuzzy match (e.g. "gpqa_diamond_cot_zeroshot" when template says "gpqa_diamond")
        block = None
        for k, v in task_results.items():
            if want_task and (k.startswith(want_task) or want_task.startswith(k)):
                block = v
                break
        if block is None:
            return None, {"error": f"no matching task block; have {list(task_results)}"}
    # Build a {metric: value} dict — only numeric, non-stderr, non-alias keys.
    score_dict = {}
    for k, v in block.items():
        if k == "alias" or "_stderr" in k:
            continue
        if isinstance(v, (int, float)):
            score_dict[k] = float(v)
    if not score_dict:
        return None, {"error": "no numeric metrics in result block"}
    # Choose the headline metric using template['scoring']:
    #   prefer "<metric>,<filter>" exact match; else first key starting with <metric>
    want_metric = scoring.get("metric", "exact_match")
    want_filter = scoring.get("filter", "")
    full_key = f"{want_metric},{want_filter}" if want_filter else None
    if full_key and full_key in score_dict:
        return score_dict[full_key], score_dict
    # Try "<metric>," prefix (any filter)
    matches = [k for k in score_dict if k.startswith(f"{want_metric},")]
    if matches:
        # Prefer "flexible-extract" / "math_verify" / "extract_chat" /
        # "prompt_level_strict_acc" canonicals over generic "none" / "strict-match".
        preferred_filters = ["flexible-extract", "math_verify", "extract_chat",
                             "remove_whitespace", "none", "strict-match"]
        for pf in preferred_filters:
            ck = f"{want_metric},{pf}"
            if ck in score_dict:
                return score_dict[ck], score_dict
        return score_dict[matches[0]], score_dict
    # Last resort: first numeric metric
    first_k = next(iter(score_dict))
    return score_dict[first_k], score_dict


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path or HF id")
    ap.add_argument("--template", required=True, help="template name or path")
    ap.add_argument("--backend", choices=("vllm", "llama"), required=True)
    ap.add_argument("--quant", default="auto",
                    help="auto|bf16|fp16|nvfp4a16|awq|gptq|q6_k|q4_k_m|...")
    ap.add_argument("--port", type=int, default=8195)
    ap.add_argument("--results-dir", default="eval_results")
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer for lm-eval (defaults to --model)")
    ap.add_argument("--served-name", default="",
                    help="vllm served-model-name (defaults to derived from path)")
    ap.add_argument("--no-server", action="store_true",
                    help="use an already-running server on --port")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0,
                    help="Pass --limit N to lm-eval (smoke runs). 0 = full set.")
    args = ap.parse_args()

    # Resolve template
    sys.path.insert(0, str(TEMPLATES_DIR))
    from template_loader import load as load_template  # type: ignore
    template = load_template(args.template)
    log(f"loaded template {template['name']} (n={template['n']}, backend={template['backend']})")

    # Resolve served name + tokenizer + out dir
    served_name = args.served_name or Path(args.model).name
    tokenizer = args.tokenizer or args.model
    # Use template['name'] for the dir (not sqlite_prefix) so it matches
    # what users grep for in chain logs (e.g. "gpqa_diamond_full" not
    # "gpqa_diamond"). sqlite_prefix is still the per-template cache key.
    out_dir = Path(args.results_dir) / template["name"] / served_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect native quant if --quant auto
    quant = args.quant
    if quant == "auto":
        quant = detect_native_quant(args.model)
        log(f"detected native quant: {quant}")

    # Launch (or skip if --no-server)
    server = None
    if not args.no_server:
        log_path = out_dir / "server.log"
        if args.backend == "vllm":
            ba = template.get("backend_args", {}) or {}
            server = launch_vllm(
                args.model, args.port, quant, log_path,
                served_name, max_model_len=args.max_model_len,
                # Per-template override (e.g. for a bench that needs eager
                # to dodge a graph-capture crash). Default is False = CUDA
                # graphs ON, which the 90.91% LCB-55 result proved safe.
                enforce_eager=bool(ba.get("vllm_enforce_eager", False)),
                max_num_batched_tokens=int(ba.get("vllm_max_num_batched_tokens", 4096)),
                # Reasoning parser: pass via template (e.g. "gemma4") to enable
                # vLLM's `<|channel>...<channel|>` splitting + thinking budget
                # enforcement. See EVAL_PROTOCOL.md §3.
                reasoning_parser=ba.get("vllm_reasoning_parser"),
                # Default chat-template kwargs (e.g. {"enable_thinking": true})
                # to activate channel-format reasoning without per-request kwargs.
                # vLLM applies these on every chat-completions call unless the
                # request overrides them. Verified end-to-end on 2026-05-12.
                default_chat_template_kwargs=ba.get("vllm_default_chat_template_kwargs"),
                extra=ba.get("vllm_extra") or None,
            )
        else:
            # Compose llama extras: bench-typed defaults + template override.
            llama_extra = llama_bench_defaults(template.get("task", ""))
            ba = template.get("backend_args", {})
            for x in ba.get("llama_extra", []) or []:
                llama_extra.append(str(x))
            # Env override wins (LLAMA_EXTRA="--flag1 value1 --flag2 value2").
            env_extra = os.environ.get("LLAMA_EXTRA", "").strip()
            if env_extra:
                llama_extra = shlex.split(env_extra)
            parallel = int(ba.get("llama_parallel", 2))
            ctx = int(ba.get("llama_ctx", 32768))
            log(f"llama extras: {llama_extra} parallel={parallel} ctx={ctx}")
            server = launch_llama(args.model, args.port, log_path,
                                  ctx=ctx, parallel=parallel,
                                  extra=llama_extra)
        try:
            wait_ready(server, served_name=served_name)
        except SystemExit:
            server.kill()
            raise
    base_url = f"http://localhost:{args.port}/v1"

    # Dispatch eval
    rc = 0
    try:
        if template["backend"] == "lm-eval":
            # Full-mode default: cap lm-eval to template["n"] so e.g. gsm8k_100
            # runs 100 questions, not the full 1314 split. The template's
            # `selection.indices` (stride-5 etc.) is NOT honored by lm-eval
            # itself; --limit takes the first-N. Accept first-N as the
            # operational sample when running unattended chains.
            full_limit = template.get("n") if template.get("n", 0) > 0 else None
            rc = dispatch_lm_eval(template, served_name, base_url, out_dir, tokenizer,
                                  limit=args.limit if args.limit > 0 else full_limit)
        elif template["backend"] == "lcb_custom":
            rc = dispatch_lcb(template, served_name, base_url, out_dir)
        else:
            fatal(10, f"unknown template backend: {template['backend']}")
    finally:
        if server is not None:
            server.kill()

    # Post-run token stats + sanity
    samples_candidates = list(out_dir.glob("**/samples_*.jsonl")) + \
                         list(out_dir.glob("**/lcb_result.samples.jsonl"))
    # Pick the most recently modified samples file. Multiple may co-exist
    # under the same out_dir (re-runs, shadow-task re-tasking, smoke vs full)
    # and the stale ones can have radically different row counts than the
    # current run — that's how today's spurious "sample count 5 != expected
    # 1172" sanity_warning showed up on the fresh ARC run.
    samples_candidates.sort(key=lambda p: p.stat().st_mtime)
    samples = samples_candidates[-1] if samples_candidates else out_dir / "samples.jsonl"
    stats = compute_token_stats(samples)
    # When --limit was used, sanity gates against the requested sample count,
    # not the full template["n"]. Per-template sanity overrides live under the
    # optional `sanity:` block (e.g. relaxed min_p10_chars for MCQ/IFEval).
    effective_n = args.limit if args.limit and args.limit > 0 else template["n"]
    warns = sanity_check(stats, effective_n, template.get("sanity"))

    # Extract canonical score so chain summary tables don't need to parse
    # samples files. score_dict: {metric_name: value} for all reported metrics.
    score, score_dict = extract_canonical_score(template, out_dir)

    # Smoke-mode score floor: a smoke run should be REJECTED, not just WARNed,
    # if the canonical metric is at/below the floor. Default floor=0.0 for
    # generate_until tasks (model must produce SOMETHING that scores >0);
    # MCQ benches can override via `smoke:floor:` in the template.
    is_smoke = (args.limit and args.limit > 0 and args.limit <= 10)
    smoke_cfg = (template.get("smoke") or {}) if isinstance(template.get("smoke"), dict) else {}
    smoke_floor = float(smoke_cfg.get("floor", 0.0))
    if is_smoke and score is not None and score <= smoke_floor:
        warns.append(f"SMOKE FLOOR FAIL: score={score:.4f} <= floor={smoke_floor:.4f} "
                     f"({score_dict})")
        # Smoke-fail rc=50 distinguishes from generic warn (rc=40) — the
        # chain treats rc=50 as a halt-the-suite signal.
        smoke_failed = True
    else:
        smoke_failed = False

    summary = {
        "template": template["name"],
        "model": served_name,
        "quant": quant,
        "backend": args.backend,
        "rc": rc,
        "samples_file": str(samples),
        "score": score,
        "scores": score_dict,
        "token_stats": stats,
        "sanity_warnings": warns,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"summary → {out_dir / 'summary.json'}")
    if score is not None:
        log(f"score: {score:.4f}  (all: {score_dict})")
    log(f"warnings: {warns or 'none'}")
    if smoke_failed:
        sys.exit(50)
    # Exit-code semantics:
    #   0   — score exists and run produced data; warnings (if any) are
    #         informational only (e.g. subset templates trip "sample count
    #         N != expected M" because process_docs filters to a v4-failure
    #         subset; the score IS valid for that subset).
    #   40  — no score AND warnings present (real failure: scorer crashed,
    #         too many empties, p10 too short, etc.).
    #   rc  — lm-eval itself returned nonzero (run aborted).
    # The wrapper bash `... && DONE || FAIL` then correctly labels real
    # failures FAIL and successful subset runs DONE. Documented in
    # memory/feedback_reeval24k_subset_warns.md.
    if rc != 0:
        sys.exit(rc)
    if score is None and warns:
        sys.exit(40)
    sys.exit(0)


if __name__ == "__main__":
    # Defensive: don't kill the orchestrator if a child catches SIGTERM
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    main()
