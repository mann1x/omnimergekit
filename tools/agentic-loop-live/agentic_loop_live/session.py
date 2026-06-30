#!/usr/bin/env python3
"""session.py — drive ONE multi-turn adversarial agent session and compact it.

Flow (all parametrized, no host-coupled paths):
  1. make a fresh session dir under <out>/sessions/<ts>_<label>_<task>/{root,wirelog}
  2. seed a per-session opencode.json that points the agent at the logging proxy
  3. start the proxy (proxy_port -> backend model port); record server props for provenance
  4. run the fixture INIT turn, then escalating follow-ups, each via `opencode run`
     with a per-turn wall budget; chain turns on the agent's session id
  5. tear the proxy down and classify the wire log (compact.compact_session)

The agent sends NO sampler params; the server-side default sampler (configured per
backend) is what the model actually uses. The proxy is the ground-truth capture.
"""
import json
import os
import re
import signal
import subprocess
import time

from . import compact, fixtures, proxy as proxy_mod

_SES_RE = re.compile(r"ses_[A-Za-z0-9]+")


def _seed_opencode_json(root, proxy_port, model_name):
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"local-llama": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Model under test (via logging proxy)",
            "options": {"baseURL": "http://127.0.0.1:%d/v1" % proxy_port, "apiKey": "local"},
            "models": {model_name: {"name": "model under test", "tools": True, "reasoning": True}}}},
        "model": "local-llama/%s" % model_name,
    }
    with open(os.path.join(root, "opencode.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)


def _start_proxy(python_bin, sdir, proxy_port, model_host, model_port):
    wirelog = os.path.join(sdir, "wirelog")
    cmd = [python_bin, "-m", "agentic_loop_live", "proxy",
           "--listen", "127.0.0.1:%d" % proxy_port,
           "--upstream", "%s:%d" % (model_host, model_port),
           "--logdir", wirelog, "--rawdir", os.path.join(wirelog, "raw")]
    log = open(os.path.join(sdir, "proxy.log"), "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    # wait for proxy /health
    import urllib.request
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % proxy_port, timeout=2) as r:
                if r.getcode() == 200:
                    break
        except Exception:
            time.sleep(0.5)
    return proc


def _grep_oc_sid(opencode_log, root):
    if not opencode_log or not os.path.isfile(opencode_log):
        return None
    try:
        sid = None
        for line in open(opencode_log, encoding="utf-8", errors="ignore"):
            if ("directory=%s" % root) in line:
                m = _SES_RE.findall(line)
                if m:
                    sid = m[-1]
        return sid
    except Exception:
        return None


def run_one_session(cfg, backend, task_id=None):
    """Drive one session against an already-running backend+proxy-able model. Returns meta dict."""
    out = cfg["_workdir"]
    label = cfg["label"]
    fx = fixtures.load_fixture(cfg["fixture"])
    task_id = task_id or fx["id"]
    model_name = backend.model_name
    proxy_port = int(cfg["proxy_port"])
    per_turn = int(cfg["per_turn_timeout_s"])
    opencode_bin = cfg["opencode_bin"]

    ts = time.strftime("%Y%m%d-%H%M%S")
    sid = "%s_%s_%s" % (ts, label, task_id)
    sdir = os.path.join(out, "sessions", sid)
    root = os.path.join(sdir, "root")
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(sdir, "wirelog"), exist_ok=True)

    _seed_opencode_json(root, proxy_port, model_name)
    # provenance: the server's sampler + n_ctx at run time
    with open(os.path.join(sdir, "server_props.json"), "w") as fh:
        json.dump(backend.props(), fh)

    proxy_proc = _start_proxy(cfg["python_bin"], sdir, proxy_port, backend.host, backend.port)

    oc_sid = [None]
    any_timeout = [False]
    turn_no = [0]
    oc_log_path = os.path.join(sdir, "opencode.log")

    def run_turn(msg):
        turn_no[0] += 1
        args = [opencode_bin, "run", "--dir", root, "--model", "local-llama/%s" % model_name,
                "--format", "json", "--dangerously-skip-permissions", "--log-level", "INFO"]
        if oc_sid[0]:
            args[2:2] = ["--session", oc_sid[0]]  # insert after "run"
        args.append(msg)
        with open(oc_log_path, "ab") as lf:
            lf.write(("\n--- TURN %d (sid=%s) ---\n" % (turn_no[0], oc_sid[0] or "NEW")).encode())
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                stdout, _ = p.communicate(timeout=per_turn)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
                stdout, _ = p.communicate()
                rc = 137
                any_timeout[0] = True
            lf.write(stdout or b"")
        text = (stdout or b"").decode("utf-8", "replace")
        if oc_sid[0] is None:
            m = _SES_RE.findall(text)
            oc_sid[0] = (m[0] if m else _grep_oc_sid(cfg.get("opencode_log"), root))
        return rc

    print("[session] %s upstream=:%d proxy=:%d per_turn=%ss followups=%d" % (
        sid, backend.port, proxy_port, per_turn, len(fx["followups"])), flush=True)
    start = time.time()
    rc = run_turn(fx["init"])
    print("[session] turn 1 (init) rc=%s sid=%s" % (rc, oc_sid[0]), flush=True)
    maxf = cfg.get("n_followups")
    followups = fx["followups"] if maxf is None else fx["followups"][:int(maxf)]
    for f in followups:
        rc = run_turn(f)
        print("[session] turn %d rc=%s" % (turn_no[0], rc), flush=True)
        if rc == 137:
            print("[session] turn %d TIMED OUT -> likely live loop, stopping early" % turn_no[0], flush=True)
            break
    wall = int(time.time() - start)

    # tear proxy down
    try:
        os.killpg(os.getpgid(proxy_proc.pid), signal.SIGTERM)
    except Exception:
        proxy_proc.terminate()

    crc = 137 if any_timeout[0] else 0
    meta = compact.compact_session(
        sdir, model_label=label, model_port=backend.port, task_id=task_id,
        task_prompt="[multi-turn adversarial] " + fx["init"], rc=crc, wall=wall,
        timeout=per_turn, index_path=os.path.join(out, "sessions", "INDEX.jsonl"))
    print("[session] DONE %s turns=%d wall=%ss verdict=%s" % (
        sid, turn_no[0], wall, meta["verdict"]), flush=True)
    return meta
