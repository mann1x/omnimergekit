#!/usr/bin/env python3
"""config.py — load + validate the harness config (YAML or JSON), no host-coupled defaults.

Everything is a parameter. The only values you MUST supply are the agent binary
(`opencode_bin`) and the backend's model (`backend.model` for a GGUF / ollama source).
"""
import json
import os
import sys

DEFAULTS = {
    "opencode_bin": "opencode",          # path to the opencode CLI (the agent under test)
    "opencode_log": None,                # optional: opencode log path for session-id fallback
    "python_bin": sys.executable,        # interpreter used to run the bundled proxy
    "out_dir": "./runs",                 # all session artifacts + sessions/INDEX.jsonl land here
    "proxy_port": 8090,                  # the logging proxy the agent talks to
    "n_sessions": 10,                    # sessions per `run`
    "per_turn_timeout_s": 600,           # wall budget per agent turn
    "fixture": "snake-adversarial",      # fixture id (bundled) or path to a fixture .json
    "n_followups": None,                 # cap follow-ups (default: all in the fixture)
    "label": None,                       # run label (default: backend.kind)
    "backend": {
        "kind": "external",              # llamacpp | llamafile | ollama | external
        "bin": None,                     # server/ollama binary (not needed for external)
        "model": None,                   # GGUF path (llamacpp/llamafile) or ollama FROM source
        "model_name": "model-under-test",  # id the agent sends; served-model-name / ollama tag
        "host": "127.0.0.1",
        "port": 8101,                    # the model server port (proxy forwards here)
        "ctx": 32768,
        "gpu": None,                     # CUDA_VISIBLE_DEVICES value (optional)
        "ngl": 99,
        "flash_attn": True,
        "sampler": {                     # server-side default sampler (agent sends none)
            "temperature": None, "top_k": None, "top_p": None,
            "min_p": None, "repeat_penalty": None,
        },
        "stop": [],                      # ollama PARAMETER stop strings (optional)
        "extra_args": [],                # extra server CLI args
        "boot_timeout_s": 360,
        "ollama_serve": False,           # spawn `ollama serve` bound to host:port
        "cache_ram": 0,                  # llamafile only
        "ctx_checkpoints": 0,            # llamafile only
    },
}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides=None):
    user = {}
    if path:
        with open(path) as fh:
            text = fh.read()
        try:
            import yaml  # optional
            user = yaml.safe_load(text) or {}
        except ImportError:
            user = json.loads(text)
    cfg = _deep_merge(DEFAULTS, user)
    cfg = _deep_merge(cfg, overrides or {})
    cfg["_workdir"] = os.path.abspath(cfg["out_dir"])
    os.makedirs(cfg["_workdir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["_workdir"], "sessions"), exist_ok=True)
    if not cfg.get("label"):
        cfg["label"] = cfg["backend"]["kind"]
    return cfg


def validate(cfg):
    errs = []
    b = cfg["backend"]
    if b["kind"] not in ("llamacpp", "llamafile", "ollama", "external"):
        errs.append("backend.kind must be llamacpp|llamafile|ollama|external")
    if b["kind"] in ("llamacpp", "llamafile", "ollama") and not b.get("model"):
        errs.append("backend.model is required for kind=%s" % b["kind"])
    if b["kind"] in ("llamacpp", "llamafile") and not b.get("bin"):
        errs.append("backend.bin (server binary) is required for kind=%s" % b["kind"])
    return errs
