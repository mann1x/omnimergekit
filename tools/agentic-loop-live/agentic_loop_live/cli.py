#!/usr/bin/env python3
"""cli.py — `python -m agentic_loop_live <subcommand>`.

Subcommands:
  run        boot the backend, drive N adversarial sessions, tear down, print the table
  session    drive ONE session against a config-described backend (run with --n 1)
  proxy      run the logging reverse proxy standalone
  compact    (re)classify one session dir
  audit      re-derive faithfulness of every loop verdict from the untruncated wire log
  tabulate   per-label loop-rate table from <out>/sessions/INDEX.jsonl
  fixtures   list bundled task fixtures
"""
import argparse
import os
import sys

from . import __version__, config as config_mod, fixtures as fixtures_mod


def _overrides_from_args(a):
    """Map common CLI flags onto a config-override dict (only set keys the user passed)."""
    o, b, s = {}, {}, {}
    if a.out is not None: o["out_dir"] = a.out
    if a.n is not None: o["n_sessions"] = a.n
    if a.per_turn_timeout is not None: o["per_turn_timeout_s"] = a.per_turn_timeout
    if a.fixture is not None: o["fixture"] = a.fixture
    if a.n_followups is not None: o["n_followups"] = a.n_followups
    if a.label is not None: o["label"] = a.label
    if a.proxy_port is not None: o["proxy_port"] = a.proxy_port
    if a.opencode_bin is not None: o["opencode_bin"] = a.opencode_bin
    if a.backend_kind is not None: b["kind"] = a.backend_kind
    if a.bin is not None: b["bin"] = a.bin
    if a.model is not None: b["model"] = a.model
    if a.model_name is not None: b["model_name"] = a.model_name
    if a.host is not None: b["host"] = a.host
    if a.port is not None: b["port"] = a.port
    if a.ctx is not None: b["ctx"] = a.ctx
    if a.gpu is not None: b["gpu"] = a.gpu
    for k, v in (("temperature", a.temp), ("top_k", a.top_k), ("top_p", a.top_p),
                 ("min_p", a.min_p), ("repeat_penalty", a.repeat_penalty)):
        if v is not None:
            s[k] = v
    if s:
        b["sampler"] = s
    if b:
        o["backend"] = b
    return o


def _add_common(p):
    p.add_argument("--config", default=None, help="YAML/JSON config file")
    p.add_argument("--out", default=None, help="output dir (sessions/ + INDEX.jsonl)")
    p.add_argument("--n", type=int, default=None, help="number of sessions")
    p.add_argument("--per-turn-timeout", type=int, default=None)
    p.add_argument("--fixture", default=None, help="fixture id or path to fixture .json")
    p.add_argument("--n-followups", type=int, default=None)
    p.add_argument("--label", default=None, help="run label (default: backend.kind)")
    p.add_argument("--proxy-port", type=int, default=None)
    p.add_argument("--opencode-bin", default=None)
    p.add_argument("--backend-kind", default=None, choices=["llamacpp", "llamafile", "ollama", "external"])
    p.add_argument("--bin", default=None, help="server/ollama binary (not needed for external)")
    p.add_argument("--model", default=None, help="GGUF path (llamacpp/llamafile) or ollama FROM source")
    p.add_argument("--model-name", default=None, help="id the agent sends / served-model-name / ollama tag")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None, help="model server port")
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repeat-penalty", type=float, default=None)


def _cmd_run(a, one=False):
    from . import backends, session, analyze
    cfg = config_mod.load_config(a.config, _overrides_from_args(a))
    if one:
        cfg["n_sessions"] = 1
    errs = config_mod.validate(cfg)
    if errs:
        for e in errs:
            print("config error:", e, file=sys.stderr)
        return 2
    print("[run] label=%s backend=%s model=%s n=%d out=%s" % (
        cfg["label"], cfg["backend"]["kind"], cfg["backend"].get("model"),
        cfg["n_sessions"], cfg["_workdir"]), flush=True)
    be = backends.start_backend(cfg, log_path=os.path.join(cfg["_workdir"], "backend_%s.log" % cfg["label"]))
    print("[run] backend up: %s  served_sampler=%s" % (
        be.base_url, {k: be.sampler.get(k) for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty")}),
        flush=True)
    try:
        for i in range(int(cfg["n_sessions"])):
            print("[run] === session %d/%d ===" % (i + 1, cfg["n_sessions"]), flush=True)
            try:
                session.run_one_session(cfg, be)
            except Exception as e:
                print("[run] session %d failed: %s" % (i + 1, e), file=sys.stderr, flush=True)
    finally:
        be.stop()
    print("\n[run] done — loop table:\n")
    analyze.tabulate(cfg["_workdir"])
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="agentic_loop_live",
                                 description="Live multi-turn agentic-loop measurement harness")
    ap.add_argument("--version", action="version", version="agentic-loop-live %s" % __version__)
    sub = ap.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="boot backend, drive N sessions, tear down, tabulate")
    _add_common(pr)
    ps = sub.add_parser("session", help="drive ONE session (run with n=1)")
    _add_common(ps)

    pp = sub.add_parser("proxy", help="run the logging reverse proxy standalone")
    pp.add_argument("rest", nargs=argparse.REMAINDER)

    pc = sub.add_parser("compact", help="(re)classify one session dir")
    pc.add_argument("rest", nargs=argparse.REMAINDER)

    pa = sub.add_parser("audit", help="faithfulness audit of loop verdicts")
    pa.add_argument("--out", default="./runs")
    pa.add_argument("--label", default=None)
    pa.add_argument("--session-id", default=None)

    pt = sub.add_parser("tabulate", help="per-label loop-rate table")
    pt.add_argument("--out", default="./runs")

    sub.add_parser("fixtures", help="list bundled fixtures")

    a = ap.parse_args(argv)
    if a.cmd in (None,):
        ap.print_help()
        return 0
    if a.cmd == "run":
        return _cmd_run(a)
    if a.cmd == "session":
        return _cmd_run(a, one=True)
    if a.cmd == "proxy":
        from . import proxy
        return proxy.main(a.rest)
    if a.cmd == "compact":
        from . import compact
        return compact.main(a.rest)
    if a.cmd == "audit":
        from . import analyze
        analyze.audit(os.path.abspath(a.out), a.label, a.session_id)
        return 0
    if a.cmd == "tabulate":
        from . import analyze
        analyze.tabulate(os.path.abspath(a.out))
        return 0
    if a.cmd == "fixtures":
        d = fixtures_mod.fixtures_dir()
        print("bundled fixtures in %s:" % d)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                print("  -", fn[:-5])
        return 0
    ap.print_help()
    return 0
