"""agentic-loop-harness CLI / orchestrator.

Reads a run profile (YAML or JSON) that contains EVERYTHING under test -- the
model GGUF, an optional chat-template override (or a list of them), the sampling
matrix, the seeds, and the Gemma-4 reasoning settings -- then:

  1. for each chat template, launches a llama-server for the GGUF with that
     template (backend=llama), or uses a running OpenAI-compatible endpoint
     (backend=endpoint);
  2. replays every fixture across the sampler matrix x seeds;
  3. writes per-(template, fixture) result JSON + a combined summary; and
  4. prints a loop/fail-rate table: rows = template x fixture, cols = sampler
     configs.

This is the file Google edits: point `model.gguf` at their 12B GGUF, set
`model.chat_template` to the jinja(s) they want to compare, and adjust
`sampling` / `server.reasoning_*` to their own settings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .replay import replay_fixture
from .server import LlamaServer, resolve_llama_server_bin

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)


def _load_profile(path):
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            sys.exit("profile is YAML but PyYAML is not installed -- "
                     "`pip install pyyaml`, or use a .json profile")
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve(path, base):
    """Resolve a profile-relative path against the profile's directory, then the
    package root, then cwd. Absolute paths pass through."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    for root in (base, PKG_ROOT, os.getcwd()):
        cand = os.path.join(root, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(base, path)  # report the profile-relative miss


def _norm_templates(spec, base):
    """Normalise model.chat_template into a list of {name, path}. None/absent ->
    a single 'embedded' cell (the GGUF's own template)."""
    if not spec:
        return [{"name": "embedded", "path": None}]
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for item in spec:
        if item is None or item == "embedded":
            out.append({"name": "embedded", "path": None})
        elif isinstance(item, dict):
            p = item.get("path")
            out.append({"name": item.get("name")
                        or (os.path.splitext(os.path.basename(p))[0] if p else "embedded"),
                        "path": _resolve(p, base) if p else None})
        else:
            out.append({"name": os.path.splitext(os.path.basename(item))[0],
                        "path": _resolve(item, base)})
    return out


def _seed_list(spec):
    if isinstance(spec, dict):
        if "start" in spec and "end" in spec:
            return list(range(int(spec["start"]), int(spec["end"]) + 1))
        if "count" in spec:
            base = int(spec.get("base", 1000))
            return [base + i for i in range(int(spec["count"]))]
        raise ValueError("seeds dict needs start+end or count[+base]")
    return [int(x) for x in spec]


def run(profile_path, overrides):
    base = os.path.dirname(os.path.abspath(profile_path))
    prof = _load_profile(profile_path)
    model = prof.get("model") or {}
    srv = dict(prof.get("server") or {})
    samp = prof.get("sampling") or {}
    runc = prof.get("run") or {}

    # apply CLI overrides
    for k in ("gpu", "port", "backend", "endpoint"):
        if overrides.get(k) is not None:
            srv[k] = overrides[k]
    out_dir = overrides.get("out_dir") or _resolve(runc.get("out_dir", "results"), base)
    os.makedirs(out_dir, exist_ok=True)

    backend = srv.get("backend", "llama")
    gguf = _resolve(model.get("gguf"), base) if model.get("gguf") else None
    templates = _norm_templates(overrides.get("templates") or model.get("chat_template"), base)

    matrix_path = _resolve(samp.get("matrix"), base)
    if not matrix_path or not os.path.exists(matrix_path):
        sys.exit("sampling.matrix not found: %s" % samp.get("matrix"))
    matrix = json.load(open(matrix_path))
    seeds = _seed_list(runc.get("seeds", {"count": 8, "base": 1000}))
    max_tokens = runc.get("max_tokens")
    timeout = float(runc.get("timeout_s", 1800))
    fixtures = [_resolve(f, base) for f in (runc.get("fixtures") or [])]
    if not fixtures:
        sys.exit("run.fixtures is empty -- list at least one fixtures/*.json")

    print("=== agentic-loop-harness ===")
    print("backend=%s  templates=%d  fixtures=%d  configs=%d  seeds=%d"
          % (backend, len(templates), len(fixtures), len(matrix), len(seeds)))
    if backend == "endpoint" and len(templates) > 1:
        sys.exit("backend=endpoint cannot switch chat templates (the template is "
                 "fixed server-side). Use backend=llama for a template sweep, or "
                 "run one endpoint per template.")

    bin_path = None
    if backend == "llama":
        if not gguf or not os.path.isfile(gguf):
            sys.exit("model.gguf not found: %s" % gguf)
        bin_path = resolve_llama_server_bin(srv.get("llama_server_bin", "auto"))
        print("llama-server: %s" % bin_path)

    table = []  # (template, fixture, results)
    combined = {"profile": os.path.abspath(profile_path), "backend": backend,
                "gguf": gguf, "matrix": matrix_path, "seeds": seeds, "cells": []}

    for tpl in templates:
        tname = tpl["name"]
        if backend == "llama":
            port = int(srv.get("port", 8080))
            log_path = os.path.join(out_dir, "llama_server_%s.log" % tname)
            server_ctx = LlamaServer(bin_path, gguf, port, srv,
                                     chat_template=tpl["path"], gpu=srv.get("gpu"),
                                     log_path=log_path)
            server_ctx.start()
            base_url = server_ctx.base_url
        else:
            base_url = srv.get("endpoint")
            if not base_url:
                sys.exit("backend=endpoint requires server.endpoint")
            server_ctx = None
        try:
            for fpath in fixtures:
                fx = json.load(open(fpath))
                fname = fx.get("name") or os.path.splitext(os.path.basename(fpath))[0]
                print("\n--- template=%s  fixture=%s ---" % (tname, fname))
                results = replay_fixture(base_url, fx, matrix, seeds,
                                         max_tokens=max_tokens, timeout=timeout,
                                         concurrency=int(srv.get("parallel", 1)))
                outp = os.path.join(out_dir, "result_%s__%s.json" % (tname, fname))
                json.dump({"template": tname, "template_path": tpl["path"],
                           "fixture": fname, "server": base_url,
                           "results": results}, open(outp, "w"), indent=2)
                table.append((tname, fname, results))
                combined["cells"].append({"template": tname, "fixture": fname,
                                          "result_file": outp, "results": results})
        finally:
            if server_ctx is not None:
                server_ctx.stop()

    json.dump(combined, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    _print_table(table, matrix)
    print("\nwrote %s" % os.path.join(out_dir, "summary.json"))


def _print_table(table, matrix):
    cfg_names = [c["name"] for c in matrix]
    w = max([20] + [len("%s/%s" % (t, f)) for t, f, _ in table]) + 1
    print("\n==== FAIL-RATE TABLE (fails = loop OR runaway, per #seeds) ====")
    header = "template/fixture".ljust(w) + "".join(c.rjust(14) for c in cfg_names)
    print(header)
    print("-" * len(header))
    for tname, fname, results in table:
        by = {r["config"]: r for r in results}
        row = ("%s/%s" % (tname, fname)).ljust(w)
        for c in cfg_names:
            r = by.get(c)
            cell = ("%d/%d" % (r["fails"], r["seeds"])) if r else "-"
            row += cell.rjust(14)
        print(row)
    print("(cell = fails/seeds; lower is better; 0/N = no loops or runaways)")


def main(argv=None):
    # stream progress live even when stdout is redirected to a file/pipe
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="agentic-loop-harness",
        description="Replay agentic conversations across chat templates x sampler "
                    "configs x seeds and report per-cell loop/runaway rates.")
    ap.add_argument("--profile", required=True, help="run profile (.yaml or .json)")
    ap.add_argument("--gpu", help="override server.gpu (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--port", type=int, help="override server.port")
    ap.add_argument("--backend", choices=["llama", "endpoint"],
                    help="override server.backend")
    ap.add_argument("--endpoint", help="override server.endpoint (backend=endpoint)")
    ap.add_argument("--out-dir", help="override run.out_dir")
    ap.add_argument("--template", action="append", dest="templates",
                    help="override model.chat_template (repeatable: one .jinja each; "
                         "use the literal 'embedded' for the GGUF's own template)")
    a = ap.parse_args(argv)
    templates = None
    if a.templates:
        templates = [None if t == "embedded" else t for t in a.templates]
    run(a.profile, {"gpu": a.gpu, "port": a.port, "backend": a.backend,
                    "endpoint": a.endpoint, "out_dir": a.out_dir,
                    "templates": templates})


if __name__ == "__main__":
    main()
