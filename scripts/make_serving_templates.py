#!/usr/bin/env python3
"""make_serving_templates.py — clone the canonical greedy 9-bench templates into
"serving-sampler" variants that differ ONLY in the sampler, so we can measure
capability at the sampler we would actually ship with (vendor_minp_rep + a
chosen temperature) instead of greedy.

Why a SEPARATE file per bench (never an in-place edit):
  The canonical greedy templates are FROZEN per CLAUDE.md (cross-cohort
  comparisons live on greedy forever). This emits `<bench>_<suffix>.yaml` with a
  distinct `name:` so omk_eval writes results into their own dir and the greedy
  summary.json is never clobbered. This is the sanctioned escape hatch.

How the sampler reaches llama-server:
  * temperature / top_p / top_k -> placed in `generation:`. omk_eval's gen_kwargs
    builder forwards exactly these per-request (plus max_gen_toks /
    thinking_token_budget), so the chat-completions body carries them.
  * min_p / repeat_penalty -> NOT forwarded per-request by omk_eval (the MPE
    generator comment is explicit: "min_p/repeat_penalty stay server-launch
    flags"). They are injected as llama-server LAUNCH flags via
    `backend_args.llama_extra` (appended, preserving the per-task reasoning
    defaults omk_eval auto-applies). They then act as server-side sampling
    defaults that the per-request body (which omits them) does not override —
    i.e. exactly the vendor_minp_rep config the 48-seed agentic gate validated.

Everything else (task, n, selection, max_gen_toks, thinking_token_budget,
scoring, cache, reports, backend_overrides, per-task reasoning auto-selection)
is preserved verbatim, so the serving-vs-greedy delta is the sampler and nothing
else. Parameterize --temperature (and the rest) and instantiate once the ship
sampler is locked; hold the eval launch until then.
"""
import argparse
import sys
from pathlib import Path

# Prefer ruamel (round-trip preserves comments + key order + backend_overrides
# nesting); fall back to PyYAML for derived files where comment loss is fine.
try:
    from ruamel.yaml import YAML
    _RUAMEL = True
except ImportError:
    import yaml
    _RUAMEL = False

CANONICAL_9 = [
    "gpqa_diamond_full", "lcb_medium_55_v4", "ifeval_100", "gsm8k_100",
    "arc_challenge_full", "aime_30", "math500_100", "humaneval_full",
    "humanevalplus_full",
]


def _load(path):
    if _RUAMEL:
        y = YAML()
        y.preserve_quotes = True
        y.width = 4096  # don't wrap long lines (chat templates, urls)
        return y, y.load(path.read_text())
    return None, yaml.safe_load(path.read_text())


def _dump(y, data, path):
    if _RUAMEL:
        import io
        buf = io.StringIO()
        y.dump(data, buf)
        path.write_text(buf.getvalue())
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False, width=4096))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates-dir", required=True,
                    help="dir holding the canonical <bench>.yaml templates "
                         "(e.g. /srv/ml/repos/omnimergekit/eval/templates)")
    ap.add_argument("--suffix", default="serving",
                    help="emitted name/file suffix -> <bench>_<suffix>.yaml "
                         "(default: serving)")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--benches", nargs="*", default=CANONICAL_9)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written; touch no files")
    args = ap.parse_args()

    tdir = Path(args.templates_dir)
    if not tdir.is_dir():
        sys.exit(f"FATAL: templates dir not found: {tdir}")

    print(f"# yaml backend: {'ruamel(round-trip)' if _RUAMEL else 'pyyaml(no-comments)'}")
    made, skipped = [], []
    for b in args.benches:
        src = tdir / f"{b}.yaml"
        if not src.exists():
            print(f"  SKIP {b}: source template missing", file=sys.stderr)
            skipped.append(b)
            continue
        y, data = _load(src)

        # 1) sampler in generation (forwarded per-request by gen_kwargs)
        gen = data.get("generation") or {}
        gen["temperature"] = args.temperature
        gen["top_p"] = args.top_p
        gen["top_k"] = args.top_k
        gen["do_sample"] = True
        data["generation"] = gen

        # 2) min_p + repeat_penalty as llama-server LAUNCH flags (append, preserve)
        ba = data.get("backend_args") or {}
        extra = list(ba.get("llama_extra") or [])
        extra += ["--min-p", str(args.min_p), "--repeat-penalty", str(args.repeat_penalty)]
        ba["llama_extra"] = extra
        data["backend_args"] = ba

        # 3) distinct name -> separate results dir; greedy summary.json untouched
        data["name"] = f"{b}_{args.suffix}"

        # 4) distinct sqlite cache prefix -> CRITICAL: never read the greedy-
        # sampler cached responses. If the serving run reused the greedy prefix
        # (+ same served-model-name) it could hit temp=0.0 cached generations and
        # silently report greedy numbers. A distinct prefix forces fresh
        # generation at the serving sampler.
        cache = data.get("cache") or {}
        if cache.get("sqlite_prefix"):
            cache["sqlite_prefix"] = f"{cache['sqlite_prefix']}_{args.suffix}"
            data["cache"] = cache

        out = tdir / f"{b}_{args.suffix}.yaml"
        if args.dry_run:
            print(f"  [dry] {out.name:<32} temp={args.temperature} top_p={args.top_p} "
                  f"top_k={args.top_k} min_p={args.min_p} rep={args.repeat_penalty} "
                  f"name={data['name']}  llama_extra+={extra[-4:]}")
        else:
            _dump(y, data, out)
            print(f"  wrote {out.name}")
        made.append(out.name)

    tag = "(dry) " if args.dry_run else ""
    print(f"{tag}{len(made)} serving templates (suffix '{args.suffix}'), "
          f"{len(skipped)} skipped")


if __name__ == "__main__":
    main()
