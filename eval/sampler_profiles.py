#!/usr/bin/env python3
"""Per-model sampling profiles for omk_eval.

A profile is a YAML under ``eval/models/<family>.yaml`` that declares:

  - ``samplers:``  named sampler presets (e.g. ``greedy`` / ``recommended`` /
    ``deployment``), each a dict of sampler knobs.
  - ``bench_policy:``  a per-template default sampler name (``default:`` plus
    optional per-bench overrides keyed by template name).
  - ``match:``  glob patterns matched against the served-name / model-dir for
    ``--sampler-profile auto`` (and for ``--sampler`` without an explicit
    profile).
  - ``protocol_overrides:``  (optional, reserved) other per-model deviations.

The runner layers a *resolved* sampler over the template's frozen
``generation:`` block. When neither ``--sampler`` nor ``--sampler-profile`` is
passed, resolution is a strict no-op — the frozen greedy templates stay
byte-identical, preserving cross-cohort greedy comparability (EVAL_PROTOCOL.md
§1.0).

Precedence inside ``resolve()``:
  1. explicit ``--sampler <name>``        → source="cli"
  2. profile ``bench_policy[template]``    → source="bench_policy"
  3. profile ``bench_policy.default``      → source="bench_policy"
  4. no profile / no policy hit            → ("", {}, "template_default")  [no-op]

This module is imported by ``omk_eval.py`` the same way as ``gpu_planner`` —
``sys.path.insert(0, <repo>/eval)`` then ``import sampler_profiles``.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("ERROR: PyYAML not installed. `pip install pyyaml`", file=sys.stderr)
    sys.exit(2)


MODELS_DIR = Path(__file__).resolve().parent / "models"

# Sampler knobs we recognise. temperature/top_p/top_k/do_sample overlay the
# template `generation:` block directly; min_p/repeat_penalty are routed
# per-backend by the runner (server-launch flags for llama-server + MultiPL-E,
# per-request fields for vLLM gen_kwargs + the LCB shim).
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "do_sample")


def resolve_path(ref: str) -> Path:
    """Resolve a profile ref to a path. Bundled family name first, then literal path."""
    if ref.endswith((".yaml", ".yml")) or "/" in ref:
        p = Path(ref)
        if not p.exists():
            raise SystemExit(f"sampler profile not found at path: {p}")
        return p
    p = MODELS_DIR / f"{ref}.yaml"
    if not p.exists():
        avail = sorted(t.stem for t in MODELS_DIR.glob("*.yaml")) if MODELS_DIR.exists() else []
        raise SystemExit(
            f"sampler profile '{ref}' not found at {p}. Available: {avail}"
        )
    return p


def load(ref: str) -> dict[str, Any]:
    """Load + lightly validate a profile YAML by family name or path."""
    path = resolve_path(ref)
    with path.open() as f:
        prof = yaml.safe_load(f)
    if not isinstance(prof, dict):
        raise SystemExit(f"{path}: profile YAML must be a mapping")
    samplers = prof.get("samplers")
    if not isinstance(samplers, dict) or not samplers:
        raise SystemExit(f"{path}: profile missing a non-empty `samplers:` mapping")
    prof["__source__"] = str(path)
    return prof


def match_profile(served_name: str = "", model_path: str = "") -> dict[str, Any] | None:
    """Auto-match: first profile in eval/models/ whose ``match:`` glob hits the
    served-name or the model-dir basename. Returns the loaded profile or None.

    Deterministic: profiles are scanned in sorted filename order so a given
    served-name always resolves to the same profile.
    """
    if not MODELS_DIR.exists():
        return None
    cand = [c for c in (served_name or "", Path(model_path).name if model_path else "") if c]
    for yml in sorted(MODELS_DIR.glob("*.yaml")):
        try:
            with yml.open() as f:
                prof = yaml.safe_load(f) or {}
        except Exception:
            continue
        for g in prof.get("match") or []:
            if any(fnmatch.fnmatch(c, g) for c in cand):
                prof["__source__"] = str(yml)
                return prof
    return None


def resolve(profile: dict[str, Any] | None, template_name: str,
            sampler_override: str | None) -> tuple[str, dict[str, Any], str]:
    """Return ``(sampler_name, sampler_dict, source)``.

    ``sampler_dict`` contains only recognised SAMPLER_KEYS. An empty name +
    empty dict + source="template_default" means "no overlay" (the frozen
    template's generation block is used as-is).
    """
    if sampler_override:
        if not profile:
            raise SystemExit(
                f"--sampler {sampler_override!r} requires a sampler profile, but "
                f"none was loaded or matched. Pass --sampler-profile <family|path>."
            )
        name, source = sampler_override, "cli"
    elif profile:
        bp = profile.get("bench_policy") or {}
        name = bp.get(template_name) or bp.get("default") or ""
        source = "bench_policy"
    else:
        return ("", {}, "template_default")

    if not name:
        return ("", {}, "template_default")

    samplers = (profile or {}).get("samplers") or {}
    if name not in samplers:
        raise SystemExit(
            f"sampler '{name}' not in profile samplers {sorted(samplers)} "
            f"({(profile or {}).get('__source__', '<none>')})"
        )
    sd = {k: v for k, v in (samplers[name] or {}).items() if k in SAMPLER_KEYS}
    return (name, sd, source)


def family(profile: dict[str, Any] | None) -> str:
    return (profile or {}).get("family", "-") if profile else "-"


def main() -> None:
    """CLI smoke: `sampler_profiles.py <family> [--template NAME] [--sampler NAME]`."""
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="family name (bundled) or path to a profile YAML")
    ap.add_argument("--template", default="lcb_medium_55_v4")
    ap.add_argument("--sampler", default=None)
    args = ap.parse_args()
    prof = load(args.profile)
    name, sd, source = resolve(prof, args.template, args.sampler)
    print(json.dumps({
        "family": family(prof), "template": args.template,
        "sampler_name": name or "template_default", "source": source,
        "resolved": sd,
    }, indent=2))


if __name__ == "__main__":
    main()
