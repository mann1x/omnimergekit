#!/usr/bin/env python3
"""fixtures.py — load an adversarial task fixture (bundled id or path to a .json).

A fixture is {id, description, init, followups[]}. The bundled `snake-adversarial`
fixture is the escalating headless-terminal task used in the published study; you can
supply your own by passing a path to a fixture JSON instead of an id.
"""
import json
import os

_FX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def fixtures_dir():
    return _FX_DIR


def load_fixture(ref):
    """ref is a bundled fixture id (e.g. 'snake-adversarial') or a path to a fixture .json."""
    path = ref
    if not os.path.isfile(path):
        path = os.path.join(_FX_DIR, ref if ref.endswith(".json") else ref + ".json")
    if not os.path.isfile(path):
        raise FileNotFoundError("fixture not found: %r (looked in %s)" % (ref, _FX_DIR))
    with open(path, encoding="utf-8") as fh:
        fx = json.load(fh)
    for k in ("id", "init", "followups"):
        if k not in fx:
            raise ValueError("fixture %s missing required key %r" % (path, k))
    return fx
