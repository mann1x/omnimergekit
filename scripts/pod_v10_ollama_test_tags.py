#!/usr/bin/env python3
"""pod_v10_ollama_test_tags.py — ON THE POD: push a LOCAL test GGUF as private test tags of an existing ollama model.

Owner 2026-10-17: "push them as private ollama test tags in https://ollama.com/mannix/gemma-4-20B-v9-agentic".
For <T> = <prefix>-<quant> the four v9 variants are created and pushed, exactly as pod_v9_xollama_publish.py
does for a published tier (same Modelfile, RENDERER/PARSER gemma4, same params, same drafter + projector):

    <T>   vision-<T>   mtp-<T>   mtp-vision-<T>

The model is private on ollama.com, so its new tags are too. Gates, reused from pod_v9_xollama_publish.py:
  0. preflight: the Modelfile rendered from --params must equal the one the publisher created the live
     :Q4_K_M from (RENDERER/PARSER + every PARAMETER) — the test tags differ from the release only in weights;
  1. local_check (renderer/parser, think budget + message, drafter, vision) on the created tag;
  2. generation_check: 3 questions through xollama itself, >= 2/3 in the visible answer;
  3. registry read-back after push (tag_ok) and the served model-layer digest == the one created locally.
A push is never trusted from its exit code. Report lines go to --report (TSV).

usage: pod_v10_ollama_test_tags.py --gguf <file> --prefix hub_k12 [--quant Q4_K_M]

Local-only (owner 2026-10-21, "don't publish them"): --no-push builds and checks the tags (gates 0-2) and KEEPS them
in the local store, for osync to other hosts; --variants picks which of the four; --host is the xollama to use.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pod_v9_xollama_publish as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--ollama", default="mannix/gemma-4-20B-v9-agentic")
    ap.add_argument("--params", default="/workspace/stage/ollama_params.json")
    ap.add_argument("--mmproj", default="/workspace/gguf/gemma-4-20B-v9-agentic/mmproj-gemma-4-20B-v9-agentic-F16.gguf")
    ap.add_argument("--drafter", default="/workspace/ollama_scratch/drafter/gemma-4-26B-A4B-it-assistant.Q8_0.gguf")
    ap.add_argument("--ref-modelfile", default="/workspace/ollama_scratch/Modelfile.Q4_K_M")
    ap.add_argument("--scratch", default="/workspace/v10/ollama")
    ap.add_argument("--report", default="/workspace/v10/logs/ollama_test_tags.tsv")
    ap.add_argument("--variants", default="plain,vision,mtp,mtp-vision", help="subset of plain,vision,mtp,mtp-vision")
    ap.add_argument("--no-push", action="store_true", help="create + check only, keep the tags local (no registry)")
    ap.add_argument("--host", default=P.HOST, help="the xollama to create and check on (host:port)")
    ap.add_argument("--store", default=str(P.STORE), help="that xollama's OLLAMA_MODELS (manifests are read from it)")
    a = ap.parse_args()
    P.HOST, P.STORE = a.host, Path(a.store)
    want = set(a.variants.split(","))
    if not want <= {"plain", "vision", "mtp", "mtp-vision"}:
        P.log(f"FATAL unknown --variants {a.variants}")
        return 2

    gguf, mmproj, drafter = Path(a.gguf), Path(a.mmproj), Path(a.drafter)
    need = [gguf] + ([mmproj] if want & {"vision", "mtp-vision"} else []) + ([drafter] if want & {"mtp", "mtp-vision"} else [])
    for p in need:
        if not p.exists():
            P.log(f"FATAL missing {p}")
            return 2
    base_params = json.load(open(a.params))
    # The live tag's blobs are not in this store, and an authenticated read-back would pull 13 GB, so the
    # preflight compares against the Modelfile the publisher created the live :Q4_K_M from (FROM lines aside).
    ref_mf = Path(a.ref_modelfile).read_text().splitlines()
    ours = P.modelfile(gguf, None, None, base_params).splitlines()
    strip = lambda ls: [x for x in ls if not x.startswith("FROM ")]  # noqa: E731
    if strip(ref_mf) != strip(ours):
        P.log(f"FATAL preflight: {a.params} renders a Modelfile different from {a.ref_modelfile}")
        return 2
    P.log(f"preflight OK: Modelfile == {a.ref_modelfile} (RENDERER/PARSER gemma4 + every PARAMETER)")

    t = f"{a.prefix}-{a.quant}"
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for kind, tag, vis, mtp in [("plain", t, False, False), ("vision", f"vision-{t}", True, False),
                                ("mtp", f"mtp-{t}", False, True), ("mtp-vision", f"mtp-vision-{t}", True, True)]:
        if kind not in want:
            continue
        p = dict(base_params)
        if mtp:
            p["draft_num_predict"] = P.DRAFT_N
        ref = f"{a.ollama}:{tag}"
        mf = scratch / f"Modelfile.{tag}"
        mf.write_text(P.modelfile(gguf, mmproj if vis else None, drafter if mtp else None, p))
        for attempt in range(1, 4):
            try:
                P.xo("rm", ref, check=False)
                P.xo("create", ref, "-f", str(mf))
                P.local_check(ref, p, vis, mtp)
                made = P.model_digest(a.ollama, tag)
                P.generation_check(ref)
                if a.no_push:
                    P.log(f"  {ref} CREATED locally on {P.HOST} (checked, not pushed), model {made[:19]}")
                    with open(a.report, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{ref}\t{made}\tlocal\n")
                    break
                P.xo("push", ref)
                info = P.registry_readback(a.ollama, tag)
                ok, why = P.tag_ok(info, p, vis, mtp)
                if not ok or info is None:
                    raise RuntimeError(f"registry read-back: {why}")
                served = info["layers"]["model"][0]["digest"]
                if served != made:
                    raise RuntimeError(f"registry serves model {served[:19]}, created {made[:19]}")
                P.log(f"  {ref} PUSHED (verified on the registry, model {made[:19]})")
                with open(a.report, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{ref}\t{made}\tok\n")
                break
            except Exception as e:  # noqa: BLE001 — logged, retried, then fatal
                P.log(f"  {ref} attempt {attempt} FAILED: {e}")
                if attempt == 3:
                    P.log(f"FATAL {ref}")
                    return 2
                time.sleep(60 * attempt)
        if not a.no_push:
            P.xo("rm", ref, check=False)
    P.log(f"###### TEST_TAGS_DONE {t} ######")
    return 0


if __name__ == "__main__":
    sys.exit(main())
