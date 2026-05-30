#!/usr/bin/env python3
'''Dump the prompt, pass/fail, and answer tail of every response that
audit_full_bench.detect_loop() flags as a degenerate loop, for one or more
variants. Use it to verify the detector's precision (no legit "repeat N times"
or list false positives) and to read what the rumination actually looks like.

Reuses the SAME detect_loop() / resp() / per_sample_pass() as audit_full_bench.py
so what this prints is exactly what the LOOP flag keys on -- no separate detector.

Usage:
  loop_precision_check.py <bench> <variant> [variant ...]
  loop_precision_check.py <bench>                 # auto: every variant under the bench
Env:
  OMK_AUDIT_ROOT   eval-results root (default /srv/ml/eval_results_tracks_2_3)
  OMK_LOOP_TAIL    chars of tail to print (default 180)

Examples:
  python scripts/loop_precision_check.py ifeval_100 gemma-4-A4B-62e-fc15_25-p8-shared130-it
  python scripts/loop_precision_check.py humanevalplus_full
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_full_bench import (  # noqa: E402  (path shim must precede import)
    ROOT, detect_loop, load_for_bench, per_sample_pass, resp,
)

SKIP_SUFFIXES = ('_OLD', '_validation', '_p8_OLD')
TAIL = int(os.environ.get('OMK_LOOP_TAIL', '180'))


def prompt_of(s):
    doc = s.get('doc', {}) or {}
    for k in ('prompt', 'question', 'turns', 'text'):
        v = doc.get(k)
        if isinstance(v, str) and v:
            return v
    return str(s.get('arguments', ''))[:120]


def discover_variants(bench):
    bd = ROOT / bench
    if not bd.is_dir():
        return []
    return [d.name for d in sorted(bd.iterdir())
            if d.is_dir() and not any(x in d.name for x in SKIP_SUFFIXES)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bench = sys.argv[1]
    variants = sys.argv[2:] or discover_variants(bench)
    if not variants:
        print('no variants found under ' + str(ROOT / bench), file=sys.stderr)
        sys.exit(2)

    chk = per_sample_pass(bench)
    grand = 0
    for variant in variants:
        samples, _ = load_for_bench(bench, variant)
        if not samples:
            print('\n##### %s : NO SAMPLES #####' % variant)
            continue
        flagged = [s for s in samples if detect_loop(resp(s))]
        grand += len(flagged)
        print('\n##### %s : %d/%d loop-flagged #####' % (variant, len(flagged), len(samples)))
        for s in flagged:
            t = resp(s)
            p = chk(s) if chk else '?'
            pr = prompt_of(s)[:90].replace('\n', ' ')
            print('  doc=%-4s len=%-6d pass=%s  PROMPT: %s' % (s.get('doc_id'), len(t), p, pr))
            print('      TAIL: %r' % (t[-TAIL:],))
    print('\nTOTAL loop-flagged across %d variant(s): %d' % (len(variants), grand))


if __name__ == '__main__':
    main()
