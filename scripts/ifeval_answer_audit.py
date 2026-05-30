#!/usr/bin/env python3
'''Deep per-answer cohort audit: length distribution, looping, empties,
over/under-thinking, and failure-cause attribution -- for a whole cohort of
variants on one bench, in a single side-by-side table.

Where audit_full_bench.py emits one flag line per variant (for the orchestrator
to grep), this prints the underlying per-answer numbers so you can see WHY a
variant differs: length percentiles, runaway (>14k) + degenerate-loop counts,
and each failure split into empty / loop / normal-wrong. It reuses the SAME
detect_loop() / resp() / per_sample_pass() as audit_full_bench.py -- one
detector, no drift.

Usage:
  ifeval_answer_audit.py <bench> [variant ...]    # audit the named variants
  ifeval_answer_audit.py <bench>                   # auto: every variant under the bench
                                                   # (skips *_OLD / *_validation* dead dirs)
Env:
  OMK_AUDIT_ROOT   eval-results root (default /srv/ml/eval_results_tracks_2_3)

Examples:
  python scripts/ifeval_answer_audit.py ifeval_100
  python scripts/ifeval_answer_audit.py humanevalplus_full variantA variantB
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_full_bench import (  # noqa: E402  (path shim must precede import)
    ROOT, SAT_CHARS, TRULY_EMPTY, detect_loop, load_for_bench, per_sample_pass, resp,
)

SKIP_SUFFIXES = ('_OLD', '_validation', '_p8_OLD')


def pctl(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def discover_variants(bench):
    bd = ROOT / bench
    if not bd.is_dir():
        return []
    out = []
    for d in sorted(bd.iterdir()):
        if not d.is_dir():
            continue
        if any(s in d.name for s in SKIP_SUFFIXES):
            continue
        out.append(d.name)
    return out


def audit_variant(bench, variant):
    samples, _ = load_for_bench(bench, variant)
    if not samples:
        return None
    chk = per_sample_pass(bench)
    texts = [resp(s) for s in samples]
    lens = [len(t) for t in texts]
    loops = [detect_loop(t) for t in texts]
    n = len(samples)

    empty = sum(1 for x in lens if x < TRULY_EMPTY)
    tiny = sum(1 for x in lens if x < 20)
    short = sum(1 for x in lens if x < 60)
    over8 = sum(1 for x in lens if x > 8000)
    over14 = sum(1 for x in lens if x > SAT_CHARS)
    loop_n = sum(1 for x in loops if x)

    fails = f_empty = f_loop = f_norm = 0
    attributed = chk is not None
    if attributed:
        for s, L, lp in zip(samples, lens, loops):
            if chk(s):
                continue
            fails += 1
            if L < TRULY_EMPTY:
                f_empty += 1
            elif lp or L > SAT_CHARS:
                f_loop += 1
            else:
                f_norm += 1
    return dict(variant=variant, n=n, empty=empty, tiny=tiny, short=short,
                p50=pctl(lens, .50), p90=pctl(lens, .90), p99=pctl(lens, .99),
                max=max(lens), over8=over8, over14=over14, loop=loop_n,
                attributed=attributed, fails=fails,
                f_empty=f_empty, f_loop=f_loop, f_norm=f_norm)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bench = sys.argv[1]
    variants = sys.argv[2:] or discover_variants(bench)
    if not variants:
        print('no variants found under ' + str(ROOT / bench), file=sys.stderr)
        sys.exit(2)

    hdr = ('{:42} {:>4} {:>5} {:>4} {:>4} {:>6} {:>6} {:>6} {:>7} {:>4} {:>5} {:>4} | '
           '{:>5} {:>3} {:>3} {:>3}').format(
        'variant', 'n', 'empty', '<20', '<60', 'p50', 'p90', 'p99', 'max',
        '>8k', '>14k', 'loop', 'fail', 'fE', 'fL', 'fN')
    print(hdr)
    print('-' * len(hdr))
    for v in variants:
        r = audit_variant(bench, v)
        if r is None:
            print('{:42} NO SAMPLES'.format(v[:42]))
            continue
        fa = ('{:>5} {:>3} {:>3} {:>3}'.format(r['fails'], r['f_empty'], r['f_loop'], r['f_norm'])
              if r['attributed'] else '{:>5} {:>3} {:>3} {:>3}'.format('n/a', '-', '-', '-'))
        print('{:42} {:>4} {:>5} {:>4} {:>4} {:>6} {:>6} {:>6} {:>7} {:>4} {:>5} {:>4} | {}'.format(
            v[:42], r['n'], r['empty'], r['tiny'], r['short'], r['p50'], r['p90'],
            r['p99'], r['max'], r['over8'], r['over14'], r['loop'], fa))


if __name__ == '__main__':
    main()
