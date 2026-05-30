#!/usr/bin/env python3
'''Audit a full-bench eval result for rumination/looping pathology + knowledge shift.

This reads the ACTUAL answer text (not just character lengths), so a degenerate
loop can never be mislabeled CLEAN, and it attributes every failure to a cause
(empty / loop / normal-wrong) so a rumination regression is not mislabeled as a
knowledge shift.

Per-bench loaders:
  humanevalplus_full / ifeval_100 : lm-eval samples_*.jsonl with per-doc pass
  multipl_e_100                    : mpe_result.samples.jsonl + summary.json pass rate

Flags (a result is CLEAN only if NONE of these fire):
  PARTIAL_BENCH    n < expected (broken eval; result not comparable)
  LOOP             >=1 answer is a genuine degenerate loop (text-level) -- HARD GATE
  LOOP_REGRESSION  loop rate materially above baseline
  SAT_COLLAPSE     >10% saturation (>14k chars)
  SAT_WARN         5-10% saturation + >2x baseline
  LEN_BLOAT        p50 > 3x baseline p50
  EMPTY_FAIL       >5% truly-empty (<15 chars) answers
  KNOWLEDGE_SHIFT  pass drop >=5pp explained by MORE normal-length wrong answers
                   (not by looping/empties) + saturation ~= baseline
  CLEAN_TRADE      pass drop 2-5pp with lower saturation and no looping
  CLEAN            no pathology

The single line printed to stdout is parsed by the orchestrator (auto-audit
shells grep `flags=[...]`). The full per-sample stats are written to audit.json.

Env overrides (so the same script serves every host -- no per-host clone):
  OMK_AUDIT_ROOT       eval-results root  (default /srv/ml/eval_results_tracks_2_3)
  OMK_AUDIT_BASELINE   default baseline variant
'''
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('OMK_AUDIT_ROOT', '/srv/ml/eval_results_tracks_2_3'))
BASELINE = os.environ.get('OMK_AUDIT_BASELINE', 'a2-62e-fc15_25-p8-s1_0p1_20')
EXPECTED_N = {'humanevalplus_full': 164, 'ifeval_100': 100, 'multipl_e_100': 300}

TRULY_EMPTY = 15        # chars; below this an answer is empty/under-thought, not just short
SAT_CHARS = 14000       # runaway-length proxy (over-thinking by sheer length)


def resp(s):
    '''The model's emitted answer text. Prefer the scored (filtered) response;
    fall back to the raw generation, then to a bare completion field.'''
    fr = s.get('filtered_resps')
    if isinstance(fr, list) and fr:
        x = fr[0]
        if isinstance(x, list):
            x = x[0] if x else ''
        if isinstance(x, str) and x:
            return x
    r = s.get('resps')
    if isinstance(r, list) and r and r[0]:
        x = r[0][0] if isinstance(r[0], list) else r[0]
        if isinstance(x, str):
            return x
    return s.get('completion', '') or ''


def detect_loop(t):
    '''High-precision degenerate-loop detector. Returns True only for runaway
    cycles -- NOT for legitimately repetitive answers (numbered lists, or a
    "repeat this phrase N times" instruction, where N is small).

    Two phase-independent signals over whitespace tokens:
      (1) the TAIL (last ~220 words) has a distinct-5-gram ratio < 0.30 -- the
          classic "stuck in a cycle until the token cap" signature; and
      (2) any 5-word shingle repeats >= 12x across the whole answer -- a count
          far beyond any plausible "repeat N times" instruction.
    Both are deliberately conservative so a hit is almost certainly pathological.
    '''
    if len(t) < 600:
        return False
    w = t.split()
    if len(w) < 60:
        return False
    # (1) tail cycle via low lexical diversity at the end
    tail = w[-220:]
    if len(tail) >= 60:
        sh = [' '.join(tail[i:i + 5]) for i in range(len(tail) - 4)]
        if sh and len(set(sh)) / len(sh) < 0.30:
            return True
    # (2) a single 5-gram dominating by raw repetition count
    sh_all = [' '.join(w[i:i + 5]) for i in range(len(w) - 4)]
    if sh_all:
        c = Counter(sh_all)
        if c.most_common(1)[0][1] >= 12:
            return True
    return False


def per_sample_pass(bench):
    '''Return a per-sample pass predicate, or None when the bench has no
    reliable per-doc pass field (then attribution is skipped).'''
    if bench == 'humanevalplus_full':
        return lambda s: s.get('pass@1', 0) >= 0.5
    if bench == 'ifeval_100':
        return lambda s: s.get('prompt_level_strict_acc', 0) >= 0.5
    return None


def stats(samples, bench, fixed_pass_rate=None):
    n = len(samples)
    if n == 0:
        return dict(n=0, pass_=0, pass_rate=0.0, p10=0, p50=0, p90=0, max_=0,
                    saturated_14k=0, sat_rate=0.0, empty=0, empty_rate=0.0,
                    loop=0, loop_rate=0.0, attributed=False,
                    f_empty=0, f_loop=0, f_norm=0, f_norm_rate=0.0)
    texts = [resp(s) for s in samples]
    lens = [len(x) for x in texts]
    loops = [detect_loop(x) for x in texts]

    chk = per_sample_pass(bench)
    attributed = chk is not None and fixed_pass_rate is None
    if fixed_pass_rate is not None:
        pass_rate = fixed_pass_rate
        passes = int(round(pass_rate * n))
    elif chk is not None:
        passes = sum(1 for s in samples if chk(s))
        pass_rate = passes / n
    else:
        passes, pass_rate = 0, 0.0

    f_empty = f_loop = f_norm = 0
    if attributed:
        for s, L, lp in zip(samples, lens, loops):
            if chk(s):
                continue
            if L < TRULY_EMPTY:
                f_empty += 1
            elif lp or L > SAT_CHARS:
                f_loop += 1
            else:
                f_norm += 1

    slens = sorted(lens)
    sat = sum(1 for x in lens if x > SAT_CHARS)
    empty = sum(1 for x in lens if x < TRULY_EMPTY)
    loop_count = sum(1 for x in loops if x)
    p10 = slens[n // 10] if n >= 10 else slens[0]
    p50 = slens[n // 2]
    p90 = slens[(n * 9) // 10] if n >= 10 else slens[-1]
    return dict(n=n, pass_=passes, pass_rate=pass_rate,
                p10=p10, p50=p50, p90=p90, max_=slens[-1],
                saturated_14k=sat, sat_rate=sat / n,
                empty=empty, empty_rate=empty / n,
                loop=loop_count, loop_rate=loop_count / n,
                attributed=attributed,
                f_empty=f_empty, f_loop=f_loop, f_norm=f_norm,
                f_norm_rate=f_norm / n)


def load_lm_eval(bench, variant):
    bd = ROOT / bench / variant / 'lm_eval_out'
    if not bd.exists():
        return None
    files = sorted(bd.glob('**/samples_*.jsonl'))
    if not files:
        return None
    return [json.loads(ln) for ln in files[-1].read_text().splitlines() if ln.strip()]


def load_mpe(bench, variant):
    f = ROOT / bench / variant / 'mpe_result.samples.jsonl'
    if not f.exists():
        return None
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


def load_summary_score(bench, variant):
    f = ROOT / bench / variant / 'summary.json'
    if not f.exists():
        return None
    return json.loads(f.read_text()).get('score')


def load_for_bench(bench, variant):
    if bench == 'multipl_e_100':
        s = load_mpe(bench, variant)
        if s is None:
            return None, None
        return s, load_summary_score(bench, variant)
    return load_lm_eval(bench, variant), None


def main():
    if len(sys.argv) < 3:
        print('usage: audit_full_bench.py <bench> <variant> [baseline]', file=sys.stderr)
        sys.exit(1)
    bench, variant = sys.argv[1], sys.argv[2]
    baseline = sys.argv[3] if len(sys.argv) >= 4 else BASELINE

    vd, vscore = load_for_bench(bench, variant)
    if vd is None:
        print('AUDIT_FAIL  bench=' + bench + '  variant=' + variant + '  reason=no_samples', flush=True)
        sys.exit(2)
    bd, bscore = load_for_bench(bench, baseline) if baseline != variant else (None, None)

    vs = stats(vd, bench, fixed_pass_rate=vscore)
    bs = stats(bd, bench, fixed_pass_rate=bscore) if bd else None

    expected = EXPECTED_N.get(bench, vs['n'])
    is_partial = vs['n'] < expected

    flags = []
    if is_partial:
        flags.append('PARTIAL_BENCH')

    # --- text-level looping: HARD GATE. Any genuine degenerate loop -> not CLEAN. ---
    if vs['loop'] >= 1:
        flags.append('LOOP')
    loop_rose = bool(bs) and vs['loop_rate'] > bs['loop_rate'] + 0.02 and vs['loop'] > bs['loop']
    if loop_rose:
        flags.append('LOOP_REGRESSION')

    # --- length pathology ---
    if vs['sat_rate'] > 0.10:
        flags.append('SAT_COLLAPSE')
    elif vs['sat_rate'] > 0.05 and (not bs or vs['sat_rate'] > 2 * bs['sat_rate']):
        flags.append('SAT_WARN')
    if vs['empty_rate'] > 0.05:
        flags.append('EMPTY_FAIL')

    # --- score-delta classification (attribution-aware so we don't mislabel) ---
    if bs and not is_partial:
        if bs['p50'] > 0 and vs['p50'] > 3 * bs['p50']:
            flags.append('LEN_BLOAT')
        dp = vs['pass_rate'] - bs['pass_rate']
        sat_match = abs(vs['sat_rate'] - bs['sat_rate']) < 0.03
        have_attr = vs['attributed'] and bs['attributed']
        # Genuine capability loss == MORE normal-length answers are wrong. This is
        # measured directly from attribution and is INDEPENDENT of looping -- a
        # variant can both ruminate more AND lose capability (then it earns both
        # LOOP_REGRESSION and KNOWLEDGE_SHIFT). When attribution is unavailable
        # (e.g. MPE scored from summary), fall back to the length-proxy heuristic.
        if have_attr:
            norm_rose = (vs['f_norm_rate'] - bs['f_norm_rate']) > 0.02
            if dp < -0.05 and norm_rose:
                flags.append('KNOWLEDGE_SHIFT')
        elif dp < -0.05 and sat_match and not loop_rose:
            flags.append('KNOWLEDGE_SHIFT')
        if dp < -0.02 and vs['sat_rate'] < bs['sat_rate'] - 0.01 and vs['loop'] == 0:
            flags.append('CLEAN_TRADE')

    if not flags:
        flags.append('CLEAN')

    audit = dict(bench=bench, variant=variant, baseline=baseline,
                 expected_n=expected, is_partial=is_partial,
                 variant_stats=vs, baseline_stats=bs, flags=flags)
    out_dir = ROOT / bench / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'audit.json').write_text(json.dumps(audit, indent=2, default=str))

    dp = (vs['pass_rate'] - bs['pass_rate']) if bs else 0.0
    n_str = str(vs['n']) + '/' + str(expected)
    if vs['attributed']:
        fails = vs['f_empty'] + vs['f_loop'] + vs['f_norm']
        fail_str = '%d(e%d/l%d/n%d)' % (fails, vs['f_empty'], vs['f_loop'], vs['f_norm'])
    else:
        fail_str = 'n/a'
    line = ('AUDIT  bench=' + bench.ljust(20) +
            '  variant=' + variant.ljust(50) +
            '  n=' + n_str.rjust(7) +
            '  score=' + format(vs['pass_rate'], '.3f') +
            '  d=' + format(dp, '+.3f') +
            '  p50=' + str(vs['p50']).rjust(5) +
            '  sat=' + format(vs['sat_rate'] * 100, '.0f') + '%' +
            '  loop=' + format(vs['loop_rate'] * 100, '.0f') + '%' +
            '  fails=' + fail_str.rjust(12) +
            '  flags=[' + ','.join(flags) + ']')
    print(line, flush=True)


if __name__ == '__main__':
    main()
