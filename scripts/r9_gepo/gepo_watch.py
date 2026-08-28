"""Stream GEPO training progress as one event line per Nth generation round.

Reads the trainer log on stdin (fed by `tail -f`). Emits:
  - one line every EVERY_N generation rounds (rounds = steps that carry
    completions/* metrics; non-generation steps carry none)
  - one line for any terminal signature, so a crash is never silent

Written as a file rather than an inline awk/shell one-liner: two attempts at
doing this with awk through ssh both failed on quoting (the dict values are
quoted strings, e.g. 'clipped_ratio': '0.3125'), printing empty fields that
looked like progress but carried nothing. See bug-628.
"""
import ast
import re
import sys

EVERY_N = 4
TOTAL_ROUNDS = 32
DICT_RE = re.compile(r"\{.*?'grad_norm'.*?\}")
TERMINAL_RE = re.compile(
    r"GATE_FAIL|FATAL|R9_GEPO_DONE|GEPO_TRAIN_DONE|Traceback|"
    r"out of memory|OutOfMemory|Killed|RuntimeError|WARNING: full leg"
)


def num(row, key):
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    n = 0
    for line in sys.stdin:
        if TERMINAL_RE.search(line):
            print(line.rstrip()[:400], flush=True)
        m = DICT_RE.search(line)
        if not m:
            continue
        try:
            row = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "completions/clipped_ratio" not in row:
            continue          # non-generation step
        n += 1
        if n % EVERY_N:
            continue
        vals = {
            "mean_len": num(row, "completions/mean_length"),
            "term_len": num(row, "completions/mean_terminated_length"),
            "clipped": num(row, "completions/clipped_ratio"),
            "reward": num(row, "reward"),
            "grad_norm": num(row, "grad_norm"),
            "step_s": num(row, "step_time"),
        }
        # A field that silently prints empty is worse than one that says so.
        parts = []
        for k, v in vals.items():
            parts.append("%s=%s" % (k, "MISSING" if v is None else
                                    ("%.4f" % v if k in ("clipped", "reward", "grad_norm")
                                     else "%.0f" % v)))
        print("run2 gen-round %d/%d | %s" % (n, TOTAL_ROUNDS, "  ".join(parts)),
              flush=True)


if __name__ == "__main__":
    main()
