import ast, sys
rows = []
for line in open(sys.argv[1], errors="ignore"):
    s = line.strip()
    if s.startswith("{") and "grad_norm" in s:
        try:
            rows.append(ast.literal_eval(s))
        except Exception:
            pass
gen = [r for r in rows if "completions/mean_length" in r]
print("%d logged steps, %d with generation\n" % (len(rows), len(gen)))
print("%4s %8s %7s %7s %6s %8s %7s" % ("step","reward","rw_std","len","clip","zero_std","grad"))
for i, r in enumerate(gen, 1):
    print("%4d %8.4f %7.4f %7.0f %6.3f %8.2f %7.4f" % (
        i, float(r["reward"]), float(r["reward_std"]),
        float(r["completions/mean_length"]), float(r["completions/clipped_ratio"]),
        float(r["frac_reward_zero_std"]), float(r["grad_norm"])))
def m(x, k): return sum(float(r[k]) for r in x) / len(x)
h, t = gen[:4], gen[-4:]
print("\nfirst4 -> last4")
print("  len    %.0f -> %.0f  (%+.1f%%)" % (m(h,"completions/mean_length"), m(t,"completions/mean_length"),
      100*(m(t,"completions/mean_length")/m(h,"completions/mean_length")-1)))
print("  reward %.4f -> %.4f" % (m(h,"reward"), m(t,"reward")))
print("  clip   %.3f -> %.3f" % (m(h,"completions/clipped_ratio"), m(t,"completions/clipped_ratio")))
