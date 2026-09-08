import json, glob, os, ast
def parses(c):
    try:
        ast.parse(c); return True
    except Exception:
        return False
n = 0
fr = {}
parsed_now = 0
print("recleaned samples (bug-606 actually changed the scored text):")
print("%-34s %-22s %-9s %-7s %-6s %s" % ("cell", "task_id", "finish", "toks", "parses", "new reason"))
for f in sorted(glob.glob("/srv/ml/eval_results/*/lcb_v6_77q*/*/lcb_result.b606.samples.jsonl")):
    cell = os.path.basename(os.path.dirname(f))
    for line in open(f, errors="ignore"):
        d = json.loads(line)
        if d.get("b606_status") != "recleaned":
            continue
        n += 1
        f_r = d.get("finish_reason") or "?"
        fr[f_r] = fr.get(f_r, 0) + 1
        ok = parses(d.get("cleaned_b606") or "")
        parsed_now += int(ok)
        print("%-34s %-22s %-9s %-7s %-6s %s" % (
            cell[:34], d.get("task_id", "")[:22], f_r,
            d.get("completion_tokens"), ok, (d.get("reason_b606") or "")[:44]))
print()
print("recleaned total:", n, " finish_reason census:", fr)
print("of those, cleaned_b606 now ast.parses:", parsed_now)
