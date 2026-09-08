import json
f = "/srv/ml/eval_results/qwen_suite/lcb_v6_77q/qwencodermpe_q6k/lcb_result.b606.samples.jsonl"
for line in open(f, errors="ignore"):
    d = json.loads(line)
    if d.get("b606_status") != "unchanged":
        continue
    if bool(d.get("passed")) == bool(d.get("passed_b606")):
        continue
    print("task_id      :", d.get("task_id"))
    print("finish_reason:", d.get("finish_reason"), " tokens:", d.get("completion_tokens"))
    print("banked passed:", d.get("passed"), " reason:", (d.get("reason") or "")[:120])
    print("b606   passed:", d.get("passed_b606"), " reason:", (d.get("reason_b606") or "")[:120])
    print("cleaned identical:", (d.get("cleaned") or "") == (d.get("cleaned_b606") or ""))
    print("cleaned len:", len(d.get("cleaned") or ""))
