#!/usr/bin/env python3
"""
Merge the 3 pieces of the 128e Q6_K GPQA Diamond eval into one canonical result.

Pieces:
1. test_6q.py partial (docs 0-178, 179 questions) — full_gpqa_128e_Q6K_def.json
2. lm-eval resume (docs 179-197 + 4 verification) — gpqa_full/128e_Q6K_resume/...jsonl
3. patch (docs 53,88,127,183 with --dry-multiplier) — gpqa_full/patch_128e_Q6K_def.json

Merge logic:
- Start with test_6q.py partial (179 entries, docs 0-178)
- Replace docs 53, 88, 127 with patch results
- Add docs 179-197 from lm-eval resume (skip verification dupes 50/100/150/178)
- Replace doc 183 with patch result
- Verify: exactly 198 entries, one per doc_id 0-197
- Save as gpqa_full/128e_Q6K_final.json
"""
import json
import glob
import os

REPO = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models"

# 1. Load test_6q.py partial
partial = json.load(open(f"{REPO}/eval_results/full_gpqa_128e_Q6K_def.json"))
partial_by_id = {r["doc_id"]: r for r in partial["results"]}
print(f"1. test_6q.py partial: {len(partial_by_id)} entries, docs {min(partial_by_id)}-{max(partial_by_id)}")

# 2. Load lm-eval resume samples (flexible-extract only)
resume_files = glob.glob(f"{REPO}/eval_results/gpqa_full/128e_Q6K_resume/128e_Q6K_resume/samples_*.jsonl")
assert len(resume_files) == 1, f"expected 1 samples file, got {resume_files}"
resume_samples = {}
with open(resume_files[0]) as f:
    for line in f:
        s = json.loads(line)
        did = s["doc_id"]
        fr = s.get("filtered_resps", [[""]])[0]
        if isinstance(fr, list):
            fr = fr[0] if fr else ""
        if fr == "[invalid]":
            continue  # skip strict-match rows
        resume_samples[did] = {
            "doc_id": did,
            "target": s["target"],
            "predicted": fr if fr else None,
            "is_correct": s.get("exact_match", 0) == 1,
            "source": "lm-eval-resume",
        }
print(f"2. lm-eval resume: {len(resume_samples)} unique entries (flexible-extract)")

# 3. Load patch results
patch = json.load(open(f"{REPO}/eval_results/gpqa_full/patch_128e_Q6K_def.json"))
patch_by_id = {r["doc_id"]: r for r in patch["results"]}
print(f"3. patch (--dry-multiplier 0.5): {len(patch_by_id)} entries, docs {sorted(patch_by_id.keys())}")

# Merge
merged = {}

# Start with test_6q.py partial (docs 0-178)
for did, r in partial_by_id.items():
    r["source"] = "test_6q_partial"
    merged[did] = r

# Replace patch docs (53, 88, 127 from partial; 183 from resume)
for did, r in patch_by_id.items():
    r["source"] = "patch_dry_multiplier_0.5"
    merged[did] = r
    print(f"   replaced doc {did} with patch result (correct={r.get('is_correct')}, truncated={r.get('truncated', False)})")

# Add resume docs 179-197 (skip verification dupes 50/100/150/178 — already in partial)
verification_ids = {50, 100, 150, 178}
for did, r in resume_samples.items():
    if did in verification_ids:
        continue  # already have from partial, verified equivalent
    if did not in merged or did == 183:  # 183 already replaced by patch
        if did != 183:  # patch takes priority for 183
            merged[did] = r
            print(f"   added doc {did} from lm-eval resume (correct={r.get('is_correct')})")

# Verify completeness
all_ids = sorted(merged.keys())
expected = list(range(198))
missing = set(expected) - set(all_ids)
extra = set(all_ids) - set(expected)
assert not missing, f"MISSING doc_ids: {missing}"
assert not extra, f"EXTRA doc_ids: {extra}"
assert len(merged) == 198, f"Expected 198 entries, got {len(merged)}"

# Score
correct = sum(1 for r in merged.values() if r.get("is_correct", False))
truncated = sum(1 for r in merged.values() if r.get("truncated", False))
no_answer = sum(1 for r in merged.values() if r.get("predicted") is None)

# Build final
final = {
    "name": "128e_Q6K_final",
    "model": "google/gemma-4-26B-A4B-it",
    "quantization": "Q6_K",
    "score": f"{correct}/{len(merged)}",
    "accuracy": round(correct / len(merged), 4),
    "correct": correct,
    "total": len(merged),
    "truncated": truncated,
    "no_answer": no_answer,
    "methodology": {
        "eval_engine": "lm-eval 0.4.11 (partial via test_6q.py with lm-eval prompts, verified equivalent)",
        "llama_cpp": "b207 (3ba12fe)",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 42,
        "max_gen_toks": 24576,
        "reasoning_budget": 16384,
        "reasoning_format": "deepseek",
        "context_size": 32768,
        "dry_multiplier": "0.5 (patch docs 53/88/127/183 only)",
    },
    "assembly": {
        "pieces": [
            "test_6q.py partial (docs 0-178, 179 entries)",
            "lm-eval resume (docs 179-197, 19 entries, 4 verification skipped)",
            "patch with --dry-multiplier 0.5 (docs 53/88/127/183, 4 entries)",
        ],
        "verification": "4 overlapping samples (Q50/100/150/178) confirmed identical between test_6q.py and lm-eval",
    },
    "sources": {
        did: r.get("source", "unknown") for did, r in sorted(merged.items())
    },
    "results": [merged[did] for did in sorted(merged.keys())],
}

out_path = f"{REPO}/eval_results/gpqa_full/128e_Q6K_final.json"
with open(out_path, "w") as f:
    json.dump(final, f, indent=2)

print(f"\n=== 128e Q6_K FINAL: {correct}/{len(merged)} = {100*correct/len(merged):.1f}% ===")
print(f"  truncated: {truncated}, no_answer: {no_answer}")
print(f"  saved: {out_path}")
