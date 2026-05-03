#!/usr/bin/env python3
"""Patch the 10 empty/invalid 109e v3 responses by querying the server directly.

Strategy: the llama.cpp PEG parser bug only triggers with --reasoning-format deepseek.
We restart the server WITHOUT that flag and reask each question. If the first seed
fails to produce a valid letter, try other seeds.
"""
import json, re, sys, time, urllib.request

SAMPLES = "eval_results/gpqa_full/109e_v3_Q6K/109e_v3_Q6K/samples_gpqa_diamond_cot_zeroshot_2026-04-11T09-52-03.920413.jsonl"
URL = "http://localhost:8099/v1/chat/completions"

EMPTY_IDS = [23, 48, 62, 79, 94, 115, 126, 127]
TRUNC_IDS = [0, 153]
TARGETS_IDS = EMPTY_IDS + TRUNC_IDS

# Extract answer: lm_eval flexible-extract regex — last (X) in response
FLEX_RE = re.compile(r"\(([A-D])\)", re.I)

SEEDS = [42, 123, 456, 789, 1337, 2026, 9999]

def load_samples():
    by_id = {}
    with open(SAMPLES) as f:
        for line in f:
            d = json.loads(line)
            if d.get("filter") != "flexible-extract":
                continue
            did = d["doc_id"]
            if did in TARGETS_IDS:
                # arg_0 is a list with one JSON string = chat messages
                chat_str = d["arguments"]["gen_args_0"]["arg_0"][0]
                messages = json.loads(chat_str)
                by_id[did] = {
                    "target": d["target"].strip("()"),  # '(D)' -> 'D'
                    "messages": messages,
                }
    return by_id

def extract_letter(text):
    """Return last (X) letter found in text, or None."""
    if not text:
        return None
    m = FLEX_RE.findall(text)
    if not m:
        return None
    return m[-1].upper()

def query(messages, seed):
    """Send chat request with given seed, return response content."""
    payload = {
        "model": "109e_v3_patch",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "seed": seed,
        "max_tokens": 24576,
        # Do NOT set reasoning_format — avoid PEG bug
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"].get("content", "")
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} questions to patch: {sorted(samples.keys())}")
    print()

    results = {}
    for did in sorted(samples.keys()):
        meta = samples[did]
        target = meta["target"]
        messages = meta["messages"]
        print(f"=== Q{did} (target={target}) ===", flush=True)

        answer = None
        attempts = []
        for seed in SEEDS:
            t0 = time.time()
            content = query(messages, seed)
            dt = time.time() - t0
            letter = extract_letter(content)
            attempts.append({
                "seed": seed,
                "letter": letter,
                "len": len(content),
                "time_s": round(dt, 1),
                "correct": letter == target,
            })
            print(f"  seed={seed:<5} -> letter={letter}  (len={len(content)}, {dt:.1f}s) {'CORRECT' if letter==target else 'WRONG' if letter else 'NO_ANSWER'}", flush=True)
            if letter is not None:
                answer = letter
                # If we got an answer, we're done — we'd bias by picking seeds until correct
                break

        results[did] = {
            "target": target,
            "answer": answer,
            "correct": answer == target if answer else False,
            "attempts": attempts,
        }
        print()

    # Summary
    print("=" * 60)
    print("PATCH SUMMARY")
    print("=" * 60)
    valid = [did for did, r in results.items() if r["answer"] is not None]
    correct = [did for did in valid if results[did]["correct"]]
    print(f"Got an answer:    {len(valid)}/{len(results)}")
    print(f"Answer correct:   {len(correct)}/{len(results)}")
    print()
    for did in sorted(results.keys()):
        r = results[did]
        mark = "OK " if r["correct"] else ("WRONG" if r["answer"] else "NONE ")
        print(f"  Q{did:3d}: target={r['target']}  answer={r['answer']}  [{mark}]")

    # Save
    with open("eval_results/gpqa_full/109e_v3_Q6K/109e_v3_patches.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to eval_results/gpqa_full/109e_v3_Q6K/109e_v3_patches.json")

if __name__ == "__main__":
    main()
