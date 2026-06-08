#!/usr/bin/env python3
"""build_imatrix_calib_corpus.py — build a category-balanced calibration corpus
for MoE imatrix computation so every expert is routed in every layer.

WHY THIS EXISTS
A code-heavy imatrix corpus (~13:1 code:other) under-routes the non-code
specialist experts **in the middle layers** of a pruned MoE. The aggregate
per-expert coverage looks even, but per-(expert, layer) some experts get
near-zero tokens — and i-quants (IQ2_*/IQ3_*) assign a fixed lattice codebook
weighted by that per-(expert, layer) importance, so a starved expert quantizes
to garbage and the model ruminates (stop-token collapse) at inference. See
`imatrix_expert_coverage.py` for the gate that detects this, and
`feedback_imatrix_expert_coverage` for the full RCA.

HYBRID CONSTRUCTION (the fix)
  1. SEED (map prompts): the producer's `TIER_A_PROMPTS`
     (math/logic/code/science/creative) + a multilingual JSONL. These are the
     exact prompts that built the competence map, so they strongly and
     reliably activate each category's expert set — the routing primer.
  2. FILL (dataset): per-category bulk samples (HF datasets or local JSONL/TXT)
     for within-category diversity, so routing spreads across ALL of a
     category's experts, not just the handful the fixed primers hit.
  Output is TOKEN-BALANCED across the 6 competence categories so no category
  dominates the routing the way the code-heavy corpus did.

WORKFLOW
  1. build_imatrix_calib_corpus.py ... --out corpus.txt
  2. llama-imatrix -m model-F16.gguf -f corpus.txt -o imatrix.dat ...
  3. imatrix_expert_coverage.py imatrix.dat        # gate: 0 starved?
  4. if a category is still short, bump it with --category-weights cat=2.0 and
     rebuild (cheaper than recomputing the imatrix blind).

The corpus is raw text (standard imatrix practice; llama-imatrix chunks by
--chunk tokens regardless of document boundaries). Categories are round-robin
interleaved so a truncated corpus still stays balanced.

Arch-agnostic: the 6 category names match the competence map
(`expert_neuron_base_v*.json` `metadata.categories`); nothing here is tied to a
specific model. Origin: 2026-06-06 Gemma-4 26B-A4B 98e i-quant rumination RCA.
"""
import argparse
import ast
import json
import random
import sys
from pathlib import Path

# Competence-map category order (matches expert_neuron_base_v*.json).
CATEGORIES = [
    "generic_math", "generic_logic", "generic_code",
    "generic_science", "generic_creative", "generic_multilingual",
]
# TIER_A_PROMPTS uses short keys; map them to the canonical category names.
TIER_A_KEY_TO_CAT = {
    "math": "generic_math", "logic": "generic_logic", "code": "generic_code",
    "science": "generic_science", "creative": "generic_creative",
}

# Default per-category dataset fill. Each entry is tried independently and
# degrades to a warning (never fatal) so one unreachable dataset can't sink the
# build. Override the whole map with --fill-spec <json>; use --no-fill for a
# seed-only corpus (useful as a quick smoke).
DEFAULT_FILL_SPEC = {
    # n is tuned to each source's token density so every category can reach the
    # token target (short-doc sources like commonsense_qa need many more rows).
    "generic_math":         {"hf": "openai/gsm8k", "config": "main", "split": "train", "field": "question", "n": 600},
    "generic_logic":        {"hf": "tau/commonsense_qa", "split": "train", "field": "question", "n": 1500},
    "generic_code":         {"hf": "jtatman/python-code-dataset-500k", "split": "train", "field": "output", "n": 300},
    "generic_science":      {"hf": "allenai/sciq", "split": "train", "field": "support", "n": 500},
    "generic_creative":     {"hf": "roneneldan/TinyStories", "split": "train", "field": "text", "n": 300},
    "generic_multilingual": {"hf": "CohereForAI/aya_dataset", "config": "default", "split": "train", "field": "inputs", "n": 800},
}


def load_tier_a(py_path):
    """Extract the TIER_A_PROMPTS dict literal from the producer .py via AST.

    We don't import the module (it pulls torch + the whole analysis stack);
    instead we parse the file, find the `TIER_A_PROMPTS = {...}` assignment, and
    literal_eval its value. The dict is pure string literals, so this is safe.
    Returns {category_name: [prompt, ...]} using canonical category names.
    """
    src = Path(py_path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TIER_A_PROMPTS":
                    raw = ast.literal_eval(node.value)
                    out = {}
                    for k, v in raw.items():
                        cat = TIER_A_KEY_TO_CAT.get(k, k if k.startswith("generic_") else f"generic_{k}")
                        out[cat] = list(v)
                    return out
    raise SystemExit(f"FATAL: TIER_A_PROMPTS not found in {py_path}")


def load_jsonl_field(path, field, limit=None):
    """Read a JSONL file and return the `field` value of each record."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = rec.get(field)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
            if limit and len(out) >= limit:
                break
    return out


def load_fill_entry(cat, spec, seed):
    """Load fill docs for one category from its spec. Never raises — returns []
    and prints a warning on any failure so the build continues (the coverage
    gate will report any resulting shortfall; nothing is silently dropped)."""
    n = int(spec.get("n", 400))
    if "local" in spec:
        path = spec["local"]
        field = spec.get("field", "prompt")
        try:
            if path.endswith(".jsonl"):
                docs = load_jsonl_field(path, field, limit=None)
            else:
                docs = [d for d in Path(path).read_text().split("\n\n") if d.strip()]
        except Exception as e:  # noqa: BLE001 — degrade, don't fail the build
            print(f"  [warn] {cat}: local fill {path} failed ({e}); skipping", file=sys.stderr)
            return []
        random.Random(seed).shuffle(docs)
        return docs[:n]
    # HF dataset path
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  [warn] {cat}: `datasets` not importable; skipping HF fill", file=sys.stderr)
        return []
    hf = spec["hf"]
    field = spec.get("field")
    try:
        # Stream (no full download) so a large dataset like TinyStories/aya only
        # costs the n rows we actually take. Shuffle through a bounded buffer for
        # diversity without materializing the whole split.
        kw = {"split": spec.get("split", "train"), "streaming": True}
        if spec.get("config"):
            ds = load_dataset(hf, spec["config"], **kw)
        else:
            ds = load_dataset(hf, **kw)
        ds = ds.shuffle(seed=seed, buffer_size=max(2000, 5 * n))
        docs = []
        for rec in ds:
            val = rec.get(field) if field else None
            if isinstance(val, str) and val.strip():
                docs.append(val.strip())
            if len(docs) >= n:
                break
        if not docs:
            print(f"  [warn] {cat}: HF {hf} field '{field}' yielded 0 docs; skipping", file=sys.stderr)
        return docs
    except Exception as e:  # noqa: BLE001 — degrade, don't fail the build
        print(f"  [warn] {cat}: HF fill {hf} failed ({e}); skipping", file=sys.stderr)
        return []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier-a-py", required=True,
                    help="producer .py containing TIER_A_PROMPTS (e.g. expert_neuron_analysis_v5_targeted.py)")
    ap.add_argument("--multilingual", help="JSONL of multilingual prompts (the 6th category seed)")
    ap.add_argument("--multilingual-field", default="prompt")
    ap.add_argument("--fill-spec", help="JSON {category: {hf|local, ...}} overriding the default fill map")
    ap.add_argument("--no-fill", action="store_true", help="seed-only corpus (no dataset fill)")
    ap.add_argument("--tokenizer", help="HF tokenizer dir for token-balancing (recommended)")
    ap.add_argument("--target-tokens-per-category", type=int, default=60000,
                    help="aim for ~this many tokens per category (default 60000)")
    ap.add_argument("--max-doc-tokens", type=int, default=1024,
                    help="truncate any single fill doc to this many tokens (default 1024)")
    ap.add_argument("--category-weights", nargs="*", default=[],
                    help="per-category token multipliers, e.g. generic_logic=2.0 (boost a starved category)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output corpus .txt")
    ap.add_argument("--report-json", help="write a per-category build report here")
    args = ap.parse_args()

    weights = {c: 1.0 for c in CATEGORIES}
    for w in args.category_weights:
        k, _, v = w.partition("=")
        if k in weights:
            weights[k] = float(v)
        else:
            sys.exit(f"FATAL: --category-weights unknown category '{k}' (valid: {CATEGORIES})")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def ntok(s):
        if tok is None:
            return max(1, len(s) // 4)  # ~4 chars/token heuristic when no tokenizer
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def clip(s):
        if tok is None or args.max_doc_tokens <= 0:
            return s
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) <= args.max_doc_tokens:
            return s
        return tok.decode(ids[:args.max_doc_tokens])

    # --- seed: map prompts ---
    seed_by_cat = load_tier_a(args.tier_a_py)
    if args.multilingual:
        ml = load_jsonl_field(args.multilingual, args.multilingual_field)
        seed_by_cat.setdefault("generic_multilingual", [])
        seed_by_cat["generic_multilingual"].extend(ml)
    for c in CATEGORIES:
        seed_by_cat.setdefault(c, [])

    # --- fill: dataset ---
    fill_spec = DEFAULT_FILL_SPEC
    if args.fill_spec:
        fill_spec = json.loads(Path(args.fill_spec).read_text())
    fill_by_cat = {c: [] for c in CATEGORIES}
    if not args.no_fill:
        for c in CATEGORIES:
            spec = fill_spec.get(c)
            if spec:
                print(f"  fill {c}: {spec.get('hf') or spec.get('local')} ...", file=sys.stderr)
                fill_by_cat[c] = [clip(d) for d in load_fill_entry(c, spec, args.seed)]

    # --- balance: per category, take seed first then fill up to the token target ---
    per_cat_docs = {}
    report = {}
    for c in CATEGORIES:
        target = int(args.target_tokens_per_category * weights[c])
        docs, used = [], 0
        for d in seed_by_cat[c] + fill_by_cat[c]:
            if used >= target:
                break
            docs.append(d)
            used += ntok(d)
        per_cat_docs[c] = docs
        short = used < target
        report[c] = {"docs": len(docs), "seed": len(seed_by_cat[c]),
                     "fill_avail": len(fill_by_cat[c]), "tokens": used,
                     "target": target, "short": short}
        if short:
            print(f"  [warn] {c}: only {used} tok < target {target} "
                  f"(seed={len(seed_by_cat[c])} fill={len(fill_by_cat[c])}) — "
                  "add more fill or lower the target", file=sys.stderr)

    # --- emit: round-robin interleave across categories ---
    rng = random.Random(args.seed)
    for c in CATEGORIES:
        rng.shuffle(per_cat_docs[c])
    iters = {c: iter(per_cat_docs[c]) for c in CATEGORIES}
    out_docs, live = [], set(CATEGORIES)
    while live:
        for c in list(CATEGORIES):
            if c in live:
                try:
                    out_docs.append(next(iters[c]))
                except StopIteration:
                    live.discard(c)
    Path(args.out).write_text("\n\n".join(out_docs) + "\n")

    total_tok = sum(r["tokens"] for r in report.values())
    print(f"\nwrote {args.out}: {len(out_docs)} docs, ~{total_tok:,} tokens "
          f"({'tokenizer' if tok else 'char-heuristic'})")
    for c in CATEGORIES:
        r = report[c]
        print(f"  {c:22s} docs={r['docs']:4d} (seed {r['seed']}, fill avail {r['fill_avail']:4d}) "
              f"tokens={r['tokens']:7,d}/{r['target']:,}{'  <SHORT>' if r['short'] else ''}")
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
