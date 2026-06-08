#!/usr/bin/env python
"""pack_pg19_math_rpv2.py — build packed jsonl shards for the Gemma 4 YaRN-2.0
(262144 -> 524288) context-extension continued-pretrain.

(Filename retained for continuity with docs/plans/gemma4_512k_plan_v2.md and the
memory index. The MIXTURE is no longer pg19-only: it is the code-leaned,
domain-balanced, per-source-length-upsampled recipe the long-context literature
converged on. See the research synthesis 2026-06-07.)

WHY THIS MIXTURE (sources, all read 2026-06-07):
  * Fu et al. 2024, "Data Engineering for Scaling LMs to 128K" (arXiv:2402.10171):
    preserve the domain mixture and UPSAMPLE LENGTH WITHIN each source; naive
    book-only upsampling is suboptimal. 0.5-5B tokens suffice.
  * Gao/Wettig et al. 2025, ProLong (arXiv:2410.02660, ACL): ~60% long
    (code repos + books, ~equal) + ~40% high-quality SHORT. 100%-long degrades.
    Code repositories are the single best long source; filter long docs >= the
    pack length; repo-LEVEL (not file-level) code concatenation.
  * Llama-2-Long (arXiv:2309.16039): up-weight long data (don't replace short) —
    also lifts short-context tasks.

OUR TARGET is a 98e-v7-*coder*, so the mix is leaned toward code, and a fat
short fraction (SlimPajama natural blend) preserves the <=64k base / IFEval /
coding competence the council flagged at risk. Default weights (overridable):
    code_long  0.35   repo-concatenated GitHub, docs >= --min-long-tokens
    book_long  0.125  PG19 books, docs >= --min-long-tokens
    sci_long   0.125  proof-pile-2 arxiv, docs >= --min-long-tokens
    short_mix  0.40   SlimPajama natural blend, any length (competence retention)

LENGTH ROUTING ("per-source length upsampling"): a `long`-role source only
contributes documents whose Gemma-4 token length >= --min-long-tokens (default =
pack_len, so a long pack is dominated by ONE coherent long document, ProLong-
faithful). A `short`-role source contributes the natural length distribution.
Cross-domain proportions are held to the target weights by greedy
produced/target balancing across sources, so length upsampling never skews the
domain mix.

PACKING: each document is bookended [BOS] ... [EOS] (ProLong) and concatenated
into pack_len chunks. NO cross-document attention mask is applied at train time
(the trainer runs causal full-attention over the whole pack, matching
"From 128K to 4M" arXiv:2504.06214) — the BOS/EOS markers are the only doc
boundary signal. Output: jsonl shards, one line {"input_ids": [int]*pack_len}.

RUN IT with the omnimergekit env (has `datasets`), NOT omk-yarn:
    /srv/ml/envs/envs/omnimergekit/bin/python pack_pg19_math_rpv2.py \
        --out /srv/ml/longctx/data_98e --tokenizer /srv/ml/google/gemma-4-26B-A4B-it \
        --tokens 500000000 --pack-len 65536 [--dry-run]

Determinism: every source is seeded-shuffled (buffer) and the balancer is
greedy-deterministic, so a fixed (seed, tokens, pack_len, weights) reproduces the
same shards. Streaming, so no full dataset download.
"""
import argparse
import json
import os
import sys
import time

# --- default source spec ----------------------------------------------------
# role: 'long'  -> only docs with >= min_long tokens are admitted
#       'short' -> natural length distribution (competence retention)
# parquet_glob:  HfFileSystem glob of Parquet files to read directly (bypasses
#                loading scripts AND lets us seed-shuffle the file list so a
#                lang-sharded code corpus interleaves instead of streaming
#                alphabetically). When set, `hf`/`config` are display-only.
# meta_filter:  case-insensitive substring matched against
#               row["meta"]["redpajama_set_name"] (SlimPajama domain routing);
#               None = accept any domain.
# min_long:     per-source long floor (0 => use the global --min-long-tokens).
# repo_concat:  repo-LEVEL concatenation from a file-level stream — buffers files
#               by repo_field and flushes a repo as one doc once it reaches
#               ~min_long tokens (Qwen2.5-Coder / ProLong long-code source type).
# row_shuffle:  per-source streaming row shuffle (default True). Set False for a
#               repo_concat code source so intra-file repo adjacency is preserved.
#
# datasets>=4.x DROPPED loading-script support, so EVERY source must be Parquet
# (codeparrot/github-code-clean, proof-pile-2, SlimPajama-627B are script-based
# and FAIL). Sources (all Parquet; starcoderdata gated but access granted):
#   * bigcode/starcoderdata  — decontaminated repo code, INLINE `content` (no S3),
#                              repo-concatenated to real >=64k long code. ~250B tok.
#   * emozilla/pg19          — long books (avg ~399k chars), `text`.
#   * DKYoon/SlimPajama-6B    — RedPajama blend (~6B tok), `text` + meta; arxiv
#                              for sci_long, full blend for short_mix.
DEFAULT_SOURCES = [
    {
        "name": "code_long", "weight": 0.35, "role": "long", "repo_concat": True,
        "hf": "bigcode/starcoderdata",
        "parquet_glob": "datasets/bigcode/starcoderdata/*/*.parquet",
        "text_field": "content", "repo_field": "max_stars_repo_name",
        "path_field": "max_stars_repo_path", "row_shuffle": False,
        "meta_filter": None, "min_long": 0,
    },
    {
        "name": "book_long", "weight": 0.125, "role": "long", "repo_concat": False,
        "hf": "emozilla/pg19", "config": None, "split": "train",
        "text_field": "text", "meta_filter": None, "min_long": 0,
    },
    {
        "name": "sci_long", "weight": 0.125, "role": "long", "repo_concat": False,
        "hf": "DKYoon/SlimPajama-6B", "config": None, "split": "train",
        "text_field": "text", "meta_filter": "arxiv", "min_long": 16384,
    },
    {
        "name": "short_mix", "weight": 0.40, "role": "short", "repo_concat": False,
        "hf": "DKYoon/SlimPajama-6B", "config": None, "split": "train",
        "text_field": "text", "meta_filter": None, "min_long": 0,
    },
]


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for jsonl shards")
    ap.add_argument("--tokenizer", required=True,
                    help="model dir with the Gemma-4 tokenizer (vocab 262144)")
    ap.add_argument("--tokens", type=int, default=500_000_000,
                    help="total packed-token budget (Fu et al.: 0.5-5B; 0.5B default)")
    ap.add_argument("--pack-len", type=int, default=65536,
                    help="tokens per packed sequence == trainer --pack-len")
    ap.add_argument("--min-long-tokens", type=int, default=0,
                    help="long-role doc min length (0 => use --pack-len, ProLong-faithful)")
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                    help="rotate to a new shard after this many tokens")
    ap.add_argument("--shuffle-buffer", type=int, default=10_000,
                    help="per-source seeded streaming shuffle buffer (0=off)")
    ap.add_argument("--max-skip", type=int, default=20_000,
                    help="consecutive too-short pulls before a long source is "
                         "declared exhausted (prevents spinning)")
    ap.add_argument("--repo-buffer", type=int, default=256,
                    help="max open repos held for repo-level code concatenation")
    ap.add_argument("--sources-json", default=None,
                    help="override DEFAULT_SOURCES with a JSON file (same schema)")
    ap.add_argument("--allow-missing-source", action="store_true",
                    help="warn-and-continue if a source fails to load (default: FATAL)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="pull a few docs per source to validate connectivity + "
                         "tokenization + length routing; write nothing")
    return ap.parse_args()


def load_sources(args):
    if args.sources_json:
        with open(args.sources_json) as f:
            srcs = json.load(f)
    else:
        srcs = [dict(s) for s in DEFAULT_SOURCES]
    tot = sum(s["weight"] for s in srcs)
    if abs(tot - 1.0) > 1e-6:
        raise SystemExit(f"FATAL: source weights sum to {tot}, must be 1.0")
    return srcs


def _iter_parquet_files(files):
    """Stream rows from a list of Parquet files ONE FILE AT A TIME. Reading each
    file as its own dataset avoids cross-file arrow schema unification — bigcode
    starcoderdata shards have heterogeneous columns (some carry a `license` column,
    some don't), and a single load_dataset(data_files=[all]) crashes mid-stream with
    'column names don't match' (observed after ~350M tokens on the full 98e pack,
    2026-06-07). Per-file is self-consistent. A bad/transient file is skipped, not
    fatal — a multi-hour pack must not die on one shard."""
    from datasets import load_dataset
    for f in files:
        try:
            ds = load_dataset("parquet", data_files=[f], split="train", streaming=True)
            yield from ds
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: parquet file {f} failed ({type(e).__name__}: {e}) — skipping",
                  file=sys.stderr)
            continue


def _load_stream(src, seed, allow_missing):
    """Open one streaming HF dataset. A `parquet_glob` source is read via direct
    Parquet (no loading script), per-file (see _iter_parquet_files) with a
    seed-shuffled FILE list so lang-sharded corpora interleave; otherwise plain
    load_dataset on a Parquet-native id."""
    from datasets import load_dataset
    try:
        if src.get("parquet_glob"):
            import random as _r
            from huggingface_hub import HfFileSystem
            files = HfFileSystem().glob(src["parquet_glob"])
            if not files:
                raise RuntimeError(f"no parquet files matched {src['parquet_glob']!r}")
            files = ["hf://" + f for f in files]
            _r.Random(seed).shuffle(files)   # deterministic lang/shard interleave
            return _iter_parquet_files(files)
        return load_dataset(src["hf"], src.get("config"),
                            split=src.get("split", "train"), streaming=True)
    except Exception as e:  # noqa: BLE001
        ident = src.get("hf") or src.get("parquet_glob")
        msg = f"source {src['name']} ({ident}) failed to load: {type(e).__name__}: {e}"
        if allow_missing:
            print(f"  WARN: {msg} — skipping (--allow-missing-source)", file=sys.stderr)
            return None
        raise SystemExit(f"FATAL: {msg}\n(pass --allow-missing-source to continue without it)")


def doc_stream(src, seed, shuffle_buffer, allow_missing):
    """Yield raw text documents from one streaming source. With meta_filter, only
    rows whose meta.redpajama_set_name matches are kept (SlimPajama routing). With
    repo_concat, files are grouped by repo_field and a repo is flushed as one
    `# path`-headed document once it reaches ~min_long tokens (or on buffer
    overflow), approximating repo-level long code."""
    ds = _load_stream(src, seed, allow_missing)
    if ds is None:
        return
    if shuffle_buffer and src.get("row_shuffle", True):
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    tf = src["text_field"]
    mfilter = src.get("meta_filter")

    def meta_ok(row):
        if not mfilter:
            return True
        m = row.get("meta")
        name = (m.get("redpajama_set_name") or "") if isinstance(m, dict) else ""
        return mfilter.lower() in name.lower()

    if not src.get("repo_concat"):
        for row in ds:
            if not meta_ok(row):
                continue
            t = row.get(tf)
            if t:
                yield t
        return

    # repo-level concatenation: accumulate files per repo, flush as one
    # `# <path>`-headed document once large or the open-repo table overflows.
    repo_field = src.get("repo_field", "repo_name")
    path_field = src.get("path_field", "path")
    repos = {}            # repo -> list[(path, code)]
    sizes = {}            # repo -> char count
    order = []            # eviction order
    repo_cap = src.get("_repo_buffer", 256)
    flush_chars = src.get("_flush_chars", 400_000)  # ~>= min_long tokens of code

    def render(name):
        return "\n\n".join(f"# {p}\n{c}" for p, c in repos[name])

    for row in ds:
        code = row.get(tf)
        if not code:
            continue
        name = row.get(repo_field) or row.get(path_field) or "_"
        if name not in sizes:
            repos[name] = []
            sizes[name] = 0
            order.append(name)
        repos[name].append((row.get(path_field, ""), code))
        sizes[name] += len(code)
        if sizes[name] >= flush_chars:
            yield render(name)
            order.remove(name)
            del repos[name], sizes[name]
        elif len(repos) > repo_cap:
            ev = order.pop(0)
            yield render(ev)
            del repos[ev], sizes[ev]
    for name in order:           # drain
        yield render(name)


def encode_doc(tok, text, bos, eos):
    ids = tok.encode(text, add_special_tokens=False)
    out = []
    if bos is not None:
        out.append(bos)
    out.extend(ids)
    if eos is not None:
        out.append(eos)
    return out


def main():
    a = parse()
    min_long = a.min_long_tokens or a.pack_len
    srcs = load_sources(a)

    print("=== pack_longctx_corpus (code-leaned YaRN mix) ===")
    print(f"  out           : {a.out}")
    print(f"  tokenizer     : {a.tokenizer}")
    print(f"  total tokens  : {a.tokens:,}")
    print(f"  pack_len      : {a.pack_len:,}")
    print(f"  min_long      : {min_long:,}  (long-role doc floor)")
    print(f"  shard_tokens  : {a.shard_tokens:,}  (~{max(1, a.tokens//a.shard_tokens)} shards)")
    print(f"  seed          : {a.seed}")
    print("  sources:")
    for s in srcs:
        eml = s.get("min_long") or min_long
        print(f"    {s['name']:10s} w={s['weight']:.3f} role={s['role']:5s} "
              f"min_long={eml:>6} {s['hf']}{('/'+s['config']) if s.get('config') else ''} "
              f"{'[repo-concat]' if s.get('repo_concat') else ''}"
              f"{(' meta=' + s['meta_filter']) if s.get('meta_filter') else ''}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    print(f"  bos={bos} eos={eos} vocab={tok.vocab_size}")

    # Build per-source generators + budgets.
    for s in srcs:
        s["_repo_buffer"] = a.repo_buffer
        s["_min_long"] = s.get("min_long") or min_long
        s["_flush_chars"] = max(200_000, s["_min_long"] * 6)  # ~6 chars/token
        s["gen"] = doc_stream(s, a.seed, a.shuffle_buffer, a.allow_missing_source)
        s["target"] = a.tokens * s["weight"]
        s["produced"] = 0
        s["docs"] = 0
        s["skips"] = 0
        s["done"] = False

    if a.dry_run:
        print("\n[dry-run] pulling up to 5 docs per source to validate...")
        for s in srcs:
            n = 0
            lens = []
            t0 = time.time()
            for text in s["gen"]:
                ids = encode_doc(tok, text, bos, eos)
                lens.append(len(ids))
                n += 1
                if n >= 5:
                    break
            if not lens:
                print(f"  {s['name']:10s} NO DOCS PULLED (gated/empty?) "
                      f"{'[ok: --allow-missing]' if a.allow_missing_source else 'FATAL'}")
                if not a.allow_missing_source:
                    return 2
                continue
            eml = s["_min_long"]
            qual = sum(1 for x in lens if x >= eml)
            print(f"  {s['name']:10s} pulled={n} tok_lens={lens} "
                  f">=min_long({eml}):{qual}/{n} ({time.time()-t0:.1f}s)")
            if s["role"] == "long" and qual == 0:
                print(f"    WARN: 0/{n} sampled {s['name']} docs reach min_long="
                      f"{eml}; long fraction may starve. Lower this source's min_long "
                      f"or raise --repo-buffer for code.")
        print("[dry-run] connectivity + tokenization OK; no shards written.")
        return 0

    os.makedirs(a.out, exist_ok=True)
    buf = []
    emitted = 0
    shard_idx = 0
    shard_emitted = 0
    shard_path = os.path.join(a.out, f"shard-{shard_idx:04d}.jsonl")
    shard_f = open(shard_path, "w")
    manifest = {"args": vars(a), "min_long": min_long, "sources": [
        {k: s.get(k) for k in ("name", "weight", "role", "hf", "config",
                               "meta_filter", "min_long", "repo_concat")} for s in srcs]}
    t0 = time.time()

    def pick():
        # greedy: the not-done source furthest below its target share.
        best, best_ratio = None, None
        for s in srcs:
            if s["done"]:
                continue
            r = s["produced"] / s["target"] if s["target"] > 0 else 1e9
            if best is None or r < best_ratio:
                best, best_ratio = s, r
        return best

    while emitted < a.tokens:
        s = pick()
        if s is None:
            print("WARN: all sources exhausted before token budget met", file=sys.stderr)
            break
        try:
            text = next(s["gen"])
        except StopIteration:
            s["done"] = True
            print(f"  [{s['name']}] stream exhausted at {s['produced']:,}/"
                  f"{int(s['target']):,} tokens", flush=True)
            continue
        ids = encode_doc(tok, text, bos, eos)
        if s["role"] == "long" and len(ids) < s["_min_long"]:
            s["skips"] += 1
            if s["skips"] >= a.max_skip:
                s["done"] = True
                print(f"  [{s['name']}] declared exhausted of long docs after "
                      f"{a.max_skip} skips ({s['produced']:,}/{int(s['target']):,} tok)",
                      flush=True)
            continue
        s["skips"] = 0
        s["produced"] += len(ids)
        s["docs"] += 1
        buf.extend(ids)
        while len(buf) >= a.pack_len:
            chunk = buf[:a.pack_len]
            del buf[:a.pack_len]
            shard_f.write(json.dumps({"input_ids": chunk}) + "\n")
            emitted += a.pack_len
            shard_emitted += a.pack_len
            if shard_emitted >= a.shard_tokens:
                shard_f.close()
                print(f"  wrote {shard_path} ({shard_emitted:,} tok, "
                      f"total {emitted:,}/{a.tokens:,}, {time.time()-t0:.0f}s)", flush=True)
                shard_idx += 1
                shard_emitted = 0
                shard_path = os.path.join(a.out, f"shard-{shard_idx:04d}.jsonl")
                shard_f = open(shard_path, "w")
    shard_f.close()
    # drop an empty trailing shard left by an exact-boundary rotation
    if shard_emitted == 0 and shard_idx > 0 and os.path.getsize(shard_path) == 0:
        os.remove(shard_path)
        shard_idx -= 1

    manifest["produced"] = {s["name"]: {"tokens": s["produced"], "docs": s["docs"],
                                        "target": int(s["target"])} for s in srcs}
    manifest["emitted_tokens"] = emitted
    manifest["shards"] = shard_idx + 1
    with open(os.path.join(a.out, "pack_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== packing complete: {emitted:,} tokens in {shard_idx + 1} shards "
          f"({time.time()-t0:.0f}s) ===")
    print("  per-source (produced/target tokens, docs):")
    for s in srcs:
        frac = s["produced"] / emitted if emitted else 0
        print(f"    {s['name']:10s} {s['produced']:>13,}/{int(s['target']):>13,} "
              f"({frac:5.1%}) docs={s['docs']:,}")
    return 0


if __name__ == "__main__":
    _rc = main()
    # Streaming HF datasets leave aiohttp/pyarrow background threads that crash
    # at interpreter finalize (PyGILState_Release / Bad file descriptor). Shards
    # + manifest are already flushed and closed by here, so skip finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc or 0)
