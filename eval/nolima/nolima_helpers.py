"""NoLiMa data loader, BookHaystack, prompt assembly, and scorer.

NoLiMa is the lexical-overlap-stripped variant of needle-in-a-haystack
(arXiv:2502.05167, ICML 2025). Needles and questions share minimal vocabulary,
so the model must rely on latent associations to locate the answer — useful
specifically for probing whether a long-context extension (YaRN, LongRoPE,
SelfExtend) actually preserved comprehension or just preserved keyword reach.

Upstream data: https://huggingface.co/datasets/amodaresi/NoLiMa
License: Adobe Research License (non-commercial research only). This module
does NOT vendor any NoLiMa data into the omk repo; needle JSONs and haystack
TXTs are pulled at runtime via `huggingface_hub.hf_hub_download` and cached to
the user's HF_DATASETS_CACHE (default `~/.cache/huggingface/`). Score artifacts
are derivative output, freely publishable per standard benchmark conventions.

Schema (one row of `needlesets/needle_set.json`):
    {
      "id": "0401",
      "reasoning_type": "world_knowledge",
      "system_prompt": "",                    # "" → use default
      "task_template": "...{haystack}...",    # body before the question
      "needle": "Actually, {CHAR} lives next to {1}.",
      "questions": {"onehop": "Which character has been to {2}?",
                    "twohop": "Which character has been to {3}?"},
      "character_set": ["Yuki", "Stuart", ...],
      "tests": {"T17_C02": {"input_args": ["...", "Helsinki", "Uusimaa"]},
                ...},
    }

Per (row, test, hop_mode, depth%, shift_seed) we:
  1. Sample a {CHAR} from character_set (deterministic, seeded by row.id+test_id+shift)
  2. Substitute {CHAR} + {1}/{2}/{3} (from test.input_args) into needle + question
  3. Embed substituted needle into the tokenized haystack at depth%
  4. Build prompt = row.task_template.format(haystack=...) + "\n" + question
  5. Submit via /v1/chat/completions
  6. Score: gold answer = [the sampled {CHAR}]; default metric "contains".
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_SYSTEM_PROMPT = (
    "You will answer a question based on the following book snippet:"
)

# Upstream's `evaluation/async_evaluate.py:_evaluate_response` — verbatim
# metric semantics. No normalization, no regex, no stemming. Case-sensitive.
VALID_METRICS = ("EM", "contains", "lastline_EM", "lastline_contains")


# ── HF dataset access ───────────────────────────────────────────────────────

def hf_pull(filename: str) -> Path:
    """Download a single file from `amodaresi/NoLiMa` and return its local path.

    Honors HF_HUB_ENABLE_HF_TRANSFER and the standard HF cache layout.
    """
    from huggingface_hub import hf_hub_download  # lazy import: optional dep
    p = hf_hub_download(repo_id="amodaresi/NoLiMa", repo_type="dataset",
                        filename=filename)
    return Path(p)


def load_needle_set(name: str = "needle_set") -> list[dict]:
    """Load one of the 6 needle variants. `name` is the JSON stem (no path,
    no extension) — one of:
        needle_set, needle_set_hard, needle_set_MC, needle_set_ONLYDirect,
        needle_set_w_CoT, needle_set_w_Distractor
    """
    fname = f"needlesets/{name}.json"
    p = hf_pull(fname)
    with p.open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{fname}: expected a list, got {type(rows).__name__}")
    return rows


def load_haystack_text(tier: str = "rand_shuffle", book_idx: int = 1) -> str:
    """Load one haystack TXT file (UTF-8). `tier` ∈ {rand_shuffle,
    rand_shuffle_long}; `book_idx` ∈ {1..5}.

    `rand_shuffle_long` is the >128k variant; use it when ctx_tokens > 128_000.
    """
    if tier not in ("rand_shuffle", "rand_shuffle_long"):
        raise ValueError(f"unknown haystack tier: {tier!r}")
    if book_idx not in (1, 2, 3, 4, 5):
        raise ValueError(f"book_idx must be 1..5, got {book_idx}")
    p = hf_pull(f"haystack/{tier}/rand_book_{book_idx}.txt")
    return p.read_text(encoding="utf-8")


# ── Substitution: {CHAR} + {1}/{2}/{3} ──────────────────────────────────────

def _format_needle(template: str, char_name: str, args: list[str]) -> str:
    """Substitute {CHAR} and positional {1}/{2}/.../{N} placeholders.

    args[0] → {1}, args[1] → {2}, etc. (1-indexed in the templates).
    """
    out = template.replace("{CHAR}", char_name)
    for i, val in enumerate(args, start=1):
        out = out.replace("{" + str(i) + "}", val)
    return out


def sample_char(character_set: list[str], seed: int) -> str:
    rng = random.Random(seed)
    return rng.choice(character_set)


# ── BookHaystack — embed a needle in tokenized text at a target depth% ─────

@dataclass
class _PlacementResult:
    text: str
    needle_token_idx: int   # position of the needle in the final token stream
    haystack_token_count: int


class BookHaystack:
    """Tokenize a haystack TXT once, then slice + embed needles at arbitrary
    depth percentages inside the tokenized window.

    Reimplemented from the upstream NoLiMa `data/book_haystack.py` spec — we
    don't copy any code. Behavior:
      1. Encode the full haystack with the model's tokenizer.
      2. Record every token index whose decoded form starts a new paragraph
         (i.e. contains '\\n'). These are the "snap points" for needle insertion
         so the needle never lands mid-word.
      3. For a given (context_length, depth_pct, shift_seed):
            a. Pick a starting offset into the haystack so the final ctx_window
               is `context_length` tokens long, deterministically rotated by
               shift_seed (allows multi-seed evaluation without re-tokenizing).
            b. Find the snap-point inside that window closest to
               `start + depth_pct * context_length`.
            c. Split the window at the snap → (pre, post) token slices.
            d. Decode pre + " " + needle + "\\n" + post.
            e. Re-encode briefly to record the needle's final token position.

    The whole class is tokenizer-agnostic: pass a callable `encode(str) -> list[int]`
    and a callable `decode(list[int]) -> str`. We don't hardcode HF AutoTokenizer
    so this stays unit-testable.
    """

    def __init__(self, full_text: str, encode, decode,
                 *, line_split: str = "\n"):
        self.full_text = full_text
        self.encode = encode
        self.decode = decode
        # Tokenize once; cache the ids.
        self._tokens: list[int] = list(encode(full_text))
        # Snap points: token indices whose decoded prefix-up-to-this-token
        # ends in a newline. We approximate by decoding successive 16-token
        # windows and finding newline positions — exact enough for needle
        # placement and ~30× faster than per-token decode on a 1M-token book.
        self._snap_points: list[int] = self._compute_snap_points(line_split)

    def _compute_snap_points(self, line_split: str) -> list[int]:
        toks = self._tokens
        out: list[int] = [0]
        step = 16
        for i in range(0, len(toks) - step, step):
            chunk = self.decode(toks[i:i + step])
            base = 0
            while True:
                j = chunk.find(line_split, base)
                if j < 0:
                    break
                # Snap to the START of the line that FOLLOWS the newline.
                # Approximate by decoding the prefix up to char j+1 and
                # counting tokens — but that's quadratic. Cheaper: assume
                # one token ≈ one char_count // 4. We only need a coarse
                # snap (the needle is human-readable text either way).
                approx_tok_in_chunk = max(1, (j + 1) * step // max(1, len(chunk)))
                out.append(i + approx_tok_in_chunk)
                base = j + 1
        # Dedupe + sort.
        out = sorted(set(out))
        return out

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def generate(self, *, context_length: int, depth_pct: float,
                 needle: str, shift_seed: int = 0,
                 distractor: str | None = None,
                 distractor_buffer_pct: float = 0.25) -> _PlacementResult:
        """Embed `needle` at `depth_pct` inside a `context_length`-token window.

        depth_pct ∈ [0.0, 1.0]. shift_seed rotates the window's start so
        running the same (ctx, depth) with multiple seeds samples different
        passages.
        """
        if not (0.0 <= depth_pct <= 1.0):
            raise ValueError(f"depth_pct must be in [0, 1], got {depth_pct}")
        if context_length > self.token_count:
            raise ValueError(
                f"ctx_length {context_length} > haystack tokens "
                f"{self.token_count} — choose a larger haystack tier"
            )
        # Window start: rotated by shift_seed. The +1 prevents start==end.
        rng = random.Random(shift_seed)
        max_start = max(1, self.token_count - context_length - 1)
        start = rng.randrange(0, max_start) if shift_seed else 0

        target_idx = start + int(depth_pct * context_length)
        snap = self._nearest_snap(target_idx, start, start + context_length)

        pre = self._tokens[start:snap]
        post = self._tokens[snap:start + context_length]

        pre_text = self.decode(pre)
        post_text = self.decode(post)

        if distractor:
            # Place distractor at a depth at least `distractor_buffer_pct`
            # away from the needle.
            d_min, d_max = max(0.05, depth_pct - distractor_buffer_pct), \
                           min(0.95, depth_pct + distractor_buffer_pct)
            # Choose a depth outside [d_min, d_max].
            if d_min > 0.05:
                d_depth = rng.uniform(0.0, d_min)
            else:
                d_depth = rng.uniform(d_max, 1.0)
            # Inject distractor into pre_text or post_text accordingly.
            if d_depth < depth_pct:
                pre_text = self._inject_text(pre_text, distractor, d_depth / depth_pct
                                             if depth_pct else 0.5)
            else:
                rel = (d_depth - depth_pct) / max(1e-6, 1.0 - depth_pct)
                post_text = self._inject_text(post_text, distractor, rel)

        composite = f"{pre_text} {needle}\n{post_text}"
        # Token position of the needle is just snap (relative to window start
        # is snap-start; absolute is snap). Callers can use either.
        return _PlacementResult(
            text=composite,
            needle_token_idx=snap - start,
            haystack_token_count=context_length,
        )

    def _nearest_snap(self, target: int, lo: int, hi: int) -> int:
        # Find the snap-point in [lo, hi] closest to `target`. Binary search.
        snaps = self._snap_points
        # Filter to range
        candidates = [s for s in snaps if lo < s < hi]
        if not candidates:
            return target
        # Closest to target
        best = min(candidates, key=lambda s: abs(s - target))
        return best

    @staticmethod
    def _inject_text(passage: str, sentence: str, frac: float) -> str:
        """Insert `sentence` into `passage` at the newline closest to
        `frac * len(passage)`. Falls back to mid-passage if no newline found.
        """
        target = int(frac * len(passage))
        nl = passage.find("\n", target)
        if nl < 0:
            nl = passage.rfind("\n", 0, target)
        if nl < 0:
            nl = target
        return passage[:nl] + "\n" + sentence + passage[nl:]


# ── Scorer ──────────────────────────────────────────────────────────────────

def score_response(response: str, gold_answers: list[str],
                   metric: str = "contains") -> bool:
    """Replicate upstream `_evaluate_response` (4 metrics, case-sensitive)."""
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS!r}, got {metric!r}")
    if metric == "EM":
        return response.strip() in gold_answers
    if metric == "contains":
        return any(g in response for g in gold_answers)
    if metric == "lastline_EM":
        return response.strip().split("\n")[-1] in gold_answers
    if metric == "lastline_contains":  # noqa: RET503 — explicit final branch
        last = response.strip().split("\n")[-1]
        return any(g in last for g in gold_answers)
    return False


# ── Test-grid expansion ─────────────────────────────────────────────────────

@dataclass
class NolimaTest:
    """One concrete (needle row, test, hop_mode, depth, shift) combination
    ready to be embedded + queried."""
    needle_id: str           # row["id"]
    test_id: str             # key in row["tests"]
    hop_mode: str            # "onehop" | "twohop"
    depth_pct: float
    shift_seed: int
    ctx_tokens: int
    # Substituted strings (after CHAR + {1}/{2}/{3} replacement):
    char_name: str
    needle_text: str
    question_text: str
    system_prompt: str
    task_template: str       # contains the literal "{haystack}" placeholder
    gold_answers: list[str]  # = [char_name] in standard NoLiMa


def expand_tests(rows: list[dict], *, hop_mode: str = "onehop",
                 tests_per_row: int | None = 1,
                 depth_intervals: int = 26,
                 ctx_tokens: int,
                 shift_seeds: Iterable[int] = (0,),
                 row_limit: int | None = None) -> list[NolimaTest]:
    """Materialize the test grid for one ctx tier.

    Defaults to one test per row (the first one) at the standard hop_mode
    "onehop" with 26 depth intervals — matches upstream's small-ctx
    configuration. Pass `tests_per_row=None` to use every test in the row.
    """
    import numpy as np  # lazy import; numpy is already an omk dep
    depths = np.linspace(0.0, 1.0, depth_intervals).tolist()

    out: list[NolimaTest] = []
    for row in rows[:row_limit] if row_limit else rows:
        if hop_mode not in row.get("questions", {}):
            continue
        char_pool = row.get("character_set") or []
        if not char_pool:
            continue
        test_keys = list(row.get("tests", {}).keys())
        if tests_per_row is not None:
            test_keys = test_keys[:tests_per_row]
        for tkey in test_keys:
            args = row["tests"][tkey].get("input_args", [])
            for shift in shift_seeds:
                # Deterministic CHAR sampling per (row, test, shift).
                char = sample_char(char_pool, hash((row["id"], tkey, shift)) & 0x7FFFFFFF)
                needle = _format_needle(row["needle"], char, args)
                question = _format_needle(row["questions"][hop_mode], char, args)
                for d in depths:
                    out.append(NolimaTest(
                        needle_id=row["id"],
                        test_id=tkey,
                        hop_mode=hop_mode,
                        depth_pct=float(d),
                        shift_seed=int(shift),
                        ctx_tokens=int(ctx_tokens),
                        char_name=char,
                        needle_text=needle,
                        question_text=question,
                        system_prompt=row.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
                        task_template=row["task_template"],
                        gold_answers=[char],
                    ))
    return out


def make_cache_key(t: NolimaTest, *, needle_set: str) -> str:
    """Sqlite cache key. Must be deterministic across resumes."""
    return (
        f"{needle_set}::{t.needle_id}::{t.test_id}::{t.hop_mode}::"
        f"{t.ctx_tokens}::{t.depth_pct:.4f}::{t.shift_seed}"
    )


# ── Tokenizer helpers ───────────────────────────────────────────────────────

@dataclass
class TokenizerWrapper:
    """Thin wrapper around HF AutoTokenizer that exposes (encode, decode) as
    plain functions — what BookHaystack expects.
    """
    tokenizer_id: str
    _tok: object = field(default=None, init=False)

    def _ensure(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                self.tokenizer_id, trust_remote_code=True, use_fast=True,
            )

    def encode(self, text: str) -> list[int]:
        self._ensure()
        return self._tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        self._ensure()
        return self._tok.decode(ids, skip_special_tokens=True)
