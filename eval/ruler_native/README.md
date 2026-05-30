# omk ruler_native backend

Thin omk-shaped backend over [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
(arXiv:[2404.06654](https://arxiv.org/abs/2404.06654)), the long-context
synthetic benchmark suite.

## Design — runtime-clone + inline scorer

We do **not** vendor RULER. The upstream repo is cloned at runtime under
`/workspace/RULER` (pods) or `/shared/dev/RULER` (solidpc), same pattern as
`/opt/llama.cpp`. The runner subprocess-calls only one piece of upstream code:
`scripts/data/prepare.py`, which generates the per-task `validation.jsonl`
staged inputs.

We **inline** the upstream scorer (`scripts/eval/synthetic/constants.py:25`,
`string_match_all`) into [`ruler_helpers.py`](ruler_helpers.py) instead of
subprocessing into `scripts/eval/evaluate.py`. The reason is operational, not
stylistic: `evaluate.py` imports
`from nemo.collections.asr.parts.utils.manifest_utils import {read,write}_manifest`,
which forces a `pip install nemo-toolkit[all]` cascade. That cascade silently
downgrades the canonical omk env pins (live verified 2026-05-28 on
linode-blackswan-2):

| pin | canonical | post-nemo-cascade |
|---|---|---|
| torch | 2.10.0+cu128 | 2.12.0+cu130 (breaks vLLM 0.20.2 wheel) |
| transformers | 5.5.0 | 4.57.6 (removes Gemma4Config / Gemma4ForCausalLM) |
| safetensors | 0.7.0 | 0.8.0-rc.0 |
| nvidia-modelopt | 0.43.0 | 0.37.0 (breaks NVFP4A16 canonical pin) |

The scorer itself is 3 lines of stdlib substring matching (case-insensitive
`r.lower() in pred.lower()` averaged per-needle, then averaged per-sample × 100).
The inline port is mathematically byte-identical to upstream output — see the
attribution + verbatim copy in `ruler_helpers.string_match_all`.

If you ever need to validate the inline scorer against upstream, install
nemo-toolkit[all] into a **dedicated** conda env (e.g. `ruler_scorer`) and
diff `evaluate.py`'s `summary.csv` against the runner's `ruler_result.json`
on the same `validation.jsonl` → `samples.jsonl` pair. Do **not** install
nemo-toolkit[all] into the canonical omk env.

## File layout

```
eval/ruler_native/
├── __init__.py             # package marker + the inline-scorer RCA
├── ruler_helpers.py        # inlined string_match_all, RULER root discovery,
│                           # prepare.py subprocess wrapper, validation.jsonl
│                           # parser, task→scorer dispatch (13 RULER tasks)
├── ruler_runner.py         # 3-phase runner (prepare → infer → score)
└── README.md               # this file

eval/templates/
├── ruler_native_smoke.yaml         # vt × ctx=4096 × 5 samples (~2 min)
├── ruler_native_vt_32k.yaml        # reference anchor vs lm-eval ruler_vt
├── ruler_native_vt_256k.yaml       # YaRN β-boundary probe at native
├── ruler_native_vt_512k.yaml       # YaRN factor=2.0 target
├── ruler_native_mk1_256k.yaml      # Multi-Key NIAH at native
└── ruler_native_mk1_512k.yaml      # Multi-Key NIAH at 512k
```

## Usage

```bash
# Smoke (assumes a 128e Q6_K llama-server already running on :8099)
omk_eval --template ruler_native_smoke \
  --model gemma-4-26B-A4B-it-Q6_K \
  --base-url http://localhost:8099 \
  --tokenizer google/gemma-4-26B-A4B-it

# Reference anchor — compare to lm-eval's bundled ruler_vt at 32k
omk_eval --template ruler_native_vt_32k --model ... --tokenizer ...

# YaRN-extended validation triad (run all five on the same served model)
for t in ruler_native_vt_256k ruler_native_vt_512k \
         ruler_native_mk1_256k ruler_native_mk1_512k; do
  omk_eval --template "$t" --model ... --tokenizer ...
done
```

## Discovery + nltk

`ruler_helpers.locate_ruler_root()` checks (first hit wins):

1. `$RULER_ROOT`
2. `/workspace/RULER` (pod canonical)
3. `/shared/dev/RULER` (solidpc canonical)

`scripts/data/prepare.py` needs the nltk `punkt` + `punkt_tab` corpora.
`ensure_nltk_data()` downloads them on first use (idempotent — silent no-op
if already cached). The corpora are tiny (~50 MB combined).

## Disk budget

Each `(task, ctx_tokens)` cell writes ~520 KB/sample to
`<stage_dir>/data/<task>/validation.jsonl`. With the default `num_samples=50`:

- ruler_native_smoke (4k × 5):  ~3 MB
- ruler_native_vt_32k (32k × 50):  ~26 MB
- ruler_native_vt_256k (256k × 50):  ~26 MB (input grows linearly but the
  jsonl is ~one line per sample regardless of ctx — the input *string* is
  large but each row is one record).
- ruler_native_vt_512k (512k × 50):  ~52 MB

Full 6-template run at 256k+512k tiers writes ~150 MB on disk plus the
per-cell sqlite resume DBs (≤10 MB each). Not in the "TB territory" range
some other long-ctx benches hit.

## License

- RULER source: Apache-2.0 (NVIDIA Corporation). The scorer port in
  `ruler_helpers.string_match_all` carries the verbatim Apache-2.0
  attribution in its header. We do not redistribute RULER source — the
  runtime clone fetches it directly from upstream.
- Paul Graham essays (haystack source for NIAH): publicly available essays,
  RULER's repo packages them under the NVIDIA repository's overall
  Apache-2.0 umbrella.
- SQuAD / HotpotQA (for `qa_1` / `qa_2` tasks): CC-BY-SA-4.0. We never
  re-publish these datasets — `prepare.py` reads them from upstream's
  cached snapshot.

Score artifacts (the `.json` and `.samples.jsonl` outputs) are derivative
output, freely publishable per standard benchmark conventions.

## When to use which template

| Goal | Template | Notes |
|---|---|---|
| Plumbing sanity | `ruler_native_smoke` | 2-min smoke; VT @ 4k × 5 |
| Calibrate runner | `ruler_native_vt_32k` | Compare to lm-eval ruler_vt within ±5pp |
| YaRN β-boundary | `ruler_native_vt_256k` | At native max ctx |
| YaRN extension | `ruler_native_vt_512k` | At factor=2.0 target |
| High-freq probe | `ruler_native_mk1_256k` | Stresses pass-through rope dims |
| YaRN extension HF | `ruler_native_mk1_512k` | Same at 512k |

The remaining 8 RULER tasks (cwe, fwe, qa_1, qa_2, niah_single_*,
niah_multikey_2/3, niah_multiquery, niah_multivalue) are runner-supported via
the `selection.ruler_task` field — drop a new template in
`eval/templates/ruler_native_<task>_<ctx>.yaml` to enable.
