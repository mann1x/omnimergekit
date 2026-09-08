# llama.cpp: a PEG parse failure must not DISCARD a served generation

`common/chat.cpp: common_chat_peg_parse()` threw `std::runtime_error` on a **full**
parse failure. `llama-server` turned that into **HTTP 500**, lm-eval stored an *empty*
completion, and a generation the model had already produced correctly scored as a fail —
or, when retries ran out, killed the whole benchmark.

The `is_partial` branch immediately above already degraded gracefully. That asymmetry was
the bug.

## Observed triggers (Ornith-1.5 184e P6, ifeval_100, 2026-09-08)

15 parse failures, only **7** of which were generation-cap truncations — so truncation was
never the whole story:

* **11 x a bare `<think>`** the model opens and never closes
* **4 x invalid UTF-8** (U+FFFD) from a split multi-byte emoji token — these were
  *complete, correct answers* thrown away over one broken byte

Cost: `ifeval_100` died at **99/100** with `rc=1`, no `samples.jsonl`, after 1h03m — the 99
good answers were lost with it. Earlier, the p24 arm's `lcb_v6_77q` cell recorded **20/77
empties** from the same cause, which is why its 0.5325 is void rather than a low score.

## The fix

On full-parse failure, degrade to content (mirroring the no-parser default
`p.content(p.rest())`). If `</think>` is present, honour the split so reasoning lands in
`reasoning_content` instead of polluting the answer. The throw is preserved behind
`LLAMA_CHAT_PEG_STRICT=1`; `tests/test-chat.cpp` was updated to opt into it rather than
have its assertion deleted.

## Gate — run BOTH directions after any rebuild

The code lives in **`libllama-common.so`**, NOT in the thin `llama-server` binary; a string
probe on the binary returns a confident false negative.

    strict=0 trigger=1 -> returned content_bytes=60
    strict=0 trigger=2 -> returned content_bytes=32
    strict=1 trigger=1 -> THREW
    strict=1 trigger=2 -> THREW
    PEGFIX_GATE_PASS

Source: `pegfix_gate.cpp` (see this dir). A one-sided pass proves nothing — a lenient-only
run cannot tell the fix from a deleted assertion.

## Two source variants — pick the right patch

llama.cpp has changed the failure message at this site. Both carry the same fix:

| patch | guards the throw | seen on |
|---|---|---|
| `llamacpp_peg_parse_degrade_not_throw.patch` | `"...does not match the expected..."` | pod, solidpc (`/opt/llama.cpp`), llamafile vendor tree |
| `llamacpp_peg_older_variant.patch` | `"Failed to parse input at pos ..."` | bs2 (rev `d6be315`), the 0103 spike |

The older variant uses `fprintf(stderr, ...)` because `LOG_WRN` is not in scope there.

**Do not scp a whole `chat.cpp` between trees** — the variants differ well beyond this
hunk (the newer one includes `json.h`, which the older tree does not have).

## Gate before trusting either

`pegfix_gate.cpp` must pass BOTH directions — a one-sided check cannot see a dead guard:

```
strict=0 trigger=1 -> returned content_bytes=60
strict=0 trigger=2 -> returned content_bytes=32
strict=1 trigger=1 -> THREW
strict=1 trigger=2 -> THREW
PEGFIX_GATE_PASS
```

A lenient-only run passed on bs2 while the guard was written `== 1` (int) instead of
`== '1'` (char) — the fallback fired unconditionally and the escape hatch was dead.
See buglog `bug-684`.
