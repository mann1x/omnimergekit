# Example chat templates

These are reference `.jinja` chat templates for the 4-cell Gemma-4 comparison in
`profiles/gemma4.example.yaml`. **Bring your own** for your model — these are here
so the example runs out of the box and so you can diff a candidate fix against the
upstream template.

| File | What it is |
|---|---|
| `google_main.jinja` | The current upstream Gemma-4 chat template (same as what's embedded in an official GGUF). Provided for reference/diffing; the `embedded` cell uses the GGUF's own copy. |
| `v7_reinject_disabled.jinja` | A fix that **disables re-injection of prior-turn reasoning** into the next-turn prompt — the mechanism behind the deep agentic rumination loop. |
| `google_pr35.jinja` | Google's PR #35 candidate template, default mode. |
| `google_pr35_ptfalse.jinja` | Google's PR #35 candidate template, pass-thinking=false mode. |

To compare a template, add it to `model.chat_template` in the profile (as a path
or `{name, path}`), or pass `--template path/to.jinja` on the CLI. The `embedded`
cell (path `null`) always tests whatever template is baked into the GGUF.

The harness applies a template via `llama-server --chat-template-file`, so any
valid llama.cpp jinja chat template works.
