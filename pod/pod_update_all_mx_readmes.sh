#!/bin/bash
# Update all 5 M1..M5 HF repo READMEs with the unified Phase 1+2 comparison
# table. Each repo's row is bolded so the variant's relative position is
# obvious at first glance.
set -uo pipefail
LOG=/workspace/logs/update_mx_readmes.log
echo "=== mass-update Mx READMEs $(date -Iseconds) ===" | tee -a "$LOG"

# ─── Common pieces ────────────────────────────────────────────────────────
read -r -d '' SOURCES <<'EOF' || true
| Role | Model |
|---|---|
| Base | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Source A | [`Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet`](https://huggingface.co/Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet) |
| Source B | [`BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5`](https://huggingface.co/BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5) |
EOF

read -r -d '' EVAL_METHOD <<'EOF' || true
Eval methodology: `llama-server` (`--reasoning-format deepseek --reasoning-budget 8192 --parallel 2 --cache-type-k q8_0 --cache-type-v q8_0 -c 32768`) → `lm_eval` `local-completions` against raw `/v1/completions`, temperature 0, max_gen_toks=2048. All five variants scored under identical conditions.
EOF

read -r -d '' VARIANT_LINKS <<'EOF' || true
- [Qwen3.5-4B-M1-Dare-Ties](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M1-Dare-Ties) — vanilla DARE-TIES
- [Qwen3.5-4B-M2-OMv2](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M2-OMv2) — OMv2 recipe (no importance signal)
- [Qwen3.5-4B-M3-Fisher](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M3-Fisher) — OMv2 + Fisher
- [Qwen3.5-4B-M4-ex-LRP](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M4-ex-LRP) — mergekit PR #682 ex-LRP
- [Qwen3.5-4B-M5-OMv2-LRP](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M5-OMv2-LRP) — OMv2 + LRP
EOF

# Helper: emit comparison table with one row bolded
# $1 = variant key (m1|m2|m3|m4|m5)
emit_table() {
    local hilite="$1"
    local rows=()
    rows[1]="| M1 | Vanilla DARE-TIES                                | dare_ties_merge.py     | none           | 51.22% | 47.00% |"
    rows[2]="| M2 | OMv2 recipe (OBIM-lite + DAREx-q + EMR election) | dare_ties_merge.py     | none           | 52.44% | 49.40% |"
    rows[3]="| M3 | OMv2 + Fisher                                    | dare_ties_merge.py     | Fisher         | **57.93%** 🥇 | 48.80% |"
    rows[4]="| M4 | ex-LRP (mergekit PR #682)                        | mergekit (PR #682)     | LRP            | 51.22% | 49.40% |"
    rows[5]="| M5 | OMv2 + LRP                                       | dare_ties_merge.py     | LRP            | 53.05% | **51.40%** 🥇 |"
    echo "| #  | Recipe                                           | Merger                 | Importance     | HumanEval pass@1 | MBPP pass@1 |"
    echo "|----|--------------------------------------------------|------------------------|----------------|:----------------:|:-----------:|"
    case "$hilite" in
        m1) echo "${rows[1]/| M1 |/| **M1 (this)** |}"; for i in 2 3 4 5; do echo "${rows[$i]}"; done ;;
        m2) echo "${rows[1]}"; echo "${rows[2]/| M2 |/| **M2 (this)** |}"; for i in 3 4 5; do echo "${rows[$i]}"; done ;;
        m3) for i in 1 2; do echo "${rows[$i]}"; done; echo "${rows[3]/| M3 |/| **M3 (this)** |}"; for i in 4 5; do echo "${rows[$i]}"; done ;;
        m4) for i in 1 2 3; do echo "${rows[$i]}"; done; echo "${rows[4]/| M4 |/| **M4 (this)** |}"; echo "${rows[5]}" ;;
        m5) for i in 1 2 3 4; do echo "${rows[$i]}"; done; echo "${rows[5]/| M5 |/| **M5 (this)** |}" ;;
    esac
}

# ─── Per-variant intro headers ────────────────────────────────────────────

# Common shared method block
read -r -d '' COMMON_METHOD <<'EOF' || true
Weights: 0.55 (A) / 0.45 (B). Density: 0.53. Seed: 42.
EOF

# ─── M1 README ────────────────────────────────────────────────────────────
build_m1() {
cat <<EOF
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- merge
- mergekit
- dare-ties
- experimental
base_model:
- Qwen/Qwen3.5-4B
- Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet
- BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5
---

# Qwen3.5-4B-M1-Dare-Ties

Vanilla DARE-TIES merge of Qwen3.5-4B with two distillation fine-tunes. Baseline of a 5-way comparison study (M1..M5) isolating the contributions of merge recipe vs importance-signal weighting on coding benchmarks.

## Sources

$SOURCES

$COMMON_METHOD

## Phase 1+2 comparison (Q6_K)

$(emit_table m1)

$EVAL_METHOD

## Other variants in this study

$VARIANT_LINKS
EOF
}

# ─── M2 README ────────────────────────────────────────────────────────────
build_m2() {
cat <<EOF
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- merge
- mergekit
- dare-ties
- omnimerge
- experimental
base_model:
- Qwen/Qwen3.5-4B
- Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet
- BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5
---

# Qwen3.5-4B-M2-OMv2

OMv2 recipe (OBIM-lite + DAREx-q + EMR election) **without** any importance-signal weighting. Isolates the recipe's contribution from the importance signal in the M3/M5 variants.

## Sources

$SOURCES

$COMMON_METHOD

## Phase 1+2 comparison (Q6_K)

$(emit_table m2)

$EVAL_METHOD

## Other variants in this study

$VARIANT_LINKS
EOF
}

# ─── M3 README ────────────────────────────────────────────────────────────
build_m3() {
cat <<EOF
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- merge
- mergekit
- dare-ties
- omnimerge
- fisher-information
- experimental
base_model:
- Qwen/Qwen3.5-4B
- Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet
- BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5
---

# Qwen3.5-4B-M3-Fisher

OMv2 recipe (OBIM-lite + DAREx-q + EMR election) **with diagonal Fisher information weighting** as the importance signal driving DAREx-q sparsification. **Best HumanEval result of the study (57.93%, +5.49 pp over recipe alone).**

> **Note:** An earlier version of this checkpoint silently dropped the Fisher signal due to a tensor-name prefix mismatch (Fisher files keyed by \`model.X\` from \`named_parameters()\`, but Qwen3.5_5's safetensors index uses \`model.language_model.X\`). This release uses a prefix-aware Fisher lookup; Fisher signal is now actually applied. The buggy variant was bit-identical to M2 — see comparison table below for the correction.

## Sources

$SOURCES

$COMMON_METHOD

Fisher: 64×256-token fp32 calibration each source (see \`fisher/\` subdir for the actual safetensors and methodology).

## Phase 1+2 comparison (Q6_K)

$(emit_table m3)

$EVAL_METHOD

## Other variants in this study

$VARIANT_LINKS
EOF
}

# ─── M4 README ────────────────────────────────────────────────────────────
build_m4() {
cat <<EOF
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- merge
- mergekit
- ex-lrp
- attnlrp
- explainable-ai
- experimental
base_model:
- Qwen/Qwen3.5-4B
- Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet
- BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5
---

# Qwen3.5-4B-M4-ex-LRP

Ex-LRP (Explainable LRP) merge per the recipe in [mergekit PR #682](https://github.com/arcee-ai/mergekit/pull/682), using \`AttnLRP\` relevance scores (lxt) as the importance signal driving merge weighting. Backbone: \`mergekit\` master + the \`pr-682-exlrp\` branch.

This is the apples-to-mergekit-recipe variant. The same LRP signal driven through our \`dare_ties_merge.py\` OMv2 recipe is published as [M5](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M5-OMv2-LRP) — comparing M4 vs M5 isolates "merger" from "signal."

## Sources

$SOURCES

$COMMON_METHOD

## Phase 1+2 comparison (Q6_K)

$(emit_table m4)

$EVAL_METHOD

## LRP signal

Under \`lrp/\` you'll find the AttnLRP relevance score safetensors used for this merge (multimodal-prefixed form: \`model.language_model.X\`), one per source model. Computed via \`scripts/lrp_compute_calibration.py\` on the same 64×256-token calibration mix as the Fisher run for M3.

The same raw scores in bare-keyed form (\`model.X\`) — what \`dare_ties_merge.py\` expects — live under \`lrp/\` in the [M5 repo](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M5-OMv2-LRP). The two are losslessly interconvertible via \`scripts/rename_lrp_keys_for_multimodal.py\`.

## Patches required for multimodal Qwen3.5_5

Running PR #682 against \`Qwen3_5ForConditionalGeneration\` required several minor patches (will be forwarded to the PR):
- \`mergekit/architecture/base.py\` — pydantic forward-ref \`model_rebuild()\` calls.
- \`mergekit/architecture/auto.py\` — layer-aware \`optional\` for hybrid attention modules.
- \`mergekit/merge_methods/lrp.py\` — base-passthrough fallback for tensors without LRP scores (vision tower, MTP).
- \`mergekit/lrp_computer.py\` — support \`qwen3_5_text\` inner-LM dispatch + tied-tensor clone before save.
- \`mergekit/config.py\` — \`str\` allowed in \`ParameterSetting\` union.
- \`lxt/efficient/models/__init__.py\` — optional \`vit_torch\` import.

## Other variants in this study

$VARIANT_LINKS
EOF
}

# ─── M5 README ────────────────────────────────────────────────────────────
build_m5() {
cat <<EOF
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- merge
- mergekit
- dare-ties
- omnimerge
- lrp
- attnlrp
- experimental
base_model:
- Qwen/Qwen3.5-4B
- Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet
- BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5
---

# Qwen3.5-4B-M5-OMv2-LRP

OMv2 recipe (OBIM-lite + DAREx-q + EMR election) with **AttnLRP relevance scores** as the importance signal driving DAREx-q sparsification. **Best MBPP result of the study (51.40%) and best balanced score (53.05% / 51.40%).**

Apples-to-apples comparison against [M3 (OMv2 + Fisher)](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M3-Fisher) — same merger, same recipe, only the importance source differs (LRP relevance vs Fisher squared-grad). And against [M4 (mergekit's ex-LRP)](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M4-ex-LRP) — same LRP signal, different merger.

## Sources

$SOURCES

$COMMON_METHOD

## Phase 1+2 comparison (Q6_K)

$(emit_table m5)

$EVAL_METHOD

## LRP signal

Under \`lrp/\` you'll find the AttnLRP relevance score safetensors used for this merge (bare-keyed form: \`model.X\`, what \`dare_ties_merge.py\` expects when the model is loaded as \`Qwen3_5ForConditionalGeneration\` and \`named_parameters()\` yields the inner-LM short paths). One file per source.

The same raw scores in multimodal-prefixed form (\`model.language_model.X\`) are published under \`lrp/\` in the [M4 repo](https://huggingface.co/ManniX-ITA/Qwen3.5-4B-M4-ex-LRP). The two are losslessly interconvertible via \`scripts/rename_lrp_keys_for_multimodal.py\`.

## Other variants in this study

$VARIANT_LINKS
EOF
}

# ─── Build + upload ───────────────────────────────────────────────────────
mkdir -p /tmp/mx_readmes
for v in m1 m2 m3 m4 m5; do
    out=/tmp/mx_readmes/README_${v}.md
    build_$v > "$out"
    echo "[*] built $out ($(wc -c < $out) bytes)" | tee -a "$LOG"
done

declare -A REPOS=(
    [m1]="ManniX-ITA/Qwen3.5-4B-M1-Dare-Ties"
    [m2]="ManniX-ITA/Qwen3.5-4B-M2-OMv2"
    [m3]="ManniX-ITA/Qwen3.5-4B-M3-Fisher"
    [m4]="ManniX-ITA/Qwen3.5-4B-M4-ex-LRP"
    [m5]="ManniX-ITA/Qwen3.5-4B-M5-OMv2-LRP"
)

for v in m1 m2 m3 m4 m5; do
    repo="${REPOS[$v]}"
    echo "[*] uploading README to $repo" | tee -a "$LOG"
    hf upload "$repo" /tmp/mx_readmes/README_${v}.md README.md --repo-type model 2>&1 | tee -a "$LOG" | tail -3
done

echo "=== mass-update done $(date -Iseconds) ===" | tee -a "$LOG"
