#!/bin/bash
# Upload M1/M2/M3 (Qwen3.5-4B Phase 1) to HF, then remove pod-side weights.
# Frees ~36 GB before M4 Q6_K + eval needs disk room.
#
# Naming per user direction (2026-04-30):
#   M1 — Qwen3.5-4B-M1-Dare-Ties (DARE-TIES baseline)
#   M2 — Qwen3.5-4B-M2-OMv2     (OBIM+DAREx+EMR, no Fisher)
#   M3 — Qwen3.5-4B-M3-Fisher   (OMv2 recipe + Fisher info)
set -uo pipefail

OWNER=ManniX-ITA
OUT=/workspace/4b_phase1

# format: M_NUM:DIR_NAME:GGUF_NAME:REPO_NAME:METHOD_LABEL
VARIANTS=(
    "1:dare_ties_baseline:DARE_TIES_baseline-Q6_K.gguf:Qwen3.5-4B-M1-Dare-Ties:DARE-TIES"
    "2:omnimerge_v2_no_fisher:Recipe_noFisher-Q6_K.gguf:Qwen3.5-4B-M2-OMv2:OMv2 recipe (OBIM-lite + DAREx-q + EMR)"
    "3:omnimerge_v2_fisher:Omnimerge_v2_Fisher-Q6_K.gguf:Qwen3.5-4B-M3-Fisher:OMv2 recipe + Fisher info weighting"
)

# Eval results (precomputed)
declare -A HE MBPP
HE[Qwen3.5-4B-M1-Dare-Ties]="51.22%"
HE[Qwen3.5-4B-M2-OMv2]="52.44%"
HE[Qwen3.5-4B-M3-Fisher]="52.44%"
MBPP[Qwen3.5-4B-M1-Dare-Ties]="47.00% (rescored 47.00%)"
MBPP[Qwen3.5-4B-M2-OMv2]="49.40% (rescored 49.40%)"
MBPP[Qwen3.5-4B-M3-Fisher]="49.40% (rescored 49.60%)"

write_readme() {
    local repo="$1" method="$2"
    local readme="/tmp/README_${repo}.md"
    cat > "$readme" <<EOF
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

# ${repo}

Phase 1 of a 4-way merge comparison on Qwen3.5-4B. This variant uses **${method}**.

The full experiment isolates the contribution of each component of the OMv2 recipe (the same recipe behind the larger Qwen3.6-27B-Omnimerge-v2/v4 series) against vanilla DARE-TIES, with Fisher information weighting and ex-LRP added on top.

## Sources

| Role | Model |
|---|---|
| Base | [\`Qwen/Qwen3.5-4B\`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Source A | [\`Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet\`](https://huggingface.co/Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet) |
| Source B | [\`BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5\`](https://huggingface.co/BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5) |

Weights: 0.55 (A) / 0.45 (B). Density: 0.53. Seed: 42.

## Phase 1 — head-to-head

| Variant | Method | HumanEval pass@1 | MBPP pass@1 | MBPP rescored (think-strip) |
|---|---|:---:|:---:|:---:|
| **M1** Dare-Ties | DARE-TIES (mergekit-equivalent) | 51.22% | 47.00% | 47.00% |
| **M2** OMv2 | OBIM-lite + DAREx-q + EMR election | **52.44%** | **49.40%** | 49.40% |
| **M3** Fisher | OMv2 recipe + Fisher info weighting | **52.44%** | **49.40%** | 49.60% |

**Eval methodology:** Q6_K quantization → llama-server (\`--reasoning-format deepseek --reasoning-budget 8192 --parallel 2\`) → lm_eval \`local-completions\` against raw \`/v1/completions\`, temperature 0, max_gen_toks=2048. Same setup as the Qwen3.6-27B Omnimerge-v4 evals.

## Headline finding (Phase 1)

**Fisher information weighting adds essentially zero uplift over the OMv2 recipe alone.** M2 and M3 produce identical HumanEval samples and differ by 1 MBPP sample (+0.20 pp, well inside ±2.24 pp stderr). The +1.22 pp HumanEval / +2.40 pp MBPP gain over plain DARE-TIES comes entirely from **OBIM-lite + DAREx-q + EMR election**.

Implication for the 27B chassis: the OMv2 recipe's value is the trio, not the importance-weighting layer. Phase 2 (M4 ex-LRP via mergekit PR #682) will determine whether learned-relevance is a better signal than Fisher for this job.

## Quants

The included \`*-Q6_K.gguf\` is the artifact used for all evaluations above. Full GGUF variants will be published separately if Phase 2 results warrant publication.

## Reproduce

\`\`\`yaml
# methods 1-3 use scripts/dare_ties_merge.py from the Omnimerge tooling
python scripts/dare_ties_merge.py \\
    --base Qwen/Qwen3.5-4B \\
    --source Jackrong/Qwen3.5-4B-Distill-Claude-4.5_Sonnet \\
    --source BlackBladeOrg/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5 \\
    --weights 0.55,0.45 --density 0.53 --darex-q 0.75 --seed 42 \\
    --method <variant>
\`\`\`

| M-N | \`--method\` | \`--v2-features\` | extra |
|---|---|---|---|
| M1 | dare_ties | — | — |
| M2 | omnimerge_v2 | obim,darex,emr | — |
| M3 | omnimerge_v2 | obim,darex,emr,fisher | + Fisher safetensors per source |

## Other variants

- [Qwen3.5-4B-M1-Dare-Ties](https://huggingface.co/${OWNER}/Qwen3.5-4B-M1-Dare-Ties)
- [Qwen3.5-4B-M2-OMv2](https://huggingface.co/${OWNER}/Qwen3.5-4B-M2-OMv2)
- [Qwen3.5-4B-M3-Fisher](https://huggingface.co/${OWNER}/Qwen3.5-4B-M3-Fisher)
- M4 (ex-LRP via mergekit PR #682) — pending

EOF
    echo "$readme"
}

upload_one() {
    local NUM="$1" DIR="$2" GGUF="$3" REPO="$4" METHOD="$5"
    local FULL_REPO="${OWNER}/${REPO}"

    echo ""
    echo "=== M${NUM} → ${FULL_REPO} ==="

    if [ ! -d "$OUT/merged/$DIR" ]; then
        echo "[!] $OUT/merged/$DIR missing — skip"
        return 1
    fi
    if [ ! -f "$OUT/gguf/$GGUF" ]; then
        echo "[!] $OUT/gguf/$GGUF missing — skip"
        return 1
    fi

    echo "[*] create repo (idempotent)"
    hf repo create "$FULL_REPO" --type model --exist-ok 2>&1 | tail -2

    local readme
    readme=$(write_readme "$REPO" "$METHOD")

    echo "[*] upload BF16 dir ($(du -sh "$OUT/merged/$DIR" | cut -f1))"
    hf upload "$FULL_REPO" "$OUT/merged/$DIR" . --repo-type model 2>&1 | tail -3

    echo "[*] upload Q6_K GGUF (3.3G) as ${REPO}-Q6_K.gguf"
    hf upload "$FULL_REPO" "$OUT/gguf/$GGUF" "${REPO}-Q6_K.gguf" --repo-type model 2>&1 | tail -3

    echo "[*] upload README"
    hf upload "$FULL_REPO" "$readme" README.md --repo-type model 2>&1 | tail -2

    echo "[*] verify (model_info)"
    if hf api repos/${FULL_REPO} 2>/dev/null | grep -q '"sha"'; then
        echo "    repo OK on HF"
    else
        # fallback verify
        local count
        count=$(hf api repos/${FULL_REPO}/tree/main 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
        if [ -n "$count" ] && [ "$count" -ge 5 ]; then
            echo "    repo has $count files — OK"
        else
            echo "[!] verify inconclusive — NOT removing local dir"
            return 1
        fi
    fi

    echo "[*] free pod-side: $OUT/merged/$DIR + $OUT/gguf/$GGUF"
    rm -rf "$OUT/merged/$DIR"
    rm -f "$OUT/gguf/$GGUF"
    echo "    freed; df now: $(df -h /workspace | tail -1 | awk '{print $5}')"
}

for v in "${VARIANTS[@]}"; do
    IFS=':' read -r NUM DIR GGUF REPO METHOD <<< "$v"
    upload_one "$NUM" "$DIR" "$GGUF" "$REPO" "$METHOD"
done

echo ""
echo "=== upload pass done $(date -Iseconds) ==="
df -h /workspace | tail -1
