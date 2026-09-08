#!/usr/bin/env bash
# Score the OLD-extractor counterfactual tree. Same docker evaluator, same
# problems, same raw generations -- only chat_to_body differs. Any pass@1 gap
# vs the b604 tree is bug-604 and nothing else.
set -uo pipefail
EV=/shared/dev/omnimergekit/eval/multipl_e/multipl_e_evaluate.sh
ROOT=/srv/ml/eval_results_b604_oldext/qwen_suite/multipl_e_100
for cell in qwencodermpe_t10_q6k qwen256e_q6k qwencodermpe_q6k qwenhybridp24_q6k; do
  for lang in rs java js; do
    g="$ROOT/$cell/generations/humaneval-$lang"
    o="$ROOT/$cell/results/humaneval-$lang"
    [ -d "$g" ] || continue
    echo ">>> $cell/$lang"
    bash "$EV" "$g" "$o" >/dev/null 2>&1 || echo "    EVAL_RC=$?"
  done
done
echo "OLDEXT_SCORING_DONE"
