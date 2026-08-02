"""ifeval_v31reg68 — the 68 IFEval prompts v30 passes and v31 fails, at GREEDY.

Produced by an-finetune/eval/ifeval_delta.py from the two canonical greedy runs
(omk/greedy/ifeval_full/{v30,v31}). These 68 docs carry the entire -4.07 pp
prompt_level_strict_acc move, so an intermediate checkpoint can be asked the same
question at a 68-doc cost instead of a 541-doc one.

Only the doc SELECTION changes. process_results / agg_inst_level_acc are re-exported
from the stock lm-eval ifeval task, so strict and loose prompt+instruction accuracy are
computed by exactly the same code that produced the v30/v31 numbers. A subset scored by
a different scorer is not a subset of the parent's result.

Selection is by dataset "key", never by row index. lm-eval's --limit takes the first N
and index arithmetic drifts the moment upstream reorders the split; a silently different
68 rows would read as a model difference. The count assert below makes that drift fatal
instead of invisible.
"""
from lm_eval.tasks.ifeval.utils import (  # noqa: F401  (re-exported for the yaml)
    agg_inst_level_acc,
    process_results,
)

_REG68_KEYS = {
    13, 179, 202, 247, 279, 321, 337, 1082, 1154, 1203, 1236, 1248,
    1265, 1325, 1367, 1476, 1480, 1582, 1609, 1629, 1645, 1659, 1686, 1733,
    1746, 1759, 1823, 1845, 1886, 1922, 1934, 1967, 2023, 2078, 2162, 2268,
    2273, 2362, 2374, 2383, 2391, 2441, 2471, 2683, 2724, 2759, 2765, 2811,
    2832, 2844, 2853, 2929, 3069, 3109, 3126, 3287, 3324, 3329, 3401, 3442,
    3445, 3505, 3506, 3513, 3536, 3619, 3623, 3748,
}


def select_ifeval_v31reg68(dataset):
    out = dataset.filter(lambda x: x.get("key") in _REG68_KEYS)
    if len(out) != len(_REG68_KEYS):
        missing = sorted(_REG68_KEYS - set(out["key"]))
        raise ValueError(
            f"ifeval_v31reg68: expected {len(_REG68_KEYS)} docs, got {len(out)} "
            f"(missing keys={missing}); every regressed key must exist in the "
            "google/IFEval train split -- a short subset is NOT a valid comparison"
        )
    return out


# --------------------------------------------------------------------------- noise control
# The 68 docs above were SELECTED as v31's failures, so re-running them cannot separate "the
# model changed" from "these docs are coin-flips": conditioning on an extreme guarantees
# regression to the mean. This second set is the control that can. It is 68 keys drawn
# UNIFORMLY at random from the same 541-doc parent population --
#     random.Random(20260801).sample(sorted(parent_keys), 68)
# -- drawn on the KEY, never on the outcome, so its self-discordance under a re-run is an
# unbiased estimate of the harness's own run-to-run noise. Compare that rate against the
# v30-vs-v31 discordance (114/541 = 21.1%): if they are the same, the -4.07 pp IFEval "delta"
# is the stack talking, not the model. The parent run passes 51 of these 68.
#
# The seed and the population are recorded here rather than in a shell variable because the
# draw IS the experiment -- a control set you cannot regenerate is not a control.
_NOISE68_SEED = 20260801
_NOISE68_KEYS = {
    164, 168, 260, 292, 295, 334, 349, 1127, 1128, 1131, 1246, 1262,
    1265, 1281, 1477, 1480, 1537, 1551, 1643, 1658, 1659, 1670, 1773, 1813,
    1823, 1845, 1897, 1936, 2070, 2071, 2084, 2136, 2142, 2195, 2225, 2266,
    2337, 2341, 2362, 2392, 2398, 2505, 2532, 2583, 2728, 2751, 2768, 2780,
    2853, 2857, 2871, 2908, 2909, 3071, 3079, 3198, 3204, 3280, 3326, 3425,
    3484, 3494, 3536, 3567, 3572, 3608, 3653, 3691,
}


def select_ifeval_v31noise68(dataset):
    out = dataset.filter(lambda x: x.get("key") in _NOISE68_KEYS)
    if len(out) != len(_NOISE68_KEYS):
        missing = sorted(_NOISE68_KEYS - set(out["key"]))
        raise ValueError(
            f"ifeval_v31noise68: expected {len(_NOISE68_KEYS)} docs, got {len(out)} "
            f"(missing keys={missing}); a short control set understates the noise floor, "
            "which is the one direction that would wrongly rescue the delta"
        )
    return out
