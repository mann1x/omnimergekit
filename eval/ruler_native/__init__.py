# ruler_native — omk backend for the NVIDIA RULER (arXiv:2404.06654) long-context
# synthetic benchmark.
#
# The runner is a thin omk-shaped layer over upstream's `scripts/data/prepare.py`
# (called as a subprocess to generate validation.jsonl), our own /v1/chat/completions
# loop with sqlite resume, and an INLINE port of upstream's `string_match_all`
# scorer (Apache-2.0 attribution in ruler_helpers.py). Why inline vs. subprocess
# into upstream's evaluate.py: that script imports
# `from nemo.collections.asr.parts.utils.manifest_utils import {read,write}_manifest`,
# which forces a `pip install nemo-toolkit[all]` cascade that silently downgrades
# transformers / torch / safetensors / modelopt out from under the canonical
# omk env pins (verified live on bs2 2026-05-28). The scorer math is 3 lines
# of pure stdlib case-insensitive substring match — inlining is byte-identical
# to upstream output. See ruler_helpers.string_match_all for the verbatim port.
