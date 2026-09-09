"""Gate: the engine must detect the MTP head from the F16 GGUF, or the rebuild stops.

These are precisely the tiers whose correctness DEPENDS on the blk.64 Q4_K floor, so
running them without detection is worse than not running them: four bail loudly, and
Q2_K builds a degraded head and overwrites a good file on HF.
"""
import pathlib
import sys

sys.path.insert(0, "/srv/ml/repos/omnimergekit/scripts")
from quantize_gguf import detect_mtp_from_gguf  # noqa: E402

f16 = pathlib.Path(sys.argv[1])
info = detect_mtp_from_gguf(f16)
if not info or info.get("mtp_block_idx") != 64:
    sys.exit(f"ABORT: MTP not detected from {f16.name} (got {info!r})")
print(f"[gate] MTP detected: blk.{info['mtp_block_idx']} "
      f"({info['mtp_tensor_count']} tensors, nextn={info['mtp_num_hidden_layers']})")
