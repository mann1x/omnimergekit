import inspect
import trl, vllm, transformers
from trl import GRPOTrainer, GRPOConfig
print("trl", trl.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
sig = inspect.signature(GRPOConfig.__init__)
keys = ["loss_type","importance_sampling_level","scale_rewards","vllm_mode",
        "vllm_tensor_parallel_size","vllm_gpu_memory_utilization","epsilon","epsilon_high",
        "beta","steps_per_generation","num_generations","generation_batch_size",
        "num_iterations","mask_truncated_completions","max_completion_length"]
for k in keys:
    p = sig.parameters.get(k)
    tag = ("PRESENT default=" + repr(p.default)) if p is not None else "*** ABSENT ***"
    print("  {:32s} {}".format(k, tag))
print()
print("=== Gemma-4 support in transformers", transformers.__version__, "===")
from transformers import AutoConfig
try:
    cfg = AutoConfig.from_pretrained("/mnt/sdc/ml/models/Jprime-p3-bf16", trust_remote_code=True)
    print("  Jprime config OK:", type(cfg).__name__, "| model_type=", getattr(cfg,"model_type",None))
except Exception as e:
    print("  Jprime config FAILED:", type(e).__name__, e)
