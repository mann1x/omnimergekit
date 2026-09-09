import inspect
from trl import GRPOConfig
sig = set(inspect.signature(GRPOConfig.__init__).parameters)
want = ["output_dir","loss_type","importance_sampling_level","scale_rewards","epsilon",
        "beta","num_iterations","num_generations","max_completion_length",
        "vllm_max_model_length","temperature","mask_truncated_completions","use_vllm",
        "vllm_mode","vllm_tensor_parallel_size","vllm_gpu_memory_utilization",
        "learning_rate","lr_scheduler_type","warmup_ratio","warmup_steps",
        "num_train_epochs","per_device_train_batch_size","gradient_accumulation_steps",
        "gradient_checkpointing","bf16","optim","max_steps","logging_steps",
        "save_strategy","save_steps","seed","report_to"]
missing = [k for k in want if k not in sig]
print("ABSENT from GRPOConfig 1.12.0:", missing or "none")
print()
print("alternatives present containing warmup/scheduler/epoch/save/log:")
for k in sorted(sig):
    if any(t in k for t in ("warmup","scheduler","epoch","save","logging","optim","report")):
        print("   ", k)
