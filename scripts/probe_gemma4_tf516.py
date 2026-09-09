import transformers
from transformers import AutoConfig, AutoTokenizer
P="/mnt/sdc/v7rework/arms/Jprime-p3-bf16"
print("transformers", transformers.__version__)
cfg = AutoConfig.from_pretrained(P, trust_remote_code=True)
print("config:", type(cfg).__name__, "| model_type =", getattr(cfg,"model_type",None))
tc = getattr(cfg, "text_config", None)
print("hidden:", getattr(tc or cfg,"hidden_size",None), "| layers:", getattr(tc or cfg,"num_hidden_layers",None))
tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
print("tokenizer:", type(tok).__name__, "| vocab", len(tok))
print("eos_token_id:", tok.eos_token_id)
ids = tok.convert_tokens_to_ids(["<end_of_turn>"])
print("<end_of_turn> ->", ids)
print("chat_template present:", bool(getattr(tok,"chat_template",None)))
