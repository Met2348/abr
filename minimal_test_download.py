
from transformers import AutoTokenizer, AutoConfig

m = "assets/models/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
cfg = AutoConfig.from_pretrained(m, local_files_only=True)

print("model_type:", cfg.model_type)
print("vocab_size:", tok.vocab_size)

