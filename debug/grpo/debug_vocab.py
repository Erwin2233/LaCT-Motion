"""Quick diagnostic: check vocab size alignment between SFT checkpoint and tokenizer."""
import torch
import yaml
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("options/grpo/t2m_grpo.yaml") as f:
    cfg = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
print(f"Base tokenizer vocab: {len(tokenizer)}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(["<Motion>", "</Motion>"])
for i in range(cfg["nb_code"]):
    tokenizer.add_tokens([f"<Motion_{i}>"])
tokenizer.add_tokens(["<think>", "</think>"])
tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
print(f"After adding tokens: {len(tokenizer)}")

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
print(f"latent_id={latent_id}, start_id={start_id}, end_id={end_id}")
print(f"eos_token_id={tokenizer.eos_token_id}, pad_token_id={tokenizer.pad_token_id}")

sft_path = cfg["sft_checkpoint"]
with open(f"{sft_path}/config.json") as f:
    model_cfg = json.load(f)
print(f"SFT checkpoint vocab_size in config: {model_cfg.get('vocab_size', 'N/A')}")

model = AutoModelForCausalLM.from_pretrained(
    sft_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
embed_size = model.get_input_embeddings().weight.shape[0]
lm_head_size = model.lm_head.weight.shape[0]
print(f"SFT model embedding rows: {embed_size}")
print(f"SFT model lm_head rows: {lm_head_size}")

model.resize_token_embeddings(len(tokenizer))
new_embed_size = model.get_input_embeddings().weight.shape[0]
print(f"After resize: {new_embed_size}")

diff = embed_size - len(tokenizer)
if diff == 0:
    print("OK: embedding size matches tokenizer vocab size")
else:
    print(f"WARNING: SFT has {embed_size} embeddings but tokenizer has {len(tokenizer)} tokens! Diff={diff}")

# Also test: load a sample batch and check token ranges
from dataset import get_dataset, get_question_latent_dataset, MotionCollator
from types import SimpleNamespace

configs = SimpleNamespace(**cfg)
print("\n--- Testing data pipeline ---")
train_tokenized = get_dataset(cfg["train_path"], tokenizer, max_steps=20)
print(f"Tokenized dataset size: {len(train_tokenized)}")

# Check token range in first few samples
max_token = 0
min_token = 999999
for sample in list(train_tokenized)[:100]:
    for tid in sample["question_tokenized"]:
        max_token = max(max_token, tid)
        min_token = min(min_token, tid)
    for tid in sample["answer_tokenized"]:
        max_token = max(max_token, tid)
        min_token = min(min_token, tid)
print(f"Token ID range in data (first 100 samples): [{min_token}, {max_token}]")
print(f"Vocab size: {len(tokenizer)}")
if max_token >= len(tokenizer):
    print(f"ERROR: max token ID {max_token} >= vocab size {len(tokenizer)}")
else:
    print("OK: all data token IDs within vocab range")

# Build question_latent_dataset and check
scheduled_stage = cfg["max_latent_stage"]
ds = get_question_latent_dataset(
    scheduled_stage, train_tokenized, configs, start_id, latent_id, end_id
)
print(f"\nQuestion latent dataset size: {len(ds)}")
sample0 = ds[0]
ids = sample0["input_ids"]
print(f"Sample 0 length: {len(ids)}")
print(f"Sample 0 token range: [{min(ids)}, {max(ids)}]")
if max(ids) >= len(tokenizer):
    print(f"ERROR: max token ID {max(ids)} >= vocab size {len(tokenizer)}")
else:
    print("OK: all token IDs within vocab range")

# Test collator
collator = MotionCollator(tokenizer=tokenizer, latent_id=latent_id)
batch = collator([ds[i] for i in range(4)])
print(f"\nCollated batch input_ids shape: {batch['input_ids'].shape}")
print(f"Collated batch token range: [{batch['input_ids'].min().item()}, {batch['input_ids'].max().item()}]")
if batch["input_ids"].max().item() >= len(tokenizer):
    print(f"ERROR: max token ID in batch >= vocab size")
else:
    print("OK: collated batch token IDs within vocab range")
print(f"Attention mask sum per sample: {batch['attention_mask'].sum(dim=1).tolist()}")
print(f"Position ids shape: {batch['position_ids'].shape}")
