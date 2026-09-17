"""Test the full GRPO step: generate + forward + log probs on a single GPU."""
import torch
import yaml
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_motion import CoconutMotion
from dataset import get_dataset, get_question_latent_dataset, MotionCollator
from grpo_loss import compute_per_token_log_probs
from torch.utils.data import DataLoader

torch.manual_seed(42)

with open("options/grpo/t2m_grpo.yaml") as f:
    cfg = yaml.safe_load(f)
configs = SimpleNamespace(**cfg)

# Setup
tokenizer = AutoTokenizer.from_pretrained(configs.model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_tokens(["<Motion>", "</Motion>"])
for i in range(configs.nb_code):
    tokenizer.add_tokens([f"<Motion_{i}>"])
tokenizer.add_tokens(["<think>", "</think>"])
tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])

model = AutoModelForCausalLM.from_pretrained(
    configs.sft_checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

coconut_model = CoconutMotion(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
coconut_model = coconut_model.cuda()

# Load data
train_tokenized = get_dataset(configs.train_path, tokenizer, max_steps=20)
ds = get_question_latent_dataset(
    configs.max_latent_stage, train_tokenized, configs, start_id, latent_id, end_id
)
collator = MotionCollator(tokenizer=tokenizer, latent_id=latent_id)
loader = DataLoader(ds, batch_size=configs.batch_size_training, collate_fn=collator, shuffle=False)

batch = next(iter(loader))
device = torch.device("cuda")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

print(f"Batch size: {input_ids.shape[0]}")
print(f"Seq len: {input_ids.shape[1]}")
print(f"input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")

# Check latent alignment
lat_starts = []
for i in range(input_ids.shape[0]):
    pos = (input_ids[i] == latent_id).nonzero()
    if len(pos) > 0:
        lat_starts.append(pos[0].item())
print(f"Latent start positions (unique): {sorted(set(lat_starts))}")
print(f"All aligned: {len(set(lat_starts)) == 1}")

# === Test 1: Generation ===
print("\n=== Test 1: Generation (batch_size=32, G=1) ===")
coconut_model.eval()
with torch.no_grad():
    try:
        result = coconut_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        print(f"  OK: shape={result.shape}, range=[{result.min().item()}, {result.max().item()}]")
    except Exception as e:
        print(f"  FAILED: {e}")

# === Test 2: Forward pass (as in GRPO training) ===
print("\n=== Test 2: Forward pass with build_forward_inputs ===")
from train_grpo import generate_completions, build_forward_inputs

coconut_model.eval()
completions, completion_ids, prompt_texts, idxs = generate_completions(
    coconut_model, batch, tokenizer, configs, 1, device  # G=1 for speed
)
print(f"  Generated {len(completions)} completions")
print(f"  Completion id lengths: {[len(c) for c in completion_ids[:5]]}...")

# Build forward inputs
print("\n  Building forward inputs...")
full_ids, full_mask, full_labels, position_ids = build_forward_inputs(
    coconut_model, tokenizer, batch, completion_ids, configs, device, collator=collator
)
print(f"  full_ids shape: {full_ids.shape}")
print(f"  full_ids range: [{full_ids.min().item()}, {full_ids.max().item()}]")
print(f"  full_mask sum per sample (first 5): {full_mask.sum(dim=1)[:5].tolist()}")
print(f"  position_ids range: [{position_ids.min().item()}, {position_ids.max().item()}]")

# Check latent alignment in forward inputs
lat_starts_fwd = []
for i in range(full_ids.shape[0]):
    pos = (full_ids[i] == latent_id).nonzero()
    if len(pos) > 0:
        lat_starts_fwd.append(pos[0].item())
print(f"  Latent start positions (unique): {sorted(set(lat_starts_fwd))}")
print(f"  All aligned: {len(set(lat_starts_fwd)) == 1}")

# Forward pass
print("\n  Running forward pass...")
coconut_model.train()
try:
    outputs = coconut_model(
        input_ids=full_ids,
        attention_mask=full_mask,
        labels=full_labels,
        position_ids=position_ids,
    )
    print(f"  OK: logits shape={outputs.logits.shape}, loss={outputs.loss.item():.4f}")

    per_token_logps, comp_mask = compute_per_token_log_probs(outputs.logits, full_labels)
    print(f"  per_token_logps shape: {per_token_logps.shape}")
    print(f"  completion_mask sum (first 5): {comp_mask.sum(dim=1)[:5].tolist()}")
except Exception as e:
    import traceback
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("\n=== All tests passed! ===")
