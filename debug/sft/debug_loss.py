"""Diagnostic script to check dataset labels and loss computation."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
from dataset import get_dataset, get_cot_latent_dataset, MotionCollator
from utils import Config
import yaml

with open("options/sft/t2m_coconut.yaml") as _f:
    configs = Config(yaml.safe_load(_f))

# Setup tokenizer (same as train.py)
tokenizer = AutoTokenizer.from_pretrained(configs.model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_tokens(["<Motion>", "</Motion>"])
for i in range(512):
    tokenizer.add_tokens([f"<Motion_{i}>"])
tokenizer.add_tokens(["<think>", "</think>"])
tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
think_close_ids = tokenizer.encode("\n</think>", add_special_tokens=False)

print(f"Vocab size: {len(tokenizer)}")
print(f"latent_id={latent_id}, start_id={start_id}, end_id={end_id}")
print(f"think_open_ids={think_open_ids}, think_close_ids={think_close_ids}")
print()

# Load a small dataset
base = get_dataset("./data/t2m_train.json", tokenizer, max_size=100, max_steps=20)

# Build Stage 0 dataset (no uniform randomness)
old_uniform = configs.uniform_prob
configs.uniform_prob = 0.0  # disable for deterministic check
ds = get_cot_latent_dataset(
    0, base, configs, start_id, latent_id, end_id,
    think_open_ids=think_open_ids, think_close_ids=think_close_ids,
    no_special_marker=False,
)
configs.uniform_prob = old_uniform

# Check a few samples
for i in range(3):
    s = ds[i]
    ids = s["input_ids"]
    labels = s["labels"]
    n_total = len(ids)
    n_masked = sum(1 for l in labels if l == -100)
    n_train = n_total - n_masked
    has_latent = latent_id in ids
    n_latent = ids.count(latent_id) if isinstance(ids, list) else 0

    # Count label types
    train_labels = [l for l in labels if l != -100]
    motion_count = sum(1 for l in train_labels if l >= 151670)
    text_count = len(train_labels) - motion_count

    print(f"Sample {i}: len={n_total}, masked={n_masked}, trainable={n_train}, "
          f"has_latent={has_latent}, n_latent={n_latent}")
    print(f"  Motion labels: {motion_count}, Text labels: {text_count}")
    first_train_idx = next(j for j, l in enumerate(labels) if l != -100)
    print(f"  First trainable at pos {first_train_idx}")
    print(f"  First 10 trainable tokens: {tokenizer.decode(train_labels[:10])}")
    print(f"  Last 10 trainable tokens: {tokenizer.decode(train_labels[-10:])}")
    print()

# Build a batch with collator
collator = MotionCollator(tokenizer, latent_id=latent_id)
batch = collator([ds[i] for i in range(4)])
print("=== Batch info ===")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

labels_t = batch["labels"]
input_ids_t = batch["input_ids"]
print(f"\nLabels -100 count per sample: {[(l == -100).sum().item() for l in labels_t]}")
print(f"Labels non-(-100) count: {[(l != -100).sum().item() for l in labels_t]}")

# Check shift alignment (as in coconut_motion loss computation)
shift_labels = labels_t[..., 1:]
print(f"Shift labels non-(-100) count: {[(l != -100).sum().item() for l in shift_labels]}")

for b in range(min(4, labels_t.shape[0])):
    non_masked = (labels_t[b] != -100).nonzero()
    if len(non_masked) > 0:
        first_pos = non_masked[0].item()
        last_pos = non_masked[-1].item()
        print(f"  Sample {b}: trainable range [{first_pos}, {last_pos}], "
              f"input_id at first={input_ids_t[b, first_pos].item()}, "
              f"label at first={labels_t[b, first_pos].item()}, "
              f"they match={input_ids_t[b, first_pos].item() == labels_t[b, first_pos].item()}")

print("\n=== Running model forward to check loss ===")
model = AutoModelForCausalLM.from_pretrained(
    configs.model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model = model.to("cuda:0").eval()

with torch.no_grad():
    # Single sample forward
    s = ds[0]
    ids = torch.tensor(s["input_ids"], device="cuda:0").unsqueeze(0)
    labs = torch.tensor(s["labels"], device="cuda:0").unsqueeze(0)

    # Direct model forward (like CoconutMotion's final pass path)
    embeddings = model.get_input_embeddings()
    inputs_embeds = embeddings(ids)
    outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=True)
    logits = outputs.logits

    # Compute loss exactly as coconut_motion.py
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print(f"Single sample loss: {loss.item():.4f}")

    # Now check what the loss would be per category
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    valid_mask = flat_labels != -100
    print(f"Valid tokens in loss: {valid_mask.sum().item()} / {flat_labels.shape[0]}")

    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    per_token_loss = torch.nn.functional.cross_entropy(
        valid_logits, valid_labels, reduction="none"
    )
    print(f"Mean loss over valid tokens: {per_token_loss.mean().item():.4f}")
    print(f"Min/Max per-token loss: {per_token_loss.min().item():.2f} / {per_token_loss.max().item():.2f}")
    print(f"Std per-token loss: {per_token_loss.std().item():.4f}")

    # Split by token category
    motion_mask = valid_labels >= 151670
    text_mask = ~motion_mask
    if motion_mask.any():
        print(f"Motion token loss: {per_token_loss[motion_mask].mean().item():.4f} ({motion_mask.sum().item()} tokens)")
    if text_mask.any():
        print(f"Text token loss: {per_token_loss[text_mask].mean().item():.4f} ({text_mask.sum().item()} tokens)")

    # Check multiple samples for loss variance
    print("\n=== Multi-sample loss check ===")
    for i in range(min(8, len(ds))):
        s = ds[i]
        ids_i = torch.tensor(s["input_ids"], device="cuda:0").unsqueeze(0)
        labs_i = torch.tensor(s["labels"], device="cuda:0").unsqueeze(0)
        inputs_embeds_i = embeddings(ids_i)
        out_i = model(inputs_embeds=inputs_embeds_i)
        shift_l = out_i.logits[..., :-1, :].contiguous()
        shift_la = labs_i[..., 1:].contiguous()
        loss_i = loss_fct(shift_l.view(-1, shift_l.size(-1)), shift_la.view(-1))
        print(f"  Sample {i}: loss={loss_i.item():.4f}")
