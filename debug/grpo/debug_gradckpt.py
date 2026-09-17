"""Diagnose: does gradient checkpointing + multi-pass forward cause the CUDA error?

Tests the exact same code path as train_grpo.py step-by-step with torch.cuda.synchronize()
between each operation to pinpoint which one triggers the CUDA assertion.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 debug_gradckpt.py
"""
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_motion import CoconutMotion
from dataset import get_dataset, get_question_latent_dataset, MotionCollator
from grpo_loss import compute_per_token_log_probs
from train_grpo import generate_completions, build_forward_inputs

torch.manual_seed(42)

# DDP setup
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def sync_print(msg):
    torch.cuda.synchronize()
    print(f"[rank {rank}] {msg}", flush=True)

with open("options/grpo/t2m_grpo.yaml") as f:
    cfg = yaml.safe_load(f)
configs = SimpleNamespace(**cfg)

# === Setup tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(configs.model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_tokens(["<Motion>", "</Motion>"])
for i in range(configs.nb_code):
    tokenizer.add_tokens([f"<Motion_{i}>"])
tokenizer.add_tokens(["<think>", "</think>"])
tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

# === Load model ===
sync_print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    configs.sft_checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))

coconut_model = CoconutMotion(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
sync_print("Model loaded")

# === Test 1: WITHOUT gradient checkpointing ===
sync_print("\n========== TEST 1: NO gradient checkpointing ==========")
coconut_model = coconut_model.to(device)
parallel_model = DDP(coconut_model, device_ids=[local_rank], find_unused_parameters=False)

# Load data
train_tokenized = get_dataset(configs.train_path, tokenizer, max_steps=20)
ds = get_question_latent_dataset(configs.max_latent_stage, train_tokenized, configs, start_id, latent_id, end_id)
collator = MotionCollator(tokenizer=tokenizer, latent_id=latent_id)
sampler = DistributedSampler(ds, shuffle=False)
loader = DataLoader(ds, batch_size=configs.batch_size_training, collate_fn=collator, sampler=sampler)

batch = next(iter(loader))
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

sync_print(f"Batch: input_ids={input_ids.shape}, range=[{input_ids.min().item()}, {input_ids.max().item()}]")

# Step 1: Generate
sync_print("Step 1: generate_completions (G=1)...")
parallel_model.eval()
completions, completion_ids, prompt_texts, idxs = generate_completions(
    parallel_model.module, batch, tokenizer, configs, 1, device
)
sync_print(f"Step 1 OK: {len(completions)} completions")

# Step 2: Build forward inputs
sync_print("Step 2: build_forward_inputs...")
full_ids, full_mask, full_labels, position_ids = build_forward_inputs(
    parallel_model.module, tokenizer, batch, completion_ids, configs, device, collator=collator
)
sync_print(f"Step 2 OK: full_ids={full_ids.shape}, range=[{full_ids.min().item()}, {full_ids.max().item()}]")

# Step 3: DDP forward (training)
sync_print("Step 3: DDP forward pass (no grad ckpt)...")
parallel_model.train()
outputs = parallel_model(
    input_ids=full_ids, attention_mask=full_mask,
    labels=full_labels, position_ids=position_ids,
)
sync_print(f"Step 3 OK: logits={outputs.logits.shape}, loss={outputs.loss.item():.4f}")

# Step 4: Backward
sync_print("Step 4: backward...")
outputs.loss.backward()
sync_print("Step 4 OK: backward complete")

# Step 5: Log probs
sync_print("Step 5: compute_per_token_log_probs...")
parallel_model.eval()
with torch.no_grad():
    outputs2 = parallel_model(
        input_ids=full_ids, attention_mask=full_mask,
        labels=full_labels, position_ids=position_ids,
    )
    per_token_logps, comp_mask = compute_per_token_log_probs(outputs2.logits, full_labels)
sync_print(f"Step 5 OK: logps={per_token_logps.shape}")

parallel_model.zero_grad()
sync_print("TEST 1 PASSED: No gradient checkpointing works fine\n")

# === Test 2: WITH gradient checkpointing ===
sync_print("========== TEST 2: WITH gradient checkpointing ==========")

# Enable gradient checkpointing
coconut_model.base_causallm.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
sync_print("Gradient checkpointing enabled")

# Step 1: Generate (eval mode, should be fine)
sync_print("Step 1: generate_completions (G=1, with grad ckpt)...")
parallel_model.eval()
completions2, completion_ids2, prompt_texts2, idxs2 = generate_completions(
    parallel_model.module, batch, tokenizer, configs, 1, device
)
sync_print(f"Step 1 OK: {len(completions2)} completions")

# Step 2: Build forward inputs
sync_print("Step 2: build_forward_inputs...")
full_ids2, full_mask2, full_labels2, position_ids2 = build_forward_inputs(
    parallel_model.module, tokenizer, batch, completion_ids2, configs, device, collator=collator
)
sync_print(f"Step 2 OK: full_ids={full_ids2.shape}")

# Step 3: DDP forward with gradient checkpointing
sync_print("Step 3: DDP forward pass (WITH grad ckpt)...")
parallel_model.train()
try:
    outputs3 = parallel_model(
        input_ids=full_ids2, attention_mask=full_mask2,
        labels=full_labels2, position_ids=position_ids2,
    )
    sync_print(f"Step 3 OK: logits={outputs3.logits.shape}, loss={outputs3.loss.item():.4f}")
except Exception as e:
    sync_print(f"Step 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
    dist.destroy_process_group()
    sys.exit(1)

# Step 4: Backward with gradient checkpointing
sync_print("Step 4: backward (WITH grad ckpt)...")
try:
    outputs3.loss.backward()
    sync_print("Step 4 OK: backward complete")
except Exception as e:
    sync_print(f"Step 4 FAILED: {e}")
    import traceback
    traceback.print_exc()
    dist.destroy_process_group()
    sys.exit(1)

parallel_model.zero_grad()
sync_print("TEST 2 PASSED: Gradient checkpointing works fine\n")

# === Test 3: WITH gradient checkpointing + reference model ===
sync_print("========== TEST 3: + Reference model ==========")

# Load reference model
sync_print("Loading reference model...")
ref_base = AutoModelForCausalLM.from_pretrained(
    configs.sft_checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
ref_base.resize_token_embeddings(len(tokenizer))
ref_model = CoconutMotion(ref_base, latent_id, start_id, end_id, tokenizer.eos_token_id)
ref_model = ref_model.to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
sync_print("Reference model loaded")

# Forward + backward like in real training
sync_print("Step 3a: Policy forward (grad ckpt on)...")
parallel_model.train()
outputs4 = parallel_model(
    input_ids=full_ids2, attention_mask=full_mask2,
    labels=full_labels2, position_ids=position_ids2,
)
sync_print(f"Step 3a OK: loss={outputs4.loss.item():.4f}")

sync_print("Step 3b: Reference forward (no grad)...")
with torch.no_grad():
    ref_outputs = ref_model(
        input_ids=full_ids2, attention_mask=full_mask2,
        labels=full_labels2, position_ids=position_ids2,
    )
sync_print(f"Step 3b OK: ref logits={ref_outputs.logits.shape}")

sync_print("Step 3c: Log probs...")
per_token_logps4, comp_mask4 = compute_per_token_log_probs(outputs4.logits, full_labels2)
ref_logps4, _ = compute_per_token_log_probs(ref_outputs.logits, full_labels2)
sync_print(f"Step 3c OK: logps={per_token_logps4.shape}, ref_logps={ref_logps4.shape}")

sync_print("Step 4: Backward with everything...")
try:
    # Simulate GRPO loss (simplified)
    ratio = torch.exp(per_token_logps4 - per_token_logps4.detach())
    loss_simple = -(ratio * comp_mask4).sum() / comp_mask4.sum().clamp(min=1)
    loss_simple.backward()
    sync_print(f"Step 4 OK: backward done, loss={loss_simple.item():.4f}")
except Exception as e:
    sync_print(f"Step 4 FAILED: {e}")
    import traceback
    traceback.print_exc()

parallel_model.zero_grad()

# === Test 4: WITH gradient checkpointing + FULL GRPO loss ===
sync_print("\n========== TEST 4: Full GRPO loss computation ==========")
from grpo_loss import compute_group_advantages, compute_grpo_loss

sync_print("Running full GRPO step with G=2...")
parallel_model.eval()
completions_g, comp_ids_g, prompts_g, idxs_g = generate_completions(
    parallel_model.module, batch, tokenizer, configs, 2, device
)
sync_print(f"Generated {len(completions_g)} completions")

# Fake rewards (to avoid loading reward model)
rewards = torch.randn(len(completions_g), device=device)
sync_print(f"Rewards: shape={rewards.shape}")

advantages = compute_group_advantages(rewards, 2, scale_rewards="group")
sync_print(f"Advantages: shape={advantages.shape}")

full_ids_g, full_mask_g, full_labels_g, pos_ids_g = build_forward_inputs(
    parallel_model.module, tokenizer, batch, comp_ids_g, configs, device, collator=collator
)
sync_print(f"Forward inputs: {full_ids_g.shape}")

parallel_model.train()
sync_print("Policy forward...")
out_g = parallel_model(
    input_ids=full_ids_g, attention_mask=full_mask_g,
    labels=full_labels_g, position_ids=pos_ids_g,
)
sync_print(f"Policy forward OK: loss={out_g.loss.item():.4f}")

sync_print("Reference forward...")
with torch.no_grad():
    ref_out_g = ref_model(
        input_ids=full_ids_g, attention_mask=full_mask_g,
        labels=full_labels_g, position_ids=pos_ids_g,
    )
sync_print("Reference forward OK")

policy_logps, cmask = compute_per_token_log_probs(out_g.logits, full_labels_g)
ref_logps, _ = compute_per_token_log_probs(ref_out_g.logits, full_labels_g)
sync_print(f"Log probs OK: policy={policy_logps.shape}, ref={ref_logps.shape}")

sync_print("Computing GRPO loss...")
loss, metrics = compute_grpo_loss(
    per_token_logps=policy_logps,
    ref_per_token_logps=ref_logps,
    advantages=advantages,
    completion_mask=cmask,
    loss_type=configs.loss_type,
    beta=configs.beta,
    epsilon=getattr(configs, "epsilon", 0.2),
    max_completion_length=configs.max_new_tokens,
    gradient_accumulation_steps=configs.gradient_accumulation_steps,
)
sync_print(f"GRPO loss OK: loss={metrics['loss']:.4f}")

sync_print("Backward...")
loss.backward()
sync_print("Backward OK")

sync_print("\n========== ALL TESTS PASSED ==========")
dist.destroy_process_group()
