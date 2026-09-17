"""Quick test: verify the new forward method works with gradient checkpointing."""
import os, sys, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut_motion import CoconutMotion
from dataset import get_dataset, get_question_latent_dataset, MotionCollator
from grpo_loss import compute_per_token_log_probs, compute_group_advantages, compute_grpo_loss
from train_grpo import generate_completions, build_forward_inputs

torch.manual_seed(42)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def sync_print(msg):
    torch.cuda.synchronize()
    print(f"[rank {rank}] {msg}", flush=True)

with open("options/grpo/t2m_grpo.yaml") as f:
    cfg = yaml.safe_load(f)
configs = SimpleNamespace(**cfg)

# Smaller batch for memory
TEST_BATCH = 4
TEST_G = 2

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

sync_print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    configs.sft_checkpoint, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))

coconut_model = CoconutMotion(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

# Enable gradient checkpointing
coconut_model.base_causallm.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
sync_print("Gradient checkpointing enabled")

coconut_model = coconut_model.to(device)
parallel_model = DDP(coconut_model, device_ids=[local_rank], find_unused_parameters=False)

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

# Load data
train_tokenized = get_dataset(configs.train_path, tokenizer, max_steps=20)
ds = get_question_latent_dataset(configs.max_latent_stage, train_tokenized, configs, start_id, latent_id, end_id)
collator = MotionCollator(tokenizer=tokenizer, latent_id=latent_id)
sampler = DistributedSampler(ds, shuffle=False)
loader = DataLoader(ds, batch_size=TEST_BATCH, collate_fn=collator, sampler=sampler)
batch = next(iter(loader))

sync_print(f"Batch: input_ids={batch['input_ids'].shape}")

# === Full GRPO step ===
sync_print(f"\n=== Full GRPO step (batch={TEST_BATCH}, G={TEST_G}) ===")

sync_print("1. Generate...")
parallel_model.eval()
completions, comp_ids, prompts, idxs = generate_completions(
    parallel_model.module, batch, tokenizer, configs, TEST_G, device
)
sync_print(f"   OK: {len(completions)} completions")

sync_print("2. Build forward inputs...")
full_ids, full_mask, full_labels, pos_ids = build_forward_inputs(
    parallel_model.module, tokenizer, batch, comp_ids, configs, device, collator=collator
)
sync_print(f"   OK: shape={full_ids.shape}")

sync_print("3. Policy forward (grad ckpt)...")
parallel_model.train()
out = parallel_model(
    input_ids=full_ids, attention_mask=full_mask,
    labels=full_labels, position_ids=pos_ids,
)
sync_print(f"   OK: logits={out.logits.shape}, loss={out.loss.item():.4f}")

sync_print("4. Reference forward...")
with torch.no_grad():
    ref_out = ref_model(
        input_ids=full_ids, attention_mask=full_mask,
        labels=full_labels, position_ids=pos_ids,
    )
sync_print(f"   OK: ref logits={ref_out.logits.shape}")

sync_print("5. Log probs + GRPO loss...")
policy_logps, cmask = compute_per_token_log_probs(out.logits, full_labels)
ref_logps, _ = compute_per_token_log_probs(ref_out.logits, full_labels)

rewards = torch.randn(len(completions), device=device)
advantages = compute_group_advantages(rewards, TEST_G, scale_rewards="group")

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
sync_print(f"   OK: GRPO loss={metrics['loss']:.4f}")

sync_print("6. Backward...")
loss.backward()
sync_print("   OK: backward complete")

# Check gradients exist
grad_norm = sum(p.grad.norm().item() for p in coconut_model.parameters() if p.grad is not None)
sync_print(f"   Grad norm: {grad_norm:.4f}")

parallel_model.zero_grad()

mem_used = torch.cuda.max_memory_allocated() / 1024**3
sync_print(f"\nPeak GPU memory: {mem_used:.1f} GB")
sync_print("\n=== ALL TESTS PASSED ===")
dist.destroy_process_group()
