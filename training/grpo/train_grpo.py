"""GRPO Training for CoconutMotion.

Custom GRPO training loop for CoconutMotion's multi-pass forward architecture.
Uses TRL reward functions and TRL-inspired loss formulas (6 loss types).

Usage:
    torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml

Reference:
    - TRL grpo_trainer.py: loss formulas, advantage computation
    - latent-cot-motion train.py: model loading, data pipeline, validation
"""

import argparse
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
try:
    import wandb
except Exception:
    wandb = None

from coconut_motion import CoconutMotion
from dataset import (
    get_dataset,
    get_cot_latent_dataset,
    get_question_latent_dataset,
    MotionCollator,
)
from grpo_loss import (
    compute_per_token_log_probs,
    compute_group_advantages,
    compute_grpo_loss,
)
from rewards import RewardComputer
from unimo_rewards import UniMoRewardComputer
from temporal_rewards import TemporalGRPO, DirectTemporalReward
from utils import parse_motion_tokens
from _paths import project_path
from gradient_accumulation import forward_with_gradient_sync

_INPUT_CAPTION_PATTERN = re.compile(
    r"###\s*Input:\s*(.*?)(?:<\|im_end\|>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

_MOTION_TAG_PATTERN = re.compile(r"<Motion>(.*?)</Motion>", re.DOTALL)
_MOTION_CODE_PATTERN = re.compile(r"<Motion_\d+>")


def is_clean_motion(text):
    """Check if a completion has proper <Motion>...</Motion> with only motion codes inside.

    Returns False if:
      - No <Motion>...</Motion> tag at all
      - Tag content contains non-motion tokens (garbage text mixed in)
      - Repeated </Motion> tokens (length exploitation artifact)
      - Content after the closing </Motion> tag
    """
    if not text:
        return False
    # Reject repeated </Motion> — the model should produce exactly one
    if text.count("</Motion>") != 1:
        return False
    m = _MOTION_TAG_PATTERN.search(text)
    if not m:
        return False
    inner = m.group(1)
    # Remove all <Motion_N> codes; anything left (non-whitespace) is garbage
    residual = _MOTION_CODE_PATTERN.sub("", inner).strip()
    if residual or len(_MOTION_CODE_PATTERN.findall(inner)) == 0:
        return False
    # Reject junk after the closing tag
    close_idx = text.index("</Motion>")
    after = text[close_idx + len("</Motion>"):].strip()
    if after:
        return False
    return True


def normalize_motion_completion(text):
    """Ensure motion codes are wrapped in <Motion>...</Motion> for reward extraction.

    TRL reward functions use ``<Motion>(.*?)</Motion>`` to extract codes.
    If the model generates motion codes without proper wrapping (common early
    in GRPO when sampling), this function wraps them so the reward functions
    can still evaluate motion quality.

    Returns the original text if it already has proper wrapping or has no codes.
    """
    if _MOTION_TAG_PATTERN.search(text):
        return text
    codes = _MOTION_CODE_PATTERN.findall(text)
    if codes:
        return "<Motion>" + "".join(codes) + "</Motion>"
    return text


@torch.no_grad()
def run_greedy_sanity_check(model, tokenizer, dataset, collator, device):
    """Generate one sample with greedy decoding to verify the model works.

    This catches model loading issues, tokenizer mismatches, and other
    fundamental problems before the training loop wastes GPU time.
    """
    model.eval()
    sample = dataset[0]
    batch = collator([{
        "input_ids": list(sample["input_ids"]),
        "attention_mask": list(sample["attention_mask"]),
        "position_ids": list(sample["position_ids"]),
        "idx": sample["idx"],
    }])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Append <Motion> to prompt (same as generate_completions)
    motion_start_id = tokenizer.convert_tokens_to_ids("<Motion>")
    motion_col = torch.full(
        (1, 1), motion_start_id, device=device, dtype=input_ids.dtype,
    )
    input_ids = torch.cat([input_ids, motion_col], dim=1)
    attn_col = torch.ones(1, 1, device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, attn_col], dim=1)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    )

    prompt_len = input_ids.shape[1]
    completion_ids = outputs[0, prompt_len:].tolist()
    trimmed = []
    for tid in completion_ids:
        if tid == tokenizer.eos_token_id:
            break
        trimmed.append(tid)
    text = "<Motion>" + tokenizer.decode(trimmed, skip_special_tokens=False)

    has_tags = bool(_MOTION_TAG_PATTERN.search(text))
    codes = _MOTION_CODE_PATTERN.findall(text)
    print(f"\n[SANITY CHECK] Greedy generation (first sample):")
    print(f"  has_<Motion>_tag: {has_tags} | codes: {len(codes)}")
    print(f"  text: {text[:500]}{'...' if len(text) > 500 else ''}")
    if not has_tags and not codes:
        print("  ⚠ WARNING: Model produces no motion output with greedy decoding!")
        print("  Check: model loading, tokenizer, checkpoint compatibility.")
    print()
    return text


def setup_wandb(configs, rank):
    """Initialize Weights & Biases on rank 0 if enabled."""
    if rank != 0:
        return None
    if not getattr(configs, "use_wandb", True):
        return None
    if wandb is None:
        print("[wandb] WARNING: wandb is not installed, disabling wandb logging.")
        return None

    project = getattr(
        configs,
        "wandb_project",
        os.environ.get("WANDB_PROJECT", "latent-cot-grpo"),
    )
    run_name = getattr(
        configs, "wandb_run_name", getattr(configs, "name", "grpo-run")
    )
    entity = getattr(
        configs, "wandb_entity", os.environ.get("WANDB_ENTITY", None)
    )
    mode = getattr(configs, "wandb_mode", None)

    init_kwargs = {
        "project": project,
        "name": run_name,
        "config": vars(configs),
    }
    if entity:
        init_kwargs["entity"] = entity
    if mode:
        init_kwargs["mode"] = mode

    run = wandb.init(**init_kwargs)
    print(f"[wandb] initialized project={project}, run={run_name}")
    return run


# =====================================================================
# Config Loading
# =====================================================================
def load_config(config_path):
    def _coerce_scalar(x):
        if not isinstance(x, str):
            return x
        s = x.strip()
        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        # int
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except ValueError:
                return x
        # float (supports "5e-5", "1.0e-6", ".5", etc.)
        if re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s
        ) or re.fullmatch(r"[+-]?\d+[eE][+-]?\d+", s):
            try:
                return float(s)
            except ValueError:
                return x
        return x

    def _coerce_value(v):
        if isinstance(v, dict):
            return {k: _coerce_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_coerce_value(val) for val in v]
        return _coerce_scalar(v)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _coerce_value(cfg)
    return SimpleNamespace(**cfg)


# =====================================================================
# Model Setup (adapted from train.py setup_tokenizer_and_model)
# =====================================================================
def setup_tokenizer_and_model(configs):
    """Initialize tokenizer and model, matching SFT setup exactly."""
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load SFT checkpoint or base model
    sft_path = getattr(configs, "sft_checkpoint", None)
    if sft_path and os.path.exists(sft_path):
        print(f"Loading SFT checkpoint from: {sft_path}")
        model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            attn_implementation=getattr(
                configs, "attn_implementation", "flash_attention_2"
            ),
            trust_remote_code=True,
        )
    else:
        print(f"Loading base model: {configs.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            attn_implementation=getattr(
                configs, "attn_implementation", "flash_attention_2"
            ),
            trust_remote_code=True,
        )

    # Add special tokens (must match SFT exactly)
    tokenizer.add_tokens(["<Motion>", "</Motion>"])
    for i in range(configs.nb_code):
        tokenizer.add_tokens([f"<Motion_{i}>"])
    tokenizer.add_tokens(["<think>", "</think>"])
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    # Vocab size sanity check
    embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"[setup] tokenizer vocab={len(tokenizer)}, embedding rows={embed_size}")
    assert embed_size == len(tokenizer), (
        f"Embedding/tokenizer mismatch: {embed_size} vs {len(tokenizer)}"
    )

    # Get latent token IDs
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    print(f"[setup] latent_id={latent_id}, start_id={start_id}, end_id={end_id}")

    return model, tokenizer, latent_id, start_id, end_id


def setup_lora(model, configs):
    """Apply LoRA to the model's base_causallm."""
    from peft import LoraConfig, get_peft_model

    target_modules = getattr(
        configs, "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"],
    )

    lora_config = LoraConfig(
        r=getattr(configs, "lora_r", 16),
        lora_alpha=getattr(configs, "lora_alpha", 32),
        target_modules=target_modules,
        lora_dropout=getattr(configs, "lora_dropout", 0.05),
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the base causal LM inside CoconutMotion
    model.base_causallm = get_peft_model(model.base_causallm, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA applied: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


# =====================================================================
# Data Loading
# =====================================================================
def extract_caption_from_question(question: str) -> str:
    """Extract the text motion description from a chat-formatted prompt."""
    if not question:
        return ""
    match = _INPUT_CAPTION_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    return ""


def load_datasets(tokenizer, configs, latent_id, start_id, end_id):
    """Load train dataset and reward side inputs."""
    max_steps = getattr(configs, "max_think_steps", 20)

    # Tokenize raw JSON data → produces question_tokenized, steps_tokenized,
    # answer_tokenized, idx fields required by get_question_latent_dataset.
    train_tokenized = get_dataset(
        configs.train_path, tokenizer, max_steps=max_steps
    )

    # Build generation dataset (question + latent tokens only)
    scheduled_stage = configs.max_latent_stage  # Use max stage for GRPO

    train_dataset = get_question_latent_dataset(
        scheduled_stage, train_tokenized, configs, start_id, latent_id, end_id
    )

    # Extract ground truth motion codes (0..nb_code-1) for reward computation.
    # answer_tokenized contains *tokenizer IDs* — decode back to text and parse
    # to get the actual VQ-VAE code indices the TRL reward functions expect.
    train_ground_truths = {}
    for sample in train_tokenized:
        answer_text = tokenizer.decode(
            sample["answer_tokenized"], skip_special_tokens=False
        )
        motion_codes = parse_motion_tokens(answer_text)
        train_ground_truths[sample["idx"]] = motion_codes

    # Extract plain-language captions for semantic reward.
    with open(configs.train_path, encoding="utf-8") as f:
        raw_train_data = json.load(f)
    train_captions = {
        idx: extract_caption_from_question(sample.get("question", ""))
        for idx, sample in enumerate(raw_train_data)
    }

    # Load POS-tagged tokens from UniMo dataset for accurate text embedding.
    # UniMo's motion_train.json has pre-tokenized tokens like ["a/DET", "man/NOUN", ...].
    # Without this, all words default to POS=OTHER, which changes text embeddings.
    train_pos_tokens = {}
    unimo_tokens_path = getattr(configs, "unimo_tokens_path", None)
    matched, total = 0, len(train_captions)
    is_rank0 = (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()

    if unimo_tokens_path and os.path.exists(unimo_tokens_path):
        with open(unimo_tokens_path, encoding="utf-8") as f:
            unimo_data = json.load(f)
        caption_to_tokens = {
            item["caption"].strip(): item["tokens"]
            for item in unimo_data
            if "tokens" in item
        }
        for idx, caption in train_captions.items():
            tokens = caption_to_tokens.get(caption)
            if tokens is not None:
                train_pos_tokens[idx] = tokens
                matched += 1
    elif unimo_tokens_path and is_rank0:
        print(f"[POS tokens] WARNING: unimo_tokens_path not found: {unimo_tokens_path}")

    # Fallback: fill remaining from HumanML3D text files (caption#word/POS#f#t)
    if matched < total:
        configured_texts_dir = getattr(configs, "humanml3d_texts_dir", None)
        text_dir_candidates = []
        if configured_texts_dir:
            text_dir_candidates.append(configured_texts_dir)
        if unimo_tokens_path:
            _base = os.path.dirname(unimo_tokens_path)
            text_dir_candidates.append(
                os.path.join(_base, "dataset", "HumanML3D", "texts")
            )
        text_dir_candidates.append(
            project_path("dataset", "HumanML3D", "texts")
        )

        humanml3d_texts_dir = next(
            (d for d in text_dir_candidates if d and os.path.isdir(d)),
            None,
        )
        if humanml3d_texts_dir:
            hml_caption_to_tokens = {}
            for fname in os.listdir(humanml3d_texts_dir):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(humanml3d_texts_dir, fname)
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("#")
                        if len(parts) >= 2:
                            cap = parts[0].strip()
                            toks = parts[1].strip().split()
                            if cap and toks:
                                hml_caption_to_tokens[cap] = toks
            fallback_count = 0
            for idx, caption in train_captions.items():
                if idx not in train_pos_tokens:
                    toks = hml_caption_to_tokens.get(caption)
                    if toks is not None:
                        train_pos_tokens[idx] = toks
                        fallback_count += 1
            matched += fallback_count
            if is_rank0:
                print(
                    f"[POS tokens] Fallback from HumanML3D texts "
                    f"({humanml3d_texts_dir}): +{fallback_count}"
                )
        elif is_rank0:
            print("[POS tokens] WARNING: no HumanML3D texts directory found for fallback.")

    if is_rank0:
        print(f"[POS tokens] Total {matched}/{total} matched")

    return train_dataset, train_ground_truths, train_captions, train_pos_tokens


# =====================================================================
# GRPO Generation
# =====================================================================
@torch.no_grad()
def generate_completions(
    model, batch, tokenizer, configs, num_generations, device
):
    """Generate G completions per prompt using sampling.

    All G completions are generated in a single batched call.  The latent
    forward pass is computed once and the KV cache is replicated G times,
    avoiding redundant multi-pass computation.

    Args:
        model: CoconutMotion model (unwrapped)
        batch: dict with input_ids, attention_mask, idx
        tokenizer: tokenizer for decoding
        configs: configuration
        num_generations: G
        device: torch device

    Returns:
        all_completions: list of (B*G) generated text strings
        all_completion_ids: list of (B*G) raw completion token ID lists
        all_prompt_texts: list of (B*G) prompt texts (empty strings)
        all_idxs: list of (B*G) sample indices
    """
    model.eval()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    idx_list = batch["idx"].tolist()
    batch_size = input_ids.shape[0]

    # Append <Motion> to prompt as context.
    # In SFT, the model learned: ...[end_id]<Motion><Motion_N>...<Motion_N></Motion>
    # By including <Motion> in the prompt, the model's next-token distribution
    # after latent forward naturally favors <Motion_N> codes, avoiding the
    # unstable transition where sampling can go off-distribution into text.
    force_motion = getattr(configs, "force_motion_start", True)
    if force_motion:
        motion_start_id = tokenizer.convert_tokens_to_ids("<Motion>")
        # Append <Motion> token to input_ids
        motion_col = torch.full(
            (batch_size, 1), motion_start_id,
            device=device, dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, motion_col], dim=1)
        # Extend attention mask
        attn_col = torch.ones(
            (batch_size, 1), device=device, dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, attn_col], dim=1)

    prompt_seq_len = input_ids.shape[1]

    # Single batched generate call — latent forward computed once,
    # KV cache expanded G times internally.
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=configs.max_new_tokens,
        temperature=configs.temperature,
        do_sample=True,
        top_p=getattr(configs, "top_p", 1.0),
        top_k=getattr(configs, "top_k", 0),
        num_generations=num_generations,
    )
    # outputs shape: (B*G, max_gen_len)
    # Layout: [prompt0_gen0, prompt0_gen1, ..., prompt0_genG-1,
    #          prompt1_gen0, ...]  (repeat_interleave order)

    all_completions = []
    all_completion_ids = []
    all_prompt_texts = []
    all_idxs = []

    for i in range(batch_size * num_generations):
        full_ids = outputs[i]
        completion_token_ids = full_ids[prompt_seq_len:].tolist()
        # Trim trailing EOS padding
        trimmed_ids = []
        for tid in completion_token_ids:
            if tid == tokenizer.eos_token_id:
                break
            trimmed_ids.append(tid)

        # Prepend <Motion> tag since it's now part of the prompt, not completion
        completion_text = "<Motion>" + tokenizer.decode(
            trimmed_ids, skip_special_tokens=False
        )

        all_completions.append(completion_text)
        all_completion_ids.append(trimmed_ids)
        all_prompt_texts.append("")  # unused — skip expensive decode
        # repeat_interleave layout: sample index = i // G
        all_idxs.append(idx_list[i // num_generations])

    model.train()
    return all_completions, all_completion_ids, all_prompt_texts, all_idxs


# =====================================================================
# Build Labels for Log Prob Computation
# =====================================================================
def build_forward_inputs(
    model, tokenizer, batch, completion_ids_per_sample, configs, device,
    collator=None,
):
    """Build input_ids and labels for forward pass (log prob computation).

    For each sample: prompt (masked) + latent tokens (masked) + completion (trainable).
    If force_motion_start is enabled, include "<Motion>" in the context here too,
    so the log-prob context exactly matches generate_completions().
    Uses MotionCollator to left-pad and align latent token positions, which is
    required by CoconutMotion's multi-pass forward.

    Args:
        model: CoconutMotion
        tokenizer: tokenizer
        batch: original batch with input_ids
        completion_ids_per_sample: list of G raw completion token ID lists per sample
        configs: configuration
        device: torch device
        collator: MotionCollator for latent-aligned padding

    Returns:
        full_input_ids: (B*G, L) padded
        full_attention_mask: (B*G, L)
        full_labels: (B*G, L) with -100 for masked positions
        full_position_ids: (B*G, L) from MotionCollator (latent-aligned)
    """
    batch_size = batch["input_ids"].shape[0]
    G = len(completion_ids_per_sample) // batch_size

    max_seq_len = getattr(configs, "max_seq_len", 1024)
    force_motion = getattr(configs, "force_motion_start", True)
    motion_start_id = None
    if force_motion:
        motion_start_id = tokenizer.convert_tokens_to_ids("<Motion>")
    features = []

    for flat_idx in range(batch_size * G):
        prompt_idx = flat_idx // G

        # Prompt tokens (from batch, already includes latent tokens)
        prompt_ids = batch["input_ids"][prompt_idx][
            batch["attention_mask"][prompt_idx].bool()
        ].tolist()

        # Completion tokens — use raw IDs directly (no re-encoding)
        completion_ids = completion_ids_per_sample[flat_idx]

        # Keep forward context aligned with generation:
        # generate_completions() appends <Motion> before sampling.
        context_ids = prompt_ids
        if force_motion:
            context_ids = context_ids + [motion_start_id]

        # Full sequence
        full_ids = context_ids + completion_ids
        context_len = len(context_ids)

        # Truncate to max_seq_len
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]

        # Labels: -100 for prompt/latent/(optional <Motion>), train on completion
        labels = [-100] * context_len + completion_ids
        if len(labels) > max_seq_len:
            labels = labels[:max_seq_len]

        features.append({
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
            "position_ids": list(range(len(full_ids))),
            "idx": flat_idx,
        })

    # Use MotionCollator for latent-aligned left-padding
    padded = collator(features)

    return (
        padded["input_ids"].to(device),
        padded["attention_mask"].to(device),
        padded["labels"].to(device),
        padded["position_ids"].to(device),
    )


# =====================================================================
# Main Training Loop
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    configs = load_config(args.config)

    # === DDP Setup ===
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Pin TRL reward-side models to each rank's local GPU.
    os.environ["T2M_DEVICE"] = f"cuda:{local_rank}"
    print(f"[rank {rank}] T2M_DEVICE={os.environ['T2M_DEVICE']}")

    if rank == 0:
        print(f"GRPO Training | {world_size} GPUs | loss_type={configs.loss_type}")
        print(
            "Reward device policy: T2M models follow LOCAL_RANK "
            "(one reward model instance per rank)."
        )
        if getattr(configs, "show_progress_bar", True) and tqdm is None:
            print("[progress] WARNING: tqdm is not installed, progress bar disabled.")

    # === Seed ===
    torch.manual_seed(getattr(configs, "seed", 42) + rank)

    # === Model Setup ===
    base_model, tokenizer, latent_id, start_id, end_id = setup_tokenizer_and_model(
        configs
    )

    coconut_model = CoconutMotion(
        base_model, latent_id, start_id, end_id, tokenizer.eos_token_id
    )

    # Apply LoRA
    if getattr(configs, "use_lora", False):
        coconut_model = setup_lora(coconut_model, configs)

    # Gradient checkpointing (saves ~40% memory, required for large G)
    if getattr(configs, "gradient_checkpointing", False):
        coconut_model.base_causallm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == 0:
            print("Gradient checkpointing enabled")

    # torch.compile acceleration for the base causal LM
    if getattr(configs, "use_torch_compile", False):
        compile_mode = getattr(configs, "torch_compile_mode", "default")
        # Limit dynamo shape cache to avoid unbounded GPU memory growth
        # from variable-length completions triggering new compilations.
        torch._dynamo.config.cache_size_limit = getattr(
            configs, "dynamo_cache_size_limit", 64
        )
        coconut_model.base_causallm = torch.compile(
            coconut_model.base_causallm, mode=compile_mode, dynamic=True,
        )
        if rank == 0:
            print(f"torch.compile enabled (mode={compile_mode}, dynamic=True)")

    coconut_model = coconut_model.to(device)
    parallel_model = DDP(
        coconut_model, device_ids=[local_rank],
        find_unused_parameters=False,  # avoid conflict with gradient checkpointing
    )

    # === Reference Model (for KL penalty) ===
    ref_model = None
    if configs.beta > 0 and not getattr(configs, "use_lora", False):
        # No LoRA → need a separate frozen copy as reference model
        if rank == 0:
            print("Loading separate reference model (no LoRA, beta > 0)...")
        ref_base_model, _, _, _, _ = setup_tokenizer_and_model(configs)
        ref_model = CoconutMotion(
            ref_base_model, latent_id, start_id, end_id, tokenizer.eos_token_id
        )
        # torch.compile for reference model too
        if getattr(configs, "use_torch_compile", False):
            compile_mode = getattr(configs, "torch_compile_mode", "default")
            ref_model.base_causallm = torch.compile(
                ref_model.base_causallm, mode=compile_mode, dynamic=True,
            )
            if rank == 0:
                print(f"torch.compile enabled for reference model (mode={compile_mode}, dynamic=True)")
        ref_model = ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        if rank == 0:
            print("Reference model loaded and frozen")

    # === Reward ===
    reward_mode = getattr(configs, "reward_mode", "trl")  # "trl" or "unimo"
    if reward_mode == "unimo":
        unimo_weights = getattr(configs, "reward_weights", None)
        reward_computer = UniMoRewardComputer(device=device, reward_weights=unimo_weights)
        if rank == 0:
            print("[Reward] Using UniMo-style reward (motion_sim + text_sim)")
    else:
        custom_funcs = getattr(configs, "reward_funcs", None)
        custom_weights = getattr(configs, "reward_weights", None)
        if custom_funcs is not None:
            # Fully custom: funcs + weights from config
            reward_computer = RewardComputer(
                custom_funcs=custom_funcs,
                custom_weights=custom_weights,
            )
        elif custom_weights is not None:
            # Preset funcs, but override weights from config
            preset = getattr(configs, "reward_preset", "basic")
            reward_computer = RewardComputer(
                preset=preset,
                custom_weights=custom_weights,
            )
        else:
            # Pure preset
            reward_computer = RewardComputer(
                preset=getattr(configs, "reward_preset", "basic"),
            )

    # === Temporal Rewards ===
    use_temporal = getattr(configs, "use_temporal_reward", False)
    temporal_mode = getattr(configs, "temporal_mode", "both")  # "contrastive", "direct", "both"

    temporal_grpo = None
    direct_temporal = None

    if use_temporal and temporal_mode in ("contrastive", "both"):
        temporal_grpo = TemporalGRPO(
            reward_computer=reward_computer,
            mu=getattr(configs, "temporal_mu", 0.8),
            alpha=getattr(configs, "temporal_alpha", 0.3),
            theta=getattr(configs, "temporal_theta", 0.1),
        )
        if rank == 0:
            print(f"  T-GRPO contrastive enabled (mu={temporal_grpo.mu}, alpha={temporal_grpo.alpha})")

    if use_temporal and temporal_mode in ("direct", "both"):
        direct_temporal = DirectTemporalReward(
            weight=getattr(configs, "temporal_direct_weight", 0.2),
        )
        if rank == 0:
            print(f"  Direct temporal reward enabled (weight={direct_temporal.weight})")

    # === Data ===
    train_dataset, train_gts, train_captions, train_pos_tokens = load_datasets(
        tokenizer, configs, latent_id, start_id, end_id
    )

    collator = MotionCollator(
        tokenizer=tokenizer,
        latent_id=latent_id,
    )

    # === Greedy Sanity Check ===
    if rank == 0:
        run_greedy_sanity_check(
            coconut_model, tokenizer, train_dataset, collator, device,
        )

    # Synchronize so rank 0 sanity check finishes before training starts
    dist.barrier()

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size_training,
        collate_fn=collator,
        sampler=train_sampler,
        num_workers=getattr(configs, "train_num_workers", 4),
        prefetch_factor=getattr(configs, "dataloader_prefetch_factor", 2),
        persistent_workers=getattr(
            configs, "dataloader_persistent_workers", True
        ),
    )

    # === Optimizer ===
    trainable_params = [p for p in coconut_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=configs.lr,
        weight_decay=getattr(configs, "weight_decay", 0.01),
    )

    # Cosine LR scheduler
    total_steps = math.ceil(
        len(train_loader) * configs.num_epochs
        / configs.gradient_accumulation_steps
    )
    warmup_steps = getattr(configs, "warmup_steps", 50)
    # Support ratio (0~1) as fraction of total steps
    if isinstance(warmup_steps, float) and 0 < warmup_steps < 1:
        warmup_steps = int(warmup_steps * total_steps)
    warmup_steps = int(warmup_steps)
    min_lr = getattr(configs, "min_lr", configs.lr * 0.1)
    lr_decay_steps = int(getattr(configs, "lr_decay_steps", total_steps) or total_steps)
    lr_decay_steps = max(lr_decay_steps, warmup_steps + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(lr_decay_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / configs.lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    wandb_run = setup_wandb(configs, rank)

    # === Training Loop ===
    G = configs.num_generations
    grad_accum = configs.gradient_accumulation_steps
    # forbidden_generation_token_ids = [latent_id, start_id, end_id]
    global_step = 0
    accum_loss = 0.0
    accum_reward = 0.0
    accum_kl = 0.0
    accum_count = 0
    accum_reward_details = {}  # per-component reward accumulators
    accum_gen_tokens = 0
    accum_gen_completions = 0
    total_gen_tokens = 0
    total_gen_completions = 0

    save_dir = getattr(configs, "save_path", "./checkpoints/grpo")
    os.makedirs(save_dir, exist_ok=True)

    # Generation log: write all completions to a JSONL file for inspection
    gen_log_path = None
    gen_log_file = None
    if rank == 0:
        gen_log_path = os.path.join(save_dir, "generations.jsonl")
        gen_log_file = open(gen_log_path, "w", encoding="utf-8")

    if rank == 0:
        print(f"\nStarting GRPO training:")
        print(f"  Epochs: {configs.num_epochs}")
        print(f"  Batch size: {configs.batch_size_training}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Generations per prompt: {G}")
        print(f"  Loss type: {configs.loss_type}")
        print(f"  Beta (KL): {configs.beta}")
        print(f"  Temperature: {configs.temperature}")
        print(f"  Total steps: {total_steps}")
        print(
            f"  LR: {configs.lr:.6g} -> min {min_lr:.6g} by step {lr_decay_steps}"
        )
        print()

    for epoch in range(configs.num_epochs):
        train_sampler.set_epoch(epoch)
        parallel_model.train()
        epoch_start = time.time()
        pbar = None
        if (
            rank == 0
            and getattr(configs, "show_progress_bar", True)
            and tqdm is not None
        ):
            pbar = tqdm(
                total=len(train_loader),
                dynamic_ncols=True,
                desc=f"Epoch {epoch+1}/{configs.num_epochs}",
            )

        for step, batch in enumerate(train_loader):
            # === 1. Generate G completions per prompt ===
            completions, completion_ids, prompt_texts, idxs = generate_completions(
                parallel_model.module, batch, tokenizer, configs,
                G, device,
            )
            # Track generated completion lengths (in token IDs after prompt).
            step_gen_lengths_local = [len(ids) for ids in completion_ids]
            step_gen_tokens_local = sum(step_gen_lengths_local)
            step_gen_completions_local = len(step_gen_lengths_local)
            step_gen_len_min_local = (
                min(step_gen_lengths_local) if step_gen_lengths_local else 0
            )
            step_gen_len_max_local = (
                max(step_gen_lengths_local) if step_gen_lengths_local else 0
            )

            # Aggregate token-length stats across ranks so rank-0 logs global values.
            if dist.is_initialized():
                step_stats = torch.tensor(
                    [step_gen_tokens_local, step_gen_completions_local],
                    device=device, dtype=torch.float64,
                )
                dist.all_reduce(step_stats, op=dist.ReduceOp.SUM)
                step_gen_tokens = int(step_stats[0].item())
                step_gen_completions = int(step_stats[1].item())

                min_val = (
                    float(step_gen_len_min_local)
                    if step_gen_completions_local > 0
                    else float("inf")
                )
                min_stat = torch.tensor(
                    min_val, device=device, dtype=torch.float64
                )
                max_stat = torch.tensor(
                    float(step_gen_len_max_local), device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(min_stat, op=dist.ReduceOp.MIN)
                dist.all_reduce(max_stat, op=dist.ReduceOp.MAX)
                step_gen_len_min = (
                    0 if not math.isfinite(min_stat.item())
                    else int(min_stat.item())
                )
                step_gen_len_max = int(max_stat.item())
            else:
                step_gen_tokens = step_gen_tokens_local
                step_gen_completions = step_gen_completions_local
                step_gen_len_min = step_gen_len_min_local
                step_gen_len_max = step_gen_len_max_local
            step_gen_len_mean = (
                step_gen_tokens / max(step_gen_completions, 1)
            )

            # === 2. Compute Rewards ===
            # Get ground truth texts for reward computation
            nb_code = getattr(configs, "nb_code", 512)
            gt_texts = []
            captions = []
            pos_tokens_list = []
            for idx in idxs:
                gt_codes = train_gts.get(idx, [])
                # Convert VQ-VAE codes to motion token text for TRL rewards
                gt_text = " ".join(
                    [f"<Motion_{c}>" for c in gt_codes if 0 <= c < nb_code]
                )
                gt_texts.append(gt_text)
                captions.append(train_captions.get(idx, ""))
                pos_tokens_list.append(train_pos_tokens.get(idx))

            reward_kwargs = {"caption": captions, "pos_tokens": pos_tokens_list}

            rewards, reward_details = reward_computer.compute_detailed(
                completions, gt_texts, **reward_kwargs
            )

            # Zero out rewards for completions that are not clean motion output.
            # When reward_mode=unimo, skip this to match UniMo's behavior
            # (bad format only loses the format component, similarity still trains).
            if getattr(configs, "reward_mode", "trl") != "unimo":
                for i, c in enumerate(completions):
                    if not is_clean_motion(c):
                        rewards[i] = 0.0

            # If compute_detailed didn't provide a format key, fill it from
            # is_clean_motion so the log always has one.
            if "format" not in reward_details:
                reward_details["format"] = [
                    1.0 if is_clean_motion(c) else 0.0 for c in completions
                ]

            rewards = rewards.to(device)

            # Diagnostic: show completion quality on first few steps
            if rank == 0 and step < 3:
                n_total = len(completions)
                n_has_motion_tag = sum(
                    1 for c in completions
                    if _MOTION_TAG_PATTERN.search(c)
                )
                n_has_codes = sum(
                    1 for c in completions if _MOTION_CODE_PATTERN.search(c)
                )
                avg_codes = sum(
                    len(_MOTION_CODE_PATTERN.findall(c)) for c in completions
                ) / max(n_total, 1)
                n_zeroed = sum(1 for c in completions if not is_clean_motion(c))
                print(f"  [diag] completions={n_total} | "
                      f"has_<Motion>_tag={n_has_motion_tag}/{n_total} | "
                      f"has_codes={n_has_codes}/{n_total} | "
                      f"avg_codes={avg_codes:.1f} | "
                      f"zeroed={n_zeroed}/{n_total}")
                # Print first completion as sample
                sample = completions[0]
                print(f"  [diag] sample completion: {sample[:300]}{'...' if len(sample) > 300 else ''}")

            # Write all completions to generation log
            if gen_log_file is not None:
                rewards_list = rewards.cpu().tolist()
                per_func_rewards = {
                    k: [float(v) for v in vals]
                    for k, vals in reward_details.items()
                }
                for i, (comp, caption, idx, r) in enumerate(
                    zip(completions, captions, idxs, rewards_list)
                ):
                    record = {
                        "epoch": epoch + 1,
                        "step": step,
                        "iter": epoch * len(train_loader) + step + 1,
                        "sample_idx": idx,
                        "gen_id": i % G,
                        "caption": caption,
                        "completion": comp,
                        "reward": round(r, 4),
                    }
                    for fname, fvals in per_func_rewards.items():
                        record[f"r_{fname}"] = round(fvals[i], 4)
                    gen_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                gen_log_file.flush()

            # === 2b. Apply Temporal Rewards (if enabled) ===
            if temporal_grpo is not None:
                rewards, t_active = temporal_grpo.apply(
                    completions, gt_texts, rewards, **reward_kwargs
                )
                if (
                    rank == 0
                    and t_active
                    and getattr(configs, "log_interval", 10)
                    and global_step % getattr(configs, "log_interval", 10) == 0
                ):
                    print("  [T-GRPO] Temporal bonus applied")

            if direct_temporal is not None:
                rewards = direct_temporal.apply(completions, rewards)

            # === 3. Compute Group Advantages ===
            advantages = compute_group_advantages(
                rewards, G,
                scale_rewards=getattr(configs, "scale_rewards", "group"),
            )

            # === 4. Build forward inputs ===
            full_ids, full_mask, full_labels, position_ids = build_forward_inputs(
                parallel_model.module, tokenizer, batch,
                completion_ids, configs, device, collator=collator,
            )

            # === 5. Compute Policy Log Probs ===
            is_accumulation_boundary = (
                (step + 1) % grad_accum == 0
                or (step + 1) == len(train_loader)
            )
            outputs = forward_with_gradient_sync(
                parallel_model,
                sync_gradients=is_accumulation_boundary,
                input_ids=full_ids,
                attention_mask=full_mask,
                labels=full_labels,
                position_ids=position_ids,
                compute_ce_loss=False,
            )
            per_token_logps, completion_mask = compute_per_token_log_probs(
                outputs.logits,
                full_labels,
                # forbidden_token_ids=forbidden_generation_token_ids,
            )
            del outputs  # Free logits & inputs_embeds early (graph retained via per_token_logps)

            # === 6. Compute Reference Log Probs (KL) ===
            ref_per_token_logps = None
            if configs.beta > 0:
                with torch.no_grad():
                    if getattr(configs, "use_lora", False):
                        # LoRA: disable adapter to get reference logits
                        with parallel_model.module.base_causallm.disable_adapter():
                            ref_outputs = parallel_model.module(
                                input_ids=full_ids,
                                attention_mask=full_mask,
                                labels=full_labels,
                                position_ids=position_ids,
                                compute_ce_loss=False,
                            )
                    else:
                        # No LoRA: use separate frozen reference model
                        ref_outputs = ref_model(
                            input_ids=full_ids,
                            attention_mask=full_mask,
                            labels=full_labels,
                            position_ids=position_ids,
                            compute_ce_loss=False,
                        )
                    ref_per_token_logps, _ = compute_per_token_log_probs(
                        ref_outputs.logits,
                        full_labels,
                        # forbidden_token_ids=forbidden_generation_token_ids,
                    )
                    del ref_outputs  # Free logits immediately (no grad graph under no_grad)

            # === 7. Compute GRPO Loss ===
            loss, metrics = compute_grpo_loss(
                per_token_logps=per_token_logps,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                completion_mask=completion_mask,
                loss_type=configs.loss_type,
                beta=configs.beta,
                epsilon=getattr(configs, "epsilon", 0.2),
                max_completion_length=configs.max_new_tokens,
                gradient_accumulation_steps=grad_accum,
                kl_clip_max=getattr(configs, "kl_clip_max", 10.0),
            )

            # === 8. Backward + Accumulate ===
            # Synchronization was selected before the policy forward pass.
            loss.backward()

            # Free large tensors before the next step's generation phase to
            # avoid holding ~2×(B*G×L×V) of dead GPU memory alongside new
            # KV-cache allocations.  ref_per_token_logps is still needed by
            # the accumulator metrics above, but per_token_logps / loss /
            # full forward tensors are no longer needed.
            del loss, per_token_logps, completion_mask, ref_per_token_logps
            del full_ids, full_mask, full_labels, position_ids
            del completions, completion_ids, prompt_texts, advantages

            accum_loss += metrics["loss"]
            reward_mean = rewards.mean().item()
            accum_reward += reward_mean
            metrics["reward"] = reward_mean
            accum_kl += metrics.get("kl", 0.0)
            accum_count += 1
            accum_gen_tokens += step_gen_tokens
            accum_gen_completions += step_gen_completions
            total_gen_tokens += step_gen_tokens
            total_gen_completions += step_gen_completions

            # Accumulate per-component reward means
            step_reward_details = {}
            for rname, rvals in reward_details.items():
                rmean = sum(rvals) / max(len(rvals), 1)
                step_reward_details[rname] = rmean
                accum_reward_details[rname] = accum_reward_details.get(rname, 0.0) + rmean

            # Per-iteration log (every micro-step)
            if rank == 0:
                detail_str = " | ".join(
                    f"r_{k}={v:.4f}" for k, v in step_reward_details.items()
                )
                print(
                    f"  [iter {step+1}/{len(train_loader)}] "
                    f"loss={metrics['loss']:.4f} | "
                    f"reward={reward_mean:.4f} | "
                    f"gen_len={step_gen_len_mean:.1f} "
                    f"(min={step_gen_len_min}, max={step_gen_len_max}) | "
                    f"{detail_str} | "
                    f"kl={metrics.get('kl', 0.0):.4f}"
                )

            del rewards, reward_details

            # === 9. Gradient step ===
            if is_accumulation_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    getattr(configs, "grad_clip", 1.0),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=getattr(configs, "zero_grad_set_to_none", True)
                )
                global_step += 1
                avg_reward_details = {}  # reset; populated at log_interval
                avg_gen_len = None

                # Periodically release CUDA cached blocks to mitigate
                # fragmentation from variable-length completion tensors.
                if global_step % getattr(configs, "empty_cache_interval", 10) == 0:
                    torch.cuda.empty_cache()

                # Logging
                log_interval = getattr(configs, "log_interval", 10)
                if rank == 0 and global_step % log_interval == 0:
                    avg_loss = accum_loss / max(accum_count, 1)
                    avg_reward = accum_reward / max(accum_count, 1)
                    avg_kl = accum_kl / max(accum_count, 1)
                    lr_now = scheduler.get_last_lr()[0]
                    avg_gen_len = (
                        accum_gen_tokens / max(accum_gen_completions, 1)
                    )
                    total_avg_gen_len = (
                        total_gen_tokens / max(total_gen_completions, 1)
                    )
                    # Per-component reward averages
                    avg_reward_details = {
                        k: v / max(accum_count, 1)
                        for k, v in accum_reward_details.items()
                    }
                    detail_str = " | ".join(
                        f"r_{k}={v:.4f}" for k, v in avg_reward_details.items()
                    )
                    print(
                        f"[Epoch {epoch+1}/{configs.num_epochs}] "
                        f"Step {global_step}/{total_steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"reward={avg_reward:.4f} | "
                        f"gen_len={avg_gen_len:.1f} | "
                        f"gen_tok_total={total_gen_tokens} | "
                        f"gen_len_total={total_avg_gen_len:.1f} | "
                        f"{detail_str} | "
                        f"kl={avg_kl:.4f} | "
                        f"lr={lr_now:.6e} | "
                        f"mem={torch.cuda.max_memory_allocated(device) / 1024**3:.1f}GB"
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                    accum_loss = 0.0
                    accum_reward = 0.0
                    accum_kl = 0.0
                    accum_count = 0
                    accum_reward_details = {}
                    accum_gen_tokens = 0
                    accum_gen_completions = 0

                if rank == 0 and pbar is not None:
                    pbar.set_postfix(
                        step=f"{global_step}/{total_steps}",
                        loss=f"{metrics['loss']:.4f}",
                        reward=f"{metrics['reward']:.4f}",
                        gen_len=f"{step_gen_len_mean:.1f}",
                        kl=f"{metrics.get('kl', 0.0):.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.6e}",
                    )

                if rank == 0 and wandb_run is not None:
                    wandb_log_dict = {
                        "train/loss": float(metrics["loss"]),
                        "train/reward": float(metrics["reward"]),
                        "train/kl": float(metrics.get("kl", 0.0)),
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "train/grad_norm": float(grad_norm.item()),
                        "train/epoch": float(epoch + 1),
                        "train/global_step": float(global_step),
                        "train/gen_len_mean_step": float(step_gen_len_mean),
                        "train/gen_len_min_step": float(step_gen_len_min),
                        "train/gen_len_max_step": float(step_gen_len_max),
                        "train/generated_tokens_total": float(total_gen_tokens),
                        "system/gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    }
                    if avg_gen_len is not None:
                        wandb_log_dict["train/gen_len_mean_window"] = float(avg_gen_len)
                    # Log per-component reward averages (computed at log_interval)
                    if avg_reward_details:
                        for rname, rval in avg_reward_details.items():
                            wandb_log_dict[f"reward/{rname}"] = rval
                    wandb_run.log(wandb_log_dict, step=global_step)

                # Save checkpoint
                save_interval = getattr(configs, "save_interval", 500)
                if (
                    save_interval > 0
                    and global_step % save_interval == 0
                    and rank == 0
                ):
                    ckpt_dir = os.path.join(
                        save_dir, f"step-{global_step}"
                    )
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Save LoRA adapter or full model
                    if getattr(configs, "use_lora", False):
                        parallel_model.module.base_causallm.save_pretrained(
                            ckpt_dir
                        )
                    else:
                        parallel_model.module.base_causallm.save_pretrained(
                            ckpt_dir
                        )
                    tokenizer.save_pretrained(ckpt_dir)
                    # training_state.pt (optimizer state + step/epoch) is only
                    # needed to resume training. It is large (full AdamW state
                    # ~2x model size). Skip it when save_training_state=false to
                    # save disk; HF weights above are enough for eval/inference.
                    if getattr(configs, "save_training_state", True):
                        torch.save(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            os.path.join(ckpt_dir, "training_state.pt"),
                        )
                    print(f"  Saved checkpoint: {ckpt_dir}")

                    # Enforce save_total_limit: remove oldest checkpoints
                    save_total_limit = getattr(configs, "save_total_limit", 0)
                    if save_total_limit > 0:
                        import glob
                        existing = sorted(
                            glob.glob(os.path.join(save_dir, "step-*")),
                            key=os.path.getmtime,
                        )
                        while len(existing) > save_total_limit:
                            oldest = existing.pop(0)
                            import shutil
                            shutil.rmtree(oldest, ignore_errors=True)
                            print(f"  Removed old checkpoint: {oldest}")

            if pbar is not None:
                pbar.update(1)

        epoch_time = time.time() - epoch_start
        if pbar is not None:
            pbar.close()
        if rank == 0:
            print(f"Epoch {epoch+1} done ({epoch_time:.1f}s)")

    # === Final Save ===
    if rank == 0:
        final_dir = os.path.join(save_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        parallel_model.module.base_causallm.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training complete. Final model saved to: {final_dir}")
        if gen_log_file is not None:
            gen_log_file.close()
            print(f"Generation log saved to: {gen_log_path}")
        if wandb_run is not None:
            wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
