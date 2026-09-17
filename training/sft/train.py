"""Coconut-style latent SFT training for Text-to-Motion.

Main training script with:
  - FSDP/DDP distributed training
  - Curriculum-based latent token replacement
  - Motion-specific validation metrics
  - Checkpoint management

Usage:
    torchrun --nproc_per_node=4 train.py configs/t2m_coconut.yaml

Reference: Coconut/run.py
"""

import argparse
import functools
import gc
import json
import math
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut_motion import CoconutMotion
from dataset import (
    MotionCollator,
    get_cot_latent_dataset,
    get_dataset,
    get_question_latent_dataset,
)
from utils import (
    Config,
    compute_motion_accuracy,
    compute_scheduled_stage_by_step,
    format_stage_info_by_step,
    parse_motion_tokens,
    set_seed,
)

try:
    from peft import LoraConfig, get_peft_model, PeftModel

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


GEN_FAIL_RATIO_THRESHOLD = 0.05


def _is_none_like(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"none", ""}


def _infer_epoch_from_path(path_like) -> int | None:
    match = re.search(r"checkpoint-epoch(\d+)", str(path_like))
    if not match:
        return None
    return int(match.group(1))


def _fmt_metric(v):
    """Format a metric value, showing '-' for NaN (empty region)."""
    return f"{v:.4f}" if not math.isnan(v) else "-"


@torch.no_grad()
def _compute_train_metrics(logits, labels, motion_start_id, motion_end_id):
    """Compute decomposed training metrics (UniMo-style).

    Splits trainable tokens into CoT region (before <Motion>) and
    Answer region (<Motion> onward, including </Motion> and <|im_end|>),
    computing accuracy and loss separately for each.

    As the curriculum progresses, CoT tokens are replaced by latent tokens
    (masked as -100 in labels), so n_cot shrinks toward 0. The returned
    token counts let the caller distinguish "low accuracy" from "no tokens
    in this region".

    Returns dict with: gen_acc, cot_acc, answer_acc, cot_loss, answer_loss,
                        n_cot, n_answer, n_total.
    """
    # Shift logits/labels for next-token prediction alignment
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    preds = shift_logits.argmax(dim=-1)  # (B, T-1)
    valid_mask = shift_labels != -100     # (B, T-1)

    # Build per-sample answer mask: from <Motion> to end of sequence.
    # This includes <Motion>, all motion codes, </Motion>, and <|im_end|>.
    # Without this, <|im_end|> would be misclassified as "cot".
    answer_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    for i in range(shift_labels.size(0)):
        lab = shift_labels[i]
        starts = (lab == motion_start_id).nonzero(as_tuple=False)
        if starts.numel() > 0:
            s = starts[0].item()
            answer_mask[i, s:] = True

    cot_mask = valid_mask & ~answer_mask
    ans_mask = valid_mask & answer_mask

    # Token counts per region
    n_cot = int(cot_mask.sum().item())
    n_answer = int(ans_mask.sum().item())
    n_total = int(valid_mask.sum().item())

    # Accuracy
    correct = preds == shift_labels
    gen_acc = correct[valid_mask].float().mean().item() if valid_mask.any() else 0.0
    cot_acc = correct[cot_mask].float().mean().item() if n_cot > 0 else float("nan")
    answer_acc = correct[ans_mask].float().mean().item() if n_answer > 0 else float("nan")

    # Per-region loss
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_cot = cot_mask.view(-1)
    flat_ans = ans_mask.view(-1)

    cot_loss = float("nan")
    if n_cot > 0:
        cot_loss = torch.nn.functional.cross_entropy(
            flat_logits[flat_cot], flat_labels[flat_cot],
        ).item()
    answer_loss = float("nan")
    if n_answer > 0:
        answer_loss = torch.nn.functional.cross_entropy(
            flat_logits[flat_ans], flat_labels[flat_ans],
        ).item()

    return {
        "gen_acc": gen_acc,
        "cot_acc": cot_acc,
        "answer_acc": answer_acc,
        "cot_loss": cot_loss,
        "answer_loss": answer_loss,
        "n_cot": n_cot,
        "n_answer": n_answer,
        "n_total": n_total,
    }


def _compute_lr(
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    update_step: int,
) -> float:
    """Compute lr with linear warmup then cosine annealing to min_lr."""
    if warmup_steps > 0 and update_step < warmup_steps:
        return base_lr * float(update_step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = float(update_step - warmup_steps) / float(total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _apply_lr(
    optimizer, base_lr: float, min_lr: float,
    warmup_steps: int, total_steps: int, update_step: int,
):
    lr = _compute_lr(base_lr, min_lr, warmup_steps, total_steps, update_step)
    for group in optimizer.param_groups:
        group["lr"] = lr


def _as_bool(value, default: bool) -> bool:
    """Parse config bools robustly (supports bool/str/int)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _build_dataloader_kwargs(
    configs,
    num_workers_key: str,
    default_num_workers: int,
    *,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """Build DataLoader kwargs with guarded worker-only options."""
    num_workers = max(int(getattr(configs, num_workers_key, default_num_workers)), 0)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if drop_last:
        kwargs["drop_last"] = True
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(
            int(getattr(configs, "dataloader_prefetch_factor", 4)), 1
        )
        kwargs["persistent_workers"] = _as_bool(
            getattr(configs, "dataloader_persistent_workers", True), True
        )
    return kwargs


def _unfreeze_new_token_embeddings(peft_model, nb_text_tokens):
    """Unfreeze embed_tokens and lm_head for newly added token rows.

    PEFT freezes the entire base model. This function re-enables gradients
    for the embedding and lm_head weight tensors so that the rows
    corresponding to new tokens (index >= nb_text_tokens) can be trained.

    Following Motion-Agent's approach: the entire embedding / lm_head
    parameter tensor is set to requires_grad=True, but only the new rows
    will receive meaningful gradients because the original rows are never
    referenced in loss computation (labels mask them out).
    """
    embed = peft_model.get_input_embeddings()
    embed.weight.requires_grad_(True)

    # lm_head location differs between PeftModel and base model
    if hasattr(peft_model, "lm_head"):
        lm_head = peft_model.lm_head
    else:
        lm_head = peft_model.base_model.model.lm_head
    lm_head.weight.requires_grad_(True)


def _save_lora_checkpoint(peft_model, nb_text_tokens, ckpt_dir):
    """Save LoRA adapters + new token embeddings (Motion-Agent pattern).

    Saves:
      - ``adapter_model/``: PEFT adapter weights via ``save_pretrained``
      - ``new_token_embeddings.pt``: embed_tokens and lm_head rows for
        tokens with index >= nb_text_tokens
    """
    # 1. Save LoRA adapter via PEFT's own method
    adapter_dir = os.path.join(ckpt_dir, "adapter_model")
    peft_model.save_pretrained(adapter_dir)

    # 2. Save new token embedding rows
    embed_weight = peft_model.get_input_embeddings().weight.data
    if hasattr(peft_model, "lm_head"):
        lm_head_weight = peft_model.lm_head.weight.data
    else:
        lm_head_weight = peft_model.base_model.model.lm_head.weight.data

    torch.save(
        {
            "embeddings": embed_weight[nb_text_tokens:].cpu(),
            "lm_head": lm_head_weight[nb_text_tokens:].cpu(),
            "nb_text_tokens": nb_text_tokens,
        },
        os.path.join(ckpt_dir, "new_token_embeddings.pt"),
    )


def _load_lora_checkpoint(peft_model, nb_text_tokens, ckpt_dir, rank=0):
    """Load LoRA adapters + new token embeddings from checkpoint.

    Expects the directory layout produced by ``_save_lora_checkpoint``.
    """
    adapter_dir = os.path.join(ckpt_dir, "adapter_model")
    emb_path = os.path.join(ckpt_dir, "new_token_embeddings.pt")

    # 1. Load LoRA adapter weights
    if os.path.isdir(adapter_dir):
        from peft import set_peft_model_state_dict
        adapter_state = torch.load(
            os.path.join(adapter_dir, "adapter_model.safetensors")
            if os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
            else os.path.join(adapter_dir, "adapter_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        set_peft_model_state_dict(peft_model, adapter_state)
        if rank == 0:
            print(f"  Loaded LoRA adapter from {adapter_dir}")

    # 2. Load new token embeddings
    if os.path.isfile(emb_path):
        saved = torch.load(emb_path, map_location="cpu", weights_only=True)
        embed = peft_model.get_input_embeddings()
        if hasattr(peft_model, "lm_head"):
            lm_head = peft_model.lm_head
        else:
            lm_head = peft_model.base_model.model.lm_head
        embed.weight.data[nb_text_tokens:] = saved["embeddings"].to(embed.weight.dtype)
        lm_head.weight.data[nb_text_tokens:] = saved["lm_head"].to(lm_head.weight.dtype)
        if rank == 0:
            print(f"  Loaded new token embeddings from {emb_path}")


def _build_model_load_kwargs(configs):
    """Build common kwargs for AutoModelForCausalLM.from_pretrained."""
    kwargs = {
        "torch_dtype": torch.bfloat16 if configs.bf16 else torch.float32,
        "trust_remote_code": True,
    }
    attn_impl = getattr(configs, "attn_implementation", "flash_attention_2")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def _get_training_state_path(load_path: Path) -> Path | None:
    if load_path.is_dir():
        candidate = load_path / "training_state.pt"
        return candidate if candidate.exists() else None
    if load_path.is_file():
        if load_path.name == "training_state.pt":
            return load_path
        candidate = load_path.parent / "training_state.pt"
        return candidate if candidate.exists() else None
    return None


def maybe_load_checkpoint(model, configs, rank, nb_text_tokens=0):
    """Load model/training states from checkpoint if configured.

    Supports:
      - FSDP save dir containing `model_state.pt`
      - DDP save_pretrained dir
      - LoRA checkpoint dir containing `adapter_model/` + `new_token_embeddings.pt`
      - Direct state-dict file path (`*.pt`, `*.pth`, `*.bin`)
    """
    out = {
        "loaded": False,
        "loaded_epoch": None,
        "optimizer_state_dict": None,
        "global_step": 0,
    }

    if not hasattr(configs, "load_model_path") or _is_none_like(configs.load_model_path):
        return out

    load_path = Path(str(configs.load_model_path))
    if not load_path.exists():
        raise FileNotFoundError(f"load_model_path does not exist: {load_path}")

    training_state_path = _get_training_state_path(load_path)
    if training_state_path is not None:
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        out["loaded_epoch"] = training_state.get("epoch")
        out["optimizer_state_dict"] = training_state.get("optimizer_state_dict")
        out["global_step"] = int(training_state.get("global_step", 0))

    # Check for LoRA checkpoint format
    use_lora = _as_bool(getattr(configs, "use_lora", False), False)
    if use_lora and load_path.is_dir() and (load_path / "adapter_model").is_dir():
        target = model.base_causallm if hasattr(model, "base_causallm") else model
        _load_lora_checkpoint(target, nb_text_tokens, str(load_path), rank)
        out["loaded"] = True
        if out["loaded_epoch"] is None:
            out["loaded_epoch"] = _infer_epoch_from_path(load_path)
        if rank == 0:
            print(f"Loaded LoRA checkpoint from: {load_path}")
        return out

    load_result = None
    loaded_source = None
    model_load_kwargs = _build_model_load_kwargs(configs)

    if load_path.is_dir():
        fsdp_state_path = load_path / "model_state.pt"
        if fsdp_state_path.exists():
            state_dict = torch.load(fsdp_state_path, map_location="cpu", weights_only=True)
            load_result = model.load_state_dict(state_dict, strict=False)
            loaded_source = fsdp_state_path
        else:
            # DDP-style save_pretrained directory (load into base model weights).
            target_model = model.base_causallm if hasattr(model, "base_causallm") else model
            loaded_model = AutoModelForCausalLM.from_pretrained(
                str(load_path),
                **model_load_kwargs,
            )
            load_result = target_model.load_state_dict(loaded_model.state_dict(), strict=False)
            loaded_source = load_path
            del loaded_model
    elif load_path.is_file():
        if load_path.name == "training_state.pt":
            model_state_candidate = load_path.parent / "model_state.pt"
            if model_state_candidate.exists():
                state_dict = torch.load(model_state_candidate, map_location="cpu", weights_only=True)
                load_result = model.load_state_dict(state_dict, strict=False)
                loaded_source = model_state_candidate
            else:
                target_model = model.base_causallm if hasattr(model, "base_causallm") else model
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    str(load_path.parent),
                    **model_load_kwargs,
                )
                load_result = target_model.load_state_dict(loaded_model.state_dict(), strict=False)
                loaded_source = load_path.parent
                del loaded_model
        else:
            state_dict = torch.load(load_path, map_location="cpu", weights_only=True)
            load_result = model.load_state_dict(state_dict, strict=False)
            loaded_source = load_path
    else:
        raise ValueError(f"Unsupported load_model_path: {load_path}")

    out["loaded"] = True
    if out["loaded_epoch"] is None:
        out["loaded_epoch"] = _infer_epoch_from_path(load_path)

    if rank == 0:
        print(f"Loaded weights from: {loaded_source}")
        if load_result is not None:
            print(
                "  load_state_dict summary:"
                f" missing={len(load_result.missing_keys)}"
                f" unexpected={len(load_result.unexpected_keys)}"
            )
        if out["loaded_epoch"] is not None:
            print(f"  checkpoint epoch: {out['loaded_epoch']}")
        if out["optimizer_state_dict"] is not None:
            print("  optimizer state found in training_state.pt")

    return out


def resolve_start_epoch(config_resume: int, only_eval: bool, loaded_epoch: int | None) -> int:
    """Resolve start epoch from config resume and loaded checkpoint epoch."""
    start_epoch = int(config_resume)
    if loaded_epoch is not None and start_epoch == 0:
        return loaded_epoch if only_eval else loaded_epoch + 1
    return start_epoch


def build_epoch_schedule(start_epoch: int, num_epochs: int, only_eval: bool):
    """Build epoch iterator for training/eval modes."""
    if only_eval:
        return [start_epoch]
    if start_epoch >= num_epochs:
        return []
    return range(start_epoch, num_epochs)


def setup_tokenizer_and_model(configs):
    """Initialize Qwen tokenizer and model with all special tokens.

    Token additions (in order):
      1. Motion markers: <Motion>, </Motion>
      2. Motion codes: <Motion_0> through <Motion_511>
      3. Think markers: <think>, </think>
      4. Latent tokens: <|start-latent|>, <|end-latent|>, <|latent|>

    Latent token embeddings are initialized from the '<<' token
    (following the Coconut paper).

    When ``configs.use_lora`` is True, LoRA adapters are applied to the base
    model **before** ``resize_token_embeddings`` so that the newly added rows
    in ``embed_tokens`` and ``lm_head`` are regular (non-LoRA) parameters and
    remain trainable while the rest of the base model is frozen.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_load_kwargs = _build_model_load_kwargs(configs)
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        **model_load_kwargs,
    )

    # Record original vocab size before adding new tokens
    nb_text_tokens = len(tokenizer)

    # Add motion tokens (following UniMo mllm.py)
    tokenizer.add_tokens(["<Motion>", "</Motion>"])
    for i in range(configs.nb_code):
        tokenizer.add_tokens([f"<Motion_{i}>"])
    tokenizer.add_tokens(["<think>", "</think>"])

    # Add latent tokens
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])

    # --- LoRA: apply BEFORE resize so adapter wraps the original layers,
    #     and the resized embedding rows stay as plain trainable parameters.
    use_lora = _as_bool(getattr(configs, "use_lora", False), False)
    if use_lora:
        if not HAS_PEFT:
            raise ImportError("peft is required for LoRA training. Install with: pip install peft>=0.10.0")
        lora_config = LoraConfig(
            r=int(getattr(configs, "lora_rank", 64)),
            lora_alpha=int(getattr(configs, "lora_alpha", 64)),
            target_modules=getattr(configs, "lora_target_modules",
                                   ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "up_proj", "down_proj", "gate_proj"]),
            lora_dropout=float(getattr(configs, "lora_dropout", 0.1)),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Resize embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))

    # When using LoRA, ensure new token embeddings are trainable.
    # PEFT freezes the base model, so we explicitly unfreeze embed_tokens
    # and lm_head for the newly added token rows.
    if use_lora:
        _unfreeze_new_token_embeddings(model, nb_text_tokens)

    # Get latent token IDs
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Initialize latent token embeddings from '<<' token (Coconut strategy)
    if configs.coconut:
        target_id = tokenizer.convert_tokens_to_ids("<<")
        if target_id is not None and target_id != tokenizer.unk_token_id:
            base = model.get_input_embeddings()
            # For PeftModel, lm_head is on the base_model.model level
            lm_head = model.lm_head if hasattr(model, "lm_head") else model.base_model.model.lm_head
            target_embedding = base.weight.data[target_id].clone()
            for token_id in [latent_id, start_id, end_id]:
                base.weight.data[token_id] = target_embedding.clone()
                lm_head.weight.data[token_id] = (
                    lm_head.weight.data[target_id].clone()
                )

    # Get think tag token IDs for dataset construction
    think_open_ids = tokenizer.encode("<think>\n", add_special_tokens=False)
    think_close_ids = tokenizer.encode("\n</think>", add_special_tokens=False)

    return model, tokenizer, latent_id, start_id, end_id, think_open_ids, think_close_ids, nb_text_tokens


def validate_loss(
    parallel_model,
    val_dataset,
    configs,
    collator,
    device,
):
    """Compute validation loss."""
    parallel_model.module.eval()
    non_blocking_transfer = _as_bool(
        getattr(configs, "non_blocking_transfer", True), True
    )

    val_bs = getattr(configs, "val_batch_size", 4)
    sampler = DistributedSampler(val_dataset, shuffle=False)
    loader_kwargs = _build_dataloader_kwargs(
        configs,
        num_workers_key="val_num_workers",
        default_num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        collate_fn=collator,
        sampler=sampler,
        **loader_kwargs,
    )

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device, non_blocking=non_blocking_transfer)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "idx"
            }
            outputs = parallel_model(**batch)
            total_loss += outputs.loss.detach().float()
            total_steps += 1

    # Average across GPUs
    loss_tensor = torch.stack(
        [total_loss, torch.tensor(float(total_steps), device=device)]
    )
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = (loss_tensor[0] / loss_tensor[1].clamp_min(1.0)).item()

    return avg_loss


def validate_generation(
    parallel_model,
    val_dataset_gen,
    val_ground_truths,
    tokenizer,
    configs,
    collator,
    local_rank,
    global_rank,
):
    """Evaluate generation accuracy on validation set (batched).

    Uses DataLoader + collator for batched generation, which is
    significantly faster than one-sample-at-a-time.
    """
    parallel_model.module.eval()
    non_blocking_transfer = _as_bool(
        getattr(configs, "non_blocking_transfer", True), True
    )

    val_bs = getattr(configs, "val_batch_size", 4)

    sampler = DistributedSampler(val_dataset_gen, shuffle=False)
    loader_kwargs = _build_dataloader_kwargs(
        configs,
        num_workers_key="val_num_workers",
        default_num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset_gen,
        batch_size=val_bs,
        collate_fn=collator,
        sampler=sampler,
        **loader_kwargs,
    )

    total_exact = 0.0
    total_token_acc = 0.0
    n_samples = 0
    n_failures = 0
    n_attempted = 0
    error_samples = []

    use_synced = getattr(configs, "use_fsdp", False) and not configs.only_eval

    with torch.no_grad():
        for batch in val_loader:
            idx_list = batch["idx"].tolist()
            input_ids = batch["input_ids"].to(
                local_rank, non_blocking=non_blocking_transfer
            )
            attention_mask = batch["attention_mask"].to(
                local_rank, non_blocking=non_blocking_transfer
            )
            position_ids = batch["position_ids"].to(
                local_rank, non_blocking=non_blocking_transfer
            )
            cur_bs = input_ids.shape[0]
            n_attempted += cur_bs

            try:
                outputs = parallel_model.module.generate(
                    input_ids,
                    attention_mask,
                    position_ids=position_ids,
                    max_new_tokens=configs.max_new_tokens,
                    synced_gpus=use_synced,
                )
            except Exception as exc:
                n_failures += cur_bs
                if len(error_samples) < 3:
                    error_samples.append(
                        f"batch error {type(exc).__name__}: {exc}"
                    )
                continue

            for i in range(cur_bs):
                generated_text = tokenizer.decode(
                    outputs[i], skip_special_tokens=False,
                )
                pred_codes = parse_motion_tokens(generated_text)

                idx = idx_list[i]
                if idx < len(val_ground_truths):
                    gt_codes = val_ground_truths[idx]
                    exact, tok_acc, _ = compute_motion_accuracy(
                        pred_codes, gt_codes,
                    )
                    total_exact += exact
                    total_token_acc += tok_acc
                    n_samples += 1

    # Average across GPUs
    stats = torch.tensor(
        [total_exact, total_token_acc, n_samples, n_failures, n_attempted],
        device=local_rank,
        dtype=torch.float32,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    n = max(stats[2].item(), 1)
    total_failures = int(stats[3].item())
    total_attempted = int(stats[4].item())
    fail_rate = float(total_failures) / max(total_attempted, 1)

    if fail_rate > GEN_FAIL_RATIO_THRESHOLD:
        err_msg = (
            f"Generation failed too often: failures={total_failures}, "
            f"attempted={total_attempted}, fail_rate={fail_rate:.3f}, "
            f"threshold={GEN_FAIL_RATIO_THRESHOLD:.3f}"
        )
        if global_rank == 0:
            print(err_msg)
            for msg in error_samples:
                print(f"  sample error: {msg}")
        raise RuntimeError(err_msg)

    return {
        "exact_match": stats[0].item() / n,
        "token_accuracy": stats[1].item() / n,
        "n_samples": int(stats[2].item()),
        "n_failures": total_failures,
        "fail_rate": fail_rate,
    }


def _build_stage_dataset_and_loader(
    scheduled_stage, base_dataset_train, configs, start_id, latent_id, end_id,
    think_open_ids, think_close_ids, collator, override_uniform_prob, epoch,
):
    """Build curriculum dataset and DataLoader for a given stage.

    Used on initial epoch entry and after mid-epoch stage transitions.
    """
    dataset_train = get_cot_latent_dataset(
        scheduled_stage,
        base_dataset_train,
        configs,
        start_id,
        latent_id,
        end_id,
        think_open_ids=think_open_ids,
        think_close_ids=think_close_ids,
        no_special_marker=(
            _as_bool(getattr(configs, "cot", False), False)
            or _as_bool(getattr(configs, "no_cot", False), False)
        ),
        shuffle=True,
        override_uniform_prob=override_uniform_prob,
    )

    sampler = DistributedSampler(dataset_train, shuffle=True)
    loader_kwargs = _build_dataloader_kwargs(
        configs,
        num_workers_key="train_num_workers",
        default_num_workers=8,
        drop_last=True,
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=configs.batch_size_training,
        collate_fn=collator,
        sampler=sampler,
        **loader_kwargs,
    )
    sampler.set_epoch(epoch)

    return dataset_train, train_loader, sampler


def _flush_accumulated_gradients(
    micro_step_in_accum, parallel_model, optimizer, configs,
    global_update_step, total_update_steps, warmup_steps, min_lr,
    zero_grad_set_to_none,
):
    """Flush partially accumulated gradients with an optimizer step.

    Returns the (possibly incremented) global_update_step.
    """
    if micro_step_in_accum == 0:
        return global_update_step
    if configs.grad_clip > 0:
        if isinstance(parallel_model, FSDP):
            parallel_model.clip_grad_norm_(configs.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(
                parallel_model.parameters(), configs.grad_clip
            )
    _apply_lr(
        optimizer,
        configs.lr,
        min_lr,
        warmup_steps,
        total_update_steps,
        global_update_step + 1,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
    return global_update_step + 1


def main():
    parser = argparse.ArgumentParser(
        description="Coconut-style latent SFT for T2M"
    )
    parser.add_argument("config_file", help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)

    # Distributed setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    deterministic = _as_bool(getattr(configs, "deterministic", False), False)
    set_seed(configs.seed, deterministic=deterministic)
    non_blocking_transfer = _as_bool(
        getattr(configs, "non_blocking_transfer", True), True
    )
    zero_grad_set_to_none = _as_bool(
        getattr(configs, "zero_grad_set_to_none", True), True
    )

    if rank == 0:
        print(f"Config: {json.dumps(config_dict, indent=2)}")
        print(
            "Performance config: "
            f"deterministic={deterministic}, "
            f"non_blocking_transfer={non_blocking_transfer}, "
            f"zero_grad_set_to_none={zero_grad_set_to_none}"
        )
        os.makedirs(configs.save_path, exist_ok=True)

    # Setup model and tokenizer
    (
        model,
        tokenizer,
        latent_id,
        start_id,
        end_id,
        think_open_ids,
        think_close_ids,
        nb_text_tokens,
    ) = setup_tokenizer_and_model(configs)

    use_lora = _as_bool(getattr(configs, "use_lora", False), False)

    if rank == 0:
        print(f"Vocab size: {len(tokenizer)}")
        print(f"Latent ID: {latent_id}, Start: {start_id}, End: {end_id}")
        if use_lora:
            trainable, total = 0, 0
            for p in model.parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            print(f"LoRA enabled: trainable={trainable:,} / total={total:,} "
                  f"({100 * trainable / total:.2f}%)")

    # Token IDs for decomposed metrics (CoT vs Answer)
    motion_start_id = tokenizer.convert_tokens_to_ids("<Motion>")
    motion_end_id = tokenizer.convert_tokens_to_ids("</Motion>")

    # Wrap in Coconut if needed
    if configs.coconut:
        model = CoconutMotion(
            model, latent_id, start_id, end_id, tokenizer.eos_token_id
        )

    # Optional load from checkpoint (before DDP/FSDP wrapping).
    checkpoint_info = maybe_load_checkpoint(model, configs, rank, nb_text_tokens)

    loaded_epoch = checkpoint_info["loaded_epoch"]
    start_epoch = resolve_start_epoch(configs.resume, configs.only_eval, loaded_epoch)

    if rank == 0:
        print(
            f"Start epoch: {start_epoch} "
            f"(resume={configs.resume}, loaded_epoch={loaded_epoch})"
        )

    model = model.to(local_rank)

    # FSDP wrapping with Qwen2DecoderLayer
    use_fsdp = getattr(configs, "use_fsdp", True) and not configs.only_eval
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        auto_wrap_cls = {Qwen2DecoderLayer}
    except ImportError:
        auto_wrap_cls = set()
        use_fsdp = False

    if not use_fsdp or len(auto_wrap_cls) == 0:
        parallel_model = DDP(model, device_ids=[local_rank])
    else:
        qwen_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=auto_wrap_cls,
        )
        parallel_model = FSDP(
            model, auto_wrap_policy=qwen_wrap_policy, device_id=local_rank
        )

    # Load datasets
    base_dataset_train = None
    if not configs.only_eval:
        if rank == 0:
            print("Loading training data...")
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_steps=configs.max_think_steps,
        )
        if rank == 0:
            print(f"  Train samples: {len(base_dataset_train)}")

    collator = MotionCollator(tokenizer, latent_id=latent_id)

    # Load validation data
    base_dataset_val = None
    val_ground_truths = None
    val_loss_every = int(getattr(configs, "val_loss_every_n_epochs", 0))
    val_gen_every = int(getattr(configs, "val_gen_every_n_epochs", 0))
    val_start_epoch = int(getattr(configs, "val_start_epoch", 0))
    run_val = (val_loss_every > 0 or val_gen_every > 0) and not configs.only_eval

    if run_val:
        if rank == 0:
            print("Loading validation data...")
        base_dataset_val = get_dataset(
            configs.val_path,
            tokenizer,
            max_steps=configs.max_think_steps,
        )
        if val_gen_every > 0:
            val_ground_truths = []
            with open(configs.val_path, encoding="utf-8") as f:
                for sample in json.load(f):
                    val_ground_truths.append(parse_motion_tokens(sample["answer"]))
        if rank == 0:
            print(f"  Val samples: {len(base_dataset_val)}")
            if val_ground_truths is not None:
                print(f"  Val GT entries: {len(val_ground_truths)}")
            print(
                f"  Val schedule: loss every {val_loss_every} epochs, "
                f"gen every {val_gen_every} epochs, start epoch {val_start_epoch}"
            )

    # WandB logging
    if rank == 0 and HAS_WANDB:
        wandb.init(project=configs.project, name=configs.name, config=config_dict)

    best_train_loss = float("inf")
    optimizer = None
    loaded_optimizer_state = checkpoint_info["optimizer_state_dict"]
    global_update_step = int(checkpoint_info.get("global_step", 0))
    prev_stage = -1

    warmup_steps = int(getattr(configs, "warmup_steps", 0))
    min_lr = float(getattr(configs, "min_lr", configs.lr * 0.1))

    # Estimate total optimizer steps for cosine schedule
    if base_dataset_train is not None:
        steps_per_epoch = max(
            len(base_dataset_train) // (configs.batch_size_training * world_size), 1
        )
        update_steps_per_epoch = max(
            steps_per_epoch // configs.gradient_accumulation_steps, 1
        )
    else:
        update_steps_per_epoch = 1
    total_update_steps = update_steps_per_epoch * configs.num_epochs

    # Step-based stage scheduling (supports fractional epochs_per_stage)
    steps_per_stage = float(configs.epochs_per_stage) * update_steps_per_epoch
    stage0_extra_steps = float(getattr(configs, "stage0_extra_epochs", 0)) * update_steps_per_epoch
    consolidation_start_step = total_update_steps - int(
        float(getattr(configs, "consolidation_epochs", 0)) * update_steps_per_epoch
    )

    if rank == 0:
        print(
            f"LR schedule: warmup={warmup_steps}, cosine {configs.lr} -> {min_lr}, "
            f"~{update_steps_per_epoch} updates/epoch, ~{total_update_steps} total"
        )
        print(
            f"Stage schedule: steps_per_stage={steps_per_stage:.1f}, "
            f"stage0_extra_steps={stage0_extra_steps:.1f}, "
            f"consolidation_start_step={consolidation_start_step}"
        )

    epochs_to_run = build_epoch_schedule(start_epoch, configs.num_epochs, configs.only_eval)
    if not configs.only_eval and len(epochs_to_run) == 0:
        if rank == 0:
            print(
                f"Nothing to train: start_epoch={start_epoch} "
                f">= num_epochs={configs.num_epochs}"
            )
        if rank == 0 and HAS_WANDB:
            wandb.finish()
        dist.destroy_process_group()
        return

    stage0_extra = float(getattr(configs, "stage0_extra_epochs", 0))
    consolidation_epochs = float(getattr(configs, "consolidation_epochs", 0))
    is_cot_or_no_cot = (
        _as_bool(getattr(configs, "cot", False), False)
        or _as_bool(getattr(configs, "no_cot", False), False)
    )

    # Main loop
    for epoch in epochs_to_run:
        # Curriculum stage scheduling (step-based for fractional epochs_per_stage)
        if is_cot_or_no_cot:
            scheduled_stage = 0
        else:
            scheduled_stage = compute_scheduled_stage_by_step(
                global_update_step, steps_per_stage, stage0_extra_steps,
            )

        is_consolidation = global_update_step >= consolidation_start_step
        override_uniform_prob = 0.0 if is_consolidation else None

        if rank == 0:
            print(f"\n{'='*60}")
            print(format_stage_info_by_step(
                global_update_step, steps_per_stage, configs.max_latent_stage,
                stage0_extra_steps=stage0_extra_steps, epoch=epoch,
            ))
            n_latent = min(scheduled_stage, configs.max_latent_stage) * configs.c_thought
            print(f"  Latent tokens: {n_latent}")
            if is_consolidation:
                print(f"  ** CONSOLIDATION EPOCH ** (uniform_prob=0)")
            print(f"{'='*60}")

        avg_train_loss = float("nan")
        stage_changed = scheduled_stage != prev_stage

        if not configs.only_eval:
            # Build curriculum dataset and loader for initial stage
            dataset_train, train_loader, sampler = _build_stage_dataset_and_loader(
                scheduled_stage, base_dataset_train, configs,
                start_id, latent_id, end_id,
                think_open_ids, think_close_ids, collator,
                override_uniform_prob, epoch,
            )

            # Reset optimizer at stage transitions (not every epoch)
            if optimizer is None or (configs.reset_optimizer and stage_changed):
                # LoRA: only optimize trainable params (adapters + new token embeddings)
                opt_params = [p for p in parallel_model.parameters() if p.requires_grad]
                optimizer = optim.AdamW(
                    opt_params,
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )
                if loaded_optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(loaded_optimizer_state)
                        if rank == 0:
                            print("  Restored optimizer state from checkpoint.")
                    except Exception as exc:  # noqa: PERF203
                        if rank == 0:
                            print(f"  Warning: failed to restore optimizer state: {exc}")
                    loaded_optimizer_state = None
                if rank == 0 and stage_changed and prev_stage >= 0:
                    print(f"  Optimizer reset (stage {prev_stage} -> {scheduled_stage})")

            parallel_model.module.train()

            total_loss = torch.zeros((), device=local_rank, dtype=torch.float32)
            n_steps = 0

            # Accumulators for decomposed metrics
            sum_gen_acc = 0.0
            sum_cot_acc = 0.0
            sum_answer_acc = 0.0
            sum_cot_loss = 0.0
            sum_answer_loss = 0.0
            n_metric_steps = 0
            n_cot_steps = 0       # steps where cot region was non-empty
            n_answer_steps = 0    # steps where answer region was non-empty
            total_n_cot = 0       # total cot tokens across metric steps
            total_n_answer = 0    # total answer tokens across metric steps

            progress = tqdm(
                total=update_steps_per_epoch,
                desc=f"Epoch {epoch}",
                disable=(rank != 0),
            )

            epoch_update_steps_done = 0
            micro_step_in_accum = 0

            while epoch_update_steps_done < update_steps_per_epoch:
                for batch in train_loader:
                    batch = {
                        k: v.to(local_rank, non_blocking=non_blocking_transfer)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor) and k != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss / configs.gradient_accumulation_steps
                    loss.backward()

                    total_loss += outputs.loss.detach().float()

                    # Compute decomposed metrics periodically (not every step to save time)
                    if outputs.logits is not None and n_steps % configs.log_interval == 0:
                        metrics = _compute_train_metrics(
                            outputs.logits.detach(), batch["labels"],
                            motion_start_id, motion_end_id,
                        )
                        sum_gen_acc += metrics["gen_acc"]
                        if not math.isnan(metrics["cot_acc"]):
                            sum_cot_acc += metrics["cot_acc"]
                            sum_cot_loss += metrics["cot_loss"]
                            n_cot_steps += 1
                        if not math.isnan(metrics["answer_acc"]):
                            sum_answer_acc += metrics["answer_acc"]
                            sum_answer_loss += metrics["answer_loss"]
                            n_answer_steps += 1
                        total_n_cot += metrics["n_cot"]
                        total_n_answer += metrics["n_answer"]
                        n_metric_steps += 1

                    if rank == 0 and n_steps % configs.log_interval == 0:
                        avg_loss = (total_loss / max(n_steps + 1, 1)).item()
                        cur_ans_acc = sum_answer_acc / max(n_answer_steps, 1) if n_answer_steps > 0 else float("nan")
                        cur_cot_acc = sum_cot_acc / max(n_cot_steps, 1) if n_cot_steps > 0 else float("nan")
                        postfix = {"loss": f"{avg_loss:.4f}"}
                        postfix["ans_acc"] = f"{cur_ans_acc:.3f}" if not math.isnan(cur_ans_acc) else "-"
                        postfix["cot_acc"] = f"{cur_cot_acc:.3f}" if not math.isnan(cur_cot_acc) else "-"
                        progress.set_postfix(**postfix)

                        # Print decomposed metrics periodically (UniMo-style)
                        if n_metric_steps > 0 and n_steps % (configs.log_interval * 20) == 0:
                            print(
                                f"  Step {n_steps} | loss: {avg_loss:.4f}, "
                                f"gen_acc: {_fmt_metric(metrics['gen_acc'])}, "
                                f"cot_acc: {_fmt_metric(metrics['cot_acc'])}, "
                                f"answer_acc: {_fmt_metric(metrics['answer_acc'])}, "
                                f"cot_loss: {_fmt_metric(metrics['cot_loss'])}, "
                                f"answer_loss: {_fmt_metric(metrics['answer_loss'])}, "
                                f"n_cot: {metrics['n_cot']}, "
                                f"n_ans: {metrics['n_answer']}"
                            )

                        if HAS_WANDB:
                            wb_log = {
                                "train/loss": avg_loss,
                                "train/epoch": epoch,
                                "train/stage": scheduled_stage,
                                "train/step": global_update_step,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/global_step": global_update_step,
                                "train/gen_acc": metrics["gen_acc"] if n_metric_steps > 0 else 0,
                                "train/n_cot": metrics["n_cot"] if n_metric_steps > 0 else 0,
                                "train/n_answer": metrics["n_answer"] if n_metric_steps > 0 else 0,
                            }
                            if n_metric_steps > 0:
                                if not math.isnan(metrics["cot_acc"]):
                                    wb_log["train/cot_acc"] = metrics["cot_acc"]
                                    wb_log["train/cot_loss"] = metrics["cot_loss"]
                                if not math.isnan(metrics["answer_acc"]):
                                    wb_log["train/answer_acc"] = metrics["answer_acc"]
                                    wb_log["train/answer_loss"] = metrics["answer_loss"]
                            wandb.log(wb_log)

                    n_steps += 1
                    micro_step_in_accum += 1

                    if micro_step_in_accum == configs.gradient_accumulation_steps:
                        if configs.grad_clip > 0:
                            if isinstance(parallel_model, FSDP):
                                parallel_model.clip_grad_norm_(configs.grad_clip)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    parallel_model.parameters(), configs.grad_clip
                                )
                        _apply_lr(
                            optimizer,
                            configs.lr,
                            min_lr,
                            warmup_steps,
                            total_update_steps,
                            global_update_step + 1,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
                        global_update_step += 1
                        epoch_update_steps_done += 1
                        micro_step_in_accum = 0
                        progress.update(1)

                        # Check for mid-epoch stage transition
                        if not is_cot_or_no_cot:
                            new_stage = compute_scheduled_stage_by_step(
                                global_update_step, steps_per_stage,
                                stage0_extra_steps,
                            )
                            new_stage = min(new_stage, configs.max_latent_stage)
                            cur_stage = min(scheduled_stage, configs.max_latent_stage)
                            if new_stage != cur_stage:
                                if rank == 0:
                                    print(
                                        f"  >> Mid-epoch stage transition: "
                                        f"stage {scheduled_stage}->{new_stage} "
                                        f"at step {global_update_step}"
                                    )
                                scheduled_stage = new_stage
                                is_consolidation = (
                                    global_update_step >= consolidation_start_step
                                )
                                override_uniform_prob = (
                                    0.0 if is_consolidation else None
                                )
                                dataset_train, train_loader, sampler = (
                                    _build_stage_dataset_and_loader(
                                        scheduled_stage, base_dataset_train,
                                        configs, start_id, latent_id, end_id,
                                        think_open_ids, think_close_ids, collator,
                                        override_uniform_prob, epoch,
                                    )
                                )
                                # Reset optimizer at stage transitions if configured
                                if configs.reset_optimizer:
                                    opt_params = [
                                        p for p in parallel_model.parameters()
                                        if p.requires_grad
                                    ]
                                    optimizer = optim.AdamW(
                                        opt_params,
                                        lr=configs.lr,
                                        weight_decay=configs.weight_decay,
                                    )
                                    if rank == 0:
                                        print(
                                            f"  Optimizer reset "
                                            f"(stage transition -> {scheduled_stage})"
                                        )
                                break  # re-enter while loop with new loader

                        if epoch_update_steps_done >= update_steps_per_epoch:
                            break
                else:
                    # DataLoader exhausted before step budget filled —
                    # re-enter while loop to create fresh iterator
                    continue
                # Broke out of for loop (stage change or epoch done)
                if epoch_update_steps_done >= update_steps_per_epoch:
                    break

            # Flush remaining gradients at epoch end
            global_update_step = _flush_accumulated_gradients(
                micro_step_in_accum, parallel_model, optimizer, configs,
                global_update_step, total_update_steps, warmup_steps, min_lr,
                zero_grad_set_to_none,
            )
            progress.close()

            avg_train_loss = (total_loss / max(n_steps, 1)).item()
            nm = max(n_metric_steps, 1)
            epoch_metrics = {
                "gen_acc": sum_gen_acc / nm,
                "cot_acc": sum_cot_acc / max(n_cot_steps, 1) if n_cot_steps > 0 else float("nan"),
                "answer_acc": sum_answer_acc / max(n_answer_steps, 1) if n_answer_steps > 0 else float("nan"),
                "cot_loss": sum_cot_loss / max(n_cot_steps, 1) if n_cot_steps > 0 else float("nan"),
                "answer_loss": sum_answer_loss / max(n_answer_steps, 1) if n_answer_steps > 0 else float("nan"),
                "n_cot": total_n_cot,
                "n_answer": total_n_answer,
                "n_cot_steps": n_cot_steps,
                "n_answer_steps": n_answer_steps,
            }
            if rank == 0:
                print(
                    f"  Train loss: {avg_train_loss:.4f}  "
                    f"cot_loss: {_fmt_metric(epoch_metrics['cot_loss'])}  "
                    f"answer_loss: {_fmt_metric(epoch_metrics['answer_loss'])}"
                )
                print(
                    f"  Train gen_acc: {epoch_metrics['gen_acc']:.4f}  "
                    f"cot_acc: {_fmt_metric(epoch_metrics['cot_acc'])}  "
                    f"answer_acc: {_fmt_metric(epoch_metrics['answer_acc'])}"
                )
                avg_cot_per_batch = total_n_cot / nm if nm > 0 else 0
                avg_ans_per_batch = total_n_answer / nm if nm > 0 else 0
                print(
                    f"  Token counts (avg/batch): cot={avg_cot_per_batch:.0f}, "
                    f"answer={avg_ans_per_batch:.0f}  "
                    f"(cot non-empty: {n_cot_steps}/{n_metric_steps} steps)"
                )
        elif rank == 0:
            print("  only_eval=true: skipping training phase.")

        prev_stage = scheduled_stage

        # WandB logging
        if rank == 0 and HAS_WANDB:
            log_dict = {
                "epoch": epoch,
                "stage": scheduled_stage,
            }
            if not configs.only_eval:
                log_dict["train/avg_loss"] = avg_train_loss
                log_dict["train/global_step"] = global_update_step
                log_dict["train/epoch_gen_acc"] = epoch_metrics["gen_acc"]
                log_dict["train/epoch_n_cot"] = epoch_metrics["n_cot"]
                log_dict["train/epoch_n_answer"] = epoch_metrics["n_answer"]
                if not math.isnan(epoch_metrics["cot_acc"]):
                    log_dict["train/epoch_cot_acc"] = epoch_metrics["cot_acc"]
                    log_dict["train/epoch_cot_loss"] = epoch_metrics["cot_loss"]
                if not math.isnan(epoch_metrics["answer_acc"]):
                    log_dict["train/epoch_answer_acc"] = epoch_metrics["answer_acc"]
                    log_dict["train/epoch_answer_loss"] = epoch_metrics["answer_loss"]
            wandb.log(log_dict)

        # Save checkpoint (skip in eval-only mode)
        if not configs.only_eval:
            save_every = int(getattr(configs, "save_every_n_epochs", 1))
            is_last_epoch = (epoch == epochs_to_run[-1] if epochs_to_run else False)
            should_save = is_last_epoch or (save_every > 0 and epoch % save_every == 0)
            # Always save during consolidation phase (every epoch matters)
            if is_consolidation:
                should_save = True
            if configs.save_only_improve and avg_train_loss >= best_train_loss:
                should_save = False

            if should_save:
                ckpt_dir = os.path.join(
                    configs.save_path, f"checkpoint-epoch{epoch}"
                )

                if isinstance(parallel_model, FSDP):
                    # FSDP: gather full state dict to rank 0 and save
                    save_policy = FullStateDictConfig(
                        offload_to_cpu=True, rank0_only=True
                    )
                    with FSDP.state_dict_type(
                        parallel_model,
                        StateDictType.FULL_STATE_DICT,
                        save_policy,
                    ):
                        cpu_state = parallel_model.state_dict()

                    # Gather optimizer state (FSDP shards it across ranks)
                    optim_state = FSDP.optim_state_dict(parallel_model, optimizer)

                    if rank == 0:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        # Save full state dict
                        torch.save(cpu_state, os.path.join(ckpt_dir, "model_state.pt"))
                        tokenizer.save_pretrained(ckpt_dir)
                        torch.save(
                            {
                                "epoch": epoch,
                                "optimizer_state_dict": optim_state,
                                "train_loss": avg_train_loss,
                                "scheduled_stage": scheduled_stage,
                                "global_step": global_update_step,
                            },
                            os.path.join(ckpt_dir, "training_state.pt"),
                        )
                        print(f"  Saved FSDP checkpoint: {ckpt_dir}")
                else:
                    if rank == 0:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        _save = parallel_model.module
                        if use_lora and hasattr(_save, "base_causallm"):
                            # LoRA + CoconutMotion: save LoRA adapters + new
                            # token embeddings following Motion-Agent's pattern.
                            _save_lora_checkpoint(
                                _save.base_causallm, nb_text_tokens, ckpt_dir,
                            )
                        elif hasattr(_save, "base_causallm"):
                            _save.base_causallm.save_pretrained(ckpt_dir)
                        else:
                            _save.save_pretrained(ckpt_dir)

                        tokenizer.save_pretrained(ckpt_dir)
                        torch.save(
                            {
                                "epoch": epoch,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_loss": avg_train_loss,
                                "scheduled_stage": scheduled_stage,
                                "global_step": global_update_step,
                            },
                            os.path.join(ckpt_dir, "training_state.pt"),
                        )
                        print(f"  Saved checkpoint: {ckpt_dir}")

        if rank == 0 and avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if run_val and epoch >= val_start_epoch:
            do_val_loss = val_loss_every > 0 and epoch % val_loss_every == 0
            do_val_gen = val_gen_every > 0 and epoch % val_gen_every == 0
            val_max = int(getattr(configs, "val_max_samples", 100))

            # Validation loss
            if do_val_loss:
                if rank == 0:
                    print("  Running validation loss...")
                val_dataset = get_cot_latent_dataset(
                    scheduled_stage,
                    base_dataset_val,
                    configs,
                    start_id,
                    latent_id,
                    end_id,
                    think_open_ids=think_open_ids,
                    think_close_ids=think_close_ids,
                    no_special_marker=is_cot_or_no_cot,
                    shuffle=False,
                )
                if 0 < val_max < len(val_dataset):
                    val_dataset = val_dataset.select(range(val_max))
                val_loss = validate_loss(
                    parallel_model, val_dataset, configs, collator, local_rank,
                )
                if rank == 0:
                    print(f"  Val loss: {val_loss:.4f}")
                    if HAS_WANDB:
                        wandb.log({
                            "val/loss": val_loss,
                            "epoch": epoch,
                            "train/global_step": global_update_step,
                        })

            # Generation evaluation
            if do_val_gen:
                n_eval_latent = min(scheduled_stage, configs.max_latent_stage) * configs.c_thought
                if rank == 0:
                    print(f"  Running generation evaluation (stage={scheduled_stage}, latent_tokens={n_eval_latent})...")
                # Use current scheduled_stage so eval matches training distribution;
                # at early stages the model has not been trained on max latent tokens.
                val_dataset_gen = get_question_latent_dataset(
                    scheduled_stage,
                    base_dataset_val,
                    configs,
                    start_id,
                    latent_id,
                    end_id,
                    no_special_marker=is_cot_or_no_cot,
                )
                if 0 < val_max < len(val_dataset_gen):
                    val_dataset_gen = val_dataset_gen.select(range(val_max))
                gen_results = validate_generation(
                    parallel_model,
                    val_dataset_gen,
                    val_ground_truths,
                    tokenizer,
                    configs,
                    collator,
                    local_rank,
                    rank,
                )
                if rank == 0:
                    print(
                        f"  Gen exact_match: {gen_results['exact_match']:.4f}  "
                        f"token_acc: {gen_results['token_accuracy']:.4f}  "
                        f"n={gen_results['n_samples']}  "
                        f"failures={gen_results['n_failures']}"
                    )
                    if HAS_WANDB:
                        wandb.log({
                            "val/gen_exact_match": gen_results["exact_match"],
                            "val/gen_token_accuracy": gen_results["token_accuracy"],
                            "val/gen_n_samples": gen_results["n_samples"],
                            "val/gen_n_failures": gen_results["n_failures"],
                            "epoch": epoch,
                            "train/global_step": global_update_step,
                        })

            # Back to training mode
            if not configs.only_eval:
                parallel_model.module.train()

        gc.collect()
        torch.cuda.empty_cache()

        if configs.only_eval:
            break

    # Cleanup
    if rank == 0 and HAS_WANDB:
        wandb.finish()

    dist.destroy_process_group()

    if rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
