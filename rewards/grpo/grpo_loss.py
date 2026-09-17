"""GRPO Loss implementations for CoconutMotion.

Supports 6 loss types following TRL's grpo_trainer.py (L2194-2242):
- grpo: Original GRPO, per-sequence length normalization (length-biased)
- dapo: Global active token count normalization (recommended, TRL default)
- dr_grpo: Constant normalization by max_completion_length
- bnpo: Local batch token count normalization
- cispo: Clipped importance sampling weights (MiniMax-M1)
- sapo: Soft Adaptive Policy Optimization with smooth gating

Reference: TRL GRPOTrainer loss formulations.
"""

import torch
import torch.nn.functional as F

def compute_per_token_log_probs(logits, labels):
# def compute_per_token_log_probs(logits, labels, forbidden_token_ids=None):
    """Compute per-token log probability from logits and labels.

    Uses selective log-softmax (gather + logsumexp) to avoid materializing
    the full (B, L, V) log-probability tensor, which is critical when V is
    large (e.g. 152 K).

    Reference: TRL grpo_trainer.py selective_log_softmax

    Args:
        logits: (B, L, V) model output logits
        labels: (B, L) with -100 for masked positions (prompt, latent tokens)

    Returns:
        per_token_logps: (B, L-1) log probs (masked positions = 0)
        mask: (B, L-1) binary mask for completion tokens
    """
    # Shift: logits[t] predicts labels[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()

    # Selective log-softmax: gather target logit then subtract logsumexp.
    # This avoids allocating a (B, L-1, V) log_softmax output.
    target_logits = shift_logits.gather(
        -1, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    log_z = shift_logits.logsumexp(dim=-1)

    # if forbidden_token_ids:
    #     blocked = [
    #         int(tid)
    #         for tid in forbidden_token_ids
    #         if tid is not None and 0 <= int(tid) < shift_logits.size(-1)
    #     ]
    #     if blocked:
    #         blocked_ids = torch.tensor(
    #             blocked, device=shift_logits.device, dtype=torch.long
    #         )
    #         blocked_log_z = shift_logits.index_select(
    #             -1, blocked_ids
    #         ).logsumexp(dim=-1)
    #         # log(sum_allowed exp) = logsumexp(all) - sum_forbidden exp,
    #         # implemented stably without cloning or editing the full logits.
    #         blocked_ratio = torch.exp(blocked_log_z - log_z).clamp(
    #             max=1.0 - 1e-7
    #         )
    #         log_z = log_z + torch.log1p(-blocked_ratio)

    per_token_logps = target_logits - log_z

    # Mask: only completion tokens (labels != -100)
    mask = (shift_labels != -100).float()
    per_token_logps = per_token_logps * mask

    return per_token_logps, mask


def compute_group_advantages(rewards, num_generations, scale_rewards="group"):
    """Compute group-relative advantages.

    Args:
        rewards: (B*G,) reward for each completion
        num_generations: G, number of completions per prompt
        scale_rewards: "group" (per-group std), "batch" (batch-wide std), "none"

    Returns:
        advantages: (B*G,) normalized advantages
    """
    B = rewards.size(0) // num_generations
    grouped = rewards.view(B, num_generations)

    # Subtract group mean
    group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = grouped - group_mean

    # Scale by standard deviation
    if scale_rewards == "group":
        # Use unbiased=False for small group sizes to avoid Bessel's correction
        # noise (N-1 denominator amplifies variance with small G like 4).
        group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
        advantages = advantages / (group_std + 1e-4)
    elif scale_rewards == "batch":
        batch_std = rewards.std()
        advantages = advantages / (batch_std + 1e-4)
    # scale_rewards == "none": no scaling

    return advantages.view(B * num_generations)


def compute_grpo_loss(
    per_token_logps,
    ref_per_token_logps,
    advantages,
    completion_mask,
    loss_type="dapo",
    beta=0.01,
    epsilon=0.2,
    max_completion_length=256,
    gradient_accumulation_steps=1,
    kl_clip_max=10.0,
):
    """Compute GRPO loss with multiple loss type variants.

    Reference: TRL grpo_trainer.py L2194-2242

    Args:
        per_token_logps: (B, T) policy log probs for completion tokens
        ref_per_token_logps: (B, T) reference model log probs (None if beta=0)
        advantages: (B,) group-relative advantages
        completion_mask: (B, T) binary mask for completion tokens
        loss_type: "grpo" / "dapo" / "dr_grpo" / "bnpo" / "cispo" / "sapo"
        beta: KL divergence coefficient (0 = no KL penalty)
        epsilon: PPO-style clipping epsilon
        max_completion_length: for dr_grpo normalization
        gradient_accumulation_steps: for loss scaling
        kl_clip_max: per-token KL clipping threshold. Prevents a few tokens
            with extreme policy/reference divergence from dominating the loss
            (common with large vocab and sparse reward signals).

    Returns:
        loss: scalar loss value
        metrics: dict of loss components for logging
    """
    advantages_expanded = advantages.unsqueeze(1)  # (B, 1)

    # Importance sampling ratio (for on-policy, old_logps ≈ per_token_logps)
    old_per_token_logps = per_token_logps.detach()
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    # KL divergence
    per_token_kl = None
    if beta > 0 and ref_per_token_logps is not None:
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )
        # Clip per-token KL to prevent extreme values from dominating.
        if kl_clip_max > 0:
            per_token_kl = torch.clamp(per_token_kl, max=kl_clip_max)

    # === Loss computation by type ===
    if loss_type == "cispo":
        # MiniMax-M1 style: clip importance weights
        clamped_ratios = torch.clamp(ratio, max=1 + epsilon).detach()
        per_token_loss = -clamped_ratios * advantages_expanded * per_token_logps

    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
        # PPO-style clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = ratio * advantages_expanded
        per_token_loss2 = clipped_ratio * advantages_expanded
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    elif loss_type == "sapo":
        # Soft Adaptive Policy Optimization
        temp_neg, temp_pos = 1.05, 1.0
        per_token_loss = torch.empty_like(ratio)
        pos_mask = advantages_expanded.expand_as(ratio) > 0
        # Positive advantages
        if pos_mask.any():
            sigmoid_input = temp_pos * (ratio[pos_mask] - 1)
            per_token_loss[pos_mask] = torch.sigmoid(sigmoid_input) * 4 / temp_pos
        # Negative advantages
        if (~pos_mask).any():
            sigmoid_input = temp_neg * (ratio[~pos_mask] - 1)
            per_token_loss[~pos_mask] = torch.sigmoid(sigmoid_input) * 4 / temp_neg
        per_token_loss = -per_token_loss * advantages_expanded

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Supported: grpo, dapo, dr_grpo, bnpo, cispo, sapo"
        )

    # Add KL term
    if beta > 0 and per_token_kl is not None:
        per_token_loss = per_token_loss + beta * per_token_kl

    # === Normalization by loss_type ===
    if loss_type in ["grpo", "sapo"]:
        # Per-sequence length normalization
        loss = (
            (per_token_loss * completion_mask).sum(-1)
            / completion_mask.sum(-1).clamp(min=1.0)
        ).mean()

    elif loss_type == "bnpo":
        # Local batch active token count normalization
        loss = (
            (per_token_loss * completion_mask).sum()
            / completion_mask.sum().clamp(min=1.0)
        )

    elif loss_type == "dr_grpo":
        # Constant normalization by max_completion_length
        batch_size = per_token_logps.size(0)
        loss = (
            (per_token_loss * completion_mask).sum()
            / (batch_size * max_completion_length)
        )

    elif loss_type in ["dapo", "cispo"]:
        # Global active token count normalization (with all_reduce for DDP)
        num_active = completion_mask.sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_active)
            # Scale by world_size to compensate for DDP gradient averaging:
            # DDP divides gradients by world_size, but we already divided by
            # the global token count. Without this, effective gradient is
            # off by 1/world_size.
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        loss = (
            (per_token_loss * completion_mask).sum()
            / num_active.clamp(min=1.0)
            * world_size
        )

    # Scale by gradient accumulation
    loss = loss / gradient_accumulation_steps

    # Compute metrics for logging
    metrics = {
        "loss": loss.detach().item() * gradient_accumulation_steps,
    }
    if per_token_kl is not None:
        kl_mean = (
            (per_token_kl * completion_mask).sum()
            / completion_mask.sum().clamp(min=1.0)
        ).item()
        metrics["kl"] = kl_mean

    return loss, metrics
