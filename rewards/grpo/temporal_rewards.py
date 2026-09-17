"""Temporal Motion Rewards for GRPO Training.

Inspired by Video-R1's T-GRPO (Temporal GRPO) algorithm, adapted for
Text-to-Motion generation. Two complementary approaches:

1. **Contrastive T-GRPO** (Section A): Compare motion quality with ordered
   vs shuffled motion tokens. If ordered is significantly better, add
   temporal bonus — indicating the model captures temporal structure.

2. **Direct Temporal Quality** (Section B): Compute frame-level temporal
   smoothness metrics (velocity, acceleration, jerk) as reward signals.

Reference:
    - Video-R1: https://arxiv.org/abs/2503.21776
    - TRL t2m_rewards.py: physics reward infrastructure
"""

import sys
import re
import random
from typing import Optional

import torch
import torch.nn.functional as F

# Import the bundled TRL VQ-VAE decoding infrastructure.

try:
    from third_party.trl_motion.rewards.t2m_rewards import (
        decode_motion_tokens_to_poses,
        parse_motion_token_ids,
        extract_motion_content,
        compute_velocity_smoothness,
        compute_foot_skating,
        compute_joint_angle_violation,
    )
    _TRL_DECODE_AVAILABLE = True
except ImportError:
    _TRL_DECODE_AVAILABLE = False
    print("[temporal_rewards] WARNING: TRL decode functions not available.")


# =====================================================================
# Section A: Contrastive T-GRPO for Motion
# =====================================================================

def compute_temporal_bonus(
    ordered_rewards,
    shuffled_rewards,
    mu=0.8,
    alpha=0.3,
    theta=0.1,
):
    """Compute T-GRPO temporal bonus by comparing ordered vs shuffled.

    Following Video-R1's T-GRPO mechanism:
    - If mean ordered reward >= mu * mean shuffled reward, temporal ordering
      matters → add alpha bonus to samples with reward > theta.

    Args:
        ordered_rewards: (B*G,) rewards from normal generation
        shuffled_rewards: (B*G_s,) rewards from shuffled motion token generation
        mu: threshold ratio (default 0.8, from Video-R1)
        alpha: temporal bonus value (default 0.3, from Video-R1)
        theta: minimum reward threshold to receive bonus (default 0.1)

    Returns:
        temporal_rewards: (B*G,) rewards with temporal bonus applied
        temporal_active: bool, whether temporal bonus was applied
    """
    ordered_mean = ordered_rewards.mean().item()
    shuffled_mean = shuffled_rewards.mean().item()

    temporal_rewards = ordered_rewards.clone()

    if ordered_mean >= mu * shuffled_mean:
        # Temporal ordering matters → add bonus to good samples
        mask = ordered_rewards > theta
        temporal_rewards[mask] = temporal_rewards[mask] + alpha
        temporal_active = True
    else:
        # Shuffled performs similarly → no temporal bonus
        temporal_active = False

    return temporal_rewards, temporal_active


def shuffle_motion_tokens_in_text(text):
    """Shuffle motion token order within <Motion>...</Motion> tags.

    Preserves the tags but randomizes the order of <Motion_N> tokens inside.

    Args:
        text: completion text containing <Motion>...</Motion>

    Returns:
        shuffled_text: text with motion tokens shuffled
    """
    # Find motion content between tags
    pattern = r"(<Motion>)(.*?)(</Motion>)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text

    prefix = text[:match.start()] + match.group(1)
    motion_content = match.group(2)
    suffix = match.group(3) + text[match.end():]

    # Parse individual motion tokens
    tokens = re.findall(r"<Motion_\d+>", motion_content)
    if len(tokens) <= 1:
        return text

    # Shuffle tokens
    random.shuffle(tokens)
    shuffled_content = " ".join(tokens)

    return prefix + shuffled_content + suffix


def compute_shuffled_rewards(
    completions, ground_truths, reward_func, **reward_kwargs
):
    """Compute rewards for shuffled versions of completions.

    Args:
        completions: list[str] — generated completion texts
        ground_truths: list[str] — GT motion token texts
        reward_func: callable(generated_texts, ground_truths, **kwargs) -> tensor

    Returns:
        shuffled_rewards: tensor of rewards for shuffled completions
    """
    shuffled_texts = [shuffle_motion_tokens_in_text(c) for c in completions]
    return reward_func(shuffled_texts, ground_truths, **reward_kwargs)


# =====================================================================
# Section B: Direct Temporal Quality Metrics
# =====================================================================

@torch.inference_mode()
def compute_jerk_penalty(poses, fps=10.0):
    """Compute jerk (derivative of acceleration) penalty.

    Jerk measures the rate of change of acceleration. High jerk indicates
    abrupt, unnatural motion transitions.

    Args:
        poses: [B, T, D] or [T, D] — HumanML3D format pose data
        fps: frame rate (VQ-VAE downsampled to 10 FPS)

    Returns:
        L_jerk: scalar penalty
    """
    if poses is None or poses.numel() == 0:
        return torch.tensor(0.0)

    if poses.dim() == 2:
        poses = poses.unsqueeze(0)

    B, T, D = poses.shape
    if T < 4:
        return torch.tensor(0.0, device=poses.device)

    dt = 1.0 / fps

    # Use position data (ric: 21 joints x 3)
    ric_start, ric_end = 4, 4 + 63
    if D < ric_end:
        return torch.tensor(0.0, device=poses.device)

    positions = poses[:, :, ric_start:ric_end]

    # Velocity
    velocity = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
    # Acceleration
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt
    # Jerk
    jerk = (acceleration[:, 1:, :] - acceleration[:, :-1, :]) / dt
    jerk_norm = torch.norm(jerk, dim=-1)  # [B, T-3]

    # Threshold: penalize excessive jerk
    jerk_threshold = 50.0  # Adjusted for normalized HumanML3D data
    excess_jerk = F.relu(jerk_norm - jerk_threshold)
    L_jerk = excess_jerk.mean()

    return L_jerk


@torch.inference_mode()
def compute_frame_transition_smoothness(poses, fps=10.0):
    """Compute frame-to-frame transition smoothness.

    Measures the variance of inter-frame distances. Low variance = smooth
    constant-speed motion. High variance = irregular transitions.

    Args:
        poses: [B, T, D] or [T, D]
        fps: frame rate

    Returns:
        smoothness_score: scalar in [0, 1], higher = smoother
    """
    if poses is None or poses.numel() == 0:
        return torch.tensor(0.5)

    if poses.dim() == 2:
        poses = poses.unsqueeze(0)

    B, T, D = poses.shape
    if T < 3:
        return torch.tensor(0.5, device=poses.device)

    # Use all pose features for transition distance
    frame_diffs = poses[:, 1:, :] - poses[:, :-1, :]
    frame_distances = torch.norm(frame_diffs, dim=-1)  # [B, T-1]

    # Smoothness = 1 / (1 + coefficient_of_variation)
    mean_dist = frame_distances.mean(dim=-1, keepdim=True)
    std_dist = frame_distances.std(dim=-1, keepdim=True)
    cv = std_dist / (mean_dist + 1e-6)  # Coefficient of variation

    smoothness = 1.0 / (1.0 + cv.mean())
    return smoothness


def r_temporal_quality(
    motion_tokens,
    lambda_jerk=1.0,
    lambda_vel=1.0,
    lambda_skating=0.5,
    lambda_smoothness=0.5,
):
    """Compute comprehensive temporal quality reward.

    Combines existing physics rewards with new temporal metrics.

    Args:
        motion_tokens: list[int] — decoded motion token IDs
        lambda_jerk: jerk penalty weight
        lambda_vel: velocity smoothness weight
        lambda_skating: foot skating weight
        lambda_smoothness: frame transition smoothness weight

    Returns:
        reward: float in [0, 1]
        details: dict of individual metrics
    """
    if not _TRL_DECODE_AVAILABLE:
        return 0.5, {"error": "TRL decode not available"}

    if not motion_tokens:
        return 0.0, {"empty": True}

    poses = decode_motion_tokens_to_poses(motion_tokens)
    if poses is None:
        return 0.0, {"decode_failed": True}

    # Existing TRL physics metrics
    L_vel = compute_velocity_smoothness(poses, fps=10.0)
    L_skating = compute_foot_skating(poses, fps=10.0)
    L_joint = compute_joint_angle_violation(poses)

    # New temporal metrics
    L_jerk = compute_jerk_penalty(poses, fps=10.0)
    smoothness = compute_frame_transition_smoothness(poses, fps=10.0)

    # Total penalty (lower = better motion)
    total_penalty = (
        lambda_vel * L_vel.item()
        + lambda_skating * L_skating.item()
        + lambda_jerk * L_jerk.item()
    )

    # Combine: penalty maps to [0, 1] via sigmoid, blend with smoothness
    physics_score = torch.sigmoid(torch.tensor(-total_penalty)).item()
    smoothness_score = smoothness.item()

    reward = (
        (1 - lambda_smoothness) * physics_score
        + lambda_smoothness * smoothness_score
    )

    details = {
        "L_vel": L_vel.item(),
        "L_skating": L_skating.item(),
        "L_joint": L_joint.item(),
        "L_jerk": L_jerk.item(),
        "smoothness": smoothness_score,
        "physics_score": physics_score,
        "temporal_reward": reward,
    }

    return reward, details


# =====================================================================
# TRL-compatible Reward Functions
# =====================================================================

def t2m_temporal_reward(
    completions,
    ground_truth,
    **kwargs,
):
    """Direct temporal quality reward (TRL-compatible interface).

    Evaluates temporal smoothness of generated motion sequences.

    Args:
        completions: list[list[dict]] — TRL message format
        ground_truth: list[str] — GT motion token texts

    Returns:
        list[float] — temporal quality rewards
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content) if _TRL_DECODE_AVAILABLE else ""
        tokens = parse_motion_token_ids(motion_content) if motion_content and _TRL_DECODE_AVAILABLE else []
        r, _ = r_temporal_quality(tokens)
        rewards.append(r)
    return rewards


def t2m_jerk_reward(
    completions,
    ground_truth,
    **kwargs,
):
    """Jerk-only temporal reward (TRL-compatible interface).

    Penalizes abrupt acceleration changes in generated motion.

    Args:
        completions: list[list[dict]]
        ground_truth: list[str]

    Returns:
        list[float]
    """
    if not _TRL_DECODE_AVAILABLE:
        return [0.5] * len(completions)

    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        tokens = parse_motion_token_ids(motion_content) if motion_content else []

        if not tokens:
            rewards.append(0.0)
            continue

        poses = decode_motion_tokens_to_poses(tokens)
        if poses is None:
            rewards.append(0.0)
            continue

        L_jerk = compute_jerk_penalty(poses, fps=10.0)
        # Map penalty to [0, 1] reward
        r = torch.sigmoid(torch.tensor(-L_jerk.item())).item()
        rewards.append(r)

    return rewards


# =====================================================================
# T-GRPO Integration for train_grpo.py
# =====================================================================

class TemporalGRPO:
    """T-GRPO integration for CoconutMotion GRPO training.

    Manages the contrastive comparison between ordered and shuffled
    motion tokens, and computes temporal bonus rewards.

    Usage in train_grpo.py:
        temporal_grpo = TemporalGRPO(reward_computer, mu=0.8, alpha=0.3)

        # After generating completions and computing base rewards:
        final_rewards, t_active = temporal_grpo.apply(
            completions, ground_truths, base_rewards
        )
    """

    def __init__(self, reward_computer, mu=0.8, alpha=0.3, theta=0.1):
        """
        Args:
            reward_computer: RewardComputer instance from rewards.py
            mu: threshold ratio for ordered vs shuffled comparison
            alpha: temporal bonus value
            theta: minimum reward to receive bonus
        """
        self.reward_computer = reward_computer
        self.mu = mu
        self.alpha = alpha
        self.theta = theta

    def apply(self, completions, ground_truths, base_rewards, **reward_kwargs):
        """Apply T-GRPO temporal bonus.

        Args:
            completions: list[str] — generated completion texts
            ground_truths: list[str] — GT motion token texts
            base_rewards: tensor (N,) — base rewards from reward_computer
            **reward_kwargs: forwarded to reward_computer (e.g., caption=...)

        Returns:
            final_rewards: tensor (N,) — rewards with temporal bonus
            temporal_active: bool — whether bonus was applied
        """
        # Compute shuffled rewards
        shuffled_rewards = compute_shuffled_rewards(
            completions, ground_truths, self.reward_computer, **reward_kwargs
        )

        # Apply T-GRPO comparison
        final_rewards, temporal_active = compute_temporal_bonus(
            ordered_rewards=base_rewards,
            shuffled_rewards=shuffled_rewards,
            mu=self.mu,
            alpha=self.alpha,
            theta=self.theta,
        )

        return final_rewards, temporal_active


# =====================================================================
# Direct Temporal Reward for train_grpo.py
# =====================================================================

class DirectTemporalReward:
    """Direct temporal quality reward (no contrastive comparison).

    Computes frame-level smoothness metrics and combines with base rewards.

    Usage:
        temporal_reward = DirectTemporalReward(weight=0.2)
        final_rewards = temporal_reward.apply(completions, base_rewards)
    """

    def __init__(self, weight=0.2):
        """
        Args:
            weight: weight of temporal reward in final combination
                    final = (1 - weight) * base + weight * temporal
        """
        self.weight = weight

    def apply(self, completions, base_rewards):
        """Blend base rewards with temporal quality rewards.

        Args:
            completions: list[str] — generated completion texts
            base_rewards: tensor (N,) — base rewards

        Returns:
            final_rewards: tensor (N,) — blended rewards
        """
        if not _TRL_DECODE_AVAILABLE:
            return base_rewards

        temporal_scores = []
        for text in completions:
            motion_content = extract_motion_content(text)
            tokens = parse_motion_token_ids(motion_content) if motion_content else []
            r, _ = r_temporal_quality(tokens)
            temporal_scores.append(r)

        temporal_tensor = torch.tensor(temporal_scores, dtype=torch.float32)
        temporal_tensor = temporal_tensor.to(base_rewards.device)

        final = (1 - self.weight) * base_rewards + self.weight * temporal_tensor
        return final
