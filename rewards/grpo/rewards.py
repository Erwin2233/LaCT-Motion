"""Reward functions for GRPO training of CoconutMotion.

Provides a unified interface to compute rewards from either:
- TRL's existing T2M reward functions (recommended, 10+ types available)
- Custom reward implementations (format_strict)

TRL T2M rewards: third_party/trl_motion/rewards/t2m_rewards.py
"""

import re
import sys
import os
import torch

# Reward functions are bundled inside this project.


# =====================================================================
# TRL Reward Function Imports (lazy, fail gracefully if TRL not found)
# =====================================================================
_TRL_REWARDS_AVAILABLE = False
_TRL_FUNC_MAP = {}

# Import each reward function individually so one missing function
# doesn't disable all others.
_TRL_REWARD_IMPORTS = {
    "format": "t2m_format_reward",
    "format_soft": "t2m_format_soft_reward",
    "motion_f1": "t2m_motion_f1_reward",
    "motion_lcs": "t2m_motion_lcs_reward",
    "motion_embedding": "t2m_motion_embedding_reward",
    "semantic": "t2m_semantic_reward",
    "phys": "t2m_phys_reward",
    "phys_joint": "t2m_phys_joint_reward",
    "phys_vel": "t2m_phys_vel_reward",
    "phys_skating": "t2m_phys_skating_reward",
    # UniMo-style combined reward functions
    "unified_format": "unified_format_reward_func",
    "unified_similarity": "unified_similarity_reward_func",
}

try:
    from third_party.trl_motion import rewards as _trl_rewards_module
    for short_name, func_name in _TRL_REWARD_IMPORTS.items():
        fn = getattr(_trl_rewards_module, func_name, None)
        if fn is not None:
            _TRL_FUNC_MAP[short_name] = fn
        else:
            print(f"[rewards.py] WARNING: {func_name} not found in TRL, skipping.")
    if _TRL_FUNC_MAP:
        _TRL_REWARDS_AVAILABLE = True
        print(f"[rewards.py] Loaded {len(_TRL_FUNC_MAP)}/{len(_TRL_REWARD_IMPORTS)} TRL reward functions.")
except ImportError:
    print("[rewards.py] WARNING: TRL rewards module not available.")


# =====================================================================
# Reward Presets (matching TRL t2m_rewards.py REWARD_PRESETS)
# =====================================================================
REWARD_PRESETS = {
    "basic": {
        "funcs": ["format_soft", "motion_f1"],
        "weights": [0.1, 0.9],
    },
    "with_physics": {
        "funcs": ["format_soft", "motion_f1", "phys"],
        "weights": [0.1, 0.7, 0.2],
    },
    "strict": {
        "funcs": ["format", "motion_lcs", "phys"],
        "weights": [0.1, 0.7, 0.2],
    },
    "semantic": {
        "funcs": ["format", "motion_embedding", "semantic"],
        "weights": [0.1, 0.45, 0.45],
    },
    "full": {
        "funcs": ["format_soft", "motion_f1", "semantic", "phys"],
        "weights": [0.1, 0.4, 0.3, 0.2],
    },
}

# Map is already built during import above.


# =====================================================================
# Local Reward Functions (not from TRL)
# =====================================================================
_MOTION_CODE_RE = re.compile(r"<Motion_\d+>")


def _coconut_format_score(text: str) -> float:
    """Score format correctness for CoconutMotion GRPO completions.

    A valid completion is exactly: <Motion><Motion_N>...<Motion_N></Motion>
    Returns 0.0 for any violation:
      - missing <Motion> or </Motion>
      - repeated </Motion> tokens  (the main bug this catches)
      - non-motion content inside the tags
      - zero motion codes
      - junk after the closing </Motion>
    """
    if not text:
        return 0.0

    # Must start with <Motion>
    if not text.startswith("<Motion>"):
        return 0.0

    # Count </Motion> occurrences — must be exactly 1
    end_count = text.count("</Motion>")
    if end_count == 0:
        return 0.0  # truncated, no closing tag
    if end_count > 1:
        return 0.0  # repeated </Motion>

    # Split at the single </Motion>
    idx = text.index("</Motion>")
    inner = text[len("<Motion>"):idx]
    after = text[idx + len("</Motion>"):]

    # Nothing should come after </Motion>
    if after.strip():
        return 0.0

    # Inner content must be non-empty valid <Motion_N> tokens only
    codes = _MOTION_CODE_RE.findall(inner)
    if not codes:
        return 0.0

    residual = _MOTION_CODE_RE.sub("", inner).strip()
    if residual:
        return 0.0  # garbage mixed in

    return 1.0


def coconut_format_strict_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    **kwargs,
) -> list[float]:
    """Strict format reward for CoconutMotion (TRL-compatible signature).

    Catches repeated </Motion>, missing tags, garbage content, etc.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        rewards.append(_coconut_format_score(content))
    return rewards


# Register local reward so RewardComputer can resolve it by name.
_TRL_FUNC_MAP["format_strict"] = coconut_format_strict_reward


# =====================================================================
# Main Reward Computation Interface
# =====================================================================
class RewardComputer:
    """Computes rewards for GRPO training.

    Usage:
        reward_computer = RewardComputer(preset="with_physics")
        rewards = reward_computer(generated_texts, ground_truths)
    """

    def __init__(self, preset="basic", custom_funcs=None, custom_weights=None):
        """
        Args:
            preset: Name of reward preset ("basic", "with_physics", "strict",
                    "semantic", "full")
            custom_funcs: List of reward function names (overrides preset funcs)
            custom_weights: List of weights. If custom_funcs is set, must match
                            its length. Otherwise overrides the preset's default
                            weights (must match the preset's func count).
        """
        if custom_funcs is not None:
            self.func_names = list(custom_funcs)
            self.weights = list(custom_weights) if custom_weights else [1.0] * len(self.func_names)
        elif preset in REWARD_PRESETS:
            cfg = REWARD_PRESETS[preset]
            self.func_names = cfg["funcs"]
            if custom_weights is not None:
                # Use preset funcs with user-specified weights
                self.weights = list(custom_weights)
            else:
                self.weights = cfg["weights"]
        else:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {list(REWARD_PRESETS.keys())}"
            )

        # Resolve function references
        self.reward_funcs = []
        for name in self.func_names:
            if name in _TRL_FUNC_MAP:
                self.reward_funcs.append(_TRL_FUNC_MAP[name])
            else:
                raise ValueError(
                    f"Unknown reward function: {name}. "
                    f"Available: {list(_TRL_FUNC_MAP.keys())}"
                )

        assert len(self.weights) == len(self.reward_funcs), (
            f"Mismatch: {len(self.weights)} weights vs "
            f"{len(self.reward_funcs)} functions"
        )

        print(f"[RewardComputer] Preset: {preset}")
        for name, weight in zip(self.func_names, self.weights):
            print(f"  {name}: weight={weight}")

    def __call__(self, generated_texts, ground_truths, **kwargs):
        """Compute weighted reward.

        Args:
            generated_texts: list[str] — decoded completion texts
            ground_truths: list[str] — GT motion token texts
            **kwargs: extra fields (caption, etc.)

        Returns:
            rewards: torch.Tensor of shape (len(generated_texts),)
        """
        # Convert to TRL message format
        completions = [[{"content": text}] for text in generated_texts]

        # Compute each reward
        all_rewards = []
        for func, weight in zip(self.reward_funcs, self.weights):
            try:
                r = func(completions, ground_truths, **kwargs)
                all_rewards.append(
                    [weight * (ri if ri is not None else 0.0) for ri in r]
                )
            except Exception as e:
                print(f"[RewardComputer] Error in {func.__name__}: {e}")
                all_rewards.append([0.0] * len(generated_texts))

        # Sum weighted rewards
        combined = [
            sum(r[i] for r in all_rewards)
            for i in range(len(generated_texts))
        ]

        return torch.tensor(combined, dtype=torch.float32)

    def compute_detailed(self, generated_texts, ground_truths, **kwargs):
        """Compute rewards with per-function breakdown for logging.

        Returns:
            rewards: torch.Tensor of shape (len(generated_texts),)
            details: dict[str, list[float]] — per-function rewards
        """
        completions = [[{"content": text}] for text in generated_texts]

        details = {}
        all_weighted = []

        for func, name, weight in zip(
            self.reward_funcs, self.func_names, self.weights
        ):
            try:
                r = func(completions, ground_truths, **kwargs)
                r_clean = [ri if ri is not None else 0.0 for ri in r]
                details[name] = r_clean
                all_weighted.append([weight * ri for ri in r_clean])
            except Exception as e:
                print(f"[RewardComputer] Error in {func.__name__}: {e}")
                details[name] = [0.0] * len(generated_texts)
                all_weighted.append([0.0] * len(generated_texts))

        combined = [
            sum(r[i] for r in all_weighted)
            for i in range(len(generated_texts))
        ]

        return torch.tensor(combined, dtype=torch.float32), details
