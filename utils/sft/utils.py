import os
import re
import random
from typing import List, Tuple

import numpy as np
import torch


class Config:
    """Convert a dictionary to an object with attribute access."""

    # Keys that must be interpreted as booleans.
    _BOOL_KEYS = frozenset({
        "coconut", "cot", "no_cot", "bf16", "only_eval",
        "save_only_improve", "reset_optimizer", "pad_latent_to_max",
        "deterministic", "use_fsdp",
    })

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if key in self._BOOL_KEYS and not isinstance(value, bool):
                value = str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
            setattr(self, key, value)

        # Validate mutually exclusive modes
        modes = [
            name for name in ("coconut", "cot", "no_cot")
            if getattr(self, name, False)
        ]
        if len(modes) > 1:
            raise ValueError(
                f"Mutually exclusive modes enabled simultaneously: {modes}. "
                "Set exactly one of coconut, cot, no_cot to true."
            )


def set_seed(seed_value: int, deterministic: bool = False):
    """Set random seed for reproducibility across all libraries.

    Args:
        seed_value: Seed for random, numpy, and torch RNGs.
        deterministic: If True, force deterministic cuDNN kernels
            (slower but reproducible).  If False (default), use
            cuDNN benchmark for throughput.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def parse_motion_tokens(text: str) -> List[int]:
    """Extract motion code indices from a motion token string.

    Example:
        '<Motion><Motion_86><Motion_301></Motion>' -> [86, 301]
    """
    return [int(x) for x in re.findall(r"<Motion_(\d+)>", text)]


def compute_motion_accuracy(
    predicted_codes: List[int],
    ground_truth_codes: List[int],
) -> Tuple[float, float, float]:
    """Compute motion generation metrics.

    Returns:
        exact_match: 1.0 if sequences are identical, else 0.0
        token_accuracy: matches in overlapping prefix / ground_truth length
        length_ratio: len(predicted) / len(ground_truth)
    """
    exact_match = 1.0 if predicted_codes == ground_truth_codes else 0.0

    min_len = min(len(predicted_codes), len(ground_truth_codes))
    gt_len = max(len(ground_truth_codes), 1)
    if min_len == 0:
        token_accuracy = 0.0
    else:
        matches = sum(
            1
            for p, g in zip(predicted_codes[:min_len], ground_truth_codes[:min_len])
            if p == g
        )
        token_accuracy = matches / gt_len

    length_ratio = len(predicted_codes) / gt_len

    return exact_match, token_accuracy, length_ratio


def compute_scheduled_stage(
    epoch: int, epochs_per_stage: int, stage0_extra_epochs: int = 0
) -> int:
    """Compute curriculum stage for a given epoch.

    Stage 0 lasts ``epochs_per_stage + stage0_extra_epochs`` epochs to allow
    the model to converge on pure CoT SFT before latent tokens are introduced.
    Subsequent stages each last ``epochs_per_stage`` epochs.

    Example with epochs_per_stage=3, stage0_extra_epochs=7:
        Epoch  0-9  -> stage 0  (10 epochs)
        Epoch 10-12 -> stage 1
        Epoch 13-15 -> stage 2
        ...
    """
    stage0_epochs = epochs_per_stage + stage0_extra_epochs
    if epoch < stage0_epochs:
        return 0
    return (epoch - stage0_epochs) // epochs_per_stage + 1


def format_stage_info(
    epoch: int, epochs_per_stage: int, max_stage: int,
    stage0_extra_epochs: int = 0,
) -> str:
    """Format curriculum stage information for logging."""
    stage = min(
        compute_scheduled_stage(epoch, epochs_per_stage, stage0_extra_epochs),
        max_stage,
    )
    return f"Epoch {epoch} | Stage {stage}/{max_stage}"


def compute_scheduled_stage_by_step(
    global_update_step: int,
    steps_per_stage: float,
    stage0_extra_steps: float = 0.0,
) -> int:
    """Compute curriculum stage based on optimizer step count.

    Like ``compute_scheduled_stage`` but operates on steps instead of epochs,
    enabling fractional ``epochs_per_stage`` (e.g. 0.5 for two stage
    transitions per epoch).

    Stage 0 lasts ``steps_per_stage + stage0_extra_steps`` optimizer steps.
    Subsequent stages each last ``steps_per_stage`` steps.
    """
    stage0_steps = steps_per_stage + stage0_extra_steps
    if global_update_step < stage0_steps:
        return 0
    return int((global_update_step - stage0_steps) // steps_per_stage) + 1


def format_stage_info_by_step(
    global_update_step: int,
    steps_per_stage: float,
    max_stage: int,
    stage0_extra_steps: float = 0.0,
    epoch: int | None = None,
) -> str:
    """Format curriculum stage information for step-based scheduling."""
    stage = min(
        compute_scheduled_stage_by_step(
            global_update_step, steps_per_stage, stage0_extra_steps,
        ),
        max_stage,
    )
    prefix = f"Epoch {epoch} | " if epoch is not None else ""
    return f"{prefix}Step {global_update_step} | Stage {stage}/{max_stage}"
