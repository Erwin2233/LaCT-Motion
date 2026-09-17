"""Build Coconut-format T2M training data.

Joins three data sources:
  1. Thinking steps (texts_think_steps.json or final_think_steps.json)
  2. HumanML3D captions (texts/{ID}.txt)
  3. Motion tokens via VQ-VAE encoding from raw .npy motion data

Outputs JSON files with schema:
  {
    "question": "<chat template prompt>",
    "steps": ["step1", "step2", ...],
    "answer": "<Motion><Motion_86>...</Motion><|im_end|>"
  }

Reference: Motion-R1_copy/create_chatml_dataset_new.py
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from _paths import project_path
from third_party.motion_r1.models.vqvae import HumanVQVAE

# T2M system prompt and instruction (from UniMo training_utils.py)
SYSTEM_PROMPT = (
    "You are an assistant who helps users generate 3D human motion "
    "representations."
)
INSTRUCTION = (
    "### Instruction:\n"
    "The users will describe a motion, your job is to generate a motion "
    "matching the following input human motion description. Show your "
    "reasoning inside <think>...</think> and output motion in "
    "<Motion>...</Motion> tags."
)

# VQ-VAE default parameters (matching Motion-R1 / UniMo)
VQVAE_ARGS = types.SimpleNamespace(
    dataname="t2m",
    nb_joints=22,
    nb_code=512,
    code_dim=512,
    output_emb_width=512,
    down_t=2,
    stride_t=2,
    width=512,
    depth=3,
    dilation_growth_rate=3,
    vq_act="relu",
    vq_norm=None,
    quantizer="ema_reset",
    mu=0.99,
    beta=1.0,
)


def build_question(caption: str) -> str:
    """Build the Qwen ChatML question prompt for T2M."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n{INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n### Input:\n{caption}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_think_steps(path: str) -> dict:
    """Load think_steps.json -> {id: [step1, step2, ...]}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry["steps"] for entry in data}


def parse_caption_line(line: str):
    """Parse a caption line from HumanML3D texts/{ID}.txt.

    Format: caption#pos_tags#f_tag#to_tag[#optional_fields]
    Returns (caption, f_tag, to_tag) or None.
    """
    parts = line.strip().split("#")
    if len(parts) < 4:
        return None
    caption = parts[0].strip()
    try:
        f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
        to_tag = float(parts[3]) if parts[3] != "nan" else 0.0
    except ValueError:
        f_tag, to_tag = 0.0, 0.0
    return caption, f_tag, to_tag


def load_split_ids(split_path: str) -> list:
    """Load motion IDs from a split file (train.txt, val.txt, etc.)."""
    return [
        line.strip()
        for line in Path(split_path).read_text().splitlines()
        if line.strip()
    ]


def load_vqvae(vqvae_path: str, device: str):
    """Load pre-trained VQ-VAE model."""
    vqvae = HumanVQVAE(
        VQVAE_ARGS,
        VQVAE_ARGS.nb_code,
        VQVAE_ARGS.code_dim,
        VQVAE_ARGS.output_emb_width,
        VQVAE_ARGS.down_t,
        VQVAE_ARGS.stride_t,
        VQVAE_ARGS.width,
        VQVAE_ARGS.depth,
        VQVAE_ARGS.dilation_growth_rate,
        VQVAE_ARGS.vq_act,
        VQVAE_ARGS.vq_norm,
    )
    ckpt = torch.load(vqvae_path, map_location="cpu", weights_only=False)
    vqvae.load_state_dict(ckpt["net"], strict=True)
    vqvae.eval()
    vqvae.to(device)
    return vqvae


def encode_motion(
    motion_np: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    vqvae,
    device: str,
    unit_length: int = 4,
) -> str:
    """Encode raw motion .npy to <Motion_N> token string.

    Steps:
      1. Align length to unit_length
      2. Normalize with mean/std
      3. VQ-VAE encode → discrete codes (0..511)
      4. Format as '<Motion_0><Motion_123>...'
    """
    L = (len(motion_np) // unit_length) * unit_length
    if L == 0:
        return ""

    motion_norm = (motion_np[:L] - mean) / np.where(std < 1e-8, 1.0, std)
    motion_t = torch.from_numpy(motion_norm).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        codes = vqvae.encode(motion_t).squeeze(0)
    if codes.ndim == 2:
        codes = codes.reshape(-1)

    code_list = codes.cpu().tolist()
    return "".join(f"<Motion_{int(c)}>" for c in code_list)


def slice_motion_tokens(
    motion_token_str: str, f_tag: float, to_tag: float, fps: float, unit_length: int
) -> str:
    """Slice motion token string for segment-level captions.

    When f_tag and to_tag are non-zero, the caption only describes a
    temporal segment of the full motion. Slice the tokens accordingly.
    """
    if f_tag == 0.0 and to_tag == 0.0:
        return motion_token_str

    # Parse all tokens
    import re

    tokens = re.findall(r"<Motion_\d+>", motion_token_str)
    if not tokens:
        return ""

    start = int(f_tag * fps / unit_length)
    end = int(to_tag * fps / unit_length)
    start = max(0, start)
    end = min(end, len(tokens))
    if end <= start:
        return ""

    return "".join(tokens[start:end])


def build_dataset(
    split_ids: list,
    texts_dir: Path,
    motion_dir: Path,
    think_steps: dict,
    vqvae,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    max_steps: int = 20,
    min_motion_len: int = 40,
    max_motion_len: int = 200,
    fps: float = 20.0,
    unit_length: int = 4,
) -> tuple:
    """Build dataset for one split by encoding motions with VQ-VAE."""
    records = []
    stats = {
        "total_ids": len(split_ids),
        "total_captions": 0,
        "matched": 0,
        "no_text_file": 0,
        "no_motion_file": 0,
        "motion_too_short": 0,
        "motion_too_long": 0,
        "no_think_steps": 0,
        "too_few_steps": 0,
        "empty_slice": 0,
    }

    # Pre-encode all motions for this split
    motion_cache = {}
    for name in tqdm(split_ids, desc="Encoding motions"):
        motion_path = motion_dir / f"{name}.npy"
        if not motion_path.exists():
            stats["no_motion_file"] += 1
            continue

        motion_np = np.load(motion_path)
        if len(motion_np) < min_motion_len:
            stats["motion_too_short"] += 1
            continue
        if len(motion_np) >= max_motion_len:
            stats["motion_too_long"] += 1
            continue

        motion_str = encode_motion(motion_np, mean, std, vqvae, device, unit_length)
        if motion_str:
            motion_cache[name] = (motion_np, motion_str)

    print(f"  Encoded {len(motion_cache)} motions")

    # Build records by iterating over captions
    for name in tqdm(split_ids, desc="Building records"):
        text_path = texts_dir / f"{name}.txt"
        if not text_path.exists():
            stats["no_text_file"] += 1
            continue
        if name not in motion_cache:
            continue

        _, full_motion_str = motion_cache[name]
        lines = text_path.read_text(encoding="utf-8").splitlines()

        for idx, line in enumerate(lines):
            parsed = parse_caption_line(line)
            if parsed is None:
                continue

            caption, f_tag, to_tag = parsed
            stats["total_captions"] += 1
            sample_id = f"{name}_{idx}"

            # Look up thinking steps
            steps = think_steps.get(sample_id)
            if steps is None:
                stats["no_think_steps"] += 1
                continue

            # Filter single-step entries (useless for curriculum)
            if len(steps) < 2:
                stats["too_few_steps"] += 1
                continue

            # Slice motion tokens for segment-level captions
            motion_str = slice_motion_tokens(
                full_motion_str, f_tag, to_tag, fps, unit_length
            )
            if not motion_str:
                stats["empty_slice"] += 1
                continue

            steps = steps[:max_steps]
            question = build_question(caption)
            answer = f"<Motion>{motion_str}</Motion><|im_end|>"

            records.append(
                {
                    "question": question,
                    "steps": steps,
                    "answer": answer,
                }
            )
            stats["matched"] += 1

    return records, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build Coconut-format T2M training data with VQ-VAE encoding."
    )
    parser.add_argument(
        "--think-steps",
        default=project_path("data", "texts_think_steps.json"),
    )
    parser.add_argument(
        "--texts-dir",
        default=project_path("dataset", "HumanML3D", "texts"),
    )
    parser.add_argument(
        "--motion-dir",
        default=project_path("dataset", "HumanML3D", "new_joint_vecs"),
        help="Directory containing {ID}.npy raw motion files (263-dim joint vectors).",
    )
    parser.add_argument(
        "--splits-dir",
        default=project_path("dataset", "HumanML3D"),
    )
    parser.add_argument(
        "--vqvae-path",
        default=project_path("ckpt", "vqvae.pth"),
        help="Path to pre-trained VQ-VAE checkpoint.",
    )
    parser.add_argument(
        "--meta-dir",
        default=project_path("checkpoints", "t2m", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta"),
        help="Directory containing mean.npy and std.npy for motion normalization.",
    )
    parser.add_argument(
        "--output-dir",
        default=project_path("data", "rebuilt"),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-motion-len", type=int, default=40)
    parser.add_argument("--max-motion-len", type=int, default=200)
    parser.add_argument(
        "--splits", default="train,val,test", help="Comma-separated splits"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load thinking steps
    print("Loading thinking steps...")
    think_steps = load_think_steps(args.think_steps)
    print(f"  Loaded {len(think_steps)} entries")

    # Load VQ-VAE
    print(f"Loading VQ-VAE from {args.vqvae_path}...")
    vqvae = load_vqvae(args.vqvae_path, args.device)
    print("  VQ-VAE loaded")

    # Load normalization parameters
    mean = np.load(str(Path(args.meta_dir) / "mean.npy"))
    std = np.load(str(Path(args.meta_dir) / "std.npy"))
    print(f"  Normalization: mean shape={mean.shape}, std shape={std.shape}")

    unit_length = 2 ** VQVAE_ARGS.down_t  # 4

    texts_dir = Path(args.texts_dir)
    motion_dir = Path(args.motion_dir)
    splits_dir = Path(args.splits_dir)

    # Process each split
    for split in args.splits.split(","):
        split = split.strip()
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Skipping {split}: {split_file} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Processing split: {split}")
        print(f"{'='*50}")

        split_ids = load_split_ids(str(split_file))
        print(f"  Split IDs: {len(split_ids)}")

        records, stats = build_dataset(
            split_ids,
            texts_dir,
            motion_dir,
            think_steps,
            vqvae,
            mean,
            std,
            args.device,
            max_steps=args.max_steps,
            min_motion_len=args.min_motion_len,
            max_motion_len=args.max_motion_len,
            unit_length=unit_length,
        )

        out_path = output_dir / f"t2m_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  Output: {out_path} ({len(records)} samples)")
        print(f"  Stats: {json.dumps(stats, indent=2)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
