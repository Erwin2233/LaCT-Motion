"""Inference script: text prompt -> 3D motion generation -> visualization.

Loads a trained CoconutMotion checkpoint, generates motion tokens from
text prompts, decodes them through VQ-VAE, and renders 3D skeleton
animations.

Usage:
    python inference.py configs/t2m_coconut.yaml \
        --checkpoint checkpoints/checkpoint-epoch29 \
        --text "a person walks forward" \
        --output results/

    python inference.py configs/t2m_coconut.yaml \
        --checkpoint checkpoints/checkpoint-epoch29 \
        --text-file prompts.txt \
        --output results/
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import yaml

from transformers import AutoModelForCausalLM

from build_data import build_question, load_vqvae
from coconut_motion import CoconutMotion
from train import setup_tokenizer_and_model, _load_lora_checkpoint, _as_bool
from utils import Config, parse_motion_tokens, set_seed

# ---------------------------------------------------------------------------
# Import motion utilities from Motion-R1 (avoid name collision with local utils.py)
# ---------------------------------------------------------------------------
from _paths import project_path

RESOURCE_ROOT = project_path()

DEFAULT_VQVAE_PATH = os.path.join(RESOURCE_ROOT, "ckpt", "vqvae.pth")
DEFAULT_META_DIR = os.path.join(
    RESOURCE_ROOT,
    "checkpoints", "t2m", "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta",
)
NUM_JOINTS = 22
FPS = 20


def _load_module(filepath, name):
    """Load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_motion_utils = _load_module(
    project_path("third_party", "motion_r1", "utils", "motion_utils.py"), "motion_utils"
)
_param_util = _load_module(
    project_path("evaluation", "common", "eval_infra", "paramUtil.py"), "paramUtil"
)
recover_from_ric = _motion_utils.recover_from_ric
plot_3d_motion = _motion_utils.plot_3d_motion
t2m_kinematic_chain = _param_util.t2m_kinematic_chain


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_dir, configs, device):
    """Load trained CoconutMotion model from a checkpoint directory.

    Uses ``setup_tokenizer_and_model`` to build the base model with all
    special tokens, then loads trained weights from *checkpoint_dir*.
    Supports both full fine-tuned checkpoints and LoRA checkpoints
    (containing ``adapter_model/`` + ``new_token_embeddings.pt``).
    """
    (
        model, tokenizer, latent_id, start_id, end_id, _, _, nb_text_tokens,
    ) = setup_tokenizer_and_model(configs)

    use_lora = _as_bool(getattr(configs, "use_lora", False), False)
    is_lora_ckpt = use_lora and os.path.isdir(os.path.join(checkpoint_dir, "adapter_model"))

    if is_lora_ckpt:
        # LoRA checkpoint: load adapter weights + new token embeddings
        _load_lora_checkpoint(model, nb_text_tokens, checkpoint_dir, rank=0)
    else:
        # Full fine-tuned checkpoint: load full model weights
        target = model
        # For non-LoRA PeftModel shouldn't happen, but handle gracefully
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            target = model.base_model.model
        loaded = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            trust_remote_code=True,
        )
        target.load_state_dict(loaded.state_dict(), strict=False)
        del loaded

    if configs.coconut:
        model = CoconutMotion(
            model, latent_id, start_id, end_id, tokenizer.eos_token_id,
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer, latent_id, start_id, end_id


def load_normalization(meta_dir):
    """Load mean/std arrays for motion denormalization."""
    mean = np.load(os.path.join(meta_dir, "mean.npy"))
    std = np.load(os.path.join(meta_dir, "std.npy"))
    return mean, std


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def build_input_ids(text, tokenizer, configs, latent_id, start_id, end_id, device):
    """Tokenize a text prompt and append latent tokens based on mode.

    - coconut mode: max-stage latent tokens
    - cot / no_cot mode: no latent tokens (question only)
    """
    prompt = build_question(text)
    question_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if getattr(configs, "cot", False) or getattr(configs, "no_cot", False):
        # Pure CoT or no-CoT: no latent tokens at inference
        tokens = question_ids
    else:
        n_latent = configs.max_latent_stage * configs.c_thought
        tokens = question_ids + [start_id] + [latent_id] * n_latent + [end_id]

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def decode_motion(code_indices, vqvae, mean, std, device):
    """Decode motion code indices to 3D joint positions.

    Returns:
        joints_3d: numpy ``(T, 22, 3)`` or *None* if empty.
        motion_denorm: numpy ``(T, 263)`` or *None*.
    """
    if len(code_indices) == 0:
        return None, None

    codes = torch.tensor(code_indices, dtype=torch.long, device=device)

    with torch.inference_mode():
        motion_norm = vqvae.forward_decoder(codes)  # (1, T, 263)

    motion_norm_np = motion_norm[0].cpu().float().numpy()
    motion_denorm = motion_norm_np * std + mean

    joints_3d = recover_from_ric(
        torch.from_numpy(motion_denorm).float(), NUM_JOINTS,
    )
    return joints_3d.numpy(), motion_denorm


def generate_motion(
    text, model, tokenizer, configs,
    latent_id, start_id, end_id,
    vqvae, mean, std, device,
):
    """Full pipeline: text -> motion codes -> 3D joints."""
    input_ids, attention_mask = build_input_ids(
        text, tokenizer, configs, latent_id, start_id, end_id, device,
    )

    with torch.inference_mode():
        generation_kwargs = {
            "max_new_tokens": configs.max_new_tokens,
            "synced_gpus": False,
        }
        if getattr(configs, "do_sample", False):
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = getattr(configs, "temperature", 1.0)
            generation_kwargs["top_p"] = getattr(configs, "top_p", 1.0)
            generation_kwargs["top_k"] = getattr(configs, "top_k", 0)
            generation_kwargs["repetition_penalty"] = getattr(
                configs, "repetition_penalty", 1.0
            )
        outputs = model.generate(
            input_ids, attention_mask,
            **generation_kwargs,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    codes = parse_motion_tokens(generated_text)
    joints_3d, motion_features = decode_motion(codes, vqvae, mean, std, device)

    return {
        "text": text,
        "joints_3d": joints_3d,
        "motion_features": motion_features,
        "codes": codes,
        "generated_text": generated_text,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(result, output_dir, idx=0, render_mode="skeleton"):
    """Save motion as .npy and render .mp4 visualization.

    Args:
        render_mode: "skeleton" for stick figure (default),
                     "body" for capsule body mesh (paper quality).
    """
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"{idx:04d}"
    text = result["text"]
    joints_3d = result["joints_3d"]

    if joints_3d is None:
        print(f"  [{prefix}] No motion generated for: {text[:60]}")
        return

    # Save joints
    npy_path = os.path.join(output_dir, f"{prefix}_joints.npy")
    np.save(npy_path, joints_3d)

    # Save features
    feat_path = os.path.join(output_dir, f"{prefix}_features.npy")
    np.save(feat_path, result["motion_features"])

    # Save metadata
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "text": text,
                "codes": result["codes"],
                "n_frames": int(joints_3d.shape[0]),
            },
            f,
            indent=2,
        )

    # Render MP4
    if render_mode == "smpl":
        from render_smpl import render_smpl_video
        mp4_path = os.path.join(output_dir, f"{prefix}_motion.mp4")
        try:
            render_smpl_video(
                joints_3d, mp4_path, title=text[:80], fps=FPS,
            )
            print(f"  [{prefix}] {joints_3d.shape[0]} frames -> {mp4_path} (SMPL mesh)")
        except Exception as exc:
            print(f"  [{prefix}] {joints_3d.shape[0]} frames, SMPL render failed: {exc}")
            print(f"           joints saved to {npy_path}")
    else:
        mp4_path = os.path.join(output_dir, f"{prefix}_motion.mp4")
        try:
            plot_3d_motion(
                mp4_path, t2m_kinematic_chain, joints_3d,
                title=text[:80], fps=FPS, radius=4,
            )
            print(f"  [{prefix}] {joints_3d.shape[0]} frames -> {mp4_path}")
        except Exception as exc:
            print(f"  [{prefix}] {joints_3d.shape[0]} frames, MP4 render failed: {exc}")
            print(f"           joints saved to {npy_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Latent-CoT-Motion: text to 3D motion generation",
    )
    parser.add_argument("config_file", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint", required=True, help="Trained checkpoint directory",
    )
    parser.add_argument("--text", type=str, default=None, help="Single text prompt")
    parser.add_argument(
        "--text-file", type=str, default=None,
        help="File with one prompt per line",
    )
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--vqvae-path", default=DEFAULT_VQVAE_PATH)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--render-mode", choices=["skeleton", "smpl"], default="smpl",
        help="Visualization mode: 'skeleton' for stick figure, "
             "'smpl' for SMPL body mesh (paper quality, requires smplx)",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    if args.max_new_tokens is not None:
        configs.max_new_tokens = args.max_new_tokens
    if args.do_sample:
        configs.do_sample = True
    if args.temperature is not None:
        configs.temperature = args.temperature
    if args.top_p is not None:
        configs.top_p = args.top_p
    if args.top_k is not None:
        configs.top_k = args.top_k
    if args.repetition_penalty is not None:
        configs.repetition_penalty = args.repetition_penalty
    set_seed(args.seed)

    device = args.device

    print("Loading model...")
    model, tokenizer, latent_id, start_id, end_id = load_model(
        args.checkpoint, configs, device,
    )

    print("Loading VQ-VAE decoder...")
    vqvae = load_vqvae(args.vqvae_path, device)

    print("Loading normalization stats...")
    mean, std = load_normalization(args.meta_dir)

    # Collect prompts
    texts = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        with open(args.text_file) as f:
            texts.extend([line.strip() for line in f if line.strip()])
    if not texts:
        parser.error("Provide --text or --text-file")

    print(f"\nGenerating {len(texts)} motion(s)...\n")
    for i, text in enumerate(texts):
        print(f"[{i + 1}/{len(texts)}] '{text}'")
        result = generate_motion(
            text, model, tokenizer, configs,
            latent_id, start_id, end_id,
            vqvae, mean, std, device,
        )
        print(f"  codes ({len(result['codes'])}): {result['codes'][:10]}{'...' if len(result['codes']) > 10 else ''}")
        save_results(result, args.output, idx=i, render_mode=args.render_mode)

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
