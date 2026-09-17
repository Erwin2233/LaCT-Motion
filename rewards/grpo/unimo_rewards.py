"""UniMo-style reward computation for GRPO training.

Faithfully replicates UniMo's reward logic:
  1. Decode pred & GT motion tokens via VQ-VAE
  2. Pad/truncate poses to fixed length (196 frames)
  3. Call get_co_embeddings TWICE (pred + GT), yielding:
       (et_pred, em_pred) and (et, em)
  4. motion_cos_sim  = cos(em_pred, em)   -- motion-motion similarity
     motion_text_sim = cos(em_pred, et)   -- motion-text similarity
  5. reward = motion_cos_sim + motion_text_sim

Drop-in replacement for RewardComputer:
    reward_computer = UniMoRewardComputer(device)
    rewards, details = reward_computer.compute_detailed(completions, gt_texts, caption=caps)

Paths follow the same conventions as evaluate.py / build_data.py.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Motion-R1 path (same as evaluate.py / build_data.py)
# ---------------------------------------------------------------------------
from _paths import project_path

RESOURCE_ROOT = project_path()

GLOVE_DIR = os.path.join(RESOURCE_ROOT, "glove")
EVAL_OPT_PATH = os.path.join(
    RESOURCE_ROOT, "checkpoints", "t2m", "Comp_v6_KLD005", "opt.txt"
)
DEFAULT_VQVAE_PATH = os.path.join(RESOURCE_ROOT, "ckpt", "vqvae.pth")

# VQ-VAE code is bundled under third_party.motion_r1.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MOTION_LENGTH = 196
MAX_TEXT_LEN = 20
_POS_TOKEN_USAGE_LOGGED = False

_MOTION_TOKEN_RE = re.compile(r"<Motion_(\d+)>")


# ---------------------------------------------------------------------------
# Lazy-loaded global model cache
# ---------------------------------------------------------------------------
_CACHE = {
    "vqvae": None,
    "eval_wrapper": None,
    "w_vectorizer": None,
    "device": None,
}


def _get_device(device=None):
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    if _CACHE["device"] is None:
        env = os.environ.get("T2M_DEVICE")
        if env:
            _CACHE["device"] = torch.device(env)
        elif torch.cuda.is_available():
            _CACHE["device"] = torch.device("cuda:0")
        else:
            _CACHE["device"] = torch.device("cpu")
    return _CACHE["device"]


def _load_vqvae(device):
    if _CACHE["vqvae"] is None:
        from third_party.motion_r1.models.vqvae import HumanVQVAE
        import types

        args = types.SimpleNamespace(
            dataname="t2m", nb_joints=22, nb_code=512, code_dim=512,
            output_emb_width=512, down_t=2, stride_t=2, width=512,
            depth=3, dilation_growth_rate=3, vq_act="relu", vq_norm=None,
            quantizer="ema_reset", mu=0.99, beta=1.0,
        )
        vqvae = HumanVQVAE(
            args, args.nb_code, args.code_dim, args.output_emb_width,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm,
        )
        ckpt = torch.load(DEFAULT_VQVAE_PATH, map_location="cpu", weights_only=False)
        vqvae.load_state_dict(ckpt["net"], strict=True)
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad = False
        vqvae.to(device)
        _CACHE["vqvae"] = vqvae
        print(f"[UniMoReward] VQ-VAE loaded from {DEFAULT_VQVAE_PATH}")
    return _CACHE["vqvae"]


def _load_eval_wrapper(device):
    if _CACHE["eval_wrapper"] is None:
        from eval_infra.evaluator_wrapper import EvaluatorModelWrapper
        from eval_infra.get_eval_option import get_opt

        opt = get_opt(EVAL_OPT_PATH, device)
        # Resolve evaluator checkpoints within the project
        opt.checkpoints_dir = os.path.join(RESOURCE_ROOT, "checkpoints")
        _CACHE["eval_wrapper"] = EvaluatorModelWrapper(opt)
        print("[UniMoReward] EvaluatorModelWrapper loaded")
    return _CACHE["eval_wrapper"]


def _load_w_vectorizer():
    if _CACHE["w_vectorizer"] is None:
        from eval_infra.word_vectorizer import WordVectorizer

        _CACHE["w_vectorizer"] = WordVectorizer(GLOVE_DIR, "our_vab")
        print(f"[UniMoReward] WordVectorizer loaded from {GLOVE_DIR}")
    return _CACHE["w_vectorizer"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def parse_motion_token_ids(text: str) -> list[int]:
    """Extract <Motion_N> token IDs from text."""
    return [int(m) for m in _MOTION_TOKEN_RE.findall(text)]


def extract_motion_content(text: str) -> str | None:
    """Extract content inside <Motion>...</Motion>."""
    m = re.search(r"<Motion>(.*?)</Motion>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def pad_or_truncate_poses(poses: torch.Tensor, max_len: int = MAX_MOTION_LENGTH):
    """Pad short / truncate long poses to fixed length.

    Args:
        poses: [T, D]

    Returns:
        (padded [max_len, D], actual_len int)
    """
    T, D = poses.shape
    actual_len = min(T, max_len)
    padded = torch.zeros(max_len, D, device=poses.device, dtype=poses.dtype)
    padded[:actual_len] = poses[:actual_len]
    return padded, actual_len


def get_text_embedding(caption: str, w_vectorizer, device, pos_tokens=None):
    """Encode caption into word embeddings + POS one-hots.

    Args:
        caption: raw text caption (used as fallback if pos_tokens is None)
        w_vectorizer: WordVectorizer instance
        device: torch device
        pos_tokens: optional list of pre-tokenized "word/POS" strings from UniMo
                    (e.g. ["a/DET", "man/NOUN", ...]). When provided, uses correct
                    POS tags instead of defaulting to OTHER.

    Returns:
        word_embeddings: [1, max_text_len+2, 300]
        pos_one_hots:    [1, max_text_len+2, pos_dim]
        sent_len:        [1]
    """
    global _POS_TOKEN_USAGE_LOGGED
    if pos_tokens:
        # Use pre-tokenized tokens with correct POS tags (same as UniMo)
        tokens = [str(tok).strip() for tok in pos_tokens if str(tok).strip()]
        if not _POS_TOKEN_USAGE_LOGGED:
            print(f"[UniMoReward] Using POS-tagged text tokens, e.g. {tokens[:4]}")
            _POS_TOKEN_USAGE_LOGGED = True
    else:
        # Fallback: tokenize caption with POS=OTHER
        words = caption.lower().strip().replace("\n", " ").replace("\r", " ").split()
        clean_words = [w.replace("/", "") for w in words if w]
        clean_words = [w for w in clean_words if w]
        tokens = [f"{w}/OTHER" for w in clean_words]

    if len(tokens) < MAX_TEXT_LEN:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
    else:
        tokens = tokens[:MAX_TEXT_LEN]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)

    word_embs, pos_ohs = [], []
    for tok in tokens:
        we, po = w_vectorizer[tok]
        word_embs.append(we)
        pos_ohs.append(po)

    word_embs = torch.tensor(
        np.stack(word_embs), dtype=torch.float32, device=device
    ).unsqueeze(0)
    pos_ohs = torch.tensor(
        np.stack(pos_ohs), dtype=torch.float32, device=device
    ).unsqueeze(0)
    sent_len_t = torch.tensor([sent_len], dtype=torch.long, device=device)

    return word_embs, pos_ohs, sent_len_t


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------
# T2M task: completion is <Motion><Motion_N>+</Motion> (no <think> tags)
# Optionally supports <think>...</think> prefix if present
_FORMAT_PATTERN = re.compile(
    r"^(?:<think>.*?</think>)?<Motion>(<Motion_\d+>)+</Motion>$", re.DOTALL
)


def unimo_format_reward(text: str) -> float:
    """Strict format check: [optional <think>...</think>]<Motion><Motion_N>+</Motion>."""
    return 1.0 if _FORMAT_PATTERN.match(text) else 0.0


# ---------------------------------------------------------------------------
# Core similarity reward (faithful UniMo replication)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def unimo_similarity_reward(
    pred_text: str,
    gt_motion_text: str,
    caption: str,
    vqvae,
    eval_wrapper,
    w_vectorizer,
    device,
    pos_tokens=None,
) -> tuple[float, float, float]:
    """Compute UniMo-style similarity reward for a single sample.

    Args:
        pos_tokens: optional pre-tokenized "word/POS" list for accurate text embedding.

    Returns:
        (motion_cos_sim, motion_text_cos_sim, total_reward)
    """
    # --- Parse motion tokens ---
    motion_content = extract_motion_content(pred_text)
    pred_ids = parse_motion_token_ids(motion_content) if motion_content else []
    gt_ids = parse_motion_token_ids(gt_motion_text)

    if not pred_ids:
        pred_ids = [1]  # fallback
    if not gt_ids:
        gt_ids = [1]

    # --- Decode via VQ-VAE ---
    try:
        pred_codes = torch.tensor(pred_ids, dtype=torch.long, device=device)
        pred_pose_raw = vqvae.forward_decoder(pred_codes.unsqueeze(0))  # [1, T, D]
    except Exception:
        pred_pose_raw = vqvae.forward_decoder(
            torch.ones(1, 1, device=device, dtype=torch.long)
        )

    gt_codes = torch.tensor(gt_ids, dtype=torch.long, device=device)
    gt_pose_raw = vqvae.forward_decoder(gt_codes.unsqueeze(0))  # [1, T, D]

    # --- Squeeze [1,T,D] -> [T,D], pad, unsqueeze back ---
    pred_2d = pred_pose_raw.squeeze(0)
    gt_2d = gt_pose_raw.squeeze(0)

    pred_padded, pred_len = pad_or_truncate_poses(pred_2d, MAX_MOTION_LENGTH)
    gt_padded, gt_len = pad_or_truncate_poses(gt_2d, MAX_MOTION_LENGTH)

    pred_pose = pred_padded.unsqueeze(0)  # [1, 196, D]
    gt_pose = gt_padded.unsqueeze(0)      # [1, 196, D]

    # --- Text embedding ---
    word_embs, pos_ohs, sent_len = get_text_embedding(
        caption, w_vectorizer, device, pos_tokens=pos_tokens
    )

    # --- Two evaluator calls (faithful UniMo) ---
    et_pred, em_pred = eval_wrapper.get_co_embeddings(
        word_embs, pos_ohs, sent_len,
        pred_pose, torch.tensor([pred_len], dtype=torch.long, device=device),
    )
    et, em = eval_wrapper.get_co_embeddings(
        word_embs, pos_ohs, sent_len,
        gt_pose, torch.tensor([gt_len], dtype=torch.long, device=device),
    )

    # --- UniMo reward: cos(em_pred, em) + cos(em_pred, et) ---
    motion_cos_sim = F.cosine_similarity(em_pred, em, dim=1).item()
    motion_text_cos_sim = F.cosine_similarity(em_pred, et, dim=1).item()
    total = motion_cos_sim + motion_text_cos_sim

    return motion_cos_sim, motion_text_cos_sim, total


# ---------------------------------------------------------------------------
# UniMoRewardComputer: drop-in replacement for RewardComputer
# ---------------------------------------------------------------------------
class UniMoRewardComputer:
    """UniMo-style reward computer.

    Provides the same interface as RewardComputer:
        rewards, details = computer.compute_detailed(completions, gt_texts, caption=...)
    """

    def __init__(self, device=None, reward_weights=None):
        self.device = _get_device(device)
        _CACHE["device"] = self.device
        self.vqvae = _load_vqvae(self.device)
        self.eval_wrapper = _load_eval_wrapper(self.device)
        self.w_vectorizer = _load_w_vectorizer()
        # reward_weights: [format_w, motion_sim_w, text_sim_w], default [1, 1, 1]
        if reward_weights is not None:
            self.w_fmt = float(reward_weights[0])
            self.w_motion = float(reward_weights[1])
            self.w_text = float(reward_weights[2])
        else:
            self.w_fmt = 1.0
            self.w_motion = 1.0
            self.w_text = 1.0
        print(f"[UniMoRewardComputer] Initialized on {self.device}, "
              f"weights: format={self.w_fmt}, motion_sim={self.w_motion}, text_sim={self.w_text}")

    def __call__(self, generated_texts, ground_truths, **kwargs):
        rewards, _ = self.compute_detailed(generated_texts, ground_truths, **kwargs)
        return rewards

    def compute_detailed(self, generated_texts, ground_truths, **kwargs):
        """Compute rewards with per-component breakdown.

        Args:
            generated_texts: list[str] - decoded completions
            ground_truths:   list[str] - GT motion token strings
            caption:         list[str] - text captions (in kwargs)
            pos_tokens:      list[list[str]|None] - pre-tokenized POS tokens (in kwargs)

        Returns:
            rewards:  torch.Tensor of shape (N,)
            details:  dict with keys 'format', 'motion_sim', 'text_sim', 'similarity'
        """
        captions = kwargs.get("caption", [""] * len(generated_texts))
        pos_tokens_list = kwargs.get("pos_tokens", [None] * len(generated_texts))

        format_scores = []
        motion_sims = []
        text_sims = []
        totals = []

        for gen, gt, cap, pt in zip(generated_texts, ground_truths, captions, pos_tokens_list):
            # Format
            fmt = unimo_format_reward(gen)
            format_scores.append(fmt)

            # Similarity
            m_sim, t_sim, total = unimo_similarity_reward(
                gen, gt, cap,
                self.vqvae, self.eval_wrapper, self.w_vectorizer, self.device,
                pos_tokens=pt,
            )
            motion_sims.append(m_sim)
            text_sims.append(t_sim)
            # Apply weights: format * w_fmt + motion_sim * w_motion + text_sim * w_text
            weighted = self.w_fmt * fmt + self.w_motion * m_sim + self.w_text * t_sim
            # If format is wrong, zero out the entire reward
            totals.append(weighted if fmt > 0 else 0.0)

        details = {
            "format": format_scores,
            "motion_sim": motion_sims,
            "text_sim": text_sims,
            "similarity": [m + t for m, t in zip(motion_sims, text_sims)],
        }

        rewards = torch.tensor(totals, dtype=torch.float32)
        return rewards, details
