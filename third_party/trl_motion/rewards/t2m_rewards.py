# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""
Text-to-Motion (T2M) Reward Functions for TRL GRPO Training

支持的奖励类型 (10种):
1. format: 严格格式检查 (0.0 或 1.0)
2. format_soft: 软格式匹配 (0.0-1.0)
3. motion_f1: Token F1 分数 (集合匹配)
4. motion_lcs: 最长公共子序列 (顺序匹配)
5. motion_embedding: Motion embedding 相似度 (需要 VQ-VAE + Evaluator)
6. semantic: 文本-动作语义相似度 (需要 caption + Evaluator)
7. phys: 物理合理性 = 关节角度 + 速度平滑 + 脚部滑动
8. phys_joint: 关节旋转幅度违规惩罚 (基于 6D rotation → 旋转角度)
9. phys_vel: 速度平滑度惩罚 (加速度阈值)
10. phys_skating: 脚部滑动惩罚 (foot contact 约束)

预定义的奖励组合:
- basic: 格式(0.1) + F1(0.9)
- with_physics: 格式(0.1) + F1(0.7) + 物理(0.2)
- strict: 严格格式(0.1) + LCS(0.7) + 物理(0.2)
- semantic: 格式(0.1) + Embedding(0.45) + 语义(0.45)
- full: 格式(0.1) + F1(0.4) + 语义(0.3) + 物理(0.2)

物理约束说明:
- HumanML3D 数据格式 (263维)，VQ-VAE 下采样后帧率 10 FPS
- 关节旋转: 从 6D continuous rotation 转换为旋转角度，检查是否超过生理限制
- 速度平滑: 计算加速度，惩罚过大的加速度 (突然的速度变化)
- 脚部滑动: 当脚接触地面时，脚的速度应接近零

任务类型感知 (Task Type Awareness):
- TRL 会通过 **kwargs 传递数据集的 extra_info 字段
- extra_info 包含 task_type 字段 ("t2m" 或 "m2t")
- 可以根据任务类型使用不同的 reward 策略
- 参考 example_task_aware_reward.py 了解详细用法

环境变量配置:
- T2M_VERBOSE_REWARD=1: 打印详细的 reward 计算信息
- T2M_REWARD_LOG_INTERVAL=N: 每 N 个样本打印一次 (默认 10)
"""

import os
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# 全局配置: 是否打印 reward 详细信息
VERBOSE_REWARD = os.environ.get("T2M_VERBOSE_REWARD", "0") == "1"
REWARD_LOG_INTERVAL = int(os.environ.get("T2M_REWARD_LOG_INTERVAL", "10"))
PRINT_SAMPLE_DETAILS = os.environ.get("T2M_PRINT_SAMPLE_DETAILS", "0") == "1"  # 是否打印单个样本详情
_reward_call_count = 0
_pos_token_usage_logged = False

# T2M reward weights (adjustable via environment variables)
T2M_FORMAT_WEIGHT = float(os.environ.get("T2M_FORMAT_WEIGHT", "1.0"))
T2M_MOTION_WEIGHT = float(os.environ.get("T2M_MOTION_WEIGHT", "1.0"))
T2M_SEMANTIC_WEIGHT = float(os.environ.get("T2M_SEMANTIC_WEIGHT", "1.0"))

# M2T reward weights (adjustable via environment variables)
M2T_FORMAT_WEIGHT = float(os.environ.get("M2T_FORMAT_WEIGHT", "1.0"))
M2T_SEMANTIC_WEIGHT = float(os.environ.get("M2T_SEMANTIC_WEIGHT", "2.0"))

# 全局统计: 累积各项 reward 用于计算平均值
_reward_stats = {
    "format": [],
    "motion_embedding": [],
    "semantic": [],
    "total": [],
    "step_count": 0,
}

# =====================================================================
# T2M 模块导入 (从 trl.t2m 包导入，无需外部 ULM 依赖)
# =====================================================================
from third_party.trl_motion.t2m import GLOVE_PATH, CHECKPOINTS_PATH, VQVAE_PATH

# =====================================================================
# 全局模型缓存 (懒加载)
# =====================================================================
_MODELS_CACHE = {
    "eval_wrapper": None,
    "vqvae": None,
    "w_vectorizer": None,
    "mean": None,
    "std": None,
    "device": None,
    "clip_model": None,  # CLIP 模型 (用于 M2T 语义相似度)
    "clip_device": None,
}


def _get_device():
    """获取设备，支持通过 T2M_DEVICE 环境变量指定"""
    if _MODELS_CACHE["device"] is None:
        env_device = os.environ.get("T2M_DEVICE")
        if env_device:
            _MODELS_CACHE["device"] = torch.device(env_device)
            print(f"[T2M Reward] Using device from T2M_DEVICE: {env_device}")
        elif torch.cuda.is_available():
            _MODELS_CACHE["device"] = torch.device("cuda:0")
            print(f"[T2M Reward] Using CUDA device: cuda:0")
        else:
            _MODELS_CACHE["device"] = torch.device("cpu")
            print(f"[T2M Reward] Warning: CUDA not available, using CPU")
    return _MODELS_CACHE["device"]


def _load_evaluator_wrapper():
    """加载 EvaluatorModelWrapper"""
    if _MODELS_CACHE["eval_wrapper"] is None:
        try:
            from third_party.trl_motion.t2m.models.evaluator_wrapper import EvaluatorModelWrapper
            from third_party.trl_motion.t2m.options.get_eval_option import get_opt

            opt = get_opt(
                os.path.join(CHECKPOINTS_PATH, "t2m/Comp_v6_KLD005/opt.txt"),
                _get_device(),
                checkpoints_dir=CHECKPOINTS_PATH
            )
            _MODELS_CACHE["eval_wrapper"] = EvaluatorModelWrapper(opt)
            print(f"[T2M Reward] Loaded EvaluatorModelWrapper")
        except Exception as e:
            print(f"[T2M Reward] Warning: Failed to load EvaluatorModelWrapper: {e}")
            _MODELS_CACHE["eval_wrapper"] = None
    return _MODELS_CACHE["eval_wrapper"]


def _load_vqvae():
    """加载 VQ-VAE 模型用于解码 motion tokens"""
    if _MODELS_CACHE["vqvae"] is None:
        try:
            from third_party.trl_motion.t2m.models import vqvae as vqvae_module

            class VQVAEArgs:
                nb_joints = 22
                dataname = 't2m'
                nb_code = 512
                code_dim = 512
                mu = 0.99
                output_emb_width = 512
                down_t = 2
                stride_t = 2
                width = 512
                depth = 3
                dilation_growth_rate = 3
                vq_act = 'relu'
                vq_norm = None
                quantizer = 'ema_reset'

            args = VQVAEArgs()
            vqvae = vqvae_module.HumanVQVAE(
                args, args.nb_code, args.code_dim, args.output_emb_width,
                args.down_t, args.stride_t, args.width, args.depth,
                args.dilation_growth_rate, args.vq_act, args.vq_norm
            )

            possible_paths = [
                VQVAE_PATH,  # trl/ckpt/vqvae.pth (首选)
                os.path.join(CHECKPOINTS_PATH, "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"),
                os.path.join(CHECKPOINTS_PATH, "t2m/VQVAE/net_best_fid.tar"),
            ]

            ckpt_loaded = False
            for ckpt_path in possible_paths:
                if os.path.exists(ckpt_path):
                    print(f"[T2M Reward] Loading VQ-VAE from {ckpt_path}")
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    if 'net' in ckpt:
                        vqvae.load_state_dict(ckpt['net'], strict=True)
                    elif 'vqvae' in ckpt:
                        vqvae.load_state_dict(ckpt['vqvae'], strict=True)
                    else:
                        vqvae.load_state_dict(ckpt, strict=True)
                    ckpt_loaded = True
                    print(f"[T2M Reward] ✅ Loaded VQ-VAE from {ckpt_path}")
                    break

            if not ckpt_loaded:
                print(f"[T2M Reward] Warning: VQ-VAE checkpoint not found")
                _MODELS_CACHE["vqvae"] = None
                return None

            vqvae = vqvae.to(_get_device())
            vqvae.eval()
            for param in vqvae.parameters():
                param.requires_grad = False
            _MODELS_CACHE["vqvae"] = vqvae

        except Exception as e:
            print(f"[T2M Reward] Warning: Failed to load VQ-VAE: {e}")
            _MODELS_CACHE["vqvae"] = None
    return _MODELS_CACHE["vqvae"]


def _load_word_vectorizer():
    """加载 WordVectorizer 用于文本编码"""
    if _MODELS_CACHE["w_vectorizer"] is None:
        try:
            from third_party.trl_motion.t2m.utils.word_vectorizer import WordVectorizer
            _MODELS_CACHE["w_vectorizer"] = WordVectorizer(GLOVE_PATH, 'our_vab')
            print(f"[T2M Reward] Loaded WordVectorizer from {GLOVE_PATH}")
        except Exception as e:
            print(f"[T2M Reward] Warning: Failed to load WordVectorizer: {e}")
            _MODELS_CACHE["w_vectorizer"] = None
    return _MODELS_CACHE["w_vectorizer"]


def _load_normalization_params():
    """加载归一化参数"""
    if _MODELS_CACHE["mean"] is None:
        try:
            possible_meta_paths = [
                os.path.join(CHECKPOINTS_PATH, "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"),
                os.path.join(CHECKPOINTS_PATH, "t2m/VQVAE/meta"),
            ]
            for meta_root in possible_meta_paths:
                mean_path = os.path.join(meta_root, "mean.npy")
                std_path = os.path.join(meta_root, "std.npy")
                if os.path.exists(mean_path) and os.path.exists(std_path):
                    _MODELS_CACHE["mean"] = np.load(mean_path)
                    _MODELS_CACHE["std"] = np.load(std_path)
                    print(f"[T2M Reward] ✅ Loaded normalization params from {meta_root}")
                    break
            if _MODELS_CACHE["mean"] is None:
                _MODELS_CACHE["mean"] = np.zeros(263)
                _MODELS_CACHE["std"] = np.ones(263)
        except Exception as e:
            print(f"[T2M Reward] Warning: Failed to load normalization params: {e}")
            _MODELS_CACHE["mean"] = np.zeros(263)
            _MODELS_CACHE["std"] = np.ones(263)
    return _MODELS_CACHE["mean"], _MODELS_CACHE["std"]


def _load_clip_model():
    """
    懒加载 CLIP 模型 (用于 M2T 语义相似度)

    通过环境变量 M2T_USE_CLIP=1 启用 CLIP 方式
    """
    if _MODELS_CACHE["clip_model"] is None:
        try:
            import clip
            device_str = os.environ.get("T2M_DEVICE", "cuda:0")
            device = torch.device(device_str)
            model, _ = clip.load("ViT-B/32", device=device)
            model.eval()
            _MODELS_CACHE["clip_model"] = model
            _MODELS_CACHE["clip_device"] = device
            print(f"[M2T Reward] CLIP model (ViT-B/32) loaded on {device}")
        except Exception as e:
            print(f"[M2T Reward] Warning: Failed to load CLIP model: {e}")
            _MODELS_CACHE["clip_model"] = None
    return _MODELS_CACHE["clip_model"], _MODELS_CACHE["clip_device"]


# =====================================================================
# Motion Token 解析
# =====================================================================
def extract_motion_content(text: str) -> Optional[str]:
    """从文本中提取 <Motion>...</Motion> 标签内的内容"""
    match = re.search(r'<Motion>(.*?)</Motion>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_motion_token_ids(motion_str: str) -> list[int]:
    """解析 motion token 字符串，返回 token ID 列表"""
    if not motion_str:
        return []
    tokens = re.findall(r'<Motion_(\d+)>', motion_str)
    return [int(t) for t in tokens]


# =====================================================================
# 1) 格式奖励 r_format
# =====================================================================
_FORMAT_PATTERN = re.compile(
    r"<think>.+?</think>\s*<Motion>.+?</Motion>",
    flags=re.DOTALL | re.IGNORECASE,
)


def r_format(text: str, use_motion_tag: bool = True) -> float:
    """
    严格格式奖励: 检查是否包含完整的 <think>...</think><Motion>...</Motion> 结构
    Returns: 1.0 如果格式完全正确, 0.0 否则
    """
    if not text:
        return 0.0
    return 1.0 if _FORMAT_PATTERN.search(text) else 0.0


def r_format_soft(text: str) -> float:
    """
    软格式奖励: 渐进式评分
    Returns: 0.0-0.5 的分数
    """
    if not text:
        return 0.0

    score = 0.0
    text_lower = text.lower()

    if "<think>" in text_lower:
        score += 0.125
    if "</think>" in text_lower:
        score += 0.125
    if "<motion>" in text_lower or "<answer>" in text_lower:
        score += 0.125
    if "</motion>" in text_lower or "</answer>" in text_lower:
        score += 0.125

    return score


# =====================================================================
# 2) Motion Token 匹配奖励
# =====================================================================
def r_motion_token_f1(pred_tokens: list[int], gt_tokens: list[int]) -> float:
    """计算 motion token 的 F1 分数 (基于集合匹配)"""
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    intersection = pred_set & gt_set

    if len(pred_set) == 0 or len(gt_set) == 0:
        return 0.0

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def r_motion_token_lcs(pred_tokens: list[int], gt_tokens: list[int]) -> float:
    """计算 motion token 的最长公共子序列 (LCS) 分数"""
    if not pred_tokens or not gt_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gt_tokens)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    lcs_length = prev[n]
    return 2 * lcs_length / (m + n) if (m + n) > 0 else 0.0


# =====================================================================
# 3) Motion Token 解码
# =====================================================================
@torch.inference_mode()
def decode_motion_tokens_to_poses(
    motion_codes: list[int],
    vqvae=None,
) -> Optional[torch.Tensor]:
    """将 motion token IDs 解码为 pose 序列"""
    if not motion_codes:
        return None

    if vqvae is None:
        vqvae = _load_vqvae()
        if vqvae is None:
            return None

    try:
        device = _get_device()
        codes = torch.tensor(motion_codes, dtype=torch.long, device=device)
        poses = vqvae.forward_decoder(codes.unsqueeze(0))
        poses = poses.squeeze(0)
        return poses
    except Exception as e:
        print(f"[T2M Reward] decode_motion_tokens_to_poses error: {e}")
        return None


# =====================================================================
# 4) 文本编码
# =====================================================================
MAX_TEXT_LEN = 20


def encode_text_simple(
    text: str,
    w_vectorizer=None,
    max_len: int = MAX_TEXT_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """简化的文本编码"""
    if w_vectorizer is None:
        w_vectorizer = _load_word_vectorizer()

    device = _get_device()

    if w_vectorizer is None:
        word_embs = torch.zeros(1, max_len + 2, 300, device=device)
        pos_ohs = torch.zeros(1, max_len + 2, 15, device=device)
        sent_len = torch.tensor([max_len + 2], dtype=torch.long, device=device)
        return word_embs, pos_ohs, sent_len

    # try:
    # 预处理：将换行符替换为空格，然后分词
    text_clean = text.lower().strip().replace('\n', ' ').replace('\r', ' ')
    words = text_clean.split()
    # 过滤掉包含 '/' 的单词，或者将 '/' 替换为其他字符
    # 因为 WordVectorizer 使用 'word/POS' 格式，单词中不能有 '/'
    clean_words = []
    for w in words:
        # 移除单词中的 '/' 字符，避免与 word/POS 格式冲突
        clean_w = w.replace('/', '')
        if clean_w:  # 确保不是空字符串
            clean_words.append(clean_w)
    tokens = [f"{w}/OTHER" for w in clean_words]

    if len(tokens) < max_len:
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        actual_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_len + 2 - actual_len)
    else:
        tokens = tokens[:max_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        actual_len = len(tokens)

    word_embs_list = []
    pos_ohs_list = []

    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        word_embs_list.append(word_emb)
        pos_ohs_list.append(pos_oh)

    word_embs = torch.tensor(
        np.stack(word_embs_list), dtype=torch.float32, device=device
    ).unsqueeze(0)
    pos_ohs = torch.tensor(
        np.stack(pos_ohs_list), dtype=torch.float32, device=device
    ).unsqueeze(0)
    sent_len = torch.tensor([actual_len], dtype=torch.long, device=device)

    return word_embs, pos_ohs, sent_len

    # except Exception as e:
    #     print(f"[T2M Reward] encode_text_simple error: {e}")
    #     word_embs = torch.zeros(1, max_len + 2, 300, device=device)
    #     pos_ohs = torch.zeros(1, max_len + 2, 15, device=device)
    #     sent_len = torch.tensor([max_len + 2], dtype=torch.long, device=device)
    #     return word_embs, pos_ohs, sent_len


def encode_text_for_reward(
    text: str = "",
    pos_tokens: Optional[list[str]] = None,
    w_vectorizer=None,
    max_len: int = MAX_TEXT_LEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode text for T2M rewards, preferring dataset POS-tagged tokens."""
    tokens = []
    if pos_tokens:
        for token in pos_tokens:
            if token is None:
                continue
            token = str(token).strip()
            if not token:
                continue
            if "/" not in token:
                token = f"{token.replace('/', '')}/OTHER"
            tokens.append(token)

    if not tokens:
        return encode_text_simple(text, w_vectorizer=w_vectorizer, max_len=max_len)

    if w_vectorizer is None:
        w_vectorizer = _load_word_vectorizer()

    device = _get_device()

    if w_vectorizer is None:
        word_embs = torch.zeros(1, max_len + 2, 300, device=device)
        pos_ohs = torch.zeros(1, max_len + 2, 15, device=device)
        sent_len = torch.tensor([max_len + 2], dtype=torch.long, device=device)
        return word_embs, pos_ohs, sent_len

    if len(tokens) < max_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        actual_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_len + 2 - actual_len)
    else:
        tokens = tokens[:max_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        actual_len = len(tokens)

    word_embs_list = []
    pos_ohs_list = []

    for token in tokens:
        try:
            word_emb, pos_oh = w_vectorizer[token]
        except Exception:
            word = token.split("/", 1)[0].replace("/", "")
            fallback = f"{word}/OTHER" if word else "unk/OTHER"
            try:
                word_emb, pos_oh = w_vectorizer[fallback]
            except Exception:
                word_emb, pos_oh = w_vectorizer["unk/OTHER"]
        word_embs_list.append(word_emb)
        pos_ohs_list.append(pos_oh)

    global _pos_token_usage_logged
    if (
        not _pos_token_usage_logged
        and os.environ.get("T2M_LOG_POS_TOKEN_USAGE", "1") == "1"
    ):
        print(f"[T2M Reward] Using POS-tagged text tokens for reward embeddings, e.g. {tokens[:4]}")
        _pos_token_usage_logged = True

    word_embs = torch.tensor(
        np.stack(word_embs_list), dtype=torch.float32, device=device
    ).unsqueeze(0)
    pos_ohs = torch.tensor(
        np.stack(pos_ohs_list), dtype=torch.float32, device=device
    ).unsqueeze(0)
    sent_len = torch.tensor([actual_len], dtype=torch.long, device=device)

    return word_embs, pos_ohs, sent_len


# =====================================================================
# 5) Embedding 奖励函数
# =====================================================================
@torch.inference_mode()
def r_motion_embedding(
    eval_wrapper,
    pred_pose: torch.Tensor,
    pred_len: torch.Tensor,
    gt_pose: torch.Tensor,
    gt_len: torch.Tensor,
    word_embeddings: torch.Tensor,
    pos_one_hots: torch.Tensor,
    sent_len: torch.Tensor,
) -> float:
    """动作相似度奖励: cos(f_m(pred_pose), f_m(gt_pose))

    Returns:
        Reward in [-1, 1] range (raw cosine similarity)
    """
    if eval_wrapper is None:
        return 0.0

    # try:
    _, em_pred = eval_wrapper.get_co_embeddings(
        word_embeddings, pos_one_hots, sent_len, pred_pose, pred_len
    )
    _, em_gt = eval_wrapper.get_co_embeddings(
        word_embeddings, pos_one_hots, sent_len, gt_pose, gt_len
    )
    cos = F.cosine_similarity(em_pred, em_gt, dim=-1)
    return cos.item()
    # except Exception as e:
    #     print(f"[T2M Reward] r_motion_embedding error: {e}")
    #     return 0.0


@torch.inference_mode()
def r_semantic_embedding(
    eval_wrapper,
    pred_pose: torch.Tensor,
    pred_len: torch.Tensor,
    word_embeddings: torch.Tensor,
    pos_one_hots: torch.Tensor,
    sent_len: torch.Tensor,
) -> float:
    """语义相似度奖励: cos(f_m(pred_pose), f_text(text))

    Returns:
        Reward in [-1, 1] range (raw cosine similarity)
    """
    if eval_wrapper is None:
        return 0.0

    try:
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose, pred_len
        )
        cos = F.cosine_similarity(em_pred, et_pred, dim=-1)
        return cos.item()
    except Exception as e:
        print(f"[T2M Reward] r_semantic_embedding error: {e}")
        return 0.0


# =====================================================================
# 6) 物理合理性奖励
# =====================================================================
"""
HumanML3D 数据格式 (263维):
- [0:1]     - root rotation velocity (1)
- [1:3]     - root linear velocity x, z (2)
- [3:4]     - root height y (1)
- [4:67]    - local joint positions (ric) = 21 joints × 3 (63)
- [67:193]  - local joint rotations (6D continuous) = 21 joints × 6 (126)
- [193:259] - local joint velocities = 22 joints × 3 (66)
- [259:263] - foot contact (4)

注意: VQ-VAE 下采样率 down_t=2，所以解码后的帧率是 20/2 = 10 FPS
"""

# HumanML3D 关节名称 (22个关节，但 rotation 只有21个，不含 root)
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

# 关节旋转幅度上限 (弧度)
# 这里使用旋转幅度 (magnitude)，即 axis-angle 表示中的角度大小
# 因为 6D rotation 转换后我们计算的是旋转的总幅度，所以只设上限
JOINT_ROTATION_LIMITS = {
    # 髋关节 - 活动范围较大
    'left_hip': 2.5, 'right_hip': 2.5,
    # 膝关节 - 只能弯曲，范围有限
    'left_knee': 2.8, 'right_knee': 2.8,
    # 踝关节 - 活动范围较小
    'left_ankle': 1.2, 'right_ankle': 1.2,
    # 肩关节 - 活动范围最大
    'left_shoulder': 3.14, 'right_shoulder': 3.14,
    # 肘关节 - 只能弯曲
    'left_elbow': 2.8, 'right_elbow': 2.8,
    # 脊柱 - 活动范围有限
    'spine1': 1.0, 'spine2': 1.0, 'spine3': 1.0,
    # 颈部
    'neck': 1.5,
    # 脚 - 活动范围小
    'left_foot': 1.0, 'right_foot': 1.0,
    # 锁骨 - 活动范围很小
    'left_collar': 0.8, 'right_collar': 0.8,
    # 头部
    'head': 1.2,
    # 手腕
    'left_wrist': 1.5, 'right_wrist': 1.5,
}


def _rotation_6d_to_angle(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    将 6D continuous rotation 转换为旋转角度 (弧度)

    6D rotation 由两个正交向量组成，表示旋转矩阵的前两列
    我们计算相对于单位矩阵的旋转角度

    Args:
        rot_6d: [..., 6] - 6D rotation representation

    Returns:
        angles: [...] - 旋转角度 (弧度)，范围 [0, π]
    """
    # 6D rotation: [a1, a2, a3, b1, b2, b3]
    # a = first column, b = second column (before orthogonalization)
    a = rot_6d[..., :3]
    b = rot_6d[..., 3:6]

    # Gram-Schmidt 正交化得到正交的 x, y 轴
    x = F.normalize(a, dim=-1)
    y = b - (x * b).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1)

    # 第三列 z = x × y
    z = torch.cross(x, y, dim=-1)

    # 旋转矩阵的迹 trace(R) = 1 + 2*cos(θ)
    # 对于正交化后的旋转矩阵，迹 = x[0] + y[1] + z[2]
    # 但由于我们只有前两列，这里用近似方法：
    # 计算与单位矩阵的 Frobenius 距离作为旋转幅度的度量

    # 构建完整的旋转矩阵 [B, T, J, 3, 3]
    R = torch.stack([x, y, z], dim=-1)  # [..., 3, 3]

    # 计算旋转角度: trace(R) = 1 + 2*cos(θ)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    # cos(θ) = (trace - 1) / 2, 限制在 [-1, 1] 范围
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angles = torch.acos(cos_angle)  # [0, π]

    return angles


@torch.inference_mode()
def compute_joint_angle_violation(poses: torch.Tensor) -> torch.Tensor:
    """
    计算关节旋转幅度违规惩罚 L_joint

    检查每个关节的旋转幅度是否超过生理限制

    Args:
        poses: [T, D] 或 [B, T, D] - HumanML3D 格式的 pose 数据

    Returns:
        L_joint: 标量 - 平均违规程度
    """
    if poses is None or poses.numel() == 0:
        return torch.tensor(0.0)

    if poses.dim() == 2:
        poses = poses.unsqueeze(0)

    B, T, D = poses.shape

    # 提取 6D rotation 数据
    rot_start, rot_end = 67, 67 + 126  # 21 joints × 6

    if D < rot_end:
        # 数据维度不足，无法计算
        return torch.tensor(0.0, device=poses.device)

    # [B, T, 126] -> [B, T, 21, 6]
    rot_6d = poses[:, :, rot_start:rot_end].view(B, T, 21, 6)

    # 将 6D rotation 转换为旋转角度 [B, T, 21]
    angles = _rotation_6d_to_angle(rot_6d)

    # 计算违规：超过上限的部分
    violation = torch.zeros_like(angles)
    for joint_idx, joint_name in enumerate(JOINT_NAMES[1:]):  # 跳过 pelvis (root)
        if joint_name in JOINT_ROTATION_LIMITS:
            max_angle = JOINT_ROTATION_LIMITS[joint_name]
            joint_angles = angles[:, :, joint_idx]
            # 只惩罚超过上限的部分
            violation[:, :, joint_idx] = F.relu(joint_angles - max_angle)

    return violation.mean()


@torch.inference_mode()
def compute_velocity_smoothness(poses: torch.Tensor, fps: float = 10.0) -> torch.Tensor:
    """
    计算速度平滑度惩罚 L_vel

    惩罚过大的加速度（突然的速度变化）

    Args:
        poses: [T, D] 或 [B, T, D] - HumanML3D 格式的 pose 数据
        fps: 帧率 (VQ-VAE 下采样后为 10 FPS)

    Returns:
        L_vel: 标量 - 平均加速度违规程度
    """
    if poses is None or poses.numel() == 0:
        return torch.tensor(0.0)

    if poses.dim() == 2:
        poses = poses.unsqueeze(0)

    B, T, D = poses.shape
    if T < 3:
        return torch.tensor(0.0, device=poses.device)

    dt = 1.0 / fps

    # 优先使用数据中的速度信息 (如果有)
    vel_start, vel_end = 193, 193 + 66  # 22 joints × 3
    if D >= vel_end:
        # 使用数据中的速度
        velocity = poses[:, :, vel_start:vel_end]
    else:
        # 从位置计算速度
        ric_start, ric_end = 4, 4 + 63  # 21 joints × 3
        if D < ric_end:
            return torch.tensor(0.0, device=poses.device)
        positions = poses[:, :, ric_start:ric_end]
        velocity = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
        velocity = F.pad(velocity, (0, 0, 0, 1), mode='replicate')

    # 计算加速度
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt
    acc_norm = torch.norm(acceleration, dim=-1)  # [B, T-1]

    # 加速度阈值 (单位: m/s² 或 rad/s²)
    # 人体正常运动的加速度通常在 5-20 m/s² 范围内
    # 这里使用归一化数据，阈值需要调整
    acc_threshold = 10.0

    # 超过阈值的部分给予惩罚
    excess_acc = F.relu(acc_norm - acc_threshold)
    L_vel = excess_acc.mean()

    return L_vel


@torch.inference_mode()
def compute_foot_skating(poses: torch.Tensor, fps: float = 10.0) -> torch.Tensor:
    """
    计算脚部滑动惩罚 (foot skating)

    当脚部接触地面时，脚的速度应该接近零

    Args:
        poses: [T, D] 或 [B, T, D] - HumanML3D 格式的 pose 数据
        fps: 帧率

    Returns:
        L_skating: 标量 - 脚部滑动惩罚
    """
    if poses is None or poses.numel() == 0:
        return torch.tensor(0.0)

    if poses.dim() == 2:
        poses = poses.unsqueeze(0)

    B, T, D = poses.shape

    # 检查数据是否包含 foot contact 信息
    if D < 263:
        return torch.tensor(0.0, device=poses.device)

    # Foot contact: [259:263] = [left_heel, left_toe, right_heel, right_toe]
    foot_contact = poses[:, :, 259:263]  # [B, T, 4]

    # 脚的位置索引 (在 ric 中)
    # left_foot: joint 10, right_foot: joint 11
    # ric_start = 4, 每个 joint 3维
    left_foot_idx = 4 + 10 * 3  # 34:37
    right_foot_idx = 4 + 11 * 3  # 37:40

    if D < right_foot_idx + 3:
        return torch.tensor(0.0, device=poses.device)

    left_foot_pos = poses[:, :, left_foot_idx:left_foot_idx + 3]
    right_foot_pos = poses[:, :, right_foot_idx:right_foot_idx + 3]

    dt = 1.0 / fps

    # 计算脚的速度
    left_vel = (left_foot_pos[:, 1:, :] - left_foot_pos[:, :-1, :]) / dt
    right_vel = (right_foot_pos[:, 1:, :] - right_foot_pos[:, :-1, :]) / dt

    left_vel_norm = torch.norm(left_vel, dim=-1)  # [B, T-1]
    right_vel_norm = torch.norm(right_vel, dim=-1)  # [B, T-1]

    # Foot contact 也截断到 T-1
    left_contact = (foot_contact[:, :-1, 0] + foot_contact[:, :-1, 1]) / 2.0  # heel + toe 平均
    right_contact = (foot_contact[:, :-1, 2] + foot_contact[:, :-1, 3]) / 2.0

    # 当脚接触地面时 (contact > 0.5)，速度应该为 0
    # 惩罚 = contact * velocity
    left_skating = left_contact * left_vel_norm
    right_skating = right_contact * right_vel_norm

    L_skating = (left_skating.mean() + right_skating.mean()) / 2.0

    return L_skating


def r_physical_plausibility(
    motion_tokens: list[int],
    lambda_joint: float = 1.0,
    lambda_vel: float = 1.0,
    lambda_skating: float = 0.5,
) -> tuple[float, dict]:
    """
    从 motion tokens 计算物理合理性奖励

    Args:
        motion_tokens: motion token ID 列表
        lambda_joint: 关节角度违规惩罚系数
        lambda_vel: 速度平滑度惩罚系数
        lambda_skating: 脚部滑动惩罚系数

    Returns:
        r_phys: [0, 1] 范围的物理合理性奖励
        details: 各项惩罚的详细值
    """
    if not motion_tokens:
        return 0.5, {"L_joint": 0.0, "L_vel": 0.0, "L_skating": 0.0}

    poses = decode_motion_tokens_to_poses(motion_tokens)
    if poses is None:
        return 0.5, {"L_joint": 0.0, "L_vel": 0.0, "L_skating": 0.0}

    # 计算各项惩罚
    L_joint = compute_joint_angle_violation(poses)
    L_vel = compute_velocity_smoothness(poses, fps=10.0)  # VQ-VAE 下采样后 10 FPS
    L_skating = compute_foot_skating(poses, fps=10.0)

    # 总惩罚
    total_penalty = (
        lambda_joint * L_joint.item() +
        lambda_vel * L_vel.item() +
        lambda_skating * L_skating.item()
    )

    # 使用 sigmoid 将惩罚映射到 [0, 1] 奖励
    # penalty 越大，奖励越低
    r_phys = torch.sigmoid(torch.tensor(-total_penalty)).item()

    return r_phys, {
        "L_joint": L_joint.item(),
        "L_vel": L_vel.item(),
        "L_skating": L_skating.item()
    }


# =====================================================================
# 7) 辅助函数：Pad/Truncate poses 到固定长度
# =====================================================================
MAX_MOTION_LENGTH = 196  # 与 Motion-R1 evaluation.py 保持一致


def pad_or_truncate_poses(
    poses: torch.Tensor,
    max_len: int = MAX_MOTION_LENGTH
) -> tuple[torch.Tensor, int]:
    """
    将 poses 放入固定大小容器，短的补0，长的截断

    Args:
        poses: [T, D] - 原始 poses
        max_len: 最大长度（默认196）

    Returns:
        padded_poses: [max_len, D] - 填充/截断后的 poses
        actual_len: int - 实际有效长度

    Following Motion-R1 evaluation.py line 92-93
    """
    T, D = poses.shape
    device = poses.device

    # 实际长度 = min(原始长度, max_len)
    actual_len = min(T, max_len)

    # 创建零矩阵容器
    padded_poses = torch.zeros(max_len, D, device=device, dtype=poses.dtype)

    # 填充有效部分（如果 T > max_len 会自动截断）
    padded_poses[:actual_len] = poses[:actual_len]

    return padded_poses, actual_len


# =====================================================================
# 8) 组合 Embedding 奖励计算
# =====================================================================
def _compute_embedding_reward(
    pred_tokens: list[int],
    gt_tokens: list[int],
    caption: str = "",
    pos_tokens: Optional[list[str]] = None,
) -> float:
    """使用 embedding 计算 motion 相似度奖励: cos(f_m(pred), f_m(gt))

    Following Motion-R1 evaluation.py:
    - Pad/truncate poses to固定长度 (196 frames)
    - 短的补0，长的截断
    """
    if not pred_tokens or not gt_tokens:
        return 0.0

    # try:
    eval_wrapper = _load_evaluator_wrapper()
    vqvae = _load_vqvae()

    if eval_wrapper is None or vqvae is None:
        return r_motion_token_f1(pred_tokens, gt_tokens)

    # 解码预测的 poses
    pred_pose_raw = decode_motion_tokens_to_poses(pred_tokens, vqvae)
    # if pred_pose_raw is None:
    #     return r_motion_token_f1(pred_tokens, gt_tokens)

    # 解码 GT poses
    gt_pose_raw = decode_motion_tokens_to_poses(gt_tokens, vqvae)
    # if gt_pose_raw is None:
    #     return r_motion_token_f1(pred_tokens, gt_tokens)

    # Pad/Truncate 到固定长度 (Following Motion-R1 evaluation.py line 92-93)
    pred_pose, pred_len = pad_or_truncate_poses(pred_pose_raw, MAX_MOTION_LENGTH)
    gt_pose, gt_len = pad_or_truncate_poses(gt_pose_raw, MAX_MOTION_LENGTH)

    # 添加 batch 维度
    pred_pose = pred_pose.unsqueeze(0)  # [1, 196, D]
    gt_pose = gt_pose.unsqueeze(0)      # [1, 196, D]
    pred_len_tensor = torch.tensor([pred_len], device=pred_pose.device)
    gt_len_tensor = torch.tensor([gt_len], device=gt_pose.device)

    # 需要 caption 用于 text embedding
    # if not caption:
    #     return r_motion_token_f1(pred_tokens, gt_tokens)

    word_embs, pos_ohs, sent_len = encode_text_for_reward(caption, pos_tokens)

    # 计算 motion embedding 相似度: cos(f_m(pred), f_m(gt))
    r_mot = r_motion_embedding(
        eval_wrapper, pred_pose, pred_len_tensor, gt_pose, gt_len_tensor,
        word_embs, pos_ohs, sent_len
    )
    return r_mot

    # except Exception as e:
    #     print(f"[T2M Reward] _compute_embedding_reward error: {e}")
    #     return r_motion_token_f1(pred_tokens, gt_tokens)


def _compute_semantic_reward(
    pred_tokens: list[int],
    caption: str = "",
    pos_tokens: Optional[list[str]] = None,
) -> float:
    """计算语义相似度奖励

    Following Motion-R1 evaluation.py:
    - Pad/truncate poses to固定长度 (196 frames)
    - 短的补0，长的截断
    """
    if not pred_tokens or not caption:
        return 0.0

    try:
        eval_wrapper = _load_evaluator_wrapper()
        pred_pose_raw = decode_motion_tokens_to_poses(pred_tokens)

        if pred_pose_raw is None or eval_wrapper is None:
            return 0.0

        # Pad/Truncate 到固定长度 (Following Motion-R1 evaluation.py line 92-93)
        pred_pose, pred_len = pad_or_truncate_poses(pred_pose_raw, MAX_MOTION_LENGTH)

        # 添加 batch 维度
        pred_pose = pred_pose.unsqueeze(0)  # [1, 196, D]
        pred_len_tensor = torch.tensor([pred_len], device=pred_pose.device)
        word_embs, pos_ohs, sent_len = encode_text_for_reward(caption, pos_tokens)

        return r_semantic_embedding(
            eval_wrapper, pred_pose, pred_len_tensor, word_embs, pos_ohs, sent_len
        )
    except Exception:
        return 0.0


# =====================================================================
# 8) 批量 Embedding 奖励计算 (高效版本)
# =====================================================================
@torch.inference_mode()
def _compute_embedding_reward_batch(
    pred_tokens_list: list[list[int]],
    gt_tokens_list: list[list[int]],
    captions: list[str],
    pos_tokens_list: Optional[list[Optional[list[str]]]] = None,
) -> list[float]:
    """批量计算 motion embedding 相似度奖励

    Args:
        pred_tokens_list: 预测的 motion tokens 列表
        gt_tokens_list: ground truth motion tokens 列表
        captions: 文本描述列表

    Returns:
        rewards: 每个样本的奖励值列表
    """
    batch_size = len(pred_tokens_list)
    if batch_size == 0:
        return []
    if pos_tokens_list is None:
        pos_tokens_list = [None] * batch_size

    # 加载模型
    eval_wrapper = _load_evaluator_wrapper()
    vqvae = _load_vqvae()

    if eval_wrapper is None or vqvae is None:
        # 退化到 F1 计算
        return [r_motion_token_f1(pred, gt) for pred, gt in zip(pred_tokens_list, gt_tokens_list)]

    device = _get_device()
    rewards = []

    # 收集有效样本的索引和数据
    valid_indices = []
    pred_poses_list = []
    pred_lens_list = []
    gt_poses_list = []
    gt_lens_list = []
    valid_captions = []
    valid_pos_tokens = []

    for i, (pred_tokens, gt_tokens, caption) in enumerate(zip(pred_tokens_list, gt_tokens_list, captions)):
        if not pred_tokens or not gt_tokens:
            continue
        pos_tokens = pos_tokens_list[i] if i < len(pos_tokens_list) else None

        # 解码 poses
        pred_pose_raw = decode_motion_tokens_to_poses(pred_tokens, vqvae)
        gt_pose_raw = decode_motion_tokens_to_poses(gt_tokens, vqvae)

        if pred_pose_raw is None or gt_pose_raw is None:
            continue

        # Pad/Truncate
        pred_pose, pred_len = pad_or_truncate_poses(pred_pose_raw, MAX_MOTION_LENGTH)
        gt_pose, gt_len = pad_or_truncate_poses(gt_pose_raw, MAX_MOTION_LENGTH)

        valid_indices.append(i)
        pred_poses_list.append(pred_pose)
        pred_lens_list.append(pred_len)
        gt_poses_list.append(gt_pose)
        gt_lens_list.append(gt_len)
        valid_captions.append(caption)
        valid_pos_tokens.append(pos_tokens)

    # 初始化所有奖励为 0
    rewards = [0.0] * batch_size

    if not valid_indices:
        return rewards

    # 批量堆叠 tensors
    pred_poses_batch = torch.stack(pred_poses_list, dim=0)  # [N, 196, D]
    gt_poses_batch = torch.stack(gt_poses_list, dim=0)      # [N, 196, D]
    pred_lens_batch = torch.tensor(pred_lens_list, device=device, dtype=torch.long)
    gt_lens_batch = torch.tensor(gt_lens_list, device=device, dtype=torch.long)

    # 批量编码 captions
    word_embs_list = []
    pos_ohs_list = []
    sent_lens_list = []

    for caption, pos_tokens in zip(valid_captions, valid_pos_tokens):
        word_embs, pos_ohs, sent_len = encode_text_for_reward(caption, pos_tokens)
        word_embs_list.append(word_embs.squeeze(0))
        pos_ohs_list.append(pos_ohs.squeeze(0))
        sent_lens_list.append(sent_len.item())

    word_embs_batch = torch.stack(word_embs_list, dim=0)  # [N, L, 300]
    pos_ohs_batch = torch.stack(pos_ohs_list, dim=0)      # [N, L, 15]
    sent_lens_batch = torch.tensor(sent_lens_list, device=device, dtype=torch.long)

    # 批量获取 embeddings
    _, em_pred = eval_wrapper.get_co_embeddings(
        word_embs_batch, pos_ohs_batch, sent_lens_batch, pred_poses_batch, pred_lens_batch
    )
    _, em_gt = eval_wrapper.get_co_embeddings(
        word_embs_batch, pos_ohs_batch, sent_lens_batch, gt_poses_batch, gt_lens_batch
    )

    # 批量计算 cosine similarity
    cos_sims = F.cosine_similarity(em_pred, em_gt, dim=-1)  # [N]

    # 填充结果
    for idx, valid_idx in enumerate(valid_indices):
        rewards[valid_idx] = cos_sims[idx].item()

    return rewards


@torch.inference_mode()
def _compute_semantic_reward_batch(
    pred_tokens_list: list[list[int]],
    captions: list[str],
    pos_tokens_list: Optional[list[Optional[list[str]]]] = None,
) -> list[float]:
    """批量计算语义相似度奖励

    Args:
        pred_tokens_list: 预测的 motion tokens 列表
        captions: 文本描述列表

    Returns:
        rewards: 每个样本的奖励值列表
    """
    batch_size = len(pred_tokens_list)
    if batch_size == 0:
        return []
    if pos_tokens_list is None:
        pos_tokens_list = [None] * batch_size

    # 加载模型
    eval_wrapper = _load_evaluator_wrapper()
    vqvae = _load_vqvae()

    if eval_wrapper is None or vqvae is None:
        return [0.0] * batch_size

    device = _get_device()

    # 收集有效样本的索引和数据
    valid_indices = []
    pred_poses_list = []
    pred_lens_list = []
    valid_captions = []
    valid_pos_tokens = []

    for i, (pred_tokens, caption) in enumerate(zip(pred_tokens_list, captions)):
        if not pred_tokens or not caption:
            continue
        pos_tokens = pos_tokens_list[i] if i < len(pos_tokens_list) else None

        # 解码 poses
        pred_pose_raw = decode_motion_tokens_to_poses(pred_tokens, vqvae)

        if pred_pose_raw is None:
            continue

        # Pad/Truncate
        pred_pose, pred_len = pad_or_truncate_poses(pred_pose_raw, MAX_MOTION_LENGTH)

        valid_indices.append(i)
        pred_poses_list.append(pred_pose)
        pred_lens_list.append(pred_len)
        valid_captions.append(caption)
        valid_pos_tokens.append(pos_tokens)

    # 初始化所有奖励为 0
    rewards = [0.0] * batch_size

    if not valid_indices:
        return rewards

    # 批量堆叠 tensors
    pred_poses_batch = torch.stack(pred_poses_list, dim=0)  # [N, 196, D]
    pred_lens_batch = torch.tensor(pred_lens_list, device=device, dtype=torch.long)

    # 批量编码 captions
    word_embs_list = []
    pos_ohs_list = []
    sent_lens_list = []

    for caption, pos_tokens in zip(valid_captions, valid_pos_tokens):
        word_embs, pos_ohs, sent_len = encode_text_for_reward(caption, pos_tokens)
        word_embs_list.append(word_embs.squeeze(0))
        pos_ohs_list.append(pos_ohs.squeeze(0))
        sent_lens_list.append(sent_len.item())

    word_embs_batch = torch.stack(word_embs_list, dim=0)  # [N, L, 300]
    pos_ohs_batch = torch.stack(pos_ohs_list, dim=0)      # [N, L, 15]
    sent_lens_batch = torch.tensor(sent_lens_list, device=device, dtype=torch.long)

    # 批量获取 embeddings (text_embedding, motion_embedding)
    et_pred, em_pred = eval_wrapper.get_co_embeddings(
        word_embs_batch, pos_ohs_batch, sent_lens_batch, pred_poses_batch, pred_lens_batch
    )

    # 批量计算 cosine similarity (motion vs text)
    cos_sims = F.cosine_similarity(em_pred, et_pred, dim=-1)  # [N]

    # 填充结果
    for idx, valid_idx in enumerate(valid_indices):
        rewards[valid_idx] = cos_sims[idx].item()

    return rewards


# =====================================================================
# 9) 预定义的奖励组合
# =====================================================================
AVAILABLE_REWARDS = {
    "format": "格式奖励 - 检查 <think>...</think><Motion>...</Motion>",
    "format_soft": "软格式奖励 - 部分匹配也给分",
    "motion_f1": "动作匹配 (F1) - 基于 token 集合的 F1 分数",
    "motion_lcs": "动作匹配 (LCS) - 基于最长公共子序列",
    "motion_embedding": "动作相似度 (Embedding) - cos(f_m(pred), f_m(gt))",
    "semantic": "语义相似度 - cos(f_m(pred), f_text(text))",
    "phys": "物理合理性 - 关节角度 + 速度平滑 + 脚部滑动",
    "phys_joint": "关节角度违规惩罚 (L_joint) - 基于 6D rotation",
    "phys_vel": "速度平滑度惩罚 (L_vel) - 加速度阈值",
    "phys_skating": "脚部滑动惩罚 (L_skating) - foot contact 约束",
}

REWARD_PRESETS = {
    "basic": {
        "description": "基础版本: 格式 + Token F1",
        "rewards": ["format_soft", "motion_f1"],
        "weights": {"format_soft": 0.1, "motion_f1": 0.9},
    },
    "with_physics": {
        "description": "含物理奖励: 格式 + Token F1 + 物理合理性",
        "rewards": ["format_soft", "motion_f1", "phys"],
        "weights": {"format_soft": 0.1, "motion_f1": 0.7, "phys": 0.2},
    },
    "strict": {
        "description": "严格版本: 格式 + LCS + 物理合理性",
        "rewards": ["format", "motion_lcs", "phys"],
        "weights": {"format": 0.1, "motion_lcs": 0.7, "phys": 0.2},
    },
    "semantic": {
        "description": "语义版本: 格式 + Embedding + 语义",
        "rewards": ["format", "motion_embedding", "semantic"],
        "weights": {"format": 0.1, "motion_embedding": 0.45, "semantic": 0.45},
    },
    "full": {
        "description": "完整版本: 所有奖励",
        "rewards": ["format_soft", "motion_f1", "semantic", "phys"],
        "weights": {"format_soft": 0.1, "motion_f1": 0.4, "semantic": 0.3, "phys": 0.2},
    },
}


# =====================================================================
# 9) TRL 格式的 Reward 函数
# =====================================================================
def t2m_format_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    **kwargs
) -> list[float]:
    """严格格式奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        rewards.append(r_format(content, use_motion_tag=True))
    return rewards


def t2m_format_soft_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    **kwargs
) -> list[float]:
    """软格式奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        rewards.append(r_format_soft(content))
    return rewards


def t2m_motion_f1_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    **kwargs
) -> list[float]:
    """Motion Token F1 奖励函数 (TRL 格式)"""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        gt_tokens = parse_motion_token_ids(gt)
        rewards.append(r_motion_token_f1(pred_tokens, gt_tokens))
    return rewards


def t2m_motion_lcs_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    **kwargs
) -> list[float]:
    """Motion Token LCS 奖励函数 (TRL 格式)"""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        gt_tokens = parse_motion_token_ids(gt)
        rewards.append(r_motion_token_lcs(pred_tokens, gt_tokens))
    return rewards


def t2m_motion_embedding_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    caption: list[str] = None,
    pos_tokens: Optional[list[Optional[list[str]]]] = None,
    **kwargs
) -> list[float]:
    """Motion Embedding 奖励函数 (TRL 格式)"""
    rewards = []
    captions = caption if caption else [""] * len(completions)
    pos_tokens_list = pos_tokens if pos_tokens is not None else [None] * len(completions)

    for i, (completion, gt, cap) in enumerate(zip(completions, ground_truth, captions)):
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        gt_tokens = parse_motion_token_ids(gt)
        cur_pos_tokens = pos_tokens_list[i] if i < len(pos_tokens_list) else None
        rewards.append(_compute_embedding_reward(pred_tokens, gt_tokens, cap, cur_pos_tokens))
    return rewards


def t2m_semantic_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    caption: list[str] = None,
    pos_tokens: Optional[list[Optional[list[str]]]] = None,
    **kwargs
) -> list[float]:
    """语义相似度奖励函数 (TRL 格式)"""
    rewards = []
    captions = caption if caption else [""] * len(completions)
    pos_tokens_list = pos_tokens if pos_tokens is not None else [None] * len(completions)

    for i, (completion, cap) in enumerate(zip(completions, captions)):
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        cur_pos_tokens = pos_tokens_list[i] if i < len(pos_tokens_list) else None
        rewards.append(_compute_semantic_reward(pred_tokens, cap, cur_pos_tokens))
    return rewards


def t2m_phys_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    lambda_joint: float = 1.0,
    lambda_vel: float = 1.0,
    lambda_skating: float = 0.5,
    **kwargs
) -> list[float]:
    """物理合理性奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        r_phys, _ = r_physical_plausibility(pred_tokens, lambda_joint, lambda_vel, lambda_skating)
        rewards.append(r_phys)
    return rewards


def t2m_phys_joint_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    lambda_joint: float = 1.0,
    **kwargs
) -> list[float]:
    """关节角度违规惩罚奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []

        if pred_tokens:
            poses = decode_motion_tokens_to_poses(pred_tokens)
            if poses is not None:
                L_joint = compute_joint_angle_violation(poses)
                reward = torch.sigmoid(-lambda_joint * L_joint).item()
            else:
                reward = 0.5
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards


def t2m_phys_vel_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    lambda_vel: float = 1.0,
    **kwargs
) -> list[float]:
    """速度平滑度惩罚奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []

        if pred_tokens:
            poses = decode_motion_tokens_to_poses(pred_tokens)
            if poses is not None:
                L_vel = compute_velocity_smoothness(poses, fps=10.0)  # VQ-VAE 下采样后 10 FPS
                reward = torch.sigmoid(-lambda_vel * L_vel).item()
            else:
                reward = 0.5
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards


def t2m_phys_skating_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    lambda_skating: float = 1.0,
    **kwargs
) -> list[float]:
    """脚部滑动惩罚奖励函数 (TRL 格式)"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []

        if pred_tokens:
            poses = decode_motion_tokens_to_poses(pred_tokens)
            if poses is not None:
                L_skating = compute_foot_skating(poses, fps=10.0)
                reward = torch.sigmoid(-lambda_skating * L_skating).item()
            else:
                reward = 0.5
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards


def t2m_combined_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    caption: list[str] = None,
    # 权重配置
    w_format: float = 0.1,
    w_motion: float = 0.9,
    use_soft_format: bool = True,
    use_lcs: bool = False,
    **kwargs
) -> list[float]:
    """组合奖励函数 (TRL 格式)"""
    rewards = []
    captions = caption if caption else [""] * len(completions)

    for completion, gt, cap in zip(completions, ground_truth, captions):
        content = completion[0]["content"] if completion else ""

        # 格式奖励
        r_fmt = r_format_soft(content) if use_soft_format else r_format(content)

        # Motion 匹配奖励
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        gt_tokens = parse_motion_token_ids(gt)

        if use_lcs:
            r_mot = r_motion_token_lcs(pred_tokens, gt_tokens)
        else:
            r_mot = r_motion_token_f1(pred_tokens, gt_tokens)

        total = w_format * r_fmt + w_motion * r_mot
        rewards.append(min(max(total, 0.0), 1.0))

    return rewards


def t2m_flexible_reward(
    completions: list[list[dict[str, str]]],
    ground_truth: list[str],
    caption: list[str] = None,
    extra_info: list[dict] = None,  # 新增：接收 extra_info (包含 task_type)
    # 配置
    preset: str = None,
    rewards_list: list[str] = None,
    weights: dict[str, float] = None,
    lambda_joint: float = 1.0,
    lambda_vel: float = 1.0,
    normalize_weights: bool = True,
    **kwargs
) -> list[float]:
    """
    灵活的奖励函数 - 支持自定义奖励组合 (TRL 格式)

    Args:
        completions: 模型生成的 completions
        ground_truth: 标准答案
        caption: 文本描述
        extra_info: 额外信息，包含 task_type 字段 ("t2m" or "m2t")
        preset: 预设名称 ("basic", "with_physics", "strict", "semantic", "full")
        rewards_list: 自定义奖励列表
        weights: 自定义权重字典
        lambda_joint: 关节角度违规惩罚系数
        lambda_vel: 速度突变惩罚系数
        normalize_weights: 是否归一化权重

    Returns:
        list[float]: 每个 completion 的 reward
    """
    global _reward_call_count

    # 确定使用的奖励和权重
    if preset is not None:
        if preset not in REWARD_PRESETS:
            raise ValueError(f"Unknown preset: {preset}, available: {list(REWARD_PRESETS.keys())}")
        config = REWARD_PRESETS[preset]
        if rewards_list is None:
            rewards_list = config["rewards"]
        if weights is None:
            weights = config["weights"]
    elif rewards_list is None:
        rewards_list = ["format_soft", "motion_f1"]
        if weights is None:
            weights = {"format_soft": 0.1, "motion_f1": 0.9}

    if weights is None:
        weights = {r: 1.0 / len(rewards_list) for r in rewards_list}

    if normalize_weights:
        total_weight = sum(weights.get(r, 0) for r in rewards_list)
        if total_weight > 0:
            weights = {r: weights.get(r, 0) / total_weight for r in rewards_list}

    captions = caption if caption else [""] * len(completions)
    pos_tokens_list = kwargs.get("pos_tokens")
    if pos_tokens_list is None:
        pos_tokens_list = [None] * len(completions)

    # 提取 task_type (如果提供了 extra_info)
    task_types = []
    if extra_info is not None:
        task_types = [info.get("task_type", "unknown") for info in extra_info]
    else:
        task_types = ["unknown"] * len(completions)

    final_rewards = []

    # 打印配置信息 (只在第一次调用时)
    if VERBOSE_REWARD and _reward_call_count == 0:
        print("\n" + "=" * 70)
        print("T2M Reward Configuration")
        print("=" * 70)
        print(f"Preset: {preset}")
        print(f"Rewards: {rewards_list}")
        print(f"Weights: {weights}")
        print(f"Batch size: {len(completions)}")

        # 打印任务类型分布
        if extra_info is not None:
            from collections import Counter
            task_dist = Counter(task_types)
            print(f"Task distribution: {dict(task_dist)}")

        print("=" * 70 + "\n")

    # ========== 预处理：收集所有样本的 tokens ==========
    all_pred_tokens = []
    all_gt_tokens = []
    all_contents = []

    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"] if completion else ""
        motion_content = extract_motion_content(content)
        pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
        gt_tokens = parse_motion_token_ids(gt)

        all_pred_tokens.append(pred_tokens)
        all_gt_tokens.append(gt_tokens)
        all_contents.append(content)

    # ========== 批量计算 embedding 奖励 (如果需要) ==========
    # 设置为 True 使用批量计算（更高效），False 使用逐样本计算
    USE_BATCH_EMBEDDING = os.environ.get("T2M_USE_BATCH_EMBEDDING", "1") == "1"

    batch_embedding_rewards = None
    batch_semantic_rewards = None

    if USE_BATCH_EMBEDDING:
        if "motion_embedding" in rewards_list:
            batch_embedding_rewards = _compute_embedding_reward_batch(
                all_pred_tokens, all_gt_tokens, captions, pos_tokens_list
            )

        if "semantic" in rewards_list:
            batch_semantic_rewards = _compute_semantic_reward_batch(
                all_pred_tokens, captions, pos_tokens_list
            )

    # ========== 主循环：计算每个样本的奖励 ==========
    for idx, (pred_tokens, gt_tokens, content, cap, task_type) in enumerate(
        zip(all_pred_tokens, all_gt_tokens, all_contents, captions, task_types)
    ):
        reward_values = {}

        for reward_name in rewards_list:
            if reward_name == "format":
                reward_values[reward_name] = r_format(content)
            elif reward_name == "format_soft":
                reward_values[reward_name] = r_format_soft(content)
            elif reward_name == "motion_f1":
                reward_values[reward_name] = r_motion_token_f1(pred_tokens, gt_tokens)
            elif reward_name == "motion_lcs":
                reward_values[reward_name] = r_motion_token_lcs(pred_tokens, gt_tokens)
            elif reward_name == "motion_embedding":
                if USE_BATCH_EMBEDDING:
                    reward_values[reward_name] = batch_embedding_rewards[idx]
                else:
                    # 逐样本计算
                    cur_pos_tokens = pos_tokens_list[idx] if idx < len(pos_tokens_list) else None
                    reward_values[reward_name] = _compute_embedding_reward(
                        pred_tokens, gt_tokens, cap, cur_pos_tokens
                    )
            elif reward_name == "semantic":
                if USE_BATCH_EMBEDDING:
                    reward_values[reward_name] = batch_semantic_rewards[idx]
                else:
                    # 逐样本计算
                    cur_pos_tokens = pos_tokens_list[idx] if idx < len(pos_tokens_list) else None
                    reward_values[reward_name] = _compute_semantic_reward(
                        pred_tokens, cap, cur_pos_tokens
                    )
            elif reward_name == "phys":
                r_phys, _ = r_physical_plausibility(pred_tokens, lambda_joint, lambda_vel, lambda_skating=0.5)
                reward_values[reward_name] = r_phys
            elif reward_name == "phys_joint":
                if pred_tokens:
                    poses = decode_motion_tokens_to_poses(pred_tokens)
                    if poses is not None:
                        L_joint = compute_joint_angle_violation(poses)
                        reward_values[reward_name] = torch.sigmoid(-lambda_joint * L_joint).item()
                    else:
                        reward_values[reward_name] = 0.5
                else:
                    reward_values[reward_name] = 0.5
            elif reward_name == "phys_vel":
                if pred_tokens:
                    poses = decode_motion_tokens_to_poses(pred_tokens)
                    if poses is not None:
                        L_vel = compute_velocity_smoothness(poses, fps=10.0)
                        reward_values[reward_name] = torch.sigmoid(-lambda_vel * L_vel).item()
                    else:
                        reward_values[reward_name] = 0.5
                else:
                    reward_values[reward_name] = 0.5
            elif reward_name == "phys_skating":
                if pred_tokens:
                    poses = decode_motion_tokens_to_poses(pred_tokens)
                    if poses is not None:
                        L_skating = compute_foot_skating(poses, fps=10.0)
                        reward_values[reward_name] = torch.sigmoid(-L_skating).item()
                    else:
                        reward_values[reward_name] = 0.5
                else:
                    reward_values[reward_name] = 0.5
            else:
                raise ValueError(f"Unknown reward type: {reward_name}")

        total = sum(reward_values.get(r, 0) * weights.get(r, 0) for r in rewards_list)
        final_reward = min(max(total, 0.0), 1.0)
        final_rewards.append(final_reward)

        # 收集统计信息（当包含 semantic 相关奖励时）
        should_collect_stats = "motion_embedding" in rewards_list or "semantic" in rewards_list
        if should_collect_stats:
            _reward_stats["format"].append(reward_values.get("format", 0))
            _reward_stats["motion_embedding"].append(reward_values.get("motion_embedding", 0))
            _reward_stats["semantic"].append(reward_values.get("semantic", 0))
            _reward_stats["total"].append(final_reward)

        # 打印单个样本的详细 reward 信息（需要同时开启 PRINT_SAMPLE_DETAILS）
        if PRINT_SAMPLE_DETAILS and VERBOSE_REWARD and (_reward_call_count % REWARD_LOG_INTERVAL == 0):
            print(f"\n{'='*70}")
            print(f"Reward Detail - Sample #{_reward_call_count} (Batch idx={idx}, Task={task_type})")
            print(f"{'='*70}")
            print(f"Caption: {cap[:80]}..." if len(cap) > 80 else f"Caption: {cap}")
            print(f"Predicted tokens: {len(pred_tokens)} tokens")
            print(f"Ground truth tokens: {len(gt_tokens)} tokens")
            print(f"\nIndividual Rewards:")
            for r_name in rewards_list:
                r_val = reward_values.get(r_name, 0)
                w_val = weights.get(r_name, 0)
                weighted = r_val * w_val
                print(f"  {r_name:20s}: {r_val:.4f} (weight={w_val:.3f}, weighted={weighted:.4f})")
            print(f"\nFinal Reward: {final_reward:.4f}")
            print(f"{'='*70}\n")

        _reward_call_count += 1

    # 每个 batch 结束后，检查是否需要打印统计信息
    _reward_stats["step_count"] += 1
    should_print_stats = "motion_embedding" in rewards_list or "semantic" in rewards_list
    if VERBOSE_REWARD and _reward_stats["step_count"] % REWARD_LOG_INTERVAL == 0 and should_print_stats:
        # 计算当前 batch 的平均值
        if len(_reward_stats["format"]) > 0:
            avg_format = sum(_reward_stats["format"]) / len(_reward_stats["format"])
            avg_motion_emb = sum(_reward_stats["motion_embedding"]) / len(_reward_stats["motion_embedding"])
            avg_semantic = sum(_reward_stats["semantic"]) / len(_reward_stats["semantic"])
            avg_total = sum(_reward_stats["total"]) / len(_reward_stats["total"])

            print(f"\n{'#'*70}")
            print(f"Step {_reward_stats['step_count']} - Batch Average Rewards")
            print(f"{'#'*70}")
            print(f"  format              : {avg_format:.4f}")
            print(f"  motion_embedding    : {avg_motion_emb:.4f}")
            print(f"  semantic            : {avg_semantic:.4f}")
            print(f"  Total (weighted)    : {avg_total:.4f}")
            print(f"  Samples in batch    : {len(_reward_stats['format'])}")
            print(f"{'#'*70}\n")

            # 清空统计信息
            _reward_stats["format"].clear()
            _reward_stats["motion_embedding"].clear()
            _reward_stats["semantic"].clear()
            _reward_stats["total"].clear()

    return final_rewards


# =====================================================================
# 10) 工厂函数: 创建带自定义权重的奖励函数
# =====================================================================
def create_t2m_reward_func(
    preset: str = None,
    rewards_list: list[str] = None,
    weights: dict[str, float] = None,
    w_format: float = 0.1,
    w_motion: float = 0.9,
    use_soft_format: bool = True,
    use_lcs: bool = False,
    lambda_joint: float = 1.0,
    lambda_vel: float = 1.0,
):
    """
    创建自定义的 T2M 奖励函数

    Example:
        >>> # 使用预设
        >>> reward_func = create_t2m_reward_func(preset="with_physics")

        >>> # 简单配置
        >>> reward_func = create_t2m_reward_func(w_format=0.2, w_motion=0.8)

        >>> # 自定义奖励组合
        >>> reward_func = create_t2m_reward_func(
        ...     rewards_list=["format_soft", "motion_f1", "phys"],
        ...     weights={"format_soft": 0.1, "motion_f1": 0.7, "phys": 0.2}
        ... )
    """
    if preset is not None or rewards_list is not None:
        # 使用灵活配置
        def reward_func(completions, ground_truth, caption=None, **kwargs):
            return t2m_flexible_reward(
                completions, ground_truth, caption,
                preset=preset,
                rewards_list=rewards_list,
                weights=weights,
                lambda_joint=lambda_joint,
                lambda_vel=lambda_vel,
                **kwargs
            )
        if preset:
            reward_func.__name__ = f"t2m_{preset}_reward"
        else:
            reward_func.__name__ = "t2m_custom_reward"
    else:
        # 使用简单配置
        def reward_func(completions, ground_truth, caption=None, **kwargs):
            return t2m_combined_reward(
                completions, ground_truth, caption,
                w_format=w_format,
                w_motion=w_motion,
                use_soft_format=use_soft_format,
                use_lcs=use_lcs,
                **kwargs
            )
        reward_func.__name__ = f"t2m_reward_f{w_format}_m{w_motion}"

    return reward_func


# =====================================================================
# 预定义的奖励函数
# =====================================================================
t2m_basic_reward = create_t2m_reward_func(preset="basic")
t2m_strict_reward = create_t2m_reward_func(preset="strict")
t2m_physics_reward = create_t2m_reward_func(preset="with_physics")
t2m_semantic_preset_reward = create_t2m_reward_func(preset="semantic")
t2m_full_reward = create_t2m_reward_func(preset="full")
t2m_lcs_reward = create_t2m_reward_func(w_format=0.1, w_motion=0.9, use_lcs=True)


# =====================================================================
# 11) M2T (Motion-to-Text) 奖励函数
# =====================================================================

def _get_completion_text(completion) -> str:
    """
    从 completion 中提取文本内容。

    completion 可能是:
    - str: 直接返回
    - list[dict]: 对话格式 [{role: ..., content: ...}]，提取 content

    Returns:
        文本内容
    """
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list) and len(completion) > 0:
        # 对话格式: [{role: "assistant", content: "..."}]
        if isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0].get("content", "")
    return ""


def m2t_format_reward(completion) -> float:
    """
    M2T 格式检查: 验证输出是否包含 <answer>...</answer> 标签

    Args:
        completion: 模型输出 (str 或 list[dict])

    Returns:
        1.0: 格式正确
        0.0: 格式错误
    """
    import re
    content = _get_completion_text(completion)
    pattern = r"<answer>.*?</answer>"
    if re.search(pattern, content, re.DOTALL):
        return 1.0
    return 0.0


def m2t_format_soft_reward(completion) -> float:
    """
    M2T 软格式检查:
    - 包含完整 <answer>...</answer>: 1.0
    - 只有开始标签: 0.5
    - 无标签: 0.0

    Args:
        completion: 模型输出 (str 或 list[dict])
    """
    content = _get_completion_text(completion)
    has_start = "<answer>" in content
    has_end = "</answer>" in content

    if has_start and has_end:
        return 1.0
    elif has_start:
        return 0.5
    return 0.0


def _extract_m2t_answer(completion) -> str:
    """
    从 M2T 输出中提取 <answer>...</answer> 标签内的文本

    Args:
        completion: 模型输出 (str 或 list[dict])

    Returns:
        提取的文本，如果没有找到则返回空字符串
    """
    import re
    content = _get_completion_text(completion)
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""  # 提取不到则返回空字符串


@torch.inference_mode()
def _compute_m2t_semantic_reward(pred_text: str, gt_text: str) -> float:
    """
    计算 M2T 语义相似度: cos(E_text(pred), E_text(gt))

    Args:
        pred_text: 生成的文本描述
        gt_text: 参考文本描述

    Returns:
        余弦相似度值 [-1, 1]
    """
    if not pred_text or not gt_text:
        return 0.0

    try:
        eval_wrapper = _load_evaluator_wrapper()
        if eval_wrapper is None:
            return 0.0

        # 编码生成文本
        word_embs_pred, pos_ohs_pred, len_pred = encode_text_simple(pred_text)
        text_emb_pred = eval_wrapper.get_text_embeddings(word_embs_pred, pos_ohs_pred, len_pred)

        # 编码参考文本
        word_embs_gt, pos_ohs_gt, len_gt = encode_text_simple(gt_text)
        text_emb_gt = eval_wrapper.get_text_embeddings(word_embs_gt, pos_ohs_gt, len_gt)

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(text_emb_pred, text_emb_gt, dim=-1)
        return cos_sim.item()
    except Exception as e:
        print(f"[M2T Reward] _compute_m2t_semantic_reward error: {e}")
        return 0.0


@torch.inference_mode()
def _compute_m2t_semantic_reward_clip(pred_text: str, gt_text: str) -> float:
    """
    使用 CLIP 计算 M2T 语义相似度: cos(CLIP(pred), CLIP(gt))

    参考 UniMo-main/train_grpo.py 实现

    Args:
        pred_text: 生成的文本描述
        gt_text: 参考文本描述 (caption)

    Returns:
        余弦相似度值 [-1, 1]，空文本返回 0.0
    """
    if not pred_text or not gt_text:
        return 0.0

    try:
        import clip
        clip_model, device = _load_clip_model()
        if clip_model is None:
            return 0.0

        # 一次性编码两个文本
        text_inputs = [gt_text, pred_text]
        text_tokens = clip.tokenize(text_inputs, truncate=True).to(device)
        text_embeddings = clip_model.encode_text(text_tokens)

        gt_emb = text_embeddings[0]
        pred_emb = text_embeddings[1]

        # 计算余弦相似度 (保持 [-1, 1] 范围)
        cos_sim = F.cosine_similarity(gt_emb.unsqueeze(0), pred_emb.unsqueeze(0), dim=-1)
        return cos_sim.item()

    except Exception as e:
        print(f"[M2T Reward] _compute_m2t_semantic_reward_clip error: {e}")
        return 0.0


def m2t_semantic_reward(
    completions: list[str],
    ground_truth: list[str] = None,
    **kwargs,
) -> list[float]:
    """
    M2T 语义相似度奖励函数

    计算生成文本与参考文本之间的 text embedding 余弦相似度

    Args:
        completions: 模型生成的输出列表
        ground_truth: 参考文本列表

    Returns:
        奖励值列表
    """
    if ground_truth is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # 提取 <answer>...</answer> 中的文本
        pred_text = _extract_m2t_answer(completion)
        reward = _compute_m2t_semantic_reward(pred_text, gt)
        rewards.append(reward)

    return rewards


# =====================================================================
# 12) 联合训练奖励函数 (Task-Aware Reward)
# =====================================================================

# M2T 奖励统计
_m2t_reward_stats = {
    "format": [],
    "semantic": [],
    "total": [],
    "step_count": 0,
    "t2m_sample_count": 0,  # 累积 T2M 样本数量
    "m2t_sample_count": 0,  # M2T 样本计数（用于详细打印）
}


def get_t2m_reward_stats_for_logging():
    """
    获取 T2M reward 统计信息用于 wandb 记录
    返回当前批次的平均值，并清空统计
    """
    global _reward_stats
    if len(_reward_stats["format"]) == 0:
        return None

    stats = {
        "t2m/format": sum(_reward_stats["format"]) / len(_reward_stats["format"]),
        "t2m/motion_embedding": sum(_reward_stats["motion_embedding"]) / len(_reward_stats["motion_embedding"]),
        "t2m/semantic": sum(_reward_stats["semantic"]) / len(_reward_stats["semantic"]),
        "t2m/total": sum(_reward_stats["total"]) / len(_reward_stats["total"]),
        "t2m/samples": len(_reward_stats["format"]),
    }

    # 清空统计以准备下一个 logging 周期
    _reward_stats["format"].clear()
    _reward_stats["motion_embedding"].clear()
    _reward_stats["semantic"].clear()
    _reward_stats["total"].clear()

    return stats


def get_m2t_reward_stats_for_logging():
    """
    获取 M2T reward 统计信息用于 wandb 记录
    返回当前批次的平均值，并清空统计
    """
    global _m2t_reward_stats
    if len(_m2t_reward_stats["format"]) == 0:
        return None

    stats = {
        "m2t/format": sum(_m2t_reward_stats["format"]) / len(_m2t_reward_stats["format"]),
        "m2t/semantic": sum(_m2t_reward_stats["semantic"]) / len(_m2t_reward_stats["semantic"]),
        "m2t/total": sum(_m2t_reward_stats["total"]) / len(_m2t_reward_stats["total"]),
        "m2t/samples": len(_m2t_reward_stats["format"]),
        "t2m/samples_in_m2t_batch": _m2t_reward_stats["t2m_sample_count"],
    }

    # 清空统计
    _m2t_reward_stats["format"].clear()
    _m2t_reward_stats["semantic"].clear()
    _m2t_reward_stats["total"].clear()
    _m2t_reward_stats["t2m_sample_count"] = 0

    return stats


def task_aware_reward(
    completions: list[str],
    prompts: list = None,
    ground_truth: list[str] = None,
    caption: list[str] = None,
    extra_info: list[dict] = None,
    # T2M 权重
    t2m_w_format: float = 0.1,
    t2m_w_embedding: float = 0.45,
    t2m_w_semantic: float = 0.45,
    # M2T 权重
    m2t_w_format: float = 0.1,
    m2t_w_semantic: float = 0.9,
    **kwargs,
) -> list[float]:
    """
    联合训练奖励函数: 根据 task_type 自动选择 T2M 或 M2T 奖励

    T2M 奖励:
    - format (t2m_w_format): 检查 <Motion>...</Motion> 格式
    - motion_embedding (t2m_w_embedding): motion embedding 相似度
    - semantic (t2m_w_semantic): motion-text 语义相似度

    M2T 奖励:
    - format (m2t_w_format): 检查 <answer>...</answer> 格式
    - semantic (m2t_w_semantic): text embedding 相似度

    Args:
        completions: 模型生成的输出列表
        prompts: prompt 列表 (可选)
        ground_truth: 标准答案列表
        caption: 文本描述列表 (T2M 任务需要)
        extra_info: 额外信息列表，包含 task_type 字段

    Returns:
        奖励值列表
    """
    global _m2t_reward_stats

    if extra_info is None:
        # 默认使用 T2M
        extra_info = [{"task_type": "t2m"}] * len(completions)

    if ground_truth is None:
        ground_truth = [""] * len(completions)

    if caption is None:
        caption = [""] * len(completions)

    rewards = []
    t2m_indices = []
    m2t_indices = []

    # 分离 T2M 和 M2T 样本
    for idx, info in enumerate(extra_info):
        task_type = info.get("task_type", "t2m") if isinstance(info, dict) else "t2m"
        if task_type == "m2t":
            m2t_indices.append(idx)
        else:
            t2m_indices.append(idx)

    # 初始化结果
    rewards = [0.0] * len(completions)

    # ========== 处理 T2M 样本 ==========
    if t2m_indices:
        t2m_completions = [completions[i] for i in t2m_indices]
        t2m_ground_truth = [ground_truth[i] for i in t2m_indices]
        t2m_caption = [caption[i] for i in t2m_indices]

        # 使用 t2m_flexible_reward 计算 T2M 奖励
        t2m_rewards = t2m_flexible_reward(
            completions=t2m_completions,
            ground_truth=t2m_ground_truth,
            caption=t2m_caption,
            preset="semantic",  # 使用 semantic preset
            weights={
                "format": t2m_w_format,
                "motion_embedding": t2m_w_embedding,
                "semantic": t2m_w_semantic,
            },
            extra_info=[extra_info[i] for i in t2m_indices],
        )

        for i, idx in enumerate(t2m_indices):
            rewards[idx] = t2m_rewards[i]

    # 累积 T2M 样本数量
    _m2t_reward_stats["t2m_sample_count"] += len(t2m_indices)

    # ========== 处理 M2T 样本 ==========
    if m2t_indices:
        for idx in m2t_indices:
            completion = completions[idx]
            gt_text = ground_truth[idx]

            # M2T format reward
            r_format = m2t_format_reward(completion)

            # M2T semantic reward (支持 CLIP 或 Evaluator 两种方式)
            pred_text = _extract_m2t_answer(completion)
            use_clip = os.environ.get("M2T_USE_CLIP", "0") == "1"
            if use_clip:
                r_semantic = _compute_m2t_semantic_reward_clip(pred_text, gt_text)
            else:
                r_semantic = _compute_m2t_semantic_reward(pred_text, gt_text)

            # 加权组合
            total_reward = r_format * m2t_w_format + r_semantic * m2t_w_semantic
            total_reward = min(max(total_reward, 0.0), 1.0)

            rewards[idx] = total_reward

            # 收集统计信息
            _m2t_reward_stats["format"].append(r_format)
            _m2t_reward_stats["semantic"].append(r_semantic)
            _m2t_reward_stats["total"].append(total_reward)
            _m2t_reward_stats["m2t_sample_count"] += 1

            # 打印单个 M2T 样本的详细信息
            if PRINT_SAMPLE_DETAILS and VERBOSE_REWARD and (_m2t_reward_stats["m2t_sample_count"] % REWARD_LOG_INTERVAL == 0):
                content = _get_completion_text(completion)
                print(f"\n{'='*70}")
                print(f"M2T Reward Detail - Sample #{_m2t_reward_stats['m2t_sample_count']} (Batch idx={idx})")
                print(f"{'='*70}")
                print(f"Ground Truth: {gt_text[:100]}..." if len(gt_text) > 100 else f"Ground Truth: {gt_text}")
                print(f"Predicted Answer: {pred_text[:100]}..." if len(pred_text) > 100 else f"Predicted Answer: {pred_text}")
                print(f"\nCompletion (first 300 chars):")
                print(f"  {content[:300]}..." if len(content) > 300 else f"  {content}")
                print(f"\nIndividual Rewards:")
                print(f"  format              : {r_format:.4f} (weight={m2t_w_format:.3f}, weighted={r_format * m2t_w_format:.4f})")
                print(f"  semantic            : {r_semantic:.4f} (weight={m2t_w_semantic:.3f}, weighted={r_semantic * m2t_w_semantic:.4f})")
                print(f"\nFinal Reward: {total_reward:.4f}")
                if r_format == 0:
                    print(f"⚠️  FORMAT FAILED: Missing <answer>...</answer> tags!")
                print(f"{'='*70}\n")

    # 打印 M2T 统计信息
    _m2t_reward_stats["step_count"] += 1
    if VERBOSE_REWARD and _m2t_reward_stats["step_count"] % REWARD_LOG_INTERVAL == 0:
        if len(_m2t_reward_stats["format"]) > 0:
            avg_format = sum(_m2t_reward_stats["format"]) / len(_m2t_reward_stats["format"])
            avg_semantic = sum(_m2t_reward_stats["semantic"]) / len(_m2t_reward_stats["semantic"])
            avg_total = sum(_m2t_reward_stats["total"]) / len(_m2t_reward_stats["total"])

            print(f"\n{'#'*70}")
            print(f"Step {_m2t_reward_stats['step_count']} - M2T Batch Average Rewards")
            print(f"{'#'*70}")
            print(f"  format              : {avg_format:.4f}")
            print(f"  semantic            : {avg_semantic:.4f}")
            print(f"  Total (weighted)    : {avg_total:.4f}")
            print(f"  M2T samples         : {len(_m2t_reward_stats['format'])}")
            print(f"  T2M samples         : {_m2t_reward_stats['t2m_sample_count']}")
            print(f"{'#'*70}\n")

            # 清空统计信息
            _m2t_reward_stats["format"].clear()
            _m2t_reward_stats["semantic"].clear()
            _m2t_reward_stats["total"].clear()
            _m2t_reward_stats["t2m_sample_count"] = 0

    return rewards


# 预定义的联合训练奖励函数
task_aware_preset_reward = task_aware_reward


# =====================================================================
# UniMo 风格 Reward 函数（两个独立函数，传给 GRPOTrainer 等权求和）
# =====================================================================

def unified_format_reward_func(completions, extra_info=None, **kwargs):
    """
    UniMo 风格格式 reward（独立函数，返回 0 或 1）

    根据 task_type 自动选择格式检查:
    - T2M: <think>...</think><Motion>...</Motion>
    - M2T: <think>...</think><answer>...</answer>

    作为独立 reward function 传给 GRPOTrainer，与 similarity reward 等权求和。
    """
    rewards = []
    extra_info_list = extra_info if extra_info else [{}] * len(completions)

    for i, completion in enumerate(completions):
        info = extra_info_list[i] if i < len(extra_info_list) else {}
        task_type = info.get("task_type", "t2m") if isinstance(info, dict) else "t2m"
        content = _get_completion_text(completion)

        if task_type == "m2t":
            pattern = r"<think>.+?</think>\s*<answer>.+?</answer>"
            weight = M2T_FORMAT_WEIGHT
        else:
            pattern = r"<think>.+?</think>\s*<Motion>.+?</Motion>"
            weight = T2M_FORMAT_WEIGHT

        rewards.append(weight if re.search(pattern, content, re.DOTALL) else 0.0)
    return rewards


@torch.inference_mode()
def unified_similarity_reward_func(completions, ground_truth=None, caption=None, extra_info=None, **kwargs):
    """
    UniMo 风格语义相似度 reward（独立函数）

    根据 task_type 自动选择相似度计算:
    - T2M: motion_embedding_sim + semantic_sim（范围 ~0-2）
    - M2T: CLIP_cosine_sim * 2（范围 ~0-2）

    作为独立 reward function 传给 GRPOTrainer，与 format reward 等权求和。
    不做手动加权、不 clamp，保留原始动态范围。
    """
    rewards = []
    ground_truth = ground_truth or [""] * len(completions)
    caption = caption or [""] * len(completions)
    pos_tokens_list = kwargs.get("pos_tokens")
    if pos_tokens_list is None:
        pos_tokens_list = [None] * len(completions)
    extra_info_list = extra_info if extra_info else [{}] * len(completions)

    for i, completion in enumerate(completions):
        info = extra_info_list[i] if i < len(extra_info_list) else {}
        task_type = info.get("task_type", "t2m") if isinstance(info, dict) else "t2m"

        if task_type == "m2t":
            # M2T: CLIP text-text cosine similarity
            pred_text = _extract_m2t_answer(completion)
            gt_text = ground_truth[i] if i < len(ground_truth) else ""
            r_semantic = _compute_m2t_semantic_reward_clip(pred_text, gt_text)
            rewards.append(M2T_SEMANTIC_WEIGHT * r_semantic)
        else:
            # T2M: embedding_sim + semantic_sim（和 UniMo 一样直接求和）
            content = _get_completion_text(completion)
            motion_content = extract_motion_content(content)
            pred_tokens = parse_motion_token_ids(motion_content) if motion_content else []
            gt_str = ground_truth[i] if i < len(ground_truth) else ""
            gt_tokens = parse_motion_token_ids(gt_str)
            cap = caption[i] if i < len(caption) else ""
            cur_pos_tokens = pos_tokens_list[i] if i < len(pos_tokens_list) else None

            r_embedding = _compute_embedding_reward(pred_tokens, gt_tokens, cap, cur_pos_tokens)
            r_semantic = _compute_semantic_reward(pred_tokens, cap, cur_pos_tokens)
            rewards.append(T2M_MOTION_WEIGHT * r_embedding + T2M_SEMANTIC_WEIGHT * r_semantic)
    return rewards
