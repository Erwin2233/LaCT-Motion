"""
T2M (Text-to-Motion) 模块

包含 Motion-R1 论文中使用的模型、工具和数据集。
原始代码来源于 ULM 项目，已整合到 TRL 中以便独立打包。
"""

import os
from _paths import project_path

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
# TRL 项目根目录 (trl/t2m -> trl -> trl_project_root)
TRL_ROOT = project_path()
DATA_ROOT = project_path()
GLOVE_PATH = os.path.join(DATA_ROOT, "glove")
CHECKPOINTS_PATH = os.path.join(DATA_ROOT, "checkpoints")
# VQ-VAE 权重路径 (在 TRL 项目根目录的 ckpt/ 下)
VQVAE_PATH = os.path.join(TRL_ROOT, "ckpt", "vqvae.pth")

__all__ = [
    "TRL_ROOT",
    "DATA_ROOT",
    "GLOVE_PATH",
    "CHECKPOINTS_PATH",
    "VQVAE_PATH",
]
