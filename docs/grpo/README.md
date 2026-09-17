# Latent-CoT-GRPO

基于 GRPO（Group Relative Policy Optimization）强化学习的 CoconutMotion Text-to-Motion 生成优化。

在 [latent-cot-motion](../latent-cot-motion/) 的 SFT（Coconut 式 curriculum learning）训练基础上，通过 reward 驱动的策略优化进一步提升 motion 生成质量。

## 核心思路

```
SFT 阶段（latent-cot-motion）          GRPO 阶段（本项目）
┌─────────────────────────┐      ┌──────────────────────────────┐
│ Coconut curriculum SFT  │      │ 采样 G 个 completion         │
│ Stage 0→8 逐步替换      │ ──→  │ 计算 reward（格式/F1/物理）  │
│ text CoT → latent token │      │ Group-relative advantage     │
│                         │      │ Policy gradient + KL 正则    │
└─────────────────────────┘      └──────────────────────────────┘
```

GRPO 优化的是**模型权重**（影响 latent hidden state 质量和 motion token 生成），而非 latent token 本身。CoconutMotion 的 multi-pass forward 对 GRPO 透明——生成阶段用 `CoconutMotion.generate()` 采样，log prob 阶段用 `CoconutMotion.forward()` 计算，latent 区域在 loss 中被 mask 掉。

## 项目结构

```
latent-cot-grpo/
├── train_grpo.py            # GRPO 训练主循环
├── grpo_loss.py             # 6 种 GRPO loss 实现
├── rewards.py               # Reward 封装（导入 TRL T2M rewards）
├── coconut_motion.py        # CoconutMotion（增加 sampling 支持）
├── configs/
│   ├── t2m_grpo.yaml        # GRPO 训练配置
│   └── t2m_coconut.yaml     # SFT 配置（从 latent-cot-motion 复制，只读）
├── dataset.py               # 数据加载（从 latent-cot-motion 复制，只读）
├── utils.py                 # 工具函数（从 latent-cot-motion 复制，只读）
├── evaluate.py              # 评估（从 latent-cot-motion 复制，只读）
├── inference.py             # 推理（从 latent-cot-motion 复制，只读）
├── data/                    # 数据目录（从 latent-cot-motion 复制）
├── checkpoints/             # SFT checkpoint（从 latent-cot-motion 复制）
└── GRPO_IMPLEMENTATION_PLAN.md
```

标注"只读"的文件从 latent-cot-motion 直接复制，无需修改。

## 支持的 Loss 类型

通过 `loss_type` 配置切换，参照 [TRL GRPOTrainer](../trl/trl/trainer/grpo_trainer.py) 实现：

| Loss Type | 归一化方式 | 来源 |
|-----------|-----------|------|
| `grpo` | 按序列长度 | DeepSeekMath |
| **`dapo`** (默认) | 按全局 active token count | DAPO |
| `dr_grpo` | 按 max_completion_length（常数） | Dr. GRPO |
| `bnpo` | 按 local batch token count | BNPO |
| `cispo` | clip importance sampling | MiniMax-M1 |
| `sapo` | adaptive soft gating | SAPO |

## Reward 函数

直接导入 [TRL T2M rewards](../trl/trl/rewards/t2m_rewards.py)，支持 10 种 reward 和 5 种预设组合：

| 预设 | 组合 | 权重 |
|------|------|------|
| `basic` | format_soft + F1 | 0.1 + 0.9 |
| **`with_physics`** (默认) | format_soft + F1 + phys | 0.1 + 0.7 + 0.2 |
| `strict` | format + LCS + phys | 0.1 + 0.7 + 0.2 |
| `semantic` | format + embedding + semantic | 0.1 + 0.45 + 0.45 |
| `full` | format_soft + F1 + semantic + phys | 0.1 + 0.4 + 0.3 + 0.2 |

## 快速开始

### 1. 训练

```bash
cd latent-cot-grpo

# 4 GPU 训练
torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml

# 2 GPU 快速验证
torchrun --nproc_per_node=2 train_grpo.py --config configs/t2m_grpo.yaml
```

### 2. 评估

```bash
python evaluate.py --model_path checkpoints/grpo/final/ --config configs/t2m_grpo.yaml
```

### 3. 推理

```bash
python inference.py --model_path checkpoints/grpo/final/ --text "a person walks forward slowly"
```

## 关键配置项

`configs/t2m_grpo.yaml` 中的主要参数：

```yaml
# 生成
num_generations: 4       # 每个 prompt 采样 G 个 completion
temperature: 0.7         # 采样温度
top_p: 0.9               # nucleus sampling

# Loss
loss_type: dapo          # 6 种可选
beta: 0.01               # KL 惩罚系数（0 = 不加 KL）
epsilon: 0.2             # PPO clipping

# 训练
lr: 1.0e-6               # 低学习率
num_epochs: 3            # RL 阶段较短
batch_size_training: 2   # 因 G 倍膨胀

# LoRA
use_lora: true
lora_r: 16
lora_alpha: 32

# Reward
reward_preset: with_physics
```

## LoRA 策略

使用 LoRA 避免同时加载 policy 和 reference 两套完整模型：

- **Policy** = base model + LoRA adapter（可训练）
- **Reference** = base model（冻结，`disable_adapter()` 即可获取 ref log probs）
- 额外显存 < 1%

## 依赖

- 本项目的 SFT checkpoint：`latent-cot-motion/checkpoints/`
- TRL（reward 函数）：`trl/`
- PyTorch, Transformers, PEFT, Datasets

## 参考

- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- [Coconut](https://arxiv.org/abs/2412.06769) — latent CoT 训练
- [TRL](https://github.com/huggingface/trl) — GRPO loss 和 reward 函数参考
