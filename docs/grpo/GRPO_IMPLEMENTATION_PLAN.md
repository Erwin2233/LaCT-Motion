# 计划：为 latent-cot-motion 加入 GRPO 强化学习

## Context

**问题**：`latent-cot-motion` 目前仅使用 SFT（Coconut 式 curriculum learning）训练 Text-to-Motion 模型。SFT 的局限在于只能模仿训练数据分布，无法通过 reward 信号探索更优的 motion 生成策略。

**目标**：在 SFT 训练完成后，加入 GRPO（Group Relative Policy Optimization）阶段，利用 motion 质量 reward 进一步优化模型。

**工作目录**：`latent-cot-grpo/` — 所有改动在此目录下实现，对其他目录只读。

**只读参考资源**：
- `latent-cot-motion/` — SFT 代码（只读，从中复制需修改的文件）
- `trl/` — TRL 框架（只读，导入 reward 函数 + 参考 loss 公式）
  - `trl/rewards/t2m_rewards.py`（2279 行）— **10+ 种 T2M reward 函数**，含 5 种预设组合
  - `trl/trainer/grpo_trainer.py`（2362 行）— GRPO loss 公式参考（6 种 loss type）
  - `trl/trainer/grpo_config.py`（874 行）— 超参配置参考
- `latent-reasoning-motion/` — 已有 latent motion GRPO 训练框架
- `Awesome-Latent-CoT/repos/HRPO/` 和 `repos/SofT-GRPO/` — GRPO 变体参考

---

## 核心挑战

CoconutMotion 模型的 multi-pass forward（每个 latent token 用前一位置的 hidden state 替换）与标准 GRPO 的兼容问题：

1. **生成阶段**：`CoconutMotion.generate()` 已能正常采样 motion token → 可直接复用（需增加 sampling 支持）
2. **Log prob 计算**：`CoconutMotion.forward()` 已能计算 logits → 可从中提取 per-token log prob
3. **Latent 区域处理**：latent token 无离散目标 → 在 GRPO loss 中 mask 掉（labels=-100）
4. **梯度流**：reward 信号通过 motion token 的 policy gradient → 反传到模型权重 → 间接改善 latent hidden state 的计算

**结论**：GRPO 优化的是**模型权重**（影响 latent hidden state 质量和 motion token 生成），而非 latent token 本身。这与标准 GRPO 的框架兼容。

---

## 实施方案：自定义 GRPO 训练循环

选择自定义循环而非 TRL GRPOTrainer，原因：
- CoconutMotion 的 multi-pass forward 不兼容 TRL 的标准 `model.forward()` / `model.generate()` 调用
- Reference model 也需要 multi-pass forward（TRL 的 ref model 只做标准 forward）
- 需要精确控制 latent 区域的 mask 和 KV cache
- 避免 text→token→text→token 的格式转换（latent tokens 是特殊 token）
- 项目已有完善的训练基础设施（DDP、checkpoint、cosine LR）

同时**直接导入 TRL 的 reward 函数**和**参照 TRL 的 loss 公式**，降低开发量。

### 项目目录结构

```
latent-cot-grpo/
├── coconut_motion.py       # 从 latent-cot-motion 复制并修改（增加 sampling）
├── train_grpo.py           # GRPO 训练主循环（新建）
├── grpo_loss.py            # GRPO loss 实现（新建，参照 TRL 6 种 loss type）
├── rewards.py              # Reward 封装（新建，导入 TRL 或自定义）
├── configs/
│   └── t2m_grpo.yaml       # GRPO 超参配置（新建）
├── dataset.py              # 从 latent-cot-motion 复制（不改）
├── utils.py                # 从 latent-cot-motion 复制（不改）
├── evaluate.py             # 从 latent-cot-motion 复制（不改）
├── inference.py            # 从 latent-cot-motion 复制（不改）
└── GRPO_IMPLEMENTATION_PLAN.md  # 本计划文件
```

**初始化命令**：
```bash
mkdir -p latent-cot-grpo/configs

# 复制不修改的文件
cd latent-cot-grpo
cp ../latent-cot-motion/dataset.py .
cp ../latent-cot-motion/utils.py .
cp ../latent-cot-motion/evaluate.py .
cp ../latent-cot-motion/inference.py .
cp ../latent-cot-motion/configs/t2m_coconut.yaml configs/
cp -r ../latent-cot-motion/data data           # 数据目录
cp -r ../latent-cot-motion/checkpoints checkpoints  # SFT checkpoint

# 复制需要修改的文件
cp ../latent-cot-motion/coconut_motion.py .
```

---

## Step 1：Reward 函数

### 方案 A：直接导入 TRL 已有的 T2M Reward（推荐）

**无需新建 rewards.py** — 直接从 TRL 导入使用。

**来源**：`trl/trl/rewards/t2m_rewards.py`（2279 行）

#### 已有的 10 种 T2M Reward

| Reward 函数 | TRL 导入名 | 说明 |
|------------|-----------|------|
| 格式检查（严格） | `t2m_format_reward` | `<think>...</think><Motion>...</Motion>` 完整性，0 或 1 |
| 格式检查（软） | `t2m_format_soft_reward` | 部分匹配也给分（0.0-1.0） |
| Token F1 | `t2m_motion_f1_reward` | 基于 token 集合的 F1 分数 |
| Token LCS | `t2m_motion_lcs_reward` | 最长公共子序列（顺序匹配） |
| Embedding 相似度 | `t2m_motion_embedding_reward` | cos(f_m(pred), f_m(gt))，需 VQ-VAE + Evaluator |
| 语义相似度 | `t2m_semantic_reward` | cos(f_m(pred), f_text(text))，需 caption + Evaluator |
| 物理合理性（综合） | `t2m_phys_reward` | 关节角度 + 速度平滑 + 脚部滑动 |
| 关节角度 | `t2m_phys_joint_reward` | 6D rotation → 旋转角度违规惩罚 |
| 速度平滑 | `t2m_phys_vel_reward` | 加速度阈值惩罚 |
| 脚部滑动 | `t2m_phys_skating_reward` | foot contact 约束 |

#### 5 种预设组合（REWARD_PRESETS）

| 预设名 | 组合 | 权重 |
|--------|------|------|
| `basic` | format_soft + F1 | 0.1 + 0.9 |
| `with_physics` | format_soft + F1 + phys | 0.1 + 0.7 + 0.2 |
| `strict` | format + LCS + phys | 0.1 + 0.7 + 0.2 |
| `semantic` | format + embedding + semantic | 0.1 + 0.45 + 0.45 |
| `full` | format_soft + F1 + semantic + phys | 0.1 + 0.4 + 0.3 + 0.2 |

#### TRL Reward 函数签名

```python
def t2m_xxx_reward(
    completions: list[list[dict[str, str]]],  # [[{"content": "..."}]]
    ground_truth: list[str],                   # GT motion token 文本
    **kwargs                                    # 数据集额外列（caption 等）
) -> list[float]:
```

#### 在自定义 GRPO 循环中使用 TRL Reward

```python
import sys
sys.path.insert(0, "trl")

from third_party.trl_motion.rewards import t2m_format_soft_reward, t2m_motion_f1_reward, t2m_phys_reward

def compute_rewards(generated_texts, ground_truths, captions=None):
    """调用 TRL reward 函数计算综合 reward"""
    # 转换为 TRL 期望的格式
    completions = [[{"content": text}] for text in generated_texts]

    # 计算各项 reward
    format_rewards = t2m_format_soft_reward(completions, ground_truths)
    f1_rewards = t2m_motion_f1_reward(completions, ground_truths)
    phys_rewards = t2m_phys_reward(completions, ground_truths)

    # 加权组合（with_physics preset）
    rewards = []
    for f, a, p in zip(format_rewards, f1_rewards, phys_rewards):
        rewards.append(0.1 * f + 0.7 * a + 0.2 * p)

    return torch.tensor(rewards, dtype=torch.float32)
```

### 方案 B：自行实现 Reward（如不想依赖 TRL）

**新建文件**：`latent-cot-grpo/rewards.py`

参考 TRL `t2m_rewards.py` 的实现，精简为 3 个核心 reward：

#### 1.1 Format Reward（格式正确性）
```
检查生成文本是否包含完整的 <Motion>...</Motion> 标签
- 有完整标签且内容非空 → 1.0
- 标签不完整 → 0.0 ~ 0.5（soft scoring）
- 完全无标签 → 0.0
```

#### 1.2 Token F1 Reward（token 级匹配）
```
解析生成的 motion token 序列，与 ground truth 基于集合计算 F1
- precision = |pred ∩ gt| / |pred|
- recall = |pred ∩ gt| / |gt|
- F1 = 2 * precision * recall / (precision + recall)
参考：TRL t2m_rewards.py 的 r_motion_token_f1() 函数
```

#### 1.3 Embedding Similarity Reward（语义匹配）
```
使用 Motion-R1 的预训练 motion encoder 计算生成 motion 与 GT motion 的 embedding 距离
- 需加载 EvaluatorModelWrapper（来自 evaluate.py）
- reward = cosine_similarity(f_m(pred), f_m(gt))
- 参考：TRL t2m_rewards.py 的 r_motion_embedding_similarity() 和 _compute_motion_embedding()
```

**关键文件参考**：
- `utils.py:parse_motion_tokens()` (L57-63) — 从文本解析 motion code
- `utils.py:compute_motion_accuracy()` (L66-93) — token 级精度计算
- `evaluate.py` — FID/R-precision 评估基础设施
- `trl/trl/rewards/t2m_rewards.py` — 完整 reward 实现参考

---

## Step 2：修改 `coconut_motion.py` 支持采样生成

**修改文件**：`latent-cot-grpo/coconut_motion.py`（从 latent-cot-motion 复制后修改）

### 2.1 新增 top_p / top_k filtering 辅助函数

在 `coconut_motion.py` 顶部添加：
```python
import torch.nn.functional as F

def top_p_filtering(logits, top_p=0.9):
    """Nucleus sampling: 只保留累积概率 <= top_p 的 token"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 移除累积概率超过 top_p 的 token（保留第一个超过的）
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k_filtering(logits, top_k=50):
    """Top-k sampling: 只保留概率最高的 k 个 token"""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

def sample_from_logits(logits, temperature=0.0, do_sample=False, top_p=1.0, top_k=0):
    """统一的采样函数：greedy 或 temperature sampling"""
    if temperature > 0 and do_sample:
        logits = logits / temperature
        if top_p < 1.0:
            logits = top_p_filtering(logits, top_p)
        if top_k > 0:
            logits = top_k_filtering(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        return torch.argmax(logits, dim=-1)
```

### 2.2 修改 `generate()` 签名

**当前** (L356-365):
```python
def generate(self, input_ids, attention_mask, position_ids=None,
             max_new_tokens=256, output_embedding=False, synced_gpus=False, **kwargs):
```

**改为**:
```python
def generate(self, input_ids, attention_mask, position_ids=None,
             max_new_tokens=256, output_embedding=False, synced_gpus=False,
             temperature=0.0, do_sample=False, top_p=1.0, top_k=0, **kwargs):
```

### 2.3 修改第一个 token 的采样 (L394)

**当前**:
```python
next_tokens = torch.argmax(last_logits[:, -1, :], dim=-1)
```

**改为**:
```python
next_tokens = sample_from_logits(
    last_logits[:, -1, :], temperature, do_sample, top_p, top_k
)
```

### 2.4 修改 autoregressive loop 中的采样 (L436)

**当前**:
```python
next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)
```

**改为**:
```python
next_tokens = sample_from_logits(
    out.logits[:, -1, :], temperature, do_sample, top_p, top_k
)
```

---

## Step 3：实现 GRPO 训练核心

**新建文件**：`latent-cot-grpo/train_grpo.py`

### 3.1 GRPO 训练循环

```python
"""
GRPO Training for CoconutMotion

Usage:
    torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml
"""

输入：SFT 训练好的 CoconutMotion 模型 (policy_model)
      冻结的 SFT 模型副本 (ref_model)  — 或 LoRA 模式下 disable adapter 获取 ref logps
      训练数据（question + latent tokens + ground_truth motion）

For each epoch:
  For each batch of prompts:

    # === 1. 采样 G 个 completion ===
    completions = []
    for g in range(num_generations):  # G=4~8
        # 使用 CoconutMotion.generate() 采样（temperature > 0）
        generated = policy_model.generate(
            input_ids=prompt_with_latent_tokens,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
        completions.append(generated)

    # === 2. 计算 Reward ===
    # 使用 TRL reward 函数（或自定义 reward）
    rewards = compute_rewards(completions, ground_truth)  # [B*G]

    # === 3. 计算 Group Advantage ===
    grouped_rewards = rewards.view(B, G)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True) + 1e-4
    advantages = (grouped_rewards - group_mean) / group_std  # [B, G]
    advantages = advantages.view(B * G)

    # === 4. 计算 Policy Log Prob ===
    # 拼接 prompt + completion，通过 CoconutMotion.forward() 获取 logits
    full_input_ids = concat(prompt_with_latent, completion_tokens)
    labels = build_labels(full_input_ids)
    # labels: latent 区域=-100, prompt=-100, completion 部分=token_ids

    outputs = policy_model.forward(full_input_ids, labels=labels)
    logits = outputs.logits

    # 提取 completion 区域的 per-token log prob
    per_token_logps, completion_mask = compute_per_token_log_probs(logits, labels)

    # === 5. 计算 Reference Log Prob（KL 正则项）===
    with torch.no_grad():
        if use_lora:
            with policy_model.disable_adapter():
                ref_outputs = policy_model.forward(full_input_ids, labels=labels)
        else:
            ref_outputs = ref_model.forward(full_input_ids, labels=labels)
        ref_per_token_logps, _ = compute_per_token_log_probs(ref_outputs.logits, labels)

    # === 6. 计算 GRPO Loss ===
    loss = compute_grpo_loss(
        per_token_logps, ref_per_token_logps, advantages,
        completion_mask, loss_type=config.loss_type, beta=config.beta,
        epsilon=config.epsilon
    )

    # === 7. 反向传播 + 更新 ===
    loss.backward()
    clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

### 3.2 多种 GRPO Loss 实现（参考 TRL）

参照 TRL `grpo_trainer.py` L2124-2286 的 loss 实现，支持多种 loss type：

```python
def compute_grpo_loss(
    per_token_logps,        # (B, T) policy log probs
    ref_per_token_logps,    # (B, T) reference log probs
    advantages,             # (B,) group-relative advantages
    completion_mask,        # (B, T) completion 区域 mask
    loss_type="dapo",       # grpo / dapo / dr_grpo / bnpo / cispo / sapo
    beta=0.01,              # KL 系数
    epsilon=0.2,            # clipping epsilon
    max_completion_length=256,  # DR-GRPO 需要
):
    """
    多种 GRPO Loss 实现（参照 TRL grpo_trainer.py L2194-2242）

    Loss types:
    - grpo: 原始 GRPO，按序列长度归一化（有长度偏差）
    - dapo: 按全局 active token count 归一化（TRL 默认，推荐）
    - dr_grpo: 按 max_completion_length 常数归一化
    - bnpo: 按 local batch token count 归一化
    - cispo: clip importance sampling weights（MiniMax-M1）
    - sapo: adaptive soft gating（温度控制的平滑门控）
    """
    advantages = advantages.unsqueeze(1)  # (B, 1)

    # Importance sampling ratio（对于 on-policy 训练，old_logps ≈ per_token_logps）
    old_per_token_logps = per_token_logps.detach()
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    # KL divergence
    if beta > 0:
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )

    # === Loss 计算（按 type 分支） ===
    if loss_type == "cispo":
        # MiniMax-M1 style: clip importance weights
        clamped_ratios = torch.clamp(ratio, max=1 + epsilon).detach()
        per_token_loss = -clamped_ratios * advantages * per_token_logps

    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
        # PPO-style clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = ratio * advantages
        per_token_loss2 = clipped_ratio * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    elif loss_type == "sapo":
        # Soft Adaptive Policy Optimization
        temp_neg, temp_pos = 1.05, 1.0
        per_token_loss = torch.empty_like(ratio)
        pos_mask = advantages.expand_as(ratio) > 0
        for temp, mask in [(temp_pos, pos_mask), (temp_neg, ~pos_mask)]:
            sigmoid_input = temp * (ratio[mask] - 1)
            sigmoid_smoothed = torch.sigmoid(sigmoid_input)
            per_token_loss[mask] = sigmoid_smoothed * 4 / temp
        per_token_loss = -per_token_loss * advantages

    # 加入 KL 项
    if beta > 0:
        per_token_loss = per_token_loss + beta * per_token_kl

    # === 归一化方式（按 loss_type 分支） ===
    if loss_type in ["grpo", "sapo"]:
        # 按序列长度归一化
        loss = ((per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)).mean()

    elif loss_type == "bnpo":
        # 按 local batch active token count 归一化
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    elif loss_type == "dr_grpo":
        # 按 max_completion_length 归一化（常数）
        batch_size = per_token_logps.size(0)
        loss = (per_token_loss * completion_mask).sum() / (batch_size * max_completion_length)

    elif loss_type in ["dapo", "cispo"]:
        # 按全局 active token count 归一化（需 all_reduce）
        num_active = completion_mask.sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_active)
        loss = (per_token_loss * completion_mask).sum() / num_active.clamp(min=1.0)

    return loss
```

### 3.3 Per-Token Log Prob 计算

```python
def compute_per_token_log_probs(logits, labels):
    """从 logits 和 labels 计算 per-token log probability

    参考 TRL grpo_trainer.py L977-979 的 selective_log_softmax

    Args:
        logits: (B, L, V) model output logits
        labels: (B, L) with -100 for masked positions

    Returns:
        per_token_logps: (B, L-1) log probs (masked positions = 0)
        mask: (B, L-1) binary mask for completion tokens
    """
    # Shift: logits[t] predicts labels[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Log softmax + gather
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(
        -1, shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Mask: only completion tokens (labels != -100)
    mask = (shift_labels != -100).float()
    per_token_logps = per_token_logps * mask

    return per_token_logps, mask
```

---

## Step 4：支持 LoRA 微调（降低显存）

**实现位置**：`train_grpo.py`

GRPO 需要同时持有 policy_model 和 ref_model。使用 LoRA：
- 在 SFT 模型基础上加 LoRA adapter
- policy = base + LoRA（可训练）
- ref = base（冻结，disable adapter 即可获取 ref logits）
- 显存仅增加 LoRA 参数量（<1% 额外）

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 对 CoconutMotion 内部的 base_causallm 应用 LoRA
policy_model.base_causallm = get_peft_model(policy_model.base_causallm, lora_config)

# 获取 ref log probs（disable LoRA adapter）：
with policy_model.base_causallm.disable_adapter():
    ref_outputs = policy_model.forward(full_input_ids, labels=labels)
```

---

## Step 5：新增 GRPO 配置文件

**新建文件**：`latent-cot-grpo/configs/t2m_grpo.yaml`

```yaml
# === 基础配置（继承 SFT） ===
model_id: Qwen/Qwen2.5-3B-Instruct
sft_checkpoint: ./checkpoints/coconut_stage8_final/  # SFT 最终模型
bf16: true
coconut: true
max_latent_stage: 8
c_thought: 2

# === GRPO 特有配置 ===
mode: grpo

# 生成
num_generations: 4          # 每个 prompt 采样 G 个 completion
temperature: 0.7            # 采样温度
top_p: 0.9                  # nucleus sampling
max_new_tokens: 256

# RL 超参
loss_type: dapo             # grpo / dapo / dr_grpo / bnpo / cispo / sapo（推荐 dapo）
beta: 0.01                  # KL 惩罚系数（0 则不计算 KL）
epsilon: 0.2                # PPO-style clipping epsilon
scale_rewards: group        # group / batch / none
lr: 1e-6                    # 低学习率（参考 TRL 默认值）
min_lr: 1e-7
warmup_steps: 50
num_epochs: 3               # RL 阶段较短
grad_clip: 1.0

# LoRA
use_lora: true
lora_r: 16
lora_alpha: 32

# Reward 配置
reward_source: trl           # trl（导入 TRL reward）或 custom（自定义）
reward_preset: with_physics  # basic / with_physics / strict / semantic / full
# 自定义权重（可覆盖 preset）：
# reward_format_weight: 0.1
# reward_f1_weight: 0.7
# reward_phys_weight: 0.2

# 数据
batch_size_training: 2      # 因 G 倍膨胀，batch 要小
gradient_accumulation_steps: 16

# Logging
log_interval: 10            # 每 N 步打印一次 metrics
save_interval: 500          # 每 N 步保存 checkpoint
eval_interval: 200          # 每 N 步运行验证
```

---

## Step 6：集成与入口脚本

**新建文件**：`latent-cot-grpo/train_grpo.py`

主函数结构：
```
1. 解析 GRPO 配置（t2m_grpo.yaml）
2. 加载 SFT checkpoint → CoconutMotion
3. 应用 LoRA → policy_model
4. 初始化 reward 函数（TRL 导入或自定义）
5. 加载训练数据（复用 dataset.py 的 get_question_latent_dataset）
6. GRPO 训练循环（Step 3 的伪代码）
   - 每 N 步 validate_generation（复用 train.py）
   - 每 N 步保存 checkpoint
7. 保存最终模型
```

**DDP 启动命令**：
```bash
cd latent-cot-grpo
torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml
```

---

## 文件变更总结

所有文件在 `latent-cot-grpo/` 下：

| 操作 | 文件 | 内容 |
|------|------|------|
| **复制+修改** | `coconut_motion.py` | 从 latent-cot-motion 复制，增加 temperature/sampling 支持 |
| **新建** | `train_grpo.py` | GRPO 训练主循环（含 log prob 提取） |
| **新建** | `grpo_loss.py` | GRPO loss 实现（6 种 loss type，参照 TRL） |
| **新建** | `rewards.py` | Reward 封装（导入 TRL reward 或自定义） |
| **新建** | `configs/t2m_grpo.yaml` | GRPO 超参配置 |
| **复制** | `dataset.py` | 从 latent-cot-motion 复制（不改） |
| **复制** | `utils.py` | 从 latent-cot-motion 复制（不改） |
| **复制** | `evaluate.py` | 从 latent-cot-motion 复制（不改） |
| **复制** | `inference.py` | 从 latent-cot-motion 复制（不改） |
| **复制** | `data/` | 从 latent-cot-motion 复制（数据目录） |
| **复制** | `checkpoints/` | 从 latent-cot-motion 复制（SFT checkpoint） |

---

## 复用的已有代码

| 来源 | 文件 | 复用内容 |
|------|------|---------|
| **TRL（导入）** | `trl/rewards/t2m_rewards.py` | 10+ 种 T2M reward 函数（可直接 import） |
| **TRL（参考）** | `trl/trainer/grpo_trainer.py` L2194-2242 | GRPO loss 公式（6 种 loss type） |
| **TRL（参考）** | `trl/trainer/grpo_config.py` | 超参默认值和范围 |
| latent-cot-motion | `utils.py:parse_motion_tokens()` | 解析 motion code |
| latent-cot-motion | `utils.py:compute_motion_accuracy()` | token 级精度计算 |
| latent-cot-motion | `dataset.py:get_question_latent_dataset()` | 构造 prompt+latent 输入 |
| latent-cot-motion | `evaluate.py:EvaluatorModelWrapper` | embedding similarity reward |
| latent-cot-motion | `coconut_motion.py:CoconutMotion.forward()` | log prob 计算 |
| latent-reasoning-motion | `rewards/unified_rewards.py` | reward 函数设计参考 |
| HRPO repo | `trl/trainer/grpo_trainer.py` | GRPO loss 公式参考 |

---

## TRL 提供的 6 种 Loss Type（本方案全部支持）

通过 `loss_type` 配置参数切换：

| Loss Type | 归一化方式 | 特点 | 来源 |
|-----------|-----------|------|------|
| `grpo` | 按序列长度 | 原始 GRPO，有长度偏差 | DeepSeekMath |
| `dapo` (推荐) | 按全局 active token count | 消除长度偏差 | DAPO |
| `dr_grpo` | 按 max_completion_length | 常数归一化 | Dr. GRPO |
| `bnpo` | 按 local batch token count | 类似 dapo 但仅本地 | BNPO |
| `cispo` | clip importance sampling | 截断重要性采样权重 | MiniMax-M1 |
| `sapo` | adaptive soft gating | 温度控制的平滑门控 | SAPO |

---

## 验证方案

### 阶段性验证

1. **Reward 函数验证**（无需训练）：
   ```bash
   # 测试 TRL reward 函数
   python -c "
   import sys; sys.path.insert(0, 'trl')
   from third_party.trl_motion.rewards import t2m_format_soft_reward, t2m_motion_f1_reward
   completions = [[{'content': '<think>reasoning</think><Motion><Motion_1> <Motion_2></Motion>'}]]
   gt = ['<Motion_1> <Motion_2> <Motion_3>']
   print('Format:', t2m_format_soft_reward(completions, gt))
   print('F1:', t2m_motion_f1_reward(completions, gt))
   "
   ```
   - 验证 reward 分布合理（均值 0.3-0.7，有区分度）
   - 对比 greedy 和 sampling 生成的 reward 差异

2. **采样生成验证**（修改 coconut_motion.py 后）：
   ```bash
   # 验证不同 temperature 产生不同输出
   python -c "
   from coconut_motion import CoconutMotion
   model = CoconutMotion.from_pretrained('checkpoints/sft/')
   for temp in [0.0, 0.3, 0.7, 1.0]:
       out = model.generate(ids, mask, temperature=temp, do_sample=temp>0)
       print(f'temp={temp}:', out[0][:20])
   "
   ```

3. **GRPO 训练基础验证**：
   ```bash
   cd latent-cot-grpo
   torchrun --nproc_per_node=2 train_grpo.py --config configs/t2m_grpo.yaml
   ```
   检查指标：
   - `loss` 下降且不 NaN
   - `reward/mean` 逐步提升
   - `kl` 不爆炸（保持 <0.1）
   - advantage 分布近似 N(0,1)

4. **端到端质量验证**：
   ```bash
   python evaluate.py --model_path checkpoints/grpo_final/ --config configs/t2m_grpo.yaml
   python inference.py --model_path checkpoints/grpo_final/ --text "a person walks forward slowly"
   ```
   预期指标：FID 下降 5-15%，R-Precision Top-1 提升 2-5%

---

## 后续扩展（Phase 2+）

完成基础 GRPO 后可进一步探索：

1. **切换 Loss Type**：对比 `dapo` vs `dr_grpo` vs `cispo` vs `sapo` 的效果
2. **Reward 升级**：从 `basic` → `with_physics` → `semantic` → `full` 逐步增加复杂度
3. **KL 调优**：调整 `beta`（0 → 0.001 → 0.01 → 0.1），找到探索/稳定平衡点
4. **Physics Reward 精细化**：分别加权 `phys_joint` + `phys_vel` + `phys_skating`
5. **SofT-GRPO**：在 motion token 生成时使用 Gumbel-softmax（参考 repos/SofT-GRPO）
6. **自适应 latent token 数量**：参考 TaH 论文，根据 prompt 难度动态分配
7. **Curriculum RL**：分阶段 RL（先大 epsilon 探索，再小 epsilon 收敛）
