# 计划：为 latent-cot-motion 加入 GRPO 强化学习

## Context

**问题**：`latent-cot-motion` 目前仅使用 SFT（Coconut 式 curriculum learning）训练 Text-to-Motion 模型。SFT 的局限在于只能模仿训练数据分布，无法通过 reward 信号探索更优的 motion 生成策略。

**目标**：在 SFT 训练完成后，加入 GRPO（Group Relative Policy Optimization）阶段，利用 motion 质量 reward 进一步优化模型。

**关键发现**：
- `latent-reasoning-motion/` 已有完整的 GRPO 训练框架（rewards、latent recurrent reasoner、训练脚本），可大量复用
- `Awesome-Latent-CoT/repos/HRPO/` 和 `repos/SofT-GRPO/` 提供 GRPO 变体参考
- CoconutMotion 的 multi-pass forward 机制需要特殊处理（非标准自回归生成）

---

## 核心挑战

CoconutMotion 模型的 multi-pass forward（每个 latent token 用前一位置的 hidden state 替换）与标准 GRPO 的兼容问题：

1. **生成阶段**：`CoconutMotion.generate()` 已能正常采样 motion token → 可直接复用
2. **Log prob 计算**：`CoconutMotion.forward()` 已能计算 logits → 可从中提取 per-token log prob
3. **Latent 区域处理**：latent token 无离散目标 → 在 GRPO loss 中 mask 掉（labels=-100）
4. **梯度流**：reward 信号通过 motion token 的 policy gradient → 反传到模型权重 → 间接改善 latent hidden state 的计算

**结论**：GRPO 优化的是**模型权重**（影响 latent hidden state 质量和 motion token 生成），而非 latent token 本身。这与标准 GRPO 的框架兼容。

---

## 实施方案：自定义 GRPO 训练循环

选择自定义循环而非 TRL GRPOTrainer，原因：
- CoconutMotion 的 multi-pass forward 不兼容 TRL 的标准 `model.forward()` 调用
- 需要精确控制 latent 区域的 mask 和 KV cache
- 项目已有完善的训练基础设施（DDP、checkpoint、cosine LR）

---

## Step 1：新增 Reward 函数模块

**新建文件**：`rewards.py`

**复用来源**：`latent-reasoning-motion/rewards/unified_rewards.py`

实现三个 reward 函数（由简到复杂）：

### 1.1 Format Reward（格式正确性）
```
检查生成文本是否包含完整的 <Motion>...</Motion> 标签
- 有完整标签且内容非空 → 1.0
- 标签不完整 → 0.0 ~ 0.5（soft scoring）
- 完全无标签 → 0.0
```

### 1.2 Token Accuracy Reward（token 级匹配）
```
解析生成的 motion token 序列，与 ground truth 逐 token 比较
- reward = matching_tokens / max(len(pred), len(gt))
- 已有 utils.py:compute_motion_accuracy() 可直接复用
```

### 1.3 Embedding Similarity Reward（语义匹配）
```
使用 Motion-R1 的预训练 motion encoder 计算生成 motion 与 GT motion 的 embedding 距离
- 需加载 EvaluatorModelWrapper（来自 evaluate.py）
- reward = exp(-distance) 或 cosine_similarity
- 参考：latent-reasoning-motion/rewards/unified_rewards.py 的 similarity reward
```

**关键文件参考**：
- `utils.py:parse_motion_tokens()` (L57-63) — 从文本解析 motion code
- `utils.py:compute_motion_accuracy()` (L66-93) — token 级精度计算
- `evaluate.py` — FID/R-precision 评估基础设施

---

## Step 2：实现 GRPO 训练核心

**新建文件**：`train_grpo.py`

### 2.1 GRPO 训练循环伪代码

```
输入：SFT 训练好的 CoconutMotion 模型 (policy_model)
      冻结的 SFT 模型副本 (ref_model)
      训练数据（question + latent tokens + ground_truth motion）

For each epoch:
  For each batch of prompts:

    # === 1. 采样 G 个 completion ===
    completions = []
    for g in range(num_generations):  # G=4~8
        # 使用 CoconutMotion.generate() 采样（temperature > 0）
        generated = policy_model.generate(
            input_ids=prompt_with_latent_tokens,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        completions.append(generated)

    # === 2. 计算 Reward ===
    rewards = compute_rewards(completions, ground_truth)  # [B*G]

    # === 3. 计算 Group Advantage ===
    # 按 prompt 分组（每组 G 个 completion）
    grouped_rewards = rewards.reshape(B, G)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    group_std = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (grouped_rewards - group_mean) / group_std  # [B, G]
    advantages = advantages.reshape(B*G)

    # === 4. 计算 Policy Log Prob ===
    # 拼接 prompt + completion，通过 CoconutMotion.forward() 获取 logits
    full_input_ids = concat(prompt_with_latent, completion_tokens)
    labels = build_labels(full_input_ids)
    # latent区域=-100, prompt=-100, completion部分=token_ids

    outputs = policy_model.forward(full_input_ids, labels=labels)
    logits = outputs.logits

    # 提取 completion 区域的 per-token log prob
    per_token_logps = compute_log_probs(logits, labels)  # 仅 completion 区域

    # === 5. 计算 Reference Log Prob（KL 正则项）===
    with torch.no_grad():
        ref_outputs = ref_model.forward(full_input_ids, labels=labels)
        ref_per_token_logps = compute_log_probs(ref_outputs.logits, labels)

    # === 6. 计算 GRPO Loss ===
    # KL 散度项
    per_token_kl = exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1

    # Policy gradient 项（带 advantage 加权）
    ratio = exp(policy_logps - policy_logps.detach())
    per_token_loss = -(ratio * advantages.unsqueeze(1) - beta * per_token_kl)

    # 在 completion mask 上求均值
    loss = (per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)
    loss = loss.mean()

    # === 7. 反向传播 + 更新 ===
    loss.backward()
    clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

### 2.2 关键实现细节

**生成修改**：`CoconutMotion.generate()` 当前仅支持 greedy decoding（argmax）。需要增加：
- `temperature` 参数控制采样随机性
- `do_sample=True` 启用采样而非贪心
- `top_k` / `top_p` 可选截断

**修改文件**：`coconut_motion.py` 的 `generate()` 方法 (L356-483)
- 在 autoregressive loop (L415-446) 中，将 `argmax` 替换为 `torch.multinomial` 采样
- 增加 temperature scaling: `logits = logits / temperature`

**Log Prob 计算**：新增辅助函数
```python
def compute_per_token_log_probs(logits, labels):
    """从 logits 和 labels 计算 per-token log probability"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logps = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    # mask 掉 labels=-100 的位置
    mask = (shift_labels != -100).float()
    return per_token_logps * mask, mask
```

**Reference Model**：
- 加载 SFT 最终 checkpoint 的冻结副本
- 占用额外显存 → 可用 LoRA（仅训练 adapter，ref_model = base model 本身）

---

## Step 3：支持 LoRA 微调（降低显存）

**修改文件**：`train_grpo.py`

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
policy_model = get_peft_model(sft_model, lora_config)

# 获取 ref log probs：
with policy_model.disable_adapter():
    ref_outputs = policy_model.forward(...)
```

---

## Step 4：新增 GRPO 配置文件

**新建文件**：`configs/t2m_grpo.yaml`

```yaml
# === 基础配置（继承 SFT） ===
model_id: Qwen/Qwen2.5-3B-Instruct
sft_checkpoint: ./checkpoints/coconut_stage8_final/
bf16: true
coconut: true
max_latent_stage: 8
c_thought: 2

# === GRPO 特有配置 ===
mode: grpo
num_generations: 4
temperature: 0.7
top_p: 0.9

# RL 超参
beta: 0.01
lr: 5e-6
min_lr: 1e-6
warmup_steps: 50
num_epochs: 3
grad_clip: 1.0
max_new_tokens: 256

# LoRA
use_lora: true
lora_r: 16
lora_alpha: 32

# Reward 权重
reward_format_weight: 0.1
reward_accuracy_weight: 0.5
reward_embedding_weight: 0.4

# 数据
batch_size_training: 4
gradient_accumulation_steps: 16
```

---

## Step 5：修改 `coconut_motion.py` 支持采样生成

**修改文件**：`coconut_motion.py`

在 `generate()` 方法 (L356-483) 中增加 sampling 支持：

**当前代码** (L434):
```python
next_tokens = torch.argmax(next_logits, dim=-1)  # Greedy
```

**修改为**:
```python
if temperature > 0 and do_sample:
    next_logits = next_logits / temperature
    if top_p < 1.0:
        next_logits = top_p_filtering(next_logits, top_p)
    probs = F.softmax(next_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
else:
    next_tokens = torch.argmax(next_logits, dim=-1)
```

同时修改 `generate()` 签名增加参数：
```python
def generate(self, input_ids, attention_mask=None, max_new_tokens=256,
             temperature=0.0, do_sample=False, top_p=1.0, **kwargs):
```

---

## Step 6：集成与入口脚本

**新建文件**：`train_grpo.py`

主函数结构：
```
1. 解析 GRPO 配置（t2m_grpo.yaml）
2. 加载 SFT checkpoint → CoconutMotion
3. 应用 LoRA → policy_model
4. 初始化 reward 函数（format + accuracy + embedding）
5. 加载训练数据（复用 dataset.py 的 get_question_latent_dataset）
6. GRPO 训练循环（Step 2 的伪代码）
7. 每 N 步验证（复用 train.py 的 validate_generation）
8. 保存 checkpoint
```

**DDP 启动命令**：
```bash
torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml
```

---

## 文件变更总结

| 操作 | 文件 | 内容 |
|------|------|------|
| **新建** | `rewards.py` | format_reward, accuracy_reward, embedding_reward |
| **新建** | `train_grpo.py` | GRPO 训练主循环 |
| **新建** | `configs/t2m_grpo.yaml` | GRPO 超参配置 |
| **修改** | `coconut_motion.py` | generate() 增加 temperature/sampling 支持 |
| **不改** | `train.py` | SFT 训练不受影响 |
| **不改** | `dataset.py` | 数据加载可直接复用 |
| **不改** | `utils.py` | 工具函数可直接复用 |

---

## 复用的已有代码

| 来源 | 文件 | 复用内容 |
|------|------|---------|
| latent-cot-motion | `utils.py:parse_motion_tokens()` | 解析 motion code |
| latent-cot-motion | `utils.py:compute_motion_accuracy()` | token accuracy reward |
| latent-cot-motion | `dataset.py:get_question_latent_dataset()` | 构造 prompt+latent 输入 |
| latent-cot-motion | `evaluate.py:EvaluatorModelWrapper` | embedding similarity reward |
| latent-cot-motion | `coconut_motion.py:CoconutMotion.forward()` | log prob 计算 |
| latent-reasoning-motion | `rewards/unified_rewards.py` | reward 函数设计参考 |
| latent-reasoning-motion | `rewards/latent_cot_reward_utils.py` | motion 解析工具 |
| latent-reasoning-motion | `config.env` | GRPO 超参参考值 |
| HRPO repo | `trl/trainer/grpo_trainer.py` | GRPO loss 公式参考 |

---

## 验证方案

### 阶段性验证

1. **Reward 函数验证**：
   - 在 SFT 模型上运行 `rewards.py`，验证 reward 分布合理（均值 0.3-0.7，有区分度）
   - 对比 greedy 和 sampling 生成的 reward 差异

2. **GRPO 训练基础验证**：
   - 运行 1 个 epoch，检查 loss 下降且不 NaN
   - 检查 advantage 分布（应近似 N(0,1)）
   - 检查 KL divergence 不爆炸（保持 <0.1）

3. **端到端质量验证**：
   - GRPO 训练后运行 `evaluate.py`，对比 SFT-only baseline
   - 预期指标：FID 下降 5-15%，R-Precision Top-1 提升 2-5%
   - 运行 `inference.py` 可视化生成 motion，目测质量

### 运行命令

```bash
# Step 1: 验证 reward 函数
python -c "from rewards import *; test_rewards()"

# Step 2: GRPO 训练
torchrun --nproc_per_node=4 train_grpo.py --config configs/t2m_grpo.yaml

# Step 3: 评估
python evaluate.py --model_path checkpoints/grpo_final/ --config configs/t2m_grpo.yaml

# Step 4: 可视化
python inference.py --model_path checkpoints/grpo_final/ --text "a person walks forward slowly"
```

---

## 后续扩展（Phase 2+）

完成基础 GRPO 后可进一步探索：

1. **DAPO / DR-GRPO**：参考 latent-reasoning-motion 的 `loss_type` 参数，切换 GRPO 变体
2. **SofT-GRPO**：在 motion token 生成时使用 Gumbel-softmax（参考 repos/SofT-GRPO）
3. **Physics Reward**：加入关节角度/速度/滑步检测（参考 latent-reasoning-motion 的 physics reward）
4. **自适应 latent token 数量**：参考 TaH 论文，根据 prompt 难度动态分配
5. **Curriculum RL**：分阶段 RL（先大 epsilon 探索，再小 epsilon 收敛）
