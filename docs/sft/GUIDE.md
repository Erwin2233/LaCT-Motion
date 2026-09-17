# Latent-CoT-Motion 运行指南

Coconut 风格的 Latent SFT 训练系统，用于 Text-to-Motion（T2M）任务。在 SFT 阶段通过课程学习，逐步将显式思维链（`<think>` 文本）替换为连续隐状态（latent token），基于 Qwen 2.5-3B-Instruct。

## 项目结构

```
latent-cot-motion/
├── configs/
│   └── t2m_coconut.yaml    # 训练配置文件
├── data/                    # 训练数据目录（由 build_data.py 生成）
├── logs/                    # 训练日志目录（由 run.sh 生成）
├── checkpoints/             # 训练 checkpoint 输出目录
├── tests/                   # 回归测试
├── build_data.py            # 数据准备脚本（VQ-VAE 编码 + 思维链拼接）
├── coconut_motion.py        # Coconut 模型包装器（适配 Qwen 2.5）
├── dataset.py               # 数据集 + Collator（含课程学习逻辑）
├── train.py                 # 分布式训练主脚本（DDP/FSDP）
├── inference.py             # 推理脚本：文本 → motion token → VQ-VAE 解码 → 3D 可视化
├── evaluate.py              # 批量评估脚本：验证集 token 级指标
├── utils.py                 # 工具函数（Config、指标计算、motion token 解析）
├── run.sh                   # 一键训练启动脚本（含数据构建 + 分布式训练）
└── requirements.txt         # Python 依赖
```

## 1. 环境准备

```bash
cd latent-cot-motion
pip install -r requirements.txt
```

主要依赖：

| 依赖包 | 最低版本 | 用途 |
|--------|---------|------|
| torch | 2.1.0 | DDP/FSDP 分布式训练 |
| transformers | 4.40.0 | Qwen 2.5 模型加载、DynamicCache 支持 |
| datasets | 2.18.0 | 数据集 map/处理 |
| pyyaml | 6.0 | 配置文件解析 |
| numpy | 1.24.0 | 数值计算 |
| tqdm | 4.66.0 | 进度条 |
| wandb | 0.16.0 | 训练日志（可选） |

## 2. 数据准备

### 2.1 前置数据

训练数据通过 VQ-VAE 实时编码生成，需要以下文件：

| 数据来源 | 默认路径 | 说明 |
|---------|---------|------|
| 思维链步骤 | `UniMo-main/dataset/texts_think_steps.json` | `{caption: steps}` 格式 |
| HumanML3D 文本描述 | `Motion-R1_copy/data/humanml3d/texts/{ID}.txt` | 每行: `caption#pos_tags#f_tag#to_tag` |
| 原始动作数据 | `Motion-R1_copy/data/humanml3d/new_joint_vecs/{ID}.npy` | 263 维关节向量 |
| 数据集划分文件 | `Motion-R1_copy/data/humanml3d/{train,val,test}.txt` | 每行一个 motion ID |
| VQ-VAE 权重 | `Motion-R1_copy/ckpt/vqvae.pth` | 预训练 VQ-VAE checkpoint |
| 归一化参数 | `Motion-R1_copy/checkpoints/t2m/.../meta/{mean,std}.npy` | motion 归一化 mean/std |

路径均相对于 ``。

### 2.2 生成训练数据

```bash
python build_data.py --device cuda:0
```

使用自定义路径：

```bash
python build_data.py \
    --think-steps data/texts_think_steps.json \
    --texts-dir dataset/HumanML3D/texts \
    --motion-dir dataset/HumanML3D/new_joint_vecs \
    --splits-dir dataset/HumanML3D \
    --vqvae-path ckpt/vqvae.pth \
    --meta-dir checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta \
    --output-dir ./data \
    --max-steps 20 \
    --min-motion-len 40 \
    --max-motion-len 200 \
    --splits train,val,test \
    --device cuda:0
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--think-steps` | `.../UniMo-main/dataset/texts_think_steps.json` | 思维链步骤文件 |
| `--texts-dir` | `.../Motion-R1_copy/data/humanml3d/texts` | HumanML3D 文本描述目录 |
| `--motion-dir` | `.../Motion-R1_copy/data/humanml3d/new_joint_vecs` | 原始动作 `.npy` 文件目录 |
| `--splits-dir` | `.../Motion-R1_copy/data/humanml3d` | 含 `{split}.txt` 划分文件的目录 |
| `--vqvae-path` | `.../Motion-R1_copy/ckpt/vqvae.pth` | VQ-VAE checkpoint 路径 |
| `--meta-dir` | `.../Motion-R1_copy/.../meta` | 含 `mean.npy`、`std.npy` 的归一化目录 |
| `--output-dir` | `./data` | 输出目录 |
| `--max-steps` | 20 | 每个样本最大思维步数 |
| `--min-motion-len` | 40 | 最短动作帧数 |
| `--max-motion-len` | 200 | 最长动作帧数 |
| `--splits` | `train,val,test` | 要处理的数据集划分 |
| `--device` | `cuda:0` | VQ-VAE 编码使用的设备 |

运行后将在 `data/` 下生成：

```
data/
├── t2m_train.json
├── t2m_val.json
└── t2m_test.json
```

### 2.3 数据格式

每条样本的 JSON 结构：

```json
{
  "question": "<|im_start|>system\nYou are an assistant...<|im_end|>\n<|im_start|>user\n### Input:\na person walks forward<|im_end|>\n<|im_start|>assistant\n",
  "steps": ["First, the person shifts weight...", "Then, they begin stepping..."],
  "answer": "<Motion><Motion_86><Motion_301>...</Motion><|im_end|>"
}
```

- `question`：Qwen ChatML 格式的对话模板（system + user + assistant 前缀）
- `steps`：思维链分步推理，训练过程中会被逐步替换为 latent token
- `answer`：VQ-VAE 编码后的 motion token 序列，包裹在 `<Motion>...</Motion>` 标签中

## 3. 配置说明

配置文件位于 `configs/t2m_coconut.yaml`，关键参数分组如下：

### 3.1 模型配置

```yaml
model_id: Qwen/Qwen2.5-3B-Instruct   # 基座模型
bf16: true                              # BFloat16 训练
nb_code: 512                            # motion token 词表大小 (Motion_0 ~ Motion_511)
```

### 3.2 训练模式

```yaml
coconut: true    # Coconut latent 课程训练（默认）
cot: false       # 纯 CoT SFT（始终 stage 0，不替换 latent）
no_cot: false    # 无推理（跳过所有思维步骤）
```

三种模式互斥：
- **coconut=true**：课程学习，逐步用 latent token 替换思维步骤
- **cot=true**：传统显式 CoT SFT，保留全部 `<think>` 文本，不使用 latent token
- **no_cot=true**：跳过思维链，输入中不包含 `<think>` 与 latent 区域，直接从 question 预测 answer

注意：`uniform_prob` 随机 stage 采样仅在 `coconut` 模式下生效，`cot` 和 `no_cot` 模式不受影响。

### 3.3 课程调度

```yaml
c_thought: 2           # 每个被替换的思维步骤对应的 latent token 数量
epochs_per_stage: 3    # 每个 stage 持续的 epoch 数
max_latent_stage: 8    # 最大 stage 数（最多替换 8 个步骤）
pad_latent_to_max: true  # 超过 max_stage 后是否补齐 latent 到最大数量
uniform_prob: 0.1      # 随机采样 stage 的概率（仅 coconut 模式）
```

课程调度公式：`scheduled_stage = epoch // epochs_per_stage`

| Epoch | Stage | Latent Tokens | 可见思维步骤 | 训练目标 |
|-------|-------|---------------|-------------|---------|
| 0-2 | 0 | 0 | 全部 | `<think>step1...stepN</think> + motion` |
| 3-5 | 1 | 2 | step2~N | `remaining steps + motion` |
| 6-8 | 2 | 4 | step3~N | `remaining steps + motion` |
| 9-11 | 3 | 6 | step4~N | `remaining steps + motion` |
| ... | ... | ... | ... | ... |
| 21-23 | 7 | 14 | step8~N | `remaining steps + motion` |
| 24-29 | 8+ | 16 | 无 | `motion tokens only` |

### 3.4 训练超参

```yaml
num_epochs: 30
batch_size_training: 16                 # 单卡 batch size
gradient_accumulation_steps: 8          # 有效 batch size = 16 * 8 * GPU数
lr: 1.0e-4
weight_decay: 0.01
grad_clip: 1.0                          # 梯度裁剪最大范数（防止 bf16 下 NaN）
warmup_steps: 200
max_seq_len: 512
max_think_steps: 20                     # tokenize 时截断的最大思维步数
use_fsdp: false                         # false=DDP，true=FSDP（3B 模型推荐 DDP）
```

- `warmup_steps`：线性 warmup 的更新步数（按 optimizer step 计数）
- `max_seq_len`：训练/验证样本在数据构造阶段会被截断到该长度
- `use_fsdp`：默认 `false`，使用 DDP。FSDP 在 latent token 数量随 batch 变化时可能导致 NCCL 死锁，3B 模型推荐 DDP

### 3.5 验证配置

```yaml
val_batch_size: 32                # 验证 batch size（loss 计算与生成评估共用）
val_loss_every_n_epochs: 1        # 每 N 个 epoch 计算一次验证 loss（0=禁用）
val_gen_every_n_epochs: 3         # 每 N 个 epoch 运行一次生成评估（0=禁用）
val_gen_max_samples: 50           # 每张卡生成评估的最大样本数
val_max_samples: 100              # 每轮验证载入的最大样本数
val_start_epoch: 0                # 从哪个 epoch 开始验证
```

- 验证 loss 和生成评估独立触发，按各自频率运行
- 生成评估使用批量生成（`CoconutMotion.generate()` 支持任意 batch size）
- 最后一个 epoch 始终运行生成评估

### 3.6 Checkpoint 与恢复

```yaml
save_path: ./checkpoints
save_only_improve: false    # true: 仅在 val loss 下降时保存
resume: 0                   # 从第几个 epoch 恢复（0 = 从头训练）
load_model_path: "None"     # 可填 checkpoint 目录、model_state.pt 或 training_state.pt
reset_optimizer: true       # 每个 stage 重置 optimizer
```

- 若 `load_model_path` 指向 checkpoint 且 `resume=0`，程序会自动推断起始 epoch：
  - 训练模式：从已保存 epoch 的下一轮继续
  - `only_eval=true`：在该 checkpoint 对应的 stage 上评估一次
- 若存在 `training_state.pt` 且包含 `optimizer_state_dict`，会尝试恢复优化器状态。

### 3.7 评估

```yaml
only_eval: false        # true: 只做评估，不训练
max_new_tokens: 256     # 生成时最大新 token 数
```

## 4. 训练

### 4.1 一键启动（推荐）

```bash
setsid bash run.sh      # 后台运行，断开终端不中断
tail -f logs/train_*.log # 查看实时日志
```

`run.sh` 会自动完成：
1. 检查数据是否已构建，未构建则先运行 `build_data.py`
2. 使用 `torchrun` 启动分布式训练
3. 日志输出到 `logs/train_{timestamp}.log`

端口策略：默认随机分配 29500-39500 范围端口，可通过环境变量覆盖：

```bash
MASTER_PORT=29600 bash run.sh
```

### 4.2 手动分布式训练

```bash
# DDP 多卡训练（默认配置）
torchrun --nproc_per_node=2 train.py configs/t2m_coconut.yaml
```

- 默认使用 DDP（`use_fsdp: false`）
- 有效 batch size = `batch_size_training * gradient_accumulation_steps * nproc_per_node`
- 默认配置下：16 * 8 * 2 = 256

### 4.3 单卡训练

`torchrun` 仍然需要，因为代码依赖 `dist.init_process_group`：

```bash
torchrun --nproc_per_node=1 train.py configs/t2m_coconut.yaml
```

### 4.4 多机训练

```bash
# 每台机器上分别运行
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py configs/t2m_coconut.yaml
```

### 4.5 训练日志

训练过程中会打印：

```
============================================================
Epoch 3 | Stage 1/8
  Latent tokens: 2
============================================================
Epoch 3: 100%|██████████| 250/250 [05:30<00:00, loss=2.3456]
  Train loss: 2.3456
  Validating...
  Val loss: 2.5678
  Gen exact_match: 0.0200
  Gen token_acc: 0.1500
  Saved checkpoint: ./checkpoints/checkpoint-epoch3
```

如果安装了 `wandb`，训练指标会自动上传到 W&B。项目名称和运行名称由配置文件中的 `project` 和 `name` 字段控制。

### 4.6 恢复训练

推荐方式：同时指定 `load_model_path`（权重来源）与可选 `resume`（手工覆盖起始 epoch）。

```yaml
load_model_path: ./checkpoints/checkpoint-epoch9
resume: 0    # 设为 0 时自动从 epoch 10 继续
```

然后正常启动训练命令。
如果你显式设置 `resume` 为非 0，会优先使用该值。

## 5. Checkpoint 输出

每个 epoch 结束后保存至 `checkpoints/checkpoint-epoch{N}/`。

DDP（默认）会保存 HuggingFace 格式：

```
checkpoints/checkpoint-epoch3/
├── config.json              # 模型配置
├── model.safetensors        # 模型权重（base_causallm 部分）
├── tokenizer.json           # Tokenizer（含所有新增 token）
├── tokenizer_config.json
├── special_tokens_map.json
└── training_state.pt        # 训练状态（epoch, optimizer, loss, stage, global_step）
```

FSDP（`use_fsdp: true`）会额外保存：

```
checkpoints/checkpoint-epoch3/
├── model_state.pt           # FSDP 聚合后的完整 state_dict
└── training_state.pt        # 训练状态（含 global_step）
```

注意：保存的是 `base_causallm`（Qwen 2.5-3B）或其等价完整 state_dict。`CoconutMotion` 包装层无额外参数，加载时会自动重新包装。

## 6. 推理

使用 `inference.py` 从文本生成 3D 动作：

```bash
# 单条文本
python inference.py configs/t2m_coconut.yaml \
    --checkpoint checkpoints/checkpoint-epoch29 \
    --text "a person walks forward" \
    --output results/

# 批量文本（每行一条）
python inference.py configs/t2m_coconut.yaml \
    --checkpoint checkpoints/checkpoint-epoch29 \
    --text-file prompts.txt \
    --output results/
```

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `config_file` | 必填 | YAML 配置文件路径 |
| `--checkpoint` | 必填 | 训练好的 checkpoint 目录 |
| `--text` | - | 单条文本提示 |
| `--text-file` | - | 文本文件（每行一条提示） |
| `--output` | `results/` | 输出目录 |
| `--vqvae-path` | `Motion-R1_copy/ckpt/vqvae.pth` | VQ-VAE checkpoint |
| `--meta-dir` | `Motion-R1_copy/.../meta` | mean/std 归一化目录 |
| `--device` | `cuda:0` | 推理设备 |
| `--seed` | 42 | 随机种子 |

输出文件（每条提示生成）：

```
results/
├── 0000_joints.npy      # 3D 关节坐标 (T, 22, 3)
├── 0000_features.npy    # 263 维动作特征 (T, 263)
├── 0000_meta.json       # 元信息（文本、codes、帧数）
└── 0000_motion.mp4      # 3D 骨骼动画
```

推理流程：`text → build_question → tokenize + latent tokens → CoconutMotion.generate() → parse motion codes → VQ-VAE decode → 反归一化 → recover_from_ric → 3D joints → MP4`

注意：推理时自动根据训练模式决定输入构造——`coconut` 模式注入最大数量的 latent token，`cot`/`no_cot` 模式不注入 latent token。

## 7. 批量评估

使用 `evaluate.py` 在验证集上计算 token 级别指标：

```bash
python evaluate.py configs/t2m_coconut.yaml \
    --checkpoint checkpoints/checkpoint-epoch29 \
    --max-samples 100

# 保存结果到 JSON
python evaluate.py configs/t2m_coconut.yaml \
    --checkpoint checkpoints/checkpoint-epoch29 \
    --output eval_results.json
```

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `config_file` | 必填 | YAML 配置文件路径 |
| `--checkpoint` | 必填 | 训练好的 checkpoint 目录 |
| `--max-samples` | 全部 | 限制评估样本数 |
| `--output` | - | 保存结果 JSON 路径 |
| `--device` | `cuda:0` | 评估设备 |
| `--seed` | 42 | 随机种子 |

输出指标：

| 指标 | 说明 |
|------|------|
| `exact_match` | motion token 序列完全匹配率 |
| `token_accuracy` | 重叠前缀中匹配位置数 / GT 长度 |
| `avg_length_ratio` | 生成长度 / GT 长度 |
| `avg_pred_length` | 平均生成 token 数 |
| `n_failures` | 生成失败次数 |
| `n_empty` | 空输出次数 |

注意：评估时同样根据训练模式自动切换输入模板。

## 8. 评估模式（train.py）

仅运行评估，不训练：

```yaml
only_eval: true
load_model_path: ./checkpoints/checkpoint-epoch29
```

```bash
torchrun --nproc_per_node=1 train.py configs/t2m_coconut.yaml
```

`only_eval=true` 时：只运行一次完整评估（不会进入训练，也不会保存 checkpoint）。
训练模式下：验证 loss 每 `val_loss_every_n_epochs` 个 epoch 运行一次，生成评估每 `val_gen_every_n_epochs` 个 epoch 运行一次，最后一个 epoch 始终运行生成评估。

## 9. 核心原理

### 9.1 Coconut 多轮前向传播

`CoconutMotion.forward()` 对每个 `<|latent|>` token 执行一轮独立的前向传播：

1. 前向传播到当前 latent token 位置
2. 提取前一位置的最后一层 hidden state
3. 用该 hidden state 替换当前 latent token 的 embedding
4. 复用 KV cache 避免重复计算已处理的 token

最终将所有 pass 的 logits 拼接，计算交叉熵损失。

### 9.2 Token 序列示意

**Stage 0**（纯 CoT）：
```
[question] <think>\n step1\n step2\n ... stepN\n \n</think> <Motion>...<Motion_86>...</Motion><|im_end|>
```

**Stage k**（部分 latent）：
```
[question] <|start-latent|> <|latent|>*k*c <|end-latent|> <think>\n step(k+1)\n ... stepN\n \n</think> <Motion>...<|im_end|>
```

**Stage > max**（全 latent）：
```
[question] <|start-latent|> <|latent|>*max*c <|end-latent|> <Motion>...<|im_end|>
```

### 9.3 损失掩码

- `question` + `<|start-latent|>` + `<|latent|>` 区域 + `<|end-latent|>` → label = -100（不计算损失）
- 剩余思维步骤 + answer（motion tokens）→ 正常计算交叉熵

### 9.4 Batch 对齐

`MotionCollator` 通过左填充对齐 batch 内所有样本的第一个 `<|latent|>` 位置，使 latent token 在 batch 维度上对齐，最大化 KV cache 复用：

```
pad pad pad xxxxx <|latent|> <|latent|> xxxxx pad pad
pad pad pad pad x <|latent|> xxxxxxxx  pad pad pad pad
pad xxxxx xxxxx x <|latent|> <|latent|> xxxxxxx pad
```

### 9.5 批量生成

`CoconutMotion.generate()` 支持任意 batch size 的批量生成：
- 对左填充的 batch 正确处理 `position_ids`（从 `attention_mask` 计算累积位置）
- 逐步贪心解码，通过 `done` 掩码追踪各样本的 EOS 状态
- 批量 GPU→CPU 转移（每步一次 `.cpu().tolist()`，避免逐元素同步）

## 10. 常见问题

**Q: 显存不足怎么办？**

- 减小 `batch_size_training`（如 8 或 4）
- 增大 `gradient_accumulation_steps` 以保持有效 batch size
- 减小 `max_seq_len`（如 384）
- 确认 `bf16: true` 已开启

**Q: 如何切换到纯 CoT SFT 训练（不使用 latent）？**

```yaml
coconut: false
cot: true
no_cot: false
```

此模式下始终保持 stage 0，所有思维步骤以 `<think>` 文本形式保留。推理和评估时也不会注入 latent token。

**Q: 如何调整课程进度？**

- 加快课程：减小 `epochs_per_stage`（如 2）
- 减慢课程：增大 `epochs_per_stage`（如 5）
- 更多 latent：增大 `max_latent_stage` 和 `c_thought`
- 更平滑的过渡：增大 `uniform_prob`（如 0.3），让更多样本随机采样 stage

**Q: 端口冲突怎么办？**

`run.sh` 默认随机分配端口。如果仍然冲突，可手动指定：

```bash
MASTER_PORT=30123 bash run.sh
```

或手动启动时指定：

```bash
torchrun --nproc_per_node=2 --master_port=30123 train.py configs/t2m_coconut.yaml
```

**Q: `build_data.py` 报匹配数量太少？**

检查数据源的关联：`build_data.py` 通过 `caption` 文本匹配思维链和 HumanML3D texts，拼写差异会导致匹配失败。运行时会打印 stats 统计各类未匹配的原因（`no_think_steps`、`no_motion_tokens`、`too_few_steps`）。

**Q: 训练后如何用于下游 GRPO？**

保存的 checkpoint 是标准 Qwen 2.5 格式 + 扩展词表。可直接作为 `latent-reasoning-motion` 项目的初始化模型，该项目会在 GRPO 阶段进一步微调 latent reasoning。

**Q: tokenizer fork 警告怎么处理？**

`run.sh` 已设置 `TOKENIZERS_PARALLELISM=false`。如果手动启动训练，请在命令前添加：

```bash
TOKENIZERS_PARALLELISM=false torchrun ...
```
