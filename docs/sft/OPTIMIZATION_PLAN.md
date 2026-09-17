# CoconutMotion.generate() 推理速度优化方案

## 1. 问题分析

当前 `generate()` 方法（`coconut_motion.py:245-373`）存在严重的 double prefill 问题：

```
当前流程:
  self.forward(input_ids, ...)           # 第1次: multi-pass 前向, 遍历整个序列
      ├── pass 0: [0, first_latent)      #   计算 question prefix
      ├── pass 1: [latent_0]             #   替换第1个 latent embedding
      ├── ...                            #   替换后续 latent embeddings
      └── final pass: [last_latent+1, end) #   计算剩余 token (use_cache=False !)
                                         #   ← KV cache 被丢弃
  self.base_causallm(inputs_embeds, ...) # 第2次: 对整个序列重新做完整前向
                                         #   ← 仅仅为了拿到 KV cache
  greedy decode loop ...                 # 自回归生成
```

**根因**：`forward()` 的 final pass 设置了 `use_cache=False`（第228行），不返回 KV cache。`generate()` 不得不对完整序列重新做一次前向传播来获取可用于自回归解码的 KV cache。

**关键洞察**：multi-pass forward 中逐步构建的 KV cache 与对替换后的完整序列做一次性前向传播所得的 KV cache 在数学上是等价的。原因如下：

- pass 0 计算 `[q0, q1, ..., start]` 的 KV，与 latent token 无关
- pass 1 用 pass 0 的 KV cache 计算 `lat0_replaced` 的 KV，等价于单次前向中 `lat0_replaced` attend to 前缀
- 以此类推，每个 latent pass 追加的 KV 都与单次前向等价
- final pass 用累积的 KV cache 计算剩余 token，与单次前向等价

因此第二次 prefill 完全冗余。

### 量化影响

以默认配置为例（`|q|≈70, M=16, M_a≈37.5`）：

| 阶段 | 当前实现 (前向次数) | 优化后 (前向次数) |
|------|-------------------|------------------|
| Multi-pass latent reasoning | ~17 steps | ~17 steps |
| **冗余 full prefill** | **~86 steps** | **0** |
| Autoregressive decoding | ~37.5 steps | ~37.5 steps |
| **总计** | **~140.5** | **~54.5** |

仅消除 double prefill，prefill+reasoning 阶段即可获得 **~2.6x** 加速。

---

## 2. 优化方案

### 优化 1（核心）：消除 double prefill

**思路**：新增一个专用于推理的 `_forward_latent_for_generation()` 方法，在 multi-pass 过程中直接积累可复用的 KV cache，最终连同 logits 一起返回。不修改训练用的 `forward()` 方法，避免影响梯度计算。

**实现要点**：

1. 在 final pass 中设置 `use_cache=True`，让模型返回完整 KV cache
2. 返回 `(past_key_values, last_logits, inputs_embeds)` 三元组
3. `generate()` 直接使用该 KV cache 进入自回归解码
4. 推理路径使用 `torch.no_grad()` / `torch.inference_mode()`，无需关心梯度

**改动位置**：`coconut_motion.py`

```python
def _forward_latent_for_generation(self, input_ids, attention_mask, position_ids):
    """Inference-only multi-pass forward that returns KV cache.

    Unlike forward(), this method:
    - Does not compute loss
    - Sets use_cache=True on the final pass to return reusable KV cache
    - Uses in-place embedding replacement (safe under torch.no_grad())
    - Returns (past_key_values, last_logits, inputs_embeds)
    """
    latent_indices = (input_ids == self.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max((len(lst) for lst in latent_lists), default=0)

    inputs_embeds = self.embedding(input_ids)
    next_compute_range = (0, input_ids.shape[1])

    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())

    kv_cache = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_states_offset = 0
        else:
            past_key_values = _slice_kv_cache(kv_cache, next_compute_range[0])
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            hidden_states_offset = next_compute_range[0]

        next_compute_range = (
            next_compute_range[1],
            input_ids.shape[1] if pass_idx + 1 >= max_n_latents
            else next_compute_range[1] + 1,
        )

        hidden_states = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        # In-place embedding replacement (safe under no_grad)
        for instance_idx, mask_list in enumerate(latent_lists):
            if len(mask_list) > pass_idx:
                token_idx = mask_list[pass_idx]
                inputs_embeds[instance_idx, token_idx, :] = hidden_states[
                    instance_idx, token_idx - 1 - hidden_states_offset, :
                ]

    self.gen_forward_cnt += max_n_latents

    # ---- Final pass: use_cache=True to return KV cache ----
    final_past_kv = None
    if kv_cache is not None:
        final_past_kv = _slice_kv_cache(kv_cache, next_compute_range[0])

    outputs = self.base_causallm(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=(
            attention_mask[:, :next_compute_range[1]]
            if kv_cache is not None
            else attention_mask[:, next_compute_range[0]:next_compute_range[1]]
        ),
        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
        past_key_values=final_past_kv,
        output_hidden_states=False,
        use_cache=True,   # <--- 关键改动
    )
    self.gen_forward_cnt += 1

    return outputs.past_key_values, outputs.logits, inputs_embeds
```

**改造后的 `generate()` 流程**：

```python
def generate(self, input_ids, attention_mask, position_ids=None, ...):
    ...
    # 单次 multi-pass forward，直接拿到 KV cache
    past_key_values, last_logits, inputs_embeds = \
        self._forward_latent_for_generation(input_ids, attention_mask, position_ids)

    # 直接从 last_logits 取首个 token，无需再次 prefill
    next_tokens = torch.argmax(last_logits[:, -1, :], dim=-1)

    # 进入自回归解码循环 ...
```

### 优化 2：消除 Python 级 tensor 拆组

**现状**（`coconut_motion.py:183-204`，训练路径）：

```python
# O(B * S) 的 Python 循环，每个 latent pass 都执行
tensor_list = [
    [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
    for batch_idx in range(inputs_embeds.shape[0])
]
for idx_pair in filling_indices:
    batch_idx, token_idx = idx_pair
    tensor_list[batch_idx][token_idx] = hidden_states[...]
inputs_embeds = torch.stack([torch.stack(t) for t in tensor_list])
```

对于 `B=16, S=512`，每 pass 需要 8192 次 Python 逐元素 tensor 操作。

**优化方案**：在**推理路径**中直接用 in-place 赋值替代（如优化 1 中已包含）。在**训练路径**中用 `clone()` + 高级索引赋值替代 Python 循环：

```python
# 训练路径：用 clone + 向量化索引替代 Python 循环
new_embeds = inputs_embeds.clone()
if filling_indices:
    batch_idxs = torch.tensor([p[0] for p in filling_indices], device=device)
    token_idxs = torch.tensor([p[1] for p in filling_indices], device=device)
    source_idxs = token_idxs - 1 - hidden_states_offset
    new_embeds[batch_idxs, token_idxs] = hidden_states[batch_idxs, source_idxs]
inputs_embeds = new_embeds
```

`clone()` 创建新 tensor，确保计算图正确；高级索引赋值是 O(M) 操作而非 O(B*S)。

### 优化 3：预分配 decode attention mask

**现状**（`coconut_motion.py:317-320`）：

```python
# 每个 decode step 都做一次 cat 扩展
decode_attn = torch.cat([
    decode_attn,
    torch.ones(batch_size, 1, dtype=torch.long, device=device),
], dim=1)
```

`max_new_tokens=256` 次迭代会产生 256 次 `torch.cat`，每次分配新内存并复制。

**优化方案**：预分配最大长度的 attention mask：

```python
# 预分配
max_total_len = attention_mask.shape[1] + max_new_tokens
decode_attn = torch.zeros(batch_size, max_total_len, dtype=torch.long, device=device)
decode_attn[:, :attention_mask.shape[1]] = attention_mask
decode_len = attention_mask.shape[1]

# 解码循环内：
decode_attn[:, decode_len] = 1
decode_len += 1
# 传给模型时切片：decode_attn[:, :decode_len]
```

---

## 3. 前后对比

### 推理流程对比

```
优化前:
  ┌─ forward() ──────────────────────────────────────┐
  │  pass 0: question prefix          (~70 tokens)   │
  │  pass 1..M: latent tokens         (16 passes)    │
  │  final pass: remaining tokens     (~0 tokens)    │  use_cache=False
  │  → 返回 inputs_embeds (KV cache 丢弃)            │
  └──────────────────────────────────────────────────┘
  ┌─ 冗余 prefill ───────────────────────────────────┐
  │  完整前向: 全部 ~86 tokens        (1 pass)        │  use_cache=True
  │  → 返回 KV cache                                 │
  └──────────────────────────────────────────────────┘
  ┌─ 自回归解码 ─────────────────────────────────────┐
  │  decode: ~37.5 tokens             (37.5 passes)  │
  └──────────────────────────────────────────────────┘
  总前向 pass 数: 17 + 86 + 37.5 ≈ 140.5

优化后:
  ┌─ _forward_latent_for_generation() ───────────────┐
  │  pass 0: question prefix          (~70 tokens)   │
  │  pass 1..M: latent tokens         (16 passes)    │
  │  final pass: remaining tokens     (~0 tokens)    │  use_cache=True
  │  → 返回 KV cache + logits                        │
  └──────────────────────────────────────────────────┘
  ┌─ 自回归解码 ─────────────────────────────────────┐
  │  decode: ~37.5 tokens             (37.5 passes)  │
  └──────────────────────────────────────────────────┘
  总前向 pass 数: 17 + 37.5 ≈ 54.5
```

### 各优化项预期收益

| 优化项 | 描述 | 预期加速 | 影响范围 |
|--------|------|---------|---------|
| **优化 1** | 消除 double prefill | **~2.6x**（推理端到端） | `generate()` 推理 |
| **优化 2** | 向量化 embedding 替换 | ~1.1-1.2x（latent pass 阶段） | `forward()` 训练 + 推理 |
| **优化 3** | 预分配 decode attention mask | ~1.05x（decode 阶段） | `generate()` 推理 |

### 对比 Explicit CoT 的理论加速

优化后的 latent CoT vs explicit CoT：

| | Explicit CoT | Latent CoT (优化后) | 加速比 |
|--|-------------|-------------------|--------|
| Prefill | ~70 | ~70 (pass 0) | 1x |
| Reasoning | ~63 (自回归) | ~17 (multi-pass) | **3.7x** |
| Motion decode | ~37.5 | ~37.5 | 1x |
| **总计** | **~170.5** | **~124.5** | **1.37x** |

优化后端到端加速从当前的 ~0.8-0.9x（反而更慢）恢复为理论值 ~1.37x。

---

## 4. 改动文件清单

| 文件 | 改动 |
|------|------|
| `coconut_motion.py` | 新增 `_forward_latent_for_generation()` 方法；重写 `generate()` 方法；训练 `forward()` 中的 tensor 拆组改为 `clone()` + 向量化索引 |
| `tests/test_regressions.py` | 新增 `generate()` 输出一致性测试（优化前后输出 token 相同） |

不涉及的文件：`train.py`、`dataset.py`、`inference.py`、`evaluate.py` —— 它们只调用 `model.forward()` 或 `model.generate()` 的公开接口，接口签名不变。

## 5. 验证方法

1. **输出一致性**：对同一输入，优化前后 `generate()` 产出的 token 序列必须完全相同（贪心解码是确定性的）
2. **训练 loss 一致性**：`forward()` 训练路径的 loss 在 `clone()` 替换前后必须一致（float 精度误差 < 1e-5）
3. **性能对比**：对 100 条验证集样本计时，对比优化前后的端到端推理耗时

## 6. 风险与注意事项

- **训练路径不受影响**：`forward()` 方法的签名和返回值不变，仅内部实现从 Python 循环改为向量化操作，`clone()` 保证计算图正确性
- **`output_embedding` 参数**：当前 `generate()` 支持 `output_embedding=True` 返回 `inputs_embeds`，优化后需保留此功能
- **FSDP `synced_gpus`**：dummy forward 逻辑需保留，确保多 GPU 同步正确
- **无 latent token 场景**：当 `cot=true` 或 `no_cot=true` 时 latent token 数为 0，multi-pass 循环不执行，直接走 final pass，此路径需要验证 KV cache 正确返回
