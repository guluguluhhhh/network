# Transformer 模型 Profiling 分析报告

**模型配置**: `Transformer(num_of_layer=1, max_seq_len=8192)`, FP16, seq_len=4455  
**GPU**: H800 (95 GB)  
**Profiling 工具**: `torch.profiler` + TensorBoard PyTorch Profiler 插件  
**日期**: 2026-06-09

---

## 1. GPU Kernel 耗时分析（Device Self Time）

> 数据来源: TensorBoard — Kernel View / Total Time 饼图

这是 GPU 上**实际执行**的 kernel 耗时分布，反映真实计算瓶颈：

| 排名 | CUDA Kernel | 占比 | 对应 PyTorch 算子 | 说明 |
|------|-------------|------|-------------------|------|
| 1 | `nvjet_hsh_256x160_64...` | **37.4%** | `aten::matmul` | Q @ K^T attention score |
| 2 | `vectorized_gather_kernel` | **29.1%** | `aten::index` | MoE 权重 gather（纯内存拷贝） |
| 3 | `nvjet_hsh_128x8_64x1...` | **13.2%** | `aten::matmul` | attn_weights @ V |
| 4 | `nvjet_hsh_256x8_64x6...` | **6.7%** | `aten::mm` | Linear projection |
| 5 | `cutlass::Kernel2` × 2 | ~7% | `aten::mm` | Linear（CUTLASS 路径） |
| 6 | `cunn_SoftMaxForward` | ~4.5% | `aten::_softmax` | attention + router softmax |
| 7 | `elementwise_kernel` 系列 | ~2% | `aten::mul` 等 | 各处 elementwise 运算 |

**Self CUDA time total: 19.50ms**

<!-- GPU Kernel 耗时饼图 -->

### 关键发现

1. **前两名占 66.5%**：朴素 attention matmul（37.4%）和 MoE weight gather（29.1%）吃掉了近三分之二的 GPU 时间。

2. **MoE weight gather 不是计算，是内存搬运**：`vectorized_gather_kernel` 只做数据拷贝，没有任何有效计算，却消耗了 29.1% 的 GPU 时间。

3. **Attention 未使用 FlashAttention**：手动 `matmul → softmax → matmul` 路径，Q @ K^T 单个 kernel 就占 37.4%，生成了 `[8, 4455, 4455]` 约 303MB 的 attention score 矩阵。

---

## 2. Tensor Core 利用率

> 数据来源: TensorBoard — Tensor Cores Utilization 饼图

| 指标 | 数值 |
|------|------|
| Using Tensor Cores | **6.9%** |
| Not Using Tensor Cores | **93.1%** |

<!-- Tensor Core 利用率饼图 -->

Tensor Core 几乎完全闲置，原因：

- `vectorized_gather_kernel`（29.1%）是纯内存操作，不涉及计算
- `aten::bmm` 的 MoE 部分 M=1，矩阵太小无法利用 Tensor Core 的 tile 大小
- 模型维度偏小（head_dim=64, embed_dim=512），GEMM 形状不利于 Tensor Core 发挥

---

## 3. CPU 端调度开销（Host Time）

> 数据来源: TensorBoard — Host Self Time / Host Total Time 饼图
>
> 注意: Host Time 只反映 CPU 侧的调度和内存分配开销，**不包含 GPU 计算时间**（PyTorch 异步提交 kernel）。唯一例外是 `cudaMalloc`，它是同步阻塞的。

### Host Self Time 分布

| 算子 | 占比 | 说明 |
|------|------|------|
| `aten::arange` | **31.5%** | CPU 密集的序列索引生成 |
| `aten::index` | **28.7%** | 主要耗时在触发 `cudaMalloc` 分配 3 × 4.35GB 显存（同步阻塞） |
| `aten::mm` | 10.3% | 参数准备 + `cudaLaunchKernel` dispatch |
| `aten::mul` | 7.9% | 17 次调用的 dispatch 累积 |
| `aten::bmm` | 5.9% | dispatch |
| 其他 | 15.7% | pow、resize\_、copy\_、empty_strided、add\_ 等 |

<!-- Host Self Time 饼图 + Host Total Time 饼图 -->

### Host Total Time 分布

| 算子 | 占比 | 说明 |
|------|------|------|
| `aten::arange` | **36.5%** | 含子调用 |
| `aten::index` | 16.6% | |
| `aten::rms_norm` | 10.6% | 含内部的 pow、mean、rsqrt 等 |
| `aten::matmul` | 8.9% | 含内部 `aten::mm` |
| `aten::linear` | 6.5% | 含内部 `aten::mm` |
| `aten::mm` | 5.9% | 叶子算子，Total ≈ Self |
| 其他 | 15% | mul、embedding、bmm、pow 等 |

### Host Self Time vs Host Total Time

- **Self Time**：算子**自身**在 CPU 上花的时间，不含子算子。各算子加起来 = 100%，适合定位 CPU 瓶颈。
- **Total Time**：含子调用的 CPU 总耗时，存在重叠计数。例如 `aten::linear`（6.5%）的 Total 包含了内部 `aten::mm` 的时间，适合看高层算子的整体开销。

---

## 4. 显存占用分析（Memory View）

> 数据来源: TensorBoard — Memory View + module hook 数据

### 4.1 总览

| 指标 | 数值 |
|------|------|
| 模型权重 + 输入 (Base) | 496 MB |
| **Allocated 峰值** | **16,251.7 MB** |
| **激活显存** (Peak - Base) | **15,756 MB** |
| 激活占总峰值比例 | **97%** |

### 4.2 显存时间线

<!-- Memory View 曲线图 -->

从 Memory View 曲线可以看到，显存从 ~496 MB 基线开始，在 5-9ms 之间阶梯式暴涨至峰值。MoE 的 3 次 weight gather 是显存的绝对主力：

| 阶段 | 显存变化 | 事件 |
|------|---------|------|
| 0-5ms | ~496 MB 平稳 | 模型权重 + 输入（基线） |
| ~5ms | 小幅上升 → ~1 GB | Attention QKV proj + score 矩阵（+347 MB） |
| **6.63ms** | **阶梯 #1 → ~5.5 GB** | `aten::index`: gather w_gate（**+4,563 MB**） |
| **7.23ms** | **阶梯 #2 → ~10 GB** | `aten::index`: gather w_up（**+4,563 MB**） |
| **7.78ms** | **阶梯 #3 → ~14.5 GB** | `aten::index`: gather w_down（**+4,563 MB**） |
| **8.70ms** | **→ 峰值 16,252 MB** | `aten::mm`: lm_head logits（+1,915 MB） |

### 4.3 最大显存分配

<!-- Memory View 分配表截图 -->

| 算子 | 单次分配 | 次数 | 总计 | 分配时刻 | 释放 |
|------|---------|------|------|---------|------|
| `aten::index` | 4,562,944 KB (4.35 GB) | 3 | **13.05 GB** | 6.63 / 7.23 / 7.78 ms | 未释放 |
| `aten::mm` | 1,914,880 KB (1.87 GB) | 1 | 1.87 GB | 8.70 ms | 未释放 |

> 注意: **Allocation Time** 列是分配**发生的时间戳**（第几毫秒），不是分配耗时。

### 4.4 MoE 显存爆炸详解

原始专家权重仅 `3 × [64, 512, 128]` = **24 MB**，但 gather 展开后：

```python
flat_indices = topk_indices.reshape(-1)    # [4455 * 8] = [35640]
gate_w = self.w_gate[flat_indices]          # [35640, 512, 128] → 4.35 GB
up_w   = self.w_up[flat_indices]            # [35640, 512, 128] → 4.35 GB
down_w = self.w_down[flat_indices]          # [35640, 128, 512] → 4.35 GB
                                            # 合计: 13.05 GB，膨胀 544 倍
```

### 4.5 各模块显存增量

| 模块 | 显存增量 | 占激活总量 | 说明 |
|------|----------|-----------|------|
| `layers.0.ffn` (MoE) | +13,477 MB | **85.5%** | 3 次 weight gather |
| `lm_head` | +1,870 MB | 11.9% | logits `[4455, 220000]` × FP16 |
| `layers.0.attn` | +347 MB | 2.2% | QKV + attention score 矩阵 |
| `final_norm` | +22 MB | 0.1% | RMSNorm 中间量 |
| `token_embedding` | +9 MB | 0.06% | 查表 + position embedding |

---

## 5. Perfetto Timeline vs TensorBoard 对比说明

两个工具看的是同一份 profiling 数据，但视角不同：

| 维度 | Perfetto Timeline | TensorBoard Profiler |
|------|-------------------|---------------------|
| 核心视图 | 时间线（横轴=时间，纵轴=线程/stream） | 聚合统计表 + 饼图 + 显存曲线 |
| 适合回答 | kernel 之间有没有空隙？是串行还是并行？CPU launch 跟不跟得上？ | 哪个算子总耗时最多？显存被谁吃了？Tensor Core 利用率？ |
| 常见误读 | CPU 行上 `aten::index` 色块大 ≠ GPU 计算慢（实为 `cudaMalloc` 阻塞）；`aten::mm` 在 CPU 行很窄（只是 launch）但在 CUDA 行是大块 | Host Time ≠ GPU 计算时间（异步提交），应看 Device Self Time |

---

## 6. 优化建议（按收益排序）

### P0: MoE 重写 — grouped GEMM 替代 index + bmm

**预期收益**: 显存 -13 GB，GPU 时间 -29%，Tensor Core 利用率大幅提升

当前实现把「按 expert 分组的稀疏计算」变成了「按 token 展开的密集拷贝」。正确做法是按 expert 分组 token，每个 expert 做一次大矩阵乘：

```python
for expert_id in range(num_experts):
    mask = (topk_indices == expert_id).any(dim=-1)
    expert_tokens = x[mask]
    out = silu(expert_tokens @ w_gate[expert_id]) * (expert_tokens @ w_up[expert_id])
    out = out @ w_down[expert_id]
```

生产环境使用 fused MoE kernel（如 vLLM `fused_moe`、Megablocks）。

### P1: FlashAttention — 替换手动 attention

**预期收益**: 显存 -303 MB，GPU 时间 -37%（消除最大单一 kernel），Tensor Core 利用率提升

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

消除 `[8, 4455, 4455]` 的 attention score 矩阵，使用 tiling 避免 O(n^2) 显存，同时利用 Tensor Core。

### P2: lm_head 分块计算

**预期收益**: 显存 -1.87 GB

vocab=220000 的 logits 输出巨大，可分 chunk 计算 loss：

```python
for chunk in h.chunk(8):
    logits_chunk = self.lm_head(chunk)
    loss += F.cross_entropy(logits_chunk, targets_chunk)
```

### P3: 删除无效的权重初始化

`network.py` 第 140-144 行的 `xavier_uniform_` 变换立刻被第 146-148 行的 `nn.init.normal_` 覆盖，前者完全无效。

---

## 7. 产出文件

| 文件 | 用途 | 打开方式 |
|------|------|----------|
| `tb_log/` | TensorBoard profiling 日志 | `tensorboard --logdir=tb_log --port=6006` |
| `/tmp/transformer_memory_snapshot.pickle` | 显存分配历史 | [PyTorch Memory Viz](https://pytorch.org/memory_viz) |
