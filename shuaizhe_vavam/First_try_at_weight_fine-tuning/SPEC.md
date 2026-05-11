# VAVIM 微调计划 - 详细规范文档

## 1. 项目概述

本计划旨在使用 nuplan 和 nuscenes 数据集对 VAVIM 进行微调，并实现视频预测功能。

**目标目录**: `/raid/zengchaolv/shuaizhe_vavam/First_try_at_weight_fine-tuning`

---

## 2. 数据集规模与划分

### 2.1 数据集统计

| 数据集 | 原始图像数量 | Token 文件 | 可用训练样本 |
|--------|-------------|-----------|-------------|
| nuplan | ~24,232 | nuplan_tokens/*.pt | 24,232 |
| nuscenes (mini) | 404 (CAM_FRONT) | nuscenes_tokens/*.pt | 344 |

### 2.2 训练/测试集划分

按照实际比例进行划分：

| 数据集 | 训练集 (80%) | 测试集 (20%) |
|--------|--------------|-------------|
| nuplan | 19,386 | 4,846 |
| nuscenes | 275 | 69 |

**划分原则**: 
- 使用 `train_test_split` 或随机打乱后按比例划分
- 保证时间序列的连续性，避免数据泄露

---

## 3. 系统架构

### 3.1 输入流程

```
数据集图像 (nuplan/nuscenes)
        ↓
    VQ Tokenizer (16x16, vocab=16384)
        ↓
    Token 序列 (每帧 576 tokens)
        ↓
    与 VAVIM 预训练权重合并
        ↓
    再次微调 (Fine-tuning)
        ↓
    更新后的权重
```

### 3.2 模型组件

| 组件 | 文件位置 | 说明 |
|------|---------|------|
| MupGPT2 | video_prediction_nuscenes_mini/vam/video_pretraining/mup_gpt2.py | 视频生成模型 |
| VQ Tokenizer | /raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_encoder.jit | 视觉 token 器 |
| 预训练权重 | /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt | Base 权重 |

---

## 4. 详细实施步骤

### 4.1 数据准备阶段

#### Step 1: 创建数据集目录结构

```
First_try_at_weight_fine-tuning/
├── data/
│   ├── nuplan/
│   │   ├── train.pkl
│   │   ├── test.pkl
│   │   └── tokens/
│   └── nuscenes/
│       ├── train.pkl
│       ├── test.pkl
│       └── tokens/
├── configs/
├── scripts/
├── checkpoints/
├── weights/
└── results/
```

#### Step 2: 数据集划分脚本

创建 `scripts/01_split_dataset.py`:
- 读取原始 pkl 文件
- 按 80/20 比例划分训练/测试集
- 保存为 train.pkl 和 test.pkl

#### Step 3: Token 化脚本

创建 `scripts/02_tokenize_data.py`:
- 加载 VQ Tokenizer
- 对训练集和测试集图像进行 tokenize
- 保存为 .pt 文件

### 4.2 微调阶段

#### Step 4: 配置微调参数

修改或创建配置文件 `configs/finetune_nuplan_nuscenes.yaml`:

```yaml
# 模型配置
model:
  _target_: vam.video_pretraining.NextTokenPredictor
  network:
    _target_: vam.video_pretraining.mup_gpt2.MupGPT2
    embedding_dim: 256
    dim_heads: 128
    nb_layers: 24
    vocabulary_size: 16385
    nb_timesteps: 8
    nb_tokens_per_timestep: 576

# 数据配置
data:
  sequence_length: 8
  batch_size: 4
  nuplan_tokens_rootdir: .../First_try_at_weight_fine-tuning/data/nuplan/tokens
  nuscenes_tokens_rootdir: .../First_try_at_weight_fine-tuning/data/nuscenes/tokens
  nuplan_train_pickle: .../data/nuplan/train.pkl
  nuplan_test_pickle: .../data/nuplan/test.pkl
  nuscenes_train_pickle: .../data/nuscenes/train.pkl
  nuscenes_test_pickle: .../data/nuscenes/test.pkl
  # 数据混合比例
  ratios: [0.9, 0.1]  # nuplan 90%, nuscenes 10%

# 优化器配置
optimizer_conf:
  lr: 1e-4  # 微调学习率
  weight_decay: 1e-8

# 训练配置
trainer:
  max_epochs: 3
  accumulate_grad_batches: 4
  precision: 16
  checkpoint_dir: .../First_try_at_weight_fine-tuning/checkpoints
```

#### Step 5: 执行微调

创建 `scripts/03_finetune.py`:
- 加载预训练权重 (width_768_pretrained_139k_total_155k.pt)
- 使用 nuplan + nuscenes 训练集进行微调
- 保存检查点到 `checkpoints/`

### 4.3 视频预测阶段

#### Step 6: 实现视频预测

参考 `video_prediction_nuscenes_mini/scripts/video_prediction_nuscenes_mini.py`，创建预测脚本：

```python
# scripts/04_predict.py
def predict_video(
    model_path: str,           # 微调后的权重
    tokenizer_path: str,      # VQ Tokenizer
    input_frames: list,      # 4帧测试集图像
    num_predict: int = 4,    # 预测4帧
):
    # 1. 加载模型和权重
    # 2. Tokenize 输入帧
    # 3. 自回归生成未来帧
    # 4. Detokenize 生成图像
    # 5. 保存结果
```

#### Step 7: 评估与测试

从测试集中随机采样 4 帧，进行 4 帧预测，并评估结果。

---

## 5. 文件清单

| 文件 | 用途 |
|------|------|
| `SPEC.md` | 本计划文档 |
| `scripts/01_split_dataset.py` | 数据集划分 |
| `scripts/02_tokenize_data.py` | Token 化处理 |
| `scripts/03_finetune.py` | 微调训练 |
| `scripts/04_predict.py` | 视频预测 |
| `scripts/05_evaluate.py` | 评估脚本 |
| `configs/finetune_config.yaml` | 微调配置 |
| `requirements.txt` | 依赖包 |

---

## 6. 评估指标

### 6.1 视频质量评估

| 指标 | 说明 |
|------|------|
| FID (Fréchet Inception Distance) | 生成视频质量 |
| SSIM | 结构相似性 |
| PSNR | 峰值信噪比 |
| LPIPS | 感知相似性 |

### 6.2 预测准确性

- 从测试集随机采样 10 个序列
- 每个序列输入 4 帧，预测 4 帧
- 计算平均 SSIM / PSNR

---

## 7. 时间估算

| 阶段 | 预估时间 |
|------|----------|
| 数据准备 | 1-2 小时 |
| Tokenize | 2-4 小时 |
| 微调训练 | 4-8 小时 |
| 预测评估 | 1-2 小时 |

**总计**: 约 8-16 小时

---

## 8. 风险与注意事项

### 8.1 可能的问题

1. **显存不足**: 降低 batch_size 或使用梯度累积
2. **过拟合**: nuplan 数据量较大，nuscenes 数据量较小，注意混合比例
3. **权重加载失败**: 检查预训练权重路径

### 8.2 建议

- 先用小 batch 测试流程
- 使用 gradient checkpointing 节省显存
- 监控训练 loss 曲线

---

## 附录: 预训练权重和Tokenizer位置

```
/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/
├── width_768_pretrained_139k_total_155k.pt
└── VAM_width_768_pretrained_139k.pt

/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/
├── VQ_ds16_16384_llamagen_encoder.jit
└── VQ_ds16_16384_llamagen_decoder.jit
```