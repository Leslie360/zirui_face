# 模型架构详细介绍

本文档详细介绍了项目中使用的神经网络模型架构，包括基础模型、增强模型以及各种组件模块。

---

## 目录
1. [核心组件](#核心组件)
2. [基础模型](#基础模型)
3. [增强模型](#增强模型)
4. [工厂函数](#工厂函数)
5. [模型对比](#模型对比)

---

## 核心组件

### 1. SEBlock (Squeeze-and-Excitation Block)

**功能**: 通道注意力机制，自适应地重新校准通道特征响应。

**结构**:
```
输入 (B, C, H, W)
    ↓
全局平均池化 → (B, C, 1, 1)
    ↓
全连接层 (C → C/16) + ReLU
    ↓
全连接层 (C/16 → C) + Sigmoid
    ↓
通道加权 → 输出 (B, C, H, W)
```

**参数**:
- `channel`: 输入通道数
- `reduction`: 降维比例，默认为 16

**特点**:
- 轻量级设计，计算开销小
- 显式建模通道间依赖关系
- 提升模型特征表达能力

---

### 2. ResBlock (残差块)

**功能**: 带有跳跃连接的卷积块，支持可选的 SE 注意力机制。

**结构**:
```
输入
  ├─→ 主路径: Conv3x3(stride) → BN → ReLU → Conv3x3 → BN → SE (可选)
  └─→ 跳跃连接: Conv1x1(stride) → BN (如需下采样)
       ↓
     相加 → ReLU → 输出
```

**参数**:
- `in_channels`: 输入通道数
- `out_channels`: 输出通道数
- `stride`: 步长（用于下采样）
- `use_se`: 是否使用 SE 注意力，默认 True

**特点**:
- 解决深层网络梯度消失问题
- 支持特征维度变换
- 集成 SE 模块增强特征表达

---

## 基础模型

### 1. MemristorCNN (基础忆阻器 CNN)

**用途**: 用于忆阻器硬件映射的基础卷积神经网络。

**架构**:
```
输入: (3, 32, 32) CIFAR-10 图像

特征提取:
  Conv2d(3 → FIRST_CONV_CHANNELS, 3x3, pad=1)
  → ReLU → MaxPool2d(2×2)
  → Conv2d(FIRST_CONV_CHANNELS → SECOND_CONV_CHANNELS, 3x3, pad=1)
  → ReLU → MaxPool2d(2×2)
  → Conv2d(SECOND_CONV_CHANNELS → THIRD_CONV_CHANNELS, 3x3, pad=1)
  → ReLU → MaxPool2d(2×2)

分类器:
  Flatten
  → Linear(THIRD_CONV_CHANNELS×4×4 → 1024)
  → ReLU
  → Linear(1024 → NUM_CLASSES)

输出: (NUM_CLASSES,) logits
```

**默认配置** (来自 config.py):
- `FIRST_CONV_CHANNELS`: 32
- `SECOND_CONV_CHANNELS`: 64
- `THIRD_CONV_CHANNELS`: 128
- `NUM_CLASSES`: 10

**特点**:
- 简单的三层卷积结构
- 适合硬件映射实验
- 全连接层可被忆阻器阵列替代

**辅助函数**:
- `extract_fc_weights(model)`: 提取全连接层权重和偏置，返回 `[(W1, b1), (W2, b2)]`

---

### 2. MemristorCNN_SingleChannel (单通道版本)

**用途**: 处理单通道（灰度）图像的忆阻器 CNN 变体。

**与 MemristorCNN 的区别**:
- 输入通道数: `1` (vs. 3)
- 保持相同的卷积通道配置
- 其他结构完全一致

**应用场景**:
- 灰度图像数据集 (如 MNIST)
- 单色传感器数据
- 通道分离实验

---

## 增强模型

### 1. MemristorCNN_Strong (强化 ResNet-like 架构)

**用途**: 高性能模型，集成残差连接、SE 注意力和对比学习投影头。

**架构**:
```
输入: (3, 32, 32)

初始卷积:
  Conv2d(3 → 64, 3x3, stride=1, pad=1, bias=False)
  → BN → ReLU

残差层:
  Layer1: 2×ResBlock(64 → 64, stride=1)   [32×32]
  Layer2: 2×ResBlock(64 → 128, stride=2)  [16×16]
  Layer3: 2×ResBlock(128 → 256, stride=2) [8×8]

分支输出:
  ├─→ 分类器:
  │    Flatten → Linear(256×8×8 → 1024)
  │    → ReLU → Dropout(0.5)
  │    → Linear(1024 → NUM_CLASSES)
  │
  └─→ 投影头 (用于对比学习):
       Flatten → Linear(256×8×8 → 512)
       → ReLU → Linear(512 → 128)

输出: logits 或 (logits, projection)
```

**参数**:
- `num_classes`: 分类数量，默认 10
- `use_se`: 是否在 ResBlock 中使用 SE 注意力，默认 True

**前向传播模式**:
- `forward(x)`: 仅返回分类 logits
- `forward(x, return_features=True)`: 返回 `(logits, projection)` 用于对比学习

**特点**:
- **更深的网络**: 6 个残差块 vs. 3 个普通卷积
- **批归一化**: 稳定训练，加速收敛
- **Dropout 正则化**: 防止过拟合（p=0.5）
- **SE 注意力**: 自适应特征重标定
- **投影头**: 支持自监督对比学习（SimCLR 等）
- **无偏置卷积**: 配合 BN 使用，减少参数

**适用场景**:
- 需要更高准确率的任务
- 对比学习预训练
- 小数据集（CIFAR-10/100）

---

### 2. ResNet18CIFAR (CIFAR 定制 ResNet18)

**用途**: 标准 ResNet18 适配 CIFAR-10 小尺寸图像（32×32）。

**关键修改**:
```python
# 原始 ResNet18 (ImageNet)
Conv2d(3 → 64, 7×7, stride=2, pad=3)
→ MaxPool2d(3×3, stride=2)
→ ...

# CIFAR 适配版本
Conv2d(3 → 64, 3×3, stride=1, pad=1)  # 小卷积核，无下采样
→ Identity (移除最大池化)             # 保持空间分辨率
→ ...
```

**改动原因**:
- CIFAR-10 图像仅 32×32，过早的下采样会丢失信息
- 7×7 卷积和 MaxPool 会将特征图快速缩小到 4×4
- 调整后保持足够的空间分辨率供后续层处理

**特点**:
- 使用 torchvision 预训练架构
- 轻量级适配，保持 ResNet 优势
- 作为基线模型对比实验效果

---

## 工厂函数

### get_model(variant, num_classes)

**功能**: 统一的模型创建接口，根据变体名称返回相应模型实例。

**参数**:
- `variant` (str): 模型变体名称
  - `"base"`: 返回 `MemristorCNN`
  - `"strong"`: 返回 `MemristorCNN_Strong`
  - `"resnet18"` 或 `"resnet"`: 返回 `ResNet18CIFAR`
- `num_classes` (int): 分类数量，默认从 `cfg.NUM_CLASSES` 读取

**示例**:
```python
from model import get_model

# 获取基础模型
base_model = get_model("base", num_classes=10)

# 获取强化模型
strong_model = get_model("strong", num_classes=10)

# 获取 ResNet18
resnet_model = get_model("resnet18", num_classes=10)
```

**错误处理**:
- 传入未知变体名称时抛出 `ValueError`

---

## 模型对比

### 复杂度对比

| 模型 | 卷积层数 | 参数量 (估算) | 特征提取能力 | 适用场景 |
|------|---------|--------------|-------------|---------|
| **MemristorCNN** | 3 | ~1.5M | 基础 | 硬件映射、快速原型 |
| **MemristorCNN_SingleChannel** | 3 | ~1.5M | 基础 | 灰度图像 |
| **MemristorCNN_Strong** | 6 (ResBlocks) | ~3.5M | 强 | 高准确率、对比学习 |
| **ResNet18CIFAR** | 18 | ~11M | 极强 | 基线对比、迁移学习 |

### 性能预期 (CIFAR-10)

| 模型 | 训练时长 (相对) | 测试准确率 (估算) | 内存占用 (相对) |
|------|----------------|------------------|----------------|
| **MemristorCNN** | 1× | ~70-75% | 1× |
| **MemristorCNN_Strong** | 1.5× | ~88-92% | 1.5× |
| **ResNet18CIFAR** | 2× | ~92-95% | 2.5× |

### 设计权衡

**MemristorCNN**:
- ✅ 简单、易于硬件映射
- ✅ 训练快速
- ❌ 准确率受限
- ❌ 缺乏正则化

**MemristorCNN_Strong**:
- ✅ 高准确率
- ✅ 支持对比学习
- ✅ 正则化充分（BN + Dropout）
- ⚠️ 参数量中等
- ❌ 硬件映射复杂度增加

**ResNet18CIFAR**:
- ✅ 最高准确率
- ✅ 可迁移预训练权重
- ❌ 参数量大
- ❌ 不适合忆阻器硬件映射

---

## 使用建议

### 1. 选择模型变体

**场景 1: 忆阻器硬件实验**
- 推荐: `MemristorCNN` (base)
- 原因: 结构简单，全连接层易于映射到忆阻器交叉阵列

**场景 2: 高准确率需求**
- 推荐: `MemristorCNN_Strong` (strong)
- 原因: 在保持可映射性的同时提供更强特征提取能力

**场景 3: 基线对比实验**
- 推荐: `ResNet18CIFAR` (resnet18)
- 原因: 作为理想浮点模型的上限参考

**场景 4: 对比学习预训练**
- 推荐: `MemristorCNN_Strong` (strong)
- 原因: 内置投影头，支持 `return_features=True`

### 2. 配置建议

在 `config.py` 中设置：
```python
# 选择模型变体
MODEL_VARIANT = "strong"  # "base" | "strong" | "resnet18"

# 根据模型调整超参数
if MODEL_VARIANT == "base":
    BATCH_SIZE = 128
    LR = 0.001
elif MODEL_VARIANT == "strong":
    BATCH_SIZE = 128
    LR = 0.001
    WEIGHT_DECAY = 5e-4
elif MODEL_VARIANT == "resnet18":
    BATCH_SIZE = 128
    LR = 0.01
    WEIGHT_DECAY = 1e-4
```

### 3. 训练技巧

**数据增强**:
- Base 模型: 基础增强（RandomCrop + Flip）
- Strong 模型: 强增强（+ ColorJitter + RandomErasing）
- ResNet18: 中等增强

**优化器选择**:
- Base: SGD (momentum=0.9)
- Strong/ResNet18: AdamW (weight_decay=5e-4)

**学习率调度**:
- 推荐: CosineAnnealingLR (T_max=100)
- 或: StepLR (step_size=30, gamma=0.1)

---

## 忆阻器集成

### 全连接层替换

所有模型的全连接层都可以被忆阻器交叉阵列替代：

```python
from train_memristor_cnn import MemFCLState, get_fc_linears

# 1. 创建模型
model = get_model("strong", num_classes=10)

# 2. 提取全连接层
fc1, fc2 = get_fc_linears(model)

# 3. 初始化忆阻器状态
fcl_state = MemFCLState(
    g_states_per_color, 
    max_pulses_per_step=3,
    color_mapping="round_robin",
    device=device
)

# 4. 将浮点权重映射到电导状态
fcl_state.init_layer_from_float("fc1", fc1.weight.detach().cpu().numpy())
fcl_state.init_layer_from_float("fc2", fc2.weight.detach().cpu().numpy())

# 5. 写回量化权重到模型
fcl_state.writeback("fc1", fc1.weight)
fcl_state.writeback("fc2", fc2.weight)
```

### 权重更新模式

**双向模式 (bidir)**:
- 权重可增可减（±ΔG）
- 更灵活，收敛更快
- 对重叠噪声更敏感

**单向模式 (unidir)**:
- 仅允许电导增加（+ΔG）
- 符合某些忆阻器物理特性
- 需要更多训练轮次
- 对噪声更鲁棒

---

## 实验配置示例

### 实验 1: 基础训练

```python
cfg.MODEL_VARIANT = "base"
cfg.EPOCHS = 100
cfg.LR = 0.001
cfg.USE_MEMRISTOR_FCL = False  # 纯浮点训练
```

预期结果: ~70-75% (无忆阻器)

### 实验 2: 强化模型 + 忆阻器

```python
cfg.MODEL_VARIANT = "strong"
cfg.EPOCHS = 300
cfg.LR = 0.001
cfg.USE_MEMRISTOR_FCL = True
cfg.UPDATE_MODE = "bidir"
cfg.BIDIR_MAX_PULSES = 3
cfg.BIDIR_MEM_SCALE = 1.0
cfg.LABEL_SMOOTHING = 0.05
```

预期结果: ~88-92% @ overlap=0

### 实验 3: 重叠敏感性测试

```python
cfg.RGB_OVERLAP_LIST = [0.0, 0.1, 0.3, 0.5]
cfg.OVERLAP_ALPHA = 1.8  # 增强重叠效应
cfg.OVERLAP_GAMMA = 1.0
```

预期结果:
- Bidir: 明显准确率下降 (0→0.5: -6~8%)
- Unidir: 更鲁棒 (0→0.5: -3~5%)

---

## 总结

本项目提供了从简单到复杂的多层次模型架构，满足不同实验需求：

1. **MemristorCNN**: 快速原型和硬件映射
2. **MemristorCNN_Strong**: 高性能与可映射性平衡
3. **ResNet18CIFAR**: 理想浮点基线

所有模型均支持忆阻器全连接层集成，可用于研究神经形态硬件的量化误差、重叠噪声和更新模式影响。

**关键创新点**:
- 🎨 **每色独立电导状态**: 支持 RGB 通道分离量化
- ⚡ **脉冲限制更新**: 模拟真实硬件约束
- 🔄 **双向/单向模式**: 对比不同忆阻器特性
- 🌈 **重叠增强机制**: 可调节的通道串扰 (OVERLAP_ALPHA)

更多细节请参考 `config.py` 和 `train_memristor_cnn.py`。
