# LiteECG-Net 说明

## 项目概述

LiteECG-Net 是一个用于ECG（心电图）信号异常检测的深度学习框架，实现了多种先进的神经网络架构，用于基于MIT-BIH心律失常数据库对ECG信号进行分类。

### 主要功能
- **多种模型架构**：LiteECGNet、ECGNet、SE_ECGNet、BiRCNN、ResNet、DeepECGNet、LDCNN
- **分层数据集支持**：支持分层和随机数据集分割策略
- **全面评估**：模型参数、FLOPs、推理延迟和内存使用等指标
- **灵活配置**：易于修改的超参数和模型配置
- **缓存系统**：预处理数据缓存，加速训练
- **可重现结果**：固定随机种子的确定性训练


## 项目结构

```
LiteECG-Net-git/
├── data/                           # 数据目录
│   └── mitbih/                     # MIT-BIH数据库文件
│       ├── *.dat                   # ECG信号数据文件
│       ├── *.atr                   # 标注文件
│       ├── *.hea                   # 头文件
│       └── cache_*.npz             # 预处理数据缓存文件
├── models/                         # 神经网络模型实现
│   ├── __init__.py                 # 包初始化文件
│   ├── LiteECGNet.py              # 轻量级ECG网络（带注意力机制）
│   ├── ECGNet.py                  # 原始ECGNet架构
│   ├── SE_ECGNet.py               # 带挤压-激励模块的ECGNet
│   ├── BiRCNN.py                  # 双向RNN与CNN结合
│   ├── ResNet.py                  # 1D ResNet用于ECG信号
│   ├── DeepECGNet.py              # 带Transformer的深度ECG网络
│   └── LDCNN.py                   # 轻量级深度CNN
├── config.py                       # 模型和训练配置
├── dataset.py                      # 随机数据集分割实现
├── stratified_dataset.py           # 分层数据集分割实现
├── trainer.py                      # 主要训练和评估脚本
├── utils.py                        # 数据处理工具函数
├── run_all_models.bat              # 运行所有模型的批处理脚本
├── run_all_models-nostra.bat       # 不带分层采样的批处理脚本
└── quick_test.bat                  # 快速测试脚本（每个模型1个epoch）
```

## 代码理解

### 核心文件说明

#### 1. 数据集处理
- **dataset.py**：实现了基于记录的随机数据集分割
- **stratified_dataset.py**：实现了基于样本的分层数据集分割，保持类分布

#### 2. 模型实现
- **models/LiteECGNet.py**：轻量级ECG网络，带有注意力机制
- **models/ECGNet.py**：原始ECGNet架构
- **models/SE_ECGNet.py**：带有挤压-激励模块的ECGNet
- **models/BiRCNN.py**：双向RNN与CNN结合的模型
- **models/ResNet.py**：1D ResNet用于ECG信号
- **models/DeepECGNet.py**：带有Transformer的深度ECG网络
- **models/LDCNN.py**：轻量级深度CNN

#### 3. 训练和评估
- **trainer.py**：主要训练和评估脚本，包含数据加载、模型训练、评估和结果保存
- **utils.py**：工具函数，包括数据预处理、符号映射等

### 数据流程

1. **数据加载**：从MIT-BIH数据库加载ECG信号和标注
2. **数据预处理**：
   - 基线移除
   - 信号分段（默认2秒一段）
   - 标准化（默认z-score）
   - 数据增强（边界填充）
3. **数据集分割**：
   - 分层采样：保持类分布
   - 随机采样：基于记录的随机分割
4. **模型训练**：
   - 支持多种模型架构
   - 支持Focal Loss处理类别不平衡
   - 早停策略
5. **模型评估**：
   - 准确率、F1分数、精确率、召回率
   - 混淆矩阵
   - 模型参数、FLOPs、推理延迟等

## 运行实验

### 快速开始

#### 1. 运行单个模型

```bash
# 训练LiteECGNet模型，30个epoch，批次大小256，学习率3e-3，启用缓存
python trainer.py --model liteecgnet --epochs 30 --batch-size 256 --lr 3e-3 --cache
```

#### 2. 运行所有模型（Windows）

```bash
# 运行所有模型，使用分层采样
run_all_models.bat

# 运行所有模型，不使用分层采样
run_all_models-nostra.bat
```

#### 3. 快速测试所有模型

```bash
# 每个模型运行1个epoch进行快速测试
quick_test.bat
```

### 命令行参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model` | str | `liteecgnet` | 要训练的模型（`liteecgnet`, `ecgnet`, `se_ecgnet`, `bircnn`, `resnet`, `deepecgnet`, `ldcnn`） |
| `--root` | str | `./data/mitbih` | MIT-BIH目录路径 |
| `--epochs` | int | `30` | 训练epoch数 |
| `--batch-size` | int | `256` | 批次大小 |
| `--lr` | float | `3e-3` | 学习率 |
| `--weight-decay` | float | `1e-4` | 正则化权重衰减 |
| `--segment-sec` | float | `2.0` | ECG分段持续时间（秒） |
| `--fs` | int | `360` | 采样频率（Hz） |
| `--cache` | flag | `False` | 启用数据缓存 |
| `--class-limit` | int | `None` | 训练集中每类样本限制 |
| `--patience` | int | `7` | 早停耐心值 |
| `--focal` | flag | `False` | 使用Focal Loss代替CrossEntropy |
| `--no-stratified` | flag | `False` | 禁用分层采样 |
| `--device` | str | `cuda` | 使用的设备（`cuda`或`cpu`） |

## 结果解释

训练完成后，结果将保存在`data/records/`目录中：
- `ecg_lightweight_{model}_{timestamp}.pt` - 训练好的模型权重
- `eval_report_{model}_{timestamp}.json` - 详细的评估报告

评估报告包含以下内容：
- 测试集准确率和F1分数
- 混淆矩阵
- 模型指标（参数数量、模型大小、FLOPs、推理延迟、内存使用）
- 训练历史（损失曲线、epoch时间等）

### 模型性能指标说明

- **参数数量**：模型中可训练参数的总数，影响模型大小和内存使用
- **FLOPs**：浮点运算次数，衡量模型计算复杂度
- **推理延迟**：模型处理单个样本所需的时间，对于实时应用很重要
- **内存使用**：模型运行时占用的内存

## 常见问题解决

### 1. CUDA内存不足
- 减少批次大小：`--batch-size 128`
- 使用CPU：`--device cpu`

### 2. 数据加载错误
- 确保MIT-BIH文件格式正确
- 检查文件权限和路径

### 3. 模型导入错误
- 验证所有依赖项已安装
- 检查Python路径和环境

### 4. 训练速度慢
- 启用缓存：`--cache`
- 使用GPU加速
- 调整`--num-workers`参数以适应您的系统

## 项目扩展

### 添加新模型
1. 在`models/`目录中创建模型类
2. 在`config.py`中添加配置
3. 在`trainer.py`中导入并添加到模型选择中

### 自定义损失函数
在`trainer.py`中实现自定义损失函数，并修改损失函数选择逻辑。

### 数据增强
在数据集类（`dataset.py`, `stratified_dataset.py`）中添加增强技术。

## 学习资源

- **MIT-BIH心律失常数据库**：https://physionet.org/content/mitdb/1.0.0/
- **ECGNet论文**：https://arxiv.org/abs/1804.00712
- **挤压-激励网络论文**：https://arxiv.org/abs/1709.01507
- **Focal Loss论文**：https://arxiv.org/abs/1708.02002

## 技术支持

如果您在使用过程中遇到问题，可以：
1. 检查代码中的注释和文档
2. 查看命令行参数说明
3. 参考项目中的示例脚本

## 注意事项

- 本框架设计用于研究和教育目的
- 对于临床应用，请确保适当的验证和监管合规
- 首次运行时会自动生成缓存文件，后续运行会更快
- 训练结果会保存在`data/records/`目录中

---

祝您使用愉快！