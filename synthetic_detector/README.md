# 合成图像检测器 (Synthetic Image Detector)

基于多特征融合和语义嵌入的合成图像检测模型，用于区分真实图像和AI生成的图像。

## 项目结构

```
synthetic_detector/
├── configs/             # 配置文件
├── data/                # 数据处理模块
├── models/              # 模型定义
├── train/               # 训练脚本
├── evaluate/            # 评估脚本
├── utils/               # 工具函数
├── demo/                # 演示脚本
├── checkpoints/         # 保存的模型检查点(训练后生成)
├── logs/                # 训练日志(训练后生成)
├── results/             # 评估结果(评估后生成)
└── run.py               # 主运行脚本(包含所有功能)
```

## 特性

- **多特征融合**: 结合CNN特征、频率特征和共现特征进行检测
- **自适应权重**: 自动学习和调整不同特征的权重
- **注意力机制**: 使用注意力机制增强特征融合效果
- **高效数据处理**: 支持大型数据集的采样和预处理
- **可视化工具**: 提供详细的评估指标和特征权重可视化

## 模型架构

- **主干网络**: ResNet50 (预训练)
- **特征提取器**:
  - CNN特征 (ResNet50)
  - 频率特征 (FFT + CNN)
  - 共现特征 (统计特征 + CNN)
- **特征融合**: 自适应权重 + 注意力机制
- **输出**: 二分类 (真实/生成)

## 使用方法

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA (推荐用于加速训练)
- 其他依赖: torchvision, numpy, matplotlib, scikit-learn, PIL, tqdm

### 统一命令行工具

项目提供了一个统一的命令行工具`run.py`，支持所有功能。使用方法：

```bash
python run.py <命令> [参数]
```

支持的命令：`prepare`, `train`, `evaluate`, `demo`

### 数据准备

从DeepFaceGen数据集中提取子集:

```bash
python run.py prepare --src_dir "原始数据集路径" --dst_dir "处理后数据集路径" --ratio 0.1 --max_samples 5000 --img_size 256 --resize
```

### 训练模型

训练检测器模型:

```bash
python run.py train --data_dir "数据集路径" --batch_size 32 --epochs 50 --lr 0.001 --pretrained --save_dir checkpoints --log_dir logs
```

### 评估模型

评估模型性能:

```bash
python run.py evaluate --data_dir "测试数据集路径" --checkpoint "checkpoints/detector_best.pth" --output_dir results --visualize
```

### 演示检测

使用模型检测图像:

```bash
python run.py demo --input "图像或目录路径" --checkpoint "checkpoints/detector_best.pth" --output_dir demo_results --visualize
```

## 防止过拟合的策略

为了提高模型的泛化能力，我们实施了以下策略:

1. **增强数据增强**:
   - 水平和垂直翻转
   - 随机旋转(±10度)
   - 亮度、对比度和饱和度变化
   - 随机裁剪和调整大小

2. **正则化技术**:
   - Dropout率提高到0.7
   - 权重衰减(L2正则化)设为1e-4
   - 早停机制(验证性能8轮无提升时停止训练)

3. **学习率调度**:
   - 学习率自动调整(ReduceLROnPlateau)
   - 当验证损失停滞时降低学习率

4. **模型改进**:
   - 多特征融合增强泛化能力
   - 注意力机制自适应调整特征重要性

## 训练技巧

- 建议从较小的数据子集开始训练，然后逐渐增加
- 监控训练过程中的特征权重变化，了解模型学习状况
- 如果CNN特征权重过高，可以手动调整初始权重
- 对于大型数据集，可以增加批处理大小(如果内存允许)或使用梯度累积
- 使用TensorBoard查看训练过程中的损失和指标变化

## 性能优化

- 为避免CUDA内存不足问题，评估时使用分批处理
- 数据加载使用多线程(num_workers)加速
- 使用pin_memory=True加速GPU数据传输
- 对于大型数据集，使用subset_ratio参数减少数据量

## 注意事项

- 首次运行时会下载预训练的ResNet50模型
- 训练日志保存在logs目录，可用TensorBoard查看
- 最佳模型保存为detector_best.pth
- 演示时，阈值参数(--threshold)可调整检测灵敏度
