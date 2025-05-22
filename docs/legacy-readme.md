# 图像生成检测与图像哈希项目

本项目实现了两个核心算法：
1. 基于Gram模块的图像生成检测算法 - 可识别GAN和扩散模型生成的图像
2. 语义嵌入与多特征融合的图像哈希算法 - 实现高效图像检索

## 项目结构

```
.
├── README.md                     # 项目说明文档
├── requirements.txt              # 项目依赖
├── data/                         # 数据集目录
│   ├── real/                     # 真实图像数据
│   ├── generated/                # 生成图像数据
│   └── dataset.py                # 数据集加载脚本
├── models/                       # 模型定义
│   ├── gram_detection/           # Gram检测模型
│   │   ├── gram_module.py        # Gram矩阵计算模块
│   │   └── detector.py           # 检测网络定义
│   └── hash/                     # 哈希模型
│       ├── semantic_module.py    # 语义嵌入模块
│       ├── feature_fusion.py     # 特征融合模块
│       └── hash_net.py           # 哈希网络定义
├── utils/                        # 工具函数
│   ├── visualization.py          # 可视化工具
│   ├── metrics.py                # 评估指标计算
│   └── transforms.py             # 数据转换和增强
├── train/                        # 训练相关
│   ├── train_detector.py         # 检测器训练脚本
│   └── train_hash.py             # 哈希网络训练脚本
├── evaluate/                     # 评估相关
│   ├── evaluate_detector.py      # 检测器评估脚本
│   └── evaluate_hash.py          # 哈希网络评估脚本
└── demo/                         # 演示应用
    ├── demo_detector.py          # 检测器演示
    └── demo_hash.py              # 哈希网络演示
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

### 图像生成检测

```bash
# 训练检测器
python train/train_detector.py --data_dir data/ --batch_size 32 --epochs 50

# 评估检测器
python evaluate/evaluate_detector.py --model_path checkpoints/detector.pth --test_dir data/test/

# 演示检测功能
python demo/demo_detector.py --image_path your_image.jpg --model_path checkpoints/detector.pth
```

### 图像哈希

```bash
# 训练哈希网络
python train/train_hash.py --data_dir data/ --hash_size 64 --batch_size 32 --epochs 50

# 评估哈希性能
python evaluate/evaluate_hash.py --model_path checkpoints/hash_net.pth --query_dir data/query/ --database_dir data/database/

# 演示哈希功能
python demo/demo_hash.py --image_path your_query.jpg --database_dir data/database/ --model_path checkpoints/hash_net.pth
```
