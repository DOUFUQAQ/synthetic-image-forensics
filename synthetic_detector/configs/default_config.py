"""
默认配置文件，包含训练和评估的默认参数
"""

# 数据参数
DATA_CONFIG = {
    'subset_ratio': 0.1,        # 使用数据集的比例
    'max_samples': 5000,        # 每类最大样本数
    'img_size': 256,            # 图像大小
}

# 模型参数
MODEL_CONFIG = {
    'feature_dim': 256,         # 特征维度
    'pretrained': True,         # 是否使用预训练的ResNet50
    'num_classes': 2,           # 类别数量(真实/生成)
}

# 训练参数
TRAIN_CONFIG = {
    'batch_size': 32,           # 批量大小
    'epochs': 50,               # 训练轮数
    'lr': 0.001,                # 学习率
    'weight_decay': 1e-4,       # 权重衰减
    'save_interval': 5,         # 每隔多少轮保存一次模型
    'num_workers': 4,           # 数据加载线程数
    'seed': 42,                 # 随机种子
}

# 路径配置
PATH_CONFIG = {
    'save_dir': 'checkpoints',  # 模型保存目录
    'log_dir': 'logs',          # 日志保存目录
    'output_dir': 'results',    # 评估结果保存目录
}
