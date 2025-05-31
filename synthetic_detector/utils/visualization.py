import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
import torch

def plot_feature_weights(weights_history):
    """
    绘制特征权重随训练进度的变化
    
    Args:
        weights_history: 形状为 [epochs, num_features] 的特征权重历史
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(weights_history) + 1)
    feature_names = ['CNN', '频率', '共现']
    
    for i, name in enumerate(feature_names):
        ax.plot(epochs, weights_history[:, i], marker='o', label=name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('权重值')
    ax.set_title('特征融合权重变化')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_roc_curve(y_true, y_score):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_score: 预测概率
        
    Returns:
        matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = np.trapz(tpr, fpr)  # 计算曲线下面积
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--')  # 随机猜测的基线
    ax.set_xlabel('假正率 (FPR)')
    ax.set_ylabel('真正率 (TPR)')
    ax.set_title('接收者操作特征曲线 (ROC)')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_precision_recall_curve(y_true, y_score):
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true: 真实标签
        y_score: 预测概率
        
    Returns:
        matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = np.trapz(precision, recall)  # 计算平均精度
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'AP = {ap:.4f}')
    ax.set_xlabel('召回率 (Recall)')
    ax.set_ylabel('精确率 (Precision)')
    ax.set_title('精确率-召回率曲线')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_confusion_matrix(y_true, y_pred, normalize=True):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        normalize: 是否归一化
        
    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', ax=ax)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('混淆矩阵')
    ax.set_xticklabels(['真实', '生成'])
    ax.set_yticklabels(['真实', '生成'])
    
    return fig

def visualize_attention_maps(model, images, device):
    """
    可视化注意力权重
    
    Args:
        model: 训练好的模型
        images: 输入图像批次
        device: 设备
        
    Returns:
        matplotlib figure
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        attention_weights = outputs['attention_weights']
    
    batch_size = min(8, images.size(0))  # 最多显示8张图像
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 2*batch_size))
    
    for i in range(batch_size):
        # 显示原始图像
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        # 获取当前图像的注意力权重
        weights = attention_weights[i].cpu().numpy()
        
        # 对于每个特征，可视化注意力权重
        for j, feature_name in enumerate(['CNN', '频率', '共现']):
            axes[i, j+1].bar([feature_name], [weights[j]])
            axes[i, j+1].set_ylim(0, 1)
            axes[i, j+1].set_title(f'权重: {weights[j]:.2f}')
    
    plt.tight_layout()
    return fig
