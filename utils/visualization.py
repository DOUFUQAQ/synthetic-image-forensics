import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
import cv2
import os
from PIL import Image


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """
    绘制训练和验证的损失和指标曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标字典，键为指标名，值为列表
        val_metrics: 验证指标字典，键为指标名，值为列表
        save_path: 保存图表的路径，如果为None则显示图表
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制关键指标曲线
    metrics_to_plot = []
    
    # 对于检测任务
    if 'accuracy' in train_metrics:
        metrics_to_plot.extend(['accuracy', 'precision', 'recall', 'f1'])
    
    # 对于哈希任务
    if 'mAP' in train_metrics:
        metrics_to_plot.extend(['mAP', 'P@10', 'P@100'])
    
    # 最多绘制3个指标
    for i, metric in enumerate(metrics_to_plot[:3]):
        if metric in train_metrics and metric in val_metrics:
            plt.subplot(2, 2, i+2)
            plt.plot(train_metrics[metric], label=f'Training {metric}')
            plt.plot(val_metrics[metric], label=f'Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.title(f'Training and Validation {metric}')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_detection_results(images, true_labels, pred_labels, prob_scores, num_samples=5, save_path=None):
    """
    可视化检测结果
    
    Args:
        images: 图像张量，形状为 [N, C, H, W]
        true_labels: 真实标签，形状为 [N]
        pred_labels: 预测标签，形状为 [N]
        prob_scores: 预测概率，形状为 [N]
        num_samples: 显示的样本数量
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 转换为numpy数组
    if isinstance(images, torch.Tensor):
        # 反归一化
        inv_normalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        images_to_show = []
        for i in range(min(num_samples, len(images))):
            img = inv_normalize(images[i])
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            images_to_show.append(img)
    else:
        images_to_show = images[:num_samples]
    
    # 创建图表
    fig, axes = plt.subplots(1, len(images_to_show), figsize=(15, 3))
    if len(images_to_show) == 1:
        axes = [axes]
    
    for i, (img, tl, pl, ps) in enumerate(zip(images_to_show, true_labels[:num_samples], 
                                            pred_labels[:num_samples], prob_scores[:num_samples])):
        axes[i].imshow(img)
        
        # 设置标题，显示真实标签、预测标签和概率
        label_map = {0: 'Real', 1: 'Generated'}
        color = 'green' if tl == pl else 'red'
        axes[i].set_title(f'True: {label_map[tl.item() if isinstance(tl, torch.Tensor) else tl]}\n'
                         f'Pred: {label_map[pl.item() if isinstance(pl, torch.Tensor) else pl]} '
                         f'({ps:.2f})', color=color)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(true_labels, pred_labels, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        true_labels: 真实标签，形状为 [N]
        pred_labels: 预测标签，形状为 [N]
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Generated'], 
                yticklabels=['Real', 'Generated'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(true_labels, prob_scores, save_path=None):
    """
    绘制ROC曲线
    
    Args:
        true_labels: 真实标签，形状为 [N]
        prob_scores: 预测概率，形状为 [N]
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_labels, prob_scores)
    
    # 计算AUC
    auc = np.trapz(tpr, fpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')  # 随机猜测的基准线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_hash_embeddings(embeddings, labels, method='tsne', n_components=2, save_path=None):
    """
    可视化哈希嵌入
    
    Args:
        embeddings: 嵌入向量，形状为 [N, embedding_dim]
        labels: 标签，形状为 [N]
        method: 降维方法，'tsne' 或 'pca'
        n_components: 降维后的维度
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 将嵌入向量转换为numpy数组
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    # 应用降维
    if method == 'tsne':
        reduced_embeddings = TSNE(n_components=n_components, random_state=42).fit_transform(embeddings)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reduced_embeddings = PCA(n_components=n_components).fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 绘制降维后的嵌入
    plt.figure(figsize=(10, 8))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            reduced_embeddings[idx, 0], 
            reduced_embeddings[idx, 1], 
            c=[colors[i]], 
            label=f'Class {label}', 
            alpha=0.7
        )
    
    plt.title(f'Hash Embeddings ({method.upper()})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_retrieval_results(query_image, retrieved_images, retrieved_labels, query_label,
                               retrieved_dists=None, top_k=5, save_path=None):
    """
    可视化图像检索结果
    
    Args:
        query_image: 查询图像，形状为 [C, H, W] 或 PIL图像
        retrieved_images: 检索到的图像列表，每个元素形状为 [C, H, W] 或 PIL图像
        retrieved_labels: 检索到的图像标签列表
        query_label: 查询图像的标签
        retrieved_dists: 检索到的图像与查询图像的距离列表，可选
        top_k: 显示的检索结果数量
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 确保图像是PIL格式
    if isinstance(query_image, torch.Tensor):
        query_image = transforms.ToPILImage()(query_image)
    
    retrieved_pil = []
    for img in retrieved_images[:top_k]:
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        retrieved_pil.append(img)
    
    # 创建图表
    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
    
    # 显示查询图像
    axes[0].imshow(query_image)
    axes[0].set_title(f'Query\nClass: {query_label}')
    axes[0].axis('off')
    
    # 显示检索结果
    for i in range(top_k):
        if i < len(retrieved_pil):
            axes[i+1].imshow(retrieved_pil[i])
            
            title = f'Rank {i+1}\nClass: {retrieved_labels[i]}'
            if retrieved_dists is not None and i < len(retrieved_dists):
                title += f'\nDist: {retrieved_dists[i]:.2f}'
                
            color = 'green' if retrieved_labels[i] == query_label else 'red'
            axes[i+1].set_title(title, color=color)
            axes[i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_gram_features(features, save_path=None):
    """
    可视化Gram特征
    
    Args:
        features: Gram特征矩阵，形状为 [B, C, C]
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 如果是批次，只取第一个样本
    if len(features.shape) == 3:
        feature = features[0]
    else:
        feature = features
    
    # 转换为numpy数组
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    
    # 绘制Gram矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature, cmap='viridis')
    plt.title('Gram Matrix Visualization')
    plt.xlabel('Channel i')
    plt.ylabel('Channel j')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_frequency_features(feature_map, use_log=True, save_path=None):
    """
    可视化频域特征
    
    Args:
        feature_map: 频域特征图，形状为 [C, H, W]
        use_log: 是否使用对数变换增强可视化效果
        save_path: 保存图表的路径，如果为None则显示图表
    """
    # 转换为numpy数组
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().numpy()
    
    # 如果是多通道，取均值或第一个通道
    if len(feature_map.shape) == 3:
        feature_map = feature_map[0]  # 取第一个通道
    
    # 对数变换增强可视化效果
    if use_log:
        feature_map = np.log(np.abs(feature_map) + 1)
    
    # 绘制频域特征
    plt.figure(figsize=(8, 6))
    plt.imshow(feature_map, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Frequency Domain Feature')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_comparison_grid(real_images, gen_images, grid_size=(3, 2), save_path=None):
    """
    创建真实图像和生成图像的比较网格
    
    Args:
        real_images: 真实图像列表，每个元素形状为 [C, H, W] 或 PIL图像
        gen_images: 生成图像列表，每个元素形状为 [C, H, W] 或 PIL图像
        grid_size: 网格大小，(行，列)
        save_path: 保存图表的路径，如果为None则显示图表
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(real_images) and idx < len(gen_images):
                # 获取图像
                real_img = real_images[idx]
                gen_img = gen_images[idx]
                
                # 转换为PIL图像
                if isinstance(real_img, torch.Tensor):
                    real_img = transforms.ToPILImage()(real_img)
                if isinstance(gen_img, torch.Tensor):
                    gen_img = transforms.ToPILImage()(gen_img)
                
                # 在一个单元格中显示两张图像
                ax = axes[i, j]
                
                # 创建组合图像
                width = max(real_img.width, gen_img.width)
                height = real_img.height + gen_img.height
                comparison = Image.new('RGB', (width, height))
                comparison.paste(real_img, (0, 0))
                comparison.paste(gen_img, (0, real_img.height))
                
                ax.imshow(comparison)
                ax.set_title(f'Real (top) vs Generated (bottom)')
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_training_info(epoch, model, optimizer, train_loss, val_loss, 
                       train_metrics, val_metrics, save_path):
    """
    保存训练信息和模型检查点
    
    Args:
        epoch: 当前训练轮次
        model: 模型
        optimizer: 优化器
        train_loss: 训练损失
        val_loss: 验证损失
        train_metrics: 训练指标字典
        val_metrics: 验证指标字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }, save_path)
