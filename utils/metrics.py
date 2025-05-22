import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, average_precision_score, precision_recall_curve


def compute_detection_metrics(preds, targets, probs=None):
    """
    计算图像生成检测的评估指标
    
    Args:
        preds: 预测类别，形状为 [N]
        targets: 真实标签，形状为 [N]
        probs: 预测概率，形状为 [N, num_classes]，可选
        
    Returns:
        metrics: 评估指标字典
    """
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(targets, preds)
    metrics['precision'] = precision_score(targets, preds, average='binary', zero_division=0)
    metrics['recall'] = recall_score(targets, preds, average='binary', zero_division=0)
    metrics['f1'] = f1_score(targets, preds, average='binary', zero_division=0)
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(targets, preds)
    metrics['true_negatives'] = conf_matrix[0, 0]
    metrics['false_positives'] = conf_matrix[0, 1]
    metrics['false_negatives'] = conf_matrix[1, 0]
    metrics['true_positives'] = conf_matrix[1, 1]
    
    # 如果提供了概率，计算AUC和AP
    if probs is not None:
        # 检查probs是一维还是二维数组
        if len(probs.shape) > 1 and probs.shape[1] == 2:  # 二分类
            probs = probs[:, 1]  # 取第二类（生成图像）的概率
        
        metrics['auc'] = roc_auc_score(targets, probs)
        metrics['ap'] = average_precision_score(targets, probs)
    
    return metrics


def compute_detection_metrics_batch(outputs, targets):
    """
    计算一个批次的检测评估指标
    
    Args:
        outputs: 模型输出，形状为 [B, num_classes]
        targets: 真实标签，形状为 [B]
        
    Returns:
        metrics: 评估指标字典
    """
    # 转换为numpy数组进行评估
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    
    return compute_detection_metrics(preds, targets, probs)


def compute_hash_metrics(query_codes, db_codes, query_labels, db_labels, k_values=[10, 100, 1000]):
    """
    计算图像哈希的检索性能指标
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]，值为0或1
        db_codes: 数据库哈希码，形状为 [num_db, hash_dim]，值为0或1
        query_labels: 查询标签，形状为 [num_query]
        db_labels: 数据库标签，形状为 [num_db]
        k_values: 计算Precision@K的K值列表
        
    Returns:
        metrics: 评估指标字典
    """
    metrics = {}
    num_query = query_codes.size(0)
    num_db = db_codes.size(0)
    
    # 计算汉明距离
    hamming_dists = torch.zeros(num_query, num_db)
    for i in range(num_query):
        hamming_dists[i] = torch.sum(query_codes[i] != db_codes, dim=1)
    
    # 计算mAP
    APs = []
    precisions = {k: [] for k in k_values}
    
    for i in range(num_query):
        # 获取与查询i相同类别的数据库项
        query_label = query_labels[i]
        relevant = (db_labels == query_label)
        num_relevant = relevant.sum().item()
        
        if num_relevant == 0:
            continue
            
        # 按汉明距离排序
        _, indices = torch.sort(hamming_dists[i])
        
        # 计算累积正确数和精度
        is_relevant = relevant[indices]
        cumulative_relevant = torch.cumsum(is_relevant, dim=0)
        precision_at_k = cumulative_relevant.float() / torch.arange(1, len(indices) + 1, device=is_relevant.device)
        
        # 计算AP
        AP = torch.sum(precision_at_k * is_relevant) / num_relevant
        APs.append(AP.item())
        
        # 计算Precision@K
        for k in k_values:
            if k <= len(indices):
                p_at_k = cumulative_relevant[k-1].float() / k
                precisions[k].append(p_at_k.item())
    
    # 计算mAP
    metrics['mAP'] = sum(APs) / len(APs) if APs else 0.0
    
    # 计算平均Precision@K
    for k in k_values:
        metrics[f'P@{k}'] = sum(precisions[k]) / len(precisions[k]) if precisions[k] else 0.0
    
    # 计算汉明半径为2范围内的检索率
    hamming_radius_2 = []
    for i in range(num_query):
        # 计算汉明距离小于等于2的检索项数量
        in_radius = (hamming_dists[i] <= 2).sum().item()
        hamming_radius_2.append(in_radius / num_db)
    
    metrics['hamming_radius_2'] = sum(hamming_radius_2) / len(hamming_radius_2)
    
    return metrics


def compute_retrieval_time(hash_model, query_images, db_codes, batch_size=32, device='cuda'):
    """
    计算检索时间
    
    Args:
        hash_model: 哈希模型
        query_images: 查询图像，形状为 [num_query, 3, H, W]
        db_codes: 数据库哈希码，形状为 [num_db, hash_dim]
        batch_size: 批处理大小
        device: 计算设备
        
    Returns:
        avg_time: 平均检索时间（毫秒）
    """
    import time
    
    hash_model.eval()
    num_query = query_images.size(0)
    
    # 分批处理查询
    total_time = 0
    with torch.no_grad():
        for i in range(0, num_query, batch_size):
            batch_query = query_images[i:i+batch_size].to(device)
            
            # 记录开始时间
            start_time = time.time()
            
            # 生成哈希码
            hash_codes, _ = hash_model(batch_query)
            binary_codes = (hash_codes > 0).float()
            
            # 计算汉明距离并排序
            for code in binary_codes:
                dists = torch.sum(code.unsqueeze(0) != db_codes, dim=1)
                _, _ = torch.sort(dists)
            
            # 记录结束时间
            end_time = time.time()
            total_time += (end_time - start_time)
    
    # 计算平均检索时间（毫秒）
    avg_time = (total_time / num_query) * 1000
    
    return avg_time
