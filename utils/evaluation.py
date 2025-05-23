#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm


def compute_hamming_distance(query_codes, database_codes):
    """
    计算查询哈希码与数据库哈希码之间的汉明距离
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        
    Returns:
        distances: 汉明距离矩阵，形状为 [num_query, num_database]
    """
    # 确保输入是二值的
    query_codes = query_codes.float()
    database_codes = database_codes.float()
    
    # 计算异或 (XOR) 结果，相同位为0，不同位为1
    xor_result = torch.logical_xor(query_codes.unsqueeze(1), database_codes.unsqueeze(0)).float()
    
    # 沿着哈希维度求和，得到汉明距离
    distances = torch.sum(xor_result, dim=2)
    
    return distances


def compute_precision_at_k(query_codes, database_codes, query_labels, database_labels, k=100):
    """
    计算P@k (Precision at k)
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        query_labels: 查询标签，形状为 [num_query]
        database_labels: 数据库标签，形状为 [num_database]
        k: 检索前k个结果
        
    Returns:
        p_at_k: P@k 值
    """
    num_query = query_codes.size(0)
    
    # 如果数据库样本少于k，则将k设为数据库大小
    k = min(k, database_codes.size(0))
    
    # 计算汉明距离
    hamming_distances = compute_hamming_distance(query_codes, database_codes)
    
    # 按距离排序获取前k个最相似样本的索引
    _, indices = torch.sort(hamming_distances, dim=1)
    indices = indices[:, :k]
    
    # 获取查询样本对应的标签
    query_labels = query_labels.view(-1, 1)
    
    # 获取检索到的样本标签
    retrieved_labels = torch.gather(database_labels.repeat(num_query, 1), 1, indices)
    
    # 检查检索到的样本是否与查询样本属于同一类
    relevant = (retrieved_labels == query_labels).float()
    
    # 计算每个查询的P@k
    precision_at_k = torch.mean(relevant, dim=1)
    
    # 计算所有查询的平均P@k
    avg_precision_at_k = torch.mean(precision_at_k).item()
    
    return avg_precision_at_k


def compute_map(query_codes, database_codes, query_labels, database_labels, top_k=None):
    """
    计算平均精度均值(mAP)
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        query_labels: 查询标签，形状为 [num_query]
        database_labels: 数据库标签，形状为 [num_database]
        top_k: 只考虑前k个检索结果，如果为None则考虑所有结果
        
    Returns:
        mAP: 平均精度均值
    """
    num_query = query_codes.size(0)
    num_database = database_codes.size(0)
    
    # 如果top_k未指定，则考虑所有数据库样本
    if top_k is None:
        top_k = num_database
    else:
        top_k = min(top_k, num_database)
    
    # 计算汉明距离
    hamming_distances = compute_hamming_distance(query_codes, database_codes)
    
    # 按距离排序获取最相似样本的索引
    _, indices = torch.sort(hamming_distances, dim=1)
    indices = indices[:, :top_k]
    
    # 初始化AP列表
    APs = []
    
    # 遍历每个查询样本
    for i in range(num_query):
        # 获取当前查询的标签
        query_label = query_labels[i]
        
        # 获取检索结果的标签
        retrieved_labels = database_labels[indices[i]]
        
        # 找出相关样本
        relevant = (retrieved_labels == query_label).float()
        
        # 如果没有相关样本，则AP为0
        if torch.sum(relevant) == 0:
            APs.append(0)
            continue
        
        # 计算累积相关样本数量
        cumulative_relevant = torch.cumsum(relevant, dim=0)
        
        # 计算位置权重
        positions = torch.arange(1, top_k + 1, device=relevant.device).float()
        
        # 计算精度
        precision = cumulative_relevant / positions
        
        # 只关注相关样本的精度
        precision_at_relevant = precision * relevant
        
        # 计算平均精度 (AP)
        AP = torch.sum(precision_at_relevant) / torch.sum(relevant)
        APs.append(AP.item())
    
    # 计算mAP
    mAP = np.mean(APs)
    
    return mAP


def visualize_retrieval_results(query_image, retrieved_images, retrieved_labels, query_label, 
                               hamming_distances=None, save_path=None):
    """
    可视化检索结果
    
    Args:
        query_image: 查询图像
        retrieved_images: 检索到的图像列表
        retrieved_labels: 检索到的图像标签
        query_label: 查询图像标签
        hamming_distances: 汉明距离（可选）
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    
    # 设置图像大小
    fig = plt.figure(figsize=(15, 8))
    
    # 创建网格
    gs = gridspec.GridSpec(2, 5)
    
    # 显示查询图像
    ax = plt.subplot(gs[0, 2])
    ax.imshow(query_image)
    ax.set_title(f'Query (Class {query_label})')
    ax.axis('off')
    
    # 显示检索结果
    for i, (img, label, dist) in enumerate(zip(retrieved_images, retrieved_labels, 
                                             hamming_distances if hamming_distances else [None]*len(retrieved_images))):
        if i >= 10:
            break
            
        row, col = (i // 5) + 1, i % 5
        ax = plt.subplot(gs[row, col])
        ax.imshow(img)
        
        # 根据标签是否匹配设置边框颜色
        if label == query_label:
            title_color = 'green'
        else:
            title_color = 'red'
            
        # 设置标题
        if dist is not None:
            title = f'Class {label}\nDist: {dist:.1f}'
        else:
            title = f'Class {label}'
            
        ax.set_title(title, color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_hash_bit_importance(query_codes, database_codes, query_labels, database_labels):
    """
    分析不同哈希位的重要性
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        query_labels: 查询标签，形状为 [num_query]
        database_labels: 数据库标签，形状为 [num_database]
        
    Returns:
        bit_importance: 每个哈希位的重要性得分
    """
    hash_dim = query_codes.size(1)
    device = query_codes.device
    
    # 初始化每个位的重要性得分
    bit_importance = torch.zeros(hash_dim, device=device)
    
    # 计算原始mAP
    original_map = compute_map(query_codes, database_codes, query_labels, database_labels)
    
    # 逐位翻转并测量mAP变化
    for bit in range(hash_dim):
        # 复制查询码
        flipped_codes = query_codes.clone()
        
        # 翻转当前位
        flipped_codes[:, bit] = 1 - flipped_codes[:, bit]
        
        # 计算翻转后的mAP
        flipped_map = compute_map(flipped_codes, database_codes, query_labels, database_labels)
        
        # 重要性 = 原始mAP - 翻转后的mAP
        bit_importance[bit] = original_map - flipped_map
    
    return bit_importance
