#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# 导入自定义模块
from models.hash.hash_net import MultiFeatureHashNet
from data.datasets import get_dataset
from utils.evaluation import compute_hamming_distance, compute_map, analyze_hash_bit_importance


def parse_args():
    parser = argparse.ArgumentParser(description='分析哈希码质量')
    parser.add_argument('--model_path', type=str, default='results/cifar10/final_model.pth',
                        help='模型路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='数据集名称 (cifar10, cifar100, imagenet)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--hash_dim', type=int, default=64,
                        help='哈希码维度')
    parser.add_argument('--fusion_mode', type=str, default='late',
                        help='特征融合模式 (early, late, hybrid)')
    parser.add_argument('--output_dir', type=str, default='hash_analysis',
                        help='输出目录')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    return parser.parse_args()


def get_binary_hash_codes(hash_codes):
    """将连续哈希码转换为二进制"""
    binary_codes = (hash_codes >= 0).float()
    return binary_codes


def analyze_bit_balance(binary_codes):
    """分析哈希码位平衡性"""
    # 计算每一位的平均值（理想情况下应该接近0.5）
    bit_means = torch.mean(binary_codes, dim=0)
    
    # 计算每一位与0.5的偏差
    bit_deviation = torch.abs(bit_means - 0.5)
    
    # 计算总体偏差
    overall_deviation = torch.mean(bit_deviation).item()
    
    return bit_means.cpu().numpy(), bit_deviation.cpu().numpy(), overall_deviation


def analyze_bit_independence(binary_codes):
    """分析哈希码位独立性"""
    # 计算位之间的相关性矩阵
    n_bits = binary_codes.size(1)
    correlation_matrix = torch.zeros((n_bits, n_bits))
    
    for i in range(n_bits):
        for j in range(i, n_bits):
            bit_i = binary_codes[:, i]
            bit_j = binary_codes[:, j]
            
            # 计算皮尔逊相关系数
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                # 中心化
                bit_i_centered = bit_i - bit_i.mean()
                bit_j_centered = bit_j - bit_j.mean()
                
                # 计算协方差
                covariance = torch.mean(bit_i_centered * bit_j_centered)
                
                # 计算标准差
                std_i = torch.std(bit_i)
                std_j = torch.std(bit_j)
                
                # 计算相关系数
                if std_i > 0 and std_j > 0:
                    correlation = covariance / (std_i * std_j)
                else:
                    correlation = 0
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
    
    # 计算平均绝对相关系数（不包括对角线）
    mask = ~torch.eye(n_bits, dtype=bool)
    avg_correlation = torch.mean(torch.abs(correlation_matrix[mask])).item()
    
    return correlation_matrix.cpu().numpy(), avg_correlation


def analyze_hamming_distance_distribution(binary_codes, labels):
    """分析汉明距离分布"""
    # 限制样本数以提高计算效率
    max_samples = min(2000, binary_codes.size(0))
    if binary_codes.size(0) > max_samples:
        indices = torch.randperm(binary_codes.size(0))[:max_samples]
        binary_codes = binary_codes[indices]
        labels = labels[indices]
    
    # 计算所有样本对之间的汉明距离
    hamming_distances = compute_hamming_distance(binary_codes, binary_codes)
    
    # 创建标签匹配矩阵
    label_matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    
    # 收集同类和不同类样本之间的距离
    same_class_distances = hamming_distances[label_matches == 1].cpu().numpy()
    diff_class_distances = hamming_distances[label_matches == 0].cpu().numpy()
    
    # 移除自身比较（距离为0）
    same_class_distances = same_class_distances[same_class_distances > 0]
    
    # 计算平均距离
    avg_same_class = np.mean(same_class_distances)
    avg_diff_class = np.mean(diff_class_distances)
    
    # 计算距离差异性
    distance_separability = (avg_diff_class - avg_same_class) / binary_codes.size(1)
    
    return same_class_distances, diff_class_distances, avg_same_class, avg_diff_class, distance_separability


def main(args):
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    _, _, test_dataset = get_dataset(args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = MultiFeatureHashNet(
        hash_dim=args.hash_dim,
        fusion_mode=args.fusion_mode
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 提取所有测试样本的哈希码和标签
    all_hash_codes = []
    all_labels = []
    
    print("提取测试集特征...")
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            hash_code, _, _ = model(data)
            
            all_hash_codes.append(hash_code)
            all_labels.append(labels)
    
    all_hash_codes = torch.cat(all_hash_codes)
    all_labels = torch.cat(all_labels)
    
    # 生成二进制哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    # 1. 分析位平衡性
    print("\n=== 分析哈希位平衡性 ===")
    bit_means, bit_deviation, overall_deviation = analyze_bit_balance(binary_codes)
    
    print(f"总体偏差: {overall_deviation:.4f} (越接近0越好，表示位分布接近0.5)")
    
    # 绘制位平衡性图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bit_means)), bit_means)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel('Bit Index')
    plt.ylabel('Average Value')
    plt.title('Hash Bit Balance (ideally close to 0.5)')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'bit_balance.png'), dpi=300, bbox_inches='tight')
    
    # 2. 分析位独立性
    print("\n=== 分析哈希位独立性 ===")
    correlation_matrix, avg_correlation = analyze_bit_independence(binary_codes)
    
    print(f"平均绝对相关系数: {avg_correlation:.4f} (越接近0越好，表示位之间独立性强)")
    
    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Bit Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'bit_correlation.png'), dpi=300, bbox_inches='tight')
    
    # 3. 分析汉明距离分布
    print("\n=== 分析汉明距离分布 ===")
    same_class_distances, diff_class_distances, avg_same_class, avg_diff_class, distance_separability = \
        analyze_hamming_distance_distribution(binary_codes, all_labels)
    
    print(f"同类样本平均汉明距离: {avg_same_class:.2f}")
    print(f"不同类样本平均汉明距离: {avg_diff_class:.2f}")
    print(f"距离可分性: {distance_separability:.4f} (越大越好)")
    
    # 绘制汉明距离分布图
    plt.figure(figsize=(10, 6))
    plt.hist(same_class_distances, bins=32, alpha=0.5, label=f'Same Class (avg={avg_same_class:.2f})')
    plt.hist(diff_class_distances, bins=32, alpha=0.5, label=f'Different Classes (avg={avg_diff_class:.2f})')
    plt.axvline(x=avg_same_class, color='blue', linestyle='--')
    plt.axvline(x=avg_diff_class, color='orange', linestyle='--')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.title('Hamming Distance Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hamming_distance_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 4. 分析哈希位重要性
    print("\n=== 分析哈希位重要性 ===")
    bit_importance = analyze_hash_bit_importance(binary_codes, binary_codes, all_labels, all_labels)
    bit_importance = bit_importance.cpu().numpy()
    
    # 按重要性排序
    sorted_indices = np.argsort(bit_importance)[::-1]
    sorted_importance = bit_importance[sorted_indices]
    
    # 绘制位重要性图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance)
    plt.xlabel('Bit Index (sorted by importance)')
    plt.ylabel('Importance Score')
    plt.title('Hash Bit Importance')
    plt.savefig(os.path.join(output_dir, 'bit_importance.png'), dpi=300, bbox_inches='tight')
    
    # 显示最重要和最不重要的位
    top_k = min(10, len(sorted_indices))
    print(f"\n最重要的 {top_k} 位:")
    for i in range(top_k):
        print(f"位 {sorted_indices[i]}: 重要性得分 = {sorted_importance[i]:.6f}")
    
    print(f"\n最不重要的 {top_k} 位:")
    for i in range(1, top_k + 1):
        idx = sorted_indices[-i]
        print(f"位 {idx}: 重要性得分 = {bit_importance[idx]:.6f}")
    
    # 5. 保存分析结果
    analysis_results = {
        'bit_balance': {
            'overall_deviation': overall_deviation,
            'bit_means': bit_means.tolist()
        },
        'bit_independence': {
            'avg_correlation': avg_correlation
        },
        'hamming_distance': {
            'avg_same_class': float(avg_same_class),
            'avg_diff_class': float(avg_diff_class),
            'distance_separability': float(distance_separability)
        },
        'bit_importance': {
            'top_bits': [int(sorted_indices[i]) for i in range(top_k)],
            'top_importance': [float(sorted_importance[i]) for i in range(top_k)],
            'bottom_bits': [int(sorted_indices[-i-1]) for i in range(top_k)],
            'bottom_importance': [float(bit_importance[sorted_indices[-i-1]]) for i in range(top_k)]
        }
    }
    
    with open(os.path.join(output_dir, 'hash_analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"\n分析结果已保存到 {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
