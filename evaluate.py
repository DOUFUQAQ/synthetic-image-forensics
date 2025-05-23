#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

# 导入自定义模块
from models.hash.hash_net import MultiFeatureHashNet
from data.datasets import get_dataset
from utils.evaluation import compute_map, compute_precision_at_k, compute_hamming_distance


def parse_args():
    parser = argparse.ArgumentParser(description='评估哈希模型性能')
    parser.add_argument('--model_path', type=str, default='results/cifar10/final_model.pth',
                        help='模型路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='数据集名称 (cifar10, cifar100, imagenet)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='测试批次大小')
    parser.add_argument('--hash_dim', type=int, default=64,
                        help='哈希码维度')
    parser.add_argument('--fusion_mode', type=str, default='late',
                        help='特征融合模式 (early, late, hybrid)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='输出目录')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    return parser.parse_args()


def get_binary_hash_codes(hash_codes):
    """将连续哈希码转换为二进制"""
    binary_codes = (hash_codes >= 0).float()
    return binary_codes


def evaluate_model(model, data_loader, args):
    """在测试集上评估模型性能"""
    model.eval()
    device = next(model.parameters()).device
    
    all_hash_codes = []
    all_embeddings = []
    all_labels = []
    all_feature_info = []
    
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)
            
            # 前向传播
            hash_code, embedding, feature_info = model(data)
            
            all_hash_codes.append(hash_code)
            all_embeddings.append(embedding)
            all_labels.append(labels)
            all_feature_info.append(feature_info)
            
    all_hash_codes = torch.cat(all_hash_codes)
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    
    # 生成二进制哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    return binary_codes, all_embeddings, all_labels, all_feature_info


def evaluate_different_hash_lengths(binary_codes, labels, max_bits=64):
    """评估不同哈希码长度的检索性能"""
    results = {}
    
    for num_bits in [8, 16, 32, 48, 64]:
        if num_bits > max_bits:
            continue
        
        # 选择前num_bits位
        truncated_codes = binary_codes[:, :num_bits]
        
        # 计算mAP和不同k值的准确率
        mAP = compute_map(truncated_codes, truncated_codes, labels, labels)
        p_at_10 = compute_precision_at_k(truncated_codes, truncated_codes, labels, labels, k=10)
        p_at_100 = compute_precision_at_k(truncated_codes, truncated_codes, labels, labels, k=100)
        
        results[num_bits] = {
            'mAP': mAP,
            'P@10': p_at_10,
            'P@100': p_at_100
        }
        
        print(f"哈希码长度 {num_bits}: mAP={mAP:.4f}, P@10={p_at_10:.4f}, P@100={p_at_100:.4f}")
    
    return results


def analyze_per_class_performance(binary_codes, labels, num_classes):
    """分析特定类别的检索性能"""
    per_class_results = {}
    
    for class_idx in range(num_classes):
        # 获取该类别的样本索引
        query_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
        
        if len(query_indices) == 0:
            continue
            
        # 使用该类别的样本作为查询
        query_codes = binary_codes[query_indices]
        query_labels = labels[query_indices]
        
        # 计算该类别的mAP和准确率
        mAP = compute_map(query_codes, binary_codes, query_labels, labels)
        p_at_10 = compute_precision_at_k(query_codes, binary_codes, query_labels, labels, k=10)
        
        per_class_results[class_idx] = {
            'mAP': mAP,
            'P@10': p_at_10,
            'num_samples': len(query_indices)
        }
        
        print(f"类别 {class_idx}: mAP={mAP:.4f}, P@10={p_at_10:.4f}, 样本数={len(query_indices)}")
    
    return per_class_results


def visualize_hash_distribution(binary_codes, labels, output_dir):
    """可视化哈希码分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 计算汉明距离矩阵
    print("计算汉明距离矩阵...")
    num_samples = min(1000, len(binary_codes))  # 限制样本数以提高可视化速度
    sampled_indices = np.random.choice(len(binary_codes), num_samples, replace=False)
    
    sampled_codes = binary_codes[sampled_indices]
    sampled_labels = labels[sampled_indices]
    
    # 计算两两之间的汉明距离
    hamming_distances = torch.zeros((num_samples, num_samples))
    for i in range(num_samples):
        hamming_distances[i] = compute_hamming_distance(sampled_codes[i:i+1], sampled_codes)
    
    # 2. 使用t-SNE将哈希码降维到2D空间
    print("使用t-SNE降维...")
    binary_np = sampled_codes.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    hash_2d = tsne.fit_transform(binary_np)
    
    # 3. 绘制t-SNE散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(hash_2d[:, 0], hash_2d[:, 1], c=sampled_labels.cpu().numpy(), 
                         cmap='tab10', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Hash Codes')
    plt.savefig(os.path.join(output_dir, 'hash_tsne.png'), dpi=300, bbox_inches='tight')
    
    # 4. 绘制汉明距离热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(hamming_distances.cpu().numpy(), cmap='viridis')
    plt.title('Hamming Distance Matrix')
    plt.savefig(os.path.join(output_dir, 'hamming_distance_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 5. 绘制同类样本和不同类样本之间的汉明距离分布
    same_class_distances = []
    diff_class_distances = []
    
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            distance = hamming_distances[i, j].item()
            if sampled_labels[i] == sampled_labels[j]:
                same_class_distances.append(distance)
            else:
                diff_class_distances.append(distance)
    
    plt.figure(figsize=(10, 6))
    plt.hist(same_class_distances, bins=32, alpha=0.5, label='Same Class')
    plt.hist(diff_class_distances, bins=32, alpha=0.5, label='Different Classes')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hamming Distances')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'hamming_distance_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 6. 绘制哈希位分布
    bit_values = binary_codes.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(bit_values)), bit_values)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel('Bit Index')
    plt.ylabel('Average Value')
    plt.title('Distribution of Hash Bit Values (ideally close to 0.5)')
    plt.savefig(os.path.join(output_dir, 'hash_bit_distribution.png'), dpi=300, bbox_inches='tight')
    
    print(f"可视化结果已保存到 {output_dir}")


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
    num_classes = len(test_dataset.classes) if hasattr(test_dataset, 'classes') else 10
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = MultiFeatureHashNet(
        hash_dim=args.hash_dim,
        fusion_mode=args.fusion_mode
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # 评估模型
    print("在测试集上评估模型...")
    binary_codes, embeddings, labels, feature_info = evaluate_model(model, test_loader, args)
    
    # 1. 在测试集上计算总体性能
    print("\n=== 总体性能评估 ===")
    mAP = compute_map(binary_codes, binary_codes, labels, labels)
    p_at_10 = compute_precision_at_k(binary_codes, binary_codes, labels, labels, k=10)
    p_at_100 = compute_precision_at_k(binary_codes, binary_codes, labels, labels, k=100)
    
    print(f"mAP: {mAP:.4f}")
    print(f"P@10: {p_at_10:.4f}")
    print(f"P@100: {p_at_100:.4f}")
    
    overall_results = {
        'mAP': mAP,
        'P@10': p_at_10,
        'P@100': p_at_100,
        'hash_dim': args.hash_dim,
        'fusion_mode': args.fusion_mode
    }
    
    with open(os.path.join(output_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    # 2. 评估不同哈希码长度的性能
    print("\n=== 不同哈希码长度的性能 ===")
    hash_length_results = evaluate_different_hash_lengths(binary_codes, labels, args.hash_dim)
    
    with open(os.path.join(output_dir, 'hash_length_results.json'), 'w') as f:
        json.dump(hash_length_results, f, indent=4)
    
    # 绘制不同长度的性能对比图
    plt.figure(figsize=(10, 6))
    lengths = list(hash_length_results.keys())
    map_values = [hash_length_results[l]['mAP'] for l in lengths]
    p10_values = [hash_length_results[l]['P@10'] for l in lengths]
    p100_values = [hash_length_results[l]['P@100'] for l in lengths]
    
    plt.plot(lengths, map_values, 'o-', label='mAP')
    plt.plot(lengths, p10_values, 's-', label='P@10')
    plt.plot(lengths, p100_values, '^-', label='P@100')
    plt.xlabel('Hash Code Length')
    plt.ylabel('Performance')
    plt.title('Performance vs. Hash Code Length')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'hash_length_performance.png'), dpi=300, bbox_inches='tight')
    
    # 3. 分析特定类别的检索性能
    print("\n=== 特定类别的检索性能 ===")
    per_class_results = analyze_per_class_performance(binary_codes, labels, num_classes)
    
    with open(os.path.join(output_dir, 'per_class_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in per_class_results.items()}, f, indent=4)
    
    # 绘制每个类别的性能条形图
    plt.figure(figsize=(14, 7))
    classes = list(per_class_results.keys())
    map_values = [per_class_results[c]['mAP'] for c in classes]
    p10_values = [per_class_results[c]['P@10'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, map_values, width, label='mAP')
    plt.bar(x + width/2, p10_values, width, label='P@10')
    
    plt.xlabel('Class')
    plt.ylabel('Performance')
    plt.title('Per-Class Retrieval Performance')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
    
    # 4. 可视化哈希码分布
    print("\n=== 可视化哈希码分布 ===")
    visualize_hash_distribution(binary_codes, labels, output_dir)
    
    print(f"\n所有评估结果已保存到 {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
