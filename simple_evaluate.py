#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# 导入哈希模型
from models.hash.hash_net import MultiFeatureHashNet


def parse_args():
    parser = argparse.ArgumentParser(description='简化版评估哈希模型')
    parser.add_argument('--model_path', type=str, default='results/cifar10/final_model.pth',
                        help='模型路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
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


def compute_hamming_distance(query_codes, database_codes):
    """
    计算查询哈希码与数据库哈希码之间的汉明距离
    """
    query_codes = query_codes.float()
    database_codes = database_codes.float()
    
    xor_result = torch.logical_xor(query_codes.unsqueeze(1), database_codes.unsqueeze(0)).float()
    distances = torch.sum(xor_result, dim=2)
    
    return distances


def compute_map(query_codes, database_codes, query_labels, database_labels, top_k=None):
    """
    计算平均精度均值(mAP)
    """
    num_query = query_codes.size(0)
    num_database = database_codes.size(0)
    
    if top_k is None:
        top_k = num_database
    else:
        top_k = min(top_k, num_database)
    
    hamming_distances = compute_hamming_distance(query_codes, database_codes)
    _, indices = torch.sort(hamming_distances, dim=1)
    indices = indices[:, :top_k]
    
    APs = []
    
    for i in range(num_query):
        query_label = query_labels[i]
        retrieved_labels = database_labels[indices[i]]
        
        relevant = (retrieved_labels == query_label).float()
        
        if torch.sum(relevant) == 0:
            APs.append(0)
            continue
        
        cumulative_relevant = torch.cumsum(relevant, dim=0)
        positions = torch.arange(1, top_k + 1, device=relevant.device).float()
        precision = cumulative_relevant / positions
        precision_at_relevant = precision * relevant
        AP = torch.sum(precision_at_relevant) / torch.sum(relevant)
        APs.append(AP.item())
    
    mAP = np.mean(APs)
    return mAP


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载CIFAR-10测试集
    print("加载CIFAR-10测试集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"成功加载了 {len(test_dataset)} 个测试样本")
    except Exception as e:
        print(f"加载CIFAR-10数据集失败: {e}")
        return
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        model = MultiFeatureHashNet(
            hash_dim=args.hash_dim,
            fusion_mode=args.fusion_mode
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 提取哈希码
    print("从测试集提取哈希码...")
    all_hash_codes = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data, labels = data.to(device), labels.to(device)
            
            try:
                hash_code, _, _ = model(data)
                all_hash_codes.append(hash_code)
                all_labels.append(labels)
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    if not all_hash_codes:
        print("没有成功提取任何哈希码，无法评估")
        return
        
    all_hash_codes = torch.cat(all_hash_codes)
    all_labels = torch.cat(all_labels)
    
    # 生成二进制哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    # 计算性能指标
    print("\n计算检索性能...")
    
    # 随机选择一部分作为查询（为了提高计算效率）
    num_samples = min(1000, len(binary_codes))
    indices = torch.randperm(len(binary_codes))[:num_samples]
    
    query_codes = binary_codes[indices]
    query_labels = all_labels[indices]
    
    # 计算mAP和P@k
    mAP = compute_map(query_codes, binary_codes, query_labels, all_labels)
    p_at_10 = compute_map(query_codes, binary_codes, query_labels, all_labels, top_k=10)
    p_at_100 = compute_map(query_codes, binary_codes, query_labels, all_labels, top_k=100)
    
    print(f"\n=== 模型性能 ===")
    print(f"mAP: {mAP:.4f}")
    print(f"P@10: {p_at_10:.4f}")
    print(f"P@100: {p_at_100:.4f}")
    
    # 分析哈希码平衡性
    bit_means = torch.mean(binary_codes, dim=0).cpu().numpy()
    overall_deviation = np.mean(np.abs(bit_means - 0.5))
    
    print("\n=== 哈希码质量 ===")
    print(f"位平衡度偏差 (理想值为0): {overall_deviation:.4f}")
    
    # 分析不同类别的性能
    unique_labels = torch.unique(all_labels).cpu().numpy()
    print("\n=== 各类别性能 ===")
    
    for label in unique_labels:
        # 获取该类别的样本
        class_indices = (all_labels == label).nonzero(as_tuple=True)[0]
        
        if len(class_indices) == 0:
            continue
            
        class_codes = binary_codes[class_indices]
        class_labels = all_labels[class_indices]
        
        # 随机选择一部分作为查询
        num_query = min(100, len(class_indices))
        query_indices = torch.randperm(len(class_indices))[:num_query]
        
        query_codes = class_codes[query_indices]
        query_labels = class_labels[query_indices]
        
        # 计算mAP
        class_mAP = compute_map(query_codes, binary_codes, query_labels, all_labels)
        print(f"类别 {int(label)}: mAP={class_mAP:.4f}, 样本数={len(class_indices)}")
    
    print(f"\n评估完成")


if __name__ == "__main__":
    main()
