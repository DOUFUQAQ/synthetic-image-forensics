#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# 导入自定义模块
from models.hash.hash_net import MultiFeatureHashNet
from data.datasets import get_dataset
from utils.evaluation import compute_hamming_distance, visualize_retrieval_results


def parse_args():
    parser = argparse.ArgumentParser(description='可视化图像检索结果')
    parser.add_argument('--model_path', type=str, default='results/cifar10/final_model.pth',
                        help='模型路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='数据集名称 (cifar10, cifar100, imagenet)')
    parser.add_argument('--num_queries', type=int, default=5,
                        help='可视化的查询样本数量')
    parser.add_argument('--num_results', type=int, default=10,
                        help='每个查询显示的检索结果数量')
    parser.add_argument('--hash_dim', type=int, default=64,
                        help='哈希码维度')
    parser.add_argument('--fusion_mode', type=str, default='late',
                        help='特征融合模式 (early, late, hybrid)')
    parser.add_argument('--output_dir', type=str, default='retrieval_visualization',
                        help='输出目录')
    parser.add_argument('--class_id', type=int, default=None,
                        help='指定要查询的类别ID，默认为随机选择')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    return parser.parse_args()


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化的图像转换回原始像素值范围"""
    # 复制防止修改原始数据
    image = image.clone().detach()
    
    # 转换尺寸为 [C, H, W]
    if image.dim() == 4:
        image = image.squeeze(0)
    
    # 反归一化
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    
    # 裁剪到 [0, 1] 范围
    image = torch.clamp(image, 0, 1)
    
    # 转换为 NumPy 数组并调整通道顺序
    image = image.cpu().permute(1, 2, 0).numpy()
    
    return image


def get_binary_hash_codes(hash_codes):
    """将连续哈希码转换为二进制"""
    binary_codes = (hash_codes >= 0).float()
    return binary_codes


def main(args):
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    _, _, test_dataset = get_dataset(args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
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
    all_images = []
    all_hash_codes = []
    all_labels = []
    
    print("提取测试集特征...")
    with torch.no_grad():
        for data, labels in test_loader:
            # 保存原始图像用于可视化
            all_images.extend([img for img in data])
            
            data, labels = data.to(device), labels.to(device)
            hash_code, _, _ = model(data)
            
            all_hash_codes.append(hash_code)
            all_labels.append(labels)
    
    all_hash_codes = torch.cat(all_hash_codes)
    all_labels = torch.cat(all_labels)
    
    # 生成二进制哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    # 选择要查询的样本
    if args.class_id is not None:
        # 从指定类别中选择样本
        class_indices = (all_labels == args.class_id).nonzero(as_tuple=True)[0]
        if len(class_indices) == 0:
            raise ValueError(f"类别 {args.class_id} 在测试集中不存在")
        
        if len(class_indices) < args.num_queries:
            print(f"警告: 类别 {args.class_id} 只有 {len(class_indices)} 个样本，少于请求的 {args.num_queries} 个")
            num_queries = len(class_indices)
        else:
            num_queries = args.num_queries
            
        query_indices = np.random.choice(class_indices.cpu().numpy(), num_queries, replace=False)
    else:
        # 随机选择样本
        query_indices = np.random.choice(len(binary_codes), args.num_queries, replace=False)
    
    # 为每个查询样本可视化检索结果
    for i, idx in enumerate(query_indices):
        query_code = binary_codes[idx].unsqueeze(0)
        query_label = all_labels[idx].item()
        query_image = all_images[idx]
        
        # 计算与所有样本的汉明距离
        hamming_distances = compute_hamming_distance(query_code, binary_codes).squeeze(0)
        
        # 排序并获取最相似的样本
        distances, indices = torch.sort(hamming_distances)
        
        # 获取前N个结果
        top_indices = indices[:args.num_results].cpu().numpy()
        top_distances = distances[:args.num_results].cpu().numpy()
        top_labels = all_labels[top_indices].cpu().numpy()
        
        # 准备可视化图像
        retrieved_images = [denormalize_image(all_images[j]) for j in top_indices]
        query_image_denorm = denormalize_image(query_image)
        
        # 可视化并保存结果
        save_path = os.path.join(output_dir, f'query_{i}_class_{query_label}.png')
        visualize_retrieval_results(
            query_image_denorm, 
            retrieved_images, 
            top_labels, 
            query_label, 
            top_distances,
            save_path
        )
        
        print(f"已保存查询 {i+1}/{len(query_indices)} (类别 {query_label}) 的检索结果")
    
    print(f"\n所有可视化结果已保存到 {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
