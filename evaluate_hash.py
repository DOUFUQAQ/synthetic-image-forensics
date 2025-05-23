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
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 导入自定义模块
from models.hash.hash_net import MultiFeatureHashNet


class SimpleImageFolder(torch.utils.data.Dataset):
    """简单的图像文件夹数据集类，直接读取图像文件"""
    def __init__(self, root, split='test', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        if not os.path.exists(self.root):
            raise ValueError(f"目录 {self.root} 不存在")
            
        # 获取所有图像文件
        self.image_files = []
        self.labels = []
        
        # 遍历所有类别目录
        class_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        if not class_dirs:  # 如果没有类别目录，直接搜索图像文件
            files = [f for f in os.listdir(self.root) if self._is_image_file(f)]
            self.image_files = [os.path.join(self.root, f) for f in files]
            self.labels = [0] * len(files)  # 均分配标签0
        else:
            # 遍历每个类别目录
            for class_idx, class_name in enumerate(sorted(class_dirs)):
                class_dir = os.path.join(self.root, class_name)
                files = [f for f in os.listdir(class_dir) if self._is_image_file(f)]
                self.image_files.extend([os.path.join(class_dir, f) for f in files])
                self.labels.extend([class_idx] * len(files))
        
        print(f"找到 {len(self.image_files)} 个图像文件，{len(set(self.labels))} 个类别")
        
        # 默认转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _is_image_file(self, filename):
        """检查文件是否为支持的图像文件"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法打开图像 {img_path}: {e}")
            # 返回一个随机生成的图像
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        # 应用转换
        if self.transform is not None:
            img = self.transform(img)
            
        label = self.labels[idx]
        
        return img, label


def parse_args():
    parser = argparse.ArgumentParser(description='评估哈希模型性能')
    parser.add_argument('--model_path', type=str, default='results/cifar10/final_model.pth',
                        help='模型路径')
    parser.add_argument('--data_dir', type=str, default='datasets/cifar10',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='测试批次大小')
    parser.add_argument('--hash_dim', type=int, default=64,
                        help='哈希码维度')
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


def compute_precision_at_k(query_codes, database_codes, query_labels, database_labels, k=100, batch_size=100):
    """
    计算P@k (Precision at k)，使用分批处理以减少内存消耗
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        query_labels: 查询标签，形状为 [num_query]
        database_labels: 数据库标签，形状为 [num_database]
        k: 检索前k个结果
        batch_size: 分批处理的批量大小，用于减少内存使用
        
    Returns:
        p_at_k: P@k 值
    """
    num_query = query_codes.size(0)
    num_database = database_codes.size(0)
    device = query_codes.device
    
    # 如果数据库样本少于k，则将k设为数据库大小
    k = min(k, num_database)
    
    # 当数据集过大时，限制查询样本数量
    if num_query > 1000:
        print(f"数据集很大 ({num_query} 查询样本)，使用子集计算P@{k}")
        indices = torch.randperm(num_query)[:1000]
        query_codes = query_codes[indices]
        query_labels = query_labels[indices]
        num_query = 1000
    
    # 初始化精度列表
    precision_list = []
    
    # 分批处理查询
    num_batches = int(np.ceil(num_query / batch_size))
    
    for batch_idx in range(num_batches):
        # 获取当前批次的查询样本
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_query)
        batch_query_codes = query_codes[start:end]
        batch_query_labels = query_labels[start:end]
        batch_size_actual = end - start
        
        # 计算批次查询与所有数据库样本的汉明距离
        hamming_distances = compute_hamming_distance(batch_query_codes, database_codes)
        
        # 按距离排序获取最相似样本的索引
        _, indices = torch.sort(hamming_distances, dim=1)
        indices = indices[:, :k]
        
        # 对于每个查询样本，计算precision
        for i in range(batch_size_actual):
            query_label = batch_query_labels[i]
            retrieved_labels = database_labels[indices[i]]
            
            # 计算相关的结果比例
            relevant = (retrieved_labels == query_label).float()
            precision = torch.mean(relevant).item()
            precision_list.append(precision)
        
        # 释放不需要的张量，减少内存使用
        del hamming_distances, indices
        torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 计算平均精度
    avg_precision = np.mean(precision_list)
    
    return avg_precision


def compute_map(query_codes, database_codes, query_labels, database_labels, top_k=None, batch_size=100):
    """
    计算平均精度均值(mAP)，使用分批处理以减少内存消耗
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]
        database_codes: 数据库哈希码，形状为 [num_database, hash_dim]
        query_labels: 查询标签，形状为 [num_query]
        database_labels: 数据库标签，形状为 [num_database]
        top_k: 只考虑前k个检索结果，如果为None则考虑所有结果
        batch_size: 分批处理的批量大小，用于减少内存使用
        
    Returns:
        mAP: 平均精度均值
    """
    num_query = query_codes.size(0)
    num_database = database_codes.size(0)
    device = query_codes.device
    
    # 如果top_k未指定，则考虑所有数据库样本
    if top_k is None:
        top_k = num_database
    else:
        top_k = min(top_k, num_database)
    
    # 当数据集过大时，限制查询样本数量
    if num_query > 1000:
        print(f"数据集很大 ({num_query} 查询样本)，使用子集计算mAP")
        indices = torch.randperm(num_query)[:1000]
        query_codes = query_codes[indices]
        query_labels = query_labels[indices]
        num_query = 1000
    
    # 初始化AP列表
    APs = []
    
    # 分批处理查询
    num_batches = int(np.ceil(num_query / batch_size))
    
    for batch_idx in range(num_batches):
        # 获取当前批次的查询样本
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_query)
        batch_query_codes = query_codes[start:end]
        batch_query_labels = query_labels[start:end]
        batch_size_actual = end - start
        
        # 计算批次查询与所有数据库样本的汉明距离
        hamming_distances = compute_hamming_distance(batch_query_codes, database_codes)
        
        # 按距离排序获取最相似样本的索引
        _, indices = torch.sort(hamming_distances, dim=1)
        indices = indices[:, :top_k]
        
        # 遍历当前批次的查询样本
        for i in range(batch_size_actual):
            query_label = batch_query_labels[i]
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
            positions = torch.arange(1, top_k + 1, device=device).float()
            
            # 计算精度
            precision = cumulative_relevant / positions
            
            # 只关注相关样本的精度
            precision_at_relevant = precision * relevant
            
            # 计算平均精度 (AP)
            AP = torch.sum(precision_at_relevant) / torch.sum(relevant)
            APs.append(AP.item())
        
        # 释放不需要的张量，减少内存使用
        del hamming_distances, indices
        torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 计算mAP
    mAP = np.mean(APs)
    
    return mAP


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
        p_at_10 = compute_map(truncated_codes, truncated_codes, labels, labels, top_k=10)
        p_at_100 = compute_map(truncated_codes, truncated_codes, labels, labels, top_k=100)
        
        results[num_bits] = {
            'mAP': mAP,
            'P@10': p_at_10,
            'P@100': p_at_100
        }
        
        print(f"哈希码长度 {num_bits}: mAP={mAP:.4f}, P@10={p_at_10:.4f}, P@100={p_at_100:.4f}")
    
    return results


def analyze_per_class_performance(binary_codes, labels):
    """分析特定类别的检索性能"""
    per_class_results = {}
    
    # 获取不同的类别
    unique_labels = torch.unique(labels).cpu().numpy()
    
    for class_idx in unique_labels:
        # 获取该类别的样本索引
        query_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
        
        if len(query_indices) == 0:
            continue
            
        # 使用该类别的样本作为查询
        query_codes = binary_codes[query_indices]
        query_labels = labels[query_indices]
        
        # 计算该类别的mAP和准确率
        mAP = compute_map(query_codes, binary_codes, query_labels, labels)
        p_at_10 = compute_map(query_codes, binary_codes, query_labels, labels, top_k=10)
        
        per_class_results[int(class_idx)] = {
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
    
    # 2. 绘制汉明距离热力图
    plt.figure(figsize=(12, 10))
    plt.imshow(hamming_distances.cpu().numpy(), cmap='viridis')
    plt.colorbar(label='Hamming Distance')
    plt.title('Hamming Distance Matrix')
    plt.savefig(os.path.join(output_dir, 'hamming_distance_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 3. 绘制同类样本和不同类样本之间的汉明距离分布
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
    
    # 4. 绘制哈希位分布
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
    output_dir = os.path.join(args.output_dir, os.path.basename(args.data_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查数据目录是否存在
    data_dir = args.data_dir
    print(f"使用数据目录: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"警告: 指定的数据目录 {data_dir} 不存在")
        print("将尝试使用torchvision下载数据集...")
        
        # 如果指定目录不存在，尝试使用torchvision
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # 尝试使用标准CIFAR10数据集
            print("加载标准CIFAR-10数据集 (不下载)...")
            cifar10_dir = os.path.join(os.path.dirname(data_dir), 'cifar-10-batches-py')
            if os.path.exists(cifar10_dir):
                print(f"检测到现有CIFAR-10数据在 {cifar10_dir}")
                download = False
            else:
                print("未找到现有CIFAR-10数据，将下载...")
                download = True
                
            test_dataset = torchvision.datasets.CIFAR10(
                root=os.path.dirname(data_dir), train=False, download=download, transform=transform)
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            print(f"成功加载了 {len(test_dataset)} 个测试样本")
        except Exception as e:
            print(f"加载 CIFAR-10 数据集失败: {e}")
            return
    else:
        # 如果指定目录存在，使用我们自定义的SimpleImageFolder类
        print(f"数据目录 {data_dir} 存在，使用SimpleImageFolder直接加载图像...")
        try:
            # 使用我们自定义的简单数据集类
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            test_dataset = SimpleImageFolder(data_dir, split='test', transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            print(f"成功加载了 {len(test_dataset)} 个测试样本")
        except Exception as e:
            print(f"加载自定义数据集失败: {e} - {type(e)}")
            print("尝试查看目录结构...")
            
            # 尝试显示目录结构，以帮助诊断
            try:
                print(f"检查 {data_dir} 目录结构:")
                test_dir = os.path.join(data_dir, 'test')
                if os.path.exists(test_dir):
                    print(f"  'test' 目录存在")
                    # 输出子目录
                    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                    if subdirs:
                        print(f"  'test' 目录中的子目录: {subdirs[:5]}..." if len(subdirs) > 5 else f"  'test' 目录中的子目录: {subdirs}")
                    else:
                        print("  'test' 目录中没有子目录")
                        # 查看文件
                        files = os.listdir(test_dir)
                        print(f"  'test' 目录中的文件: {files[:5]}..." if len(files) > 5 else f"  'test' 目录中的文件: {files}")
                else:
                    print(f"  'test' 目录不存在")
            except Exception as e:
                print(f"检查目录结构失败: {e}")
                
            return
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        # 首先加载模型文件
        checkpoint = torch.load(args.model_path, map_location=device)
        print(f"加载了检查点文件")
        
        # 检查是否是检查点字典
        if isinstance(checkpoint, dict):
            # 显示检查点信息
            keys = list(checkpoint.keys())
            print(f"检查点包含以下键: {keys}")
            
            # 如果是完整的检查点，提取model_state_dict
            if 'model_state_dict' in checkpoint:
                print("从检查点中提取模型状态字典")
                model_state = checkpoint['model_state_dict']
                
                # 创建模型实例
                try:
                    model = MultiFeatureHashNet(hash_dim=args.hash_dim)
                except TypeError as e:
                    print(f"初始化模型失败，尝试使用默认参数: {e}")
                    model = MultiFeatureHashNet()
                
                # 加载模型状态
                model.load_state_dict(model_state)
                model = model.to(device)
                model.eval()
                print("模型加载成功")
                
                # 显示额外信息
                if 'epoch' in checkpoint:
                    print(f"模型培训轮数: {checkpoint['epoch']}")
                if 'train_metrics' in checkpoint and isinstance(checkpoint['train_metrics'], dict) and 'mAP' in checkpoint['train_metrics']:
                    print(f"训练集 mAP: {checkpoint['train_metrics']['mAP']:.4f}")
                if 'val_metrics' in checkpoint and isinstance(checkpoint['val_metrics'], dict) and 'mAP' in checkpoint['val_metrics']:
                    print(f"验证集 mAP: {checkpoint['val_metrics']['mAP']:.4f}")
            else:
                # 如果没有model_state_dict键，尝试直接加载
                print("检查点中没有model_state_dict键，尝试直接加载检查点")
                try:
                    model = MultiFeatureHashNet(hash_dim=args.hash_dim)
                    model.load_state_dict(checkpoint)
                    model = model.to(device)
                    model.eval()
                    print("模型直接加载成功")
                except Exception as e:
                    print(f"尝试直接加载检查点失败: {e}")
                    raise e
        else:
            # 如果不是字典，直接尝试加载
            print("加载的不是检查点字典，尝试直接加载")
            model = MultiFeatureHashNet(hash_dim=args.hash_dim)
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()
            print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 详细分析错误
        print("尝试查看模型文件详情...")
        try:
            checkpoint = torch.load(args.model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                # 显示检查点结构
                print(f"检查点类型: {type(checkpoint)}")
                print(f"检查点键: {list(checkpoint.keys())}")
                
                # 检查model_state_dict的结构
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    print(f"model_state_dict类型: {type(model_state)}")
                    print(f"model_state_dict中的键数量: {len(model_state.keys()) if isinstance(model_state, dict) else 'N/A'}")
                    
                    # 打印一些键作为样本
                    if isinstance(model_state, dict) and len(model_state) > 0:
                        sample_keys = list(model_state.keys())[:5]
                        print(f"model_state_dict的样本键: {sample_keys}")
                
                # 如果有模型构造参数，尝试显示
                if 'model_config' in checkpoint:
                    print(f"model_config: {checkpoint['model_config']}")
        except Exception as detailed_e:
            print(f"详细分析模型文件时出错: {detailed_e}")
        
        return
    
    # 评估模型
    print("在测试集上评估模型...")
    binary_codes, embeddings, labels, feature_info = evaluate_model(model, test_loader, args)
    
    # 1. 在测试集上计算总体性能
    print("\n=== 总体性能评估 ===")
    mAP = compute_map(binary_codes, binary_codes, labels, labels)
    p_at_10 = compute_map(binary_codes, binary_codes, labels, labels, top_k=10)
    p_at_100 = compute_map(binary_codes, binary_codes, labels, labels, top_k=100)
    
    print(f"mAP: {mAP:.4f}")
    print(f"P@10: {p_at_10:.4f}")
    print(f"P@100: {p_at_100:.4f}")
    
    overall_results = {
        'mAP': mAP,
        'P@10': p_at_10,
        'P@100': p_at_100,
        'hash_dim': args.hash_dim
    }
    
    with open(os.path.join(output_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    # 2. 评估不同哈希码长度的性能
    print("\n=== 不同哈希码长度的性能 ===")
    hash_length_results = evaluate_different_hash_lengths(binary_codes, labels, args.hash_dim)
    
    with open(os.path.join(output_dir, 'hash_length_results.json'), 'w') as f:
        json.dump({str(k): v for k, v in hash_length_results.items()}, f, indent=4)
    
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
    per_class_results = analyze_per_class_performance(binary_codes, labels)
    
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
