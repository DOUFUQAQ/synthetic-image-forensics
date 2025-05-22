import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import json
from PIL import Image
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hash.hash_net import MultiFeatureHashNet, get_binary_hash_codes, compute_map
from data.dataset import HashingDataset
from utils.transforms import get_hashing_transforms
from utils.metrics import compute_hash_metrics, compute_retrieval_time
from utils.visualization import visualize_retrieval_results, plot_hash_embeddings


def evaluate_hash_model(args):
    """
    评估图像哈希模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = MultiFeatureHashNet(
        hash_dim=args.hash_dim,
        backbone_name=args.backbone,
        fusion_type=args.fusion_type,
        pretrained=False
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded model from {args.model_path}')
    
    # 数据变换
    _, test_transform = get_hashing_transforms(img_size=args.img_size)
    
    # 创建查询和数据库数据集
    query_dataset = HashingDataset(
        args.query_dir, 
        transform=test_transform,
        mode='test'
    )
    
    database_dataset = HashingDataset(
        args.database_dir, 
        transform=test_transform,
        mode='database'
    )
    
    query_loader = DataLoader(
        query_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    database_loader = DataLoader(
        database_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Query dataset size: {len(query_dataset)}')
    print(f'Database dataset size: {len(database_dataset)}')
    
    # 提取数据库特征
    print('Extracting database features...')
    db_hash_codes = []
    db_embeddings = []
    db_labels = []
    db_paths = []
    
    with torch.no_grad():
        for images, labels, paths in database_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            hash_codes, embeddings = model(images)
            
            db_hash_codes.append(hash_codes.cpu())
            db_embeddings.append(embeddings.cpu())
            db_labels.append(labels.cpu())
            db_paths.extend(paths)
    
    db_hash_codes = torch.cat(db_hash_codes)
    db_embeddings = torch.cat(db_embeddings)
    db_labels = torch.cat(db_labels)
    
    # 转换为二进制哈希码
    db_binary_codes = get_binary_hash_codes(db_hash_codes)
    
    # 提取查询特征
    print('Extracting query features...')
    query_hash_codes = []
    query_embeddings = []
    query_labels = []
    query_paths = []
    query_images_tensor = []
    
    # 记录查询时间
    total_query_time = 0
    num_queries = 0
    
    with torch.no_grad():
        for images, labels, paths in query_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 记录开始时间
            start_time = time.time()
            
            # 前向传播
            hash_codes, embeddings = model(images)
            
            # 记录结束时间
            end_time = time.time()
            batch_time = (end_time - start_time) * 1000  # 转为毫秒
            total_query_time += batch_time
            num_queries += images.size(0)
            
            query_hash_codes.append(hash_codes.cpu())
            query_embeddings.append(embeddings.cpu())
            query_labels.append(labels.cpu())
            query_paths.extend(paths)
            
            # 保存一些查询图像用于可视化
            if len(query_images_tensor) < args.num_vis_samples:
                query_images_tensor.append(images[:min(args.batch_size, args.num_vis_samples - len(query_images_tensor))].cpu())
    
    # 合并查询结果
    query_hash_codes = torch.cat(query_hash_codes)
    query_embeddings = torch.cat(query_embeddings)
    query_labels = torch.cat(query_labels)
    
    if query_images_tensor:
        query_images_tensor = torch.cat(query_images_tensor)
    
    # 转换为二进制哈希码
    query_binary_codes = get_binary_hash_codes(query_hash_codes)
    
    # 计算平均查询时间
    avg_query_time = total_query_time / num_queries
    print(f'Average query time per image: {avg_query_time:.2f} ms')
    
    # 计算检索时间（包括距离计算和排序）
    if len(query_images_tensor) > 0:
        retrieval_time = compute_retrieval_time(
            model, query_images_tensor[:min(100, len(query_images_tensor))], 
            db_binary_codes, batch_size=args.batch_size, device=device
        )
        print(f'Average retrieval time per image: {retrieval_time:.2f} ms')
    else:
        retrieval_time = 0
    
    # 计算评估指标
    metrics = compute_hash_metrics(
        query_binary_codes, db_binary_codes, 
        query_labels, db_labels,
        k_values=[10, 100, 1000]
    )
    
    # 添加时间指标
    metrics['avg_query_time_ms'] = float(avg_query_time)
    metrics['avg_retrieval_time_ms'] = float(retrieval_time)
    
    # 打印评估结果
    print("\nEvaluation Results:")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"P@10: {metrics['P@10']:.4f}")
    print(f"P@100: {metrics['P@100']:.4f}")
    print(f"P@1000: {metrics['P@1000']:.4f}")
    print(f"Hamming Radius 2 Rate: {metrics['hamming_radius_2']:.4f}")
    
    # 保存评估结果到JSON文件
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 可视化嵌入
    embed_path = os.path.join(args.output_dir, 'hash_embeddings.png')
    # 限制点数，避免图表过于拥挤
    if len(query_embeddings) > 1000:
        indices = np.random.choice(len(query_embeddings), 1000, replace=False)
        plot_hash_embeddings(
            query_embeddings[indices], query_labels[indices], 
            method='tsne', save_path=embed_path
        )
    else:
        plot_hash_embeddings(
            query_embeddings, query_labels, 
            method='tsne', save_path=embed_path
        )
    print(f'Saved embeddings visualization to {embed_path}')
    
    # 可视化检索结果
    if len(query_images_tensor) > 0:
        # 选择一些查询图像进行可视化
        vis_indices = np.random.choice(len(query_images_tensor), min(args.num_vis_samples, len(query_images_tensor)), replace=False)
        
        for i, idx in enumerate(vis_indices):
            if i >= 5:  # 最多可视化5个查询
                break
                
            query_img = query_images_tensor[idx]
            query_label = query_labels[idx]
            query_code = query_binary_codes[idx]
            
            # 计算汉明距离
            hamming_dists = torch.sum(query_code.unsqueeze(0) != db_binary_codes, dim=1)
            
            # 获取最近的图像
            _, indices = torch.sort(hamming_dists)
            top_indices = indices[:10]  # 取前10个结果
            
            # 获取检索结果
            retrieved_paths = [db_paths[j] for j in top_indices]
            retrieved_labels = db_labels[top_indices].numpy()
            retrieved_dists = hamming_dists[top_indices].numpy()
            
            # 加载检索图像
            retrieved_images = []
            for path in retrieved_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    retrieved_images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
            
            # 可视化检索结果
            result_path = os.path.join(args.output_dir, f'retrieval_result_{i}.png')
            visualize_retrieval_results(
                query_img, retrieved_images, retrieved_labels, query_label.item(),
                retrieved_dists=retrieved_dists, top_k=5, save_path=result_path
            )
            print(f'Saved retrieval result {i} to {result_path}')
    
    print(f'All evaluation results saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Hashing Model')
    
    # 数据参数
    parser.add_argument('--query_dir', type=str, required=True,
                       help='Path to query image directory')
    parser.add_argument('--database_dir', type=str, required=True,
                       help='Path to database image directory')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       help='Backbone architecture (resnet18, resnet34, mobilenet_v2)')
    parser.add_argument('--hash_dim', type=int, default=64,
                       help='Hash code dimension')
    parser.add_argument('--fusion_type', type=str, default='late',
                       choices=['early', 'late', 'hybrid'],
                       help='Feature fusion type')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/hash',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_vis_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    evaluate_hash_model(args)
