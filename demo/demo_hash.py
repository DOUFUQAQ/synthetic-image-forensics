import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hash.hash_net import MultiFeatureHashNet, get_binary_hash_codes


def preprocess_image(image_path, img_size=224):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
        img_size: 目标图像大小
        
    Returns:
        image_tensor: 预处理后的图像张量
        original_image: 原始PIL图像
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return image_tensor, image


def visualize_retrieval(query_image, retrieved_images, distances, save_path=None):
    """
    可视化检索结果
    
    Args:
        query_image: 查询图像
        retrieved_images: 检索到的图像列表
        distances: 汉明距离列表
        save_path: 保存路径，如果为None则显示图像
    """
    n_results = len(retrieved_images)
    plt.figure(figsize=(15, 3))
    
    # 显示查询图像
    plt.subplot(1, n_results + 1, 1)
    plt.imshow(query_image)
    plt.title('查询图像', fontsize=12)
    plt.axis('off')
    
    # 显示检索结果
    for i, (img, dist) in enumerate(zip(retrieved_images, distances)):
        plt.subplot(1, n_results + 1, i + 2)
        plt.imshow(img)
        plt.title(f'结果 #{i+1}\n距离: {dist}', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_hash_code(model, image_tensor, device):
    """
    生成图像哈希码
    
    Args:
        model: 哈希模型
        image_tensor: 输入图像张量
        device: 计算设备
        
    Returns:
        hash_code: 二进制哈希码
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        hash_code, _ = model(image_tensor)
        
        # 转换为二进制哈希码
        binary_code = get_binary_hash_codes(hash_code)
    
    return binary_code


def create_database_hash_codes(model, database_dir, img_size, device):
    """
    为数据库中的所有图像创建哈希码
    
    Args:
        model: 哈希模型
        database_dir: 数据库目录
        img_size: 图像大小
        device: 计算设备
        
    Returns:
        db_codes: 数据库哈希码列表
        db_images: 数据库图像列表
        db_paths: 数据库图像路径列表
    """
    db_codes = []
    db_images = []
    db_paths = []
    
    # 获取目录中的所有图像
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(database_dir)
        if os.path.isfile(os.path.join(database_dir, f)) and 
        os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print(f'数据库目录 {database_dir} 中没有有效图像文件')
        return [], [], []
    
    print(f'正在处理数据库中的 {len(image_files)} 张图像...')
    
    for i, filename in enumerate(image_files):
        try:
            # 预处理图像
            img_path = os.path.join(database_dir, filename)
            image_tensor, original_image = preprocess_image(img_path, img_size)
            
            # 生成哈希码
            hash_code = generate_hash_code(model, image_tensor, device)
            
            # 保存结果
            db_codes.append(hash_code.squeeze(0))
            db_images.append(original_image)
            db_paths.append(img_path)
            
            if (i + 1) % 10 == 0:
                print(f'已处理 {i+1}/{len(image_files)} 张图像')
                
        except Exception as e:
            print(f'处理图像 {filename} 时出错: {e}')
    
    # 转换为张量
    if db_codes:
        db_codes = torch.stack(db_codes)
    
    return db_codes, db_images, db_paths


def retrieve_similar_images(query_code, db_codes, db_images, top_k=5):
    """
    检索相似图像
    
    Args:
        query_code: 查询哈希码
        db_codes: 数据库哈希码
        db_images: 数据库图像
        top_k: 返回的最相似图像数量
        
    Returns:
        similar_images: 相似图像列表
        distances: 汉明距离列表
    """
    # 计算汉明距离
    hamming_dists = torch.sum(query_code.unsqueeze(0) != db_codes, dim=1)
    
    # 获取最相似的图像
    values, indices = torch.topk(hamming_dists, k=min(top_k, len(db_images)), largest=False)
    
    similar_images = [db_images[i] for i in indices]
    distances = values.tolist()
    
    return similar_images, distances


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建结果目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    try:
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
        print(f'成功加载模型: {args.model_path}')
    except Exception as e:
        print(f'加载模型时出错: {e}')
        return
    
    # 创建数据库哈希码
    print('正在为数据库创建哈希码...')
    db_codes, db_images, db_paths = create_database_hash_codes(
        model, args.database_dir, args.img_size, device
    )
    
    if not db_codes:
        print('数据库为空，无法执行检索')
        return
    
    print(f'数据库包含 {len(db_codes)} 张图像')
    
    # 处理查询图像
    if not os.path.isfile(args.image_path):
        print(f'查询图像不存在: {args.image_path}')
        return
    
    print(f'正在处理查询图像: {args.image_path}')
    
    try:
        # 预处理查询图像
        query_tensor, query_image = preprocess_image(args.image_path, args.img_size)
        
        # 记录开始时间
        start_time = time.time()
        
        # 生成查询哈希码
        query_code = generate_hash_code(model, query_tensor, device)
        
        # 检索相似图像
        similar_images, distances = retrieve_similar_images(
            query_code.squeeze(0), db_codes, db_images, args.top_k
        )
        
        # 记录结束时间
        end_time = time.time()
        
        # 计算检索时间（毫秒）
        retrieval_time = (end_time - start_time) * 1000
        print(f'检索时间: {retrieval_time:.2f} 毫秒')
        
        # 打印结果
        print(f'找到 {len(similar_images)} 个相似图像:')
        for i, (img, dist) in enumerate(zip(similar_images, distances)):
            print(f'结果 #{i+1}: 汉明距离 = {dist}')
        
        # 可视化结果
        if args.output_dir:
            save_path = os.path.join(args.output_dir, 'retrieval_results.png')
            visualize_retrieval(query_image, similar_images, distances, save_path)
            print(f'结果已保存至: {save_path}')
        else:
            visualize_retrieval(query_image, similar_images, distances)
        
    except Exception as e:
        print(f'处理查询图像时出错: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像哈希检索演示')
    
    parser.add_argument('--image_path', type=str, required=True,
                       help='查询图像路径')
    parser.add_argument('--database_dir', type=str, required=True,
                       help='数据库图像目录')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       help='骨干网络架构 (resnet18, resnet34, mobilenet_v2)')
    parser.add_argument('--hash_dim', type=int, default=64,
                       help='哈希码维度')
    parser.add_argument('--fusion_type', type=str, default='late',
                       choices=['early', 'late', 'hybrid'],
                       help='特征融合类型')
    parser.add_argument('--img_size', type=int, default=224,
                       help='输入图像大小')
    parser.add_argument('--top_k', type=int, default=5,
                       help='返回的最相似图像数量')
    parser.add_argument('--output_dir', type=str, default='',
                       help='保存结果的目录，如果为空则显示结果')
    
    args = parser.parse_args()
    
    main(args)
