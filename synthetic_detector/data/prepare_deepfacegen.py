"""
DeepFaceGen数据集预处理脚本
用于从大型DeepFaceGen数据集中抽取子集，并进行必要的预处理
"""

import os
import sys
import argparse
import random
import shutil
from tqdm import tqdm
from PIL import Image
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare DeepFaceGen dataset subset')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='源数据集目录(原始DeepFaceGen数据集路径)')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='目标目录(处理后的数据集保存路径)')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='采样比例(0-1之间的小数)')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每个类别的最大样本数')
    parser.add_argument('--img_size', type=int, default=256,
                        help='调整图像大小')
    parser.add_argument('--resize', action='store_true',
                        help='是否调整图像大小')
    parser.add_argument('--use_all', action='store_true',
                        help='使用所有图像，不进行采样')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='处理线程数')
    return parser.parse_args()

def process_image(src_path, dst_path, resize=False, img_size=256):
    """处理单个图像：复制并可选地调整大小"""
    try:
        if resize:
            img = Image.open(src_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img.save(dst_path, quality=95)
        else:
            shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"处理图像 {src_path} 时出错: {e}")
        return False

def prepare_dataset(args):
    """准备数据集子集"""
    random.seed(args.seed)
    
    # 确保目标目录存在
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # 创建真实和生成图像的目录
    real_dir = os.path.join(args.dst_dir, 'real')
    fake_dir = os.path.join(args.dst_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # 处理真实图像
    real_src_dir = os.path.join(args.src_dir, 'real')
    if not os.path.exists(real_src_dir):
        print(f"警告: 真实图像目录 {real_src_dir} 不存在")
    else:
        print(f"处理真实图像...")
        process_class(real_src_dir, real_dir, args)
    
    # 处理生成图像
    fake_src_dir = os.path.join(args.src_dir, 'fake')
    if not os.path.exists(fake_src_dir):
        print(f"警告: 生成图像目录 {fake_src_dir} 不存在")
        
        # 尝试查找其他可能的生成图像目录
        possible_fake_dirs = [d for d in os.listdir(args.src_dir) 
                           if os.path.isdir(os.path.join(args.src_dir, d)) and d != 'real']
        
        if possible_fake_dirs:
            print(f"找到可能的生成图像目录: {possible_fake_dirs}")
            # 合并所有生成图像子目录
            for subdir in possible_fake_dirs:
                subdir_path = os.path.join(args.src_dir, subdir)
                print(f"处理生成图像子目录: {subdir}...")
                process_class(subdir_path, fake_dir, args)
        else:
            print("未找到生成图像目录")
    else:
        print(f"处理生成图像...")
        process_class(fake_src_dir, fake_dir, args)
    
    # 打印统计信息
    real_count = len([f for f in os.listdir(real_dir) 
                    if os.path.isfile(os.path.join(real_dir, f))])
    fake_count = len([f for f in os.listdir(fake_dir) 
                    if os.path.isfile(os.path.join(fake_dir, f))])
    
    print("\n数据集准备完成!")
    print(f"真实图像: {real_count}")
    print(f"生成图像: {fake_count}")
    print(f"总样本数: {real_count + fake_count}")
    print(f"数据集保存在: {args.dst_dir}")

def process_class(src_class_dir, dst_class_dir, args):
    """处理单个类别的图像"""
    # 获取所有图像文件
    image_files = [f for f in os.listdir(src_class_dir) 
                  if os.path.isfile(os.path.join(src_class_dir, f)) and 
                  f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # 随机采样 - 如果不是使用所有图像
    if not args.use_all:
        if args.ratio < 1.0:
            num_samples = min(int(len(image_files) * args.ratio), args.max_samples)
            image_files = random.sample(image_files, num_samples)
        elif len(image_files) > args.max_samples:
            image_files = random.sample(image_files, args.max_samples)
    
    print(f"处理 {len(image_files)} 个图像...")
    
    # 使用线程池并行处理图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for img_file in image_files:
            src_path = os.path.join(src_class_dir, img_file)
            dst_path = os.path.join(dst_class_dir, img_file)
            future = executor.submit(
                process_image, src_path, dst_path, args.resize, args.img_size
            )
            futures.append(future)
        
        # 显示进度条
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

def main():
    args = parse_args()
    prepare_dataset(args)

if __name__ == '__main__':
    main()
