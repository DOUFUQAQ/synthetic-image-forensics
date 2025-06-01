"""
AIGC数据集组织脚本
将解压后的AIGC数据集整理为模型训练所需的格式
"""

import os
import sys
import argparse
import random
import shutil
from tqdm import tqdm
import concurrent.futures
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='组织AIGC数据集')
    parser.add_argument('--src_dir', type=str, required=True,
                        help='源数据目录（解压后的AIGC数据集路径）')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='目标目录（组织后的数据集保存路径）')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='采样比例(0-1之间的小数)，如果指定了real_ratio和fake_ratio，则忽略此参数')
    parser.add_argument('--real_ratio', type=float, default=None,
                        help='真实图像采样比例(0-1之间的小数)，默认等于--ratio')
    parser.add_argument('--fake_ratio', type=float, default=None,
                        help='生成图像采样比例(0-1之间的小数)，默认等于--ratio')
    parser.add_argument('--real_max', type=int, default=None,
                        help='真实图像的最大样本数，默认等于--max_samples')
    parser.add_argument('--fake_max', type=int, default=None,
                        help='生成图像的最大样本数，默认等于--max_samples')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='每个类别的最大样本数')
    parser.add_argument('--use_symlinks', action='store_true',
                        help='使用符号链接而不是复制文件（节省空间）')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='处理线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()

def get_image_files(directory):
    """获取目录中的所有图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def sample_files(files, ratio, max_samples, seed=42):
    """从文件列表中采样指定比例的文件"""
    random.seed(seed)
    
    # 确定采样数量
    sample_size = min(int(len(files) * ratio), max_samples)
    
    # 随机采样
    if sample_size < len(files):
        sampled_files = random.sample(files, sample_size)
    else:
        sampled_files = files
    
    return sampled_files

def process_files(src_files, dst_dir, use_symlinks=False):
    """处理文件（复制或创建符号链接）"""
    os.makedirs(dst_dir, exist_ok=True)
    
    for src_file in tqdm(src_files, desc=f"处理文件到 {dst_dir}"):
        filename = os.path.basename(src_file)
        dst_file = os.path.join(dst_dir, filename)
        
        # 如果目标文件已存在，生成唯一文件名
        if os.path.exists(dst_file):
            base, ext = os.path.splitext(filename)
            dst_file = os.path.join(dst_dir, f"{base}_{random.randint(1000, 9999)}{ext}")
        
        # 复制或创建符号链接
        try:
            if use_symlinks:
                os.symlink(os.path.abspath(src_file), dst_file)
            else:
                shutil.copy2(src_file, dst_file)
        except Exception as e:
            print(f"处理文件 {src_file} 时出错: {e}")

def process_directory(src_dir, dst_subdir, ratio, max_samples, use_symlinks, seed=42):
    """处理单个目录"""
    # 获取图像文件
    image_files = get_image_files(src_dir)
    print(f"在 {src_dir} 中找到 {len(image_files)} 个图像文件")
    
    # 采样文件
    sampled_files = sample_files(image_files, ratio, max_samples, seed)
    print(f"采样 {len(sampled_files)} 个文件 (ratio={ratio}, max={max_samples})")
    
    # 处理文件
    process_files(sampled_files, dst_subdir, use_symlinks)
    
    return len(sampled_files)

def process_with_labels(src_dir, csv_file, dst_subdir, label_value, ratio, max_samples, use_symlinks, seed=42):
    """使用CSV标签文件处理图像"""
    try:
        # 读取CSV文件
        df = pd.read_csv(os.path.join(src_dir, csv_file))
        
        # 根据标签筛选
        if 'label' in df.columns:
            filtered_df = df[df['label'] == label_value]
        else:
            # 如果没有label列，尝试找其他可能的列
            if 'generated' in df.columns:
                filtered_df = df[df['generated'] == (1 if label_value == 'fake' else 0)]
            elif 'real' in df.columns:
                filtered_df = df[df['real'] == (1 if label_value == 'real' else 0)]
            else:
                print(f"警告: CSV文件 {csv_file} 中没有找到标签列，跳过")
                return 0
        
        # 获取图像文件
        if 'relative_path' in df.columns:
            image_files = [os.path.join(src_dir, path) for path in filtered_df['relative_path']]
        elif 'file_path' in df.columns:
            image_files = [os.path.join(src_dir, path) for path in filtered_df['file_path']]
        elif 'filename' in df.columns:
            image_files = [os.path.join(src_dir, path) for path in filtered_df['filename']]
        else:
            print(f"警告: CSV文件 {csv_file} 中没有找到路径列，跳过")
            return 0
        
        # 验证文件是否存在
        image_files = [f for f in image_files if os.path.exists(f)]
        print(f"在CSV {csv_file} 中找到 {len(image_files)} 个 {label_value} 图像")
        
        # 采样文件
        sampled_files = sample_files(image_files, ratio, max_samples, seed)
        print(f"采样 {len(sampled_files)} 个文件 (ratio={ratio}, max={max_samples})")
        
        # 处理文件
        process_files(sampled_files, dst_subdir, use_symlinks)
        
        return len(sampled_files)
    except Exception as e:
        print(f"处理CSV文件 {csv_file} 时出错: {e}")
        return 0

def organize_aigc_dataset(args):
    """组织AIGC数据集"""
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    
    # 处理采样比例
    ratio = args.ratio
    real_ratio = args.real_ratio if args.real_ratio is not None else ratio
    fake_ratio = args.fake_ratio if args.fake_ratio is not None else ratio
    
    # 处理最大样本数
    max_samples = args.max_samples
    real_max = args.real_max if args.real_max is not None else max_samples
    fake_max = args.fake_max if args.fake_max is not None else max_samples
    
    use_symlinks = args.use_symlinks
    seed = args.seed
    
    print("开始组织AIGC数据集:")
    print(f"- 源目录: {src_dir}")
    print(f"- 目标目录: {dst_dir}")
    print(f"- 真实图像采样比例: {real_ratio}")
    print(f"- 生成图像采样比例: {fake_ratio}")
    print(f"- 真实图像最大样本数: {real_max}")
    print(f"- 生成图像最大样本数: {fake_max}")
    print(f"- 使用符号链接: {use_symlinks}")
    
    os.makedirs(dst_dir, exist_ok=True)
    real_dir = os.path.join(dst_dir, 'real')
    fake_dir = os.path.join(dst_dir, 'fake')
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    
    real_nested_dir = os.path.join(src_dir, 'real')
    fake_nested_dir = os.path.join(src_dir, 'fake')
    
    if os.path.isdir(real_nested_dir):
        print("\n处理嵌套的real目录...")
        
        image_real_dir = os.path.join(real_nested_dir, 'Image_real')
        if os.path.isdir(image_real_dir):
            print(f"\n处理 Image_real 目录...")
            for subdir in os.listdir(image_real_dir):
                subdir_path = os.path.join(image_real_dir, subdir)
                if os.path.isdir(subdir_path):
                    print(f"\n处理 {subdir} 子目录...")
                    count = process_directory(
                        subdir_path, 
                        real_dir, 
                        real_ratio, 
                        real_max // 2,  
                        use_symlinks,
                        seed
                    )
                    real_count += count
        
        for item in os.listdir(real_nested_dir):
            item_path = os.path.join(real_nested_dir, item)
            if os.path.isdir(item_path) and item != 'Image_real':
                print(f"\n处理 {item} 目录...")
                count = process_directory(
                    item_path, 
                    real_dir, 
                    real_ratio, 
                    real_max // 2,
                    use_symlinks,
                    seed
                )
                real_count += count
    
    if os.path.isdir(fake_nested_dir):
        print("\n处理嵌套的fake目录...")
        
        fake_image_dirs = [
            "Midjourney", "Midjourney_new", "SD1", "SD2", "SDXL", "DF-GAN",
            "Chatgpt", "Baidu", "img2img", "pix2pix", "variations", "OJ"
        ]
        
        for item in os.listdir(fake_nested_dir):
            if item.startswith("tra_") and os.path.isdir(os.path.join(fake_nested_dir, item)):
                fake_image_dirs.append(item)
        
        available_dirs = [d for d in fake_image_dirs if os.path.isdir(os.path.join(fake_nested_dir, d))]
        samples_per_dir = fake_max // max(1, len(available_dirs)) if len(available_dirs) > 0 else 0
        
        print("\n处理生成图像...")
        print(f"- 可用生成图像目录数量: {len(available_dirs)}")
        print(f"- 每个目录采样数量: {samples_per_dir}")
        
        for dirname in fake_image_dirs:
            dir_path = os.path.join(fake_nested_dir, dirname)
            if os.path.isdir(dir_path):
                print(f"\n处理 {dirname} 目录...")
                count = process_directory(
                    dir_path, 
                    fake_dir, 
                    fake_ratio, 
                    samples_per_dir,  
                    use_symlinks,
                    seed + fake_image_dirs.index(dirname)  
                )
                fake_count += count
    
    csv_files = [
        "i2i_image_labels_relative.csv",
        "t2i_image_labels_relative.csv",
        "tra_image_labels_relative.csv"
    ]
    
    for csv_file in csv_files:
        csv_path = os.path.join(fake_nested_dir, csv_file)
        if os.path.exists(csv_path):
            print(f"\n处理CSV标签文件 {csv_file}...")
            count = process_with_labels(
                fake_nested_dir, 
                csv_file, 
                real_dir, 
                'real', 
                real_ratio * 0.5,  
                real_max // 2,
                use_symlinks,
                seed
            )
            real_count += count
            
            count = process_with_labels(
                fake_nested_dir, 
                csv_file, 
                fake_dir, 
                'fake', 
                fake_ratio * 0.5,
                fake_max // 2,
                use_symlinks,
                seed
            )
            fake_count += count
    
    image_real_dir = os.path.join(real_nested_dir, 'Image_real') if os.path.isdir(real_nested_dir) else None
    if image_real_dir and os.path.isdir(image_real_dir):
        real_csv_files = [
            "real_celebA-HQ_labels_relative.csv",
            "real_image_labels_relative.csv"
        ]
        
        for csv_file in real_csv_files:
            csv_path = os.path.join(image_real_dir, csv_file)
            if os.path.exists(csv_path):
                print(f"\n处理真实图像 CSV标签文件 {csv_file}...")
                count = process_with_labels(
                    image_real_dir, 
                    csv_file, 
                    real_dir, 
                    'real', 
                    real_ratio,
                    real_max,
                    use_symlinks,
                    seed
                )
                real_count += count
    
    print("\n数据集组织完成!")
    print(f"- 真实图像: {real_count}")
    print(f"- 生成图像: {fake_count}")
    print(f"- 总计: {real_count + fake_count}")
    print(f"\n请使用以下命令进行数据预处理:")
    print(f"python run.py prepare --src_dir \"{dst_dir}\" --dst_dir \"processed_data\" --img_size 256 --resize")

def main():
    args = parse_args()
    organize_aigc_dataset(args)

if __name__ == '__main__':
    main()
