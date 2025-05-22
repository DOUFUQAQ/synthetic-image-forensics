import os
import argparse
import shutil
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


def check_valid_image(file_path):
    """
    检查文件是否为有效的图像文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为有效图像
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext not in valid_extensions:
        return False
    
    # 尝试打开图像
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False


def prepare_detection_dataset(real_dir, generated_dir, output_dir, limit=None, split_ratio=0.8):
    """
    准备用于图像生成检测的数据集
    
    Args:
        real_dir: 真实图像目录
        generated_dir: 生成图像目录
        output_dir: 输出目录
        limit: 每类图像的最大数量
        split_ratio: 训练/测试集划分比例
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    train_real_dir = os.path.join(output_dir, 'train', 'real')
    train_gen_dir = os.path.join(output_dir, 'train', 'generated')
    test_real_dir = os.path.join(output_dir, 'test', 'real')
    test_gen_dir = os.path.join(output_dir, 'test', 'generated')
    
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_gen_dir, exist_ok=True)
    os.makedirs(test_real_dir, exist_ok=True)
    os.makedirs(test_gen_dir, exist_ok=True)
    
    # 收集并过滤真实图像
    real_images = []
    for root, _, files in os.walk(real_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if check_valid_image(file_path):
                real_images.append(file_path)
    
    # 收集并过滤生成图像
    gen_images = []
    for root, _, files in os.walk(generated_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if check_valid_image(file_path):
                gen_images.append(file_path)
    
    # 随机打乱图像
    random.seed(42)
    random.shuffle(real_images)
    random.shuffle(gen_images)
    
    # 限制图像数量
    if limit:
        real_images = real_images[:limit]
        gen_images = gen_images[:limit]
    
    print(f"收集到 {len(real_images)} 张真实图像和 {len(gen_images)} 张生成图像")
    
    # 划分训练集和测试集
    n_real_train = int(len(real_images) * split_ratio)
    n_gen_train = int(len(gen_images) * split_ratio)
    
    real_train = real_images[:n_real_train]
    real_test = real_images[n_real_train:]
    gen_train = gen_images[:n_gen_train]
    gen_test = gen_images[n_gen_train:]
    
    # 复制图像到目标目录
    print("正在复制真实图像到训练集...")
    for i, img_path in enumerate(tqdm(real_train)):
        dst_path = os.path.join(train_real_dir, f"real_{i:06d}{os.path.splitext(img_path)[1]}")
        shutil.copy2(img_path, dst_path)
    
    print("正在复制真实图像到测试集...")
    for i, img_path in enumerate(tqdm(real_test)):
        dst_path = os.path.join(test_real_dir, f"real_{i:06d}{os.path.splitext(img_path)[1]}")
        shutil.copy2(img_path, dst_path)
    
    print("正在复制生成图像到训练集...")
    for i, img_path in enumerate(tqdm(gen_train)):
        dst_path = os.path.join(train_gen_dir, f"gen_{i:06d}{os.path.splitext(img_path)[1]}")
        shutil.copy2(img_path, dst_path)
    
    print("正在复制生成图像到测试集...")
    for i, img_path in enumerate(tqdm(gen_test)):
        dst_path = os.path.join(test_gen_dir, f"gen_{i:06d}{os.path.splitext(img_path)[1]}")
        shutil.copy2(img_path, dst_path)
    
    print(f"数据集准备完成！")
    print(f"训练集: {len(real_train)} 张真实图像, {len(gen_train)} 张生成图像")
    print(f"测试集: {len(real_test)} 张真实图像, {len(gen_test)} 张生成图像")


def prepare_hash_dataset(image_dirs, output_dir, limit_per_class=None, split_ratio=0.8):
    """
    准备用于图像哈希的数据集
    
    Args:
        image_dirs: 包含多个类别目录的顶级目录
        output_dir: 输出目录
        limit_per_class: 每个类别的最大图像数量
        split_ratio: 训练/测试集划分比例
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    query_dir = os.path.join(output_dir, 'query')
    database_dir = os.path.join(output_dir, 'database')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(database_dir, exist_ok=True)
    
    # 获取所有类别目录
    class_dirs = []
    if os.path.isdir(image_dirs):
        for item in os.listdir(image_dirs):
            item_path = os.path.join(image_dirs, item)
            if os.path.isdir(item_path):
                class_dirs.append(item_path)
    else:
        print(f"错误: {image_dirs} 不是有效目录")
        return
    
    if not class_dirs:
        print(f"错误: 在 {image_dirs} 中未找到类别目录")
        return
    
    print(f"找到 {len(class_dirs)} 个类别目录")
    
    # 处理每个类别
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        print(f"处理类别: {class_name}")
        
        # 创建对应的输出目录
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        query_class_dir = os.path.join(query_dir, class_name)
        database_class_dir = os.path.join(database_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        os.makedirs(query_class_dir, exist_ok=True)
        os.makedirs(database_class_dir, exist_ok=True)
        
        # 收集并过滤图像
        images = []
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            if os.path.isfile(file_path) and check_valid_image(file_path):
                images.append(file_path)
        
        # 随机打乱图像
        random.shuffle(images)
        
        # 限制图像数量
        if limit_per_class and len(images) > limit_per_class:
            images = images[:limit_per_class]
        
        print(f"  收集到 {len(images)} 张图像")
        
        # 划分数据集
        n_train = int(len(images) * split_ratio)
        n_test = len(images) - n_train
        n_query = min(50, n_test // 2)  # 查询集用一半测试集，但最多50张
        
        train_images = images[:n_train]
        test_images = images[n_train:]
        query_images = test_images[:n_query]
        
        # 所有图像都放入数据库（包括训练集）
        database_images = images
        
        # 复制图像到目标目录
        print(f"  正在复制训练图像...")
        for i, img_path in enumerate(tqdm(train_images)):
            dst_path = os.path.join(train_class_dir, f"{class_name}_{i:06d}{os.path.splitext(img_path)[1]}")
            shutil.copy2(img_path, dst_path)
        
        print(f"  正在复制测试图像...")
        for i, img_path in enumerate(tqdm(test_images)):
            dst_path = os.path.join(test_class_dir, f"{class_name}_{i:06d}{os.path.splitext(img_path)[1]}")
            shutil.copy2(img_path, dst_path)
        
        print(f"  正在复制查询图像...")
        for i, img_path in enumerate(tqdm(query_images)):
            dst_path = os.path.join(query_class_dir, f"{class_name}_{i:06d}{os.path.splitext(img_path)[1]}")
            shutil.copy2(img_path, dst_path)
        
        print(f"  正在复制数据库图像...")
        for i, img_path in enumerate(tqdm(database_images)):
            dst_path = os.path.join(database_class_dir, f"{class_name}_{i:06d}{os.path.splitext(img_path)[1]}")
            shutil.copy2(img_path, dst_path)
    
    print(f"哈希数据集准备完成！")


def resize_dataset(input_dir, output_dir, target_size=(256, 256), format='jpg'):
    """
    调整数据集中图像的大小
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_size: 目标大小 (宽度, 高度)
        format: 输出图像格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入目录中的所有文件和子目录
    for root, dirs, files in os.walk(input_dir):
        # 计算相对路径
        rel_path = os.path.relpath(root, input_dir)
        
        # 创建对应的输出子目录
        if rel_path != '.':
            out_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(out_subdir, exist_ok=True)
        else:
            out_subdir = output_dir
        
        # 处理当前目录下的所有文件
        for file in tqdm(files, desc=f"处理 {rel_path}"):
            file_path = os.path.join(root, file)
            
            # 检查是否为有效图像
            if not check_valid_image(file_path):
                continue
            
            # 输出文件路径
            file_base = os.path.splitext(file)[0]
            out_file = f"{file_base}.{format}"
            out_path = os.path.join(out_subdir, out_file)
            
            try:
                # 打开图像并调整大小
                with Image.open(file_path) as img:
                    # 转换为RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整大小
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    
                    # 保存调整后的图像
                    img_resized.save(out_path, format=format.upper(), quality=95)
            except Exception as e:
                print(f"处理 {file_path} 时出错: {e}")
    
    print(f"调整大小完成！所有图像已保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='数据集准备工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 检测数据集准备命令
    detection_parser = subparsers.add_parser('detection', help='准备检测数据集')
    detection_parser.add_argument('--real_dir', type=str, required=True,
                                 help='真实图像目录')
    detection_parser.add_argument('--generated_dir', type=str, required=True,
                                 help='生成图像目录')
    detection_parser.add_argument('--output_dir', type=str, required=True,
                                 help='输出目录')
    detection_parser.add_argument('--limit', type=int, default=None,
                                 help='每类图像的最大数量')
    detection_parser.add_argument('--split_ratio', type=float, default=0.8,
                                 help='训练/测试集划分比例')
    
    # 哈希数据集准备命令
    hash_parser = subparsers.add_parser('hash', help='准备哈希数据集')
    hash_parser.add_argument('--image_dirs', type=str, required=True,
                            help='包含多个类别目录的顶级目录')
    hash_parser.add_argument('--output_dir', type=str, required=True,
                            help='输出目录')
    hash_parser.add_argument('--limit_per_class', type=int, default=None,
                            help='每个类别的最大图像数量')
    hash_parser.add_argument('--split_ratio', type=float, default=0.8,
                            help='训练/测试集划分比例')
    
    # 调整图像大小命令
    resize_parser = subparsers.add_parser('resize', help='调整数据集中图像的大小')
    resize_parser.add_argument('--input_dir', type=str, required=True,
                              help='输入目录')
    resize_parser.add_argument('--output_dir', type=str, required=True,
                              help='输出目录')
    resize_parser.add_argument('--width', type=int, default=256,
                              help='目标宽度')
    resize_parser.add_argument('--height', type=int, default=256,
                              help='目标高度')
    resize_parser.add_argument('--format', type=str, default='jpg',
                              help='输出图像格式 (jpg, png)')
    
    args = parser.parse_args()
    
    if args.command == 'detection':
        prepare_detection_dataset(
            args.real_dir, args.generated_dir, args.output_dir,
            args.limit, args.split_ratio
        )
    elif args.command == 'hash':
        prepare_hash_dataset(
            args.image_dirs, args.output_dir,
            args.limit_per_class, args.split_ratio
        )
    elif args.command == 'resize':
        resize_dataset(
            args.input_dir, args.output_dir,
            (args.width, args.height), args.format
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
