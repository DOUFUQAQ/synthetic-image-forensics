import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class DeepFaceGenDataset(Dataset):
    """
    Dataset class for loading and preprocessing DeepFaceGen dataset images
    Supports sampling a subset of the full dataset to manage computational resources
    """
    def __init__(self, 
                 root_dir, 
                 transform=None, 
                 subset_ratio=0.1,  # 默认只使用10%的数据
                 max_samples_per_class=5000,  # 每类最多采样数量
                 img_size=256,
                 seed=42):
        """
        Args:
            root_dir (str): 数据集根目录
            transform (callable, optional): 额外的图像变换
            subset_ratio (float): 采样比例，1.0表示使用全部数据
            max_samples_per_class (int): 每个类别最大样本数
            img_size (int): 图像大小
            seed (int): 随机种子，保证可重复采样
        """
        self.root_dir = root_dir
        self.seed = seed
        random.seed(seed)
        
        # 基础变换: 调整大小和标准化
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 额外的变换
        self.transform = transform
        
        # 采样参数
        self.subset_ratio = subset_ratio
        self.max_samples_per_class = max_samples_per_class
        
        # 加载数据
        self.images, self.labels = self._load_dataset()
        
    def _load_dataset(self):
        """加载数据集，采样子集"""
        # 假设子目录结构: root_dir/class_name/image_files
        # 类别: 0-真实图像, 1-生成图像
        
        images = []
        labels = []
        
        # 遍历子目录(类别)
        class_dirs = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_path = os.path.join(self.root_dir, class_dir)
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # 随机采样
            if self.subset_ratio < 1.0:
                num_samples = min(
                    int(len(image_files) * self.subset_ratio),
                    self.max_samples_per_class
                )
                image_files = random.sample(image_files, num_samples)
            elif len(image_files) > self.max_samples_per_class:
                image_files = random.sample(image_files, self.max_samples_per_class)
                
            # 添加图像路径和标签
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                images.append(img_path)
                labels.append(class_idx)
                
        print(f"Loaded {len(images)} images: {len([l for l in labels if l==0])} real, "
              f"{len([l for l in labels if l==1])} synthetic")
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
            pil_image = image  # 保存PIL图像用于类别特定的变换
            
            # 先检查transform是否为ClassSpecificTransform类型
            if hasattr(self.transform, '__call__') and hasattr(self.transform, 'real_transform'):
                # 应用基础变换前的类别特定变换
                transformed_pil = self.transform(pil_image, label)
                # 然后应用基础变换
                image = self.base_transform(transformed_pil)
            else:
                # 应用基础变换
                image = self.base_transform(image)
                
                # 应用普通额外变换(如果有)
                if self.transform:
                    image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个占位图像和标签
            return torch.zeros(3, 256, 256), label
            

def get_dataloaders(data_dir, batch_size=32, subset_ratio=0.1, max_samples=5000, 
                  img_size=256, num_workers=4, seed=42, advanced_augmentation=False):
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir (str): 数据集根目录
        batch_size (int): 批量大小
        subset_ratio (float): 数据采样比例
        max_samples (int): 每类最大样本数
        img_size (int): 图像大小
        num_workers (int): 数据加载线程数
        seed (int): 随机种子
        advanced_augmentation (bool): 是否启用高级数据增强
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 针对不同类别图像的不同增强策略函数
    def get_real_image_transform(advanced=False):
        # 针对真实图像的数据增强
        if advanced:
            # 高级增强策略
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.7),  # 提高翻转概率
                transforms.RandomVerticalFlip(p=0.3),  # 提高垂直翻转概率
                transforms.RandomRotation(30),  # 更大的旋转角度
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 更强的颜色干扰
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # 位移和缩放
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 透视变换
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),  # 更大的裁剪范围
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.4),  # 更强的高斯模糊
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))  # 更强的擦除
            ])
        else:
            # 标准增强策略
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1)
            ])
    
    def get_synthetic_image_transform(advanced=False):
        # 针对合成图像的数据增强
        if advanced:
            # 高级增强策略
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15))
            ])
        else:
            # 标准增强策略
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0))
            ])
    
    # 创建自定义的数据增强类
    class ClassSpecificTransform:
        def __init__(self, real_transform, synthetic_transform):
            self.real_transform = real_transform
            self.synthetic_transform = synthetic_transform
            
        def __call__(self, img, label):
            if label == 0:  # 真实图像
                return self.real_transform(img)
            else:  # 合成图像
                return self.synthetic_transform(img)
    
    # 使用类别特定的数据增强，根据advanced_augmentation参数决定使用哪种级别的增强
    train_transform = ClassSpecificTransform(
        real_transform=get_real_image_transform(advanced=advanced_augmentation),
        synthetic_transform=get_synthetic_image_transform(advanced=advanced_augmentation)
    )
    
    if advanced_augmentation:
        print("使用高级数据增强：更强的旋转、垂直翻转、高斯模糊和色彩扰动")
    
    # 创建数据集
    dataset = DeepFaceGenDataset(
        root_dir=data_dir,
        transform=train_transform,
        subset_ratio=subset_ratio,
        max_samples_per_class=max_samples,
        img_size=img_size,
        seed=seed
    )
    
    # 分割训练集和验证集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # 验证集不使用数据增强
    val_dataset.transform = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
