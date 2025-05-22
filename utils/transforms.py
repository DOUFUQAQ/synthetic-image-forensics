import torch
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_detection_transforms(img_size=256):
    """
    获取用于图像生成检测的数据增强变换
    
    Args:
        img_size: 目标图像大小
        
    Returns:
        train_transform, val_transform: 训练和验证数据增强
    """
    # 训练集变换 - 增强版
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # 添加垂直翻转
        transforms.RandomRotation(15),  # 添加旋转
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # 增强颜色变化
        transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)  # 增强仙变换
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # 添加高斯模糊
        ], p=0.3),
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机擦除
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_hashing_transforms(img_size=224):
    """
    获取用于图像哈希的数据增强变换
    
    Args:
        img_size: 目标图像大小
        
    Returns:
        train_transform, val_transform: 训练和验证数据增强
    """
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_albumentations_transforms(img_size=256):
    """
    使用Albumentations库获取数据增强变换
    提供更丰富和高效的数据增强操作
    
    Args:
        img_size: 目标图像大小
        
    Returns:
        train_transform, val_transform: 训练和验证数据增强
    """
    # 训练集变换
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # 验证集变换
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


class AlbumentationsWrapper:
    """
    将Albumentations变换包装为torchvision风格的接口
    便于与PyTorch数据加载器集成
    """
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, img):
        # 将PIL图像转换为numpy数组
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        # 应用Albumentations变换
        result = self.transform(image=img)
        
        return result['image']
