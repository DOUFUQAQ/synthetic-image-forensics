import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gram_detection.detector import GramBasedDetector


def preprocess_image(image_path, img_size=256):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
        img_size: 目标图像大小
        
    Returns:
        image_tensor: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return image_tensor, image


def visualize_prediction(image, prob, pred_label, save_path=None):
    """
    可视化预测结果
    
    Args:
        image: PIL图像
        prob: 生成图像的概率
        pred_label: 预测标签
        save_path: 保存路径，如果为None则显示图像
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    
    label_text = "生成图像" if pred_label == 1 else "真实图像"
    confidence = prob if pred_label == 1 else 1 - prob
    
    plt.title(f"预测: {label_text}\n置信度: {confidence:.2%}", fontsize=16)
    
    color = "red" if pred_label == 1 else "green"
    plt.text(10, 30, f"生成概率: {prob:.2%}", fontsize=14, color=color, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def detect_image(model, image_tensor, device):
    """
    检测图像是否为生成图像
    
    Args:
        model: 检测模型
        image_tensor: 输入图像张量
        device: 计算设备
        
    Returns:
        pred_label: 预测标签 (0: 真实, 1: 生成)
        prob: 生成图像的概率
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        logits, _ = model(image_tensor)
        
        # 获取预测结果
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        
        # 获取生成图像的概率
        gen_prob = probs[0, 1].item()
    
    return pred_label, gen_prob


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建结果目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    try:
        model = GramBasedDetector(
            backbone_name=args.backbone,
            pretrained=False,
            num_classes=2
        ).to(device)
        
        # 加载模型参数 - 支持两种格式
        try:
            # 尝试加载完整检查点格式
            checkpoint = torch.load(args.model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接加载 state_dict
                model.load_state_dict(checkpoint)
        except Exception as inner_e:
            # 如果上述方法失败，直接加载 state_dict
            state_dict = torch.load(args.model_path, map_location=device)
            model.load_state_dict(state_dict)
            
        model.eval()
        print(f'成功加载模型: {args.model_path}')
    except Exception as e:
        print(f'加载模型时出错: {str(e)}')
        return
    
    # 处理单个图像或目录中的所有图像
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_paths = [
            os.path.join(args.image_path, f) for f in os.listdir(args.image_path)
            if os.path.isfile(os.path.join(args.image_path, f)) and 
            os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        if not image_paths:
            print(f'目录 {args.image_path} 中没有有效图像文件')
            return
    else:
        print(f'无效的路径: {args.image_path}')
        return
    
    # 处理每个图像
    results = []
    
    for i, img_path in enumerate(image_paths):
        try:
            print(f'处理图像 {i+1}/{len(image_paths)}: {img_path}')
            
            # 预处理图像
            image_tensor, original_image = preprocess_image(img_path, args.img_size)
            
            # 检测图像
            pred_label, gen_prob = detect_image(model, image_tensor, device)
            
            # 记录结果
            result = {
                'path': img_path,
                'pred_label': '生成图像' if pred_label == 1 else '真实图像',
                'gen_prob': gen_prob
            }
            results.append(result)
            
            # 打印结果
            print(f'预测结果: {result["pred_label"]}, 生成概率: {gen_prob:.2%}')
            
            # 可视化结果
            if args.output_dir:
                output_filename = f'result_{i}.png'
                if len(image_paths) == 1:
                    output_filename = 'result.png'
                
                save_path = os.path.join(args.output_dir, output_filename)
                visualize_prediction(original_image, gen_prob, pred_label, save_path)
                print(f'结果已保存至: {save_path}')
            else:
                visualize_prediction(original_image, gen_prob, pred_label)
            
        except Exception as e:
            print(f'处理图像 {img_path} 时出错: {e}')
    
    # 如果有多个图像，显示摘要
    if len(results) > 1:
        print("\n检测结果摘要:")
        for result in results:
            print(f"{os.path.basename(result['path'])}: {result['pred_label']} ({result['gen_prob']:.2%})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像生成检测演示')
    
    parser.add_argument('--image_path', type=str, required=True,
                       help='要检测的图像路径或包含图像的目录')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型检查点路径')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='骨干网络架构 (resnet50, resnet34, efficientnet_b0)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='输入图像大小')
    parser.add_argument('--output_dir', type=str, default='',
                       help='保存结果的目录，如果为空则显示结果')
    
    args = parser.parse_args()
    
    main(args)
