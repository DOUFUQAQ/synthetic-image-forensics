"""
演示脚本：使用训练好的模型检测单张图像或目录中的图像
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import create_model

def parse_args():
    parser = argparse.ArgumentParser(description='演示合成图像检测器')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或包含图像的目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='输出目录')
    parser.add_argument('--img_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备(cuda或cpu)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='检测阈值')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化特征权重')
    return parser.parse_args()

def preprocess_image(image_path, img_size=256):
    """预处理单张图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        return img_tensor, img
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None, None

def detect_image(model, image_path, device, img_size=256, threshold=0.5, visualize=False):
    """检测单张图像"""
    # 预处理图像
    img_tensor, orig_img = preprocess_image(image_path, img_size)
    if img_tensor is None:
        return None, None, None
    
    # 转换为批次并移动到设备
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(img_batch)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        feature_weights = outputs['feature_weights']
        attention_weights = outputs['attention_weights']
    
    # 获取预测结果
    prob_synthetic = probs[0, 1].item()
    is_synthetic = prob_synthetic > threshold
    
    result = {
        'is_synthetic': is_synthetic,
        'prob_synthetic': prob_synthetic,
        'feature_weights': feature_weights.cpu().numpy(),
        'attention_weights': attention_weights[0].cpu().numpy()
    }
    
    # 可视化
    if visualize:
        # 创建结果图像
        plt.figure(figsize=(12, 6))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(orig_img))
        plt.title('输入图像')
        plt.axis('off')
        
        # 显示检测结果和置信度
        plt.subplot(1, 2, 2)
        
        # 使用饼图显示真实vs生成的概率
        labels = ['真实', '生成']
        sizes = [1 - prob_synthetic, prob_synthetic]
        colors = ['#3498db', '#e74c3c'] if is_synthetic else ['#2ecc71', '#e74c3c']
        explode = (0, 0.1) if is_synthetic else (0.1, 0)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        
        result_text = '检测结果: ' + ('生成图像' if is_synthetic else '真实图像')
        plt.title(result_text)
        
        # 特征权重条形图
        plt.figure(figsize=(8, 4))
        feature_names = ['CNN特征', '频率特征', '共现特征']
        plt.bar(feature_names, feature_weights.cpu().numpy())
        plt.title('特征权重')
        plt.ylim(0, 1)
        
        # 注意力权重条形图
        plt.figure(figsize=(8, 4))
        plt.bar(feature_names, attention_weights[0].cpu().numpy())
        plt.title('注意力权重')
        plt.ylim(0, 1)
        
        vis_result = plt.gcf()
    else:
        vis_result = None
    
    return result, orig_img, vis_result

def process_directory(model, input_dir, output_dir, device, img_size, threshold, visualize):
    """处理目录中的所有图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                 if os.path.isfile(os.path.join(input_dir, f)) and 
                 f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    results = []
    
    for img_file in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(input_dir, img_file)
        result, orig_img, vis_result = detect_image(
            model, img_path, device, img_size, threshold, visualize
        )
        
        if result is not None:
            # 保存结果
            result['filename'] = img_file
            results.append(result)
            
            if visualize:
                # 保存可视化结果
                vis_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_result.png")
                vis_result.savefig(vis_path)
                plt.close()
    
    # 统计结果
    synthetic_count = sum(1 for r in results if r['is_synthetic'])
    real_count = len(results) - synthetic_count
    
    print(f"\n处理完成! 共检测 {len(results)} 张图像:")
    print(f"真实图像: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"生成图像: {synthetic_count} ({synthetic_count/len(results)*100:.1f}%)")
    
    # 保存结果摘要
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"检测结果摘要\n")
        f.write(f"总计: {len(results)} 张图像\n")
        f.write(f"真实图像: {real_count} ({real_count/len(results)*100:.1f}%)\n")
        f.write(f"生成图像: {synthetic_count} ({synthetic_count/len(results)*100:.1f}%)\n\n")
        
        f.write("详细结果:\n")
        for r in results:
            label = "生成" if r['is_synthetic'] else "真实"
            f.write(f"{r['filename']}: {label} (置信度: {r['prob_synthetic']*100:.1f}%)\n")
    
    print(f"结果保存到: {output_dir}")

def main():
    args = parse_args()
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = create_model()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"从 {args.checkpoint} 加载模型 (Epoch {checkpoint['epoch']})")
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        print(f"处理单个图像: {args.input}")
        result, orig_img, vis_result = detect_image(
            model, args.input, device, args.img_size, args.threshold, args.visualize
        )
        
        if result is not None:
            # 打印结果
            print(f"检测结果: {'生成图像' if result['is_synthetic'] else '真实图像'}")
            print(f"生成图像概率: {result['prob_synthetic']*100:.2f}%")
            
            # 特征权重
            feature_names = ['CNN特征', '频率特征', '共现特征']
            print("\n特征权重:")
            for i, name in enumerate(feature_names):
                print(f"{name}: {result['feature_weights'][i]:.4f}")
            
            # 保存可视化结果
            if args.visualize:
                output_path = os.path.join(output_dir, "detection_result.png")
                vis_result.savefig(output_path)
                print(f"可视化结果保存到: {output_path}")
                plt.show()
    else:
        # 处理目录
        print(f"处理目录: {args.input}")
        process_directory(
            model, args.input, output_dir, device, 
            args.img_size, args.threshold, args.visualize
        )

if __name__ == '__main__':
    main()
