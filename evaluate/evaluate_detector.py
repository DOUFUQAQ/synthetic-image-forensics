import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import json
import time
import numpy as np

# 自定义JSON编码器来处理NumPy数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gram_detection.detector import GramBasedDetector
from data.dataset import DetectionDataset
from utils.transforms import get_detection_transforms
from utils.metrics import compute_detection_metrics
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_detection_results, create_comparison_grid,
    visualize_gram_features
)


def evaluate_detector(args):
    """
    评估图像生成检测器
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
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
    except Exception as e:
        # 如果上述方法失败，直接加载 state_dict
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded model from {args.model_path}')
    
    # 数据变换
    _, test_transform = get_detection_transforms(img_size=args.img_size)
    
    # 创建测试数据集
    test_dataset = DetectionDataset(
        real_dir=os.path.join(args.test_dir, 'real'),
        generated_dir=os.path.join(args.test_dir, 'generated'),
        transform=test_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Test dataset size: {len(test_dataset)}')
    
    # 开始评估
    all_preds = []
    all_probs = []
    all_targets = []
    all_images = []
    all_gram_features = []
    
    # Record running time
    total_time = 0
    num_samples = 0
    
    # Setup timing mechanism based on device
    using_cuda_timing = torch.cuda.is_available()
    if using_cuda_timing:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Record start time based on device
            if using_cuda_timing:
                start_event.record()
                start_time = 0  # Not used in CUDA timing path
            else:
                start_time = time.time()
            
            # Forward propagation
            logits, texture_embedding = model(images)
            
            # Record end time based on device
            if using_cuda_timing:
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event)  # Returns milliseconds
            else:
                end_time = time.time()
                batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
            total_time += batch_time
            num_samples += images.size(0)
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 收集结果
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            
            # 保存一些图像用于可视化
            if len(all_images) < args.num_vis_samples:
                all_images.append(images[:min(args.batch_size, args.num_vis_samples - len(all_images))].cpu())
            
            # 保存一些Gram特征用于可视化
            if args.visualize_features and len(all_gram_features) < 10:
                # 获取内部Gram矩阵特征
                # 注意：这需要模型支持访问内部特征
                if hasattr(model, 'texture_extractor') and hasattr(model.texture_extractor, 'features'):
                    gram_features = model.texture_extractor.features.get('layer3', None)
                    if gram_features is not None:
                        all_gram_features.append(gram_features[:1].cpu())  # 只保存第一个样本
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1}/{len(test_loader)} batches')
    
    # 计算平均推理时间
    avg_time_per_sample = total_time / num_samples
    print(f'Average inference time per sample: {avg_time_per_sample:.2f} ms')
    
    # 合并所有结果
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    if all_images:
        all_images = torch.cat(all_images)
    
    # 计算评估指标
    metrics = compute_detection_metrics(all_preds, all_targets, all_probs[:, 1])
    
    # 打印评估结果
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"AP: {metrics['ap']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # 保存评估结果到JSON文件
    # 使用float()处理所有应该是浮点数的值
    metrics['avg_inference_time_ms'] = float(avg_time_per_sample)
    
    # 使用自定义编码器处理NumPy数据类型
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_targets, all_preds, save_path=cm_path)
    print(f'Saved confusion matrix to {cm_path}')
    
    # 绘制ROC曲线
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(all_targets, all_probs[:, 1], save_path=roc_path)
    print(f'Saved ROC curve to {roc_path}')
    
    # 可视化一些预测结果
    if len(all_images) > 0:
        vis_indices = np.random.choice(len(all_images), min(args.num_vis_samples, len(all_images)), replace=False)
        vis_images = all_images[vis_indices]
        vis_targets = all_targets[vis_indices]
        vis_preds = all_preds[vis_indices]
        vis_probs = all_probs[vis_indices, 1]  # 取第二类（生成图像）的概率
        
        results_path = os.path.join(args.output_dir, 'detection_results.png')
        plot_detection_results(
            vis_images, vis_targets, vis_preds, vis_probs, 
            num_samples=min(5, len(vis_images)), save_path=results_path
        )
        print(f'Saved detection results visualization to {results_path}')
    
    # 可视化Gram特征
    if args.visualize_features and all_gram_features:
        for i, gram_feature in enumerate(all_gram_features[:3]):  # 最多保存3个
            feature_path = os.path.join(args.output_dir, f'gram_feature_{i}.png')
            visualize_gram_features(gram_feature, save_path=feature_path)
            print(f'Saved Gram feature visualization to {feature_path}')
    
    # 对于错误分类的案例进行分析
    if len(all_images) > 0:
        error_indices = np.where(all_preds != all_targets)[0]
        if len(error_indices) > 0:
            # 取最多10个错误样本
            error_indices = error_indices[:min(10, len(error_indices))]
            error_images = all_images[error_indices]
            error_targets = all_targets[error_indices]
            error_preds = all_preds[error_indices]
            error_probs = all_probs[error_indices, 1]
            
            error_path = os.path.join(args.output_dir, 'error_cases.png')
            plot_detection_results(
                error_images, error_targets, error_preds, error_probs, 
                num_samples=min(5, len(error_indices)), save_path=error_path
            )
            print(f'Saved error cases visualization to {error_path}')
    
    print(f'All evaluation results saved to {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GAN/Diffusion Detection Model')
    
    # 数据参数
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Path to test directory containing real and generated images')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Input image size')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture (resnet50, resnet34, efficientnet_b0)')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/detector',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_vis_samples', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--visualize_features', action='store_true',
                       help='Visualize Gram features')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    evaluate_detector(args)
