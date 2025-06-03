import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_ensemble_model import load_ensemble_model
from models.detector import create_model
from evaluate.evaluate_detector import evaluate_model, DeepFaceGenDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='评估合成图像检测器集成模型')
    parser.add_argument('--data_dir', type=str, required=True, help='测试数据目录')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--subset_ratio', type=float, default=1.0, help='子集比例，1.0表示全集')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--calibrate', action='store_true', help='是否校准模型')
    parser.add_argument('--optimize_threshold', action='store_true', help='是否优化阈值')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--feature_dim', type=int, default=2048, help='特征维度')
    parser.add_argument('--temperature', type=float, default=None, help='温度参数')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据集
    test_dataset = DeepFaceGenDataset(
        root_dir=args.data_dir,
        transform=None,  # 测试时不使用额外的变换
        subset_ratio=args.subset_ratio,
        img_size=args.img_size,
        seed=args.seed
    )
    
    print(f"加载了 {len(test_dataset)} 张图像，其中真实图像 {test_dataset.real_count} 张，合成图像 {test_dataset.synthetic_count} 张")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 使用专用函数加载集成模型
    print(f"正在加载模型检查点: {args.checkpoint}")
    model = load_ensemble_model(args.checkpoint, device=device)
    
    # 设置温度参数（如果提供）
    if args.temperature is not None:
        print(f"设置温度参数为: {args.temperature}")
        model.temperature.data = torch.tensor(args.temperature, device=device)
    
    # 执行评估
    print("开始评估模型...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        calibrate=args.calibrate,
        optimize_threshold=args.optimize_threshold,
        visualize=args.visualize
    )
    
    # 输出评估结果摘要
    print("\n===== 评估结果摘要 =====")
    print(f"准确率: {results['metrics']['accuracy']:.2f}%")
    print(f"AUC: {results['metrics']['auc']:.4f}")
    print(f"F1分数: {results['metrics']['f1']:.4f}")
    print(f"真实图像准确率: {results['class_acc']['class_0']:.2f}%")
    print(f"合成图像准确率: {results['class_acc']['class_1']:.2f}%")
    
    print(f"\n详细结果已保存到: {output_dir}")
    
if __name__ == "__main__":
    main()
