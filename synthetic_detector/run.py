"""
合成图像检测器启动脚本
提供训练、评估和演示功能的统一入口
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='合成图像检测器命令行工具')
    subparsers = parser.add_subparsers(dest='command', help='要执行的命令')
    
    # 数据集下载命令
    download_parser = subparsers.add_parser('download', help='下载数据集样本')
    download_parser.add_argument('--output_dir', type=str, default='sample_dataset',
                              help='输出目录')
    download_parser.add_argument('--num_samples', type=int, default=1000,
                              help='每个类别的样本数量')
    download_parser.add_argument('--source', type=str, default='celeba_hq',
                              help='数据源(celeba_hq, ffhq, thispersondoesnotexist)')
    download_parser.add_argument('--api_key', type=str, default='',
                              help='API密钥（如果需要）')
    
    # 数据预处理命令
    prepare_parser = subparsers.add_parser('prepare', help='准备数据集')
    prepare_parser.add_argument('--src_dir', type=str, required=True,
                              help='源数据集目录(原始DeepFaceGen数据集路径)')
    prepare_parser.add_argument('--dst_dir', type=str, required=True,
                              help='目标目录(处理后的数据集保存路径)')
    prepare_parser.add_argument('--ratio', type=float, default=0.1,
                              help='采样比例(0-1之间的小数)')
    prepare_parser.add_argument('--max_samples', type=int, default=5000,
                              help='每个类别的最大样本数')
    prepare_parser.add_argument('--img_size', type=int, default=256,
                              help='调整图像大小')
    prepare_parser.add_argument('--resize', action='store_true',
                              help='是否调整图像大小')
    prepare_parser.add_argument('--use_all', action='store_true',
                              help='使用所有图像，不进行采样')
    prepare_parser.add_argument('--num_workers', type=int, default=4,
                              help='处理线程数')
    prepare_parser.add_argument('--seed', type=int, default=42,
                              help='随机种子')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data_dir', type=str, required=True,
                            help='数据集路径')
    train_parser.add_argument('--subset_ratio', type=float, default=0.1,
                            help='数据集采样比例')
    train_parser.add_argument('--max_samples', type=int, default=5000,
                            help='每类最大样本数')
    train_parser.add_argument('--img_size', type=int, default=256,
                            help='图像大小')
    train_parser.add_argument('--batch_size', type=int, default=32,
                            help='批量大小')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='训练轮数')
    train_parser.add_argument('--lr', type=float, default=0.001,
                            help='学习率')
    train_parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='权重衰减')
    train_parser.add_argument('--pretrained', action='store_true',
                            help='使用预训练模型')
    train_parser.add_argument('--class_weights', type=str, default=None,
                            help='类别权重，格式为"1,3"，表示真实:1，生成:3')
    train_parser.add_argument('--use_focal_loss', action='store_true',
                            help='使用焦点损失(Focal Loss)而不是交叉熵损失')
    train_parser.add_argument('--focal_gamma', type=float, default=2.0,
                            help='焦点损失的gamma参数，控制降低简单样本权重的程度')
    train_parser.add_argument('--real_boost', type=float, default=1.8,
                            help='真实图像类别的增强因子，提高真实图像识别率')
    train_parser.add_argument('--feature_weights', type=str, default=None,
                            help='五种特征的权重，格式为"0.3,0.2,0.2,0.15,0.15"，表示CNN:0.3,频率:0.2,共现:0.2,语义:0.15,哈希:0.15')
    train_parser.add_argument('--noise_scale', type=float, default=0.01,
                            help='高斯噪声注入比例，用于提高模型鲁棒性')
    train_parser.add_argument('--advanced_augmentation', action='store_true',
                            help='使用高级数据增强策略，包括更强的随机旋转、缩放、色彩抖动等')
    train_parser.add_argument('--dropout_rate', type=float, default=0.7,
                             help='Dropout比率，控制正则化强度')
    train_parser.add_argument('--temperature', type=float, default=1.5,
                             help='温度缩放参数，控制输出概率分布的平滑度')
    train_parser.add_argument('--save_dir', type=str, default='checkpoints',
                            help='模型保存目录')
    train_parser.add_argument('--log_dir', type=str, default='logs',
                            help='日志保存目录')
    train_parser.add_argument('--device', type=str, default='cuda',
                            help='设备(cuda或cpu)')
    train_parser.add_argument('--early_stopping', action='store_true',
                            help='使用早停机制')
    train_parser.add_argument('--patience', type=int, default=20,
                            help='早停耐心值，验证性能无提升的轮数')
    
    # 评估命令
    evaluate_parser = subparsers.add_parser('evaluate', help='评估模型')
    evaluate_parser.add_argument('--data_dir', type=str, required=True,
                               help='测试数据集路径')
    evaluate_parser.add_argument('--checkpoint', type=str, required=True,
                               help='模型检查点路径')
    evaluate_parser.add_argument('--subset_ratio', type=float, default=0.2,
                               help='数据集采样比例')
    evaluate_parser.add_argument('--img_size', type=int, default=256,
                               help='图像大小')
    evaluate_parser.add_argument('--batch_size', type=int, default=32,
                               help='批量大小')
    evaluate_parser.add_argument('--temperature', type=float, default=1.5,
                               help='温度缩放参数，控制输出概率分布的平滑度，值越大输出越平缓')
    evaluate_parser.add_argument('--feature_weights', type=str, default=None,
                               help='覆盖模型默认特征权重，格式为"0.3,0.2,0.2,0.15,0.15"')
    evaluate_parser.add_argument('--calibrate', action='store_true',
                               help='使用模型校准层对输出进行校准')
    evaluate_parser.add_argument('--balance_threshold', type=float, default=0.2,
                               help='类别平衡阈值，控制logits差距的最大值')
    evaluate_parser.add_argument('--output_dir', type=str, default='results',
                               help='输出目录')
    evaluate_parser.add_argument('--detailed_analysis', action='store_true',
                               help='输出详细分析结果，包括每个样本的置信度和特征权重')
    evaluate_parser.add_argument('--visualize', action='store_true',
                               help='可视化特征权重和类别分布')
    evaluate_parser.add_argument('--device', type=str, default='cuda',
                               help='设备(cuda或cpu)')
    evaluate_parser.add_argument('--threshold', type=float, default=0.5,
                               help='决策阈值，用于类别判定(默认: 0.5)')
    evaluate_parser.add_argument('--optimize_threshold', action='store_true',
                               help='自动搜索最佳阈值以平衡类别准确率')
    evaluate_parser.add_argument('--calibration_strength', type=float, default=0.3,
                               help='校准强度，越大平衡效果越强')
    evaluate_parser.add_argument('--seed', type=int, default=42,
                               help='随机种子')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='模型演示')
    demo_parser.add_argument('--checkpoint', type=str, required=True,
                           help='模型检查点路径')
    demo_parser.add_argument('--image', type=str, default=None,
                           help='输入图像路径')
    demo_parser.add_argument('--folder', type=str, default=None,
                           help='输入文件夹路径，用于批量检测')
    demo_parser.add_argument('--output_dir', type=str, default='demo_results',
                           help='输出目录')
    demo_parser.add_argument('--img_size', type=int, default=256,
                           help='图像大小')
    demo_parser.add_argument('--temperature', type=float, default=1.5,
                           help='温度缩放参数，控制输出概率的平滑度')
    demo_parser.add_argument('--feature_weights', type=str, default=None,
                           help='覆盖模型默认特征权重，格式为"0.3,0.2,0.2,0.15,0.15"')
    demo_parser.add_argument('--use_calibration', action='store_true',
                           help='使用模型校准层对输出进行校准')
    demo_parser.add_argument('--balance_threshold', type=float, default=0.2,
                           help='类别平衡阈值，控制logits差距的最大值')
    demo_parser.add_argument('--show_feature_analysis', action='store_true',
                           help='显示特征分析结果，包括各特征的贡献和重要性')
    demo_parser.add_argument('--device', type=str, default='cuda',
                           help='设备(cuda或cpu)')
    demo_parser.add_argument('--visualize', action='store_true',
                           help='可视化特征权重和类别分布')
    demo_parser.add_argument('--threshold', type=float, default=0.5,
                           help='检测阈值')
    
    return parser.parse_args()

def run_download(args):
    """运行数据集下载"""
    from data.download_sample import main as download_main
    # 将参数传递给下载脚本
    sys.argv = [sys.argv[0]]
    for key, value in vars(args).items():
        if key != 'command':  # 跳过命令本身
            if isinstance(value, bool) and value:
                sys.argv.append(f'--{key}')
            elif not isinstance(value, bool):
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
    download_main()

def run_prepare(args):
    """运行数据预处理"""
    from data.prepare_deepfacegen import prepare_dataset
    prepare_dataset(args)

def run_train(args):
    """运行训练"""
    from train.train_detector import main
    sys.argv = [sys.argv[0]]  # 清除现有参数
    
    # 添加所有参数
    for key, value in vars(args).items():
        if key != 'command':  # 跳过命令本身
            if isinstance(value, bool) and value:
                sys.argv.append(f'--{key}')
            elif not isinstance(value, bool):
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
    
    main()

def run_evaluate(args):
    """运行评估"""
    from evaluate.evaluate_detector import main
    sys.argv = [sys.argv[0]]  # 清除现有参数
    
    # 添加所有参数
    for key, value in vars(args).items():
        if key != 'command':  # 跳过命令本身
            if isinstance(value, bool) and value:
                sys.argv.append(f'--{key}')
            elif not isinstance(value, bool):
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
    
    main()

def run_demo(args):
    """运行演示"""
    from demo.demo import main
    sys.argv = [sys.argv[0]]  # 清除现有参数
    
    # 添加所有参数
    for key, value in vars(args).items():
        if key != 'command':  # 跳过命令本身
            if isinstance(value, bool) and value:
                sys.argv.append(f'--{key}')
            elif not isinstance(value, bool):
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))
    
    main()

def main():
    """主函数"""
    # 将项目根目录添加到系统路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        sys.path.append(os.path.dirname(current_dir))  # 添加父目录
    
    args = parse_args()
    
    if args.command == 'download':
        run_download(args)
    elif args.command == 'prepare':
        run_prepare(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        print(f"未知命令: {args.command}")
        print("可用命令: download, prepare, train, evaluate, demo")
        sys.exit(1)

if __name__ == '__main__':
    main()
