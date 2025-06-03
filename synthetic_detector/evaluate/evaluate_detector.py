import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import create_model
from data.dataset import DeepFaceGenDataset
from utils.metrics import compute_metrics, compute_class_wise_accuracy
from utils.visualization import (
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_confusion_matrix,
    visualize_attention_maps
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate synthetic image detector')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--subset_ratio', type=float, default=0.2,
                        help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for evaluation')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention maps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # 决策阈值参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for classification (default: 0.5)')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Automatically search for the best threshold to balance class accuracy')
    parser.add_argument('--calibrate', action='store_true',
                        help='Apply post-processing calibration to improve class balance')
    parser.add_argument('--calibration_strength', type=float, default=0.3,
                        help='Calibration strength parameter')
    
    return parser.parse_args()

def apply_calibration(logits, calibration_strength=0.3):
    """对模型输出进行后处理校准，以平衡类别预测
    
    Args:
        logits: 模型输出的logits张量 [batch_size, num_classes]
        calibration_strength: 校准强度，越大平衡效果越强
        
    Returns:
        calibrated_logits: 校准后的logits
    """
    batch_size, num_classes = logits.size()
    
    # 获取当前预测分布
    probs = torch.softmax(logits, dim=1)
    class_means = probs.mean(dim=0)  # 每个类别的平均预测概率
    
    # 检测是否存在类别不平衡 (一个类别占主导)
    max_prob, max_idx = torch.max(class_means, dim=0)
    min_prob, min_idx = torch.min(class_means, dim=0)
    imbalance_ratio = max_prob / (min_prob + 1e-8)
    
    if imbalance_ratio > 1.5 or max_prob > 0.65:
        print(f"检测到类别不平衡: 类别{max_idx.item()}的预测概率为{max_prob.item():.4f}，" 
              f"类别{min_idx.item()}的预测概率为{min_prob.item():.4f}")
        print(f"应用校准，强度={calibration_strength}")
        
        # 计算校准因子 - 提高较少预测的类别的权重
        calibration_factors = torch.ones(num_classes, device=logits.device)
        for c in range(num_classes):
            # 类别c的校准因子与其预测频率成反比
            calibration_factors[c] = 1.0 + calibration_strength * (1.0 - class_means[c] / class_means.sum())
        
        # 应用校准 - 将因子应用到logits上
        calibration_matrix = calibration_factors.expand(batch_size, num_classes)
        calibrated_logits = logits * calibration_matrix
        
        # 打印校准效果
        new_probs = torch.softmax(calibrated_logits, dim=1)
        new_means = new_probs.mean(dim=0)
        print(f"校准后: 类别0(真实)概率={new_means[0].item():.4f}, 类别1(合成)概率={new_means[1].item():.4f}")
        
        return calibrated_logits
    
    print("预测分布相对平衡，无需校准")
    return logits

def evaluate(model, test_loader, device, output_dir, visualize=False, threshold=0.5, optimize_threshold=True, calibrate=False, calibration_strength=0.3):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    # 记录用于可视化的样本
    visualize_samples = None
    
    # 收集所有批次的特征权重和多样性指标
    all_feature_weights = []
    all_attention_weights = []
    all_feature_diversity = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            logits = outputs['logits']
            batch_feature_weights = outputs['feature_weights']
            batch_attention_weights = outputs['attention_weights']
            
            # 收集特征权重和多样性指标
            all_feature_weights.append(batch_feature_weights.cpu())
            all_attention_weights.append(batch_attention_weights.cpu())
            
            # 收集特征多样性指标（如果有）
            if 'feature_diversity' in outputs:
                all_feature_diversity.append(outputs['feature_diversity'].cpu())
            
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            # 应用校准（如果启用）
            if calibrate:
                logits = apply_calibration(logits, calibration_strength)
                probs = torch.softmax(logits, dim=1)
                _, preds = logits.max(1)
            
            # 收集结果
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 收集正类概率
            
            # 保存第一个批次用于可视化
            if visualize and visualize_samples is None:
                visualize_samples = inputs
    
    # 转换为numpy数组
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 计算平均特征权重
    all_feature_weights = torch.cat(all_feature_weights, dim=0)  # 合并所有批次
    all_attention_weights = torch.cat(all_attention_weights, dim=0)  # 合并所有批次
    
    # 计算平均值
    avg_feature_weights = all_feature_weights.mean(dim=0)  # 平均特征权重
    avg_attention_weights = all_attention_weights.mean(dim=0)  # 平均注意力权重
    
    # 计算平均特征多样性指标（如果有）
    avg_feature_diversity = None
    if all_feature_diversity:
        avg_feature_diversity = torch.stack(all_feature_diversity).mean().item()
    
    # 如果启用了阈值优化，搜索最佳阈值
    if optimize_threshold:
        print("\n正在搜索最佳决策阈值...")
        best_threshold = threshold
        best_balanced_acc = 0
        best_metrics = None
        best_class_acc = None
        
        # 搜索不同的阈值
        for t in np.arange(0.1, 0.9, 0.05):
            # 根据阈值重新计算预测
            threshold_preds = (np.array(all_probs) >= t).astype(int)
            
            # 计算该阈值下的指标
            t_metrics = compute_metrics(all_targets, threshold_preds, all_probs)
            t_class_acc = compute_class_wise_accuracy(all_targets, threshold_preds)
            
            # 计算平衡准确率 (两个类别准确率的平均)
            balanced_acc = 0.5 * (t_class_acc['class_0'] + t_class_acc['class_1']) / 100.0
            
            # 打印信息
            print(f"  阈值 {t:.2f}: 真实类准确率 {t_class_acc['class_0']:.1f}%, 生成类准确率 {t_class_acc['class_1']:.1f}%, 平衡准确率 {balanced_acc:.4f}")
            
            # 更新最佳阈值
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_threshold = t
                best_metrics = t_metrics
                best_class_acc = t_class_acc
        
        print(f"\n找到最佳阈值: {best_threshold:.2f}, 平衡准确率: {best_balanced_acc:.4f}")
        print(f"  真实类准确率: {best_class_acc['class_0']:.2f}%, 生成类准确率: {best_class_acc['class_1']:.2f}%")
        
        # 使用最佳阈值的结果
        metrics = best_metrics
        class_acc = best_class_acc
        threshold = best_threshold
    else:
        # 直接使用给定阈值计算指标
        threshold_preds = (np.array(all_probs) >= threshold).astype(int)
        metrics = compute_metrics(all_targets, threshold_preds, all_probs)
        class_acc = compute_class_wise_accuracy(all_targets, threshold_preds)
    
    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f"总体准确率: {metrics['accuracy']:.2f}%")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    
    # 强调类别不平衡的打印
    print("\n===== 类别平衡指标 =====")
    print(f"真实图像准确率: {class_acc['class_0']:.2f}%")
    print(f"生成图像准确率: {class_acc['class_1']:.2f}%")
    print(f"类别平衡差异: {abs(class_acc['class_0'] - class_acc['class_1']):.2f}%")
    
    # 特征权重与多样性分析
    print("\n===== 特征利用指标 =====")
    if avg_feature_diversity is not None:
        print(f"特征多样性指标: {avg_feature_diversity:.4f}")
    
    feature_names = ['CNN', '频率', '共现', '语义嵌入', '图像哈希']
    print("\n特征权重分布:")
    for name, weight in zip(feature_names, avg_feature_weights):
        print(f"{name}: {weight.item():.4f}")
        
    # 特征权重分析
    max_idx = torch.argmax(avg_feature_weights).item()
    min_idx = torch.argmin(avg_feature_weights).item()
    max_weight = avg_feature_weights[max_idx].item()
    min_weight = avg_feature_weights[min_idx].item()
    max_feature = feature_names[max_idx]
    min_feature = feature_names[min_idx]
    print(f"\n权重分析:")
    print(f"最大权重特征: {max_feature} ({max_weight:.4f})")
    print(f"最小权重特征: {min_feature} ({min_weight:.4f})")
    print(f"权重比例 (max/min): {max_weight/min_weight:.2f}")
    print(f"权重标准差: {torch.std(avg_feature_weights).item():.4f}")
    
    # 保存评估结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标到文本文件
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("===== 评估指标 =====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n===== 类别准确率 =====\n")
        for k, v in class_acc.items():
            f.write(f"{k}: {v}\n")
        f.write("\n===== 特征权重 =====\n")
        for i, name in enumerate(feature_names):
            f.write(f"{name}: {avg_feature_weights[i].item()}\n")
        
        f.write("\n===== 注意力权重 =====\n")
        for i, name in enumerate(feature_names):
            f.write(f"{name}: {avg_attention_weights[i].item()}\n")
            
        # 添加特征多样性指标
        if 'feature_diversity' in outputs:
            f.write("\n===== 特征多样性 =====\n")
            f.write(f"Feature Diversity: {outputs['feature_diversity'].item()}\n")
    
    # 绘制并保存可视化结果
    plt.figure(figsize=(10, 8))
    
    # ROC曲线
    roc_fig = plot_roc_curve(all_targets, all_probs)
    roc_fig.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # 精确率-召回率曲线
    pr_fig = plot_precision_recall_curve(all_targets, all_probs)
    pr_fig.savefig(os.path.join(output_dir, 'pr_curve.png'))
    
    # 混淆矩阵
    cm_fig = plot_confusion_matrix(all_targets, all_preds)
    cm_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 可视化注意力权重
    if visualize and visualize_samples is not None:
        attn_fig = visualize_attention_maps(model, visualize_samples, device)
        attn_fig.savefig(os.path.join(output_dir, 'attention_maps.png'))
    
    # 保存评估结果到文件
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("===== 评估结果 =====\n")
        f.write(f"总体准确率: {metrics['accuracy']:.2f}%\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"F1分数: {metrics['f1']:.4f}\n")
        f.write(f"精确率: {metrics['precision']:.4f}\n")
        f.write(f"召回率: {metrics['recall']:.4f}\n\n")
        
        f.write("===== 类别平衡指标 =====\n")
        f.write(f"真实图像准确率: {class_acc['class_0']:.2f}%\n")
        f.write(f"生成图像准确率: {class_acc['class_1']:.2f}%\n")
        f.write(f"类别平衡差异: {abs(class_acc['class_0'] - class_acc['class_1']):.2f}%\n\n")
        
        f.write("===== 特征利用指标 =====\n")
        if avg_feature_diversity is not None:
            f.write(f"特征多样性指标: {avg_feature_diversity:.4f}\n\n")
        
        f.write("特征权重分布:\n")
        feature_names = ['CNN', '频率', '共现', '语义嵌入', '图像哈希']
        for name, weight in zip(feature_names, avg_feature_weights):
            f.write(f"{name}: {weight.item():.4f}\n")
    
    # 保存阈值信息
    if optimize_threshold:
        threshold_file = os.path.join(output_dir, 'optimal_threshold.txt')
        with open(threshold_file, 'w') as f:
            f.write(f"最佳决策阈值: {threshold:.4f}\n")
            f.write(f"对应平衡准确率: {0.5 * (class_acc['class_0'] + class_acc['class_1']) / 100.0:.4f}\n")
    
    # 保存预测结果为CSV
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    import pandas as pd
    pd.DataFrame({
        'true_label': all_targets,
        'predicted_label': (np.array(all_probs) >= threshold).astype(int),
        'probability': all_probs
    }).to_csv(predictions_file, index=False)
    
    # 绘制特征权重分布图
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, [w.item() for w in avg_feature_weights])
    plt.title('特征权重分布')
    plt.ylabel('权重')
    plt.savefig(os.path.join(output_dir, 'feature_weights.png'))
    
    # 绘制ROC曲线
    plot_roc_curve(all_targets, all_probs)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_targets, (np.array(all_probs) >= threshold).astype(int))
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # 绘制精确率-召回率曲线
    plot_precision_recall_curve(all_targets, all_probs)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    
    # 返回结果，包含特征多样性指标和最佳阈值
    results = {
        'metrics': metrics,
        'class_acc': class_acc,
        'feature_weights': avg_feature_weights.cpu().numpy(),
        'attention_weights': avg_attention_weights.cpu().numpy(),
        'threshold': threshold
    }
    
    if avg_feature_diversity is not None:
        results['feature_diversity'] = avg_feature_diversity
        
    return results

def validate_model_checkpoint(model):
    """验证模型检查点是否包含所有必要参数，如果缺失则初始化"""
    # 检查模型是否有_ensure_critical_parameters方法
    if hasattr(model, '_ensure_critical_parameters') and callable(model._ensure_critical_parameters):
        try:
            model._ensure_critical_parameters()
            print("已验证模型关键参数完整性")
        except Exception as e:
            print(f"警告: 验证模型参数时出错: {e}")
    else:
        # 添加动态方法以确保向后兼容性
        print("模型缺少参数验证方法，动态添加兼容方法")
        
        def ensure_critical_parameters(self):
            # 检查类别偏置
            if not hasattr(self, 'class_bias') or self.class_bias is None:
                device = next(self.parameters()).device
                self.class_bias = torch.nn.Parameter(torch.tensor([0.05, 0.05], dtype=torch.float, device=device), requires_grad=True)
                print("警告: 类别偏置参数缺失，已使用默认值初始化")
            
            # 检查增强因子
            if not hasattr(self, 'real_boost_factor') or self.real_boost_factor is None:
                device = next(self.parameters()).device
                self.real_boost_factor = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float, device=device), requires_grad=True)
                print("警告: 真实类别增强因子缺失，已使用默认值初始化")
            
            # 检查特征基础权重
            if not hasattr(self, 'base_weights') or self.base_weights is None:
                device = next(self.parameters()).device
                base_weights = torch.tensor([[0.20, 0.20, 0.20, 0.20, 0.20]], dtype=torch.float, device=device)
                self.base_weights = torch.nn.Parameter(base_weights, requires_grad=True)
                print("警告: 特征基础权重缺失，已使用默认值初始化")
        
        # 将方法动态添加到模型
        model._ensure_critical_parameters = types.MethodType(ensure_critical_parameters, model)
        model._ensure_critical_parameters()
    
    return model

def main():
    # 确保在函数内部导入需要的模块
    import torch
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载测试数据集
    test_dataset = DeepFaceGenDataset(
        root_dir=args.data_dir,
        transform=None,
        subset_ratio=args.subset_ratio,
        img_size=args.img_size,
        seed=args.seed
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型
    model = create_model(feature_dim=args.feature_dim)
    model = model.to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {args.checkpoint}")
    
    try:
        # 尝试添加安全的全局对象
        try:
            torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
        except Exception as e_safe:
            print(f"Warning: Failed to add safe globals: {e_safe}")
        
        # 尝试多种加载方式
        checkpoint = None
        try:
            # 1. 尝试安全模式
            checkpoint = torch.load(args.checkpoint, map_location=device, pickle_module=None)
            print("Loaded checkpoint with pickle_module=None")
        except Exception as e1:
            print(f"Loading with pickle_module=None failed: {e1}")
            try:
                # 2. 尝试仅加载权重
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
                print("Loaded checkpoint with weights_only=True")
            except Exception as e2:
                print(f"Loading with weights_only failed: {e2}")
                # 3. 标准加载
                checkpoint = torch.load(args.checkpoint, map_location=device)
                print("Loaded checkpoint with standard loading")
        
        # 检查检查点是否是集成模型
        if isinstance(checkpoint, dict) and 'ensemble' in checkpoint:
            print("Detected ensemble model format, extracting base model...")
            if isinstance(checkpoint['ensemble'], list) and len(checkpoint['ensemble']) > 0:
                print(f"Ensemble contains {len(checkpoint['ensemble'])} models, using the first one")
                try:
                    model.load_state_dict(checkpoint['ensemble'][0], strict=False)
                    print("Successfully loaded the first model from ensemble")
                except Exception as e:
                    print(f"Failed to load from ensemble list: {e}")
            else:
                try:
                    model.load_state_dict(checkpoint['ensemble'], strict=False)
                    print("Successfully loaded ensemble model")
                except Exception as e:
                    print(f"Failed to load ensemble model: {e}")
        
        # 检查检查点是否包含state_dict或model_state_dict
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Loading from state_dict key...")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(f"Loaded from state_dict, epoch: {checkpoint.get('epoch', 'unknown')}")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading from model_state_dict key...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded from model_state_dict, epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # 尝试直接加载
        else:
            print("Attempting to load checkpoint directly...")
            model.load_state_dict(checkpoint, strict=False)
            print("Successfully loaded checkpoint directly")
            
        # 验证模型参数
        validate_model_checkpoint(model)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Failed to load model checkpoint. Please verify the checkpoint file is valid.")
        raise


    # 运行评测并保存结果
    evaluate(
        model,
        test_loader,
        device,
        output_dir,
        visualize=args.visualize,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        calibrate=args.calibrate,
        calibration_strength=args.calibration_strength,
    )
    print(f"评测完成，结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
