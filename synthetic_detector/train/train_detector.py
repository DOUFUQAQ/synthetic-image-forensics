import os
import sys
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import create_model
from data.dataset import get_dataloaders
from utils.metrics import compute_metrics
from utils.visualization import plot_feature_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train synthetic image detector')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to DeepFaceGen dataset')
    parser.add_argument('--subset_ratio', type=float, default=0.1,
                        help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of samples per class')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer regularization')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ResNet50')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='类别权重，格式为"1,3"，表示真实:1，生成:3')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='使用焦点损失(Focal Loss)而不是交叉熵损失')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='焦点损失的gamma参数，控制降低简单样本权重的程度')
    parser.add_argument('--real_boost', type=float, default=3.0,
                        help='真实图像类别的增强因子，用于提高真实图像识别率')
    parser.add_argument('--feature_weights', type=str, default=None,
                        help='各特征类型权重，格式为"0.3,0.2,0.2,0.15,0.15"，分别对应CNN,频率,共现,语义,哈希特征')
    parser.add_argument('--noise_scale', type=float, default=0.05,
                        help='特征噪声注入比例，用于提高模型鲁棒性')
    parser.add_argument('--advanced_augmentation', action='store_true',
                        help='启用高级数据增强，包括旋转、垂直翻转、高斯模糊等')
    parser.add_argument('--dropout_rate', type=float, default=0.7,
                        help='Dropout率，控制正则化强度，防止过拟合')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='温度缩放参数，用于校准模型输出概率')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', action='store_true',
                        help='使用早停机制')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值，验证性能无提升的轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 训练一个epoch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        logits = outputs['logits']
        # 兼容增强后的模型输出结构
        feature_weights = outputs.get('feature_weights', None)
        
        # 如果是加强版模型，还可能有其他特征信息
        raw_features = outputs.get('raw_features', {})
        features = outputs.get('features', None)
        
        # 计算损失
        loss = criterion(logits, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每100个批次打印一次
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    # 计算平均损失和准确率
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    
    # 增强的特征权重跟踪与可视化
    
    # 检查模型是否有可训练的base_weights参数
    has_learnable_weights = hasattr(model, 'base_weights') and isinstance(model.base_weights, nn.Parameter)
    if has_learnable_weights:
        learned_weights = model.base_weights.detach().cpu()
        print(f"\n\033[1;32m[训练中] Epoch {epoch} 学习到的特征权重: {learned_weights.numpy()}\033[0m")
        
        # 计算与初始权重的变化
        if hasattr(model, 'initial_weights'):
            init_weights = model.initial_weights.cpu()
            weight_changes = learned_weights - init_weights
            print(f"\033[0;33m权重变化(相对初始值): {weight_changes.numpy()}\033[0m")
            
            # 记录到TensorBoard
            for i, (w, change) in enumerate(zip(learned_weights, weight_changes)):
                writer.add_scalar(f'LearnedFeatureWeight/{i}', w.item(), epoch)
                writer.add_scalar(f'WeightChange/{i}', change.item(), epoch)
            
            # 计算权重分布的均匀度（entropy）
            weight_entropy = -torch.sum(learned_weights * torch.log(learned_weights + 1e-10))
            writer.add_scalar('WeightDistribution/Entropy', weight_entropy.item(), epoch)
    
    # 仍然处理从前向传播返回的特征权重
    if feature_weights is not None:
        avg_feature_weights = torch.mean(feature_weights, dim=0)  # 对批次计算平均值
        for i, w in enumerate(avg_feature_weights):
            writer.add_scalar(f'BatchFeatureWeight/{i}', w.item(), epoch)
    else:
        avg_feature_weights = None
    
    # 注意力权重可能不存在，检查模型输出
    attention_weights = outputs.get('attention_weights', None)
    if attention_weights is not None:
        avg_attention = torch.mean(attention_weights, dim=0)
        for i, w in enumerate(avg_attention):
            writer.add_scalar(f'AttentionWeight/{i}', w.item(), epoch)
    
    # 选择返回的特征权重 - 优先使用学习到的参数
    if has_learnable_weights:
        feature_weights_np = learned_weights.numpy()
    elif avg_feature_weights is not None:
        feature_weights_np = avg_feature_weights.detach().cpu().numpy()
    else:
        # 如果没有特征权重，返回默认值
        feature_weights_np = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 平等权重
        
    # 创建特征权重条形图并添加到TensorBoard
    try:
        feature_names = ['CNN', '频率', '共现', '语义', '哈希']
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(feature_names, feature_weights_np)
        ax.set_ylim([0, max(1.0, feature_weights_np.max() * 1.2)])
        ax.set_xlabel('特征类型')
        ax.set_ylabel('权重')
        ax.set_title(f'Epoch {epoch} 特征权重分布')
        
        # 添加百分比标签
        for bar, weight in zip(bars, feature_weights_np):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}\n({weight*100:.1f}%)',
                    ha='center', va='bottom')
        
        writer.add_figure('FeatureWeights/Distribution', fig, epoch)
        plt.close(fig)
    except Exception as e:
        print(f"\033[0;31m创建特征权重可视化图表失败: {e}\033[0m")
        
    return train_loss, train_acc, feature_weights_np

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    
    # 添加每个类别的识别结果跟踪
    class_correct = [0, 0]  # [real_correct, synthetic_correct]
    class_total = [0, 0]    # [real_total, synthetic_total]
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 测试时增强 - 多角度评估
            batch_size = inputs.size(0)
            tta_preds = []
            
            # 原始图像预测
            outputs = model(inputs)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            tta_preds.append(preds)
            
            # 水平翻转图像预测
            flipped_inputs = torch.flip(inputs, dims=[3])  # 水平翻转
            flipped_outputs = model(flipped_inputs)
            flipped_logits = flipped_outputs['logits']
            _, flipped_preds = flipped_logits.max(1)
            tta_preds.append(flipped_preds)
            
            # 合并测试时增强的预测结果
            tta_preds = torch.stack(tta_preds)
            
            # 简化测试时增强预测结果处理
            # 为了避免使用torch.mode可能带来的兼容性问题，使用更简单的方法
            
            # 默认使用原始模型的预测
            final_preds = preds
            
            # 我们在这里使用简化版的TTA:
            # 如果翻转前后预测都认为是真实图像，则增强认为是真实图像的信心
            # 这样可以减少误检率，提高对真实图像的识别率
            if tta_preds.size(0) >= 2:
                # 如果原始和翻转后预测都为类别0(真实图像)
                both_predict_real = (tta_preds[0] == 0) & (tta_preds[1] == 0)
                # 对于两个模型都预测为真实图像的样本，我们确定其为真实图像
                final_preds[both_predict_real] = 0  # 0代表真实图像类别
            
            # 计算损失
            loss = criterion(logits, targets)
            running_loss += loss.item()
            
            # 跟踪每个类别的准确率
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if final_preds[i].item() == label:
                    class_correct[label] += 1
            
            # 收集结果
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 收集正类概率
    
    # 计算平均损失
    val_loss = running_loss / len(val_loader)
    
    # 计算各种指标
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 计算各类别的准确率
    real_class_accuracy = class_correct[0] / max(class_total[0], 1) if class_total[0] > 0 else 0.0
    synthetic_class_accuracy = class_correct[1] / max(class_total[1], 1) if class_total[1] > 0 else 0.0
    
    # 准确率、精度、召回率、F1分数
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    
    # 计算ROC AUC
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError as e:
        print(f"警告: ROC AUC计算错误: {e}")
        auc = 0.5  # 默认值为随机猜测的AUC
    
    # 计算类别平衡指标
    class_balance = min(real_class_accuracy / max(synthetic_class_accuracy, 0.001), 
                        synthetic_class_accuracy / max(real_class_accuracy, 0.001))
    
    # 写入TensorBoard
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    writer.add_scalar('Validation/Precision', precision, epoch)
    writer.add_scalar('Validation/Recall', recall, epoch)
    writer.add_scalar('Validation/F1', f1, epoch)
    writer.add_scalar('Validation/AUC', auc, epoch)
    writer.add_scalar('Validation/RealAccuracy', real_class_accuracy, epoch)
    writer.add_scalar('Validation/SyntheticAccuracy', synthetic_class_accuracy, epoch)
    writer.add_scalar('Validation/ClassBalance', class_balance, epoch)
    
    # 检查模型是否有可训练的base_weights
    has_learnable_weights = hasattr(model, 'base_weights') and isinstance(model.base_weights, nn.Parameter)
    
    if has_learnable_weights:
        # 提取当前学习到的权重
        learned_weights = model.base_weights.detach().cpu()
        print(f"\n\033[1;36m[验证中] Epoch {epoch} 特征权重: {learned_weights.numpy()}\033[0m")
        
        # 检查权重的分布情况 - 计算标准差衡量权重的集中程度
        weight_std = torch.std(learned_weights)
        writer.add_scalar('WeightDistribution/StdDev', weight_std.item(), epoch)
        print(f"\033[0;36m权重标准差: {weight_std.item():.4f} (越大表示权重分布越不均匀)\033[0m")
        
        # 提取权重数据用于可视化
        weights_np = learned_weights.numpy()
        
        # 创建特征权重条形图
        try:
            feature_names = ['CNN', '频率', '共现', '语义', '哈希']
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 创建特征权重条形图
            bars = ax.bar(feature_names, weights_np, alpha=0.7, color='skyblue')
            ax.set_ylim([0, max(1.0, weights_np.max() * 1.2)])
            
            # 突出显示最大权重
            max_idx = np.argmax(weights_np)
            bars[max_idx].set_color('darkred')
            bars[max_idx].set_alpha(1.0)
            
            # 添加标签和标题
            ax.set_xlabel('特征类型')
            ax.set_ylabel('权重值')
            ax.set_title(f'验证 Epoch {epoch} - 特征权重分布 (AUC={auc:.4f})')
            
            # 添加权重值标签
            for bar, weight in zip(bars, weights_np):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{weight:.3f}\n({weight*100:.1f}%)',
                        ha='center', va='bottom')
            
            # 保存到TensorBoard
            writer.add_figure('Validation/FeatureWeights', fig, epoch)
            plt.close(fig)
        except Exception as e:
            print(f"\033[0;31m创建验证特征权重可视化失败: {e}\033[0m")
        
        # 将特征权重添加到指标字典中
        feature_weights = {f'weight_{i}': w for i, w in enumerate(weights_np)}
    else:
        feature_weights = {}
        
    # 返回增强的指标字典
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'real_class_accuracy': real_class_accuracy,
        'synthetic_class_accuracy': synthetic_class_accuracy,
        'class_balance': class_balance,
        **feature_weights  # 添加特征权重到指标中
    }
    
    return val_loss, metrics

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.save_dir)
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.log_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建TensorBoard日志
    timestr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(log_dir, f'detector_{timestr}'))
    
    # 获取数据加载器 - 数据增强已在dataset.py中定义
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        subset_ratio=args.subset_ratio,
        max_samples=args.max_samples,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        advanced_augmentation=args.advanced_augmentation  # 传递高级数据增强选项
    )
    
    if args.advanced_augmentation:
        print("启用高级数据增强: 旋转、垂直翻转、高斯模糊等已激活")
    # 解析特征权重参数（如果提供）
    feature_weights = None
    if args.feature_weights:
        try:
            feature_weights = [float(w) for w in args.feature_weights.split(',')]
            if len(feature_weights) != 5:  # 确保正好有5个权重值
                print(f"警告: 特征权重必须包含5个值，但提供了{len(feature_weights)}个。使用默认权重。")
                feature_weights = None
            else:
                print(f"使用自定义特征权重: {feature_weights}")
        except Exception as e:
            print(f"解析特征权重时出错: {e}，使用默认权重。")
            feature_weights = None
    
    model = create_model(
        pretrained=args.pretrained,
        feature_dim=args.feature_dim,
        feature_weights=feature_weights,
        noise_scale=args.noise_scale,
        dropout_rate=args.dropout_rate,
        temperature=args.temperature,
        real_boost=args.real_boost
    )
    model = model.to(device)
    
    # 增强的焦点损失类 - 包含类别平衡机制
    class EnhancedFocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean', real_boost=1.5):
            super(EnhancedFocalLoss, self).__init__()
            self.alpha = alpha  # 类别权重
            self.gamma = gamma  # 降低简单样本的权重
            self.reduction = reduction
            self.real_boost = real_boost  # 真实图像类别增强因子
            self.epoch_count = 0  # 记录训练轮数
            
        def update_epoch(self):
            """ 更新训练轮数计数器 """
            self.epoch_count += 1
            
            # 不再自动增加真实图像类别权重，保持初始设置的权重
            
            # 每10轮打印当前权重状态
            if self.epoch_count % 10 == 0 and self.alpha is not None:
                print(f"[Epoch {self.epoch_count}] 当前类别权重: 真实={self.alpha[0]:.2f}, 合成={self.alpha[1]:.2f}")
                print(f"[Epoch {self.epoch_count}] 当前图像增强因子: {self.real_boost:.2f}")
            
        
        def forward(self, inputs, targets):
            batch_size = targets.size(0)
            
            # 生成平衡的样本权重
            sample_weights = torch.ones_like(targets, dtype=torch.float)
            
            # 标记不同类别的样本
            real_indices = (targets == 0)
            synthetic_indices = (targets == 1)
            
            # 平等对待两个类别，无需应用自适应因子
            # 我们使用固定的boost_factor，不再每轮增加
            if real_indices.sum() > 0:
                sample_weights[real_indices] = self.real_boost
            if synthetic_indices.sum() > 0:
                sample_weights[synthetic_indices] = self.real_boost  # 对合成图像也使用相同的权重
            
            # 计算交叉熟损失
            if self.alpha is not None:
                ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
            else:
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
            # 计算预测概率
            pt = torch.exp(-ce_loss)
            
            # 应用焦点损失
            focal_weight = (1 - pt) ** self.gamma
            
            # 平等对待两个类别，不再特别增强真实图像
            
            # 应用焦点权重
            focal_loss = focal_weight * ce_loss
            
            # 应用样本权重
            weighted_focal_loss = focal_loss * sample_weights
            
            # 添加类别平衡惩罚项，鼓励两个类别的损失接近
            if real_indices.sum() > 0 and synthetic_indices.sum() > 0:
                real_loss = weighted_focal_loss[real_indices].mean()
                synthetic_loss = weighted_focal_loss[synthetic_indices].mean()
                # 增大平衡惩罚力度，更强调平衡两个类别
                imbalance_penalty = torch.abs(real_loss - synthetic_loss) * 0.5
                final_loss = weighted_focal_loss.mean() + imbalance_penalty
            else:
                final_loss = weighted_focal_loss.mean()
                
            if self.reduction == 'mean':
                return final_loss
            elif self.reduction == 'sum':
                return final_loss * batch_size
            else:
                return weighted_focal_loss
    
    # 定义损失函数和优化器
    # 处理类别权重 - 尽量保持用户指定的权重
    if args.class_weights:
        try:
            weights = [float(w) for w in args.class_weights.split(',')]
            # 不再强制真实图像权重高于合成图像，尊重用户选择
            class_weights = torch.FloatTensor(weights).to(device)
            print(f"使用用户指定的类别权重: 真实={weights[0]}, 合成={weights[1]}")
            
            if args.use_focal_loss:
                print(f"使用焦点损失，gamma={args.focal_gamma}, 图像增强因子={args.real_boost}")
                criterion = EnhancedFocalLoss(alpha=class_weights, gamma=args.focal_gamma, real_boost=args.real_boost)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        except Exception as e:
            print(f"警告: 解析类别权重出错: {e}")
            print("使用平衡的默认类别权重")
            
            # 使用平衡的默认权重
            default_weights = torch.FloatTensor([1.0, 1.0]).to(device)
            
            if args.use_focal_loss:
                print(f"使用焦点损失，gamma={args.focal_gamma}, 图像增强因子={args.real_boost}")
                criterion = EnhancedFocalLoss(alpha=default_weights, gamma=args.focal_gamma, real_boost=args.real_boost)
            else:
                criterion = nn.CrossEntropyLoss(weight=default_weights)
    else:
        # 如果没有指定类别权重，使用平衡的默认权重
        default_weights = torch.FloatTensor([1.0, 1.0]).to(device)  # 平等对待两个类别
        print(f"使用平衡的默认类别权重: 真实=1.0, 合成=1.0")
        
        if args.use_focal_loss:
            print(f"使用焦点损失，gamma={args.focal_gamma}, 图像增强因子={args.real_boost}")
            criterion = EnhancedFocalLoss(alpha=default_weights, gamma=args.focal_gamma, real_boost=args.real_boost)
        else:
            criterion = nn.CrossEntropyLoss(weight=default_weights)
    
    # 为特征融合权重参数设置更高的学习率，促进快速学习
    base_lr = args.lr
    fusion_lr = args.lr * 10.0  # 特征融合权重使用10倍学习率
    
    # 查找模型中的base_weights参数
    fusion_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'base_weights' in name:
            print(f"为特征融合权重'{name}'设置高学习率: {fusion_lr}")
            fusion_params.append(param)
        else:
            other_params.append(param)
    
    # 设置不同参数组的不同学习率
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': args.weight_decay * 2.0}  # 特征权重使用更高的权重衰减
    ], weight_decay=args.weight_decay * 1.5)  # 基础权重衰减不变
    
    # 打印初始特征权重
    if hasattr(model, 'base_weights'):
        print(f"初始特征权重: {model.base_weights.data}")
    if hasattr(model, 'initial_weights'):
        print(f"参考初始权重: {model.initial_weights}")
    
    
    # 学习率调度器
    try:
        # 尝试使用verbose参数（新版PyTorch支持）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5, verbose=True
        )
    except TypeError:
        # 如果失败，尝试不使用verbose参数（兼容旧版PyTorch）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )
        print("注意: 使用不带verbose参数的ReduceLROnPlateau调度器")
    
    # 训练模型
    best_val_acc = 0.0
    best_balanced_score = 0.0  # 跟踪平衡评分而非单纯准确率
    best_val_loss = float('inf')
    best_balance_rating = 0.0  # 跟踪类别平衡度
    best_min_class_acc = 0.0  # 跟踪最低类别准确率
    feature_weights_history = []
    
    # 初始化模型集成列表
    model_ensemble = []
    max_ensemble_size = 3  # 最多保存3个模型进行集成
    
    # 早停相关变量
    if args.early_stopping:
        # 确保耐心值正确传递
        patience = args.patience if hasattr(args, 'patience') and args.patience > 0 else 20
        print(f"启用早停机制，耐心值: {patience} 轮")
        patience_counter = 0
        best_val_loss = float('inf')
    else:
        patience = 15  # 默认耐心值
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        # 如果使用增强焦点损失，更新epoch计数器
        if hasattr(criterion, 'update_epoch'):
            criterion.update_epoch()
            
        # 训练一个epoch
        train_loss, train_acc, feature_weights = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 验证
        val_loss, metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存特征权重历史
        feature_weights_history.append(feature_weights)
        
        # 从验证指标中提取数据
        val_acc = metrics['accuracy']
        real_class_acc = metrics['real_class_accuracy']
        synthetic_class_acc = metrics['synthetic_class_accuracy']
        
        # 打印训练状态
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Real Class Acc: {real_class_acc:.4f}, Synthetic Class Acc: {synthetic_class_acc:.4f}, '
              f'Gap: {abs(real_class_acc - synthetic_class_acc):.4f}')
        
        # 打印特征权重
        weights_str = ', '.join([f'{w:.4f}' for w in feature_weights])
        print(f'Feature Weights: [{weights_str}]')
        
        # 使用真正的平衡评分 - 平等看待两个类别
        # 平衡分数 = 0.5 * 真实类别准确率 + 0.5 * 合成类别准确率
        balanced_score = 0.5 * real_class_acc + 0.5 * synthetic_class_acc
        print(f'平衡评分: {balanced_score:.4f} (50% 真实类别 + 50% 合成类别)')
        
        # 添加类别差距评估 - 稳定结果更重要
        class_gap = abs(real_class_acc - synthetic_class_acc)
        balance_rating = 1.0 - class_gap  # 差距越小，平衡度越高
        print(f'类别平衡度: {balance_rating:.4f} (1 - |真实准确率 - 合成准确率|)')
        
        # 早停检查 - 根据验证集损失和类别平衡评分综合考虑
        if args.early_stopping:
            improved = False
            
            # 如果验证集损失降低
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                print(f"验证集损失降低至 {val_loss:.4f}")
            
            # 平等考虑两个类别的性能
            # 考虑平衡评分和类别平衡度
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                improved = True
                print(f"平衡评分提高至 {balanced_score:.4f}, 真实准确率: {real_class_acc:.4f}, 合成准确率: {synthetic_class_acc:.4f}")
            
            # 当两个类别的准确率差距变小时，也认为是改进
            if balance_rating > 0.7 and balance_rating > best_balance_rating:
                best_balance_rating = balance_rating
                improved = True
                print(f"类别平衡度提高至 {balance_rating:.4f}，差距减小")
                
            # 确保两个类别的准确率都达到最低要求
            min_class_acc = min(real_class_acc, synthetic_class_acc)
            if min_class_acc > best_min_class_acc:
                best_min_class_acc = min_class_acc
                improved = True
                print(f"最低类别准确率提高至 {min_class_acc:.4f}")

                
            # 更新耐心计数器
            if improved:
                patience_counter = 0
                print(f"模型性能改善，重置耐心计数器")
            else:
                patience_counter += 1
                print(f"模型性能未改善，耐心计数: {patience_counter}/{args.patience}")
                
                # 如果达到耐心限制，触发早停
                if patience_counter >= patience:
                    print(f"早停触发! {patience} 轮内模型性能未改善")
                    break
        
        # 根据真实图像准确率和平衡分数判断是否为最佳模型
        is_best = balanced_score > best_val_acc
        best_val_acc = max(balanced_score, best_val_acc)
        
        # 定期保存检查点
        if epoch % args.save_interval == 0 or is_best:
            checkpoint_path = os.path.join(save_dir, f'detector_epoch_{epoch}.pth')
            best_model_path = os.path.join(save_dir, 'detector_best.pth')
            ensemble_path = os.path.join(save_dir, f'ensemble_model_{epoch}.pth')
            
            # 保存当前检查点
            try:
                model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'real_class_acc': real_class_acc,
                    'synthetic_class_acc': synthetic_class_acc,
                    'balanced_score': balanced_score,
                    'feature_weights': feature_weights
                }
                
                torch.save(model_state, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')
                
                # 保存最佳模型
                if is_best:
                    torch.save(model_state, best_model_path)
                    print(f'Best model saved with balanced score: {balanced_score:.4f} '
                          f'(real: {real_class_acc:.4f}, synthetic: {synthetic_class_acc:.4f})')
                    
                # 更新模型集成列表
                if real_class_acc > 0.5 and balanced_score > 0.65:  # 只添加真实图像识别率较高的模型
                    model_ensemble.append({
                        'state_dict': model.state_dict().copy(),
                        'real_acc': real_class_acc,
                        'synthetic_acc': synthetic_class_acc,
                        'epoch': epoch
                    })
                    print(f'Model added to ensemble (ensemble size: {len(model_ensemble)})')
                    
                    # 如果集成模型太多，保留最好的N个
                    if len(model_ensemble) > max_ensemble_size:
                        # 根据真实类别准确率排序
                        model_ensemble = sorted(model_ensemble, key=lambda x: x['real_acc'], reverse=True)[:max_ensemble_size]
                    
                    # 保存集成模型信息
                    torch.save({
                        'ensemble': model_ensemble,
                        'epoch': epoch
                    }, ensemble_path)
                    print(f'Ensemble models saved to {ensemble_path}')
                    
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
                print("继续训练而不中断...")
    
    # 绘制特征权重变化
    weight_fig = plot_feature_weights(np.array(feature_weights_history))
    writer.add_figure('FeatureWeightsHistory', weight_fig, args.epochs)
    
    writer.close()
    print("Training completed!")

if __name__ == '__main__':
    main()
