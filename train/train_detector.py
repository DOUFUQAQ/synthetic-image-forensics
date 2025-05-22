import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gram_detection.detector import GramBasedDetector, DetectionLoss
from data.dataset import DetectionDataset
from torch.utils.data import DataLoader
from utils.transforms import get_detection_transforms
from utils.metrics import compute_detection_metrics, compute_detection_metrics_batch
from utils.visualization import plot_training_curves, save_training_info, plot_confusion_matrix, plot_roc_curve


def to_numpy(tensor):
    """
    安全地将张量转换为numpy数组，处理需要梯度的张量
    """
    return tensor.detach().cpu().numpy()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个轮次
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        avg_loss: 平均损失
        metrics: 训练指标
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_text_loss = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        
        # 前向传播
        logits, texture_embedding = model(images)
        
        # 计算损失
        loss, cls_loss, text_loss = criterion(logits, texture_embedding, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_text_loss += text_loss.item()
        
        # 收集预测结果
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_text_loss = total_text_loss / len(dataloader)
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_detection_metrics(
        to_numpy(all_preds), to_numpy(all_targets), to_numpy(all_probs[:, 1])
    )
    
    metrics['cls_loss'] = avg_cls_loss
    metrics['text_loss'] = avg_text_loss
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """
    在验证集上评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        avg_loss: 平均损失
        metrics: 验证指标
    """
    model.eval()
    total_loss = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            # 前向传播
            logits, texture_embedding = model(images)
            
            # 计算损失
            loss, _, _ = criterion(logits, texture_embedding, targets)
            
            total_loss += loss.item()
            
            # 收集预测结果
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_detection_metrics(
        to_numpy(all_preds), to_numpy(all_targets), to_numpy(all_probs[:, 1])
    )
    
    return avg_loss, metrics


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取数据变换
    train_transform, val_transform = get_detection_transforms(img_size=args.img_size)
    
    # 创建数据集和数据加载器
    # 创建训练数据集和数据加载器
    train_loader = DataLoader(
        DetectionDataset(
            os.path.join(args.data_dir, 'train', 'real'), 
            os.path.join(args.data_dir, 'train', 'generated'),
            transform=train_transform,
            mode='train'
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建验证数据集和数据加载器
    val_loader = DataLoader(
        DetectionDataset(
            os.path.join(args.data_dir, 'test', 'real'), 
            os.path.join(args.data_dir, 'test', 'generated'),
            transform=val_transform,
            mode='test'
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Validation dataset size: {len(val_loader.dataset)}')
    
    # 创建模型 - 增加dropout率
    model = GramBasedDetector(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=2,
        dropout_rate=0.7  # 增加dropout率以减少过拟合
    ).to(device)
    
    # 创建损失函数
    criterion = DetectionLoss().to(device)
    
    # 创建优化器 - 增强正则化
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4  # 增加权重衰减以加强正则化
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 训练状态追踪
    best_val_loss = float('inf')
    best_val_metric = 0
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # 添加早停相关属性
    patience = 8  # 连续8轮没有改善则停止训练
    patience_counter = 0
    min_epochs = 15  # 最少训练15轮
    
    # 开始训练
    print('Starting training...')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练一个轮次
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录损失和指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append([train_metrics[m] for m in metrics_names])
        val_metrics_history.append([val_metrics[m] for m in metrics_names])
        
        # 写入TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for metric in metrics_names:
            writer.add_scalar(f'Metrics/{metric}/train', train_metrics[metric], epoch)
            writer.add_scalar(f'Metrics/{metric}/val', val_metrics[metric], epoch)
        
        # 保存最佳模型（基于验证损失）- 添加异常处理
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                # 先保存到临时文件再重命名，避免部分写入导致的文件损坏
                temp_path = os.path.join(args.save_dir, 'temp_best_loss_model.pth')
                torch.save(model.state_dict(), temp_path)
                final_path = os.path.join(args.save_dir, 'best_loss_model.pth')
                
                # 如果存在旧文件，先删除
                if os.path.exists(final_path):
                    os.remove(final_path)
                    
                # 重命名临时文件
                os.rename(temp_path, final_path)
                print(f'Saved best loss model at epoch {epoch + 1}')
            except Exception as e:
                print(f'Warning: Failed to save best loss model: {str(e)}')
                # 继续训练而不中断
        
        # 保存最佳模型（基于验证指标，这里使用AUC）- 添加异常处理
        if val_metrics['auc'] > best_val_metric:
            best_val_metric = val_metrics['auc']
            try:
                # 先保存到临时文件再重命名，避免部分写入导致的文件损坏
                temp_path = os.path.join(args.save_dir, 'temp_best_metric_model.pth')
                torch.save(model.state_dict(), temp_path)
                final_path = os.path.join(args.save_dir, 'best_metric_model.pth')
                
                # 如果存在旧文件，先删除
                if os.path.exists(final_path):
                    os.remove(final_path)
                    
                # 重命名临时文件
                os.rename(temp_path, final_path)
                print(f'Saved best metric model at epoch {epoch + 1}')
            except Exception as e:
                print(f'Warning: Failed to save best metric model: {str(e)}')
                # 继续训练而不中断
                
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1  # 增加耐心计数器
            
        # 检查是否应该早停
        if patience_counter >= patience and epoch + 1 >= min_epochs:
            print(f'Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.')
            break
        
        # 定期保存检查点 - 添加异常处理
        if (epoch + 1) % args.save_interval == 0:
            try:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'best_val_loss': best_val_loss,
                    'best_val_metric': best_val_metric
                }
                
                # 先保存到临时文件
                temp_path = os.path.join(args.save_dir, f'temp_checkpoint_epoch_{epoch + 1}.pth')
                torch.save(checkpoint_data, temp_path)
                
                final_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                # 如果存在旧文件，先删除
                if os.path.exists(final_path):
                    os.remove(final_path)
                    
                # 重命名临时文件
                os.rename(temp_path, final_path)
                print(f'Saved checkpoint at epoch {epoch + 1}')
            except Exception as e:
                print(f'Warning: Failed to save checkpoint: {str(e)}')
                # 继续训练而不中断
        
        # 打印进度
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch + 1}/{args.epochs}, Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Metrics: Accuracy={train_metrics["accuracy"]:.4f}, '
              f'Precision={train_metrics["precision"]:.4f}, '
              f'Recall={train_metrics["recall"]:.4f}, '
              f'F1={train_metrics["f1"]:.4f}, '
              f'AUC={train_metrics["auc"]:.4f}')
        print(f'Val Metrics: Accuracy={val_metrics["accuracy"]:.4f}, '
              f'Precision={val_metrics["precision"]:.4f}, '
              f'Recall={val_metrics["recall"]:.4f}, '
              f'F1={val_metrics["f1"]:.4f}, '
              f'AUC={val_metrics["auc"]:.4f}')
        print('-' * 80)
    
    # 训练结束
    total_time = time.time() - start_time
    print(f'Training completed in {total_time / 60:.2f} minutes')
    
    # 保存最终模型 - 添加异常处理
    try:
        # 先保存到临时文件
        temp_path = os.path.join(args.save_dir, 'temp_final_model.pth')
        torch.save(model.state_dict(), temp_path)
        
        final_path = os.path.join(args.save_dir, 'final_model.pth')
        # 如果存在旧文件，先删除
        if os.path.exists(final_path):
            os.remove(final_path)
            
        # 重命名临时文件
        os.rename(temp_path, final_path)
        print('Saved final model')
    except Exception as e:
        print(f'Warning: Failed to save final model: {str(e)}')
    
    # 绘制训练曲线
    # 修改函数调用以符合函数定义
    # 将指标历史转换为字典格式
    train_metrics_dict = {metrics_names[i]: np.array(train_metrics_history)[:, i] 
                         for i in range(len(metrics_names))}
    val_metrics_dict = {metrics_names[i]: np.array(val_metrics_history)[:, i] 
                       for i in range(len(metrics_names))}
    
    plot_training_curves(
        train_losses, val_losses,
        train_metrics_dict, val_metrics_dict,
        save_path=os.path.join(args.save_dir, 'training_curves.png')
    )
    
    # 保存训练信息
    save_training_info(
        args, train_losses, val_losses,
        train_metrics_history, val_metrics_history,
        metrics_names, total_time, 
        save_path=os.path.join(args.save_dir, 'training_info.json')
    )
    
    # 获取验证集上的预测结果
    _, val_metrics = validate(model, val_loader, criterion, device)
    
    # 计算混淆矩阵
    cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        to_numpy(all_targets), to_numpy(all_preds), 
        ['Real', 'Generated'], 
        save_path=cm_path
    )
    
    # 计算ROC曲线
    roc_path = os.path.join(args.save_dir, 'roc_curve.png')
    plot_roc_curve(
        to_numpy(all_targets), to_numpy(all_probs[:, 1]), 
        save_path=roc_path
    )
    
    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN/Diffusion Detection Model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory containing train and test folders')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'],
                       help='Backbone network architecture')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/validation split ratio')
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save checkpoints and results')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
