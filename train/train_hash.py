import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hash.hash_net import MultiFeatureHashNet, HashingLoss, get_binary_hash_codes, compute_map
from data.dataset import HashingDataset, get_dataloaders
from utils.transforms import get_hashing_transforms
from utils.visualization import plot_training_curves, save_training_info, plot_hash_embeddings


def train_one_epoch(model, dataloader, criterion, optimizer, device, writer=None, epoch=None):
    """
    训练一个轮次
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        writer: TensorBoard写入器（可选）
        epoch: 当前轮次（可选）
        
    Returns:
        avg_loss: 平均损失
        metrics: 训练指标
    """
    model.train()
    total_loss = 0
    loss_components = {'similarity': 0, 'quantization': 0, 'balance': 0}
    
    all_hash_codes = []
    all_embeddings = []
    all_labels = []
    fusion_weights = {}  # 记录融合权重信息
    all_paths = []
    
    for batch_idx, (images, labels, paths) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播 - 现在返回哈希码、嵌入和特征信息
        hash_codes, embeddings, feature_info = model(images)
        
        # 收集融合权重信息（用于分析）
        if 'late_fusion_weights' in feature_info:
            for k, v in feature_info['late_fusion_weights'].items():
                if k not in fusion_weights:
                    fusion_weights[k] = []
                fusion_weights[k].append(v)
        
        # 计算损失
        loss, losses_dict = criterion(hash_codes, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in losses_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v
        
        # 收集哈希码和标签
        all_hash_codes.append(hash_codes.detach().cpu())
        all_embeddings.append(embeddings.detach().cpu())
        all_labels.append(labels.cpu())
        all_paths.extend(paths)
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    
    # 计算指标
    all_hash_codes = torch.cat(all_hash_codes)
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    
    # 计算二进制哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    # 分析融合权重
    fusion_weights_avg = {}
    if fusion_weights:
        for feature_type, weights in fusion_weights.items():
            # 确保权重是标量值而不是张量
            weights_list = [w.mean().item() if torch.is_tensor(w) else w for w in weights]
            fusion_weights_avg[feature_type] = sum(weights_list) / len(weights_list)
            print(f"\u5e73均{feature_type}特征权重: {fusion_weights_avg[feature_type]:.4f}")
    
    # 计算mAP（使用部分数据加速计算）
    if len(binary_codes) > 1000:
        indices = np.random.choice(len(binary_codes), 1000, replace=False)
        query_codes = binary_codes[indices]
        query_labels = all_labels[indices]
        mAP = compute_map(query_codes, binary_codes, query_labels, all_labels, top_k=100)
    else:
        mAP = compute_map(binary_codes, binary_codes, all_labels, all_labels, top_k=100)
    
    # 汇总指标
    metrics = {
        'mAP': mAP,
        'similarity_loss': loss_components['similarity'],
        'quantization_loss': loss_components['quantization'],
        'balance_loss': loss_components['balance']
    }
    
    # 将融合权重添加到指标中
    for k, v in fusion_weights_avg.items():
        metrics[f'weight_{k}'] = v
    
    # 记录TensorBoard日志（如果提供）
    if writer is not None and epoch is not None:
        # 记录损失和指标
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/mAP', mAP, epoch)
        for k, v in loss_components.items():
            writer.add_scalar(f'Train/Loss_{k}', v, epoch)
        
        # 记录融合权重
        for feature_type, weight in fusion_weights_avg.items():
            writer.add_scalar(f'Train/Fusion_Weight_{feature_type}', weight, epoch)
        
        # 每10个轮次添加一次嵌入可视化
        if epoch % 10 == 0 and epoch > 0:
            try:
                writer.add_embedding(
                    all_embeddings[:500],
                    metadata=all_labels[:500].tolist(),
                    tag=f'embeddings_epoch_{epoch}',
                    global_step=epoch
                )
            except Exception as e:
                print(f"TensorBoard嵌入可视化错误: {e}")
    
    return avg_loss, metrics, (all_embeddings, all_labels, all_paths, fusion_weights_avg)


def validate(model, dataloader, criterion, device, writer=None, epoch=None):
    """
    在验证集上评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        writer: TensorBoard写入器（可选）
        epoch: 当前轮次（可选）
        
    Returns:
        avg_loss: 平均损失
        metrics: 验证指标
    """
    model.eval()
    total_loss = 0
    loss_components = {'similarity': 0, 'quantization': 0, 'balance': 0}
    
    all_hash_codes = []
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    # 用于收集融合权重信息
    fusion_weights = {}
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播 - 增强型多特征融合哈希网络现在返回三个值
            hash_codes, embeddings, feature_info = model(images)
            
            # 收集融合权重信息
            if 'late_fusion_weights' in feature_info:
                for k, v in feature_info['late_fusion_weights'].items():
                    if k not in fusion_weights:
                        fusion_weights[k] = []
                    fusion_weights[k].append(v)
            
            # 计算损失
            loss, losses_dict = criterion(hash_codes, labels)
            
            total_loss += loss.item()
            for k, v in losses_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v
            
            # 收集哈希码和标签
            all_hash_codes.append(hash_codes.cpu())
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_paths.extend(paths)
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    
    # 计算指标
    all_hash_codes = torch.cat(all_hash_codes)
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    
    # 计算二值哈希码
    binary_codes = get_binary_hash_codes(all_hash_codes)
    
    # 分析融合权重
    fusion_weights_avg = {}
    if fusion_weights:
        for feature_type, weights in fusion_weights.items():
            # 确保权重是标量值而不是张量
            weights_list = [w.mean().item() if torch.is_tensor(w) else w for w in weights]
            fusion_weights_avg[feature_type] = sum(weights_list) / len(weights_list)
            print(f"[验证] 平均{feature_type}特征权重: {fusion_weights_avg[feature_type]:.4f}")
    
    # 计算mAP和P@K
    mAP = compute_map(binary_codes, binary_codes, all_labels, all_labels)
    p_at_10 = compute_map(binary_codes, binary_codes, all_labels, all_labels, top_k=10)
    p_at_100 = compute_map(binary_codes, binary_codes, all_labels, all_labels, top_k=100)
    
    # 汇总指标
    metrics = {
        'mAP': mAP,
        'P@10': p_at_10,
        'P@100': p_at_100,
        'similarity_loss': loss_components['similarity'],
        'quantization_loss': loss_components['quantization'],
        'balance_loss': loss_components['balance']
    }
    
    # 将融合权重添加到指标中
    for k, v in fusion_weights_avg.items():
        metrics[f'weight_{k}'] = v
    
    # 记录TensorBoard日志（如果提供）
    if writer is not None and epoch is not None:
        # 记录损失和指标
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/mAP', mAP, epoch)
        writer.add_scalar('Val/P@10', p_at_10, epoch)
        writer.add_scalar('Val/P@100', p_at_100, epoch)
        for k, v in loss_components.items():
            writer.add_scalar(f'Val/Loss_{k}', v, epoch)
        
        # 记录融合权重
        for feature_type, weight in fusion_weights_avg.items():
            writer.add_scalar(f'Val/Fusion_Weight_{feature_type}', weight, epoch)
    
    return avg_loss, metrics, (all_embeddings, all_labels, all_paths, fusion_weights_avg)


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建TensorBoard日志目录
    log_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取数据变换
    train_transform, val_transform = get_hashing_transforms(img_size=args.img_size)
    
    # 判断数据集类型和结构
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # 如果数据集有train和test子目录，说明是标准格式
        print(f'Found standard dataset format with train and test directories')
        
        # 获取类别目录
        class_dirs_train = []
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                class_dirs_train.append(class_path)
        
        class_dirs_test = []
        for class_name in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_name)
            if os.path.isdir(class_path):
                class_dirs_test.append(class_path)
        
        print(f'Found {len(class_dirs_train)} class directories in train set')
        print(f'Found {len(class_dirs_test)} class directories in test set')
        
        # 使用所有类别目录作为数据目录
        data_dirs = class_dirs_train + class_dirs_test
    else:
        # 如果数据集没有train/test目录，则假设每个目录是一个类别
        data_dirs = []
        for subdir in os.listdir(args.data_dir):
            path = os.path.join(args.data_dir, subdir)
            if os.path.isdir(path):
                data_dirs.append(path)
        print(f'Found {len(data_dirs)} data directories')
    
    if not data_dirs:
        raise ValueError(f"No valid data directories found in {args.data_dir}")
    
    # 创建数据集和数据加载器
    train_loader, val_loader = get_dataloaders(
        dataset_type='hashing',
        data_dirs=data_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=train_transform,
        val_transform=val_transform,
        split_ratio=args.split_ratio
    )
    
    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Validation dataset size: {len(val_loader.dataset)}')
    
    # 创建模型
    model = MultiFeatureHashNet(
        hash_dim=args.hash_dim,
        backbone_name=args.backbone
    )
    
    # 先手动调整特征融合权重（如果模型支持此操作）
    try:
        # 尝试设置特征融合权重 - 增加CNN特征的权重
        if hasattr(model, 'late_fusion') and hasattr(model.late_fusion, 'feature_weights'):
            # 原始权重可能是自动计算或均匀分布的
            # 修改为更强调CNN特征的分布
            print("手动调整特征融合权重，增强CNN特征重要性")
            model.late_fusion.feature_weights = {
                'cnn_global': 0.5,       # 增加CNN特征权重（从约0.33提高到0.5）
                'frequency_global': 0.3,  # 略微提高频率特征
                'cooccurrence_global': 0.2  # 降低共现特征权重
            }
            print(f"调整后的特征权重: {model.late_fusion.feature_weights}")
    except Exception as e:
        print(f"无法手动调整特征融合权重: {e}")
        
    # 将模型移动到设备上（重要！）
    model = model.to(device)
    print(f"模型已移动到 {device} 设备上")
    
    # 创建损失函数
    criterion = HashingLoss(
        alpha=args.sim_weight,
        beta=args.quant_weight,
        gamma=args.balance_weight,
        hash_dim=args.hash_dim
    )
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay * 1.5  # 增加权重衰减以减轻过拟合
    )
    
    print(f"使用增强的权重衰减: {args.weight_decay * 1.5}")
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 最大化mAP
        factor=0.5,
        patience=5
    )
    # 学习率变化时手动打印日志
    print(f"Initial learning rate: {args.lr}")
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_metrics_history = {'mAP': [], 'similarity_loss': [], 'quantization_loss': [], 'balance_loss': []}
    val_metrics_history = {'mAP': [], 'P@10': [], 'P@100': []}
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_map = 0
    
    # 初始化早停计数器 - 增加耐心参数以允许更长时间的训练
    patience = 15  # 增加早停耐心参数，适应更长的训练过程
    patience_counter = 0
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=log_dir)
    
    # 记录模型结构
    try:
        sample_images, _, _ = next(iter(train_loader))
        sample_images = sample_images[:1].to(device)
        writer.add_graph(model, sample_images)
    except Exception as e:
        print(f"Failed to add model graph to TensorBoard: {e}")
    
    print('Starting training...')
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练一个轮次
        train_loss, train_metrics, train_data = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        
        # 验证
        val_loss, val_metrics, val_data = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        
        # 更新学习率
        scheduler.step(val_metrics['mAP'])
        
        # 记录损失和指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        for k, v in train_metrics.items():
            if k in train_metrics_history:
                train_metrics_history[k].append(v)
        
        for k, v in val_metrics.items():
            if k in val_metrics_history:
                val_metrics_history[k].append(v)
        
        # 检查是否是最佳模型
        is_best_loss = val_loss < best_val_loss
        is_best_map = val_metrics['mAP'] > best_val_map
        
        if is_best_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 重置早停计数器
        else:
            patience_counter += 1  # 增加早停计数器
        
        if is_best_map:
            best_val_map = val_metrics['mAP']
        
        # 准备保存的检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
            'best_val_map': best_val_map,
            'fusion_type': args.fusion_type,
            'hash_dim': args.hash_dim,
            'backbone': args.backbone
        }
        
        # 定义模型保存路径
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
        best_loss_path = os.path.join(args.save_dir, 'best_loss_model.pth')
        best_map_path = os.path.join(args.save_dir, 'best_map_model.pth')
        
        # 实现健壮的模型保存机制
        def safe_save_model(model_dict, save_path):
            try:
                # 首先保存到临时文件
                temp_path = save_path + '.tmp'
                torch.save(model_dict, temp_path)
                
                # 成功保存到临时文件后，重命名为最终文件
                if os.path.exists(save_path):
                    os.remove(save_path)  # 删除旧文件
                os.rename(temp_path, save_path)
                return True
            except Exception as e:
                print(f"Error saving model to {save_path}: {e}")
                return False
        
        # 保存最新检查点
        safe_save_model(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best_loss:
            if safe_save_model(checkpoint, best_loss_path):
                print(f"Saved best loss model. Val Loss: {best_val_loss:.4f}")
        
        if is_best_map:
            if safe_save_model(checkpoint, best_map_path):
                print(f"Saved best mAP model. Val mAP: {best_val_map:.4f}")
        
        # 检查是否触发早停
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            break
        
        # 打印当前结果
        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{args.epochs}, Time: {elapsed:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Metrics: mAP={train_metrics["mAP"]:.4f}')
        print(f'Val Metrics: mAP={val_metrics["mAP"]:.4f}, '
              f'P@10={val_metrics["P@10"]:.4f}, P@100={val_metrics["P@100"]:.4f}')
        
        # 写入TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for metric in train_metrics:
            if isinstance(train_metrics[metric], (int, float)):
                writer.add_scalar(f'Metrics/train_{metric}', train_metrics[metric], epoch)
        
        for metric in val_metrics:
            if isinstance(val_metrics[metric], (int, float)):
                writer.add_scalar(f'Metrics/val_{metric}', val_metrics[metric], epoch)
        
        # 打印训练进度总结
        print(f"Epoch {epoch+1}/{args.epochs} complete. " + 
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " + 
              f"Train mAP: {train_metrics['mAP']:.4f}, Val mAP: {val_metrics['mAP']:.4f}\n")
        
        # 每隔几个轮次保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_training_info(
                epoch, model, optimizer, train_loss, val_loss,
                train_metrics, val_metrics, save_path
            )
            print(f'Saved checkpoint to {save_path}')
            
            # 可视化嵌入
            # 安全地解包val_data，防止值的数量不匹配
            if isinstance(val_data, tuple) and len(val_data) >= 2:
                val_embeddings = val_data[0]  # 第一个元素是嵌入
                val_labels = val_data[1]      # 第二个元素是标签
                print(f"获取到验证集嵌入 shape={val_embeddings.shape} 和标签 shape={val_labels.shape}")
            else:
                print(f"警告：val_data格式不符合预期 (类型: {type(val_data)})，跳过嵌入可视化")
                continue
                
            if epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
                embed_path = os.path.join(args.save_dir, f'embeddings_epoch_{epoch+1}.png')
                if len(val_embeddings) > 1000:  # 限制点数，避免图表过于拥挤
                    indices = np.random.choice(len(val_embeddings), 1000, replace=False)
                    plot_hash_embeddings(
                        val_embeddings[indices], val_labels[indices], 
                        method='tsne', save_path=embed_path
                    )
                else:
                    plot_hash_embeddings(
                        val_embeddings, val_labels, 
                        method='tsne', save_path=embed_path
                    )
                print(f'Saved embeddings visualization to {embed_path}')
        
        print('-' * 60)
    
    # 训练结束，保存最终模型
    final_save_path = os.path.join(args.save_dir, 'final_model.pth')
    save_training_info(
        args.epochs-1, model, optimizer, train_loss, val_loss,
        train_metrics, val_metrics, final_save_path
    )
    print(f'Saved final model to {final_save_path}')
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 绘制训练曲线
    plot_path = os.path.join(args.save_dir, 'training_curves.png')
    plot_training_curves(
        train_losses, val_losses,
        train_metrics_history, val_metrics_history,
        save_path=plot_path
    )
    print(f'Saved training curves to {plot_path}')
    
    # 分析最终的融合权重
    fusion_weights_avg = val_data[3] if len(val_data) > 3 else {}
    if fusion_weights_avg:
        print("\n=== 特征融合权重分析 ===")
        for feature_type, weight in fusion_weights_avg.items():
            print(f"{feature_type} 特征权重: {weight:.4f}")
            
        # 将融合权重保存到文件
        weights_path = os.path.join(args.save_dir, 'fusion_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(fusion_weights_avg, f, indent=4)
    
    # 可视化哈希码嵌入
    val_embeddings, val_labels, val_paths = val_data[:3]  # 正确提取前三个元素
    final_embed_path = os.path.join(args.save_dir, 'final_embeddings.png')
    if len(val_embeddings) > 1000:
        indices = np.random.choice(len(val_embeddings), 1000, replace=False)
        plot_hash_embeddings(
            val_embeddings[indices], val_labels[indices], 
            method='tsne', save_path=final_embed_path
        )
    else:
        plot_hash_embeddings(
            val_embeddings, val_labels,
            method='tsne', save_path=final_embed_path
        )
    print(f'Saved embedding visualization to {final_embed_path}')
    
    # 打印最终性能报告
    print(f"\n=== 最终性能报告 ===")
    print(f"\n1. 训练结果:")
    print(f"  - 最终训练损失: {train_losses[-1]:.4f}")
    print(f"  - 最终验证损失: {val_losses[-1]:.4f}")
    print(f"  - 最终训练mAP: {train_metrics_history['mAP'][-1]:.4f}")
    print(f"  - 最终验证mAP: {val_metrics_history['mAP'][-1]:.4f}")
    
    print(f"\n2. 最佳模型性能:")
    print(f"  - 最佳验证损失: {best_val_loss:.4f}")
    print(f"  - 最佳验证mAP: {best_val_map:.4f}")
    print(f"  - 最佳P@10: {max(val_metrics_history['P@10']):.4f}")
    print(f"  - 最佳P@100: {max(val_metrics_history['P@100']):.4f}")
    
    print(f"\n3. 训练配置:")
    print(f"  - 模型类型: 多特征融合哈希网络")
    print(f"  - 融合类型: {args.fusion_type}")
    print(f"  - 哈希码维度: {args.hash_dim}")
    print(f"  - 骨干网络: {args.backbone}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 权重衰减: {args.weight_decay}")
    print(f"  - 早停耐心参数: {patience}")
    
    print("\n训练完成! 可使用TensorBoard查看详细可视化结果:\ntensorboard --logdir=" + log_dir)
    
    return best_val_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Feature Fusion Image Hashing Model')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save models and results')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'], 
                       help='Backbone network')
    parser.add_argument('--hash_dim', type=int, default=64, help='Hash code dimension')
    parser.add_argument('--fusion_type', type=str, default='late', 
                       choices=['early', 'late', 'hybrid'], 
                       help='Feature fusion type: early (at feature extraction), late (after pooling), or hybrid')
    
    # 多特征融合参数
    parser.add_argument('--use_frequency', type=bool, default=True, 
                       help='Use frequency domain features (DCT transform)')
    parser.add_argument('--use_cooccurrence', type=bool, default=True, 
                       help='Use co-occurrence matrix texture features')
    parser.add_argument('--cooc_angles', type=int, default=4, 
                       help='Number of angles for co-occurrence matrix (1-4)')
    parser.add_argument('--use_attention', type=bool, default=True, 
                       help='Use attention mechanism for feature fusion')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate for regularization')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=8, 
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--sim_weight', type=float, default=1.0, help='Similarity loss weight')
    parser.add_argument('--quant_weight', type=float, default=1.0, help='Quantization loss weight')
    parser.add_argument('--balance_weight', type=float, default=0.1, help='Balance loss weight')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/val split ratio')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no_early_stop', action='store_true', 
                       help='Disable early stopping')
    
    args = parser.parse_args()
    
    main(args)
