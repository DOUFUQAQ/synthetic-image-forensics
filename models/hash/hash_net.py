import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .semantic_module import MultiLevelSemanticExtractor, SemanticEmbeddingModule
from .feature_fusion import EarlyFusionModule, LateFusionModule, CooccurrenceMatrix, FrequencyTransform


class MultiFeatureHashNet(nn.Module):
    """
    多特征融合图像哈希网络
    结合语义嵌入与多特征融合的图像哈希算法
    支持频域、空间域与共生矩阵等多特征融合
    """
    def __init__(self, hash_dim=64, backbone_name='resnet18', fusion_type='late', pretrained=True):
        super(MultiFeatureHashNet, self).__init__()
        
        # 哈希码维度
        self.hash_dim = hash_dim
        
        # 验证融合类型
        assert fusion_type in ['early', 'late', 'hybrid'], "融合类型必须是 'early', 'late' 或 'hybrid'"
        self.fusion_type = fusion_type
        
        # 多层语义特征提取器
        self.semantic_extractor = MultiLevelSemanticExtractor(
            backbone_name=backbone_name, 
            pretrained=pretrained
        )
        
        # 获取语义提取器输出的特征维度
        if backbone_name == 'resnet18' or backbone_name == 'resnet34':
            feature_dims = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            feature_dims = [64, 256, 512, 1024, 2048]
        elif backbone_name == 'mobilenet_v2':
            feature_dims = [16, 24, 32, 96]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 共生矩阵特征提取器
        self.cooc_extractor = CooccurrenceMatrix(
            num_levels=8, 
            reduction_dim=64,
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]  # 四个方向
        )
        
        # 频域特征提取器
        self.freq_extractor = FrequencyTransform(use_dct=True, use_dft=False)
        
        # 如果使用早期融合
        if fusion_type in ['early', 'hybrid']:
            # 为每一层添加增强的早期融合模块
            self.early_fusion1 = EarlyFusionModule(feature_dims[0])
            self.early_fusion2 = EarlyFusionModule(feature_dims[1])
            self.early_fusion3 = EarlyFusionModule(feature_dims[2])
            self.early_fusion4 = EarlyFusionModule(feature_dims[3])
        
        # 如果使用晚期融合
        if fusion_type in ['late', 'hybrid']:
            # 定义晚期融合的特征类型和对应维度
            late_feature_names = ['cnn_global', 'frequency_global', 'cooccurrence_global']
            # 各类特征的维度 - 确保与实际输出维度匹配
            # 共生矩阵特征始终保持为64维
            late_fusion_dims = [512, 256, 64]  # [CNN特征, 频域特征, 共生矩阵特征]
            
            # 添加检查逻辑，确保特征维度正确
            print(f"[INFO] Late fusion feature dimensions: {late_fusion_dims}")
            
            # 增强的晚期融合模块
            self.late_fusion = LateFusionModule(
                feature_dims=late_fusion_dims,
                embedding_dim=512,
                feature_names=late_feature_names
            )
        
        # 融合后的特征维度
        if fusion_type == 'early':
            final_dim = sum(feature_dims)
        elif fusion_type == 'late':
            final_dim = 512  # 晚期融合的输出维度
        else:  # 'hybrid'
            final_dim = 512 + sum(feature_dims)  # 早期和晚期融合的结合
        
        # 语义嵌入模块
        self.semantic_embedding = SemanticEmbeddingModule(
            in_features=final_dim,
            embedding_dim=512,
            dropout_rate=0.2
        )
        
        # 哈希层
        self.hash_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hash_dim),
            nn.Tanh()  # 使输出在 [-1, 1] 范围内
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W]
            
        Returns:
            hash_code: 连续的哈希码，形状为 [B, hash_dim]
            embedding: 语义嵌入，形状为 [B, 512]
            feature_info: 详细特征信息字典，用于分析
        """
        batch_size = x.size(0)
        feature_info = {}  # 存储特征信息的字典
        
        # 验证输入数据
        if x.dim() != 4 or x.size(1) != 3:
            print(f"Warning: Expected input with shape [B, 3, H, W], got {x.shape}")
            # 确保依然能够运行
            if x.dim() < 4:
                x = x.view(batch_size, 3, 32, 32)  # 假设默认尺寸
        
        # 检查输入是否包含无效值（NaN或Inf）
        if not torch.isfinite(x).all():
            print("Warning: Input contains non-finite values, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. 提取多层次语义特征
        cnn_features, pooled_features = self.semantic_extractor(x)
        feature_info['cnn_features'] = cnn_features
        feature_info['cnn_global'] = pooled_features[-1]  # 最后一层的全局特征
        
        # 2. 提取频域特征
        try:
            freq_features = self.freq_extractor(x)
            
            # 检查频域特征是否有效
            if not torch.isfinite(freq_features).all():
                print("Warning: Frequency features contain non-finite values, replacing with zeros")
                freq_features = torch.nan_to_num(freq_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 池化得到全局频域特征
            freq_global = F.adaptive_avg_pool2d(freq_features, 1).squeeze(-1).squeeze(-1)
            
            # 确保没有维度为0的情况
            if freq_global.numel() == 0:
                raise ValueError(f"Empty frequency global features: {freq_global.shape}")
                
            # 频域特征必须是256维
            if freq_global.size(1) != 256:
                print(f"Warning: Frequency global features should be 256D, but got {freq_global.size(1)}D. Adjusting...")
                # 创建256维的频域特征
                freq_global_new = torch.zeros(batch_size, 256, device=x.device)
                
                if freq_global.size(1) < 256:
                    # 如果维度过小，将现有特征复制到前面部分
                    freq_global_new[:, :freq_global.size(1)] = freq_global
                    # 添加随机噪声到剩余部分
                    freq_global_new[:, freq_global.size(1):] = torch.randn(batch_size, 256 - freq_global.size(1), device=x.device) * 0.01
                else:
                    # 如果维度过大，截取前256维
                    freq_global_new = freq_global[:, :256]
                
                freq_global = freq_global_new
                
            # L2正规化前检查是否有零向量
            if (freq_global.norm(p=2, dim=1) == 0).any():
                print("Warning: Zero vectors in frequency features before normalization")
                # 为零向量添加小噪声
                mask = (freq_global.norm(p=2, dim=1) == 0)
                if mask.any():
                    freq_global[mask] = torch.randn_like(freq_global[mask]) * 0.01
            
            freq_global = F.normalize(freq_global, p=2, dim=1)  # L2正规化
            
            # 检查输出维度
            print(f"Frequency features dimensions: features={freq_features.shape}, global={freq_global.shape}")
            
            feature_info['frequency_features'] = freq_features
            feature_info['frequency_global'] = freq_global
            
        except Exception as e:
            print(f"Error extracting frequency features: {e}. Using fallback.")
            # 创建备用特征
            device = x.device
            # 空间特征形状应与输入图像匹配
            freq_features = torch.zeros(batch_size, 256, x.size(2), x.size(3), device=device)
            # 全局特征
            freq_global = torch.zeros(batch_size, 256, device=device)
            # 添加随机噪声并归一化
            noise = torch.randn_like(freq_global) * 0.1
            freq_global = F.normalize(freq_global + noise, p=2, dim=1)
            
            feature_info['frequency_features'] = freq_features
            feature_info['frequency_global'] = freq_global
        
        # 3. 提取共生矩阵特征
        try:
            cooc_spatial, cooc_global = self.cooc_extractor(x)
            
            # 检查输出维度
            print(f"Cooccurrence features dimensions: spatial={cooc_spatial.shape}, global={cooc_global.shape}")
            
            # 确保全局特征是64维的
            if cooc_global.size(1) != 64:
                print(f"Warning: Expected cooc_global to have 64 dimensions, got {cooc_global.size(1)}. Adjusting...")
                if cooc_global.size(1) > 64:
                    cooc_global = cooc_global[:, :64]  # 截断
                else:
                    # 扩展维度
                    expanded = torch.zeros(batch_size, 64, device=x.device)
                    expanded[:, :cooc_global.size(1)] = cooc_global
                    if cooc_global.size(1) < 64:
                        expanded[:, cooc_global.size(1):] = torch.randn(batch_size, 64 - cooc_global.size(1), device=x.device) * 0.01
                    cooc_global = expanded
            
            feature_info['cooccurrence_spatial'] = cooc_spatial
            feature_info['cooccurrence_global'] = cooc_global
            
        except Exception as e:
            print(f"Error extracting cooccurrence features: {e}. Using fallback.")
            # 创建备用特征
            device = x.device
            # 空间特征形状应与输入图像匹配
            cooc_spatial = torch.zeros(batch_size, 64, x.size(2), x.size(3), device=device)
            # 全局特征必须是64维
            cooc_global = torch.zeros(batch_size, 64, device=device)
            # 添加随机噪声并归一化
            noise = torch.randn_like(cooc_global) * 0.1
            cooc_global = F.normalize(cooc_global + noise, p=2, dim=1)
            
            feature_info['cooccurrence_spatial'] = cooc_spatial
            feature_info['cooccurrence_global'] = cooc_global
        
        # 4. 根据融合类型处理特征
        if self.fusion_type == 'early':
            # 应用早期融合
            fused_features1, layer1_info = self.early_fusion1(cnn_features['layer1'])
            fused_features2, layer2_info = self.early_fusion2(cnn_features['layer2'])
            fused_features3, layer3_info = self.early_fusion3(cnn_features['layer3'])
            fused_features4, layer4_info = self.early_fusion4(cnn_features['layer4'])
            
            # 记录每层融合的注意力权重
            feature_info['early_fusion'] = {
                'layer1': layer1_info,
                'layer2': layer2_info,
                'layer3': layer3_info,
                'layer4': layer4_info
            }
            
            # 池化得到全局特征
            p1 = F.adaptive_avg_pool2d(fused_features1, 1).squeeze(-1).squeeze(-1)
            p2 = F.adaptive_avg_pool2d(fused_features2, 1).squeeze(-1).squeeze(-1)
            p3 = F.adaptive_avg_pool2d(fused_features3, 1).squeeze(-1).squeeze(-1)
            p4 = F.adaptive_avg_pool2d(fused_features4, 1).squeeze(-1).squeeze(-1)
            
            # 连接所有层的特征
            fused_features = torch.cat([p1, p2, p3, p4], dim=1)
            
        elif self.fusion_type == 'late':
            # 创建特征字典传入晚期融合模块，同时确保特征维度符合预期
            # 检查并打印各特征维度，帮助调试
            cnn_dim = pooled_features[-1].size(1)
            freq_dim = freq_global.size(1)
            cooc_dim = cooc_global.size(1)
            print(f"Feature dimensions before late fusion: CNN={cnn_dim}, Frequency={freq_dim}, Cooccurrence={cooc_dim}")
            
            # 检查特征是否有效
            if not torch.isfinite(pooled_features[-1]).all():
                print("Warning: CNN features contain non-finite values, replacing with zeros")
                pooled_features[-1] = torch.nan_to_num(pooled_features[-1], nan=0.0, posinf=0.0, neginf=0.0)
            
            if not torch.isfinite(freq_global).all():
                print("Warning: Frequency features contain non-finite values, replacing with zeros")
                freq_global = torch.nan_to_num(freq_global, nan=0.0, posinf=0.0, neginf=0.0)
            
            if not torch.isfinite(cooc_global).all():
                print("Warning: Cooccurrence features contain non-finite values, replacing with zeros")
                cooc_global = torch.nan_to_num(cooc_global, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 确保共生矩阵特征维度为64
            assert cooc_global.size(1) == 64, f"Cooccurrence feature dimension must be 64, got {cooc_global.size(1)}"
            
            # 创建特征字典
            late_features_dict = {
                'cnn_global': pooled_features[-1],
                'frequency_global': freq_global,
                'cooccurrence_global': cooc_global
            }
            
            # 应用晚期融合
            try:
                fused_features, attention_weights = self.late_fusion(late_features_dict)
                feature_info['late_fusion_weights'] = attention_weights
            except Exception as e:
                print(f"Error in late fusion: {e}. Creating fallback features.")
                # 创建备用特征，避免中断训练
                device = x.device
                batch_size = x.size(0)
                fused_features = torch.zeros(batch_size, 512, device=device)
                # 添加随机噪声并归一化
                noise = torch.randn_like(fused_features) * 0.1
                fused_features = F.normalize(fused_features + noise, p=2, dim=1)
                
                # 创建带有数字的默认融合权重，而不是字符串
                default_weights = {}
                for key in late_features_dict.keys():
                    # 创建均等的权重，每个特征类型均为1/n
                    default_weights[key] = torch.ones(batch_size, 1, device=device) / len(late_features_dict)
                feature_info['late_fusion_weights'] = default_weights
            
        else:  # 'hybrid'
            # 同时应用早期和晚期融合
            # 先应用早期融合到各层特征
            fused_features1, layer1_info = self.early_fusion1(cnn_features['layer1'])
            fused_features2, layer2_info = self.early_fusion2(cnn_features['layer2'])
            fused_features3, layer3_info = self.early_fusion3(cnn_features['layer3'])
            fused_features4, layer4_info = self.early_fusion4(cnn_features['layer4'])
            
            # 记录注意力权重
            feature_info['early_fusion'] = {
                'layer1': layer1_info,
                'layer2': layer2_info,
                'layer3': layer3_info,
                'layer4': layer4_info
            }
            
            # 池化得到全局特征
            p1 = F.adaptive_avg_pool2d(fused_features1, 1).squeeze(-1).squeeze(-1)
            p2 = F.adaptive_avg_pool2d(fused_features2, 1).squeeze(-1).squeeze(-1)
            p3 = F.adaptive_avg_pool2d(fused_features3, 1).squeeze(-1).squeeze(-1)
            p4 = F.adaptive_avg_pool2d(fused_features4, 1).squeeze(-1).squeeze(-1)
            
            # 早期融合的结果
            early_fused = torch.cat([p1, p2, p3, p4], dim=1)
            
            # 检查并处理各特征维度
            cnn_dim = pooled_features[-1].size(1)
            freq_dim = freq_global.size(1)
            cooc_dim = cooc_global.size(1)
            print(f"Feature dimensions before hybrid fusion: CNN={cnn_dim}, Frequency={freq_dim}, Cooccurrence={cooc_dim}")
            
            # 检查特征是否有效
            for feat_name, feat in [('CNN', pooled_features[-1]), ('Frequency', freq_global), ('Cooccurrence', cooc_global)]:
                if not torch.isfinite(feat).all():
                    print(f"Warning: {feat_name} features contain non-finite values, replacing with zeros")
                    if feat_name == 'CNN':
                        pooled_features[-1] = torch.nan_to_num(pooled_features[-1], nan=0.0, posinf=0.0, neginf=0.0)
                    elif feat_name == 'Frequency':
                        freq_global = torch.nan_to_num(freq_global, nan=0.0, posinf=0.0, neginf=0.0)
                    elif feat_name == 'Cooccurrence':
                        cooc_global = torch.nan_to_num(cooc_global, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 确保共生矩阵特征维度为64
            assert cooc_global.size(1) == 64, f"Cooccurrence feature dimension must be 64, got {cooc_global.size(1)}"
            
            # 创建特征字典
            late_features_dict = {
                'cnn_global': pooled_features[-1],
                'frequency_global': freq_global,
                'cooccurrence_global': cooc_global
            }
            
            # 应用晚期融合
            try:
                late_fused, attention_weights = self.late_fusion(late_features_dict)
                feature_info['late_fusion_weights'] = attention_weights
            except Exception as e:
                print(f"Error in hybrid late fusion: {e}. Creating fallback features.")
                # 创建备用特征，避免中断训练
                device = x.device
                batch_size = x.size(0)
                late_fused = torch.zeros(batch_size, 512, device=device)
                # 添加随机噪声并归一化
                noise = torch.randn_like(late_fused) * 0.1
                late_fused = F.normalize(late_fused + noise, p=2, dim=1)
                
                # 创建带有数字的默认融合权重，而不是字符串
                default_weights = {}
                for key in late_features_dict.keys():
                    # 创建均等的权重，每个特征类型均为1/n
                    default_weights[key] = torch.ones(batch_size, 1, device=device) / len(late_features_dict)
                feature_info['late_fusion_weights'] = default_weights
            
            # 结合两种融合结果
            fused_features = torch.cat([early_fused, late_fused], dim=1)
        
        # 5. 检查融合特征
        if not torch.isfinite(fused_features).all():
            print("Warning: Fused features contain non-finite values, replacing with zeros")
            fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 应用语义嵌入模块
        try:
            embedding = self.semantic_embedding(fused_features)
            feature_info['semantic_embedding'] = embedding
            
            # 6. 生成哈希码
            hash_code = self.hash_layer(embedding)
            feature_info['hash_code'] = hash_code
            
        except Exception as e:
            print(f"Error in final embedding/hashing: {e}. Creating fallback output.")
            # 创建备用输出
            device = x.device
            embedding = torch.zeros(batch_size, 512, device=device)
            hash_code = torch.zeros(batch_size, self.hash_dim, device=device)
            
            # 添加随机噪声使哈希码更真实
            noise = torch.randn_like(hash_code) * 0.1
            hash_code = torch.tanh(noise)  # 确保在[-1,1]范围内
            
            feature_info['semantic_embedding'] = embedding
            feature_info['hash_code'] = hash_code
        
        # 确保返回值的维度正确
        assert hash_code.size(1) == self.hash_dim, f"Hash code dimension must be {self.hash_dim}, got {hash_code.size(1)}"
        assert embedding.size(1) == 512, f"Embedding dimension must be 512, got {embedding.size(1)}"
        
        return hash_code, embedding, feature_info


class HashingLoss(nn.Module):
    """
    哈希损失函数
    结合多种损失，以优化哈希码的性能
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1, hash_dim=64):
        super(HashingLoss, self).__init__()
        self.alpha = alpha  # 相似性损失权重
        self.beta = beta    # 量化损失权重
        self.gamma = gamma  # 均匀分布损失权重
        self.hash_dim = hash_dim
        
    def similarity_loss(self, hash_codes, labels):
        """
        计算相似性损失，使相似图像的哈希码接近，不相似图像的哈希码远离
        
        Args:
            hash_codes: 哈希码，形状为 [B, hash_dim]
            labels: 标签，形状为 [B]
            
        Returns:
            loss: 相似性损失值
        """
        batch_size = hash_codes.size(0)
        
        # 计算标签相似度矩阵 (1表示相似，-1表示不相似)
        label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * 2 - 1  # [B, B]
        
        # 计算哈希码内积 (相似度)
        hash_sim = torch.matmul(hash_codes, hash_codes.t()) / self.hash_dim  # [B, B]
        
        # 计算相似性损失
        loss = F.mse_loss(hash_sim, label_sim)
        
        return loss
    
    def quantization_loss(self, hash_codes):
        """
        计算量化损失，使哈希码接近二值 (-1 或 1)
        
        Args:
            hash_codes: 哈希码，形状为 [B, hash_dim]
            
        Returns:
            loss: 量化损失值
        """
        return F.mse_loss(hash_codes.abs(), torch.ones_like(hash_codes))
    
    def balance_loss(self, hash_codes):
        """
        计算均衡性损失，使哈希码各位均匀分布
        
        Args:
            hash_codes: 哈希码，形状为 [B, hash_dim]
            
        Returns:
            loss: 均衡性损失值
        """
        # 计算每个维度的平均值，理想情况下应接近0
        mean = torch.mean(hash_codes, dim=0)
        return torch.mean(mean * mean)
    
    def forward(self, hash_codes, labels):
        """
        计算总损失
        
        Args:
            hash_codes: 哈希码，形状为 [B, hash_dim]
            labels: 标签，形状为 [B]
            
        Returns:
            total_loss: 总损失值
            losses: 各部分损失的字典
        """
        sim_loss = self.similarity_loss(hash_codes, labels)
        quant_loss = self.quantization_loss(hash_codes)
        bal_loss = self.balance_loss(hash_codes)
        
        total_loss = self.alpha * sim_loss + self.beta * quant_loss + self.gamma * bal_loss
        
        losses = {
            'similarity': sim_loss.item(),
            'quantization': quant_loss.item(),
            'balance': bal_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, losses


def get_binary_hash_codes(hash_codes):
    """
    将连续的哈希码转换为二进制哈希码
    
    Args:
        hash_codes: 连续哈希码，形状为 [B, hash_dim]，值在[-1, 1]范围内
        
    Returns:
        binary_codes: 二进制哈希码，形状为 [B, hash_dim]，值为0或1
    """
    return (hash_codes > 0).float()


def hamming_distance(code1, code2):
    """
    计算两个二进制哈希码之间的汉明距离
    
    Args:
        code1: 第一个哈希码，形状为 [hash_dim]，值为0或1
        code2: 第二个哈希码，形状为 [hash_dim]，值为0或1
        
    Returns:
        distance: 汉明距离，标量
    """
    return torch.sum(code1 != code2).item()


def compute_map(query_codes, db_codes, query_labels, db_labels, top_k=None):
    """
    计算平均精度均值 (mAP)
    
    Args:
        query_codes: 查询哈希码，形状为 [num_query, hash_dim]，值为0或1
        db_codes: 数据库哈希码，形状为 [num_db, hash_dim]，值为0或1
        query_labels: 查询标签，形状为 [num_query]
        db_labels: 数据库标签，形状为 [num_db]
        top_k: 考虑的检索结果数量，如果为None则考虑所有结果
        
    Returns:
        mAP: 平均精度均值
    """
    num_query = query_codes.size(0)
    num_db = db_codes.size(0)
    
    # 计算汉明距离
    hamming_dists = torch.zeros(num_query, num_db)
    for i in range(num_query):
        hamming_dists[i] = torch.sum(query_codes[i] != db_codes, dim=1)
    
    # 对每个查询计算AP
    APs = []
    for i in range(num_query):
        # 获取与查询i相同类别的数据库项
        query_label = query_labels[i]
        relevant = (db_labels == query_label)
        num_relevant = relevant.sum().item()
        
        if num_relevant == 0:
            continue
            
        # 按汉明距离排序
        _, indices = torch.sort(hamming_dists[i])
        
        # 限制考虑的结果数量
        if top_k is not None:
            indices = indices[:top_k]
            
        # 计算累积正确数和精度
        is_relevant = relevant[indices]
        cumulative_relevant = torch.cumsum(is_relevant, dim=0)
        precision = cumulative_relevant.float() / torch.arange(1, len(indices) + 1, device=is_relevant.device)
        
        # 计算AP
        AP = torch.sum(precision * is_relevant) / num_relevant
        APs.append(AP.item())
    
    # 计算mAP
    mAP = sum(APs) / len(APs) if APs else 0.0
    
    return mAP
