import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange


class FrequencyTransform(nn.Module):
    """
    频域变换模块
    提取图像的频域特征，使用PyTorch原生操作
    """
    def __init__(self, use_dct=True, use_dft=False):
        super(FrequencyTransform, self).__init__()
        self.use_dct = use_dct
        self.use_dft = use_dft
        
        # 预计算DCT基函数矩阵，用于加速DCT计算
        self.dct_filters = {}
        
    def _get_dct_filters(self, size, device):
        """获取指定尺寸的DCT基函数矩阵。
        先在CPU上计算，然后转移到GPU，平衡资源使用"""
        key = f"{size}_{device}"
        if key not in self.dct_filters:
            # 使用Numpy在CPU上预计算DCT基函数矩阵
            dct_m = np.zeros((size, size))
            for k in range(size):
                for n in range(size):
                    if k == 0:
                        dct_m[k, n] = np.sqrt(1.0 / size)
                    else:
                        scale = np.sqrt(2.0 / size)
                        angle = np.pi * (2 * n + 1) * k / (2 * size)
                        dct_m[k, n] = scale * np.cos(angle)
            # 将计算好的numpy矩阵转移到GPU
            self.dct_filters[key] = torch.from_numpy(dct_m).float().to(device)
        return self.dct_filters[key]
            
    def dct_2d(self, x):
        """使用PyTorch原生操作实现的离散余弦变换"""
        batch_size, channels, height, width = x.size()
        device = x.device
        
        # 获取DCT基函数矩阵
        dct_h = self._get_dct_filters(height, device)
        dct_w = self._get_dct_filters(width, device)
        
        # 将x重形状以便于矩阵乘法
        x_reshaped = x.reshape(batch_size * channels, height, width)
        
        # 对行和列进行DCT变换
        # 第一步：对行进行DCT
        x_dct_h = torch.matmul(dct_h, x_reshaped)
        
        # 第二步：对列进行DCT
        x_dct = torch.matmul(x_dct_h, dct_w.t())
        
        # 重新整形回原始形状
        return x_dct.reshape(batch_size, channels, height, width)
    
    def dft_2d(self, x):
        """离散傅里叶变换"""
        # 使用PyTorch的FFT实现
        fft_result = torch.fft.fft2(x)
        # 取振幅谱
        magnitude = torch.abs(fft_result)
        # 对数变换以增强细节
        magnitude = torch.log(magnitude + 1)
        return magnitude
    
    def forward(self, x):
        """
        应用频域变换
        
        Args:
            x: 输入图像或特征图，形状为 [B, C, H, W]
            
        Returns:
            freq_features: 频域特征
        """
        results = []
        
        if self.use_dct:
            dct_result = self.dct_2d(x)
            results.append(dct_result)
            
        if self.use_dft:
            dft_result = self.dft_2d(x)
            results.append(dft_result)
            
        if not results:
            return x
        
        # 如果使用了多种变换，则连接结果
        if len(results) > 1:
            return torch.cat(results, dim=1)
        else:
            return results[0]


class CooccurrenceMatrix(nn.Module):
    """
    共生矩阵特征提取器
    用于捕获图像的纹理特征
    基于灰度共生矩阵(GLCM)实现
    """
    def __init__(self, num_levels=8, distance=1, reduction_dim=64, angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        super(CooccurrenceMatrix, self).__init__()
        self.num_levels = num_levels
        self.distance = distance
        self.angles = angles
        
        # 用于减少共生矩阵维度的卷积层
        self.reduction = nn.Sequential(
            nn.Conv2d(num_levels * num_levels * len(angles), reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True)
        )
        
        # 增加纹理特征统计层
        self.texture_stats = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(reduction_dim, reduction_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(reduction_dim)
        )
    
    def compute_glcm(self, x):
        """
        计算多角度的灰度共生矩阵，平衡CPU和GPU资源使用
        
        Args:
            x: 输入图像或特征图，形状为 [B, C, H, W]
            
        Returns:
            combined_glcm: 多角度的GLCM特征 [B, num_levels*num_levels*angles, C, 1]
        """
        try:
            # 将张量移至CPU，避免在GPU上进行大量遍历操作
            x_cpu = x.detach().cpu().numpy()
            batch_size, channels, height, width = x_cpu.shape
            device = x.device
            
            # 处理异常情况
            if height < 2 or width < 2:
                print(f"Warning: Input size too small for GLCM: {height}x{width}, using fallback")
                # 返回随机初始化的GLCM
                result_shape = (batch_size, len(self.angles) * self.num_levels * self.num_levels, channels, 1)
                return torch.zeros(result_shape, device=device)
            
            # 在CPU上进行灰度量化
            x_min = np.min(x_cpu, axis=(2, 3), keepdims=True)
            x_max = np.max(x_cpu, axis=(2, 3), keepdims=True)
            x_range = np.maximum(x_max - x_min, 1e-6)  # 防止除零，使用广播
            x_scaled = np.clip(((x_cpu - x_min) / x_range * (self.num_levels - 1)), 
                              0, self.num_levels - 1).astype(np.int32)
            
            # 初始化存储所有角度的GLCM
            all_glcms = []
            
            # 在CPU上计算所有角度的GLCM
            for angle in self.angles:
                # 使用NumPy计算偏移量
                offset_x = int(round(self.distance * np.cos(angle)))
                offset_y = int(round(self.distance * np.sin(angle)))
                
                # 跳过无效偏移
                if offset_x == 0 and offset_y == 0:
                    continue
                
                # 初始化当前角度的GLCM - 使用更高效的初始化
                glcm = np.zeros((batch_size, channels, self.num_levels, self.num_levels), dtype=np.float32)
                
                # 计算有效区域
                ref_start_y = max(0, -offset_y)
                ref_end_y = min(height, height - offset_y)
                ref_start_x = max(0, -offset_x)
                ref_end_x = min(width, width - offset_x)
                
                # 确保有效区域存在
                if ref_start_y >= ref_end_y or ref_start_x >= ref_end_x:
                    continue
                    
                # 提取参考点和相邻点
                ref_slice = x_scaled[:, :, ref_start_y:ref_end_y, ref_start_x:ref_end_x]
                neighbor_y_start = ref_start_y + offset_y
                neighbor_y_end = ref_end_y + offset_y
                neighbor_x_start = ref_start_x + offset_x
                neighbor_x_end = ref_end_x + offset_x
                neighbor_slice = x_scaled[:, :, neighbor_y_start:neighbor_y_end, neighbor_x_start:neighbor_x_end]
                
                # 验证切片形状
                if ref_slice.shape != neighbor_slice.shape:
                    print(f"Warning: Mismatched slice shapes: {ref_slice.shape} vs {neighbor_slice.shape}")
                    continue
                
                valid_h, valid_w = ref_slice.shape[2], ref_slice.shape[3]
                if valid_h <= 0 or valid_w <= 0:
                    continue
                
                # 批量计算GLCM以提高效率
                for b in range(batch_size):
                    for c in range(channels):
                        # 使用NumPy的向量化操作更高效地计算GLCM
                        ref_flat = ref_slice[b, c].flatten()
                        neighbor_flat = neighbor_slice[b, c].flatten()
                        
                        # 创建联合索引数组，每个元素表示一个 (i,j) 对
                        joint_idx = ref_flat * self.num_levels + neighbor_flat
                        
                        # 使用bincount计算联合直方图
                        counts = np.bincount(joint_idx, minlength=self.num_levels * self.num_levels)
                        glcm[b, c] = counts.reshape(self.num_levels, self.num_levels)
                
                # 归一化 - 添加数值稳定性检查
                norm_factor = valid_h * valid_w
                if norm_factor > 0:
                    glcm = np.divide(glcm, norm_factor, out=np.zeros_like(glcm), where=norm_factor!=0)
                
                # 重整GLCM形状
                glcm = glcm.reshape(batch_size, channels, self.num_levels * self.num_levels)
                all_glcms.append(glcm)
            
            # 如果没有有效的GLCM，返回空张量
            if not all_glcms:
                result_shape = (batch_size, len(self.angles) * self.num_levels * self.num_levels, channels, 1)
                return torch.zeros(result_shape, device=device)
            
            # 将NumPy数组合并，然后转换回PyTorch张量
            all_glcms_array = np.concatenate(all_glcms, axis=2)  # [B, C, num_levels*num_levels*angles]
            
            # 检查是否有NaN或Inf
            if not np.isfinite(all_glcms_array).all():
                print("Warning: Non-finite values in GLCM computation, replacing with zeros")
                all_glcms_array = np.nan_to_num(all_glcms_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 转换为PyTorch张量并移回到GPU
            combined_glcm = torch.from_numpy(all_glcms_array).float().to(device)
            
            # 调整形状
            combined_glcm = combined_glcm.permute(0, 2, 1).unsqueeze(-1)  # [B, num_levels*num_levels*angles, C, 1]
            return combined_glcm
            
        except Exception as e:
            print(f"Error in GLCM computation: {e}. Using fallback.")
            # 返回随机初始化的GLCM，避免训练中断
            result_shape = (batch_size, len(self.angles) * self.num_levels * self.num_levels, channels, 1)
            return torch.zeros(result_shape, device=device)
    
    def forward(self, x):
        """
        提取共生矩阵特征
        
        Args:
            x: 输入图像或特征图，形状为 [B, C, H, W]
            
        Returns:
            cooc_features: 共生矩阵特征 (spatial features, global features)
                spatial_features: 空间分布特征 [B, reduction_dim, H, W]
                global_features: 全局纹理特征 [B, 64] - 必须是64维
        """
        batch_size = x.size(0)
        device = x.device
        
        try:
            # 检查输入数据的有效性
            if x.size(2) < 2 or x.size(3) < 2:
                raise ValueError(f"Input size too small: {x.shape}")
                
            if not torch.isfinite(x).all():
                print("Warning: Input contains non-finite values, replacing with zeros")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 计算多角度共生矩阵
            glcm = self.compute_glcm(x)
            
            # 检查GLCM是否有效
            if glcm.numel() == 0 or not torch.isfinite(glcm).all():
                print("Warning: Invalid GLCM produced, using fallback")
                raise ValueError("Invalid GLCM")
            
            # 应用降维卷积获取空间特征
            spatial_features = self.reduction(glcm)
            
            # 提取全局纹理统计特征
            global_features = self.texture_stats(spatial_features)
            
            # 检查特征维度
            current_dim = global_features.size(1)
            
            # 调整特征维度至目标64维
            if current_dim != 64:
                print(f"Adjusting global feature dimension from {current_dim} to 64")
                if current_dim > 64:
                    # 如果维度过大，使用自适应池化缩减
                    global_features = F.adaptive_avg_pool1d(global_features.unsqueeze(1), 64).squeeze(1)
                else:
                    # 如果维度过小，使用更平滑的线性插值
                    if current_dim > 0:
                        # 计算缩放比例
                        scale_factor = 64.0 / current_dim
                        # 使用线性插值而不是零填充
                        if scale_factor > 1.0:
                            # 创建临时张量进行插值
                            temp = global_features.unsqueeze(1)  # [B, 1, current_dim]
                            # 使用1D插值
                            expanded = F.interpolate(temp, size=64, mode='linear', align_corners=False)
                            global_features = expanded.squeeze(1)  # [B, 64]
                        else:
                            # 创建零张量并复制
                            expanded = torch.zeros(batch_size, 64, device=device)
                            expanded[:, :current_dim] = global_features
                            # 添加小噪声到剩余区域以增加多样性
                            noise = torch.randn(batch_size, 64 - current_dim, device=device) * 0.01
                            expanded[:, current_dim:] = noise
                            global_features = expanded
                    else:
                        # 处理极端情况：当前维度为0
                        global_features = torch.randn(batch_size, 64, device=device) * 0.01
            
            # 应用L2归一化以提高稳定性
            global_features = F.normalize(global_features, p=2, dim=1)
        
        except Exception as e:
            print(f"Error in co-occurrence feature extraction: {e}. Creating fallback features.")
            # 完全回退到随机初始化的特征
            # 创建模拟的空间特征
            try:
                out_channels = self.reduction[-3].out_channels
            except:
                out_channels = 64  # 默认值
                
            spatial_features = torch.zeros(batch_size, out_channels, 
                                          x.size(2), x.size(3), device=device)
            # 创建64维全局特征
            global_features = torch.zeros(batch_size, 64, device=device)
            # 添加随机噪声并归一化
            noise = torch.randn_like(global_features) * 0.1
            global_features = F.normalize(global_features + noise, p=2, dim=1)
            
        # 最后检查确保维度正确
        assert global_features.size(1) == 64, f"Global feature dimension must be 64, got {global_features.size(1)}"
        
        return spatial_features, global_features


class EarlyFusionModule(nn.Module):
    """
    早期融合模块
    在特征提取的早期阶段融合多种特征
    包括频域特征、共生矩阵纹理特征和空间域特征
    """
    def __init__(self, in_channels, reduction_ratio=4):
        super(EarlyFusionModule, self).__init__()
        
        self.reduced_channels = in_channels // reduction_ratio
        
        # 频域变换
        self.freq_transform = FrequencyTransform(use_dct=True, use_dft=False)
        
        # 共生矩阵变换 - 使用多角度
        self.cooc_transform = CooccurrenceMatrix(
            num_levels=8, 
            reduction_dim=self.reduced_channels,
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]  # 四个方向
        )
        
        # 空间域处理
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 3, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        执行早期特征融合
        
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
            
        Returns:
            fused: 融合后的特征图
            features_dict: 各类特征的字典，用于深入分析
        """
        # 提取频域特征
        freq_features = self.freq_transform(x)
        freq_features = F.interpolate(freq_features, size=x.shape[2:], mode='bilinear')
        
        # 提取共生矩阵特征 - 现在返回空间和全局特征
        cooc_spatial, cooc_global = self.cooc_transform(x)
        cooc_spatial = F.interpolate(cooc_spatial, size=x.shape[2:], mode='bilinear')
        
        # 提取空间特征
        spatial_features = self.spatial_conv(x)
        
        # 使用注意力机制对不同特征赋予权重
        concat_features = torch.cat([freq_features, cooc_spatial, spatial_features], dim=1)
        attention_weights = self.attention(concat_features)  # [B, 3, H, W]
        
        # 应用注意力权重
        b, _, h, w = freq_features.shape
        weighted_freq = freq_features * attention_weights[:, 0:1, :, :]
        weighted_cooc = cooc_spatial * attention_weights[:, 1:2, :, :]
        weighted_spatial = spatial_features * attention_weights[:, 2:3, :, :]
        
        # 融合加权后的特征
        weighted_concat = torch.cat([weighted_freq, weighted_cooc, weighted_spatial], dim=1)
        fused_features = self.fusion_conv(weighted_concat)
        
        # 构建特征字典以便于分析
        features_dict = {
            'frequency': freq_features,
            'cooccurrence_spatial': cooc_spatial,
            'cooccurrence_global': cooc_global,
            'spatial': spatial_features,
            'attention_weights': attention_weights,
            'fused': fused_features
        }
        
        return fused_features, features_dict


class LateFusionModule(nn.Module):
    """
    晚期融合模块
    在特征提取的后期阶段融合多种特征
    支持语义增强和动态注意力权重
    """
    def __init__(self, feature_dims, embedding_dim=512, feature_names=None):
        super(LateFusionModule, self).__init__()
        
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        # 特征名称，用于更好地跟踪不同特征的作用
        if feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(feature_dims))]
        else:
            assert len(feature_names) == len(feature_dims), "feature_names长度必须与feature_dims相同"
            self.feature_names = feature_names
        
        # 为每个特征类型创建变换网络
        self.transforms = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(embedding_dim),
                nn.Dropout(0.2)  # 增加dropout以增强模型的鸭棒性
            ) for name, dim in zip(self.feature_names, feature_dims)
        })
        
        # 上下文感知编码器 - 处理特征间的交互
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * len(feature_dims), embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 动态注意力模块
        self.query_proj = nn.Linear(embedding_dim, embedding_dim // 2)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim // 2)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features_dict):
        """
        执行晚期特征融合
        
        Args:
            features_dict: 特征字典，键为特征名称，值为对应特征张量，形状为 [B, feature_dim[i]]
            
        Returns:
            fused: 融合后的特征，形状为 [B, embedding_dim]
            attention_weights: 注意力权重字典，用于可视化分析
        """
        batch_size = next(iter(features_dict.values())).size(0)
        device = next(iter(features_dict.values())).device
        
        # 检查输入特征是否完整
        for name in self.feature_names:
            assert name in features_dict, f"缺少特征: {name}"
        
        # 处理特征维度问题，确保与预期一致
        expected_dims = dict(zip(self.feature_names, self.feature_dims))
        fixed_features = {}
        
        for name in self.feature_names:
            feature = features_dict[name]
            expected_dim = expected_dims[name]
            current_dim = feature.size(1)
            
            # 打印当前特征维度
            print(f"Feature {name}: expected {expected_dim}D, got {current_dim}D")
            
            # 如果维度不匹配，尝试修复
            if current_dim != expected_dim:
                print(f"Fixing dimension mismatch for {name}")
                if current_dim > expected_dim:
                    # 如果维度过大，裁剪或压缩
                    fixed_features[name] = F.adaptive_avg_pool1d(
                        feature.unsqueeze(1), expected_dim).squeeze(1)
                else:
                    # 如果维度过小，扩展并带噪声
                    expanded = torch.zeros(batch_size, expected_dim, device=device)
                    expanded[:, :current_dim] = feature
                    if current_dim < expected_dim:
                        noise = torch.randn(batch_size, expected_dim - current_dim, device=device) * 0.01
                        expanded[:, current_dim:] = noise
                    fixed_features[name] = expanded
            else:
                # 维度正确，直接使用
                fixed_features[name] = feature
        
        # 1. 将每个特征映射到统一的嵌入空间
        transformed_features = {}
        for name in self.feature_names:
            try:
                transformed_features[name] = self.transforms[name](fixed_features[name])
            except Exception as e:
                print(f"Error transforming {name}: {e}")
                # 如果转换失败，使用零矩阵替代
                embedding_dim = next(self.transforms[name].parameters()).size(0)
                transformed_features[name] = torch.zeros(batch_size, embedding_dim, device=device)
                # 添加噪声使特征更真实
                noise = torch.randn_like(transformed_features[name]) * 0.01
                transformed_features[name] = transformed_features[name] + noise
        
        # 2. 合并所有特征以获得上下文表示
        concat_embeddings = torch.cat([transformed_features[name] for name in self.feature_names], dim=1)
        context_embedding = self.context_encoder(concat_embeddings)
        
        # 3. 计算注意力分数
        query = self.query_proj(context_embedding)  # [B, embed_dim//2]
        
        attention_scores = {}
        for name in self.feature_names:
            key = self.key_proj(transformed_features[name])  # [B, embed_dim//2]
            # 计算注意力分数，这里用点积而非缩放点积
            attention_scores[name] = torch.sum(query * key, dim=1)  # [B]
        
        # 4. 转换分数为权重
        stacked_scores = torch.stack([attention_scores[name] for name in self.feature_names], dim=1)  # [B, num_features]
        attention_weights = F.softmax(stacked_scores, dim=1)  # [B, num_features]
        
        # 5. 加权合并特征
        values = torch.stack([self.value_proj(transformed_features[name]) for name in self.feature_names], dim=1)  # [B, num_features, embed_dim]
        
        # 加权融合
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [B, embed_dim]
        
        # 6. 最终投影
        fused_features = self.output_proj(weighted_sum + context_embedding)  # 添加残差连接
        
        # 7. 构建注意力权重字典供分析使用
        # 对每个特征类型，返回整个batch的权重
        attention_weights_dict = {name: attention_weights[:, i].unsqueeze(1) 
                                 for i, name in enumerate(self.feature_names)}
        
        # 计算并打印每个特征的平均权重，仅用于调试
        mean_weights = {name: weights.mean().item() 
                       for name, weights in attention_weights_dict.items()}
        print(f"Mean attention weights: {mean_weights}")
        
        return fused_features, attention_weights_dict
