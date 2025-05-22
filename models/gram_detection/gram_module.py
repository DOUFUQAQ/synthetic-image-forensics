import torch
import torch.nn as nn
import torch.nn.functional as F


class GramMatrix(nn.Module):
    """
    计算特征图的Gram矩阵
    Gram矩阵捕获特征间的统计相关性，对于识别生成图像很有效
    """
    def __init__(self, normalize=True):
        super(GramMatrix, self).__init__()
        self.normalize = normalize
        
    def forward(self, x):
        """
        计算输入特征图的Gram矩阵
        
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
        
        Returns:
            gram: Gram矩阵，形状为 [B, C, C]
        """
        B, C, H, W = x.size()
        
        # 将特征图重塑为二维矩阵
        features = x.view(B, C, H * W)
        
        # 计算Gram矩阵 (C x C)
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # 归一化
        if self.normalize:
            gram = gram / (C * H * W)
        
        return gram


class MultiScaleGramModule(nn.Module):
    """
    多尺度Gram矩阵提取模块
    从不同尺度的特征图计算Gram矩阵，更全面地捕获纹理信息
    """
    def __init__(self, in_channels, reduction=8, scales=(1, 2, 4)):
        super(MultiScaleGramModule, self).__init__()
        self.scales = scales
        self.gram = GramMatrix(normalize=True)
        
        # 在每个尺度上的降维卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
            for _ in scales
        ])
        
        # 自适应池化层
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((H, H)) for H in scales
        ])
        
        # 合并后的特征映射
        reduced_dim = in_channels // reduction
        gram_dim = reduced_dim * reduced_dim * len(scales)
        self.fc = nn.Sequential(
            nn.Linear(gram_dim, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        计算多尺度Gram特征
        
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
        
        Returns:
            out: 增强的特征图，形状为 [B, C, H, W]
        """
        B, C, _, _ = x.size()
        gram_features = []
        
        # 在不同尺度上计算Gram矩阵
        for i, scale in enumerate(self.scales):
            # 降维
            feat = self.convs[i](x)
            # 池化到指定尺度
            feat = self.pools[i](feat)
            # 计算Gram矩阵
            gram = self.gram(feat)
            # 平坦化Gram矩阵
            gram = gram.view(B, -1)
            gram_features.append(gram)
        
        # 连接所有尺度的Gram特征
        gram_features = torch.cat(gram_features, dim=1)
        
        # 通过全连接层生成通道注意力权重
        channel_weights = self.fc(gram_features).view(B, C, 1, 1)
        
        # 应用注意力机制
        out = x * channel_weights
        
        return out


class GramTextureExtractor(nn.Module):
    """
    纹理特征提取器，用于检测图像中的伪造纹理模式
    """
    def __init__(self, backbone, feature_layers=(1, 2, 3), reduction=8):
        super(GramTextureExtractor, self).__init__()
        self.backbone = backbone
        self.feature_layers = feature_layers
        
        # 获取各层的特征维度
        # 这里假设backbone已经按照layer1, layer2等命名
        feature_dims = []
        for i in feature_layers:
            layer = getattr(backbone, f'layer{i}')
            # 获取该层最后一个块的输出通道数
            last_block = layer[-1]
            feature_dims.append(last_block.conv3.out_channels if hasattr(last_block, 'conv3') else 
                               last_block.conv2.out_channels if hasattr(last_block, 'conv2') else
                               last_block.conv1.out_channels)
        
        # 为每一层创建Gram模块
        self.gram_modules = nn.ModuleList([
            MultiScaleGramModule(dim, reduction=reduction)
            for dim in feature_dims
        ])
        
        # 初始化hooks和特征图存储
        self.features = {}
        self.hooks = []
        
        # 注册hooks以获取中间特征图
        for i in feature_layers:
            layer = getattr(backbone, f'layer{i}')
            self.hooks.append(
                layer.register_forward_hook(self._get_features_hook(f'layer{i}'))
            )
    
    def _get_features_hook(self, layer_name):
        """Hook函数，用于保存中间特征图"""
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x):
        """
        提取输入图像的纹理特征
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W]
        
        Returns:
            texture_features: 纹理特征列表
        """
        # 清除之前的特征图
        self.features = {}
        
        # 通过backbone提取特征
        _ = self.backbone(x)
        
        # 应用Gram模块处理特征图
        texture_features = []
        for i, layer_idx in enumerate(self.feature_layers):
            layer_name = f'layer{layer_idx}'
            feature = self.features[layer_name]
            enhanced_feature = self.gram_modules[i](feature)
            texture_features.append(enhanced_feature)
        
        return texture_features
