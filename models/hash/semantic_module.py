import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SemanticEmbeddingModule(nn.Module):
    """
    语义嵌入模块
    提取和嵌入图像的高级语义信息
    """
    def __init__(self, in_features, embedding_dim=512, dropout_rate=0.2):
        super(SemanticEmbeddingModule, self).__init__()
        
        # 语义嵌入网络
        self.embedding_network = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        生成语义嵌入
        
        Args:
            x: 输入特征，形状为 [B, in_features]
            
        Returns:
            embedding: 语义嵌入，形状为 [B, embedding_dim]
        """
        return self.embedding_network(x)


class SemanticAttention(nn.Module):
    """
    语义注意力模块
    根据语义信息对特征图的不同区域赋予不同的重要性
    """
    def __init__(self, in_channels, reduction=16):
        super(SemanticAttention, self).__init__()
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        应用注意力机制
        
        Args:
            x: 输入特征图，形状为 [B, C, H, W]
            
        Returns:
            out: 增强的特征图，形状为 [B, C, H, W]
        """
        # 通道注意力
        channel_att = self.channel_att(x)
        x = x * channel_att
        
        # 空间注意力
        spatial_att = self.spatial_att(x)
        x = x * spatial_att
        
        return x


class MultiLevelSemanticExtractor(nn.Module):
    """
    多层级语义特征提取器
    从多个层次的特征图中提取语义信息
    """
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(MultiLevelSemanticExtractor, self).__init__()
        
        # 加载预训练的骨干网络
        if backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.out_features = 512
            self.layer_dims = [64, 128, 256, 512]  # 各层输出通道数
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            self.out_features = 512
            self.layer_dims = [64, 128, 256, 512]
        elif backbone_name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
            self.out_features = 1280
            self.layer_dims = [16, 24, 32, 96]  # 需要根据实际情况调整
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # 提取各层特征
        if backbone_name.startswith('resnet'):
            self.layer0 = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            )
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        else:  # mobilenet_v2
            # 为MobileNet设计合适的层划分
            features = backbone.features
            self.layer0 = features[0:2]   # 初始卷积和第一个倒残差块
            self.layer1 = features[2:4]   # 2-3倒残差块
            self.layer2 = features[4:7]   # 4-6倒残差块
            self.layer3 = features[7:14]  # 7-13倒残差块
            self.layer4 = features[14:]   # 剩余层
        
        # 添加注意力模块
        self.attention1 = SemanticAttention(self.layer_dims[0])
        self.attention2 = SemanticAttention(self.layer_dims[1])
        self.attention3 = SemanticAttention(self.layer_dims[2])
        self.attention4 = SemanticAttention(self.layer_dims[3])
        
        # 特征整合
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        提取多层次语义特征
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W]
            
        Returns:
            features: 各层特征的字典
            pooled_features: 池化后的各层特征的列表
        """
        # 提取各层特征
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 应用注意力
        x1_att = self.attention1(x1)
        x2_att = self.attention2(x2)
        x3_att = self.attention3(x3)
        x4_att = self.attention4(x4)
        
        # 全局池化
        p1 = self.global_pool(x1_att).squeeze(-1).squeeze(-1)
        p2 = self.global_pool(x2_att).squeeze(-1).squeeze(-1)
        p3 = self.global_pool(x3_att).squeeze(-1).squeeze(-1)
        p4 = self.global_pool(x4_att).squeeze(-1).squeeze(-1)
        
        # 组织输出
        features = {
            'layer1': x1_att,
            'layer2': x2_att,
            'layer3': x3_att,
            'layer4': x4_att
        }
        
        pooled_features = [p1, p2, p3, p4]
        
        return features, pooled_features
