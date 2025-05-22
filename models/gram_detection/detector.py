import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .gram_module import GramTextureExtractor


class GramBasedDetector(nn.Module):
    """
    基于Gram矩阵的图像生成检测器
    利用纹理特征区分真实图像和AI生成图像
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, num_classes=2, dropout_rate=0.5):
        super(GramBasedDetector, self).__init__()
        
        # 加载预训练的骨干网络
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_dim = self.backbone.fc.in_features
            # 移除最后的全连接层
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_out_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Gram纹理特征提取器
        self.texture_extractor = GramTextureExtractor(
            self.backbone, 
            feature_layers=(2, 3, 4),  # ResNet中的中间层
            reduction=8
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 根据所选层计算融合后的特征维度
        # 这需要根据具体的backbone和所选层调整
        if backbone_name.startswith('resnet'):
            if backbone_name == 'resnet50':
                feature_dims = [512, 1024, 2048]  # ResNet50的layer2,3,4的输出通道数
            else:  # resnet34
                feature_dims = [128, 256, 512]    # ResNet34的layer2,3,4的输出通道数
        else:  # efficientnet_b0
            feature_dims = [24, 40, 112]          # 需要根据实际情况调整
            
        fusion_dim = sum(feature_dims)
        
        # 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 纹理损失网络
        self.texture_fc = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W]
            
        Returns:
            logits: 分类logits，形状为 [B, num_classes]
            texture_embedding: 纹理嵌入，形状为 [B, 128]
        """
        batch_size = x.size(0)
        
        # 提取纹理特征
        texture_features = self.texture_extractor(x)
        
        # 融合特征
        pooled_features = []
        for feature in texture_features:
            pooled = self.fusion(feature)
            pooled_features.append(pooled)
        
        # 连接所有特征
        fused_features = torch.cat(pooled_features, dim=1)
        
        # 生成分类结果
        logits = self.classifier(fused_features)
        
        # 生成纹理嵌入，用于辅助损失计算
        texture_embedding = self.texture_fc(fused_features)
        
        return logits, texture_embedding


class TextureTripletLoss(nn.Module):
    """
    纹理三元组损失
    用于增强模型对真实和生成图像纹理差异的学习
    """
    def __init__(self, margin=1.0):
        super(TextureTripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        """
        计算三元组损失
        
        Args:
            embeddings: 纹理嵌入，形状为 [B, embedding_dim]
            labels: 图像标签，0表示真实，1表示生成，形状为 [B]
            
        Returns:
            loss: 三元组损失值
        """
        # 分离真实和生成图像的嵌入
        real_indices = (labels == 0).nonzero(as_tuple=True)[0]
        fake_indices = (labels == 1).nonzero(as_tuple=True)[0]
        
        if len(real_indices) == 0 or len(fake_indices) == 0:
            # 如果批次中只有一种类型的图像，返回零损失
            return torch.tensor(0.0, device=embeddings.device)
        
        real_embeddings = embeddings[real_indices]
        fake_embeddings = embeddings[fake_indices]
        
        # 计算所有可能的真实-真实和真实-假对
        n_real = real_embeddings.size(0)
        n_fake = fake_embeddings.size(0)
        
        # 扩展为所有可能的对
        anchor = real_embeddings.unsqueeze(1).repeat(1, n_real, 1)  # [n_real, n_real, dim]
        positive = real_embeddings.unsqueeze(0).repeat(n_real, 1, 1)  # [n_real, n_real, dim]
        negative = fake_embeddings.unsqueeze(0).repeat(n_real, 1, 1)  # [n_real, n_fake, dim]
        
        # 重塑为二维张量以方便计算
        anchor = anchor.view(-1, embeddings.size(1))  # [(n_real*n_real), dim]
        positive = positive.view(-1, embeddings.size(1))  # [(n_real*n_real), dim]
        negative = negative.view(n_real * n_fake, embeddings.size(1))  # [(n_real*n_fake), dim]
        
        # 创建所有可能的三元组
        n_triplets = min(n_real * n_real, n_real * n_fake)
        triplet_anchor = anchor[:n_triplets]  # [n_triplets, dim]
        triplet_positive = positive[:n_triplets]  # [n_triplets, dim]
        triplet_negative = negative[:n_triplets]  # [n_triplets, dim]
        
        # 计算距离
        pos_dist = F.pairwise_distance(triplet_anchor, triplet_positive, p=2)
        neg_dist = F.pairwise_distance(triplet_anchor, triplet_negative, p=2)
        
        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss
    对难以分类的样本给予更高的权重
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        """
        计算focal loss
        
        Args:
            logits: 预测logits，形状为 [B, num_classes]
            targets: 目标标签，形状为 [B]
            
        Returns:
            loss: focal loss值
        """
        # 将目标转换为one-hot编码
        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes).float()
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        
        # 计算focal loss
        pt = torch.sum(targets_onehot * probs, dim=1)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-10)
        
        return loss.mean()


class DetectionLoss(nn.Module):
    """
    组合损失函数
    结合分类损失和纹理损失
    """
    def __init__(self, alpha=1.0, beta=0.5, focal_gamma=2.0):
        super(DetectionLoss, self).__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta    # 三元组损失权重
        
        self.cls_loss = FocalLoss(gamma=focal_gamma)
        self.texture_loss = TextureTripletLoss(margin=1.0)
        
    def forward(self, logits, embeddings, targets):
        """
        计算总损失
        
        Args:
            logits: 分类logits，形状为 [B, num_classes]
            embeddings: 纹理嵌入，形状为 [B, embedding_dim]
            targets: 目标标签，形状为 [B]
            
        Returns:
            total_loss: 总损失值
            cls_loss: 分类损失值
            text_loss: 纹理损失值
        """
        cls_loss = self.cls_loss(logits, targets)
        text_loss = self.texture_loss(embeddings, targets)
        
        total_loss = self.alpha * cls_loss + self.beta * text_loss
        
        return total_loss, cls_loss, text_loss
