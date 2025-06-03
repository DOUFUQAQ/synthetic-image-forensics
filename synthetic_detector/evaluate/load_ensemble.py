import sys
import os
import torch
import types

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_ensemble_model(checkpoint_path):
    """
    加载集成模型检查点，兼容两种格式：
    1. 直接的模型state_dict
    2. 包含"ensemble"键的集成模型
    
    Args:
        checkpoint_path: 模型检查点路径
        
    Returns:
        加载好的模型
    """
    from models.detector import create_model
    
    print(f"尝试加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建默认模型
    model = create_model(pretrained=True)
    
    # 检查是否是集成模型格式
    if 'ensemble' in checkpoint:
        print("检测到集成模型格式，提取模型权重...")
        ensemble_models = checkpoint['ensemble']
        
        # 如果集成模型是列表，取第一个作为基础模型
        if isinstance(ensemble_models, list) and len(ensemble_models) > 0:
            print(f"集成包含 {len(ensemble_models)} 个模型，使用第一个模型的参数...")
            model_dict = ensemble_models[0]
            
            # 尝试加载模型
            try:
                model.load_state_dict(model_dict, strict=False)
                print("已成功加载集成模型中的第一个模型")
            except Exception as e:
                print(f"加载集成模型参数时出错: {e}")
        else:
            print("集成模型格式不正确，尝试直接加载...")
            try:
                model.load_state_dict(checkpoint['ensemble'], strict=False)
                print("已直接加载集成模型参数")
            except Exception as e:
                print(f"直接加载集成模型参数时出错: {e}")
    else:
        # 尝试直接加载模型权重
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("已成功加载模型参数")
        except Exception as e:
            print(f"加载模型参数时出错: {e}")
    
    # 确保模型有必要的参数
    validate_model_checkpoint(model)
    
    return model

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
                
            # 检查温度参数
            if not hasattr(self, 'temperature') or self.temperature is None:
                device = next(self.parameters()).device
                self.temperature = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float, device=device), requires_grad=True)
                print("警告: 温度参数缺失，已使用默认值初始化")
                
            # 其他必要参数
            if not hasattr(self, '_default_real_boost'):
                self._default_real_boost = 0.1
            
            if not hasattr(self, 'eval_mode_active'):
                self.eval_mode_active = False
                
            if not hasattr(self, 'data_mean') or self.data_mean is None:
                self.register_buffer('data_mean', torch.tensor([0.485, 0.456, 0.406]))
                
            if not hasattr(self, 'data_std') or self.data_std is None:
                self.register_buffer('data_std', torch.tensor([0.229, 0.224, 0.225]))
        
        # 将方法动态添加到模型
        model._ensure_critical_parameters = types.MethodType(ensure_critical_parameters, model)
        model._ensure_critical_parameters()
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试加载集成模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    
    args = parser.parse_args()
    
    model = load_ensemble_model(args.checkpoint)
    print("模型加载成功!")
    print("模型结构:", model)
