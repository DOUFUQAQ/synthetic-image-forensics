#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查点分析工具 - 用于检查训练好的模型检查点内容
"""

import torch
import os
import sys
import argparse
from torch.serialization import add_safe_globals

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检查点分析工具')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/detector_best.pth', help='模型checkpoint路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"分析检查点文件: {args.checkpoint}")
    
    try:
        # 添加允许的全局类以支持NumPy数组加载
        add_safe_globals(['numpy._core.multiarray._reconstruct'])
        
        # 加载checkpoint，禁用weights_only限制
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        
        # 检查检查点结构
        print(f"\n检查点类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"\n检查点键: {checkpoint.keys()}")
            
            # 检查模型状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("\n模型状态字典存在于 'model_state_dict' 键")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("\n模型状态字典存在于 'state_dict' 键")
            else:
                state_dict = checkpoint
                print("\n检查点本身似乎是状态字典")
            
            # 寻找特征权重参数
            weight_keys = [k for k in state_dict.keys() if 'weight' in k]
            print(f"\n权重参数键 ({len(weight_keys)}):")
            for k in weight_keys[:20]:  # 只打印前20个以避免过多输出
                print(f"  - {k}: {state_dict[k].shape}")
            
            if len(weight_keys) > 20:
                print(f"  ... 还有 {len(weight_keys) - 20} 个权重参数")
            
            # 特别查找base_weights参数
            if 'base_weights' in state_dict:
                print(f"\nbase_weights参数存在: {state_dict['base_weights']}")
            else:
                # 尝试寻找可能包含特征权重的参数
                for k in state_dict.keys():
                    if 'base' in k.lower() or 'feature' in k.lower() and 'weight' in k.lower():
                        print(f"\n可能的特征权重参数: {k}")
                        print(f"  值: {state_dict[k]}")
                        print(f"  形状: {state_dict[k].shape}")
            
            # 检查训练信息
            if 'epoch' in checkpoint:
                print(f"\n训练轮次: {checkpoint['epoch']}")
            if 'best_metric' in checkpoint:
                print(f"最佳指标: {checkpoint['best_metric']}")
            if 'optimizer_state_dict' in checkpoint:
                print(f"优化器状态存在")
            if 'scheduler_state_dict' in checkpoint:
                print(f"学习率调度器状态存在")
        else:
            print("检查点不是字典类型，无法进一步分析")
    
    except Exception as e:
        print(f"分析检查点时出错: {e}")

if __name__ == '__main__':
    main()
