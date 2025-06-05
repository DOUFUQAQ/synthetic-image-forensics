#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Contribution Analyzer - For evaluating the contribution of different feature subsets to detection performance
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到路径以导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 根据项目结构决定导入路径
try:
    # 先尝试根目录的detector和dataset
    from detector import MultiFeatureDetector, create_detector
    from dataset import get_dataloaders
    print("从项目根目录导入检测器和数据集模块")
except ImportError:
    # 如果失败，尝试使用models和data子目录
    from models.detector import create_detector, MultiFeatureDetector
    from data.dataset import get_dataloaders
    print("从models和data子目录导入检测器和数据集模块")

# 其他必要的导入
from utils.metrics import compute_metrics
from evaluate.load_ensemble import validate_model_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Feature Contribution Analyzer')
    
    # Basic parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading threads')
    parser.add_argument('--device', type=str, default='cuda', help='Device type')
    
    return parser.parse_args()


def get_top_k_weights(weights, k=3):
    """Get the top k features with the highest weights, set other feature weights to 0"""
    weights_np = np.array(weights)
    result = np.zeros_like(weights_np)
    # Get the indices of the top k features with the highest absolute values
    top_indices = np.abs(weights_np).argsort()[-k:]
    for idx in top_indices:
        result[idx] = weights_np[idx]
    # Normalize the weights so that their sum is 1
    if np.sum(np.abs(result)) > 0:
        result = result / np.sum(np.abs(result))
    return result.tolist()

def normalize_selected_weights(weights, selected_indices):
    """Only keep selected feature weights and normalize them"""
    weights_np = np.array(weights)
    result = np.zeros_like(weights_np)
    for idx in selected_indices:
        result[idx] = weights_np[idx]
    # Normalize the weights so that their sum is 1
    if np.sum(np.abs(result)) > 0:
        result = result / np.sum(np.abs(result))
    return result.tolist()

def adjust_weights_for_real_detection(weights):
    """Adjust weights to enhance real image detection capability"""
    weights_np = np.array(weights)
    # Find the indices of semantic features (usually index 3) and frequency features (index 1)
    semantic_idx = 3
    freq_idx = 1
    
    # Weight adjustment strategy:
    # 1. Significantly reduce the weight of semantic features, as they may overemphasize the semantic features of generated images
    weights_np[semantic_idx] *= 0.3  # More significantly reduce the weight of semantic features
    
    # 2. Significantly increase the weight of CNN features, as they contain texture details of real images
    weights_np[0] *= 2.5  # Significantly increase the weight of CNN features
    
    # 3. Moderately increase the weight of frequency features
    weights_np[1] *= 1.8  # Increase the weight of frequency features
    
    # Normalize the weights
    return (weights_np / np.sum(np.abs(weights_np))).tolist()

def strong_real_detection_weights(weights):
    """More aggressively adjust weights to significantly improve real image detection rate"""
    # Emphasize CNN and frequency features, reduce semantic features
    return [0.45, 0.45, 0.08, 0.02, 0.0]  # CNN and frequency features dominate

def real_boost_adjustment(model):
    """Adjust the model's real_boost_factor to enhance real image detection"""
    if hasattr(model, 'real_boost_factor'):
        # Save the original value
        original_boost = model.real_boost_factor
        # Increase the real class boost factor from 0.2 to 0.8 (significant increase)
        model.real_boost_factor = 0.8
        print(f"\nAdjust real_boost_factor: {original_boost} -> {model.real_boost_factor}")
        return original_boost
    return None

def adjust_class_bias(model):
    """Adjust the model's class bias to improve real image detection rate"""
    if hasattr(model, 'class_bias'):
        # Save the original bias
        original_bias = model.class_bias.clone() if hasattr(model.class_bias, 'clone') else model.class_bias.copy()
        
        # Significantly bias towards the real class (class 0) - correctly handle Tensor type
        if isinstance(model.class_bias, torch.Tensor):
            # Check the original shape
            print(f"Class bias tensor shape: {model.class_bias.shape}")
            
            # Create a new initial value, keeping the same shape as the original
            if model.class_bias.shape == torch.Size([2]):  # If the shape is [2]
                new_bias = torch.tensor([0.6, -0.3], device=model.class_bias.device)
            elif model.class_bias.shape == torch.Size([1, 2]):  # If the shape is [1, 2]
                new_bias = torch.tensor([[0.6, -0.3]], device=model.class_bias.device)
            else:  # For other shapes, use full_like to fill
                # Create a tensor with the same shape as the original and fill the first element with 0.6, the second element with -0.3
                new_bias = torch.zeros_like(model.class_bias)
                if len(model.class_bias.shape) > 1:
                    # Multi-dimensional shape
                    new_bias[..., 0] = 0.6
                    new_bias[..., 1] = -0.3
                else:
                    # One-dimensional shape
                    new_bias[0] = 0.6
                    if len(new_bias) > 1:
                        new_bias[1] = -0.3
            
            # Use the copy_ method to update the tensor value
            model.class_bias.copy_(new_bias)
        else:
            # If it's not a Tensor type, try to assign directly
            model.class_bias = torch.tensor([[0.6, -0.3]], device='cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"\nAdjust class_bias: {original_bias} -> {model.class_bias}")
        return original_bias
    return None

def adjust_temperature(model, new_temp=1.0):
    """Adjust model temperature parameter to make predictions more certain"""
    if hasattr(model, 'temperature'):
        original_temp = model.temperature
        # The temperature parameter may be of type torch.nn.Buffer, requiring the correct data type
        if isinstance(model.temperature, torch.Tensor):
            # Check the shape and handle correctly
            if model.temperature.dim() == 0:  # Scalar shape
                # For scalar shape, use scalar form of tensor
                model.temperature.copy_(torch.tensor(new_temp, device=model.temperature.device))
            else:  # Shape with dimensions
                # Maintain consistency with the original shape
                new_tensor = torch.full_like(model.temperature, new_temp)
                model.temperature.copy_(new_tensor)
        else:
            # If it's not a Tensor type, assign directly
            model.temperature = new_temp
            
        print(f"\nAdjust temperature: {original_temp} -> {model.temperature}")
        return original_temp
    return None

def analyze_feature_importance(model, val_loader, device, feature_count=5, apply_bias_correction=False, adjust_temp=False):
    """
    Evaluate feature importance by testing each feature individually and in combinations
    
    Analyze the contribution of different features to model performance, including:
    1. Single Feature Tests - Evaluate the contribution of each feature individually
    2. Feature Combination Tests - Evaluate the synergistic effects of feature combinations
    3. Learned Weight Tests - Evaluate the weight combinations learned during model training
    4. Optimized Combination Tests - Targeted optimization based on learned weights
    """
    model.eval()
    
    # Save results for different feature combinations
    results = {}
    
    # First save the original model's learned weights as a baseline
    if hasattr(model, 'base_weights'):
        learned_weights = model.base_weights.data.clone()
        print(f"\nOriginal learned weights: {learned_weights.cpu().numpy()}")
        
        # Add learned weights as one scheme
        weight_schemes = {
            # Test each feature individually
            'CNN_Feature': [1.0, 0.0, 0.0, 0.0, 0.0],
            'Frequency_Feature': [0.0, 1.0, 0.0, 0.0, 0.0],
            'Co-occurrence_Feature': [0.0, 0.0, 1.0, 0.0, 0.0],
            'Semantic_Feature': [0.0, 0.0, 0.0, 1.0, 0.0],
            'Hash_Feature': [0.0, 0.0, 0.0, 0.0, 1.0],
            
            # Feature combination tests
            'CNN+Frequency': [0.5, 0.5, 0.0, 0.0, 0.0],
            'CNN+Co-occurrence': [0.5, 0.0, 0.5, 0.0, 0.0],
            'Frequency+Co-occurrence': [0.0, 0.5, 0.5, 0.0, 0.0],
            'Semantic+Frequency': [0.0, 0.5, 0.0, 0.5, 0.0],
            'Semantic+Co-occurrence': [0.0, 0.0, 0.5, 0.5, 0.0],
            
            # Uniform weights
            'Uniform_Weights': [0.2, 0.2, 0.2, 0.2, 0.2],
            
            # Learned weights
            'Learned_Weights': learned_weights.cpu().numpy().tolist(),
            
            # Based on learned weights ranking, using only the top three most important features
            'Top_Three_Features': get_top_k_weights(learned_weights.cpu().numpy(), k=3),
            
            # Only use semantic and frequency features (two most important features)
            'Semantic+Frequency_Normalized': normalize_selected_weights(learned_weights.cpu().numpy(), [1, 3]),
            
            # Enhanced real image detection version - reduce semantic feature weight
            'Real_Image_Enhanced': adjust_weights_for_real_detection(learned_weights.cpu().numpy()),
            
            # Extreme real image detection scheme
            'Extreme_Real_Image_Detection': strong_real_detection_weights(learned_weights.cpu().numpy()),
            
            # Only use CNN and frequency features
            'CNN+Frequency_Enhanced': [0.5, 0.5, 0.0, 0.0, 0.0],
        }
    else:
        # If model doesn't have base_weights attribute, use default weight schemes
        weight_schemes = {
            # Test each feature individually
            'CNN_Feature': [1.0, 0.0, 0.0, 0.0, 0.0],
            'Frequency_Feature': [0.0, 1.0, 0.0, 0.0, 0.0],
            'Co-occurrence_Feature': [0.0, 0.0, 1.0, 0.0, 0.0],
            'Semantic_Feature': [0.0, 0.0, 0.0, 1.0, 0.0],
            'Hash_Feature': [0.0, 0.0, 0.0, 0.0, 1.0],
            
            # Uniform weights
            'Uniform_Weights': [0.2, 0.2, 0.2, 0.2, 0.2],
        }
    
    # Save original weights for restoration after testing
    original_weights = model.base_weights.data.clone() if hasattr(model, 'base_weights') else None

    # Test each weight scheme sequentially
    for scheme_name, weights in weight_schemes.items():
        print(f"\nTesting feature combination: {scheme_name} - weights: {[round(w, 4) for w in weights]}")
        
        # Set model weights
        
        # Apply bias correction if specified
        original_bias = None
        original_temp = None
        if apply_bias_correction and scheme_name in ['Extreme_Real_Image_Detection', 'CNN+Frequency_Enhanced']:
            original_bias = adjust_class_bias(model)
            if adjust_temp:
                original_temp = adjust_temperature(model, new_temp=1.0)
        if isinstance(model, MultiFeatureDetector) and hasattr(model, 'base_weights'):
            model.base_weights.data = torch.tensor(weights, device=device)
            
        # Evaluate current weight scheme
        metrics = evaluate_model(model, val_loader, device)
        
        # Save results
        results[scheme_name] = {
            'weights': weights,
            'metrics': metrics,
            'bias_correction_applied': apply_bias_correction and scheme_name in ['Extreme_Real_Image_Detection', 'CNN+Frequency_Enhanced'],
            'temperature_adjusted': adjust_temp,
        }
        
        # Print key metrics
        auc = metrics.get('auc', 0.0)
        balanced_acc = metrics.get('balanced_accuracy', metrics.get('balanced_acc', 0.0))
        real_acc = metrics.get('real_class_accuracy', 0.0)
        synth_acc = metrics.get('synthetic_class_accuracy', 0.0)
        
        print(f"  AUC: {auc:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Real Image Accuracy: {real_acc:.4f}, Synthetic Image Accuracy: {synth_acc:.4f}")
        
        # Calculate detection bias (synthetic detection rate - real image detection rate)
        detection_bias = synth_acc - real_acc
        print(f"  Detection Bias: {detection_bias:.4f} (0 means no bias, positive values indicate synthetic detection bias)")
        
        # Restore bias and temperature parameters (if changed)
        if apply_bias_correction and original_bias is not None:
            model.class_bias = original_bias
        if adjust_temp and original_temp is not None:
            model.temperature = original_temp
        
        # Show available metric keys only on first iteration
        if scheme_name == list(weight_schemes.keys())[0]:
            print(f"  Available metric keys: {list(metrics.keys())}")
    
    # Restore original weights
    if original_weights is not None:
        model.base_weights.data = original_weights
        
    return results


def evaluate_model(model, val_loader, device):
    """Evaluate model performance, returns a dictionary of metrics"""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    
    # Add tracking for recognition results of each class
    class_correct = [0, 0]  # [real_correct, synthetic_correct]
    class_total = [0, 0]    # [real_total, synthetic_total]
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluation Progress", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # Process various output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs)
                if isinstance(logits, dict):
                    logits = logits.get('logits', list(logits.values())[0])
            else:
                logits = outputs
            
            # Calculate prediction results
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Track accuracy for each class
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1
            
            # Collect results
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Collect positive class probability
    
    # Calculate various metrics
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy for each class
    real_class_accuracy = class_correct[0] / max(class_total[0], 1) if class_total[0] > 0 else 0.0
    synthetic_class_accuracy = class_correct[1] / max(class_total[1], 1) if class_total[1] > 0 else 0.0
    
    # Manually calculate balanced accuracy
    balanced_accuracy = (real_class_accuracy + synthetic_class_accuracy) / 2.0
    
    # Use utils.metrics to calculate detailed metrics
    try:
        from utils.metrics import compute_metrics
        # Correctly pass parameters, using all_probs as y_prob rather than probabilities
        metrics = compute_metrics(all_targets, all_preds, y_prob=all_probs)
        
        # Add missing key metrics
        metrics['balanced_accuracy'] = balanced_accuracy
        metrics['real_class_accuracy'] = real_class_accuracy
        metrics['synthetic_class_accuracy'] = synthetic_class_accuracy
        
        # Print original metric keys for debugging
        print(f"  Original metric keys: {list(metrics.keys())}")
        print(f"  After adding key metrics: balanced_accuracy={balanced_accuracy:.4f}, real_acc={real_class_accuracy:.4f}, synth_acc={synthetic_class_accuracy:.4f}")
    except Exception as e:
        print(f"Warning: Cannot use utils.metrics, using basic metrics: {e}")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
        
        # Accuracy, precision, recall, F1 score
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='binary', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='binary', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='binary', zero_division=0),
            'balanced_accuracy': balanced_accuracy,  # Use manually calculated balanced accuracy
            'real_class_accuracy': real_class_accuracy,
            'synthetic_class_accuracy': synthetic_class_accuracy,
        }
        
        # Calculate ROC AUC
        try:
            metrics['auc'] = roc_auc_score(all_targets, all_probs)
        except ValueError:
            print("Warning: Unable to calculate AUC")
            metrics['auc'] = 0.5
    
    return metrics


def load_model_and_data(args, device):
    """Load model and evaluation dataset"""
    # Import plotting library
    import matplotlib
    matplotlib.use('Agg')  # Use a headless backend, suitable for server environments
    import matplotlib.pyplot as plt
    
    # Load model
    print(f"Loading model: {args.checkpoint}")
    try:
        # Create an initial model instance with 256-dimensional features to match the trained model
        model = create_detector(feature_dim=256).to(device)
        
        # Add allowed global classes to support NumPy array loading
        try:
            # Try different versions of PyTorch's loading methods
            if hasattr(torch, '_load') and hasattr(torch._load, 'add_safe_globals'):
                # New version of PyTorch
                torch._load.add_safe_globals(['numpy._core.multiarray._reconstruct'])
            elif hasattr(torch.serialization, 'add_safe_globals'):
                # Old version of PyTorch
                from torch.serialization import add_safe_globals
                add_safe_globals(['numpy._core.multiarray._reconstruct'])
        except Exception as e:
            print(f"Failed to add safe global classes, but will continue to load: {e}")
        
        # Load checkpoint, disable weights_only restriction
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Handle different formats of checkpoints
        if isinstance(checkpoint, dict):
            # Check and print feature weight information (if available)
            if 'feature_weights' in checkpoint:
                print(f"Checkpoint contains feature weights: {checkpoint['feature_weights']}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Check the model state dictionary for base_weights
                if 'base_weights' in state_dict:
                    print(f"Model state dictionary contains base_weights: {state_dict['base_weights']}")
                model.load_state_dict(state_dict)
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                if 'base_weights' in state_dict:
                    print(f"Model state dictionary contains base_weights: {state_dict['base_weights']}")
                model.load_state_dict(state_dict)
            elif 'ensemble' in checkpoint and len(checkpoint['ensemble']) > 0:
                model.load_state_dict(checkpoint['ensemble'][0])
            else:
                # Directly try to load
                if 'base_weights' in checkpoint:
                    print(f"Directly loaded state dictionary contains base_weights: {checkpoint['base_weights']}")
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # Verify model checkpoint to ensure key parameters exist
        validate_model_checkpoint(model)
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise
    
    # Set to evaluation mode
    model.eval()
    
    # Output model's base_weights information
    if hasattr(model, 'base_weights'):
        weights = model.base_weights.detach().cpu().numpy()
        print(f"Model's current feature weights: {weights}")
        print(f"Weight type: {type(model.base_weights)}")
        if isinstance(model.base_weights, nn.Parameter):
            print("Feature weights are set as trainable parameters")
        else:
            print("Feature weights are non-trainable parameters (buffer)")
    else:
        print("Model does not contain base_weights parameter")
    
    # Get validation data loader
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        subset_ratio=0.2,  # Use a portion of the data for quick evaluation
        seed=42
    )
    
    return model, val_loader


def validate_model_checkpoint(model):
    """Validate model parameter integrity"""
    # Check key parameters
    if not hasattr(model, 'base_weights'):
        print("Warning: Model is missing base_weights parameter")
        
    if hasattr(model, 'temperature') and model.temperature is None:
        print("Warning: Model's temperature parameter is None")
        
    if hasattr(model, 'class_bias') and model.class_bias is None:
        print("Warning: Model's class_bias parameter is None")
    
    # Check model parameter status
    if hasattr(model, 'get_parameter_status'):
        print("MODEL PARAMETER STATUS:", model.get_parameter_status())
    
    # Check if model is in evaluation mode
    if hasattr(model, 'eval_mode_active'):
        print(f"Model evaluation mode: {model.eval_mode_active}")
    
    return True


def visualize_results(results, save_path='feature_importance_results.png'):
    """Visualize performance comparison of different feature combinations"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract scheme names and corresponding metrics
    schemes = list(results.keys())
    auc_scores = []
    balanced_acc = []
    real_acc = []
    synthetic_acc = []
    detection_bias = []  # Detection bias metric
    
    for scheme in schemes:
        metrics = results[scheme]['metrics']
        # Get AUC
        auc_scores.append(metrics.get('auc', 0.0))
        
        # Get balanced accuracy, compatible with different key names
        if 'balanced_accuracy' in metrics:
            balanced_acc.append(metrics['balanced_accuracy'])
        elif 'balanced_acc' in metrics:
            balanced_acc.append(metrics['balanced_acc'])
        else:
            balanced_acc.append(0.0)
            
        # Get real image accuracy
        real_acc.append(metrics.get('real_class_accuracy', 0.0))
        
        # Get synthetic image accuracy
        synthetic_acc.append(metrics.get('synthetic_class_accuracy', 0.0))
        
        # Calculate detection bias
        bias = metrics.get('synthetic_class_accuracy', 0.0) - metrics.get('real_class_accuracy', 0.0)
        detection_bias.append(bias)
    
    # Create multi-subplot layout to accommodate more content
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))
    
    # Set bar chart positions
    x = np.arange(len(schemes))
    width = 0.2  # Reduce bar width to fit more metrics
    
    # Upper part: Main performance metrics
    rects1 = ax1.bar(x - width*1.5, auc_scores, width, label='AUC', color='royalblue')
    rects2 = ax1.bar(x - width*0.5, balanced_acc, width, label='Balanced Accuracy', color='forestgreen')
    rects3 = ax1.bar(x + width*0.5, real_acc, width, label='Real Image Accuracy', color='darkorange')
    rects4 = ax1.bar(x + width*1.5, synthetic_acc, width, label='Synthetic Image Accuracy', color='crimson')
    
    # Lower part: Detection bias
    rects5 = ax2.bar(x, detection_bias, width*1.5, label='Detection Bias (Synth-Real)', color='purple')
    
    # Add a zero line for bias reference
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    
    # Add labels and legend - Upper part
    ax1.set_xlabel('')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    ax1.set_ylim(0, 1.0)  # Ensure y-axis from 0 to 1
    
    # Add labels and legend - Lower part
    ax2.set_xlabel('Feature Combinations')
    ax2.set_ylabel('Detection Bias (Positive = Synthetic Bias)')
    ax2.set_title('Detection Bias by Feature Combination')
    ax2.set_xticks(x)
    ax2.set_xticklabels(schemes, rotation=45, ha='right')
    ax2.legend()
    
    # Set appropriate ylim to highlight detection bias
    max_bias = max(abs(max(detection_bias)), abs(min(detection_bias))) if detection_bias else 0.5
    ax2.set_ylim(-max_bias*1.2, max_bias*1.2)
    
    # Add labels to each bar
    def add_labels(rects, ax, format_str='{:.2f}'):
        """Add numeric labels to each bar"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
    
    add_labels(rects1, ax1)
    add_labels(rects2, ax1)
    add_labels(rects3, ax1)
    add_labels(rects4, ax1)
    add_labels(rects5, ax2)
    
    # Add scheme weight annotations
    for i, scheme in enumerate(schemes):
        weights = results[scheme]['weights']
        weight_str = '[' + ', '.join([f'{w:.2f}' for w in weights]) + ']'
        ax2.annotate(weight_str, 
                    xy=(i, -max_bias*1.05),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='top', rotation=90, fontsize=7,
                    alpha=0.7)
    
    # Display chart
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance analysis results saved to {save_path}")
    plt.close()


def main():
    """Main function: Load model, data, and analyze feature importance"""
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and data
    model, val_loader = load_model_and_data(args, device)
    
    # Print current model parameters
    if hasattr(model, 'class_bias'):
        print(f"Original class_bias: {model.class_bias}")
    if hasattr(model, 'real_boost_factor'):
        print(f"Original real_boost_factor: {model.real_boost_factor}")
    if hasattr(model, 'temperature'):
        print(f"Original temperature: {model.temperature}")
    
    # Analyze feature importance - without bias correction
    print("\n=== Analyzing Feature Importance (No Bias Correction) ===")
    results_no_correction = analyze_feature_importance(model, val_loader, device, apply_bias_correction=False)
    
    # Analyze feature importance - with bias correction
    print("\n=== Analyzing Feature Importance (With Bias Correction) ===")
    results_with_correction = analyze_feature_importance(model, val_loader, device, apply_bias_correction=True, adjust_temp=True)
    
    # Merge results
    results = {}
    results.update({f"{k}_NoBiasCorrection": v for k, v in results_no_correction.items()})
    results.update({f"{k}_WithBiasCorrection": v for k, v in results_with_correction.items() 
                   if v.get('bias_correction_applied', False)})
    
    # Visualize results
    visualize_results(results, save_path='feature_importance_results.png')
    
    print("Analysis complete, results saved to feature_importance_results.png")


if __name__ == '__main__':
    main()
