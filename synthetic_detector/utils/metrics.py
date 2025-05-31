import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率 (用于计算AUC和AP)
        
    Returns:
        包含各种指标的字典
    """
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100  # 转为百分比
    metrics['precision'] = precision_score(y_true, y_pred, average='binary')
    metrics['recall'] = recall_score(y_true, y_pred, average='binary')
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['true_positive'] = tp
    
    # 假阳性率和假阴性率
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # 如果提供了概率，计算AUC和平均精度
    if y_prob is not None:
        # 处理极端情况
        if len(np.unique(y_true)) > 1:  # 确保有至少两个类别
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['ap'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auc'] = 0.5  # 只有一个类别时的默认值
            metrics['ap'] = np.mean(y_true)  # 正类比例
    
    return metrics

def compute_class_wise_accuracy(y_true, y_pred):
    """
    计算每个类别的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        每个类别的准确率字典
    """
    classes = np.unique(y_true)
    class_acc = {}
    
    for c in classes:
        # 获取当前类别的样本索引
        idx = (y_true == c)
        if np.sum(idx) > 0:
            # 计算该类别的准确率
            acc = np.mean(y_pred[idx] == y_true[idx]) * 100
            class_acc[f'class_{c}'] = acc
    
    return class_acc
