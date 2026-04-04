import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix, roc_curve, precision_recall_curve, cohen_kappa_score)
from sklearn.inspection import permutation_importance
import shap
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def load_data(train_file, test_file, feature_mapping_file):
    """加载训练集、测试集和特征映射文件"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    feature_mapping = pd.read_csv(feature_mapping_file)
    
    # 创建特征名称映射字典
    feature_name_mapping = dict(zip(feature_mapping['Feature Name'], feature_mapping['Abbreviation']))
    
    return train_data, test_data, feature_name_mapping

def prepare_data(data, subject_col='PATNO', label_col='COHORT'):
    """准备数据，分离特征和标签"""
    # 移除受试者ID列
    X = data.drop([subject_col, label_col], axis=1)
    y = data[label_col]
    subjects = data[subject_col]
    
    return X, y, subjects

def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """
    受试者级别交叉验证 - 采用"降维、分割、升维"逻辑。
    """
    # 降维到受试者级别
    subject_labels_series = pd.Series(y, index=subjects).groupby(level=0).max()
    unique_subjects = subject_labels_series.index.values
    subject_labels = subject_labels_series.values
    
    # 初始化分层交叉验证
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    
    # 在受试者级别完成分割
    for train_subject_idx, val_subject_idx in sgkf.split(unique_subjects, subject_labels, groups=unique_subjects):
        train_subjects_fold = unique_subjects[train_subject_idx]
        val_subjects_fold = unique_subjects[val_subject_idx]
        
        # 扩展回样本级别
        train_sample_idx = np.where(np.isin(subjects, train_subjects_fold))[0]
        val_sample_idx = np.where(np.isin(subjects, val_subjects_fold))[0]
        
        folds.append((train_sample_idx, val_sample_idx))
    
    return folds

def f1_score_threshold(y_true, y_prob):
    """
    计算最大F1分数对应的最优阈值。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    f1_scores = f1_scores[:-1]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def youden_threshold(y_true, y_prob):
    """基于约登指数计算最优阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_scores = tpr - fpr  # 约登J统计量
    optimal_idx = np.argmax(youden_scores)
    return thresholds[optimal_idx]

def calculate_all_metrics(y_true, y_pred, y_prob, threshold=None):
    """计算所有评估指标"""
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 按指定顺序返回结果
    result = {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Optimal Threshold': threshold if threshold is not None else 0.5,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-score': f1,
        "Cohen's Kappa": cohen_kappa
    }
    
    return result

def plot_roc_pr_curves(y_true, y_prob, model_name, save_prefix="5_", save_dir="."):
    """绘制ROC曲线和PR曲线"""
    # 如果保存目录不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.4f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    ax2.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.4f})', linewidth=2)
    ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline ({np.mean(y_true):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{model_name} Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_{model_name}_roc_pr_curves.pdf'), bbox_inches='tight')
    plt.close()

    # 保存数据
    predictions_data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    predictions_data.to_csv(os.path.join(save_dir, f'{save_prefix}_{model_name}_predictions.csv'), index=False)

def plot_feature_importance_comparison(importances_dict, feature_names, model_name, save_prefix="5_", save_dir="."):
    """绘制特征重要性对比图"""
    # 如果保存目录不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    
    n_methods = len(importances_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, importance_values) in enumerate(importances_dict.items()):
        # 获取排序索引
        sorted_idx = np.argsort(np.abs(importance_values))
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = importance_values[sorted_idx]
        
        # 绘制水平条形图
        bars = axes[idx].barh(range(len(sorted_values)), sorted_values)
        axes[idx].set_yticks(range(len(sorted_values)))
        axes[idx].set_yticklabels(sorted_features)
        axes[idx].set_xlabel('Importance')
        axes[idx].set_title(f'{model_name} - {method_name}')
        axes[idx].grid(True, alpha=0.3)
        
        # 负值使用不同颜色
        for i, bar in enumerate(bars):
            if sorted_values[i] < 0:
                bar.set_color('#FF0051')  # 255, 0, 81
            else:
                bar.set_color('#008BFB')  # 0, 139, 251
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_{model_name}_feature_importance.pdf'), bbox_inches='tight')
    plt.close()
    
    # 保存数据
    importance_df = pd.DataFrame(importances_dict, index=feature_names)
    importance_df.to_csv(os.path.join(save_dir, f'{save_prefix}_{model_name}_feature_importance.csv'))

def plot_shap_summary(shap_values, X, feature_names, model_name, save_prefix="5_", save_dir="."):
    """绘制SHAP摘要图"""
    # 如果保存目录不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    
    # 条形图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", max_display=10, show=False)
    # 将SHAP柱状图的X轴标签汉化为"SHAP值"
    ax_bar = plt.gca()
    ax_bar.set_xlabel('SHAP值')
    plt.title(f'{model_name} - SHAP Feature Importance (Bar)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_{model_name}_shap_bar.pdf'), bbox_inches='tight')
    plt.close()
    
    # 摘要图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=10, show=False)
    ax_summary = plt.gca()
    ax_summary.set_xlabel('SHAP值')
    fig = plt.gcf()
    for cax in fig.axes:
        if cax is ax_summary:
            continue
        if cax.get_ylabel() == 'Feature value':
            cax.set_ylabel('特征值')
        if cax.get_xlabel() == 'Feature value':
            cax.set_xlabel('特征值')
        
        # 处理y轴刻度
        yticks = cax.get_yticks()
        yticklabels = [label.get_text() for label in cax.get_yticklabels()]
        new_yticklabels = []
        for text in yticklabels:
            if text == 'Low':
                new_yticklabels.append('低')
            elif text == 'High':
                new_yticklabels.append('高')
            else:
                new_yticklabels.append(text)
        if new_yticklabels:
            cax.set_yticks(yticks)
            cax.set_yticklabels(new_yticklabels)
        
        # 处理x轴刻度
        xticks = cax.get_xticks()
        xticklabels = [label.get_text() for label in cax.get_xticklabels()]
        new_xticklabels = []
        for text in xticklabels:
            if text == 'Low':
                new_xticklabels.append('低')
            elif text == 'High':
                new_xticklabels.append('高')
            else:
                new_xticklabels.append(text)
        if new_xticklabels:
            cax.set_xticks(xticks)
            cax.set_xticklabels(new_xticklabels)
    
    plt.title(f'{model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_{model_name}_shap_summary.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{save_prefix}_{model_name}_shap_summary.svg'), bbox_inches='tight')
    plt.close()
    
    # 保存SHAP值数据
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(save_dir, f'{save_prefix}_{model_name}_shap_values.csv'), index=False)

def get_permutation_importance(model, X, y, feature_names, random_state=42):
    """计算排列重要性"""
    perm_importance = permutation_importance(model, X, y, 
                                           n_repeats=10, 
                                           random_state=random_state, 
                                           scoring='roc_auc')
    return perm_importance.importances_mean

def print_results(model_name, metrics, optimal_threshold):
    """打印结果"""
    print(f"\n{'='*50}")
    print(f"{model_name} Model Results")
    print(f"{'='*50}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['Confusion Matrix'])

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value}")
