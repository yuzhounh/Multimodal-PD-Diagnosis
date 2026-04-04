import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import optuna
import shap
from utils import *
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

class XGBoostModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.trial_count = 0
        # 三种阈值
        self.threshold_default = 0.5
        self.threshold_youden = None
        self.threshold_f1 = None
        
    def create_model(self, params):
        """根据参数创建模型"""
        (n_estimators, learning_rate, max_depth, subsample, colsample_bytree,
         reg_alpha, reg_lambda, gamma, min_child_weight, scale_pos_weight) = params
        
        model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'gamma': gamma,
            'min_child_weight': min_child_weight,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'verbose': False
        }
        
        return xgb.XGBClassifier(**model_params)
    
    def objective_function(self, trial, X, y, subjects, folds, study):
        """Optuna 贝叶斯优化的目标函数"""
        self.trial_count += 1
        
        try:
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 2, 6)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1.0, 10.0, log=True)
            gamma = trial.suggest_float('gamma', 0.0, 0.5)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
            scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, 5.0)
            
            params = [n_estimators, learning_rate, max_depth, subsample, colsample_bytree,
                     reg_alpha, reg_lambda, gamma, min_child_weight, scale_pos_weight]
            
            model = self.create_model(params)
            
            auc_scores = []
            for train_idx, val_idx in folds:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                y_val_prob = model.predict_proba(X_val_fold)[:, 1]
                auc_roc = roc_auc_score(y_val_fold, y_val_prob)
                auc_scores.append(auc_roc)
            
            return np.mean(auc_scores)
                
        except Exception as e:
            print(f"XGBoost 目标函数错误: {e}")
            return 0.0
    
    def optimize_hyperparameters(self, X, y, subjects, n_calls=100):
        """使用 Optuna 贝叶斯优化进行超参数调优"""
        print("开始 XGBoost 贝叶斯优化...")
        print("=" * 80)
        
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        self.trial_count = 0
        
        def callback(study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
                return
            
            current_trial = self.trial_count
            
            params_list = []
            params_list.append(f"n_estimators={trial.params['n_estimators']}")
            params_list.append(f"learning_rate={trial.params['learning_rate']:.4f}")
            params_list.append(f"max_depth={trial.params['max_depth']}")
            params_list.append(f"subsample={trial.params['subsample']:.4f}")
            params_list.append(f"colsample_bytree={trial.params['colsample_bytree']:.4f}")
            params_list.append(f"reg_alpha={trial.params['reg_alpha']:.4f}")
            params_list.append(f"reg_lambda={trial.params['reg_lambda']:.4f}")
            params_list.append(f"gamma={trial.params['gamma']:.4f}")
            params_list.append(f"min_child_weight={trial.params['min_child_weight']}")
            params_list.append(f"scale_pos_weight={trial.params['scale_pos_weight']:.4f}")
            
            params_str = ", ".join(params_list)
            
            print(f"Trial: {current_trial}/{n_calls}")
            print(f"参数: {params_str}")
            print(f"AUC-ROC: {trial.value:.6f}")
            print(f"最佳 AUC-ROC (Trial {study.best_trial.number + 1}): {study.best_value:.6f}")
            print()
        
        study.optimize(
            lambda trial: self.objective_function(trial, X, y, subjects, folds, study),
            n_trials=n_calls,
            callbacks=[callback]
        )
        
        print("=" * 80)
        print("搜索完成")
        
        best_trial = study.best_trial
        self.best_params = best_trial.params.copy()
        
        print(f"最优参数: {self.best_params}")
        print(f"最优 AUC-ROC: {best_trial.value:.6f}")
        
        return study
    
    def train_final_model(self, X, y, subjects):
        """在完整训练集上训练最终模型"""
        print("训练 XGBoost 最终模型...")
        
        self.model = self.create_model([
            self.best_params['n_estimators'], 
            self.best_params['learning_rate'],
            self.best_params['max_depth'], 
            self.best_params['subsample'], 
            self.best_params['colsample_bytree'],
            self.best_params['reg_alpha'], 
            self.best_params['reg_lambda'], 
            self.best_params['gamma'],
            self.best_params['min_child_weight'], 
            self.best_params['scale_pos_weight']
        ])
        
        self.model.fit(X, y, verbose=False)
        
        # 计算最优阈值
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        youden_thresholds = []
        f1_thresholds = []
        
        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            fold_model = self.create_model([
                self.best_params['n_estimators'], 
                self.best_params['learning_rate'],
                self.best_params['max_depth'], 
                self.best_params['subsample'], 
                self.best_params['colsample_bytree'],
                self.best_params['reg_alpha'], 
                self.best_params['reg_lambda'], 
                self.best_params['gamma'],
                self.best_params['min_child_weight'], 
                self.best_params['scale_pos_weight']
            ])
            fold_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # 计算两种最优阈值
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            
            youden_thresholds.append(youden_thresh)
            f1_thresholds.append(f1_thresh)
        
        # 计算中位数
        self.threshold_f1 = np.median(f1_thresholds)
        self.threshold_youden = np.median(youden_thresholds)
        
        print("\n" + "="*60)
        print("最优阈值计算结果对比:")
        print("="*60)
        print(f"1. 默认阈值:     {self.threshold_default:.4f}")
        print(f"2. F1-score:     {self.threshold_f1:.4f}")
        print(f"3. Youden 指数:  {self.threshold_youden:.4f}")
        print("="*60 + "\n")
    
    def get_cross_validation_performance(self, X, y, subjects):
        """获取交叉验证性能结果"""
        print("计算训练集交叉验证性能...")
        
        folds = subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=self.random_state)
        
        # 存储三种阈值的结果
        fold_results = {
            'default': [],
            'f1': [],
            'youden': []
        }
        
        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            fold_model = self.create_model([
                self.best_params['n_estimators'], 
                self.best_params['learning_rate'],
                self.best_params['max_depth'], 
                self.best_params['subsample'], 
                self.best_params['colsample_bytree'],
                self.best_params['reg_alpha'], 
                self.best_params['reg_lambda'], 
                self.best_params['gamma'],
                self.best_params['min_child_weight'], 
                self.best_params['scale_pos_weight']
            ])
            
            fold_model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # 计算三种阈值下的性能
            thresholds = {
                'default': self.threshold_default,
                'f1': self.threshold_f1,
                'youden': self.threshold_youden
            }
            
            for thresh_type, threshold in thresholds.items():
                y_pred = (y_val_prob >= threshold).astype(int)
                metrics = calculate_all_metrics(y_val_fold, y_pred, y_val_prob, threshold)
                fold_results[thresh_type].append(metrics)
        
        # 计算均值和标准差
        summary_results = {}
        
        for thresh_type in ['default', 'f1', 'youden']:
            metrics_df = pd.DataFrame(fold_results[thresh_type])
            means = metrics_df.mean()
            stds = metrics_df.std()
            
            summary_results[f'train_{thresh_type}_mean'] = means.to_dict()
            summary_results[f'train_{thresh_type}_std'] = stds.to_dict()
        
        return summary_results
    
    def evaluate_feature_importance(self, X, y, feature_names, dataset_name, save_dir):
        """评估特征重要性"""
        print("计算 XGBoost 特征重要性...")
        
        importances = {}
        
        # 1. XGBoost 内置特征重要性
        importance_gain = self.model.get_booster().get_score(importance_type='gain')
        gain_values = np.zeros(len(feature_names))
        for i, feature in enumerate(feature_names):
            if f'f{i}' in importance_gain:
                gain_values[i] = importance_gain[f'f{i}']
        importances['XGB_Gain'] = gain_values
        
        # 2. SHAP
        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        importances['SHAP'] = np.mean(np.abs(shap_values), axis=0)
        
        plot_shap_summary(shap_values, X_sample, feature_names, 'XGBoost', dataset_name, save_dir)
        
        # 3. 排列重要性
        perm_importance = get_permutation_importance(self.model, X, y, feature_names, self.random_state)
        importances['Permutation'] = perm_importance
        
        plot_feature_importance_comparison(importances, feature_names, 'XGBoost', dataset_name, save_dir)
        
        return importances

    def evaluate_test_set(self, X_test, y_test, dataset_name, save_dir):
        """评估测试集性能"""
        print("计算测试集性能...")
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 计算三种阈值下的性能
        test_performance = {}
        
        # 默认阈值
        y_pred_default = (y_prob >= self.threshold_default).astype(int)
        test_performance['test_default'] = calculate_all_metrics(y_test, y_pred_default, y_prob, self.threshold_default)
        
        # F1 阈值
        y_pred_f1 = (y_prob >= self.threshold_f1).astype(int)
        test_performance['test_f1'] = calculate_all_metrics(y_test, y_pred_f1, y_prob, self.threshold_f1)
        
        # Youden 阈值
        y_pred_youden = (y_prob >= self.threshold_youden).astype(int)
        test_performance['test_youden'] = calculate_all_metrics(y_test, y_pred_youden, y_prob, self.threshold_youden)
        
        plot_roc_pr_curves(y_test, y_prob, 'XGBoost', dataset_name, save_dir)
        
        return test_performance, y_prob

def load_roi_mapping(roi_mapping_file='AAL3v1_with_flags.csv'):
    """加载 ROI 名称映射文件，将 ROI_{i} 映射为对应的脑区名称"""
    roi_df = pd.read_csv(roi_mapping_file)
    # 创建映射字典: ROI_{ID} -> Name
    roi_mapping = {f'ROI_{row["ID"]}': row['Name'] for _, row in roi_df.iterrows()}
    return roi_mapping

def process_single_dataset(train_file, test_file, feature_mapping_file, n_calls=100):
    """处理单个数据集"""
    dataset_name = train_file.replace('_train.csv', '').replace('.csv', '')
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*80}\n")
    
    train_data, test_data, feature_name_mapping = load_data(train_file, test_file, feature_mapping_file)
    
    # 加载 ROI 名称映射
    roi_mapping = load_roi_mapping('AAL3v1_with_flags.csv')
    
    X_train, y_train, subjects_train = prepare_data(train_data)
    X_test, y_test, subjects_test = prepare_data(test_data)
    
    X_train = X_train.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test.values
    y_test = y_test.values
    
    # 构建特征名列表，先使用 feature_name_mapping，再对 ROI 特征进行映射
    feature_names = []
    for col in train_data.columns:
        if col not in ['PATNO', 'COHORT']:
            # 首先使用原有的特征名映射
            mapped_name = feature_name_mapping.get(col, col)
            # 如果是 ROI 特征，再使用 ROI 映射
            if mapped_name in roi_mapping:
                mapped_name = roi_mapping[mapped_name]
            feature_names.append(mapped_name)
    
    xgb_model = XGBoostModel(random_state=42)
    
    # 超参数优化
    xgb_model.optimize_hyperparameters(X_train, y_train, subjects_train, n_calls=n_calls)
    
    # 训练最终模型
    xgb_model.train_final_model(X_train, y_train, subjects_train)
    
    # 获取交叉验证性能
    train_performance = xgb_model.get_cross_validation_performance(X_train, y_train, subjects_train)
    
    # 特征重要性评估
    xgb_model.evaluate_feature_importance(X_train, y_train, feature_names, dataset_name, save_dir)
    
    # 测试集评估
    test_performance, y_prob = xgb_model.evaluate_test_set(X_test, y_test, dataset_name, save_dir)

    # 保存训练集性能
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv(f'{save_dir}/{dataset_name}_XGBoost_train_performance.csv')
    print(f"\n训练集性能结果已保存至: {save_dir}/{dataset_name}_XGBoost_train_performance.csv")
    
    # 保存测试集性能
    test_df = pd.DataFrame(test_performance)
    test_df.to_csv(f'{save_dir}/{dataset_name}_XGBoost_test_performance.csv')
    print(f"测试集性能结果已保存至: {save_dir}/{dataset_name}_XGBoost_test_performance.csv")
    
    # 保存模型
    with open(f'{save_dir}/{dataset_name}_XGBoost_model.pkl', 'wb') as f:
        pickle.dump({
            'model': xgb_model.model,
            'best_params': xgb_model.best_params,
            'threshold_default': xgb_model.threshold_default,
            'threshold_f1': xgb_model.threshold_f1,
            'threshold_youden': xgb_model.threshold_youden
        }, f)
    
    # 打印结果概览
    print("\n" + "="*80)
    print("XGBoost 模型训练集性能结果 (mean±std):")
    print("="*80)
    print(train_df.round(4))
    
    print("\n" + "="*80)
    print("XGBoost 模型测试集性能结果:")
    print("="*80)
    print(test_df.round(4))
    
    return train_performance, test_performance

def main():
    np.random.seed(42)
    
    # 定义数据集
    dataset_bases = [
        'PPMI_8_data_1_weak_feature_set',
        'PPMI_8_data_2_moderate_feature_set',
        'PPMI_8_data_3_strong_feature_set',
        'PPMI_8_data_4_MRI',
        'PPMI_8_data_5_MRI_with_weak_feature_set',
        'PPMI_8_data_6_MRI_with_moderate_feature_set',
        'PPMI_8_data_7_MRI_with_strong_feature_set'
    ]
    
    feature_mapping_file = 'PPMI_feature_mapping.csv'
    all_results = {}
    
    # 处理每个数据集
    for dataset_base in dataset_bases:
        train_file = f'{dataset_base}_train.csv'
        test_file = f'{dataset_base}_test.csv'
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_perf, test_perf = process_single_dataset(
                train_file, test_file, feature_mapping_file, n_calls=100
            )
            all_results[dataset_base] = {'train': train_perf, 'test': test_perf}
        else:
            print(f"警告: {train_file} 或 {test_file} 未找到,跳过...")
    
    # 保存汇总结果
    summary_results = []
    for dataset_name, results in all_results.items():
        test_perf = results['test']
        
        # 为每种阈值策略创建一行
        for strategy in ['default', 'f1', 'youden']:
            row = {
                'Dataset': dataset_name,
                'Strategy': strategy,
                **test_perf[f'test_{strategy}']
            }
            summary_results.append(row)
    
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/XGBoost_all_datasets_summary.csv', index=False)
    print(f"\n所有数据集汇总结果已保存至: results/XGBoost_all_datasets_summary.csv")

if __name__ == "__main__":
    main()