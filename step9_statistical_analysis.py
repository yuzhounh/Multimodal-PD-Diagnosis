import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')

def determine_variable_type(series):
    """确定变量类型"""
    unique_values = series.dropna().unique()
    
    # 如果只有两个唯一值，认为是二分类变量
    if len(unique_values) == 2:
        return 'Binary'
    # 如果唯一值少于10个且都是整数，认为是分类变量
    elif len(unique_values) < 10 and all(isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()) for x in unique_values):
        return 'Categorical'
    else:
        return 'Continuous'

def perform_statistical_test(data, group_col, feature_col, var_type):
    """执行统计检验"""
    group_0 = data[data[group_col] == 0][feature_col].dropna()
    group_1 = data[data[group_col] == 1][feature_col].dropna()
    
    if var_type == 'Binary' or var_type == 'Categorical':
        # 卡方检验
        contingency_table = pd.crosstab(data[feature_col], data[group_col])
        if contingency_table.min().min() >= 5:  # 期望频数≥5
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            test_method = 'Chi-square test'
        else:
            # 使用Fisher精确检验
            from scipy.stats import fisher_exact
            if contingency_table.shape == (2, 2):
                _, p_value = fisher_exact(contingency_table)
                test_method = 'Fisher exact test'
            else:
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                test_method = 'Chi-square test'
    else:
        # 连续变量：首先检验正态性
        _, p_norm_0 = stats.shapiro(group_0.sample(min(5000, len(group_0))))
        _, p_norm_1 = stats.shapiro(group_1.sample(min(5000, len(group_1))))
        
        if p_norm_0 > 0.05 and p_norm_1 > 0.05:
            # 正态分布，使用t检验
            _, p_value = ttest_ind(group_0, group_1)
            test_method = 'Independent t-test'
        else:
            # 非正态分布，使用Mann-Whitney U检验
            _, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
            test_method = 'Mann-Whitney U test'
    
    return test_method, p_value

def calculate_statistics(data, group_col, feature_col, var_type):
    """计算描述性统计"""
    overall_data = data[feature_col].dropna()
    group_0_data = data[data[group_col] == 0][feature_col].dropna()
    group_1_data = data[data[group_col] == 1][feature_col].dropna()
    
    if var_type == 'Continuous':
        # 连续变量：均值±标准差
        overall_stats = f"{overall_data.mean():.2f}±{overall_data.std():.2f}"
        group_0_stats = f"{group_0_data.mean():.2f}±{group_0_data.std():.2f}"
        group_1_stats = f"{group_1_data.mean():.2f}±{group_1_data.std():.2f}"
    else:
        # 分类变量：众数和比例
        overall_mode = overall_data.mode()[0] if len(overall_data.mode()) > 0 else 'N/A'
        group_0_mode = group_0_data.mode()[0] if len(group_0_data.mode()) > 0 else 'N/A'
        group_1_mode = group_1_data.mode()[0] if len(group_1_data.mode()) > 0 else 'N/A'
        
        overall_prop = (overall_data == overall_mode).mean() if overall_mode != 'N/A' else 0
        group_0_prop = (group_0_data == group_0_mode).mean() if group_0_mode != 'N/A' else 0
        group_1_prop = (group_1_data == group_1_mode).mean() if group_1_mode != 'N/A' else 0
        
        overall_stats = f"{overall_mode} ({overall_prop:.2f})"
        group_0_stats = f"{group_0_mode} ({group_0_prop:.2f})"
        group_1_stats = f"{group_1_mode} ({group_1_prop:.2f})"
    
    return overall_stats, group_0_stats, group_1_stats

def get_range_string(data, feature_col, var_type):
    """获取变量范围字符串"""
    feature_data = data[feature_col].dropna()
    
    if var_type == 'Continuous':
        min_val = feature_data.min()
        max_val = feature_data.max()
        return f"[{min_val:.2f}, {max_val:.2f}]"
    else:
        unique_vals = sorted(feature_data.unique())
        return "{" + ", ".join(map(str, unique_vals)) + "}"

def main():
    # 读取数据
    file_path = 'PPMI_7_propensity_score_matching.csv'
    data = pd.read_csv(file_path)
    
    # 读取特征名对应关系
    feature_names_df = pd.read_csv('PPMI_feature_mapping.csv')
    feature_to_abbr = dict(zip(feature_names_df['Feature Name'], feature_names_df['Abbreviation']))
    
    if 'COHORT' not in data.columns:
        print("警告：未找到COHORT列，请检查数据文件")
        return
    
    # 定义要分析的特征列
    feature_columns = [
        'age_at_visit', 'SEX', 'EDUCYRS', 'fampd_bin', 'BMI', 'LEDD', 
        'moca', 'MCI_testscores', 'quip', 'ess', 'rem', 'gds', 
        'stai_state', 'stai_trait', 'scopa', 'orthostasis', 'NHY', 
        'pigd', 'td_pigd', 'updrs1_score', 'updrs2_score', 'updrs3_score'
    ]
    
    # 过滤存在的列
    existing_columns = [col for col in feature_columns if col in data.columns]
    
    print(f"数据总数: {len(data)}")
    print(f"PD组 (Class_0): {len(data[data['COHORT'] == 0])}")
    print(f"NC组 (Class_1): {len(data[data['COHORT'] == 1])}")
    print(f"分析特征数: {len(existing_columns)}")
    
    # 存储结果
    results = []
    p_values = []
    
    # 添加样本大小行
    results.append({
        'Variable': 'Sample Size',
        'Range': '',
        'Overall': len(data),
        'Class_0': len(data[data['COHORT'] == 0]),
        'Class_1': len(data[data['COHORT'] == 1]),
        'Variable_Type': '',
        'Test_Method': '',
        'p_value': ''
    })
    
    # 对每个特征进行分析
    for feature in existing_columns:
        print(f"正在分析: {feature}")
        
        # 跳过缺失值过多的特征
        if data[feature].isnull().sum() / len(data) > 0.5:
            print(f"  跳过 {feature}：缺失值过多")
            continue
        
        # 确定变量类型
        var_type = determine_variable_type(data[feature])
        
        # 获取范围
        range_str = get_range_string(data, feature, var_type)
        
        # 计算描述性统计
        overall_stats, group_0_stats, group_1_stats = calculate_statistics(
            data, 'COHORT', feature, var_type
        )
        
        # 执行统计检验
        test_method, p_value = perform_statistical_test(
            data, 'COHORT', feature, var_type
        )
        
        # 使用特征缩写代替原始变量名
        feature_abbr = feature_to_abbr.get(feature, feature)
        
        results.append({
            'Variable': feature_abbr,  # 使用缩写名
            'Original_Variable': feature,  # 保留原始变量名
            'Range': range_str,
            'Overall': overall_stats,
            'Class_0': group_0_stats,
            'Class_1': group_1_stats,
            'Variable_Type': var_type,
            'Test_Method': test_method,
            'p_value': p_value
        })
        
        p_values.append(p_value)
    
    # FDR校正
    if p_values:
        rejected, p_adjusted = fdrcorrection(p_values, alpha=0.05)
        
        # 更新p值
        for i, result in enumerate(results[1:]):  # 跳过样本大小行
            if p_adjusted[i] < 0.001:
                result['p_value'] = '<0.001'
            else:
                result['p_value'] = f'{p_adjusted[i]:.3f}'
    
    # 保存结果
    results_df = pd.DataFrame(results)
    
    # 移除 Original_Variable 列以便最终输出更简洁
    if 'Original_Variable' in results_df.columns:
        results_df = results_df.drop('Original_Variable', axis=1)
    
    output_file = 'result_9_statistical_analysis.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print("\n前5行结果预览:")
    print(results_df.head())
    
    return results_df

if __name__ == "__main__":
    results = main()
