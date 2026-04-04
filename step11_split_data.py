import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_subject_proxy_labels(y, subjects):
    """
    创建被试级别的代理标签（任一样本为1，则被试为1的规则）
    
    Parameters:
    -----------
    y : array-like
        样本级别的标签
    subjects : array-like  
        样本对应的被试ID
        
    Returns:
    --------
    y_proxy : array
        样本级别的代理标签（每个样本使用其被试的代理标签）
    subject_labels_map : dict
        被试到代理标签的映射
    """
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    
    # 创建被试标签映射：每个被试使用"任一为1则为1"的规则
    subject_labels_map = pd.Series(y, index=subjects).groupby(level=0).max()
    
    # 为每个样本分配其被试的代理标签
    y_proxy = pd.Series(subjects).map(subject_labels_map).values
    
    return y_proxy, subject_labels_map.to_dict()


def print_split_statistics(train_data, test_data, subject_col='PATNO', label_col='COHORT'):
    """
    打印数据划分统计信息
    """
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    # 样本级别统计
    print(f"\n样本级别标签分布:")
    print(f"训练集标签分布: {train_data[label_col].value_counts().to_dict()}")
    print(f"测试集标签分布: {test_data[label_col].value_counts().to_dict()}")

    train_class_1_ratio = train_data[label_col].value_counts(normalize=True).get(1, 0)
    test_class_1_ratio = test_data[label_col].value_counts(normalize=True).get(1, 0)
    difference = train_class_1_ratio - test_class_1_ratio
    
    print(f"训练集中类别1的比例: {train_class_1_ratio:.4f}")
    print(f"测试集中类别1的比例: {test_class_1_ratio:.4f}")
    print(f"比例差异: {difference:.4f}")
    
    # 被试级别统计
    train_subjects = train_data[subject_col].unique()
    test_subjects = test_data[subject_col].unique()
    
    print(f"\n训练集被试数: {len(train_subjects)}")
    print(f"测试集被试数: {len(test_subjects)}")
    
    # 检查重叠
    overlapping_subjects = set(train_subjects) & set(test_subjects)
    if overlapping_subjects:
        print(f"警告: 训练集和测试集之间存在被试重叠: {overlapping_subjects}")
    else:
        print("训练集和测试集之间无被试重叠。")
    
    # 被试级别标签分布
    _, train_subject_labels_map = create_subject_proxy_labels(
        train_data[label_col].values, train_data[subject_col].values
    )
    _, test_subject_labels_map = create_subject_proxy_labels(
        test_data[label_col].values, test_data[subject_col].values
    )
    
    train_subject_labels = pd.Series(list(train_subject_labels_map.values()))
    test_subject_labels = pd.Series(list(test_subject_labels_map.values()))
    
    print(f"\n被试级别标签分布（任一为1则为1规则）:")
    print(f"训练集被试标签分布: {train_subject_labels.value_counts().to_dict()}")
    print(f"测试集被试标签分布: {test_subject_labels.value_counts().to_dict()}")


def stratified_group_split(data, test_size=0.3, random_state=42, 
                          subject_col='PATNO', label_col='COHORT'):
    """
    使用被试级别的分层抽样划分数据
    确保同一被试的数据要么在训练集，要么在测试集
    """
    # 创建代理标签
    y_proxy, subject_labels_map = create_subject_proxy_labels(
        data[label_col].values, data[subject_col].values
    )
    
    # 获取唯一被试及其对应的代理标签
    unique_subjects = np.array(list(subject_labels_map.keys()))
    subject_proxy_labels = np.array(list(subject_labels_map.values()))
    
    # 在被试级别进行分层划分
    train_subjects, test_subjects = train_test_split(
        unique_subjects, 
        test_size=test_size, 
        stratify=subject_proxy_labels, 
        random_state=random_state
    )
    
    # 根据被试划分数据
    train_data = data[data[subject_col].isin(train_subjects)]
    test_data = data[data[subject_col].isin(test_subjects)]
    
    return train_data, test_data


def process_single_file(input_file, test_size=0.3, random_state=42):
    """
    处理单个文件并保存划分结果
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {input_file}")
    print(f"{'='*60}")
    
    # 读取数据
    data = pd.read_csv(input_file)
    print(f"原始数据形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")
    
    # 划分数据
    train_data, test_data = stratified_group_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 打印统计信息
    print_split_statistics(train_data, test_data)
    
    # 生成输出文件名
    base_name = input_file.replace('.csv', '')
    train_file = f"{base_name}_train.csv"
    test_file = f"{base_name}_test.csv"
    
    # 保存文件
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"\n已保存:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    
    return train_file, test_file


def main():
    """
    主函数：处理所有7个数据集
    """
    # 定义所有数据集文件名
    dataset_files = [
        'PPMI_8_data_1_weak_feature_set.csv',
        'PPMI_8_data_2_moderate_feature_set.csv',
        'PPMI_8_data_3_strong_feature_set.csv',
        'PPMI_8_data_4_MRI.csv',
        'PPMI_8_data_5_MRI_with_weak_feature_set.csv',
        'PPMI_8_data_6_MRI_with_moderate_feature_set.csv',
        'PPMI_8_data_7_MRI_with_strong_feature_set.csv'
    ]
    
    print("开始处理所有数据集...")
    print(f"训练集:测试集比例 = 7:3")
    print(f"随机种子 = 42")
    
    # 存储所有结果
    results = []
    
    # 处理每个文件
    for input_file in dataset_files:
        try:
            train_file, test_file = process_single_file(
                input_file, 
                test_size=0.3, 
                random_state=42
            )
            results.append({
                'input': input_file,
                'train': train_file,
                'test': test_file,
                'status': 'Success'
            })
        except Exception as e:
            print(f"\n错误: 处理 {input_file} 时出错")
            print(f"错误信息: {str(e)}")
            results.append({
                'input': input_file,
                'train': None,
                'test': None,
                'status': f'Failed: {str(e)}'
            })
    
    # 打印总结
    print(f"\n{'='*60}")
    print("处理完成！总结:")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['input']}")
        if result['status'] == 'Success':
            print(f"   ✓ {result['train']}")
            print(f"   ✓ {result['test']}")
        else:
            print(f"   ✗ {result['status']}")
    
    # 保存处理结果摘要
    results_df = pd.DataFrame(results)
    outfile = 'result_11_data_split_summary.csv'
    results_df.to_csv(outfile, index=False)
    print(f"\n处理摘要已保存至: {outfile}")


if __name__ == "__main__":
    main()