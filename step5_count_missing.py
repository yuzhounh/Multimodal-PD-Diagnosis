import pandas as pd
import numpy as np

# 读取CSV文件
print("读取 PPMI_4_PD_HC.csv...")

df = pd.read_csv('PPMI_4_PD_HC.csv')
print(f"数据形状: {df.shape}")

# 筛选COHORT列为0的行
cohort_0_df = df[df['COHORT'] == 0]
print(f"COHORT为0的行数: {len(cohort_0_df)}")

if len(cohort_0_df) == 0:
    print("没有找到COHORT为0的行")
else:
    # 计算每个特征的缺失值比例
    missing_percentages = (cohort_0_df.isna().sum() / len(cohort_0_df)) * 100
    
    # 按缺失值比例从大到小排序
    sorted_missing = missing_percentages.sort_values(ascending=False)
    
    # 输出结果
    print("\nCOHORT为0的行中，各特征缺失值比例（从大到小排序）:")
    for feature, percentage in sorted_missing.items():
        print(f"{feature}: {percentage:.2f}%")
        
    # 保存结果到CSV文件
    sorted_missing_df = pd.DataFrame({
        'Feature': sorted_missing.index,
        'Missing_Percentage': sorted_missing.values
    })
    sorted_missing_df.to_csv('result_5_missing_percentages.csv', index=False)
    print("\n结果已保存到 result_5_missing_percentages.csv")
