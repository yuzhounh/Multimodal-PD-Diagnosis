import pandas as pd

# 读取保留特征后的数据文件
print("读取 PPMI_3_remove_abnormal_data.csv...")
df = pd.read_csv('PPMI_3_remove_abnormal_data.csv')
print(f"原始数据形状: {df.shape}")

# 筛选 COHORT 列为1或2的行
pd_hc_df = df[df['COHORT'].isin([1, 2])].copy()
print(f"COHORT为1的行数: {len(pd_hc_df[pd_hc_df['COHORT'] == 1])}")
print(f"COHORT为2的行数: {len(pd_hc_df[pd_hc_df['COHORT'] == 2])}")
print(f"筛选后数据形状: {pd_hc_df.shape}")

# 转换 COHORT 列: {1,2}->{1,0}
pd_hc_df['COHORT'] = pd_hc_df['COHORT'].map({1: 1, 2: 0})
print("COHORT列已转换: 1->1, 2->0")
print(f"转换后COHORT为1的行数: {len(pd_hc_df[pd_hc_df['COHORT'] == 1])}")
print(f"转换后COHORT为0的行数: {len(pd_hc_df[pd_hc_df['COHORT'] == 0])}")

# 保存结果
pd_hc_df.to_csv('PPMI_4_PD_HC.csv', index=False)
print("已将COHORT为1和2的行保存到 PPMI_4_PD_HC.csv")
