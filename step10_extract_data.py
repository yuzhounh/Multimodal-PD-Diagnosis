import pandas as pd

# 读取特征关系数据
feature_relation_df = pd.read_csv('PPMI_feature_strength.csv')

# 读取主数据
data_df = pd.read_csv('PPMI_7_propensity_score_matching.csv')

# 提取strength为1的特征
strength_1_feature_set = feature_relation_df[feature_relation_df['strength'] == 1]['feature name'].tolist()
# 提取strength为1或2的特征
strength_1_2_feature_set = feature_relation_df[feature_relation_df['strength'].isin([1, 2])]['feature name'].tolist()
# 提取strength为1或2或3的特征
strength_1_2_3_feature_set = feature_relation_df[feature_relation_df['strength'].isin([1, 2, 3])]['feature name'].tolist()

# 读取AAL_with_flags.csv文件
aal_df = pd.read_csv('AAL3v1_with_flags.csv')

# 筛选Flag >= threshold 且 Value 不为空的行，提取对应的Value
threshold = 6
selected_values = aal_df[(aal_df['Flag'] >= threshold) & (aal_df['Value'].notna())]['Value'].tolist()

# 构建ROI列名列表
roi_columns = [f'ROI_{int(value)}' for value in selected_values]

# 1. strength为1的特征 + COHORT
selected_columns_1 = ['PATNO'] + strength_1_feature_set + ['COHORT']
data_df[selected_columns_1].to_csv('PPMI_8_data_1_weak_feature_set.csv', index=False)

# 2. strength为1,2的特征 + COHORT
selected_columns_2 = ['PATNO'] + strength_1_2_feature_set + ['COHORT']
data_df[selected_columns_2].to_csv('PPMI_8_data_2_moderate_feature_set.csv', index=False)

# 3. strength为1,2,3的特征 + COHORT
selected_columns_3 = ['PATNO'] + strength_1_2_3_feature_set + ['COHORT']
data_df[selected_columns_3].to_csv('PPMI_8_data_3_strong_feature_set.csv', index=False)

# 4. 所有ROI特征
selected_columns_4 = ['PATNO'] + roi_columns + ['COHORT']
data_df[selected_columns_4].to_csv('PPMI_8_data_4_MRI.csv', index=False)

# 5. 所有ROI特征 + strength为1的特征 + COHORT
selected_columns_5 = ['PATNO'] + roi_columns + strength_1_feature_set + ['COHORT']
data_df[selected_columns_5].to_csv('PPMI_8_data_5_MRI_with_weak_feature_set.csv', index=False)

# 6. 所有ROI特征 + strength为1,2的特征 + COHORT
selected_columns_6 = ['PATNO'] + roi_columns + strength_1_2_feature_set + ['COHORT']
data_df[selected_columns_6].to_csv('PPMI_8_data_6_MRI_with_moderate_feature_set.csv', index=False)

# 7. 所有ROI特征 + strength为1,2,3的特征 + COHORT
selected_columns_7 = ['PATNO'] + roi_columns + strength_1_2_3_feature_set + ['COHORT']
data_df[selected_columns_7].to_csv('PPMI_8_data_7_MRI_with_strong_feature_set.csv', index=False)

print("数据提取完成，已保存到对应的CSV文件中。")
