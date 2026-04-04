import pandas as pd


# 读取风险因素文件以获取需要保留的特征
risk_factors = pd.read_csv('PPMI_risk_factors.csv')

# 筛选 Flag = 1 的特征
features_to_keep = risk_factors[risk_factors['Flag'] == 1]['Feature'].tolist()

# 读取 PPMI 数据
ppmi_data = pd.read_excel('PPMI_Curated_Data_Cut_Public_20250321.xlsx')

# 仅保留 features_to_keep 中指定的列
ppmi_filtered = ppmi_data[features_to_keep]

# 查找并显示 PATNO=127741, visit_date=12/2022 的 BMI 值
target_row = ppmi_filtered[(ppmi_filtered['PATNO'] == 127741) & (ppmi_filtered['visit_date'] == '12/2022')]


# 输出 PATNO=127741, visit_date=12/2022 的 BMI 值，并将其设为空
if not target_row.empty:
    bmi_value = target_row['BMI'].values[0]
    print(f"PATNO=127741, visit_date=12/2022, BMI={bmi_value}")
    
    # 将 BMI 值设为 NaN（空值）
    ppmi_filtered.loc[(ppmi_filtered['PATNO'] == 127741) & (ppmi_filtered['visit_date'] == '12/2022'), 'BMI'] = pd.NA
    print(f"已将 PATNO=127741, visit_date=12/2022 的 BMI 值设为空")
else:
    print("警告：未找到 PATNO=127741, visit_date=12/2022 的行")


# 重新编码 fampd_bin 和 td_pigd 列：{1, 2} --> {1, 0}
if 'fampd_bin' in ppmi_filtered.columns:
    ppmi_filtered.loc[:, 'fampd_bin'] = ppmi_filtered['fampd_bin'].replace({1: 1, 2: 0})
    print("已重新编码 fampd_bin: {1, 2} --> {1, 0}")

if 'td_pigd' in ppmi_filtered.columns:
    ppmi_filtered.loc[:, 'td_pigd'] = ppmi_filtered['td_pigd'].replace({1: 1, 2: 0})
    print("已重新编码 td_pigd: {1, 2} --> {1, 0}")


# 将筛选后的数据保存到新的 CSV 文件
ppmi_filtered.to_csv('PPMI_1_features.csv', index=False)

print(f"筛选后的数据已保存到 PPMI_1_features.csv")
print(f"从 {ppmi_data.shape[1]} 个总特征中保留了 {len(features_to_keep)} 个特征")
