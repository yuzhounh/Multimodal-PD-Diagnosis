import pandas as pd
from datetime import datetime
import numpy as np

# 读取两个CSV文件
print("读取数据文件...")
df1 = pd.read_csv('PPMI_1_features.csv')
df2 = pd.read_csv('PPMI_ROI_values_MNI.csv')

# 将visit_date转换为datetime格式
df1['visit_date'] = pd.to_datetime(df1['visit_date'], format='%m/%Y')

# 将ses转换为datetime格式（假设格式为YYYYMMDD）
df2['ses'] = pd.to_datetime(df2['ses'], format='%Y%m%d')

# 存储匹配结果
matched_rows = []
date_info = []

print(f"开始匹配，共需处理 {len(df2)} 行...")

# 遍历第二个表格的每一行
for idx, row2 in df2.iterrows():
    sub = row2['sub']
    ses = row2['ses']
    
    # 在第一个表格中查找匹配的PATNO
    matched_df1 = df1[df1['PATNO'] == sub]
    
    if len(matched_df1) == 0:
        print(f"警告: sub={sub} 的MRI数据在精选表格中没有匹配到PATNO")
        continue
    
    # 计算时间差，找到最接近的日期
    matched_df1 = matched_df1.copy()
    matched_df1['time_diff'] = abs((matched_df1['visit_date'] - ses).dt.days)
    
    # 找到时间差最小的行
    min_diff_idx = matched_df1['time_diff'].idxmin()
    best_match = matched_df1.loc[min_diff_idx]
    
    # 合并两行数据
    combined_row = pd.concat([best_match, row2])
    matched_rows.append(combined_row)
    
    # 保存日期信息
    date_info.append({
        'sub': sub,
        'ses': ses,
        'visit_date': best_match['visit_date'],
        'EVENT_ID': best_match['EVENT_ID'],
        'YEAR': best_match['YEAR'],
        'date_diff_days': best_match['time_diff']
    })
    
    # if (idx + 1) % 100 == 0:
    #     print(f"已处理 {idx + 1}/{len(df2)} 行...")

# 创建结果DataFrame
print("生成结果文件...")
result_df = pd.DataFrame(matched_rows)

# 删除重复的列（如果有的话）
result_df = result_df.loc[:, ~result_df.columns.duplicated()]

# 保存第一个结果文件
result_df.to_csv('PPMI_2_ROI_values.csv', index=False)
print(f"已保存 PPMI_2_ROI_values.csv ({len(result_df)} 行)")

# 创建日期差异DataFrame
date_diff_df = pd.DataFrame(date_info)

# 保存第二个结果文件
date_diff_df.to_csv('PPMI_2_date_difference.csv', index=False)
print(f"已保存 PPMI_2_date_difference.csv ({len(date_diff_df)} 行)")

# 显示统计信息
print("\n匹配统计:")
print(f"第二个表格总行数: {len(df2)}")
print(f"成功匹配行数: {len(matched_rows)}")
print(f"未匹配行数: {len(df2) - len(matched_rows)}")
print(f"\n时间差统计 (天):")
print(date_diff_df['date_diff_days'].describe())

# 统计时间差分布
time_diff_stats = date_diff_df['date_diff_days']
total_matches = len(time_diff_stats)

# 计算各时间差范围的数量和比例
less_than_30 = (time_diff_stats < 30).sum()
less_than_60 = (time_diff_stats < 60).sum()

print(f"\n时间差分布统计:")
print(f"时间差 < 30天: {less_than_30} 行 ({less_than_30/total_matches*100:.2f}%)")
print(f"时间差 < 60天: {less_than_60} 行 ({less_than_60/total_matches*100:.2f}%)")
print(f"时间差 >= 60天: {total_matches - less_than_60} 行 ({(total_matches - less_than_60)/total_matches*100:.2f}%)")

print("\n完成!")
