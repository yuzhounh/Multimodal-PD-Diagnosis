import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
df = pd.read_csv('PPMI_5_key_features.csv')

# 创建一个日志文件记录填充信息
log_file_name = 'result_7_fill_missing.txt'
log_file = open(log_file_name, 'w', encoding='utf-8')
log_file.write("缺失值填充日志\n")
log_file.write("=" * 80 + "\n")

# 判断列是离散值还是连续值
def is_discrete(series):
    # 去除NaN后的唯一值数量
    unique_count = series.dropna().nunique()
    # 如果唯一值数量小于等于总数的20%或小于10个，认为是离散值
    return unique_count <= min(10, len(series.dropna()) * 0.2)

# 获取列的类型（离散或连续）
column_types = {}
for col in df.columns:
    if col not in ['sub', 'ses', 'COHORT', 'subgroup']:  # 跳过标识列
        column_types[col] = 'discrete' if is_discrete(df[col]) else 'continuous'

# 统计每列的缺失值数量和类型
log_file.write("\n列统计信息:\n")
log_file.write("=" * 80 + "\n")
log_file.write(f"{'列名':<40} {'缺失值数量':<15} {'类型':<10}\n")
log_file.write("-" * 80 + "\n")

# 统计每列的缺失值
nan_stats = df.isna().sum()
nan_stats = nan_stats[nan_stats > 0]  # 只保留有缺失值的列

# 计算总缺失值数量
total_nan_count = nan_stats.sum()

# 写入日志文件
for col in nan_stats.index:
    if col not in ['sub', 'ses', 'COHORT', 'subgroup']:
        col_type = column_types[col]
        log_file.write(f"{col:<40} {nan_stats[col]:<15} {col_type:<10}\n")

log_file.write("\n" + "=" * 80 + "\n\n")

# 打印到控制台
print("\n每列缺失值统计：")
print("=" * 50)
for col in nan_stats.index:
    if col not in ['sub', 'ses', 'COHORT', 'subgroup']:
        col_type = column_types[col]
        print(f"{col:<20} {nan_stats[col]:<10} {col_type}")


# 对每个缺失值进行填充
for index, row in df.iterrows():
    for col in df.columns:
        if pd.isna(row[col]) and col not in ['sub', 'ses', 'COHORT', 'subgroup']:
            # 获取当前行的COHORT值
            cohort_value = row['COHORT']
            
            # 找出同一COHORT的所有行
            same_cohort_rows = df[df['COHORT'] == cohort_value]
            
            # 获取该列的值（排除NaN）
            col_values = same_cohort_rows[col].dropna()
            
            # 根据列类型选择填充方法
            if column_types[col] == 'discrete':
                # 使用众数填充
                mode_result = stats.mode(col_values)
                fill_value = mode_result[0] if isinstance(mode_result[0], (np.ndarray, list)) else mode_result[0]
                fill_method = '众数'
            else:
                # 使用中值填充
                fill_value = col_values.median()
                fill_method = '中值'
            
            # 填充缺失值
            df.at[index, col] = fill_value
            
            # 记录填充信息
            log_file.write(f"行 {index}:\n")
            log_file.write(f"  sub: {row['sub']}, ses: {row['ses']}, COHORT: {row['COHORT']}, subgroup: {row['subgroup']}\n")
            log_file.write(f"  列: {col}, 类型: {column_types[col]}, 填充方法: {fill_method}, 填充值: {fill_value}\n")
            log_file.write("-" * 80 + "\n")

# 关闭日志文件
log_file.close()

# 保存填充后的数据
df.to_csv('PPMI_6_filled.csv', index=False)
print(f"填充完成，共处理了 {total_nan_count} 个缺失值。详细信息请查看 {log_file_name}")
