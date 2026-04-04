import pandas as pd
import re
import os
from pathlib import Path
from datetime import datetime

def extract_sub_ses_from_filename(filename):
    """
    从文件名中提取 sub 和 ses
    例如: sub-00312220130522_MNIPD25_three_views.png
    返回: (3122, 20130522)
    """
    # 使用正则表达式匹配 sub-数字+日期 的模式
    pattern = r'sub-(\d+)(\d{8})_'
    match = re.search(pattern, filename)
    
    if match:
        sub_str = match.group(1)
        ses_str = match.group(2)
        
        # 将sub转换为整数（去除前导零）
        sub = int(sub_str)
        # ses保持为字符串或转换为整数，根据CSV中的类型
        ses = int(ses_str)
        
        return (sub, ses)
    return None

def get_abnormal_files(directory_path):
    """
    获取目录中所有.png文件的 (sub, ses) 列表
    """
    abnormal_pairs = []
    
    # 如果传入的是目录路径
    if os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith('.png'):
                result = extract_sub_ses_from_filename(filename)
                if result:
                    abnormal_pairs.append(result)
    else:
        # 如果需要手动输入文件名列表
        print("请提供目录路径")
    
    return abnormal_pairs

def convert_date_to_yyyymmdd(date_str):
    """
    将日期字符串从 YYYY-MM-DD 格式转换为 YYYYMMDD 格式
    例如: '2012-02-21' -> 20120221
    """
    try:
        # 解析日期字符串
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        # 转换为 YYYYMMDD 格式
        return int(date_obj.strftime('%Y%m%d'))
    except ValueError:
        # 如果已经是数字格式，直接返回
        try:
            return int(date_str)
        except ValueError:
            return None

def remove_abnormal_data(csv_input_path, png_directory, csv_output_path):
    """
    主函数：读取CSV，删除异常数据行，保存结果，并检查PNG中存在但CSV中不存在的数据
    
    参数:
        csv_input_path: 输入CSV文件路径
        png_directory: 包含异常PNG文件的目录路径
        csv_output_path: 输出CSV文件路径
    """
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_input_path}")
    df = pd.read_csv(csv_input_path)
    print(f"原始数据行数: {len(df)}")
    
    # 获取所有异常文件的 (sub, ses)
    print(f"\n正在扫描目录: {png_directory}")
    abnormal_pairs = get_abnormal_files(png_directory)
    print(f"找到 {len(abnormal_pairs)} 个异常文件")
    
    if abnormal_pairs:
        print("\n异常数据 (sub, ses) - 全部输出:")
        for pair in abnormal_pairs:
            print(f"  {pair}")
    
    # 创建CSV中存在的 (sub, ses) 组合集合
    csv_pairs = set()
    for _, row in df.iterrows():
        csv_pairs.add((int(row['sub']), convert_date_to_yyyymmdd(row['ses'])))
    
    # 检查PNG中存在但CSV中不存在的 (sub, ses) 组合
    png_only_pairs = []
    for pair in abnormal_pairs:
        if pair not in csv_pairs:
            png_only_pairs.append(pair)
    
    if png_only_pairs:
        print(f"\n发现 {len(png_only_pairs)} 个存在于PNG但不存在于CSV中的 (sub, ses) 组合:")
        for pair in png_only_pairs:
            print(f"  {pair}")
    else:
        print("\n所有PNG文件对应的 (sub, ses) 组合都在CSV中存在")
    
    # 删除匹配的行
    initial_count = len(df)
    
    # 创建一个布尔掩码，标记需要删除的行
    # 需要将CSV中的日期格式转换为与PNG文件名匹配的格式
    mask = df.apply(lambda row: (
        int(row['sub']), 
        convert_date_to_yyyymmdd(row['ses'])
    ) in abnormal_pairs, axis=1)
    
    # 获取被删除的行，并输出其 COHORT 信息
    removed_rows = df[mask].copy()
    
    if len(removed_rows) > 0:
        print(f"\n删除的异常值详细信息 (共 {len(removed_rows)} 行):")
        print("-" * 80)
        for idx, row in removed_rows.iterrows():
            cohort_value = row['COHORT'] if 'COHORT' in row else 'N/A'
            ses_converted = convert_date_to_yyyymmdd(row['ses'])
            print(f"  sub: {int(row['sub'])}, ses: {ses_converted}, COHORT: {cohort_value}")
        print("-" * 80)
    
    # 删除匹配的行
    df_cleaned = df[~mask].copy()
    
    removed_count = initial_count - len(df_cleaned)
    print(f"\n删除了 {removed_count} 行数据")
    print(f"剩余数据行数: {len(df_cleaned)}")
    
    # 保存结果
    print(f"\n正在保存到: {csv_output_path}")
    df_cleaned.to_csv(csv_output_path, index=False)
    print("完成！")
    
    return df_cleaned

# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    csv_input = "PPMI_2_ROI_values.csv"
    png_dir = "abnormal_preprocessed_viz/"
    csv_output = "PPMI_3_remove_abnormal_data.csv"
    
    # 执行删除操作
    result_df = remove_abnormal_data(csv_input, png_dir, csv_output)
    
    # 可选：显示一些统计信息
    print("\n--- 数据摘要 ---")
    print(f"保留的数据包含 {result_df['sub'].nunique()} 个不同的受试者")
    print(f"保留的数据包含 {result_df['ses'].nunique()} 个不同的会话")
