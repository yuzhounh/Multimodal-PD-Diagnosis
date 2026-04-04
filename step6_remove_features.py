import pandas as pd

# 读取原始特征文件
df = pd.read_csv('PPMI_4_PD_HC.csv')

# 需要删除的特征列表
features_to_remove = [
    'sym_posins', 'ageonset', 'sym_other', 'sym_brady', 'sym_rigid', 'sym_tremor', 'duration',
    'updrs4_score', 'NFL_CSF', 'DVS_BNT', 'DVT_FAS', 'DVZ_TMTB', 'DVT_CLCKDRAW', 'DVZ_TMTA',
    'MSEADLG', 'abeta', 'ptau', 'hemohi', 'asyn', 'tau', 'nfl_serum', 'con_putamen',
    'mean_striatum', 'lowput_expected', 'upsit_pctl', 'urate', 'APOE_e4'
]

# 检查哪些特征实际存在于数据集中
existing_features = [col for col in features_to_remove if col in df.columns]
if len(existing_features) < len(features_to_remove):
    missing_features = set(features_to_remove) - set(existing_features)
    print(f"警告: 以下特征在数据集中不存在: {', '.join(missing_features)}")

# 从数据框中删除这些特征
df_reserved = df.drop(columns=existing_features, errors='ignore')

# 保存结果
df_reserved.to_csv('PPMI_5_key_features.csv', index=False)

print(f"原始特征数量: {df.shape[1]}")
print(f"保留后特征数量: {df_reserved.shape[1]}")
print(f"删除的特征数量: {df.shape[1] - df_reserved.shape[1]}")
print("已将保留的特征保存到 PPMI_5_key_features.csv")
