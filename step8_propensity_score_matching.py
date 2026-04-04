import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
print("Reading data...")
df = pd.read_csv('PPMI_6_filled.csv')

# 检查数据
print(f"Total samples: {len(df)}")
print(f"PD samples: {sum(df['COHORT'] == 1)}")
print(f"NC samples: {sum(df['COHORT'] == 0)}")

# 提取匹配变量和目标变量
match_vars = ['age_at_visit', 'SEX', 'EDUCYRS']  # 假设这些是列名，根据实际数据调整
X = df[match_vars]
y = (df['COHORT'] == 1).astype(int)  # 1 for PD, 0 for NC

# 标准化数值特征
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled['age_at_visit'] = scaler.fit_transform(X[['age_at_visit']])
X_scaled['EDUCYRS'] = scaler.fit_transform(X[['EDUCYRS']])

# 使用逻辑回归计算倾向性得分
print("Calculating propensity scores...")
logistic = LogisticRegression(random_state=42)
logistic.fit(X_scaled, y)
df['propensity_score'] = logistic.predict_proba(X_scaled)[:, 1]

# 分离PD和NC样本
pd_samples = df[df['COHORT'] == 1].copy()
nc_samples = df[df['COHORT'] == 0].copy()

print(f"Before matching: {len(pd_samples)} PD samples, {len(nc_samples)} NC samples")

# 为每个NC样本找到最匹配的PD样本
print("Performing propensity score matching...")
matched_indices = []
used_pd_indices = set()

for nc_idx, nc_row in nc_samples.iterrows():
    # 计算所有PD样本与当前NC样本的倾向性得分差异
    pd_samples['score_diff'] = abs(pd_samples['propensity_score'] - nc_row['propensity_score'])
    
    # 按差异排序并找到最匹配的未使用PD样本
    sorted_pd = pd_samples.sort_values('score_diff')
    
    for pd_idx, pd_row in sorted_pd.iterrows():
        if pd_idx not in used_pd_indices:
            matched_indices.append(pd_idx)
            used_pd_indices.add(pd_idx)
            break

# 创建匹配后的数据集
matched_pd_samples = df.loc[matched_indices]
matched_dataset = pd.concat([matched_pd_samples, nc_samples])

print(f"After matching: {len(matched_pd_samples)} PD samples, {len(nc_samples)} NC samples")

# 评估匹配效果
print("Evaluating matching quality...")
for var in match_vars:
    before_pd_mean = pd_samples[var].mean()
    before_nc_mean = nc_samples[var].mean()
    after_pd_mean = matched_pd_samples[var].mean()
    
    before_diff = abs(before_pd_mean - before_nc_mean)
    after_diff = abs(after_pd_mean - before_nc_mean)
    
    print(f"{var}: Before matching diff = {before_diff:.4f}, After matching diff = {after_diff:.4f}")

# 可视化匹配前后的倾向性得分分布
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 第一个子图
ax1 = axes[0]
sns.histplot(pd_samples['propensity_score'], color='red', alpha=0.5, label='PD', ax=ax1)
sns.histplot(nc_samples['propensity_score'], color='blue', alpha=0.5, label='HC', ax=ax1)
# ax1.set_title('匹配前')
ax1.set_xlabel('倾向性评分')
ax1.set_ylabel('计数')
ax1.legend(loc='upper left')
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

# 第二个子图
ax2 = axes[1]
sns.histplot(matched_pd_samples['propensity_score'], color='red', alpha=0.5, label='PD', ax=ax2)
sns.histplot(nc_samples['propensity_score'], color='blue', alpha=0.5, label='HC', ax=ax2)
# ax2.set_title('匹配后')
ax2.set_xlabel('倾向性评分')
ax2.set_ylabel('计数')
ax2.legend(loc='upper left')
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

plt.tight_layout()
# plt.savefig(r'result_8_propensity_score_matching.pdf')
# plt.savefig(r'result_8_propensity_score_matching.tif', dpi=300, format='tif')
plt.savefig(r'result_8_propensity_score_matching.svg', format='svg')
# plt.show()
plt.close()

# 保存匹配后的结果
matched_dataset.to_csv('PPMI_7_propensity_score_matching.csv', index=False)
print("Matched dataset saved to PPMI_7_propensity_score_matching.csv")
