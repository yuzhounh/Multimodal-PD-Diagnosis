# Multimodal Parkinson's Disease Diagnosis

基于 PPMI（Parkinson's Progression Markers Initiative）数据集的多模态帕金森病（PD）与健康对照（HC）分类诊断系统。本项目融合临床/人口学特征与 MNI 空间 MRI ROI 影像特征，通过多级特征组合与机器学习模型实现帕金森病的辅助诊断。

## 项目概述

本项目实现了一套完整的端到端分析流程，包括：

- **数据清洗与预处理**：特征筛选、MRI-临床数据时间匹配、异常扫描剔除、缺失值填充
- **倾向性评分匹配**：基于年龄、性别、教育年限进行 PD/HC 1:1 匹配，消除混杂因素
- **多模态特征集构建**：按临床特征强度（弱/中/强）与 MRI ROI 特征组合，生成 7 组不同的特征集
- **机器学习建模**：Random Forest 与 XGBoost，搭配 Optuna 超参数调优、受试者级别交叉验证
- **模型解释性分析**：SHAP 值分析、排列重要性、ROC/PR 曲线绘制

## 项目结构

```
Multimodal-PD-Diagnosis/
├── main.py                      # 流水线入口，按顺序执行所有 step*.py
├── utils.py                     # 共享工具函数（数据加载、交叉验证、指标计算、SHAP、绘图）
├── step1_reserve_features.py    # 根据标记筛选保留特征
├── step2_match_tables.py        # MRI 与临床数据按最近日期匹配合并
├── step3_remove_abnormal.py     # 剔除异常预处理的 MRI 扫描
├── step4_reserve_samples.py     # 保留 PD/HC 样本，编码标签
├── step5_count_missing.py       # 统计各特征缺失率
├── step6_remove_features.py     # 移除高缺失率或数据泄漏特征
├── step7_fill_missing.py        # 缺失值填充（离散→众数，连续→中位数）
├── step8_propensity_score_matching.py  # 倾向性评分匹配
├── step9_statistical_analysis.py       # 基线统计分析（PD vs HC），FDR 校正
├── step10_extract_data.py       # 构建 7 组多模态特征集
├── step11_split_data.py         # 受试者级别分层 70/30 训练-测试划分
├── step12_RF.py                 # Random Forest 训练与评估
├── step12_XGBoost.py            # XGBoost 训练与评估
├── PPMI_risk_factors.csv        # 特征筛选标记表
├── PPMI_ROI_values_MNI.csv      # MNI 空间 ROI 提取值
├── PPMI_feature_strength.csv    # 临床特征强度分级（弱/中/强）
├── PPMI_feature_mapping.csv     # 特征名称映射（缩写 ↔ 全称）
├── AAL3v1_with_flags.csv        # AAL3 脑区元数据与筛选标记
└── README.md
```

## 数据流水线

```
PPMI_Curated_Data (Excel)
  │
  ├─ step1 ──→ PPMI_1_features.csv          # 筛选后的临床特征
  ├─ step2 ──→ PPMI_2_ROI_values.csv  # 临床+MRI 合并表
  ├─ step3 ──→ PPMI_3_remove_abnormal_data.csv
  ├─ step4 ──→ PPMI_4_PD_HC.csv
  ├─ step5 ──→ result_5_missing_percentages.csv（统计报告）
  ├─ step6 ──→ PPMI_5_key_features.csv
  ├─ step7 ──→ PPMI_6_filled.csv
  ├─ step8 ──→ PPMI_7_propensity_score_matching.csv
  ├─ step9 ──→ result_9_statistical_analysis.csv（统计报告）
  ├─ step10 ─→ PPMI_8_data_1.csv … PPMI_8_data_7.csv
  ├─ step11 ─→ *_train.csv / *_test.csv
  └─ step12 ─→ results/（模型、指标、SHAP、ROC/PR 曲线）
```

## 七组特征集说明

| 编号 | 特征组合 | 描述 |
|------|----------|------|
| 1 | 弱临床特征 | 年龄、性别、教育年限、BMI、焦虑量表等 |
| 2 | 弱/中等临床特征 | 追加 MoCA、MCI、REM、GDS 等认知/睡眠/情绪量表 |
| 3 | 弱/中等/强临床特征 | 追加 UPDRS、NHY、LEDD、家族史等核心运动指标 |
| 4 | MRI | AAL3 脑区 ROI 值（Flag ≥ 6） |
| 5 | 弱临床 + MRI | 特征集 1 + 特征集 4 |
| 6 | 弱/中等临床 + MRI | 特征集 2 + 特征集 4 |
| 7 | 弱/中等/强临床 + MRI | 特征集 3 + 特征集 4 |

## 环境依赖

- Python 3.12+
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- shap
- matplotlib
- seaborn
- scipy
- statsmodels
- openpyxl

安装依赖：

```bash
pip install pandas numpy scikit-learn xgboost optuna shap matplotlib seaborn scipy statsmodels openpyxl
```

## 使用方法

### 一键执行全部流水线

```bash
python main.py
```

`main.py` 将自动按编号顺序执行 `step1` 到 `step12` 的所有脚本，任何步骤失败将中止后续执行。

### 单独执行某一步

```bash
python step1_reserve_features.py
python step2_match_tables.py
……
python step12_RF.py
```

## 前置数据要求

运行完整流水线前，需确保以下文件就位：

| 文件 | 说明 |
|------|------|
| `PPMI_Curated_Data_Cut_Public_20250321.xlsx` | PPMI 官方导出的临床数据表 |
| `abnormal_preprocessed_viz/` | 异常扫描可视化图片目录（供 step3 匹配剔除） |

## 输出结果

所有模型结果保存在 `results/` 目录下，包括：

- 模型文件（`.pkl`）
- 分类性能指标（Accuracy、AUC-ROC、AUC-PR、F1 等）
- SHAP 特征重要性图（PDF/SVG）
- ROC 曲线与 PR 曲线（PDF）
- 排列重要性分析结果
- 详细预测结果与 SHAP 值（CSV）

## 评估指标

模型在三种决策阈值下分别评估：

- **默认阈值**（0.5）
- **最优 F1 阈值**：使 F1-score 最大化
- **Youden 阈值**：使约登指数（TPR - FPR）最大化

每种阈值下报告：AUC-ROC、AUC-PR、Accuracy、Balanced Accuracy、Precision、Recall、Specificity、F1-score、Cohen's Kappa。

## 许可证

本项目仅用于学术研究。PPMI 数据使用须遵循其[数据使用协议](https://www.ppmi-info.org/access-data-specimens/download-data)。
