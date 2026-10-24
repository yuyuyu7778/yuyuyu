# 项目：2026 MathorCup 大数据挑战赛 · 赛道 A
# 啤酒酿造化合物分析与啤酒风味预测
# 版本：终极版 v3（含 CMF 框架、失败回退、评审预演）

> 本文件放在项目根目录，Claude Code 每次启动自动读取。
> 作用：统一规则、统一框架、统一验收，避免每次对话重新交代。

---

## 一、任务背景

题目三问：

1. **问题1 数据缺失条件下的感官评分**
   以 1.csv 化合物数据为基础，给出 3.xlsx 定义的 Aroma 与 Flavour 两套、约 40 个感官维度的 0–6 分分级评分结果。

2. **问题2 化合物–风味分析**
   用 2.csv 构建化合物共表达图，结合 1.csv 特征，给出每个感官维度的 Top-3 关键化合物，并做稳定性（bootstrap 入选频率 / 方法间 Jaccard）与化学合理性论证。

3. **问题3 风味与"风味指纹"生成**
   以 tasting_category_fine 为风格标签，学习风格条件生成模型，给定风格生成化合物谱与感官画像，并做跨风格迁移（如 Pils/Lager → Tripel），要求缺陷化合物不恶化（DMS / Diacetyl / 4VG / trans-2-nonenal / Stale hops）。

数据文件（data/raw/，只读）：
- `1.csv` 样本 × 化合物丰度（226 样本，部分列对数变换，负值对应浓度 < 1；ethanol/color/bitterness/pH/CO2 保留原始量纲）
- `2.csv` 化合物 × 化合物 皮尔逊相关矩阵
- `3.xlsx` 10 个 sheet（beer_1~beer_10），0–6 离散评分
- `4.csv` 同 40 维，标准化/中心化连续值（不是原始分）
- `5.csv` / `6.csv` 相关矩阵（6.csv 为 46 化学 × 50 感官 非方阵）
- `7.csv` 250 样本 × 187 感官属性（标准化，含 NA）
- `8.csv` 综合参考（可能含缺陷化合物阈值）

---

## 二、核心框架：CMF（Compound Module Framework）

**三问不许做成三个孤立模型，必须共享同一套"化合物模块"表示。**

```text
数据层：X ∈ R^{226×p}，p ≈ 220 个化合物
表示层：共识聚类学得模块映射 M : R^p → R^K，K ∈ [8, 15]
        模块得分 Z = X M ∈ R^{226×K}

三问统一：
  Q1  f : Z → Y ∈ R^{40}      多任务有序回归 + 图正则
  Q2  g : (Z, Y) → Top-3      两层归因 + Stability Selection
  Q3  G : (c, Z) → Ẑ → X̂    条件生成 + 约束投影 + 反事实
```

**这条主线必须在论文摘要、模型总览图、结论中三次出现。**

---

## 三、硬规则（不可违反）

1. **原始数据只读**，`data/raw/` 下文件绝不修改；清洗结果存 `data/clean/`。
2. **每一步必须落盘**：脚本、结果 csv、图 png、日志 log，不允许只在对话里给结论。
3. **每阶段 git commit**，message 写明阶段：`stage2: Q1 baseline`。
4. **数字必须可追溯**：注明脚本、函数、运行时间。
5. **严禁编造**数据、文献、参数；不确定写 `[待核实]` 并给出核实方法。
6. **汇报固定格式**：做了什么 / 产出文件 / 关键数字（附输出） / 我的假设 / 可能出错处。
7. **破坏性操作先说明**（删文件、覆盖结果、装依赖、改原始数据）。
8. **不许跳过验证**：无验证指标的结论视为未完成。
9. **创新点必须有降级方案**（见第五节回退树）。
10. **所有随机过程固定 `SEED = 42`**。

---

## 四、目录约定

```
项目根/
├── CLAUDE.md
├── data/
│   ├── raw/          原始附件（只读）
│   └── clean/        清洗后数据（labels_10x40.csv, module_scores.csv ...）
├── src/              脚本，按阶段命名：s1_check.py, s2_q1.py, s3_q2.py, s4_q3.py, s5_paper.py
├── out/
│   ├── fig/          图（png, 300dpi, 中文 SimHei）
│   ├── tab/          表（csv, utf-8-sig）
│   └── log/          运行日志
├── report/           阶段报告：01_数据核验.md, 02_问题1.md, 03_问题2.md, 04_问题3.md, 05_评审.md
└── paper/            论文.md, 附录_评审预演.md, 消融总表.md, 框架总览图.png
```

---

## 五、技术约束

- Python 3.10+；依赖：pandas / numpy / scipy / sklearn / statsmodels / networkx / matplotlib / seaborn / xgboost / torch / mord / python-louvain
- 图表统一：`plt.rcParams['font.sans-serif']=['SimHei']`，`figsize=(8,5)`，`dpi=300`
- 所有脚本可独立运行：`python src/xxx.py`，不许依赖 notebook 状态
- 模块划分必须做**共识聚类**（consensus clustering）：Louvain 多阈值多种子，统计两两同社区频率，层次聚类切 K，附轮廓系数与稳定性

---

## 六、三问创新点与加固（写进论文“创新点”节）

### Q1｜图正则多任务有序回归

- 特征 = 模块得分（K 维）+ 5 个理化参数
- 目标函数：`min ||Zβ − Y||² + λ·tr(βᵀ Lβ)`，L 为模块相关图 Laplacian
- 输出层：有序回归（cumulative link），**不用普通回归 + 四舍五入**
- 迁移：用 7.csv（250×187）预训练，10 个样本微调
- 指标：MAE / RMSE / Spearman / **±1 容差准确率**
- 不确定性：bootstrap 200 次给 95% 预测区间，宽区间主动标出

### Q2｜两层归因 + Stability Selection

- 第一层：模块级归因（40 维感官 × K 模块，Spearman + RF）
- 第二层：只在 Top-3 模块内下钻单化合物（LASSO + Fisher z）
- 稳定性：Stability Selection（子采样 500 次，π_thr = 0.6），同时给 bootstrap 频率与 Jaccard 对比
- 显著性：置换检验 1000 次 + BH-FDR 校正
- 化学解释：类别 → 风味 → 依据来源

### Q3｜模块空间条件生成 + 约束 + 反事实

- CVAE 在模块空间（K 维），β 退火防后验坍缩
- 约束：`min ||Ẑ − Z_target||² s.t. defect_j ≤ τ_j`，超阈值用梯度投影
- 反事实：`min ||Δ||₀ s.t. 分类器(Ẑ) = 目标风格`（最少改几个模块）
- 闭环：生成 → Q1 模型 → 感官画像 → 与真实同风格画像余弦相似度
- 评价：MMD / 缺陷合格率 / 画像相似度 / **分类器可分性**

---

## 七、失败回退树（创新点做不出时必须降级，不许卡死）

| 创新点 | 失败现象 | 降级到 | 论文口径 |
|---|---|---|---|
| 共识聚类 | K 摇摆 / 不稳定 | 单阈值 0.6 + Louvain + 稳定性报告 | 改口“数据驱动分组” |
| 图正则 | 不收敛 / 效果更差 | 普通多任务 Ridge | 消融表保留对照行 |
| 有序回归 | 库报错 / 难调 | 回归 + 阈值切分 | 改口“阈值映射” |
| CVAE | 后验坍缩 | 模块空间条件高斯混合采样 | 改口“条件分布采样” |
| 反事实 | 优化不收敛 | 只给 Top-3 模块改动方向建议 | 删优化过程，保留结论 |
| 闭环验证 | Q1 模型不准 | 只用真实 Tripel 画像对比 | 不用 Q1 反推 |

**原则：任何创新点卡住超过 4 小时，立即执行降级方案，并在论文里如实写“尝试了 X，因 Y 改为 Z”。**

---

## 八、必做验证清单（缺一不可）

| 验证 | 目的 | 合格标准 |
|---|---|---|
| 负对照（打乱标签） | 查数据泄漏 | 打乱后指标明显变差 |
| 随机模块对照 | 证明模块不是玄学 | 真实模块显著优于随机分组 |
| 置换检验（Q2） | 显著性 | p < 0.05 且经 BH-FDR 校正 |
| 闭环验证（Q3） | 生成合理性 | 生成画像与真实画像相似度 > 0.7 |
| 缺陷约束对比（Q3） | 不恶化 | 迁移后超阈值项数 ≤ 迁移前 |
| 陌生化测试 | 摘要可读性 | 新会话仅凭摘要+总览图能复述框架 |

---

## 九、最终交付物清单

- [ ] `data/clean/labels_10x40.csv`、`module_scores.csv`
- [ ] `out/tab/`：q1_pred、q1_metrics、q1_ablation、q2_top3、q2_stability、q2_jaccard、q3_gen、q3_transfer、q3_metrics、ablation_all
- [ ] `out/fig/`：q1_scatter、q2_graph、q2_module_sensory_heatmap、q3_pca、q3_defect_bar、框架总览图
- [ ] `report/`：01–05 五份阶段报告
- [ ] `paper/`：论文.md、附录_评审预演.md、消融总表.md
- [ ] `git log` 每阶段一个 commit
- [ ] 附录代码可复现：`python src/s2_q1.py` 一键跑出主结果

---

## 十、当前进度

- [ ] Step 0 环境与数据清点
- [ ] Step 1 数据核验（log 底数 / sheet 对齐 / 4.csv 反标准化 / 7.csv 匹配 / 8.csv 阈值）
- [ ] 框架搭建：共识聚类定 K、模块划分、module_scores.csv
- [ ] Step 2 问题1（含消融 6 组 + 负对照）
- [ ] Step 3 问题2（两层归因 + stability selection + 置换检验）
- [ ] Step 4 问题3（CVAE + 约束 + 反事实 + 闭环）
- [ ] Step 5 论文 + 消融总表 + 评审预演 + 陌生化测试
