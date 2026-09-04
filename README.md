# 个人量化研究回测与实盘成果

本仓库展示个人量化研究的**回测、模拟与实盘结果**，包含两大模块：**可转债多因子策略** 与 **A股多因子选股策略**。仓库仅包含回测产出（绩效图表、汇总数据、模拟盘持仓记录）与实盘收益截图，**不含因子与原始数据**。





## 模块一：可转债多因子策略

研究路径：**单因子测试 → 机器学习合成 → 模拟盘验证**

### 1. 单因子测试（`convertible_bond/single_factor/`）

对 19 个量价类因子（残差波动率、日内振幅、隔夜动量、低 Beta、换手率、多空拉锯等）**逐一进行分组回测**——每个因子一个子图，展示该因子各分组的净值走势，检验分层单调性：

![逐因子 5 分组回测（每因子一个子图）](convertible_bond/single_factor/results/layered_5q.png)

![逐因子 10 分组回测（每因子一个子图）](convertible_bond/single_factor/results/layered_10q.png)

因子间相关性矩阵：

![因子相关性热力图](convertible_bond/single_factor/results/factor_correlation_heatmap.png)

<details>
<summary>等权复合合成结果</summary>

![单因子等权复合 5 分层](convertible_bond/single_factor/results/EW_5q.png)
![单因子等权复合 10 分层](convertible_bond/single_factor/results/EW_10q.png)
![等权复合净值](convertible_bond/single_factor/results/ew_composite.png)

</details>



### 2. LightGBM 因子合成（`convertible_bond/lightgbm/`）

用 LightGBM 对单因子池做非线性合成，对比三种方案：等权（EW）、标准 LGB、浅层 LGB（LGB_Shallow，控过拟合）。

![各方案净值对比](convertible_bond/lightgbm/results/lgb_nav_comparison.png)

<details>
<summary>更多 LightGBM 图表</summary>

![因子重要性（条形图）](convertible_bond/lightgbm/results/lgb_feature_importance_bar.png)
![LGB 10 分层](convertible_bond/lightgbm/results/lgb_LGB_10层.png)
![LGB 5 分层](convertible_bond/lightgbm/results/lgb_LGB_5层.png)

</details>

### 3. 模拟交易验证（`convertible_bond/paper_trading/`）

将合成因子落地的**每日买入清单**（`buy_list_*.json`），按持仓数 N=8/10/12/15、等权/加权两种权重模拟：

![模拟盘净值对比](convertible_bond/paper_trading/results/paper_trading_comparison.png)

<details>
<summary>更多 模拟交易 图表</summary>

![模拟盘绩效汇总](convertible_bond/paper_trading/results/paper_trading_results.png)

</details>
---

## 模块二：A股多因子选股策略

### 1. 单因子测试（`stock_factors/single_factors/`）

对 32 个 alpha 因子逐一测试，每个因子产出**分层净值曲线、多空收益柱状图、滚动 IC** 三类图表：


- 多数因子 Q1→Q5 分层收益单调
- 完整指标见 `factor_metrics.csv`（IR、年化、Sharpe、最大回撤、IC 均值、分层收益）

![因子相关性矩阵](stock_factors/single_factors/factor_correlation.png)

<details>
<summary>单因子图表示例</summary>

![alpha_001 分层收益](stock_factors/single_factors/layered/effective_alpha_001_bar.png)
![alpha_001 分层净值](stock_factors/single_factors/layered/effective_alpha_001_wealth.png)
![alpha_001 滚动 IC](stock_factors/single_factors/rolling_ic/effective_alpha_001.png)

（其余 31 个因子图表位于 `layered/` 与 `rolling_ic/` 目录）

</details>

### 2. 等权合成（`stock_factors/equal_weight_agg/`）

32 个因子等权合成综合分，分别以 **2016** 与 **2020** 为起点做样本外对照，并进行 5/10/20/50 分层测试与参数稳健性扫描。

**分层单调性**（10 分层年化收益）：

| 起点 | Q1（最差组） | Q10（最好组） | 单调性 |
|:---|---:|---:|:---:|
| 2016 | -28.4% | **+24.2%** | 逐层递增 ✓ |
| 2020 | -20.8% | **+29.1%** | 逐层递增 ✓ |

![2016 起点 10 分层](stock_factors/equal_weight_agg/results_2016/layered_10layer_bar.png)
![2020 起点 10 分层](stock_factors/equal_weight_agg/results_2020/layered_10layer_bar.png)

**50 分层 Top 组净值**（合成分的头部选股能力）：

![2016 起点 50 分层 Top5 净值](stock_factors/equal_weight_agg/results_2016/layered_50layer_top5_ts.png)
![2020 起点 50 分层 Top5 净值](stock_factors/equal_weight_agg/results_2020/layered_50layer_top5_ts.png)


<details>
<summary>更多稳健性图表（持仓数 × 阈值组合净值）</summary>

![size10](stock_factors/equal_weight_agg/results_2016/strategy_size10_thr10.png)
![size20](stock_factors/equal_weight_agg/results_2016/strategy_size20_thr10.png)
![size30](stock_factors/equal_weight_agg/results_2016/strategy_size30_thr10.png)

</details>

---

## 实盘表现

| 账户 | 区间 | 累计收益 | 
|:---|:---|---:|
| 账户1 | 2021-09 ~ 2024-05 | **+36.33%** | 
| 账户2 | 2024-07 ~ 2026-09 | **+38.51%** | 

![账户1收益： 2021-09 ~ 2024-05](实盘收益1.png)

![账户2收益： +38.51%](实盘收益2.png)

## 目录结构

```
├── 实盘收益1.png / 实盘收益2.png # 实盘账户收益截图
├── convertible_bond/            # 可转债多因子
│   ├── single_factor/           #   19 个单因子逐因子分组回测 + 相关性
│   ├── lightgbm/                #   LightGBM 因子合成（EW / LGB / LGB_Shallow）
│   └── paper_trading/           #   模拟盘（每日买入清单 + 绩效）
└── stock_factors/               # A股多因子
    ├── single_factors/          #   32 个 alpha 因子（分层 / 滚动 IC）
    └── equal_weight_agg/        #   等权合成（2016 / 2020 双起点 + 稳健性）
```

