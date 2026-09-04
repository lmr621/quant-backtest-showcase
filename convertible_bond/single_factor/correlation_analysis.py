# -*- coding: utf-8 -*-
"""Effective因子相关性分析：因子收益相关性 + 热力图"""
import sys, os, re, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.dirname(HERE)
OVERNIGHT_DIR = os.path.dirname(BACKTEST_DIR)
CB_DIR = os.path.dirname(OVERNIGHT_DIR)
sys.path.insert(0, CB_DIR)

import config, operators as op

ALPHA_DIR = os.path.join(CB_DIR, 'QRdatabase', 'alpha', 'overnight', 'effective')
OUTPUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_alphas():
    """加载所有effective因子信号，返回 {name: array}"""
    alphas = {}
    for fn in sorted(os.listdir(ALPHA_DIR)):
        if fn.endswith('.npy'):
            name = fn[:-4]
            alphas[name] = np.load(os.path.join(ALPHA_DIR, fn))
    return alphas


def compute_factor_returns(alphas, fwd_ret, valids):
    """计算每个因子的多空日收益序列"""
    factor_rets = {}
    n_b, n_d = fwd_ret.shape

    for name, alpha in alphas.items():
        raw = alpha  # 已预处理
        daily_ret = np.full(n_d, np.nan)
        for d in range(n_d):
            col = raw[:, d]
            vm = ~np.isnan(col) & valids[:, d]
            if vm.sum() < 40: continue
            # 做多 top 20%，做空 bottom 20%
            vf = col[vm]
            th_u = np.percentile(vf, 80)
            th_l = np.percentile(vf, 20)
            long_mask = vm & (col >= th_u)
            short_mask = vm & (col <= th_l)
            if long_mask.sum() == 0 or short_mask.sum() == 0: continue
            long_w = 1.0 / long_mask.sum()
            short_w = 1.0 / short_mask.sum()
            daily_ret[d] = (np.nansum(long_w * fwd_ret[:, d][long_mask]) -
                            np.nansum(short_w * fwd_ret[:, d][short_mask]))
        factor_rets[name] = daily_ret

    return factor_rets


def main():
    print("=" * 70)
    print("Effective 因子相关性分析")
    print("=" * 70)

    # 加载数据
    alphas = load_all_alphas()
    print(f"\n加载 {len(alphas)} 个因子信号")

    bonds, dates_all = op.load_meta()
    dates_pd = pd.to_datetime(dates_all.astype(str))
    mask = dates_pd >= pd.to_datetime(config.BACKTEST_START_DATE)
    n_d = mask.sum()
    fwd_ret = op.returns('overnight')[:, mask]
    valids = op.valids('overnight')[:, mask]

    # 计算因子收益
    print("计算因子多空收益...")
    factor_rets = compute_factor_returns(alphas, fwd_ret, valids)

    # 构建收益DataFrame
    ret_df = pd.DataFrame(factor_rets)
    ret_df = ret_df.dropna(how='all')

    # 相关性矩阵
    corr_matrix = ret_df.corr()

    # 简化标签名
    labels = []
    for name in corr_matrix.columns:
        m = re.search(r'cb_alpha(\d+)_(.+)', name)
        if m:
            labels.append(f'{m.group(1)}_{m.group(2)[:15]}')
        else:
            labels.append(name[:25])
    corr_matrix.index = labels
    corr_matrix.columns = labels

    # 平均相关性
    n = len(corr_matrix)
    avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))  # 排除对角线
    print(f"\n平均相关系数: {avg_corr:.3f}")

    # Top 5 最高相关对
    print("\n最高相关的因子对:")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((labels[i], labels[j], corr_matrix.iloc[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    for a, b, v in pairs[:10]:
        print(f"  {a} <-> {b}: {v:+.3f}")

    # 最低相关对
    print("\n最低相关的因子对:")
    pairs.sort(key=lambda x: abs(x[2]))
    for a, b, v in pairs[:10]:
        print(f"  {a} <-> {b}: {v:+.3f}")

    # 热力图
    fig, ax = plt.subplots(figsize=(16, 13))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # 在格子里标数值
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=5.5, color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

    ax.set_title(f'Effective因子相关性热力图 (平均相关={avg_corr:.3f})', fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'factor_correlation_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"\n热力图已保存")

    # 保存CSV
    csv_path = os.path.join(OUTPUT_DIR, 'factor_correlation.csv')
    corr_matrix.to_csv(csv_path, encoding='utf-8-sig', float_format='%.4f')
    print(f"CSV已保存: {csv_path}")

    # 每个因子的年报
    print(f"\n{'='*70}")
    print("因子年化收益 (Q5-Q1 spread)")
    print(f"{'='*70}")
    for name in corr_matrix.columns:
        rets = factor_rets[ret_df.columns[labels.index(name)]]
        valid = rets[np.isfinite(rets)]
        ann = np.nanmean(valid) * 250 if len(valid) > 0 else 0
        ir = np.nanmean(valid) / np.nanstd(valid) * np.sqrt(250) if np.nanstd(valid) > 0 else 0
        print(f"  {name:30s} ann={ann:+7.1%} IR={ir:+6.2f}")

    print(f"\n结果保存: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
