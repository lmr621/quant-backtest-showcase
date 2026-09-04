# -*- coding: utf-8 -*-
"""Effective因子完整回测分析。

对 effective/ 下所有因子：
  - 5层/10层分层等权回测 (long-short)
  - Top10%/Top5% 纯多头回测 (long-only)
  - 等权多因子合成回测
  - 输出分层净值图 + 汇总CSV

用法: python run_backtest.py
"""
import sys, os, time, json, re, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.dirname(HERE)
OVERNIGHT_DIR = os.path.dirname(BACKTEST_DIR)
CB_DIR = os.path.dirname(OVERNIGHT_DIR)
sys.path.insert(0, CB_DIR)

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import config, operators as op

EFFECTIVE_DIR = os.path.join(OVERNIGHT_DIR, 'updates_alphas', 'effective')
OUTPUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

C10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']


ALPHA_DIR = os.path.join(CB_DIR, 'QRdatabase', 'alpha', 'overnight', 'effective')

def load_effective_alphas():
    """从预计算的 alpha 文件中加载因子信号，返回 [(name, num, factor_matrix), ...]"""
    factors = []
    files = sorted(f for f in os.listdir(ALPHA_DIR) if f.endswith('.npy'))
    for fn in files:
        name = fn[:-4]  # remove .npy
        match = re.search(r'cb_alpha(\d+)', name)
        if match:
            num = int(match.group(1))
            alpha = np.load(os.path.join(ALPHA_DIR, fn))
            factors.append((name, num, alpha))
    return factors


def layered_analysis(name, factor_raw, fwd_ret, valids, dates_pd, n_q=5):
    """n_q 层等权分层分析，返回 layers 结果和每日收益序列"""
    n_b, n_d = factor_raw.shape
    layers_ret = []
    layers_wealth = []

    for q in range(1, n_q + 1):
        s = np.zeros((n_b, n_d))
        for d in range(n_d):
            col = factor_raw[:, d]
            vm = ~np.isnan(col) & valids[:, d]
            if vm.sum() < n_q * 2: continue
            vf = col[vm]; pct = 100.0 / n_q
            lo = np.percentile(vf, (q - 1) * pct)
            hi = np.percentile(vf, q * pct)
            if q < n_q:
                lm = vm & (col >= lo) & (col < hi)
            else:
                lm = vm & (col >= lo)
            if lm.sum() > 0:
                s[lm, d] = 1.0 / lm.sum()

        dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
        w = np.cumprod(1.0 + dr)
        layers_ret.append(dr)
        layers_wealth.append(w)

    anns = [np.nanmean(dr) * 250 if len(dr) > 0 else np.nan for dr in layers_ret]
    return {'anns': anns, 'layers_ret': layers_ret, 'layers_wealth': layers_wealth}


def long_only_analysis(factor_raw, fwd_ret, valids, top_pct=10):
    """纯多头：做多因子值最高的 top_pct%"""
    n_b, n_d = factor_raw.shape
    s = np.zeros((n_b, n_d))
    for d in range(n_d):
        col = factor_raw[:, d]
        vm = ~np.isnan(col) & valids[:, d]
        if vm.sum() < 20: continue
        th = np.percentile(col[vm], 100 - top_pct)
        lm = vm & (col >= th)
        if lm.sum() > 0: s[lm, d] = 1.0 / lm.sum()

    dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
    w = np.cumprod(1.0 + dr)
    ann = np.nanmean(dr) * 250 if len(dr) > 0 else 0
    dd = np.max(np.maximum.accumulate(w) - w) / np.max(np.maximum.accumulate(w))
    ir_val = np.nanmean(dr) / np.nanstd(dr) * np.sqrt(250) if np.nanstd(dr) > 0 else 0
    return {'ann': ann, 'maxdd': dd, 'IR': ir_val, 'ret': dr, 'wealth': w}


def plot_layered(factors_results, n_q, dates_pd, out_dir):
    """绘制所有因子的分层净值图"""
    n_cols = 4
    n_rows = (len(factors_results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for idx, (name, num, lr) in enumerate(factors_results):
        ax = axes[idx]
        for qi in range(n_q):
            w = lr['layers_wealth'][qi]
            if len(w) > 0:
                ax.plot(dates_pd[-len(w):], w, label=f'Q{qi+1}', color=C10[qi * (10 // n_q)], lw=1.2)
        ax.set_title(f'{num:03d}', fontsize=10)
        ax.set_yscale('log')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)

    for idx in range(len(factors_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Effective因子 {n_q}层分层净值', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'layered_{n_q}q.png'), dpi=150)
    plt.close(fig)


def plot_ew_composite(ew_ret, dates_pd, out_dir):
    """等权合成因子收益曲线"""
    fig, ax = plt.subplots(figsize=(12, 5))
    w = np.cumprod(1.0 + ew_ret)
    ax.plot(dates_pd[-len(w):], w, lw=1.5, color='#1f77b4')
    ax.set_yscale('log')
    ax.set_title('Effective因子等权合成 — 多空组合净值')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ew_composite.png'), dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print("=" * 70)
    print("Effective 因子回测分析")
    print("=" * 70)

    # 加载预计算的alpha信号
    factors = load_effective_alphas()
    print(f"\n发现 {len(factors)} 个因子:")
    for name, num, alpha in factors:
        print(f"  [{num:03d}] {name}")

    # 加载数据
    bonds, dates_all = op.load_meta()
    dates_pd = pd.to_datetime(dates_all.astype(str))
    mask = dates_pd >= pd.to_datetime(config.BACKTEST_START_DATE)
    dates_masked = dates_pd[mask]
    n_d = mask.sum()
    fwd_ret = op.returns('overnight')[:, mask]
    valids = op.valids('overnight')[:, mask]
    print(f"\n回测区间: {dates_masked[0].strftime('%Y-%m-%d')} ~ {dates_masked[-1].strftime('%Y-%m-%d')}, {n_d}天")

    # 逐个因子分析
    all_results_5 = []
    all_results_10 = []
    summary_rows = []

    for name, num, alpha in factors:
        print(f"\n[{num:03d}] {name}...", end=' ', flush=True)
        t1 = time.time()

        # alpha信号已过滤日期，直接使用
        raw = alpha

        # 5层分层
        r5 = layered_analysis(name, raw, fwd_ret, valids, dates_masked, 5)
        # 10层分层
        r10 = layered_analysis(name, raw, fwd_ret, valids, dates_masked, 10)
        # Top10% 纯多头
        t10 = long_only_analysis(raw, fwd_ret, valids, 10)
        # Top5% 纯多头
        t5 = long_only_analysis(raw, fwd_ret, valids, 5)

        all_results_5.append((name, num, r5))
        all_results_10.append((name, num, r10))
        summary_rows.append({
            'num': num, 'name': name,
            'Q1_ann': r5['anns'][0], 'Q5_ann': r5['anns'][4],
            'Q5_Q1_spread': r5['anns'][4] - r5['anns'][0],
            'Top10_ann': t10['ann'], 'Top10_IR': t10['IR'], 'Top10_maxDD': t10['maxdd'],
            'Top5_ann': t5['ann'], 'Top5_IR': t5['IR'], 'Top5_maxDD': t5['maxdd'],
        })

        spread = r5['anns'][4] - r5['anns'][0]
        print(f"Q5-Q1={spread:+.1%} Top10={t10['ann']:+.1%} IR={t10['IR']:+.1f} ({time.time()-t1:.0f}s)")

    # 等权合成
    print(f"\n{'='*70}\n等权多因子合成...")
    all_signals = []
    for name, num, alpha in factors:
        all_signals.append(alpha)

    ew_composite = np.nanmean(np.stack(all_signals, axis=2), axis=2)
    ew_ret = np.array([np.nansum(ew_composite[:, t] / np.nansum(np.abs(ew_composite[:, t])) * fwd_ret[:, t])
                       for t in range(n_d) if np.nansum(np.abs(ew_composite[:, t])) > 0])
    ew_wealth = np.cumprod(1.0 + ew_ret)
    ew_ann = np.nanmean(ew_ret) * 250
    ew_dd = np.max(np.maximum.accumulate(ew_wealth) - ew_wealth) / np.max(np.maximum.accumulate(ew_wealth))
    ew_ir = np.nanmean(ew_ret) / np.nanstd(ew_ret) * np.sqrt(250)
    print(f"EW Composite: ann={ew_ann:+.1%} DD={ew_dd:.1%} IR={ew_ir:.2f}")

    # 绘图
    print("\n绘制图表...")
    plot_layered(all_results_5, 5, dates_masked, OUTPUT_DIR)
    plot_layered(all_results_10, 10, dates_masked, OUTPUT_DIR)
    plot_ew_composite(ew_ret, dates_masked, OUTPUT_DIR)

    # 保存汇总CSV
    df = pd.DataFrame(summary_rows)
    df = df.sort_values('Q5_Q1_spread', ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, 'summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig', float_format='%.4f')

    # 打印汇总
    print(f"\n{'='*70}")
    print("因子汇总 (按 Q5-Q1 spread 排序)")
    print(f"{'='*70}")
    print(f"{'#':>3s} {'因子':25s} {'Q5-Q1':>8s} {'Top10':>8s} {'Top10_IR':>8s} {'Top5':>8s}")
    for _, row in df.iterrows():
        print(f"{row['num']:3d} {row['name']:25s} {row['Q5_Q1_spread']:+8.1%} {row['Top10_ann']:+8.1%} {row['Top10_IR']:+8.2f} {row['Top5_ann']:+8.1%}")

    print(f"\n等权合成: ann={ew_ann:+.1%} DD={ew_dd:.1%} IR={ew_ir:.2f}")
    print(f"\n结果保存: {OUTPUT_DIR}")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
