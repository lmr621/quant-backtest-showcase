# -*- coding: utf-8 -*-
"""Effective因子聚合回测: EW等权分层"""
import sys, os, re, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CB_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, CB_DIR)
import config, operators as op

ALPHA_DIR = os.path.join(CB_DIR, 'QRdatabase', 'alpha', 'overnight', 'effective')
OUTPUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
C10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']


def load_alphas():
    """加载所有因子信号，对齐到最小公共维度"""
    names = []
    signals = []
    for fn in sorted(os.listdir(ALPHA_DIR)):
        if fn.endswith('.npy'):
            names.append(fn[:-4])
            signals.append(np.load(os.path.join(ALPHA_DIR, fn)))
    min_b = min(s.shape[0] for s in signals)
    min_d = min(s.shape[1] for s in signals)
    aligned = [s[:min_b, :min_d] for s in signals]
    return names, np.stack(aligned, axis=2)


def layered_portfolio(composite, fwd_ret, valids, n_q=5):
    """n_q层等权分层"""
    n_b, n_d = composite.shape
    layer_rets = []
    layer_wealths = []

    for q in range(1, n_q + 1):
        s = np.zeros((n_b, n_d))
        for d in range(n_d):
            col = composite[:, d]
            vm = ~np.isnan(col) & valids[:, d]
            if vm.sum() < n_q * 2: continue
            vf = col[vm]; pct = 100.0 / n_q
            lo = np.percentile(vf, (q - 1) * pct)
            hi = np.percentile(vf, q * pct)
            if q < n_q:
                lm = vm & (col >= lo) & (col < hi)
            else:
                lm = vm & (col >= lo)
            if lm.sum() > 0: s[lm, d] = 1.0 / lm.sum()

        dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
        w = np.cumprod(1.0 + dr)
        layer_rets.append(dr)
        layer_wealths.append(w)

    anns = [np.nanmean(dr) * 250 if len(dr) > 0 else np.nan for dr in layer_rets]
    return anns, layer_wealths


def plot_results(anns, wealths, n_q, name, dates_pd, out_dir):
    """绘制分层净值图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for qi in range(n_q):
        if len(wealths[qi]) > 0:
            ax.plot(dates_pd[-len(wealths[qi]):], wealths[qi],
                    label=f'Q{qi+1} ({anns[qi]:+.1%})', color=C10[qi * (10 // n_q)], lw=1.5)
    ax.set_yscale('log'); ax.set_title(f'{name} — {n_q}层分层净值')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_{n_q}q.png'), dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    print("=" * 70)
    print("Effective因子聚合回测: EW等权")
    print("=" * 70)

    names, factors = load_alphas()
    n_b, n_d, n_f = factors.shape
    print(f"\n因子数: {n_f}, 日期数: {n_d}")

    bonds, dates_all = op.load_meta()
    dates_pd = pd.to_datetime(dates_all.astype(str))
    mask = dates_pd >= pd.to_datetime(config.BACKTEST_START_DATE)
    fwd_ret_full = op.returns('overnight')
    valids_full = op.valids('overnight')
    fwd_ret = fwd_ret_full[:n_b, :][:, mask][:, :n_d]
    valids = valids_full[:n_b, :][:, mask][:, :n_d]
    dates_masked = dates_pd[mask][:n_d]
    print(f"回测区间: {dates_masked[0].strftime('%Y-%m-%d')} ~ {dates_masked[-1].strftime('%Y-%m-%d')}")

    # EW 等权聚合
    ew_composite = np.nanmean(factors, axis=2)

    # 5层
    ew5_anns, ew5_wealths = layered_portfolio(ew_composite, fwd_ret, valids, 5)
    print(f"\n5层: Q1={ew5_anns[0]:+.1%} Q2={ew5_anns[1]:+.1%} Q3={ew5_anns[2]:+.1%} Q4={ew5_anns[3]:+.1%} Q5={ew5_anns[4]:+.1%}")
    print(f"  Q5-Q1 spread: {ew5_anns[4]-ew5_anns[0]:+.1%}")

    # 10层
    ew10_anns, ew10_wealths = layered_portfolio(ew_composite, fwd_ret, valids, 10)
    print(f"\n10层: Q1={ew10_anns[0]:+.1%} Q5={ew10_anns[4]:+.1%} Q10={ew10_anns[9]:+.1%}")
    print(f"  Q10-Q1 spread: {ew10_anns[9]-ew10_anns[0]:+.1%}")

    plot_results(ew5_anns, ew5_wealths, 5, 'EW', dates_masked, OUTPUT_DIR)
    plot_results(ew10_anns, ew10_wealths, 10, 'EW', dates_masked, OUTPUT_DIR)

    print(f"\n结果保存: {OUTPUT_DIR}")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
