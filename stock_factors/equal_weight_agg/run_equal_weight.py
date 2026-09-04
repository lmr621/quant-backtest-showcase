# -*- coding: utf-8 -*-
"""等权聚合策略（生产用）。

方法：32 个有效因子 cs_rank 到 [-1,1] 后等权平均，得到复合因子；
     方向检查（高值→高收益）后，做分层回测 + Top-N 选股。

Usage:
    python run_equal_weight.py                # 2016 起
    python run_equal_weight.py 2020           # 2020 起

Outputs (into equal_weight/results_<year>/):
  - layered_<5/10/20/50>layer_bar.png / _ts.png   分层收益柱状图 + 净值时序图
  - robustness_grid.csv                            持仓数 x 调仓阈值 稳健性网格
  - strategy_size<thr>_thr<thr>.png                各组合净值 + 回撤曲线
  - buy_list_EW.json                               每日选股列表
"""
import sys, os, time, json, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, r'D:\investment')

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings; warnings.filterwarnings('ignore')

from agg_utils import (EFFECTIVE_NAMES, load_all_factors, cs_rank, calc_metrics,
                        layered_backtest, build_top10_strategy)
from operators import simulate, load_var

LAYERS = [5, 10, 20, 50]
TOP_K_FOR_50 = 5
SIZES = [10, 15, 20, 30]
THRESHOLDS = [20, 10, 5, 2.5]
START_DATES = {2016: '2016-01-06', 2020: '2020-01-06'}
C_EW = '#3498db'


def layered_returns_from(composite, n_layers, n_s, n_d, start_id):
    """每层年化收益 + 净值曲线（从 start_id 起）。"""
    wealths, returns = [], []
    for layer in range(1, n_layers + 1):
        strategy = np.zeros((n_s, n_d))
        for d in range(n_d):
            df = composite[:, d]; vm = np.isfinite(df)
            if vm.sum() < n_layers * 3: continue
            lo = np.percentile(df[vm], (layer - 1) * (100.0 / n_layers))
            hi = np.percentile(df[vm], layer * (100.0 / n_layers))
            lm = (vm & (df >= lo) & (df < hi)) if layer < n_layers else (vm & (df >= lo))
            if lm.sum() > 0: strategy[lm, d] = 1.0 / lm.sum()
        with contextlib.redirect_stdout(io.StringIO()):
            sr = simulate(strategy, buffer=0.98, tc_bps=0)
        met = calc_metrics(sr['dailypnl'][start_id:])
        wealths.append(met['wealth']); returns.append(met['ann_ret'])
    return wealths, returns


def plot_layers_bar(returns, n_layers, labels, title, out_path):
    x = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, [r * 100 for r in returns], color=C_EW)
    for b, r in zip(bars, returns):
        if np.isfinite(r): ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{r*100:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Annualized Return (%)'); ax.set_title(title)
    ax.grid(axis='y', alpha=0.3); ax.axhline(0, color='black', lw=0.8)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_layers_ts(wealths, labels, dates, n_layers, title, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
    d = dates[:len(wealths[0])]
    for li in range(n_layers):
        ax.plot(d[:len(wealths[li])], wealths[li], color=colors[li], lw=1.1, label=labels[li])
    ax.set_title(title); ax.legend(ncol=2, fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_strategy_ts(met, dates, size, thr, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    w = met['wealth']; d = dates[:len(w)]
    axes[0].plot(d, w, color=C_EW, lw=1.3, label=f'EW (AnnRet={met["ann_ret"]:.1%})')
    axes[0].set_title(f'Wealth — size={size}, threshold={thr}%'); axes[0].legend(); axes[0].grid(alpha=0.3)
    dd = -np.asarray(met['drawdown']) * 100
    axes[1].fill_between(d[:len(dd)], dd, 0, color=C_EW, alpha=0.35, label=f'MaxDD={met["max_dd"]:.1%}')
    axes[1].set_title('Dynamic Drawdown (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(np.nanmin(dd) * 1.1, 5)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    START_DATE = START_DATES.get(start_year, f'{start_year}-01-06')
    OUT_DIR = os.path.join(HERE, f'results_{start_year}')
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    print('=' * 60)
    print(f'等权聚合策略 (start {START_DATE})')
    print('=' * 60)

    # ── 加载 + 等权复合 ──
    print('\n[1/3] 加载因子 + 构建等权复合...')
    factors_raw, valid_names = load_all_factors()
    n_f = factors_raw.shape[2]
    dates_raw = load_var('dates'); dates = pd.to_datetime(dates_raw.astype(str))
    try:
        start_id = dates.get_loc(START_DATE)
    except KeyError:
        start_id = dates.searchsorted(pd.Timestamp(START_DATE))
    n_s, n_d = factors_raw.shape[0], factors_raw.shape[1]

    factors = np.zeros_like(factors_raw)
    for fi in range(n_f):
        factors[:, :, fi] = cs_rank(factors_raw[:, :, fi])
    comp = np.nanmean(factors, axis=2)

    # 方向检查（高值 → 高收益）
    lw, lm = layered_backtest(comp, 5, n_s, n_d, tc_bps=0)
    if lm[0]['ann_ret'] > lm[4]['ann_ret']:
        comp = -comp
        print('  方向已翻转（反转因子统一取负）')
    else:
        print('  方向 OK')

    # ── 分层回测 ──
    print('\n[2/3] 分层回测 (5/10/20/50)...')
    layered_rows = []
    for n_layers in LAYERS:
        wealths, returns = layered_returns_from(comp, n_layers, n_s, n_d, start_id)
        if n_layers == 50:
            idx = list(range(n_layers - TOP_K_FOR_50, n_layers))
            labels = [f'Q{i+1}' for i in idx]
            plot_layers_bar([returns[i] for i in idx], TOP_K_FOR_50, labels,
                            f'{n_layers}-Layer (Top {TOP_K_FOR_50})', os.path.join(OUT_DIR, f'layered_{n_layers}layer_top5_bar.png'))
            plot_layers_ts([wealths[i] for i in idx], labels, dates[start_id + 1:], TOP_K_FOR_50,
                           f'{n_layers}-Layer Top {TOP_K_FOR_50} Wealth', os.path.join(OUT_DIR, f'layered_{n_layers}layer_top5_ts.png'))
        else:
            labels = [f'Q{i+1}' for i in range(n_layers)]
            plot_layers_bar(returns, n_layers, labels, f'{n_layers}-Layer',
                            os.path.join(OUT_DIR, f'layered_{n_layers}layer_bar.png'))
            plot_layers_ts(wealths, labels, dates[start_id + 1:], n_layers,
                           f'{n_layers}-Layer Wealth', os.path.join(OUT_DIR, f'layered_{n_layers}layer_ts.png'))
        for i in range(n_layers):
            layered_rows.append({'layers': n_layers, 'quantile': i + 1, 'ann_ret': returns[i]})
        print(f'  {n_layers} layers: Q1={returns[0]:.1%} Q{n_layers}={returns[-1]:.1%}')
    pd.DataFrame(layered_rows).to_csv(os.path.join(OUT_DIR, 'layered_summary.csv'),
                                      encoding='utf-8-sig', index=False, float_format='%.4f')

    # ── Top-N 选股（稳健性网格） ──
    print('\n[3/3] Top-N 选股稳健性（持仓 x 调仓阈值）...')
    stock_list = pd.read_csv(r'D:\QRdatabase\stock_list.csv')
    robust_rows = []
    met_cache = {}
    for size in SIZES:
        for thr in THRESHOLDS:
            sa, avg_tvr = build_top10_strategy(comp, n_s, n_d, threshold_pct=thr, size=size)
            with contextlib.redirect_stdout(io.StringIO()):
                sr = simulate(sa, buffer=0.98, tc_bps=1)
            met = calc_metrics(sr['dailypnl'][start_id:])
            robust_rows.append({'size': size, 'threshold_pct': thr, 'ann_ret': met['ann_ret'],
                                'max_dd': met['max_dd'], 'IR': met['IR'], 'sharpe': met['sharpe'],
                                'tvr': avg_tvr})
            met_cache[(size, thr)] = met
            plot_strategy_ts(met, dates[start_id + 1:], size, thr,
                             os.path.join(OUT_DIR, f'strategy_size{size}_thr{thr}.png'))
            print(f'  size={size:>2} thr={thr:>4}%: AnnRet={met["ann_ret"]:+.1%} Sharpe={met["sharpe"]:.2f} TVR={avg_tvr:.1%}')
    df_robust = pd.DataFrame(robust_rows)
    df_robust.to_csv(os.path.join(OUT_DIR, 'robustness_grid.csv'), encoding='utf-8-sig', index=False, float_format='%.4f')

    # buy_list（默认 size=10, thr=5%）
    sa, _ = build_top10_strategy(comp, n_s, n_d, threshold_pct=5, size=10)
    prev_top = None; buy_list = {}
    for day in range(n_d):
        df = comp[:, day]; vm = np.isfinite(df)
        if vm.sum() < 10: continue
        top_th = np.percentile(df[vm], 95)
        cur = set()
        if prev_top is not None:
            for si in prev_top:
                if np.isfinite(df[si]) and df[si] >= top_th: cur.add(si)
        if len(cur) < 10:
            for si in np.argsort(df)[::-1]:
                if not np.isfinite(df[si]): continue
                if si not in cur:
                    cur.add(si)
                    if len(cur) >= 10: break
        prev_top = set(cur)
        buy_list[str(dates[day])] = {'date': str(dates[day]),
                                     'stocks': stock_list.iloc[list(cur)].to_dict('records')}
    with open(os.path.join(OUT_DIR, 'buy_list_EW.json'), 'w', encoding='utf-8') as f:
        json.dump(buy_list, f, ensure_ascii=False, indent=2)

    print(f'\n总耗时 {time.time() - t0:.0f}s → {OUT_DIR}')


if __name__ == '__main__':
    main()
