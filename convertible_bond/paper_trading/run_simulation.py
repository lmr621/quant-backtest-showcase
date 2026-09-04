# -*- coding: utf-8 -*-
"""模拟盘测试：EW聚合因子值，每日等权做多N只CB，绘制收益和回撤

参考: /d/investment/strategies/factor_choose_and_aggregation_strategy.py
"""
import sys, os, re, time, json
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
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

PORTFOLIO_SIZES = [8, 10, 12, 15]
C10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
       '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']


def compute_metrics(daily_ret):
    """计算累计净值、年化收益、最大回撤"""
    wealth = np.cumprod(1.0 + daily_ret)
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    ann_ret = np.nanmean(daily_ret) * 250 if len(daily_ret) > 0 else 0
    ir = np.nanmean(daily_ret) / np.nanstd(daily_ret) * np.sqrt(250) if np.nanstd(daily_ret) > 0 else 0
    max_dd = np.nanmax(drawdown)
    return wealth, drawdown, ann_ret, ir, max_dd


def simulate_long_only(composite, fwd_ret, valids, limit_up, n_hold, cost_oneway_bps=1):
    """每日等权做多因子值最高的N只CB（剔除14:45已涨停的）。

    成本假设: 收盘买入 + 次日开盘卖出 = 每日两次交易
    单次费率 = cost_oneway_bps (默认万分之一=1bp)
    每日成本 = 2 × 万分之一 = 0.02%
    """
    n_b, n_d = composite.shape
    daily_cost = 2 * cost_oneway_bps / 10000  # 每日全仓买卖两次的成本比例
    daily_ret = np.full(n_d, np.nan)
    holdings = []
    excluded_days = 0

    for d in range(n_d):
        col = composite[:, d]
        vm = ~np.isnan(col) & valids[:, d] & ~limit_up[:, d]
        n_valid = vm.sum()
        if n_valid < n_hold:
            excluded_days += 1
            continue

        sorted_idx = np.argsort(col[vm])[::-1][:n_hold]
        vm_indices = np.where(vm)[0]
        top_indices = vm_indices[sorted_idx]

        w = 1.0 / n_hold
        ret = np.nansum(w * fwd_ret[top_indices, d])

        # 每日全仓买卖两次，扣除固定成本
        ret -= daily_cost

        daily_ret[d] = ret
        holdings.append(top_indices)

    if excluded_days > 0:
        print(f"({excluded_days}天持仓不足)")
    annual_cost = daily_cost * 250
    print(f"[单次万{cost_oneway_bps}, 每日2次=万{cost_oneway_bps*2}, 年化成本={annual_cost:.1%}]")
    return daily_ret, holdings


def simulate_long_only_weighted(composite, fwd_ret, valids, limit_up, n_hold, cost_oneway_bps=1):
    """每日按因子值比例做多因子值最高的N只CB（剔除14:45已涨停的）。

    权重分配: 根据因子值大小按比例分配权重
    成本假设: 收盘买入 + 次日开盘卖出 = 每日两次交易
    单次费率 = cost_oneway_bps (默认万分之一=1bp)
    每日成本 = 2 × 万分之一 = 0.02%
    """
    n_b, n_d = composite.shape
    daily_cost = 2 * cost_oneway_bps / 10000  # 每日全仓买卖两次的成本比例
    daily_ret = np.full(n_d, np.nan)
    holdings = []
    weights_list = []
    excluded_days = 0

    for d in range(n_d):
        col = composite[:, d]
        vm = ~np.isnan(col) & valids[:, d] & ~limit_up[:, d]
        n_valid = vm.sum()
        if n_valid < n_hold:
            excluded_days += 1
            continue

        sorted_idx = np.argsort(col[vm])[::-1][:n_hold]
        vm_indices = np.where(vm)[0]
        top_indices = vm_indices[sorted_idx]

        # 按因子值比例分配权重
        factor_values = col[top_indices]
        # 确保因子值为正（如果有负值，先平移到正值区间）
        min_factor = factor_values.min()
        if min_factor <= 0:
            factor_values = factor_values - min_factor + 1e-8
        # 归一化权重
        weights = factor_values / factor_values.sum()

        ret = np.nansum(weights * fwd_ret[top_indices, d])

        # 每日全仓买卖两次，扣除固定成本
        ret -= daily_cost

        daily_ret[d] = ret
        holdings.append(top_indices)
        weights_list.append(weights)

    if excluded_days > 0:
        print(f"({excluded_days}天持仓不足)")
    annual_cost = daily_cost * 250
    print(f"[单次万{cost_oneway_bps}, 每日2次=万{cost_oneway_bps*2}, 年化成本={annual_cost:.1%}]")
    return daily_ret, holdings, weights_list


def main():
    t0 = time.time()
    print("=" * 70)
    print("模拟盘测试: EW聚合 + 每日做多N只CB")
    print("=" * 70)

    # 加载数据
    names = []; signals = []
    for fn in sorted(os.listdir(ALPHA_DIR)):
        if fn.endswith('.npy'):
            names.append(fn[:-4])
            signals.append(np.load(os.path.join(ALPHA_DIR, fn)))
    factors = np.stack(signals, axis=2)
    n_f = len(factors)

    bonds, dates_all = op.load_meta()
    dates_pd = pd.to_datetime(dates_all.astype(str))
    mask = dates_pd >= pd.to_datetime(config.BACKTEST_START_DATE)
    dates_masked = dates_pd[mask]
    fwd_ret = op.returns('overnight')[:, mask]
    valids = op.valids('overnight')[:, mask]
    n_b, n_d = factors.shape[0], factors.shape[1]

    # 计算14:45涨停状态 (价格≥prev_close * 1.195 = 接近20%涨停)
    px_1455 = op.load_var('px_1455')[:, mask]
    pre_close = op.load_var('pre_close')[:, mask]
    ret_1445 = np.full_like(px_1455, np.nan)
    valid_p = pre_close > 0
    ret_1445[valid_p] = px_1455[valid_p] / pre_close[valid_p] - 1.0
    limit_up = (ret_1445 >= 0.195)
    limit_up_count = limit_up.sum()
    print(f"14:45涨停次数: {limit_up_count} ({limit_up_count/limit_up.size*100:.2f}%)")

    print(f"因子数: {n_f}, 回测区间: {dates_masked[0].strftime('%Y-%m-%d')} ~ {dates_masked[-1].strftime('%Y-%m-%d')}")
    print(f"持仓数: {PORTFOLIO_SIZES}")

    # EW聚合
    composite = np.nanmean(factors, axis=2)
    print("EW聚合因子值完成")

    # 对每个持仓数做模拟 (等权 vs 加权)
    results_equal = {}
    results_weighted = {}
    for n_hold in PORTFOLIO_SIZES:
        print(f"\n模拟 N={n_hold}...", end=' ', flush=True)
        # 等权配置
        daily_ret_eq, holdings_eq = simulate_long_only(composite, fwd_ret, valids, limit_up, n_hold)
        valid_ret_eq = daily_ret_eq[np.isfinite(daily_ret_eq)]
        wealth_eq, dd_eq, ann_eq, ir_eq, max_dd_eq = compute_metrics(valid_ret_eq)
        results_equal[n_hold] = {
            'daily_ret': valid_ret_eq, 'wealth': wealth_eq, 'drawdown': dd_eq,
            'ann_ret': ann_eq, 'IR': ir_eq, 'max_dd': max_dd_eq,
        }
        print(f"[等权] ann={ann_eq:+.1%} IR={ir_eq:+.2f} maxDD={max_dd_eq:.1%}", end=' ')

        # 加权配置
        daily_ret_wt, holdings_wt, weights_wt = simulate_long_only_weighted(composite, fwd_ret, valids, limit_up, n_hold)
        valid_ret_wt = daily_ret_wt[np.isfinite(daily_ret_wt)]
        wealth_wt, dd_wt, ann_wt, ir_wt, max_dd_wt = compute_metrics(valid_ret_wt)
        results_weighted[n_hold] = {
            'daily_ret': valid_ret_wt, 'wealth': wealth_wt, 'drawdown': dd_wt,
            'ann_ret': ann_wt, 'IR': ir_wt, 'max_dd': max_dd_wt,
        }
        print(f"[加权] ann={ann_wt:+.1%} IR={ir_wt:+.2f} maxDD={max_dd_wt:.1%}")

    # =========== 绘制: 净值 + 回撤 (等权 vs 加权对比) ===========
    fig, axes = plt.subplots(len(PORTFOLIO_SIZES), 2, figsize=(14, 3 * len(PORTFOLIO_SIZES)))

    for i, n_hold in enumerate(PORTFOLIO_SIZES):
        r_eq = results_equal[n_hold]
        r_wt = results_weighted[n_hold]
        ax_w, ax_d = axes[i, 0], axes[i, 1]

        # 累计净值对比
        ax_w.plot(dates_masked[-len(r_eq['wealth']):], r_eq['wealth'], color=C10[i], lw=1.5, label='等权')
        ax_w.plot(dates_masked[-len(r_wt['wealth']):], r_wt['wealth'], color=C10[i], lw=1.5, ls='--', label='加权')
        ax_w.axhline(1.0, color='grey', ls='--', lw=0.5)
        ax_w.set_title(f'N={n_hold}  累计净值 (等权={r_eq["ann_ret"]:+.1%} vs 加权={r_wt["ann_ret"]:+.1%})')
        ax_w.legend(loc='best', fontsize=8)
        ax_w.grid(alpha=0.3)

        # 回撤对比
        ax_d.fill_between(dates_masked[-len(r_eq['drawdown']):], 0, r_eq['drawdown'] * 100,
                          color=C10[i], alpha=0.3, label='等权')
        ax_d.plot(dates_masked[-len(r_eq['drawdown']):], r_eq['drawdown'] * 100,
                  color=C10[i], lw=1)
        ax_d.plot(dates_masked[-len(r_wt['drawdown']):], r_wt['drawdown'] * 100,
                  color='#d62728', lw=1, ls='--', label='加权')
        ax_d.set_title(f'N={n_hold}  动态回撤 (等权={r_eq["max_dd"]:.1%} vs 加权={r_wt["max_dd"]:.1%})')
        ax_d.set_ylabel('回撤 (%)')
        ax_d.legend(loc='best', fontsize=8)
        ax_d.grid(alpha=0.3)
        ax_d.invert_yaxis()

    plt.suptitle('模拟盘测试 — 等权 vs 加权配置对比', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'paper_trading_results.png'), dpi=150)
    plt.close(fig)

    # =========== 汇总对比图 ===========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 净值对比
    for i, n_hold in enumerate(PORTFOLIO_SIZES):
        r_eq = results_equal[n_hold]
        r_wt = results_weighted[n_hold]
        ax = axes[0]
        ax.plot(dates_masked[-len(r_eq['wealth']):], r_eq['wealth'],
                label=f'N={n_hold} 等权 ({r_eq["ann_ret"]:+.1%})', color=C10[i], lw=1.5)
        ax.plot(dates_masked[-len(r_wt['wealth']):], r_wt['wealth'],
                label=f'N={n_hold} 加权 ({r_wt["ann_ret"]:+.1%})', color=C10[i], lw=1.5, ls='--')
    ax.axhline(1.0, color='grey', ls='--', lw=0.5)
    ax.set_title('累计净值对比 (实线=等权, 虚线=加权)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 回撤对比
    for i, n_hold in enumerate(PORTFOLIO_SIZES):
        r_eq = results_equal[n_hold]
        r_wt = results_weighted[n_hold]
        ax = axes[1]
        ax.plot(dates_masked[-len(r_eq['drawdown']):], r_eq['drawdown'] * 100,
                label=f'N={n_hold} 等权 ({r_eq["max_dd"]:.1%})', color=C10[i], lw=1)
        ax.plot(dates_masked[-len(r_wt['drawdown']):], r_wt['drawdown'] * 100,
                label=f'N={n_hold} 加权 ({r_wt["max_dd"]:.1%})', color=C10[i], lw=1, ls='--')
    ax.set_title('动态回撤对比 (实线=等权, 虚线=加权)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.invert_yaxis(); ax.set_ylabel('回撤 (%)')

    plt.suptitle('模拟盘汇总对比', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'paper_trading_comparison.png'), dpi=150)
    plt.close(fig)

    # =========== 导出买入清单 (参考 buy_list.json) ===========
    print(f"\n{'='*70}")
    print("导出买入清单...")

    # 加载债券名称映射
    bond_list = pd.read_csv(os.path.join(CB_DIR, 'QRdatabase', 'bond_list.csv'), dtype=str)
    code_to_name = dict(zip(bond_list['bond_code'], bond_list['bond_name']))
    code_to_short = dict(zip(bond_list['bond_code'], bond_list['bond_short_name']))
    code_to_stock = dict(zip(bond_list['bond_code'], bond_list['stock_name']))

    for n_hold in PORTFOLIO_SIZES:
        # 等权配置
        _, holdings_eq = simulate_long_only(composite, fwd_ret, valids, limit_up, n_hold)
        buy_list_eq = {}
        for d, top_indices in enumerate(holdings_eq):
            date_str = dates_masked[d].strftime('%Y-%m-%d')
            stocks = []
            for idx in top_indices:
                code = bonds[idx]
                stocks.append({
                    'code': code,
                    'name': code_to_short.get(code, code_to_name.get(code, code)),
                    'stock_name': code_to_stock.get(code, ''),
                    'weight': 1.0 / n_hold,
                })
            buy_list_eq[date_str] = {'date': dates_masked[d].strftime('%Y-%m-%d %H:%M:%S'), 'stocks': stocks}

        out_json_eq = os.path.join(OUT_DIR, f'buy_list_N{n_hold}_equal.json')
        with open(out_json_eq, 'w', encoding='utf-8') as f:
            json.dump(buy_list_eq, f, ensure_ascii=False, indent=2)
        print(f"  N={n_hold} [等权]: {len(buy_list_eq)} 天 → {out_json_eq}")

        # 加权配置
        _, holdings_wt, weights_wt = simulate_long_only_weighted(composite, fwd_ret, valids, limit_up, n_hold)
        buy_list_wt = {}
        for d, (top_indices, weights) in enumerate(zip(holdings_wt, weights_wt)):
            date_str = dates_masked[d].strftime('%Y-%m-%d')
            stocks = []
            for idx, w in zip(top_indices, weights):
                code = bonds[idx]
                stocks.append({
                    'code': code,
                    'name': code_to_short.get(code, code_to_name.get(code, code)),
                    'stock_name': code_to_stock.get(code, ''),
                    'weight': float(w),
                })
            buy_list_wt[date_str] = {'date': dates_masked[d].strftime('%Y-%m-%d %H:%M:%S'), 'stocks': stocks}

        out_json_wt = os.path.join(OUT_DIR, f'buy_list_N{n_hold}_weighted.json')
        with open(out_json_wt, 'w', encoding='utf-8') as f:
            json.dump(buy_list_wt, f, ensure_ascii=False, indent=2)
        print(f"  N={n_hold} [加权]: {len(buy_list_wt)} 天 → {out_json_wt}")

    # =========== 打印汇总 ===========
    print(f"\n{'='*70}")
    print("模拟盘汇总")
    print(f"{'='*70}")
    print("【等权配置】")
    print(f"{'持仓数':>6s} {'年化收益':>10s} {'IR':>8s} {'最大回撤':>10s} {'Calmar':>8s}")
    for n_hold in PORTFOLIO_SIZES:
        r = results_equal[n_hold]
        calmar = r['ann_ret'] / r['max_dd'] if r['max_dd'] > 0 else 0
        print(f"{n_hold:6d} {r['ann_ret']:+10.1%} {r['IR']:+8.2f} {r['max_dd']:+10.1%} {calmar:+8.2f}")

    print(f"\n【加权配置】")
    print(f"{'持仓数':>6s} {'年化收益':>10s} {'IR':>8s} {'最大回撤':>10s} {'Calmar':>8s}")
    for n_hold in PORTFOLIO_SIZES:
        r = results_weighted[n_hold]
        calmar = r['ann_ret'] / r['max_dd'] if r['max_dd'] > 0 else 0
        print(f"{n_hold:6d} {r['ann_ret']:+10.1%} {r['IR']:+8.2f} {r['max_dd']:+10.1%} {calmar:+8.2f}")

    print(f"\n【对比 (加权 - 等权)】")
    print(f"{'持仓数':>6s} {'年化收益差':>10s} {'IR差':>8s} {'最大回撤差':>10s}")
    for n_hold in PORTFOLIO_SIZES:
        r_eq = results_equal[n_hold]
        r_wt = results_weighted[n_hold]
        ann_diff = r_wt['ann_ret'] - r_eq['ann_ret']
        ir_diff = r_wt['IR'] - r_eq['IR']
        dd_diff = r_wt['max_dd'] - r_eq['max_dd']
        print(f"{n_hold:6d} {ann_diff:+10.1%} {ir_diff:+8.2f} {dd_diff:+10.1%}")

    # 保存汇总JSON
    summary = {'equal': {}, 'weighted': {}}
    for n_hold in PORTFOLIO_SIZES:
        r_eq = results_equal[n_hold]
        r_wt = results_weighted[n_hold]
        summary['equal'][f'N={n_hold}'] = {
            'ann_ret': float(r_eq['ann_ret']), 'IR': float(r_eq['IR']),
            'max_dd': float(r_eq['max_dd']),
            'final_wealth': float(r_eq['wealth'][-1]) if len(r_eq['wealth']) > 0 else 0,
        }
        summary['weighted'][f'N={n_hold}'] = {
            'ann_ret': float(r_wt['ann_ret']), 'IR': float(r_wt['IR']),
            'max_dd': float(r_wt['max_dd']),
            'final_wealth': float(r_wt['wealth'][-1]) if len(r_wt['wealth']) > 0 else 0,
        }
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果保存: {OUT_DIR}")
    print(f"耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
