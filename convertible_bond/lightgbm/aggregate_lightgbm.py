# -*- coding: utf-8 -*-
"""因子聚合 —— LightGBM 非线性聚合 (拓展窗口)。

核心思路:
  1. 将各因子值作为特征，隔夜收益作为目标，训练 LightGBM 回归模型
  2. 拓展窗口 (expanding window): 每天用 [0, t) 所有历史数据训练，预测第 t 天
  3. 模型的预测值即为聚合后的因子值
  4. 前 MIN_WINDOW 天数据不足，用 EW 作为预热

优势:
  - 能捕捉因子间的非线性交互
  - 自动学习各因子的最优权重（可能是非线性的）
  - 树模型天然处理特征共线性

与线性聚合对比:
  - IC/IC_IR 加权: 线性组合，权重基于历史 IC
  - 单变量回归: 线性组合，避免多元共线性
  - LightGBM: 非线性组合，可捕捉交互项

超参（偏保守以防过拟合）:
  - n_estimators=100, max_depth=4, min_child_samples=100
  - learning_rate=0.05, reg_alpha=0.1, reg_lambda=1.0
  - subsample=0.8, colsample_bytree=0.8
"""
import sys, os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
CB_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CB_DIR)
sys.path.insert(0, os.path.dirname(CB_DIR))

import config
import operators as op

# ── 全局参数 ──────────────────────────────────────────────
MIN_WINDOW = 252  # 最小训练窗口（天），用于样本内外分割

# 使用全部 19 个有效隔夜因子
FACTOR_NAMES = [
    '001_turnover', '002_morning_diff', '003_ov_momentum',
    '004_asym_ov', '005_tug_of_war', '006_rv_momentum',
    '007', '008', '009', '010',
    '011', '012', '013', '014', '015',
    '016', '017', '018', '019',
]

# LightGBM 参数
LGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'min_child_samples': 100,
    'learning_rate': 0.05,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
}


# ── 数据加载 ──────────────────────────────────────────────

def load_all_factors(alphas_dir, mask):
    """加载全部有效因子。返回 (n_bonds, n_days, n_factors) 数组。"""
    bonds, _ = op.load_meta()
    n_b, n_d = len(bonds), mask.sum()
    factors = np.full((n_b, n_d, len(FACTOR_NAMES)), np.nan, dtype=np.float32)
    for fi, fname in enumerate(FACTOR_NAMES):
        num = fname.split('_')[0]
        flist = [f for f in os.listdir(alphas_dir)
                 if f.startswith(f'cb_alpha{num}') and f.endswith('.py')]
        if not flist:
            print(f"  [WARN] 因子 {fname} (cb_alpha{num}) 未找到, 跳过")
            continue
        import importlib.util
        spec = importlib.util.spec_from_file_location(fname, os.path.join(alphas_dir, flist[0]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inst = getattr(mod, f'CBAlpha{int(num):03d}')()
        factors[:, :, fi] = inst.generate_factor()[:, mask]
    return factors


# ── 基础算子 ──────────────────────────────────────────────

def cs_rank(arr):
    """截面排序，映射到 [-1, 1]。"""
    out = np.full_like(arr, np.nan, dtype=float)
    for d in range(arr.shape[1]):
        col = arr[:, d]
        m = np.isfinite(col)
        n = m.sum()
        if n < 3:
            continue
        order = np.argsort(np.argsort(col[m]))
        out[np.where(m)[0], d] = order / (n - 1) * 2.0 - 1.0
    return out


# ── LightGBM 聚合 ─────────────────────────────────────────

def lightgbm_aggregate(factors_std, fwd_ret, valids, min_w=MIN_WINDOW):
    """拓展窗口 LightGBM 聚合。

    每天 t >= min_w:
      - 训练集: 所有 [0, t) 天中 valid 且非 NaN 的 (bond, day) 样本
      - 特征 X: (n_samples, n_factors) 因子值
      - 目标 y: (n_samples,) 隔夜收益
      - 预测: 对第 t 天所有 valid bond 预测，得到聚合因子值

    返回:
      lgb_factor:   (n_b, n_d)  LightGBM 预测聚合因子
      lgb_shallow:  (n_b, n_d)  浅层模型 (max_depth=2) 聚合因子（更保守的对照）
      feature_imp:  (n_d, n_f)  每日特征重要性 (gain)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[ERROR] 请先安装 lightgbm: pip install lightgbm")
        raise

    n_b, n_d, n_f = factors_std.shape
    lgb_factor = np.full((n_b, n_d), np.nan, dtype=np.float64)
    lgb_shallow = np.full((n_b, n_d), np.nan, dtype=np.float64)
    feature_imp = np.full((n_d, n_f), np.nan)

    # NaN → 0 (cs_rank 后 0 = 中性排名，不影响模型学习)
    X_all = np.nan_to_num(factors_std, nan=0.0)

    ew = np.nanmean(factors_std, axis=2)

    # 浅层模型参数（更保守）
    shallow_params = {**LGB_PARAMS, 'max_depth': 2, 'n_estimators': 50}

    n_trained = 0
    skip_count = {'no_data': 0, 'no_valid_pred': 0, 'train_error': 0}

    for t in range(min_w, n_d):
        # ── 构建训练集：pool [0, t) 所有有效样本 ──
        X_list, y_list = [], []
        for d in range(t):
            v = valids[:, d]
            if v.sum() < 10:
                continue
            X_d = X_all[v, d, :]       # (n_valid, n_f) — NaN 已填充为 0
            y_d = fwd_ret[v, d]        # (n_valid,)
            ok = np.isfinite(y_d)       # 仅过滤目标 NaN（特征已无 NaN）
            if ok.sum() < 10:
                continue
            X_list.append(X_d[ok])
            y_list.append(y_d[ok])

        if not X_list:
            skip_count['no_data'] += 1
            continue

        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)

        if len(X_train) < 500:
            skip_count['no_data'] += 1
            continue

        # ── 构建当日预测集 ──
        v_t = valids[:, t]
        if v_t.sum() < 10:
            skip_count['no_valid_pred'] += 1
            continue
        X_pred = X_all[v_t, t, :]  # NaN 已填充
        # 所有 valid bond 都可用于预测（特征已无 NaN）
        pred_idx = np.where(v_t)[0]

        # ── 训练 LightGBM ──
        try:
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train)],
                      eval_metric='l2')
        except Exception:
            skip_count['train_error'] += 1
            continue

        try:
            model_shallow = lgb.LGBMRegressor(**shallow_params)
            model_shallow.fit(X_train, y_train,
                              eval_set=[(X_train, y_train)],
                              eval_metric='l2')
        except Exception:
            model_shallow = None

        # ── 预测 ──
        lgb_factor[pred_idx, t] = model.predict(X_pred)

        if model_shallow is not None:
            lgb_shallow[pred_idx, t] = model_shallow.predict(X_pred)

        # ── 记录特征重要性 ──
        imp = model.feature_importances_
        feature_imp[t, :] = imp / imp.sum() if imp.sum() > 0 else imp

        n_trained += 1
        if n_trained % 50 == 0:
            train_samples = len(X_train)
            print(f"  已训练 {n_trained} 天 (day {t}/{n_d}, train_samples={train_samples:,})")

    # ── 前 min_w 天用 EW ──
    for t in range(min(min_w, n_d)):
        lgb_factor[:, t] = ew[:, t]
        lgb_shallow[:, t] = ew[:, t]

    # ── 缺失值用 EW 填充 ──
    for t in range(min_w, n_d):
        for arr in [lgb_factor, lgb_shallow]:
            mask_nan = np.isnan(arr[:, t])
            arr[mask_nan, t] = ew[mask_nan, t]

    print(f"  训练完成: 共 {n_trained} 个交易日")
    if skip_count['no_data']:
        print(f"    跳过(数据不足): {skip_count['no_data']}天")
    if skip_count['no_valid_pred']:
        print(f"    跳过(预测无有效券): {skip_count['no_valid_pred']}天")
    if skip_count['train_error']:
        print(f"    跳过(训练异常): {skip_count['train_error']}天")
    return lgb_factor, lgb_shallow, feature_imp


# ── 分层分析 ──────────────────────────────────────────────

def layered_analysis(name, agg_factor, fwd_ret, valids, dates_pd, out_dir):
    """分层收益分析（5层/10层/Top10%/Top5%）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    c10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    n_b, n_d = agg_factor.shape
    results = {}

    # ── 5层/10层 ──
    for n_q in [5, 10]:
        anss = []
        for q in range(1, n_q + 1):
            s = np.zeros((n_b, n_d))
            for d in range(n_d):
                df = agg_factor[:, d]
                vm = ~np.isnan(df) & valids[:, d]
                if vm.sum() < n_q * 2:
                    continue
                vf = df[vm]
                pct = 100 / n_q
                lo = np.percentile(vf, (q - 1) * pct)
                hi = np.percentile(vf, q * pct)
                lm = vm & (df >= lo) & (df < hi) if q < n_q else vm & (df >= lo)
                if lm.sum() > 0:
                    s[lm, d] = 1.0 / lm.sum()
            dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
            anss.append(np.nanmean(dr) * 250 if len(dr) > 0 else np.nan)
        results[f'{n_q}层'] = {'anns': anss}

        # 分层净值图
        fig, ax = plt.subplots(figsize=(12, 5))
        for qi in range(n_q):
            s = np.zeros((n_b, n_d))
            for d in range(n_d):
                df = agg_factor[:, d]
                vm = ~np.isnan(df) & valids[:, d]
                if vm.sum() < n_q * 2:
                    continue
                vf = df[vm]
                pct = 100 / n_q
                lo = np.percentile(vf, qi * pct)
                hi = np.percentile(vf, (qi + 1) * pct)
                lm = vm & (df >= lo) & (df < hi) if qi < n_q - 1 else vm & (df >= lo)
                if lm.sum() > 0:
                    s[lm, d] = 1.0 / lm.sum()
            dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
            ax.plot(dates_pd[-len(dr):], np.cumprod(1.0 + dr),
                    label=f'Q{qi+1}', color=c10[qi * (10 // n_q)], lw=1.5)
        ax.set_yscale('log')
        ax.set_title(f'{name} ({n_q}层)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{name}_{n_q}层.png'), dpi=150)
        plt.close(fig)

    # ── Top10% / Top5% ──
    for top_pct, tag in [(10, 'Top10%'), (5, 'Top5%')]:
        s = np.zeros((n_b, n_d))
        for d in range(n_d):
            df = agg_factor[:, d]
            vm = ~np.isnan(df) & valids[:, d]
            if vm.sum() < 20:
                continue
            lm = vm & (df >= np.percentile(df[vm], 100 - top_pct))
            if lm.sum() > 0:
                s[lm, d] = 1.0 / lm.sum()
        dr = np.array([np.nansum(s[:, t] * fwd_ret[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
        w = np.cumprod(1.0 + dr)
        ann = np.nanmean(dr) * 250 if len(dr) > 0 else 0
        dd = np.max(np.maximum.accumulate(w) - w) / np.max(np.maximum.accumulate(w))
        ir_val = np.nanmean(dr) / np.nanstd(dr) * np.sqrt(250) if np.nanstd(dr) > 0 else 0
        results[tag] = {'ann': ann, 'maxdd': dd, 'IR': ir_val}

    return results


# ── 主流程 ────────────────────────────────────────────────

def main():
    t0 = time.time()
    out_dir = os.path.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("因子聚合 —— LightGBM 非线性聚合 (拓展窗口)")
    print(f"因子数量: {len(FACTOR_NAMES)}")
    print("=" * 60)

    # ── 加载数据 ──
    bonds, dates_all = op.load_meta()
    dates_pd = pd.to_datetime(dates_all.astype(str))
    mask = dates_pd >= pd.to_datetime(config.BACKTEST_START_DATE)
    n_b, n_d = len(bonds), mask.sum()
    dates_masked = dates_pd[mask]
    print(f"区间: {dates_masked[0].strftime('%Y-%m-%d')} ~ {dates_masked[-1].strftime('%Y-%m-%d')}, {n_d}天")
    print(f"训练窗口: expanding, min_window={MIN_WINDOW}天")

    fwd = op.returns('overnight')[:, mask]
    v = op.valids('overnight')[:, mask]
    alphas_dir = os.path.join(CB_DIR, 'overnight', 'updates_alphas', 'effective')

    print("\n加载因子...")
    factors_raw = load_all_factors(alphas_dir, mask)

    print("标准化 (cs_rank)...")
    factors_std = np.zeros_like(factors_raw)
    for fi in range(len(FACTOR_NAMES)):
        factors_std[:, :, fi] = cs_rank(factors_raw[:, :, fi])

    ew = np.nanmean(factors_std, axis=2)

    # ── LightGBM 聚合 ──
    print(f"\nLightGBM 拓展窗口聚合 (min_window={MIN_WINDOW})...")
    t_lgb = time.time()
    lgb_factor, lgb_shallow, feature_imp = lightgbm_aggregate(
        factors_std, fwd, v, MIN_WINDOW
    )
    print(f"  耗时: {time.time() - t_lgb:.0f}s")

    agg = {
        'EW': ew,
        'LGB': lgb_factor,
        'LGB_Shallow': lgb_shallow,
    }

    # ── 分层分析 ──
    print("\n分层分析...")
    all_results = {}
    for m_name in agg:
        print(f"  {m_name}...")
        all_results[m_name] = layered_analysis(
            f'lgb_{m_name}', agg[m_name], fwd, v, dates_masked, out_dir
        )

    # ── 打印结果 ──
    print("\n" + "=" * 60)
    print("LightGBM vs EW vs LGB_Shallow")
    print("=" * 60)
    for m_name in agg:
        anns = all_results[m_name]['5层']['anns']
        t10 = all_results[m_name]['Top10%']
        t5 = all_results[m_name]['Top5%']
        tag = {'EW': 'EW           ', 'LGB': 'LGB          ',
               'LGB_Shallow': 'LGB_Shallow  '}[m_name]
        print(f"\n{tag}: Q1={anns[0]:+.1%} Q2={anns[1]:+.1%} Q3={anns[2]:+.1%} "
              f"Q4={anns[3]:+.1%} Q5={anns[4]:+.1%}")
        print(f"  Top10%: ann={t10['ann']:+.1%} DD={t10['maxdd']:.1%} IR={t10['IR']:.2f}")
        print(f"  Top5%:  ann={t5['ann']:+.1%} DD={t5['maxdd']:.1%} IR={t5['IR']:.2f}")

    # ── 特征重要性时序 ──
    print("\n绘制特征重要性时序...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 平均重要性排名
    imp_mean = np.nanmean(feature_imp[MIN_WINDOW:], axis=0)
    sorted_idx = np.argsort(imp_mean)[::-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.tab20
    for rank, j in enumerate(sorted_idx):
        s = pd.Series(feature_imp[:, j], index=dates_masked)
        s = s.dropna()
        if len(s) > 0:
            ax.plot(s.index, s.values, lw=0.6, alpha=0.7,
                    color=cmap(rank % 20),
                    label=f'{FACTOR_NAMES[j]} ({imp_mean[j]:.3f})')
    ax.set_title('LightGBM 特征重要性时序 (gain, 拓展窗口)')
    ax.legend(fontsize=6, ncol=3, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'lgb_feature_importance.png'), dpi=150)
    plt.close(fig)

    # 平均重要性柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [cmap(i % 20) for i in range(len(sorted_idx))]
    names_sorted = [FACTOR_NAMES[j] for j in sorted_idx]
    vals_sorted = [imp_mean[j] for j in sorted_idx]
    ax.barh(range(len(sorted_idx)), vals_sorted[::-1], color=colors[::-1])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(names_sorted[::-1])
    ax.set_xlabel('平均特征重要性 (gain)')
    ax.set_title('LightGBM 因子平均重要性排名')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'lgb_feature_importance_bar.png'), dpi=150)
    plt.close(fig)

    # ── 保存特征重要性 CSV ──
    df_imp = pd.DataFrame(feature_imp, columns=FACTOR_NAMES, index=dates_masked)
    df_imp.to_csv(os.path.join(out_dir, 'lgb_feature_importance.csv'), encoding='utf-8-sig')

    # ── 聚合净值对比图 ──
    print("绘制聚合净值对比...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top10% 对比
    for m_name, ls in [('EW', '-'), ('LGB', '-'), ('LGB_Shallow', '--')]:
        s = np.zeros((n_b, n_d))
        for d in range(n_d):
            df = agg[m_name][:, d]
            vm = ~np.isnan(df) & v[:, d]
            if vm.sum() < 20:
                continue
            lm = vm & (df >= np.percentile(df[vm], 90))
            if lm.sum() > 0:
                s[lm, d] = 1.0 / lm.sum()
        dr = np.array([np.nansum(s[:, t] * fwd[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
        cum = np.cumprod(1.0 + dr)
        axes[0].plot(dates_masked[-len(dr):], cum, ls, lw=1.5, label=m_name)
    axes[0].set_yscale('log')
    axes[0].set_title('Top10% 多头净值对比')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Top5% 对比
    for m_name, ls in [('EW', '-'), ('LGB', '-'), ('LGB_Shallow', '--')]:
        s = np.zeros((n_b, n_d))
        for d in range(n_d):
            df = agg[m_name][:, d]
            vm = ~np.isnan(df) & v[:, d]
            if vm.sum() < 20:
                continue
            lm = vm & (df >= np.percentile(df[vm], 95))
            if lm.sum() > 0:
                s[lm, d] = 1.0 / lm.sum()
        dr = np.array([np.nansum(s[:, t] * fwd[:, t]) for t in range(n_d) if np.sum(s[:, t]) > 0])
        cum = np.cumprod(1.0 + dr)
        axes[1].plot(dates_masked[-len(dr):], cum, ls, lw=1.5, label=m_name)
    axes[1].set_yscale('log')
    axes[1].set_title('Top5% 多头净值对比')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'lgb_nav_comparison.png'), dpi=150)
    plt.close(fig)

    # ── 汇总 CSV ──
    summary_rows = []
    for m_name in agg:
        res = all_results[m_name]
        for layer in ['5层', '10层']:
            anns = res[layer]['anns']
            row = {'method': m_name, 'layer': layer}
            for qi, ann in enumerate(anns):
                row[f'Q{qi+1}'] = f'{ann:+.4f}'
            summary_rows.append(row)
        for tag in ['Top10%', 'Top5%']:
            r = res[tag]
            summary_rows.append({
                'method': m_name, 'layer': tag,
                'ann': f"{r['ann']:+.4f}",
                'maxdd': f"{r['maxdd']:.4f}",
                'IR': f"{r['IR']:.2f}",
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, 'lgb_summary.csv'), encoding='utf-8-sig', index=False)

    elapsed = time.time() - t0
    print(f"\n完成 ({elapsed:.0f}s). 结果: {out_dir}")


if __name__ == '__main__':
    main()
