# 用法：
#   python plot_training.py                        # 自动读 ./logs/ 下所有 csv
#   python plot_training.py logs/xxx_log.csv       # 指定文件

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, PercentFormatter

# ── 配置 ─────────────────────────────────────────────────────────
LOG_DIR     = "./logs/"
SMOOTH_WIN  = 80        # reward 平滑窗口
OUTPUT_DIR  = "./figures/"
DPI         = 300

# 学术风格配色（Colorblind-friendly）
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161']


def load_logs(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df['source'] = os.path.basename(p).replace('_log.csv', '')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def smooth(series, win):
    return series.rolling(window=win, min_periods=1, center=True).mean()


def fmt_M(x, _):
    return f'{x/1e6:.1f}M'


def fmt_K(x, _):
    if x >= 1000:
        return f'{x/1000:.0f}K'
    return str(int(x))


def setup_academic_style():
    """设置学术期刊风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  plot_reward_learning_curve 
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_learning_curve(df, filename="mean_reward_curve.png",
                                title_prefix="PPO"):
    """
    生成仿 DDQN 风格的 Reward 曲线：
    - 横坐标更长（全部训练步数，可用 xlim 参数延伸）
    - 原始 episode reward 显示为半透明浅蓝色 fill_between
    - 100-ep 滚动均值显示为深蓝色实线
    - First Clear 用红色虚线标注
    - best checkpoint 用红星标注
    """
    sources = df['source'].unique()
    fig, ax = plt.subplots(figsize=(10, 5))   

    for i, src in enumerate(sources):
        sub = df[df['source'] == src].copy().reset_index(drop=True)
        c_smooth = COLORS[i % len(COLORS)]
        c_raw    = '#AED6F1'   

        x = sub['step'].values

        # --- Episode Reward (raw fill) ---
        if 'ep_reward' in sub.columns:
            y_ep = sub['ep_reward'].values
            ax.fill_between(x, 0, y_ep, alpha=0.25, color=c_raw, linewidth=0)
            ax.plot(x, y_ep, color=c_raw, alpha=0.4, linewidth=0.5)

        # --- Mean Reward 100-ep (smooth) ---
        if 'mean_reward_100' in sub.columns:
            y_sm = smooth(sub['mean_reward_100'], SMOOTH_WIN).values
            ax.plot(x, y_sm, color=c_smooth, linewidth=2.2,
                    label=f'Mean Reward (100-ep rolling)', zorder=3)

        # --- First Clear marker ---
        if 'flag_get' in sub.columns:
            clears = sub[sub['flag_get'] == 1]
            if len(clears) > 0:
                first_step = int(clears['step'].iloc[0])
                first_ep   = int(clears['episode'].iloc[0]) if 'episode' in sub.columns else '?'
                ax.axvline(first_step, color='#E74C3C', alpha=0.85,
                           linewidth=1.8, linestyle='--',
                           label=f'First Clear  (Step {first_step/1000:.0f}K)',
                           zorder=4)

                # Subtle subsequent clears
                total_clears = len(clears)
                if total_clears > 80:
                    max_step = clears['step'].max()
                    interval = max(int(max_step / 60), 1)
                    clears = clears.copy()
                    clears['_bucket'] = (clears['step'] // interval).astype(int)
                    sampled = clears.drop_duplicates(subset='_bucket')
                else:
                    sampled = clears
                for _, row in sampled.iterrows():
                    if int(row['step']) == first_step:
                        continue
                    ax.axvline(row['step'], color='#E74C3C', alpha=0.08,
                               linewidth=0.5, linestyle='-', zorder=2)

        # --- Best checkpoint star ---
        if 'mean_reward_100' in sub.columns:
            best_step = 4423680
            best_row = sub[sub['step'] == best_step]
            if not best_row.empty: 
                best_rew = best_row['mean_reward_100'].iloc[0]
            else:
                # 若数据中没有该步数，可回退原逻辑
                best_idx = sub['mean_reward_100'].idxmax()
                best_step = sub.loc[best_idx, 'step']
                best_rew = sub.loc[best_idx, 'mean_reward_100']
            ax.plot(best_step, best_rew, marker='*', markersize=14,
                    color='#E74C3C', zorder=5,
                    label=f'best_eval.chkpt\nStep {best_step/1e6:.2f}M  |  Reward {best_rew:.0f}')
            ax.annotate(
                f'best_eval.chkpt\nStep {best_step/1e6:.2f}M | Reward {best_rew:.0f}',
                xy=(best_step, best_rew),
                xytext=(best_step + (x.max() - x.min()) * 0.05, best_rew * 0.85),
                fontsize=9,
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#BDBDBD', lw=0.8),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2),
            )

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Mean Reward (100-ep Rolling)')
    ax.set_title(f'{title_prefix} — Mean Reward vs Environment Steps\n'
                 'Super Mario Bros World 1-1', fontsize=13)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_M))

    # Raw data legend patch
    raw_patch = mpatches.Patch(color=c_raw, alpha=0.5, label='Episode Reward (raw)')
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=[raw_patch] + handles, labels=['Episode Reward (raw)'] + labels_leg,
              frameon=True, fancybox=False, edgecolor='#CCCCCC',
              loc='upper left', framealpha=0.9, fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, facecolor='white', edgecolor='none')
    print(f'[Saved] {out}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Stage Clear Rate 
# ─────────────────────────────────────────────────────────────────────────────
def plot_clear_rate(df, win=500, filename="clear_rate.png", title_prefix="PPO"):
    """
    计算并绘制 500-episode rolling clear rate（百分比），
    仿 DDQN Stage Clear Rate 图风格。
    需要列：episode (或 index), flag_get
    """
    if 'flag_get' not in df.columns:
        print("[Skip] 'flag_get' column not found — cannot plot clear rate")
        return

    sources = df['source'].unique()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, src in enumerate(sources):
        sub = df[df['source'] == src].copy().reset_index(drop=True)
        c = COLORS[i % len(COLORS)]

       
        if 'episode' in sub.columns:
            ep = sub['episode'].values
        else:
            ep = np.arange(len(sub))

        flags = sub['flag_get'].fillna(0).values
        cr = pd.Series(flags).rolling(window=win, min_periods=1).mean().values * 100

        # Fill area
        ax.fill_between(ep, 0, cr, alpha=0.18, color=c)
        ax.plot(ep, cr, color=c, linewidth=2.2,
                label=src if len(sources) > 1 else None)

        # First clear marker
        clear_idx = np.where(flags > 0)[0]
        if len(clear_idx) > 0:
            fc_ep = ep[clear_idx[0]]
            ax.axvline(fc_ep, color='#E74C3C', alpha=0.85,
                       linewidth=1.8, linestyle='--',
                       label=f'First Clear (Ep {fc_ep:,})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Clear Rate (%)')
    ax.set_title(f'{title_prefix} — Stage Clear Rate ({win}-ep Rolling)')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_K))
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC',
              loc='upper left', framealpha=0.9)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, facecolor='white', edgecolor='none')
    print(f'[Saved] {out}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Original single-metric plots (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def plot_single_metric(df, ycol, ylabel, title, filename,
                        show_raw=True, show_clear=True):
    sources = df['source'].unique()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, src in enumerate(sources):
        sub = df[df['source'] == src].copy()
        if ycol not in sub.columns:
            continue
        c = COLORS[i % len(COLORS)]
        x = sub['step'].values
        y = sub[ycol].values
        if show_raw:
            ax.plot(x, y, color=c, alpha=0.15, linewidth=0.6)
        y_sm = smooth(pd.Series(y), SMOOTH_WIN).values
        label_str = src if len(sources) > 1 else None
        ax.plot(x, y_sm, color=c, linewidth=2.0, label=label_str)

    if show_clear and ycol == 'mean_reward_100' and 'flag_get' in df.columns:
        clears = df[df['flag_get'] == 1].copy()
        if len(clears) > 0:
            first_step = int(clears['step'].iloc[0])
            first_ep = int(clears['episode'].iloc[0]) if 'episode' in df.columns else '?'
            ax.axvline(first_step, color='#E69F00', alpha=0.8,
                       linewidth=1.5, linestyle='--',
                       label=f'First Clear (Ep {first_ep})')
            total_clears = len(clears)
            if total_clears > 50:
                max_step = clears['step'].max()
                interval = max(int(max_step / 50), 1)
                clears['_bucket'] = (clears['step'] // interval).astype(int)
                sampled = clears.drop_duplicates(subset='_bucket')
            else:
                sampled = clears
            for _, row_data in sampled.iterrows():
                if int(row_data['step']) == first_step:
                    continue
                ax.axvline(row_data['step'], color='#E69F00', alpha=0.12,
                           linewidth=0.5, linestyle='-')

    ax.set_xlabel('Training Steps (Millions)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_M))
    if len(sources) > 1 or (show_clear and ycol == 'mean_reward_100'):
        ax.legend(frameon=True, fancybox=False, edgecolor='black',
                  loc='best', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, facecolor='white', edgecolor='none')
    print(f'[Saved] {out}')
    plt.close()


def generate_all_plots(df, algo_name="PPO"):
    metrics = [
        ('mean_reward_100', 'Mean Reward', 'Mean Reward (100-episode)', 'reward_mean100.png'),
        ('ep_reward',       'Episode Reward', 'Episode Reward',         'reward_episode.png'),
        ('x_pos',           'X Position',  'X Position per Episode',    'x_position.png'),
        ('entropy',         'Entropy',     'Policy Entropy',            'entropy.png'),
        ('value_loss',      'Value Loss',  'Value Loss',                'value_loss.png'),
        ('policy_loss',     'Policy Loss', 'Policy Loss',               'policy_loss.png'),
        ('approx_kl',       'Approx KL',  'Approx KL Divergence',      'approx_kl.png'),
        ('clip_fraction',   'Clip Fraction','PPO Clip Fraction',        'clip_fraction.png'),
        ('ep_length',       'Steps',       'Episode Length',            'episode_length.png'),
    ]

    print("\n[Generating individual plots...]")
    for ycol, ylabel, title, fname in metrics:
        if ycol in df.columns:
            plot_single_metric(df, ycol, ylabel, title, fname,
                               show_clear=(ycol == 'mean_reward_100'))
        else:
            print(f"[Skip] Column '{ycol}' not found")

    # ── NEW: DDQN-style reward curve ──────────────────────────────
    print("\n[Generating mean reward curve...]")
    plot_reward_learning_curve(df, filename="mean_reward_curve.png",
                           title_prefix=algo_name)

    # ── NEW: Clear rate plot ───────────────────────────────────────
    print("[Generating clear rate plot...]")
    plot_clear_rate(df, win=500, filename="clear_rate.png",
                    title_prefix=algo_name)

    # ── Compare if multiple sources ────────────────────────────────
    sources = df['source'].unique()
    if len(sources) > 1:
        print("\n[Generating comparison plots...]")
        key_metrics = ['mean_reward_100', 'entropy', 'value_loss', 'policy_loss']
        for ycol in key_metrics:
            if ycol in df.columns:
                for m in metrics:
                    if m[0] == ycol:
                        ylabel, title = m[1], m[2]
                        break
                plot_single_metric(df, ycol, ylabel, f'Comparison: {title}',
                                   f'compare_{ycol}.png')


def generate_summary_text(df):
    total_steps  = df['step'].max()
    total_eps    = len(df)
    total_clears = int(df['flag_get'].sum()) if 'flag_get' in df.columns else 0
    best_reward  = df['mean_reward_100'].max() if 'mean_reward_100' in df.columns else 0

    summary = f"""
Training Summary:
----------------
Total Steps: {total_steps/1e6:.2f}M
Total Episodes: {total_eps:,}
Best Mean Reward (100-ep): {best_reward:.2f}
Stage Clears: {total_clears:,}
Clear Rate: {total_clears/total_eps*100:.2f}%
"""
    print(summary)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'training_summary.txt'), 'w') as f:
        f.write(summary)
    print(f"[Saved] {OUTPUT_DIR}training_summary.txt")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob(os.path.join(LOG_DIR, '*_log.csv')))

    if not paths:
        print(f'[Error] No CSV files found in {LOG_DIR}')
        sys.exit(1)

    print(f'[Load] {len(paths)} file(s): {[os.path.basename(p) for p in paths]}')
    df = load_logs(paths)
    print(f'[Info] {len(df):,} episodes, steps: {df["step"].min():,} → {df["step"].max():,}')

    setup_academic_style()
    generate_all_plots(df, algo_name="PPO")
    generate_summary_text(df)

    print(f"\n[Done] All figures saved to: {OUTPUT_DIR}")
