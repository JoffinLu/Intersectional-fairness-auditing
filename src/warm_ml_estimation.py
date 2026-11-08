"""Estimation and screening figures drawn from retained numerical exports."""

import numpy as np
import matplotlib.pyplot as plt

from warm_ml_style import (
    WIDTH,
    COLORS,
    FILLS,
    clean,
    panel_header,
    method_handle,
)


def figure1(data):
    """Show exact fixed-prior risk and screening without changing the curves."""
    required = {
        'kappa', 'raw_macro_mse', 'shrink_macro_mse',
        'raw_target_screen_probability', 'shrink_target_screen_probability',
    }
    if not required.issubset(data.columns) or len(data) != 151:
        raise ValueError('Figure 1 requires all 151 retained concentration values.')
    values = data.sort_values('kappa')
    kappa = values.kappa.to_numpy()
    if not np.array_equal(kappa, np.arange(151)):
        raise ValueError('The retained Figure 1 concentration grid must be 0 to 150.')
    if not np.isfinite(values[list(required)].to_numpy(dtype=float)).all():
        raise ValueError('Figure 1 contains a non-finite value.')

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95))
    fig.subplots_adjust(left=.105, right=.98, bottom=.225, top=.755,
                        wspace=.43)

    for ax in axes:
        ax.set(xlim=(-3, 153), xlabel='Prior concentration κ')
        ax.set_xticks([0, 50, 100, 150])
        ax.axvline(90, color=COLORS['raw'], linewidth=.9,
                   linestyle=(0, (1.2, 2.5)), alpha=.72, zorder=1)
        clean(ax)

    axes[0].plot(kappa, 1000 * values.raw_macro_mse,
                 color=COLORS['raw'], linewidth=1.55,
                 linestyle=(0, (4.5, 2.4)), zorder=2)
    axes[0].plot(kappa, 1000 * values.shrink_macro_mse,
                 color=COLORS['shrink'], linewidth=1.85, zorder=3)
    axes[0].set(ylabel='Macro MSE × 1,000', ylim=(5.5, 12.5),
                yticks=[6, 8, 10, 12])
    risk90 = float(values.loc[values.kappa.eq(90), 'shrink_macro_mse'].iloc[0])
    axes[0].plot(90, 1000 * risk90, marker='o', markersize=5,
                 markerfacecolor=FILLS['shrink'], markeredgecolor=COLORS['shrink'],
                 markeredgewidth=1.2, linestyle='none', zorder=5)
    axes[0].annotate(
        f'κ = 90: {1000 * risk90:.2f}', xy=(90, 1000 * risk90),
        xytext=(5, 10.72), fontsize=8.8, color=COLORS['raw'],
        arrowprops={'arrowstyle': '->', 'linewidth': .75,
                    'color': COLORS['raw'], 'shrinkA': 4, 'shrinkB': 4},
        zorder=6,
    )

    axes[1].plot(kappa, 100 * values.raw_target_screen_probability,
                 color=COLORS['raw'], linewidth=1.55,
                 linestyle=(0, (4.5, 2.4)), zorder=2)
    axes[1].plot(kappa, 100 * values.shrink_target_screen_probability,
                 color=COLORS['shrink'], linewidth=1.85, zorder=3)
    axes[1].set(ylabel='Target screen probability (%)', ylim=(-3, 103),
                yticks=[0, 25, 50, 75, 100])
    screen90 = float(values.loc[
        values.kappa.eq(90), 'shrink_target_screen_probability'].iloc[0])
    axes[1].plot(90, 100 * screen90, marker='o', markersize=5,
                 markerfacecolor=FILLS['shrink'], markeredgecolor=COLORS['shrink'],
                 markeredgewidth=1.2, linestyle='none', zorder=5)
    axes[1].annotate(
        f'κ = 90: {100 * screen90:g}%', xy=(90, 100 * screen90),
        xytext=(96, 32), fontsize=8.8, color=COLORS['raw'],
        arrowprops={'arrowstyle': '->', 'linewidth': .75,
                    'color': COLORS['raw'], 'shrinkA': 4, 'shrinkB': 4},
        zorder=6,
    )

    panel_header(axes[0], 'a', 'Average estimation risk')
    panel_header(axes[1], 'b', 'Harmed-cell screening')
    fig.legend(
        handles=[
            method_handle('raw', 'Raw proportions', marker='', linestyle='--'),
            method_handle('shrink', 'Fixed-prior shrinkage', marker='', linestyle='-'),
        ],
        loc='upper center', bbox_to_anchor=(.54, .985), ncol=2,
        frameon=False, handlelength=2.9, columnspacing=2.1,
        handletextpad=.7,
    )
    return fig, list(axes)


def figure2(data, metadata):
    """Show the four retained settings with their original uncertainty bounds."""
    methods = [
        ('raw', 'Raw', 'raw', 'o'),
        ('moment_eb', 'Moment EB', 'eb', 's'),
        ('sr_point_adapter', 'SR point', 'sr', '^'),
        ('suremap_author_adapter', 'SureMap', 'suremap', 'D'),
    ]
    required = {
        'scenario_id', 'method', 'x', 'macro_mse', 'macro_mse_mcse',
        'target_screen_probability', 'screen_wilson_lower',
        'screen_wilson_upper', 'repetitions',
    }
    if not required.issubset(data.columns) or len(data) != 16:
        raise ValueError('Figure 2 requires all 16 retained setting/method rows.')
    if set(data.method) != {entry[0] for entry in methods}:
        raise ValueError('Figure 2 method identities do not match the retained design.')
    if data.duplicated(['scenario_id', 'method']).any():
        raise ValueError('Figure 2 contains duplicate setting/method rows.')
    numeric = data[list(required - {'scenario_id', 'method'})].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not data.repetitions.eq(120).all():
        raise ValueError('Figure 2 requires finite retained values from 120 repetitions.')

    order = data.loc[data.method.eq('raw')].sort_values('x').scenario_id.tolist()
    if len(order) != 4 or len(set(order)) != 4:
        raise ValueError('Figure 2 requires four distinct ordered settings.')
    meta = metadata
    if 'family' in meta:
        meta = meta.loc[meta.family.eq('sparse_rare')]
    meta = meta[['scenario_id', 'target_n', 'effect_above_margin']].drop_duplicates()
    if meta.scenario_id.duplicated().any():
        raise ValueError('Figure 2 setting metadata are ambiguous.')
    meta = meta.set_index('scenario_id').loc[order]
    ticks = [f'n = {int(row.target_n)}\n' + r'$\varepsilon$' + f' = {row.effect_above_margin:.2f}'
             for row in meta.itertuples()]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.35))
    fig.subplots_adjust(left=.105, right=.98, bottom=.295, top=.77,
                        wspace=.43)

    for method, label, key, marker in methods:
        values = data.loc[data.method.eq(method)].set_index('scenario_id').loc[order]
        if len(values) != 4:
            raise ValueError(f'Figure 2 is missing a retained row for {method}.')
        x = values.x.to_numpy()
        point = dict(
            fmt=marker, linestyle='none', markersize=4.8,
            markerfacecolor=FILLS[key], markeredgecolor=COLORS[key],
            markeredgewidth=1.05, color=COLORS[key],
            elinewidth=1.05, capsize=2.5, capthick=.9, zorder=4,
        )
        axes[0].errorbar(
            x, 1000 * values.macro_mse,
            yerr=1960 * values.macro_mse_mcse,
            **point,
        )
        probability = values.target_screen_probability.to_numpy()
        lower = values.screen_wilson_lower.to_numpy()
        upper = values.screen_wilson_upper.to_numpy()
        if np.any(lower > probability) or np.any(upper < probability):
            raise ValueError('Figure 2 Wilson intervals do not contain their estimates.')
        axes[1].errorbar(
            x, 100 * probability,
            yerr=100 * np.array([probability - lower, upper - probability]),
            **point,
        )

    for ax in axes:
        ax.set_xticks(range(4), labels=ticks)
        ax.tick_params(axis='x', pad=6, labelsize=8.2)
        ax.set(xlim=(-.5, 3.5), xlabel='Target count n and excess ε')
        ax.xaxis.labelpad = 9
        clean(ax)
    axes[0].set(ylabel='Macro MSE × 1,000', ylim=(0, 4.4),
                yticks=[0, 1, 2, 3, 4])
    axes[1].set(ylabel='Target screen probability (%)', ylim=(-3, 103),
                yticks=[0, 25, 50, 75, 100])
    panel_header(axes[0], 'a', 'Average estimation risk')
    panel_header(axes[1], 'b', 'Target screening')
    fig.legend(
        handles=[method_handle(key, label, marker=marker)
                 for _, label, key, marker in methods],
        loc='upper center', bbox_to_anchor=(.54, .985), ncol=4,
        frameon=False, columnspacing=1.9, handlelength=1.5,
        handletextpad=.5,
    )
    return fig, list(axes)
