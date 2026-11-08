"""Power-cost and real-support panels drawn from the retained figure tables.

The input tables provide all displayed rates, paired contrasts and Monte Carlo
standard errors.  These functions change presentation only and return their
quantitative axes to the common alignment and export workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from warm_ml_style import WIDTH, COLORS, FILLS, clean, panel_header, method_handle


def figure4(data):
    """Display all 24 paired power costs on one symmetric, zero-centred scale."""
    required = {
        'target_n', 'confirmation', 'floor', 'paired_difference',
        'paired_mcse', 'displayed_difference_pp',
    }
    assert required.issubset(data.columns)
    assert len(data) == 24
    assert not data.duplicated(['target_n', 'confirmation', 'floor']).any()
    assert np.isfinite(data[list(required)].to_numpy(dtype=float)).all()
    assert np.allclose(data.displayed_difference_pp, 100 * data.paired_difference,
                       rtol=0, atol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95), sharey=True)
    fig.subplots_adjust(left=.115, right=.875, bottom=.27, top=.765, wspace=.23)
    cmap = LinearSegmentedColormap.from_list(
        'warm_ml_signed_power',
        [COLORS['heat_negative'], COLORS['heat_zero'], COLORS['heat_positive']],
        N=257,
    )
    norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    levels = [.5, .8, .9]
    floors = [.05, .10, .50, 1.0]

    for ax, count, letter in zip(axes, (20, 80), 'ab'):
        subset = data[data.target_n == count]
        assert len(subset) == 12
        grid = subset.pivot(index='confirmation', columns='floor',
                            values='displayed_difference_pp').loc[levels, floors]
        values = grid.to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (np.abs(values) <= 40).all()
        image = ax.pcolormesh(
            np.arange(5) - .5, np.arange(4) - .5, values,
            cmap=cmap, norm=norm, edgecolors='white', linewidth=1.2,
            antialiased=True, rasterized=False,
        )
        ax.set(xlim=(-.5, 3.5), ylim=(2.5, -.5))
        ax.set_xticks(range(4), labels=['.05', '.10', '.50', '1.0'])
        ax.set_yticks(range(3), labels=['.5', '.8', '.9'])
        ax.tick_params(axis='both', which='both', length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(3):
            for column in range(4):
                ax.text(column, row, f'{values[row, column]:+.1f}',
                        ha='center', va='center', fontsize=9.5,
                        color='#293340', fontweight='semibold')
        panel_header(ax, letter, f'Target count n = {count}')

    axes[0].set_ylabel('Confirmation fraction c', labelpad=8)
    axes[1].tick_params(labelleft=False)
    color_axis = fig.add_axes([.912, .27, .018, .495])
    colorbar = fig.colorbar(image, cax=color_axis, ticks=[-40, -20, 0, 20, 40])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2, pad=4, labelsize=8.5)
    colorbar.solids.set_rasterized(False)
    fig.text(.495, .135, 'Uniform weight floor ρ', ha='center', fontsize=9)
    fig.text(.495, .045, 'Paired power difference: split conflict − full (pp)',
             ha='center', fontsize=9)
    return fig, list(axes)


def figure5(data):
    """Keep absolute power and paired differences aligned across all 16 settings."""
    required = {
        'scenario_id', 'dataset', 'model', 'geometry', 'y', 'full_power',
        'split_power', 'full_mcse', 'split_mcse', 'paired_difference', 'paired_mcse',
    }
    assert required.issubset(data.columns)
    assert len(data) == 16 and data.scenario_id.nunique() == 16
    numeric = ['y', 'full_power', 'split_power', 'full_mcse', 'split_mcse',
               'paired_difference', 'paired_mcse']
    assert np.isfinite(data[numeric].to_numpy(dtype=float)).all()
    assert ((data[['full_power', 'split_power']] >= 0) &
            (data[['full_power', 'split_power']] <= 1)).all().all()
    assert (data[['full_mcse', 'split_mcse', 'paired_mcse']] >= 0).all().all()
    assert np.allclose(data.paired_difference, data.split_power - data.full_power,
                       rtol=0, atol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.0), sharey=True)
    fig.subplots_adjust(left=.235, right=.975, bottom=.13, top=.825, wspace=.24)
    records = data.sort_values('y', kind='stable')
    datasets = [
        ('adult', 'Adult'), ('diabetes', 'Diabetes'),
        ('acs_income', 'ACS Income'), ('acs_employment', 'ACS Employment'),
    ]
    labels = []
    ticks = []
    for group, (key, title) in enumerate(datasets):
        subset = records[records.dataset == key]
        assert len(subset) == 4
        assert subset.y.tolist() == [5 * group + j for j in (1, 2, 3, 4)]
        ticks.append(5 * group)
        labels.append(title)
        for row in subset.itertuples(index=False):
            assert row.model in {'lr', 'hgb'}
            assert row.geometry in {'homogeneous', 'calibration_heterogeneous'}
            ticks.append(row.y)
            labels.append(row.model.upper() + ' · ' +
                          ('H' if row.geometry == 'homogeneous' else 'V'))

            for rate, error, method, marker, shift in (
                (row.full_power, row.full_mcse, 'full', 'o', -.12),
                (row.split_power, row.split_mcse, 'conflict', 's', .12),
            ):
                axes[0].errorbar(
                    100 * rate, row.y + shift, xerr=196 * error,
                    fmt=marker, ms=4.1, color=COLORS[method],
                    markerfacecolor='white' if method == 'full' else FILLS[method],
                    markeredgecolor=COLORS[method], markeredgewidth=.9,
                    capsize=2, elinewidth=.9, capthick=.8, zorder=3,
                )
            axes[1].errorbar(
                100 * row.paired_difference, row.y, xerr=196 * row.paired_mcse,
                fmt='D', ms=4.1, color=COLORS['difference'],
                markerfacecolor=FILLS['difference'],
                markeredgecolor=COLORS['difference'], markeredgewidth=.9,
                capsize=2, elinewidth=.9, capthick=.8, zorder=3,
            )

    for ax in axes:
        ax.set(ylim=(19.8, -.75))
        clean(ax, 'x')
        ax.tick_params(axis='y', length=0, pad=7)
        for boundary in (4.6, 9.6, 14.6):
            ax.axhline(boundary, color='#E7EBEF', linewidth=.6, zorder=0)

    axes[0].set_yticks(ticks, labels=labels)
    group_titles = {label for _, label in datasets}
    for label in axes[0].get_yticklabels():
        if label.get_text() in group_titles:
            label.set_fontweight('bold')
            label.set_fontsize(9)
            label.set_color('#293340')
    axes[1].tick_params(labelleft=False)
    axes[0].set(xlim=(0, 8), xticks=[0, 2, 4, 6, 8],
                xlabel='Target flagging (%)')
    axes[1].set(xlim=(-1, 5), xticks=[0, 2, 4],
                xlabel='Split − full (pp)')
    axes[1].axvline(0, color=COLORS['raw'], linewidth=.85,
                    linestyle=(0, (3, 3)), zorder=1)
    panel_header(axes[0], 'a', 'Absolute target power')
    panel_header(axes[1], 'b', 'Paired difference')
    handles = [method_handle('full', 'Full HB + Holm', 'o'),
               method_handle('conflict', 'Split conflict', 's')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.595, .975),
               ncol=2, columnspacing=2.0, handletextpad=.65)
    return fig, list(axes)
