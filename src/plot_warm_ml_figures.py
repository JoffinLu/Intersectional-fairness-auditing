"""Restyle all five figures from frozen source-data CSVs, without new experiments.

The 174 mm canvas matches the Sage manuscript. All quantitative observations,
interval endpoints and display offsets are read from retained source data.
Exports are editable vector PDF/SVG and 600-dpi PNG, with source-data hashes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import NullLocator

from warm_ml_style import (ROOT, WIDTH, COLORS, FILLS, apply_style, clean,
                           panel_header, method_handle, add_card, measure_alignment)


def figure3(data):
    """Absolute rates and paired contrasts from the same 2,000 outcomes."""
    assert len(data) == 60 and data.scenario_id.nunique() == 15
    assert (data.repetitions == 2000).all()
    methods = [('full_hb_holm', 'Full HB + Holm', 'full', 'o'),
               ('split_conflict_holm', 'Split conflict', 'conflict', 's'),
               ('split_raw_holm', 'Split raw', 'split_raw', '^'),
               ('split_eb_holm', 'Split EB', 'split_eb', 'D')]
    fig, grid = plt.subplots(2, 3, figsize=(WIDTH, 4.65), sharex=True, sharey='row',
                            gridspec_kw={'height_ratios': [1.25, 1]})
    fig.subplots_adjust(left=.105, right=.978, bottom=.18, top=.80,
                        wspace=.23, hspace=.49)
    for col, gap in enumerate((.10, .20, .30)):
        upper, lower = grid[:, col]
        for j, (method, label, key, marker) in enumerate(methods):
            z = data[(data.method == method) & np.isclose(data.effect_above_margin, gap)]
            z = z.sort_values('target_n')
            assert z.target_n.tolist() == [10, 20, 40, 80, 160]
            x = z.target_n.to_numpy() * 2 ** z.log2_absolute_display_offset.to_numpy()
            values = 100 * z.target_flag_rate.to_numpy()
            error = np.vstack([values-z.absolute_lower_pct, z.absolute_upper_pct-values])
            upper.errorbar(x, values, yerr=error, fmt=marker, ls='none',
                           color=COLORS[key], markerfacecolor='white' if j == 0 else FILLS[key],
                           markeredgewidth=.9, ms=4, capsize=1.7, elinewidth=.8, zorder=6-j)
            if j:
                x = z.target_n.to_numpy() * 2 ** z.log2_contrast_display_offset.to_numpy()
                values = z.paired_difference_pp.to_numpy()
                error = np.vstack([values-z.paired_lower_pp, z.paired_upper_pp-values])
                lower.errorbar(x, values, yerr=error, fmt=marker, ls='none',
                               color=COLORS[key], markerfacecolor=FILLS[key],
                               markeredgewidth=.9, ms=4, capsize=1.7,
                               elinewidth=.8, zorder=6-j)
        for ax in (upper, lower):
            ax.set(xscale='log', xlim=(7.3, 218))
            ax.set_xticks([10, 20, 40, 80, 160], labels=['10', '20', '40', '80', '160'])
            ax.xaxis.set_minor_locator(NullLocator())
            clean(ax)
        upper.set(ylim=(-3, 103), yticks=[0, 25, 50, 75, 100])
        lower.set(ylim=(-8.5, 6.5), yticks=[-8, -4, 0, 4])
        lower.axhline(0, color=COLORS['full'], lw=.85, zorder=2)
        panel_header(upper, 'abc'[col], f'ε = {gap:.2f}')
        panel_header(lower, 'def'[col], f'ε = {gap:.2f}')
    grid[0, 0].set_ylabel('Target flagging (%)')
    grid[1, 0].set_ylabel('Split − full (pp)')
    fig.legend(handles=[method_handle(key, label, marker)
                        for _, label, key, marker in methods], ncol=4,
               loc='upper center', bbox_to_anchor=(.535, .98),
               handletextpad=.45, columnspacing=1.3)
    fig.text(.543, .084, 'Target positive count n (log scale)', ha='center', fontsize=9)
    fig.text(.543, .025, 'Within-count marker offsets are for visibility only',
             ha='center', fontsize=8, color='#657181')
    return fig, list(grid.flat)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--figures', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--qa-helper', type=Path,
                        help='Optional directory containing an external panel-alignment auditor.')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (ROOT/'paper').mkdir(exist_ok=True)
    qa = out/'qa'
    qa.mkdir(exist_ok=True)
    apply_style()
    plt.rcParams.update({'font.family': 'sans-serif', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    width_mm = 174
    assert np.isclose(WIDTH * 25.4, width_mm)
    from warm_ml_estimation import figure1, figure2
    from warm_ml_robustness import figure4, figure5
    functions = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}
    expected_rows = {1: 151, 2: 16, 3: 60, 4: 24, 5: 16}
    sources = {}
    exports = {}
    for number in args.figures:
        if number not in functions:
            raise ValueError('Figure number must be between 1 and 5.')
        source = ROOT/f'figure_data/Fig{number}_data.csv'
        before = sha256(source)
        data = pd.read_csv(source)
        assert len(data) == expected_rows[number]
        if number == 2:
            metadata = pd.read_csv(ROOT/'analysis/point_screen_controls.csv')
            fig, axes = functions[number](data, metadata)
        else:
            fig, axes = functions[number](data)
        # Nunito has no Greek alphabet: mathematical symbols use STIX mathtext.
        for artist in fig.findobj(match=Text):
            text = artist.get_text()
            if '$' not in text:
                for symbol, command in [('κ', r'$\kappa$'), ('ε', r'$\varepsilon$'),
                                        ('ρ', r'$\rho$')]:
                    text = text.replace(symbol, command)
                artist.set_text(text)
        for ax in axes:
            add_card(fig, ax)
        measure_alignment(fig, axes, qa/f'Fig{number}.geometry.json')
        if args.qa_helper:
            sys.path.insert(0, str(args.qa_helper.resolve()))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(fig, axes=axes,
                panel_ids=[chr(97+i) for i in range(len(axes))],
                json_out=qa/f'Fig{number}.alignment.json', strict=True)
        fig.savefig(out/f'Fig{number}.pdf')
        fig.savefig(out/f'Fig{number}.svg')
        fig.savefig(out/f'Fig{number}.png', dpi=600)
        fig.savefig(out/f'Fig{number}_preview.png', dpi=300)
        plt.close(fig)
        assert before == sha256(source), 'Source data must remain byte-identical.'
        shutil.copy2(out/f'Fig{number}.pdf', ROOT/f'paper/Fig{number}.pdf')
        sources[source.relative_to(ROOT).as_posix()] = before
        exports[f'Fig{number}'] = {suffix: sha256(out/f'Fig{number}.{suffix}')
                                  for suffix in ('pdf', 'svg', 'png')}
        print(f'Fig{number}: {len(data)} source rows retained; vector PDF/SVG exported.')
    import matplotlib
    provenance = {'style': 'Warm ML; white canvas and panels; Nunito; pastel markers with dark edges',
                  'width_mm': 174, 'backend': 'Matplotlib', 'matplotlib_version': matplotlib.__version__,
                  'new_experiments': False, 'source_data_unchanged': True,
                  'source_data_sha256': sources, 'colors': COLORS, 'fills': FILLS,
                  'font_sha256': sha256(ROOT/'vendor/Nunito/Nunito-Variable.ttf'),
                  'uncertainty': 'Retained original pointwise Monte Carlo intervals and paired contrasts.',
                  'exports_sha256': exports}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
