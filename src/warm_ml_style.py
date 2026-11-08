"""Shared physical-size typography and semantic encodings for the five figures."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 174 / 25.4
COLORS = {
    'raw': '#707B8B', 'shrink': '#238F80', 'eb': '#238F80',
    'sr': '#8971B3', 'suremap': '#4D85AA', 'full': '#4D85AA',
    'conflict': '#CD626F', 'split_raw': '#707B8B', 'split_eb': '#238F80',
    'difference': '#CD626F', 'heat_negative': '#DC929A',
    'heat_zero': '#FFFFFF', 'heat_positive': '#8DC4B4',
}
FILLS = {
    'raw': '#D8DEE7', 'shrink': '#C8E6DF', 'eb': '#C8E6DF',
    'sr': '#E1BEE7', 'suremap': '#BBDEFB', 'full': '#BBDEFB',
    'conflict': '#FFD0D0', 'split_raw': '#D8DEE7', 'split_eb': '#C8E6DF',
    'difference': '#FFD0D0', 'heat_negative': '#FFD0D0',
    'heat_zero': '#FFFFFF', 'heat_positive': '#C8E6DF',
}


def _register_nunito():
    """Instantiate weight-specific local caches from the licensed variable font."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    source = ROOT / 'vendor/Nunito/Nunito-Variable.ttf'
    if not source.is_file():
        raise FileNotFoundError('The bundled Nunito font and its OFL license are required.')
    cache = ROOT / '.cache/fonts'
    cache.mkdir(parents=True, exist_ok=True)
    for weight, name in ((400, 'Regular'), (600, 'SemiBold'), (700, 'Bold')):
        target = cache / f'Nunito-{name}.ttf'
        if not target.is_file():
            font = instantiateVariableFont(TTFont(source), {'wght': weight},
                                           inplace=True, updateFontNames=True)
            replacement = {1: 'Nunito', 2: name,
                           3: f'NunitoFigure-{name}', 4: f'Nunito {name}',
                           6: f'NunitoFigure-{name}', 16: 'Nunito', 17: name}
            for record in list(font['name'].names):
                if record.nameID in replacement:
                    font['name'].setName(replacement[record.nameID], record.nameID,
                                         record.platformID, record.platEncID, record.langID)
            font.save(target)
        font_manager.fontManager.addfont(str(target))


def apply_style():
    _register_nunito()
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Nunito', 'DejaVu Sans'],
        'mathtext.fontset': 'stix', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 9,
        'text.color': '#293340', 'axes.labelcolor': '#293340',
        'xtick.color': '#526071', 'ytick.color': '#526071',
        'axes.edgecolor': '#AAB3BF', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': '#FFFFFF', 'figure.facecolor': '#FFFFFF',
        'savefig.facecolor': '#FFFFFF', 'savefig.bbox': None,
        'grid.color': '#E8EBEF', 'grid.linewidth': .55,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'lines.linewidth': 1.7, 'lines.markersize': 4.3,
        'legend.frameon': False, 'legend.handlelength': 1.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def clean(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)


def panel_header(ax, letter, title, color='full'):
    """Fixed-point anchors keep labels aligned across different panel heights."""
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(0, 13), textcoords='offset points',
                ha='left', va='baseline', fontsize=10, fontweight='bold',
                color=COLORS[color], annotation_clip=False,
                bbox={'boxstyle': 'round,pad=.16,rounding_size=.18',
                      'facecolor': FILLS[color], 'edgecolor': 'none'})
    ax.annotate(title, (0, 1), xycoords='axes fraction', xytext=(15, 13), textcoords='offset points',
                ha='left', va='baseline', fontsize=10, fontweight='semibold',
                annotation_clip=False)


def method_handle(key, label, marker='o', linestyle='none'):
    return Line2D([], [], label=label, color=COLORS[key], marker=marker,
                  linestyle=linestyle, markersize=4.7, markeredgewidth=1,
                  markerfacecolor='white' if key == 'full' else FILLS[key])


def add_card(fig, ax):
    """A barely raised white panel, with no colored panel background."""
    p = ax.get_position()
    width_pt, height_pt = fig.get_size_inches() * 72
    xpad, bottom, top = 1 / width_pt, 1 / height_pt, 29 / height_pt
    card = FancyBboxPatch((p.x0 - xpad, p.y0 - bottom), p.width + 2*xpad,
                          p.height + bottom + top, transform=fig.transFigure,
                          boxstyle='round,pad=.004,rounding_size=.016',
                          facecolor='#FFFFFF', edgecolor='none', zorder=-2)
    card.set_path_effects([patheffects.SimplePatchShadow(offset=(0, -.65),
                                                       shadow_rgbFace='#536072', alpha=.055),
                           patheffects.Normal()])
    fig.add_artist(card)


def measure_alignment(fig, axes, destination):
    """Check plot rectangles in physical points; retain their measured geometry."""
    fig.canvas.draw()
    width, height = fig.get_size_inches() * 72
    boxes = []
    for i, ax in enumerate(axes):
        p = ax.get_position()
        boxes.append({'panel': chr(97+i),
                      'bbox_pt': [p.x0*width, p.y0*height, p.x1*width, p.y1*height]})
    deviations = []
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            aa, bb = a['bbox_pt'], b['bbox_pt']
            if abs(aa[1]-bb[1]) < 2 and abs(aa[3]-bb[3]) < 2:
                deviations.extend([abs(aa[1]-bb[1]), abs(aa[3]-bb[3])])
            if abs(aa[0]-bb[0]) < 2 and abs(aa[2]-bb[2]) < 2:
                deviations.extend([abs(aa[0]-bb[0]), abs(aa[2]-bb[2])])
    assert deviations and max(deviations) <= 1.5, 'Panel edges must align within 1.5 pt.'
    report = {'width_pt': float(width), 'height_pt': float(height), 'panels': boxes,
              'tolerance_pt': 1.5, 'maximum_shared_edge_deviation_pt': max(deviations)}
    Path(destination).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
