"""Physical-size typography and shared encodings for manuscript figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIDTH = 5.14  # 370.08 PDF points, essentially the actual 370.6-point text width.
# NPG-inspired family, with a darker cyan for white-background contrast.
# One mapping is shared by every quantitative panel in the manuscript.
PALETTE = {'raw': '#606B73', 'shrink': '#00A087', 'eb': '#00A087',
           'sr': '#8491B4', 'suremap': '#3B8EA5', 'full': '#3C5488',
           'conflict': '#E64B35', 'split_raw': '#606B73', 'split_eb': '#00A087',
           'difference': '#E64B35', 'heat_negative': '#E64B35',
           'heat_zero': '#F7F8FA', 'heat_positive': '#3C5488'}


def style():
    plt.rcParams.update({
        'font.family': 'Arial', 'mathtext.fontset': 'stix',
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 9.5,
        'axes.labelcolor': '#222222', 'text.color': '#222222',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'axes.edgecolor': '#777777', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'grid.color': '#E3E6E8', 'grid.linewidth': .55,
        'lines.linewidth': 1.5, 'lines.markersize': 4,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.bbox': None, 'figure.facecolor': 'white',
    })


def clean(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis)


def panel(ax, text):
    ax.set_title(text, loc='left', pad=9, fontweight='normal')


def ink_on(rgb):
    """Choose legible numeric-label ink from the actual heatmap fill."""
    def luminance(values):
        linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values[:3]]
        return sum(w*v for w,v in zip((.2126,.7152,.0722),linear))
    light=luminance(rgb)
    dark=luminance((17/255,17/255,17/255))
    return 'white' if 1.05/(light+.05) > (light+.05)/(dark+.05) else '#111111'
