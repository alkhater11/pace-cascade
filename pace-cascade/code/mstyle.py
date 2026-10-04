"""mstyle.py -- MATLAB-like Matplotlib style shared by all figure scripts."""
import matplotlib as mpl
from cycler import cycler

# MATLAB default colour order (R2014b+)
MATLAB = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": "#262626", "axes.labelcolor": "#262626",
        "axes.grid": True, "grid.color": "#dcdcdc", "grid.linestyle": "-", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.prop_cycle": cycler(color=MATLAB),
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": True, "legend.edgecolor": "#262626", "legend.fancybox": False, "legend.framealpha": 1.0,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
