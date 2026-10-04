"""Revision figures: redesigned schematic + new-experiment figures (EPS+PDF)."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
import mstyle; mstyle.apply()   # MATLAB-like style (second revision)

DSN = {"mnist": "MNIST", "fmnist": "Fashion-MNIST"}
COL = {"static": "#77AC30", "pace": "#D95319", "int8": "#0072BD", "fp32": "#000000",
       "cnn": "#A2142F", "mem": "#7E2F8E", "env": "#D95319"}
def save(fig, name):
    for ext in ("eps", "pdf"): fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)

# ---------- Fig8 (schematic) redesigned: clearer, professional ----------
fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.axis("off")
ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
def box(x, y, w, h, text, fc, ec="#333333", fs=8.2, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                 fc=fc, ec=ec, lw=lw)); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2, txt=None, col="#333333", fs=7.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                 lw=1.1, color=col, connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((x1+x2)/2, (y1+y2)/2, txt, ha="center", va="center", fontsize=fs,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
# input
box(0.15, 2.1, 1.25, 1.0, "input\n$x$", "#eaeaf5")
# cheap config block
box(1.95, 1.95, 2.35, 1.3, "cheap config $C_{\\mathrm{lo}}$\n(per-layer $b_i,t_i$)\n"
    r"energy $E_{\mathrm{lo}}$", "#ffe9d6", fs=8.0)
# gate
box(4.95, 1.95, 2.05, 1.3, "confidence\ngate $g(x)$\n"r"vs. threshold $\tau$", "#e2f3e2", fs=8.0)
# accept
box(8.05, 3.35, 3.2, 1.15, "ACCEPT  $\\hat{y}_{C_{\\mathrm{lo}}}$\n"r"cost $E_{\mathrm{lo}}$  (majority)", "#e6f0fb", fs=8.0)
# escalate
box(8.05, 0.75, 3.2, 1.15, "ESCALATE $\\to C_{\\mathrm{hi}}$: $\\hat{y}_{C_{\\mathrm{hi}}}$\n"
    r"cost $E_{\mathrm{lo}}+E_{\mathrm{hi}}$  (few)", "#fde6e6", fs=8.0)
arr(1.40, 2.6, 1.95, 2.6)
arr(4.30, 2.6, 4.95, 2.6)
arr(7.00, 2.9, 8.05, 3.9, r"$g(x)\geq\tau$", col="#2a7a2a", rad=0.15)
arr(7.00, 2.3, 8.05, 1.3, r"$g(x)<\tau$", col="#b33", rad=-0.15)
# feedback loop for escalation (dashed): escalate re-runs high-precision on same x
ax.add_patch(FancyArrowPatch((9.65, 0.75), (3.1, 1.5), arrowstyle="-|>", mutation_scale=10,
             lw=0.9, color="#b33", ls=(0, (4, 2)), connectionstyle="arc3,rad=0.32"))
ax.text(6.0, 0.30, "escalated input re-processed at higher precision (cheap pass already paid)",
        ha="center", fontsize=6.8, color="#b33", style="italic")
# energy annotation
ax.text(6.0, 5.0, r"Expected energy per sample:  $\mathbb{E}[E]=E_{\mathrm{lo}}+p_{\mathrm{esc}}\,E_{\mathrm{hi}}$,"
        r"   $p_{\mathrm{esc}}=\Pr[g(x)<\tau]$", ha="center", fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fbfbe7", ec="#cccc99"))
# newcommand-free math fallbacks
save(fig, "Fig8")

# ---------- Fig9: multi-pair envelope vs static frontier ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, ds in zip(axes, DSN):
    E = json.load(open(os.path.join(OUT, f"multipair_env_{ds}.json")))
    R = json.load(open(os.path.join(OUT, "revision_r4.json")))[ds]["multipair"]
    # all pair curves faint
    for p in R["pairs"]:
        c = np.array(p["curve"])  # (E_nJ, acc, esc)
        ax.plot(c[:, 0], c[:, 1]*100, color="#f0b0b0", lw=0.5, alpha=0.5, zorder=1)
    front = sorted(E["front_test"])
    fe = np.array([e for e, a in front]); fa = np.array([a for e, a in front])
    ax.plot(fe, fa*100, "-o", color=COL["static"], ms=3, lw=1.4, label="static Pareto frontier", zorder=4)
    grid = np.array(E["grid_acc"]); env = np.array(E["cascade_env_nJ"])
    good = np.isfinite(env)
    ax.plot(env[good], grid[good]*100, "-", color=COL["env"], lw=1.8,
            label="best cascade envelope\n(over 36 pairs)", zorder=5)
    ax.set_title(DSN[ds]); ax.set_xlabel("energy (nJ / sample)"); ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(right=min(75, fe.max()*1.4))
fig.tight_layout(); save(fig, "Fig9")

# ---------- Fig10: CNN results (energy bars, both datasets) ----------
cnn = json.load(open(os.path.join(OUT, "cnn_results.json")))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, ds in zip(axes, DSN):
    h = cnn[ds]["headline"]
    names = ["FP32", "INT8", "static MP", "PACE"]
    en = [h["fp32_energy_pj"]/1000, h["int8_energy_pj"]/1000, h["static_energy_pj"]/1000, h["pace_energy_pj"][0]/1000]
    err = [0, 0, 0, h["pace_energy_pj"][1]/1000]
    cols = [COL["fp32"], COL["int8"], COL["static"], COL["pace"]]
    bars = ax.bar(names, en, yerr=err, color=cols, capsize=3, width=0.62)
    ax.set_yscale("log"); ax.set_title(f"{DSN[ds]} (CNN)"); ax.set_ylabel("energy (nJ / sample, log)")
    for b, v in zip(bars, en):
        ax.text(b.get_x()+b.get_width()/2, v*1.09, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="x", visible=False)
    # annotate accuracies
    accs = [h["fp32_acc"][0], h["int8_acc"][0], h["static_acc"][0], h["pace_acc"][0]]
    ax.text(0.5, 0.94, "acc: " + " / ".join(f"{a*100:.1f}" for a in accs),
            transform=ax.transAxes, ha="center", fontsize=6.6, color="#555")
fig.tight_layout(); save(fig, "Fig10")

# ---------- Fig11: energy-model fair comparison (compute vs memory-augmented) ----------
fair = json.load(open(os.path.join(OUT, "energy_fair.json")))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
labels = ["compute-bound (mu=0)", "memory-augmented (mu=0.15)"]
short = ["compute-only", "memory-aug."]
for ax, ds in zip(axes, DSN):
    x = np.arange(len(labels)); w = 0.36
    stat = [fair[ds][l]["static_nJ"] for l in labels]
    pace = [fair[ds][l]["pace_nJ"] for l in labels]
    b1 = ax.bar(x-w/2, stat, w, color=COL["static"], label="static MP (re-selected)")
    b2 = ax.bar(x+w/2, pace, w, color=COL["pace"], label="PACE")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title(f"{DSN[ds]}: energy model"); ax.set_ylabel("energy (nJ / sample, log)")
    for bb in (b1, b2):
        for b in bb:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.06, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=6.6)
    ax.grid(axis="x", visible=False); ax.legend(loc="upper left", fontsize=6.8)
fig.tight_layout(); save(fig, "Fig11")

# ---------- Fig12: PACE workflow flowchart (offline design + online) ----------
fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 7)
def fbox(x, y, w, h, t, fc, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                 fc=fc, ec="#333", lw=0.9)); ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs)
def fdia(cx, cy, w, h, t, fc):  # decision diamond
    ax.add_patch(plt.Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                 closed=True, fc=fc, ec="#333", lw=0.9)); ax.text(cx, cy, t, ha="center", va="center", fontsize=7.2)
def far(x1, y1, x2, y2, t=None, col="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.0, color=col))
    if t: ax.text((x1+x2)/2+0.12, (y1+y2)/2, t, ha="left", va="center", fontsize=6.8)
ax.text(2.5, 6.7, "OFFLINE DESIGN", ha="center", fontsize=8.5, weight="bold", color="#555")
ax.text(7.7, 6.7, "ONLINE INFERENCE", ha="center", fontsize=8.5, weight="bold", color="#555")
ax.axvline(5.0, color="#bbb", ls="--", lw=0.8)
fbox(0.5, 5.6, 4.0, 0.75, "calibrate ranges on $\\mathcal{D}_c$", "#eef")
fbox(0.5, 4.55, 4.0, 0.75, "enumerate $\\mathcal{O}^L$: acc + energy on $\\mathcal{D}_v$", "#eef")
fbox(0.5, 3.5, 4.0, 0.75, "extract Pareto frontier $\\mathcal{P}$", "#eef")
fbox(0.5, 2.45, 4.0, 0.75, "select $C_{\\mathrm{lo}},C_{\\mathrm{hi}}$; fit temperature $T^\\star$", "#ffe9d6")
fbox(0.5, 1.4, 4.0, 0.75, "choose $\\tau^\\star$ (meet acc target)", "#ffe9d6")
for (y1, y2) in [(5.6, 5.3), (4.55, 4.25), (3.5, 3.2), (2.45, 2.15)]:
    far(2.5, y1, 2.5, y2)
far(2.5, 1.4, 2.5, 0.9); fbox(0.5, 0.15, 4.0, 0.7, "deploy $(C_{\\mathrm{lo}},C_{\\mathrm{hi}},T^\\star,\\tau^\\star)$", "#e2f3e2")
far(4.5, 0.5, 5.5, 0.5); # cross to online
fbox(5.6, 5.6, 3.9, 0.75, "input $x$ arrives", "#eaeaf5")
fbox(5.6, 4.55, 3.9, 0.75, "cheap pass $C_{\\mathrm{lo}}$ $\\to$ $g(x)$  (cost $E_{\\mathrm{lo}}$)", "#ffe9d6")
fdia(7.55, 3.3, 3.4, 1.1, "$g(x)\\geq\\tau^\\star$?", "#e2f3e2")
fbox(5.6, 1.5, 1.8, 0.8, "accept\n$\\hat{y}_{C_{\\mathrm{lo}}}$", "#e6f0fb")
fbox(7.7, 1.5, 1.8, 0.8, "run $C_{\\mathrm{hi}}$\n$\\hat{y}_{C_{\\mathrm{hi}}}$", "#fde6e6")
far(7.55, 5.6, 7.55, 5.3); far(7.55, 4.55, 7.55, 3.85)
far(6.8, 3.0, 6.5, 2.3, "yes", "#2a7a2a"); far(8.3, 3.0, 8.6, 2.3, "no", "#b33")
save(fig, "Fig12")

print("revision figures written:", [f for f in sorted(os.listdir(FIG)) if f.endswith('.pdf')])
