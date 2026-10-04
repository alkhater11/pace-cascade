"""make_figs.py -- all figures as EPS + PDF (Fig1..Fig8) and CSV tables from the
real result JSONs. Publication style, vector output, no chartjunk.
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
import mstyle; mstyle.apply()   # MATLAB-like style (second revision)

DS_NAME = {"mnist": "MNIST", "fmnist": "Fashion-MNIST"}
C = {"fp32": "#000000", "int8": "#0072BD", "static": "#77AC30",
     "pace": "#D95319", "oracle": "#7E2F8E", "random": "#bbbbbb",
     "lo": "#EDB120", "hi": "#4DBEEE"}

def save(fig, name):
    for ext in ("eps", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)

E = {ds: json.load(open(os.path.join(OUT, f"explore_{ds}.json"))) for ds in DS_NAME}
P = {ds: json.load(open(os.path.join(OUT, f"pace_{ds}.json"))) for ds in DS_NAME}

# ---------- Fig1: static Pareto frontier (both datasets) ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, ds in zip(axes, DS_NAME):
    e = E[ds]; recs = e["records"]
    en = np.array([r["energy_pj"] for r in recs]) / 1000.0
    va = np.array([r["val_acc"] for r in recs]) * 100
    ax.scatter(en, va, s=9, c="#cccccc", edgecolors="none", label="all configs", zorder=1)
    fr = e["pareto_idx"]
    fe = np.array([recs[i]["energy_pj"] for i in fr]) / 1000.0
    fa = np.array([recs[i]["val_acc"] for i in fr]) * 100
    o = np.argsort(fe)
    ax.plot(fe[o], fa[o], "-o", color=C["static"], ms=3, lw=1.2, label="Pareto frontier", zorder=3)
    i8 = recs[e["uniform_idx"]["(8, 0)"]]
    ax.scatter([i8["energy_pj"]/1000], [i8["val_acc"]*100], marker="s", s=42,
               color=C["int8"], zorder=4, label="uniform INT8")
    ax.axhline(e["fp32_test_acc"]*100, color=C["fp32"], ls="--", lw=1.0, label="FP32")
    ax.set_title(DS_NAME[ds]); ax.set_xlabel("energy (nJ / sample)")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.9)
fig.tight_layout(); save(fig, "Fig1")

# ---------- Fig2: per-layer sensitivity ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, ds in zip(axes, DS_NAME):
    sens = E[ds]["sensitivity"]
    for l in sorted(sens, key=int):
        rows = sens[l]
        effs = [r["eff"] for r in rows]; drops = [r["drop"]*100 for r in rows]
        o = np.argsort(effs)
        ax.plot(np.array(effs)[o], np.array(drops)[o], "-o", ms=3,
                label=f"layer {int(l)+1}")
    ax.set_title(DS_NAME[ds]); ax.set_xlabel("effective multiplier width (bits)")
    ax.set_ylabel("accuracy drop vs INT8 (pp)")
    ax.legend(loc="upper right")
fig.tight_layout(); save(fig, "Fig2")

# ---------- Fig3: PACE cascade vs static frontier (KEY) ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, ds in zip(axes, DS_NAME):
    pj = P[ds]; ps = pj["pareto_static"]
    se = np.array([p["energy_pj"] for p in ps]) / 1000.0
    sa = np.array([p["test_acc"] for p in ps]) * 100
    o = np.argsort(se)
    ax.plot(se[o], sa[o], "-o", color=C["static"], ms=3, label="static MP frontier", zorder=3)
    cur = pj["curves"]["msp"]
    ce = np.array([c["energy_pj"] for c in cur]) / 1000.0
    ca = np.array([c["test_acc"] for c in cur]) * 100
    ax.plot(ce, ca, "-", color=C["pace"], lw=1.6, label="PACE (max-softmax)", zorder=4)
    orc = pj["oracle"]
    ax.scatter([orc["energy_pj"]/1000], [orc["test_acc"]*100], marker="*", s=90,
               color=C["oracle"], zorder=5, label="oracle gate")
    rc = pj["random_curve"]
    re = np.array([c["energy_pj"] for c in rc]) / 1000.0
    ra = np.array([c["test_acc"] for c in rc]) * 100
    ax.plot(re, ra, ":", color="#999999", lw=1.1, label="random gate", zorder=2)
    ax.scatter([pj["headline"]["e_hi_pj"]/1000], [pj["headline"]["int8_acc"][0]*100],
               marker="s", s=36, color=C["int8"], zorder=5, label="uniform INT8")
    ax.set_title(DS_NAME[ds]); ax.set_xlabel("energy (nJ / sample)")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.9)
fig.tight_layout(); save(fig, "Fig3")

# ---------- Fig4: escalation rate & accuracy vs tau ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, ds in zip(axes, DS_NAME):
    cur = P[ds]["curves"]["msp"]
    tau = [c["tau"] for c in cur]
    esc = [c["esc"]*100 for c in cur]
    acc = [c["test_acc"]*100 for c in cur]
    ax2 = ax.twinx()
    l1, = ax.plot(tau, esc, "-", color=C["pace"], label="escalation rate")
    l2, = ax2.plot(tau, acc, "--", color=C["int8"], label="test accuracy")
    ax.set_title(DS_NAME[ds]); ax.set_xlabel(r"confidence threshold $\tau$")
    ax.set_ylabel("escalation rate (%)", color=C["pace"])
    ax2.set_ylabel("test accuracy (%)", color=C["int8"])
    ax.axvline(P[ds]["headline"]["tau_star"], color="#888888", ls=":", lw=1.0)
    ax.tick_params(axis="y", colors=C["pace"]); ax2.tick_params(axis="y", colors=C["int8"])
    ax2.grid(False)
    ax.legend([l1, l2], ["escalation", "accuracy"], loc="center right")
fig.tight_layout(); save(fig, "Fig4")

# ---------- Fig5: gate-type ablation ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
labels = {"msp": "max-softmax", "margin": "margin", "negentropy": "neg-entropy",
          "msp_calibrated": "max-softmax (T-scaled)"}
styles = {"msp": "-", "margin": "--", "negentropy": "-.", "msp_calibrated": ":"}
for ax, ds in zip(axes, DS_NAME):
    for k in ["msp", "margin", "negentropy", "msp_calibrated"]:
        cur = P[ds]["curves"][k]
        ce = np.array([c["energy_pj"] for c in cur]) / 1000.0
        ca = np.array([c["test_acc"] for c in cur]) * 100
        ax.plot(ce, ca, styles[k], lw=1.3, label=labels[k])
    ax.set_title(DS_NAME[ds]); ax.set_xlabel("energy (nJ / sample)")
    ax.set_ylabel("test accuracy (%)"); ax.legend(loc="lower right")
fig.tight_layout(); save(fig, "Fig5")

# ---------- Fig6: reliability diagrams (C_lo raw vs T-scaled) ----------
# rebuild reliability from a quick recompute
import axlib as A
from run_train import SIZES, split
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, ds in zip(axes, DS_NAME):
    Xtr,ytr,Xval,yval,Xte,yte,Xcal = split(ds)
    m = A.NumpyMLP.load(os.path.join(OUT, "models", f"{ds}_seed0.npz"))
    am, wm = A.calibrate(m, Xcal)
    C_lo = P[ds]["headline"]["C_lo"]; T = P[ds]["temperature"]
    lg, pr = A.forward_quant(m, Xte, C_lo, am, wm, return_probs=True)
    prc = A._softmax(lg / T)
    nb = 12; bins = np.linspace(0, 1, nb + 1)
    for probs, col, lab in [(pr, C["pace"], f"raw (ECE={P[ds]['ece_raw']:.3f})"),
                            (prc, C["int8"], f"T={T:.2f} (ECE={P[ds]['ece_calibrated']:.3f})")]:
        conf = probs.max(1); pred = probs.argmax(1); corr = (pred == yte).astype(float)
        xs, ys = [], []
        for i in range(nb):
            msk = (conf > bins[i]) & (conf <= bins[i+1])
            if msk.sum() > 20:
                xs.append(conf[msk].mean()*100); ys.append(corr[msk].mean()*100)
        ax.plot(xs, ys, "-o", ms=3, color=col, label=lab)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="perfect")
    ax.set_title(DS_NAME[ds]); ax.set_xlabel("confidence (%)"); ax.set_ylabel("accuracy (%)")
    ax.legend(loc="upper left")
fig.tight_layout(); save(fig, "Fig6")

# ---------- Fig7: energy comparison bars (headline) ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
for ax, ds in zip(axes, DS_NAME):
    h = P[ds]["headline"]
    names = ["FP32", "INT8", "static MP", "PACE"]
    en = [h["fp32_energy_pj"]/1000, h["e_hi_pj"]/1000,
          h["static_mp_energy_pj"]/1000, h["pace_energy_pj"][0]/1000]
    err = [0, 0, 0, h["pace_energy_pj"][1]/1000]
    cols = [C["fp32"], C["int8"], C["static"], C["pace"]]
    bars = ax.bar(names, en, yerr=err, color=cols, capsize=3, width=0.62)
    ax.set_yscale("log"); ax.set_title(DS_NAME[ds])
    ax.set_ylabel("energy (nJ / sample, log)")
    for b, v in zip(bars, en):
        ax.text(b.get_x()+b.get_width()/2, v*1.08, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "Fig7")

# ---------- Fig8: PACE schematic ----------
fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec="#333333", lw=0.9))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8)
def arrow(x1, y1, x2, y2, txt=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#333333"))
    if txt: ax.text((x1+x2)/2, (y1+y2)/2+0.18, txt, ha="center", fontsize=7)
box(0.2, 1.5, 1.3, 1.0, "input\n$x$", "#eef")
box(2.0, 1.5, 1.9, 1.0, "cheap config\n$C_{lo}$  ($E_{lo}$)", "#ffe9d6")
box(4.5, 1.5, 1.9, 1.0, "confidence\ngate $g(x)$", "#e7f6e7")
box(7.0, 2.55, 2.6, 1.0, "accept: $\\hat{y}_{lo}$\n(cost $E_{lo}$)", "#e6f0fb")
box(7.0, 0.35, 2.6, 1.0, "escalate $C_{hi}$: $\\hat{y}_{hi}$\n(cost $E_{lo}+E_{hi}$)", "#fde6e6")
arrow(1.5, 2.0, 2.0, 2.0)
arrow(3.9, 2.0, 4.5, 2.0)
arrow(6.4, 2.1, 7.0, 3.0, r"$g(x)\geq\tau$")
arrow(6.4, 1.9, 7.0, 0.9, r"$g(x)<\tau$")
ax.text(5.0, 3.6, r"expected energy: $E[E]=E_{lo}+p_{esc}\,E_{hi}$,  "
                  r"$p_{esc}=\Pr[g(x)<\tau]$", ha="center", fontsize=8.5)
save(fig, "Fig8")

# ---------- CSV tables ----------
def w_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

# headline table
rows = []
for ds in DS_NAME:
    h = P[ds]["headline"]
    rows.append([DS_NAME[ds], "FP32", f"{h['fp32_acc'][0]*100:.2f}", f"{h['fp32_acc'][1]*100:.2f}",
                 f"{h['fp32_energy_pj']/1000:.1f}", "1.0"])
    rows.append([DS_NAME[ds], "INT8", f"{h['int8_acc'][0]*100:.2f}", f"{h['int8_acc'][1]*100:.2f}",
                 f"{h['e_hi_pj']/1000:.2f}", f"{h['fp32_energy_pj']/h['e_hi_pj']:.1f}"])
    rows.append([DS_NAME[ds], "static-MP", f"{h['static_mp_acc'][0]*100:.2f}", f"{h['static_mp_acc'][1]*100:.2f}",
                 f"{h['static_mp_energy_pj']/1000:.2f}", f"{h['fp32_energy_pj']/h['static_mp_energy_pj']:.1f}"])
    rows.append([DS_NAME[ds], "PACE", f"{h['pace_acc'][0]*100:.2f}", f"{h['pace_acc'][1]*100:.2f}",
                 f"{h['pace_energy_pj'][0]/1000:.2f}+/-{h['pace_energy_pj'][1]/1000:.2f}",
                 f"{h['fp32_energy_pj']/h['pace_energy_pj'][0]:.1f}"])
w_csv("table_headline.csv", ["dataset", "method", "acc_mean", "acc_std", "energy_nJ", "speedup_vs_fp32"], rows)

# pareto points table (seed0)
for ds in DS_NAME:
    rows = []
    for p in P[ds]["pareto_static"]:
        rows.append([str(p["config"]).replace(",", ";"), f"{p['energy_pj']/1000:.2f}",
                     f"{p['val_acc']*100:.2f}", f"{p['test_acc']*100:.2f}" if p["test_acc"] else ""])
    w_csv(f"table_pareto_{ds}.csv", ["config", "energy_nJ", "val_acc", "test_acc"], rows)

print("Figures written:", sorted(os.listdir(FIG)))
print("CSV tables:", [f for f in os.listdir(OUT) if f.endswith(".csv")])
