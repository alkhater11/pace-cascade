"""make_figs_r2.py -- figures of the second revision (Fig13-Fig17), PDF + EPS,
MATLAB-like styling (mstyle.py).  Missing result files are skipped."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mstyle

mstyle.apply()
M = mstyle.MATLAB
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)
COL = {"static": M[4], "pace": M[1], "int8": M[0], "oracle": M[3], "env": M[6], "narrow": M[2], "fp32": "k"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ Fig13 ----
def fig13():
    R = {"ResNet-8 / CIFAR-10": load("r2_resnet8.json"), "ViT-4 / Fashion-MNIST": load("r2_vit4.json")}
    if not all(R.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (title, r) in zip(axes, R.items()):
        sc = 1e-3
        allp = np.array(r["all_test"]); fr = np.array(sorted(r["frontier_test"]))
        a8 = r["oracle_compare"]["a_int8"]
        ax.plot(allp[:, 0] * sc, allp[:, 1] * 100, ".", color="#c8c8c8", ms=2.5, label="all configurations", zorder=1)
        ax.plot(fr[:, 0] * sc, fr[:, 1] * 100, "-o", color=COL["static"], ms=3, label="static frontier", zorder=3)
        ax.plot(np.array(r["trace"]["E"]) * sc, np.array(r["trace"]["acc"]) * 100, "-", color=COL["pace"], lw=1.6,
                label="PACE cascade ($C_{lo}\\rightarrow$INT8)", zorder=4)
        g = np.array(r["oracle_compare"]["grid"]); env = np.array(r["oracle_compare"]["envelope"])
        ok = np.isfinite(env)
        ax.plot(env[ok] * sc, g[ok] * 100, "--", color=COL["env"], lw=1.2, label="best cascade envelope", zorder=4)
        og = r["oracle_gate"]
        ax.plot([og["E"] * sc], [og["acc"] * 100], "p", color=COL["oracle"], ms=8, label="oracle gate", zorder=5)
        ax.plot([r["selection"]["E_hi"] * sc], [a8 * 100], "s", color=COL["int8"], ms=6, label="uniform INT8", zorder=5)
        ax.set_xlim(0.9 * fr[:, 0].min() * sc, 1.25 * (r["selection"]["E_lo"] + r["selection"]["E_hi"]) * sc)
        ax.set_ylim(a8 * 100 - 12, max(fr[:, 1].max(), og["acc"]) * 100 + 1.0)
        ax.set_title(title); ax.set_xlabel("energy ($\\mu$J / sample)"); ax.set_ylabel("test accuracy (%)")
        ax.legend(loc="lower right")
    fig.tight_layout(); save(fig, "Fig13")


# ------------------------------------------------------------------ Fig14 ----
def _sev_series(per_set, key):
    m, s = [], []
    for sev in [0, 1, 2, 3]:
        if sev == 0:
            v = per_set["clean|0"][key]; m.append(v[0]); s.append(v[1])
        else:
            vals = [per_set[f"{k}|{sev}"][key] for k in ["gaussian_noise", "impulse_noise", "gaussian_blur", "contrast"]]
            m.append(np.mean([v[0] for v in vals])); s.append(np.mean([v[1] for v in vals]))
    return np.array(m), np.array(s)


def fig14():
    mlp = load("r2_mlp.json"); rn = load("r2_resnet8.json"); vt = load("r2_vit4.json")
    panels = []
    if mlp:
        c = mlp["fmnist"]["corruption"]
        panels.append(("MLP / Fashion-MNIST", c["per_set"], c["configs"]["E_smp"], c["configs"]["E_hi"], 1e-3, "nJ"))
    for title, r in [("ResNet-8 / CIFAR-10", rn), ("ViT-4 / Fashion-MNIST", vt)]:
        if r:
            panels.append((title, r["corruption"]["per_set"], r["selection"]["E_smp"], r["selection"]["E_hi"],
                           1e-6 if "ResNet" in title else 1e-3, "$\\mu$J" if "ResNet" in title else "nJ"))
    if not panels:
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(2.45 * len(panels), 4.6), squeeze=False)
    x = np.arange(4)
    for j, (title, ps, e_smp, e_hi, sc, unit) in enumerate(panels):
        ax = axes[0, j]
        for key, lab, col, mk in [("int8", "INT8", COL["int8"], "s"), ("smp", "static mixed-prec.", COL["static"], "o"),
                                  ("casc", "PACE cascade", COL["pace"], "^")]:
            m, s = _sev_series(ps, key)
            ax.errorbar(x, m * 100, yerr=s * 100, fmt="-" + mk, color=col, ms=4, capsize=2, label=lab)
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(["clean", "1", "2", "3"])
        ax.set_ylabel("test accuracy (%)") if j == 0 else None
        if j == 0:
            ax.legend(loc="lower left")
        ax = axes[1, j]
        m, s = _sev_series(ps, "E"); esc, _ = _sev_series(ps, "esc")
        ax.errorbar(x, m * sc, yerr=s * sc, fmt="-^", color=COL["pace"], ms=4, capsize=2, label="PACE cascade")
        ax.axhline(e_smp * sc, color=COL["static"], ls="-", lw=1.2, label="static mixed-prec.")
        ax.axhline(e_hi * sc, color=COL["int8"], ls="--", lw=1.2, label="INT8")
        for xi, (mi, ei) in enumerate(zip(m, esc)):
            ax.annotate(f"{100*ei:.0f}%", (xi, mi * sc), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6.5, color=COL["pace"])
        ax.set_xticks(x); ax.set_xticklabels(["clean", "1", "2", "3"]); ax.set_xlabel("corruption severity")
        ax.set_ylabel(f"energy ({unit} / sample)") if j == 0 else None
        ax.set_ylim(0, e_hi * sc * 1.25)
        if j == 0:
            ax.legend(loc="lower left")
    fig.tight_layout(); save(fig, "Fig14")


# ------------------------------------------------------------------ Fig15 ----
def fig15():
    rg = load("r2_regimes.json")
    if not rg:
        return
    W = rg["workloads"]; gs = list(rg["family"]); phis = list(rg["family"][gs[0]])
    cmap = LinearSegmentedColormap.from_list("mdiv", [M[0], "#ffffff", M[1]])
    n = len(W); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.25 * rows), squeeze=False)
    for k, wn in enumerate(W):
        ax = axes[k // cols, k % cols]
        Z = np.array([[100 * (rg["family"][g][p][wn]["S_star"] if rg["family"][g][p][wn]["S_star"] is not None else -100)
                       for p in phis] for g in gs])
        Zc = np.clip(Z, -60, 60)
        ax.imshow(Zc, cmap=cmap, vmin=-60, vmax=60, aspect="auto")
        for i in range(len(gs)):
            for j in range(len(phis)):
                ax.text(j, i, f"{Z[i, j]:+.0f}", ha="center", va="center", fontsize=6.3,
                        color="k" if abs(Zc[i, j]) < 40 else "w")
        ax.set_xticks(range(len(phis))); ax.set_xticklabels([f"{float(p):.2g}" for p in phis], fontsize=7)
        ax.set_yticks(range(len(gs))); ax.set_yticklabels(["linear", "quadratic", "cubic", "exponential"][:len(gs)], fontsize=7)
        ax.grid(False); ax.set_title(wn, fontsize=8.5)
        if "quad" in gs:                                   # mark the main-text model H0
            ax.add_patch(plt.Rectangle((phis.index("0.333") - 0.5, gs.index("quad") - 0.5), 1, 1,
                                       fill=False, ec="k", lw=1.4))
        if k // cols == rows - 1:
            ax.set_xlabel("fixed-energy fraction $\\phi$")
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.tight_layout(); save(fig, "Fig15")


# ------------------------------------------------------------------ Fig16 ----
def fig16():
    rg = load("r2_regimes.json")
    if not rg:
        return
    titles = {"H0": "H0: array MAC (main text)", "H2": "H2: bit-serial", "H5": "H5: analog IMC (ADC)"}
    marks = ["o", "s", "^", "v", "D", "P"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, key in zip(axes, ["H0", "H2", "H5"]):
        for k, wn in enumerate(rg["workloads"]):
            pts = np.array(rg["scatter"][key][wn])
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 0], np.clip(pts[:, 1] * 100, -95, 200), marks[k], color=M[k % len(M)], ms=3,
                    mfc="none", mew=0.7, label=wn)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_title(titles[key]); ax.set_xlabel("$\\rho = E_{lo}/E_{hi}$")
    axes[0].set_ylabel("cascade $-$ static energy (%)")
    axes[0].legend(loc="upper left", fontsize=6.2, ncol=1)
    fig.tight_layout(); save(fig, "Fig16")


# ------------------------------------------------------------------ Fig17 ----
def fig17():
    mlp = load("r2_mlp.json")
    if not mlp:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, ds, title in zip(axes, ["mnist", "fmnist"], ["MNIST", "Fashion-MNIST"]):
        c = mlp[ds]["capacity"]; sc = 1e-3
        fw = np.array(sorted(c["frontier_wide"])); fn = np.array(sorted(c["frontier_narrow"]))
        ax.plot(fw[:, 0] * sc, fw[:, 1] * 100, "-o", color=COL["static"], ms=3, label="static frontier, wide MLP")
        ax.plot(fn[:, 0] * sc, fn[:, 1] * 100, "-s", color=COL["narrow"], ms=3, label="static frontier, narrow MLP")
        ax.plot(np.array(c["trace"]["E"]) * sc, np.array(c["trace"]["acc"]) * 100, "-", color=COL["pace"], lw=1.6,
                label="capacity cascade (narrow$\\rightarrow$wide INT8)")
        oc = c["oracle_compare"]; g = np.array(oc["grid"]); env = np.array(oc["envelope"]); ok = np.isfinite(env)
        ax.plot(env[ok] * sc, g[ok] * 100, "--", color=COL["env"], lw=1.2, label="best cascade envelope")
        ax.plot([c["E_hi"] * sc], [oc["a_int8"] * 100], "s", color=COL["int8"], ms=6, label="uniform INT8 (wide)")
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("energy (nJ / sample)")
        ax.set_ylabel("test accuracy (%)")
        top = max(fw[:, 1].max(), oc["ensemble_best_acc"]) * 100
        ax.set_ylim(top - 8, top + 0.6); ax.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(); save(fig, "Fig17")


if __name__ == "__main__":
    for f in (fig13, fig14, fig15, fig16, fig17):
        try:
            f()
        except Exception as e:                       # keep going; report
            print("FAILED", f.__name__, repr(e))
