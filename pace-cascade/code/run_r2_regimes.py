"""run_r2_regimes.py -- where does the conclusion hold? (second revision)

For every workload with an exhaustive per-configuration cache
  MLP/MNIST, MLP/Fashion-MNIST            (729 configurations, extended options)
  ResNet-8/CIFAR-10, ViT-4/Fashion-MNIST   (4^G stage-wise configurations)
  Capacity/MNIST, Capacity/Fashion-MNIST   (wide 729 + narrow 81 configurations)
and for every hardware energy model (named models H0-H6 and the
(g, phi) family grid of configs/energy_models.json) we
  * re-select every baseline under that model (fair re-optimisation),
  * compare the exact static frontier with the best cascade envelope (test),
  * predict the outcome from validation quantities (Proposition 3) and check the
    prediction against the realised test outcome.
Accuracies never depend on the energy model, so only the energy ranking moves.
Output: results/r2_regimes.json
"""
import os, json, time
import numpy as np
import r2lib as L
import deepquant as Q

MLP_MACS = [200704, 32768, 1280]
NMACS = [25088, 320]


def concat(a, b):
    c = L.Cache.__new__(L.Cache)
    c.configs = a.configs + b.configs
    c.index = {cf: i for i, cf in enumerate(c.configs)}
    c.y = a.y; c.correct = np.concatenate([a.correct, b.correct]); c.msp = np.concatenate([a.msp, b.msp])
    c.N = a.N; c.acc = np.concatenate([a.acc, b.acc]); c._order = {}
    return c


def workloads():
    W = {}
    ld = lambda n: L.Cache.load(os.path.join(L.CACHE, n))
    for ds, name in [("mnist", "MLP / MNIST"), ("fmnist", "MLP / Fashion-MNIST")]:
        vc, tc = ld(f"mlp_{ds}_val_ext.npz"), ld(f"mlp_{ds}_test_ext.npz")
        W[name] = {"vc": vc, "tc": tc, "i8": ((8, 0),) * 3,
                   "E": (lambda m, c=vc.configs: L.energies(c, MLP_MACS, m))}
    for tag, name, GM in [("resnet8", "ResNet-8 / CIFAR-10", Q.ResNet8Q.GROUP_MACS),
                          ("vit4", "ViT-4 / Fashion-MNIST", Q.ViT4Q.GROUP_MACS)]:
        if os.path.exists(os.path.join(L.CACHE, f"{tag}_test.npz")):
            vc, tc = ld(f"{tag}_val.npz"), ld(f"{tag}_test.npz")
            W[name] = {"vc": vc, "tc": tc, "i8": ((8, 0),) * len(GM),
                       "E": (lambda m, c=vc.configs, g=GM: L.energies(c, g, m))}
    for ds, name in [("mnist", "Capacity / MNIST"), ("fmnist", "Capacity / Fashion-MNIST")]:
        wv, wt = ld(f"mlp_{ds}_val_ext.npz"), ld(f"mlp_{ds}_test_ext.npz")
        nv, nt = ld(f"narrow_{ds}_val.npz"), ld(f"narrow_{ds}_test.npz")
        def Eu(m, wc=wv.configs, nc=nv.configs):
            return np.concatenate([L.energies(wc, MLP_MACS, m), L.energies([c[:2] for c in nc], NMACS, m)])
        W[name] = {"vc": concat(wv, nv), "tc": concat(wt, nt), "i8": ((8, 0),) * 3, "E": Eu}
    return W


def finite(x):
    return float(x) if np.isfinite(x) else None


def evaluate(w, m, keep_curves=False):
    vc, tc = w["vc"], w["tc"]
    E = w["E"](m)
    i8 = vc.index[w["i8"]]
    cmp = L.compare_frontier(tc, E, i8)
    if not keep_curves:
        for k in ("grid", "static_curve", "envelope"):
            cmp.pop(k)
    for k in ("S_star", "S_max", "cascade_E_star"):
        cmp[k] = finite(cmp[k])
    sel = L.select_protocol(vc, E, i8)
    feas = L.feasibility(vc, E, sel)
    obs = L.observed_on_cache(tc, E, sel)
    feas = {k: (finite(v) if isinstance(v, float) else v) for k, v in feas.items()}
    return {"compare": cmp, "feasibility": feas, "observed": obs,
            "C_lo": [list(o) for o in vc.configs[sel["lo"]]], "C_smp": [list(o) for o in vc.configs[sel["smp"]]],
            "E_lo": sel["E_lo"], "E_hi": sel["E_hi"], "E_smp": sel["E_smp"]}


def scatter(w, m, max_frontier=40):
    """(rho, relative cascade-minus-static energy) for every frontier pair, at the
    accuracy of the high arm minus 0.2 pp (test)."""
    tc = w["tc"]; E = w["E"](m); acc = tc.acc
    fr = L.pareto_idx(E, acc)
    fr = sorted(fr, key=lambda i: -acc[i])[:max_frontier]
    pts = []
    for lo in fr:
        for hi in fr:
            if E[lo] >= E[hi]:
                continue
            a = acc[hi] - 0.002
            if a <= acc[lo]:
                continue
            ec, _, _ = L.cascade_energy_at(tc, E, lo, hi, [a])
            es = L.static_energy_at(E, acc, [a])
            if np.isfinite(ec[0]) and np.isfinite(es[0]):
                pts.append([float(E[lo] / E[hi]), float((ec[0] - es[0]) / es[0])])
    return pts


def main():
    t0 = time.time()
    EM = L.load_energy_models()
    W = workloads()
    print("workloads:", list(W), flush=True)
    out = {"workloads": list(W), "named": {}, "family": {}, "scatter": {}}
    for key in ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]:
        m = EM[key]
        out["named"][key] = {"name": m["name"],
                             "ratio_e3_e8": L.emac(3, 0, m) / L.emac(8, 0, m),
                             "ratio_e4_e8": L.emac(4, 0, m) / L.emac(8, 0, m),
                             "results": {wn: evaluate(w, m, keep_curves=(key == "H0")) for wn, w in W.items()}}
        print(f"  {key} done ({time.time()-t0:.0f}s)", flush=True)
    fg = EM["family_grid"]
    for g in fg["g"]:
        out["family"][g] = {}
        for phi in fg["phi"]:
            m = L.family_model(g, phi, fg["e8"])
            out["family"][g][str(phi)] = {wn: {"S_star": r["compare"]["S_star"], "S_max": r["compare"]["S_max"],
                                               "frac_win": r["compare"]["frac_win"],
                                               "pred": r["feasibility"]["pred_gate_wins"],
                                               "obs": r["observed"]["observed_win"],
                                               "rho": r["feasibility"]["rho"]}
                                          for wn, r in ((wn, evaluate(w, m)) for wn, w in W.items())}
        print(f"  family g={g} done ({time.time()-t0:.0f}s)", flush=True)
    for key in ["H0", "H2", "H5"]:
        out["scatter"][key] = {wn: scatter(w, EM[key]) for wn, w in W.items()}
    # agreement between validation-based prediction and test outcome
    agree, total = 0, 0
    for key in out["named"]:
        for wn, r in out["named"][key]["results"].items():
            total += 1; agree += int(r["feasibility"]["pred_gate_wins"] == r["observed"]["observed_win"])
    for g in out["family"]:
        for phi in out["family"][g]:
            for wn, r in out["family"][g][phi].items():
                total += 1; agree += int(r["pred"] == r["obs"])
    out["prediction_agreement"] = {"agree": agree, "total": total}
    L.dump(out, "r2_regimes.json")
    print(f"saved r2_regimes.json ({time.time()-t0:.0f}s); prediction agreement {agree}/{total}")


if __name__ == "__main__":
    main()
