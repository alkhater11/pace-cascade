"""run_r2_interp.py -- is a precision-only cascade win genuine input-adaptivity or
interpolation between sparse static points?  For every cached workload (H0):
 (a) oracle-tuned best-pair envelope vs. the step-wise static frontier AND the
     time-sharing (convex-hull) static baseline, on test;
 (b) best pair, static and time-sharing all SELECTED ON VALIDATION, evaluated on test."""
import os, json, numpy as np
import r2lib as L
import deepquant as Q
H0 = L.load_energy_models()["H0"]
ld = lambda n: L.Cache.load(os.path.join(L.CACHE, n))
W = []
for ds in ["mnist", "fmnist"]:
    if os.path.exists(os.path.join(L.CACHE, f"mlp_{ds}_test_ext.npz")):
        vc, tc = ld(f"mlp_{ds}_val_ext.npz"), ld(f"mlp_{ds}_test_ext.npz")
        idx = [i for i, c in enumerate(vc.configs) if all(o in L.OPTIONS for o in c)]
        W.append((f"MLP / {ds} (216)", vc.subset(idx), tc.subset(idx), [200704, 32768, 1280]))
        W.append((f"MLP / {ds} (729)", vc, tc, [200704, 32768, 1280]))
for tag, GM in [("vit4", Q.ViT4Q.GROUP_MACS), ("resnet8", Q.ResNet8Q.GROUP_MACS)]:
    if os.path.exists(os.path.join(L.CACHE, f"{tag}_test.npz")):
        W.append((tag, ld(f"{tag}_val.npz"), ld(f"{tag}_test.npz"), GM))
out = {}
for name, vc, tc, GM in W:
    E = L.energies(vc.configs, GM, H0)
    i8 = vc.index[tuple([(8, 0)] * len(GM))]
    a_star = tc.acc[i8] - 0.002
    fr = sorted(L.pareto_idx(E, tc.acc), key=lambda i: -tc.acc[i])[:60]
    pairs = [(l, h) for l in fr for h in fr if E[l] < E[h]]
    grid = np.linspace(max(tc.acc[i8] - 0.03, tc.acc[fr].min()), tc.acc.max(), 121)
    tg = np.append(grid, a_star)
    st = L.static_energy_at(E, tc.acc, tg); hu = L.hull_energy_at(E, tc.acc, tg)
    env, arg, _ = L.envelope(tc, E, pairs, tg)
    rs, rh = (st - env) / st, (hu - env) / hu
    j = int(arg[-1])
    out[name] = {"S_step_star": float(rs[-1]), "S_hull_star": float(rh[-1]),
                 "S_step_max": float(rs[:-1].max()), "S_hull_max": float(rh[:-1].max()),
                 "frac_step": float((rs[:-1] > 1e-9).mean()), "frac_hull": float((rh[:-1] > 1e-9).mean()),
                 "pair_star": [[list(o) for o in tc.configs[pairs[j][0]]], [list(o) for o in tc.configs[pairs[j][1]]]] if j >= 0 else None,
                 "rho_star": float(E[pairs[j][0]] / E[pairs[j][1]]) if j >= 0 else None,
                 "val_selected": L.best_pair_val_to_test(vc, tc, E, i8)}
    v = out[name]["val_selected"]
    print(f"{name:22s} oracle: S*step={100*rs[-1]:+.1f}% S*hull={100*rh[-1]:+.1f}% "
          f"maxStep={100*rs[:-1].max():+.1f}% maxHull={100*rh[:-1].max():+.1f}% rho*={out[name]['rho_star']}")
    if "cas_E" in v:
        print(f"{'':22s} val->test: cascade E={v['cas_E']/1e3:.2f} acc={100*v['cas_test_acc']:.2f} | "
              f"static E={v['static_E']/1e3:.2f} acc={100*v['static_test_acc']:.2f} | "
              f"mix E={v['mix_E']/1e3:.2f} acc={100*v['mix_test_acc']:.2f} | rho={v['rho']:.2f} pesc={100*v['cas_pesc']:.1f}%")
# five-seed evaluation of the validation-selected choices (MLP workloads)
import axlib as A
from run_train import SEEDS, split
for name in list(out):
    if not name.startswith("MLP"):
        continue
    ds = name.split("/")[1].split()[0]; v = out[name]["val_selected"]
    if "cas_E" not in v:
        continue
    Xtr, ytr, Xval, yval, Xte, yte, Xcal = split(ds)
    r = {"cas_acc": [], "cas_E": [], "cas_pesc": [], "static_acc": [], "mix_acc": []}
    for s in SEEDS:
        m = A.NumpyMLP.load(os.path.join(L.RES, "models", f"{ds}_seed{s}.npz")); am, wm = A.calibrate(m, Xcal)
        P = lambda c: L.softmax(A.forward_quant(m, Xte, c, am, wm))
        pl, ph = P(v["cas_lo"]), P(v["cas_hi"]).argmax(1)
        esc = pl.max(1) < v["tau"]
        r["cas_acc"].append(float((np.where(esc, ph, pl.argmax(1)) == yte).mean()))
        r["cas_pesc"].append(float(esc.mean())); r["cas_E"].append(v["E_lo"] + float(esc.mean()) * v["E_hi"])
        r["static_acc"].append(float((P(v["static_cfg"]).argmax(1) == yte).mean()))
        if "mix_cfgs" in v:
            a1 = float((P(v["mix_cfgs"][0]).argmax(1) == yte).mean()); a2 = float((P(v["mix_cfgs"][1]).argmax(1) == yte).mean())
            r["mix_acc"].append(a1 + v["mix_w"] * (a2 - a1))
        else:
            r["mix_acc"].append(r["static_acc"][-1])
    out[name]["five_seed"] = {k: L.ms(x) for k, x in r.items()}
    f = out[name]["five_seed"]
    print(f"{name:22s} 5 seeds: cascade E={f['cas_E'][0]/1e3:.2f}+/-{f['cas_E'][1]/1e3:.2f} acc={100*f['cas_acc'][0]:.2f}+/-{100*f['cas_acc'][1]:.2f} "
          f"esc={100*f['cas_pesc'][0]:.1f}% | static E={v['static_E']/1e3:.2f} acc={100*f['static_acc'][0]:.2f}+/-{100*f['static_acc'][1]:.2f} "
          f"| mix E={v['mix_E']/1e3:.2f} acc={100*f['mix_acc'][0]:.2f}", flush=True)
L.dump(out, "r2_interp.json")
