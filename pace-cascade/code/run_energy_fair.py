"""Fair memory-augmented comparison: under EACH energy model, re-select the static
mixed-precision config AND the cascade's cheap config C_lo from the full 216-config
enumeration (accuracies are energy-model-independent; only energies change), then
recompute the PACE cascade by real forward passes. Reveals whether the negative
result is regime-dependent.  Key point: truncation (t>0) cuts compute energy but
NOT operand-fetch (memory) energy, so genuine low-bit configs are favoured when
memory dominates.
"""
import os, json, numpy as np
import axlib as A
from run_train import SIZES, SEEDS, split

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")
KAPPA = 0.2/64.0; E_ACC = 0.1
def e_mac(b, t, mu): return KAPPA*(b-t)**2 + E_ACC + mu*b
def cfg_e(cfg, macs, mu): return sum(m*e_mac(b, t, mu) for m, (b, t) in zip(macs, cfg))
def msp(p): return p.max(1)

def pareto(pairs):  # pairs: list (energy, val_acc, idx)
    order = sorted(range(len(pairs)), key=lambda i:(pairs[i][0], -pairs[i][1]))
    fr, best = [], -1
    for i in order:
        if pairs[i][1] > best+1e-12: fr.append(pairs[i][2]); best = pairs[i][1]
    return fr

def main():
    mus = {"compute-bound (mu=0)": 0.0, "memory-augmented (mu=0.15)": 0.15}
    SW = 50.0   # pJ per escalation (reconfiguration), fixed representative value
    out = {}
    for ds in ["mnist", "fmnist"]:
        expl = json.load(open(os.path.join(OUT, f"explore_{ds}.json")))
        recs = expl["records"]; macs = expl["layer_macs"]
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = split(ds)
        m = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
        am, wm = A.calibrate(m, Xcal)
        int8_idx = expl["uniform_idx"]["(8, 0)"]
        int8_val = recs[int8_idx]["val_acc"]; C_hi = recs[int8_idx]["config"]
        out[ds] = {}
        for label, mu in mus.items():
            # re-rank all configs by TOTAL energy under this model
            pairs = [(cfg_e(r["config"], macs, mu), r["val_acc"], i) for i, r in enumerate(recs)]
            fr = pareto(pairs)
            e_of = {i: cfg_e(recs[i]["config"], macs, mu) for i in range(len(recs))}
            # static-MP: cheapest (total energy) config matching int8 val within 0.2pp
            match = [i for i in fr if recs[i]["val_acc"] >= int8_val-0.002]
            static_idx = min(match, key=lambda i: e_of[i]) if match else int8_idx
            static_cfg = recs[static_idx]["config"]; e_static = e_of[static_idx]
            # C_lo: cheapest (total energy) Pareto config within 5pp of int8 val
            cl = [i for i in fr if recs[i]["val_acc"] >= int8_val-0.05]
            lo_idx = min(cl, key=lambda i: e_of[i]); C_lo = recs[lo_idx]["config"]
            e_lo = e_of[lo_idx]; e_hi = e_of[int8_idx]
            # recompute cascade at tau matching int8 accuracy on val
            _, prv = A.forward_quant(m, Xval, C_lo, am, wm, return_probs=True)
            pv = prv.argmax(1); phv = A.forward_quant(m, Xval, C_hi, am, wm).argmax(1)
            taus = np.linspace(0.2, 1.0001, 120); scv = msp(prv); tau = 1.0
            for t in taus:
                esc = scv < t
                if float((np.where(esc, phv, pv) == yval).mean()) >= int8_val-0.002: tau=float(t); break
            # multi-seed test energy/acc under this model
            accs, ens, escs = [], [], []
            for s in SEEDS:
                ms = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed{s}.npz"))
                ams, wms = A.calibrate(ms, Xcal)
                _, prt = A.forward_quant(ms, Xte, C_lo, ams, wms, return_probs=True)
                pht = A.forward_quant(ms, Xte, C_hi, ams, wms).argmax(1)
                esc = msp(prt) < tau
                accs.append(float((np.where(esc, pht, prt.argmax(1)) == yte).mean()))
                ens.append(e_lo + float(esc.mean())*(e_hi + SW)); escs.append(float(esc.mean()))
            pace_e = float(np.mean(ens)); pace_acc = float(np.mean(accs)); pesc = float(np.mean(escs))
            out[ds][label] = {
                "mu": mu, "switch_pJ": SW, "static_cfg": static_cfg, "C_lo": C_lo,
                "static_nJ": round(e_static/1000, 2), "pace_nJ": round(pace_e/1000, 2),
                "pace_acc": round(pace_acc, 4), "pace_esc": round(pesc, 4),
                "gap_nJ": round((pace_e-e_static)/1000, 2),
                "pace_dominated": bool(pace_e > e_static+1e-9)}
            print(f"[{ds} | {label}] static={static_cfg} ({e_static/1000:.2f}nJ)  "
                  f"C_lo={C_lo}  PACE={pace_e/1000:.2f}nJ acc={pace_acc:.4f} esc={pesc*100:.1f}%  "
                  f"gap={(pace_e-e_static)/1000:+.2f}nJ  {'DOMINATED' if pace_e>e_static else 'PACE WINS'}")
    json.dump(out, open(os.path.join(OUT, "energy_fair.json"), "w"), indent=2)
    print("\nsaved energy_fair.json")

if __name__ == "__main__":
    main()
