"""Per-class metrics, escalation-by-class, and threshold operating points (real)."""
import os, json, numpy as np
import axlib as A
from run_train import SIZES, split

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")

def per_class(pred, y, K=10):
    prec, rec, f1, sup = [], [], [], []
    for k in range(K):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        prec.append(p); rec.append(r); f1.append(f); sup.append(int((y == k).sum()))
    return prec, rec, f1, sup

def gate_msp(probs):
    return probs.max(1)

out_all = {}
for ds in ["mnist", "fmnist"]:
    expl = json.load(open(os.path.join(OUT, f"explore_{ds}.json")))
    pace = json.load(open(os.path.join(OUT, f"pace_{ds}.json")))
    h = pace["headline"]
    C_lo, C_hi = h["C_lo"], h["C_hi"]
    static_mp = h["static_mp_config"]
    tau = h["tau_star"]; T = pace["temperature"]

    Xtr, ytr, Xval, yval, Xte, yte, Xcal = split(ds)
    m = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
    am, wm = A.calibrate(m, Xcal)

    # predictions for each method (seed 0, test)
    pred_fp = m.predict_fp(Xte) if hasattr(m, "predict_fp") else m.forward_fp(Xte).argmax(1)
    pred_i8 = A.forward_quant(m, Xte, C_hi, am, wm).argmax(1)
    pred_mp = A.forward_quant(m, Xte, static_mp, am, wm).argmax(1)
    lg_lo, pr_lo = A.forward_quant(m, Xte, C_lo, am, wm, return_probs=True)
    pred_hi = pred_i8
    pred_lo = pr_lo.argmax(1)
    sc = gate_msp(A._softmax(lg_lo / T))
    esc = sc < tau
    pred_pace = np.where(esc, pred_hi, pred_lo)

    pc = {}
    for name, pred in [("fp32", pred_fp), ("int8", pred_i8), ("static_mp", pred_mp), ("pace", pred_pace)]:
        p, r, f, s = per_class(pred, yte)
        pc[name] = {"precision": p, "recall": r, "f1": f, "support": s,
                    "macro_f1": float(np.mean(f)), "acc": float((pred == yte).mean())}

    # escalation by true class
    esc_by_class = []
    for k in range(10):
        mask = (yte == k)
        esc_by_class.append({"class": k, "esc_rate": float(esc[mask].mean()),
                             "lo_err": float((pred_lo[mask] != k).mean()),
                             "support": int(mask.sum())})

    # threshold operating points from msp curve
    cur = pace["curves"]["msp"]
    ops = []
    for target in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]:
        # nearest by escalation
        best = min(cur, key=lambda c: abs(c["esc"] - target))
        ops.append({"tau": round(best["tau"], 3), "esc": round(best["esc"], 4),
                    "energy_nJ": round(best["energy_pj"]/1000, 2), "test_acc": round(best["test_acc"], 4)})
    # dedup by tau
    seen = set(); ops2 = []
    for o in ops:
        if o["tau"] not in seen:
            seen.add(o["tau"]); ops2.append(o)

    out_all[ds] = {"per_class": pc, "esc_by_class": esc_by_class, "operating_points": ops2,
                   "C_lo": C_lo, "C_hi": C_hi, "static_mp": static_mp}
    print(f"[{ds}] macro-F1: FP32={pc['fp32']['macro_f1']:.4f} INT8={pc['int8']['macro_f1']:.4f} "
          f"MP={pc['static_mp']['macro_f1']:.4f} PACE={pc['pace']['macro_f1']:.4f}")
    print(f"  escalation by class (rate): " +
          " ".join(f"{e['class']}:{e['esc_rate']:.2f}" for e in esc_by_class))

json.dump(out_all, open(os.path.join(OUT, "perclass.json"), "w"), indent=2)
print("saved perclass.json")
