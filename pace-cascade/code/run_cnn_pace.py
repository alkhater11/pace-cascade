"""CNN generality study: sensitivity, reduced static Pareto, and the PACE cascade
on SmallCNN. Reduced option set (4 opts, 4 layers = 256 configs) evaluated on a
3000-image validation subset for tractability; selected configs use full test."""
import os, json, itertools, numpy as np
import axlib as A
import cnnlib as C
from run_cnn_train import split4, SEEDS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "cnn_models")
OPTIONS = [(8, 0), (6, 0), (4, 1)]     # eff 8,6,3 (reduced for CNN tractability)
NVAL_SWEEP = 1500

def pareto(pts):
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], -pts[i][1]))
    fr, best = [], -1
    for i in order:
        if pts[i][1] > best + 1e-12:
            fr.append(i); best = pts[i][1]
    return fr

def msp(p): return p.max(1)

def main():
    fp_sum = json.load(open(os.path.join(OUT, "cnn_fp_summary.json")))
    res_path = os.path.join(OUT, "cnn_results.json")
    out_all = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for ds in ["mnist", "fmnist"]:
        if ds in out_all:
            print(f"[{ds}] already done, skipping"); continue
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = split4(ds)
        m = C.SmallCNN.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
        am, wm = C.cnn_calibrate(m, Xcal)
        macs = C.cnn_layer_macs(m)
        fp32_e = sum(macs) * 4.6
        Xvs, yvs = Xval[:NVAL_SWEEP], yval[:NVAL_SWEEP]

        # sensitivity: one layer approximated, others (8,0)
        base = [(8, 0)] * 4
        base_acc = A.accuracy(C.forward_quant_cnn(m, Xvs, base, am, wm), yvs)
        sens = {}
        for l in range(4):
            sens[l] = []
            for opt in OPTIONS:
                cfg = list(base); cfg[l] = opt
                acc = A.accuracy(C.forward_quant_cnn(m, Xvs, cfg, am, wm), yvs)
                sens[l].append({"opt": list(opt), "eff": opt[0]-opt[1], "drop": base_acc-acc})

        # reduced enumeration (256) on val subset
        records = []
        for combo in itertools.product(OPTIONS, repeat=4):
            cfg = [list(c) for c in combo]
            e = C.cnn_config_energy(cfg, macs)
            va = A.accuracy(C.forward_quant_cnn(m, Xvs, cfg, am, wm), yvs)
            records.append({"config": cfg, "energy_pj": e, "val_acc": va, "test_acc": None})
        fr = pareto([(r["energy_pj"], r["val_acc"], i) for i, r in enumerate(records)])
        int8_cfg = [(8, 0)]*4
        # test acc for pareto + int8
        int8_idx = next(i for i, r in enumerate(records) if r["config"] == [list(o) for o in int8_cfg])
        for i in set(fr) | {int8_idx}:
            records[i]["test_acc"] = A.accuracy(C.forward_quant_cnn(m, Xte, records[i]["config"], am, wm), yte)

        # select C_hi=INT8, C_lo=cheapest pareto within 5pp of int8 val
        int8_val = records[int8_idx]["val_acc"]
        cl = [i for i in fr if records[i]["val_acc"] >= int8_val - 0.05]
        c_lo_idx = min(cl, key=lambda i: records[i]["energy_pj"])
        C_lo = records[c_lo_idx]["config"]; C_hi = records[int8_idx]["config"]
        e_lo = records[c_lo_idx]["energy_pj"]; e_hi = records[int8_idx]["energy_pj"]
        # static MP: cheapest pareto matching int8 val within 0.2pp
        mm = [i for i in fr if records[i]["val_acc"] >= int8_val - 0.002]
        static_idx = min(mm, key=lambda i: records[i]["energy_pj"]) if mm else int8_idx
        static_cfg = records[static_idx]["config"]; static_e = records[static_idx]["energy_pj"]

        # tau* on seed0 val subset (full val) to match int8 acc within 0.2pp
        _, prv = C.forward_quant_cnn(m, Xval, C_lo, am, wm, return_probs=True)
        pv_lo = prv.argmax(1); ph_v = C.forward_quant_cnn(m, Xval, C_hi, am, wm).argmax(1)
        int8_val_full = A.accuracy(C.forward_quant_cnn(m, Xval, C_hi, am, wm), yval)
        taus = np.linspace(0.2, 1.0001, 100); scv = msp(prv); tau = 1.0
        for t in taus:
            esc = scv < t
            if float((np.where(esc, ph_v, pv_lo) == yval).mean()) >= int8_val_full - 0.002:
                tau = float(t); break

        # cascade test curve (seed0)
        _, prt = C.forward_quant_cnn(m, Xte, C_lo, am, wm, return_probs=True)
        pt_lo = prt.argmax(1); ph_t = C.forward_quant_cnn(m, Xte, C_hi, am, wm).argmax(1)
        sct = msp(prt); curve = []
        for t in taus:
            esc = sct < t
            curve.append({"tau": float(t), "esc": float(esc.mean()),
                          "energy_pj": e_lo + float(esc.mean())*e_hi,
                          "test_acc": float((np.where(esc, ph_t, pt_lo) == yte).mean())})

        # multi-seed headline
        ps = {"fp32": [], "int8": [], "static": [], "pace_e": [], "pace_acc": [], "pace_esc": []}
        for s in SEEDS:
            ms = C.SmallCNN.load(os.path.join(MODELS, f"{ds}_seed{s}.npz"))
            ams, wms = C.cnn_calibrate(ms, Xcal)
            ps["fp32"].append(fp_sum[ds][str(s)]["test_acc"])
            ps["int8"].append(A.accuracy(C.forward_quant_cnn(ms, Xte, C_hi, ams, wms), yte))
            ps["static"].append(A.accuracy(C.forward_quant_cnn(ms, Xte, static_cfg, ams, wms), yte))
            _, pr = C.forward_quant_cnn(ms, Xte, C_lo, ams, wms, return_probs=True)
            p_hi = C.forward_quant_cnn(ms, Xte, C_hi, ams, wms).argmax(1)
            esc = msp(pr) < tau
            ps["pace_e"].append(e_lo + float(esc.mean())*e_hi)
            ps["pace_acc"].append(float((np.where(esc, p_hi, pr.argmax(1)) == yte).mean()))
            ps["pace_esc"].append(float(esc.mean()))
        ms_ = lambda x: [float(np.mean(x)), float(np.std(x))]

        out_all[ds] = {
            "macs": macs, "fp32_energy_pj": fp32_e, "options": OPTIONS,
            "sensitivity": sens, "base_acc_int8": base_acc,
            "pareto": [{"config": records[i]["config"], "energy_pj": records[i]["energy_pj"],
                        "val_acc": records[i]["val_acc"], "test_acc": records[i]["test_acc"]} for i in fr],
            "C_lo": C_lo, "C_hi": C_hi, "e_lo_pj": e_lo, "e_hi_pj": e_hi,
            "static_cfg": static_cfg, "static_e_pj": static_e, "tau_star": tau,
            "curve": curve,
            "headline": {"fp32_acc": ms_(ps["fp32"]), "int8_acc": ms_(ps["int8"]),
                         "static_acc": ms_(ps["static"]), "pace_acc": ms_(ps["pace_acc"]),
                         "pace_energy_pj": ms_(ps["pace_e"]), "pace_esc": ms_(ps["pace_esc"]),
                         "int8_energy_pj": e_hi, "static_energy_pj": static_e, "fp32_energy_pj": fp32_e},
        }
        h = out_all[ds]["headline"]
        print(f"\n=== {ds} CNN ===")
        print(f"C_lo={C_lo} E={e_lo/1000:.2f}nJ  C_hi(INT8)={C_hi} E={e_hi/1000:.2f}nJ  "
              f"static={static_cfg} E={static_e/1000:.2f}nJ  tau*={tau:.3f}")
        print(f"[3-seed] FP32 {h['fp32_acc'][0]*100:.2f}±{h['fp32_acc'][1]*100:.2f} @ {fp32_e/1000:.1f}nJ")
        print(f"[3-seed] INT8 {h['int8_acc'][0]*100:.2f}±{h['int8_acc'][1]*100:.2f} @ {e_hi/1000:.2f}nJ "
              f"({fp32_e/e_hi:.1f}x)")
        print(f"[3-seed] static-MP {h['static_acc'][0]*100:.2f}±{h['static_acc'][1]*100:.2f} @ "
              f"{static_e/1000:.2f}nJ ({fp32_e/static_e:.1f}x)")
        print(f"[3-seed] PACE {h['pace_acc'][0]*100:.2f}±{h['pace_acc'][1]*100:.2f} @ "
              f"{h['pace_energy_pj'][0]/1000:.2f}±{h['pace_energy_pj'][1]/1000:.2f}nJ "
              f"esc={h['pace_esc'][0]*100:.1f}% ({fp32_e/h['pace_energy_pj'][0]:.1f}x)")
        json.dump(out_all, open(res_path, "w"), indent=2)   # checkpoint per dataset
    print("\nsaved cnn_results.json")

if __name__ == "__main__":
    main()
