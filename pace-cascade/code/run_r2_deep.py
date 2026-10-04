"""run_r2_deep.py -- deeper / modern architectures (second revision).

  python run_r2_deep.py --model resnet8     # ResNet-8 on CIFAR-10 (JPEG mirror)
  python run_r2_deep.py --model vit4        # ViT-4 on Fashion-MNIST

Steps (all on top of the trained weights from deepnets.py):
  1. fidelity check: NumPy floating-point engine vs. the JAX-trained model;
  2. exhaustive enumeration of all |O|^G stage-wise configurations on a
     validation subset and a test subset (seed 0), cached to results/caches;
  3. the paper's deployment protocol (C_lo, static-MP and tau chosen on
     validation; C_hi = uniform INT8), applied unchanged to every seed on the
     full test set;
  4. oracle-tuned comparison: exact static frontier vs. the lower-energy envelope
     of cascades over all frontier pairs, plus the oracle gate;
  5. challenging inputs: 4 corruptions x 3 severities (fixed clean-data choices).
Output: results/r2_<model>.json
"""
import os, json, time, argparse
import numpy as np
import r2lib as L
import deepquant as Q
import deepnets as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet8", "vit4"])
    args = ap.parse_args()
    t00 = time.time()
    if args.model == "resnet8":
        cfg = D.load_config("resnet8_cifar10.json")
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = D.cifar10_splits(cfg)
        f = lambda X: X.astype(np.float32) / 255.0
        Xval, Xte, Xcal = f(Xval), f(Xte), f(Xcal)
        mk = lambda s: Q.ResNet8Q(os.path.join(D.OUT, f"resnet8_cifar10_seed{s}.npz"), Xcal)
        vsub, tsub, csub = slice(0, None, 3), slice(0, None, 4), slice(0, None, 10)   # 1667/2500/1000 (class-balanced)
        GM = Q.ResNet8Q.GROUP_MACS
    else:
        cfg = D.load_config("vit4_fmnist.json")
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = D.fmnist_splits()
        mk = lambda s: Q.ViT4Q(os.path.join(D.OUT, f"vit4_fmnist_seed{s}.npz"), cfg, Xcal)
        vsub, tsub, csub = slice(0, None, 2), slice(0, None, 2), slice(0, None, 5)   # 5000/5000/2000
        GM = Q.ViT4Q.GROUP_MACS
    tag = args.model
    H0 = L.load_energy_models()["H0"]
    fps = json.load(open(os.path.join(L.RES, f"{tag}_fp_summary.json")))
    seeds = cfg["seeds"]
    out = {"model": tag, "group_macs": GM, "total_macs": int(sum(GM)), "options": L.DEEP_OPTIONS,
           "n_val_enum": int(len(yval[vsub])), "n_test_enum": int(len(yte[tsub])),
           "n_test_corruption": int(len(yte[csub]))}

    eng0 = mk(0)
    acc_np = float((eng0.forward(Xte, None).argmax(1) == yte).mean())
    out["fidelity"] = {"fp32_numpy_test_acc_seed0": acc_np, "fp32_jax_test_acc_seed0": fps["0"]["test_acc"]}
    print(f"[{tag}] fidelity: numpy FP32 {acc_np:.4f} vs JAX {fps['0']['test_acc']:.4f}", flush=True)

    # ---- 2. enumeration ------------------------------------------------------
    t0 = time.time()
    configs, pv, mv = Q.enumerate_all(eng0, Xval[vsub], L.DEEP_OPTIONS)
    L.save_cache(os.path.join(L.CACHE, f"{tag}_val.npz"), configs, pv, mv, yval[vsub])
    print(f"[{tag}] val enumeration of {len(configs)} configs: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    _, pt, mt = Q.enumerate_all(eng0, Xte[tsub], L.DEEP_OPTIONS)
    L.save_cache(os.path.join(L.CACHE, f"{tag}_test.npz"), configs, pt, mt, yte[tsub])
    print(f"[{tag}] test enumeration: {time.time()-t0:.0f}s", flush=True)
    vc, tc = L.Cache(configs, pv, mv, yval[vsub]), L.Cache(configs, pt, mt, yte[tsub])
    E = L.energies(configs, GM, H0)
    G = len(GM)
    i8 = vc.index[tuple([(8, 0)] * G)]

    # ---- 3. protocol -------------------------------------------------------
    sel = L.select_protocol(vc, E, i8)
    c_lo, c_hi, c_smp = configs[sel["lo"]], configs[i8], configs[sel["smp"]]
    out["selection"] = {"C_lo": c_lo, "C_hi": c_hi, "C_smp": c_smp, "tau": sel["tau"],
                        "E_lo": sel["E_lo"], "E_hi": sel["E_hi"], "E_smp": sel["E_smp"],
                        "E_fp32": float(sum(GM) * L.FP32_PJ),
                        "val_acc_lo": float(vc.acc[sel["lo"]]), "val_acc_int8": float(vc.acc[i8]),
                        "val_acc_smp": float(vc.acc[sel["smp"]])}
    out["feasibility"] = L.feasibility(vc, E, sel)
    out["observed_seed0_cache"] = L.observed_on_cache(tc, E, sel)

    # ---- 4. oracle-tuned comparison (seed 0, test subset) --------------------
    out["oracle_compare"] = L.compare_frontier(tc, E, i8)
    fr = L.pareto_idx(E, tc.acc)
    out["frontier_test"] = [[float(E[i]), float(tc.acc[i])] for i in fr]
    out["all_test"] = [[float(E[i]), float(tc.acc[i])] for i in range(len(configs))]
    pesc, cacc = L.pair_curve(tc, sel["lo"], i8)
    sub = np.linspace(0, len(pesc) - 1, 400).astype(int)
    out["trace"] = {"E": (sel["E_lo"] + pesc[sub] * sel["E_hi"]).tolist(), "acc": cacc[sub].tolist()}
    wrong = ~tc.correct[sel["lo"]]
    out["oracle_gate"] = {"acc": float(np.where(wrong, tc.correct[i8], tc.correct[sel["lo"]]).mean()),
                          "E": sel["E_lo"] + float(wrong.mean()) * sel["E_hi"], "esc": float(wrong.mean())}
    print(f"[{tag}] C_lo={c_lo} E_lo={sel['E_lo']/1e3:.1f}nJ  static-MP={c_smp} E={sel['E_smp']/1e3:.1f}nJ  "
          f"INT8 E={sel['E_hi']/1e3:.1f}nJ  tau={sel['tau']:.3f}  S*={out['oracle_compare']['S_star']:+.3f}",
          flush=True)

    # ---- 3b + 5. multi-seed headline (full test) and corruptions --------------
    hl = {k: [] for k in ["fp32", "int8", "smp", "lo", "casc", "esc", "E"]}
    sets = [("clean", 0)] + L.corruption_sets()
    per = {f"{k}|{s}": {"int8": [], "smp": [], "casc": [], "esc": [], "E": []} for k, s in sets}
    Xc0 = Xte[csub]; yc = yte[csub]
    for s in seeds:
        eng = eng0 if s == 0 else mk(s)
        def preds(X, c):
            p = L.softmax(eng.forward(X, c))
            return p.argmax(1), p.max(1)
        p8, _ = preds(Xte, c_hi); psm, _ = preds(Xte, c_smp); plo, mlo = preds(Xte, c_lo)
        esc = mlo < sel["tau"]
        hl["fp32"].append(fps[str(s)]["test_acc"]); hl["int8"].append(float((p8 == yte).mean()))
        hl["smp"].append(float((psm == yte).mean())); hl["lo"].append(float((plo == yte).mean()))
        hl["casc"].append(float((np.where(esc, p8, plo) == yte).mean())); hl["esc"].append(float(esc.mean()))
        hl["E"].append(sel["E_lo"] + float(esc.mean()) * sel["E_hi"])
        for k, sv in sets:
            Xc = Xc0 if k == "clean" else L.corrupt(Xc0, k, sv)
            q8, _ = preds(Xc, c_hi); qsm, _ = preds(Xc, c_smp); qlo, mq = preds(Xc, c_lo)
            e = mq < sel["tau"]
            r = per[f"{k}|{sv}"]
            r["int8"].append(float((q8 == yc).mean())); r["smp"].append(float((qsm == yc).mean()))
            r["casc"].append(float((np.where(e, q8, qlo) == yc).mean())); r["esc"].append(float(e.mean()))
            r["E"].append(sel["E_lo"] + float(e.mean()) * sel["E_hi"])
        print(f"[{tag}] seed {s} done ({time.time()-t00:.0f}s): int8={hl['int8'][-1]:.4f} smp={hl['smp'][-1]:.4f} "
              f"casc={hl['casc'][-1]:.4f} esc={hl['esc'][-1]:.3f}", flush=True)
    out["headline"] = {k: L.ms(v) for k, v in hl.items()}
    out["headline_per_seed"] = hl
    out["corruption"] = {"per_set": {key: {kk: L.ms(v) for kk, v in r.items()} for key, r in per.items()},
                         "per_seed": per}
    L.dump(out, f"r2_{tag}.json")
    print(f"saved r2_{tag}.json ({time.time()-t00:.0f}s)")


if __name__ == "__main__":
    main()
