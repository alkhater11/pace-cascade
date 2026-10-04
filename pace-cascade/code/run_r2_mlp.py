"""run_r2_mlp.py -- MLP experiments added in the second revision.

  (0) reproduction audit: retrained weights vs. the archived R1 results
  (1) per-configuration caches over the EXTENDED option set (9 options incl.
      effective widths 3 and 2 -> 729 configurations), seed 0, val + test
  (2) larger precision gaps: static frontier vs. best cascade envelope with the
      main 216-configuration space and with the extended 729-configuration space
  (3) challenging inputs: 4 corruptions x 3 severities, 5 seeds, configurations
      and threshold frozen from the clean protocol of the paper; plus the
      oracle comparison on corrupted data (seed 0, all 216 configurations)
  (4) capacity cascade: narrow 784-32-10 MLP (cheap arm) -> wide MLP (high arm),
      compared with the static frontier of the UNION of both design spaces
Outputs: results/r2_mlp.json and results/caches/*.npz
"""
import os, json, itertools, time
import numpy as np
import axlib as A
import r2lib as L
from run_train import SIZES, SEEDS, split

MACS = A.layer_macs(SIZES)                     # [200704, 32768, 1280]
NSIZES = [784, 32, 10]; NMACS = A.layer_macs(NSIZES)
EM = L.load_energy_models(); H0 = EM["H0"]
ORIG = os.path.join(L.ROOT, "results_orig")
MOD = os.path.join(L.RES, "models"); NMOD = os.path.join(L.RES, "models_narrow")
CFG_EXT = [[list(o) for o in c] for c in itertools.product(L.OPTIONS_EXT, repeat=3)]
IDX216 = [i for i, c in enumerate(CFG_EXT) if all(tuple(o) in L.OPTIONS for o in c)]
CFG_N = [[list(o) for o in c] for c in itertools.product(L.OPTIONS_EXT, repeat=2)]


def predict(m, X, cfg, am, wm):
    p = L.softmax(A.forward_quant(m, X, cfg, am, wm))
    return p.argmax(1), p.max(1)


def build_cache(m, X, configs, am, wm):
    pred = np.zeros((len(configs), len(X)), np.int16); msp = np.zeros((len(configs), len(X)), np.float32)
    for i, c in enumerate(configs):
        pred[i], msp[i] = predict(m, X, c, am, wm)
        if (i + 1) % 243 == 0:
            print(f"    cache {i+1}/{len(configs)}", flush=True)
    return pred, msp


def corr_flat(X, kind, sev, seed=0):
    return L.corrupt(X.reshape(-1, 28, 28, 1), kind, sev, seed).reshape(len(X), -1)


def main():
    out = {}
    t00 = time.time()
    fp_new = json.load(open(os.path.join(L.RES, "fp_summary.json")))
    fp_old = json.load(open(os.path.join(ORIG, "fp_summary.json")))
    for ds in ["mnist", "fmnist"]:
        R = out[ds] = {}
        Xtrain, ytrain, Xval, yval, Xte, yte, Xcal = split(ds)
        m0 = A.NumpyMLP.load(os.path.join(MOD, f"{ds}_seed0.npz")); am0, wm0 = A.calibrate(m0, Xcal)

        # ---- (0) reproduction audit ----------------------------------------
        d_fp = max(abs(fp_new[ds][str(s)]["test_acc"] - fp_old[ds][str(s)]["test_acc"]) for s in SEEDS)

        # ---- (1) caches ----------------------------------------------------
        t0 = time.time()
        fv, ft = os.path.join(L.CACHE, f"mlp_{ds}_val_ext.npz"), os.path.join(L.CACHE, f"mlp_{ds}_test_ext.npz")
        if os.path.exists(fv) and os.path.exists(ft):          # resume
            zv, zt = np.load(fv), np.load(ft); pv, mv, pt, mt = zv["pred"], zv["msp"], zt["pred"], zt["msp"]
        else:
            pv, mv = build_cache(m0, Xval, CFG_EXT, am0, wm0)
            pt, mt = build_cache(m0, Xte, CFG_EXT, am0, wm0)
            L.save_cache(fv, CFG_EXT, pv, mv, yval); L.save_cache(ft, CFG_EXT, pt, mt, yte)
        vc, tc = L.Cache(CFG_EXT, pv, mv, yval), L.Cache(CFG_EXT, pt, mt, yte)
        print(f"[{ds}] caches built ({time.time()-t0:.0f}s)", flush=True)
        expl = json.load(open(os.path.join(ORIG, f"explore_{ds}.json")))
        d_val = max(abs(vc.acc[vc.index[tuple(tuple(o) for o in r["config"])]] - r["val_acc"])
                    for r in expl["records"])
        R["reproduction"] = {"max_abs_diff_fp32_test_acc": d_fp, "max_abs_diff_216_val_acc": float(d_val)}
        print(f"[{ds}] reproduction: |dFP32 test|max={d_fp:.5f}  |d val acc (216 cfgs)|max={d_val:.5f}", flush=True)

        # ---- (2) larger precision gaps -------------------------------------
        E_ext = L.energies(CFG_EXT, MACS, H0)
        i8 = vc.index[((8, 0), (8, 0), (8, 0))]
        tc216, vc216 = tc.subset(IDX216), vc.subset(IDX216)
        E216 = E_ext[IDX216]; i8_216 = vc216.index[((8, 0), (8, 0), (8, 0))]
        R["gap_216"] = L.compare_frontier(tc216, E216, i8_216)
        R["gap_729"] = L.compare_frontier(tc, E_ext, i8)
        sel = L.select_protocol(vc, E_ext, i8)
        R["gap_729_protocol"] = {"lo": CFG_EXT[sel["lo"]], "smp": CFG_EXT[sel["smp"]], "tau": sel["tau"],
                                 **L.feasibility(vc, E_ext, sel), **L.observed_on_cache(tc, E_ext, sel),
                                 "E_lo": sel["E_lo"], "E_hi": sel["E_hi"], "E_smp": sel["E_smp"]}
        sel216 = L.select_protocol(vc216, E216, i8_216)
        R["gap_216_protocol"] = {"lo": [list(o) for o in vc216.configs[sel216["lo"]]], "tau": sel216["tau"],
                                 **L.feasibility(vc216, E216, sel216), **L.observed_on_cache(tc216, E216, sel216)}
        print(f"[{ds}] precision gaps: S*(216)={R['gap_216']['S_star']:+.3f} S*(729)={R['gap_729']['S_star']:+.3f} "
              f"Smax(729)={R['gap_729']['S_max']:+.3f}", flush=True)

        # ---- (3) corruptions (paper protocol, 5 seeds) ---------------------
        P = json.load(open(os.path.join(ORIG, f"pace_{ds}.json")))["headline"]
        C_lo, C_hi, C_smp, tau = P["C_lo"], P["C_hi"], P["static_mp_config"], P["tau_star"]
        E_lo, E_hi = A.config_energy(C_lo, SIZES), A.config_energy(C_hi, SIZES)
        E_smp = A.config_energy(C_smp, SIZES)
        sets = [("clean", 0)] + L.corruption_sets()
        per = {f"{k}|{s}": {"int8": [], "smp": [], "casc": [], "esc": [], "E": []} for k, s in sets}
        for sd in SEEDS:
            ms_ = A.NumpyMLP.load(os.path.join(MOD, f"{ds}_seed{sd}.npz")); am, wm = A.calibrate(ms_, Xcal)
            for k, s in sets:
                Xc = Xte if k == "clean" else corr_flat(Xte, k, s)
                p8, _ = predict(ms_, Xc, C_hi, am, wm); psm, _ = predict(ms_, Xc, C_smp, am, wm)
                plo, mlo = predict(ms_, Xc, C_lo, am, wm)
                esc = mlo < tau
                r = per[f"{k}|{s}"]
                r["int8"].append(float((p8 == yte).mean())); r["smp"].append(float((psm == yte).mean()))
                r["casc"].append(float((np.where(esc, p8, plo) == yte).mean()))
                r["esc"].append(float(esc.mean())); r["E"].append(E_lo + float(esc.mean()) * E_hi)
        R["corruption"] = {"configs": {"C_lo": C_lo, "C_hi": C_hi, "C_smp": C_smp, "tau": tau,
                                       "E_lo": E_lo, "E_hi": E_hi, "E_smp": E_smp},
                           "per_set": {key: {kk: L.ms(v) for kk, v in r.items()} for key, r in per.items()},
                           "per_seed": per}
        # oracle comparison on corrupted data (seed 0, 216 configs, severity 2)
        R["corruption_oracle"] = {}
        for k in L.CORRUPTIONS:
            Xc = corr_flat(Xte, k, 2)
            pc, mc = build_cache(m0, Xc, [CFG_EXT[i] for i in IDX216], am0, wm0)
            cc = L.Cache([CFG_EXT[i] for i in IDX216], pc, mc, yte)
            cmpk = L.compare_frontier(cc, E216, i8_216)
            for kk in ("grid", "static_curve", "envelope"):
                cmpk.pop(kk)
            R["corruption_oracle"][k] = cmpk
        print(f"[{ds}] corruptions done ({time.time()-t00:.0f}s)", flush=True)

        # ---- (4) capacity (width-and-precision) cascade ----------------------
        n0 = A.NumpyMLP.load(os.path.join(NMOD, f"{ds}_seed0.npz")); amn, wmn = A.calibrate(n0, Xcal)
        npv, nmv = build_cache(n0, Xval, CFG_N, amn, wmn)
        npt, nmt = build_cache(n0, Xte, CFG_N, amn, wmn)
        CFG_N3 = [c + [[0, 0]] for c in CFG_N]
        L.save_cache(os.path.join(L.CACHE, f"narrow_{ds}_val.npz"), CFG_N3, npv, nmv, yval)
        L.save_cache(os.path.join(L.CACHE, f"narrow_{ds}_test.npz"), CFG_N3, npt, nmt, yte)
        U = CFG_EXT + CFG_N3
        uv = L.Cache(U, np.concatenate([pv, npv]), np.concatenate([mv, nmv]), yval)
        ut = L.Cache(U, np.concatenate([pt, npt]), np.concatenate([mt, nmt]), yte)
        E_u = np.concatenate([E_ext, L.energies(CFG_N, NMACS, H0)])
        selu = L.select_protocol(uv, E_u, i8)
        cap = {"lo": U[selu["lo"]], "smp": U[selu["smp"]], "tau": selu["tau"],
               "lo_is_narrow": bool(selu["lo"] >= len(CFG_EXT)), "smp_is_narrow": bool(selu["smp"] >= len(CFG_EXT)),
               "E_lo": selu["E_lo"], "E_hi": selu["E_hi"], "E_smp": selu["E_smp"],
               **L.feasibility(uv, E_u, selu), **L.observed_on_cache(ut, E_u, selu)}
        cmpu = L.compare_frontier(ut, E_u, i8)
        cap["oracle_compare"] = cmpu
        # cascade trace (seed 0, test) for the figure
        pesc, cacc = L.pair_curve(ut, selu["lo"], i8)
        sub = np.linspace(0, len(pesc) - 1, 400).astype(int)
        cap["trace"] = {"E": (selu["E_lo"] + pesc[sub] * selu["E_hi"]).tolist(), "acc": cacc[sub].tolist()}
        fr_w = L.pareto_idx(E_ext, tc.acc); fr_n = L.pareto_idx(E_u[len(CFG_EXT):], ut.acc[len(CFG_EXT):])
        cap["frontier_wide"] = [[float(E_ext[i]), float(tc.acc[i])] for i in fr_w]
        cap["frontier_narrow"] = [[float(E_u[len(CFG_EXT) + i]), float(ut.acc[len(CFG_EXT) + i])] for i in fr_n]
        # five-seed headline with the seed-0 selection
        lo_c, smp_c, lo_n, smp_n = cap["lo"], cap["smp"], cap["lo_is_narrow"], cap["smp_is_narrow"]
        hl = {"int8": [], "smp": [], "casc": [], "esc": [], "E": [], "lo": [], "fp32_narrow": []}
        nsum = json.load(open(os.path.join(L.RES, "narrow_fp_summary.json")))
        for sd in SEEDS:
            mw = A.NumpyMLP.load(os.path.join(MOD, f"{ds}_seed{sd}.npz")); amw, wmw = A.calibrate(mw, Xcal)
            mn = A.NumpyMLP.load(os.path.join(NMOD, f"{ds}_seed{sd}.npz")); ama, wma = A.calibrate(mn, Xcal)
            def run(c, narrow):
                return predict(mn, Xte, c[:2], ama, wma) if narrow else predict(mw, Xte, c, amw, wmw)
            p8, _ = predict(mw, Xte, [[8, 0]] * 3, amw, wmw)
            psm, _ = run(smp_c, smp_n); plo, mlo = run(lo_c, lo_n)
            esc = mlo < selu["tau"]
            hl["int8"].append(float((p8 == yte).mean())); hl["smp"].append(float((psm == yte).mean()))
            hl["lo"].append(float((plo == yte).mean()))
            hl["casc"].append(float((np.where(esc, p8, plo) == yte).mean())); hl["esc"].append(float(esc.mean()))
            hl["E"].append(selu["E_lo"] + float(esc.mean()) * selu["E_hi"])
            hl["fp32_narrow"].append(nsum[ds][str(sd)]["test_acc"])
        cap["headline"] = {k: L.ms(v) for k, v in hl.items()}
        cap["headline_per_seed"] = hl
        R["capacity"] = cap
        print(f"[{ds}] capacity: lo={lo_c} (narrow={lo_n}) E_lo={selu['E_lo']/1e3:.2f}nJ "
              f"cascade E={cap['headline']['E'][0]/1e3:.2f}nJ acc={cap['headline']['casc'][0]:.4f} | "
              f"static-MP E={selu['E_smp']/1e3:.2f}nJ acc={cap['headline']['smp'][0]:.4f} | "
              f"S*={cmpu['S_star']:+.3f}", flush=True)
    L.dump(out, "r2_mlp.json")
    print(f"saved r2_mlp.json ({time.time()-t00:.0f}s)")


if __name__ == "__main__":
    main()
