"""run_revision.py -- experiments added in revision.

(A) Multi-pair cascade sweep (Reviewer 4.1): instead of a single (C_lo,C_hi)
    pair, sweep MANY pairs drawn from the static Pareto frontier and test whether
    ANY pair's input-adaptive cascade dominates the static frontier.
(B) Per-seed selection (Reviewer 4.2): select C_lo, C_hi and tau independently on
    each seed's own validation split and evaluate on that seed's test split;
    compare against the seed-0-fixed protocol used in the main paper.
"""
import os, json, numpy as np
import axlib as A
from run_train import SIZES, SEEDS, split

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")
DATASETS = ["mnist", "fmnist"]

def msp(probs):
    return probs.max(1)

def cascade_curve(pr_lo, pred_lo, pred_hi, y, e_lo, e_hi, taus):
    sc = msp(pr_lo)
    pts = []
    for t in taus:
        esc = sc < t
        pred = np.where(esc, pred_hi, pred_lo)
        pts.append((e_lo + float(esc.mean()) * e_hi, float((pred == y).mean()), float(esc.mean())))
    return pts

def static_energy_at_acc(front_pts, target):
    cand = [e for (e, a) in front_pts if a >= target - 1e-9]
    return min(cand) if cand else float("inf")

def main():
    taus = np.linspace(0.20, 1.0001, 120)
    summary = {}
    for ds in DATASETS:
        expl = json.load(open(os.path.join(OUT, f"explore_{ds}.json")))
        recs = expl["records"]; front = expl["pareto_idx"]
        fp32_e = expl["fp32_energy_pj"]
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = split(ds)
        m = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
        am, wm = A.calibrate(m, Xcal)

        # static frontier in TEST space (energy, test_acc)
        front_test = []
        for i in front:
            r = recs[i]
            ta = r["test_acc"]
            if ta is None:
                ta = A.accuracy(A.forward_quant(m, Xte, r["config"], am, wm), yte)
            front_test.append((r["energy_pj"], ta))

        # candidate cheap configs: 6 cheapest Pareto points; candidate hi: 5 most
        # accurate Pareto points + uniform INT8
        fr_sorted_e = sorted(front, key=lambda i: recs[i]["energy_pj"])
        fr_sorted_a = sorted(front, key=lambda i: -recs[i]["val_acc"])
        lo_cands = fr_sorted_e[:6]
        int8_idx = expl["uniform_idx"]["(8, 0)"]
        hi_cands = list(dict.fromkeys(fr_sorted_a[:5] + [int8_idx]))

        # precompute predictions/probs for all candidate configs on test
        cache = {}
        for i in set(lo_cands) | set(hi_cands):
            lg, pr = A.forward_quant(m, Xte, recs[i]["config"], am, wm, return_probs=True)
            cache[i] = (pr, pr.argmax(1), recs[i]["energy_pj"])

        pairs = []
        for li in lo_cands:
            for hi in hi_cands:
                e_lo = recs[li]["energy_pj"]; e_hi = recs[hi]["energy_pj"]
                if e_lo >= e_hi:
                    continue
                pr_lo, pred_lo, _ = cache[li]
                _, pred_hi, _ = cache[hi]
                curve = cascade_curve(pr_lo, pred_lo, pred_hi, yte, e_lo, e_hi, taus)
                # domination margin: min over curve of (cascade_E - static_E at same acc)
                # (negative => cascade beats static frontier somewhere)
                margin = min(ce - static_energy_at_acc(front_test, ca) for (ce, ca, _) in curve)
                # cascade energy to reach C_hi accuracy target (within 0.2pp)
                hi_acc = A.accuracy(A.forward_quant(m, Xte, recs[hi]["config"], am, wm), yte)
                reach = [ce for (ce, ca, _) in curve if ca >= hi_acc - 0.002]
                casc_E_at_hi = min(reach) if reach else float("inf")
                static_E_at_hi = static_energy_at_acc(front_test, hi_acc - 0.002)
                pairs.append({
                    "lo_cfg": recs[li]["config"], "hi_cfg": recs[hi]["config"],
                    "e_lo_nJ": e_lo/1000, "e_hi_nJ": e_hi/1000,
                    "domination_margin_nJ": margin/1000,
                    "cascade_E_at_hiacc_nJ": casc_E_at_hi/1000 if np.isfinite(casc_E_at_hi) else None,
                    "static_E_at_hiacc_nJ": static_E_at_hi/1000 if np.isfinite(static_E_at_hi) else None,
                    "hi_acc": hi_acc,
                    "curve": [(round(ce/1000,3), round(ca,4), round(pe,4)) for (ce,ca,pe) in curve],
                })
        n_dominating = sum(1 for p in pairs if p["domination_margin_nJ"] < -1e-6)
        best = min(pairs, key=lambda p: p["domination_margin_nJ"])
        summary.setdefault(ds, {})["multipair"] = {
            "n_pairs": len(pairs), "n_dominating": n_dominating,
            "best_margin_nJ": best["domination_margin_nJ"],
            "best_pair": {"lo": best["lo_cfg"], "hi": best["hi_cfg"],
                          "e_lo_nJ": best["e_lo_nJ"], "e_hi_nJ": best["e_hi_nJ"]},
            "pairs": pairs, "front_test": [(round(e/1000,3), round(a,4)) for (e,a) in front_test],
        }
        print(f"\n[{ds}] MULTI-PAIR: {len(pairs)} pairs tested, {n_dominating} dominate the static frontier.")
        print(f"  best (most negative) domination margin = {best['domination_margin_nJ']:+.2f} nJ "
              f"(lo={best['lo_cfg']} hi={best['hi_cfg']})")
        print(f"  -> a positive margin means EVERY cascade operating point costs at least that much")
        print(f"     MORE energy than the static config at equal accuracy.")

        # ---------------- (B) per-seed selection ----------------
        # candidate C_lo restricted to Pareto configs; C_hi = uniform INT8.
        C_hi = recs[int8_idx]["config"]
        per_seed_fixed = json.load(open(os.path.join(OUT, f"pace_{ds}.json")))["headline"]
        rows = []
        for s in SEEDS:
            ms = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed{s}.npz"))
            ams, wms = A.calibrate(ms, Xcal)
            # this seed's val accuracies
            int8_val = A.accuracy(A.forward_quant(ms, Xval, C_hi, ams, wms), yval)
            # choose C_lo: cheapest Pareto cfg with THIS seed's val_acc >= int8_val-5pp
            best_lo, best_e = None, 1e18
            for i in front:
                cfg = recs[i]["config"]; e = recs[i]["energy_pj"]
                if e >= best_e:
                    continue
                va = A.accuracy(A.forward_quant(ms, Xval, cfg, ams, wms), yval)
                if va >= int8_val - 0.05 and e < best_e:
                    best_lo, best_e = cfg, e
            C_lo = best_lo; e_lo = best_e; e_hi = A.config_energy(C_hi, SIZES)
            # fit tau on THIS seed val to reach int8_val-0.2pp
            _, prv = A.forward_quant(ms, Xval, C_lo, ams, wms, return_probs=True)
            pv_lo = prv.argmax(1); ph_v = A.forward_quant(ms, Xval, C_hi, ams, wms).argmax(1)
            scv = msp(prv); tau = 1.0
            for t in taus:
                esc = scv < t
                acc = float((np.where(esc, ph_v, pv_lo) == yval).mean())
                if acc >= int8_val - 0.002:
                    tau = float(t); break
            # evaluate on THIS seed test
            _, prt = A.forward_quant(ms, Xte, C_lo, ams, wms, return_probs=True)
            pt_lo = prt.argmax(1); ph_t = A.forward_quant(ms, Xte, C_hi, ams, wms).argmax(1)
            esc = msp(prt) < tau
            acc = float((np.where(esc, ph_t, pt_lo) == yte).mean())
            en = e_lo + float(esc.mean()) * e_hi
            rows.append({"seed": s, "C_lo": C_lo, "tau": round(tau,3),
                         "esc": round(float(esc.mean()),4), "energy_nJ": round(en/1000,3),
                         "test_acc": round(acc,4)})
        acc_arr = np.array([r["test_acc"] for r in rows]); en_arr = np.array([r["energy_nJ"] for r in rows])
        summary[ds]["per_seed"] = {
            "rows": rows,
            "acc_mean": float(acc_arr.mean()), "acc_std": float(acc_arr.std()),
            "energy_mean_nJ": float(en_arr.mean()), "energy_std_nJ": float(en_arr.std()),
            "fixed_seed0": {"acc": per_seed_fixed["pace_acc"], "energy_nJ": [per_seed_fixed["pace_energy_pj"][0]/1000, per_seed_fixed["pace_energy_pj"][1]/1000]},
            "static_mp_energy_nJ": per_seed_fixed["static_mp_energy_pj"]/1000,
            "static_mp_acc": per_seed_fixed["static_mp_acc"],
        }
        print(f"[{ds}] PER-SEED selection: PACE acc={acc_arr.mean():.4f}+/-{acc_arr.std():.4f} "
              f"E={en_arr.mean():.2f}+/-{en_arr.std():.2f} nJ")
        print(f"        (paper's seed-0-fixed: acc={per_seed_fixed['pace_acc'][0]:.4f} "
              f"E={per_seed_fixed['pace_energy_pj'][0]/1000:.2f} nJ; "
              f"static-MP E={per_seed_fixed['static_mp_energy_pj']/1000:.2f} nJ)")
        for r in rows:
            print(f"          seed{r['seed']}: C_lo={r['C_lo']} tau={r['tau']} "
                  f"esc={r['esc']*100:.1f}% E={r['energy_nJ']:.2f}nJ acc={r['test_acc']:.4f}")

    json.dump(summary, open(os.path.join(OUT, "revision_r4.json"), "w"), indent=2)
    print("\nsaved revision_r4.json")

if __name__ == "__main__":
    main()
