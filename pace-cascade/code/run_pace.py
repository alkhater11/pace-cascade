"""run_pace.py -- PACE input-adaptive cascade: tau-sweep, multi-seed headline,
gate-type / temperature-scaling / oracle-random ablations.

Cascade semantics (honest energy accounting):
  stage 1 runs the cheap config C_lo on every input.
  if a per-input confidence gate g(x) < tau  ->  escalate: re-run C_hi.
  escalated inputs therefore pay  E_lo + E_hi  (the cheap pass already happened).
  accepted inputs pay E_lo and keep C_lo's prediction.
  expected per-sample energy = E_lo + p_escalate * E_hi.
Gate compute (max-softmax / margin / entropy over 10 logits) is a few ops vs
2.35e5 MACs/sample -> treated as negligible (<0.01%), reported as such.
"""
import os, json, itertools, numpy as np
import axlib as A
from run_train import SIZES, SEEDS, split

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")
DATASETS = ["mnist", "fmnist"]

# ----------------------------- confidence gates ----------------------------- #
def gate_scores(probs, kind="msp"):
    """Higher score = more confident. Escalate when score < tau."""
    p = np.sort(probs, axis=1)
    if kind == "msp":                       # maximum softmax probability
        return p[:, -1]
    if kind == "margin":                    # top1 - top2
        return p[:, -1] - p[:, -2]
    if kind == "negentropy":                # 1 - normalized entropy  (in [0,1])
        ent = -(probs * np.log(probs + 1e-12)).sum(1)
        return 1.0 - ent / np.log(probs.shape[1])
    raise ValueError(kind)

def fit_temperature(logits, y, grid=None):
    """1-D temperature scaling: minimise NLL on (logits, y)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.0, 11), np.linspace(1.05, 5.0, 80)])
    yl = logits[np.arange(len(y)), y]
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        z = logits / T
        lse = z.max(1) + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1))
        nll = float((lse - yl / T).mean())
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

# ----------------------------- cascade evaluation --------------------------- #
def cascade_eval(p_lo, pred_lo, pred_hi, y, e_lo, e_hi, scores, tau):
    """Return (accuracy, energy_pj, escalation_rate) for a given gate threshold."""
    escalate = scores < tau
    pred = np.where(escalate, pred_hi, pred_lo)
    acc = float((pred == y).mean())
    p_esc = float(escalate.mean())
    energy = e_lo + p_esc * e_hi
    return acc, energy, p_esc

def select_configs(expl):
    """C_hi = uniform INT8 (canonical safe). C_lo = cheapest Pareto config whose
    standalone val_acc >= int8_val - GAP. Returns (C_lo, C_hi, info)."""
    recs = expl["records"]; front = expl["pareto_idx"]
    int8_idx = expl["uniform_idx"]["(8, 0)"]
    int8_val = recs[int8_idx]["val_acc"]
    GAP = 0.05
    floor = int8_val - GAP
    cand = [i for i in front if recs[i]["val_acc"] >= floor]
    c_lo_idx = min(cand, key=lambda i: recs[i]["energy_pj"])
    C_lo = recs[c_lo_idx]["config"]
    C_hi = recs[int8_idx]["config"]
    # cheapest static config matching INT8 accuracy within 0.2% (static MP baseline)
    matchers = [i for i in front if recs[i]["val_acc"] >= int8_val - 0.002]
    static_mp_idx = min(matchers, key=lambda i: recs[i]["energy_pj"]) if matchers else int8_idx
    info = {"int8_val": int8_val, "floor_lo": floor,
            "C_lo_standalone_val": recs[c_lo_idx]["val_acc"],
            "C_hi_standalone_val": recs[int8_idx]["val_acc"],
            "static_mp_config": recs[static_mp_idx]["config"],
            "static_mp_energy_pj": recs[static_mp_idx]["energy_pj"],
            "static_mp_val": recs[static_mp_idx]["val_acc"]}
    return C_lo, C_hi, info

def main():
    summary = json.load(open(os.path.join(OUT, "fp_summary.json")))
    for ds in DATASETS:
        expl = json.load(open(os.path.join(OUT, f"explore_{ds}.json")))
        C_lo, C_hi, info = select_configs(expl)
        e_lo = A.config_energy(C_lo, SIZES)
        e_hi = A.config_energy(C_hi, SIZES)
        fp32_e = expl["fp32_energy_pj"]

        Xtr, ytr, Xval, yval, Xte, yte, Xcal = split(ds)
        m0 = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
        am0, wm0 = A.calibrate(m0, Xcal)

        # cheap/hi predictions + gate scores on val and test (seed 0)
        lg_lo_v, pr_lo_v = A.forward_quant(m0, Xval, C_lo, am0, wm0, return_probs=True)
        lg_lo_t, pr_lo_t = A.forward_quant(m0, Xte,  C_lo, am0, wm0, return_probs=True)
        pred_hi_v = A.forward_quant(m0, Xval, C_hi, am0, wm0).argmax(1)
        pred_hi_t = A.forward_quant(m0, Xte,  C_hi, am0, wm0).argmax(1)
        pred_lo_v, pred_lo_t = pr_lo_v.argmax(1), pr_lo_t.argmax(1)

        # temperature scaling fit on val (cheap model logits)
        T = fit_temperature(lg_lo_v, yval)
        ece_raw = A.expected_calibration_error(pr_lo_t, yte)
        ece_cal = A.expected_calibration_error(A._softmax(lg_lo_t / T), yte)

        # ---- tau-sweep curves for each gate kind (raw) + msp calibrated ----
        taus = np.linspace(0.30, 1.0001, 80)
        curves = {}
        for kind in ["msp", "margin", "negentropy"]:
            sv = gate_scores(pr_lo_v, kind); st = gate_scores(pr_lo_t, kind)
            pts = []
            for tau in taus:
                acc, en, pe = cascade_eval(pr_lo_t, pred_lo_t, pred_hi_t, yte, e_lo, e_hi, st, tau)
                accv, env, pev = cascade_eval(pr_lo_v, pred_lo_v, pred_hi_v, yval, e_lo, e_hi, sv, tau)
                pts.append({"tau": float(tau), "test_acc": acc, "energy_pj": en, "esc": pe,
                            "val_acc": accv, "val_energy_pj": env, "val_esc": pev})
            curves[kind] = pts
        # msp calibrated (scores from temperature-scaled probs)
        prc_v = A._softmax(lg_lo_v / T); prc_t = A._softmax(lg_lo_t / T)
        sv = gate_scores(prc_v, "msp"); st = gate_scores(prc_t, "msp")
        pts = []
        for tau in taus:
            acc, en, pe = cascade_eval(prc_t, pred_lo_t, pred_hi_t, yte, e_lo, e_hi, st, tau)
            pts.append({"tau": float(tau), "test_acc": acc, "energy_pj": en, "esc": pe})
        curves["msp_calibrated"] = pts

        # ---- oracle gate (escalate exactly C_lo's errors) & random gate ----
        wrong_t = (pred_lo_t != yte)
        oracle_acc = float(np.where(wrong_t, pred_hi_t, pred_lo_t)[np.arange(len(yte))] == yte).mean() \
            if False else float((np.where(wrong_t, pred_hi_t, pred_lo_t) == yte).mean())
        oracle_energy = e_lo + float(wrong_t.mean()) * e_hi
        rng = np.random.default_rng(0); random_curve = []
        for pe in np.linspace(0, 1, 21):
            esc = rng.random(len(yte)) < pe
            acc = float((np.where(esc, pred_hi_t, pred_lo_t) == yte).mean())
            random_curve.append({"esc": float(pe), "test_acc": acc, "energy_pj": e_lo + pe * e_hi})

        # ---- pick tau* on seed-0 val to match INT8 accuracy within 0.2% ----
        int8_val = info["int8_val"]
        msp_v = gate_scores(pr_lo_v, "msp")
        tau_star, found = 1.0, False
        for tau in taus:
            accv, _, _ = cascade_eval(pr_lo_v, pred_lo_v, pred_hi_v, yval, e_lo, e_hi, msp_v, tau)
            if accv >= int8_val - 0.002:
                tau_star, found = float(tau), True
                break

        # ---- multi-seed headline at tau* (msp, raw) ----
        per_seed = {"fp32": [], "int8": [], "static_mp": [], "pace": [],
                    "pace_esc": [], "pace_acc": []}
        static_mp_cfg = info["static_mp_config"]
        for s in SEEDS:
            ms = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed{s}.npz"))
            ams, wms = A.calibrate(ms, Xcal)
            # int8
            int8_acc = A.accuracy(A.forward_quant(ms, Xte, C_hi, ams, wms), yte)
            # static MP
            smp_acc = A.accuracy(A.forward_quant(ms, Xte, static_mp_cfg, ams, wms), yte)
            smp_e = A.config_energy(static_mp_cfg, SIZES)
            # pace
            _, pr_lo = A.forward_quant(ms, Xte, C_lo, ams, wms, return_probs=True)
            p_hi = A.forward_quant(ms, Xte, C_hi, ams, wms).argmax(1)
            p_lo = pr_lo.argmax(1); sc = gate_scores(pr_lo, "msp")
            acc, en, pe = cascade_eval(pr_lo, p_lo, p_hi, yte, e_lo, e_hi, sc, tau_star)
            per_seed["fp32"].append(summary[ds][str(s)]["test_acc"])
            per_seed["int8"].append(int8_acc)
            per_seed["static_mp"].append(smp_acc)
            per_seed["pace"].append(en)
            per_seed["pace_esc"].append(pe)
            per_seed["pace_acc"].append(acc)

        def ms_(x): return [float(np.mean(x)), float(np.std(x))]
        headline = {
            "C_lo": C_lo, "C_hi": C_hi, "e_lo_pj": e_lo, "e_hi_pj": e_hi,
            "fp32_energy_pj": fp32_e, "tau_star": tau_star, "tau_star_found": found,
            "static_mp_config": static_mp_cfg, "static_mp_energy_pj": info["static_mp_energy_pj"],
            "fp32_acc": ms_(per_seed["fp32"]), "int8_acc": ms_(per_seed["int8"]),
            "static_mp_acc": ms_(per_seed["static_mp"]),
            "pace_acc": ms_(per_seed["pace_acc"]), "pace_energy_pj": ms_(per_seed["pace"]),
            "pace_esc": ms_(per_seed["pace_esc"]),
            "int8_energy_pj": e_hi,
        }
        out = {"dataset": ds, "select_info": info, "temperature": T,
               "ece_raw": ece_raw, "ece_calibrated": ece_cal,
               "curves": curves, "oracle": {"test_acc": oracle_acc, "energy_pj": oracle_energy,
                                            "esc": float(wrong_t.mean())},
               "random_curve": random_curve, "headline": headline,
               "pareto_static": [{"energy_pj": expl["records"][i]["energy_pj"],
                                  "test_acc": expl["records"][i]["test_acc"],
                                  "val_acc": expl["records"][i]["val_acc"],
                                  "config": expl["records"][i]["config"]}
                                 for i in expl["pareto_idx"]],
               "sizes": SIZES}
        json.dump(out, open(os.path.join(OUT, f"pace_{ds}.json"), "w"), indent=2)

        print(f"\n===== {ds} =====")
        print(f"C_lo={C_lo} E={e_lo/1000:.2f}nJ standalone_val={info['C_lo_standalone_val']:.4f}")
        print(f"C_hi(INT8)={C_hi} E={e_hi/1000:.2f}nJ standalone_val={info['C_hi_standalone_val']:.4f}")
        print(f"static-MP={static_mp_cfg} E={info['static_mp_energy_pj']/1000:.2f}nJ")
        print(f"temperature T={T:.3f}  ECE raw={ece_raw:.4f} -> cal={ece_cal:.4f}")
        print(f"tau*={tau_star:.3f} (found={found})")
        print(f"oracle: acc={oracle_acc:.4f} E={oracle_energy/1000:.2f}nJ esc={wrong_t.mean():.3f}")
        h = headline
        print(f"[5-seed] FP32 acc={h['fp32_acc'][0]:.4f}+/-{h['fp32_acc'][1]:.4f} E={fp32_e/1000:.1f}nJ")
        print(f"[5-seed] INT8 acc={h['int8_acc'][0]:.4f}+/-{h['int8_acc'][1]:.4f} E={e_hi/1000:.2f}nJ")
        print(f"[5-seed] staticMP acc={h['static_mp_acc'][0]:.4f}+/-{h['static_mp_acc'][1]:.4f} "
              f"E={info['static_mp_energy_pj']/1000:.2f}nJ")
        print(f"[5-seed] PACE acc={h['pace_acc'][0]:.4f}+/-{h['pace_acc'][1]:.4f} "
              f"E={h['pace_energy_pj'][0]/1000:.2f}+/-{h['pace_energy_pj'][1]/1000:.2f}nJ "
              f"esc={h['pace_esc'][0]:.3f}")

if __name__ == "__main__":
    main()
