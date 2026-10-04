"""Memory- and switch-augmented energy model (Reviewers 2.3, 3.10).

Compute-only model (paper main text):   e = kappa*(b-t)^2 + e_acc          [pJ/MAC]
Augmented model adds operand-fetch:      e = kappa*(b-t)^2 + e_acc + mu*b   [pJ/MAC]
  (memory/data-movement energy is linear in the number of operand BITS moved,
   so low precision reduces it; mu sets the memory:compute balance.)
Cascade escalation additionally pays a fixed reconfiguration penalty E_switch per
escalated input (precision-scalable datapath mode switch + weight re-fetch).

We recompute the headline MLP comparison over a grid of (mu, E_switch) and report
whether PACE remains Pareto-dominated by static mixed-precision.
"""
import os, json, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
KAPPA = 0.2 / 64.0; E_ACC = 0.1
MACS = {"mnist": [200704, 32768, 1280], "fmnist": [200704, 32768, 1280]}

def e_mac(b, t, mu): return KAPPA * (b - t) ** 2 + E_ACC + mu * b
def cfg_energy(cfg, macs, mu): return sum(m * e_mac(b, t, mu) for m, (b, t) in zip(macs, cfg))

def main():
    mus = [0.0, 0.15, 0.3125, 0.625]        # pJ/bit/MAC: mem = 0, ~0.5x, ~1x, ~2x compute@8b
    switches = [0.0, 10.0, 50.0, 200.0]      # pJ per escalation
    tab = {}
    for ds in ["mnist", "fmnist"]:
        P = json.load(open(os.path.join(OUT, f"pace_{ds}.json")))
        h = P["headline"]; macs = MACS[ds]
        C_lo, C_hi, static_cfg = h["C_lo"], h["C_hi"], h["static_mp_config"]
        p_esc = h["pace_esc"][0]                 # mean escalation at tau*
        rows = []
        for mu in mus:
            e_lo = cfg_energy(C_lo, macs, mu); e_hi = cfg_energy(C_hi, macs, mu)
            e_static = cfg_energy(static_cfg, macs, mu)
            e_fp32 = sum(macs) * (4.6 + mu * 32)      # FP32 also pays memory (32-bit operands)
            for sw in switches:
                e_pace = e_lo + p_esc * (e_hi + sw)
                dominated = e_pace > e_static + 1e-9
                rows.append({"mu": mu, "switch_pJ": sw,
                             "static_nJ": round(e_static/1000, 2),
                             "pace_nJ": round(e_pace/1000, 2),
                             "gap_nJ": round((e_pace - e_static)/1000, 2),
                             "pace_dominated": bool(dominated),
                             "fp32_nJ": round(e_fp32/1000, 1),
                             "static_vs_fp32": round(e_fp32/e_static, 1)})
        tab[ds] = {"C_lo": C_lo, "C_hi": C_hi, "static_cfg": static_cfg,
                   "p_esc": p_esc, "rows": rows}
        n_dom = sum(r["pace_dominated"] for r in rows)
        print(f"\n=== {ds} : PACE dominated in {n_dom}/{len(rows)} (mu,switch) settings ===")
        print(f"{'mu':>7} {'switch':>7} {'static':>8} {'PACE':>8} {'gap':>7} {'dom?':>5} {'stat/FP32':>9}")
        for r in rows:
            print(f"{r['mu']:>7.3f} {r['switch_pJ']:>7.0f} {r['static_nJ']:>8.2f} "
                  f"{r['pace_nJ']:>8.2f} {r['gap_nJ']:>+7.2f} {str(r['pace_dominated']):>5} "
                  f"{r['static_vs_fp32']:>9.1f}")
    json.dump(tab, open(os.path.join(OUT, "energy_sensitivity.json"), "w"), indent=2)
    print("\nsaved energy_sensitivity.json")

if __name__ == "__main__":
    main()
