"""Corrected multi-pair analysis: build the LOWER ENERGY ENVELOPE over all
(C_lo,C_hi) pairs and all thresholds, and compare it to the static Pareto
frontier at matched accuracy. Handles honestly the regime where the cascade
reaches accuracy ABOVE the best static config (ensemble effect)."""
import os, json, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

for ds in ["mnist", "fmnist"]:
    R = json.load(open(os.path.join(OUT, "revision_r4.json")))[ds]["multipair"]
    front = sorted(R["front_test"])  # (energy_nJ, test_acc)
    fr_e = np.array([e for e, a in front]); fr_a = np.array([a for e, a in front])
    fr_amax = fr_a.max()

    def static_E_at(acc):
        cand = fr_e[fr_a >= acc - 1e-9]
        return cand.min() if len(cand) else np.inf

    # gather all cascade operating points across all pairs
    all_pts = []
    for p in R["pairs"]:
        for (ce, ca, pe) in p["curve"]:
            all_pts.append((ce, ca))
    all_pts = np.array(all_pts)

    # accuracy grid within the static frontier's support
    grid = np.linspace(fr_a.min(), fr_amax, 60)
    casc_env, stat_curve = [], []
    for a in grid:
        # cheapest cascade point reaching >= a
        m = all_pts[:, 1] >= a - 1e-9
        casc_env.append(all_pts[m, 0].min() if m.any() else np.inf)
        stat_curve.append(static_E_at(a))
    casc_env = np.array(casc_env); stat_curve = np.array(stat_curve)
    # domination check within frontier support
    valid = np.isfinite(casc_env) & np.isfinite(stat_curve)
    diff = casc_env[valid] - stat_curve[valid]     # >0 => static cheaper (cascade dominated)
    frac_dom = float((diff < -1e-6).mean())
    min_diff = float(diff.min()); max_diff = float(diff.max())

    # ensemble effect: best cascade accuracy above static max, and its energy
    above = all_pts[all_pts[:, 1] > fr_amax + 1e-9]
    best_above_acc = float(above[:, 1].max()) if len(above) else None
    best_above_e = float(above[above[:, 1].argmax(), 0]) if len(above) else None

    print(f"=== {ds} ===")
    print(f"static frontier accuracy support: [{fr_a.min():.4f}, {fr_amax:.4f}], "
          f"min static energy {fr_e.min():.2f} nJ")
    print(f"cascade lower-envelope vs static frontier (within support, {valid.sum()} grid pts):")
    print(f"  cascade cheaper anywhere? {'YES' if frac_dom>0 else 'NO'} "
          f"(fraction of acc levels where cascade wins on energy: {frac_dom*100:.1f}%)")
    print(f"  energy gap cascade-static: min {min_diff:+.2f} nJ, max {max_diff:+.2f} nJ "
          f"(positive => static cheaper)")
    if best_above_acc:
        print(f"  ensemble effect: cascade reaches {best_above_acc:.4f} (> static max {fr_amax:.4f}) "
              f"but at {best_above_e:.2f} nJ vs INT8-region energies")
    # save
    d = {"grid_acc": grid.tolist(), "cascade_env_nJ": casc_env.tolist(),
         "static_curve_nJ": stat_curve.tolist(), "frac_levels_cascade_wins": frac_dom,
         "min_gap_nJ": min_diff, "max_gap_nJ": max_diff,
         "ensemble_best_acc": best_above_acc, "ensemble_best_e_nJ": best_above_e,
         "front_test": R["front_test"]}
    json.dump(d, open(os.path.join(OUT, f"multipair_env_{ds}.json"), "w"), indent=2)
print("\nsaved multipair_env_{mnist,fmnist}.json")
