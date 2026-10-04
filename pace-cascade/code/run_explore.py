"""run_explore.py -- per-layer sensitivity + exhaustive static Pareto frontier (seed 0).

For speed, val_acc is computed for ALL enumerated configs (full 10k val set) and
the Pareto front is selected on val_acc; test_acc is then evaluated only for the
configs that matter downstream (Pareto front + uniform baselines). This avoids
redundant full test-set passes over dominated configurations.
"""
import os, json, itertools, time, numpy as np
import axlib as A
from run_train import SIZES, split

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")

# Per-layer approximation option set (b, t). Energy depends on effective width b-t.
#   (8,0) eff8=0.300pJ  (6,0) eff6=0.2125  (5,0) eff5=0.178  (4,0) eff4=0.150
#   (6,2) eff4=0.150    (4,1) eff3=0.128   -- two eff4 opts test trunc-mult vs low-bit
OPTIONS = [(8, 0), (6, 0), (5, 0), (4, 0), (6, 2), (4, 1)]
DATASETS = ["mnist", "fmnist"]

def pareto_front(points):
    """points: list of (energy, accuracy, idx). Lower energy / higher accuracy front."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best_acc = [], -1.0
    for i in order:
        e, acc, idx = points[i]
        if acc > best_acc + 1e-12:
            front.append(idx); best_acc = acc
    return front

def main():
    for ds in DATASETS:
        Xtrain, ytrain, Xval, yval, Xte, yte, Xcal = split(ds)
        m = A.NumpyMLP.load(os.path.join(MODELS, f"{ds}_seed0.npz"))
        act_max, w_max = A.calibrate(m, Xcal)
        fp32_test = json.load(open(os.path.join(OUT, "fp_summary.json")))[ds]["0"]["test_acc"]

        # ---- per-layer sensitivity: approximate ONE layer to each option, others (8,0) ----
        base = [(8, 0)] * (len(SIZES) - 1)
        base_val = A.forward_quant(m, Xval, base, act_max, w_max)
        base_acc = A.accuracy(base_val, yval)
        sens = {}
        for l in range(len(SIZES) - 1):
            sens[l] = []
            for opt in OPTIONS:
                cfg = list(base); cfg[l] = opt
                acc = A.accuracy(A.forward_quant(m, Xval, cfg, act_max, w_max), yval)
                sens[l].append({"opt": list(opt), "eff": opt[0] - opt[1],
                                "val_acc": acc, "drop": base_acc - acc})

        # ---- exhaustive enumeration: val_acc for every config ----
        t0 = time.time()
        records = []
        for combo in itertools.product(OPTIONS, repeat=len(SIZES) - 1):
            cfg = [list(c) for c in combo]
            e = A.config_energy(cfg, SIZES)
            va = A.accuracy(A.forward_quant(m, Xval, cfg, act_max, w_max), yval)
            records.append({"config": cfg, "energy_pj": e, "val_acc": va, "test_acc": None})
        dt = time.time() - t0

        pts = [(r["energy_pj"], r["val_acc"], i) for i, r in enumerate(records)]
        front = pareto_front(pts)

        uniform = {}
        for opt in OPTIONS:
            target = [list(opt) for _ in range(len(SIZES) - 1)]
            for i, r in enumerate(records):
                if r["config"] == target:
                    uniform[str(tuple(opt))] = i; break

        # ---- test_acc only for configs that matter (Pareto + uniform) ----
        for i in set(front) | set(uniform.values()):
            r = records[i]
            r["test_acc"] = A.accuracy(A.forward_quant(m, Xte, r["config"], act_max, w_max), yte)

        fp32_e = sum(A.layer_macs(SIZES)) * 4.6  # FP32 MAC = 3.7pJ mult + 0.9pJ add
        int8_e = A.config_energy([(8, 0)] * (len(SIZES) - 1), SIZES)

        out = {"options": OPTIONS, "sizes": SIZES, "layer_macs": A.layer_macs(SIZES),
               "base_acc_int8": base_acc, "sensitivity": sens, "records": records,
               "pareto_idx": front, "uniform_idx": uniform,
               "fp32_energy_pj": fp32_e, "int8_energy_pj": int8_e,
               "fp32_test_acc": fp32_test}
        json.dump(out, open(os.path.join(OUT, f"explore_{ds}.json"), "w"), indent=2)
        print(f"[{ds}] enumerated {len(records)} cfgs in {dt:.1f}s | INT8 base val={base_acc:.4f} "
              f"test={records[uniform[str((8,0))]]['test_acc']:.4f} | FP32 test={fp32_test:.4f} "
              f"| pareto={len(front)} | E: FP32={fp32_e/1000:.1f}nJ INT8={int8_e/1000:.2f}nJ")
        for idx in front:
            r = records[idx]
            print(f"    E={r['energy_pj']/1000:5.2f}nJ ({fp32_e/r['energy_pj']:4.1f}x)  "
                  f"val={r['val_acc']:.4f} test={r['test_acc']:.4f}  cfg={r['config']}")

if __name__ == "__main__":
    main()
