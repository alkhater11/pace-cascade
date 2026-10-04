"""run_capacity_train.py -- train the narrow MLP (784-32-10) used as the cheap arm
of the width-and-precision cascade (second revision).  Same optimiser, split and
seeds as the main MLPs (configs/mlp_narrow.json)."""
import os, json, time, numpy as np
import axlib as A
from run_train import split
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = json.load(open(os.path.join(ROOT, "configs", "mlp_narrow.json")))
OUT = os.path.join(ROOT, "results"); MOD = os.path.join(OUT, "models_narrow"); os.makedirs(MOD, exist_ok=True)
summ = {}
for ds in CFG["datasets"]:
    Xtrain, ytrain, Xval, yval, Xte, yte, Xcal = split(ds); summ[ds] = {}
    for s in CFG["seeds"]:
        t0 = time.time(); m = A.NumpyMLP(CFG["sizes"], seed=s)
        m.train(Xtrain, ytrain, epochs=CFG["epochs"], bs=CFG["batch_size"], lr=CFG["lr"], wd=CFG["weight_decay"], seed=s)
        m.save(os.path.join(MOD, f"{ds}_seed{s}.npz"))
        va = float((m.predict_fp(Xval) == yval).mean()); te = float((m.predict_fp(Xte) == yte).mean())
        summ[ds][str(s)] = {"val_acc": va, "test_acc": te, "sec": time.time() - t0}
        print(f"[narrow {ds} seed{s}] val={va:.4f} test={te:.4f}", flush=True)
json.dump(summ, open(os.path.join(OUT, "narrow_fp_summary.json"), "w"), indent=2)
