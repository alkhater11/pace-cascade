"""run_train.py -- train compact MLPs for both datasets across seeds; cache weights."""
import os, json, time, numpy as np
import axlib as A

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "models")
os.makedirs(MODELS, exist_ok=True)

SIZES = [784, 256, 128, 10]
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["mnist", "fmnist"]
EPOCHS = 15

def split(name):
    Xtr, ytr, Xte, yte = A.get_dataset(name)
    Xtrain, ytrain = Xtr[:50000], ytr[:50000]
    Xval, yval = Xtr[50000:], ytr[50000:]
    Xcal = Xtr[:2000]
    return Xtrain, ytrain, Xval, yval, Xte, yte, Xcal

def main():
    summary = {}
    for ds in DATASETS:
        Xtrain, ytrain, Xval, yval, Xte, yte, Xcal = split(ds)
        summary[ds] = {}
        for seed in SEEDS:
            path = os.path.join(MODELS, f"{ds}_seed{seed}.npz")
            t0 = time.time()
            m = A.NumpyMLP(SIZES, seed=seed)
            m.train(Xtrain, ytrain, epochs=EPOCHS, bs=128, lr=1e-3, seed=seed)
            m.save(path)
            val_acc = float((m.predict_fp(Xval) == yval).mean())
            test_acc = float((m.predict_fp(Xte) == yte).mean())
            dt = time.time() - t0
            summary[ds][seed] = {"val_acc": val_acc, "test_acc": test_acc, "sec": dt}
            print(f"[{ds} seed{seed}] val={val_acc:.4f} test={test_acc:.4f} ({dt:.1f}s)", flush=True)
    # aggregate
    for ds in DATASETS:
        accs = [summary[ds][s]["test_acc"] for s in SEEDS]
        print(f"== {ds}: FP32 test acc {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}")
    json.dump(summary, open(os.path.join(OUT, "fp_summary.json"), "w"), indent=2)
    print("saved fp_summary.json")

if __name__ == "__main__":
    main()
