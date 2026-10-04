"""Train SmallCNN, 3 seeds x 2 datasets, on a 20k subset (compute-tractable).
Full 10k val / 10k test / 2k calibration, matching the MLP protocol otherwise."""
import os, json, time, numpy as np
import axlib as A
import cnnlib as C

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = os.path.join(OUT, "cnn_models"); os.makedirs(MODELS, exist_ok=True)
SEEDS = [0, 1, 2]; EPOCHS = 8; NTRAIN = 20000

def split4(ds):
    Xtr, ytr, Xte, yte = A.get_dataset(ds)
    X = Xtr.reshape(-1, 1, 28, 28)
    Xtrain, ytrain = X[:NTRAIN], ytr[:NTRAIN]
    Xval, yval = X[50000:60000], ytr[50000:60000]
    Xtest, ytest = Xte.reshape(-1, 1, 28, 28), yte
    Xcal = X[:2000]
    return Xtrain, ytrain, Xval, yval, Xtest, ytest, Xcal

if __name__ == "__main__":
    summary = {}
    for ds in ["mnist", "fmnist"]:
        Xtr, ytr, Xval, yval, Xte, yte, Xcal = split4(ds)
        summary[ds] = {}
        for s in SEEDS:
            path = os.path.join(MODELS, f"{ds}_seed{s}.npz")
            if os.path.exists(path):
                m = C.SmallCNN.load(path)
            else:
                m = C.SmallCNN(seed=s)
                t = time.time(); m.train(Xtr, ytr, epochs=EPOCHS, bs=128, lr=1e-3, seed=s)
                dt = time.time() - t; m.save(path)
            va = (m.predict(Xval) == yval).mean(); ta = (m.predict(Xte) == yte).mean()
            summary[ds][str(s)] = {"val_acc": float(va), "test_acc": float(ta)}
            print(f"[{ds} seed{s}] val={va:.4f} test={ta:.4f}", flush=True)
        accs = [summary[ds][str(s)]["test_acc"] for s in SEEDS]
        print(f"== {ds}: CNN FP32 test {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}")
    json.dump(summary, open(os.path.join(OUT, "cnn_fp_summary.json"), "w"), indent=2)
    print("saved cnn_fp_summary.json")
