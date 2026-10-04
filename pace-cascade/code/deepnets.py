"""deepnets.py -- deeper / modern architectures added in the second revision.

Two models are defined and trained in JAX (CPU, float32, single thread):

  * ResNet-8 for CIFAR-10: a 3-stage residual CNN (16-32-64 channels, one basic
    block per stage, 1x1 projection shortcuts, batch normalisation, global
    average pooling).  10 weight layers, 12.50 M MACs per 32x32x3 image.
  * ViT-4 for Fashion-MNIST: a compact Vision Transformer (7x7 patches -> 16
    tokens + CLS, width 64, 4 pre-LN encoder blocks, 4 heads, MLP ratio 2,
    GELU).  2.43 M MACs per image.

Trained parameters are exported to .npz (NumPy) so that the approximate,
integer-exact inference in deepquant.py does not depend on JAX.

All hyper-parameters are read from configs/resnet8_cifar10.json and
configs/vit4_fmnist.json.  Seeds control initialisation, data shuffling and
augmentation; with XLA's single-threaded CPU backend runs are deterministic.
"""
import os, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
CONFIGS = os.path.join(ROOT, "configs")
OUT = os.path.join(ROOT, "results", "deep_models")


def load_config(name):
    with open(os.path.join(CONFIGS, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def cifar10_splits(cfg, split_seed=2026):
    """CIFAR-10 (GitHub JPEG mirror, see data/README.md).  Returns uint8 NHWC.
    A fixed, class-stratified 5,000-image validation split is held out of the
    50,000 training images (identical for all seeds); calibration = first
    2,000 images of the remaining training portion."""
    d = np.load(os.path.join(DATA, "cifar10_jpgmirror.npz"))
    X, y = d["X_train"], d["y_train"]
    rng = np.random.default_rng(split_seed)
    val_idx = []
    for k in range(10):
        idx = np.where(y == k)[0]
        val_idx.append(rng.choice(idx, cfg["n_val"] // 10, replace=False))
    val_idx = np.sort(np.concatenate(val_idx))
    mask = np.ones(len(y), bool); mask[val_idx] = False
    tr_idx = np.where(mask)[0]
    tr_idx = tr_idx[rng.permutation(len(tr_idx))]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xval, yval = X[val_idx], y[val_idx]
    Xcal = Xtr[:cfg["n_cal"]]
    return Xtr, ytr, Xval, yval, d["X_test"], d["y_test"], Xcal


def fmnist_splits():
    """Same split as the MLP study: 50k train / 10k val / 10k test, cal = first 2k."""
    import axlib as A
    Xtr, ytr, Xte, yte = A.get_dataset("fmnist")
    Xtr = Xtr.reshape(-1, 28, 28, 1); Xte = Xte.reshape(-1, 28, 28, 1)
    return Xtr[:50000], ytr[:50000], Xtr[50000:], ytr[50000:], Xte, yte, Xtr[:2000]


def to_float(X):
    return X.astype(np.float32) / 255.0 if X.dtype == np.uint8 else X.astype(np.float32)


def augment_crop_flip(xb, rng, pad=4, flip=True):
    """Random crop with zero padding + optional horizontal flip (NHWC float)."""
    N, H, W, C = xb.shape
    xp = np.zeros((N, H + 2 * pad, W + 2 * pad, C), dtype=xb.dtype)
    xp[:, pad:pad + H, pad:pad + W] = xb
    ox = rng.integers(0, 2 * pad + 1, N); oy = rng.integers(0, 2 * pad + 1, N)
    out = np.empty_like(xb)
    for i in range(N):
        out[i] = xp[i, oy[i]:oy[i] + H, ox[i]:ox[i] + W]
    if flip:
        f = rng.random(N) < 0.5
        out[f] = out[f, :, ::-1]
    return out


# --------------------------------------------------------------------------- #
#  ResNet-8
# --------------------------------------------------------------------------- #
RESNET_CONVS = [  # name, kh, cin, cout, stride
    ("conv0", 3, 3, 16, 1),
    ("s1c1", 3, 16, 16, 1), ("s1c2", 3, 16, 16, 1),
    ("s2c1", 3, 16, 32, 2), ("s2c2", 3, 32, 32, 1), ("s2sc", 1, 16, 32, 2),
    ("s3c1", 3, 32, 64, 2), ("s3c2", 3, 64, 64, 1), ("s3sc", 1, 32, 64, 2),
]


def resnet8_init(seed):
    rng = np.random.default_rng(seed)
    params, stats = {}, {}
    for name, k, cin, cout, _ in RESNET_CONVS:
        fan_in = k * k * cin
        params[name + "_w"] = (rng.standard_normal((k, k, cin, cout)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        params[name + "_g"] = np.ones(cout, np.float32)
        params[name + "_b"] = np.zeros(cout, np.float32)
        stats[name + "_m"] = np.zeros(cout, np.float32)
        stats[name + "_v"] = np.ones(cout, np.float32)
    params["fc_w"] = (rng.standard_normal((64, 10)) * np.sqrt(1.0 / 64)).astype(np.float32)
    params["fc_b"] = np.zeros(10, np.float32)
    return params, stats


def resnet8_apply(params, stats, x, train, bn_mom=0.9, eps=1e-5):
    import jax, jax.numpy as jnp
    from jax import lax
    new_stats = {}

    def conv_bn(h, name, stride):
        w = params[name + "_w"]
        z = lax.conv_general_dilated(h, w, (stride, stride), "SAME",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
        if train:
            mu = z.mean((0, 1, 2)); var = z.var((0, 1, 2))
            new_stats[name + "_m"] = bn_mom * stats[name + "_m"] + (1 - bn_mom) * mu
            new_stats[name + "_v"] = bn_mom * stats[name + "_v"] + (1 - bn_mom) * var
        else:
            mu, var = stats[name + "_m"], stats[name + "_v"]
        return (z - mu) / jnp.sqrt(var + eps) * params[name + "_g"] + params[name + "_b"]

    h = jax.nn.relu(conv_bn(x, "conv0", 1))
    r = jax.nn.relu(conv_bn(h, "s1c1", 1)); r = conv_bn(r, "s1c2", 1); h = jax.nn.relu(h + r)
    r = jax.nn.relu(conv_bn(h, "s2c1", 2)); r = conv_bn(r, "s2c2", 1)
    h = jax.nn.relu(conv_bn(h, "s2sc", 2) + r)
    r = jax.nn.relu(conv_bn(h, "s3c1", 2)); r = conv_bn(r, "s3c2", 1)
    h = jax.nn.relu(conv_bn(h, "s3sc", 2) + r)
    h = h.mean((1, 2))
    logits = h @ params["fc_w"] + params["fc_b"]
    return logits, (new_stats if train else stats)


def train_resnet8(seed, cfg, verbose=True):
    import jax, jax.numpy as jnp
    Xtr, ytr, Xval, yval, Xte, yte, Xcal = cifar10_splits(cfg)
    params, stats = resnet8_init(seed)
    params = {k: jnp.asarray(v) for k, v in params.items()}
    stats = {k: jnp.asarray(v) for k, v in stats.items()}
    mom = {k: jnp.zeros_like(v) for k, v in params.items()}
    decay_keys = {k for k in params if k.endswith("_w")}
    bs, E = cfg["batch_size"], cfg["epochs"]
    steps_per_epoch = len(Xtr) // bs
    total = steps_per_epoch * E; warm = steps_per_epoch * cfg["warmup_epochs"]

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return cfg["lr"] * 0.5 * (1 + np.cos(np.pi * p))

    def loss_fn(p, s, xb, yb):
        lg, ns = resnet8_apply(p, s, xb, True)
        ll = jax.nn.log_softmax(lg)
        return -jnp.mean(ll[jnp.arange(xb.shape[0]), yb]), ns

    @jax.jit
    def step(p, s, m, xb, yb, lr):
        (l, ns), g = jax.value_and_grad(loss_fn, has_aux=True)(p, s, xb, yb)
        newp, newm = {}, {}
        for k in p:
            gk = g[k] + (cfg["weight_decay"] * p[k] if k in decay_keys else 0.0)
            mk = cfg["momentum"] * m[k] + gk
            upd = gk + cfg["momentum"] * mk if cfg["nesterov"] else mk
            newp[k] = p[k] - lr * upd; newm[k] = mk
        return newp, ns, newm, l

    @jax.jit
    def predict(p, s, xb):
        return resnet8_apply(p, s, xb, False)[0]

    def evaluate(X, y):
        out = []
        for i in range(0, len(X), 1000):
            out.append(np.asarray(predict(params, stats, jnp.asarray(to_float(X[i:i + 1000])))).argmax(1))
        return float((np.concatenate(out) == y).mean())

    rng = np.random.default_rng(1000 + seed)
    t0 = time.time(); gstep = 0
    for ep in range(E):
        perm = rng.permutation(len(Xtr)); tl = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            xb = augment_crop_flip(to_float(Xtr[idx]), rng, pad=4, flip=True)
            params, stats, mom, l = step(params, stats, mom, jnp.asarray(xb), jnp.asarray(ytr[idx]),
                                         float(lr_at(gstep)))
            gstep += 1; tl += float(l)
        if verbose:
            va = evaluate(Xval, yval) if (ep == E - 1 or ep % 3 == 2) else float("nan")
            print(f"  [resnet8 s{seed}] ep {ep+1}/{E} loss {tl/steps_per_epoch:.4f} val {va:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    va, te = evaluate(Xval, yval), evaluate(Xte, yte)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"resnet8_cifar10_seed{seed}.npz"),
             **{k: np.asarray(v) for k, v in params.items()},
             **{k: np.asarray(v) for k, v in stats.items()})
    return {"val_acc": va, "test_acc": te, "sec": time.time() - t0}


# --------------------------------------------------------------------------- #
#  ViT-4 (compact Vision Transformer)
# --------------------------------------------------------------------------- #
def vit_init(seed, cfg):
    rng = np.random.default_rng(seed)
    d, P, depth, mr = cfg["dim"], cfg["patch"], cfg["depth"], cfg["mlp_ratio"]
    npatch = (28 // P) ** 2
    def lin(i, o, scale=None):
        s = np.sqrt(1.0 / i) if scale is None else scale
        return (rng.standard_normal((i, o)) * s).astype(np.float32), np.zeros(o, np.float32)
    p = {}
    p["emb_w"], p["emb_b"] = lin(P * P, d)
    p["cls"] = (rng.standard_normal((1, 1, d)) * 0.02).astype(np.float32)
    p["pos"] = (rng.standard_normal((1, npatch + 1, d)) * 0.02).astype(np.float32)
    for l in range(depth):
        p[f"b{l}_ln1_g"] = np.ones(d, np.float32); p[f"b{l}_ln1_b"] = np.zeros(d, np.float32)
        p[f"b{l}_qkv_w"], p[f"b{l}_qkv_b"] = lin(d, 3 * d)
        p[f"b{l}_proj_w"], p[f"b{l}_proj_b"] = lin(d, d)
        p[f"b{l}_ln2_g"] = np.ones(d, np.float32); p[f"b{l}_ln2_b"] = np.zeros(d, np.float32)
        p[f"b{l}_fc1_w"], p[f"b{l}_fc1_b"] = lin(d, mr * d)
        p[f"b{l}_fc2_w"], p[f"b{l}_fc2_b"] = lin(mr * d, d)
    p["lnf_g"] = np.ones(d, np.float32); p["lnf_b"] = np.zeros(d, np.float32)
    p["head_w"], p["head_b"] = lin(d, 10, scale=0.02)
    return p


def patchify(x, P):
    """x: (N,28,28,1) -> (N, (28/P)^2, P*P), row-major patch order."""
    N = x.shape[0]; g = 28 // P
    x = x.reshape(N, g, P, g, P).transpose(0, 1, 3, 2, 4)
    return x.reshape(N, g * g, P * P)


def vit_apply(p, x, cfg):
    import jax, jax.numpy as jnp
    d, depth, H = cfg["dim"], cfg["depth"], cfg["heads"]
    dh = d // H

    def ln(h, g, b, eps=1e-6):
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        return (h - mu) / jnp.sqrt(var + eps) * g + b

    xp = patchify(x, cfg["patch"])
    h = xp @ p["emb_w"] + p["emb_b"]
    N = h.shape[0]
    h = jnp.concatenate([jnp.broadcast_to(p["cls"], (N, 1, d)), h], 1) + p["pos"]
    T = h.shape[1]
    for l in range(depth):
        a = ln(h, p[f"b{l}_ln1_g"], p[f"b{l}_ln1_b"])
        qkv = a @ p[f"b{l}_qkv_w"] + p[f"b{l}_qkv_b"]
        q, k, v = jnp.split(qkv, 3, -1)
        q = q.reshape(N, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(N, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(N, T, H, dh).transpose(0, 2, 1, 3)
        att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh), -1)
        ctx = (att @ v).transpose(0, 2, 1, 3).reshape(N, T, d)
        h = h + ctx @ p[f"b{l}_proj_w"] + p[f"b{l}_proj_b"]
        a = ln(h, p[f"b{l}_ln2_g"], p[f"b{l}_ln2_b"])
        m = jax.nn.gelu(a @ p[f"b{l}_fc1_w"] + p[f"b{l}_fc1_b"], approximate=False)
        h = h + m @ p[f"b{l}_fc2_w"] + p[f"b{l}_fc2_b"]
    hc = ln(h[:, 0], p["lnf_g"], p["lnf_b"])
    return hc @ p["head_w"] + p["head_b"]


def train_vit(seed, cfg, verbose=True):
    import jax, jax.numpy as jnp
    Xtr, ytr, Xval, yval, Xte, yte, Xcal = fmnist_splits()
    params = {k: jnp.asarray(v) for k, v in vit_init(seed, cfg).items()}
    m1 = {k: jnp.zeros_like(v) for k, v in params.items()}
    m2 = {k: jnp.zeros_like(v) for k, v in params.items()}
    decay_keys = {k for k in params if k.endswith("_w")}
    bs, E = cfg["batch_size"], cfg["epochs"]
    spe = len(Xtr) // bs; total = spe * E; warm = spe * cfg["warmup_epochs"]

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        pr = (step - warm) / max(1, total - warm)
        return cfg["lr"] * 0.5 * (1 + np.cos(np.pi * pr))

    def loss_fn(p, xb, yb):
        lg = vit_apply(p, xb, cfg)
        return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(xb.shape[0]), yb])

    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(p, m1, m2, xb, yb, lr, t):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        np_, n1, n2 = {}, {}, {}
        for k in p:
            n1[k] = b1 * m1[k] + (1 - b1) * g[k]
            n2[k] = b2 * m2[k] + (1 - b2) * g[k] * g[k]
            mh = n1[k] / (1 - b1 ** t); vh = n2[k] / (1 - b2 ** t)
            upd = mh / (jnp.sqrt(vh) + eps)
            if k in decay_keys:
                upd = upd + cfg["weight_decay"] * p[k]       # decoupled (AdamW)
            np_[k] = p[k] - lr * upd
        return np_, n1, n2, l

    @jax.jit
    def predict(p, xb):
        return vit_apply(p, xb, cfg)

    def evaluate(X, y):
        out = [np.asarray(predict(params, jnp.asarray(X[i:i + 2000]))).argmax(1) for i in range(0, len(X), 2000)]
        return float((np.concatenate(out) == y).mean())

    rng = np.random.default_rng(1000 + seed)
    t0 = time.time(); gstep = 0
    for ep in range(E):
        perm = rng.permutation(len(Xtr)); tl = 0.0
        for b in range(spe):
            idx = perm[b * bs:(b + 1) * bs]
            xb = augment_crop_flip(Xtr[idx], rng, pad=cfg["shift_pad"], flip=False)
            gstep += 1
            params, m1, m2, l = step(params, m1, m2, jnp.asarray(xb), jnp.asarray(ytr[idx]),
                                     float(lr_at(gstep - 1)), float(gstep))
            tl += float(l)
        if verbose:
            va = evaluate(Xval, yval) if (ep == E - 1 or ep % 4 == 3) else float("nan")
            print(f"  [vit4 s{seed}] ep {ep+1}/{E} loss {tl/spe:.4f} val {va:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    va, te = evaluate(Xval, yval), evaluate(Xte, yte)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"vit4_fmnist_seed{seed}.npz"), **{k: np.asarray(v) for k, v in params.items()})
    return {"val_acc": va, "test_acc": te, "sec": time.time() - t0}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet8", "vit4"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    cfgname = {"resnet8": "resnet8_cifar10.json", "vit4": "vit4_fmnist.json"}[args.model]
    cfg = load_config(cfgname)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    summ_path = os.path.join(ROOT, "results", f"{args.model}_fp_summary.json")
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
    for s in seeds:
        r = (train_resnet8 if args.model == "resnet8" else train_vit)(s, cfg)
        summ[str(s)] = r
        print(f"[{args.model} seed{s}] val={r['val_acc']:.4f} test={r['test_acc']:.4f} ({r['sec']:.0f}s)", flush=True)
        json.dump(summ, open(summ_path, "w"), indent=2)
