"""
axlib.py -- Core library for the PACE study.

Components:
  * IDX dataset loading (MNIST / Fashion-MNIST)
  * A compact NumPy MLP trained with mini-batch Adam (deterministic, CPU-only)
  * Post-training fixed-point quantization with an approximate multiplier model
    based on operand LSB-truncation (a deterministic, reproducible approximate
    computing technique).
  * An analytical per-MAC compute-energy model calibrated to published 45 nm
    arithmetic energy figures (Horowitz, ISSCC 2014).

All approximation is simulated exactly in integer arithmetic (no random error
injection): activations are quantized to unsigned b-bit, weights to signed
per-output-channel b-bit, and the t least-significant bits of BOTH operands are
forced to zero before an exact integer multiply-accumulate. This mirrors a
truncated-operand array multiplier whose energy scales with the squared
effective operand width.
"""

import gzip, os, struct, json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        return arr.reshape(dims)

def get_dataset(name):
    """name in {'mnist','fmnist'} -> (Xtr, ytr, Xte, yte), X float32 in [0,1]."""
    p = {"mnist": "mn", "fmnist": "fm"}[name]
    Xtr = _read_idx(os.path.join(DATA_DIR, f"{p}_train_images.gz")).astype(np.float32) / 255.0
    ytr = _read_idx(os.path.join(DATA_DIR, f"{p}_train_labels.gz")).astype(np.int64)
    Xte = _read_idx(os.path.join(DATA_DIR, f"{p}_test_images.gz")).astype(np.float32) / 255.0
    yte = _read_idx(os.path.join(DATA_DIR, f"{p}_test_labels.gz")).astype(np.int64)
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    return Xtr, ytr, Xte, yte

# --------------------------------------------------------------------------- #
#  Compact MLP (NumPy, Adam)
# --------------------------------------------------------------------------- #
def _relu(x):
    return np.maximum(x, 0.0)

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class NumpyMLP:
    """Fully-connected ReLU network: sizes e.g. [784,256,128,10]."""
    def __init__(self, sizes, seed=0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)  # He init
            self.W.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))

    def forward_fp(self, X, return_hidden=False):
        a = X
        hidden_inputs = []
        for i in range(len(self.W)):
            hidden_inputs.append(a)                 # input activation to layer i
            z = a @ self.W[i] + self.b[i]
            a = _relu(z) if i < len(self.W) - 1 else z
        if return_hidden:
            return a, hidden_inputs                 # logits, list of per-layer input acts
        return a

    def predict_fp(self, X, bs=2048):
        out = []
        for s in range(0, len(X), bs):
            out.append(self.forward_fp(X[s:s + bs]).argmax(1))
        return np.concatenate(out)

    def train(self, X, y, epochs=12, bs=128, lr=1e-3, wd=0.0, seed=0, verbose=False,
              val=None):
        rng = np.random.default_rng(1000 + seed)
        n = len(X)
        K = self.sizes[-1]
        Y = np.eye(K, dtype=np.float32)[y]
        # Adam state
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                xb, yb = X[idx], Y[idx]
                # ---- forward (cache) ----
                acts = [xb]; pre = []
                a = xb
                for i in range(len(self.W)):
                    z = a @ self.W[i] + self.b[i]
                    pre.append(z)
                    a = _relu(z) if i < len(self.W) - 1 else z
                    acts.append(a)
                probs = _softmax(acts[-1])
                # ---- backward ----
                t += 1
                g = (probs - yb) / len(idx)          # dL/dlogits
                for i in reversed(range(len(self.W))):
                    gW = acts[i].T @ g + wd * self.W[i]
                    gb = g.sum(0)
                    # Adam update
                    mW[i] = b1 * mW[i] + (1 - b1) * gW; vW[i] = b2 * vW[i] + (1 - b2) * gW * gW
                    mb[i] = b1 * mb[i] + (1 - b1) * gb; vb[i] = b2 * vb[i] + (1 - b2) * gb * gb
                    mWh = mW[i] / (1 - b1 ** t); vWh = vW[i] / (1 - b2 ** t)
                    mbh = mb[i] / (1 - b1 ** t); vbh = vb[i] / (1 - b2 ** t)
                    self.W[i] -= lr * mWh / (np.sqrt(vWh) + eps)
                    self.b[i] -= lr * mbh / (np.sqrt(vbh) + eps)
                    if i > 0:
                        ga = g @ self.W[i].T
                        g = ga * (pre[i - 1] > 0)     # relu grad
            if verbose:
                msg = f"  epoch {ep+1}/{epochs}"
                if val is not None:
                    acc = (self.predict_fp(val[0]) == val[1]).mean()
                    msg += f"  val_acc={acc:.4f}"
                print(msg, flush=True)
        return self

    def save(self, path):
        d = {}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"] = w; d[f"b{i}"] = b
        d["sizes"] = np.array(self.sizes)
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        sizes = list(z["sizes"])
        m = cls(sizes)
        m.W = [z[f"W{i}"] for i in range(len(sizes) - 1)]
        m.b = [z[f"b{i}"] for i in range(len(sizes) - 1)]
        return m

# --------------------------------------------------------------------------- #
#  Quantization helpers + approximate (truncated-operand) multiplier
# --------------------------------------------------------------------------- #
def calibrate(model, Xcal):
    """Return per-layer activation max (|.|) and per-output-channel weight max."""
    _, hidden_inputs = model.forward_fp(Xcal, return_hidden=True)
    act_max = [float(np.abs(h).max()) + 1e-8 for h in hidden_inputs]
    w_max = [np.abs(W).max(axis=0) + 1e-8 for W in model.W]   # per output channel
    return act_max, w_max

def _q_unsigned(x, xmax, b, t):
    """Quantize non-negative activations to b-bit unsigned, truncate t LSBs."""
    levels = (1 << b) - 1
    step = xmax / levels
    q = np.rint(x / step)
    q = np.clip(q, 0, levels).astype(np.int64)
    if t > 0:
        q = (q >> t) << t
    return q, step

def _q_signed_w(W, wmax, b, t):
    """Quantize weights to signed b-bit per-output-channel, truncate t LSBs (magnitude)."""
    levels = (1 << (b - 1)) - 1
    step = wmax / levels                    # per output channel (vector over cols)
    q = np.rint(W / step)
    q = np.clip(q, -levels, levels).astype(np.int64)
    if t > 0:
        mag = (np.abs(q) >> t) << t
        q = np.sign(q) * mag
    return q, step

def forward_quant(model, X, config, act_max, w_max, bs=4096, return_probs=False):
    """
    Approximate forward pass.
    config: list of (b, t) per weight layer.
    Returns logits (and softmax probs if requested).
    """
    outs = []
    for s in range(0, len(X), bs):
        a = X[s:s + bs]
        for i in range(len(model.W)):
            b, t = config[i]
            aq, a_step = _q_unsigned(a, act_max[i], b, t)
            wq, w_step = _q_signed_w(model.W[i], w_max[i], b, t)
            # Quantized operands are integers; products+sums fit exactly in float64
            # (|z| < 2^53), so we use fast BLAS matmul while remaining bit-exact.
            z_int = aq.astype(np.float64) @ wq.astype(np.float64)   # (N, out), exact
            z = z_int * (a_step * w_step)               # w_step is per-col vector
            z = z + model.b[i]
            a = np.maximum(z, 0.0) if i < len(model.W) - 1 else z
        outs.append(a)
    logits = np.concatenate(outs, axis=0)
    if return_probs:
        return logits, _softmax(logits)
    return logits

# --------------------------------------------------------------------------- #
#  Energy model (45 nm, Horowitz ISSCC 2014 calibration)
# --------------------------------------------------------------------------- #
# 8-bit integer multiply ~ 0.2 pJ ; 32-bit integer multiply ~ 3.1 pJ  => ~ quadratic
#   E_mult(n) = ALPHA * n^2  with ALPHA = 0.2 / 8^2
# 32-bit integer add ~ 0.1 pJ (accumulator)            => E_acc = 0.1 pJ (fixed 32-bit)
ALPHA_MULT = 0.2 / (8.0 ** 2)     # pJ per bit^2
E_ACC = 0.1                       # pJ, 32-bit accumulate

def e_mac(b, t):
    """Per-MAC compute energy (pJ) for effective multiplier width (b-t)."""
    eff = max(b - t, 1)
    return ALPHA_MULT * (eff ** 2) + E_ACC

def layer_macs(sizes):
    """MACs per layer per sample for an MLP with given sizes."""
    return [sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)]

def config_energy(config, sizes):
    """Total per-sample compute energy (pJ) for a per-layer config."""
    macs = layer_macs(sizes)
    return float(sum(m * e_mac(b, t) for m, (b, t) in zip(macs, config)))

# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def accuracy(logits, y):
    return float((logits.argmax(1) == y).mean())

def expected_calibration_error(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)

if __name__ == "__main__":
    # quick self-test of the energy model monotonicity
    for (b, t) in [(8, 0), (6, 0), (6, 2), (5, 1), (4, 0), (4, 2), (3, 0)]:
        print(f"(b={b},t={t}) eff={b-t} E_MAC={e_mac(b,t):.4f} pJ")
