"""cnnlib.py -- compact NumPy CNN with im2col, for the revision generality study.
Approximate/quantized inference reuses the operand-truncation multiplier model
from axlib (exact integer simulation). Deterministic, CPU-only."""
import numpy as np
import axlib as A

def im2col(X, kh, kw, stride=1):
    """X: (N,C,H,W) -> cols: (N, out_h*out_w, C*kh*kw), plus (out_h,out_w).
    Vectorised with sliding_window_view (no Python loop)."""
    N, C, H, W = X.shape
    oh = (H - kh) // stride + 1; ow = (W - kw) // stride + 1
    sw = np.lib.stride_tricks.sliding_window_view(X, (kh, kw), axis=(2, 3))  # N,C,oh,ow,kh,kw
    if stride != 1:
        sw = sw[:, :, ::stride, ::stride, :, :]
    # -> N, oh, ow, C, kh, kw -> N, oh*ow, C*kh*kw
    sw = sw.transpose(0, 2, 3, 1, 4, 5).reshape(N, oh * ow, C * kh * kw)
    return np.ascontiguousarray(sw), oh, ow

def col2im(dcols, X_shape, kh, kw, stride=1):
    N, C, H, W = X_shape
    oh = (H - kh) // stride + 1; ow = (W - kw) // stride + 1
    dX = np.zeros(X_shape, dtype=dcols.dtype)
    idx = 0
    for i in range(oh):
        for j in range(ow):
            patch = dcols[:, idx, :].reshape(N, C, kh, kw)
            dX[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw] += patch
            idx += 1
    return dX

def maxpool2(X):
    """2x2 stride-2 max pool. X:(N,C,H,W). Returns out and argmax mask."""
    N, C, H, W = X.shape
    H2, W2 = H // 2, W // 2
    Xr = X[:, :, :H2*2, :W2*2].reshape(N, C, H2, 2, W2, 2)
    out = Xr.max(axis=(3, 5))
    mask = (Xr == out[:, :, :, None, :, None])
    return out, mask, (H, W)

def maxpool2_back(dout, mask, shape, Cn):
    N, C, H, W = dout.shape[0], dout.shape[1], shape[0], shape[1]
    H2, W2 = H // 2, W // 2
    dXr = mask * dout[:, :, :, None, :, None]
    dX = np.zeros((N, C, H2*2, W2*2), dtype=dout.dtype)
    dX[:] = dXr.reshape(N, C, H2*2, W2*2)
    if H2*2 != H or W2*2 != W:
        full = np.zeros((N, C, H, W), dtype=dout.dtype); full[:, :, :H2*2, :W2*2] = dX; return full
    return dX

class SmallCNN:
    """conv(1->c1,3x3)-relu-pool - conv(c1->c2,3x3)-relu-pool - fc-relu - fc."""
    def __init__(self, c1=8, c2=16, fc=64, seed=0, in_hw=28):
        rng = np.random.default_rng(seed)
        self.k = 3
        self.W1 = rng.standard_normal((c1, 1, 3, 3)) * np.sqrt(2.0 / 9)
        self.b1 = np.zeros(c1)
        self.W2 = rng.standard_normal((c2, c1, 3, 3)) * np.sqrt(2.0 / (c1*9))
        self.b2 = np.zeros(c2)
        h1 = (in_hw - 2) // 2                     # after conv1(valid)+pool
        h2 = (h1 - 2) // 2                        # after conv2(valid)+pool
        self.flat = c2 * h2 * h2
        self.Wf = rng.standard_normal((self.flat, fc)) * np.sqrt(2.0 / self.flat)
        self.bf = np.zeros(fc)
        self.Wo = rng.standard_normal((fc, 10)) * np.sqrt(2.0 / fc)
        self.bo = np.zeros(10)
        self.c1, self.c2, self.fcn = c1, c2, fc
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in
                   ["W1","b1","W2","b2","Wf","bf","Wo","bo"]}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._m}
        self._t = 0

    def _conv_fwd(self, X, W, b):
        N = X.shape[0]; oc = W.shape[0]
        cols, oh, ow = im2col(X, 3, 3)                 # N, oh*ow, C*9
        Wm = W.reshape(oc, -1).T                        # C*9, oc
        out = cols @ Wm + b                             # N, oh*ow, oc
        out = out.transpose(0, 2, 1).reshape(N, oc, oh, ow)
        return out, cols, oh, ow

    def forward(self, X, cache=False):
        N = X.shape[0]
        z1, cols1, oh1, ow1 = self._conv_fwd(X, self.W1, self.b1)
        a1 = np.maximum(z1, 0.0)
        p1, m1, s1 = maxpool2(a1)
        z2, cols2, oh2, ow2 = self._conv_fwd(p1, self.W2, self.b2)
        a2 = np.maximum(z2, 0.0)
        p2, m2, s2 = maxpool2(a2)
        flat = p2.reshape(N, -1)
        zf = flat @ self.Wf + self.bf
        af = np.maximum(zf, 0.0)
        zo = af @ self.Wo + self.bo
        if cache:
            self._cache = dict(X=X, cols1=cols1, oh1=oh1, ow1=ow1, z1=z1, a1=a1, m1=m1, s1=s1,
                               p1=p1, cols2=cols2, oh2=oh2, ow2=ow2, z2=z2, a2=a2, m2=m2, s2=s2,
                               p2=p2, flat=flat, zf=zf, af=af)
        return zo

    def train(self, X, y, epochs=6, bs=128, lr=1e-3, seed=0, verbose=False):
        rng = np.random.default_rng(1000 + seed)
        N = X.shape[0]
        for ep in range(epochs):
            perm = rng.permutation(N)
            for s in range(0, N, bs):
                idx = perm[s:s+bs]; xb = X[idx]; yb = y[idx]
                zo = self.forward(xb, cache=True)
                g = self._backward(yb)
                self._adam_step(g, lr)
            if verbose:
                acc = (self.predict(X[:5000]) == y[:5000]).mean()
                print(f"    ep{ep} train_acc~{acc:.4f}")

    def _backward(self, y):
        c = self._cache; N = y.shape[0]
        probs = A._softmax(c["zo"] if "zo" in c else self.forward(c["X"]))
        # recompute zo from af to get grad (cache af)
        zo = c["af"] @ self.Wo + self.bo
        probs = A._softmax(zo)
        dzo = probs; dzo[np.arange(N), y] -= 1; dzo /= N
        dWo = c["af"].T @ dzo; dbo = dzo.sum(0)
        daf = dzo @ self.Wo.T
        dzf = daf * (c["zf"] > 0)
        dWf = c["flat"].T @ dzf; dbf = dzf.sum(0)
        dflat = dzf @ self.Wf.T
        dp2 = dflat.reshape(c["p2"].shape)
        da2 = maxpool2_back(dp2, c["m2"], c["s2"], self.c2)
        dz2 = da2 * (c["z2"] > 0)
        # conv2 backward
        N_, oc, oh2, ow2 = dz2.shape
        dz2m = dz2.reshape(N_, oc, -1).transpose(0, 2, 1)   # N, oh*ow, oc
        Wm2 = self.W2.reshape(oc, -1)                        # oc, C*9
        dWm2 = np.einsum('npo,npc->oc', dz2m, c["cols2"])
        dW2 = dWm2.reshape(self.W2.shape); db2 = dz2.sum(axis=(0, 2, 3))
        dcols2 = dz2m @ Wm2                                  # N, oh*ow, C*9
        dp1 = col2im(dcols2, c["p1"].shape, 3, 3)
        da1 = maxpool2_back(dp1, c["m1"], c["s1"], self.c1)
        dz1 = da1 * (c["z1"] > 0)
        N_, oc1, oh1, ow1 = dz1.shape
        dz1m = dz1.reshape(N_, oc1, -1).transpose(0, 2, 1)
        dWm1 = np.einsum('npo,npc->oc', dz1m, c["cols1"])
        dW1 = dWm1.reshape(self.W1.shape); db1 = dz1.sum(axis=(0, 2, 3))
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, Wf=dWf, bf=dbf, Wo=dWo, bo=dbo)

    def _adam_step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in g:
            self._m[k] = b1*self._m[k] + (1-b1)*g[k]
            self._v[k] = b2*self._v[k] + (1-b2)*(g[k]**2)
            mh = self._m[k]/(1-b1**self._t); vh = self._v[k]/(1-b2**self._t)
            setattr(self, k, getattr(self, k) - lr*mh/(np.sqrt(vh)+eps))

    def predict(self, X, bs=2048):
        out = []
        for s in range(0, X.shape[0], bs):
            out.append(self.forward(X[s:s+bs]).argmax(1))
        return np.concatenate(out)

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in
                          ["W1","b1","W2","b2","Wf","bf","Wo","bo"]},
                 dims=np.array([self.c1, self.c2, self.fcn]))
    @classmethod
    def load(cls, path):
        d = np.load(path); dims = d["dims"]
        m = cls(int(dims[0]), int(dims[1]), int(dims[2]))
        for k in ["W1","b1","W2","b2","Wf","bf","Wo","bo"]:
            setattr(m, k, d[k])
        return m

# MAC counts per layer for energy model
def cnn_layer_macs(m, in_hw=28):
    oh1 = in_hw - 2; h1 = oh1 // 2
    oh2 = h1 - 2; h2 = oh2 // 2
    mac_c1 = oh1*oh1 * m.c1 * (1*9)
    mac_c2 = oh2*oh2 * m.c2 * (m.c1*9)
    mac_fc = m.flat * m.fcn
    mac_out = m.fcn * 10
    return [mac_c1, mac_c2, mac_fc, mac_out]

# ---------------- quantized / approximate CNN inference ---------------- #
def cnn_calibrate(m, Xcal):
    """Per-layer activation max (abs) and per-output-channel weight max (abs)."""
    N = Xcal.shape[0]
    act_max, w_max = [], []
    # layer 1: input image
    cols1, oh1, ow1 = im2col(Xcal, 3, 3)
    act_max.append(float(np.abs(cols1).max()))
    Wm1 = m.W1.reshape(m.c1, -1).T                       # (C*9, oc)
    w_max.append(np.abs(Wm1).max(axis=0))                # per out-channel
    # forward (fp) to get inputs to subsequent layers
    z1 = (cols1 @ Wm1 + m.b1).transpose(0, 2, 1).reshape(N, m.c1, oh1, ow1)
    p1, _, _ = maxpool2(np.maximum(z1, 0.0))
    cols2, oh2, ow2 = im2col(p1, 3, 3)
    act_max.append(float(np.abs(cols2).max()))
    Wm2 = m.W2.reshape(m.c2, -1).T
    w_max.append(np.abs(Wm2).max(axis=0))
    z2 = (cols2 @ Wm2 + m.b2).transpose(0, 2, 1).reshape(N, m.c2, oh2, ow2)
    p2, _, _ = maxpool2(np.maximum(z2, 0.0))
    flat = p2.reshape(N, -1)
    act_max.append(float(np.abs(flat).max()))
    w_max.append(np.abs(m.Wf).max(axis=0))
    af = np.maximum(flat @ m.Wf + m.bf, 0.0)
    act_max.append(float(np.abs(af).max()))
    w_max.append(np.abs(m.Wo).max(axis=0))
    return act_max, w_max

def _qmatmul(a, W, amax, wmax, b, t):
    aq, a_step = A._q_unsigned(a, amax, b, t)
    wq, w_step = A._q_signed_w(W, wmax, b, t)
    z = (aq.astype(np.float64) @ wq.astype(np.float64)) * (a_step * w_step)
    return z

def forward_quant_cnn(m, X, config, act_max, w_max, bs=2048, return_probs=False):
    """config: list of 4 (b,t) for [conv1, conv2, fc1, fc2]."""
    outs = []
    for s in range(0, X.shape[0], bs):
        xb = X[s:s+bs]; N = xb.shape[0]
        # conv1
        b, t = config[0]
        cols1, oh1, ow1 = im2col(xb, 3, 3)
        Wm1 = m.W1.reshape(m.c1, -1).T
        z1 = _qmatmul(cols1, Wm1, act_max[0], w_max[0], b, t) + m.b1
        z1 = z1.transpose(0, 2, 1).reshape(N, m.c1, oh1, ow1)
        p1, _, _ = maxpool2(np.maximum(z1, 0.0))
        # conv2
        b, t = config[1]
        cols2, oh2, ow2 = im2col(p1, 3, 3)
        Wm2 = m.W2.reshape(m.c2, -1).T
        z2 = _qmatmul(cols2, Wm2, act_max[1], w_max[1], b, t) + m.b2
        z2 = z2.transpose(0, 2, 1).reshape(N, m.c2, oh2, ow2)
        p2, _, _ = maxpool2(np.maximum(z2, 0.0))
        flat = p2.reshape(N, -1)
        # fc1
        b, t = config[2]
        af = np.maximum(_qmatmul(flat, m.Wf, act_max[2], w_max[2], b, t) + m.bf, 0.0)
        # fc2
        b, t = config[3]
        zo = _qmatmul(af, m.Wo, act_max[3], w_max[3], b, t) + m.bo
        outs.append(zo)
    logits = np.concatenate(outs, 0)
    if return_probs:
        return logits, A._softmax(logits)
    return logits

def cnn_config_energy(config, macs):
    return float(sum(mac * A.e_mac(b, t) for mac, (b, t) in zip(macs, config)))
