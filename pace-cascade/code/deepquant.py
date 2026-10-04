"""deepquant.py -- approximate (quantized + operand-truncated) inference for the
deeper models of the second revision, simulated EXACTLY in integer arithmetic.

Conventions are identical to axlib.forward_quant (main MLP study):
  * activations: per-tensor, b-bit; unsigned after ReLU / softmax, symmetric
    signed after LayerNorm, GELU and inside attention;
  * weights: per-output-channel symmetric b-bit (BatchNorm folded first);
  * operand truncation t zeroes the t LSBs of the magnitude of BOTH operands;
  * every product and sum is formed in float64 on integer-valued operands
    (|sums| << 2^53), hence bit-exact; dequantisation and the non-MAC operators
    (residual add, pooling, LayerNorm, softmax, GELU) are floating point.

A configuration assigns one option (b, t) to each *stage*:
  ResNet-8 : [stem], [stage 1], [stage 2 + shortcut], [stage 3 + shortcut], [fc]
  ViT-4    : [patch embedding + block 1], [block 2], [block 3], [block 4 + head]
Because stages are sequential, the exhaustive enumeration re-uses the output of
every configuration prefix (depth-first search), which cuts the cost of
enumerating all |O|^G configurations by about an order of magnitude.
"""
import itertools
import numpy as np
from scipy.special import erf


# --------------------------------------------------------------------------- #
#  quantisers (float64, integer-valued)
# --------------------------------------------------------------------------- #
def _trunc(q, t, signed):
    if not t:
        return q
    f = float(1 << t)
    return (np.sign(q) * np.floor(np.abs(q) / f) * f) if signed else np.floor(q / f) * f


def qact_u(x, amax, b, t):
    levels = float((1 << b) - 1); step = amax / levels
    q = np.clip(np.rint(np.asarray(x, np.float64) / step), 0.0, levels)
    return _trunc(q, t, False), step


def qact_s(x, amax, b, t):
    levels = float((1 << (b - 1)) - 1); step = amax / levels
    q = np.clip(np.rint(np.asarray(x, np.float64) / step), -levels, levels)
    return _trunc(q, t, True), step


def qweight(W2, b, t):
    """W2: (K, cout) -> per-output-channel signed b-bit."""
    wmax = np.abs(W2).max(0) + 1e-8
    levels = float((1 << (b - 1)) - 1); step = wmax / levels
    q = np.clip(np.rint(W2 / step), -levels, levels)
    return _trunc(q, t, True), step


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


relu = lambda x: np.maximum(x, 0.0)


# --------------------------------------------------------------------------- #
#  ResNet-8
# --------------------------------------------------------------------------- #
_PAD = {(3, 1): (1, 1), (3, 2): (0, 1), (1, 2): (0, 0), (1, 1): (0, 0)}   # XLA 'SAME', even inputs


def im2col(x, k, s):
    lo, hi = _PAD[(k, s)]
    if lo or hi:
        x = np.pad(x, ((0, 0), (lo, hi), (lo, hi), (0, 0)))
    sw = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))   # B,H',W',C,k,k
    if s > 1:
        sw = sw[:, ::s, ::s]
    B, Ho, Wo = sw.shape[:3]
    return sw.reshape(B * Ho * Wo, -1), (B, Ho, Wo)


class ResNet8Q:
    LAYERS = {"conv0": (3, 1), "s1c1": (3, 1), "s1c2": (3, 1), "s2c1": (3, 2), "s2c2": (3, 1),
              "s2sc": (1, 2), "s3c1": (3, 2), "s3c2": (3, 1), "s3sc": (1, 2)}
    GROUP_MACS = [442368, 4718592, 3670016, 3670016, 640]
    n_groups = 5
    CHUNK = 250

    def __init__(self, npz_path, Xcal):
        d = np.load(npz_path)
        self.W, self.B = {}, {}
        for name, (k, s) in self.LAYERS.items():
            sc = d[name + "_g"] / np.sqrt(d[name + "_v"] + 1e-5)
            Wf = (d[name + "_w"] * sc).astype(np.float64)                       # HWIO
            self.W[name] = Wf.transpose(2, 0, 1, 3).reshape(-1, Wf.shape[3])    # (cin*k*k, cout)
            self.B[name] = (d[name + "_b"] - d[name + "_m"] * sc).astype(np.float64)
        self.W["fc"] = d["fc_w"].astype(np.float64); self.B["fc"] = d["fc_b"].astype(np.float64)
        self._wq = {}
        self.amax = {}
        self._calibrate(np.asarray(Xcal, np.float64))

    def _wget(self, name, opt):
        key = (name, opt)
        if key not in self._wq:
            self._wq[key] = qweight(self.W[name], *opt)
        return self._wq[key]

    def conv(self, x, name, opt):
        k, s = self.LAYERS[name]
        if opt is None:
            xa, Wm, sa, sw = x, self.W[name], 1.0, 1.0
            if self._calib:
                self.amax[name] = max(self.amax.get(name, 0.0), float(np.abs(x).max()))
        else:
            xa, sa = qact_u(x, self.amax[name], *opt)
            Wm, sw = self._wget(name, opt)
        out = []
        for i in range(0, x.shape[0], 64):
            cols, (B, Ho, Wo) = im2col(xa[i:i + 64], k, s)
            out.append((cols @ Wm).reshape(B, Ho, Wo, -1))
        return np.concatenate(out) * (sa * sw) + self.B[name]

    def fc(self, h, opt):
        if opt is None:
            if self._calib:
                self.amax["fc"] = max(self.amax.get("fc", 0.0), float(np.abs(h).max()))
            return h @ self.W["fc"] + self.B["fc"]
        qa, sa = qact_u(h, self.amax["fc"], *opt)
        Wq, sw = self._wget("fc", opt)
        return (qa @ Wq) * (sa * sw) + self.B["fc"]

    def stage(self, g, h, opt):
        if g == 0:
            return relu(self.conv(np.asarray(h, np.float64), "conv0", opt))
        if g == 1:
            r = relu(self.conv(h, "s1c1", opt)); r = self.conv(r, "s1c2", opt)
            return relu(h + r)
        if g == 2:
            r = relu(self.conv(h, "s2c1", opt)); r = self.conv(r, "s2c2", opt)
            return relu(self.conv(h, "s2sc", opt) + r)
        if g == 3:
            r = relu(self.conv(h, "s3c1", opt)); r = self.conv(r, "s3c2", opt)
            return relu(self.conv(h, "s3sc", opt) + r).mean((1, 2))
        return self.fc(h, opt)

    def _calibrate(self, Xcal):
        self._calib = True
        for i in range(0, len(Xcal), self.CHUNK):
            h = Xcal[i:i + self.CHUNK]
            for g in range(self.n_groups):
                h = self.stage(g, h, None)
        self._calib = False
        self.amax = {k: v + 1e-8 for k, v in self.amax.items()}

    def forward(self, X, config):
        out = []
        for i in range(0, len(X), self.CHUNK):
            h = X[i:i + self.CHUNK]
            for g in range(self.n_groups):
                h = self.stage(g, h, None if config is None else tuple(config[g]))
            out.append(h)
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
#  ViT-4
# --------------------------------------------------------------------------- #
def _ln(h, g, b, eps=1e-6):
    mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
    return (h - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def patchify(x, P):
    N = x.shape[0]; g = 28 // P
    return x.reshape(N, g, P, g, P).transpose(0, 1, 3, 2, 4).reshape(N, g * g, P * P)


class ViT4Q:
    GROUP_MACS = [644224, 594048, 594048, 594688]
    n_groups = 4
    CHUNK = 2000

    def __init__(self, npz_path, cfg, Xcal):
        z = np.load(npz_path)
        self.p = {k: z[k].astype(np.float64) for k in z.files}
        self.d, self.H, self.depth, self.P = cfg["dim"], cfg["heads"], cfg["depth"], cfg["patch"]
        self.dh = self.d // self.H
        self._wq = {}
        self.amax = {}
        self._calib = True
        for i in range(0, len(Xcal), self.CHUNK):
            h = Xcal[i:i + self.CHUNK]
            for g in range(self.n_groups):
                h = self.stage(g, h, None)
        self._calib = False
        self.amax = {k: v + 1e-8 for k, v in self.amax.items()}

    def _rec(self, key, x):
        if self._calib:
            self.amax[key] = max(self.amax.get(key, 0.0), float(np.abs(x).max()))

    def _wget(self, name, opt):
        if (name, opt) not in self._wq:
            self._wq[(name, opt)] = qweight(self.p[name], *opt)
        return self._wq[(name, opt)]

    def lin(self, a, wname, bname, key, opt, signed=True):
        """a: (..., din) -> (..., dout), quantised operands if opt is given."""
        sh = a.shape
        a2 = a.reshape(-1, sh[-1])
        if opt is None:
            self._rec(key, a2)
            z = a2 @ self.p[wname]
        else:
            qa, sa = (qact_s if signed else qact_u)(a2, self.amax[key], *opt)
            Wq, sw = self._wget(wname, opt)
            z = (qa @ Wq) * (sa * sw)
        return (z + self.p[bname]).reshape(*sh[:-1], -1)

    def amul(self, x, y, kx, ky, opt, ux=False):
        """activation x activation product  x @ y  (batched)."""
        if opt is None:
            self._rec(kx, x); self._rec(ky, y)
            return x @ y
        qx, sx = (qact_u if ux else qact_s)(x, self.amax[kx], *opt)
        qy, sy = qact_s(y, self.amax[ky], *opt)
        return (qx @ qy) * (sx * sy)

    def block(self, h, l, opt):
        p, N, T, H, dh = self.p, h.shape[0], h.shape[1], self.H, self.dh
        a = _ln(h, p[f"b{l}_ln1_g"], p[f"b{l}_ln1_b"])
        qkv = self.lin(a, f"b{l}_qkv_w", f"b{l}_qkv_b", f"b{l}_ln1", opt)
        q, k, v = np.split(qkv, 3, -1)
        q = q.reshape(N, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(N, T, H, dh).transpose(0, 2, 3, 1)               # pre-transposed K^T
        v = v.reshape(N, T, H, dh).transpose(0, 2, 1, 3)
        att = softmax(self.amul(q, k, f"b{l}_q", f"b{l}_k", opt) / np.sqrt(dh), -1)
        ctx = self.amul(att, v, f"b{l}_att", f"b{l}_v", opt, ux=True)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(N, T, self.d)
        h = h + self.lin(ctx, f"b{l}_proj_w", f"b{l}_proj_b", f"b{l}_ctx", opt)
        a = _ln(h, p[f"b{l}_ln2_g"], p[f"b{l}_ln2_b"])
        m = _gelu(self.lin(a, f"b{l}_fc1_w", f"b{l}_fc1_b", f"b{l}_ln2", opt))
        return h + self.lin(m, f"b{l}_fc2_w", f"b{l}_fc2_b", f"b{l}_gelu", opt)

    def stage(self, g, h, opt):
        p = self.p
        if g == 0:
            x = np.asarray(h, np.float64)
            t = self.lin(patchify(x, self.P), "emb_w", "emb_b", "emb", opt, signed=False)
            N = t.shape[0]
            t = np.concatenate([np.broadcast_to(p["cls"], (N, 1, self.d)), t], 1) + p["pos"]
            return self.block(t, 0, opt)
        h = self.block(h, g, opt)
        if g == self.n_groups - 1:
            hc = _ln(h[:, 0], p["lnf_g"], p["lnf_b"])
            return self.lin(hc, "head_w", "head_b", "lnf", opt)
        return h

    def forward(self, X, config):
        out = []
        for i in range(0, len(X), self.CHUNK):
            h = X[i:i + self.CHUNK]
            for g in range(self.n_groups):
                h = self.stage(g, h, None if config is None else tuple(config[g]))
            out.append(h)
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
#  exhaustive enumeration with prefix re-use
# --------------------------------------------------------------------------- #
def enumerate_all(engine, X, options, log_every=None):
    G, n = engine.n_groups, len(options)
    C, N = n ** G, len(X)
    pred = np.zeros((C, N), np.int16); msp = np.zeros((C, N), np.float32)
    for s0 in range(0, N, engine.CHUNK):
        x0 = X[s0:s0 + engine.CHUNK]; nb = len(x0)

        def rec(g, h, idx):
            for oi, opt in enumerate(options):
                h2 = engine.stage(g, h, opt)
                j = idx * n + oi
                if g == G - 1:
                    pr = softmax(h2)
                    pred[j, s0:s0 + nb] = pr.argmax(1); msp[j, s0:s0 + nb] = pr.max(1)
                else:
                    rec(g + 1, h2, j)
        rec(0, x0, 0)
        if log_every:
            print(f"    enumerated {min(s0 + nb, N)}/{N} inputs", flush=True)
    configs = [[list(o) for o in c] for c in itertools.product(options, repeat=G)]
    return configs, pred, msp
