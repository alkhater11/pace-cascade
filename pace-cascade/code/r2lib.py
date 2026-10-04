"""r2lib.py -- shared machinery for the experiments added in the second revision.

Contents
  * energy models: the paper's 45 nm model (H0) plus a family of hardware
    energy models (configs/energy_models.json) used to test under which hardware
    architectures the conclusions hold (Reviewer 2, comment 3);
  * exact cascade curves: for a cheap/high pair (lo, hi) and the max-softmax gate,
    escalating the k least-confident inputs gives accuracy
        acc(k) = acc_lo + (1/N) * sum_{i<=k} (c_hi[i] - c_lo[i])
    over inputs sorted by the cheap pass's confidence, and energy
        E(k)   = E_lo + (k/N) * (E_hi + E_switch).
    Every threshold is therefore evaluated exactly (no tau grid);
  * the static Pareto frontier, the lower-energy envelope over all pairs, and the
    feasibility quantities of Proposition 3 (break-even and minimum escalation);
  * the deployment protocol of the paper (selection on validation, evaluation
    on test);
  * deterministic input corruptions (Reviewer 2, comment 4).
"""
import os, json, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(RES, "caches")
os.makedirs(CACHE, exist_ok=True)

OPTIONS = [(8, 0), (6, 0), (5, 0), (4, 0), (6, 2), (4, 1)]            # main-text option set
OPTIONS_EXT = OPTIONS + [(3, 0), (2, 0), (4, 2)]                      # + effective width 3 and 2
DEEP_OPTIONS = [(8, 0), (6, 0), (5, 0), (4, 0)]                       # ResNet-8 / ViT-4 (per stage)
FP32_PJ = 4.6


# --------------------------------------------------------------------------- #
#  Energy models
# --------------------------------------------------------------------------- #
def load_energy_models():
    with open(os.path.join(ROOT, "configs", "energy_models.json")) as f:
        return json.load(f)


_G = {"lin": lambda x: x, "quad": lambda x: x * x, "cubic": lambda x: x ** 3, "exp": lambda x: 2.0 ** x}


def emac(b, t, m):
    """Per-MAC energy (pJ) of option (b, t) under hardware energy model m."""
    beta = max(b - t, 1)
    k = m["kind"]
    if k == "poly":
        e = (m.get("a0", 0.0) + m.get("a1", 0.0) * beta + m.get("a2", 0.0) * beta ** 2
             + m.get("aE", 0.0) * 2.0 ** beta)
    elif k == "dvafs":                       # supply voltage scaled with effective width
        v = (m["v0"] + m["v1"] * beta / 8.0) / (m["v0"] + m["v1"])
        e = (m["a0"] + m["a2"] * beta ** 2) * v * v
    elif k == "family":                      # e8 * [phi + (1 - phi) g(beta) / g(8)]
        g = _G[m["g"]]
        e = m["e8"] * (m["phi"] + (1.0 - m["phi"]) * g(beta) / g(8))
    elif k == "flat":                        # fixed-width datapath: no saving below 8 bits
        e = m["e8"]
    else:
        raise ValueError(k)
    return float(e + m.get("mu", 0.0) * b)


def family_model(g, phi, e8=0.3):
    return {"kind": "family", "g": g, "phi": phi, "e8": e8}


def energies(configs, group_macs, m):
    """Per-sample energy (pJ) of each configuration (list of per-group options)."""
    return np.array([sum(M * emac(b, t, m) for M, (b, t) in zip(group_macs, cfg)) for cfg in configs])


# --------------------------------------------------------------------------- #
#  Per-configuration prediction caches
# --------------------------------------------------------------------------- #
class Cache:
    """Predictions of every configuration on one evaluation split."""

    def __init__(self, configs, pred, msp, y):
        self.configs = [tuple(tuple(int(v) for v in o) for o in c) for c in configs]
        self.index = {c: i for i, c in enumerate(self.configs)}
        self.y = np.asarray(y)
        self.correct = (np.asarray(pred) == self.y[None, :])
        self.msp = np.asarray(msp, dtype=np.float32)
        self.N = len(self.y)
        self.acc = self.correct.mean(1)
        self._order = {}

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["configs"], z["pred"], z["msp"], z["y"])

    def order(self, i):
        if i not in self._order:
            self._order[i] = np.argsort(self.msp[i], kind="stable")
        return self._order[i]

    def subset(self, idx):
        idx = list(idx)
        c = Cache.__new__(Cache)
        c.configs = [self.configs[i] for i in idx]
        c.index = {cf: j for j, cf in enumerate(c.configs)}
        c.y = self.y; c.correct = self.correct[idx]; c.msp = self.msp[idx]
        c.N = self.N; c.acc = self.acc[idx]; c._order = {}
        return c


def save_cache(path, configs, pred, msp, y):
    np.savez_compressed(path, configs=np.array(configs, dtype=np.int16),
                        pred=np.asarray(pred, dtype=np.int16), msp=np.asarray(msp, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int16))


# --------------------------------------------------------------------------- #
#  Frontier, cascade curves and envelopes
# --------------------------------------------------------------------------- #
def pareto_idx(E, acc):
    order = np.lexsort((-acc, E))
    fr, best = [], -1.0
    for i in order:
        if acc[i] > best + 1e-12:
            fr.append(int(i)); best = acc[i]
    return fr


def static_energy_at(E, acc, targets):
    """Least energy of any configuration whose accuracy is >= each target."""
    o = np.argsort(-acc, kind="stable")
    acc_s, emin = acc[o], np.minimum.accumulate(E[o])
    k = np.searchsorted(-acc_s, -np.asarray(targets) + 1e-12, side="right")
    out = np.full(len(k), np.inf)
    m = k > 0
    out[m] = emin[k[m] - 1]
    return out


def pair_curve(cache, lo, hi):
    """Escalation fraction and accuracy for k = 0..N escalated inputs (max-softmax gate)."""
    o = cache.order(lo)
    clo = cache.correct[lo][o].astype(np.int32)
    chi = cache.correct[hi][o].astype(np.int32)
    acc = (clo.sum() + np.concatenate([[0], np.cumsum(chi - clo)])) / cache.N
    return np.arange(cache.N + 1) / cache.N, acc


def cascade_energy_at(cache, E, lo, hi, targets, e_switch=0.0):
    pesc, acc = pair_curve(cache, lo, hi)
    cm = np.maximum.accumulate(acc)
    k = np.searchsorted(cm, np.asarray(targets) - 1e-12, side="left")
    out = np.full(len(k), np.inf)
    ok = k <= cache.N
    out[ok] = E[lo] + pesc[k[ok]] * (E[hi] + e_switch)
    return out, pesc, acc


def envelope(cache, E, pairs, targets, e_switch=0.0):
    env = np.full(len(targets), np.inf)
    arg = np.full(len(targets), -1)
    best_acc = -1.0
    for j, (lo, hi) in enumerate(pairs):
        e, _, acc = cascade_energy_at(cache, E, lo, hi, targets, e_switch)
        better = e < env
        env[better] = e[better]; arg[better] = j
        best_acc = max(best_acc, float(acc.max()))
    return env, arg, best_acc


def compare_frontier(cache, E, int8_idx, e_switch=0.0, band=0.03, ngrid=121, max_pairs_frontier=60):
    """Oracle-tuned comparison on one split: exact static frontier versus the
    lower-energy envelope of cascades built from any two frontier configurations."""
    acc = cache.acc
    fr = pareto_idx(E, acc)
    if len(fr) > max_pairs_frontier:                 # keep the most relevant (accurate end)
        fr = sorted(fr, key=lambda i: -acc[i])[:max_pairs_frontier]
    pairs = [(l, h) for l in fr for h in fr if E[l] < E[h]]
    a8 = acc[int8_idx]; a_star = a8 - 0.002
    amax = acc.max()
    lo_band = max(a8 - band, acc[fr].min())
    grid = np.linspace(lo_band, amax, ngrid)
    tg = np.append(grid, a_star)
    stat = static_energy_at(E, acc, tg)
    env, arg, best_acc = envelope(cache, E, pairs, tg, e_switch)
    rel = (stat - env) / stat                        # > 0: cascade cheaper at equal accuracy
    rg = rel[:-1]
    j_star = int(arg[-1])
    out = {"a_int8": float(a8), "a_star": float(a_star), "static_E_star": float(stat[-1]),
           "cascade_E_star": float(env[-1]), "S_star": float(rel[-1]),
           "S_max": float(np.max(rg)), "frac_win": float(np.mean(rg > 1e-9)),
           "n_frontier": len(fr), "n_pairs": len(pairs),
           "ensemble_best_acc": best_acc, "static_best_acc": float(amax)}
    if j_star >= 0:
        lo, hi = pairs[j_star]
        out["best_pair_star"] = {"lo": [list(o) for o in cache.configs[lo]],
                                 "hi": [list(o) for o in cache.configs[hi]],
                                 "rho": float(E[lo] / E[hi])}
    out["grid"] = grid.tolist(); out["static_curve"] = stat[:-1].tolist(); out["envelope"] = env[:-1].tolist()
    return out


# --------------------------------------------------------------------------- #
#  Deployment protocol (as in the paper): select on validation, evaluate on test
# --------------------------------------------------------------------------- #
def select_protocol(vcache, E, int8_idx, gap=0.05, match=0.002):
    acc = vcache.acc
    fr = pareto_idx(E, acc)
    a8 = acc[int8_idx]
    cand = [i for i in fr if acc[i] >= a8 - gap]
    lo = min(cand, key=lambda i: E[i])
    mm = [i for i in fr if acc[i] >= a8 - match]
    smp = min(mm, key=lambda i: E[i]) if mm else int8_idx
    pesc, cacc = pair_curve(vcache, lo, int8_idx)
    cm = np.maximum.accumulate(cacc)
    k = int(np.searchsorted(cm, a8 - match - 1e-12, side="left"))
    if k > vcache.N:
        tau = 1.0001
    else:
        o = vcache.order(lo)
        tau = float(vcache.msp[lo][o][k]) if k < vcache.N else 1.0001   # escalate msp < tau
    p_gate = float(pesc[min(k, vcache.N)])
    return {"lo": lo, "hi": int8_idx, "smp": smp, "tau": tau, "a8_val": float(a8),
            "acc_lo_val": float(acc[lo]), "p_gate_val": p_gate,
            "E_lo": float(E[lo]), "E_hi": float(E[int8_idx]), "E_smp": float(E[smp])}


def feasibility(vcache, E, sel, e_switch=0.0, match=0.002):
    """Quantities of Proposition 3 on the validation split."""
    a_t = sel["a8_val"] - match
    e_stat = float(static_energy_at(E, vcache.acc, [a_t])[0])
    p_be = (e_stat - sel["E_lo"]) / (sel["E_hi"] + e_switch)
    p_min = max(a_t - sel["acc_lo_val"], 0.0)
    return {"rho": sel["E_lo"] / sel["E_hi"], "p_be": p_be, "p_min": p_min,
            "p_gate": sel["p_gate_val"], "headroom_H": (p_be / p_min if p_min > 0 else np.inf),
            "kappa_g": (sel["p_gate_val"] / p_min if p_min > 0 else np.nan),
            "pred_oracle_can_win": bool(p_min < p_be), "pred_gate_wins": bool(sel["p_gate_val"] < p_be)}


def observed_on_cache(tcache, E, sel, e_switch=0.0):
    """Apply the validation-selected cascade (lo, INT8, tau) to a test cache and
    compare with the static frontier at the cascade's own test accuracy."""
    lo, hi = sel["lo"], sel["hi"]
    esc = tcache.msp[lo] < sel["tau"]
    acc_c = float(np.where(esc, tcache.correct[hi], tcache.correct[lo]).mean())
    e_c = sel["E_lo"] + float(esc.mean()) * (sel["E_hi"] + e_switch)
    e_s = float(static_energy_at(E, tcache.acc, [acc_c])[0])
    return {"acc_cascade": acc_c, "E_cascade": e_c, "p_esc": float(esc.mean()),
            "E_static_at_acc": e_s, "gap_rel": (e_c - e_s) / e_s, "observed_win": bool(e_c < e_s)}


# --------------------------------------------------------------------------- #
#  Corruptions (deterministic; MNIST-C / ImageNet-C style, three severities)
# --------------------------------------------------------------------------- #
CORRUPTIONS = ["gaussian_noise", "impulse_noise", "gaussian_blur", "contrast"]
SEVERITY = {"gaussian_noise": [0.08, 0.16, 0.24], "impulse_noise": [0.03, 0.06, 0.10],
            "gaussian_blur": [0.6, 1.0, 1.5], "contrast": [0.6, 0.4, 0.25]}


def corrupt(X, kind, sev, seed=0):
    """X: float32 (N,H,W,C) in [0,1].  sev in {1,2,3}."""
    rng = np.random.default_rng(10007 * (CORRUPTIONS.index(kind) + 1) + 101 * sev + seed)
    p = SEVERITY[kind][sev - 1]
    X = X.astype(np.float32)
    if kind == "gaussian_noise":
        Y = X + rng.normal(0.0, p, X.shape).astype(np.float32)
    elif kind == "impulse_noise":
        Y = X.copy(); m = rng.random(X.shape) < p
        Y[m] = (rng.random(int(m.sum())) < 0.5).astype(np.float32)
    elif kind == "gaussian_blur":
        from scipy.ndimage import gaussian_filter
        Y = gaussian_filter(X, sigma=(0, p, p, 0), mode="nearest")
    elif kind == "contrast":
        mu = X.mean(axis=(1, 2, 3), keepdims=True)
        Y = (X - mu) * p + mu
    else:
        raise ValueError(kind)
    return np.clip(Y, 0.0, 1.0).astype(np.float32)


def corruption_sets():
    return [(k, s) for k in CORRUPTIONS for s in (1, 2, 3)]


# --------------------------------------------------------------------------- #
#  misc
# --------------------------------------------------------------------------- #
def ms(x):
    x = np.asarray(x, dtype=float)
    return [float(x.mean()), float(x.std())]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def dump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# --------------------------------------------------------------------------- #
#  Time-sharing (convex-hull) static baseline and validation-selected best pair
# --------------------------------------------------------------------------- #
def hull_energy_at(E, acc, targets, cand=None):
    """Least energy of an input-agnostic static mixture of two configurations
    (run C_i on a fraction w of inputs and C_j on the rest): the lower convex
    hull of the frontier.  Needs no gate and no re-computation."""
    idx = np.array(pareto_idx(E, acc) if cand is None else cand)
    e, a = E[idx], acc[idx]
    out = []
    for t in np.atleast_1d(targets):
        best = e[a >= t - 1e-12].min() if (a >= t - 1e-12).any() else np.inf
        lo_m, hi_m = a < t, a >= t
        if lo_m.any() and hi_m.any():
            el, al = e[lo_m][:, None], a[lo_m][:, None]
            eh, ah = e[hi_m][None, :], a[hi_m][None, :]
            w = (t - al) / (ah - al)
            best = min(best, float((el + w * (eh - el)).min()))
        out.append(best)
    return np.array(out)


def best_pair_val_to_test(vc, tc, E, int8_idx, match=0.002, max_frontier=60):
    """Choose the cheapest (C_lo, C_hi, tau) reaching INT8-0.2pp on VALIDATION,
    evaluate on TEST; compare with the static configuration and the static
    time-sharing mixture chosen the same way."""
    a_t = vc.acc[int8_idx] - match
    fr = sorted(pareto_idx(E, vc.acc), key=lambda i: -vc.acc[i])[:max_frontier]
    best = (np.inf, None)
    for lo in fr:
        for hi in fr:
            if E[lo] >= E[hi] or vc.acc[lo] >= a_t:
                continue
            e, pesc, acc = cascade_energy_at(vc, E, lo, hi, [a_t])
            if e[0] < best[0]:
                best = (e[0], (lo, hi))
    st = [i for i in fr if vc.acc[i] >= a_t - 1e-12]
    s_idx = min(st, key=lambda i: E[i]) if st else int8_idx
    out = {"target_val": float(a_t), "static_cfg": [list(o) for o in vc.configs[s_idx]],
           "static_E": float(E[s_idx]), "static_test_acc": float(tc.acc[s_idx])}
    # time-sharing mixture chosen on validation
    hb = (np.inf, None)
    for i in fr:
        for j in fr:
            if vc.acc[i] < a_t <= vc.acc[j]:
                w = (a_t - vc.acc[i]) / (vc.acc[j] - vc.acc[i])
                e = E[i] + w * (E[j] - E[i])
                if e < hb[0]:
                    hb = (e, (i, j, w))
    if hb[1] is not None and hb[0] < E[s_idx]:
        i, j, w = hb[1]
        out.update({"mix_E": float(hb[0]), "mix_test_acc": float(tc.acc[i] + w * (tc.acc[j] - tc.acc[i])),
                    "mix_cfgs": [[list(o) for o in vc.configs[i]], [list(o) for o in vc.configs[j]]], "mix_w": float(w)})
    else:
        out.update({"mix_E": float(E[s_idx]), "mix_test_acc": float(tc.acc[s_idx])})
    if best[1] is not None:
        lo, hi = best[1]
        pesc, cacc = pair_curve(vc, lo, hi)
        k = int(np.searchsorted(np.maximum.accumulate(cacc), a_t - 1e-12, side="left"))
        o = vc.order(lo)
        tau = float(vc.msp[lo][o][k]) if k < vc.N else 1.0001
        esc = tc.msp[lo] < tau
        out.update({"cas_lo": [list(x) for x in vc.configs[lo]], "cas_hi": [list(x) for x in vc.configs[hi]],
                    "cas_E_val": float(best[0]), "rho": float(E[lo] / E[hi]),
                    "cas_test_acc": float(np.where(esc, tc.correct[hi], tc.correct[lo]).mean()),
                    "cas_E": float(E[lo] + esc.mean() * E[hi]), "cas_pesc": float(esc.mean()), "tau": tau,
                    "E_lo": float(E[lo]), "E_hi": float(E[hi])})
    return out
