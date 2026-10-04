"""Unit tests for the PACE code base.

Run with  `python -m pytest tests/`  or simply  `python tests/test_pace.py`.
The engine tests need JAX (used only to train the deeper models)."""
import os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import axlib as A
import r2lib as L


def test_energy_model_calibration():
    assert abs(A.e_mac(8, 0) - 0.3) < 1e-12           # 0.2 pJ multiply + 0.1 pJ accumulate
    assert abs(A.e_mac(4, 1) - 0.128125) < 1e-9       # effective width 3
    assert A.e_mac(6, 2) == A.e_mac(4, 0)             # same effective width, same compute energy
    EM = L.load_energy_models()
    for k in ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]:
        assert 0.2 < L.emac(8, 0, EM[k]) - EM[k].get("mu", 0) * 8 < 0.32, k
    fam = L.family_model("quad", 1.0 / 3.0, 0.3)       # the family contains the paper's model
    for b, t in L.OPTIONS_EXT:
        assert abs(L.emac(b, t, fam) - A.e_mac(b, t)) < 1e-9


def test_integer_mac_is_exact():
    rng = np.random.default_rng(0)
    m = A.NumpyMLP([20, 16, 5], seed=1)
    X = rng.random((50, 20)).astype(np.float32)
    am, wm = A.calibrate(m, X)
    for b, t in [(8, 0), (6, 2), (4, 1), (2, 0)]:
        aq, sa = A._q_unsigned(X, am[0], b, t)
        wq, sw = A._q_signed_w(m.W[0], wm[0], b, t)
        exact = aq.astype(np.int64) @ wq.astype(np.int64)            # pure integer arithmetic
        flt = aq.astype(np.float64) @ wq.astype(np.float64)          # what the simulator does
        assert np.array_equal(exact, flt.astype(np.int64))


def test_cascade_curve_matches_brute_force():
    rng = np.random.default_rng(1)
    N, C = 300, 4
    y = rng.integers(0, 3, N)
    pred = rng.integers(0, 3, (C, N)); msp = rng.random((C, N)).astype(np.float32)
    cache = L.Cache([[[8, 0]]] * C, pred, msp, y)
    pesc, acc = L.pair_curve(cache, 1, 3)
    o = np.argsort(msp[1], kind="stable")
    for k in [0, 1, 17, 150, N]:
        esc = np.zeros(N, bool); esc[o[:k]] = True
        brute = np.mean(np.where(esc, pred[3] == y, pred[1] == y))
        assert abs(brute - acc[k]) < 1e-12 and abs(pesc[k] - k / N) < 1e-12


def test_static_energy_at():
    E = np.array([5.0, 3.0, 9.0, 4.0]); acc = np.array([0.90, 0.80, 0.95, 0.91])
    r = L.static_energy_at(E, acc, [0.79, 0.85, 0.905, 0.95, 0.99])
    assert list(r[:4]) == [3.0, 4.0, 4.0, 9.0] and np.isinf(r[4])
    assert L.pareto_idx(E, acc) == [1, 3, 2]


def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def test_resnet_engine_matches_jax(tmp_path=None):
    if not _jax_ok():
        return
    import jax.numpy as jnp
    import deepnets as D, deepquant as Q
    rng = np.random.default_rng(3)
    p, s = D.resnet8_init(0)
    for k in s:                                     # non-trivial BN statistics
        s[k] = (rng.random(s[k].shape) + (0.5 if k.endswith("_v") else -0.5)).astype(np.float32)
    path = os.path.join(L.CACHE, "_test_resnet.npz"); np.savez(path, **p, **s)
    X = rng.random((6, 32, 32, 3)).astype(np.float32)
    ref = np.asarray(D.resnet8_apply({k: jnp.asarray(v) for k, v in p.items()},
                                     {k: jnp.asarray(v) for k, v in s.items()}, jnp.asarray(X), False)[0])
    eng = Q.ResNet8Q(path, X)
    assert np.max(np.abs(eng.forward(X, None) - ref)) < 1e-3
    q8 = eng.forward(X, [(8, 0)] * 5)
    assert np.mean(q8.argmax(1) == ref.argmax(1)) >= 0.8
    os.remove(path)


def test_vit_engine_matches_jax():
    if not _jax_ok():
        return
    import jax.numpy as jnp
    import deepnets as D, deepquant as Q
    cfg = D.load_config("vit4_fmnist.json")
    p = D.vit_init(0, cfg)
    path = os.path.join(L.CACHE, "_test_vit.npz"); np.savez(path, **p)
    X = np.random.default_rng(4).random((7, 28, 28, 1)).astype(np.float32)
    ref = np.asarray(D.vit_apply({k: jnp.asarray(v) for k, v in p.items()}, jnp.asarray(X), cfg))
    eng = Q.ViT4Q(path, cfg, X)
    assert np.max(np.abs(eng.forward(X, None) - ref)) < 1e-4
    os.remove(path)


def test_enumeration_matches_direct_forward():
    if not _jax_ok():
        return
    import deepnets as D, deepquant as Q
    cfg = D.load_config("vit4_fmnist.json")
    p = D.vit_init(1, cfg)
    path = os.path.join(L.CACHE, "_test_vit2.npz"); np.savez(path, **p)
    X = np.random.default_rng(5).random((9, 28, 28, 1)).astype(np.float32)
    eng = Q.ViT4Q(path, cfg, X)
    opts = [(8, 0), (4, 0)]
    configs, pred, msp = Q.enumerate_all(eng, X, opts)
    for j in [0, 5, 10, 15]:
        pr = L.softmax(eng.forward(X, configs[j]))
        assert np.array_equal(pr.argmax(1), pred[j]) and np.allclose(pr.max(1), msp[j], atol=1e-6)
    os.remove(path)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
