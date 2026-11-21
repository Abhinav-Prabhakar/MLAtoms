"""Wasserstein GAN — the earth-mover distance and four Lipschitz constraints.

Chapter:   14-generative
Companion: wgan.md
Tier 2 · 4 variants: wgan, wgan_gp, wgan_lp, spectral_norm
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("wgan", "wgan_gp", "wgan_lp", "spectral_norm")

DIM = 2
LATENT = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 256
HIDDEN = 64
BATCH = 128
TRAIN_STEPS = 3000
N_CRITIC = 4
LR = 1e-4
BETA1 = 0.5
BETA2 = 0.999
LEAK = 0.2
CLIP = 0.01
LAM = 10.0

CONSTRAINT = {
    "wgan": "clip",
    "wgan_gp": "two_sided",
    "wgan_lp": "one_sided",
    "spectral_norm": "spectral",
}


# --------------------------------------------------------------------------- #
# Data: eight Gaussians on a circle
# --------------------------------------------------------------------------- #
def centres() -> np.ndarray:
    """The eight mode centres, (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def make_data(seed: int, per_mode: int = PER_MODE) -> np.ndarray:
    """Return points drawn uniformly from the eight modes."""
    rng = np.random.default_rng(seed)
    flat = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, DIM))
                      for c in centres()])
    return flat[rng.permutation(len(flat))]


def mode_stats(samples: np.ndarray) -> tuple:
    """Return (modes covered, largest share, mean distance to nearest centre).

    A mode counts as covered when it wins more than 3% of the samples.
    """
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return int(np.sum(share > 0.03)), float(share.max()), float(d.min(1).mean())


# --------------------------------------------------------------------------- #
# Networks: three weight matrices each, one hidden activation
# --------------------------------------------------------------------------- #
def _leaky(x: np.ndarray, slope: float) -> np.ndarray:
    return np.where(x > 0.0, x, slope * x)


def _init(rng: np.random.Generator, sizes: list) -> list:
    """He-initialised stack."""
    layers = []
    for a, b in zip(sizes, sizes[1:], strict=False):
        layers.append({"W": rng.normal(0.0, np.sqrt(2.0 / a), (a, b)),
                       "b": np.zeros(b)})
    return layers


def _forward(params: list, x: np.ndarray, slope: float,
             out_act: str) -> tuple:
    """Run the stack; returns (output, cache) with one entry per layer."""
    h, cache = x, []
    last = len(params) - 1
    for i, layer in enumerate(params):
        pre = h @ layer["W"] + layer["b"]
        if i == last and out_act == "tanh":
            out = np.tanh(pre)
        elif i == last:
            out = pre
        else:
            out = _leaky(pre, slope)
        cache.append((h, pre, out))
        h = out
    return h, cache


def _backward(params: list, cache: list, d: np.ndarray, slope: float,
              out_act: str) -> tuple:
    """Return (parameter gradients, gradient with respect to the input)."""
    grads = [None] * len(params)
    last = len(params) - 1
    for i in range(last, -1, -1):
        h, pre, out = cache[i]
        if i == last and out_act == "tanh":
            d = d * (1.0 - out ** 2)
        elif i != last:
            d = d * np.where(pre > 0.0, 1.0, slope)
        grads[i] = {"W": h.T @ d, "b": d.sum(axis=0)}
        d = d @ params[i]["W"].T
    return grads, d


def _input_grad(params: list, x: np.ndarray) -> np.ndarray:
    """The critic's gradient with respect to its input, (n, DIM).

    The critic is piecewise linear, so the gradient is a product of the
    layer matrices with the activation masks taken at `x`.
    """
    _, cache = _forward(params, x, LEAK, "linear")
    w = params[-1]["W"][:, 0]
    d = np.broadcast_to(w, (len(x), len(w)))
    for i in range(len(params) - 2, -1, -1):
        pre = cache[i][1]
        d = (d * np.where(pre > 0.0, 1.0, LEAK)) @ params[i]["W"].T
    return d


def _penalty(params: list, xh: np.ndarray,
             one_sided: bool) -> tuple:
    """Return (mean gradient norm, parameter gradients of the penalty).

    The penalty is the mean over the batch of (||grad D|| - 1)^2, or of
    max(0, ||grad D|| - 1)^2. Its parameter gradient is a double backward
    pass written out by hand for the critic's three weight matrices.
    """
    _, cache = _forward(params, xh, LEAK, "linear")
    m1 = np.where(cache[0][1] > 0.0, 1.0, LEAK)
    m2 = np.where(cache[1][1] > 0.0, 1.0, LEAK)
    w1, w2, w3 = params[0]["W"], params[1]["W"], params[2]["W"][:, 0]
    v = m2 * w3
    t = m1 * (v @ w2.T)
    g = t @ w1.T
    n = np.linalg.norm(g, axis=1)
    coef = 2.0 * (n - 1.0)
    if one_sided:
        coef = np.where(n > 1.0, coef, 0.0)
    coef = coef / len(xh)
    ghat = g / np.maximum(n, 1e-12)[:, None]
    gw1 = np.einsum("n,nd,nh->dh", coef, ghat, t)
    w1g = ghat @ w1
    gw2 = np.einsum("n,nh,nk->hk", coef, w1g * m1, v)
    p = (m1 * w1g) @ w2
    gw3 = np.einsum("n,nh->h", coef, m2 * p)
    return float(np.mean(n)), [
        {"W": gw1, "b": np.zeros_like(params[0]["b"])},
        {"W": gw2, "b": np.zeros_like(params[1]["b"])},
        {"W": gw3[:, None], "b": np.zeros_like(params[2]["b"])},
    ]


def _spectral(params: list, us: list) -> None:
    """Divide each weight matrix by its top singular value, in place.

    One power iteration per call, with the left vector carried over from the
    previous call; after the division the ratio is one exactly, because the
    vector it was measured with is unchanged.
    """
    for layer, u in zip(params, us, strict=True):
        w = layer["W"]
        v = w.T @ u
        v /= np.linalg.norm(v) + 1e-12
        u[...] = w @ v
        u /= np.linalg.norm(u) + 1e-12
        layer["W"] = w / (np.linalg.norm(w @ v) + 1e-12)


class _Adam:
    """Adam with a separately tunable first-moment decay (0.5 here)."""

    def __init__(self, params: list, lr: float = LR) -> None:
        self.lr = lr
        self.t = 0
        self.m = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]
        self.v = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        c1 = 1.0 - BETA1 ** self.t
        c2 = 1.0 - BETA2 ** self.t
        for L, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            for key in L:
                m[key] = BETA1 * m[key] + (1.0 - BETA1) * g[key]
                v[key] = BETA2 * v[key] + (1.0 - BETA2) * g[key] ** 2
                L[key] -= self.lr * (m[key] / c1) / (
                    np.sqrt(v[key] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class WGAN:
    """A Wasserstein GAN with one of four Lipschitz constraints.

    The critic is not a classifier: it is trained to maximise
    E[D(real)] - E[D(fake)] under a constraint on its slope, and the generator
    to maximise E[D(fake)] against it. The constraint is weight clipping, a
    two-sided or one-sided gradient penalty, or spectral normalisation.
    """

    def __init__(self, variant: str = "wgan_gp", *, hidden: int = HIDDEN,
                 train_steps: int = TRAIN_STEPS, batch: int = BATCH,
                 n_critic: int = N_CRITIC, lr: float = LR,
                 lam: float = LAM, clip: float = CLIP, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if hidden < 1:
            raise ValueError("hidden must be at least one")
        if train_steps < 1 or batch < 1 or n_critic < 1:
            raise ValueError("train_steps, batch and n_critic must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if not np.isfinite(lam) or lam < 0.0:
            raise ValueError("lam must be finite and non-negative")
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip must be positive and finite")
        self.variant = variant
        self.constraint = CONSTRAINT[variant]
        self.hidden = hidden
        self.train_steps = train_steps
        self.batch = batch
        self.n_critic = n_critic
        self.lr = lr
        self.lam = lam
        self.clip = clip
        self.seed = seed
        self.critic_params: list | None = None
        self.gen_params: list | None = None

    def n_params(self) -> tuple:
        """(critic parameters, generator parameters)."""
        if self.critic_params is None:
            raise RuntimeError("call fit first")
        return (sum(L["W"].size + L["b"].size for L in self.critic_params),
                sum(L["W"].size + L["b"].size for L in self.gen_params))

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray) -> WGAN:
        x = _check_data(x)
        rng = np.random.default_rng(self.seed)
        d_params = _init(rng, [DIM, self.hidden, self.hidden, 1])
        g_params = _init(rng, [LATENT, self.hidden, self.hidden, DIM])
        opt_d = _Adam(d_params, self.lr)
        opt_g = _Adam(g_params, self.lr)
        us = [rng.normal(size=(L["W"].shape[0], 1)) for L in d_params]
        for _ in range(self.train_steps):
            for _ in range(self.n_critic):
                xr = x[rng.integers(0, len(x), self.batch)]
                z = rng.normal(0.0, 1.0, (self.batch, LATENT))
                fake, _ = _forward(g_params, z, 0.0, "tanh")
                _, cr = _forward(d_params, xr, LEAK, "linear")
                _, cf = _forward(d_params, fake, LEAK, "linear")
                gr, _ = _backward(d_params, cr,
                                  np.full((self.batch, 1), -1.0 / self.batch),
                                  LEAK, "linear")
                gf, _ = _backward(d_params, cf,
                                  np.full((self.batch, 1), 1.0 / self.batch),
                                  LEAK, "linear")
                grads = [{key: gr[i][key] + gf[i][key] for key in gr[i]}
                         for i in range(len(d_params))]
                if self.constraint in ("two_sided", "one_sided"):
                    eps = rng.random((self.batch, 1))
                    xh = eps * xr + (1.0 - eps) * fake
                    _, pg = _penalty(d_params, xh,
                                     self.constraint == "one_sided")
                    grads = [{key: grads[i][key] + self.lam * pg[i][key]
                              for key in grads[i]}
                             for i in range(len(d_params))]
                opt_d.step(d_params, grads)
                if self.constraint == "clip":
                    for L in d_params:
                        for key in L:
                            np.clip(L[key], -self.clip, self.clip, out=L[key])
                elif self.constraint == "spectral":
                    _spectral(d_params, us)
            z = rng.normal(0.0, 1.0, (self.batch, LATENT))
            fake, cg = _forward(g_params, z, 0.0, "tanh")
            _, cf = _forward(d_params, fake, LEAK, "linear")
            _, dx = _backward(d_params, cf,
                              np.full((self.batch, 1), -1.0 / self.batch),
                              LEAK, "linear")
            gg, _ = _backward(g_params, cg, dx, 0.0, "tanh")
            opt_g.step(g_params, gg)
        self.critic_params = d_params
        self.gen_params = g_params
        return self

    # -- evaluation ----------------------------------------------------------- #
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n points from the generator."""
        if self.gen_params is None:
            raise RuntimeError("call fit first")
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 1.0, (n, LATENT))
        return _forward(self.gen_params, z, 0.0, "tanh")[0]

    def critic(self, x: np.ndarray) -> np.ndarray:
        """The critic's value at each row of x."""
        if self.critic_params is None:
            raise RuntimeError("call fit first")
        x = _check_data(x)
        return _forward(self.critic_params, x, LEAK, "linear")[0][:, 0]

    def critic_grad(self, x: np.ndarray) -> np.ndarray:
        """The critic's input gradient at each row of x, (n, DIM)."""
        if self.critic_params is None:
            raise RuntimeError("call fit first")
        return _input_grad(self.critic_params, _check_data(x))

    def grad_norm(self, x: np.ndarray) -> float:
        """Mean ||grad D|| over the rows of x."""
        return float(np.linalg.norm(self.critic_grad(x), axis=1).mean())

    def wasserstein(self, x: np.ndarray, n: int = 1024,
                    seed: int = 0) -> float:
        """E[D(real)] - E[D(fake)] on a fresh sample."""
        fake = self.sample(n, seed=seed)
        return float(self.critic(np.asarray(x, dtype=float)).mean()
                     - self.critic(fake).mean())


def _check_data(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"x must be (n, {DIM})")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _grad_check(model: WGAN, x: np.ndarray, h: float = 1e-6) -> float:
    """Max absolute difference between the analytic and numeric input gradient."""
    analytic = model.critic_grad(x)
    numeric = np.zeros_like(analytic)
    for i in range(x.shape[1]):
        step = np.zeros_like(x)
        step[:, i] = h
        numeric[:, i] = (model.critic(x + step) - model.critic(x - step)) / (
            2.0 * h)
    return float(np.max(np.abs(analytic - numeric)))


def demo() -> None:
    x = make_data(0)
    test = make_data(1)
    print(f"wgan. eight Gaussians on a circle of radius {RADIUS}, "
          f"{len(x)} points, spread {SPREAD}:")
    print(f"  generator {LATENT} -> {HIDDEN} -> {HIDDEN} -> {DIM} (ReLU, "
          f"tanh), critic {DIM} -> {HIDDEN} -> {HIDDEN} -> 1 (leaky)")
    print(f"  {TRAIN_STEPS} steps, {N_CRITIC} critic passes each, batch "
          f"{BATCH}, Adam lr {LR}, beta1 {BETA1}")

    runs = {}
    print("wgan. the four Lipschitz constraints:")
    print(f"  {'variant':<15s} {'constraint':<13s} {'modes':>5s} "
          f"{'share':>6s} {'dist':>6s} {'W':>7s} {'|g|':>6s} {'D/G':>9s}")
    for name in VARIANTS:
        model = WGAN(name).fit(x)
        runs[name] = model
        many, share, dist = mode_stats(model.sample(1024, seed=1))
        gap = model.wasserstein(test, seed=2)
        d_params, g_params = model.n_params()
        print(f"  {name:<15s} {model.constraint:<13s} {many:>3d}/{MODES} "
              f"{share:6.2f} {dist:6.3f} {gap:+7.3f} "
              f"{model.grad_norm(test):6.3f} {d_params}/{g_params}")

    print("wgan. the critic at a mode centre and halfway between two modes:")
    print(f"  {'variant':<15s} {'D centre':>9s} {'D mid':>7s} "
          f"{'|g| centre':>10s} {'|g| mid':>8s}")
    mids = 0.5 * (centres() + np.roll(centres(), 1, axis=0))
    profile = {}
    for name in VARIANTS:
        model = runs[name]
        dc, dm = float(model.critic(centres()).mean()), float(
            model.critic(mids).mean())
        gc, gm = model.grad_norm(centres()), model.grad_norm(mids)
        profile[name] = (dc, dm, gc, gm)
        print(f"  {name:<15s} {dc:+9.2f} {dm:+7.2f} {gc:10.3f} {gm:8.3f}")

    print("wgan. the same two-sided penalty at lambda=1:")
    relaxed = WGAN("wgan_gp", lam=1.0).fit(x)
    many, share, dist = mode_stats(relaxed.sample(1024, seed=1))
    print(f"  modes {many}/{MODES}, share {share:.2f}, distance {dist:.3f}, "
          f"|g| centre {relaxed.grad_norm(centres()):.3f}, "
          f"mid {relaxed.grad_norm(mids):.3f}")

    print("wgan. the analytic input gradient against central differences:")
    error = _grad_check(runs["wgan_lp"], test[:16])
    print(f"  max absolute error {error:.1e} on 16 points")

    print("wgan. the spectral norm's top singular value per layer:")
    sigma = [float(np.linalg.svd(L["W"], compute_uv=False)[0])
             for L in runs["spectral_norm"].critic_params]
    print("  " + "  ".join(f"{s:.9f}" for s in sigma))

    print("wgan. malformed calls are refused:")
    bad = [
        lambda: WGAN("wgan_gp_v2"),
        lambda: WGAN("wgan", hidden=0),
        lambda: WGAN("wgan", train_steps=0),
        lambda: WGAN("wgan", n_critic=0),
        lambda: WGAN("wgan", lr=-1.0),
        lambda: WGAN("wgan", lam=-1.0),
        lambda: WGAN("wgan", clip=0.0),
        lambda: WGAN("wgan").sample(8),
        lambda: runs["wgan"].sample(0),
        lambda: runs["wgan"].critic(np.zeros((8, 3))),
        lambda: runs["wgan"].fit(np.full((8, DIM), np.nan)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    print("wgan. what the table says:")
    assert all(mode_stats(runs[n].sample(1024, 1))[0] == MODES
               for n in ("wgan", "wgan_lp", "spectral_norm"))
    assert mode_stats(runs["wgan_gp"].sample(1024, 1))[0] < MODES
    assert mode_stats(relaxed.sample(1024, 1))[0] == MODES
    assert 0.95 < profile["wgan_gp"][2] < 1.10
    assert profile["wgan"][2] < 0.01
    assert profile["wgan_lp"][2] < profile["wgan_lp"][3]
    assert profile["wgan_lp"][0] > profile["wgan_lp"][1]
    assert abs(profile["wgan_gp"][0] - profile["wgan_gp"][1]) < 0.1
    assert error < 1e-6
    assert all(abs(s - 1.0) < 1e-5 for s in sigma)
    assert np.all(np.abs(runs["wgan"].sample(64, 7)) <= 1.0)
    print("  clipping leaves a dead critic (|g| ~ 0); the two-sided penalty")
    print("  forces unit slope at the peaks and cannot separate five modes;")
    print("  the one-sided penalty keeps flat plateaus on the modes and")
    print("  slopes between them, and the relaxed two-sided weight recovers.")
    print("wgan demo passed")


if __name__ == "__main__":
    demo()
