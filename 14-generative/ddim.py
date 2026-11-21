"""DDIM — deterministic sampling, the eta knob, inversion and few steps.

Chapter:   14-generative
Companion: ddim.md
Tier 2 · 4 variants: ddim, eta_sampling, ddim_inversion, few_step
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("ddim", "eta_sampling", "ddim_inversion", "few_step")

DIM = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 256
HIDDEN = 64
BATCH = 256
TRAIN_STEPS = 3000
LR = 2e-3
TIME_FEATURES = 4
STEPS = 200
COSINE_S = 0.008

CONFIG = {
    "ddim": {"steps": 50, "eta": 0.0},
    "eta_sampling": {"steps": 50, "eta": 1.0},
    "ddim_inversion": {"steps": 200, "eta": 0.0},
    "few_step": {"steps": 10, "eta": 0.0},
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
    """Return (modes covered, largest share, mean distance to nearest centre)."""
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres()[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return int(np.sum(share > 0.03)), float(share.max()), float(d.min(1).mean())


# --------------------------------------------------------------------------- #
# Schedule and network
# --------------------------------------------------------------------------- #
def make_schedule(steps: int = STEPS) -> tuple:
    """Cosine schedule; `abar` has a leading one so abar[t] is the t-th rung."""
    t = np.arange(steps + 1) / steps
    f = np.cos((t + COSINE_S) / (1.0 + COSINE_S) * np.pi / 2.0) ** 2
    abar = f / f[0]
    beta = np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)
    alpha = 1.0 - beta
    return beta, alpha, np.concatenate([[1.0], np.cumprod(alpha)])


def _sig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _time_features(t: np.ndarray, steps: int) -> np.ndarray:
    """Four features: sin and cos at one and two cycles over the schedule."""
    phase = (np.asarray(t, dtype=float) / steps) * np.pi
    return np.stack([np.sin(phase), np.cos(phase),
                     np.sin(2.0 * phase), np.cos(2.0 * phase)], axis=1)


def _init_net(rng: np.random.Generator, sizes: list) -> list:
    layers = []
    for a, b in zip(sizes, sizes[1:], strict=False):
        layers.append({"W": rng.normal(0.0, np.sqrt(2.0 / a), (a, b)),
                       "b": np.zeros(b)})
    return layers


def _forward(params: list, x: np.ndarray, t: np.ndarray,
             steps: int) -> tuple:
    h = np.concatenate([x, _time_features(t, steps)], axis=1)
    cache, last = [], len(params) - 1
    for i, layer in enumerate(params):
        pre = h @ layer["W"] + layer["b"]
        out = pre if i == last else pre * _sig(pre)
        cache.append((h, pre, out))
        h = out
    return h, cache


def _backward(params: list, cache: list, dout: np.ndarray) -> list:
    grads = [None] * len(params)
    d, last = dout, len(params) - 1
    for i in range(last, -1, -1):
        h, pre, out = cache[i]
        if i != last:
            s = _sig(pre)
            d = d * (s + pre * s * (1.0 - s))
        grads[i] = {"W": h.T @ d, "b": d.sum(axis=0)}
        d = d @ params[i]["W"].T
    return grads


class _Adam:
    def __init__(self, params: list, lr: float = LR) -> None:
        self.lr, self.t = lr, 0
        self.m = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]
        self.v = [{k: np.zeros_like(v) for k, v in L.items()} for L in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        c1 = 1.0 - 0.9 ** self.t
        c2 = 1.0 - 0.999 ** self.t
        for L, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            for key in L:
                m[key] = 0.9 * m[key] + 0.1 * g[key]
                v[key] = 0.999 * v[key] + 0.001 * g[key] ** 2
                L[key] -= self.lr * (m[key] / c1) / (
                    np.sqrt(v[key] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #
class DDIM:
    """A v-prediction denoiser read with the DDIM update.

    `sample` walks the trajectory from noise to data with a step count and a
    stochasticity `eta`; `invert` walks it the other way, which is only
    possible at eta = 0. The model is trained as in `diffusion.py`, with the
    velocity target; that choice is what makes ten steps work.
    """

    def __init__(self, variant: str = "ddim", *, hidden: int = HIDDEN,
                 train_steps: int = TRAIN_STEPS, batch: int = BATCH,
                 lr: float = LR, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if hidden < 1:
            raise ValueError("hidden must be at least one")
        if train_steps < 1 or batch < 1:
            raise ValueError("train_steps and batch must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.steps = CONFIG[variant]["steps"]
        self.eta = CONFIG[variant]["eta"]
        self.hidden = hidden
        self.train_steps = train_steps
        self.batch = batch
        self.lr = lr
        self.seed = seed
        self.beta, self.alpha, self.abar = make_schedule(STEPS)
        self.params: list | None = None

    def n_params(self) -> int:
        if self.params is None:
            raise RuntimeError("call fit first")
        return sum(L["W"].size + L["b"].size for L in self.params)

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray) -> DDIM:
        x = _check_data(x)
        rng = np.random.default_rng(self.seed)
        params = _init_net(rng, [DIM + TIME_FEATURES, self.hidden,
                                 self.hidden, DIM])
        opt = _Adam(params, self.lr)
        for _ in range(self.train_steps):
            x0 = x[rng.integers(0, len(x), self.batch)]
            t = rng.integers(1, STEPS + 1, self.batch)
            eps = rng.normal(0.0, 1.0, x0.shape)
            abar = self.abar[t][:, None]
            xt = np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps
            target = np.sqrt(abar) * eps - np.sqrt(1.0 - abar) * x0
            out, cache = _forward(params, xt, t, STEPS)
            dout = 2.0 * (out - target) / (len(x0) * x0.shape[1])
            opt.step(params, _backward(params, cache, dout))
        self.params = params
        return self

    # -- the trajectory ------------------------------------------------------- #
    def _times(self, steps: int) -> np.ndarray:
        """The descending rungs visited, from STEPS to zero inclusive."""
        if steps < 1 or steps > STEPS:
            raise ValueError(f"steps must lie in [1, {STEPS}]")
        grid = np.rint(np.linspace(0.0, STEPS, steps + 1)).astype(int)
        return grid[::-1]

    def _predict(self, x: np.ndarray, t: int) -> np.ndarray:
        out, _ = _forward(self.params, x, np.full(len(x), t), STEPS)
        return out

    def _split(self, x: np.ndarray, t: int) -> tuple:
        """The velocity prediction turned into (x_hat_0, eps_hat)."""
        v = self._predict(x, t)
        abar = self.abar[t]
        return (np.sqrt(abar) * x - np.sqrt(1.0 - abar) * v,
                np.sqrt(abar) * v + np.sqrt(1.0 - abar) * x)

    def _sigma(self, t: int, s: int, eta: float) -> float:
        at, as_ = self.abar[t], self.abar[s]
        if eta == 0.0 or s == 0:
            return 0.0
        return float(eta * np.sqrt((1.0 - as_) / (1.0 - at)
                                   * (1.0 - at / as_)))

    def sample(self, n: int = 1024, *, steps: int | None = None,
               eta: float | None = None, seed: int = 0,
               start: np.ndarray | None = None) -> np.ndarray:
        """Run the reverse trajectory from noise (or from a supplied `start`)."""
        if self.params is None:
            raise RuntimeError("call fit first")
        if n < 1:
            raise ValueError("n must be positive")
        steps = self.steps if steps is None else steps
        eta = self.eta if eta is None else eta
        if eta < 0.0 or not np.isfinite(eta):
            raise ValueError("eta must be finite and non-negative")
        times = self._times(steps)
        rng = np.random.default_rng(seed)
        x = (rng.normal(0.0, 1.0, (n, DIM)) if start is None
             else np.array(start, dtype=float, copy=True))
        for t, s in zip(times[:-1], times[1:], strict=True):
            x0, eps = self._split(x, t)
            sig = self._sigma(t, s, eta)
            x = (np.sqrt(self.abar[s]) * x0
                 + np.sqrt(max(1.0 - self.abar[s] - sig ** 2, 0.0)) * eps)
            if sig > 0.0:
                x = x + sig * rng.normal(0.0, 1.0, x.shape)
        return x

    def invert(self, x: np.ndarray, *, steps: int | None = None) -> np.ndarray:
        """Map data to its latent by running the eta = 0 trajectory backwards."""
        if self.params is None:
            raise RuntimeError("call fit first")
        if self.eta != 0.0:
            raise ValueError("inversion needs eta = 0")
        x = _check_data(x)
        times = self._times(self.steps if steps is None else steps)
        z = np.array(x, dtype=float, copy=True)
        for i in range(len(times) - 1, 0, -1):
            s, t = times[i], times[i - 1]
            x0, eps = self._split(z, s)
            z = (np.sqrt(self.abar[t]) * x0
                 + np.sqrt(1.0 - self.abar[t]) * eps)
        return z

    def round_trip(self, x: np.ndarray, *, steps: int | None = None) -> tuple:
        """Return (reconstruction, mean error, max error) for data points."""
        x = _check_data(x)
        z = self.invert(x, steps=steps)
        recon = self.sample(len(z), steps=steps, eta=0.0, start=z)
        err = np.linalg.norm(recon - x, axis=1)
        return recon, float(err.mean()), float(err.max())


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
def demo() -> None:
    x, test = make_data(0), make_data(1)
    print(f"ddim. eight Gaussians on a circle of radius {RADIUS}, "
          f"{len(x)} points, spread {SPREAD}:")
    print(f"  v-prediction, cosine schedule, {STEPS} training rungs, "
          f"{HIDDEN}-wide SiLU net, {TRAIN_STEPS} steps")
    model = DDIM().fit(x)
    print(f"  {model.n_params()} parameters in the denoiser")

    print("ddim. the four variants:")
    print(f"  {'variant':<16s} {'steps':>5s} {'eta':>4s} {'modes':>5s} "
          f"{'share':>6s} {'dist':>6s}")
    runs = {}
    for name in VARIANTS:
        runs[name] = model if name == "ddim" else DDIM(name).fit(x)
        m = runs[name]
        many, share, dist = mode_stats(m.sample(1024, seed=1))
        print(f"  {name:<16s} {m.steps:>5d} {m.eta:>4.1f} {many:>3d}/{MODES} "
              f"{share:6.2f} {dist:6.3f}")

    print("ddim. the trajectory at different lengths (eta = 0):")
    print(f"  {'steps':>5s} {'modes':>5s} {'share':>6s} {'dist':>6s} "
          f"{'mean shift':>17s}")
    ladder = {}
    base = runs["ddim_inversion"].sample(1024, steps=STEPS, seed=1)
    for k in (STEPS, 50, 20, 10, 5):
        s = runs["ddim_inversion"].sample(1024, steps=k, seed=1)
        ladder[k] = (mode_stats(s), float(np.linalg.norm(
            s.mean(axis=0) - base.mean(axis=0))))
        many, share, dist = ladder[k][0]
        print(f"  {k:>5d} {many:>3d}/{MODES} {share:6.2f} {dist:6.3f} "
              f"{ladder[k][1]:>17.4f}")

    print("ddim. the eta knob at 50 steps, from the same starting noise:")
    start = np.random.default_rng(7).normal(0.0, 1.0, (1024, DIM))
    reference = runs["ddim"].sample(1024, steps=50, eta=0.0, start=start)
    print(f"  {'eta':>4s} {'modes':>5s} {'dist':>6s} "
          f"{'mean |sample - eta=0|':>22s}")
    for eta in (0.0, 0.5, 1.0):
        s = runs["ddim"].sample(1024, steps=50, eta=eta, seed=0, start=start)
        many, share, dist = mode_stats(s)
        gap = float(np.linalg.norm(s - reference, axis=1).mean())
        print(f"  {eta:>4.1f} {many:>3d}/{MODES} {dist:6.3f} {gap:>22.4f}")

    print("ddim. inversion and back, at three resolutions:")
    print(f"  {'steps':>5s} {'mean err':>9s} {'max err':>8s} "
          f"{'latent std':>10s} {'latent mean':>11s}")
    inversions = {}
    inv = runs["ddim_inversion"]
    for k in (STEPS, 50, 20):
        recon, mean, worst = inv.round_trip(test[:256], steps=k)
        z = inv.invert(test[:256], steps=k)
        inversions[k] = (mean, worst, float(z.std()), float(z.mean()))
        print(f"  {k:>5d} {mean:9.4f} {worst:8.4f} {z.std():10.3f} "
              f"{z.mean():+11.3f}")

    print("ddim. malformed calls are refused:")
    bad = [
        lambda: DDIM("ddim_v2"),
        lambda: DDIM("ddim", hidden=0),
        lambda: DDIM("ddim", train_steps=0),
        lambda: DDIM("ddim", lr=0.0),
        lambda: DDIM("ddim").sample(8),
        lambda: model.sample(0),
        lambda: model.sample(8, steps=0),
        lambda: model.sample(8, steps=STEPS + 1),
        lambda: model.sample(8, eta=-1.0),
        lambda: runs["eta_sampling"].invert(test[:8]),
        lambda: model.fit(np.zeros((8, 3))),
        lambda: model.invert(np.full((8, DIM), np.nan)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    print("ddim. what the table says:")
    assert all(mode_stats(runs[n].sample(1024, seed=1))[0] == MODES
               for n in VARIANTS)
    assert runs["few_step"].steps == 10
    assert ladder[10][0][2] < 0.25
    assert abs(ladder[10][0][2] - ladder[STEPS][0][2]) < 0.05
    assert ladder[5][0][2] > ladder[10][0][2]
    assert ladder[10][1] > ladder[50][1]
    assert all(np.array_equal(a["W"], b["W"]) and np.array_equal(a["b"], b["b"])
               for a, b in zip(runs["ddim"].params, runs["few_step"].params,
                               strict=True))
    first = runs["ddim"].sample(128, seed=3)
    second = runs["ddim"].sample(128, seed=3)
    assert np.array_equal(first, second)
    stamp = runs["ddim"].sample(128, steps=50, eta=1.0, seed=0, start=start[:128])
    still = runs["ddim"].sample(128, steps=50, eta=0.0, seed=0, start=start[:128])
    assert not np.allclose(stamp, still)
    assert inversions[STEPS][0] < 0.01
    assert inversions[50][0] > inversions[STEPS][0]
    assert inversions[20][0] > inversions[50][0]
    assert inversions[STEPS][1] < 0.02
    assert inversions[STEPS][2] > 0.9
    print("  ten steps match two hundred; eta changes the sample without")
    print("  changing coverage; inversion is accurate at full resolution and")
    print("  loses a little at every longer stride.")
    print("ddim demo passed")


if __name__ == "__main__":
    demo()
