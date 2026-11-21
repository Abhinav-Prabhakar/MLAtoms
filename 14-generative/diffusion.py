"""Diffusion models — the forward noising process and its learned reverse.

Chapter:   14-generative
Companion: diffusion.md
Tier 1 · 6 variants: ddpm, linear_schedule, cosine_schedule,
epsilon_prediction, v_prediction, latent_diffusion
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("ddpm", "linear_schedule", "cosine_schedule",
            "epsilon_prediction", "v_prediction", "latent_diffusion")

DIM = 8
LATENT = 2
MODES = 8
RADIUS = 0.8
SPREAD = 0.07
PER_MODE = 100
BASIS_SEED = 7
HID = 64
BATCH = 256
TRAIN_STEPS = 3000
LR = 2e-3
TIME_FEATURES = 4
LINEAR_LO = 1e-4
LINEAR_HI = 2e-2
SHORT_LO = 1e-3
SHORT_HI = 5e-2
COSINE_S = 0.008

CONFIG = {
    "ddpm": {"steps": 1000, "schedule": "linear", "target": "eps",
             "space": "data"},
    "linear_schedule": {"steps": 200, "schedule": "linear", "target": "eps",
                        "space": "data"},
    "cosine_schedule": {"steps": 200, "schedule": "cosine", "target": "eps",
                        "space": "data"},
    "epsilon_prediction": {"steps": 100, "schedule": "cosine", "target": "eps",
                           "space": "data"},
    "v_prediction": {"steps": 100, "schedule": "cosine", "target": "v",
                     "space": "data"},
    "latent_diffusion": {"steps": 100, "schedule": "cosine", "target": "v",
                         "space": "latent"},
}


# --------------------------------------------------------------------------- #
# Data: eight Gaussians on a circle, isometrically embedded in eight dimensions
# --------------------------------------------------------------------------- #
def _basis() -> np.ndarray:
    """The (2, 8) matrix whose rows span the data's plane."""
    rng = np.random.default_rng(BASIS_SEED)
    q, _ = np.linalg.qr(rng.normal(size=(DIM, LATENT)))
    return q.T


BASIS = _basis()


def centres() -> np.ndarray:
    """The eight mode centres in two dimensions, (8, 2)."""
    angle = 2.0 * np.pi * np.arange(MODES) / MODES
    return RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)


def encode(x: np.ndarray) -> np.ndarray:
    """Project ambient points onto the two-dimensional data plane."""
    return np.asarray(x, dtype=float) @ BASIS.T


def decode(z: np.ndarray) -> np.ndarray:
    """Lift latent codes back into the ambient space, exactly."""
    return np.asarray(z, dtype=float) @ BASIS


def make_data(seed: int, per_mode: int = PER_MODE) -> tuple:
    """Return (x, y): ambient points and their mode labels."""
    rng = np.random.default_rng(seed)
    flat = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, 2))
                      for c in centres()])
    y = np.repeat(np.arange(MODES), per_mode)
    order = rng.permutation(len(y))
    return decode(flat)[order], y[order]


def mode_stats(samples: np.ndarray) -> tuple:
    """Return (modes covered, largest share, mean distance to nearest centre).

    A mode counts as covered when it wins more than 3% of the samples.
    """
    centres_ambient = decode(centres())
    s = np.asarray(samples, dtype=float)
    d = np.linalg.norm(s[:, None, :] - centres_ambient[None], axis=2)
    share = np.bincount(d.argmin(axis=1), minlength=MODES) / len(s)
    return int(np.sum(share > 0.03)), float(share.max()), float(d.min(1).mean())


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def make_schedule(steps: int, kind: str) -> tuple:
    """Return (beta, alpha, abar), each of length `steps`.

    `linear` interpolates the per-step variance from LINEAR_LO to LINEAR_HI;
    `cosine` places the cumulative signal on a cosine curve (Nichol & Dhariwal
    2021), which keeps more signal at small t and destroys it faster at the end.
    """
    if kind == "linear":
        lo, hi = (LINEAR_LO, LINEAR_HI) if steps >= 500 else (SHORT_LO, SHORT_HI)
        beta = np.linspace(lo, hi, steps)
    elif kind == "cosine":
        t = np.arange(steps + 1) / steps
        f = np.cos((t + COSINE_S) / (1.0 + COSINE_S) * np.pi / 2.0) ** 2
        abar = f / f[0]
        beta = np.clip(1.0 - abar[1:] / abar[:-1], 1e-5, 0.999)
    else:
        raise ValueError(f"unknown schedule {kind!r}")
    alpha = 1.0 - beta
    return beta, alpha, np.cumprod(alpha)


# --------------------------------------------------------------------------- #
# Network: a two-hidden-layer MLP with SiLU, plus a Fourier time embedding
# --------------------------------------------------------------------------- #
def _sig(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _time_features(t: np.ndarray, steps: int) -> np.ndarray:
    """Four features: sin and cos at one and two cycles over the schedule."""
    phase = (np.asarray(t, dtype=float) / steps) * np.pi
    return np.stack([np.sin(phase), np.cos(phase),
                     np.sin(2.0 * phase), np.cos(2.0 * phase)], axis=1)


def _init_net(rng: np.random.Generator, sizes: list) -> list:
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:], strict=False)):
        last = i == len(sizes) - 2
        scale = 0.0 if last else np.sqrt(2.0 / a)
        layers.append({"W": rng.normal(0.0, scale, (a, b)), "b": np.zeros(b)})
    return layers


def _forward(params: list, x: np.ndarray, t: np.ndarray,
             steps: int) -> tuple:
    """Run the stack; returns (prediction, cache) with one entry per layer."""
    h = np.concatenate([x, _time_features(t, steps)], axis=1)
    cache = []
    last = len(params) - 1
    for i, layer in enumerate(params):
        pre = h @ layer["W"] + layer["b"]
        out = pre if i == last else pre * _sig(pre)
        cache.append((h, pre, out))
        h = out
    return h, cache


def _backward(params: list, cache: list, dout: np.ndarray) -> list:
    """Return parameter gradients; `dout` carries the 1/(n*d) of a mean."""
    grads = [None] * len(params)
    d = dout
    last = len(params) - 1
    for i in range(last, -1, -1):
        h, pre, out = cache[i]
        if i != last:
            s = _sig(pre)
            d = d * (s + pre * s * (1.0 - s))
        grads[i] = {"W": h.T @ d, "b": d.sum(axis=0)}
        if i:
            d = d @ params[i]["W"].T
    return grads


class _Adam:
    """Adam with state mirrored to the layer list."""

    def __init__(self, params: list, lr: float = LR) -> None:
        self.lr = lr
        self.t = 0
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
                L[key] -= self.lr * (m[key] / c1) / (np.sqrt(v[key] / c2) + 1e-8)


# --------------------------------------------------------------------------- #
# The two parameterisations
# --------------------------------------------------------------------------- #
def _target(x0: np.ndarray, eps: np.ndarray, abar: np.ndarray,
            kind: str) -> np.ndarray:
    """The regression target at the batch's own noise levels."""
    if kind == "eps":
        return eps
    return np.sqrt(abar) * eps - np.sqrt(1.0 - abar) * x0


def _split(out: np.ndarray, xt: np.ndarray, abar: np.ndarray,
           kind: str) -> tuple:
    """Turn a prediction into (x_hat_0, eps_hat) at one noise level."""
    if kind == "eps":
        eps = out
        return (xt - np.sqrt(1.0 - abar) * out) / np.sqrt(abar), eps
    x0 = np.sqrt(abar) * xt - np.sqrt(1.0 - abar) * out
    eps = np.sqrt(abar) * out + np.sqrt(1.0 - abar) * xt
    return x0, eps


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class Diffusion:
    """A denoising diffusion model for one of the six configurations.

    `fit` learns to predict either the noise (`eps`) or the velocity (`v`) that
    took a clean point to its noisy version. `sample` runs the ancestral
    reverse process from pure noise; `denoise_mse` measures the implied clean
    point at a chosen noise level.
    """

    def __init__(self, variant: str = "ddpm", *, hidden: int = HID,
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
        cfg = CONFIG[variant]
        self.variant = variant
        self.steps = cfg["steps"]
        self.schedule = cfg["schedule"]
        self.target = cfg["target"]
        self.latent = cfg["space"] == "latent"
        self.hidden = hidden
        self.train_steps = train_steps
        self.batch = batch
        self.lr = lr
        self.seed = seed
        self.beta, self.alpha, self.abar = make_schedule(self.steps,
                                                         self.schedule)
        self.params: list | None = None

    # -- shapes --------------------------------------------------------------- #
    @property
    def dim(self) -> int:
        """The dimension the diffusion itself runs in."""
        return LATENT if self.latent else DIM

    def n_params(self) -> int:
        """Parameter count, excluding the fixed basis."""
        return sum(L["W"].size + L["b"].size for L in self.params)

    # -- training ------------------------------------------------------------- #
    def fit(self, x: np.ndarray) -> Diffusion:
        x = _check_data(x)
        z = encode(x) if self.latent else x
        rng = np.random.default_rng(self.seed)
        params = _init_net(rng, [self.dim + TIME_FEATURES, self.hidden,
                                 self.hidden, self.dim])
        opt = _Adam(params, self.lr)
        for _ in range(self.train_steps):
            idx = rng.integers(0, len(z), self.batch)
            x0 = z[idx]
            t = rng.integers(1, self.steps + 1, self.batch)
            eps = rng.normal(0.0, 1.0, x0.shape)
            abar = self.abar[t - 1][:, None]
            xt = np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps
            y = _target(x0, eps, abar, self.target)
            out, cache = _forward(params, xt, t, self.steps)
            dout = 2.0 * (out - y) / (len(x0) * x0.shape[1])
            opt.step(params, _backward(params, cache, dout))
        self.params = params
        return self

    # -- sampling ------------------------------------------------------------- #
    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw n points by running the reverse process from pure noise."""
        if self.params is None:
            raise RuntimeError("call fit first")
        if n < 1:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, (n, self.dim))
        for t in range(self.steps, 0, -1):
            out, _ = _forward(self.params, x, np.full(n, t), self.steps)
            _, eps = _split(out, x, self.abar[t - 1], self.target)
            noise = (rng.normal(0.0, 1.0, (n, self.dim))
                     if t > 1 else None)
            x = self._step(x, t, eps, noise)
        return decode(x) if self.latent else x

    def _step(self, x: np.ndarray, t: int, eps: np.ndarray,
              noise: np.ndarray | None) -> np.ndarray:
        """One ancestral update; `noise=None` makes it deterministic."""
        abar = self.abar[t - 1]
        mean = (x - self.beta[t - 1] / np.sqrt(1.0 - abar) * eps)
        mean = mean / np.sqrt(self.alpha[t - 1])
        if noise is None or t == 1:
            return mean
        var = self.beta[t - 1] * (1.0 - self.abar[t - 2]) / (1.0 - abar)
        return mean + np.sqrt(var) * noise

    # -- diagnostics ---------------------------------------------------------- #
    def denoise_mse(self, x: np.ndarray, t: int, repeats: int = 4,
                    seed: int = 0) -> float:
        """Mean squared error of the implied clean point at step `t`.

        Everything is scored in the data's own two dimensions: an ambient
        model gets six coordinates for free, and comparing its raw error with
        the latent model's would flatter it by a factor of four.
        """
        if self.params is None:
            raise RuntimeError("call fit first")
        if t < 1 or t > self.steps:
            raise ValueError(f"t must lie in [1, {self.steps}]")
        z = encode(x)
        rng = np.random.default_rng(seed)
        abar = self.abar[t - 1]
        total = 0.0
        for _ in range(repeats):
            eps = rng.normal(0.0, 1.0, z.shape)
            if self.latent:
                xt = np.sqrt(abar) * z + np.sqrt(1.0 - abar) * eps
            else:
                xt = (np.sqrt(abar) * decode(z)
                      + np.sqrt(1.0 - abar) * rng.normal(0.0, 1.0,
                                                         (len(z), DIM)))
            out, _ = _forward(self.params, xt, np.full(len(z), t), self.steps)
            x0, _ = _split(out, xt, abar, self.target)
            x0 = x0 if self.latent else encode(x0)
            total += float(np.mean((x0 - z) ** 2))
        return total / repeats


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
    x, y = make_data(0)
    test, _ = make_data(1)
    print(f"diffusion. eight Gaussians on a circle embedded in {DIM} "
          f"dimensions, {len(x)} points:")
    print(f"  ambient {DIM}, intrinsic {LATENT}, {PER_MODE} per mode, "
          f"spread {SPREAD}")
    print(f"  model {DIM + TIME_FEATURES} -> {HID} -> {HID} -> {DIM}, SiLU, "
          f"{TIME_FEATURES} time features")
    print(f"  {TRAIN_STEPS} Adam steps, batch {BATCH}, lr {LR}")

    runs = {}
    print("diffusion. the six configurations:")
    print(f"  {'variant':<19s} {'T':>5s} {'schedule':>8s} {'target':>7s} "
          f"{'params':>7s} {'modes':>6s} {'share':>6s} {'dist':>6s}")
    for name in VARIANTS:
        model = Diffusion(name).fit(x)
        runs[name] = model
        many, share, dist = mode_stats(model.sample(1024, seed=1))
        print(f"  {name:<19s} {model.steps:>5d} {model.schedule:>8s} "
              f"{model.target:>7s} {model.n_params():>7d} "
              f"{many:>4d}/{MODES} {share:6.2f} {dist:6.3f}")

    print("diffusion. the schedules at their last rung:")
    print(f"  {'schedule':<19s} {'T':>5s} {'abar_T':>10s} "
          f"{'middle step':>11s} {'mean beta':>10s}")
    for name in ("ddpm", "linear_schedule", "cosine_schedule"):
        model = runs[name]
        cross = int(np.argmax(model.abar < 0.5)) + 1
        print(f"  {name:<19s} {model.steps:>5d} {model.abar[-1]:>10.2e} "
              f"{cross:>11d} {np.mean(model.beta):>10.4f}")

    print("diffusion. the implied clean point at three noise levels:")
    print(f"  {'variant':<19s} {'t=10':>10s} {'t=50':>10s} {'t=T':>12s}")
    mse = {}
    for name in VARIANTS:
        model = runs[name]
        row = tuple(model.denoise_mse(test, t) for t in (10, 50,
                                                         model.steps))
        mse[name] = row
        print(f"  {name:<19s} {row[0]:>10.4f} {row[1]:>10.4f} "
              f"{row[2]:>12.4f}")

    print("diffusion. an oracle noise run through the same update:")
    model = runs["cosine_schedule"]
    rng = np.random.default_rng(4)
    hidden = encode(test)
    xt = np.sqrt(model.abar[-1]) * hidden + np.sqrt(
        1.0 - model.abar[-1]) * rng.normal(0.0, 1.0, hidden.shape)
    for t in range(model.steps, 0, -1):
        abar = model.abar[t - 1]
        eps = (xt - np.sqrt(abar) * hidden) / np.sqrt(1.0 - abar)
        xt = model._step(xt, t, eps, None)
    recovery = float(np.max(np.abs(xt - hidden)))
    print(f"  max |x0 - recovered| after {model.steps} deterministic steps: "
          f"{recovery:.2e}")

    print("diffusion. the latent model, and the ambient model it replaces:")
    latent = runs["latent_diffusion"]
    round_trip = float(np.mean((decode(encode(test)) - test) ** 2))
    print(f"  encode/decode round trip {round_trip:.2e}, "
          f"{latent.n_params()} parameters against "
          f"{runs['v_prediction'].n_params()}")
    print(f"  distance to nearest centre {mode_stats(latent.sample(1024, 1))[2]:.3f}"
          f" against {mode_stats(runs['v_prediction'].sample(1024, 1))[2]:.3f}")

    print("diffusion. malformed calls are refused:")
    bad = [
        lambda: Diffusion("ddim"),
        lambda: Diffusion("ddpm", hidden=0),
        lambda: Diffusion("ddpm", train_steps=0),
        lambda: Diffusion("ddpm", lr=-1.0),
        lambda: Diffusion("ddpm").sample(8),
        lambda: runs["ddpm"].sample(0),
        lambda: runs["ddpm"].fit(np.zeros((8, LATENT))),
        lambda: runs["ddpm"].fit(np.full((8, DIM), np.nan)),
        lambda: runs["ddpm"].denoise_mse(test, 0),
        lambda: runs["ddpm"].denoise_mse(test, runs["ddpm"].steps + 1),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)} malformed calls refused")

    print("diffusion. what the table says:")
    assert recovery < 1e-9
    assert round_trip < 1e-12
    assert all(mode_stats(runs[n].sample(1024, 1))[0] == MODES
               for n in VARIANTS)
    assert runs["ddpm"].steps == 1000
    assert runs["cosine_schedule"].abar[-1] < 1e-6
    assert 1e-3 < runs["linear_schedule"].abar[-1] < 2e-2
    assert mse["epsilon_prediction"][2] > 1e3
    assert mse["v_prediction"][2] < 1.0
    assert mse["v_prediction"][0] < 0.05
    assert mode_stats(latent.sample(1024, 1))[2] < 0.3
    assert (mode_stats(latent.sample(1024, 1))[2]
            < mode_stats(runs["v_prediction"].sample(1024, 1))[2] - 0.1)
    assert latent.n_params() < runs["v_prediction"].n_params()
    print("  every variant covers all eight modes; the cosine epsilon model's")
    print("  implied x0 at t=T is meaningless while v-prediction's stays")
    print("  below one; the latent model is lossless and samples closer.")
    print("diffusion demo passed")


if __name__ == "__main__":
    demo()
