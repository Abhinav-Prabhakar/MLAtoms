"""Dropout and friends — noise that is scaled to disappear at inference.

Chapter:   12-neural-blocks
Companion: dropout.md
Tier 2 · 8 variants: standard, spatial, gaussian, variational, dropout1d,
droppath, stochastic_depth, alpha_dropout
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("standard", "spatial", "gaussian", "variational", "dropout1d",
            "droppath", "stochastic_depth", "alpha_dropout")

PATHS = ("droppath", "stochastic_depth")

# SELU constants, reused by alpha dropout.
ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #
class Dropout:
    """Multiplicative noise with inverted scaling.

    `forward` returns the perturbed tensor and a cache; `backward` applies the
    same mask to the upstream gradient. At inference every variant is the
    identity, except the path variants, which add the residual back.
    """

    def __init__(self, variant: str = "standard", rate: float = 0.5, *,
                 sigma: float = 0.5, depth: int = 0, max_depth: int = 8,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not np.isfinite(rate) or rate < 0.0 or rate >= 1.0:
            raise ValueError("rate must be finite and in [0, 1)")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.variant = variant
        self.rate = rate
        self.sigma = sigma
        self.depth = depth
        self.max_depth = max_depth
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.alpha = ALPHA
        self.scale = SCALE

    # -- helpers ------------------------------------------------------------- #
    def _check(self, x: np.ndarray, name: str = "x") -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim < 1 or x.size == 0:
            raise ValueError(f"{name} must have at least one element")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite")
        return x

    def _rate_at(self, depth: int | None) -> float:
        if self.variant != "stochastic_depth":
            return self.rate
        level = self.depth if depth is None else depth
        if level < 0 or level >= self.max_depth:
            raise ValueError("depth must be in [0, max_depth)")
        return self.rate * (level + 1) / self.max_depth

    def _channel_shape(self, x: np.ndarray) -> tuple[int, ...]:
        """One mask entry per sample and channel, broadcast over space."""
        return (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)

    def _time_shape(self, x: np.ndarray) -> tuple[int, ...]:
        """One mask entry per sample and time step, broadcast over features."""
        return (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)

    def _sample_shape(self, x: np.ndarray) -> tuple[int, ...]:
        """One mask entry per sample, broadcast over everything else."""
        return (x.shape[0],) + (1,) * (x.ndim - 1)

    # -- forward ------------------------------------------------------------- #
    def forward(self, x: np.ndarray, *, residual: np.ndarray | None = None,
                depth: int | None = None, training: bool = True):
        x = self._check(x)
        if self.variant in PATHS:
            if residual is None:
                raise ValueError("a path variant needs a residual")
            residual = self._check(residual, "residual")
            if residual.shape != x.shape:
                raise ValueError("residual must have the shape of the branch")
            if not training:
                return residual + x, {"eval": True}
        elif not training:
            return x, {"eval": True}

        keep = 1.0 - self._rate_at(depth)
        cache = {"eval": False, "variant": self.variant, "keep": keep,
                 "rate": 1.0 - keep}

        if self.variant in ("standard", "alpha_dropout"):
            mask = self.rng.random(x.shape) < keep
            kept = mask.astype(float)
            if self.variant == "alpha_dropout":
                alpha_p = -self.alpha * self.scale
                a = (keep * (1.0 + (1.0 - keep) * alpha_p ** 2)) ** -0.5
                b = -a * alpha_p * (1.0 - keep)
                output = x * kept * a + alpha_p * (1.0 - kept) * a + b
                scale = kept * a
                cache.update(scale=scale, a=a, b=b, alpha_p=alpha_p)
                return output, cache
            scale = kept / keep
            cache["scale"] = scale
            return x * scale, cache

        if self.variant == "gaussian":
            noise = 1.0 + self.sigma * self.rng.standard_normal(x.shape)
            cache["noise"] = noise
            return x * noise, cache

        if self.variant == "spatial":
            if x.ndim < 2:
                raise ValueError("spatial needs a sample and channel axis")
            shape = self._channel_shape(x)
            mask = self.rng.random(shape) < keep
            scale = mask.astype(float) / keep
            cache["scale"] = scale
            return x * scale, cache

        if self.variant == "variational":
            if x.ndim < 2:
                raise ValueError("variational needs a sample and feature axis")
            shape = self._channel_shape(x)
            noise = 1.0 + self.sigma * self.rng.standard_normal(shape)
            cache["noise"] = noise
            return x * noise, cache

        if self.variant == "dropout1d":
            if x.ndim < 2:
                raise ValueError("dropout1d needs a sample and time axis")
            shape = self._time_shape(x)
            mask = self.rng.random(shape) < keep
            scale = mask.astype(float) / keep
            cache["scale"] = scale
            return x * scale, cache

        # droppath and stochastic_depth: keep or drop the whole sample.
        shape = self._sample_shape(x)
        mask = self.rng.random(shape) < keep
        scale = mask.astype(float) / keep
        cache.update(scale=scale, rate=1.0 - keep)
        return residual + x * scale, cache

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        if cache.get("eval"):
            return gradient
        if "noise" in cache:
            return gradient * cache["noise"]
        return gradient * cache["scale"]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)

    print("dropout. inverted scaling keeps the expectation and kills train/test")
    print("  skew:")
    x = np.ones((20000, 8))
    model = Dropout("standard", rate=0.25, seed=2)
    out, cache = model.forward(x)
    kept = out != 0.0
    assert abs(kept.mean() - 0.75) < 0.005
    assert np.allclose(out[kept], 1.0 / 0.75)
    assert abs(out.mean() - 1.0) < 0.005
    assert abs(out.var() - (1.0 / 0.75 - 1.0)) < 0.02
    assert np.allclose(model.forward(x, training=False)[0], x)
    assert np.allclose(model.backward(np.ones_like(out), cache),
                       kept.astype(float) / 0.75)
    print(f"  keep rate {kept.mean():.4f}, surviving values exactly"
          f" {1.0 / 0.75:.4f},")
    print(f"  mean {out.mean():.4f} and variance {out.var():.4f} = p/(1-p);"
          " inference is the identity")

    print("dropout. spatial drops whole channels:")
    x = rng.normal(size=(6, 3, 4, 5))
    model = Dropout("spatial", rate=0.5, seed=3)
    out, _ = model.forward(x)
    zero = out == 0.0
    assert np.all(zero == zero[:, :1, :1, :])
    channels = (~zero[:, 0, 0, :]).sum()
    assert 12 <= channels <= 21
    assert np.allclose(out[~zero], (x / 0.5)[~zero])
    print(f"  {channels} of 24 feature maps survive; each is either zero")

    print("dropout. Gaussian noise is multiplicative with mean one:")
    x = np.ones((20000, 8))
    model = Dropout("gaussian", sigma=0.5, seed=4)
    out, cache = model.forward(x)
    assert abs(out.mean() - 1.0) < 0.01
    assert abs(out.var() - 0.25) < 0.02
    assert np.allclose(model.backward(np.ones_like(out), cache), out)
    print(f"  mean {out.mean():.4f} and variance {out.var():.4f} for sigma"
          " 0.5, with no mask at all")

    print("dropout. variational noise is fixed across time:")
    x = rng.normal(size=(5, 7, 6))
    model = Dropout("variational", sigma=0.5, seed=5)
    out, cache = model.forward(x)
    noise = cache["noise"]
    assert noise.shape == (5, 1, 6)
    assert np.allclose(out, x * noise)
    assert np.allclose(out[:, 0, :] / x[:, 0, :],
                       out[:, -1, :] / x[:, -1, :])
    print(f"  {noise.shape[0]} x {noise.shape[2]} noises, each reused at all"
          " seven steps")

    print("dropout. dropout1d drops whole timesteps:")
    x = rng.normal(size=(5, 7, 6))
    model = Dropout("dropout1d", rate=0.5, seed=6)
    out, _ = model.forward(x)
    zero = out == 0.0
    assert np.all(zero == zero[:, :, :1])
    steps = int(zero[:, :, 0].sum())
    assert 5 <= steps <= 25
    print(f"  {steps} of 35 time steps dropped, each across all six features")

    print("dropout. drop path removes a whole residual branch:")
    branch = rng.normal(size=(16, 4))
    residual = rng.normal(size=(16, 4))
    model = Dropout("droppath", rate=0.5, seed=7)
    out, cache = model.forward(branch, residual=residual)
    dropped = np.all(out == residual, axis=1)
    survived = ~dropped
    assert np.allclose(out[dropped], residual[dropped])
    assert np.allclose(out[survived],
                       residual[survived] + 2.0 * branch[survived])
    gradient = model.backward(np.ones_like(branch), cache)
    assert np.allclose(gradient[survived], 2.0)
    assert np.allclose(gradient[dropped], 0.0)
    assert np.allclose(model.forward(branch, residual=residual,
                                     training=False)[0], residual + branch)
    print(f"  {int(dropped.sum())} of 16 samples skip the branch entirely and"
          " their gradient is zero")

    print("dropout. stochastic depth grows the rate with depth:")
    x = rng.normal(size=(2000, 4))
    residual = np.zeros_like(x)
    model = Dropout("stochastic_depth", rate=0.5, max_depth=6, seed=8)
    rates, drops = [], []
    for depth in range(6):
        out, cache = model.forward(x, residual=residual, depth=depth)
        rates.append(cache["rate"])
        drops.append(float(np.mean(np.all(out == 0.0, axis=1))))
    assert np.all(np.diff(rates) > 0.0)
    assert abs(rates[-1] - 0.5) < 1e-12
    assert abs(drops[0] - rates[0]) < 0.03
    assert abs(drops[-1] - rates[-1]) < 0.03
    print("  rates " + ", ".join(f"{rate:.3f}" for rate in rates))
    print("  and measured drops " + ", ".join(f"{drop:.3f}" for drop in drops))

    print("dropout. alpha dropout preserves mean zero and variance one:")
    x = rng.normal(size=(50000, 8))
    model = Dropout("alpha_dropout", rate=0.25, seed=9)
    out, cache = model.forward(x)
    a, b = cache["a"], cache["b"]
    alpha_p = cache["alpha_p"]
    assert abs(out.mean()) < 0.02
    assert abs(out.var() - 1.0) < 0.03
    dropped = np.isclose(out, alpha_p * a + b)
    assert dropped.any()
    assert np.allclose(out[dropped], alpha_p * a + b)
    assert abs(a ** 2 * 0.75 * (1.0 + 0.25 * alpha_p ** 2) - 1.0) < 1e-9
    plain = Dropout("standard", rate=0.25, seed=9).forward(x)[0]
    assert plain.var() > 1.2
    print(f"  mean {out.mean():.4f}, variance {out.var():.4f}; standard"
          f" dropout on the same input has {plain.var():.4f}")

    print("dropout. a zero rate is exactly the identity:")
    x = rng.normal(size=(4, 3, 2))
    for variant in VARIANTS:
        model = Dropout(variant, rate=0.0, sigma=0.0, seed=10)
        if variant in PATHS:
            out, _ = model.forward(x, residual=x)
            assert np.allclose(out, 2.0 * x)
        else:
            out, _ = model.forward(x)
            assert np.allclose(out, x)
    print("  all eight variants reproduce their input at rate zero")

    print("dropout. malformed calls are refused:")
    guard = Dropout("standard", rate=0.5, seed=11)
    bad = [
        lambda: Dropout("inverted"),
        lambda: Dropout("standard", rate=-0.1),
        lambda: Dropout("standard", rate=1.0),
        lambda: Dropout("standard", rate=np.nan),
        lambda: Dropout("gaussian", sigma=-0.1),
        lambda: Dropout("gaussian", sigma=np.nan),
        lambda: Dropout("stochastic_depth", max_depth=0),
        lambda: Dropout("stochastic_depth", max_depth=3).forward(
            np.zeros((2, 2)), residual=np.zeros((2, 2)), depth=3),
        lambda: Dropout("droppath").forward(np.zeros((2, 2))),
        lambda: Dropout("droppath").forward(np.zeros((2, 2)),
                                            residual=np.zeros((3, 2))),
        lambda: Dropout("spatial").forward(np.zeros(4)),
        lambda: guard.forward(np.zeros((2, 2, 2)) * np.nan),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a rate outside")
    print("  [0, 1), a negative or non-finite sigma, a zero depth budget, a")
    print("  depth past it, a path without a residual or with the wrong shape,")
    print("  a spatial call on a vector, and a non-finite input")
    print("dropout demo passed")


if __name__ == "__main__":
    demo()
