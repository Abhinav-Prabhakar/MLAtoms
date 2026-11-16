"""Normalisation — 8 layers that control the scale of activations.

Chapter:   00-primitives
Companion: normalization.md
Tier 1 · 8 variants: batch, layer, group, instance, rms, weight_std, batch_renorm,
power_norm

All functions are stateless forwards: they compute statistics from the tensor they
are given. Keeping running statistics is a training-loop concern and lives in the
atom's doc, not here.
"""

from __future__ import annotations

import numpy as np


def _stats_axes(x: np.ndarray, mode: str) -> tuple[int, ...]:
    if x.ndim == 2:
        n, f = x.shape
        if mode in ("batch", "layer", "group", "instance"):
            # 2-D input: batch over rows, layer over columns. Group/instance are
            # per-sample reductions, which for 2-D means the row.
            return (0,) if mode == "batch" else (1,)
        raise ValueError(mode)
    if x.ndim == 4:
        if mode == "batch":
            return (0, 2, 3)
        if mode == "layer":
            return (1, 2, 3)
        if mode == "instance":
            return (2, 3)
    raise ValueError(f"unsupported shape {x.shape} for {mode}")


def batch_norm(
    x: np.ndarray, gamma: np.ndarray | None = None, beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalise each feature over the batch (and spatial dims)."""
    axes = _stats_axes(x, "batch")
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    return _affine(y, gamma, beta)


def layer_norm(
    x: np.ndarray, gamma: np.ndarray | None = None, beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalise each sample over all non-batch dims. Independent of batch size."""
    axes = _stats_axes(x, "layer")
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    return _affine(y, gamma, beta)


def instance_norm(
    x: np.ndarray, gamma: np.ndarray | None = None, beta: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalise each channel of each sample over its spatial dims."""
    axes = _stats_axes(x, "instance")
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    return _affine(y, gamma, beta)


def group_norm(
    x: np.ndarray, groups: int, gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None, eps: float = 1e-5,
) -> np.ndarray:
    """Normalise within groups of channels. groups=1 -> layer norm,
    groups=C -> instance norm; anything between is a compute/accuracy knob."""
    if x.ndim == 2:
        return layer_norm(x, gamma, beta, eps)
    n, c = x.shape[0], x.shape[1]
    if c % groups:
        raise ValueError(f"{c} channels not divisible into {groups} groups")
    shape = (n, groups, c // groups) + x.shape[2:]
    xg = x.reshape(shape)
    axes = tuple(range(2, xg.ndim))  # everything except batch and group index
    mu = xg.mean(axis=axes, keepdims=True)
    var = xg.var(axis=axes, keepdims=True)
    y = ((xg - mu) / np.sqrt(var + eps)).reshape(x.shape)
    return _affine(y, gamma, beta)


def rms_norm(
    x: np.ndarray, gamma: np.ndarray | None = None, eps: float = 1e-6
) -> np.ndarray:
    """Divide by the root-mean-square only: no mean subtraction, no beta.
    Cheaper than LayerNorm and the current default in LLMs."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return _affine(x / rms, gamma, None)


def weight_std(W: np.ndarray, gain: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """Standardise each output row of a weight matrix: a reparameterisation that
    makes the *effective* weights scale-free, not the activations."""
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    return gain * (W - mu) / (sd + eps)


def batch_renorm(
    x: np.ndarray, running_mean: np.ndarray, running_var: np.ndarray,
    rmax: float = 3.0, dmax: float = 5.0, eps: float = 1e-5,
) -> np.ndarray:
    """Batch norm with clipped statistics (Ioffe, 2017).

    r = clip(std_batch / std_running, 1/rmax, rmax) and
    d = clip((mean_batch - mean_running) / std_running, -dmax, dmax).
    Output is (x - mean_batch)/std_batch * r + d, so a batch whose statistics drift
    far from the running estimate cannot move the activations more than dmax.
    """
    axes = _stats_axes(x, "batch")
    mu = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    shape = (1, -1) + (1,) * (x.ndim - 2)
    rm = np.asarray(running_mean, dtype=float).reshape(shape)
    rv = np.asarray(running_var, dtype=float).reshape(shape)
    r = np.clip(np.sqrt(var + eps) / np.sqrt(rv + eps), 1.0 / rmax, rmax)
    d = np.clip((mu - rm) / np.sqrt(rv + eps), -dmax, dmax)
    return (x - mu) / np.sqrt(var + eps) * r + d


def power_norm(x: np.ndarray, p: float = 2.0, eps: float = 1e-12) -> np.ndarray:
    """Project each vector onto the sphere of radius d^(1/p): l_p normalisation,
    used to stabilise very deep nets whose activations grow without bound."""
    norm = np.linalg.norm(x, ord=p, axis=-1, keepdims=True)
    return x / (norm + eps) * x.shape[-1] ** (1.0 / p)


def _affine(y: np.ndarray, gamma: np.ndarray | None, beta: np.ndarray | None) -> np.ndarray:
    if gamma is not None:
        g = np.asarray(gamma).reshape((1, -1) + (1,) * (y.ndim - 2)) if y.ndim > 2 else gamma
        y = y * g
    if beta is not None:
        b = np.asarray(beta).reshape((1, -1) + (1,) * (y.ndim - 2)) if y.ndim > 2 else beta
        y = y + b
    return y


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(16, 8))

    # 1. Batch norm: zero mean, unit variance per feature.
    y = batch_norm(x)
    assert np.allclose(y.mean(0), 0, atol=1e-8) and np.allclose(y.var(0), 1, atol=1e-4)
    # 2. Layer norm: zero mean, unit variance per sample, and it ignores batch composition.
    y = layer_norm(x)
    assert np.allclose(y.mean(1), 0, atol=1e-8) and np.allclose(y.var(1), 1, atol=1e-4)
    single = layer_norm(x[3:4])
    assert np.allclose(single, y[3:4], atol=1e-10), "layer norm must not depend on the batch"
    print("batch/layer norm statistics ok")

    # 3. Group norm interpolates between layer norm and instance norm.
    img = rng.normal(size=(4, 6, 5, 5))
    assert np.allclose(group_norm(img, 1), layer_norm(img), atol=1e-10)
    assert np.allclose(group_norm(img, 6), instance_norm(img), atol=1e-10)
    per_plane = instance_norm(img)
    assert np.allclose(per_plane.mean(axis=(2, 3)), 0, atol=1e-8)
    print("group norm: groups=1 -> layer, groups=C -> instance")

    # 4. RMS norm: unit root-mean-square, mean untouched.
    z = rng.normal(size=(7, 11)) + 2.0
    r = rms_norm(z)
    assert np.allclose(np.sqrt((r**2).mean(-1)), 1.0, atol=1e-5)
    assert not np.allclose(r.mean(-1), 0, atol=1e-3), "rms norm must not centre"
    print("rms norm: unit RMS, non-zero mean")

    # 5. Weight standardisation makes rows zero-mean/unit-std.
    W = rng.normal(size=(12, 20)) * 3 + 1
    ws = weight_std(W)
    assert np.allclose(ws.mean(1), 0, atol=1e-8) and np.allclose(ws.std(1), 1, atol=1e-6)
    print("weight standardisation ok")

    # 6. Batch renorm equals batch norm when statistics match, and clips when they drift.
    x0 = rng.normal(size=(64, 8))
    assert np.allclose(batch_renorm(x0, x0.mean(0), x0.var(0)), batch_norm(x0), atol=1e-8)
    shifted = rng.normal(50.0, 10.0, size=(16, 8))
    br = batch_renorm(shifted, np.zeros(8), np.ones(8), rmax=3.0, dmax=5.0)
    assert np.abs(br.mean(0)).max() <= 5.0 + 1e-6
    print("batch renorm: matches BN when stable, clips drift to dmax")

    # 7. Power norm gives every row the same l_p norm.
    pn = power_norm(rng.normal(size=(5, 9)) * np.arange(1, 6)[:, None], p=2)
    assert np.allclose(np.linalg.norm(pn, axis=1), 3.0, atol=1e-6)
    print("power norm: constant row norm")
    print("normalization demo passed")


if __name__ == "__main__":
    main()
