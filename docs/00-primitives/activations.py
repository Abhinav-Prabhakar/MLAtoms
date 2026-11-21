"""Activations — 17 pointwise nonlinearities plus the simplex projections.

Chapter:   00-primitives
Companion: activations.md
Tier 1 · 17 variants: relu, leaky_relu, prelu, elu, selu, gelu_tanh, gelu_erf,
silu, mish, softplus, glu, swiglu, geglu, softmax, log_softmax, sparsemax, entmax

Every pointwise activation exposes an exact derivative. The simplex family
(softmax, sparsemax, entmax) is not pointwise: it maps a vector to a
probability vector, so it exposes a Jacobian-vector product instead.
"""

from __future__ import annotations

import numpy as np

SELU_SCALE = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772
GELU_C = np.sqrt(2.0 / np.pi)
GELU_K = 0.044715


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(__import__("math").erf, otypes=[float])(x)


POINTWISE = frozenset(
    ["relu", "leaky_relu", "prelu", "elu", "selu", "gelu_tanh", "gelu_erf",
     "silu", "mish", "softplus"]
)
SIMPLEX = frozenset(["softmax", "log_softmax", "sparsemax", "entmax"])
GATED = frozenset(["glu", "swiglu", "geglu"])


def activation(x: np.ndarray, variant: str = "relu", alpha: float = 0.01) -> np.ndarray:
    """Pointwise activation. x: (...,) -> same shape.

    alpha is the negative slope for leaky_relu/prelu, and the saturation value
    for elu (selu ignores it and uses the published constants).
    """
    x = np.asarray(x, dtype=float)
    if variant == "relu":
        return np.maximum(x, 0.0)
    if variant in ("leaky_relu", "prelu"):
        return np.where(x > 0.0, x, alpha * x)
    if variant == "elu":
        return np.where(x > 0.0, x, alpha * (np.expm1(np.minimum(x, 0.0))))
    if variant == "selu":
        return SELU_SCALE * np.where(
            x > 0.0, x, SELU_ALPHA * np.expm1(np.minimum(x, 0.0))
        )
    if variant == "gelu_tanh":
        return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + GELU_K * x**3)))
    if variant == "gelu_erf":
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
    if variant == "silu":
        return x * _sigmoid(x)
    if variant == "mish":
        return x * np.tanh(_softplus(x))
    if variant == "softplus":
        return _softplus(x)
    raise ValueError(f"unknown pointwise variant {variant!r}")


def activation_grad(x: np.ndarray, variant: str = "relu", alpha: float = 0.01) -> np.ndarray:
    """d(activation)/dx at x, elementwise."""
    x = np.asarray(x, dtype=float)
    if variant == "relu":
        return (x > 0.0).astype(float)
    if variant in ("leaky_relu", "prelu"):
        return np.where(x > 0.0, 1.0, alpha)
    if variant == "elu":
        return np.where(x > 0.0, 1.0, alpha * np.exp(np.minimum(x, 0.0)))
    if variant == "selu":
        return np.where(x > 0.0, SELU_SCALE, SELU_SCALE * SELU_ALPHA * np.exp(np.minimum(x, 0.0)))
    if variant == "gelu_tanh":
        t = np.tanh(GELU_C * (x + GELU_K * x**3))
        du = GELU_C * (1.0 + 3.0 * GELU_K * x**2)
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du
    if variant == "gelu_erf":
        phi = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
        return 0.5 * (1.0 + _erf(x / np.sqrt(2.0))) + x * phi
    if variant == "silu":
        s = _sigmoid(x)
        return s * (1.0 + x * (1.0 - s))
    if variant == "mish":
        sp = _softplus(x)
        t = np.tanh(sp)
        return t + x * (1.0 - t**2) * _sigmoid(x)
    if variant == "softplus":
        return _sigmoid(x)
    raise ValueError(f"unknown pointwise variant {variant!r}")


# --------------------------------------------------------------------------- #
# Simplex projections: sum to one, non-negative.
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def sparsemax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Euclidean projection onto the simplex (Martins & Astudillo, 2016).

    Output is sparse: coordinates whose z-score is below the threshold are
    exactly zero.
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    zs = -np.sort(-z, axis=axis)
    css = np.cumsum(zs, axis=axis) - 1.0
    n = z.shape[axis]
    shape = [1] * z.ndim
    shape[axis] = n
    k = np.arange(1, n + 1).reshape(shape)
    support = zs - css / k > 0
    k_z = support.sum(axis=axis, keepdims=True)
    tau = np.take_along_axis(css, (k_z - 1).astype(int), axis=axis) / k_z
    return np.maximum(z - tau, 0.0)


def entmax15(x: np.ndarray, axis: int = -1, iters: int = 40) -> np.ndarray:
    """alpha-entmax with alpha=1.5: p_i = max(0, z_i/2 - tau)^2, sum p_i = 1.

    Temperature-free interpolation between softmax (dense) and sparsemax
    (sparse), by bisection on tau. Exact for alpha=1.5, unlike iterative
    approximations used for general alpha.
    """
    z = np.moveaxis(np.asarray(x, dtype=float), axis, -1)
    flat = z.reshape(-1, z.shape[-1])
    out = np.empty_like(flat)
    for row in range(flat.shape[0]):
        zz = flat[row]
        lo = zz.max() / 2.0 - 1.0
        hi = zz.max() / 2.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            p = np.maximum(zz / 2.0 - mid, 0.0) ** 2
            if p.sum() > 1.0:      # tau too small: masses still too large
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        out[row] = np.maximum(zz / 2.0 - tau, 0.0) ** 2
    return np.moveaxis(out.reshape(z.shape), -1, axis)


def softmax_jvp(s: np.ndarray, ds: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jacobian-vector product: J_softmax(s) @ ds, given s = softmax(x)."""
    return s * (ds - np.sum(s * ds, axis=axis, keepdims=True))


def sparsemax_jvp(s: np.ndarray, ds: np.ndarray, axis: int = -1) -> np.ndarray:
    """JVP on the support: zero outside, centred inside."""
    mask = (s > 0).astype(float)
    d = ds * mask
    return d - mask * (np.sum(d, axis=axis, keepdims=True) / np.maximum(mask.sum(axis=axis, keepdims=True), 1.0))


# --------------------------------------------------------------------------- #
# Gated units: two inputs, one output.
# --------------------------------------------------------------------------- #
def gated(a: np.ndarray, b: np.ndarray, variant: str = "swiglu") -> np.ndarray:
    """Gated linear unit. a is the value branch, b is the gate branch."""
    if variant == "glu":
        return a * _sigmoid(b)
    if variant == "swiglu":
        return activation(a, "silu") * b
    if variant == "geglu":
        return activation(a, "gelu_tanh") * b
    raise ValueError(f"unknown gated variant {variant!r}")


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(scale=2.0, size=(7, 11))

    # 1. Every pointwise derivative matches a central difference.
    for v in sorted(POINTWISE):
        g = activation_grad(x, v, alpha=0.3)
        h = 1e-6
        num = (activation(x + h, v, alpha=0.3) - activation(x - h, v, alpha=0.3)) / (2 * h)
        err = np.max(np.abs(g - num))
        assert err < 2e-5, (v, err)
    print(f"pointwise gradients ok ({len(POINTWISE)} variants, max err < 2e-5)")

    # 2. Simplex projections are valid distributions.
    for v, f in [("softmax", softmax), ("sparsemax", sparsemax), ("entmax", entmax15)]:
        p = f(x)
        assert np.allclose(p.sum(-1), 1.0), v
        assert (p >= -1e-12).all(), v
    assert np.allclose(np.exp(log_softmax(x)), softmax(x))
    print("simplex: all rows sum to 1")

    # 3. Sparsity ordering: sparsemax <= entmax <= softmax in non-zeros.
    zeros = {v: int((f(x) == 0).sum()) for v, f in
             [("softmax", softmax), ("entmax", entmax15), ("sparsemax", sparsemax)]}
    assert zeros["softmax"] <= zeros["entmax"] <= zeros["sparsemax"], zeros
    print(f"sparsity (zeros): {zeros}")

    # 4. softmax JVP matches the finite-difference Jacobian.
    z = x[0]
    s = softmax(z)
    dz = rng.normal(size=z.shape)
    h = 1e-6
    fd = (softmax(z + h * dz) - softmax(z - h * dz)) / (2 * h)
    assert np.allclose(softmax_jvp(s, dz), fd, atol=1e-6)
    print("softmax JVP ok")

    # 5. Gated units are elementwise products of the right branch.
    a, b = rng.normal(size=(2, 4))
    assert np.allclose(gated(a, b, "swiglu"), activation(a, "silu") * b)
    print("gated units ok")
    print("activations demo passed")


if __name__ == "__main__":
    main()
