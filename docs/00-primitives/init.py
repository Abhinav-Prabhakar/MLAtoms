"""Initialisation — 10 schemes that decide whether a deep net trains.

Chapter:   00-primitives
Companion: init.md
Tier 1 · 10 variants: zeros, xavier_uniform, xavier_normal, he_uniform, he_normal,
orthogonal, lsuv, fixup, truncated_normal, scaled_variance

Slow or exploding training at step 0 is almost always an initialisation bug: the
activation variance either collapses or doubles once per layer. Every scheme here
is a different answer to "what variance should the weights have?".
"""

from __future__ import annotations

import numpy as np


def initialize(
    shape: tuple[int, int],
    variant: str = "he_normal",
    *,
    gain: float = 1.0,
    depth: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Weight matrix of `shape = (fan_out, fan_in)`.

    depth is only used by fixup (number of residual branches).
    """
    rng = np.random.default_rng(rng)
    fan_out, fan_in = shape
    if variant == "zeros":
        return np.zeros(shape)
    if variant == "xavier_uniform":
        a = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-a, a, size=shape)
    if variant == "xavier_normal":
        return rng.normal(0.0, gain * np.sqrt(2.0 / (fan_in + fan_out)), size=shape)
    if variant == "he_uniform":
        a = gain * np.sqrt(6.0 / fan_in)
        return rng.uniform(-a, a, size=shape)
    if variant == "he_normal":
        return rng.normal(0.0, gain * np.sqrt(2.0 / fan_in), size=shape)
    if variant == "orthogonal":
        return _orthogonal(fan_out, fan_in, gain, rng)
    if variant == "fixup":
        # Depth-aware shrinkage of the residual branch, in the spirit of Fixup:
        # each branch contributes 1/sqrt(N), so the residual stream's variance
        # grows linearly with depth instead of exponentially.
        scale = 1.0 / np.sqrt(max(depth - 1, 1))
        return rng.normal(0.0, gain * np.sqrt(1.0 / fan_in), size=shape) * scale
    if variant == "truncated_normal":
        sd = gain * np.sqrt(2.0 / fan_in)
        w = rng.normal(0.0, sd, size=shape)
        return np.clip(w, -2.0 * sd, 2.0 * sd)
    if variant == "scaled_variance":
        # Unit-variance weights scaled by 1/sqrt(fan_in): the recent default for
        # residual streams that are not layer-normalised.
        return rng.normal(0.0, 1.0, size=shape) / np.sqrt(fan_in)
    raise ValueError(f"unknown variant {variant!r}")


def _orthogonal(fan_out: int, fan_in: int, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Semi-orthogonal matrix: W W^T = gain^2 I when fan_out <= fan_in."""
    a = rng.normal(size=(max(fan_out, fan_in), min(fan_out, fan_in)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if fan_out < fan_in:
        q = q.T
    return gain * q[:fan_out, :fan_in]


def _orthogonalize(W: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(W)
    return q * np.sign(np.diag(r))


def lsuv(
    W: np.ndarray, x: np.ndarray, activation=None, target: float = 1.0, iters: int = 8
) -> np.ndarray:
    """Layer-sequential unit variance: orthogonalise, then rescale so the
    post-activation output standard deviation is `target` on real data.

    The rescale must be iterated because a nonlinearity does not commute with a
    change of weight scale (a fact most one-shot implementations get wrong).
    """
    q = _orthogonalize(W)
    for _ in range(iters):
        y = x @ q.T
        if activation is not None:
            y = activation(y)
        q = q * (target / max(float(np.std(y)), 1e-12))
    return q


def main() -> None:
    rng = np.random.default_rng(0)
    d_in, d_out, n = 512, 512, 4096
    x = rng.normal(size=(n, d_in))

    # 1. Xavier keeps variance constant across a linear layer.
    for v in ("xavier_uniform", "xavier_normal"):
        W = initialize((d_out, d_in), v, rng=rng)
        ratio = np.var(x @ W.T) / np.var(x)
        assert abs(ratio - 1.0) < 0.1, (v, ratio)
    print("xavier: output variance ~ input variance")

    # 2. He accounts for the ReLU halving: the second moment is preserved.
    # (Variance is not: ReLU output has a positive mean, so var = E[y^2] - E[y]^2.)
    for v in ("he_uniform", "he_normal", "truncated_normal"):
        W = initialize((d_out, d_in), v, rng=rng)
        y = np.maximum(x @ W.T, 0.0)
        ratio = np.mean(y**2) / np.mean(x**2)
        assert abs(ratio - 1.0) < 0.12, (v, ratio)
    print("he: post-ReLU second moment preserved")

    # 3. Orthogonal really is orthogonal (spectrum is flat).
    W = initialize((128, 128), "orthogonal", rng=rng)
    sv = np.linalg.svd(W, compute_uv=False)
    assert np.allclose(sv, sv[0], rtol=1e-8)
    assert np.allclose(W @ W.T, np.eye(128), atol=1e-8)
    print(f"orthogonal: condition number {sv[0] / sv[-1]:.3f}")

    # 4. LSUV reaches unit output standard deviation on real activations.
    relu = lambda t: np.maximum(t, 0.0)  # noqa: E731
    W = lsuv(initialize((256, 128), "he_normal", rng=rng), x[:, :128], relu)
    assert abs(np.std(relu(x[:, :128] @ W.T)) - 1.0) < 1e-6
    print("lsuv: unit output std")

    # 5. Truncated normal is bounded at 2 sigma; zeros stay zero.
    W = initialize((1000, 1000), "truncated_normal", rng=rng)
    assert np.abs(W).max() <= 2.0 * np.sqrt(2.0 / 1000) + 1e-12
    assert np.count_nonzero(initialize((4, 4), "zeros")) == 0
    print("truncated/zeros ok")

    # 6. Fixup shrinks with depth.
    shallow = initialize((64, 64), "fixup", depth=2, rng=rng).std()
    deep = initialize((64, 64), "fixup", depth=50, rng=rng).std()
    assert deep < shallow
    print(f"fixup: depth 2 -> {shallow:.4f}, depth 50 -> {deep:.4f}")
    print("init demo passed")


if __name__ == "__main__":
    main()
