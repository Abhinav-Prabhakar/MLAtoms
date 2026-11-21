"""Pooling — nine ways to throw away spatial resolution.

Chapter:   12-neural-blocks
Companion: pooling.md
Tier 1 · 9 variants: max, avg, global_max, global_avg, adaptive, attention_pool,
gem, lp_pool, stochastic
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("max", "avg", "global_max", "global_avg", "adaptive",
            "attention_pool", "gem", "lp_pool", "stochastic")


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def _tuple(value, dims: int, name: str):
    """Expand a scalar to one entry per spatial axis."""
    if isinstance(value, (int, np.integer)):
        values = (int(value),) * dims
    else:
        values = tuple(int(entry) for entry in value)
        if len(values) != dims:
            raise ValueError(f"{name} must have one entry per spatial axis")
    if any(entry < 1 for entry in values):
        raise ValueError(f"{name} entries must be positive")
    return values


def _windows(x: np.ndarray, size, stride) -> np.ndarray:
    """Every window, as `(batch, *output, channels, *window)`.

    `sliding_window_view` makes the windows a read-only view; the stride
    subsamples the output positions afterwards, so no copy happens until the
    caller reduces over the trailing window axes.
    """
    dims = len(size)
    view = np.lib.stride_tricks.sliding_window_view(
        x, size, axis=tuple(range(1, dims + 1)))
    steps = (slice(None),) + tuple(slice(None, None, step) for step in stride)
    return view[steps]


def _reduce(windows: np.ndarray, dims: int, how: str,
            power: float = 1.0) -> np.ndarray:
    axes = tuple(range(windows.ndim - dims, windows.ndim))
    if how == "max":
        return windows.max(axis=axes)
    if how == "mean":
        return windows.mean(axis=axes)
    if how == "gem":
        if np.any(windows < 0.0):
            raise ValueError("gem needs non-negative inputs")
        if power == 0.0:
            if np.any(windows <= 0.0):
                raise ValueError("the geometric mean needs positive inputs")
            return np.exp(np.log(windows).mean(axis=axes))
        moment = np.mean(windows ** power, axis=axes)
        return np.power(moment, 1.0 / power)
    if how == "lp":
        if np.any(windows < 0.0):
            raise ValueError("lp_pool needs non-negative inputs")
        if power == 0.0:
            raise ValueError("lp_pool has no geometric-mean form")
        return np.power(np.sum(windows ** power, axis=axes), 1.0 / power)
    raise ValueError(f"unknown reduction {how!r}")


# --------------------------------------------------------------------------- #
# Adaptive and attention pooling
# --------------------------------------------------------------------------- #
def _bounds(size: int, out: int, index: int) -> tuple[int, int]:
    """The torch adaptive rule: floor at the start, ceiling at the end."""
    start = index * size // out
    end = -(-(index + 1) * size // out)
    return start, max(end, start + 1)


def adaptive(x: np.ndarray, output_size, reduce: str = "mean") -> np.ndarray:
    """Pool to a fixed output size regardless of the input size."""
    if reduce not in ("mean", "max"):
        raise ValueError("adaptive reduce must be 'mean' or 'max'")
    x = np.asarray(x, dtype=float)
    dims = x.ndim - 2
    if dims < 1:
        raise ValueError("x must have a batch axis and at least one spatial")
    target = _tuple(output_size, dims, "output_size")
    out = np.empty((x.shape[0], *target, x.shape[-1]))
    for index in np.ndindex(*target):
        window = tuple(slice(*_bounds(x.shape[1 + axis], target[axis], start))
                       for axis, start in enumerate(index))
        piece = x[(slice(None), *window)]
        out[(slice(None), *index)] = piece.max(axis=tuple(range(1, dims + 1))) \
            if reduce == "max" else piece.mean(axis=tuple(range(1, dims + 1)))
    return out


def attention_pool(x: np.ndarray, weight: np.ndarray, *,
                   temperature: float = 1.0, return_weights: bool = False):
    """Softmax-weighted average over positions, scores from a linear map.

    `x` is `(batch, *spatial, channels)` and `weight` is `(channels,)`.
    """
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float).reshape(-1)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if x.shape[-1] != weight.size:
        raise ValueError("weight must have one entry per channel")
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    scores = (flat @ weight) / temperature
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    pooled = np.einsum("bn,bnc->bc", weights, flat)
    if return_weights:
        return pooled, weights
    return pooled


def pool_grad(x: np.ndarray, upstream: np.ndarray, size, stride=None,
              mode: str = "max") -> np.ndarray:
    """Route an upstream gradient back to the inputs that produced each pool."""
    x = np.asarray(x, dtype=float)
    dims = x.ndim - 2
    size = _tuple(size, dims, "size")
    stride = size if stride is None else _tuple(stride, dims, "stride")
    if mode not in ("max", "mean"):
        raise ValueError("pool_grad mode must be 'max' or 'mean'")
    gradient = np.zeros_like(x)
    count = float(np.prod(size))
    for index in np.ndindex(*_windows(x, size, stride).shape[1:dims + 1]):
        window = tuple(slice(start * stride[axis],
                             start * stride[axis] + size[axis])
                       for axis, start in enumerate(index))
        piece = x[(slice(None), *window)]
        flows = upstream[(slice(None), *index)]
        if mode == "mean":
            gradient[(slice(None), *window)] += (flows / count)[(slice(None),
                                                                 *([None] * dims))]
            continue
        best = piece.max(axis=tuple(range(1, dims + 1)), keepdims=True)
        hits = piece == best
        hits = hits / hits.sum(axis=tuple(range(1, dims + 1)), keepdims=True)
        gradient[(slice(None), *window)] += \
            hits * flows[(slice(None), *([None] * dims))]
    return gradient


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #
def pool(x: np.ndarray, *, variant: str = "max", size=2, stride=None,
         output_size=2, power: float = 2.0, temperature: float = 1.0,
         weight=None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Pool over the spatial axes of `(batch, *spatial, channels)`."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    x = np.asarray(x, dtype=float)
    if x.ndim < 3:
        raise ValueError("x must be (batch, *spatial, channels)")
    dims = x.ndim - 2
    spatial = x.shape[1:dims + 1]
    if any(length < 1 for length in spatial):
        raise ValueError("the spatial axes must be non-empty")

    if variant in ("global_max", "global_avg"):
        axes = tuple(range(1, dims + 1))
        return x.max(axis=axes) if variant == "global_max" else x.mean(axis=axes)
    if variant == "adaptive":
        return adaptive(x, output_size)
    if variant == "attention_pool":
        if weight is None:
            raise ValueError("attention_pool needs a weight vector")
        return attention_pool(x, weight, temperature=temperature)

    size = _tuple(size, dims, "size")
    stride = size if stride is None else _tuple(stride, dims, "stride")
    if any(size[axis] > spatial[axis] for axis in range(dims)):
        raise ValueError("the window is larger than the input")

    if variant in ("gem", "lp_pool"):
        if power <= 0.0 and variant == "lp_pool":
            raise ValueError("lp_pool needs a positive power")
        how = "gem" if variant == "gem" else "lp"
        return _reduce(_windows(x, size, stride), dims, how, power)

    windows = _windows(x, size, stride)
    if variant in ("max", "avg"):
        return _reduce(windows, dims, "max" if variant == "max" else "mean")
    if np.any(windows < 0.0):
        raise ValueError("stochastic needs non-negative activations")
    if power <= 0.0:
        raise ValueError("stochastic needs a positive power")
    if rng is None:
        raise ValueError("stochastic needs a generator")
    flat = windows.reshape(*windows.shape[:dims + 1], windows.shape[dims + 1],
                           -1)
    weights = flat ** power
    total = weights.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("every window needs a positive activation")
    cumulative = np.cumsum(weights / total, axis=-1)
    draw = rng.random(cumulative.shape[:-1] + (1,))
    chosen = np.minimum((cumulative < draw).sum(axis=-1), flat.shape[-1] - 1)
    return np.take_along_axis(flat, chosen[..., None], axis=-1)[..., 0]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _grid(shape, rng):
    return rng.normal(0.0, 1.0, shape)


def demo() -> None:
    rng = np.random.default_rng(0)
    print("pooling. max and mean are what the words say:")
    x = _grid((2, 5, 6, 3), rng)
    biggest = pool(x, variant="max", size=2)
    average = pool(x, variant="avg", size=2)
    assert biggest.shape == (2, 2, 3, 3)
    manual = np.array([[[[x[b, 2 * i:2 * i + 2, 2 * j:2 * j + 2, c].max()
                          for c in range(3)] for j in range(3)]
                        for i in range(2)] for b in range(2)])
    assert np.allclose(biggest, manual)
    assert np.allclose(average, pool(x, variant="avg", size=2, stride=2))
    print(f"  a {tuple(x.shape)} input pools to {biggest.shape} with a 2x2")
    print("  window, and every output equals the maximum of its own window")

    print("pooling. global pooling collapses space and nothing else:")
    flat_max = pool(x, variant="global_max")
    flat_avg = pool(x, variant="global_avg")
    assert flat_max.shape == (2, 3)
    assert np.allclose(flat_max, x.max(axis=(1, 2)))
    assert np.allclose(flat_avg, x.mean(axis=(1, 2)))
    assert np.allclose(flat_avg,
                       pool(x, variant="avg", size=(5, 6)).reshape(2, 3))
    print(f"  {tuple(x.shape)} becomes {flat_max.shape}, matching numpy's own")
    print("  reduction along the two spatial axes")

    print("pooling. adaptive output sizes use floor starts and ceil ends:")
    small = _grid((1, 7, 1), rng)
    three = pool(small, variant="adaptive", output_size=3)
    assert three.shape == (1, 3, 1)
    assert np.allclose(three[0, 0, 0], small[0, 0:3, 0].mean())
    assert np.allclose(three[0, 1, 0], small[0, 2:5, 0].mean())
    assert np.allclose(three[0, 2, 0], small[0, 4:7, 0].mean())
    identity = pool(small, variant="adaptive", output_size=7)
    assert np.allclose(identity, small)
    halves = pool(_grid((1, 5, 1), rng), variant="adaptive", output_size=2)
    print("  seven samples to three use the intervals [0,3) [2,5) [4,7),")
    print("  which overlap by one at each seam; five to two gives")
    print(f"  {halves.shape[1]} outputs, and seven to seven is the identity")

    print("pooling. attention pooling learns where to look:")
    table = np.zeros((1, 10, 1))
    table[0, 6, 0] = 4.0
    table[0, 3, 0] = 1.0
    table[0, 8, 0] = 0.5
    weight = np.array([0.01])
    start_value = float(attention_pool(table, weight, temperature=0.05)[0, 0])
    losses = []
    for _ in range(80):
        pooled = attention_pool(table, weight, temperature=0.05)
        loss = float((pooled[0, 0] - 4.0) ** 2)
        losses.append(loss)
        slope = (attention_pool(table, weight + 1e-6, temperature=0.05)[0, 0]
                 - pooled[0, 0]) / 1e-6
        weight = weight - 0.5 * slope * 2.0 * (pooled[0, 0] - 4.0)
    final_value = float(attention_pool(table, weight, temperature=0.05)[0, 0])
    assert losses[-1] < 1e-6 * losses[0]
    assert abs(final_value - 4.0) < 1e-3
    _, weights = attention_pool(table, weight, temperature=0.05,
                                return_weights=True)
    assert int(weights[0].argmax()) == 6
    print(f"  gradient descent on one scalar takes the loss from "
          f"{losses[0]:.3f} to {losses[-1]:.2e}")
    print(f"  and the pooled value from {start_value:.3f} to "
          f"{final_value:.4f}, which is the peak")
    print(f"  at a sharp temperature the weight sits on position "
          f"{int(weights[0].argmax())}, which is the maximum")

    print("pooling. the generalised mean interpolates mean and max:")
    positive = np.abs(_grid((1, 4, 4, 1), rng)) + 0.1
    means = {p: float(pool(positive, variant="gem", size=4, power=p)[0, 0, 0, 0])
             for p in (0.0, 1.0, 2.0, 8.0, 64.0)}
    assert abs(means[1.0] - float(positive.mean())) < 1e-12
    assert abs(means[0.0] - float(np.exp(np.log(positive).mean()))) < 1e-12
    assert all(means[a] <= means[b] + 1e-12
               for a, b in zip((0.0, 1.0, 2.0, 8.0), (1.0, 2.0, 8.0, 64.0),
                               strict=True))
    assert means[64.0] <= float(positive.max()) + 1e-12
    print(f"  p=0 geometric {means[0.0]:.4f}, p=1 mean {means[1.0]:.4f},")
    print(f"  p=2 rms {means[2.0]:.4f}, p=64 near max {means[64.0]:.4f}")
    print(f"  the maximum itself is {float(positive.max()):.4f}; the sequence")
    print("  is monotone in p, as Jensen's inequality requires")

    print("pooling. lp pooling is the unnormalised generalised mean:")
    lp = pool(positive, variant="lp_pool", size=4, power=2.0)
    assert np.allclose(lp, pool(positive, variant="gem", size=4, power=2.0)
                       * 16 ** 0.5)
    print("  with a 4x4 window, lp equals gem times sqrt(16): the only")
    print("  difference between the two is the 1/N inside the root")

    print("pooling. stochastic pooling samples instead of maximising:")
    single = np.array([[[0.2], [0.5], [0.9], [0.4]]])
    draws = np.array([pool(single, variant="stochastic", size=4, power=1.0,
                           rng=np.random.default_rng(seed))[0, 0, 0]
                      for seed in range(4000)])
    frequencies = {value: float(np.mean(np.isclose(draws, value)))
                   for value in (0.2, 0.5, 0.9, 0.4)}
    expected = single[0, :, 0] / single.sum()
    for value, share in zip((0.2, 0.5, 0.9, 0.4), expected, strict=True):
        assert abs(frequencies[value] - float(share)) < 0.03
    assert set(np.round(draws, 6)) <= {0.2, 0.5, 0.9, 0.4}
    print(f"  4000 draws land on the four inputs with shares "
          f"{[round(frequencies[v], 3) for v in (0.2, 0.5, 0.9, 0.4)]}")
    print(f"  against the exact probabilities {np.round(expected, 3).tolist()}")
    print("  every draw is one of the window's own values, which max and")
    print("  mean cannot promise")

    print("pooling. the gradient follows the route the forward pass took:")
    tiny = np.array([[[1.0], [3.0], [2.0], [4.0]]])
    upstream = np.array([[[1.0], [10.0]]])
    d_max = pool_grad(tiny, upstream, size=2, mode="max")
    d_mean = pool_grad(tiny, upstream, size=2, mode="mean")
    assert np.allclose(d_max[0, :, 0], [0.0, 1.0, 0.0, 10.0])
    assert np.allclose(d_mean[0, :, 0], [0.5, 0.5, 5.0, 5.0])
    assert abs(float(d_max.sum()) - 11.0) < 1e-12
    print(f"  max sends 1 to index 1 and 10 to index 3: "
          f"{d_max[0, :, 0].tolist()}")
    print(f"  mean splits them: {d_mean[0, :, 0].tolist()}")
    print("  both conserve the total upstream mass of 11")

    bad = [
        lambda: pool(x, variant="median"),
        lambda: pool(x[0, 0], variant="max"),
        lambda: pool(x, variant="max", size=0),
        lambda: pool(x, variant="max", size=(2,)),
        lambda: pool(x, variant="max", size=9),
        lambda: pool(x, variant="max", size=2, stride=(1, 0)),
        lambda: pool(x, variant="gem", size=2, power=-1.0),
        lambda: pool(-positive, variant="gem", size=2),
        lambda: pool(np.zeros_like(positive), variant="gem", size=2,
                     power=0.0),
        lambda: pool(positive, variant="lp_pool", size=2, power=0.0),
        lambda: pool(x, variant="stochastic", size=2),
        lambda: pool(x, variant="stochastic", size=2, power=0.0,
                     rng=rng),
        lambda: pool(-np.abs(x) - 0.1, variant="stochastic", size=2,
                     rng=rng),
        lambda: pool(np.zeros_like(x), variant="stochastic", size=2,
                     rng=rng),
        lambda: pool(x, variant="attention_pool", size=2),
        lambda: pool(x, variant="attention_pool", weight=np.ones(4)),
        lambda: pool(x, variant="attention_pool", weight=np.ones(3),
                     temperature=0.0),
        lambda: adaptive(x, output_size=0),
        lambda: adaptive(x, output_size=(3, 3), reduce="median"),
        lambda: pool_grad(x, upstream, size=2, mode="median"),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, an input with no spatial axis, a zero or wrong-length")
    print("  window, a window larger than the input, a zero stride, a power")
    print("  that is negative, negative inputs to gem or stochastic, a")
    print("  missing generator, an all-zero stochastic window, a missing")
    print("  or wrong-sized attention weight, a zero temperature, an empty")
    print("  or invalid adaptive target, and an unknown gradient mode")
    print("pooling demo passed")


if __name__ == "__main__":
    demo()
