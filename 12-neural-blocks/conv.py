"""Convolution — one engine, ten variants, and its exact adjoint.

Chapter:   12-neural-blocks
Companion: conv.md
Tier 1 · 10 variants: conv1d, conv2d, conv3d, dilated, transposed, depthwise,
separable, group, causal, circular_pad
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("conv1d", "conv2d", "conv3d", "dilated", "transposed", "depthwise",
            "separable", "group", "causal", "circular_pad")


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def _tuple(value, dims: int, name: str):
    """Expand a scalar to one entry per spatial axis."""
    if isinstance(value, (int, np.integer)):
        return (int(value),) * dims
    values = tuple(int(entry) for entry in value)
    if len(values) != dims:
        raise ValueError(f"{name} must have one entry per spatial axis")
    return values


def _padding(size: int, kernel: int, stride: int, dilation: int,
             mode: str) -> tuple[int, int]:
    """Pad widths that produce the requested output length."""
    reach = dilation * (kernel - 1)
    if mode == "valid":
        return 0, 0
    if mode == "same":
        before = reach // 2
        return before, reach - before
    if mode == "causal":
        return reach, 0
    if mode == "full":
        return reach, reach
    raise ValueError(f"unknown padding mode {mode!r}")


def _pad(x: np.ndarray, pads, wrap: bool) -> np.ndarray:
    widths = [(0, 0), *pads, (0, 0)]
    if wrap:
        return np.pad(x, widths, mode="wrap")
    return np.pad(x, widths)


def _patches(x: np.ndarray, kernel, stride, dilation) -> np.ndarray:
    """Every kernel-sized window, as (batch, *output, channels, *kernel).

    `x` must already be padded. The loop is over output positions, which is
    slow but leaves no doubt about what a window is.
    """
    dims = len(kernel)
    shape = tuple((x.shape[1 + axis] - dilation[axis] * (kernel[axis] - 1) - 1)
                  // stride[axis] + 1 for axis in range(dims))
    if any(length < 1 for length in shape):
        raise ValueError("the kernel does not fit the padded input")
    windows = np.empty((x.shape[0], *shape, x.shape[-1], *kernel))
    for index in np.ndindex(*shape):
        window = tuple(slice(start * stride[axis],
                             start * stride[axis]
                             + dilation[axis] * (kernel[axis] - 1) + 1,
                             dilation[axis])
                       for axis, start in enumerate(index))
        windows[(slice(None), *index)] = np.moveaxis(
            x[(slice(None), *window)], -1, 1)
    return windows


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def convolve(x: np.ndarray, weight: np.ndarray, *, stride=1, dilation=1,
             groups: int = 1, padding: str = "same",
             circular: bool = False) -> np.ndarray:
    """N-dimensional cross-correlation.

    Parameters
    ----------
    x : ndarray
        `(batch, *spatial, in_channels)`.
    weight : ndarray
        `(out_channels, in_channels / groups, *kernel)`.
    stride, dilation : int or tuple
        One value per spatial axis, or one value for all of them.
    groups : int
        Split the channels into this many independent blocks.
    padding : str
        `valid`, `same`, `causal` or `full`.
    circular : bool
        Wrap the border instead of zero-padding it.

    Returns
    -------
    ndarray
        `(batch, *output, out_channels)`.
    """
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)
    dims = weight.ndim - 2
    if dims not in (1, 2, 3):
        raise ValueError("weight must have one to three spatial axes")
    if x.ndim != dims + 2:
        raise ValueError(f"x must have {dims + 2} axes, got {x.ndim}")
    if weight.shape[1] * groups != x.shape[-1]:
        raise ValueError("weight and groups do not match the input channels")
    if x.shape[-1] % groups or weight.shape[0] % groups:
        raise ValueError("channels must divide evenly into groups")
    stride = _tuple(stride, dims, "stride")
    dilation = _tuple(dilation, dims, "dilation")
    kernel = weight.shape[2:]
    pads = [_padding(x.shape[1 + axis], kernel[axis], stride[axis],
                     dilation[axis], padding) for axis in range(dims)]
    padded = _pad(x, pads, circular)
    windows = _patches(padded, kernel, stride, dilation)
    width = int(np.prod(kernel))
    channels = x.shape[-1]
    flat = windows.reshape(*windows.shape[:dims + 1], channels, width)
    blocks = flat.reshape(*flat.shape[:dims + 1], groups, channels // groups,
                          width)
    kernels = weight.reshape(groups, weight.shape[0] // groups,
                             channels // groups, width)
    out = np.einsum("...gck,gock->...go", blocks, kernels)
    return out.reshape(*out.shape[:dims + 1], weight.shape[0])


def transpose(y: np.ndarray, weight: np.ndarray, *, stride=1,
              dilation=1) -> np.ndarray:
    """The adjoint of `convolve` with `padding="valid"` and one group.

    Scatter-adds every output position back onto the inputs it read, which is
    exactly what the backward pass of a convolution does.
    """
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    dims = weight.ndim - 2
    if y.ndim != dims + 2:
        raise ValueError(f"y must have {dims + 2} axes, got {y.ndim}")
    if y.shape[-1] != weight.shape[0]:
        raise ValueError("y has the wrong number of channels")
    stride = _tuple(stride, dims, "stride")
    dilation = _tuple(dilation, dims, "dilation")
    kernel = weight.shape[2:]
    length = tuple((y.shape[1 + axis] - 1) * stride[axis]
                   + dilation[axis] * (kernel[axis] - 1) + 1
                   for axis in range(dims))
    out = np.zeros((y.shape[0], *length, weight.shape[1]))
    spread = tuple(range(2, 2 + dims))
    for index in np.ndindex(*y.shape[1:1 + dims]):
        window = tuple(slice(start * stride[axis],
                             start * stride[axis]
                             + dilation[axis] * (kernel[axis] - 1) + 1,
                             dilation[axis])
                       for axis, start in enumerate(index))
        piece = np.einsum("bo,oc...->bc...", y[(slice(None), *index)], weight)
        out[(slice(None), *window)] += piece.transpose(0, *spread, 1)
    return out


def gradients(x: np.ndarray, upstream: np.ndarray, weight: np.ndarray, *,
              stride=1, dilation=1):
    """Gradients of the valid convolution with respect to `x` and `weight`."""
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)
    dims = weight.ndim - 2
    stride = _tuple(stride, dims, "stride")
    dilation = _tuple(dilation, dims, "dilation")
    kernel = weight.shape[2:]
    windows = _patches(x, kernel, stride, dilation)
    flat = windows.reshape(*windows.shape[:dims + 1], x.shape[-1], -1)
    axes = list(range(dims + 1))
    d_weight = np.tensordot(flat, upstream, axes=(axes, axes))
    d_weight = d_weight.transpose(2, 0, 1)
    d_input = transpose(upstream, weight, stride=stride, dilation=dilation)
    return d_input, d_weight.reshape(weight.shape)


# --------------------------------------------------------------------------- #
# The variants
# --------------------------------------------------------------------------- #
def conv(x: np.ndarray, weight, *, variant: str = "conv2d", stride=1,
         dilation=1, groups: int = 1, padding: str = "same") -> np.ndarray:
    """Dispatch a convolution by variant name.

    `separable` takes `weight` as a `(depth, point)` pair; every other variant
    takes a single array. `transposed` treats `weight` as `(out, in, *kernel)`
    and returns the adjoint of the valid convolution.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    x = np.asarray(x, dtype=float)
    if variant in ("conv1d", "conv2d", "conv3d"):
        wanted = {"conv1d": 1, "conv2d": 2, "conv3d": 3}[variant]
        if np.ndim(weight) != wanted + 2:
            raise ValueError(f"{variant} needs a {wanted}-D kernel")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding=padding)
    if variant == "dilated":
        if dilation == 1:
            raise ValueError("dilated needs a dilation above one")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding=padding)
    if variant == "group":
        if groups < 2:
            raise ValueError("group needs at least two groups")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding=padding)
    if variant == "depthwise":
        channels = x.shape[-1]
        if np.shape(weight)[0] != channels or np.shape(weight)[1] != 1:
            raise ValueError("depthwise needs one kernel per input channel")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=channels, padding=padding)
    if variant == "separable":
        depth, point = weight
        mixed = convolve(x, depth, stride=stride, dilation=dilation,
                         groups=x.shape[-1], padding=padding)
        return convolve(mixed, point, padding="same")
    if variant == "causal":
        if x.ndim != 3:
            raise ValueError("causal is a one-dimensional padding mode")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding="causal")
    if variant == "circular_pad":
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding=padding, circular=True)
    if padding == "causal":
        raise ValueError("transposed has no causal form")
    return transpose(x, weight, stride=stride, dilation=dilation)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _box(kernel: int) -> np.ndarray:
    return np.ones((1, 1, kernel)) / kernel


def demo() -> None:
    print("conv. a delta in, the kernel out:")
    rng = np.random.default_rng(0)
    kernel = rng.normal(0.0, 1.0, (3, 2, 4))
    delta = np.zeros((1, 12, 2))
    delta[0, 5] = 1.0
    answer = convolve(delta, kernel, padding="valid")
    taps = answer[0, 2:6]
    expected = np.stack([kernel[:, :, k].sum(axis=1) for k in (3, 2, 1, 0)])
    assert np.allclose(taps, expected)
    print(f"  a delta at position 5 leaves taps at 2..5 of a {answer.shape[1]}")
    print("  long valid output, with the taps in reverse order because this")
    print("  is cross-correlation rather than the textbook flip")

    print("conv. a box kernel is a moving average:")
    signal = rng.normal(0.0, 1.0, (1, 20, 1))
    smoothed = convolve(signal, _box(4), padding="valid")[0, :, 0]
    assert np.allclose(smoothed, np.convolve(signal[0, :, 0],
                                             np.ones(4) / 4, mode="valid"))
    print(f"  four taps reproduce numpy's own valid convolution on "
          f"{signal.shape[1]} points")

    print("conv. stride, dilation and padding reshape the answer:")
    x2 = rng.normal(0.0, 1.0, (2, 9, 9, 3))
    w2 = rng.normal(0.0, 0.2, (4, 3, 3, 3))
    shapes = {"same": convolve(x2, w2).shape,
              "valid": convolve(x2, w2, padding="valid").shape,
              "full": convolve(x2, w2, padding="full").shape,
              "stride 2": convolve(x2, w2, stride=2).shape,
              "dilation 2": convolve(x2, w2, dilation=2,
                                     padding="valid").shape}
    for name, shape in shapes.items():
        print(f"    {name:<10s} {shape}")
    assert shapes["same"][1:3] == (9, 9)
    assert shapes["valid"][1:3] == (7, 7)
    assert shapes["full"][1:3] == (11, 11)
    assert shapes["stride 2"][1:3] == (5, 5)
    assert shapes["dilation 2"][1:3] == (5, 5)

    print("conv. the transpose is the exact adjoint:")
    x = rng.normal(0.0, 1.0, (2, 10, 3))
    w = rng.normal(0.0, 0.5, (4, 3, 3))
    y = rng.normal(0.0, 1.0, (2, 8, 4))
    forward = float(np.sum(convolve(x, w, padding="valid") * y))
    backward = float(np.sum(x * transpose(y, w)))
    assert abs(forward - backward) < 1e-9
    assert abs(forward) > 1.0
    print(f"  <conv(x), y> = {forward:.6f} and <x, conv'(y)> = "
          f"{backward:.6f}")
    print("  the inner products agree to machine precision, which is what")
    print("  makes the transpose usable as the backward pass")

    print("conv. a group convolution is independent convs stacked:")
    xg = rng.normal(0.0, 1.0, (1, 7, 6))
    grouped_w = rng.normal(0.0, 0.3, (6, 2, 3))
    grouped = convolve(xg, grouped_w, groups=3, padding="valid")
    pieces = np.concatenate([
        convolve(xg[..., 2 * g:2 * g + 2], grouped_w[2 * g:2 * g + 2],
                 padding="valid")
        for g in range(3)], axis=-1)
    assert np.allclose(grouped, pieces)
    depth_w = rng.normal(0.0, 0.3, (6, 1, 3))
    depth = convolve(xg, depth_w, groups=6, padding="valid")
    assert np.allclose(depth, np.concatenate([
        convolve(xg[..., g:g + 1], depth_w[g:g + 1], padding="valid")
        for g in range(6)], axis=-1))
    print("  three groups of two channels match three separate convolutions")
    print(f"  and six groups reduce to one {depth_w.shape[0]}x"
          f"{depth_w.shape[2]} kernel")
    print("  per channel, which is the depthwise case")

    print("conv. separable is depthwise then pointwise:")
    point_w = rng.normal(0.0, 0.4, (5, 6, 1))
    together = conv(xg, (depth_w, point_w), variant="separable",
                    padding="valid")
    apart = convolve(convolve(xg, depth_w, groups=6, padding="valid"),
                     point_w, padding="valid")
    assert np.allclose(together, apart)
    full = int(5 * 6 * 3)
    split = int(6 * 3 + 5 * 6)
    print("  the two-step form matches the dispatch exactly")
    print(f"  a full {full}-weight layer becomes {split} weights: "
          f"{split / full:.2f} of the cost")

    print("conv. causal padding cannot see the future:")
    series = rng.normal(0.0, 1.0, (1, 30, 2))
    filter_1d = rng.normal(0.0, 0.4, (3, 2, 4))
    causal = conv(series, filter_1d, variant="causal")
    assert causal.shape[1] == series.shape[1]
    changed = series.copy()
    changed[0, 20:] += 100.0
    after = conv(changed, filter_1d, variant="causal")
    assert np.allclose(after[0, :20], causal[0, :20])
    assert not np.allclose(after[0, 20:], causal[0, 20:])
    jump = float(np.abs(after[0, 20:] - causal[0, 20:]).max())
    print("  outputs before step 20 are unchanged when the future is")
    print(f"  perturbed, and a {jump:.2f} shift arrives exactly at step 20")

    print("conv. circular padding commutes with rolling:")
    xc = rng.normal(0.0, 1.0, (1, 12, 1))
    wc = rng.normal(0.0, 0.4, (1, 1, 3))
    rolled = np.roll(xc, 4, axis=1)
    direct = convolve(xc, wc, circular=True)
    shifted = convolve(rolled, wc, circular=True)
    assert np.allclose(shifted, np.roll(direct, 4, axis=1))
    print("  a four-step roll of the input is a four-step roll of the output")
    print("  with no seam anywhere, which zero padding cannot promise")

    print("conv. the backward pass checks against finite differences:")
    small_x = rng.normal(0.0, 1.0, (2, 6, 2))
    small_w = rng.normal(0.0, 0.5, (3, 2, 3))
    upstream = rng.normal(0.0, 1.0, (2, 4, 3))
    d_x, d_w = gradients(small_x, upstream, small_w)
    eps = 1e-6
    worst = 0.0
    for index in np.ndindex(*small_w.shape):
        old = small_w[index]
        small_w[index] = old + eps
        plus = float(np.sum(convolve(small_x, small_w, padding="valid")
                            * upstream))
        small_w[index] = old - eps
        minus = float(np.sum(convolve(small_x, small_w, padding="valid")
                            * upstream))
        small_w[index] = old
        worst = max(worst, abs((plus - minus) / (2 * eps) - d_w[index]))
    for index in [(0, 0, 0), (1, 3, 1), (0, 5, 1)]:
        old = small_x[index]
        small_x[index] = old + eps
        plus = float(np.sum(convolve(small_x, small_w, padding="valid")
                            * upstream))
        small_x[index] = old - eps
        minus = float(np.sum(convolve(small_x, small_w, padding="valid")
                            * upstream))
        small_x[index] = old
        worst = max(worst, abs((plus - minus) / (2 * eps) - d_x[index]))
    assert worst < 1e-6
    print(f"  the worst of {small_w.size + 3} finite differences is "
          f"{worst:.2e}")
    print("  d_input is the transpose of the forward pass; d_weight is a")
    print("  windowed correlation with the upstream gradient")

    bad = [
        lambda: conv(x2, w2, variant="conv4d"),
        lambda: conv(x2, w2, variant="conv1d"),
        lambda: conv(x2, w2, variant="conv3d"),
        lambda: convolve(x2, w2, padding="cheap"),
        lambda: convolve(x2, w2, padding="valid", stride=(2,)),
        lambda: convolve(x2, w2, padding="valid", groups=5),
        lambda: convolve(x2[:, :, :, :1], w2),
        lambda: convolve(x2, w2[:, :2]),
        lambda: convolve(x2, w2, padding="valid", dilation=(1, 9)),
        lambda: transpose(y, w, stride=(1, 2)),
        lambda: transpose(y[..., :2], w),
        lambda: conv(x2, w2, variant="dilated"),
        lambda: conv(x2, w2, variant="group", groups=1),
        lambda: conv(x2, w2, variant="depthwise"),
        lambda: conv(xg, (point_w, depth_w), variant="separable"),
        lambda: conv(np.ones((2, 8, 7, 4)), w2, variant="causal"),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a one-dimensional kernel given to a two- or")
    print("  three-dimensional variant, an unknown padding mode, a stride or")
    print("  dilation with the wrong number of entries, a group count that")
    print("  does not divide the channels, a kernel that does not match the")
    print("  input channels, a dilation wider than the input, transposed")
    print("  arguments with the wrong rank or channel count, a dilation of")
    print("  one, a single group, a depthwise kernel with the wrong shape,")
    print("  a separable pair in the wrong order, and a causal call on")
    print("  two-dimensional data")
    print("conv demo passed")


if __name__ == "__main__":
    demo()
