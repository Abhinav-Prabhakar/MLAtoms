"""ConvNeXt blocks — depthwise spatial mixing, GRN and layer scale.

Chapter:   12-neural-blocks
Companion: convnext.md
Tier 2 · 4 variants: convnext, convnext_v2, grn, patchify_stem
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("convnext", "convnext_v2", "grn", "patchify_stem")
BLOCKS = ("convnext", "convnext_v2", "grn")

_GELU = 0.7978845608028654
_CUBIC = 0.044715
GRN_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _gelu(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    inner = _GELU * (x + _CUBIC * x ** 3)
    tanh = np.tanh(inner)
    return (0.5 * (1.0 + tanh)
            + 0.5 * x * (1.0 - tanh ** 2)
            * _GELU * (1.0 + 3.0 * _CUBIC * x ** 2))


def _layer_norm(x: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv, {"x": x, "mean": mean, "inv": inv}


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    x, mean, inv = cache["x"], cache["mean"], cache["inv"]
    centred = x - mean
    shift = gradient.mean(axis=-1, keepdims=True)
    scale = (gradient * centred).mean(axis=-1, keepdims=True)
    return inv * (gradient - shift - centred * inv ** 2 * scale)


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #
class ConvNeXt:
    """A patchify stem followed by one depthwise block.

    `forward(x)` takes `(batch, height, width, channels)` with both spatial
    sizes divisible by the stem stride, and returns the feature map. The
    `patchify_stem` variant stops after the stem.
    """

    def __init__(self, variant: str = "convnext", *, channels_in: int = 3,
                 dim: int = 8, hidden: int = 32, kernel: int = 7,
                 stride: int = 4, scale: float = 1e-6, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(channels_in, dim, hidden, kernel, stride) < 1:
            raise ValueError("all sizes must be positive")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.variant = variant
        self.channels_in = channels_in
        self.dim = dim
        self.hidden = hidden
        self.kernel = kernel
        self.stride = stride
        self.scale = scale
        self.seed = seed
        rng = np.random.default_rng(seed)
        p: dict[str, np.ndarray] = {}
        fan_in = stride * stride * channels_in
        p["stem.W"] = rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                                 (fan_in, dim))
        p["stem.b"] = np.zeros(dim)
        if variant in BLOCKS:
            p["dw.W"] = rng.normal(0.0, 1.0 / np.sqrt(kernel * kernel),
                                   (dim, kernel, kernel))
            p["dw.b"] = np.zeros(dim)
            p["expand.W"] = rng.normal(0.0, 1.0 / np.sqrt(dim),
                                       (dim, hidden))
            p["expand.b"] = np.zeros(hidden)
            p["project.W"] = rng.normal(0.0, 1.0 / np.sqrt(hidden),
                                        (hidden, dim))
            p["project.b"] = np.zeros(dim)
            if variant in ("convnext", "grn"):
                p["layer_scale"] = np.full(dim, scale)
            if variant in ("convnext_v2", "grn"):
                p["grn.gamma"] = np.zeros(hidden)
                p["grn.beta"] = np.ones(hidden)
        self.parameters = p
        self.gradients = {key: np.zeros_like(value)
                          for key, value in p.items()}

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @property
    def grid(self) -> int:
        return self.stride

    # -- validation ---------------------------------------------------------- #
    def _check(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 4:
            raise ValueError("x must be (batch, height, width, channels)")
        if x.shape[3] != self.channels_in:
            raise ValueError("x has the wrong channel count")
        if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
            raise ValueError("batch and spatial sizes must be positive")
        if x.shape[1] % self.stride or x.shape[2] % self.stride:
            raise ValueError("height and width must divide by the stride")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        return x

    # -- stem ---------------------------------------------------------------- #
    def _patchify(self, x: np.ndarray) -> np.ndarray:
        batch, height, width, channels = x.shape
        size = self.stride
        rows, columns = height // size, width // size
        patches = x.reshape(batch, rows, size, columns, size, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(batch, rows, columns, size * size * channels)

    def _unpatchify(self, gradient: np.ndarray) -> np.ndarray:
        batch, rows, columns, _ = gradient.shape
        size, channels = self.stride, self.channels_in
        blocks = gradient.reshape(batch, rows, columns, size, size, channels)
        blocks = blocks.transpose(0, 1, 3, 2, 4, 5)
        return blocks.reshape(batch, rows * size, columns * size, channels)

    def _stem_forward(self, x: np.ndarray):
        p = self.parameters
        patches = self._patchify(x)
        out = patches @ p["stem.W"] + p["stem.b"]
        return out, {"patches": patches}

    def _stem_backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        g, p = self.gradients, self.parameters
        patches = cache["patches"]
        g["stem.W"] += patches.reshape(-1, patches.shape[-1]).T @ gradient.reshape(
            -1, self.dim)
        g["stem.b"] += gradient.sum(axis=(0, 1, 2))
        return self._unpatchify(gradient @ p["stem.W"].T)

    # -- depthwise convolution ---------------------------------------------- #
    def _depthwise(self, x: np.ndarray):
        p = self.parameters
        pad = self.kernel // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        windows = np.stack(
            [np.stack([padded[:, i:i + x.shape[1], j:j + x.shape[2], :]
                       for j in range(self.kernel)], axis=-1)
             for i in range(self.kernel)], axis=-2)
        out = np.einsum("bhwcij,cij->bhwc", windows, p["dw.W"])
        return out + p["dw.b"], {"windows": windows}

    def _depthwise_backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        g, p = self.gradients, self.parameters
        windows = cache["windows"]
        g["dw.W"] += np.einsum("bhwc,bhwcij->cij", gradient, windows)
        g["dw.b"] += gradient.sum(axis=(0, 1, 2))
        pad = self.kernel // 2
        batch, height, width = gradient.shape[:3]
        d_padded = np.zeros((batch, height + 2 * pad, width + 2 * pad,
                             self.dim))
        for i in range(self.kernel):
            for j in range(self.kernel):
                d_padded[:, i:i + height, j:j + width, :] += (
                    gradient * p["dw.W"][None, None, None, :, i, j])
        return d_padded[:, pad:pad + height, pad:pad + width, :]

    # -- global response normalisation --------------------------------------- #
    def _grn(self, x: np.ndarray):
        p = self.parameters
        norms = np.sqrt((x ** 2).sum(axis=(1, 2), keepdims=True))
        mean = norms.mean(axis=-1, keepdims=True)
        normalised = norms / (mean + GRN_EPS)
        out = x + p["grn.gamma"] * x * normalised + p["grn.beta"]
        return out, {"x": x, "norms": norms, "mean": mean,
                     "normalised": normalised}

    def _grn_backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        g, p = self.gradients, self.parameters
        x, norms = cache["x"], cache["norms"]
        normalised, mean = cache["normalised"], cache["mean"]
        g["grn.gamma"] += (gradient * x * normalised).sum(axis=(0, 1, 2))
        g["grn.beta"] += gradient.sum(axis=(0, 1, 2))
        denominator = mean + GRN_EPS
        totals = (gradient * x).sum(axis=(1, 2), keepdims=True)
        weighted = p["grn.gamma"] * totals
        share = (weighted * norms).sum(axis=-1, keepdims=True)
        bracket = (weighted / denominator
                   - share / denominator ** 2 / self.hidden)
        d_x = gradient * (1.0 + p["grn.gamma"] * normalised)
        d_x += bracket * x / np.maximum(norms, GRN_EPS)
        return d_x

    # -- forward ------------------------------------------------------------- #
    def _block(self, x: np.ndarray):
        p = self.parameters
        depth, dw_cache = self._depthwise(x)
        norm, norm_cache = _layer_norm(depth)
        pre = norm @ p["expand.W"] + p["expand.b"]
        hidden = _gelu(pre)
        cache = {"x": x, "dw": dw_cache, "norm": norm,
                 "norm_cache": norm_cache, "pre": pre, "hidden": hidden}
        if self.variant in ("convnext_v2", "grn"):
            hidden, grn_cache = self._grn(hidden)
            cache["grn"] = grn_cache
        cache["project_input"] = hidden
        projected = hidden @ p["project.W"] + p["project.b"]
        cache["projected"] = projected
        if self.variant in ("convnext", "grn"):
            projected = p["layer_scale"] * projected
        return x + projected, cache

    def forward(self, x: np.ndarray):
        x = self._check(x)
        stem, stem_cache = self._stem_forward(x)
        if self.variant == "patchify_stem":
            return stem, {"x": x, "stem": stem_cache}
        out, block_cache = self._block(stem)
        return out, {"x": x, "stem": stem_cache, "block": block_cache}

    # -- backward ------------------------------------------------------------ #
    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        expected = (cache["x"].shape[0], cache["x"].shape[1] // self.stride,
                    cache["x"].shape[2] // self.stride, self.dim)
        if gradient.shape != expected:
            raise ValueError("gradient has the wrong shape")
        for key in self.gradients:
            self.gradients[key][...] = 0.0
        if self.variant == "patchify_stem":
            return self._stem_backward(gradient, cache["stem"])
        g, p = self.gradients, self.parameters
        block = cache["block"]
        d_projected = gradient
        if self.variant in ("convnext", "grn"):
            g["layer_scale"] += (gradient * block["projected"]).sum(
                axis=(0, 1, 2))
            d_projected = gradient * p["layer_scale"]
        g["project.W"] += block["project_input"].reshape(
            -1, self.hidden).T @ d_projected.reshape(-1, self.dim)
        g["project.b"] += d_projected.sum(axis=(0, 1, 2))
        d_hidden = (d_projected @ p["project.W"].T)
        if self.variant in ("convnext_v2", "grn"):
            d_hidden = self._grn_backward(d_hidden, block["grn"])
        d_pre = d_hidden * _gelu_grad(block["pre"])
        g["expand.W"] += block["norm"].reshape(
            -1, self.dim).T @ d_pre.reshape(-1, self.hidden)
        g["expand.b"] += d_pre.sum(axis=(0, 1, 2))
        d_norm = d_pre @ p["expand.W"].T
        d_depth = _layer_norm_backward(d_norm, block["norm_cache"])
        d_stem = gradient + self._depthwise_backward(d_depth, block["dw"])
        return self._stem_backward(d_stem, cache["stem"])


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    rng = np.random.default_rng(1)
    image = rng.normal(size=(2, 8, 8, 3))

    print("convnext. the stem patchifies, the block refines:")
    for variant in VARIANTS:
        model = ConvNeXt(variant, channels_in=3, dim=8, hidden=32, kernel=7,
                         stride=4, seed=2)
        out, _ = model.forward(image)
        assert out.shape == (2, 2, 2, 8)
        print(f"    {variant:<13s} {model.n_parameters:4d} parameters")
    stem = ConvNeXt("patchify_stem", channels_in=3, dim=8, seed=2)
    assert stem.n_parameters == 392
    print("  8x8x3 becomes 2x2x8 through 4x4 patches; the stem alone costs")
    print("  392 numbers, and the V1 block adds 960")

    print("convnext. patchify is a linear map over each 4x4x3 block:")
    model = ConvNeXt("patchify_stem", channels_in=3, dim=8, stride=4, seed=3)
    out, _ = model.forward(image)
    patches = model._patchify(image)
    assert patches.shape == (2, 2, 2, 48)
    manual = patches @ model.parameters["stem.W"] + model.parameters["stem.b"]
    assert np.allclose(out, manual)
    block = image.reshape(2, 2, 4, 2, 4, 3).transpose(0, 1, 3, 2, 4, 5)
    flat = block.reshape(2, 2, 2, 48)
    assert np.allclose(patches, flat)
    print("  the same 48 numbers per patch, so a stride-4 convolution and a")
    print("  patch extraction followed by a linear map are the same operation")

    print("convnext. the depthwise convolution is local:")
    model = ConvNeXt("convnext", channels_in=3, dim=8, hidden=32, kernel=7,
                     stride=4, seed=4)
    features = rng.normal(size=(2, 8, 8, 8))
    base, _ = model._depthwise(features)
    moved = features.copy()
    moved[:, 2, 2, :] += 1.0
    shifted, _ = model._depthwise(moved)
    difference = np.abs(shifted - base).sum(axis=-1)
    assert difference[0, 2, 2] > 0.0
    radius = 3
    rows = np.arange(8)
    outside = np.abs(rows[:, None] - 2) > radius
    outside = outside | (np.abs(rows[None, :] - 2) > radius)
    assert np.allclose(difference[0][outside], 0.0)
    print("  one feature pixel reaches exactly the 7x7 window around it: a")
    print("  radius of 3 in an 8x8 grid, and nothing outside")

    print("convnext. layer scale starts the block almost at the identity:")
    for variant in ("convnext", "convnext_v2"):
        model = ConvNeXt(variant, channels_in=3, dim=8, hidden=32, kernel=7,
                         stride=4, seed=5)
        stem_out = model._stem_forward(image)[0]
        out, _ = model.forward(image)
        change = float(np.linalg.norm(out - stem_out))
        size = float(np.linalg.norm(stem_out))
        print(f"    {variant:<13s} relative change {change / size:.2e}")
        if variant == "convnext":
            assert change / size < 1e-4
    print("  gamma = 1e-6 makes a V1 block a small perturbation; V2 drops the")
    print("  scale and lets the block act at full strength")

    print("convnext. GRN normalises the channel norms over the image:")
    model = ConvNeXt("grn", channels_in=3, dim=8, hidden=32, kernel=7,
                     stride=4, seed=6)
    out, cache = model.forward(image)
    grn = cache["block"]["grn"]
    norms = grn["norms"]
    assert np.allclose(norms / norms.mean(axis=-1, keepdims=True),
                       grn["normalised"])
    assert np.allclose(grn["normalised"].mean(axis=-1), 1.0)
    assert np.allclose(model.parameters["grn.beta"], 1.0)
    assert np.all(model.parameters["grn.gamma"] == 0.0)
    print("  each channel is divided by the mean channel norm, so the")
    print("  normalised norms average one; gamma starts at zero and beta at")
    print("  one, which makes GRN a shift by one at initialisation")

    print("convnext. finite differences agree with the analytic gradients:")
    small = rng.normal(size=(2, 8, 8, 2))
    target = rng.normal(size=(2, 2, 2, 3))
    for variant in VARIANTS:
        model = ConvNeXt(variant, channels_in=2, dim=3, hidden=5, kernel=3,
                         stride=4, seed=7)
        if "layer_scale" in model.parameters:
            model.parameters["layer_scale"] = np.ones(3)
        if "grn.gamma" in model.parameters:
            model.parameters["grn.gamma"] = rng.normal(0.0, 0.5, 5)
        out, cache = model.forward(small)
        d_x = model.backward(target, cache)
        analytic = {key: value.copy() for key, value in model.gradients.items()}
        worst = 0.0
        for key, value in model.parameters.items():
            for index in np.ndindex(value.shape):
                old = value[index]
                value[index] = old + 1e-6
                plus = model.forward(small)[0]
                value[index] = old - 1e-6
                minus = model.forward(small)[0]
                value[index] = old
                numeric = float(np.sum((plus - minus) * target)) / 2e-6
                worst = max(worst, abs(numeric - analytic[key][index]))
        for index in np.ndindex(small.shape):
            old = small[index]
            small[index] = old + 1e-6
            plus = model.forward(small)[0]
            small[index] = old - 1e-6
            minus = model.forward(small)[0]
            small[index] = old
            numeric = float(np.sum((plus - minus) * target)) / 2e-6
            worst = max(worst, abs(numeric - d_x[index]))
        assert worst < 1e-6
        print(f"  {variant:<13s} worst error {worst:.1e}")

    print("convnext. a short fit reduces the loss:")
    model = ConvNeXt("convnext", channels_in=3, dim=8, hidden=32, kernel=7,
                     stride=4, seed=8)
    goal = rng.normal(size=(2, 2, 2, 8))
    start = 0.5 * float(np.sum((model.forward(image)[0] - goal) ** 2))
    for _ in range(250):
        out, cache = model.forward(image)
        model.backward(out - goal, cache)
        for key in model.gradients:
            model.parameters[key] -= 0.02 * model.gradients[key]
    end = 0.5 * float(np.sum((model.forward(image)[0] - goal) ** 2))
    assert end < 0.1 * start
    print(f"  250 steps take the loss from {start:.3f} to {end:.4f}")

    print("convnext. malformed calls are refused:")
    guard = ConvNeXt("convnext", channels_in=3, dim=8, hidden=32, kernel=7,
                     stride=4, seed=9)
    bad = [
        lambda: ConvNeXt("convnext_v3"),
        lambda: ConvNeXt("convnext", channels_in=0),
        lambda: ConvNeXt("convnext", dim=0),
        lambda: ConvNeXt("convnext", hidden=0),
        lambda: ConvNeXt("convnext", kernel=0),
        lambda: ConvNeXt("convnext", kernel=4),
        lambda: ConvNeXt("convnext", stride=0),
        lambda: ConvNeXt("convnext", scale=-1.0),
        lambda: guard.forward(image[0]),
        lambda: guard.forward(image[..., :2]),
        lambda: guard.forward(np.zeros((2, 7, 8, 3))),
        lambda: guard.forward(np.zeros((0, 8, 8, 3))),
        lambda: guard.forward(np.full_like(image, np.nan)),
        lambda: guard.backward(np.zeros((2, 2, 2, 4)), guard.forward(image)[1]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, an")
    print("  even or zero kernel, a negative scale, a three-dimensional")
    print("  input, the wrong channel count, a size that does not divide by")
    print("  the stride, an empty batch, non-finite values, and a wrong")
    print("  gradient shape")
    print("convnext demo passed")


if __name__ == "__main__":
    demo()
