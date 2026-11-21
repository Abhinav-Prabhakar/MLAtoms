"""Quantisation — ten ways to store a weight in fewer bits.

Chapter:   13-training
Companion: quantization.md
Tier 2 · 10 variants: absmax, zero_point, per_channel, per_group, ptq, qat,
gptq, awq, int4, int8
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("absmax", "zero_point", "per_channel", "per_group", "ptq", "qat",
            "gptq", "awq", "int4", "int8")
MATRIX_NAMES = ("W1", "W2")
PRESETS = {
    "absmax": {"bits": 8, "mode": "tensor", "symmetric": True, "acts": False},
    "zero_point": {"bits": 8, "mode": "tensor", "symmetric": False,
                   "acts": False},
    "per_channel": {"bits": 8, "mode": "channel", "symmetric": True,
                    "acts": False},
    "per_group": {"bits": 4, "mode": "group", "symmetric": True,
                  "acts": False},
    "ptq": {"bits": 4, "mode": "group", "symmetric": True, "acts": False,
            "clip": 99.5},
    "qat": {"bits": 4, "mode": "group", "symmetric": True, "acts": False},
    "gptq": {"bits": 4, "mode": "group", "symmetric": True, "acts": False,
             "damping": 0.01},
    "awq": {"bits": 4, "mode": "group", "symmetric": True, "acts": False,
            "alpha": 0.5},
    "int4": {"bits": 4, "mode": "group", "symmetric": True, "acts": True},
    "int8": {"bits": 8, "mode": "channel", "symmetric": True, "acts": True},
}
PROCEDURAL = ("ptq", "qat", "gptq", "awq", "int4", "int8")


# --------------------------------------------------------------------------- #
# The quantiser
# --------------------------------------------------------------------------- #
def quantise(weight: np.ndarray, *, bits: int = 8, mode: str = "tensor",
             group: int = 16, symmetric: bool = True,
             clip: float | None = None) -> dict:
    """Round a matrix onto a `bits`-wide grid.

    `mode` picks the scale granularity: one scale for the tensor, one per row,
    or one per group of `group` entries along the second axis. The returned
    `values` are dequantised and can be used in place of the original.
    """
    values = np.asarray(weight, dtype=float)
    if values.ndim != 2:
        raise ValueError("weight must be a matrix")
    if values.size < 1:
        raise ValueError("weight must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("weight must be finite")
    if not 2 <= bits <= 16:
        raise ValueError("bits must lie in [2, 16]")
    if group < 1:
        raise ValueError("group must be positive")
    if clip is not None and not 0.0 < clip <= 100.0:
        raise ValueError("clip must lie in (0, 100]")
    rows, cols = values.shape
    if clip is not None:
        limit = float(np.percentile(np.abs(values), clip))
        values = np.clip(values, -limit, limit)
    levels = 2 ** bits - 1
    qmax = 2 ** (bits - 1) - 1
    if mode == "tensor":
        blocks = values.reshape(1, 1, -1)
    elif mode == "channel":
        blocks = values.reshape(rows, 1, cols)
    elif mode == "group":
        groups = (cols + group - 1) // group
        width = groups * group
        padded = np.zeros((rows, width))
        padded[:, :cols] = values
        blocks = padded.reshape(rows, groups, group)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if symmetric:
        scale = np.abs(blocks).max(axis=-1, keepdims=True) / qmax
        scale = np.where(scale > 0.0, scale, 1.0)
        codes = np.clip(np.round(blocks / scale), -qmax, qmax)
        dequant = codes * scale
        zeros = None
    else:
        low = blocks.min(axis=-1, keepdims=True)
        high = blocks.max(axis=-1, keepdims=True)
        scale = (high - low) / levels
        scale = np.where(scale > 0.0, scale, 1.0)
        zero = np.clip(np.round(-low / scale), 0, levels)
        codes = np.clip(np.round(blocks / scale) + zero, 0, levels)
        dequant = (codes - zero) * scale
        zeros = zero
    flat = dequant.reshape(rows, -1)[:, :cols]
    return {"values": flat, "codes": codes.astype(np.int32), "scale": scale,
            "zero": zeros, "bits": bits, "mode": mode, "group": group,
            "shape": values.shape, "count": int(values.size)}


def quantised_bytes(info: dict) -> int:
    """Codes plus fp16 scales and zero points, plus a small header."""
    code = (info["count"] * info["bits"] + 7) // 8
    total = code + int(info["scale"].size) * 2
    if info["zero"] is not None:
        total += int(info["zero"].size) * 2
    return total + 8


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _forward(weights: dict, x: np.ndarray, act_bits: int | None = None):
    """One tanh-hidden-layer MLP, with optional activation quantisation."""
    x = np.asarray(x, dtype=float)
    if act_bits is not None:
        x = quantise(x, bits=act_bits)["values"]
    pre = x @ weights["W1"].T + weights["b1"]
    hidden = np.tanh(pre)
    if act_bits is not None:
        hidden = quantise(hidden, bits=act_bits)["values"]
    logits = hidden @ weights["W2"].T + weights["b2"]
    return logits, {"x": x, "pre": pre, "hidden": hidden}


def _loss_and_grads(weights: dict, x: np.ndarray, y: np.ndarray,
                    act_bits: int | None = None):
    logits, cache = _forward(weights, x, act_bits)
    probs = _softmax(logits)
    batch = x.shape[0]
    loss = -float(np.mean(np.log(probs[np.arange(batch), y] + 1e-12)))
    dlogits = probs.copy()
    dlogits[np.arange(batch), y] -= 1.0
    dlogits /= batch
    hidden = cache["hidden"]
    dhidden = (dlogits @ weights["W2"]) * (1.0 - hidden ** 2)
    grads = {
        "W1": dhidden.T @ cache["x"],
        "b1": dhidden.sum(axis=0),
        "W2": dlogits.T @ hidden,
        "b2": dlogits.sum(axis=0),
    }
    return loss, grads, cache


def _train(weights: dict, x: np.ndarray, y: np.ndarray, steps: int,
           lr: float) -> list[float]:
    history = []
    for _ in range(steps):
        loss, grads, _ = _loss_and_grads(weights, x, y)
        for key in weights:
            weights[key] -= lr * grads[key]
        history.append(loss)
    return history


# --------------------------------------------------------------------------- #
# Calibrated methods
# --------------------------------------------------------------------------- #
def _gptq_row(row: np.ndarray, inverse: np.ndarray, bits: int,
              group: int) -> np.ndarray:
    """Sequential rounding with OBQ compensation, one output unit."""
    qmax = 2 ** (bits - 1) - 1
    scales = np.empty((len(row) + group - 1) // group)
    for start in range(0, len(row), group):
        block = row[start:start + group]
        scale = np.abs(block).max() / qmax
        scales[start // group] = scale if scale > 0.0 else 1.0
    result = np.empty_like(row)
    work = row.copy()
    for index in range(len(row)):
        scale = scales[index // group]
        result[index] = np.clip(np.round(work[index] / scale),
                                -qmax, qmax) * scale
        error = (work[index] - result[index]) / inverse[index, index]
        if index + 1 < len(row):
            work[index + 1:] -= error * inverse[index, index + 1:]
    return result


def _layer_error(weight: np.ndarray, quantised: np.ndarray,
                 inputs: np.ndarray) -> float:
    difference = (weight - quantised) @ np.asarray(inputs, dtype=float).T
    return float(np.mean(difference ** 2))


# --------------------------------------------------------------------------- #
# The compressor
# --------------------------------------------------------------------------- #
class Quantizer:
    """A small MLP, ten quantisation recipes, and the accounting that matters.

    `fit(x, y)` trains the fp32 model and then applies the variant's procedure.
    `accuracy` evaluates the quantised weights, with activation quantisation
    for the two integer recipes. `n_bytes` counts what the compressed model
    actually occupies.
    """

    def __init__(self, variant: str = "absmax", *, dim: int = 32,
                 hidden: int = 24, classes: int = 5, group: int = 16,
                 bits: int | None = None, alpha: float = 0.5,
                 damping: float = 0.01, percentile: float = 99.5,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(dim, hidden, classes) < 1:
            raise ValueError("all sizes must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if group < 1:
            raise ValueError("group must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must lie in (0, 100]")
        if bits is not None and not 2 <= bits <= 16:
            raise ValueError("bits must lie in [2, 16]")
        self.variant = variant
        self.preset = PRESETS[variant]
        self.bits = self.preset["bits"] if bits is None else int(bits)
        self.dim = dim
        self.hidden = hidden
        self.classes = classes
        self.group = group
        self.alpha = alpha
        self.damping = damping
        self.percentile = percentile
        rng = np.random.default_rng(seed)
        self.weights = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(dim), (hidden, dim)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden), (classes, hidden)),
            "b2": np.zeros(classes),
        }
        self.quantised: dict[str, np.ndarray] | None = None
        self.info: dict[str, dict] = {}
        self.history: list[float] = []
        self.act_bits = self.bits if self.preset["acts"] else None

    # -- validation ---------------------------------------------------------- #
    def _check(self, x, y):
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("x must be (batch, dim)")
        if values.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if values.shape[1] != self.dim:
            raise ValueError("x has the wrong feature count")
        if not np.all(np.isfinite(values)):
            raise ValueError("x must be finite")
        labels = np.asarray(y)
        if labels.shape != (values.shape[0],):
            raise ValueError("y must hold one label per example")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return values, labels

    # -- quantisation -------------------------------------------------------- #
    def _scheme(self) -> dict:
        return {"bits": self.bits, "mode": self.preset["mode"],
                "group": self.group, "symmetric": self.preset["symmetric"]}

    def _clip(self) -> float | None:
        return self.percentile if self.variant == "ptq" else None

    def _quantise_layer(self, name: str, inputs: np.ndarray) -> np.ndarray:
        weight = self.weights[name]
        if self.variant == "gptq":
            hessian = inputs.T @ inputs / inputs.shape[0]
            scale = np.trace(hessian) / hessian.shape[0]
            hessian = hessian + self.damping * scale * np.eye(len(hessian))
            inverse = np.linalg.inv(hessian)
            result = np.empty_like(weight)
            for row in range(weight.shape[0]):
                result[row] = _gptq_row(
                    weight[row], inverse, self.bits, self.group)
            return result
        if self.variant == "awq":
            activation = np.mean(np.abs(inputs), axis=0)
            activation = np.maximum(activation, 1e-8)
            scale = activation ** self.alpha
            scale = scale / np.exp(np.mean(np.log(scale)))
            scaled = weight * scale[None, :]
            info = quantise(scaled, **self._scheme())
            return info["values"] / scale[None, :]
        info = quantise(weight, **self._scheme(), clip=self._clip())
        return info["values"]

    def quantize(self, x) -> None:
        """Apply the variant's procedure to the trained fp32 weights."""
        values, _ = self._check(x, y=np.zeros(x.shape[0], dtype=int))
        self.quantised = {key: value.copy()
                          for key, value in self.weights.items()}
        for name in MATRIX_NAMES:
            inputs = values
            if name == "W2":
                hidden = np.tanh(values @ self.quantised["W1"].T
                                 + self.weights["b1"])
                inputs = hidden
            self.quantised[name] = self._quantise_layer(name, inputs)
        self.info = {}
        for name in MATRIX_NAMES:
            self.info[name] = quantise(self.quantised[name], **self._scheme(),
                                       clip=self._clip())

    # -- training ------------------------------------------------------------ #
    def _fake_weights(self) -> dict:
        weights = {key: value.copy() for key, value in self.weights.items()}
        for name in MATRIX_NAMES:
            weights[name] = quantise(
                self.weights[name], **self._scheme())["values"]
        return weights

    def fit(self, x, y, *, steps: int = 400, lr: float = 0.3) -> list[float]:
        values, labels = self._check(x, y)
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.history = []
        if self.variant == "qat":
            warmup = max(1, steps // 2)
            self.history += _train(self.weights, values, labels, warmup, lr)
            for _ in range(steps - warmup):
                fake = self._fake_weights()
                loss, grads, _ = _loss_and_grads(fake, values, labels)
                for key in self.weights:
                    self.weights[key] -= lr * grads[key]
                self.history.append(loss)
        else:
            self.history += _train(self.weights, values, labels, steps, lr)
        self.quantize(values)
        return self.history

    # -- evaluation ---------------------------------------------------------- #
    def _active(self) -> dict:
        return self.quantised if self.quantised is not None else self.weights

    def accuracy(self, x, y) -> float:
        values, labels = self._check(x, y)
        logits, _ = _forward(self._active(), values, self.act_bits)
        return float(np.mean(np.argmax(logits, axis=-1) == labels))

    def loss(self, x, y) -> float:
        values, labels = self._check(x, y)
        return _loss_and_grads(self._active(), values, labels,
                               self.act_bits)[0]

    @property
    def n_bytes(self) -> int:
        if self.quantised is None:
            return int(sum(value.nbytes for value in self.weights.values()))
        total = int(self.weights["b1"].size + self.weights["b2"].size) * 4
        for name in MATRIX_NAMES:
            total += quantised_bytes(self.info[name])
        return total

    @property
    def n_fp32(self) -> int:
        return int(sum(value.size for value in self.weights.values())) * 4


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _centres(seed: int, classes: int = 5, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(classes, dim)) * 1.4


def _task(seed: int, centres: np.ndarray, per_class: int = 40,
          noise: float = 0.8):
    rng = np.random.default_rng(seed)
    classes, dim = centres.shape
    labels = np.repeat(np.arange(classes), per_class)
    x = centres[labels] + rng.normal(0.0, noise, (labels.size, dim))
    order = rng.permutation(labels.size)
    return x[order], labels[order]


def demo() -> None:
    centres = _centres(1)
    train_x, train_y = _task(2, centres)
    test_x, test_y = _task(3, centres, per_class=40)

    print("quantisation. one matrix, one grid:")
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(16, 16))
    info = quantise(matrix, bits=4)
    step = float(np.abs(info["scale"]).max())
    worst = float(np.abs(info["values"] - matrix).max())
    used = np.unique(info["codes"]).size
    print(f"  a 4-bit tensor scale is {step:.4f}; the largest rounding")
    print(f"  error is {worst:.4f}, at most half a step; {used} of 16 codes")
    print("  are used on this symmetric matrix")
    assert worst <= step / 2 + 1e-12
    assert used <= 16

    print("quantisation. a skewed range wastes a symmetric grid:")
    skewed = np.concatenate([np.zeros(64), np.linspace(0.0, 2.0, 192)])
    symmetric = quantise(skewed.reshape(16, 16), bits=4)
    asymmetric = quantise(skewed.reshape(16, 16), bits=4, symmetric=False)
    error_s = float(np.sqrt(np.mean((symmetric["values"] - skewed.reshape(
        16, 16)) ** 2)))
    error_a = float(np.sqrt(np.mean((asymmetric["values"] - skewed.reshape(
        16, 16)) ** 2)))
    print(f"  values in [0, 2] with a spike at zero: rmse {error_s:.4f}")
    print(f"  symmetric against {error_a:.4f} asymmetric, which spends its")
    print("  16 codes on the occupied half of the range")
    assert error_a < error_s

    print("quantisation. granularity follows the scale spread:")
    rows = np.logspace(-2.0, 2.0, 16)
    spread = rng.normal(size=(16, 16)) * rows[:, None]
    errors = {}
    for mode, group in (("tensor", 16), ("channel", 16), ("group", 4)):
        result = quantise(spread, bits=4, mode=mode, group=group)
        errors[mode] = float(np.sqrt(np.mean((result["values"] - spread) ** 2)))
        print(f"    {mode:<8s} rmse {errors[mode]:.4f}"
              f"  {quantised_bytes(result):4d} bytes")
    assert errors["channel"] < errors["tensor"]
    assert errors["group"] <= errors["channel"]

    print("quantisation. bits buy resolution, not free accuracy:")
    errors = {}
    for bits in (8, 6, 4, 3, 2):
        result = quantise(matrix, bits=bits)
        errors[bits] = float(np.sqrt(np.mean((result["values"] - matrix) ** 2)))
        print(f"    {bits:2d} bits  rmse {errors[bits]:.4f}"
              f"  {quantised_bytes(result):5d} bytes")
    ordered = [errors[bits] for bits in (2, 3, 4, 6, 8)]
    assert all(left > right for left, right in zip(ordered, ordered[1:],
                                                  strict=False))

    print("quantisation. gptq rounds against the input, not the weight:")
    rng = np.random.default_rng(5)
    factor = rng.normal(size=(64, 1))
    inputs = factor + 0.2 * rng.normal(size=(64, 32))
    weight = rng.normal(size=(24, 32)) / np.sqrt(32)
    naive = quantise(weight, bits=4, mode="group",
                     group=16)["values"]
    hessian = inputs.T @ inputs / inputs.shape[0]
    hessian += 0.01 * np.trace(hessian) / 32 * np.eye(32)
    inverse = np.linalg.inv(hessian)
    compensated = np.empty_like(weight)
    for row in range(weight.shape[0]):
        compensated[row] = _gptq_row(
            weight[row], inverse, 4, 16)
    error_naive = _layer_error(weight, naive, inputs)
    error_gptq = _layer_error(weight, compensated, inputs)
    print(f"  round-to-nearest leaves {error_naive:.2e} output error")
    print(f"  the compensated rounding leaves {error_gptq:.2e}:")
    print("  correlated inputs make a rounding error in one column")
    print("  predictable from the columns already fixed")
    assert error_gptq < error_naive

    print("quantisation. awq scales the channels it must protect:")
    scales = np.logspace(-2.0, 2.0, 32)
    inputs = rng.normal(size=(128, 32)) * scales
    naive = quantise(weight, bits=4, mode="group", group=16)["values"]
    activation = np.mean(np.abs(inputs), axis=0)
    scale = activation ** 0.5
    scale = scale / np.exp(np.mean(np.log(scale)))
    scaled = quantise(weight * scale[None, :], bits=4, mode="group",
                      group=16)["values"] / scale[None, :]
    error_naive = _layer_error(weight, naive, inputs)
    error_awq = _layer_error(weight, scaled, inputs)
    print(f"  plain 4-bit groups leave {error_naive:.2e}")
    print(f"  activation-aware scaling leaves {error_awq:.2e}; the")
    print("  weights feeding large activations get the resolution")
    assert error_awq < error_naive

    print("quantisation. ten recipes on one three-layer model:")
    shape = {"dim": 24, "hidden": 16, "classes": 8}
    centres = _centres(1, classes=8, dim=24)
    train_x, train_y = _task(2, centres, per_class=20, noise=1.2)
    test_x, test_y = _task(3, centres, per_class=40, noise=1.2)
    reference = Quantizer("absmax", seed=6, **shape)
    reference.fit(train_x, train_y, steps=600, lr=0.3)
    dense_weights = {key: value.copy()
                     for key, value in reference.weights.items()}
    dense = reference.accuracy(test_x, test_y)
    print(f"  the fp32 model scores {dense:.3f} and occupies"
          f" {reference.n_fp32} bytes")
    table = {}
    for variant in VARIANTS:
        model = Quantizer(variant, seed=6, **shape)
        model.weights = {key: value.copy()
                         for key, value in dense_weights.items()}
        if variant == "qat":
            model.fit(train_x, train_y, steps=300, lr=0.1)
        else:
            model.quantize(train_x)
        score = model.accuracy(test_x, test_y)
        table[variant] = (score, model.loss(test_x, test_y), model.n_bytes)
        ratio = reference.n_fp32 / model.n_bytes
        print(f"    {variant:<12s} {score:.3f} accuracy"
              f"  loss {model.loss(test_x, test_y):.3f}"
              f"  {model.n_bytes:4d} bytes  {ratio:.1f}x")
        assert model.n_bytes < reference.n_fp32
    assert table["int8"][0] >= dense - 0.02
    assert table["per_group"][0] >= dense - 0.02
    assert table["int4"][0] < table["per_group"][0]
    print("  weight-only 4-bit is nearly free; activating at 4 bits is not")

    print("quantisation. at two bits the recipe is the difference:")
    stress = {}
    for variant in ("per_group", "qat", "gptq", "awq"):
        model = Quantizer(variant, seed=6, bits=2, **shape)
        model.weights = {key: value.copy()
                         for key, value in dense_weights.items()}
        if variant == "qat":
            model.fit(train_x, train_y, steps=300, lr=0.1)
        else:
            model.quantize(train_x)
        stress[variant] = (model.accuracy(test_x, test_y),
                           model.loss(test_x, test_y))
        print(f"    {variant:<10s} {stress[variant][0]:.3f} accuracy"
              f"  loss {stress[variant][1]:.3f}")
    assert stress["qat"][1] < stress["per_group"][1]
    assert stress["qat"][0] >= stress["per_group"][0]
    print("  straight-through training keeps the smallest loss; the")
    print("  calibration methods optimise a layer error that a two-bit")
    print("  network does not convert into the same downstream gain")

    print("quantisation. malformed calls are refused:")
    guard = Quantizer("absmax", dim=8, hidden=8, classes=2, seed=7)
    good_x = np.random.default_rng(8).normal(size=(8, 8))
    good_y = np.arange(8) % 2
    guard.fit(good_x, good_y, steps=2, lr=0.1)
    bad = [
        lambda: Quantizer("absmax_v2"),
        lambda: Quantizer("absmax", dim=0),
        lambda: Quantizer("absmax", hidden=0),
        lambda: Quantizer("absmax", classes=1),
        lambda: Quantizer("absmax", group=0),
        lambda: Quantizer("awq", alpha=0.0),
        lambda: Quantizer("gptq", damping=-1.0),
        lambda: Quantizer("ptq", percentile=0.0),
        lambda: Quantizer("absmax", bits=1),
        lambda: Quantizer("absmax", bits=17),
        lambda: quantise(np.zeros((2, 2)), bits=1),
        lambda: quantise(np.zeros((2, 2)), bits=17),
        lambda: quantise(np.zeros((2, 2)), group=0),
        lambda: quantise(np.zeros(4), bits=8),
        lambda: quantise(np.full((2, 2), np.nan), bits=8),
        lambda: guard.fit(good_x[0], good_y, steps=2),
        lambda: guard.fit(good_x[:, :4], good_y, steps=2),
        lambda: guard.fit(np.zeros((0, 8)), good_y[:0], steps=2),
        lambda: guard.fit(np.full_like(good_x, np.nan), good_y, steps=2),
        lambda: guard.fit(good_x, good_y[:3], steps=2),
        lambda: guard.fit(good_x, good_y.astype(float), steps=2),
        lambda: guard.fit(good_x, good_y + 5, steps=2),
        lambda: guard.fit(good_x, good_y, steps=0),
        lambda: guard.fit(good_x, good_y, lr=0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero size, one")
    print("  class, a zero group, a zero or negative calibration constant,")
    print("  a bit width outside [2, 16], a non-matrix or non-finite weight,")
    print("  a 1-d input, wrong features, an empty or non-finite batch,")
    print("  wrong or float or out-of-range labels, and a non-positive step")
    print("  count or learning rate")
    print("quantisation demo passed")


if __name__ == "__main__":
    demo()
