"""Mixed precision — low-precision arithmetic with fp32 bookkeeping.

Chapter:   13-training
Companion: mixed_precision.md
Tier 2 · 6 variants: fp16, bf16, fp8, loss_scaling, dynamic_scaling,
master_weights
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("fp16", "bf16", "fp8", "loss_scaling", "dynamic_scaling",
            "master_weights")
FORMATS = {"fp16": (5, 10), "bf16": (8, 7), "fp8": (4, 3)}
SATURATES = {"fp16": False, "bf16": False, "fp8": True}
MAX_VALUE = {"fp16": 65504.0, "bf16": 3.3895313892515355e38,
             "fp8": 448.0}


# --------------------------------------------------------------------------- #
# Format emulation
# --------------------------------------------------------------------------- #
def cast(x, fmt: str) -> np.ndarray:
    """Round `x` to the nearest value of a floating-point format.

    `fmt` names the exponent and mantissa widths: fp16 is (5, 10), bf16 is
    (8, 7) and fp8 is E4M3 at (4, 3). Rounding is to nearest even, subnormals
    are represented, and overflow becomes infinity except in fp8, which
    saturates at its largest finite value.
    """
    if fmt == "fp32":
        return np.asarray(x, dtype=float)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)):
        return values.copy()
    ebits, mbits = FORMATS[fmt]
    max_exp = (1 << (ebits - 1)) - 1
    min_exp = 1 - max_exp
    step = 2.0 ** (min_exp - mbits)
    magnitude = np.abs(values)
    mantissa, exponent = np.frexp(magnitude)
    rounded = np.rint(mantissa * 2.0 ** (mbits + 1)) / 2.0 ** (mbits + 1)
    out = np.copysign(np.ldexp(rounded, exponent), values)
    below = magnitude < 2.0 ** min_exp
    out = np.where(below, np.rint(values / step) * step, out)
    limit = MAX_VALUE[fmt]
    if SATURATES[fmt]:
        out = np.clip(out, -limit, limit)
    else:
        edge = limit + 2.0 ** (max_exp - mbits - 1)
        out = np.where(np.abs(out) >= edge, np.copysign(np.inf, values), out)
    return out


def limits(fmt: str) -> dict[str, float]:
    """The largest finite value, the smallest normal, and the ulp exponent."""
    ebits, mbits = FORMATS[fmt]
    max_exp = (1 << (ebits - 1)) - 1
    return {"max": MAX_VALUE[fmt],
            "min_normal": 2.0 ** (1 - max_exp),
            "eps": 2.0 ** -mbits}


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    rows = np.arange(logits.shape[0])
    loss = -float(np.mean(np.log(probs[rows, labels] + 1e-12)))
    probs[rows, labels] -= 1.0
    return loss, probs / logits.shape[0]


# --------------------------------------------------------------------------- #
# Dynamic loss scaling
# --------------------------------------------------------------------------- #
class DynamicScaler:
    """Halve the loss scale on overflow, double it after a run of good steps."""

    def __init__(self, scale: float = 1024.0, window: int = 4,
                 growth: float = 2.0, backoff: float = 0.5,
                 minimum: float = 1.0, maximum: float = 65536.0):
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        if window < 1:
            raise ValueError("window must be positive")
        if not 0.0 < backoff < 1.0 or growth <= 1.0:
            raise ValueError("backoff must shrink and growth must grow")
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("the scale bounds are inconsistent")
        self.scale = float(scale)
        self.window = window
        self.growth = growth
        self.backoff = backoff
        self.minimum = minimum
        self.maximum = maximum
        self.good = 0

    def update(self, overflow: bool) -> float:
        if overflow:
            self.scale = max(self.minimum, self.scale * self.backoff)
            self.good = 0
        else:
            self.good += 1
            if self.good >= self.window:
                self.scale = min(self.maximum, self.scale * self.growth)
                self.good = 0
        return self.scale


# --------------------------------------------------------------------------- #
# The training loop
# --------------------------------------------------------------------------- #
class MixedPrecision:
    """A one-hidden-layer classifier trained in a low-precision format.

    `forward(x, y)` casts the weights and activations to the format but
    accumulates in float64. `backward(cache)` returns the gradients and an
    overflow flag; `apply(gradients, lr)` updates the stored parameters.
    """

    def __init__(self, variant: str = "fp16", *, in_dim: int = 4,
                 hidden: int = 8, classes: int = 3, scale: float = 1024.0,
                 window: int = 4, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if min(in_dim, hidden, classes) < 1:
            raise ValueError("all sizes must be positive")
        if classes < 2:
            raise ValueError("classes must be at least two")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        if window < 1:
            raise ValueError("window must be positive")
        self.variant = variant
        self.fmt = variant if variant in FORMATS else "fp16"
        self.in_dim = in_dim
        self.hidden = hidden
        self.classes = classes
        self.scale = float(scale)
        self.master = variant == "master_weights"
        self.scaler = DynamicScaler(scale=scale, window=window)
        rng = np.random.default_rng(seed)
        params = {
            "W1": rng.normal(0.0, 1.0 / np.sqrt(in_dim), (in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, classes)),
            "b2": np.zeros(classes),
        }
        if not self.master:
            params = {key: cast(value, self.fmt)
                      for key, value in params.items()}
        self.parameters = params

    # -- properties ---------------------------------------------------------- #
    @property
    def n_parameters(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    @property
    def current_scale(self) -> float:
        if self.variant == "loss_scaling":
            return self.scale
        if self.variant == "dynamic_scaling":
            return self.scaler.scale
        return 1.0

    # -- validation ---------------------------------------------------------- #
    def _check(self, x, y):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError("x must be (batch, in_dim)")
        if x.shape[0] < 1:
            raise ValueError("the batch must be non-empty")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        labels = np.asarray(y)
        if labels.shape != (x.shape[0],):
            raise ValueError("y must hold one label per example")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("y must be integer labels")
        if labels.min() < 0 or labels.max() >= self.classes:
            raise ValueError("y is outside the class range")
        return x, labels

    # -- forward ------------------------------------------------------------- #
    def _working(self) -> dict[str, np.ndarray]:
        return {key: cast(value, self.fmt)
                for key, value in self.parameters.items()}

    def forward(self, x, y):
        x, y = self._check(x, y)
        p = self._working()
        pre = cast(x, self.fmt) @ p["W1"] + p["b1"]
        hidden = cast(np.tanh(pre), self.fmt)
        logits = hidden @ p["W2"] + p["b2"]
        loss, _ = _cross_entropy(logits, y)
        return loss, {"x": x, "y": y, "p": p, "pre": pre, "h": hidden,
                      "z": logits}

    # -- backward ------------------------------------------------------------ #
    def backward(self, cache):
        p, x, hidden = cache["p"], cache["x"], cache["h"]
        _, dz = _cross_entropy(cache["z"], cache["y"])
        dpre = (dz @ p["W2"].T) * (1.0 - np.tanh(cache["pre"]) ** 2)
        grads = {
            "W2": hidden.T @ dz,
            "b2": dz.sum(axis=0),
            "W1": x.T @ dpre,
            "b1": dpre.sum(axis=0),
        }
        scale = self.current_scale
        stored = {key: cast(value * scale, self.fmt)
                  for key, value in grads.items()}
        if any(not np.all(np.isfinite(value)) for value in stored.values()):
            return None, True
        return {key: value / scale for key, value in stored.items()}, False

    def apply(self, gradients, lr: float) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        for key, gradient in gradients.items():
            value = self.parameters[key] - lr * gradient
            self.parameters[key] = (value if self.master
                                    else cast(value, self.fmt))

    # -- training ------------------------------------------------------------ #
    def fit(self, x, y, *, steps: int = 200, lr: float = 0.05) -> list[float]:
        if steps < 1:
            raise ValueError("steps must be positive")
        history = []
        for _ in range(steps):
            loss, cache = self.forward(x, y)
            grads, overflow = self.backward(cache)
            if overflow:
                if self.variant == "dynamic_scaling":
                    self.scaler.update(True)
                continue
            self.apply(grads, lr)
            if self.variant == "dynamic_scaling":
                self.scaler.update(False)
            history.append(loss)
        return history

    def accuracy(self, x, y) -> float:
        x, labels = self._check(x, y)
        p = self._working()
        hidden = np.tanh(cast(x, self.fmt) @ p["W1"] + p["b1"])
        logits = hidden @ p["W2"] + p["b2"]
        return float(np.mean(np.argmax(logits, axis=-1) == labels))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _blobs(seed: int, per_class: int = 25, dim: int = 4, classes: int = 3):
    rng = np.random.default_rng(seed)
    centres = np.eye(classes, dim) * 2.0
    labels = np.repeat(np.arange(classes), per_class)
    x = centres[labels] + rng.normal(0.0, 0.4, (labels.size, dim))
    order = rng.permutation(labels.size)
    return x[order], labels[order]


def demo() -> None:
    print("mixed precision. three formats, three failure modes:")
    for fmt in FORMATS:
        info = limits(fmt)
        rng = np.random.default_rng(1)
        sample = 1.0 + rng.random(500)
        error = np.abs(cast(sample, fmt) - sample) / sample
        print(f"    {fmt:<5s} max {info['max']:9.3e}"
              f"  min normal {info['min_normal']:9.2e}"
              f"  eps {info['eps']:7.2e}  rel err {error.max():7.2e}")
    assert limits("fp16")["max"] == 65504.0
    assert limits("bf16")["max"] > 3.0e38
    assert limits("fp8")["max"] == 448.0
    print("  fp16 has range but loses tiny values; bf16 keeps the exponent")
    print("  and loses mantissa; fp8 has almost no mantissa at all")

    print("mixed precision. the emulation matches a native float16:")
    rng = np.random.default_rng(8)
    probe = np.concatenate([
        rng.normal(size=1000) * 10.0,
        rng.normal(size=500) * 1e-7,
        np.array([65519.0, 65520.0, 1e-45, 0.0, -3.0]),
    ])
    with np.errstate(over="ignore"):
        native = probe.astype(np.float16).astype(float)
    ours = cast(probe, "fp16")
    assert np.array_equal(ours, native)
    finite = np.isfinite(native)
    worst = float(np.max(np.abs(ours[finite] - native[finite])))
    print(f"  {probe.size} draws, including subnormals and overflow, agree")
    print(f"  bit for bit with numpy's own float16 (largest gap {worst:.1e})")

    print("mixed precision. fp16 flushes tiny gradients to zero:")
    tiny = np.array([1e-9, 1e-8, 1e-7, 1e-6])
    direct = cast(tiny, "fp16")
    scaled = cast(tiny * 1024.0, "fp16") / 1024.0
    assert direct[0] == 0.0 and direct[1] == 0.0
    assert scaled[0] > 0.0 and scaled[1] > 0.0
    print("    gradient     " + "  ".join(f"{v:9.1e}" for v in tiny))
    print("    fp16         " + "  ".join(f"{v:9.1e}" for v in direct))
    print("    scaled/1024  " + "  ".join(f"{v:9.1e}" for v in scaled))
    print("  multiplying the loss by 1024 moves the gradient into the")
    print("  representable range; dividing it back is exact by a power of 2")

    print("mixed precision. bf16 rounds every weight to seven bits:")
    values = np.array([1.0 + 0.001 * k for k in range(4)])
    errors = np.abs(cast(values, "bf16") - values) / values
    assert errors.max() <= 2.0 ** -8
    print("    " + "  ".join(f"{v:.4f}" for v in values))
    print("    " + "  ".join(f"{v:.4f}" for v in cast(values, "bf16")))
    print(f"  worst relative error {errors.max():.2e}, against a 2^-8 half")
    print("  ulp: the range is fine, the resolution is not")

    print("mixed precision. fp8 saturates instead of overflowing:")
    values = np.array([400.0, 448.0, 500.0, 10000.0])
    result = cast(values, "fp8")
    assert np.all(result <= 448.0)
    print("    " + "  ".join(f"{v:7.1f}" for v in values))
    print("    " + "  ".join(f"{v:7.1f}" for v in result))
    print("  E4M3 has no infinity encoding, so the top values clamp at 448")

    print("mixed precision. a dynamic scaler tracks overflow:")
    scaler = DynamicScaler(scale=1024.0, window=3)
    trace = []
    for overflow in (False, False, False, True, False, False, False, True):
        trace.append(scaler.update(overflow))
    print("    " + " -> ".join(f"{value:.0f}" for value in trace))
    assert trace[2] == 2048.0 and trace[3] == 1024.0 and trace[6] == 2048.0
    print("  three clean steps double the scale; one overflow halves it")
    print("  and resets the counter")

    x, y = _blobs(seed=2)

    print("mixed precision. master weights keep the small updates:")
    frozen = MixedPrecision("fp16", in_dim=4, hidden=8, seed=3)
    master = MixedPrecision("master_weights", in_dim=4, hidden=8, seed=3)
    frozen_start = {key: value.copy()
                    for key, value in frozen.parameters.items()}
    master_start = {key: value.copy()
                    for key, value in master.parameters.items()}
    frozen.fit(x, y, steps=100, lr=1e-5)
    master.fit(x, y, steps=100, lr=1e-5)
    kept = sum(int(np.sum(frozen.parameters[key] == frozen_start[key]))
               for key in frozen_start)
    total = sum(value.size for value in frozen_start.values())
    moved_low = max(float(np.abs(frozen.parameters[key]
                                 - frozen_start[key]).max())
                    for key in frozen_start)
    moved_master = max(float(np.abs(master.parameters[key]
                                    - master_start[key]).max())
                       for key in master_start)
    print(f"  at lr 1e-5, {kept} of {total} fp16 weights are bit-identical")
    print("  after 100 steps; the largest one that did move travelled")
    print(f"  {moved_low:.1e}, while the fp32 masters moved {moved_master:.1e}")
    assert kept > total // 2
    assert moved_master > 2.0 * moved_low
    print("  an update smaller than half an ulp rounds to no change at all,")
    print("  so most of the low-precision copy stalls while every master")
    print("  keeps accumulating the same gradient")

    print("mixed precision. precision sets the floor on what is learnable:")
    scores = {}
    for variant in VARIANTS:
        model = MixedPrecision(variant, in_dim=4, hidden=8, seed=4,
                               scale=1024.0)
        start = model.forward(x, y)[0]
        history = model.fit(x, y, steps=400, lr=0.05)
        end = model.forward(x, y)[0]
        scores[variant] = (end, model.accuracy(x, y), model.current_scale,
                           len(history))
        print(f"    {variant:<15s} {start:.4f} -> {end:.4f}"
              f"  accuracy {scores[variant][1]:.3f}"
              f"  scale {scores[variant][2]:>6.0f}")
    assert scores["fp16"][1] >= 0.95
    assert scores["bf16"][1] >= 0.95
    assert scores["master_weights"][1] >= 0.95
    assert scores["dynamic_scaling"][1] >= 0.95
    assert scores["fp8"][1] < 0.8
    assert scores["loss_scaling"][3] == 400
    print("  fp16 and bf16 lose resolution but keep the exponents and")
    print("  converge; fp8's three mantissa bits cannot resolve an update")
    print("  this small, so its loss stalls far from zero")

    print("mixed precision. malformed calls are refused:")
    guard = MixedPrecision("fp16", in_dim=4, hidden=8, seed=5)
    guard.forward(x, y)
    bad = [
        lambda: MixedPrecision("fp64"),
        lambda: MixedPrecision("fp16", in_dim=0),
        lambda: MixedPrecision("fp16", hidden=0),
        lambda: MixedPrecision("fp16", classes=1),
        lambda: MixedPrecision("fp16", scale=0.0),
        lambda: MixedPrecision("fp16", window=0),
        lambda: MixedPrecision("loss_scaling", scale=-1.0),
        lambda: DynamicScaler(scale=0.0),
        lambda: DynamicScaler(window=0),
        lambda: DynamicScaler(backoff=1.5),
        lambda: cast(np.zeros(2), "fp64"),
        lambda: guard.forward(x[0], y),
        lambda: guard.forward(x[:, :3], y),
        lambda: guard.forward(x, y[:2]),
        lambda: guard.forward(x, y.astype(float)),
        lambda: guard.forward(x, y + 3),
        lambda: guard.forward(np.full_like(x, np.nan), y),
        lambda: guard.apply({"W1": np.zeros((4, 8))}, lr=0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown format or variant, a zero")
    print("  size, one class, a non-positive scale, a zero window, a")
    print("  growing backoff, a one-dimensional input, wrong labels, a")
    print("  non-finite input, and a non-positive learning rate")
    print("mixed precision demo passed")


if __name__ == "__main__":
    demo()
