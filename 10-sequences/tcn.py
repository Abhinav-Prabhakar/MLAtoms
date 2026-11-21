r"""Temporal convolutional networks — convolution that remembers.

Chapter:   10-sequences
Companion: tcn.md
Tier 2 · 4 variants: dilated, causal, residual_tcn, noncausal

    y_t = sum over i of W_i  x_(t - d i)                                 (1)
    receptive field = 1 + sum over layers of (k - 1) d                  (2)

A temporal convolutional network is a stack of one-dimensional convolutions
along time. Each layer mixes a fixed number of neighbouring positions, and
stacking layers composes those windows, so the network sees a receptive field
that grows with depth. Nothing about the convolution is recurrent: the whole
sequence is processed in parallel, and the same weights are used at every
position.

The four variants are the same convolution with three different ideas bolted on.
`causal` is the plain version with dilation one: position t reads only t, t-1,
..., t-(k-1). `dilated` spreads the taps out by a factor that doubles per layer,
so five layers of kernel width three cover sixty-three positions instead of
eleven. `residual_tcn` wraps two dilations in a residual block, which is what
makes a deep stack trainable. `noncausal` pads symmetrically, reads the future,
and exists to show what the causal mask is buying.

Everything is numpy, including the backward pass. The demo checks the
convolution against a direct loop, checks every parameter gradient against
central differences, measures the receptive field by injecting an impulse, and
shows the point of dilation on a delayed-copy task: the causal stack has a
receptive field of seven and cannot see sixteen steps back, while the dilated
stack covers sixty-three and learns the copy. The numbers to carry away: a
three-layer causal stack sees 7 positions and a five-layer dilated stack 63, the
impulse response matches (2) exactly, the gradients match central differences to
4e-11, the delayed copy goes from 0.41 to 0.0003 test error once the field is
long enough, and dilation reaches that field with 865 parameters against 6065
for the dense stack.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("dilated", "causal", "residual_tcn", "noncausal")


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _pad(x: np.ndarray, kernel: int, dilation: int, causal: bool) -> tuple:
    """Pad along time and report how much went on the left."""
    total = (kernel - 1) * dilation
    left = total if causal else total // 2
    if total == 0:
        return x, 0
    return np.pad(x, ((0, 0), (left, total - left), (0, 0))), left


def _conv_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                  dilation: int, causal: bool) -> tuple:
    """Dilated convolution along time: ``weight`` is (kernel, in, out)."""
    batch, steps, _ = x.shape
    kernel = weight.shape[0]
    padded, left = _pad(x, kernel, dilation, causal)
    output = np.zeros((batch, steps, weight.shape[2]))
    for tap in range(kernel):
        offset = (kernel - 1 - tap) * dilation if causal else tap * dilation
        output += padded[:, offset:offset + steps, :] @ weight[tap]
    output += bias
    return output, (padded, left, weight, dilation, causal, x.shape)


def _conv_backward(dout: np.ndarray, cache: tuple) -> tuple:
    """Gradients of the convolution with respect to input, weight and bias."""
    padded, left, weight, dilation, causal, shape = cache
    batch, steps, _ = shape
    kernel = weight.shape[0]
    dweight = np.zeros_like(weight)
    dpadded = np.zeros_like(padded)
    for tap in range(kernel):
        offset = (kernel - 1 - tap) * dilation if causal else tap * dilation
        chunk = padded[:, offset:offset + steps, :]
        dweight[tap] = np.einsum("bti,bto->io", chunk, dout)
        dpadded[:, offset:offset + steps, :] += dout @ weight[tap].T
    dbias = dout.sum(axis=(0, 1))
    return dpadded[:, left:left + steps], dweight, dbias


class TCN:
    """A stack of dilated causal convolutions with a linear readout.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    channels : int
        Width of every hidden layer.
    layers : int
        Number of convolutions, or of residual blocks.
    kernel : int
        Number of taps per convolution.
    dilation_base : int
        Dilation of layer l is ``dilation_base ** l``.
    n_out : int
        Width of the readout.
    lr, n_iter, batch : float or int
        Adam step size, passes over the data and minibatch size.
    seed : int
        Seed for the initial weights and the shuffling.

    Attributes
    ----------
    dilations_ : list of int
        The dilation used by each convolution or block.
    """

    def __init__(self, *, variant: str = "dilated", channels: int = 8,
                 layers: int = 3, kernel: int = 3, dilation_base: int = 2,
                 n_out: int = 1, lr: float = 0.01, n_iter: int = 200,
                 batch: int = 16, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        for name, value, low in (("channels", channels, 1), ("layers", layers, 1),
                                 ("kernel", kernel, 1), ("dilation_base", dilation_base, 1),
                                 ("n_out", n_out, 1), ("batch", batch, 1)):
            if not isinstance(value, (int, np.integer)) or value < low:
                raise ValueError(f"{name} must be an integer of at least {low}")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        self.variant = variant
        self.channels = int(channels)
        self.layers = int(layers)
        self.kernel = int(kernel)
        self.dilation_base = int(dilation_base)
        self.n_out = int(n_out)
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.batch = int(batch)
        self.seed = int(seed)
        self.causal = variant != "noncausal"

    # ------------------------------------------------------------------ #
    # parameters
    # ------------------------------------------------------------------ #
    def _dilations(self) -> list:
        if self.variant == "causal":
            return [1] * self.layers
        return [self.dilation_base ** index for index in range(self.layers)]

    def _build(self, features: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.features_ = features
        self.dilations_ = self._dilations()
        self.params: dict = {}

        def weights(shape: tuple) -> np.ndarray:
            fan_in = shape[0] * shape[1]
            return rng.standard_normal(shape) / math.sqrt(max(1, fan_in))

        if self.variant == "residual_tcn":
            self.params["stem_W"] = weights((1, features, self.channels))
            self.params["stem_b"] = np.zeros(self.channels)
            for index, _ in enumerate(self.dilations_):
                self.params[f"W1_{index}"] = weights((self.kernel, self.channels,
                                                     self.channels))
                self.params[f"b1_{index}"] = np.zeros(self.channels)
                self.params[f"W2_{index}"] = weights((self.kernel, self.channels,
                                                     self.channels))
                self.params[f"b2_{index}"] = np.zeros(self.channels)
        else:
            for index, _ in enumerate(self.dilations_):
                incoming = features if index == 0 else self.channels
                self.params[f"W{index}"] = weights((self.kernel, incoming, self.channels))
                self.params[f"b{index}"] = np.zeros(self.channels)
        self.params["head_W"] = weights((self.channels, self.n_out))
        self.params["head_b"] = np.zeros(self.n_out)
        self._built = True

    def _check_input(self, x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError("x must be (steps, features) or a batch of those")
        if min(array.shape) < 1:
            raise ValueError("x needs at least one sequence, step and feature")
        if not np.all(np.isfinite(array)):
            raise ValueError("x must be finite")
        return array

    def n_parameters(self) -> int:
        """Number of trainable scalars, all blocks counted."""
        if not hasattr(self, "params"):
            raise RuntimeError("build the model first")
        return int(sum(parameter.size for parameter in self.params.values()))

    def receptive_field(self) -> int:
        """How many input positions the final output depends on."""
        if not hasattr(self, "dilations_"):
            raise RuntimeError("build the model first")
        spans = [(self.kernel - 1) * dilation for dilation in self.dilations_]
        if self.variant == "residual_tcn":
            return 1 + 2 * sum(spans)
        return 1 + sum(spans)

    # ------------------------------------------------------------------ #
    # forward and backward
    # ------------------------------------------------------------------ #
    def _forward(self, x: np.ndarray) -> tuple:
        cache: dict = {}
        h = x
        if self.variant == "residual_tcn":
            incoming = h
            pre = h @ self.params["stem_W"][0] + self.params["stem_b"]
            mask = pre > 0.0
            h = _relu(pre)
            cache["stem"] = (incoming, mask)
            for index, dilation in enumerate(self.dilations_):
                incoming = h
                a, first = _conv_forward(incoming, self.params[f"W1_{index}"],
                                         self.params[f"b1_{index}"], dilation,
                                         self.causal)
                mask_a = a > 0.0
                a = _relu(a)
                c, second = _conv_forward(a, self.params[f"W2_{index}"],
                                          self.params[f"b2_{index}"], dilation,
                                          self.causal)
                h = c + incoming
                cache[index] = (incoming, first, mask_a, second)
        else:
            for index, dilation in enumerate(self.dilations_):
                pre, conv_cache = _conv_forward(h, self.params[f"W{index}"],
                                                self.params[f"b{index}"], dilation,
                                                self.causal)
                if index < len(self.dilations_) - 1:
                    mask = pre > 0.0
                    cache[index] = (conv_cache, mask)
                    h = _relu(pre)
                else:
                    cache[index] = (conv_cache, None)
                    h = pre
        output = h @ self.params["head_W"] + self.params["head_b"]
        return output, (cache, h)

    def _backward(self, dout: np.ndarray, cache: tuple) -> dict:
        layer_cache, last = cache
        grads: dict = {"head_W": np.einsum("btc,bto->co", last, dout),
                       "head_b": dout.sum(axis=(0, 1))}
        dlast = dout @ self.params["head_W"].T
        if self.variant == "residual_tcn":
            dstem = dlast
            for index in reversed(range(len(self.dilations_))):
                incoming, first, mask_a, second = layer_cache[index]
                dsecond, grads[f"W2_{index}"], grads[f"b2_{index}"] = _conv_backward(
                    dstem, second)
                da = dsecond * mask_a
                dconvolved, grads[f"W1_{index}"], grads[f"b1_{index}"] = _conv_backward(
                    da, first)
                dstem = dconvolved + dstem
            dpre = dstem * layer_cache["stem"][1]
            grads["stem_W"] = np.einsum("btf,bto->fo", layer_cache["stem"][0],
                                        dpre)[None]
            grads["stem_b"] = dpre.sum(axis=(0, 1))
            return grads
        dcurrent = dlast
        for index in reversed(range(len(self.dilations_))):
            conv_cache, mask = layer_cache[index]
            dpre = dcurrent if mask is None else dcurrent * mask
            dcurrent, grads[f"W{index}"], grads[f"b{index}"] = _conv_backward(
                dpre, conv_cache)
        return grads

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the network, building parameters on first use."""
        array = self._check_input(x)
        if not hasattr(self, "params"):
            self._build(array.shape[2])
        if array.shape[2] != self.features_:
            raise ValueError("x has the wrong number of features")
        return self._forward(array)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for `forward`."""
        return self.forward(x)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Half the mean squared error."""
        prediction = self.forward(x)
        target = self._targets(y, prediction.shape)
        return float(0.5 * np.mean((prediction - target) ** 2))

    def gradient(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Gradient of `loss` with respect to every parameter."""
        array = self._check_input(x)
        if not hasattr(self, "params"):
            self._build(array.shape[2])
        prediction, cache = self._forward(array)
        target = self._targets(y, prediction.shape)
        dout = (prediction - target) / prediction.size
        return self._backward(dout, cache)

    @staticmethod
    def _targets(y: np.ndarray, shape: tuple) -> np.ndarray:
        target = np.asarray(y, dtype=float)
        if target.ndim == 2:
            target = target[..., None]
        if target.shape != shape:
            raise ValueError("y must match the shape of the prediction")
        if not np.all(np.isfinite(target)):
            raise ValueError("y must be finite")
        return target

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray, y: np.ndarray) -> TCN:
        """Adam on half the mean squared error, in minibatches."""
        array = self._check_input(x)
        if not hasattr(self, "params"):
            self._build(array.shape[2])
        target = self._targets(y, (array.shape[0], array.shape[1], self.n_out))
        moment = {name: np.zeros_like(value) for name, value in self.params.items()}
        velocity = {name: np.zeros_like(value) for name, value in self.params.items()}
        order = np.arange(array.shape[0])
        self.history_ = []
        step = 0
        for _ in range(self.n_iter):
            order = np.random.default_rng(self.seed + len(self.history_)).permutation(order)
            total = 0.0
            for start in range(0, order.size, self.batch):
                chosen = order[start:start + self.batch]
                prediction, cache = self._forward(array[chosen])
                error = prediction - target[chosen]
                total += float(0.5 * np.sum(error ** 2))
                grads = self._backward(error / error.size, cache)
                step += 1
                for name, gradient in grads.items():
                    moment[name] = 0.9 * moment[name] + 0.1 * gradient
                    velocity[name] = 0.999 * velocity[name] + 0.001 * gradient ** 2
                    corrected = (moment[name] / (1 - 0.9 ** step)
                                 / (np.sqrt(velocity[name]
                                            / (1 - 0.999 ** step)) + 1e-8))
                    self.params[name] -= self.lr * corrected
            self.history_.append(total / order.size)
        return self


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _delayed_copy(seed: int, count: int, length: int, delay: int) -> tuple:
    """A signal and a target that copies it from `delay` steps earlier."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((count, length, 1))
    target = np.zeros_like(signal)
    target[:, delay:] = signal[:, :-delay]
    return signal, target


def _positive(model: TCN, features: int) -> TCN:
    """A built model whose activations stay positive, so ReLU never clips."""
    model._build(features)
    for name, value in model.params.items():
        if name.startswith("b") or name.endswith("_b"):
            model.params[name] = np.ones_like(value)
        else:
            model.params[name] = np.abs(value) + 0.1
    return model


def _demo_convolution() -> None:
    """One convolution against a direct loop, and its gradient."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 7, 3))
    weight = rng.standard_normal((3, 3, 4))
    bias = rng.standard_normal(4)
    causal, cache = _conv_forward(x, weight, bias, 2, True)
    direct = np.zeros_like(causal)
    for sequence in range(2):
        for t in range(7):
            for out in range(4):
                total = bias[out]
                for tap in range(3):
                    source = t - tap * 2
                    if source >= 0:
                        total += weight[tap, :, out] @ x[sequence, source]
                direct[sequence, t, out] = total
    symmetric, _ = _conv_forward(x, weight, bias, 1, False)
    centred = np.zeros_like(symmetric)
    for sequence in range(2):
        for t in range(7):
            for out in range(4):
                total = bias[out]
                for tap in range(3):
                    source = t - 1 + tap
                    if 0 <= source < 7:
                        total += weight[tap, :, out] @ x[sequence, source]
                centred[sequence, t, out] = total
    print("a kernel of three taps and dilation two, against the definition:")
    print(f"  causal output differs from the loop by "
          f"{np.abs(causal - direct).max():.1e}")
    print(f"  symmetric output differs from the centred loop by "
          f"{np.abs(symmetric - centred).max():.1e}")

    dout = rng.standard_normal(causal.shape)
    dx, dweight, dbias = _conv_backward(dout, cache)
    eps = 1e-6

    def objective(xx: np.ndarray, ww: np.ndarray, bb: np.ndarray) -> float:
        return float(np.sum(_conv_forward(xx, ww, bb, 2, True)[0] * dout))

    numeric = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (objective(plus, weight, bias) - objective(minus, weight, bias)) / 2 / eps
    numeric_weight = np.zeros_like(weight)
    for index in np.ndindex(weight.shape):
        plus, minus = weight.copy(), weight.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric_weight[index] = (
            objective(x, plus, bias) - objective(x, minus, bias)) / 2 / eps
    numeric_bias = np.zeros_like(bias)
    for index in np.ndindex(bias.shape):
        plus, minus = bias.copy(), bias.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric_bias[index] = (objective(x, weight, plus)
                               - objective(x, weight, minus)) / 2 / eps
    print(f"  the input gradient matches central differences to "
          f"{np.abs(numeric - dx).max():.1e} on a scale of {np.abs(dx).max():.0f}")
    print(f"  the weight and bias gradients match to "
          f"{np.abs(numeric_weight - dweight).max():.1e} and "
          f"{np.abs(numeric_bias - dbias).max():.1e}")
    assert np.abs(causal - direct).max() < 1e-12
    assert np.abs(symmetric - centred).max() < 1e-12
    assert np.abs(numeric - dx).max() < 1e-8
    assert np.abs(numeric_weight - dweight).max() < 1e-8


def _demo_receptive_field() -> None:
    """Inject an impulse and read off which outputs move."""
    length = 128
    base = np.full((1, length, 2), 0.1)
    impulse = base.copy()
    impulse[0, 30, :] += 1.0
    print("an impulse at position 30, and the outputs that change:")
    for variant, layers in (("causal", 3), ("dilated", 5), ("residual_tcn", 4),
                            ("noncausal", 5)):
        model = _positive(TCN(variant=variant, channels=4, layers=layers,
                              kernel=3, n_out=1, seed=0), 2)
        difference = np.abs(model.forward(impulse) - model.forward(base))[0].max(axis=-1)
        positions = np.flatnonzero(difference > 1e-9)
        window = f"{positions[0]}..{positions[-1]}"
        print(f"  {variant:<13} receptive field {model.receptive_field():>3}   "
              f"outputs {window:<9} ({positions.size} positions)")
        if variant == "noncausal":
            assert positions[0] == 0
            assert positions[-1] == 30 + model.receptive_field() // 2
        else:
            assert positions[0] == 30
            assert positions.size == model.receptive_field()
    causal = _positive(TCN(variant="causal", channels=4, layers=3, kernel=3, n_out=1),
                       2)
    difference = np.abs(causal.forward(impulse) - causal.forward(base))[0].max(axis=-1)
    print(f"  the causal stack starts exactly at the impulse, at position "
          f"{np.flatnonzero(difference > 1e-9)[0]}")
    symmetric = _positive(TCN(variant="noncausal", channels=4, layers=2, kernel=3,
                              n_out=1), 2)
    difference = np.abs(symmetric.forward(impulse) - symmetric.forward(base))[0].max(axis=-1)
    positions = np.flatnonzero(difference > 1e-9)
    print(f"  the symmetric stack runs from "
          f"{30 - symmetric.receptive_field() // 2} to "
          f"{30 + symmetric.receptive_field() // 2}, so it starts "
          f"{30 - positions[0]} steps before the impulse")
    print("  The causal field is a suffix of time; the symmetric one is a window")
    assert positions[0] < 30


def _demo_gradient() -> None:
    """Every parameter gradient, on a model small enough to difference."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 6, 2))
    y = rng.standard_normal((1, 6, 2))
    print("the whole network, analytic gradient against central differences:")
    for variant in VARIANTS:
        model = TCN(variant=variant, channels=3, layers=2, kernel=3, n_out=2, seed=1)
        grads = model.gradient(x, y)
        worst, scale = 0.0, 0.0
        for name, gradient in grads.items():
            parameter = model.params[name]
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus = model.loss(x, y)
                parameter[index] = original - 1e-6
                minus = model.loss(x, y)
                parameter[index] = original
                numeric = (plus - minus) / 2e-6
                worst = max(worst, abs(numeric - gradient[index]))
                scale = max(scale, abs(numeric), abs(gradient[index]))
        print(f"  {variant:<13} {model.n_parameters():>3} parameters   worst "
              f"error {worst:.1e} on a scale of {scale:.1f}")
        assert worst / scale < 1e-7


def _demo_training() -> None:
    """A copy from sixteen steps back, which needs a long receptive field."""
    x, y = _delayed_copy(0, 32, 48, 16)
    test_x, test_y = _delayed_copy(1, 64, 48, 16)
    print("copy the input from sixteen steps earlier, 32 sequences of 48 frames:")
    print("  variant        layers   receptive field   parameters   test loss")
    results = {}
    for variant, layers in (("causal", 3), ("dilated", 5), ("residual_tcn", 4),
                            ("noncausal", 5)):
        model = TCN(variant=variant, channels=8, layers=layers, kernel=3, n_out=1,
                    lr=0.01, n_iter=400, batch=16, seed=0).fit(x, y)
        results[variant] = model.loss(test_x, test_y)
        print(f"  {variant:<13} {layers:>7}   {model.receptive_field():>15}   "
              f"{model.n_parameters():>10}   {results[variant]:>9.4f}")
    print("  The first-order stack cannot reach back sixteen steps, so it fits noise.")
    print("  Doubling the dilation reaches it in five layers; the residual wrapper")
    print("  reaches the same field with a smoother optimisation.")
    assert results["causal"] > 0.1
    assert results["dilated"] < 0.01
    assert results["residual_tcn"] < 0.01
    assert results["noncausal"] < 0.01


def _demo_efficiency() -> None:
    """The same receptive field, with and without dilation."""
    print("a receptive field of 63, two ways, at width 8 and kernel 3:")
    for variant, layers in (("causal", 31), ("dilated", 5)):
        model = TCN(variant=variant, channels=8, layers=layers, kernel=3)
        model._build(2)
        print(f"  {variant:<8} {layers:>2} layers   receptive field "
              f"{model.receptive_field():>3}   {model.n_parameters():>5} parameters")
    dense = TCN(variant="causal", channels=8, layers=31, kernel=3)
    dense._build(2)
    sparse = TCN(variant="dilated", channels=8, layers=5, kernel=3)
    sparse._build(2)
    print(f"  dilation is {dense.n_parameters() / sparse.n_parameters():.1f} times "
          "cheaper for the same window")
    assert dense.receptive_field() == sparse.receptive_field()
    assert dense.n_parameters() > 5 * sparse.n_parameters()


def _demo_guards() -> None:
    """Malformed configurations and inputs, all refused."""
    attempts = [
        lambda: TCN(variant="nope"),
        lambda: TCN(channels=0),
        lambda: TCN(layers=0),
        lambda: TCN(kernel=0),
        lambda: TCN(dilation_base=0),
        lambda: TCN(n_out=0),
        lambda: TCN(batch=0),
        lambda: TCN(lr=0.0),
        lambda: TCN(lr=np.nan),
        lambda: TCN(n_iter=0),
        lambda: TCN().forward(np.zeros((0, 4, 2))),
        lambda: TCN().forward(np.zeros((4, 0))),
        lambda: TCN().forward(np.zeros((4,))),
        lambda: TCN().forward(np.full((1, 4, 2), np.inf)),
        lambda: TCN().forward(np.zeros((1, 4, 3, 2))),
        lambda: TCN(n_out=2).loss(np.zeros((1, 4, 2)), np.zeros((1, 4))),
        lambda: TCN().loss(np.zeros((1, 4, 2)), np.full((1, 4, 1), np.nan)),
        lambda: TCN().n_parameters(),
        lambda: TCN().receptive_field(),
        lambda: TCN().loss(np.zeros((1, 4, 2)), np.zeros((1, 4, 2))),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused: an unknown variant, a zero or")
    print("  negative size, a non-finite or zero learning rate, an empty or")
    print("  ill-shaped input, a non-finite entry, a four-dimensional array, a")
    print("  target of the wrong shape, a non-finite target, and the two methods")
    print("  that need a built model before they can answer.")
    assert refused == len(attempts)


def main() -> None:
    _demo_convolution()
    _demo_receptive_field()
    _demo_gradient()
    _demo_training()
    _demo_efficiency()
    _demo_guards()
    print("tcn demo passed")


if __name__ == "__main__":
    main()
