"""MLP — the affine layer, and four ways to make it deep.

Chapter:   12-neural-blocks
Companion: mlp.md
Tier 1 · 6 variants: linear, mlp, residual_mlp, gated_mlp, swiglu_mlp,
deep_mlp
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("linear", "mlp", "residual_mlp", "gated_mlp", "swiglu_mlp",
            "deep_mlp")

EPSILON = 1e-5


# --------------------------------------------------------------------------- #
# Activation functions and their derivatives
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _silu_grad(x: np.ndarray) -> np.ndarray:
    curve = _sigmoid(x)
    return curve * (1.0 + x * (1.0 - curve))


ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "silu": (_silu, _silu_grad),
    "identity": (_identity, lambda x: np.ones_like(x)),
}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    probabilities = _softmax(logits)
    rows = np.arange(labels.size)
    loss = float(-np.mean(np.log(probabilities[rows, labels] + 1e-12)))
    gradient = probabilities.copy()
    gradient[rows, labels] -= 1.0
    return loss, gradient / labels.size


# --------------------------------------------------------------------------- #
# Layers: each owns its parameters and reports its own gradients
# --------------------------------------------------------------------------- #
class _Dense:
    """One affine map followed by a pointwise activation."""

    def __init__(self, name: str, inputs: int, outputs: int,
                 rng: np.random.Generator, activation: str = "relu"):
        scale = np.sqrt(1.0 / inputs)
        if activation == "relu":
            scale = np.sqrt(2.0 / inputs)
        self.name = name
        self.activation = activation
        self.parameters = {
            f"{name}.W": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b": np.zeros(outputs),
        }
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, x: np.ndarray):
        p = self.parameters
        function, derivative = ACTIVATIONS[self.activation]
        pre = x @ p[f"{self.name}.W"] + p[f"{self.name}.b"]
        return function(pre), (x, pre, derivative(pre))

    def backward(self, gradient: np.ndarray, cache):
        x, _pre, slope = cache
        p, g = self.parameters, self.gradients
        d_pre = gradient * slope
        g[f"{self.name}.W"][...] = x.T @ d_pre
        g[f"{self.name}.b"][...] = d_pre.sum(axis=0)
        return d_pre @ p[f"{self.name}.W"].T


class _GLU:
    """A ReLU branch multiplied by a learned sigmoid gate."""

    def __init__(self, name: str, inputs: int, outputs: int,
                 rng: np.random.Generator):
        scale = np.sqrt(2.0 / inputs)
        self.name = name
        self.parameters = {
            f"{name}.W": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b": np.zeros(outputs),
            f"{name}.V": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.c": np.zeros(outputs),
        }
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, x: np.ndarray):
        p = self.parameters
        pre = x @ p[f"{self.name}.W"] + p[f"{self.name}.b"]
        active = _relu(pre)
        logits = x @ p[f"{self.name}.V"] + p[f"{self.name}.c"]
        gate = _sigmoid(logits)
        return active * gate, (x, pre, active, gate)

    def backward(self, gradient: np.ndarray, cache):
        x, pre, active, gate = cache
        p, g = self.parameters, self.gradients
        d_active = gradient * gate
        d_logits = gradient * active * gate * (1.0 - gate)
        d_pre = d_active * _relu_grad(pre)
        g[f"{self.name}.W"][...] = x.T @ d_pre
        g[f"{self.name}.b"][...] = d_pre.sum(axis=0)
        g[f"{self.name}.V"][...] = x.T @ d_logits
        g[f"{self.name}.c"][...] = d_logits.sum(axis=0)
        return d_pre @ p[f"{self.name}.W"].T + d_logits @ p[f"{self.name}.V"].T


class _SwiGLU:
    """SiLU(W1 x) times (W3 x), projected back by W2."""

    def __init__(self, name: str, inputs: int, outputs: int,
                 rng: np.random.Generator):
        scale = np.sqrt(2.0 / inputs)
        self.name = name
        self.parameters = {
            f"{name}.W1": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b1": np.zeros(outputs),
            f"{name}.W3": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b3": np.zeros(outputs),
            f"{name}.W2": rng.normal(0.0, np.sqrt(1.0 / outputs),
                                     (outputs, outputs)),
            f"{name}.b2": np.zeros(outputs),
        }
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, x: np.ndarray):
        p = self.parameters
        upper = x @ p[f"{self.name}.W1"] + p[f"{self.name}.b1"]
        value = x @ p[f"{self.name}.W3"] + p[f"{self.name}.b3"]
        product = _silu(upper) * value
        out = product @ p[f"{self.name}.W2"] + p[f"{self.name}.b2"]
        return out, (x, upper, value, product)

    def backward(self, gradient: np.ndarray, cache):
        x, upper, value, product = cache
        p, g = self.parameters, self.gradients
        g[f"{self.name}.W2"][...] = product.T @ gradient
        g[f"{self.name}.b2"][...] = gradient.sum(axis=0)
        d_product = gradient @ p[f"{self.name}.W2"].T
        d_upper = d_product * value * _silu_grad(upper)
        d_value = d_product * _silu(upper)
        g[f"{self.name}.W1"][...] = x.T @ d_upper
        g[f"{self.name}.b1"][...] = d_upper.sum(axis=0)
        g[f"{self.name}.W3"][...] = x.T @ d_value
        g[f"{self.name}.b3"][...] = d_value.sum(axis=0)
        return (d_upper @ p[f"{self.name}.W1"].T
                + d_value @ p[f"{self.name}.W3"].T)


def _layer_norm(x: np.ndarray):
    centred = x - x.mean(axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt((centred ** 2).mean(axis=-1, keepdims=True)
                            + EPSILON)
    return centred * inverse, (centred, inverse)


def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)


class _Residual:
    """x plus a branch, optionally normalising x on the way in."""

    def __init__(self, block, normalise: bool = False):
        self.block = block
        self.normalise = normalise
        self.parameters = block.parameters
        self.gradients = block.gradients

    def forward(self, x: np.ndarray):
        normed, norm_cache = _layer_norm(x) if self.normalise else (x, None)
        branch, block_cache = self.block.forward(normed)
        return x + branch, (norm_cache, block_cache)

    def backward(self, gradient: np.ndarray, cache):
        norm_cache, block_cache = cache
        d_normed = self.block.backward(gradient, block_cache)
        if self.normalise:
            d_normed = _layer_norm_backward(d_normed, norm_cache)
        return gradient + d_normed


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class MLP:
    """A stack of blocks trained with minibatch Adam.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `linear` is the affine head alone, `mlp` is a plain
        ReLU stack, `residual_mlp` adds skip connections, `gated_mlp` and
        `swiglu_mlp` replace the activation with a product of two branches, and
        `deep_mlp` makes the skips pre-normalised so that a deep stack trains.
    width : int
        Hidden width. Ignored by `linear`.
    depth : int
        Number of hidden blocks.
    epochs, batch_size, learning_rate : int, int, float
        Minibatch Adam schedule.
    seed : int
        Seeds the initialisation and the reshuffling.

    Notes
    -----
    `fit` standardises the inputs with the training mean and spread, builds the
    blocks, and stores `losses_`, `n_params_` and `blocks`. Every block owns a
    parameter dict and a gradient dict, and the optimiser walks the union.
    `predict` returns labels, `predict_proba` the softmax.
    """

    def __init__(self, variant: str = "mlp", *, width: int = 32, depth: int = 2,
                 epochs: int = 200, batch_size: int = 64,
                 learning_rate: float = 0.01, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        for name, value in (("width", width), ("depth", depth),
                            ("epochs", epochs), ("batch_size", batch_size)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        self.variant = variant
        self.width = int(width)
        self.depth = int(depth)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self._fitted = False

    # -- data ------------------------------------------------------------- #
    @staticmethod
    def _check(x: np.ndarray, y: np.ndarray):
        features = np.asarray(x, dtype=float)
        labels = np.asarray(y)
        if features.ndim != 2:
            raise ValueError("x must be two-dimensional")
        if labels.ndim != 1 or labels.size != features.shape[0]:
            raise ValueError("y must be one-dimensional with one label per row")
        if not np.all(np.isfinite(features)):
            raise ValueError("x must be finite")
        if labels.size < 2 * 2:
            raise ValueError("need at least four rows")
        if np.any(labels < 0) or np.any(labels - np.round(labels) > 1e-8):
            raise ValueError("labels must be non-negative integers")
        levels = np.unique(labels)
        if levels.size < 2:
            raise ValueError("need at least two classes")
        mapping = {level: index for index, level in enumerate(levels)}
        return features, np.array([mapping[value] for value in labels]), levels

    # -- graph ------------------------------------------------------------ #
    def _build(self):
        rng = np.random.default_rng(self.seed)
        if self.variant == "linear":
            blocks = [_Dense("head", self.n_features_, self.n_classes_, rng,
                             "identity")]
        else:
            blocks = [_Dense("stem", self.n_features_, self.width, rng,
                             "identity")]
            for index in range(self.depth):
                name = f"h{index}"
                if self.variant == "mlp":
                    blocks.append(_Dense(name, self.width, self.width, rng))
                elif self.variant == "residual_mlp":
                    blocks.append(_Residual(_Dense(name, self.width,
                                                   self.width, rng)))
                elif self.variant == "gated_mlp":
                    blocks.append(_GLU(name, self.width, self.width, rng))
                elif self.variant == "swiglu_mlp":
                    blocks.append(_SwiGLU(name, self.width, self.width, rng))
                else:
                    blocks.append(_Residual(_Dense(name, self.width,
                                                   self.width, rng),
                                            normalise=True))
            blocks.append(_Dense("head", self.width, self.n_classes_, rng,
                                 "identity"))
        self.blocks = blocks
        self.parameters = {}
        self.gradients = {}
        for block in blocks:
            self.parameters.update(block.parameters)
            self.gradients.update(block.gradients)
        self.n_params_ = int(sum(value.size
                                 for value in self.parameters.values()))

    def _forward(self, x: np.ndarray):
        cache = []
        for block in self.blocks:
            x, block_cache = block.forward(x)
            cache.append(block_cache)
        return x, cache

    def _backward(self, gradient: np.ndarray, cache) -> None:
        for block, block_cache in zip(reversed(self.blocks),
                                      reversed(cache), strict=True):
            gradient = block.backward(gradient, block_cache)

    # -- training --------------------------------------------------------- #
    def fit(self, x: np.ndarray, y: np.ndarray) -> MLP:
        """Train on features `x` and integer labels `y`."""
        features, labels, levels = self._check(x, y)
        self.classes_ = levels
        self.n_features_ = features.shape[1]
        self.n_classes_ = int(levels.size)
        self.mean_ = features.mean(axis=0)
        spread = features.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1.0)
        scaled = (features - self.mean_) / self.scale_
        self._build()
        rng = np.random.default_rng(self.seed)
        first = {key: np.zeros_like(value)
                 for key, value in self.parameters.items()}
        second = {key: np.zeros_like(value)
                  for key, value in self.parameters.items()}
        step = 0
        self.losses_ = []
        for _ in range(self.epochs):
            order = rng.permutation(labels.size)
            total, count = 0.0, 0
            for start in range(0, labels.size, self.batch_size):
                chosen = order[start:start + self.batch_size]
                logits, cache = self._forward(scaled[chosen])
                loss, gradient = _cross_entropy(logits, labels[chosen])
                self._backward(gradient, cache)
                step += 1
                for key, value in self.parameters.items():
                    update = self.gradients[key]
                    first[key] = 0.9 * first[key] + 0.1 * update
                    second[key] = 0.999 * second[key] + 0.001 * update ** 2
                    corrected = first[key] / (1.0 - 0.9 ** step)
                    scale = second[key] / (1.0 - 0.999 ** step)
                    value -= (self.learning_rate * corrected
                              / (np.sqrt(scale) + 1e-8))
                total += loss * chosen.size
                count += chosen.size
            self.losses_.append(total / count)
        self._fitted = True
        return self

    # -- prediction ------------------------------------------------------- #
    def _prepare(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        features = np.asarray(x, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.n_features_:
            raise ValueError(f"x must have shape (n, {self.n_features_})")
        if not np.all(np.isfinite(features)):
            raise ValueError("x must be finite")
        return (features - self.mean_) / self.scale_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scaled = self._prepare(x)
        logits, _ = self._forward(scaled)
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most likely class, mapped back to the original label values."""
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Accuracy against the original labels."""
        labels = np.asarray(y)
        predicted = self.predict(x)
        return float(np.mean(predicted == labels))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _spirals(n: int, classes: int, rng: np.random.Generator):
    """`classes` interleaved arms, the classic test for a nonlinear stack."""
    per = n // classes
    points, labels = [], []
    for index in range(classes):
        radius = np.linspace(0.15, 1.0, per)
        angle = np.linspace(index * 2.0 * np.pi / classes,
                            index * 2.0 * np.pi / classes + 2.2 * np.pi, per)
        arms = radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]
        points.append(arms + rng.normal(0.0, 0.05, arms.shape))
        labels.append(np.full(per, index))
    features = np.vstack(points)
    targets = np.concatenate(labels)
    order = rng.permutation(features.shape[0])
    return features[order], targets[order]


def _path_ratio(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Norm of the gradient at the input over the norm at the logits."""
    scaled = (x - model.mean_) / model.scale_
    logits, cache = model._forward(scaled)
    _, gradient = _cross_entropy(logits, y)
    start = float(np.linalg.norm(gradient))
    for block, block_cache in zip(reversed(model.blocks), reversed(cache),
                                  strict=True):
        gradient = block.backward(gradient, block_cache)
    return float(np.linalg.norm(gradient)) / start


def _fresh(variant: str, depth: int, x: np.ndarray, classes: int) -> MLP:
    """A built-but-untrained model with the training scaler installed."""
    model = MLP(variant, width=32, depth=depth, epochs=1, seed=0)
    model.n_features_ = x.shape[1]
    model.n_classes_ = classes
    model.classes_ = np.arange(classes)
    model.mean_ = x.mean(axis=0)
    spread = x.std(axis=0)
    model.scale_ = np.where(spread > 1e-12, spread, 1.0)
    model._build()
    return model


def demo() -> None:
    split = 300
    features, labels = _spirals(360, 3, np.random.default_rng(0))
    train_x, train_y = features[:split], labels[:split]
    test_x, test_y = features[split:], labels[split:]
    print("mlp. three interleaved spiral arms, 300 points to fit, 60 held out:")
    models = {}
    for variant in VARIANTS:
        depth = 6 if variant == "deep_mlp" else 3
        models[variant] = MLP(variant, width=32, depth=depth, epochs=300,
                              seed=0).fit(train_x, train_y)
    scores = {name: model.score(test_x, test_y)
              for name, model in models.items()}
    for variant in VARIANTS:
        model = models[variant]
        print(f"    {variant:<13s} depth {model.depth}  {model.n_params_:5d} "
              f"parameters  train {model.score(train_x, train_y):.3f}  "
              f"test {scores[variant]:.3f}")

    print("mlp. the affine model cannot bend, everything else can:")
    assert scores["linear"] <= 0.4
    assert all(scores[variant] >= 0.95 for variant in VARIANTS[1:])
    assert all(np.isfinite(model.losses_[-1]) for model in models.values())
    print(f"  the affine map lands at {scores['linear']:.3f}, which is chance")
    print(f"  ({1.0 / 3.0:.3f}) for three balanced classes, and the five stacks")
    print(f"  land between {min(scores[v] for v in VARIANTS[1:]):.3f} and "
          f"{max(scores[v] for v in VARIANTS[1:]):.3f} on sixty test points")

    print("mlp. a skip connection costs nothing to add:")
    plain, skipped = models["mlp"], models["residual_mlp"]
    assert plain.n_params_ == skipped.n_params_
    print(f"  both have {plain.n_params_} parameters; the residual adds x and")
    print(f"  nothing else. Held out they land at {scores['mlp']:.3f} and "
          f"{scores['residual_mlp']:.3f}")
    residual = skipped.blocks[1]
    for key in residual.parameters:
        residual.parameters[key][...] = 0.0
    identity, _ = residual.forward(np.ones((3, 32)))
    assert np.allclose(identity, 1.0)
    print("  with the branch zeroed the block returns its input exactly")

    print("mlp. the gate is a real modulation, not a constant:")
    gate_block = models["gated_mlp"].blocks[1]
    scaled = (train_x - models["gated_mlp"].mean_) \
        / models["gated_mlp"].scale_
    stem, _ = models["gated_mlp"].blocks[0].forward(scaled)
    _, gate_cache = gate_block.forward(stem)
    gate = gate_cache[3]
    assert 0.2 < float(gate.mean()) < 0.8
    assert float(gate.min()) < 0.5 < float(gate.max())
    print(f"  mean gate {float(gate.mean()):.3f}, and "
          f"{float(np.mean(gate > 0.5)):.3f} of the units sit above one half")
    print(f"  the gate matrix is why `gated_mlp` carries "
          f"{models['gated_mlp'].n_params_ - plain.n_params_} parameters more "
          "than the plain stack")

    print("mlp. at twelve layers the plain stack is a gradient lottery:")
    ratios = {}
    for variant in ("mlp", "deep_mlp"):
        ratios[variant] = [_path_ratio(_fresh(variant, depth, train_x, 3),
                                       train_x, train_y)
                           for depth in (2, 4, 8, 12)]
    assert ratios["deep_mlp"][-1] > 2.0 * ratios["mlp"][-1]
    print("  depth          2      4      8     12")
    for variant in ("mlp", "deep_mlp"):
        row = "  ".join(f"{value:6.3f}" for value in ratios[variant])
        print(f"  {variant:<12s} {row}")
    print("  the plain column hovers near one whatever the depth; the")
    print("  residual column grows, because x reaches the top untouched")

    print("mlp. zeroing every branch leaves the identity path alone:")
    bridged = _fresh("deep_mlp", 12, train_x, 3)
    chain = bridged.blocks[1:-1]
    for block in chain:
        for key in block.parameters:
            block.parameters[key][...] = 0.0
    scaled = (train_x - bridged.mean_) / bridged.scale_
    hidden, _ = bridged.blocks[0].forward(scaled)
    caches = []
    for block in chain:
        hidden, block_cache = block.forward(hidden)
        caches.append(block_cache)
    probe = np.random.default_rng(1).normal(0.0, 1.0, hidden.shape)
    carried = probe
    for block, block_cache in zip(reversed(chain), reversed(caches),
                                  strict=True):
        carried = block.backward(carried, block_cache)
    assert np.allclose(carried, probe)
    print("  with all twelve branches at zero a gradient crosses the chain")
    print(f"  unchanged: norm ratio "
          f"{float(np.linalg.norm(carried) / np.linalg.norm(probe)):.6f}")

    print("mlp. and depth still trains, just not predictably:")
    plain_twelve = MLP("mlp", width=32, depth=12, epochs=300, seed=0).fit(
        train_x, train_y)
    deep_twelve = MLP("deep_mlp", width=32, depth=12, epochs=300, seed=0).fit(
        train_x, train_y)
    assert plain_twelve.n_params_ == deep_twelve.n_params_
    for model in (plain_twelve, deep_twelve):
        assert np.isfinite(model.losses_[-1])
        assert model.score(test_x, test_y) > scores["linear"]
    print(f"  both are twelve blocks of width 32 and "
          f"{plain_twelve.n_params_} parameters")
    print(f"  the plain stack ends at {plain_twelve.score(test_x, test_y):.3f} "
          f"and the residual at {deep_twelve.score(test_x, test_y):.3f},")
    print("  but the order flips with the epoch budget and the seed, so the")
    print("  gradient path is the claim and the accuracy is an anecdote")

    print("  the pieces are what they claim:")
    probe = np.array([[1.0, 2.0, 3.0, 4.0]])
    normalised, _ = _layer_norm(probe)
    assert np.allclose(normalised.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(normalised.std(axis=1), 1.0, atol=1e-4)
    assert abs(float(_silu(probe)[0, 1]) - 2.0 * _sigmoid(2.0)) < 1e-12
    print(f"  layer norm maps [1, 2, 3, 4] to "
          f"{np.round(normalised[0], 4).tolist()}")
    print(f"  mean {float(normalised[0].mean()):+.1e}, spread "
          f"{float(normalised[0].std()):.6f}")
    print(f"  silu(2) is {float(_silu(probe)[0, 1]):.6f} = 2 * sigmoid(2)")

    bad = [
        lambda: MLP("mlp_lite"),
        lambda: MLP("mlp", width=0),
        lambda: MLP("mlp", depth=0),
        lambda: MLP("mlp", epochs=0),
        lambda: MLP("mlp", batch_size=0),
        lambda: MLP("mlp", depth=2.5),
        lambda: MLP("mlp", epochs=1, learning_rate=0.0),
        lambda: MLP("mlp", epochs=1, learning_rate=np.nan),
        lambda: MLP("mlp", epochs=1).fit(train_x[0], train_y),
        lambda: MLP("mlp", epochs=1).fit(train_x, train_y[:-1]),
        lambda: MLP("mlp", epochs=1).fit(np.full_like(train_x, np.nan),
                                         train_y),
        lambda: MLP("mlp", epochs=1).fit(train_x[:3], train_y[:3]),
        lambda: MLP("mlp", epochs=1).fit(train_x, -train_y - 1),
        lambda: MLP("mlp", epochs=1).fit(train_x, train_y + 0.5),
        lambda: MLP("mlp", epochs=1).fit(train_x, np.zeros_like(train_y)),
        lambda: models["mlp"].predict_proba(train_x[:, :1]),
        lambda: models["mlp"].predict_proba(
            np.full_like(train_x, np.inf)),
        lambda: MLP("mlp", epochs=1).predict(train_x),
        lambda: MLP("mlp", epochs=1).predict_proba(train_x),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive width, depth, epoch count or batch size,")
    print("  a non-integer depth, a zero or non-finite learning rate, a")
    print("  one-dimensional feature matrix, a label vector of the wrong")
    print("  length, non-finite features, fewer than four rows, negative or")
    print("  fractional labels, a single class, a prediction with the wrong")
    print("  number of columns or a non-finite entry, and prediction before")
    print("  fitting")
    print("mlp demo passed")


if __name__ == "__main__":
    demo()
