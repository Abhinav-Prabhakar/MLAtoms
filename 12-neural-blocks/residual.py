"""Residual blocks — the identity path and six ways to dress it up.

Chapter:   12-neural-blocks
Companion: residual.md
Tier 1 · 7 variants: basic, bottleneck, preactivation, densenet, highway,
scaled_residual, stochastic_depth
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("basic", "bottleneck", "preactivation", "densenet", "highway",
            "scaled_residual", "stochastic_depth")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(pre: np.ndarray) -> np.ndarray:
    return (pre > 0.0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    tail = np.exp(z[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    spread = np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    inverse = 1.0 / spread
    return centred * inverse, (centred, inverse)


def _layer_norm_backward(gradient: np.ndarray, cache):
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)


def _affine(name: str, inputs: int, outputs: int, rng: np.random.Generator):
    scale = np.sqrt(2.0 / inputs)
    return {f"{name}.W": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b": np.zeros(outputs)}


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
# Blocks
# --------------------------------------------------------------------------- #
class _Block:
    """A width-to-width map with its own parameters and gradient buffers."""

    def __init__(self, name: str, width: int):
        self.name = name
        self.width = width
        self.out_width = width
        self.parameters: dict[str, np.ndarray] = {}
        self.gradients: dict[str, np.ndarray] = {}
        self._finish()

    def _finish(self) -> None:
        for key, value in self.parameters.items():
            self.gradients[key] = np.zeros_like(value)

    def forward(self, x: np.ndarray, *, training: bool = True,
                rng: np.random.Generator | None = None):
        raise NotImplementedError

    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        raise NotImplementedError


class _Basic(_Block):
    """Two affine layers with a ReLU between, added to the input."""

    def __init__(self, name: str, width: int, rng: np.random.Generator):
        super().__init__(name, width)
        self.parameters = {**_affine(f"{name}.first", width, width, rng),
                           **_affine(f"{name}.second", width, width, rng)}
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        pre = x @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        return x + branch, (x, pre, hidden)

    def backward(self, gradient, cache):
        x, pre, hidden = cache
        p, g = self.parameters, self.gradients
        g[f"{self.name}.second.W"][...] = hidden.T @ gradient
        g[f"{self.name}.second.b"][...] = gradient.sum(axis=0)
        d_hidden = gradient @ p[f"{self.name}.second.W"].T
        d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.first.W"][...] = x.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        return gradient + d_pre @ p[f"{self.name}.first.W"].T


class _Bottleneck(_Block):
    """Project down, transform at low width, project back up."""

    def __init__(self, name: str, width: int, rng: np.random.Generator,
                 ratio: int = 4):
        super().__init__(name, width)
        narrow = max(width // ratio, 1)
        self.narrow = narrow
        self.parameters = {**_affine(f"{name}.down", width, narrow, rng),
                           **_affine(f"{name}.inner", narrow, narrow, rng),
                           **_affine(f"{name}.up", narrow, width, rng)}
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        down = x @ p[f"{self.name}.down.W"] + p[f"{self.name}.down.b"]
        pre = _relu(down) @ p[f"{self.name}.inner.W"] + p[f"{self.name}.inner.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.up.W"] + p[f"{self.name}.up.b"]
        return x + branch, (x, down, pre, hidden)

    def backward(self, gradient, cache):
        x, down, pre, hidden = cache
        p, g = self.parameters, self.gradients
        g[f"{self.name}.up.W"][...] = hidden.T @ gradient
        g[f"{self.name}.up.b"][...] = gradient.sum(axis=0)
        d_hidden = gradient @ p[f"{self.name}.up.W"].T
        d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.inner.W"][...] = _relu(down).T @ d_pre
        g[f"{self.name}.inner.b"][...] = d_pre.sum(axis=0)
        d_down = d_pre @ p[f"{self.name}.inner.W"].T * _relu_grad(down)
        g[f"{self.name}.down.W"][...] = x.T @ d_down
        g[f"{self.name}.down.b"][...] = d_down.sum(axis=0)
        return gradient + d_down @ p[f"{self.name}.down.W"].T


class _PreActivation(_Block):
    """Normalise on the way into the branch, never on the way out."""

    def __init__(self, name: str, width: int, rng: np.random.Generator):
        super().__init__(name, width)
        self.parameters = {**_affine(f"{name}.first", width, width, rng),
                           **_affine(f"{name}.second", width, width, rng)}
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        normed, norm_cache = _layer_norm(x)
        pre = normed @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        return x + branch, (normed, norm_cache, pre, hidden)

    def backward(self, gradient, cache):
        normed, norm_cache, pre, hidden = cache
        p, g = self.parameters, self.gradients
        g[f"{self.name}.second.W"][...] = hidden.T @ gradient
        g[f"{self.name}.second.b"][...] = gradient.sum(axis=0)
        d_hidden = gradient @ p[f"{self.name}.second.W"].T
        d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.first.W"][...] = normed.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        d_normed = d_pre @ p[f"{self.name}.first.W"].T
        return gradient + _layer_norm_backward(d_normed, norm_cache)


class _Dense(_Block):
    """Concatenate instead of add: the block's output contains its input."""

    def __init__(self, name: str, width: int, growth: int,
                 rng: np.random.Generator):
        super().__init__(name, width)
        self.growth = growth
        self.out_width = width + growth
        self.parameters = _affine(f"{name}.transform", width, growth, rng)
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        pre = x @ p[f"{self.name}.transform.W"] + p[f"{self.name}.transform.b"]
        branch = _relu(pre)
        return np.concatenate([x, branch], axis=1), (x, pre, branch)

    def backward(self, gradient, cache):
        x, pre, branch = cache
        p, g = self.parameters, self.gradients
        width = x.shape[1]
        d_branch = gradient[:, width:] * _relu_grad(pre)
        g[f"{self.name}.transform.W"][...] = x.T @ d_branch
        g[f"{self.name}.transform.b"][...] = d_branch.sum(axis=0)
        return gradient[:, :width] + d_branch @ p[f"{self.name}.transform.W"].T


class _Highway(_Block):
    """A learned gate mixes the branch and the identity, per unit."""

    def __init__(self, name: str, width: int, rng: np.random.Generator):
        super().__init__(name, width)
        self.parameters = {**_affine(f"{name}.first", width, width, rng),
                           **_affine(f"{name}.second", width, width, rng),
                           **_affine(f"{name}.gate", width, width, rng)}
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        pre = x @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        gate = _sigmoid(x @ p[f"{self.name}.gate.W"]
                        + p[f"{self.name}.gate.b"])
        mixed = gate * branch + (1.0 - gate) * x
        return mixed, (x, pre, hidden, gate, branch)

    def backward(self, gradient, cache):
        x, pre, hidden, gate, branch = cache
        p, g = self.parameters, self.gradients
        d_branch = gradient * gate
        d_gate = gradient * (branch - x)
        g[f"{self.name}.second.W"][...] = hidden.T @ d_branch
        g[f"{self.name}.second.b"][...] = d_branch.sum(axis=0)
        d_hidden = d_branch @ p[f"{self.name}.second.W"].T
        d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.first.W"][...] = x.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        d_gate_pre = d_gate * gate * (1.0 - gate)
        g[f"{self.name}.gate.W"][...] = x.T @ d_gate_pre
        g[f"{self.name}.gate.b"][...] = d_gate_pre.sum(axis=0)
        d_x = gradient * (1.0 - gate) + d_gate_pre @ p[f"{self.name}.gate.W"].T
        return d_x + d_pre @ p[f"{self.name}.first.W"].T


class _Scaled(_Basic):
    """Multiply the branch by one learned scalar, initialised at zero."""

    def __init__(self, name: str, width: int, rng: np.random.Generator):
        super().__init__(name, width, rng)
        self.parameters[f"{name}.alpha"] = np.zeros(1)
        self._finish()

    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        pre = x @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        alpha = p[f"{self.name}.alpha"]
        return x + alpha * branch, (x, pre, hidden, branch)

    def backward(self, gradient, cache):
        x, pre, hidden, branch = cache
        p, g = self.parameters, self.gradients
        alpha = p[f"{self.name}.alpha"]
        g[f"{self.name}.alpha"][...] = float(np.sum(gradient * branch))
        d_branch = gradient * alpha
        g[f"{self.name}.second.W"][...] = hidden.T @ d_branch
        g[f"{self.name}.second.b"][...] = d_branch.sum(axis=0)
        d_hidden = d_branch @ p[f"{self.name}.second.W"].T
        d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.first.W"][...] = x.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        return gradient + d_pre @ p[f"{self.name}.first.W"].T


class _StochasticDepth(_Basic):
    """Drop the whole branch, per sample, during training."""

    def __init__(self, name: str, width: int, survival: float,
                 rng: np.random.Generator):
        super().__init__(name, width, rng)
        if not 0.0 < survival <= 1.0:
            raise ValueError("survival must be in (0, 1]")
        self.survival = survival

    def forward(self, x, *, training=True, rng=None):
        output, cache = super().forward(x, training=training, rng=rng)
        branch = output - x
        if not training or self.survival == 1.0:
            return x + branch, (cache, None)
        if rng is None:
            raise ValueError("stochastic_depth needs a generator while training")
        keep = (rng.random((x.shape[0], 1)) < self.survival).astype(float)
        return x + keep * branch / self.survival, (cache, keep)

    def backward(self, gradient, cache):
        basic_cache, keep = cache
        if keep is None:
            return super().backward(gradient, basic_cache)
        return super().backward(gradient * keep / self.survival, basic_cache)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class ResNet:
    """A stem, a stack of residual blocks, and a linear head."""

    def __init__(self, variant: str = "basic", *, width: int = 16,
                 depth: int = 16, growth: int = 2, classes: int = 2,
                 epochs: int = 200, batch_size: int = 32,
                 learning_rate: float = 0.01, final_survival: float = 0.5,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if width < 1 or depth < 1 or growth < 1:
            raise ValueError("width, depth and growth must be positive")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0.0 < final_survival <= 1.0:
            raise ValueError("final_survival must be in (0, 1]")
        self.variant = variant
        self.width = width
        self.depth = depth
        self.growth = growth
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.final_survival = final_survival
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fitted = False

    def _build(self) -> None:
        rng = self.rng
        self.parameters = _affine("stem", self.n_features_, self.width, rng)
        self.blocks = []
        current = self.width
        for index in range(self.depth):
            name = f"block{index}"
            if self.variant == "basic":
                block = _Basic(name, current, rng)
            elif self.variant == "bottleneck":
                block = _Bottleneck(name, current, rng)
            elif self.variant == "preactivation":
                block = _PreActivation(name, current, rng)
            elif self.variant == "densenet":
                block = _Dense(name, current, self.growth, rng)
            elif self.variant == "highway":
                block = _Highway(name, current, rng)
            elif self.variant == "scaled_residual":
                block = _Scaled(name, current, rng)
            else:
                survival = 1.0 - (1.0 - self.final_survival) * index \
                    / max(self.depth - 1, 1)
                block = _StochasticDepth(name, current, survival, rng)
            self.blocks.append(block)
            current = block.out_width
        self.parameters.update(_affine("head", current, self.classes, rng))
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}
        for block in self.blocks:
            self.parameters.update(block.parameters)
            self.gradients.update(block.gradients)
        self.n_params_ = int(sum(value.size for value in self.parameters.values()))

    def _forward(self, x: np.ndarray, *, training: bool):
        p = self.parameters
        pre = x @ p["stem.W"] + p["stem.b"]
        hidden = _relu(pre)
        block_caches = []
        for block in self.blocks:
            hidden, block_cache = block.forward(hidden, training=training,
                                                rng=self.rng)
            block_caches.append(block_cache)
        logits = hidden @ p["head.W"] + p["head.b"]
        return logits, (x, pre, hidden, block_caches)

    def _backward(self, gradient: np.ndarray, cache) -> None:
        x, pre, hidden, block_caches = cache
        p, g = self.parameters, self.gradients
        g["head.W"][...] = hidden.T @ gradient
        g["head.b"][...] = gradient.sum(axis=0)
        d_hidden = gradient @ p["head.W"].T
        for block, block_cache in zip(reversed(self.blocks),
                                      reversed(block_caches), strict=True):
            d_hidden = block.backward(d_hidden, block_cache)
        d_pre = d_hidden * _relu_grad(pre)
        g["stem.W"][...] = x.T @ d_pre
        g["stem.b"][...] = d_pre.sum(axis=0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> ResNet:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2:
            raise ValueError("x must be two-dimensional")
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError("y must be a label vector of the same length")
        if x.shape[0] < 4:
            raise ValueError("need at least four rows to fit")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        if np.any(y != y.astype(int)):
            raise ValueError("labels must be integers")
        labels = y.astype(int)
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("need at least two classes")
        if classes.size != self.classes:
            raise ValueError("the label count does not match `classes`")
        self.classes_ = classes
        self.n_features_ = x.shape[1]
        self.mean_ = x.mean(axis=0)
        spread = x.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1.0)
        scaled = (x - self.mean_) / self.scale_
        self._build()
        self.losses_ = []
        n = scaled.shape[0]
        steps = 0
        first = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        second = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        for _ in range(self.epochs):
            order = self.rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                rows = order[start:start + self.batch_size]
                logits, cache = self._forward(scaled[rows], training=True)
                loss, gradient = _cross_entropy(logits, labels[rows])
                total += loss * rows.size
                self._backward(gradient, cache)
                steps += 1
                for key, value in self.parameters.items():
                    grad = self.gradients[key]
                    first[key] = 0.9 * first[key] + 0.1 * grad
                    second[key] = 0.999 * second[key] + 0.001 * grad ** 2
                    corrected_first = first[key] / (1.0 - 0.9 ** steps)
                    corrected_second = second[key] / (1.0 - 0.999 ** steps)
                    value -= self.learning_rate * corrected_first \
                        / (np.sqrt(corrected_second) + 1e-8)
            self.losses_.append(total / n)
        self._fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_:
            raise ValueError("x has the wrong number of columns")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        logits, _ = self._forward((x - self.mean_) / self.scale_, training=False)
        return _softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the model before predicting")
        return self.classes_[self.predict_proba(x).argmax(axis=1)]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        if np.any(y != y.astype(int)):
            raise ValueError("labels must be integers")
        return float(np.mean(self.predict(x) == y.astype(int)))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _circles(n: int, rng: np.random.Generator):
    """Two rings: an inner disk and an outer annulus, equally sized."""
    half = n // 2
    radius = np.concatenate([rng.uniform(0.05, 0.45, half),
                             rng.uniform(0.65, 1.05, n - half)])
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    features = np.c_[radius * np.cos(angle), radius * np.sin(angle)]
    features = features + rng.normal(0.0, 0.03, features.shape)
    labels = np.concatenate([np.zeros(half, dtype=int),
                             np.ones(n - half, dtype=int)])
    order = rng.permutation(n)
    return features[order], labels[order]


def demo() -> None:
    rng = np.random.default_rng(0)
    x, y = _circles(400, rng)
    order = rng.permutation(x.shape[0])
    x, y = x[order], y[order]
    train_x, train_y = x[:320], y[:320]
    test_x, test_y = x[320:], y[320:]
    print("residual. two rings, 320 points to fit and 80 held out:")

    models = {}
    for variant in VARIANTS:
        models[variant] = ResNet(variant, width=16, depth=12, epochs=80,
                                 final_survival=0.8,
                                 seed=0).fit(train_x, train_y)
    scores = {name: model.score(test_x, test_y)
              for name, model in models.items()}
    for variant in VARIANTS:
        model = models[variant]
        print(f"    {variant:<16s} width {model.width:2d} to "
              f"{model.blocks[-1].out_width:2d}  {model.n_params_:5d} "
              f"parameters  test {scores[variant]:.3f}")
    assert all(score > 0.85 for score in scores.values())
    print("  every variant clears 0.85; a line through the origin is at")
    print("  chance on two concentric rings")

    print("residual. a zeroed branch leaves the identity path alone:")
    identity = np.random.default_rng(1).normal(0.0, 1.0, (32, 16))
    for variant in ("basic", "bottleneck", "preactivation", "scaled_residual"):
        fresh = ResNet(variant, width=16, depth=1, epochs=1, seed=3)
        fresh.n_features_ = 16
        fresh._build()
        block = fresh.blocks[0]
        for key in block.parameters:
            block.parameters[key][...] = 0.0
        carried, _ = block.forward(identity, training=False)
        assert np.allclose(carried, identity)
    print("  basic, bottleneck, preactivation and scaled residual all return")
    print("  their input exactly when the branch is zero")

    print("residual. depth does not change the width unless it is dense:")
    for variant in ("basic", "bottleneck", "preactivation", "highway",
                    "scaled_residual", "stochastic_depth"):
        model = models[variant]
        assert all(block.out_width == 16 for block in model.blocks)
    dense = models["densenet"]
    assert dense.blocks[-1].out_width == 16 + 2 * 12
    assert np.allclose(dense.blocks[0].forward(
        np.ones((1, 16)), training=False)[0][0, :16], 1.0)
    print(f"  densenet grows from 16 to {dense.blocks[-1].out_width} over 12")
    print("  blocks, and each block's first columns are its input verbatim")

    print("residual. the bottleneck pays for its depth:")
    plain = models["basic"].n_params_
    thin = models["bottleneck"].n_params_
    wide = models["highway"].n_params_
    assert thin < plain < wide
    print(f"  at width 16 and depth 12: basic {plain}, bottleneck {thin},")
    print(f"  highway {wide} parameters; the bottleneck's three matrices")
    print(f"  cost {thin / plain:.2f} of the basic block's two")

    print("residual. scaling starts at zero, so the block starts as identity:")
    model = models["scaled_residual"]
    alphas = np.array([float(block.parameters[f"{block.name}.alpha"][0])
                       for block in model.blocks])
    assert np.all(np.abs(alphas) > 0.0)
    assert np.all(np.isfinite(alphas))
    print(f"  after 80 epochs the {alphas.size} scalars lie between "
          f"{alphas.min():.3f} and {alphas.max():.3f}")
    print("  and none is still zero, so every branch has been admitted")

    print("residual. a highway gate is a per-unit mixture, not a switch:")
    model = models["highway"]
    hidden = np.ones((64, 16))
    for block in model.blocks[:3]:
        hidden, _ = block.forward(hidden, training=False)
    gate = _sigmoid(hidden @ model.parameters["block0.gate.W"]
                    + model.parameters["block0.gate.b"])
    assert 0.05 < float(gate.mean()) < 0.95
    assert float(gate.min()) < 0.5 < float(gate.max())
    print(f"  the first gate has mean {float(gate.mean()):.3f} and spans")
    print(f"  {float(gate.min()):.3f} to {float(gate.max()):.3f}, so units")
    print("  carry different mixtures of branch and input")

    print("residual. stochastic depth is identity at evaluation:")
    model = models["stochastic_depth"]
    block = model.blocks[5]
    probe = np.random.default_rng(2).normal(0.0, 1.0, (8, 16))
    branch, _ = _Basic.forward(block, probe, training=False)
    evaluated, _ = block.forward(probe, training=False)
    assert np.allclose(evaluated, branch)
    masks = [block.forward(probe, training=True,
                           rng=np.random.default_rng(seed))[0]
             for seed in range(30)]
    assert max(float(np.abs(part - branch).max()) for part in masks) > 0.0

    class _KeepAll:
        def random(self, shape):
            return np.zeros(shape)

    dropped = _StochasticDepth("probe", 8, 0.5, np.random.default_rng(0))
    trained, _ = dropped.forward(probe[:, :8], training=True, rng=_KeepAll())
    full, _ = _Basic.forward(dropped, probe[:, :8], training=False)
    assert np.allclose(trained, full + (full - probe[:, :8]))
    print(f"  block 5 keeps {block.survival:.3f} of its branches per draw;")
    print("  evaluation uses the full branch with no rescaling, so depth at")
    print("  test time is deterministic and deeper than most training draws")

    print("residual. the reported loss falls for every variant:")
    for variant in VARIANTS:
        losses = models[variant].losses_
        assert min(losses) < 0.5 * losses[0]
        assert np.all(np.isfinite(losses))
    print("  every variant at least halves its first-epoch loss; stochastic")
    print("  depth's curve is the noisiest because each step sees a")
    print("  different subset of branches")

    bad = [
        lambda: ResNet("residual_lite"),
        lambda: ResNet("basic", width=0),
        lambda: ResNet("basic", depth=0),
        lambda: ResNet("basic", growth=0),
        lambda: ResNet("basic", epochs=0),
        lambda: ResNet("basic", batch_size=0),
        lambda: ResNet("basic", learning_rate=0.0),
        lambda: ResNet("basic", learning_rate=np.nan),
        lambda: ResNet("basic", final_survival=0.0),
        lambda: ResNet("basic", final_survival=1.5),
        lambda: ResNet("basic", epochs=1).fit(train_x[0], train_y),
        lambda: ResNet("basic", epochs=1).fit(train_x, train_y[:-1]),
        lambda: ResNet("basic", epochs=1).fit(train_x[:3], train_y[:3]),
        lambda: ResNet("basic", epochs=1).fit(
            np.full_like(train_x, np.inf), train_y),
        lambda: ResNet("basic", epochs=1).fit(train_x, train_y + 0.5),
        lambda: ResNet("basic", epochs=1).fit(train_x, np.zeros_like(train_y)),
        lambda: ResNet("basic", epochs=1, classes=3).fit(train_x, train_y),
        lambda: models["basic"].predict_proba(train_x[:, :1]),
        lambda: models["basic"].predict_proba(
            np.full_like(train_x, np.nan)),
        lambda: ResNet("basic", epochs=1).predict(train_x),
        lambda: models["basic"].score(train_x, train_y + 0.5),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive width, depth, growth, epoch count or")
    print("  batch size, a zero or non-finite learning rate, a survival")
    print("  outside (0, 1], a one-dimensional feature matrix, a label vector")
    print("  of the wrong length, fewer than four rows, non-finite features,")
    print("  fractional or single-class labels, a class count that disagrees")
    print("  with the labels, a prediction with the wrong width or a")
    print("  non-finite entry, prediction before fitting, and a score with")
    print("  fractional labels")
    print("residual demo passed")


if __name__ == "__main__":
    demo()
