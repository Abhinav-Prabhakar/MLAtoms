"""Transformer blocks — attention, norms and the order they are applied in.

Chapter:   12-neural-blocks
Companion: transformer.md
Tier 1 · 8 variants: post_ln, pre_ln, encoder, decoder, cross_attention,
swiglu_ffn, rmsnorm_block, parallel_attention
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("post_ln", "pre_ln", "encoder", "decoder", "cross_attention",
            "swiglu_ffn", "rmsnorm_block", "parallel_attention")


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(pre: np.ndarray) -> np.ndarray:
    return (pre > 0.0).astype(float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    negative = np.exp(x[~positive])
    out[~positive] = negative / (1.0 + negative)
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _silu_grad(x: np.ndarray) -> np.ndarray:
    sigmoid = _sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    inverse = 1.0 / np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    return centred * inverse, (centred, inverse)


def _layer_norm_backward(gradient: np.ndarray, cache):
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)


def _rms_norm(x: np.ndarray):
    inverse = 1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    return x * inverse, (x, inverse)


def _rms_norm_backward(gradient: np.ndarray, cache):
    x, inverse = cache
    width = x.shape[-1]
    inner = (gradient * x).sum(axis=-1, keepdims=True)
    return inverse * gradient - (inverse ** 3) * x * inner / width


def _affine(name: str, inputs: int, outputs: int, rng: np.random.Generator):
    scale = np.sqrt(2.0 / inputs)
    return {f"{name}.W": rng.normal(0.0, scale, (inputs, outputs)),
            f"{name}.b": np.zeros(outputs)}


def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    probabilities = _softmax(logits)
    rows = np.arange(labels.size)
    loss = float(-np.mean(np.log(probabilities[rows, labels] + 1e-12)))
    gradient = probabilities.copy()
    gradient[rows, labels] -= 1.0
    return loss, gradient / labels.size


# --------------------------------------------------------------------------- #
# Multi-head attention
# --------------------------------------------------------------------------- #
class _Attention:
    """Scaled dot-product attention with an optional separate memory."""

    def __init__(self, name: str, dim: int, heads: int,
                 rng: np.random.Generator):
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        self.name = name
        self.dim = dim
        self.heads = heads
        self.depth = dim // heads
        self.scale = 1.0 / np.sqrt(self.depth)
        self.parameters = {}
        for tag in ("q", "k", "v", "o"):
            self.parameters.update(_affine(f"{name}.{tag}", dim, dim, rng))
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, x: np.ndarray, *, memory: np.ndarray | None = None,
                causal: bool = False):
        batch, length, dim = x.shape
        source = x if memory is None else memory
        places = source.shape[1]
        p = self.parameters
        query = x @ p[f"{self.name}.q.W"] + p[f"{self.name}.q.b"]
        key = source @ p[f"{self.name}.k.W"] + p[f"{self.name}.k.b"]
        value = source @ p[f"{self.name}.v.W"] + p[f"{self.name}.v.b"]
        heads = self.heads
        depth = self.depth
        q = query.reshape(batch, length, heads, depth).transpose(0, 2, 1, 3)
        k = key.reshape(batch, places, heads, depth).transpose(0, 2, 1, 3)
        v = value.reshape(batch, places, heads, depth).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = None
        if causal:
            mask = np.triu(np.ones((length, places), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        weights = _softmax(scores)
        context = weights @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        output = merged @ p[f"{self.name}.o.W"] + p[f"{self.name}.o.b"]
        cache = (x, source, memory is not None, mask, q, k, v, weights,
                 merged, batch, length, places)
        return output, cache

    def backward(self, gradient: np.ndarray, cache):
        (x, source, cross, mask, q, k, v, weights, merged,
         batch, length, places) = cache
        p, g = self.parameters, self.gradients
        dim = self.dim
        g[f"{self.name}.o.W"][...] = merged.reshape(-1, dim).T \
            @ gradient.reshape(-1, dim)
        g[f"{self.name}.o.b"][...] = gradient.sum(axis=(0, 1))
        d_merged = gradient @ p[f"{self.name}.o.W"].T
        d_context = d_merged.reshape(batch, length, self.heads,
                                     self.depth).transpose(0, 2, 1, 3)
        d_weights = d_context @ v.transpose(0, 1, 3, 2)
        d_v = weights.transpose(0, 1, 3, 2) @ d_context
        d_scores = weights * (d_weights
                              - (d_weights * weights).sum(axis=-1,
                                                          keepdims=True))
        if mask is not None:
            d_scores = np.where(mask, 0.0, d_scores)
        d_scores *= self.scale
        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q
        dq = d_q.transpose(0, 2, 1, 3).reshape(batch * length, dim)
        dk = d_k.transpose(0, 2, 1, 3).reshape(batch * places, dim)
        dv = d_v.transpose(0, 2, 1, 3).reshape(batch * places, dim)
        flat_x = x.reshape(batch * length, dim)
        flat_source = source.reshape(batch * places, dim)
        g[f"{self.name}.q.W"][...] = flat_x.T @ dq
        g[f"{self.name}.q.b"][...] = dq.sum(axis=0)
        g[f"{self.name}.k.W"][...] = flat_source.T @ dk
        g[f"{self.name}.k.b"][...] = dk.sum(axis=0)
        g[f"{self.name}.v.W"][...] = flat_source.T @ dv
        g[f"{self.name}.v.b"][...] = dv.sum(axis=0)
        d_query = dq @ p[f"{self.name}.q.W"].T
        d_source = dk @ p[f"{self.name}.k.W"].T + dv @ p[f"{self.name}.v.W"].T
        d_query = d_query.reshape(batch, length, dim)
        d_source = d_source.reshape(batch, places, dim)
        if cross:
            return d_query, d_source
        return d_query + d_source, None


# --------------------------------------------------------------------------- #
# Feed-forward network
# --------------------------------------------------------------------------- #
class _FeedForward:
    """Two matrices, ReLU, and optionally a SwiGLU gate."""

    def __init__(self, name: str, dim: int, hidden: int, kind: str,
                 rng: np.random.Generator):
        self.name = name
        self.kind = kind
        self.parameters = {**_affine(f"{name}.first", dim, hidden, rng),
                           **_affine(f"{name}.second", hidden, dim, rng)}
        if kind == "swiglu":
            self.parameters.update(_affine(f"{name}.gate", dim, hidden, rng))
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}

    def forward(self, x: np.ndarray):
        p = self.parameters
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        pre = flat @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        if self.kind == "swiglu":
            gate_pre = flat @ p[f"{self.name}.gate.W"] \
                + p[f"{self.name}.gate.b"]
            hidden = _silu(pre) * gate_pre
            output = hidden @ p[f"{self.name}.second.W"] \
                + p[f"{self.name}.second.b"]
            return output.reshape(shape), (flat, pre, gate_pre, hidden)
        hidden = _relu(pre)
        output = hidden @ p[f"{self.name}.second.W"] \
            + p[f"{self.name}.second.b"]
        return output.reshape(shape), (flat, pre, hidden)

    def backward(self, gradient: np.ndarray, cache):
        p, g = self.parameters, self.gradients
        shape = gradient.shape
        flat_gradient = gradient.reshape(-1, shape[-1])
        if self.kind == "swiglu":
            x, pre, gate_pre, hidden = cache
        else:
            x, pre, hidden = cache
        g[f"{self.name}.second.W"][...] = hidden.T @ flat_gradient
        g[f"{self.name}.second.b"][...] = flat_gradient.sum(axis=0)
        d_hidden = flat_gradient @ p[f"{self.name}.second.W"].T
        d_gate = None
        if self.kind == "swiglu":
            d_pre = d_hidden * gate_pre * _silu_grad(pre)
            d_gate = d_hidden * _silu(pre)
            g[f"{self.name}.gate.W"][...] = x.T @ d_gate
            g[f"{self.name}.gate.b"][...] = d_gate.sum(axis=0)
        else:
            d_pre = d_hidden * _relu_grad(pre)
        g[f"{self.name}.first.W"][...] = x.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        d_input = d_pre @ p[f"{self.name}.first.W"].T
        if d_gate is not None:
            d_input = d_input + d_gate @ p[f"{self.name}.gate.W"].T
        return d_input.reshape(shape)


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #
class _Block:
    """One transformer block, with the layout selected by the variant."""

    def __init__(self, name: str, variant: str, dim: int, heads: int,
                 hidden: int, rng: np.random.Generator):
        self.name = name
        self.variant = variant
        self.norm = "rms" if variant == "rmsnorm_block" else "layer"
        self.causal = variant in ("pre_ln", "decoder")
        self.parallel = variant == "parallel_attention"
        self.parameters: dict[str, np.ndarray] = {}
        self.gradients: dict[str, np.ndarray] = {}

        def merge(module) -> None:
            self.parameters.update(module.parameters)
            self.gradients.update(module.gradients)

        if variant != "cross_attention":
            self.self_attention = _Attention(f"{name}.self", dim, heads, rng)
            merge(self.self_attention)
        if variant in ("decoder", "cross_attention"):
            self.cross_attention = _Attention(f"{name}.cross", dim, heads,
                                              rng)
            merge(self.cross_attention)
        self.feed_forward = _FeedForward(f"{name}.ffn", dim, hidden,
                                         "swiglu" if variant == "swiglu_ffn"
                                         else "relu", rng)
        merge(self.feed_forward)

    def _norm(self, x: np.ndarray):
        if self.norm == "rms":
            return _rms_norm(x)
        return _layer_norm(x)

    def _norm_backward(self, gradient: np.ndarray, cache):
        if self.norm == "rms":
            return _rms_norm_backward(gradient, cache)
        return _layer_norm_backward(gradient, cache)

    def forward(self, x: np.ndarray, *, memory: np.ndarray | None = None,
                training: bool = True):
        if self.variant == "post_ln":
            attention, attention_cache = self.self_attention.forward(x)
            normed, norm_cache = self._norm(x + attention)
            branch, ffn_cache = self.feed_forward.forward(normed)
            output, final_cache = self._norm(normed + branch)
            return output, ("post", attention_cache, norm_cache, ffn_cache,
                            final_cache)
        if self.variant == "cross_attention":
            normed, norm_cache = self._norm(x)
            branch, attention_cache = self.cross_attention.forward(
                normed, memory=memory)
            mixed = x + branch
            normed2, norm_cache2 = self._norm(mixed)
            forward_out, ffn_cache = self.feed_forward.forward(normed2)
            return mixed + forward_out, ("cross", norm_cache, attention_cache,
                                         norm_cache2, ffn_cache)
        if self.variant == "decoder":
            normed, norm_cache = self._norm(x)
            attention, attention_cache = self.self_attention.forward(
                normed, causal=True)
            mixed = x + attention
            normed2, norm_cache2 = self._norm(mixed)
            branch, cross_cache = self.cross_attention.forward(
                normed2, memory=memory, causal=True)
            mixed2 = mixed + branch
            normed3, norm_cache3 = self._norm(mixed2)
            forward_out, ffn_cache = self.feed_forward.forward(normed3)
            return mixed2 + forward_out, ("decoder", norm_cache,
                                          attention_cache, norm_cache2,
                                          cross_cache, norm_cache3, ffn_cache)
        normed, norm_cache = self._norm(x)
        if self.parallel:
            attention, attention_cache = self.self_attention.forward(normed)
            branch, ffn_cache = self.feed_forward.forward(normed)
            return x + attention + branch, ("parallel", norm_cache,
                                            attention_cache, ffn_cache)
        attention, attention_cache = self.self_attention.forward(
            normed, causal=self.causal)
        mixed = x + attention
        normed2, norm_cache2 = self._norm(mixed)
        branch, ffn_cache = self.feed_forward.forward(normed2)
        return mixed + branch, ("pre", norm_cache, attention_cache,
                                norm_cache2, ffn_cache)

    def backward(self, gradient: np.ndarray, cache):
        kind = cache[0]
        if kind == "post":
            _, attention_cache, norm_cache, ffn_cache, final_cache = cache
            d_mixed = self._norm_backward(gradient, final_cache)
            d_normed = d_mixed + self.feed_forward.backward(d_mixed,
                                                           ffn_cache)
            d_residual = self._norm_backward(d_normed, norm_cache)
            d_attention, _ = self.self_attention.backward(d_residual,
                                                          attention_cache)
            return d_residual + d_attention, None
        if kind == "cross":
            _, norm_cache, attention_cache, norm_cache2, ffn_cache = cache
            d_normed = self.feed_forward.backward(gradient, ffn_cache)
            d_mixed = gradient + self._norm_backward(d_normed, norm_cache2)
            d_normed_x, d_memory = self.cross_attention.backward(
                d_mixed, attention_cache)
            d_x = d_mixed + self._norm_backward(d_normed_x, norm_cache)
            return d_x, d_memory
        if kind == "decoder":
            (_, norm_cache, attention_cache, norm_cache2, cross_cache,
             norm_cache3, ffn_cache) = cache
            d_normed = self.feed_forward.backward(gradient, ffn_cache)
            d_mixed2 = gradient + self._norm_backward(d_normed, norm_cache3)
            d_normed2, d_memory = self.cross_attention.backward(
                d_mixed2, cross_cache)
            d_mixed = d_mixed2 + self._norm_backward(d_normed2, norm_cache2)
            d_normed1, _ = self.self_attention.backward(d_mixed,
                                                        attention_cache)
            d_x = d_mixed + self._norm_backward(d_normed1, norm_cache)
            return d_x, d_memory
        if kind == "parallel":
            _, norm_cache, attention_cache, ffn_cache = cache
            d_attention, _ = self.self_attention.backward(gradient,
                                                          attention_cache)
            d_branch = self.feed_forward.backward(gradient, ffn_cache)
            d_normed = d_attention + d_branch
            return gradient + self._norm_backward(d_normed, norm_cache), None
        _, norm_cache, attention_cache, norm_cache2, ffn_cache = cache
        d_normed = self.feed_forward.backward(gradient, ffn_cache)
        d_mixed = gradient + self._norm_backward(d_normed, norm_cache2)
        d_normed1, _ = self.self_attention.backward(d_mixed, attention_cache)
        d_x = d_mixed + self._norm_backward(d_normed1, norm_cache)
        return d_x, None


# --------------------------------------------------------------------------- #
# The sequence model
# --------------------------------------------------------------------------- #
class SeqModel:
    """Stack of blocks over tokens, read at token zero by a linear head."""

    def __init__(self, variant: str = "encoder", *, dim: int = 8,
                 heads: int = 2, hidden: int = 16, depth: int = 2,
                 classes: int = 2, epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.01, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if depth < 1 or hidden < 1:
            raise ValueError("depth and hidden must be positive")
        if dim % heads:
            raise ValueError("dim must divide evenly into heads")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.variant = variant
        self.dim = dim
        self.heads = heads
        self.hidden = hidden
        self.depth = depth
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._fitted = False

    def _build(self) -> None:
        rng = self.rng
        self.blocks = [_Block(f"block{index}", self.variant, self.dim,
                              self.heads, self.hidden, rng)
                       for index in range(self.depth)]
        self.parameters = {}
        self.gradients = {}
        for block in self.blocks:
            self.parameters.update(block.parameters)
            self.gradients.update(block.gradients)
        head = _affine("head", self.dim, self.classes, rng)
        self.parameters.update(head)
        self.gradients.update({key: np.zeros_like(value)
                               for key, value in head.items()})
        self.n_params_ = int(sum(value.size
                                 for value in self.parameters.values()))

    def _forward(self, x: np.ndarray):
        hidden = x
        caches = []
        for block in self.blocks:
            hidden, block_cache = block.forward(hidden, memory=x)
            caches.append(block_cache)
        pooled = hidden[:, 0]
        logits = pooled @ self.parameters["head.W"] + self.parameters["head.b"]
        return logits, (hidden, pooled, caches)

    def _backward(self, gradient: np.ndarray, cache) -> None:
        hidden, pooled, caches = cache
        p, g = self.parameters, self.gradients
        g["head.W"][...] = pooled.T @ gradient
        g["head.b"][...] = gradient.sum(axis=0)
        d_hidden = np.zeros_like(hidden)
        d_hidden[:, 0] = gradient @ p["head.W"].T
        d_memory = np.zeros_like(hidden)
        for block, block_cache in zip(reversed(self.blocks),
                                      reversed(caches), strict=True):
            d_hidden, d_cross = block.backward(d_hidden, block_cache)
            if d_cross is not None:
                d_memory = d_memory + d_cross
        if not np.all(np.isfinite(d_memory)):
            raise RuntimeError("the cross-attention gradient diverged")

    def fit(self, x: np.ndarray, y: np.ndarray) -> SeqModel:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 3:
            raise ValueError("x must be (batch, sequence, features)")
        if y.ndim != 1 or y.size != x.shape[0]:
            raise ValueError("y must be a label vector of the same length")
        if x.shape[0] < 4:
            raise ValueError("need at least four sequences to fit")
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
        self.n_features_ = x.shape[2]
        self._build()
        self.losses_ = []
        n = x.shape[0]
        steps = 0
        first = {key: np.zeros_like(value)
                 for key, value in self.parameters.items()}
        second = {key: np.zeros_like(value)
                  for key, value in self.parameters.items()}
        for _ in range(self.epochs):
            order = self.rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                rows = order[start:start + self.batch_size]
                logits, cache = self._forward(x[rows])
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
        if x.ndim != 3 or x.shape[2] != self.n_features_:
            raise ValueError("x has the wrong shape")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        logits, _ = self._forward(x)
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
def _marker_task(n: int, length: int, dim: int, rng: np.random.Generator):
    """Two markers per sequence; the label is the sign of their product."""
    features = rng.normal(0.0, 0.05, (n, length, dim))
    labels = np.zeros(n, dtype=int)
    for row in range(n):
        places = rng.choice(length, 2, replace=False)
        values = rng.choice([-1.0, 1.0], 2)
        features[row, places, 0] = 1.0
        features[row, places, 1] = values
        labels[row] = int(values[0] * values[1] > 0.0)
    return features, labels


def demo() -> None:
    rng = np.random.default_rng(0)
    features, labels = _marker_task(640, 8, 8, rng)
    order = rng.permutation(features.shape[0])
    features, labels = features[order], labels[order]
    train_x, train_y = features[:480], labels[:480]
    test_x, test_y = features[480:], labels[480:]
    print("transformer. two markers, label is the sign of their product:")
    print("  480 sequences of 8 tokens, 160 held out, read from token 0")

    models = {}
    for variant in VARIANTS:
        models[variant] = SeqModel(variant, dim=8, heads=2, hidden=16,
                                   depth=2, epochs=60, batch_size=64,
                                   seed=0).fit(train_x, train_y)
    scores = {name: model.score(test_x, test_y)
              for name, model in models.items()}
    for variant in VARIANTS:
        model = models[variant]
        print(f"    {variant:<17s} {model.n_params_:5d} parameters  "
              f"test {scores[variant]:.3f}")
    bidirectional = [v for v in VARIANTS if v not in ("pre_ln", "decoder")]
    assert all(scores[v] >= 0.9 for v in bidirectional)
    assert all(scores[v] <= 0.65 for v in ("pre_ln", "decoder"))
    print("  the six bidirectional blocks solve it; the two causal blocks")
    print("  cannot, because the second marker may arrive after the first")

    print("transformer. the causal mask is exactly the past:")
    sequence = features[:2, :, :]
    for variant in ("pre_ln", "decoder"):
        block = models[variant].blocks[0]
        base, _ = block.forward(sequence, memory=sequence)
        changed = sequence.copy()
        changed[:, 5:, 0] += 10.0
        after, _ = block.forward(changed, memory=sequence)
        assert np.allclose(after[:, :5], base[:, :5])
        assert not np.allclose(after[:, 5:], base[:, 5:])
    for variant in ("encoder", "post_ln", "cross_attention"):
        block = models[variant].blocks[0]
        base, _ = block.forward(sequence, memory=sequence)
        changed = sequence.copy()
        changed[:, 5:, 0] += 10.0
        after, _ = block.forward(changed, memory=sequence)
        if variant != "cross_attention":
            assert not np.allclose(after[:, :5], base[:, :5])
    print("  perturbing tokens 5 to 7 leaves 0 to 4 bit-identical for")
    print("  `pre_ln` and `decoder`, and changes them for the rest")

    print("transformer. cross-attention reads a memory the others ignore:")
    memory = rng.normal(0.0, 1.0, features[:2].shape)
    for variant in ("decoder", "cross_attention"):
        block = models[variant].blocks[0]
        before, _ = block.forward(sequence, memory=sequence)
        after, _ = block.forward(sequence, memory=memory)
        assert not np.allclose(before, after)
    for variant in ("encoder", "post_ln", "swiglu_ffn"):
        block = models[variant].blocks[0]
        before, _ = block.forward(sequence, memory=sequence)
        after, _ = block.forward(sequence, memory=memory)
        assert np.allclose(before, after)
        assert not hasattr(block, "cross_attention")
    print("  replacing the memory changes `decoder` and `cross_attention`")
    print("  and leaves the self-attention-only blocks exactly unchanged")

    print("transformer. post-LN normalises the block's output:")
    block = models["post_ln"].blocks[0]
    out, _ = block.forward(sequence, memory=sequence)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=2e-3)
    pre, _ = models["encoder"].blocks[0].forward(sequence, memory=sequence)
    assert not np.allclose(pre.mean(axis=-1), 0.0, atol=1e-3)
    print(f"  mean {float(out.mean()):+.1e}, spread "
          f"{float(out.std()):.6f} per token; pre-LN leaves the residual")
    print("  stream unnormalised, as it must")

    print("transformer. RMSNorm keeps the scale, not the mean:")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    rms, _ = _rms_norm(x)
    normed, _ = _layer_norm(x)
    assert abs(float(np.sqrt((rms ** 2).mean())) - 1.0) < 1e-5
    assert abs(float(normed.mean())) < 1e-12
    assert abs(float(rms.mean())) > 1e-3
    print(f"  [1, 2, 3, 4] becomes {np.round(rms[0], 4).tolist()} with rms")
    print(f"  {float(np.sqrt((rms ** 2).mean())):.6f}, and")
    print(f"  {np.round(normed[0], 4).tolist()} with mean "
          f"{float(normed.mean()):+.1e}")

    print("transformer. parallel attention is one sum, not two stages:")
    block = models["parallel_attention"].blocks[0]
    normed, _ = block._norm(sequence)
    attention, _ = block.self_attention.forward(normed)
    branch, _ = block.feed_forward.forward(normed)
    fused, _ = block.forward(sequence, memory=sequence)
    assert np.allclose(fused, sequence + attention + branch)
    print("  the block output equals input plus attention plus feed-forward")
    print("  of the same normalised input, which is why it can run the two")
    print("  matrix products concurrently")

    print("transformer. SwiGLU buys a gate for one more matrix:")
    wide = models["swiglu_ffn"].blocks[0].feed_forward
    narrow = models["encoder"].blocks[0].feed_forward
    assert wide.kind == "swiglu"
    assert wide.parameters["block0.ffn.gate.W"].shape == (
        narrow.parameters["block0.ffn.first.W"].shape)
    probe = rng.normal(0.0, 1.0, (2, 3, 8))
    gate_pre = probe @ wide.parameters["block0.ffn.gate.W"] \
        + wide.parameters["block0.ffn.gate.b"]
    first = probe @ wide.parameters["block0.ffn.first.W"] \
        + wide.parameters["block0.ffn.first.b"]
    expected = (_silu(first) * gate_pre) @ wide.parameters["block0.ffn.second.W"] \
        + wide.parameters["block0.ffn.second.b"]
    actual, _ = wide.forward(probe)
    assert np.allclose(actual, expected)
    extra = wide.parameters["block0.ffn.gate.W"].size
    print(f"  the gate adds {extra} weights and 16 biases per block, and the")
    print("  hidden state is silu(projection) times the gate projection")

    print("transformer. every variant's backward matches differences:")
    check = np.random.default_rng(21)
    probe_x = check.normal(0.0, 0.5, (6, 4, 6))
    probe_y = np.array([0, 1, 1, 0, 1, 0])
    probe_w = check.normal(size=(6, 2))
    for variant in VARIANTS:
        model = SeqModel(variant, dim=6, heads=2, hidden=8, depth=2,
                         epochs=1, batch_size=6, seed=5)
        model.fit(probe_x, probe_y)
        _, cache = model._forward(probe_x)
        model._backward(probe_w, cache)
        worst = 0.0
        for key, value in model.parameters.items():
            numeric = np.zeros_like(value)
            steps = np.nditer(value, flags=["multi_index"])
            for _ in steps:
                index = steps.multi_index
                old = value[index]
                value[index] = old + 1e-6
                up = float(np.sum(model._forward(probe_x)[0] * probe_w))
                value[index] = old - 1e-6
                down = float(np.sum(model._forward(probe_x)[0] * probe_w))
                value[index] = old
                numeric[index] = (up - down) / 2e-6
            worst = max(worst, float(np.abs(model.gradients[key]
                                            - numeric).max()))
        assert worst < 1e-6
        print(f"    {variant:<18s} worst {worst:.1e} across"
              f" {model.n_params_} parameters")

    print("transformer. the parameter counts follow the layout:")
    counts = {name: models[name].n_params_ for name in VARIANTS}
    assert counts["post_ln"] == counts["pre_ln"] == counts["encoder"]
    assert counts["encoder"] == counts["rmsnorm_block"]
    assert counts["swiglu_ffn"] > counts["encoder"]
    assert counts["decoder"] > counts["cross_attention"] + 500
    assert counts["cross_attention"] == counts["encoder"]
    print(f"  the two-block stack is {counts['encoder']} parameters;")
    print(f"  swiglu is {counts['swiglu_ffn']} and the decoder, with a second")
    print(f"  attention, is {counts['decoder']}")

    bad = [
        lambda: SeqModel("encoder_lite"),
        lambda: SeqModel("encoder", depth=0),
        lambda: SeqModel("encoder", hidden=0),
        lambda: SeqModel("encoder", dim=8, heads=3),
        lambda: SeqModel("encoder", epochs=0),
        lambda: SeqModel("encoder", batch_size=0),
        lambda: SeqModel("encoder", learning_rate=0.0),
        lambda: SeqModel("encoder", learning_rate=np.nan),
        lambda: SeqModel("encoder", epochs=1).fit(train_x[0], train_y),
        lambda: SeqModel("encoder", epochs=1).fit(train_x, train_y[:-1]),
        lambda: SeqModel("encoder", epochs=1).fit(train_x[:3], train_y[:3]),
        lambda: SeqModel("encoder", epochs=1).fit(
            np.full_like(train_x, np.nan), train_y),
        lambda: SeqModel("encoder", epochs=1).fit(train_x, train_y + 0.5),
        lambda: SeqModel("encoder", epochs=1).fit(
            train_x, np.zeros_like(train_y)),
        lambda: SeqModel("encoder", epochs=1, classes=3).fit(train_x,
                                                             train_y),
        lambda: models["encoder"].predict_proba(train_x[:, :, :4]),
        lambda: models["encoder"].predict_proba(train_x[:, 0]),
        lambda: models["encoder"].predict_proba(
            np.full_like(train_x, np.inf)),
        lambda: SeqModel("encoder", epochs=1).predict(train_x),
        lambda: models["encoder"].score(train_x, train_y + 0.5),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a non-positive depth or hidden width, a head count that")
    print("  does not divide the width, a non-positive epoch count or batch")
    print("  size, a zero or non-finite learning rate, a two-dimensional")
    print("  input, a label vector of the wrong length, fewer than four")
    print("  sequences, non-finite tokens, fractional or single-class labels,")
    print("  a class count that disagrees with the labels, a prediction with")
    print("  the wrong feature width or rank, a non-finite token, prediction")
    print("  before fitting, and a score with fractional labels")
    print("transformer demo passed")


if __name__ == "__main__":
    demo()
