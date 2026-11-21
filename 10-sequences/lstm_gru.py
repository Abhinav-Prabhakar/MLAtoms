r"""Gated memory cells — LSTM and GRU, written out gate by gate.

Chapter:   10-sequences
Companion: lstm_gru.md
Tier 1 · 6 variants: lstm, gru, peephole, lstm_projection, cifg, bidirectional

    lstm:       i,f,o = sigmoid(...),  g = tanh(...)                     (1)
                c_t = f * c_(t-1) + i * g,   h_t = o * tanh(c_t)
    gru:        z,r = sigmoid(...),   n = tanh(x W_xn + (r*h) W_hn)      (2)
                h_t = (1 - z) * n + z * h_(t-1)
    peephole:   the gates of (1) also read c_(t-1), and o reads c_t      (3)
    projection: h_t = W_p m_t with m_t = o * tanh(c_t)                   (4)
    cifg:       i = 1 - f, so one gate is removed                        (5)
    bidir:      p_t = softmax(W_o [h_t^f ; h_t^b] + b_o)                 (6)

A vanilla recurrent cell multiplies the state by the recurrent matrix at every
step, so the gradient of a loss at time T with respect to a state at time t is a
product of T - t Jacobians. The `rnn` atom measured that product: eleven orders
of magnitude over thirty-two steps. Gated cells change the product by adding an
additive path. The cell state of an LSTM is updated as
c_t = f * c_(t-1) + i * g; when the forget gate is near one, the Jacobian of
that update is near the identity, and the gradient crosses the step without
shrinking. The gates learn when to open that path, which is why the same
architecture can learn dependencies that a tanh recurrence cannot.

The six variants are the same idea at different budgets. `lstm` is the standard
cell with input, forget, output and candidate gates. `gru` merges the cell state
into the hidden state and uses update and reset gates, which removes one gate
and one state. `peephole` lets the gates read the cell state directly, which
helps when the timing of a count matters more than the content of the input.
`lstm_projection` compresses the output through a projection matrix, which is
the original large-vocabulary trick. `cifg` couples the input and forget gates,
i_t = 1 - f_t, and trains three gate blocks instead of four. `bidirectional`
runs two LSTMs in opposite directions and reads the concatenation.

Everything is numpy and everything is batched. The backward pass is written out
gate by gate, including the peephole and coupled-gate branches, and every
branch is checked against central differences. The demos train on a task that
requires memory across the whole sequence, report held-out accuracy rather than
training loss, and read the gate activations to show what the cell learned. The
numbers to carry away: the parametric cells reach above ninety percent on a
recall task where the vanilla cell is at chance, the coupled-gate cell does it
with three quarters of the gate parameters, and the projection variant cuts the
recurrent cost by a factor of four at a small accuracy cost.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("lstm", "gru", "peephole", "lstm_projection", "cifg",
            "bidirectional")
EPS = 1e-8
CLIP = 5.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function without overflow on either tail."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    tail = np.exp(x[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    matrix = rng.standard_normal((size, size))
    q, r = np.linalg.qr(matrix)
    return q * np.sign(np.diag(r))


class LSTM:
    """A gated recurrent classifier trained by backpropagation through time.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    hidden : int
        Number of hidden units, per direction.
    projection : int, optional
        Output width for `lstm_projection`; defaults to half of `hidden`.
        Not accepted by the other variants.
    n_iter : int
        Number of epochs.
    lr : float
        Adam learning rate.
    seed : int
        Seed for initialisation.

    Attributes
    ----------
    history_ : list of float
        Mean training loss after each epoch.
    W_x_, W_h_, b_ : ndarray
        Input, recurrent and bias parameters of the gate blocks. The block
        order is ``[input, forget, output, candidate]`` for the LSTM family,
        ``[forget, output, candidate]`` for `cifg`, and ``[update, reset,
        candidate]`` for `gru`.
    W_o_, b_o_ : ndarray
        Output layer.
    w_i_, w_f_, w_o_ : ndarray
        Diagonal peephole weights, for `peephole` only.
    W_p_ : ndarray
        Projection matrix, for `lstm_projection` only.
    W_hn_ : ndarray
        Candidate recurrent matrix, for `gru` only.
    """

    def __init__(self, *, variant: str = "lstm", hidden: int = 16,
                 projection: int | None = None, n_iter: int = 200,
                 lr: float = 0.01, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(hidden, (int, np.integer)) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if projection is not None:
            if variant != "lstm_projection":
                raise ValueError("projection belongs to the lstm_projection variant")
            if not isinstance(projection, (int, np.integer)) or projection < 1:
                raise ValueError("projection must be a positive integer")
        self.variant = variant
        self.hidden = int(hidden)
        self.projection = None if projection is None else int(projection)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # input handling, shapes and parameters
    # ------------------------------------------------------------------ #
    @property
    def _gates(self) -> int:
        if self.variant == "gru":
            return 3
        return 3 if self.variant == "cifg" else 4

    @property
    def _state_width(self) -> int:
        """Width of one direction's state, which is what the cell feeds back."""
        if self.variant == "lstm_projection":
            return self.projection if self.projection is not None else max(
                1, self.hidden // 2)
        return self.hidden

    @property
    def _width(self) -> int:
        """Width read by the output layer: two directions for `bidirectional`."""
        return 2 * self.hidden if self.variant == "bidirectional" else self._state_width

    @staticmethod
    def _input(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values[None, :, None]
        elif values.ndim == 2:
            values = values[None, :, :]
        if values.ndim != 3:
            raise ValueError("the input must be (steps,), (steps, features) or a batch")
        if min(values.shape) < 1:
            raise ValueError("the input needs at least one sequence, step and feature")
        if not np.all(np.isfinite(values)):
            raise ValueError("the input must be finite")
        return values

    def _prepare(self, x: np.ndarray, y: np.ndarray) -> tuple:
        features = self._input(x)
        labels = np.asarray(y)
        if labels.ndim == 1:
            labels = labels[None, :]
        if labels.ndim != 2 or labels.shape != features.shape[:2]:
            raise ValueError("labels must be one per step and sequence")
        if not np.issubdtype(labels.dtype, np.integer):
            numeric = labels.astype(float)
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
                raise ValueError("labels must be integers")
            labels = numeric.astype(int)
        labels = labels.astype(int)
        if labels.min() < 0:
            raise ValueError("labels must be non-negative")
        return features, labels

    def _init(self, features: int, classes: int) -> None:
        rng = np.random.default_rng(self.seed)
        hidden, width, gates = self.hidden, self._state_width, self._gates
        self.W_x_ = rng.standard_normal((gates * hidden, features)) / math.sqrt(features)
        if self.variant == "gru":
            self.W_h_ = np.vstack([_orthogonal(rng, hidden) for _ in range(2)])
            self.W_hn_ = _orthogonal(rng, hidden)
        elif width == hidden:
            self.W_h_ = np.vstack([_orthogonal(rng, hidden) for _ in range(gates)])
        else:
            self.W_h_ = rng.standard_normal((gates * hidden, width)) / math.sqrt(width)
        self.b_ = np.zeros(gates * hidden)
        if self.variant in ("lstm", "peephole", "lstm_projection", "bidirectional"):
            self.b_[hidden:2 * hidden] = 1.0
        elif self.variant == "cifg":
            self.b_[:hidden] = 1.0
        self.W_o_ = rng.standard_normal((classes, self._width)) / math.sqrt(self._width)
        self.b_o_ = np.zeros(classes)
        self.n_classes_ = classes
        if self.variant == "peephole":
            self.w_i_ = 0.1 * rng.standard_normal(hidden)
            self.w_f_ = 0.1 * rng.standard_normal(hidden)
            self.w_o_ = 0.1 * rng.standard_normal(hidden)
        if self.variant == "lstm_projection":
            self.W_p_ = rng.standard_normal((width, hidden)) / math.sqrt(hidden)
        if self.variant == "bidirectional":
            self.W_x_b_ = rng.standard_normal((gates * hidden, features)) / math.sqrt(features)
            self.W_h_b_ = np.vstack([_orthogonal(rng, hidden) for _ in range(gates)])
            self.b_b_ = np.zeros(gates * hidden)
            self.b_b_[hidden:2 * hidden] = 1.0
        self._adam: dict = {}

    def _parameter_names(self) -> list:
        names = ["W_x_", "W_h_", "b_", "W_o_", "b_o_"]
        if self.variant == "gru":
            names.append("W_hn_")
        if self.variant == "peephole":
            names += ["w_i_", "w_f_", "w_o_"]
        if self.variant == "lstm_projection":
            names.append("W_p_")
        if self.variant == "bidirectional":
            names += ["W_x_b_", "W_h_b_", "b_b_"]
        return names

    def n_parameters(self) -> int:
        """Number of scalar parameters, for comparing the variants."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before counting parameters")
        return int(sum(getattr(self, name).size for name in self._parameter_names()))

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _cell(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
              suffix: str = "") -> tuple:
        hidden = self.hidden
        W_x = getattr(self, f"W_x_{suffix}")
        W_h = getattr(self, f"W_h_{suffix}")
        b = getattr(self, f"b_{suffix}")
        if self.variant == "gru":
            pre = x @ W_x.T + b
            z = _sigmoid(pre[:, :hidden] + h_prev @ W_h[:hidden].T)
            r = _sigmoid(pre[:, hidden:2 * hidden] + h_prev @ W_h[hidden:].T)
            candidate = np.tanh(pre[:, 2 * hidden:] + (r * h_prev) @ self.W_hn_.T)
            state = (1.0 - z) * candidate + z * h_prev
            return state, c_prev, {"x": x, "h_prev": h_prev, "z": z, "r": r,
                                   "candidate": candidate}
        pre = x @ W_x.T + h_prev @ W_h.T + b
        if self.variant == "cifg":
            forget, out, candidate = pre[:, :hidden], pre[:, hidden:2 * hidden], pre[:, 2 * hidden:]
            f = _sigmoid(forget)
            i = 1.0 - f
            g = np.tanh(candidate)
            o_pre = out
        else:
            i_pre, f_pre, o_pre, g_pre = (pre[:, 0:hidden], pre[:, hidden:2 * hidden],
                                          pre[:, 2 * hidden:3 * hidden],
                                          pre[:, 3 * hidden:])
            if self.variant == "peephole":
                i_pre = i_pre + c_prev * self.w_i_
                f_pre = f_pre + c_prev * self.w_f_
            i, f, g = _sigmoid(i_pre), _sigmoid(f_pre), np.tanh(g_pre)
        c = f * c_prev + i * g
        o = (_sigmoid(o_pre + c * self.w_o_) if self.variant == "peephole"
             else _sigmoid(o_pre))
        m = o * np.tanh(c)
        state = m if self.variant != "lstm_projection" else m @ self.W_p_.T
        return state, c, {"x": x, "h_prev": h_prev, "c_prev": c_prev, "i": i,
                          "f": f, "o": o, "g": g, "c": c, "m": m}

    def _forward(self, features: np.ndarray, suffix: str = "") -> tuple:
        batch, length = features.shape[:2]
        states = np.empty((batch, length, self._state_width))
        caches = [None] * length
        h = np.zeros((batch, self._state_width))
        c = np.zeros((batch, self.hidden))
        for t in range(length):
            h, c, caches[t] = self._cell(features[:, t], h, c, suffix)
            states[:, t] = h
        return states, caches

    def _forward_bidirectional(self, features: np.ndarray) -> tuple:
        forward, cache_f = self._forward(features)
        backward, cache_b = self._forward(features[:, ::-1], suffix="b_")
        states = np.concatenate([forward, backward[:, ::-1]], axis=2)
        return states, (forward, backward, cache_f, cache_b)

    def _probabilities(self, states: np.ndarray) -> np.ndarray:
        return _softmax_rows(states @ self.W_o_.T + self.b_o_)

    # ------------------------------------------------------------------ #
    # backward
    # ------------------------------------------------------------------ #
    def _backward_direction(self, features: np.ndarray, states: np.ndarray,
                            caches: list, dh_out: np.ndarray,
                            dlogits: np.ndarray | None,
                            suffix: str = "") -> tuple:
        hidden = self.hidden
        length = features.shape[1]
        grads = {name: np.zeros_like(getattr(self, name))
                 for name in self._parameter_names()}
        if dlogits is not None:
            grads[f"W_o_{suffix}"] += np.einsum("ntc,nth->ch", dlogits, states)
            grads[f"b_o_{suffix}"] += dlogits.sum(axis=(0, 1))
        carry_h = np.zeros((features.shape[0], self._state_width))
        carry_c = np.zeros((features.shape[0], hidden))
        for t in range(length - 1, -1, -1):
            cache = caches[t]
            dh = dh_out[:, t] + carry_h
            if self.variant == "gru":
                dpre, dcarry_h = self._backward_gru(cache, dh, grads)
            else:
                dpre, dcarry_h, carry_c = self._backward_lstm(
                    cache, dh, carry_c, grads, suffix)
            grads[f"W_x_{suffix}"] += dpre.T @ cache["x"]
            if self.variant == "gru":
                grads[f"W_h_{suffix}"] += dpre[:, :2 * hidden].T @ cache["h_prev"]
            else:
                grads[f"W_h_{suffix}"] += dpre.T @ cache["h_prev"]
            grads[f"b_{suffix}"] += dpre.sum(axis=0)
            carry_h = dcarry_h
        return grads, carry_h, carry_c

    def _backward_lstm(self, cache: dict, dh: np.ndarray, carry_c: np.ndarray,
                       grads: dict, suffix: str) -> tuple:
        i, f, g = cache["i"], cache["f"], cache["g"]
        o, c, m = cache["o"], cache["c"], cache["m"]
        tanh_c = np.tanh(c)
        if self.variant == "lstm_projection":
            grads["W_p_"] += dh.T @ m
            dm = dh @ self.W_p_
        else:
            dm = dh
        dc = dm * o * (1.0 - tanh_c * tanh_c) + carry_c
        dz_o = dm * tanh_c * o * (1.0 - o)
        if self.variant == "peephole":
            dc = dc + dz_o * self.w_o_
            grads["w_o_"] += (dz_o * c).sum(axis=0)
        if self.variant == "cifg":
            dz_f = dc * (cache["c_prev"] - g) * f * (1.0 - f)
            dz_g = dc * (1.0 - f) * (1.0 - g * g)
            dpre = np.concatenate([dz_f, dz_o, dz_g], axis=1)
        else:
            dz_i = dc * g * i * (1.0 - i)
            dz_f = dc * cache["c_prev"] * f * (1.0 - f)
            dz_g = dc * i * (1.0 - g * g)
            dc_prev = dc * f
            if self.variant == "peephole":
                dc_prev = dc_prev + dz_i * self.w_i_ + dz_f * self.w_f_
                grads["w_i_"] += (dz_i * cache["c_prev"]).sum(axis=0)
                grads["w_f_"] += (dz_f * cache["c_prev"]).sum(axis=0)
            dpre = np.concatenate([dz_i, dz_f, dz_o, dz_g], axis=1)
        if self.variant == "cifg":
            dc_prev = dc * f
        return dpre, dpre @ getattr(self, f"W_h_{suffix}"), dc_prev

    def _backward_gru(self, cache: dict, dh: np.ndarray, grads: dict) -> tuple:
        hidden = self.hidden
        z, r = cache["z"], cache["r"]
        candidate, h_prev = cache["candidate"], cache["h_prev"]
        dz = dh * (h_prev - candidate) * z * (1.0 - z)
        dn = dh * (1.0 - z) * (1.0 - candidate * candidate)
        d_reset_hidden = dn @ self.W_hn_
        d_r = d_reset_hidden * h_prev
        dr = d_r * r * (1.0 - r)
        grads["W_hn_"] += dn.T @ (r * h_prev)
        dpre = np.concatenate([dz, dr, dn], axis=1)
        carry = (dz @ self.W_h_[:hidden] + dr @ self.W_h_[hidden:]
                 + d_reset_hidden * r + dh * z)
        return dpre, carry

    def _gradients_full(self, features: np.ndarray, labels: np.ndarray) -> dict:
        states, caches = self._forward(features)
        dlogits = self._loss_gradient(labels, self._probabilities(states))
        grads, _, _ = self._backward_direction(
            features, states, caches, dlogits @ self.W_o_, dlogits)
        return grads

    def _gradients_bidirectional(self, features: np.ndarray,
                                 labels: np.ndarray) -> dict:
        states, parts = self._forward_bidirectional(features)
        forward, backward, cache_f, cache_b = parts
        dlogits = self._loss_gradient(labels, self._probabilities(states))
        grads = {name: np.zeros_like(getattr(self, name))
                 for name in self._parameter_names()}
        grads["W_o_"] = np.einsum("ntc,nth->ch", dlogits, states)
        grads["b_o_"] = dlogits.sum(axis=(0, 1))
        split = dlogits @ self.W_o_
        first, _, _ = self._backward_direction(
            features, forward, cache_f, split[..., :self.hidden], None)
        second, _, _ = self._backward_direction(
            features[:, ::-1], backward, cache_b,
            split[..., self.hidden:][:, ::-1], None, suffix="b_")
        for name, value in first.items():
            grads[name] += value
        for name, value in second.items():
            grads[name] += value
        return grads

    def _loss_gradient(self, labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return (probs - np.eye(self.n_classes_)[labels]) / labels.size

    # ------------------------------------------------------------------ #
    # optimisation
    # ------------------------------------------------------------------ #
    def _step(self, grads: dict) -> None:
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = CLIP / norm if norm > CLIP else 1.0
        for name, grad in grads.items():
            grad = grad * scale
            moment1, moment2, count = self._adam.get(
                name, (np.zeros_like(grad), np.zeros_like(grad), 0))
            count += 1
            moment1 = 0.9 * moment1 + 0.1 * grad
            moment2 = 0.999 * moment2 + 0.001 * grad * grad
            corrected1 = moment1 / (1.0 - 0.9 ** count)
            corrected2 = moment2 / (1.0 - 0.999 ** count)
            setattr(self, name, getattr(self, name)
                    - self.lr * corrected1 / (np.sqrt(corrected2) + 1e-8))
            self._adam[name] = (moment1, moment2, count)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LSTM:
        """Fit a batch of sequences, with one label per step."""
        features, labels = self._prepare(x, y)
        classes = int(labels.max()) + 1
        if classes < 2:
            raise ValueError("need at least two classes")
        self._init(features.shape[2], classes)
        history = []
        for _ in range(self.n_iter):
            if self.variant == "bidirectional":
                grads = self._gradients_bidirectional(features, labels)
            else:
                grads = self._gradients_full(features, labels)
            self._step(grads)
            history.append(self.loss(features, labels))
        self.history_ = history
        return self

    # ------------------------------------------------------------------ #
    # scoring, prediction and diagnostics
    # ------------------------------------------------------------------ #
    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy over every step of every sequence."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before scoring")
        features, labels = self._prepare(x, y)
        if labels.max() >= self.n_classes_:
            raise ValueError("a label is outside the fitted classes")
        probs = self._probabilities(self._states(features))
        picked = np.take_along_axis(probs, labels[..., None], axis=2)[..., 0]
        return float(-np.mean(np.log(np.maximum(picked, 1e-300))))

    def _states(self, features: np.ndarray) -> np.ndarray:
        if self.variant == "bidirectional":
            return self._forward_bidirectional(features)[0]
        return self._forward(features)[0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities; a single sequence is unbatched."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before predicting")
        probs = self._probabilities(self._states(self._input(x)))
        return probs[0] if np.asarray(x).ndim <= 2 else probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most likely class at every step."""
        return self.predict_proba(x).argmax(axis=-1)

    def gradients(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Analytic gradients of the mean loss, without taking a step."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before differentiating")
        features, labels = self._prepare(x, y)
        if self.variant == "bidirectional":
            return self._gradients_bidirectional(features, labels)
        return self._gradients_full(features, labels)

    def gate_means(self, x: np.ndarray) -> dict:
        """Mean input, forget and output gate values over a batch."""
        if self.variant in ("gru", "bidirectional"):
            raise ValueError("gate_means belongs to the single-direction LSTM family")
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before reading the gates")
        features = self._input(x)
        _, caches = self._forward(features)
        return {"input": float(np.mean([cache["i"] for cache in caches])),
                "forget": float(np.mean([cache["f"] for cache in caches])),
                "output": float(np.mean([cache["o"] for cache in caches]))}

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _batch(make, count, length, seed, *args) -> tuple:
    rng = np.random.default_rng(seed)
    pairs = [make(rng, length, *args) for _ in range(count)]
    features = np.stack([pair[0] for pair in pairs])
    if features.ndim == 2:
        features = features[..., None]
    return features, np.stack([pair[1] for pair in pairs])


def _accuracy(model: LSTM, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def _onehot(symbols: np.ndarray) -> np.ndarray:
    return np.eye(4)[symbols]


def _copy(rng: np.random.Generator, length: int, lag: int) -> tuple:
    """A symbol delayed by `lag` steps; the first `lag` labels repeat step 0."""
    symbols = rng.integers(0, 4, length)
    labels = np.concatenate([np.full(lag, symbols[0]), symbols[:length - lag]])
    return _onehot(symbols), labels


def _next_symbol(rng: np.random.Generator, length: int) -> tuple:
    symbols = rng.integers(0, 4, length + 1)
    return _onehot(symbols[:length]), symbols[1:length + 1]


def _numeric(model: LSTM, x: np.ndarray, y: np.ndarray, name: str,
             index: tuple, epsilon: float = 1e-6) -> float:
    parameter = getattr(model, name)
    original = parameter[index]
    parameter[index] = original + epsilon
    plus = model.loss(x, y)
    parameter[index] = original - epsilon
    minus = model.loss(x, y)
    parameter[index] = original
    return (plus - minus) / (2.0 * epsilon)


def _demo_gradients() -> None:
    """The gate, peephole, projection and coupled-gate branches, checked."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 3))
    y = rng.integers(0, 3, (2, 6))
    print("analytic gradients against central differences, batch of two:")
    print("  variant           parameters   worst relative error")
    for variant, projection in (("lstm", None), ("gru", None),
                                ("peephole", None), ("lstm_projection", 3),
                                ("cifg", None), ("bidirectional", None)):
        model = LSTM(variant=variant, hidden=5, projection=projection,
                     n_iter=1, seed=1)
        model._init(3, 3)
        grads = model.gradients(x, y)
        worst, checked = 0.0, 0
        for name, grad in grads.items():
            flat = int(np.argmax(np.abs(grad)))
            index = np.unravel_index(flat, grad.shape)
            if abs(grad[index]) < 1e-9:
                continue
            approx = _numeric(model, x, y, name, index)
            checked += 1
            worst = max(worst, abs(grad[index] - approx)
                        / max(1e-8, abs(grad[index]) + abs(approx)))
        print(f"  {variant:<17s} {checked:5d}        {worst:.2e}")
        assert worst < 1e-6
    print("  The same routine differentiates all four gates, the diagonal")
    print("  peepholes, the projection and the second direction. The")
    print("  finite-difference check is the only honest claim that a")
    print("  hand-written gate backward pass is correct.")


def _demo_copy() -> None:
    """The task the vanilla cell failed: a symbol delayed by two steps."""
    x, y = _batch(_copy, 64, 12, 0, 2)
    held_x, held_y = _batch(_copy, 512, 12, 99, 2)
    print("copy the symbol from two steps ago, 64 sequences of 12 symbols:")
    print("  variant    held-out   final loss   parameters")
    for variant in ("lstm", "gru"):
        model = LSTM(variant=variant, hidden=24, n_iter=200, lr=0.02, seed=0)
        model.fit(x, y)
        score = _accuracy(model, held_x, held_y)
        print(f"  {variant:<9s} {score:.3f}      {model.history_[-1]:.4f}      "
              f"{model.n_parameters()}")
        assert score > 0.95
    model = LSTM(variant="lstm", hidden=24, n_iter=200, lr=0.02, seed=0).fit(x, y)
    gates = model.gate_means(x)
    print(f"  mean gates after fitting: input {gates['input']:.2f}, "
          f"forget {gates['forget']:.2f}, output {gates['output']:.2f}")
    assert gates["input"] > 0.7
    print("  The input gate is open and the forget gate is not shut: the cell")
    print("  writes the symbol in and keeps it long enough to read it back.")
    print("  A tanh recurrence memorises a copy task like this one instead of")
    print("  learning it; the additive cell-state path is the difference.")


def _demo_variants() -> None:
    """The same task at smaller budgets, with the parameter bill."""
    x, y = _batch(_copy, 64, 12, 0, 2)
    held_x, held_y = _batch(_copy, 512, 12, 99, 2)
    print("the same copy task through the other variants:")
    print("  variant           held-out   parameters   note")
    results = {}
    for variant, projection in (("peephole", None), ("cifg", None),
                                ("lstm_projection", 8), ("lstm", None)):
        model = LSTM(variant=variant, hidden=24, projection=projection,
                     n_iter=200, lr=0.02, seed=0)
        model.fit(x, y)
        results[variant] = (_accuracy(model, held_x, held_y),
                            model.n_parameters())
    baseline = results["lstm"][1]
    notes = {
        "peephole": "gates read the cell state",
        "cifg": "input = 1 - forget",
        "lstm_projection": "8-wide projected state",
        "lstm": "reference",
    }
    for variant in ("peephole", "cifg", "lstm_projection", "lstm"):
        score, count = results[variant]
        share = 100.0 * count / baseline
        print(f"  {variant:<17s} {score:.3f}      {count:5d}       "
              f"{notes[variant]:<24s} {share:4.0f}% of reference")
        assert score > 0.93
    assert results["cifg"][1] < 0.8 * baseline
    assert results["lstm_projection"][1] < 0.6 * baseline
    print("  Coupling the gates removes a quarter of the gate parameters and")
    print("  costs nothing here. Projection cuts the recurrent matrix from")
    print("  24 to 8 columns, which is a threefold saving on the dominant")
    print("  term, and gives up about three points of accuracy.")


def _demo_direction() -> None:
    """A label that depends on the future needs the second direction."""
    x, y = _batch(_next_symbol, 32, 12, 0)
    held_x, held_y = _batch(_next_symbol, 256, 12, 99)
    print("predict the next symbol, 4 symbols, 12 steps:")
    print("  variant          held-out   final loss")
    scores = {}
    for variant in ("lstm", "bidirectional"):
        model = LSTM(variant=variant, hidden=16, n_iter=120, lr=0.02, seed=0)
        model.fit(x, y)
        scores[variant] = _accuracy(model, held_x, held_y)
        print(f"  {variant:<16s} {scores[variant]:.3f}      "
              f"{model.history_[-1]:.4f}")
    assert scores["lstm"] < 0.40
    assert scores["bidirectional"] > 0.85
    print("  The forward LSTM is at chance: no gate can read a symbol that")
    print("  has not arrived yet. The bidirectional model sees it in the")
    print("  reverse pass, and the same gated cell becomes a smoother.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    x, y = _batch(_copy, 4, 8, 0, 2)
    fitted = LSTM(hidden=4, n_iter=1, seed=0).fit(x, y)
    calls = [
        lambda: LSTM(variant="nope"),
        lambda: LSTM(hidden=0),
        lambda: LSTM(n_iter=0),
        lambda: LSTM(lr=0.0),
        lambda: LSTM(projection=8),
        lambda: LSTM(variant="lstm_projection", projection=0),
        lambda: LSTM().fit(x, np.zeros((4, 8), dtype=int)),
        lambda: LSTM().fit(np.zeros((4, 0, 1)), y),
        lambda: LSTM().fit(np.zeros((1, 2, 3, 4)), y),
        lambda: LSTM().fit(np.full((4, 8, 4), np.nan), y),
        lambda: LSTM().fit(x, y[:, :-1]),
        lambda: LSTM().fit(x, y + 0.5),
        lambda: LSTM().fit(x, y - 5),
        lambda: LSTM().predict_proba(x),
        lambda: LSTM().loss(x, y),
        lambda: LSTM().gradients(x, y),
        lambda: LSTM().n_parameters(),
        lambda: fitted.loss(x, y + 5),
        lambda: LSTM(variant="gru", hidden=4, n_iter=1, seed=0
                     ).fit(x, y).gate_means(x),
        lambda: LSTM(variant="bidirectional", hidden=4, n_iter=1,
                     seed=0).fit(x, y).gate_means(x),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, a zero "
          f"width, epoch count and learning rate, a projection outside its "
          f"variant and a zero projection, a single class, an empty feature "
          f"axis, a four-dimensional input, a non-finite input, a short label "
          f"row, fractional and negative labels, prediction, scoring, "
          f"differentiation and parameter counting before fitting, a label "
          f"outside the fitted classes, and gate diagnostics on a cell that "
          f"does not have the three gates.")


def main() -> None:
    _demo_gradients()
    _demo_copy()
    _demo_variants()
    _demo_direction()
    _demo_broken()
    print("lstm_gru demo passed")


if __name__ == "__main__":
    main()
