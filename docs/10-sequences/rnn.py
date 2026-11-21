r"""Recurrent networks — backpropagation through time, written out by hand.

Chapter:   10-sequences
Companion: rnn.md
Tier 1 · 7 variants: vanilla, elman, jordan, bptt, truncated_bptt,
layer_norm_rnn, bidirectional

    cell:       h_t = tanh(W_x x_t + W_h h_(t-1) + b_h)                  (1)
    jordan:     h_t = tanh(W_x x_t + W_h h_(t-1) + W_y y_(t-1) + b_h)    (2)
    output:     p_t = softmax(W_o h_t + b_o),  L = -mean_t log p_t[y_t]  (3)
    bptt:       dh_t = W_o^T (p_t - e_t) + W_h^T dpre_(t+1)              (4)
    layer norm: h_t = tanh(g * (pre - mean pre) / std pre + beta)        (5)
    truncated:  the same recursion inside windows of length k            (6)
    bidir:      p_t = softmax(W_o [h_t^f ; h_t^b] + b_o)                 (7)

A recurrent network is a feedforward network with a memory: the same cell
applied at every step, with its own previous output fed back in. That one
change shares the parameters across time, turns the computation graph into a
cycle, and makes training an exercise in applying the chain rule along a chain
of copies. Backpropagation through time is ordinary backpropagation on the
unrolled graph, and writing it out by hand is the clearest way to see where
gradients vanish, where they explode, and what each variant changes.

The seven variants are a cell and a schedule. The cell is the Elman recurrence
of (1), the same recurrence with a learned initial context (`elman`), Jordan's
output recurrence of (2), or a layer-normalised cell (5). The schedule is full
backpropagation (`vanilla`, `bptt`), windows that cut the gradient at their
boundaries (6), or two passes in opposite directions (7). `bptt` is the
diagnostic setting: the same cell and the same descent, plus the per-step
Jacobian norms that make the vanishing visible.

Everything is hand-written numpy, and everything is batched: the input is a
stack of sequences, the cell is one matrix product per step over the whole
batch, and every demo trains on a sample and reports held-out accuracy. The
demos check what is easy to get wrong. Analytic gradients are compared with
central differences for the plain, Jordan, layer-normalised and bidirectional
cells, so every branch of the backward pass is covered. On cumulative parity
the network generalises to held-out sequences. On a target that
depends only on the next symbol, a one-directional network is at chance while
the bidirectional one is near perfect, which is the whole argument for (7). And
the recurrent Jacobian of a fitted network falls by more than eleven orders of
magnitude over thirty-two steps, which is the phenomenon the `lstm_gru` atom
exists to fix.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("vanilla", "elman", "jordan", "bptt", "truncated_bptt",
            "layer_norm_rnn", "bidirectional")
EPS = 1e-8
CLIP = 5.0


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    """A random orthogonal matrix, so the recurrence starts norm-preserving."""
    matrix = rng.standard_normal((size, size))
    q, r = np.linalg.qr(matrix)
    return q * np.sign(np.diag(r))


class RNN:
    """A recurrent classifier trained by backpropagation through time.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`: the cell and the training schedule.
    hidden : int
        Number of hidden units, per direction.
    n_iter : int
        Number of epochs; each epoch is one forward and one backward pass.
    lr : float
        Adam learning rate.
    chunk : int
        Window length for `truncated_bptt`.
    seed : int
        Seed for the initial weights and the initial context.

    Attributes
    ----------
    history_ : list of float
        Mean training loss after each epoch.
    W_x_, W_h_, b_h_ : ndarray
        Input, recurrent and bias parameters of the cell.
    W_o_, b_o_ : ndarray
        Output layer, shaped ``(n_classes, hidden)`` for one direction and
        ``(n_classes, 2 * hidden)`` for `bidirectional`.
    h0_ : ndarray
        Learned initial context, for `elman` only.
    W_y_ : ndarray
        Output-to-hidden weights of (2), for `jordan` only.
    ln_gain_, ln_bias_ : ndarray
        The affine pair of (5), for `layer_norm_rnn` only.
    W_x_b_, W_h_b_, b_h_b_ : ndarray
        Parameters of the backward pass, for `bidirectional` only.
    """

    def __init__(self, *, variant: str = "vanilla", hidden: int = 16,
                 n_iter: int = 200, lr: float = 0.01, chunk: int = 10,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(hidden, (int, np.integer)) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not isinstance(chunk, (int, np.integer)) or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        if variant != "truncated_bptt" and chunk != 10:
            raise ValueError("chunk belongs to the truncated_bptt variant")
        self.variant = variant
        self.hidden = int(hidden)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.chunk = int(chunk)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # input handling and parameters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _input(x: np.ndarray) -> np.ndarray:
        """Return a batch of sequences, shaped ``(batch, steps, features)``."""
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
        hidden = self.hidden
        self.W_x_ = rng.standard_normal((hidden, features)) / math.sqrt(features)
        self.W_h_ = _orthogonal(rng, hidden)
        self.b_h_ = np.zeros(hidden)
        width = 2 * hidden if self.variant == "bidirectional" else hidden
        self.W_o_ = rng.standard_normal((classes, width)) / math.sqrt(width)
        self.b_o_ = np.zeros(classes)
        self.n_classes_ = classes
        if self.variant == "elman":
            self.h0_ = 0.1 * rng.standard_normal(hidden)
        if self.variant == "jordan":
            self.W_y_ = 0.1 * rng.standard_normal((hidden, classes))
        if self.variant == "layer_norm_rnn":
            self.ln_gain_ = np.ones(hidden)
            self.ln_bias_ = np.zeros(hidden)
        if self.variant == "bidirectional":
            self.W_x_b_ = rng.standard_normal((hidden, features)) / math.sqrt(features)
            self.W_h_b_ = _orthogonal(rng, hidden)
            self.b_h_b_ = np.zeros(hidden)
        self._adam: dict = {}

    def _parameter_names(self) -> list:
        names = ["W_x_", "W_h_", "b_h_", "W_o_", "b_o_"]
        if self.variant == "elman":
            names.append("h0_")
        if self.variant == "jordan":
            names.append("W_y_")
        if self.variant == "layer_norm_rnn":
            names += ["ln_gain_", "ln_bias_"]
        if self.variant == "bidirectional":
            names += ["W_x_b_", "W_h_b_", "b_h_b_"]
        return names

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _cell(self, x: np.ndarray, h_prev: np.ndarray, y_prev: np.ndarray,
              suffix: str = "") -> tuple:
        """One copy of the cell for a whole batch, and the cache to invert it."""
        W_x = getattr(self, f"W_x_{suffix}")
        W_h = getattr(self, f"W_h_{suffix}")
        b_h = getattr(self, f"b_h_{suffix}")
        pre = x @ W_x.T + h_prev @ W_h.T + b_h
        if self.variant == "jordan":
            pre = pre + y_prev @ self.W_y_.T
        if self.variant == "layer_norm_rnn":
            deviation = pre.std(axis=1, keepdims=True)
            centred = (pre - pre.mean(axis=1, keepdims=True)) / (deviation + EPS)
            normalised = self.ln_gain_ * centred + self.ln_bias_
        else:
            deviation, centred, normalised = 0.0, None, pre
        return np.tanh(normalised), (normalised, centred, deviation, h_prev)

    def _forward(self, features: np.ndarray, suffix: str = "",
                 start: np.ndarray | None = None) -> tuple:
        """Run one direction over a batch, returning states, outputs and caches."""
        batch, length = features.shape[:2]
        hidden = np.empty((batch, length, self.hidden))
        caches = [None] * length
        if start is not None:
            state = start
        elif self.variant == "elman" and suffix == "":
            state = np.repeat(self.h0_[None, :], batch, axis=0)
        else:
            state = np.zeros((batch, self.hidden))
        previous_y = np.zeros((batch, self.n_classes_))
        for t in range(length):
            state, caches[t] = self._cell(features[:, t], state, previous_y, suffix)
            hidden[:, t] = state
            if self.variant == "jordan":
                previous_y = _softmax_rows(state @ self.W_o_.T + self.b_o_)
        if self.variant == "bidirectional":
            return hidden, None, caches
        return hidden, _softmax_rows(hidden @ self.W_o_.T + self.b_o_), caches

    def _forward_bidirectional(self, features: np.ndarray) -> tuple:
        forward, _, cache_f = self._forward(features)
        backward, _, cache_b = self._forward(features[:, ::-1], suffix="b_")
        hidden = np.concatenate([forward, backward[:, ::-1]], axis=2)
        return hidden, _softmax_rows(hidden @ self.W_o_.T + self.b_o_), (
            forward, backward, cache_f, cache_b)

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        if self.variant == "bidirectional":
            return self._forward_bidirectional(features)[1]
        return self._forward(features)[1]

    # ------------------------------------------------------------------ #
    # backward
    # ------------------------------------------------------------------ #
    def _backward_direction(self, features: np.ndarray, probs: np.ndarray,
                            hidden: np.ndarray, caches: list,
                            dh_out: np.ndarray, dlogits: np.ndarray | None,
                            suffix: str = "") -> tuple:
        """Backpropagate one direction; returns grads, start grad and step norms.

        ``dh_out`` is the gradient reaching each hidden state from above. For a
        single direction that is ``dlogits @ W_o``; for one half of a
        bidirectional pair it is the matching column block. When ``dlogits`` is
        given the output layer belongs to this direction and its gradients are
        accumulated here too.
        """
        length = features.shape[1]
        grads = {name: np.zeros_like(getattr(self, name))
                 for name in self._parameter_names()}
        if dlogits is not None:
            grads[f"W_o_{suffix}"] += np.einsum("ntc,nth->ch", dlogits, hidden)
            grads[f"b_o_{suffix}"] += dlogits.sum(axis=(0, 1))
        carry = np.zeros((features.shape[0], self.hidden))
        for t in range(length - 1, -1, -1):
            normalised, centred, deviation, h_prev = caches[t]
            dh = dh_out[:, t] + carry
            if centred is not None:
                dz = dh * (1.0 - np.tanh(normalised) ** 2)
                grads["ln_gain_"] += (dz * centred).sum(axis=0)
                grads["ln_bias_"] += dz.sum(axis=0)
                dcentred = dz * self.ln_gain_
                dpre = (dcentred - dcentred.mean(axis=1, keepdims=True)
                        - centred * (dcentred * centred).mean(axis=1, keepdims=True)
                        ) / (deviation + EPS)
            else:
                dpre = dh * (1.0 - np.tanh(normalised) ** 2)
            grads[f"W_x_{suffix}"] += dpre.T @ features[:, t]
            grads[f"W_h_{suffix}"] += dpre.T @ h_prev
            grads[f"b_h_{suffix}"] += dpre.sum(axis=0)
            if self.variant == "jordan" and t > 0:
                grads["W_y_"] += dpre.T @ probs[:, t - 1]
                feedback = self._jordan_feedback(dpre, probs[:, t - 1])
                grads["W_o_"] += feedback.T @ hidden[:, t - 1]
                grads["b_o_"] += feedback.sum(axis=0)
                dlogits[:, t - 1] += feedback
                dh_out[:, t - 1] += feedback @ self.W_o_
            carry = dpre @ getattr(self, f"W_h_{suffix}")
        return grads, carry

    def _jordan_feedback(self, dpre: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Push a gradient on ``y_(t-1)`` back through its softmax."""
        dy = dpre @ self.W_y_
        jacobian = np.einsum("nc,nd->ncd", probs, probs)
        jacobian = -jacobian + np.einsum("nc,cd->ncd", probs,
                                         np.eye(self.n_classes_))
        return np.einsum("ncd,nd->nc", jacobian, dy)

    def _loss_gradient(self, labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return (probs - np.eye(self.n_classes_)[labels]) / labels.size

    def _gradients_full(self, features: np.ndarray, labels: np.ndarray) -> dict:
        hidden, probs, caches = self._forward(features)
        dlogits = self._loss_gradient(labels, probs)
        grads, start = self._backward_direction(
            features, probs, hidden, caches, dlogits @ self.W_o_, dlogits)
        if self.variant == "elman":
            grads["h0_"] = start.sum(axis=0)
        else:
            del start
        return grads

    def _gradients_bidirectional(self, features: np.ndarray,
                                 labels: np.ndarray) -> dict:
        hidden, probs, parts = self._forward_bidirectional(features)
        forward, backward, cache_f, cache_b = parts
        dlogits = self._loss_gradient(labels, probs)
        grads = {name: np.zeros_like(getattr(self, name))
                 for name in self._parameter_names()}
        grads["W_o_"] = np.einsum("ntc,nth->ch", dlogits, hidden)
        grads["b_o_"] = dlogits.sum(axis=(0, 1))
        split = dlogits @ self.W_o_
        first, _ = self._backward_direction(
            features, probs, forward, cache_f, split[..., :self.hidden], None)
        second, _ = self._backward_direction(
            features[:, ::-1], probs[:, ::-1], backward, cache_b,
            split[..., self.hidden:][:, ::-1], None, suffix="b_")
        for name, value in first.items():
            grads[name] += value
        for name, value in second.items():
            grads[name] += value
        return grads

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

    def fit(self, x: np.ndarray, y: np.ndarray) -> RNN:
        """Fit a batch of sequences, with one label per step."""
        features, labels = self._prepare(x, y)
        classes = int(labels.max()) + 1
        if classes < 2:
            raise ValueError("need at least two classes")
        if self.variant == "truncated_bptt" and self.chunk > features.shape[1]:
            raise ValueError("chunk must not exceed the sequence length")
        self._init(features.shape[2], classes)
        history = []
        for _ in range(self.n_iter):
            if self.variant == "bidirectional":
                grads = self._gradients_bidirectional(features, labels)
            elif self.variant == "truncated_bptt":
                grads = self._window_gradients(features, labels)
            else:
                grads = self._gradients_full(features, labels)
            self._step(grads)
            history.append(self.loss(features, labels))
        self.history_ = history
        return self

    def _window_gradients(self, features: np.ndarray,
                          labels: np.ndarray) -> dict:
        """Average the per-window gradients, cutting the tape at each boundary."""
        batch, length = features.shape[:2]
        total = {name: np.zeros_like(getattr(self, name))
                 for name in self._parameter_names()}
        for n in range(batch):
            state = (self.h0_[None, :] if self.variant == "elman"
                     else np.zeros((1, self.hidden)))
            for begin in range(0, length, self.chunk):
                end = min(length, begin + self.chunk)
                window = features[n:n + 1, begin:end]
                hidden, probs, caches = self._forward(window, start=state)
                dlogits = self._loss_gradient(labels[n:n + 1, begin:end], probs)
                grads, _ = self._backward_direction(
                    window, probs, hidden, caches, dlogits @ self.W_o_, dlogits)
                if self.variant == "elman":
                    grads["h0_"] = np.zeros_like(self.h0_)
                for name, value in grads.items():
                    total[name] += value
                state = hidden[:, -1]
        return {name: value / batch for name, value in total.items()}

    # ------------------------------------------------------------------ #
    # scoring, prediction and introspection
    # ------------------------------------------------------------------ #
    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy over every step of every sequence."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before scoring")
        features, labels = self._prepare(x, y)
        if labels.max() >= self.n_classes_:
            raise ValueError("a label is outside the fitted classes")
        probs = self._probabilities(features)
        picked = np.take_along_axis(probs, labels[..., None], axis=2)[..., 0]
        return float(-np.mean(np.log(np.maximum(picked, 1e-300))))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities at every step; a single sequence is unbatched."""
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before predicting")
        features = self._input(x)
        probs = self._probabilities(features)
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
        if self.variant == "truncated_bptt":
            raise ValueError("truncated_bptt has no single full-sequence gradient")
        return self._gradients_full(features, labels)

    def jacobian_norms(self, x: np.ndarray) -> np.ndarray:
        """Norm of ``d h_(T-1) / d h_t`` for each step, for `bptt`.

        This is the product of the recurrent Jacobians along the chain. Its
        decay with the distance from the end is the vanishing gradient,
        measured without reference to any particular loss.
        """
        if self.variant != "bptt":
            raise ValueError("jacobian norms belong to the bptt variant")
        if not hasattr(self, "W_o_"):
            raise RuntimeError("fit before inspecting the recurrence")
        features = self._input(x)[:1]
        hidden, _, _ = self._forward(features)
        length = features.shape[1]
        steps = [np.diag(1.0 - hidden[0, t] ** 2) @ self.W_h_
                 for t in range(length)]
        product = np.eye(self.hidden)
        norms = np.empty(length)
        norms[-1] = 1.0
        for t in range(length - 1, 0, -1):
            product = product @ steps[t]
            norms[t - 1] = float(np.linalg.norm(product))
        return norms

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _batch(make, count, length, seed, *args) -> tuple:
    """Stack independent draws of a task into a batch."""
    rng = np.random.default_rng(seed)
    pairs = [make(rng, length, *args) for _ in range(count)]
    features = np.stack([pair[0] for pair in pairs])
    if features.ndim == 2:
        features = features[..., None]
    return features, np.stack([pair[1] for pair in pairs])


def _accuracy(model: RNN, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == y))


def _parity(rng: np.random.Generator, length: int) -> tuple:
    """Cumulative parity: the label at every step depends on the whole prefix."""
    bits = rng.integers(0, 2, length)
    return bits.astype(float), np.cumsum(bits) % 2


def _next_symbol(rng: np.random.Generator, length: int) -> tuple:
    """A one-hot symbol and the symbol that follows it."""
    symbols = rng.integers(0, 4, length + 1)
    return np.eye(4)[symbols[:length]], symbols[1:length + 1]


def _numeric(model: RNN, x: np.ndarray, y: np.ndarray, name: str,
             index: tuple, epsilon: float = 1e-6) -> float:
    """Central difference of the loss in one parameter entry."""
    parameter = getattr(model, name)
    original = parameter[index]
    parameter[index] = original + epsilon
    plus = model.loss(x, y)
    parameter[index] = original - epsilon
    minus = model.loss(x, y)
    parameter[index] = original
    return (plus - minus) / (2.0 * epsilon)


def _demo_gradients() -> None:
    """Every branch of the backward pass against finite differences."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 7, 3))
    y = rng.integers(0, 3, (2, 7))
    print("analytic gradients against central differences, batch of two:")
    print("  variant           parameters   worst relative error")
    for variant in ("vanilla", "elman", "jordan", "layer_norm_rnn",
                    "bidirectional"):
        model = RNN(variant=variant, hidden=5, n_iter=1, seed=1)
        features, labels = model._prepare(x, y)
        model._init(features.shape[2], 3)
        if variant == "bidirectional":
            grads = model._gradients_bidirectional(features, labels)
        else:
            grads = model._gradients_full(features, labels)
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
    print("  Every parameter with a non-zero derivative was perturbed, so the")
    print("  output layer, the recurrence, the learned context, Jordan's")
    print("  feedback and the layer-norm affine all have a checked inverse.")


def _demo_bptt() -> RNN:
    """The memory task, its learning curve, and the effect of clipping."""
    x, y = _batch(_parity, 32, 12, 0)
    held_x, held_y = _batch(_parity, 256, 12, 99)
    model = RNN(variant="bptt", hidden=24, n_iter=300, lr=0.02, seed=0)
    model.fit(x, y)
    print("cumulative parity, 32 sequences of 12 bits, 24 hidden units:")
    print(f"  loss at the first epoch  {model.history_[0]:.4f}")
    print(f"  loss at the last epoch   {model.history_[-1]:.4f}")
    print(f"  training accuracy        {_accuracy(model, x, y):.3f}")
    print(f"  held-out accuracy        {_accuracy(model, held_x, held_y):.3f}")
    assert model.history_[-1] < 0.05 < model.history_[0]
    assert _accuracy(model, held_x, held_y) > 0.95
    print("  Parity cannot be read off one symbol: the state has to carry the")
    print("  running sum, and backpropagation through time is what teaches it")
    print("  to do so. The held-out number is the one that matters.")
    return model


def _demo_cells() -> None:
    """The other cells and the schedule variants on the same task."""
    x, y = _batch(_parity, 32, 12, 0)
    held_x, held_y = _batch(_parity, 256, 12, 99)
    print("the same parity task through every cell:")
    print("  variant          held-out   final loss   learned parameter")
    for variant in ("jordan", "elman", "layer_norm_rnn", "vanilla"):
        model = RNN(variant=variant, hidden=24, n_iter=300, lr=0.02, seed=0)
        model.fit(x, y)
        score = _accuracy(model, held_x, held_y)
        if variant == "elman":
            note = f"|h0| = {np.linalg.norm(model.h0_):.2f}"
        elif variant == "layer_norm_rnn":
            note = (f"gain {model.ln_gain_.min():.2f} to "
                    f"{model.ln_gain_.max():.2f}")
        elif variant == "jordan":
            note = f"|W_y| = {np.linalg.norm(model.W_y_):.2f}"
        else:
            note = "shared recurrence"
        print(f"  {variant:<16s} {score:.3f}      {model.history_[-1]:.4f}    {note}")
        assert score > 0.90
        if variant == "elman":
            assert np.linalg.norm(model.h0_) > 0.5
        if variant == "layer_norm_rnn":
            assert model.ln_gain_.max() > 2.0 * model.ln_gain_.min()
    print("  The learned initial state and the layer-norm gains both move well")
    print("  away from their neutral values: the extra parameters are used.")


def _demo_direction() -> None:
    """A label that depends on the future is unreachable in one direction."""
    x, y = _batch(_next_symbol, 32, 12, 0)
    held_x, held_y = _batch(_next_symbol, 256, 12, 99)
    print("predict the next symbol from the current one, 4 symbols:")
    print("  variant          held-out   final loss")
    scores = {}
    for variant in ("vanilla", "bidirectional"):
        model = RNN(variant=variant, hidden=16, n_iter=150, lr=0.02, seed=0)
        model.fit(x, y)
        scores[variant] = _accuracy(model, held_x, held_y)
        print(f"  {variant:<16s} {scores[variant]:.3f}      "
              f"{model.history_[-1]:.4f}")
    assert scores["vanilla"] < 0.40
    assert scores["bidirectional"] > 0.85
    print("  The next symbol is independent of everything the forward pass has")
    print("  seen, so it is at chance — 0.25 — and stays there. The backward")
    print("  pass sees the answer before it is needed, and the concatenation")
    print("  of the two directions makes it a smoothing problem.")


def _demo_truncation() -> None:
    """Truncation is a different gradient, not a different forward pass."""
    x, y = _batch(_parity, 8, 12, 0)
    full = RNN(variant="vanilla", hidden=8, n_iter=1, seed=0)
    full._init(1, 2)
    features, labels = full._prepare(x, y)
    reference = full._gradients_full(features, labels)
    scale = max(float(np.abs(grad).max()) for grad in reference.values())
    print("windowed gradient against full backpropagation, 8 sequences of 12:")
    print("  chunk   max absolute difference   relative to gradient scale")
    for chunk in (12, 6, 4, 3):
        windowed = RNN(variant="truncated_bptt", hidden=8, n_iter=1,
                       chunk=chunk, seed=0)
        windowed._init(1, 2)
        for name in ("W_x_", "W_h_", "b_h_", "W_o_", "b_o_"):
            setattr(windowed, name, getattr(full, name).copy())
        grads = windowed._window_gradients(features, labels)
        difference = max(float(np.abs(reference[name] - grads[name]).max())
                         for name in reference)
        print(f"  {chunk:5d}   {difference:.3e}                 "
              f"{difference / scale:.3e}")
        if chunk == 12:
            assert difference < 1e-12
        else:
            assert difference > 1e-3
    assert scale > 0.0
    print("  One window recovers the exact gradient; shorter windows drop only")
    print("  the terms that crossed a boundary, so the difference is a clean")
    print("  bias rather than an error. It is not automatically a worse model:")
    short_x, short_y = _batch(_parity, 16, 12, 0)
    short_held = _batch(_parity, 256, 12, 99)
    print("  training the same 16 sequences for 200 epochs:")
    for variant, chunk in (("vanilla", 10), ("truncated_bptt", 4)):
        model = RNN(variant=variant, hidden=16, n_iter=200, lr=0.02,
                    chunk=chunk, seed=0)
        model.fit(short_x, short_y)
        print(f"    {variant:<15s} chunk {chunk:<3d} held-out "
              f"{_accuracy(model, short_held[0], short_held[1]):.3f}")
    print("  On parity the forward state carries the answer, so the truncated")
    print("  model is not worse here; the difference is what the gradient can")
    print("  attribute, not what the forward pass can represent. On a task")
    print("  where credit assignment is the whole difficulty, (6) is the")
    print("  limit, and the bias is no longer free.")


def _demo_vanishing() -> None:
    """The recurrent Jacobian, measured along the chain."""
    x, y = _batch(_parity, 16, 32, 0)
    held_x, held_y = _batch(_parity, 256, 32, 99)
    model = RNN(variant="bptt", hidden=24, n_iter=400, lr=0.02, seed=0)
    model.fit(x, y)
    norms = model.jacobian_norms(x)
    decade = float(np.log10(norms[-1] / norms[0]))
    print("norm of the Jacobian from step t to the last step, 32 steps:")
    print(f"  held-out accuracy              {_accuracy(model, held_x, held_y):.3f}")
    print(f"  norm at the last step          {norms[-1]:.3e}")
    print(f"  norm at the first step         {norms[0]:.3e}")
    print(f"  orders of magnitude lost       {decade:.2f}")
    print(f"  steps per order of magnitude   {31.0 / abs(decade):.2f}")
    assert norms[-1] == 1.0
    assert norms[0] < 1e-6
    print("  The profile at every eighth step:")
    print("   ", np.array2string(norms[::8], precision=3))
    print("  Every factor in the product is a tanh derivative times W_h, and")
    print("  tanh' < 1 wherever the state is not at rest. Thirty-two steps")
    print("  cost eleven orders of magnitude: a dependency a hundred steps back")
    print("  is invisible, which is why the gated cells exist.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    x, y = _batch(_parity, 4, 8, 0)
    fitted = RNN(hidden=4, n_iter=1, seed=0).fit(x, y)
    calls = [
        lambda: RNN(variant="nope"),
        lambda: RNN(hidden=0),
        lambda: RNN(n_iter=0),
        lambda: RNN(lr=0.0),
        lambda: RNN(chunk=0),
        lambda: RNN(chunk=4),
        lambda: RNN(variant="truncated_bptt", chunk=50).fit(x, y),
        lambda: RNN().fit(x, np.zeros((4, 8), dtype=int)),
        lambda: RNN().fit(np.zeros((4, 0)), y),
        lambda: RNN().fit(np.zeros((1, 2, 3, 4)), y),
        lambda: RNN().fit(np.full((4, 8, 1), np.nan), y),
        lambda: RNN().fit(x, y[:, :-1]),
        lambda: RNN().fit(x, y + 0.5),
        lambda: RNN().fit(x, y - 5),
        lambda: RNN().predict_proba(x),
        lambda: RNN().loss(x, y),
        lambda: RNN().gradients(x, y),
        lambda: fitted.loss(x, y + 5),
        lambda: fitted.jacobian_norms(x),
        lambda: RNN(variant="truncated_bptt", chunk=4).fit(x, y).gradients(x, y),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, a zero "
          f"width, a zero epoch count, a zero learning rate, a zero and a "
          f"misplaced window, a window longer than the sequence, a single "
          f"class, an empty feature axis, a four-dimensional input, a "
          f"non-finite input, a short label row, fractional and negative "
          f"labels, prediction and scoring before fitting, a label outside "
          f"the fitted classes, Jacobian norms outside the bptt variant, and "
          f"a full-sequence gradient outside the truncated schedule.")


def main() -> None:
    _demo_gradients()
    _demo_bptt()
    _demo_cells()
    _demo_direction()
    _demo_truncation()
    _demo_vanishing()
    _demo_broken()
    print("rnn demo passed")


if __name__ == "__main__":
    main()
