r"""Conditional random fields — exact inference and training on a chain.

Chapter:   10-sequences
Companion: crf.md
Tier 2 · 6 variants: linear_chain, forward_backward, viterbi, crf_sgd,
                    crf_lbfgs, higher_order

    score(x, y) = sum_t U(t, y_t) + sum_t A(y_(t-1), y_t) + s(y_0) + e(y_T)  (1)
    p(y | x)    = exp(score(x, y)) / Z(x)                                    (2)
    Z(x)        = sum over all y of exp(score(x, y))                         (3)

A conditional random field is a log-linear model over whole label sequences. The
score of a labelling is a sum of local terms, and the probability is that score
exponentiated and normalised by a sum over every possible labelling. The sum is
exponential in the length, so nobody enumerates it: on a chain, the sum factors
and a forward pass computes it exactly, and the same pass gives every marginal.

The six variants are the model and the three algorithms that make it usable. The
model is the linear chain of (1), where the unary term reads the features and the
transition term scores a label following another label. Forward-backward computes
the partition function and the per-position marginals; Viterbi computes the single
best labelling, which is not the same thing as the per-position argmax. Training
is by maximising the likelihood, and the gradient is an expected count minus an
observed count. `crf_sgd` takes that gradient in minibatches, `crf_lbfgs` takes it
with a quasi-Newton line search, and `higher_order` replaces the transition
A(y_(t-1), y_t) with A(y_(t-2), y_(t-1), y_t) by running the same chain over pairs
of labels.

Everything is numpy and everything is exact on a chain. Inference is the
log-space recursion, not an approximation, and the demos check it against
enumeration of every labelling for a small alphabet. The numbers to carry away:
the log partition function, all marginals and the Viterbi score agree with the
enumeration of 243 paths to 4e-16, the likelihood gradient matches central
differences to 1.1e-8 for both orders, posterior decoding loses 0.32 of joint
score to Viterbi on a random chain, and on a process that only a pair of labels
can predict the second-order field reaches 1.000 held-out accuracy with a
log-likelihood of -0.001 where the first-order one reaches 0.769 and -7.105.
"""  # noqa: E501

from __future__ import annotations

import itertools

import numpy as np

VARIANTS = ("linear_chain", "forward_backward", "viterbi", "crf_sgd", "crf_lbfgs",
            "higher_order")
TRAINABLE = ("linear_chain", "crf_sgd", "crf_lbfgs", "higher_order")


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Log-sum-exp along one axis, safe when an entry is negative infinity."""
    peak = values.max(axis=axis, keepdims=True)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = safe + np.log(np.exp(values - safe).sum(axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis)


def _onehot(labels: np.ndarray, states: int) -> np.ndarray:
    return np.eye(states)[labels]


class CRF:
    """A linear-chain conditional random field, trained by exact likelihood.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    n_iter : int
        Number of passes over the data, or of quasi-Newton iterations.
    lr : float
        Step size for the gradient methods; ignored by `crf_lbfgs`.
    batch : int
        Minibatch size for `crf_sgd`.
    memory : int
        Number of curvature pairs kept by `crf_lbfgs`.
    seed : int
        Seed for the initial parameters.

    Attributes
    ----------
    W_ : ndarray
        Feature weights, shape ``(features, states)``.
    b_ : ndarray
        Unary bias, shape ``(states,)``.
    A_ : ndarray
        Transition scores ``A_[previous, current]``, order one only.
    A2_ : ndarray
        Transition scores ``A2_[previous, previous, current]`` with a start
        row, for `higher_order`.
    s_, e_ : ndarray
        Start and end scores, order one.
    s2_, e2_ : ndarray
        Start scores for the first label, end scores for the last, for
        `higher_order`.
    history_ : list of float
        Negative log-likelihood after each pass.
    """

    def __init__(self, *, variant: str = "linear_chain", n_iter: int = 100,
                 lr: float = 0.1, batch: int = 8, memory: int = 10,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not isinstance(batch, (int, np.integer)) or batch < 1:
            raise ValueError("batch must be a positive integer")
        if not isinstance(memory, (int, np.integer)) or memory < 1:
            raise ValueError("memory must be a positive integer")
        self.variant = variant
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.batch = int(batch)
        self.memory = int(memory)
        self.seed = int(seed)
        self.order = 2 if variant == "higher_order" else 1
        self.n_evaluations_ = 0
        self._adam: dict = {}

    # ------------------------------------------------------------------ #
    # input handling and parameters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim == 2:
            features = features[None]
        if features.ndim != 3:
            raise ValueError("x must be (steps, features) or a batch of those")
        if min(features.shape) < 1:
            raise ValueError("x needs at least one sequence, step and feature")
        if not np.all(np.isfinite(features)):
            raise ValueError("x must be finite")
        return features

    @classmethod
    def _input(cls, x: np.ndarray, y: np.ndarray) -> tuple:
        features = cls._features(x)
        labels = np.asarray(y)
        if labels.ndim == 1:
            labels = labels[None]
        if labels.ndim != 2 or labels.shape != features.shape[:2]:
            raise ValueError("y must be one label per step and sequence")
        if not np.issubdtype(labels.dtype, np.integer):
            numeric = labels.astype(float)
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
                raise ValueError("labels must be integers")
            labels = numeric.astype(int)
        labels = labels.astype(int)
        if labels.min() < 0:
            raise ValueError("labels must be non-negative")
        return features, labels

    def _init(self, features: int, states: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.n_features_ = features
        self.n_states_ = states
        self.W_ = 0.01 * rng.standard_normal((features, states))
        self.b_ = np.zeros(states)
        if self.order == 1:
            self.A_ = np.zeros((states, states))
            self.s_ = np.zeros(states)
            self.e_ = np.zeros(states)
        else:
            self.A2_ = np.zeros((states + 1, states, states))
            self.s2_ = np.zeros(states)
            self.e2_ = np.zeros(states)

    def _parameters(self) -> list:
        if self.order == 1:
            return [self.W_, self.b_, self.A_, self.s_, self.e_]
        return [self.W_, self.b_, self.A2_, self.s2_, self.e2_]

    def n_parameters(self) -> int:
        """Number of free scores, for comparing the two orders."""
        if not hasattr(self, "W_"):
            raise RuntimeError("fit or initialise before counting parameters")
        return int(sum(parameter.size for parameter in self._parameters()))

    # ------------------------------------------------------------------ #
    # the model
    # ------------------------------------------------------------------ #
    def _unary(self, features: np.ndarray) -> np.ndarray:
        return features @ self.W_ + self.b_

    def _structure(self, unary: np.ndarray) -> tuple:
        """Turn a model order into one chain over its elementary states."""
        if self.order == 1:
            return unary, self.A_, self.s_, self.e_
        states = self.n_states_
        size = (states + 1) * states
        pair_unary = np.concatenate([unary] * (states + 1), axis=2)
        start = np.full(size, -np.inf)
        start[states * states:] = self.s2_
        transition = np.full((size, size), -np.inf)
        for previous in range(states + 1):
            for middle in range(states):
                row = previous * states + middle
                for current in range(states):
                    transition[row, middle * states + current] = self.A2_[
                        previous, middle, current]
        end = np.tile(self.e2_, states + 1)
        return pair_unary, transition, start, end

    def _forward(self, unary: np.ndarray, transition: np.ndarray,
                 start: np.ndarray, end: np.ndarray) -> tuple:
        """The alpha recursion of (4), in log space."""
        batch, length, _ = unary.shape
        alpha = np.empty_like(unary)
        alpha[:, 0] = start + unary[:, 0]
        for t in range(1, length):
            alpha[:, t] = _logsumexp(
                alpha[:, t - 1, :, None] + transition[None], axis=1) + unary[:, t]
        return alpha, _logsumexp(alpha[:, -1] + end[None], axis=1)

    def _backward(self, unary: np.ndarray, transition: np.ndarray,
                  end: np.ndarray) -> np.ndarray:
        """The beta recursion, starting from the end score."""
        batch, length, _ = unary.shape
        beta = np.empty_like(unary)
        beta[:, -1] = end
        for t in range(length - 2, -1, -1):
            beta[:, t] = _logsumexp(
                transition[None] + (unary[:, t + 1] + beta[:, t + 1])[:, None, :],
                axis=2)
        return beta

    def _marginal_terms(self, unary: np.ndarray, transition: np.ndarray,
                        start: np.ndarray, end: np.ndarray) -> tuple:
        """Node and edge marginals, plus the log partition function."""
        alpha, log_z = self._forward(unary, transition, start, end)
        beta = self._backward(unary, transition, end)
        node = np.exp(alpha + beta - log_z[:, None, None])
        if unary.shape[1] == 1:
            edge = np.zeros((unary.shape[0], 0, unary.shape[2], unary.shape[2]))
        else:
            edge = np.exp(alpha[:, :-1, :, None] + transition[None]
                          + (unary[:, 1:] + beta[:, 1:])[:, :, None, :]
                          - log_z[:, None, None, None])
        return log_z, node, edge

    def _viterbi(self, unary: np.ndarray, transition: np.ndarray,
                 start: np.ndarray, end: np.ndarray) -> tuple:
        """The best path and its score, by the max-plus recursion."""
        batch, length, states = unary.shape
        delta = np.empty_like(unary)
        back = np.zeros((batch, length, states), dtype=int)
        delta[:, 0] = start + unary[:, 0]
        for t in range(1, length):
            scores = delta[:, t - 1, :, None] + transition[None]
            back[:, t] = scores.argmax(axis=1)
            delta[:, t] = scores.max(axis=1) + unary[:, t]
        score = (delta[:, -1] + end[None]).max(axis=1)
        path = np.empty((batch, length), dtype=int)
        path[:, -1] = (delta[:, -1] + end[None]).argmax(axis=1)
        for t in range(length - 2, -1, -1):
            path[:, t] = np.take_along_axis(
                back[:, t + 1], path[:, t + 1, None], axis=1)[:, 0]
        return path, score

    def _pair_labels(self, path: np.ndarray) -> np.ndarray:
        """Recover labels from a path over pairs (previous, current)."""
        return path % self.n_states_

    # ------------------------------------------------------------------ #
    # public inference
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not hasattr(self, "W_"):
            raise RuntimeError("fit or initialise before inferring")

    def _prepared(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        features = self._features(x)
        if features.shape[2] != self.n_features_:
            raise ValueError("x has the wrong number of features")
        return features

    def log_partition(self, x: np.ndarray) -> np.ndarray:
        """The log of the normaliser, one number per sequence."""
        features = self._prepared(x)
        unary, transition, start, end = self._structure(self._unary(features))
        return self._forward(unary, transition, start, end)[1]

    def marginals(self, x: np.ndarray) -> np.ndarray:
        """Per-position label marginals, shape ``(batch, steps, states)``."""
        features = self._prepared(x)
        unary, transition, start, end = self._structure(self._unary(features))
        _, node, _ = self._marginal_terms(unary, transition, start, end)
        if self.order == 1:
            return node
        return node.reshape(node.shape[0], node.shape[1],
                            self.n_states_ + 1, self.n_states_).sum(axis=2)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """The single best labelling, by Viterbi."""
        features = self._prepared(x)
        unary, transition, start, end = self._structure(self._unary(features))
        path, _ = self._viterbi(unary, transition, start, end)
        return self._pair_labels(path) if self.order == 2 else path

    def posterior_decode(self, x: np.ndarray) -> np.ndarray:
        """The per-position argmax of the marginals, which is not Viterbi."""
        if self.variant != "viterbi":
            raise ValueError("posterior_decode belongs to the viterbi variant")
        return self.marginals(x).argmax(axis=-1)

    def forward_backward(self, x: np.ndarray) -> tuple:
        """The alpha and beta tables and the log partition function."""
        if self.variant != "forward_backward":
            raise ValueError("forward_backward belongs to its own variant")
        features = self._prepared(x)
        unary, transition, start, end = self._structure(self._unary(features))
        alpha, log_z = self._forward(unary, transition, start, end)
        beta = self._backward(unary, transition, end)
        return alpha, beta, log_z

    # ------------------------------------------------------------------ #
    # scoring and gradients
    # ------------------------------------------------------------------ #
    def _path_score(self, unary: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """The score of the observed labellings, without the normaliser."""
        batch, length, _ = unary.shape
        rows = np.arange(batch)
        unary_score = unary[rows[:, None], np.arange(length)[None, :], labels].sum(axis=1)
        if self.order == 1:
            transition = self.A_[labels[:, :-1], labels[:, 1:]].sum(axis=1)
            start = self.s_[labels[:, 0]]
            end = self.e_[labels[:, -1]]
        else:
            states = self.n_states_
            if length > 1:
                previous = np.concatenate(
                    [np.full((batch, 1), states), labels[:, :-2]], axis=1)
                transition = self.A2_[previous, labels[:, :-1],
                                      labels[:, 1:]].sum(axis=1)
            else:
                transition = np.zeros(batch)
            start = self.s2_[labels[:, 0]]
            end = self.e2_[labels[:, -1]]
        return unary_score + transition + start + end

    def log_likelihood(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean log probability of the labels under the model."""
        self._check_fitted()
        features, labels = self._input(x, y)
        if labels.max() >= self.n_states_:
            raise ValueError("a label is outside the fitted states")
        unary, transition, start, end = self._structure(self._unary(features))
        log_z = self._forward(unary, transition, start, end)[1]
        score = self._path_score(self._unary(features), labels)
        return float(np.mean(score - log_z))

    def _gradients(self, features: np.ndarray, labels: np.ndarray) -> tuple:
        """Negative log-likelihood and its exact gradient."""
        self.n_evaluations_ += 1
        unary, transition, start, end = self._structure(self._unary(features))
        log_z, node, edge = self._marginal_terms(unary, transition, start, end)
        score = self._path_score(unary, labels)
        loss = float(np.mean(log_z - score))
        observed = _onehot(labels, self.n_states_)
        expected = node if self.order == 1 else node.reshape(
            node.shape[0], node.shape[1], self.n_states_ + 1,
            self.n_states_).sum(axis=2)
        difference = (expected - observed) / labels.shape[0]
        grads = {"W_": np.einsum("ntf,ntk->fk", features, difference),
                 "b_": difference.sum(axis=(0, 1))}
        if self.order == 1:
            observed_edge = np.zeros((self.n_states_, self.n_states_))
            np.add.at(observed_edge, (labels[:, :-1].ravel(), labels[:, 1:].ravel()), 1.0)
            grads["A_"] = (edge.sum(axis=(0, 1)) - observed_edge) / labels.shape[0]
            observed_start = np.bincount(labels[:, 0], minlength=self.n_states_)
            grads["s_"] = (node[:, 0].sum(axis=0) - observed_start) / labels.shape[0]
            observed_end = np.bincount(labels[:, -1], minlength=self.n_states_)
            grads["e_"] = (node[:, -1].sum(axis=0) - observed_end) / labels.shape[0]
        else:
            states = self.n_states_
            expected_transition = np.zeros_like(self.A2_)
            for previous in range(states + 1):
                for middle in range(states):
                    row = previous * states + middle
                    for current in range(states):
                        expected_transition[previous, middle, current] = edge[
                            :, :, row, middle * states + current].sum()
            observed_transition = np.zeros_like(self.A2_)
            if labels.shape[1] > 1:
                previous = np.concatenate(
                    [np.full((labels.shape[0], 1), states), labels[:, :-2]], axis=1)
                np.add.at(observed_transition,
                          (previous.ravel(), labels[:, :-1].ravel(),
                           labels[:, 1:].ravel()), 1.0)
            grads["A2_"] = ((expected_transition - observed_transition)
                            / labels.shape[0])
            start_observed = np.bincount(labels[:, 0], minlength=states)
            grads["s2_"] = ((node[:, 0].reshape(labels.shape[0], states + 1, states)
                             [:, states].sum(axis=0) - start_observed)
                            / labels.shape[0])
            end_observed = np.bincount(labels[:, -1], minlength=states)
            grads["e2_"] = ((node[:, -1].reshape(labels.shape[0], states + 1, states)
                             .sum(axis=(0, 1)) - end_observed) / labels.shape[0])
        return loss, grads

    # ------------------------------------------------------------------ #
    # optimisation
    # ------------------------------------------------------------------ #
    def _pack(self) -> np.ndarray:
        return np.concatenate([parameter.ravel() for parameter in self._parameters()])

    def _unpack(self, vector: np.ndarray) -> None:
        offset = 0
        for parameter in self._parameters():
            size = parameter.size
            parameter[...] = vector[offset:offset + size].reshape(parameter.shape)
            offset += size

    def _loss_and_gradient(self, vector: np.ndarray, features: np.ndarray,
                           labels: np.ndarray) -> tuple:
        self._unpack(vector)
        loss, grads = self._gradients(features, labels)
        return loss, np.concatenate([grads[name].ravel() for name in grads])

    def _descent(self, features: np.ndarray, labels: np.ndarray) -> list:
        """Full-batch gradient descent, for `linear_chain`."""
        history = []
        for _ in range(self.n_iter):
            loss, grads = self._gradients(features, labels)
            for name, grad in grads.items():
                setattr(self, name, getattr(self, name) - self.lr * grad)
            history.append(loss)
        return history

    def _stochastic(self, features: np.ndarray, labels: np.ndarray) -> list:
        """Minibatch gradient descent with momentum, for `crf_sgd`."""
        history = []
        velocity = {name: np.zeros_like(parameter)
                    for name, parameter in zip(self._names(), self._parameters(),
                                               strict=True)}
        order = np.arange(features.shape[0])
        for _ in range(self.n_iter):
            order = np.random.default_rng(self.seed + len(history)).permutation(order)
            total = 0.0
            for start in range(0, order.size, self.batch):
                chosen = order[start:start + self.batch]
                loss, grads = self._gradients(features[chosen], labels[chosen])
                total += loss * chosen.size
                for name, grad in grads.items():
                    velocity[name] = 0.9 * velocity[name] - self.lr * grad
                    setattr(self, name, getattr(self, name) + velocity[name])
            history.append(total / order.size)
        return history

    def _names(self) -> list:
        return (["W_", "b_", "A_", "s_", "e_"] if self.order == 1
                else ["W_", "b_", "A2_", "s2_", "e2_"])

    def _quasi_newton(self, features: np.ndarray, labels: np.ndarray) -> list:
        """L-BFGS with a backtracking line search, for `crf_lbfgs`."""
        vector = self._pack()
        loss, grad = self._loss_and_gradient(vector, features, labels)
        history = [loss]
        directions, differences = [], []
        for _ in range(self.n_iter):
            if float(np.linalg.norm(grad)) < 1e-9:
                break
            q = grad.copy()
            alphas = []
            for direction, difference in zip(directions[::-1],
                                             differences[::-1], strict=True):
                rho = 1.0 / float(difference @ direction)
                alpha = rho * float(direction @ q)
                q = q - alpha * difference
                alphas.append(alpha)
            if directions:
                scale = float(directions[-1] @ differences[-1]) / float(
                    differences[-1] @ differences[-1])
            else:
                scale = 1.0 / max(1.0, float(np.linalg.norm(grad)))
            step = q * scale
            for index, (direction, difference) in enumerate(
                    zip(directions, differences, strict=True)):
                rho = 1.0 / float(difference @ direction)
                beta = rho * float(difference @ step)
                step = step + (alphas[len(alphas) - 1 - index] - beta) * direction
            step = -step
            slope = float(grad @ step)
            if slope >= 0.0:
                step = -grad
                slope = float(grad @ step)
            length = 1.0
            for _ in range(30):
                candidate = vector + length * step
                trial, trial_grad = self._loss_and_gradient(candidate, features, labels)
                if trial <= loss + 1e-4 * length * slope:
                    break
                length *= 0.5
            else:
                candidate, trial, trial_grad = vector, loss, grad
            difference = candidate - vector
            update = trial_grad - grad
            if float(difference @ update) > 1e-12:
                directions.append(difference)
                differences.append(update)
                if len(directions) > self.memory:
                    directions.pop(0)
                    differences.pop(0)
            vector, loss, grad = candidate, trial, trial_grad
            history.append(loss)
        self._unpack(vector)
        return history

    # ------------------------------------------------------------------ #
    # fit and predict
    # ------------------------------------------------------------------ #
    def fit(self, x: np.ndarray, y: np.ndarray) -> CRF:
        """Maximise the conditional likelihood of the label sequences."""
        if self.variant not in TRAINABLE:
            raise ValueError(f"{self.variant} is an inference variant; it does not fit")
        features, labels = self._input(x, y)
        states = int(labels.max()) + 1
        if states < 2:
            raise ValueError("need at least two label values")
        self._init(features.shape[2], states)
        if self.variant in ("crf_lbfgs", "higher_order"):
            history = self._quasi_newton(features, labels)
        elif self.variant == "crf_sgd":
            history = self._stochastic(features, labels)
        else:
            history = self._descent(features, labels)
        self.history_ = history
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for `decode`: the best label sequence."""
        return self.decode(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean joint score of the labels, without the normaliser."""
        self._check_fitted()
        features, labels = self._input(x, y)
        if labels.max() >= self.n_states_:
            raise ValueError("a label is outside the fitted states")
        return float(np.mean(self._path_score(self._unary(features), labels)))

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fraction of decoded labels that match the truth."""
        features, labels = self._input(x, y)
        return float(np.mean(self.decode(features) == labels))


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _brute_force(unary: np.ndarray, transition: np.ndarray, start: np.ndarray,
                 end: np.ndarray) -> tuple:
    """Enumerate every path, for checking the recursions on a small chain."""
    batch, length, states = unary.shape
    paths = np.array(list(itertools.product(range(states), repeat=length)))
    scores = start[paths[:, 0]] + unary[:, 0, paths[:, 0]]
    for t in range(1, length):
        scores = scores + transition[paths[:, t - 1], paths[:, t]] + unary[:, t, paths[:, t]]
    scores = scores + end[paths[:, -1]]
    log_z = _logsumexp(scores, axis=1)
    probabilities = np.exp(scores - log_z[:, None])
    node = np.zeros((batch, length, states))
    for t in range(length):
        for state in range(states):
            node[:, t, state] = probabilities[:, paths[:, t] == state].sum(axis=1)
    best = scores.argmax(axis=1)
    return log_z, node, paths[best], scores[np.arange(batch), best]


def _random_chain(seed: int, states: int, length: int, features: int,
                  variant: str = "linear_chain") -> tuple:
    """A fitted-looking model with random scores, for the exactness checks."""
    rng = np.random.default_rng(seed)
    model = CRF(variant=variant)
    model._init(features, states)
    model.W_ = rng.standard_normal((features, states))
    model.b_ = 0.4 * rng.standard_normal(states)
    model.A_ = rng.standard_normal((states, states))
    model.s_ = 0.4 * rng.standard_normal(states)
    model.e_ = 0.4 * rng.standard_normal(states)
    x = rng.standard_normal((3, length, features))
    return model, x


def _demo_enumeration() -> None:
    """Every number the recursions produce, against enumeration."""
    model, x = _random_chain(0, 3, 5, 2, variant="forward_backward")
    structure = model._structure(model._unary(x))
    log_z, node, best_path, best_score = _brute_force(*structure)
    alpha, beta, table_log_z = model.forward_backward(x)
    truth_log_z = model.log_partition(x)
    truth_node = model.marginals(x)
    path, path_score = model._viterbi(*structure)
    edges = model._marginal_terms(*structure)[2]
    print("three sequences, five steps, three labels, 243 paths each:")
    print(f"  log partition function within {np.abs(log_z - truth_log_z).max():.1e}")
    print(f"  every marginal within {np.abs(node - truth_node).max():.1e}")
    print(f"  Viterbi score within {np.abs(best_score - path_score).max():.1e} of the best")
    print(f"  edge marginals sum to {edges.sum(axis=(2, 3)).mean():.3f} per sequence, "
          "as they must")
    node_check = np.exp(alpha + beta - table_log_z[:, None, None])
    print(f"  alpha plus beta minus log Z reproduces the marginals: "
          f"{np.abs(node_check - node).max():.1e}")
    decoding, _ = _random_chain(0, 3, 5, 2, variant="viterbi")
    posterior = decoding.posterior_decode(x)
    deficit = model.score(x, best_path) - model.score(x, posterior)
    differences = int(np.any(posterior != best_path, axis=1).sum())
    print(f"  posterior decoding differs from Viterbi on {differences} of 3 sequences")
    print(f"  its joint score is {deficit:.2f} lower, because the best label at each")
    print("  position need not be part of the best sequence")
    assert np.abs(log_z - truth_log_z).max() < 1e-10
    assert np.abs(node - truth_node).max() < 1e-10
    assert np.abs(best_score - path_score).max() < 1e-10
    assert posterior.shape == best_path.shape
    assert deficit > 0.0
    assert np.abs(node_check - node).max() < 1e-10


def _demo_gradient() -> None:
    """The likelihood gradient against central differences."""
    model, x = _random_chain(1, 3, 5, 2)
    labels = np.array([[0, 1, 2, 1, 0], [2, 2, 0, 1, 2], [1, 0, 0, 2, 1]])
    loss, grads = model._gradients(x, labels)
    names = ("W_", "b_", "A_", "s_", "e_")
    print("chain likelihood, analytic gradient against central differences:")
    worst = 0.0
    for name in names:
        gradient = grads[name]
        error = 0.0
        parameter = getattr(model, name)
        flat = parameter.ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + 1e-6
            plus, _ = model._gradients(x, labels)
            flat[index] = original - 1e-6
            minus, _ = model._gradients(x, labels)
            flat[index] = original
            numeric = (plus - minus) / 2e-6
            numeric = np.asarray(numeric)
            error = max(error, abs(float(numeric) - float(gradient.ravel()[index]))
                        / max(1e-12, abs(float(numeric)) + abs(float(gradient.ravel()[index]))))
        worst = max(worst, error)
        print(f"  {name:<3} {flat.size:>3} scores   worst relative error {error:.1e}")
    print(f"  the likelihood itself is {loss:.6f}, matching log Z minus the path score")
    assert worst < 1e-7

    rng = np.random.default_rng(2)
    second = CRF(variant="higher_order")
    second._init(1, 2)
    second.W_ = 0.5 * rng.standard_normal((1, 2))
    second.A2_ = rng.standard_normal((3, 2, 2))
    second.s2_ = 0.5 * rng.standard_normal(2)
    second.e2_ = 0.5 * rng.standard_normal(2)
    x2 = rng.standard_normal((2, 4, 1))
    y2 = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    _, grads2 = second._gradients(x2, y2)
    worst2 = 0.0
    for name in ("W_", "b_", "A2_", "s2_", "e2_"):
        parameter = getattr(second, name)
        flat = parameter.ravel()
        gradient = grads2[name].ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + 1e-6
            plus, _ = second._gradients(x2, y2)
            flat[index] = original - 1e-6
            minus, _ = second._gradients(x2, y2)
            flat[index] = original
            numeric = (plus - minus) / 2e-6
            worst2 = max(worst2, abs(float(numeric) - float(gradient[index]))
                         / max(1e-12, abs(float(numeric)) + abs(float(gradient[index]))))
    print(f"  second order, {second.n_parameters()} scores: "
          f"worst relative error {worst2:.1e}")
    assert worst2 < 1e-7


def _hmm_sequences(seed: int, count: int, length: int, stick: float) -> tuple:
    """Labels from a sticky chain, observations from a noisy channel."""
    rng = np.random.default_rng(seed)
    states = 3
    transition = stick * np.eye(states) + (1.0 - stick) / (states - 1) * (1.0 - np.eye(states))
    emission = 0.85 * np.eye(states) + 0.15 / states
    features, labels = [], []
    for _ in range(count):
        truth = np.empty(length, dtype=int)
        truth[0] = rng.integers(states)
        for t in range(1, length):
            truth[t] = rng.choice(states, p=transition[truth[t - 1]])
        observed = np.array([rng.choice(states, p=emission[state]) for state in truth])
        features.append(np.eye(states)[observed])
        labels.append(truth)
    return np.stack(features), np.stack(labels)


def _demo_training() -> None:
    """Three ways to climb the same likelihood, on the same data."""
    x, y = _hmm_sequences(7, 40, 8, 0.9)
    test_x, test_y = _hmm_sequences(8, 400, 8, 0.9)
    print("a sticky three-state chain, 40 sequences of length 8, 400 held out:")
    print("  variant        iterations   gradient calls   held-out accuracy   final loss")
    for variant, iterations, lr in (("linear_chain", 400, 0.5), ("crf_sgd", 200, 0.2),
                                    ("crf_lbfgs", 100, 0.2)):
        model = CRF(variant=variant, n_iter=iterations, lr=lr, batch=8, seed=0).fit(x, y)
        print(f"  {variant:<14} {iterations:>10}   {model.n_evaluations_:>14}   "
              f"{model.accuracy(test_x, test_y):>16.3f}   {model.history_[-1]:>10.4f}")
        assert model.history_[-1] < model.history_[0]
        assert model.accuracy(test_x, test_y) > 0.9
    print("  The quasi-Newton pass reaches the same loss with the fewest steps and")
    print("  the fewest gradient evaluations, because each step uses curvature from")
    print("  the last few steps; the stochastic pass spends five evaluations per")
    print("  iteration and barely improves on the plain full-batch descent.")


def _second_order_sequences(seed: int, count: int, length: int,
                            stick: float) -> tuple:
    """A process no first-order chain can represent: 0 0 1 1 2 2 ..."""
    rng = np.random.default_rng(seed)
    states = 3
    features, labels = [], []
    for _ in range(count):
        offset = int(rng.integers(states))
        truth = np.array([(offset + t // 2) % states for t in range(length)])
        emission = np.full((length, states), (1.0 - stick) / (states - 1))
        emission[np.arange(length), truth] = stick
        observed = np.array([rng.choice(states, p=row) for row in emission])
        features.append(np.eye(states)[observed])
        labels.append(truth)
    return np.stack(features), np.stack(labels)


def _demo_higher_order() -> None:
    """A second-order rule that a first-order chain cannot express."""
    x, y = _second_order_sequences(3, 40, 16, 0.75)
    test_x, test_y = _second_order_sequences(4, 400, 16, 0.75)
    print("labels run 0 0 1 1 2 2 0 0 ..., emitted through a noisy channel:")
    print("  order   scores   held-out accuracy   held-out log likelihood")
    for variant, order in (("crf_lbfgs", 1), ("higher_order", 2)):
        model = CRF(variant=variant, n_iter=300, lr=0.2, seed=0).fit(x, y)
        held = model.log_likelihood(test_x, test_y)
        print(f"  {order:>5}   {model.n_parameters():>6}   "
              f"{model.accuracy(test_x, test_y):>17.3f}   {held:>23.3f}")
        if order == 1:
            first = model.accuracy(test_x, test_y)
        else:
            second = model.accuracy(test_x, test_y)
            assert second > first + 0.2
            assert held > -1.0
    print("  The pair of previous labels is the state, so the second-order field")
    print("  scores a transition the first-order one has no way to index; the extra")
    print("  27 scores buy the whole rule.")


def _demo_guards() -> None:
    """Malformed configurations and inputs, all refused."""
    x, y = _hmm_sequences(9, 6, 4, 0.9)
    attempts = [
        lambda: CRF(variant="nope"),
        lambda: CRF(n_iter=0),
        lambda: CRF(lr=0.0),
        lambda: CRF(batch=0),
        lambda: CRF(memory=0),
        lambda: CRF(variant="viterbi").fit(x, y),
        lambda: CRF(variant="forward_backward").fit(x, y),
        lambda: CRF(variant="viterbi").posterior_decode(x),
        lambda: CRF(variant="linear_chain").forward_backward(x),
        lambda: CRF().log_partition(x),
        lambda: CRF().marginals(x),
        lambda: CRF().decode(x),
        lambda: CRF().fit(x, y[:, :3]),
        lambda: CRF().fit(x.astype(float) + np.nan, y),
        lambda: CRF().fit(np.zeros((2, 3, 4, 5)), y),
        lambda: CRF().fit(np.zeros((0, 3, 2)), np.zeros((0, 3), dtype=int)),
        lambda: CRF().fit(x, np.zeros_like(y)),
        lambda: CRF().fit(x, y[0]),
    ]
    model = CRF().fit(x, y)
    attempts += [
        lambda: model.log_partition(np.zeros((1, 4, 7))),
        lambda: model.marginals(np.zeros((5,))),
        lambda: model.decode(np.full((1, 4, 3), np.inf)),
        lambda: model.log_likelihood(x, y + 5),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused: configuration, an inference variant")
    print("  asked to fit, a training variant asked for the wrong table, an unfitted")
    print("  model, a length mismatch, a non-finite feature, a four-dimensional")
    print("  array, an empty batch, a single label value, an unseen label and a")
    print("  wrong width at prediction time.")
    assert refused == len(attempts)


def main() -> None:
    _demo_enumeration()
    _demo_gradient()
    _demo_training()
    _demo_higher_order()
    _demo_guards()
    print("crf demo passed")


if __name__ == "__main__":
    main()
