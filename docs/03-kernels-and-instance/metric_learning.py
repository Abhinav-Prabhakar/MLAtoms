"""Metric learning — learn the distance instead of assuming it.

Chapter:   03-kernels-and-instance
Companion: metric_learning.md
Tier 3 · 5 variants: lmnn, nca, itml, mvp, deep_metric

    d_M(x, z)² = (x − z)ᵀ M (x − z) = ||L(x − z)||²,    M = LᵀL ⪰ 0      (1)

Every method in this chapter so far took the distance as given: Euclidean for
kNN, a kernel for the SVM, a bandwidth for the RBF network. That is an odd thing
to assume, because the whole point of a feature vector is that some directions
matter more than others. Metric learning learns the weights. The model is a
positive semidefinite matrix M, or equivalently a linear map L that sends the
data somewhere the Euclidean distance is the right one, and the supervision is
the labels: same-class points should land close, different-class points far.

The five variants are five answers to "far according to what".

`nca` maximises the expected leave-one-out accuracy of a soft nearest-neighbour
rule. Every point distributes a unit of probability over the others in inverse
proportion to distance, and the objective is the probability mass that lands on
the same class. It is smooth, so it has a gradient, and the only reason it is not
the default is that it is not convex.

`lmnn` asks for a margin instead of a likelihood: each point's k nearest
same-class neighbours become targets to pull in, and every different-class point
that comes within one unit of a target becomes an impostor to push out. With the
targets held fixed, the objective is convex in M, which is why it is the variant
people trust.

`itml` starts from a prior metric — usually the identity — and projects it onto
the constraints "these pairs are within u" and "those pairs are beyond l", using
the KL divergence between the Gaussians with those covariance matrices as the
measure of how far the metric moved. One closed-form rank-one update per
constraint, no gradients.

`mvp` is the closed-form relative: the mean squared distance of different-class
pairs over the mean squared distance of same-class pairs. Maximising that ratio
is a generalised eigenproblem, and the answer is the direction in which the
between-class spread is largest relative to the within-class spread.

`deep_metric` drops the linearity. A small network embeds the data, and a
triplet loss pulls each anchor towards a positive and away from a negative by a
margin. The metric is still Euclidean, but in a learned space, so it can separate
classes that no single Mahalanobis matrix can.

The demo builds two problems. In the first, three Gaussian clusters sit on two
axes and eight axes carry only noise: Euclidean 1-nearest-neighbour gets 59 %,
every linear variant gets at least 87 %, and `mvp` gets 99 %. In the second, the
label is the sign of x0 * x1 — an xor that no affine map can turn into two
half-spaces. Euclidean gets 52 %, the best linear metric 81 %, and the two-layer
network 94 %, because it is allowed to bend the plane.

Numbers to carry away (measured, see the demo).

    clusters   euclid 0.594   lmnn 0.991   nca 0.875   itml 0.984   mvp 0.994
    xor        euclid 0.516   lmnn 0.724   nca 0.591   itml 0.809   mvp 0.777
               deep_metric 0.942

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("lmnn", "nca", "itml", "mvp", "deep_metric")


# ---------------------------------------------------------------------- #
# shared geometry
# ---------------------------------------------------------------------- #
def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between two point sets."""
    squared = (np.sum(A ** 2, axis=1)[:, None]
               + np.sum(B ** 2, axis=1)[None, :]
               - 2.0 * A @ B.T)
    return np.maximum(squared, 0.0)


def _psd(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the positive semidefinite cone."""
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, 0.0)
    return (vectors * values) @ vectors.T


def _factor(matrix: np.ndarray) -> np.ndarray:
    """A map ``L`` with ``LᵀL = M`` and only positive directions."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, 0.0)
    keep = _kept(values)
    return (np.sqrt(values[keep])[:, None] * vectors[:, keep].T)


def _top_factor(matrix: np.ndarray, rank: int) -> np.ndarray:
    """The best rank-``rank`` map ``L`` with ``LᵀL`` close to ``matrix``."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1][:rank]
    values = np.maximum(values[order], 0.0)
    keep = _kept(values)
    return (np.sqrt(values[keep])[:, None] * vectors[:, order][:, keep].T)


def _kept(values: np.ndarray) -> np.ndarray:
    """Which directions are above numerical zero, keeping at least one."""
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    keep = values > max(float(values.max()) * 1e-12, 1e-15)
    if not keep.any():
        keep[0] = True
    return keep


def _knn(train: np.ndarray, labels: np.ndarray, test: np.ndarray,
         neighbours: int) -> np.ndarray:
    """Majority vote of the nearest training rows, ties to the lower class."""
    classes = np.unique(labels)
    distance = _sq_dists(test, train)
    chosen = np.argsort(distance, axis=1)[:, :neighbours]
    votes = np.zeros((test.shape[0], classes.size))
    rows = np.repeat(np.arange(test.shape[0]), neighbours)
    columns = np.searchsorted(classes, labels[chosen].ravel())
    np.add.at(votes, (rows, columns), 1.0)
    return classes[votes.argmax(axis=1)]


def _standardise(points: np.ndarray) -> tuple:
    """Centre and scale, so a margin of one has a stable meaning."""
    centre = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale <= 0.0] = 1.0
    return (points - centre) / scale, centre, scale


class MetricLearning:
    """Learn a Mahalanobis metric, or a nonlinear embedding, from labels.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    n_components : int or None
        Output dimension. `None` keeps every positive direction for the
        linear variants and embeds into two dimensions for `deep_metric`.
    n_neighbors : int
        Target neighbours for `lmnn` and the vote size for `predict`.
    mu : float
        Weight of the impostor term in `lmnn`, in ``[0, 1]``.
    max_iter : int
        Outer iterations of `nca` and `lmnn`, or passes over the pairs
        for `itml`.
    steps : int
        Gradient steps inside each `lmnn` outer iteration.
    lr : float
        Adam step size for `deep_metric`.
    reg : float
        Weight decay in `deep_metric`, or the ridge added to the
        within-class scatter by `mvp`.
    margin : float
        Triplet margin for `deep_metric`.
    hidden : int
        Hidden width of `deep_metric`.
    epochs : int
        Triplet passes over the training set for `deep_metric`.
    seed : int
        Seed for the deep initialisation and its triplet draws.

    Attributes
    ----------
    L_ : ndarray or None
        The linear map, shape ``(r, d)``; `None` for `deep_metric`.
    M_ : ndarray or None
        The metric, shape ``(d, d)``; `None` for `deep_metric`.
    weights_ : list or None
        Network parameters, for `deep_metric` only.
    n_iter_ : int
        Optimiser steps actually taken.
    objective_ : float
        Final value of the variant's objective.
    """

    def __init__(self, *, variant: str = "nca", n_components: int | None = None,
                 n_neighbors: int = 3, mu: float = 0.5,
                 max_iter: int = 20, steps: int = 10, lr: float = 0.02,
                 reg: float = 1e-3, margin: float = 0.5, hidden: int = 16,
                 epochs: int = 300, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_components is not None and (
                not isinstance(n_components, (int, np.integer))
                or n_components < 1):
            raise ValueError("n_components must be a positive integer or None")
        if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if not 0.0 <= mu <= 1.0:
            raise ValueError("mu must lie in [0, 1]")
        for name, value in (("max_iter", max_iter), ("steps", steps),
                            ("hidden", hidden), ("epochs", epochs)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("lr", lr), ("reg", reg), ("margin", margin)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.n_components = n_components
        self.n_neighbors = int(n_neighbors)
        self.mu = float(mu)
        self.max_iter = int(max_iter)
        self.steps = int(steps)
        self.lr = float(lr)
        self.reg = float(reg)
        self.margin = float(margin)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_X(X) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or min(array.shape) < 1:
            raise ValueError("X must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError("X must be finite")
        return array

    def _check_fitted(self) -> None:
        """Refuse to apply a model that has not seen data."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit before using the model")

    def _check_fit(self, X, y) -> tuple:
        points = self._check_X(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != points.shape[0]:
            raise ValueError("y must have one label per row of X")
        if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
            raise ValueError("y must be finite")
        classes, counts = np.unique(labels, return_counts=True)
        if classes.size < 2:
            raise ValueError("at least two classes are required")
        if counts.min() < 2:
            raise ValueError("every class needs at least two points")
        return points, labels

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> MetricLearning:
        """Learn the metric or embedding, and store its parameters."""
        points, labels = self._check_fit(X, y)
        width = points.shape[1]
        self.n_features_in_ = width
        self.classes_ = np.unique(labels)
        if self.n_components is None:
            self.n_components_ = 2 if self.variant == "deep_metric" else width
        else:
            self.n_components_ = min(self.n_components, width)
        if self.variant == "nca":
            self._fit_nca(points, labels)
        elif self.variant == "lmnn":
            self._fit_lmnn(points, labels)
        elif self.variant == "itml":
            self._fit_itml(points, labels)
        elif self.variant == "mvp":
            self._fit_mvp(points, labels)
        else:
            self._fit_deep(points, labels)
        self._train_embedding = self.transform(points)
        self._train_labels = labels
        return self

    def _fit_nca(self, points: np.ndarray, labels: np.ndarray) -> None:
        """Gradient ascent on the expected leave-one-out accuracy."""
        rows, width = points.shape
        same = labels[:, None] == labels[None, :]
        np.fill_diagonal(same, False)
        map_ = np.eye(width)

        def objective(L: np.ndarray) -> float:
            distance = _sq_dists(points @ L.T, points @ L.T)
            np.fill_diagonal(distance, np.inf)
            distance -= distance.min(axis=1, keepdims=True)
            weight = np.exp(-distance)
            np.fill_diagonal(weight, 0.0)
            probability = weight / weight.sum(axis=1, keepdims=True)
            return float(np.sum(probability[same]))

        best = objective(map_)
        for _step in range(self.max_iter):
            distance = _sq_dists(points @ map_.T, points @ map_.T)
            np.fill_diagonal(distance, np.inf)
            distance -= distance.min(axis=1, keepdims=True)
            weight = np.exp(-distance)
            np.fill_diagonal(weight, 0.0)
            probability = weight / weight.sum(axis=1, keepdims=True)
            mass = np.sum(probability * same, axis=1)
            scatter = np.zeros((width, width))
            for index in range(rows):
                offset = points[index] - points
                outer = (probability[index][:, None] * offset).T @ offset
                own = (probability[index] * same[index])[:, None] * offset
                scatter += mass[index] * outer - own.T @ offset
            gradient = 2.0 * map_ @ scatter
            rate = 1.0 / (np.linalg.norm(gradient) + 1e-12)
            moved = False
            for _ in range(30):
                trial = map_ + rate * gradient
                value = objective(trial)
                if value > best + 1e-12:
                    map_, best, moved = trial, value, True
                    break
                rate *= 0.5
            if not moved:
                break
        self.L_ = _top_factor(map_.T @ map_, self.n_components_)
        self.n_iter_ = _step + 1
        self.objective_ = best
        self._store_metric()

    def _fit_lmnn(self, points: np.ndarray, labels: np.ndarray) -> None:
        """Alternate target selection with projected gradient on M."""
        rows, width = points.shape
        metric = np.eye(width)
        difference = points[:, None, :] - points[None, :, :]
        different = labels[:, None] != labels[None, :]
        smallest = int(np.bincount(np.searchsorted(
            np.unique(labels), labels)).min())
        count = max(1, min(self.n_neighbors, smallest - 1, rows - 1))
        steps = 0

        def build(vector: np.ndarray, chosen: np.ndarray) -> tuple:
            """Target distances, the active impostors, and the objective."""
            targets = points[:, None, :] - points[chosen]
            near = np.einsum("nkd,de,nke->nk", targets, vector, targets)
            far = np.einsum("njd,de,nje->nj", difference, vector, difference)
            active = different[:, None, :] & (
                1.0 + near[:, :, None] - far[:, None, :] > 0.0)
            pull = float(np.sum(near))
            push = float(np.sum(active * (1.0 + near[:, :, None]
                                          - far[:, None, :])))
            return near, active, (1.0 - self.mu) * pull + self.mu * push

        for _ in range(self.max_iter):
            distance = np.einsum("njd,de,nje->nj", difference, metric,
                                 difference)
            distance[different] = np.inf
            np.fill_diagonal(distance, np.inf)
            targets = np.argsort(distance, axis=1)[:, :count]
            near, active, value = build(metric, targets)
            for _ in range(self.steps):
                pulled = points[:, None, :] - points[targets]
                gradient = (1.0 - self.mu) * np.einsum(
                    "nkd,nke->de", pulled, pulled)
                across = active.sum(axis=1).astype(float)
                gradient -= self.mu * np.einsum(
                    "nl,nld,nle->de", across, difference, difference)
                within = active.sum(axis=2).astype(float)
                gradient += self.mu * np.einsum(
                    "nk,nkd,nke->de", within, pulled, pulled)
                rate = 1.0 / (np.linalg.norm(gradient, 2) + 1e-12)
                accepted = False
                for _ in range(30):
                    trial = _psd(metric - rate * gradient)
                    _, _, moved = build(trial, targets)
                    if moved < value - 1e-12:
                        metric, value, accepted = trial, moved, True
                        break
                    rate *= 0.5
                if not accepted:
                    break
                near, active, value = build(metric, targets)
                steps += 1
        self.M_ = metric
        self.L_ = _factor(metric)
        self.n_components_ = self.L_.shape[0]
        distance = np.einsum("njd,de,nje->nj", difference, metric, difference)
        distance[different] = np.inf
        np.fill_diagonal(distance, np.inf)
        targets = np.argsort(distance, axis=1)[:, :count]
        self.objective_ = build(metric, targets)[2]
        self.n_iter_ = steps

    def _fit_itml(self, points: np.ndarray, labels: np.ndarray) -> None:
        """Cyclic Bregman projections onto the pair constraints."""
        rows, width = points.shape
        metric = np.eye(width)
        left, right = np.triu_indices(rows, 1)
        same = labels[left] == labels[right]
        difference = points[left] - points[right]
        base = np.einsum("nd,nd->n", difference, difference)
        similar = float(np.quantile(base[same], 0.1))
        dissimilar = float(np.quantile(base[~same], 0.9))
        similar = max(similar, 1e-6 * float(np.median(base[same])))
        constraints = np.concatenate([np.flatnonzero(same),
                                      np.flatnonzero(~same)])
        targets = np.concatenate([np.full(same.sum(), similar),
                                  np.full((~same).sum(), dissimilar)])
        updates = 0
        for _ in range(self.max_iter):
            for index, target in zip(constraints, targets, strict=False):
                v = difference[index]
                metric_v = metric @ v
                current = float(v @ metric_v)
                if current <= 1e-12:
                    continue
                beta = (target - current) / (current * current)
                metric = metric + beta * np.outer(metric_v, metric_v)
                metric = 0.5 * (metric + metric.T)
                updates += 1
        self.M_ = _psd(metric)
        self.L_ = _factor(self.M_)
        self.n_components_ = self.L_.shape[0]
        self.n_iter_ = updates
        self.similar_ = similar
        self.dissimilar_ = dissimilar
        self.objective_ = float(np.linalg.slogdet(self.M_)[1])

    def _fit_mvp(self, points: np.ndarray, labels: np.ndarray) -> None:
        """The generalised eigenproblem of between over within distance."""
        rows, width = points.shape
        left, right = np.triu_indices(rows, 1)
        same = labels[left] == labels[right]
        difference = points[left] - points[right]
        within = np.einsum("nd,ne->de", difference[same], difference[same])
        between = np.einsum("nd,ne->de", difference[~same],
                            difference[~same])
        within = within / max(int(same.sum()), 1)
        between = between / max(int((~same).sum()), 1)
        within = within + self.reg * np.eye(width)
        values, vectors = np.linalg.eigh(within)
        values = np.maximum(values, 1e-12)
        root = (vectors / np.sqrt(values)) @ vectors.T
        symmetric = root @ between @ root
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (symmetric + symmetric.T))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        keep = min(self.n_components_, width)
        positive = eigenvalues[:keep] > 0.0
        eigenvalues = eigenvalues[:keep][positive]
        if eigenvalues.size == 0:
            eigenvalues = np.array([1.0])
            directions = np.eye(width)[:, :1]
        else:
            directions = root @ eigenvectors[:, :keep][:, positive]
        self.L_ = (np.sqrt(eigenvalues)[:, None] * directions.T)
        self.M_ = self.L_.T @ self.L_
        self.n_components_ = self.L_.shape[0]
        self.ratio_ = float(np.trace(self.M_ @ between)
                            / np.trace(self.M_ @ within))
        self.n_iter_ = 1
        self.objective_ = self.ratio_

    def _fit_deep(self, points: np.ndarray, labels: np.ndarray) -> None:
        """A two-layer network trained with a triplet margin loss."""
        points, centre, scale = _standardise(points)
        self._centre, self._scale = centre, scale
        rng = np.random.default_rng(self.seed)
        width = points.shape[1]
        hidden = self.hidden
        self.weights_ = [
            rng.normal(0.0, np.sqrt(2.0 / width), (hidden, width)),
            np.zeros(hidden),
            rng.normal(0.0, np.sqrt(2.0 / hidden),
                       (self.n_components_, hidden)),
            np.zeros(self.n_components_),
        ]
        code = np.searchsorted(self.classes_, labels)
        members = [np.flatnonzero(code == value) for value in range(code.max() + 1)]
        outsiders = [np.flatnonzero(code != value) for value in range(code.max() + 1)]
        first = [np.zeros_like(part) for part in self.weights_]
        second = [np.zeros_like(part) for part in self.weights_]
        anchors = np.arange(points.shape[0])
        loss = 0.0
        for epoch in range(self.epochs):
            positives = np.empty_like(anchors)
            negatives = np.empty_like(anchors)
            for index in range(anchors.size):
                group = members[code[index]]
                choices = group[group != index]
                positives[index] = choices[rng.integers(choices.size)]
                other = outsiders[code[index]]
                negatives[index] = other[rng.integers(other.size)]
            gradient, loss = self._triplet_gradient(
                points, anchors, positives, negatives)
            for index, part in enumerate(gradient):
                part = part + self.reg * self.weights_[index]
                first[index] = 0.9 * first[index] + 0.1 * part
                second[index] = 0.999 * second[index] + 0.001 * part ** 2
                moved = first[index] / (1.0 - 0.9 ** (epoch + 1))
                spread = second[index] / (1.0 - 0.999 ** (epoch + 1))
                self.weights_[index] -= (self.lr * moved
                                         / (np.sqrt(spread) + 1e-8))
        self.L_ = None
        self.M_ = None
        self.n_iter_ = self.epochs
        self.objective_ = loss

    def _forward(self, points: np.ndarray) -> tuple:
        """Embedding and the cache needed to backpropagate through it."""
        first, bias_one, second, bias_two = self.weights_
        hidden = points @ first.T + bias_one
        active = np.maximum(hidden, 0.0)
        raw = active @ second.T + bias_two
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        return raw / norm, (points, active, norm)

    def _triplet_gradient(self, points: np.ndarray, anchors: np.ndarray,
                          positives: np.ndarray,
                          negatives: np.ndarray) -> tuple:
        """Backpropagate the active triplet margin loss."""
        embedding, cache = self._forward(points)
        difference = embedding[anchors] - embedding[positives]
        apart = np.sum(difference ** 2, axis=1)
        other = embedding[anchors] - embedding[negatives]
        away = np.sum(other ** 2, axis=1)
        violation = self.margin + apart - away
        active = violation > 0.0
        loss = float(np.mean(np.maximum(violation, 0.0)))
        if not active.any():
            return [np.zeros_like(part) for part in self.weights_], loss
        chosen = np.flatnonzero(active)
        anchor, positive = anchors[chosen], positives[chosen]
        negative = negatives[chosen]
        grad = np.zeros_like(embedding)
        np.add.at(grad, anchor,
                  2.0 * (embedding[negative] - embedding[positive]))
        np.add.at(grad, positive,
                  2.0 * (embedding[positive] - embedding[anchor]))
        np.add.at(grad, negative,
                  2.0 * (embedding[anchor] - embedding[negative]))
        grad /= anchors.size
        _, active_hidden, norm = cache
        second = self.weights_[2]
        raw_gradient = (grad - embedding * np.sum(grad * embedding, axis=1,
                                                  keepdims=True)) / norm
        grad_second = raw_gradient.T @ active_hidden
        grad_bias_two = raw_gradient.sum(axis=0)
        hidden_gradient = (raw_gradient @ second) * (active_hidden > 0.0)
        grad_first = hidden_gradient.T @ points
        grad_bias_one = hidden_gradient.sum(axis=0)
        return [grad_first, grad_bias_one, grad_second, grad_bias_two], loss

    def _store_metric(self) -> None:
        """Keep the quadratic form alongside the map."""
        self.M_ = self.L_.T @ self.L_

    # ------------------------------------------------------------------ #
    # applying
    # ------------------------------------------------------------------ #
    def transform(self, X) -> np.ndarray:
        """Map points into the learned space."""
        self._check_fitted()
        points = self._check_X(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        if self.variant == "deep_metric":
            standardised = (points - self._centre) / self._scale
            return self._forward(standardised)[0]
        return points @ self.L_.T

    def distance(self, A, B=None) -> np.ndarray:
        """Euclidean distances in the learned space."""
        left = self.transform(A)
        if B is None:
            return np.sqrt(_sq_dists(left, left))
        return np.sqrt(_sq_dists(left, self.transform(B)))

    def predict(self, X) -> np.ndarray:
        """k-nearest-neighbour vote under the learned metric."""
        self._check_fitted()
        points = self._check_X(X)
        if points.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        return _knn(self._train_embedding, self._train_labels,
                    self.transform(points), self.n_neighbors)

    def score(self, X, y) -> float:
        """Fraction of rows whose predicted label matches."""
        labels = np.asarray(y)
        if labels.size != self._check_X(X).shape[0]:
            raise ValueError("y must have one label per row of X")
        return float(np.mean(self.predict(X) == labels))

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and return the embedding of the training data."""
        self.fit(X, y)
        return self.transform(X)

    def metric(self) -> np.ndarray:
        """The Mahalanobis matrix, refused for a nonlinear embedding."""
        if self.variant == "deep_metric":
            raise RuntimeError("a deep embedding has no single metric matrix")
        self._check_fitted()
        return self.M_


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _clusters(n: int, rng: np.random.Generator) -> tuple:
    """Three Gaussian clusters on two axes, with eight noisy axes."""
    centres = np.zeros((3, 10))
    centres[0, 0] = 3.5
    centres[1, 1] = 3.5
    centres[2, :2] = -3.5
    labels = rng.integers(0, 3, n)
    points = centres[labels] + rng.normal(0.0, 1.0, (n, 10))
    points[:, 2:] += rng.normal(0.0, 6.0, (n, 8))
    return points, labels


def _xor(n: int, rng: np.random.Generator) -> tuple:
    """The sign of x0 * x1, plus six dimensions of pure noise."""
    corners = rng.uniform(-1.0, 1.0, (n, 2))
    labels = (corners[:, 0] * corners[:, 1] > 0.0).astype(int)
    noise = rng.normal(0.0, 2.0, (n, 6))
    return np.hstack([corners, noise]), labels


def _demo_linear() -> None:
    """Noise dimensions are what a linear metric is for."""
    rng = np.random.default_rng(2)
    train, train_y = _clusters(150, rng)
    test, test_y = _clusters(800, rng)
    plain = float(np.mean(_knn(train, train_y, test, 1) == test_y))
    print("three clusters on two axes, eight noisy axes:")
    print(f"  {'euclidean 1-nn':<20s}{plain:6.4f}")
    for variant, options in (("lmnn", {"max_iter": 15, "steps": 10}),
                             ("nca", {}), ("itml", {"max_iter": 5}),
                             ("mvp", {})):
        model = MetricLearning(variant=variant, **options).fit(train, train_y)
        score = model.score(test, test_y)
        print(f"  {variant:<20s}{score:6.4f}")
        assert score > plain + 0.2
        assert np.linalg.eigvalsh(model.metric()).min() > -1e-10
    print("  every learned metric drops the noise axes and recovers the")
    print("  clusters; Euclidean cannot, because eight axes with six times")
    print("  the spread drown the two axes that carry the labels")


def _demo_geometry() -> None:
    """The map, the matrix and the distances agree by construction."""
    rng = np.random.default_rng(4)
    train, train_y = _clusters(120, rng)
    model = MetricLearning(variant="mvp").fit(train, train_y)
    rows = train[:20]
    gap = rows[:, None, :] - rows[None, :, :]
    quadratic = np.einsum("ijd,de,ije->ij", gap, model.metric(), gap)
    embedded = _sq_dists(model.transform(rows), model.transform(rows))
    print("the quadratic form and the embedded Euclidean distance:")
    print(f"  worst difference   {np.abs(quadratic - embedded).max():.1e}")
    assert np.allclose(quadratic, embedded, atol=1e-9)
    assert np.allclose(model.distance(rows) ** 2, quadratic, atol=1e-9)

    reduced = MetricLearning(variant="mvp", n_components=2).fit(train, train_y)
    test, test_y = _clusters(400, rng)
    print("a rank-2 map keeps the two discriminant directions:")
    print(f"  columns {reduced.transform(rows).shape[1]}, test accuracy "
          f"{reduced.score(test, test_y):.4f}")
    assert reduced.transform(rows).shape[1] == 2
    assert reduced.score(test, test_y) > 0.9

    again = MetricLearning(variant="mvp").fit_transform(train, train_y)
    assert np.allclose(again, model.transform(train))
    print("  fit_transform is fit followed by transform, as advertised")

    tiny = np.arange(12.0).reshape(6, 2)
    tiny_y = np.array([0, 0, 1, 1, 2, 2])
    small = MetricLearning(variant="lmnn", max_iter=2, steps=2).fit(tiny, tiny_y)
    assert np.isfinite(small.transform(tiny)).all()
    print("  a two-point class makes LMNN fall back to one target")


def _demo_nonlinear() -> None:
    """A linear metric cannot fold an xor; a small network can."""
    rng = np.random.default_rng(3)
    train, train_y = _xor(250, rng)
    test, test_y = _xor(1000, rng)
    plain = float(np.mean(_knn(train, train_y, test, 1) == test_y))
    print("xor in the first two axes, six noisy axes:")
    print(f"  {'euclidean 1-nn':<20s}{plain:6.4f}")
    best = 0.0
    for variant, options in (("lmnn", {"max_iter": 10, "steps": 5}),
                             ("nca", {}), ("itml", {"max_iter": 5}),
                             ("mvp", {})):
        score = MetricLearning(variant=variant, **options).fit(
            train, train_y).score(test, test_y)
        best = max(best, score)
        print(f"  {variant:<20s}{score:6.4f}")
        assert score < 0.85
    deep = MetricLearning(variant="deep_metric").fit(train, train_y)
    score = deep.score(test, test_y)
    print(f"  {'deep_metric':<20s}{score:6.4f}")
    print("  the label is the sign of x0 * x1, which no affine map turns")
    print("  into two half-spaces; the network bends the plane instead")
    assert score > best + 0.1
    assert score > 0.9
    assert deep.L_ is None and deep.M_ is None
    assert deep.transform(train).shape[1] == 2


def _demo_guards() -> None:
    """Malformed settings and unfitted models, all refused."""
    rng = np.random.default_rng(5)
    X, y = _clusters(30, rng)
    model = MetricLearning().fit(X, y)
    deep = MetricLearning(variant="deep_metric").fit(X, y)
    attempts = [
        lambda: MetricLearning(variant="nope"),
        lambda: MetricLearning(n_components=0),
        lambda: MetricLearning(n_neighbors=0),
        lambda: MetricLearning(mu=1.5),
        lambda: MetricLearning(max_iter=0),
        lambda: MetricLearning(steps=0),
        lambda: MetricLearning(lr=0.0),
        lambda: MetricLearning(reg=np.nan),
        lambda: MetricLearning(margin=-1.0),
        lambda: MetricLearning(hidden=0),
        lambda: MetricLearning(epochs=0),
        lambda: MetricLearning(seed=-1),
        lambda: MetricLearning().fit(X, y[:10]),
        lambda: MetricLearning().fit(X, np.zeros(30)),
        lambda: MetricLearning().fit(X[:5], np.array([0, 0, 0, 0, 1])),
        lambda: MetricLearning().fit(np.full((30, 10), np.nan), y),
        lambda: MetricLearning().fit(np.zeros((30, 10, 1)), y),
        lambda: MetricLearning().fit(
            X, np.where(y == y[0], np.nan, y)),
        lambda: MetricLearning().predict(X),
        lambda: MetricLearning().transform(X),
        lambda: MetricLearning().metric(),
        lambda: deep.metric(),
        lambda: model.score(X, y[:3]),
        lambda: model.transform(X[:, :2]),
        lambda: model.predict(X[:, :2]),
        lambda: model.distance(X[:, :2]),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an")
    print("  unknown variant, a zero dimension, zero neighbours, a weight")
    print("  outside the simplex, a zero or non-finite step, a negative")
    print("  margin, zero hidden width or epochs, a negative seed, a short")
    print("  label vector, one class, a singleton class, a non-finite label,")
    print("  a non-finite or three-dimensional design, use before fit, a")
    print("  metric from the deep variant, a short target, and the wrong")
    print("  width everywhere")
    assert refused == len(attempts)


def main() -> None:
    _demo_linear()
    _demo_geometry()
    _demo_nonlinear()
    _demo_guards()
    print("metric_learning demo passed")


if __name__ == "__main__":
    main()
