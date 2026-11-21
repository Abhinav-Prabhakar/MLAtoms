"""Semi-supervised learning — seven ways to use unlabelled points.

Chapter:   13-training
Companion: semi_supervised.md
Tier 2 · 7 variants: self_training, pseudo_label, fixmatch, mean_teacher,
mixmatch, label_propagation, ladder
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("self_training", "pseudo_label", "fixmatch", "mean_teacher",
            "mixmatch", "label_propagation", "ladder")
CLASSES = 2


# --------------------------------------------------------------------------- #
# Data and augmentations
# --------------------------------------------------------------------------- #
def moons(seed: int, n: int = 400, noise: float = 0.12):
    """Two interleaving half circles, the standard semi-supervised toy set."""
    rng = np.random.default_rng(seed)
    half = n // 2
    low = rng.uniform(0.0, np.pi, half)
    high = rng.uniform(0.0, np.pi, n - half)
    first = np.column_stack([np.cos(low), np.sin(low)])
    second = np.column_stack([1.0 - np.cos(high), 1.0 - np.sin(high) - 0.5])
    x = np.vstack([first, second])
    x = x + rng.normal(0.0, noise, x.shape)
    labels = np.repeat([0, 1], [half, n - half])
    order = rng.permutation(n)
    return x[order], labels[order]


def _augment(x: np.ndarray, rng: np.random.Generator,
             strong: bool) -> np.ndarray:
    """Jitter alone is weak; a rotation and a rescale make it strong."""
    values = x.copy()
    if strong:
        angle = rng.uniform(-0.6, 0.6, len(values))
        scale = rng.uniform(0.8, 1.2, len(values))
        cosine, sine = np.cos(angle), np.sin(angle)
        centre = values.mean(axis=0)
        shifted = values - centre
        values = np.column_stack([
            cosine * shifted[:, 0] - sine * shifted[:, 1],
            sine * shifted[:, 0] + cosine * shifted[:, 1]])
        values = values * scale[:, None] + centre
        values += rng.normal(0.0, 0.15, values.shape)
    else:
        values += rng.normal(0.0, 0.05, values.shape)
    return values


# --------------------------------------------------------------------------- #
# A two-hidden-layer MLP, small enough to train inside every variant
# --------------------------------------------------------------------------- #
def _init(rng: np.random.Generator, hidden: int) -> dict:
    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / np.sqrt(rows), (rows, cols))

    return {"W1": matrix(2, hidden), "b1": np.zeros(hidden),
            "W2": matrix(hidden, hidden), "b2": np.zeros(hidden),
            "W3": matrix(hidden, CLASSES), "b3": np.zeros(CLASSES)}


def _forward(params: dict, x: np.ndarray):
    first = np.tanh(x @ params["W1"] + params["b1"])
    second = np.tanh(first @ params["W2"] + params["b2"])
    logits = second @ params["W3"] + params["b3"]
    return logits, (x, first, second)


def _backward_second(params: dict, cache, d_second: np.ndarray) -> dict:
    """Gradients from d(loss)/d(second hidden activation)."""
    x, first, second = cache
    del second
    grads = {"W2": first.T @ d_second, "b2": d_second.sum(axis=0)}
    d_first = (d_second @ params["W2"].T) * (1.0 - first ** 2)
    grads["W1"] = x.T @ d_first
    grads["b1"] = d_first.sum(axis=0)
    return grads


def _backward(params: dict, cache, dlogits: np.ndarray) -> dict:
    x, first, second = cache
    del x, first
    grads = {"W3": second.T @ dlogits, "b3": dlogits.sum(axis=0)}
    d_second = (dlogits @ params["W3"].T) * (1.0 - second ** 2)
    grads.update(_backward_second(params, cache, d_second))
    return grads


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _loss_grad(params: dict, cache, target: np.ndarray,
               weight: np.ndarray) -> np.ndarray:
    """Weighted soft-target cross-entropy, returned as d(loss)/d(logits)."""
    logits = cache[2] @ params["W3"] + params["b3"]
    probs = _softmax(logits)
    return (probs - target) * weight[:, None] / weight.sum()


def _loss_value(params: dict, cache, target: np.ndarray,
                weight: np.ndarray) -> float:
    logits = cache[2] @ params["W3"] + params["b3"]
    probs = _softmax(logits)
    return -float(np.sum(weight[:, None] * target
                         * np.log(probs + 1e-12)) / weight.sum())


def _onehot(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((len(labels), CLASSES))
    out[np.arange(len(labels)), labels] = 1.0
    return out


class _Adam:
    def __init__(self, params: dict, lr: float):
        self.lr = lr
        self.first = {k: np.zeros_like(v) for k, v in params.items()}
        self.second = {k: np.zeros_like(v) for k, v in params.items()}
        self.step_count = 0

    def step(self, params: dict, grads: dict, decay: float) -> None:
        self.step_count += 1
        correction1 = 1.0 - 0.9 ** self.step_count
        correction2 = 1.0 - 0.999 ** self.step_count
        for key, value in params.items():
            grad = grads[key] + (decay * value if key.startswith("W") else 0.0)
            self.first[key] = 0.9 * self.first[key] + 0.1 * grad
            self.second[key] = 0.999 * self.second[key] + 0.001 * grad ** 2
            value -= self.lr * (self.first[key] / correction1) / (
                np.sqrt(self.second[key] / correction2) + 1e-8)


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class SemiSupervised:
    """A labelled set, an unlabelled set, and one way to combine them.

    `fit(x_lab, y_lab, x_unlab)` trains the chosen variant and returns the
    learner. Every variant ends with a probabilistic classifier, so `predict`
    and `accuracy` are shared. `n_added` records how many pseudo-labels a
    variant committed to, which is zero for the consistency and graph methods.
    """

    def __init__(self, variant: str = "self_training", *, hidden: int = 32,
                 steps: int = 400, lr: float = 0.01,
                 weight_decay: float = 3e-3, threshold: float = 0.9,
                 lam: float = 1.0, ema: float = 0.99, alpha: float = 0.8,
                 neighbours: int = 7, rounds: int = 3, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if hidden < 1 or steps < 1:
            raise ValueError("hidden and steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if lam < 0.0:
            raise ValueError("lam must be non-negative")
        if not 0.0 < ema < 1.0:
            raise ValueError("ema must lie in (0, 1)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if neighbours < 1 or rounds < 1:
            raise ValueError("neighbours and rounds must be positive")
        self.variant = variant
        self.hidden = hidden
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.lam = lam
        self.ema = ema
        self.alpha = alpha
        self.neighbours = neighbours
        self.rounds = rounds
        self.rng = np.random.default_rng(seed)
        self.params = _init(self.rng, hidden)
        self.teacher: dict = {k: v.copy() for k, v in self.params.items()}
        self.pool_x: np.ndarray | None = None
        self.pool_scores: np.ndarray | None = None
        self.n_added = 0
        self.losses: list[float] = []

    # -- shared training ------------------------------------------------------ #
    def _sup_step(self, x, y, optimizer, weight=None, augment=False):
        if augment:
            x = _augment(x, self.rng, strong=True)
        target = _onehot(y)
        weights = (np.ones(len(y)) if weight is None
                   else np.full(len(y), weight))
        logits, cache = _forward(self.params, x)
        probs = _softmax(logits)
        loss = -float(np.sum(weights * np.log(
            probs[np.arange(len(y)), y] + 1e-12)) / weights.sum())
        optimizer.step(self.params, _backward(
            self.params, cache, _loss_grad(self.params, cache, target,
                                           weights)),
            self.weight_decay)
        return loss

    def _train(self, x, y, steps, weight=None, augment=False):
        optimizer = _Adam(self.params, self.lr)
        for _ in range(steps):
            self.losses.append(self._sup_step(x, y, optimizer, weight,
                                              augment))

    def _probs(self, x: np.ndarray) -> np.ndarray:
        return _softmax(_forward(self.params, x)[0])

    # -- variants ------------------------------------------------------------- #
    def _self_training(self, x_lab, y_lab, x_unlab) -> None:
        added = np.zeros(len(x_unlab), dtype=bool)
        for _ in range(self.rounds):
            self._train(x_lab, y_lab, self.steps)
            probs = self._probs(x_unlab)
            confidence = probs.max(axis=1)
            fresh = (confidence >= self.threshold) & ~added
            keep = np.where(fresh)[0]
            if not keep.size:
                break
            added[keep] = True
            x_lab = np.vstack([x_lab, x_unlab[keep]])
            y_lab = np.concatenate([y_lab, probs[keep].argmax(axis=1)])
        self.n_added = int(added.sum())

    def _pseudo_label(self, x_lab, y_lab, x_unlab) -> None:
        self._train(x_lab, y_lab, self.steps)
        soft = self._probs(x_unlab)
        x = np.vstack([x_lab, x_unlab])
        target = np.vstack([_onehot(y_lab), soft])
        weight = np.concatenate([np.ones(len(y_lab)),
                                 np.full(len(x_unlab), self.lam)])
        optimizer = _Adam(self.params, self.lr)
        for _ in range(self.steps):
            _, cache = _forward(self.params, x)
            self.losses.append(_loss_value(self.params, cache, target,
                                           weight))
            optimizer.step(self.params, _backward(
                self.params, cache, _loss_grad(self.params, cache, target,
                                               weight)),
                self.weight_decay)
        self.n_added = int(np.sum(soft.max(axis=1) >= self.threshold))

    def _fixmatch(self, x_lab, y_lab, x_unlab) -> None:
        optimizer = _Adam(self.params, self.lr)
        for _ in range(self.steps):
            weak = _augment(x_unlab, self.rng, strong=False)
            strong = _augment(x_unlab, self.rng, strong=True)
            probs = _softmax(_forward(self.params, weak)[0])
            confidence = probs.max(axis=1)
            mask = confidence >= self.threshold
            x = np.vstack([x_lab, strong[mask]])
            y = np.concatenate([y_lab, probs[mask].argmax(axis=1)])
            weight = np.concatenate([np.ones(len(y_lab)),
                                     np.full(int(mask.sum()), self.lam)])
            if not mask.any():
                weight = np.ones(len(y_lab))
                x, y = x_lab, y_lab
            logits, cache = _forward(self.params, x)
            self.losses.append(float(-np.mean(np.log(_softmax(logits)[
                np.arange(len(y)), y] + 1e-12))))
            optimizer.step(self.params, _backward(
                self.params, cache, _loss_grad(self.params, cache,
                                               _onehot(y), weight)),
                self.weight_decay)
            self.n_added = int(mask.sum())

    def _mean_teacher(self, x_lab, y_lab, x_unlab) -> None:
        optimizer = _Adam(self.params, self.lr)
        for _ in range(self.steps):
            weak = _augment(x_unlab, self.rng, strong=False)
            strong = _augment(x_unlab, self.rng, strong=True)
            teacher = _softmax(_forward(self.teacher, weak)[0])
            _, cache = _forward(self.params, strong)
            student = _softmax(cache[2] @ self.params["W3"]
                               + self.params["b3"])
            consistency = _backward(self.params, cache, self.lam * (
                student - teacher) / len(strong))
            _, lab_cache = _forward(self.params, x_lab)
            supervised = _backward(
                self.params, lab_cache,
                _loss_grad(self.params, lab_cache, _onehot(y_lab),
                           np.ones(len(y_lab))))
            grads = {k: supervised[k] + consistency[k] for k in self.params}
            optimizer.step(self.params, grads, self.weight_decay)
            for key in self.teacher:
                self.teacher[key] = (self.ema * self.teacher[key]
                                     + (1.0 - self.ema) * self.params[key])
            self.losses.append(float(-np.mean(np.log(student.max(axis=1)))))
        self.n_added = 0

    def _mixmatch(self, x_lab, y_lab, x_unlab) -> None:
        self._train(x_lab, y_lab, self.steps)
        guesses = np.mean([self._probs(x_unlab) for _ in range(2)], axis=0)
        sharp = guesses ** (1.0 / 0.5)
        sharp = sharp / sharp.sum(axis=1, keepdims=True)
        x = np.vstack([x_lab, x_unlab])
        target = np.vstack([_onehot(y_lab), sharp])
        weight = np.concatenate([np.ones(len(y_lab)),
                                 np.full(len(x_unlab), self.lam)])
        optimizer = _Adam(self.params, self.lr)
        for _ in range(self.steps):
            order = self.rng.permutation(len(x))
            shuffle = self.rng.permutation(len(x))
            fraction = self.rng.beta(0.75, 0.75, len(x))
            lam = np.maximum(fraction, 1.0 - fraction)[:, None]
            mixed_x = lam * x[order] + (1.0 - lam) * x[shuffle]
            mixed_y = lam * target[order] + (1.0 - lam) * target[shuffle]
            _, cache = _forward(self.params, mixed_x)
            self.losses.append(_loss_value(self.params, cache, mixed_y,
                                           weight[order]))
            optimizer.step(self.params, _backward(
                self.params, cache, _loss_grad(self.params, cache, mixed_y,
                                               weight[order])),
                self.weight_decay)
        self.n_added = len(x_unlab)

    def _label_propagation(self, x_lab, y_lab, x_unlab) -> None:
        pool = np.vstack([x_lab, x_unlab])
        gap = pool[:, None, :] - pool[None, :, :]
        distance = np.sqrt(np.sum(gap ** 2, axis=-1))
        order = np.argsort(distance, axis=1)
        adjacency = np.zeros_like(distance)
        for row in range(len(pool)):
            near = order[row, 1:self.neighbours + 1]
            adjacency[row, near] = np.exp(
                -distance[row, near] ** 2 / (2.0 * 0.3 ** 2))
        adjacency = np.maximum(adjacency, adjacency.T)
        degree = adjacency.sum(axis=1) + 1e-12
        inverse = 1.0 / np.sqrt(degree)
        normalised = adjacency * inverse[:, None] * inverse[None, :]
        seed = np.zeros((len(pool), CLASSES))
        seed[np.arange(len(y_lab)), y_lab] = 1.0
        scores = np.linalg.solve(
            np.eye(len(pool)) - self.alpha * normalised, seed)
        self.pool_x = pool
        self.pool_scores = scores
        self.n_added = len(x_unlab)

    def _ladder(self, x_lab, y_lab, x_unlab) -> None:
        optimizer = _Adam(self.params, self.lr)
        for _ in range(self.steps):
            logits, cache = _forward(self.params, x_lab)
            probs = _softmax(logits)
            self.losses.append(float(-np.mean(np.log(
                probs[np.arange(len(y_lab)), y_lab] + 1e-12))))
            clean = _backward(self.params, cache, _loss_grad(
                self.params, cache, _onehot(y_lab), np.ones(len(y_lab))))
            _, clean_cache = _forward(self.params, x_unlab)
            noisy_x = x_unlab + self.rng.normal(0.0, 0.3, x_unlab.shape)
            _, noisy_cache = _forward(self.params, noisy_x)
            cost = 2.0 * self.lam * (noisy_cache[2] - clean_cache[2]) / len(
                x_unlab)
            denoise = _backward_second(self.params, noisy_cache, cost)
            grads = dict(clean)
            for key, value in denoise.items():
                grads[key] = grads[key] + value
            optimizer.step(self.params, grads, self.weight_decay)
        self.n_added = 0

    # -- entry point ---------------------------------------------------------- #
    def fit(self, x_lab, y_lab, x_unlab) -> SemiSupervised:
        labelled = np.asarray(x_lab, dtype=float)
        unlabelled = np.asarray(x_unlab, dtype=float)
        labels = np.asarray(y_lab)
        if labelled.ndim != 2 or unlabelled.ndim != 2:
            raise ValueError("both splits must be matrices")
        if labelled.shape[1] != 2 or unlabelled.shape[1] != 2:
            raise ValueError("this demo lives in two dimensions")
        if len(labelled) < 1 or len(unlabelled) < 1:
            raise ValueError("both splits must be non-empty")
        if labels.shape != (len(labelled),):
            raise ValueError("labels must be one per labelled row")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integers")
        if labels.min() < 0 or labels.max() >= CLASSES:
            raise ValueError("labels are out of range")
        if not np.all(np.isfinite(labelled)) or not np.all(
                np.isfinite(unlabelled)):
            raise ValueError("the data must be finite")
        self.losses = []
        self.n_added = 0
        if self.variant == "self_training":
            self._self_training(labelled, labels, unlabelled)
        elif self.variant == "pseudo_label":
            self._pseudo_label(labelled, labels, unlabelled)
        elif self.variant == "fixmatch":
            self._fixmatch(labelled, labels, unlabelled)
        elif self.variant == "mean_teacher":
            self._mean_teacher(labelled, labels, unlabelled)
        elif self.variant == "mixmatch":
            self._mixmatch(labelled, labels, unlabelled)
        elif self.variant == "label_propagation":
            self._label_propagation(labelled, labels, unlabelled)
        else:
            self._ladder(labelled, labels, unlabelled)
        return self

    # -- inference ------------------------------------------------------------ #
    def predict_proba(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        if self.variant == "label_propagation":
            gap = values[:, None, :] - self.pool_x[None, :, :]
            distance = np.sqrt(np.sum(gap ** 2, axis=-1))
            order = np.argsort(distance, axis=1)[:, :self.neighbours]
            weights = np.exp(-np.take_along_axis(
                distance, order, axis=1) ** 2 / (2.0 * 0.3 ** 2))
            scores = np.take_along_axis(
                self.pool_scores[None, :, :], order[:, :, None], axis=1)
            probed = np.sum(weights[:, :, None] * scores, axis=1)
            total = probed.sum(axis=1, keepdims=True)
            return probed / np.where(total > 0.0, total, 1.0)
        return self._probs(values)

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def accuracy(self, x, y) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y)))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    pool_x, pool_y = moons(seed=1)
    test_x, test_y = moons(seed=2)
    rng = np.random.default_rng(5)
    labelled = []
    for label in range(CLASSES):
        members = np.where(pool_y == label)[0]
        labelled.extend(rng.choice(members, 4, replace=False))
    labelled = np.array(labelled)
    mask = np.ones(len(pool_x), dtype=bool)
    mask[labelled] = False
    lab_x, lab_y = pool_x[labelled], pool_y[labelled]
    unl_x = pool_x[mask]

    print("semi-supervised. two moons, eight labels, and the rest unlabelled:")
    print(f"  the labelled set is {len(lab_x)} points, four per class")
    print(f"  the unlabelled pool is {len(unl_x)} points")

    print("semi-supervised. supervised-only training on the eight labels:")
    baseline = SemiSupervised("ladder", lam=0.0, seed=4).fit(
        lab_x, lab_y, unl_x)
    base_accuracy = baseline.accuracy(test_x, test_y)
    print(f"    no unlabelled data       {base_accuracy:.3f} test accuracy")

    print("semi-supervised. the seven recipes:")
    table = {}
    for variant in VARIANTS:
        learner = SemiSupervised(variant, seed=3).fit(lab_x, lab_y, unl_x)
        table[variant] = learner.accuracy(test_x, test_y)
        print(f"    {variant:<20s} {table[variant]:.3f} accuracy"
              f"  {learner.n_added:3d} pseudo-labels")

    print("semi-supervised. what the pseudo-labels look like:")
    stretch = SemiSupervised("self_training", threshold=0.9, seed=3).fit(
        lab_x, lab_y, unl_x)
    confidence = stretch.predict_proba(unl_x).max(axis=1)
    print(f"    self training kept {stretch.n_added} of {len(unl_x)} points"
          f" above {stretch.threshold:.2f}")
    print(f"    median confidence of the kept set"
          f" {np.median(confidence[confidence >= 0.9]):.3f}")

    print("semi-supervised. agreement with the graph method:")
    graph = SemiSupervised("label_propagation", seed=3).fit(
        lab_x, lab_y, unl_x)
    graph_labels = graph.predict(test_x)
    for variant in ("self_training", "fixmatch", "mean_teacher"):
        learner = SemiSupervised(variant, seed=3).fit(lab_x, lab_y, unl_x)
        agree = float(np.mean(learner.predict(test_x) == graph_labels))
        print(f"    {variant:<14s} agrees with propagation on"
              f" {agree:.3f} of the test set")

    print("semi-supervised. a threshold sweep for self training:")
    for threshold in (0.6, 0.8, 0.9, 0.99):
        learner = SemiSupervised("self_training", threshold=threshold,
                                 seed=5).fit(lab_x, lab_y, unl_x)
        print(f"    threshold {threshold:.2f}  {learner.n_added:3d} labels"
              f"  {learner.accuracy(test_x, test_y):.3f} accuracy")
        table[f"threshold_{threshold}"] = learner.accuracy(test_x, test_y)

    print("semi-supervised. seven methods, one table:")
    winner = max(table, key=table.get)
    print(f"  {winner} wins with {table[winner]:.3f} against the"
          f" supervised {base_accuracy:.3f}")
    assert all(value >= base_accuracy - 0.03 for value in table.values())
    assert table["label_propagation"] > base_accuracy + 0.2
    assert table["mean_teacher"] > base_accuracy + 0.2
    assert table["fixmatch"] > base_accuracy
    assert table["ladder"] > base_accuracy
    assert table["self_training"] < base_accuracy + 0.03
    assert table[winner] > 0.95

    print("semi-supervised. malformed calls are refused:")
    guard = SemiSupervised("self_training", seed=6)
    bad = [
        lambda: SemiSupervised("self_training_v2"),
        lambda: SemiSupervised("self_training", hidden=0),
        lambda: SemiSupervised("self_training", steps=0),
        lambda: SemiSupervised("self_training", lr=0.0),
        lambda: SemiSupervised("self_training", weight_decay=-1.0),
        lambda: SemiSupervised("self_training", threshold=0.0),
        lambda: SemiSupervised("self_training", threshold=1.0),
        lambda: SemiSupervised("self_training", lam=-1.0),
        lambda: SemiSupervised("self_training", ema=1.0),
        lambda: SemiSupervised("label_propagation", alpha=1.5),
        lambda: SemiSupervised("label_propagation", neighbours=0),
        lambda: SemiSupervised("self_training", rounds=0),
        lambda: guard.fit(lab_x[:, :1], lab_y, unl_x),
        lambda: guard.fit(lab_x, lab_y[:-1], unl_x),
        lambda: guard.fit(lab_x, lab_y.astype(float), unl_x),
        lambda: guard.fit(lab_x, lab_y + 2, unl_x),
        lambda: guard.fit(np.full_like(lab_x, np.nan), lab_y, unl_x),
        lambda: guard.fit(lab_x, lab_y, unl_x[:0]),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, a zero width or")
    print("  step count, a non-positive learning rate, a negative decay,")
    print("  a threshold outside (0, 1), a negative lambda, an EMA of one,")
    print("  an alpha outside (0, 1), zero neighbours or rounds, a")
    print("  one-dimensional input, a label-length mismatch, float or")
    print("  out-of-range labels, a non-finite input, and an empty pool")
    print("semi-supervised demo passed")


if __name__ == "__main__":
    demo()
