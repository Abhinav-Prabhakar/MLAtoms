"""Class imbalance — eight remedies for a skewed dataset.

Chapter:   13-training
Companion: class_imbalance.md
Tier 2 · 8 variants: smote, adasyn, borderline_smote, tomek, enn, focal,
class_weight, threshold_moving
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("smote", "adasyn", "borderline_smote", "tomek", "enn", "focal",
            "class_weight", "threshold_moving")
RESAMPLE = ("smote", "adasyn", "borderline_smote", "tomek", "enn")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def blobs(seed: int, majority: int = 300, minority: int = 30,
          shift: float = 1.3):
    """Two overlapping Gaussians, one of them ten times rarer."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, (majority, 2))
    rare = rng.normal(shift, 0.9, (minority, 2))
    x = np.vstack([common, rare])
    y = np.concatenate([np.zeros(majority, dtype=int),
                        np.ones(minority, dtype=int)])
    order = rng.permutation(len(x))
    return x[order], y[order]


def _distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    gap = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(gap ** 2, axis=-1))


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def _neighbours(x: np.ndarray, k: int) -> np.ndarray:
    distance = _distances(x, x)
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1)[:, :k]


def _smote(x: np.ndarray, y: np.ndarray, k: int, rng, *, borderline=False):
    """Interpolate between a minority point and one of its minority peers."""
    minority = x[y == 1]
    majority_count = int(np.sum(y == 0)) - int(np.sum(y == 1))
    if majority_count <= 0 or len(minority) < 2:
        return x, y
    order = _neighbours(minority, k)
    if borderline:
        overall = _neighbours(x, k)
        index = np.where(y == 1)[0]
        danger = np.array([np.sum(y[overall[row]] == 0) * 2 >= k
                           for row in index])
        source = np.where(danger)[0]
        if not len(source):
            source = np.arange(len(minority))
    else:
        source = np.arange(len(minority))
    picks = rng.choice(source, majority_count, replace=True)
    peers = order[picks, rng.integers(0, order.shape[1], majority_count)]
    fraction = rng.random((majority_count, 1))
    synthetic = minority[picks] + fraction * (minority[peers]
                                              - minority[picks])
    return (np.vstack([x, synthetic]),
            np.concatenate([y, np.ones(majority_count, dtype=int)]))


def _adasyn(x: np.ndarray, y: np.ndarray, k: int, rng):
    """Spend synthetic points where the minority is most outnumbered."""
    minority = x[y == 1]
    majority = x[y == 0]
    target = len(majority) - len(minority)
    if target <= 0 or len(minority) < 2:
        return x, y
    order = _neighbours(minority, k)
    overall = _neighbours(x, k)
    index = np.where(y == 1)[0]
    ratio = np.array([np.mean(y[overall[row]] == 0) for row in index])
    if ratio.sum() <= 0.0:
        ratio = np.ones(len(minority))
    share = np.floor(target * ratio / ratio.sum()).astype(int)
    synthetic = []
    for row, count in enumerate(share):
        for _ in range(count):
            peer = order[row, rng.integers(0, order.shape[1])]
            fraction = rng.random()
            synthetic.append(minority[row] + fraction * (
                minority[peer] - minority[row]))
    if not synthetic:
        return x, y
    synthetic = np.array(synthetic)
    return (np.vstack([x, synthetic]),
            np.concatenate([y, np.ones(len(synthetic), dtype=int)]))


def _tomek(x: np.ndarray, y: np.ndarray) -> tuple:
    """Drop majority points that are each other's nearest neighbour."""
    distance = _distances(x, x)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    drop = np.zeros(len(x), dtype=bool)
    for index in range(len(x)):
        partner = nearest[index]
        if y[index] != y[partner] and nearest[partner] == index:
            if y[index] == 0:
                drop[index] = True
            else:
                drop[partner] = True
    return x[~drop], y[~drop]


def _enn(x: np.ndarray, y: np.ndarray, k: int) -> tuple:
    """Drop points whose class disagrees with the majority of their k peers."""
    distance = _distances(x, x)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :k]
    votes = np.mean(y[nearest], axis=1)
    keep = (votes >= 0.5).astype(int) == y
    return x[keep], y[keep]


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #
class Imbalance:
    """A logistic classifier plus one treatment for the class skew.

    `fit` either resamples the training set or changes the loss, then trains a
    linear model on two features. `predict_proba` returns the minority
    probability, `predict` applies `threshold_`, and the counters
    `n_synthetic` and `n_removed` report what a resampler did.
    """

    def __init__(self, variant: str = "smote", *, k: int = 5,
                 gamma: float = 2.0, alpha: float = 0.9, steps: int = 1200,
                 lr: float = 0.5, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if k < 1:
            raise ValueError("k must be positive")
        if gamma < 0.0:
            raise ValueError("gamma must be non-negative")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if steps < 1:
            raise ValueError("steps must be positive")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive and finite")
        self.variant = variant
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        self.steps = steps
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(2)
        self.b = 0.0
        self.threshold_ = 0.5
        self.n_synthetic = 0
        self.n_removed = 0
        self.n_train = 0
        self.class_weights = {0: 1.0, 1: 1.0}

    # -- training ------------------------------------------------------------- #
    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Inverse frequency: each class contributes the same total mass."""
        counts = np.bincount(y, minlength=2).astype(float)
        return np.where(y == 1, 0.5 / max(counts[1], 1.0),
                        0.5 / max(counts[0], 1.0)) * len(y)

    def _gradient_step(self, x, y, weights, gamma, w, b):
        logits = x @ w + b
        probs = 1.0 / (1.0 + np.exp(-logits))
        if gamma == 0.0:
            derivative = probs - y
        else:
            true = np.where(y == 1, probs, 1.0 - probs)
            base = (1.0 - true) ** gamma
            positive = base * (gamma * probs * np.log(probs + 1e-12)
                               - (1.0 - probs))
            negative = probs * base * (1.0 - gamma * np.log(
                1.0 - probs + 1e-12))
            derivative = np.where(y == 1, positive, negative)
        weighted = derivative * weights / weights.sum()
        return w - self.lr * (x.T @ weighted), b - self.lr * weighted.sum()

    def _train(self, x, y, weights=None, gamma=0.0):
        w, b = np.zeros(2), 0.0
        if weights is None:
            weights = np.ones(len(y))
        for _ in range(self.steps):
            w, b = self._gradient_step(x, y, weights, gamma, w, b)
        return w, b

    # -- entry point ---------------------------------------------------------- #
    def fit(self, x, y) -> Imbalance:
        values = np.asarray(x, dtype=float)
        labels = np.asarray(y)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("this demo lives in two dimensions")
        if labels.shape != (len(values),):
            raise ValueError("labels must be one per row")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integers")
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError("labels must be binary")
        if np.bincount(labels, minlength=2).min() < 2:
            raise ValueError("both classes need at least two points")
        if not np.all(np.isfinite(values)):
            raise ValueError("the data must be finite")
        self.n_synthetic = 0
        self.n_removed = 0
        counts = np.bincount(labels, minlength=2).astype(float)
        self.class_weights = {0: 1.0 / counts[0], 1: 1.0 / counts[1]}
        train_x, train_y = values, labels
        if self.variant in ("smote", "borderline_smote"):
            before = len(train_x)
            train_x, train_y = _smote(
                train_x, train_y, self.k, self.rng,
                borderline=self.variant == "borderline_smote")
            self.n_synthetic = len(train_x) - before
        elif self.variant == "adasyn":
            before = len(train_x)
            train_x, train_y = _adasyn(train_x, train_y, self.k, self.rng)
            self.n_synthetic = len(train_x) - before
        elif self.variant == "tomek":
            before = len(train_x)
            train_x, train_y = _tomek(train_x, train_y)
            self.n_removed = before - len(train_x)
        elif self.variant == "enn":
            before = len(train_x)
            train_x, train_y = _enn(train_x, train_y, self.k)
            self.n_removed = before - len(train_x)

        self.n_train = len(train_x)
        if self.variant == "focal":
            weights = np.where(train_y == 1, self.alpha, 1.0 - self.alpha)
            self.w, self.b = self._train(train_x, train_y, weights,
                                         self.gamma)
        elif self.variant == "class_weight":
            self.w, self.b = self._train(
                train_x, train_y, self._sample_weights(train_y))
        elif self.variant == "threshold_moving":
            self._fit_threshold(train_x, train_y)
        else:
            self.w, self.b = self._train(train_x, train_y)
        return self

    def _fit_threshold(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(7)
        index = np.arange(len(x))
        validation = []
        for label in (0, 1):
            members = index[y == label]
            take = max(1, len(members) // 4)
            validation.extend(rng.choice(members, take, replace=False))
        validation = np.array(validation)
        keep = np.setdiff1d(index, validation)
        self.w, self.b = self._train(x[keep], y[keep])
        probs = self._probs(x[validation])
        best, chosen = -1.0, 0.5
        for threshold in np.linspace(0.02, 0.98, 25):
            score = _balanced(y[validation], probs >= threshold)
            if score > best:
                best, chosen = score, float(threshold)
        self.threshold_ = chosen
        self.w, self.b = self._train(x, y)

    # -- inference ------------------------------------------------------------ #
    def _probs(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))

    def predict_proba(self, x) -> np.ndarray:
        return self._probs(np.asarray(x, dtype=float))

    def predict(self, x) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold_).astype(int)

    def accuracy(self, x, y) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def balanced_accuracy(self, x, y) -> float:
        return _balanced(np.asarray(y), self.predict(x))

    def minority_recall(self, x, y) -> float:
        labels = np.asarray(y)
        prediction = self.predict(x)
        return float(np.mean(prediction[labels == 1] == 1))


def _balanced(y: np.ndarray, prediction: np.ndarray) -> float:
    positive = float(np.mean(prediction[y == 1] == 1))
    negative = float(np.mean(prediction[y == 0] == 0))
    return 0.5 * (positive + negative)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def demo() -> None:
    train_x, train_y = blobs(seed=1)
    test_x, test_y = blobs(seed=11)
    prior = float(np.mean(train_y == 1))

    print("imbalance. three hundred majority points and thirty minority:")
    print(f"  the training prior is {prior:.3f}; accuracy alone is useless")

    plain = Imbalance("smote", seed=2)
    plain.w, plain.b = plain._train(train_x, train_y)
    base = plain.balanced_accuracy(test_x, test_y)
    print(f"  plain logistic regression scores {base:.3f} balanced")
    print(f"  and finds {plain.minority_recall(test_x, test_y):.3f}"
          " of the minority")
    print(f"  while its plain accuracy is {plain.accuracy(test_x, test_y):.3f}")

    print("imbalance. resampling the training set:")
    table = {}
    for variant in RESAMPLE:
        model = Imbalance(variant, seed=3).fit(train_x, train_y)
        table[variant] = model.balanced_accuracy(test_x, test_y)
        print(f"    {variant:<16s} {table[variant]:.3f} balanced"
              f"  +{model.n_synthetic:3d} synthetic"
              f"  -{model.n_removed:3d} removed")

    print("imbalance. changing the loss instead:")
    for variant in ("focal", "class_weight"):
        model = Imbalance(variant, seed=3).fit(train_x, train_y)
        table[variant] = model.balanced_accuracy(test_x, test_y)
        print(f"    {variant:<16s} {table[variant]:.3f} balanced")
    print(f"    inverse frequency puts"
          f" {1.0 / prior:.0f}x more weight on the minority point")

    print("imbalance. moving the threshold after training:")
    moved = Imbalance("threshold_moving", seed=3).fit(train_x, train_y)
    table["threshold_moving"] = moved.balanced_accuracy(test_x, test_y)
    print(f"    threshold {moved.threshold_:.2f}  "
          f"{table['threshold_moving']:.3f} balanced"
          f"  {moved.minority_recall(test_x, test_y):.3f} recall")
    sweep = []
    for threshold in (0.05, 0.1, 0.25, 0.5, 0.75):
        sweep.append((threshold, _balanced(
            test_y, plain.predict_proba(test_x) >= threshold)))
    print("    the same plain model at fixed thresholds:")
    for threshold, score in sweep:
        print(f"      {threshold:.2f}  {score:.3f}")

    print("imbalance. how many rows the treatment leaves:")
    for variant in RESAMPLE:
        model = Imbalance(variant, seed=3).fit(train_x, train_y)
        print(f"    {variant:<16s} {len(train_x):3d} rows become"
              f" {model.n_train:3d}")

    print("imbalance. eight treatments, one table:")
    winner = max(table, key=table.get)
    print(f"  {winner} wins with {table[winner]:.3f} against the"
          f" plain {base:.3f}")
    assert all(score >= base - 0.05 for score in table.values())
    assert table["class_weight"] > base + 0.05
    assert table["threshold_moving"] > base + 0.05
    assert table["smote"] > base
    assert table[winner] > 0.8

    print("imbalance. malformed calls are refused:")
    guard = Imbalance("smote", seed=4)
    bad = [
        lambda: Imbalance("smote_v2"),
        lambda: Imbalance("smote", k=0),
        lambda: Imbalance("focal", gamma=-1.0),
        lambda: Imbalance("focal", alpha=0.0),
        lambda: Imbalance("focal", alpha=1.0),
        lambda: Imbalance("smote", steps=0),
        lambda: Imbalance("smote", lr=0.0),
        lambda: guard.fit(train_x[:, :1], train_y),
        lambda: guard.fit(train_x, train_y[:-1]),
        lambda: guard.fit(train_x, train_y.astype(float)),
        lambda: guard.fit(train_x, train_y + 2),
        lambda: guard.fit(train_x, np.zeros(len(train_y), dtype=int)),
        lambda: guard.fit(np.full_like(train_x, np.nan), train_y),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except ValueError:
            refused += 1
    assert refused == len(bad)
    print(f"  {refused} of {len(bad)}: an unknown variant, zero neighbours,")
    print("  a negative gamma, an alpha outside (0, 1), a zero step or")
    print("  learning rate, a one-dimensional input, a label mismatch,")
    print("  float labels, non-binary labels, a single-class training set,")
    print("  and a non-finite input")
    print("imbalance demo passed")


if __name__ == "__main__":
    demo()
