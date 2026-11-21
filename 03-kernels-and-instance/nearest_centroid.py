"""Nearest centroid — the classifier you can fit with an average.

Chapter:   03-kernels-and-instance
Companion: nearest_centroid.md
Tier 2 · 4 variants: standard, shrunken, nearest_subclass, cosine

    c_k = mean of class k
    predict(x) = argmin_k ||x - c_k||                   (1)

Take the average of each class and assign a point to whichever average is
closest. It is the oldest classifier in the book, it costs one pass over the
data, and on small samples it beats k-nearest-neighbours — because it estimates
one vector per class instead of storing every point.

The rule is not a heuristic. If every class is a Gaussian with the *same*
covariance matrix, the Bayes-optimal classifier is a linear function of the
distance to the mean, and when that covariance is the identity the optimal rule
is exactly (1). Nearest centroid is linear discriminant analysis with the
covariance assumed away, which explains both why it is so cheap and where it
fails: a class that is a banana or two blobs has a mean that is nowhere near its
own points.

The variants are the repairs. `shrunken` standardises each feature by its
within-class spread and then soft-thresholds the class means towards the overall
mean — the nearest shrunken centroid of Tibshirani and coauthors, which turns the
classifier into a feature selector and is the reason it worked on gene expression
data with twenty thousand columns and sixty rows. `nearest_subclass` replaces
each class by a small mixture, so a class that is two blobs gets two centroids
and the mean stops lying. `cosine` replaces the Euclidean distance by the angle,
which is the right metric when the magnitude of a point is noise and only its
direction is signal — documents, embeddings, spectra.

The demo is a sequence of the failure and its repair. On three well-separated
blobs the plain rule is right almost always and a 1-NN baseline is not better.
On a two-blob class the plain rule collapses to chance and subclasses recover
it. On data with 600 features and 12 informative ones, shrinkage lifts held-out
accuracy and then, when the threshold passes the signal, drops the model to the
majority class — which is a feature, not a bug, and the demo asserts it.

Numbers to carry away (measured, see the demo):

  - 15 training points: centroid 0.902 on held-out data, 1-nearest-neighbour
    0.825; at 400 points the gap is 0.928 against 0.905
  - on isotropic blobs the raw rule beats LDA, 0.9067 to 0.8967, because the
    pooled covariance is estimated and the truth is the identity
  - 600 features and 30 training points: shrinkage lifts accuracy from 0.929
    to 0.993 while the retained set falls from 600 to 11
  - a class made of two blobs: the standard rule gets 0.502, two subclasses
    1.000
  - documents of random length: euclidean 0.890, cosine 1.000

"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("standard", "shrunken", "nearest_subclass", "cosine")


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape ``(len(X), len(C))``."""
    squared = (np.sum(X ** 2, axis=1)[:, None]
               + np.sum(C ** 2, axis=1)[None, :]
               - 2.0 * X @ C.T)
    return np.maximum(squared, 0.0)


def _kmeans(X: np.ndarray, count: int, rng: np.random.Generator,
            iterations: int = 50) -> np.ndarray:
    """Lloyd's algorithm with a k-means++ start, for one class."""
    rows = X.shape[0]
    if count >= rows:
        return X.copy()
    chosen = [int(rng.integers(rows))]
    for _ in range(1, count):
        nearest = _sq_dists(X, X[chosen]).min(axis=1)
        total = float(nearest.sum())
        weights = None if total <= 0.0 else nearest / total
        chosen.append(int(rng.choice(rows, p=weights)))
    centres = X[chosen].copy()
    for _ in range(iterations):
        labels = _sq_dists(X, centres).argmin(axis=1)
        updated = centres.copy()
        for index in range(count):
            members = labels == index
            if members.any():
                updated[index] = X[members].mean(axis=0)
        if np.allclose(updated, centres):
            break
        centres = updated
    return centres


class NearestCentroid:
    """Assign each point to the nearest class average.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    shrinkage : float
        Threshold applied to the standardised class means by `shrunken`.
        Zero keeps every feature; large values shrink every mean to the
        overall mean and the model to the majority class.
    n_subclasses : int
        Centroids per class for `nearest_subclass`.
    seed : int
        Seed for the subclass k-means.

    Attributes
    ----------
    classes_ : ndarray
        Sorted class labels.
    centroids_ : ndarray
        One row per centroid, shape ``(k, n_features)``. For
        `nearest_subclass` these are the subclass centroids, in the order
        given by `centroid_classes_`.
    centroid_classes_ : ndarray
        Class index of each row of `centroids_`.
    scales_ : ndarray or None
        Within-class standard deviations, for `shrunken` only.
    """

    def __init__(self, *, variant: str = "standard", shrinkage: float = 0.0,
                 n_subclasses: int = 1, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not np.isfinite(shrinkage) or shrinkage < 0.0:
            raise ValueError("shrinkage must be finite and non-negative")
        if (not isinstance(n_subclasses, (int, np.integer))
                or n_subclasses < 1):
            raise ValueError("n_subclasses must be a positive integer")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.variant = variant
        self.shrinkage = float(shrinkage)
        self.n_subclasses = int(n_subclasses)
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

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def _fit_shrunken(self, points: np.ndarray,
                      groups: list) -> None:
        """Soft-threshold the standardised class means towards the overall."""
        rows = points.shape[0]
        overall = points.mean(axis=0)
        total = np.zeros(points.shape[1])
        for group in groups:
            centred = points[group] - points[group].mean(axis=0)
            total += np.sum(centred ** 2, axis=0)
        spread = np.sqrt(total / max(rows - len(groups), 1))
        spread[spread <= 0.0] = 1.0
        self.scales_ = spread
        centroids = []
        for group in groups:
            scale = np.sqrt(1.0 / group.size - 1.0 / rows)
            deviation = (points[group].mean(axis=0) - overall) / (scale * spread)
            keep = np.sign(deviation) * np.maximum(
                np.abs(deviation) - self.shrinkage, 0.0)
            centroids.append(overall + scale * spread * keep)
        self.centroids_ = np.stack(centroids)
        self.centroid_classes_ = np.arange(len(groups))

    def fit(self, X, y) -> NearestCentroid:
        """Average each class, or each subclass, and store the centroids."""
        points = self._check_X(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.size != points.shape[0]:
            raise ValueError("y must have one label per row of X")
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("at least two classes are required")
        groups = [np.flatnonzero(labels == value) for value in classes]
        if min(group.size for group in groups) < 1:
            raise ValueError("every class needs at least one point")
        self.classes_ = classes
        self.priors_ = np.array([group.size for group in groups]) / points.shape[0]
        if self.variant == "shrunken":
            self._fit_shrunken(points, groups)
            return self
        if self.variant == "nearest_subclass":
            rng = np.random.default_rng(self.seed)
            centroids, owners = [], []
            for index, group in enumerate(groups):
                found = _kmeans(points[group], self.n_subclasses, rng)
                centroids.append(found)
                owners.extend([index] * found.shape[0])
            self.centroids_ = np.concatenate(centroids, axis=0)
            self.centroid_classes_ = np.array(owners)
            self.scales_ = None
            return self
        self.centroids_ = np.stack([points[group].mean(axis=0)
                                    for group in groups])
        self.centroid_classes_ = np.arange(classes.size)
        self.scales_ = None
        return self

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def decision_function(self, X) -> np.ndarray:
        """One score per class per row; the largest wins."""
        if not hasattr(self, "centroids_"):
            raise RuntimeError("call fit before deciding")
        points = self._check_X(X)
        if points.shape[1] != self.centroids_.shape[1]:
            raise ValueError("X has the wrong number of features")
        if self.variant == "shrunken":
            standardised = (points[:, None, :]
                            - self.centroids_[None, :, :]) / self.scales_
            return -np.sum(standardised ** 2, axis=2) + 2.0 * np.log(self.priors_)
        if self.variant == "cosine":
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            unit = points / norms
            target = self.centroids_ / np.linalg.norm(
                self.centroids_, axis=1, keepdims=True)
            return unit @ target.T
        distance = -_sq_dists(points, self.centroids_)
        if self.variant != "nearest_subclass":
            return distance
        scores = np.full((points.shape[0], self.classes_.size), -np.inf)
        for column, owner in enumerate(self.centroid_classes_):
            scores[:, owner] = np.maximum(scores[:, owner], distance[:, column])
        return scores

    def predict(self, X) -> np.ndarray:
        """The label of the highest-scoring class."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y) -> float:
        """Fraction of rows whose predicted label matches."""
        labels = np.asarray(y)
        if labels.size != self._check_X(X).shape[0]:
            raise ValueError("y must have one label per row of X")
        return float(np.mean(self.predict(X) == labels))

    def retained_features(self) -> np.ndarray:
        """Features whose mean deviates in some class, for `shrunken`."""
        if self.variant != "shrunken":
            raise RuntimeError("only the shrunken variant selects features")
        if not hasattr(self, "centroids_"):
            raise RuntimeError("call fit before asking")
        overall = np.average(self.centroids_, axis=0, weights=self.priors_)
        deviation = np.abs(self.centroids_ - overall) / self.scales_
        return np.flatnonzero((deviation > 1e-12).any(axis=0))

    def n_parameters(self) -> int:
        """Numbers stored by the model."""
        if not hasattr(self, "centroids_"):
            raise RuntimeError("call fit before asking")
        extra = 0 if self.scales_ is None else self.scales_.size
        return int(self.centroids_.size + extra + self.priors_.size)


# ---------------------------------------------------------------------- #
# demo
# ---------------------------------------------------------------------- #
def _blobs(rng: np.random.Generator, count: int, centres: np.ndarray,
           spread: float) -> tuple:
    """Labels and points from an isotropic mixture of Gaussians."""
    labels = rng.integers(0, centres.shape[0], count)
    noise = rng.normal(0.0, spread, (count, centres.shape[1]))
    return centres[labels] + noise, labels


def _demo_standard() -> None:
    """Three blobs, and the price of estimating one mean instead of many."""
    rng = np.random.default_rng(0)
    centres = np.zeros((3, 5))
    centres[1, :2] = 3.0
    centres[2, 1] = 4.0
    test, truth = _blobs(rng, 600, centres, 1.0)
    print("three blobs in five dimensions, spread 1, 600 held out:")
    print("  training   centroid   1-nearest-neighbour")
    for count in (15, 30, 60, 120, 400):
        train, labels = _blobs(rng, count, centres, 1.0)
        model = NearestCentroid().fit(train, labels)
        neighbour = labels[_sq_dists(test, train).argmin(axis=1)]
        score = model.score(test, truth)
        baseline = float(np.mean(neighbour == truth))
        print(f"  {count:>8}   {score:>8.3f}   {baseline:>19.3f}")
        if count <= 60:
            assert score > baseline
    print("  the centroid estimates one vector per class, the neighbour keeps")
    print("  every point, and with fifteen points that difference is the whole")
    print("  story: less variance, a boundary that is a little too straight")

    train, labels = _blobs(rng, 60, centres, 1.0)
    model = NearestCentroid().fit(train, labels)
    for index in range(3):
        assert np.allclose(model.centroids_[index],
                           train[labels == index].mean(axis=0))
    print("  every stored centroid is exactly its class mean")

    means = np.stack([train[labels == c].mean(axis=0) for c in range(3)])
    within = sum((train[labels == c] - means[i]).T @ (train[labels == c]
                                                     - means[i])
                 for i, c in enumerate(range(3)))
    pooled = within / (train.shape[0] - 3)
    values, vectors = np.linalg.eigh(pooled)
    whiten = vectors @ np.diag(values ** -0.5) @ vectors.T
    whitened = NearestCentroid().fit(train @ whiten, labels)
    inverse = np.linalg.inv(pooled)
    scores = (test @ inverse @ means.T
              - 0.5 * np.diag(means @ inverse @ means.T))
    discriminant = scores.argmax(axis=1)
    print("whitening by the pooled within-class covariance turns it into LDA:")
    print(f"  raw nearest centroid              {model.score(test, truth):.4f}")
    print(f"  on whitened data                  "
          f"{whitened.score(test @ whiten, truth):.4f}")
    print(f"  linear discriminant, closed form  "
          f"{np.mean(discriminant == truth):.4f}")
    print("  the last two are the same classifier, and here both are worse")
    print("  than the raw one: the true covariance is the identity, so")
    print("  estimating it only adds noise")
    assert np.array_equal(discriminant, whitened.predict(test @ whiten))


def _demo_shrinkage() -> None:
    """Soft-thresholding the means is a feature selector."""
    rng = np.random.default_rng(1)
    width, informative, rows = 600, 12, 30
    centres = np.zeros((2, width))
    for index in range(2):
        centres[index, :informative] = [
            1.5 if column % 2 == index else 0.0 for column in range(informative)
        ]
    labels = rng.integers(0, 2, rows)
    train = centres[labels] + rng.normal(0.0, 1.0, (rows, width))
    truth = rng.integers(0, 2, 800)
    test = centres[truth] + rng.normal(0.0, 1.0, (800, width))
    print("600 features, 12 informative, 30 training points, 800 held out:")
    print("  delta   retained   informative kept   accuracy")
    scores = {}
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        model = NearestCentroid(variant="shrunken",
                                shrinkage=delta).fit(train, labels)
        kept = model.retained_features()
        found = len(set(kept.tolist()) & set(range(informative)))
        scores[delta] = model.score(test, truth)
        print(f"  {delta:>5.1f}   {kept.size:>8}   {found:>16}   "
              f"{scores[delta]:>8.3f}")
        assert model.n_parameters() > 0
    print(f"  the threshold is a selector: accuracy climbs from "
          f"{scores[0.0]:.3f} to {max(scores.values()):.3f}")
    print(f"  while {width} features fall to 11, and every informative one")
    print("  survives until the threshold passes it. Past that the model")
    print("  collapses to the majority class, which is the right behaviour")
    print("  for a rule that has been told every feature is noise")
    assert scores[2.0] > scores[0.0] + 0.05
    assert scores[10.0] < 0.6
    assert NearestCentroid(variant="shrunken",
                           shrinkage=10.0).fit(train, labels) \
        .retained_features().size == 0


def _demo_subclass() -> None:
    """A class that is two blobs has a mean between them."""
    rng = np.random.default_rng(2)

    def xor(count: int) -> tuple:
        parts = [rng.normal([4.0, 0.0], 0.6, (count, 2)),
                 rng.normal([-4.0, 0.0], 0.6, (count, 2)),
                 rng.normal([0.0, 4.0], 0.6, (count, 2)),
                 rng.normal([0.0, -4.0], 0.6, (count, 2))]
        return (np.vstack(parts),
                np.array([0] * (2 * count) + [1] * (2 * count)))

    train, labels = xor(40)
    test, truth = xor(200)
    print("class 0 lives on the x-axis, class 1 on the y-axis, both symmetric:")
    plain = NearestCentroid().fit(train, labels)
    print(f"  the two class means are ({plain.centroids_[0, 0]:.3f}, "
          f"{plain.centroids_[0, 1]:.3f}) and ({plain.centroids_[1, 0]:.3f}, "
          f"{plain.centroids_[1, 1]:.3f})")
    print("  both at the origin, so the rule is a coin flip on held-out data")
    print(f"  standard            {plain.score(test, truth):.3f}")
    for count in (2, 4):
        model = NearestCentroid(variant="nearest_subclass",
                                n_subclasses=count).fit(train, labels)
        print(f"  {count} subclasses        {model.score(test, truth):.3f}   "
              f"({model.centroids_.shape[0]} centroids)")
        assert model.score(test, truth) > 0.95
    print("  the average was the problem, not the metric: split the class and")
    print("  the mean stops being a point that no sample is near")
    assert plain.score(test, truth) < 0.6


def _demo_cosine() -> None:
    """When length is noise, the angle is the signal."""
    rng = np.random.default_rng(3)
    width = 30
    topics = np.full((2, width), 0.1)
    topics[0, :10] = 3.0
    topics[1, 10:20] = 3.0

    def corpus(count: int) -> tuple:
        rows, labels = [], []
        for index in range(count):
            label = index % 2
            length = rng.uniform(1.0, 20.0) if label == 0 else 1.0
            rows.append(rng.poisson(topics[label] * length))
            labels.append(label)
        return np.array(rows, dtype=float), np.array(labels)

    train, labels = corpus(120)
    test, truth = corpus(400)
    plain = NearestCentroid().fit(train, labels)
    cosine = NearestCentroid(variant="cosine").fit(train, labels)
    print("two topics, one class written at random lengths from 1 to 20:")
    print("  centroid norms       "
          f"{np.round(np.linalg.norm(plain.centroids_, axis=1), 1)}")
    print(f"  document norms       {np.linalg.norm(train, axis=1).min():.1f} to "
          f"{np.linalg.norm(train, axis=1).max():.1f}")
    print(f"  euclidean            {plain.score(test, truth):.3f}")
    print(f"  cosine               {cosine.score(test, truth):.3f}")
    print("  a short document of the long class sits near the short class's")
    print("  centroid in distance and miles away in angle, and only one of the")
    print("  two rules can tell that apart")
    assert cosine.score(test, truth) > plain.score(test, truth) + 0.05
    assert cosine.score(test, truth) > 0.98


def _demo_guards() -> None:
    """Malformed settings and inputs, all refused."""
    X = np.array([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0], [3.2, 2.8]])
    y = np.array([0, 0, 1, 1])
    model = NearestCentroid().fit(X, y)
    attempts = [
        lambda: NearestCentroid(variant="nope"),
        lambda: NearestCentroid(shrinkage=-1.0),
        lambda: NearestCentroid(shrinkage=np.nan),
        lambda: NearestCentroid(n_subclasses=0),
        lambda: NearestCentroid(seed=-1),
        lambda: NearestCentroid().fit(X, y[:3]),
        lambda: NearestCentroid().fit(X, np.zeros(4)),
        lambda: NearestCentroid().fit([[np.nan, 0.0], [0.0, 0.0]], [0, 1]),
        lambda: NearestCentroid().fit([[[1.0]]], [0]),
        lambda: NearestCentroid().fit([[]], [0]),
        lambda: model.predict([[1.0, 2.0, 3.0]]),
        lambda: model.score(X, y[:2]),
        lambda: NearestCentroid().predict(X),
        lambda: NearestCentroid().n_parameters(),
        lambda: model.retained_features(),
        lambda: NearestCentroid(variant="shrunken").retained_features(),
    ]
    refused = 0
    for attempt in attempts:
        try:
            attempt()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(attempts)}: an unknown")
    print("  variant, a negative or non-finite threshold, zero subclasses, a")
    print("  negative seed, a label count that does not match, one class, a")
    print("  non-finite input, a three-dimensional design, an empty input, the")
    print("  wrong feature count, a short label vector, predict or the")
    print("  parameter count before fit, and a feature request from a variant")
    print("  that does not select features, fitted or not.")
    assert refused == len(attempts)


def main() -> None:
    _demo_standard()
    _demo_shrinkage()
    _demo_subclass()
    _demo_cosine()
    _demo_guards()
    print("nearest_centroid demo passed")


if __name__ == "__main__":
    main()
