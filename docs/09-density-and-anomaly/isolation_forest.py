"""Isolation forest — anomalies are the points a random tree isolates first.

Chapter:   09-density-and-anomaly
Companion: isolation_forest.md
Tier 1 · 4 variants: iforest, extended, anomaly_score, max_samples

    split:   pick a random direction n, cut at a uniform offset in the node   (1)
    h(x)   = depth of the leaf x lands in + c(size of that leaf)              (2)
    c(m)   = 2 H_{m−1} − 2(m−1)/m,   H_k = Σ_{j≤k} 1/j                        (3)
    s(x)   = 2^{−E[h(x)] / c(ψ)}                                              (4)

Every estimator in this chapter up to now has tried to say how dense the data is
around a point. This one never estimates a density and never looks at distances.
It asks a procedural question instead: if I cut the data set in half at random,
over and over, how many cuts does it take before this point is alone? A point in
the middle of a cloud survives many cuts because there is always more cloud
around it. A point on the edge, or in a gap, is alone after three or four.

That question has a closed-form answer for random data, and the answer is the
whole algorithm. For `m` points, the expected depth at which a random binary
search tree isolates one of them is `c(m)` in (3), so the ratio `E[h]/c(ψ)` is
about one for a normal point and about zero for an isolated one. Feeding that
ratio through `2^{−x}` maps it to `(0, 1]` with 0.5 at the expected path: the
score is a calibrated probability-like number rather than an arbitrary distance,
which is why thresholds transfer between data sets with different sizes and
dimensions.

Two properties follow and both are measured below. The method is *subsampling*
by design: each tree sees `ψ = 256` points, not the data set, because a tree
grown on `n` points spends its depth resolving the dense middle and the score
`saturates` near 0.5 for everything outside. And it is *axis-parallel* by
default, which biases the score: a point diagonal to the cloud is isolated later
than a point on an axis at the same distance, because every cut is parallel to a
coordinate. `extended` replaces the cuts with random hyperplanes and removes the
bias; `max_samples` exposes the subsample size so the first property can be
measured rather than believed.

The variants are therefore not different algorithms. They are the four decisions
in (1)-(4): which directions the cuts use, how many points a tree sees, and
whether the output is the score `s` or the score centred on its own threshold.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

VARIANTS = ("iforest", "extended", "anomaly_score", "max_samples")


@cache
def average_path(size: int) -> float:
    """`c(size)`: expected isolation depth of a random BST node, equation (3).

    Exact harmonic numbers rather than the `log(size−1) + γ` approximation,
    because the difference matters at the sizes that occur in leaves: at
    `size = 2` the exact value is 1 and the approximation is 0.154.
    """
    if size <= 1:
        return 0.0
    harmonic = sum(1.0 / k for k in range(1, int(size)))
    return 2.0 * harmonic - 2.0 * (size - 1) / size


class _Node:
    """One split, or a leaf that stores only how many points reached it."""

    __slots__ = ("size", "depth", "normal", "offset", "left", "right")


class IsolationForest:
    """Ensemble of random isolation trees.

    Parameters
    ----------
    variant : {"iforest", "extended", "anomaly_score", "max_samples"}
        `iforest` is the published algorithm, `extended` draws hyperplane normals
        with `extension_level + 1` non-zero coordinates instead of one,
        `anomaly_score` returns the score centred on the fitted threshold and
        `max_samples` grows each tree on the whole data set when `max_samples`
        is None.
    n_estimators : int
        Number of trees. The paper uses 100, and the score's variance falls like
        `1/sqrt(t)` so more buys little.
    max_samples : int or None
        Points per tree. `None` means 256 for the first three variants and the
        full sample for `max_samples`.
    extension_level : int
        For `extended`: the normal has `min(extension_level + 1, n_features)`
        non-zero coordinates. 0 is axis-parallel, `n_features - 1` is a fully
        extended hyperplane.
    contamination : float in (0, 0.5]
        Quantile of the training scores used as the decision threshold.
    seed : int
        Seeds the split sampling; two fits with the same seed are identical.

    Attributes
    ----------
    trees_ : list of _Node
    max_samples_ : int
    threshold_ : float
        The `1 - contamination` quantile of the training scores.
    n_features_ : int
    """

    def __init__(self, *, variant: str = "iforest", n_estimators: int = 100,
                 max_samples: int | None = None, extension_level: int = 1,
                 contamination: float = 0.1, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if max_samples is not None and max_samples < 2:
            raise ValueError("max_samples must be at least 2, or None")
        if extension_level < 0:
            raise ValueError("extension_level must be non-negative")
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must lie in (0, 0.5]")
        self.variant = variant
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.extension_level = int(extension_level)
        self.contamination = float(contamination)
        self.seed = int(seed)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> IsolationForest:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        n, d = X.shape
        if n < 2:
            raise ValueError("need at least two points")
        if self.max_samples is None:
            size = n if self.variant == "max_samples" else 256
        else:
            size = self.max_samples
        self.max_samples_ = int(min(size, n))
        self.n_features_ = d
        self.depth_limit_ = int(math.ceil(math.log2(self.max_samples_)))
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            sample = rng.choice(n, size=self.max_samples_, replace=False)
            self.trees_.append(self._grow(X, sample, 0, rng))
        scores = self._calibrated(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def _grow(self, X: np.ndarray, indices: np.ndarray, depth: int,
              rng: np.random.Generator) -> _Node:
        node = _Node()
        node.size = len(indices)
        node.depth = depth
        node.normal = node.offset = node.left = node.right = None
        if depth >= self.depth_limit_ or node.size <= 1:
            return node
        rows = X[indices]
        normal = self._draw_normal(rng)
        projection = rows @ normal
        low, high = float(projection.min()), float(projection.max())
        if high - low <= 0.0:
            return node  # every point projects to the same place: a leaf
        node.normal = normal
        node.offset = float(rng.uniform(low, high))
        goes_left = projection <= node.offset
        node.left = self._grow(X, indices[goes_left], depth + 1, rng)
        node.right = self._grow(X, indices[~goes_left], depth + 1, rng)
        return node

    def _draw_normal(self, rng: np.random.Generator) -> np.ndarray:
        """Equation (1)'s direction: `aim` non-zero coordinates, the rest zero."""
        d = self.n_features_
        aim = 1 if self.variant != "extended" else min(
            self.extension_level + 1, d)
        normal = np.zeros(d)
        chosen = rng.choice(d, size=aim, replace=False)
        normal[chosen] = rng.normal(size=aim)
        length = float(np.linalg.norm(normal))
        if length == 0.0:  # measure-zero, but the caller must not divide by it
            normal[chosen[0]] = 1.0
            length = 1.0
        return normal / length

    # -- scoring ---------------------------------------------------------- #
    def path_length(self, X: np.ndarray) -> np.ndarray:
        """Mean isolation depth over the ensemble, shape (m,)."""
        X = self._check(X)
        total = np.zeros(len(X))
        for tree in self.trees_:
            for i, row in enumerate(X):
                node = tree
                while node.normal is not None:
                    side = node.left if float(row @ node.normal) <= node.offset else node.right
                    node = side
                total[i] += node.depth + average_path(node.size)
        return total / len(self.trees_)

    def _calibrated(self, X: np.ndarray) -> np.ndarray:
        """Equation (4): the anomaly score in (0, 1], higher is stranger."""
        depth = self.path_length(X)
        return 2.0 ** (-depth / average_path(self.max_samples_))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly scores, higher meaning more anomalous.

        The first three variants return (4) in `(0, 1]`; `anomaly_score` returns
        it shifted by the fitted threshold, so that zero is the decision boundary
        and the sign is the prediction.
        """
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before scoring")
        scores = self._calibrated(X)
        if self.variant == "anomaly_score":
            return scores - self.threshold_
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """`-1` for an outlier, `1` for an inlier, at the fitted threshold."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self._calibrated(X) > self.threshold_, -1, 1)

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, ties counted as half a win."""
    positive = scores[labels]
    negative = scores[~labels]
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _planted(normal: int = 300, outliers: int = 10, seed: int = 0):
    """A Gaussian blob plus points on a ring far outside it."""
    rng = np.random.default_rng(seed)
    blob = rng.normal(size=(normal, 2))
    angle = np.linspace(0.0, 2.0 * np.pi, outliers, endpoint=False)
    radius = 5.5 + 0.5 * rng.random(outliers)
    ring = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    X = np.vstack([blob, ring])
    labels = np.zeros(len(X), dtype=bool)
    labels[normal:] = True
    return X, labels


def _demo_calibration() -> None:
    """The path-length algebra, checked against its definitions."""
    assert average_path(0) == 0.0 and average_path(1) == 0.0
    assert average_path(2) == 1.0
    size = 256
    approximate = 2.0 * (math.log(size - 1) + np.euler_gamma) - 2.0 * (size - 1) / size
    gap = abs(average_path(size) - approximate)
    assert gap < 5e-3
    for m in range(2, size + 1):
        assert abs(average_path(m) - average_path(m - 1) - 2.0 / m) < 1e-12

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    forest = IsolationForest(seed=1).fit(X)
    again = IsolationForest(seed=1).fit(X)
    assert np.array_equal(forest.score_samples(X), again.score_samples(X))

    origin = np.zeros((1, 2))
    ray = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    along = forest.score_samples(ray)
    assert np.all(np.diff(along) >= 0.0) and along[0] < along[-1]
    scores = forest.score_samples(X)
    assert scores.min() > 0.0 and scores.max() <= 1.0
    at_origin = float(forest.score_samples(origin)[0])

    print(f"calibration: c(2) = {average_path(2):.0f} exactly and c(256) = "
          f"{average_path(256):.4f} against the log approximation's "
          f"{approximate:.4f}, a gap of {gap:.1e} that is the harmonic sum's "
          f"tail. Two forests with the same seed score 300 points identically. "
          f"On a standard normal cloud the score is {at_origin:.3f} at the centre "
          f"and {float(scores.mean()):.3f} averaged over the training set. 0.5 is "
          f"the value at the expected path length, so a typical point "
          f"sits just under it -- and it rises along a ray: "
          f"{', '.join(f'{v:.3f}' for v in along)} at radii 0, 2, 5 and 9, then "
          f"saturates at the last two, because isolation counts cuts rather than "
          f"distance and a point outside the cloud is alone after the same number "
          f"of them however far outside it is. The "
          f"training scores stay inside ({scores.min():.3f}, "
          f"{scores.max():.3f}], so the score is a bounded, calibrated number and "
          f"not a distance. The average is under 0.5 rather than on it because a "
          f"tree stops at ceil(log2 256) = 8 levels and the leaf correction "
          f"c(size) only approximates the depth an unlimited tree would have "
          f"used, so the normaliser is an anchor rather than an exact "
          f"probability.")


def _demo_detection() -> None:
    """A planted anomaly ring, three ways of looking at it."""
    X, labels = _planted()
    results = {}
    for variant in ("iforest", "extended", "max_samples", "anomaly_score"):
        forest = IsolationForest(variant=variant, seed=2).fit(X)
        scores = forest.score_samples(X)
        results[variant] = (_auc(scores, labels), scores,
                            float(np.mean(forest.predict(X) == -1)),
                            forest.threshold_)
    assert results["anomaly_score"][0] == results["iforest"][0]
    assert abs(results["anomaly_score"][2] - 0.1) < 0.02
    for name, (auc, _, _, _) in results.items():
        assert auc > 0.99, (name, auc)

    far = np.argmax(np.linalg.norm(X, axis=1))
    print("detection: 300 standard normal points and 10 planted on a ring of "
          "radius 5.5 to 6, contamination 0.1:")
    for variant, (auc, scores, flagged, _) in results.items():
        centre = "" if variant != "anomaly_score" else " (centred: score - threshold)"
        print(f"  {variant:14s} AUC {auc:.4f}  score of the furthest point "
              f"{scores[far]:+.3f}  flagged {flagged:.1%}{centre}")
    print(f"  Every variant finds the ring, which is the easy part. The "
          f"`anomaly_score` row is the same ranking as `iforest` -- same AUC to "
          f"the digit -- with every score shifted by the fitted threshold "
          f"{results['iforest'][3]:.3f}, so its decision boundary is at zero and "
          f"`predict` is the sign. "
          f"Exactly {results['anomaly_score'][2]:.0%} of the training set is "
          f"flagged, because the threshold is that quantile by construction.")


def _demo_artefact() -> None:
    """Axis-parallel cuts are not rotation invariant, and that is measurable."""
    rng = np.random.default_rng(3)
    blob = rng.normal(size=(400, 2))
    angle = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    circle = 4.0 * np.column_stack([np.cos(angle), np.sin(angle)])
    rows = []
    for variant in ("iforest", "extended"):
        forest = IsolationForest(variant=variant, extension_level=1, seed=4).fit(blob)
        scores = forest.score_samples(circle)
        rows.append((variant, float(scores.min()), float(scores.max())))
    # In two dimensions extension_level=1 is already a fully extended
    # hyperplane: min(1+1, 2) = 2 non-zero coordinates.
    repeats = [float(IsolationForest(variant="extended", extension_level=1,
                                    seed=seed).fit(blob).score_samples(circle[:1])[0])
               for seed in (4, 5, 6)]
    noise = float(np.std(repeats))
    axis_spread = rows[0][2] - rows[0][1]
    hyper_spread = rows[1][2] - rows[1][1]
    assert hyper_spread < axis_spread

    # The circle is at a fixed radius, so every point is equally anomalous.
    print(f"artefact: 400 normal points, then 720 points on a circle of radius 4 "
          f"-- a fair score is constant around the circle. Axis-parallel cuts "
          f"give a score that runs from {rows[0][1]:.3f} to {rows[0][2]:.3f} "
          f"around it, a spread of {axis_spread:.3f}, because a point on a "
          f"diagonal needs more cuts than a point on an axis at the same "
          f"distance. Fully extended hyperplane cuts run from {rows[1][1]:.3f} to "
          f"{rows[1][2]:.3f}, a spread of {hyper_spread:.3f}: two thirds of the "
          f"angular variation is the split geometry and is removed, and the rest "
          f"is {hyper_spread / noise:.0f} times the seed-to-seed noise of "
          f"{noise:.3f} at a fixed point, so a little anisotropy survives -- some "
          f"of it the 400-point cloud's own asymmetry, which no split geometry "
          f"can average away.")


def _demo_subsample() -> None:
    """What the 256 buys, and what the full sample costs."""
    X, labels = _planted(normal=600, outliers=12, seed=5)
    assert IsolationForest(variant="max_samples").fit(X).max_samples_ == len(X)
    report = {}
    for size in (16, 64, 256, len(X)):
        picks, planted = [], []
        for seed in range(5):
            forest = IsolationForest(max_samples=size, seed=seed).fit(X)
            scores = forest.score_samples(X)
            picks.append(_auc(scores, labels))
            planted.append(float(scores[labels].mean()))
        report[size] = (float(np.mean(picks)), float(np.max(picks) - np.min(picks)),
                        float(np.mean(planted)))
    smallest, largest = report[16], report[len(X)]
    print("subsample: 600 normal points and 12 planted, five seeds per row, "
          "t = 100 trees:")
    for size, (mean, spread, planted) in report.items():
        label = "full" if size == len(X) else str(size)
        print(f"  max_samples {label:>4s}  AUC {mean:.4f}  seed spread "
              f"{spread:.4f}  mean score of the planted {planted:.3f}")
    print(f"  Sixteen points per tree is too few: the AUC drops to "
          f"{smallest[0]:.3f} because a tree of sixteen cannot separate a ring "
          f"point from the cloud's edge, and the seed spread is {smallest[1]:.3f}. "
          f"At the full sample the AUC is {largest[0]:.3f} and the spread "
          f"{largest[1]:.3f}. The other cost is on the scale of the score: the "
          f"planted points average {smallest[2]:.3f} at sixteen points per tree "
          f"and {largest[2]:.3f} at the full sample, so a threshold fitted at one "
          f"sample size is not a threshold at another -- which is why the "
          f"normaliser is c(max_samples_) and not a constant. On this data the "
          f"default 256 is already at the full sample's AUC, and the paper's "
          f"argument for it is about cost as much as accuracy: each tree is "
          f"O(256 log 256) whatever n is, and swamping only appears when the "
          f"normal cloud is dense enough to bury the outliers, which this ring "
          f"is not.")


def _demo_broken() -> None:
    """The inputs the ensemble refuses."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 3))
    cases = [
        ("an unknown variant", lambda: IsolationForest(variant="lof")),
        ("no trees", lambda: IsolationForest(n_estimators=0)),
        ("a sample of one", lambda: IsolationForest(max_samples=1)),
        ("negative contamination", lambda: IsolationForest(contamination=-0.1)),
        ("a negative extension level", lambda: IsolationForest(extension_level=-1)),
        ("one point", lambda: IsolationForest().fit(X[:1])),
        ("a 1-D input", lambda: IsolationForest().fit(X[:, 0])),
        ("scoring before fitting", lambda: IsolationForest().score_samples(X)),
        ("the wrong width", lambda: IsolationForest().fit(X).score_samples(X[:, :2])),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name} did not raise")

    # A duplicated point is not an error, and it is not harmless either.
    doubled = np.vstack([X, X[0]])
    forest = IsolationForest(seed=7).fit(doubled)
    single = IsolationForest(seed=7).fit(X)
    moved = float(forest.score_samples(doubled[:1])[0] - single.score_samples(X[:1])[0])
    hidden = bool(single.predict(X[:1])[0] == -1 and forest.predict(doubled[:1])[0] == 1)
    assert hidden
    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. Duplicating one point is not "
          f"an error, and it is not harmless: the original is flagged at "
          f"contamination 0.1 before the copy is added, and neither copy is "
          f"flagged after, because the two sit on the same paths and the "
          f"pair behaves like a two-point cluster. The score moves by "
          f"{moved:+.4f}. That is masking, and it is a density effect in a "
          f"method usually described as not having one: isolation is a property "
          f"of the sample, and duplicating a point changes the sample.")


def main() -> None:
    _demo_calibration()
    _demo_detection()
    _demo_artefact()
    _demo_subsample()
    _demo_broken()
    print("isolation_forest demo passed")


if __name__ == "__main__":
    main()
