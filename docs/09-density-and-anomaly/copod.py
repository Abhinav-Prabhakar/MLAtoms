r"""COPOD — the empirical copula, one marginal at a time.

Chapter:   09-density-and-anomaly
Companion: copod.md
Tier 3 · 3 variants: copod, copod_skew, ecod

    left:      L_j(x) = ( #{i : x_ij < x} + 1/2 ) / (n + 1)              (1)
    right:     R_j(x) = ( #{i : x_ij > x} + 1/2 ) / (n + 1)              (2)
    copod:     s(x) = sum_j -1/2 [ log L_j(x) + log R_j(x) ]             (3)
    skewness:  b_j = m3_j / m2_j^(3/2)                                   (4)
    copod_skew: s(x) = sum_j max( skew tail chosen by sign, symmetric )  (5)
    ecod:      s(x) = sum_j max( -log L_j, -log R_j, skew term )         (6)

Sklar's theorem says a joint distribution is its marginals plus a copula, and
the copula is the part that carries the dependence. COPOD makes the bravest
possible assumption about it: the copula is the independence copula. Then the
joint density is the product of the marginals, the negative log-likelihood is a
sum of one-dimensional terms, and every term can be estimated by counting.

That is the whole method. There is nothing to tune, no bandwidth, no tree
depth, no distance metric: the model is the empirical CDF of each column, and
the score is how improbable a point is under those CDFs. It is the simplest
multivariate density estimator in this chapter, and on tabular data it is
embarrassingly hard to beat.

The two tails are handled separately. For a point $x$ and feature $j$, (1) is
the fraction of the training sample strictly below it and (2) the fraction
strictly above, both smoothed by half a count so that neither is ever zero.
The symmetric score (3) is the average of the two negative log-probabilities:
it charges a point for being far into *either* tail, and it gives the same
score to the 3rd smallest and the 3rd largest value. That symmetry is wrong
when the distribution is skewed — in a lognormal sample the left tail is
bounded and the right tail is not — which is what (4) and (5) are for.

The skewness correction is a heuristic, and the demo shows what it does. On a
right-skewed sample it leaves the short tail alone and *doubles* the long one:
the symmetric score 3.546 stays 3.546, and the long-tail point goes to 7.092.
That is a prior about where anomalies live, and on the beta sample both probes
are outside the support, so the prior is arbitrary: $-0.5$ is as impossible as
$1.5$ and gets half the weight. ECOD takes the maximum of both tails at full
weight, which is why it is the safer default.

The structural limitation is independence. On a Gaussian with correlation 0.95,
the points $(2.5, 2.5)$ and $(2.5, -2.5)$ have identical marginal coordinates,
so every variant gives them the *same* score to machine precision — while the
true squared Mahalanobis distance is 6.4 for the first and 250 for the second.
Neither is flagged, because along the ridge a 2.5-sigma marginal is ordinary:
the clean 99th percentile is higher than both. The first probe is a normal draw
from the joint and the second is impossible, and the independence copula cannot
tell which is which.

The first demo's table is the cost of summing logs: on uniform data the mean
`copod` score is exactly the number of features, so a score of 6 is extreme at
$p=2$ and unremarkable at $p=16$. A raw COPOD score means nothing without the
dimension it came from.
"""  # noqa: E501

from __future__ import annotations

import numpy as np

VARIANTS = ("copod", "copod_skew", "ecod")


def _tail_probabilities(sorted_values: np.ndarray,
                        x: np.ndarray) -> tuple:
    """Left and right tail probabilities, equations (1) and (2).

    The half-count smoothing is what keeps the logs finite: the smallest value
    in the sample has L = 0.5/(n+1), not zero, and the largest has
    R = 0.5/(n+1). Values outside the training range clamp to those floors.
    """
    n = sorted_values.size
    below = np.searchsorted(sorted_values, x, side="left")
    above = n - np.searchsorted(sorted_values, x, side="right")
    return (below + 0.5) / (n + 1.0), (above + 0.5) / (n + 1.0)


def _skewness(X: np.ndarray) -> np.ndarray:
    """Per-column sample skewness, equation (4), with zero for a constant."""
    centred = X - X.mean(axis=0)
    m2 = np.mean(centred ** 2, axis=0)
    m3 = np.mean(centred ** 3, axis=0)
    safe = np.where(m2 > 0.0, m2, 1.0)
    b = np.where(m2 > 0.0, m3 / safe ** 1.5, 0.0)
    # the sign of a skewness of 1e-17 is rounding noise, not a shape
    return np.where(np.abs(b) < 1e-12, 0.0, b)


class COPOD:
    """Copula-based outlier detection with empirical marginals.

    Parameters
    ----------
    variant : {"copod", "copod_skew", "ecod"}
        Which estimator to use. `copod` is the symmetric score (3);
        `copod_skew` adds the skewness-corrected term (5); `ecod` takes the
        maximum of the two tail scores and the skew term (6).
    contamination : float
        Fraction of the training data labelled anomalous. Sets `threshold_`
        as the corresponding quantile of the training scores.

    Attributes
    ----------
    skew_ : ndarray of shape (p,)
        Sample skewness of each feature, equation (4).
    tail_ : ndarray of shape (p,)
        Which tail the skew term uses: "left", "right" or "both" (the last
        when the skewness is exactly zero and both tails are summed).
    scores_ : ndarray of shape (n,)
        Training scores, higher meaning more anomalous.
    threshold_ : float
        Score above which `predict` returns -1.
    """

    def __init__(self, *, variant: str = "copod",
                 contamination: float = 0.1) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must lie in (0, 0.5)")
        self.variant = variant
        self.contamination = float(contamination)

    def fit(self, X: np.ndarray) -> COPOD:
        if hasattr(self, "n_features_"):
            del self.n_features_
        X = self._check(X)
        n, p = X.shape
        if n < 2:
            raise ValueError(f"need at least two rows, got {n}")
        self.n_features_ = p
        self.sorted_ = [np.sort(X[:, j]) for j in range(p)]
        self.skew_ = _skewness(X)
        self.tail_ = np.where(self.skew_ < 0.0, "left",
                              np.where(self.skew_ > 0.0, "right", "both"))
        self.scores_ = self.score_samples(X)
        self.threshold_ = float(np.quantile(self.scores_,
                                            1.0 - self.contamination))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def feature_scores(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contributions to the score, shape (n, p).

        The sum along the second axis is `score_samples`. A row is the
        explanation of one point: which features made it anomalous, and by how
        much, which is the property that makes COPOD usable in a report.
        """
        if not hasattr(self, "sorted_"):
            raise RuntimeError("fit before scoring")
        X = self._check(X)
        per_dimension = np.empty_like(X)
        for j in range(self.n_features_):
            left, right = _tail_probabilities(self.sorted_[j], X[:, j])
            u_left = -np.log(left)
            u_right = -np.log(right)
            symmetric = 0.5 * (u_left + u_right)
            if self.variant == "copod":
                per_dimension[:, j] = symmetric
                continue
            skew = self.skew_[j]
            if skew < 0.0:
                skew_term = u_left
            elif skew > 0.0:
                skew_term = u_right
            else:
                skew_term = u_left + u_right
            if self.variant == "copod_skew":
                per_dimension[:, j] = np.maximum(skew_term, symmetric)
            else:
                per_dimension[:, j] = np.maximum(
                    np.maximum(u_left, u_right), skew_term)
        return per_dimension

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score; higher means more anomalous."""
        return self.feature_scores(X).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """-1 for points above the threshold, +1 for the rest."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one column")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        if hasattr(self, "n_features_") and X.shape[1] != self.n_features_:
            raise ValueError("X has the wrong number of columns")
        return X

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_uniform() -> None:
    """On uniform data the mean score has an exact value: one nat per feature."""
    rng = np.random.default_rng(0)
    print("uniform data, n = 2000, score against the closed form mean p:")
    print("  p   mean copod   mean copod_skew   mean ecod   copod 99%   ecod 99%")
    previous = 0.0
    for p in (1, 2, 4, 8, 16):
        X = rng.uniform(size=(2000, p))
        scores = {v: COPOD(variant=v).fit(X).scores_ for v in VARIANTS}
        q_plain = float(np.quantile(scores["copod"], 0.99))
        print(f"  {p:2d} {scores['copod'].mean():11.3f}"
              f" {scores['copod_skew'].mean():16.3f}"
              f" {scores['ecod'].mean():10.3f} {q_plain:11.3f}"
              f" {float(np.quantile(scores['ecod'], 0.99)):9.3f}")
        assert abs(scores["copod"].mean() - p) < 0.05
        assert q_plain > previous
        previous = q_plain

    # the tails and the atom at the point itself partition the sample
    X = rng.uniform(size=(500, 1))
    left, right = _tail_probabilities(np.sort(X[:, 0]), X[:, 0])
    assert np.allclose(left + right, X.shape[0] / (X.shape[0] + 1.0))
    assert left.min() > 0.0 and right.min() > 0.0

    # the symmetric score is symmetric in rank: the k-th smallest and the
    # k-th largest value score the same
    model = COPOD().fit(np.sort(X, axis=0))
    assert np.allclose(model.scores_, model.scores_[::-1])
    print("  The mean is the closed form: the probability integral transform")
    print("  makes every marginal uniform, and E[-log U] = 1, so the score is")
    print("  additive in the features and grows with p. The 99th percentile")
    print("  grows with it, so a threshold fitted at one dimension cannot be")
    print("  reused at another: 6 is extreme at p = 2 and ordinary at p = 16.")


def _demo_skew() -> None:
    """A skewed marginal has one long tail, and the correction picks it."""
    rng = np.random.default_rng(1)
    train = rng.beta(2.0, 5.0, size=(600, 1))
    low = float(train.min())
    high = float(train.max())
    clean = low + (high - low) * (0.05 + 0.9 * rng.beta(2.0, 5.0, size=(600, 1)))
    assert clean.min() > low and clean.max() < high

    print(f"beta(2, 5), n = 600, training range [{low:.4f}, {high:.4f}], "
          f"probes beyond both ends:")
    print("  variant       skewness   score at -0.5   score at 1.5   ratio"
          "   clean max")
    table = {}
    for variant in VARIANTS:
        model = COPOD(variant=variant).fit(train)
        probes = model.score_samples(np.array([[-0.5], [1.5]]))
        clean_max = float(model.score_samples(clean).max())
        table[variant] = (float(probes[0]), float(probes[1]), clean_max)
        print(f"  {variant:<13s} {model.skew_[0]:8.3f} {probes[0]:15.3f}"
              f" {probes[1]:14.3f} {probes[1] / probes[0]:7.2f}"
              f" {clean_max:10.3f}")

    assert abs(table["copod"][0] - table["copod"][1]) < 1e-12
    assert table["copod_skew"][1] > 1.9 * table["copod_skew"][0]
    assert abs(table["ecod"][0] - table["ecod"][1]) < 1e-12
    assert table["ecod"][0] > 1.9 * table["copod"][0]
    assert all(clean_max < low_score for low_score in
               (row[0] for row in table.values()))
    print("  The skewness is positive, so the long tail is the right one. The")
    print("  correction leaves the symmetric score alone and doubles the")
    print("  long-tail point; ECOD takes the maximum of the two tails and")
    print("  doubles both. Both probes lie outside the support, so the prior")
    print("  is arbitrary: -0.5 is as impossible as 1.5 and gets half the")
    print("  weight. On a lognormal it would be right, because the left probe")
    print("  is impossible and the right one merely rare.")


def _demo_correlation() -> None:
    """The independence copula cannot see dependence, and that is the cost."""
    rng = np.random.default_rng(2)
    rho = 0.95
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    # antithetic pairs make the training marginals exactly symmetric, so the
    # two probes really do have identical marginal counts
    half = rng.multivariate_normal(np.zeros(2), covariance, size=400)
    train = np.vstack([half, -half])
    clean = rng.multivariate_normal(np.zeros(2), covariance, size=400)
    ridge = np.array([[2.5, 2.5]])
    off = np.array([[2.5, -2.5]])

    inverse = np.linalg.inv(covariance)
    d2 = [float(point @ inverse @ point) for point in (ridge[0], off[0])]

    print(f"Gaussian with correlation {rho}, n = 800, two probes with the "
          f"same marginals:")
    print("  variant       score(2.5, 2.5)   score(2.5, -2.5)   clean 99%"
          "   flagged")
    for variant in VARIANTS:
        model = COPOD(variant=variant).fit(train)
        scores = model.score_samples(np.vstack([ridge, off]))
        cut = float(np.quantile(model.score_samples(clean), 0.99))
        flagged = (float(np.mean(model.score_samples(ridge) > cut)),
                   float(np.mean(model.score_samples(off) > cut)))
        print(f"  {variant:<13s} {scores[0]:16.6f} {scores[1]:19.6f}"
              f" {cut:11.3f}   {flagged[0]:.0f} / {flagged[1]:.0f}")
        assert abs(scores[0] - scores[1]) < 1e-12
        assert flagged == (0.0, 0.0)

    print(f"  squared Mahalanobis distance: {d2[0]:.3f} and {d2[1]:.3f}")
    assert d2[0] < 7.0 and d2[1] > 200.0
    print("  Every variant gives the two probes the same score to machine")
    print("  precision, because the score is a sum of marginal terms and the")
    print("  marginals are identical. Neither is flagged: along the ridge the")
    print("  clean 99th percentile is higher than a 2.5-sigma marginal point,")
    print("  so the score has learned that 2.5 sigma is unremarkable here.")
    print("  The first probe is an ordinary draw from the joint and the second")
    print("  is 250 chi-square units out, and the independence copula cannot")
    print("  tell which is which.")


def _demo_range() -> None:
    """Beyond the training range the empirical CDF has nothing left to say."""
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(300, 1))
    model = COPOD().fit(X)
    probes = np.array([[2.0], [10.0], [1000.0], [-1.0], [-1000.0]])
    scores = model.score_samples(probes)
    print("training uniform(0, 1), n = 300, probes far outside the range:")
    for value, score in zip(probes[:, 0], scores, strict=True):
        print(f"  x = {value:8.1f}   score = {score:.6f}")
    assert scores[0] == scores[1] == scores[2]
    assert scores[3] == scores[4]
    assert scores[0] == scores[3]
    largest = float(model.scores_.max())
    assert abs(largest - scores[0]) < 0.01
    print(f"  Every point beyond the maximum scores the same, {scores[0]:.6f},")
    print(f"  within {largest - scores[0]:.4f} of the maximum's own score. The")
    print("  ECDF saturates, so 'largest value ever seen' and 'a thousand")
    print("  times larger' are one event.")
    print("  Tail extrapolation is the fix, and it is the `extreme_value` atom.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(40, 3))
    calls = [
        lambda: COPOD(variant="nope"),
        lambda: COPOD(contamination=0.0),
        lambda: COPOD(contamination=0.5),
        lambda: COPOD(contamination=0.9),
        lambda: COPOD().fit(np.empty((0, 3))),
        lambda: COPOD().fit(np.r_[X, np.full((1, 3), np.nan)]),
        lambda: COPOD().fit(X[:1]),
        lambda: COPOD().fit(X[:, :0]),
        lambda: COPOD().fit(np.zeros((3, 3, 3))),
        lambda: COPOD().predict(X),
        lambda: COPOD().score_samples(X),
        lambda: COPOD().fit(X).score_samples(X[:, :2]),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, three "
          f"out-of-range contamination values, an empty sample, a non-finite "
          f"value, a single row, no columns, a three-dimensional array, "
          f"predicting or scoring before fitting, and a width mismatch.")


def main() -> None:
    _demo_uniform()
    _demo_skew()
    _demo_correlation()
    _demo_range()
    _demo_broken()
    print("copod demo passed")


if __name__ == "__main__":
    main()
