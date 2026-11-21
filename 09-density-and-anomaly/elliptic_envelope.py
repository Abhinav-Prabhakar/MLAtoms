r"""Elliptic envelope — the robust ellipsoid as a decision region.

Chapter:   09-density-and-anomaly
Companion: elliptic_envelope.md
Tier 2 · 4 variants: mcd_ellipse, min_cov_det, shrinkage, chi2_threshold

    distance:  d2(x) = (x - mu)^T Sigma^-1 (x - mu)                        (1)
    region:    R = { x : d2(x) <= t }                                      (2)
    volume:    V_p t^(p/2) sqrt(det Sigma),  V_p = pi^(p/2) / Gamma(p/2+1) (3)
    gaussian:  P(d2 <= t) = chi2_p(t)                                      (4)
    mcd:       min over |H| = h of det( cov(H) )                           (5)
    correction: c = alpha / F_{chi2_{p+2}}(chi2_p(alpha)),  alpha = h/n    (6)

The Mahalanobis atom asks how far a point is from a centre. This one asks a
different question: which points are *inside*. The answer is an ellipsoid, and
an ellipsoid is three choices — where its centre is, what shape it has, and how
big it is. Almost every difference between elliptic-envelope implementations is
one of those three choices.

The centre and shape come from a covariance estimate. The classical estimate is
the sample mean and covariance, and it is not robust: a tenth of the points
moved away can drag the centre into the gap and inflate the shape until the
outliers sit comfortably inside their own envelope. The minimum covariance
determinant estimator (5) instead looks for the h points whose covariance has
the smallest determinant — the most concentrated h-subset — and uses their mean
and covariance. Since the determinant is a volume, "smallest determinant" means
"tightest ellipsoid containing h points", and outliers cannot make a set tighter
by being far away.

The size comes from a threshold. The Gaussian answer is (4): under a
multivariate normal the squared distance is chi-square with p degrees of
freedom, so the envelope containing a fraction 1-alpha is the ellipsoid at the
chi-square quantile. That is exact when the data are Gaussian and wrong when
they are not. The distribution-free answer is the empirical quantile of the
training distances, which is exact in-sample by construction and correct
out-of-sample only if the training set is representative.

The four variants are the useful corners of that grid:

- `min_cov_det` is the raw estimator with the chi-square boundary. It is what
  you get by implementing (5) and stopping, and it under-covers badly.
- `mcd_ellipse` adds the small-sample correction (6) and a reweighting step, and
  sizes the envelope by the empirical quantile. This is the complete
  distribution-free detector.
- `chi2_threshold` keeps the corrected estimator and the theoretical boundary,
  which is what scikit-learn's `EllipticEnvelope` does.
- `shrinkage` replaces the robust estimator with Ledoit-Wolf shrinkage, which is
  the only one of the four that works when the dimension approaches the sample
  size.

On clean two-dimensional Gaussian data, 400 points with correlation 0.6 and a
nominal 90 percent coverage, the four behave as the algebra says:

| variant | centre error | fresh coverage | volume | scale |
|:--|--:|--:|--:|--:|
| `mcd_ellipse` | 0.047 | 0.897 | 11.656 | 1.344 |
| `min_cov_det` | 0.046 | 0.488 | 3.568 | 1.000 |
| `shrinkage` | 0.091 | 0.899 | 11.714 | 1.000 |
| `chi2_threshold` | 0.046 | 0.876 | 11.555 | 3.238 |

The analytic volume of the 90 percent ellipse of the true covariance is 11.574,
and the three corrected estimators land on it to within one percent. The raw
determinant minimiser has a volume of 3.568 and covers 49 percent of fresh
points, because the covariance of the h most concentrated points is biased low.
That bias is (6), and it is 3.26 at h/n = 0.5 in two dimensions — not a detail
you can skip.

Move a tenth of the data into a tight cluster at distance 2.5 and the classical
ellipse does the opposite of its job: its variance grows by
1 + 0.1 * 2.5^2, its volume swells to 14.56, and it flags 49 percent of the
cluster as anomalous while covering 93 percent of the clean points. Masking: the
anomalies are inside the envelope built to find them. The corrected MCD with the
chi-square boundary flags 98 percent of the cluster at 90 percent clean
coverage. Shrinkage has the same mean and the same failure, because it fixes
conditioning, not contamination. The empirical boundary has its own trap: with a
10 percent budget and 10 percent contamination the quantile lands on the largest
clean distance, so `mcd_ellipse` over-covers at 0.976 and misses a quarter of the
cluster. The budget must exceed the contamination you actually have.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("mcd_ellipse", "min_cov_det", "shrinkage", "chi2_threshold")


# --------------------------------------------------------------------------- #
# Chi-square, from the incomplete gamma function
# --------------------------------------------------------------------------- #
def _gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series, then continued fraction."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_cdf(x: float, k: int) -> float:
    """P(chi2_k <= x)."""
    return _gammainc(0.5 * k, 0.5 * x)


def _chi2_ppf(q: float, k: int) -> float:
    """Quantile of chi2_k by bisection on the CDF; exact to machine precision."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    lo, hi = 0.0, 1.0
    while _chi2_cdf(hi, k) < q:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("chi-square quantile did not converge")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, k) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _ellipsoid_volume(t: float, covariance: np.ndarray) -> float:
    """Volume of { x : (x-mu)^T S^-1 (x-mu) <= t }, equation (3)."""
    p = covariance.shape[0]
    unit = math.pi ** (0.5 * p) / math.gamma(0.5 * p + 1.0)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return 0.0
    return unit * t ** (0.5 * p) * math.exp(0.5 * logdet)


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def _quadratic(X: np.ndarray, location: np.ndarray,
               covariance: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distances, one linear solve for the whole matrix."""
    centred = X - location
    solved = np.linalg.solve(covariance, centred.T)
    return np.einsum("ij,ij->j", centred.T, solved)


def _correction(alpha: float, p: int) -> float:
    """Factor that undoes the variance lost by keeping the tightest alpha of a
    Gaussian, equation (6).

    Two places need it: the MCD support is the tightest h/n of the sample, and
    the reweighted set is the part of the sample inside the 1 - contamination
    ellipse. Both are truncations, and a Gaussian truncated at the alpha
    quantile of chi2_p has covariance alpha / chi2_{p+2}(chi2_p(alpha)) times
    the full one.
    """
    return alpha / _chi2_cdf(_chi2_ppf(alpha, p), p + 2)


def _mcd(X: np.ndarray, h: int, n_init: int,
         seed: int) -> tuple:
    """Minimum covariance determinant by C-steps from random h-subsets.

    Returns (location, covariance, support, logdet, steps). The covariance is
    the maximum-likelihood covariance of the support, with no correction: the
    caller decides whether to apply (6).
    """
    n, p = X.shape
    rng = np.random.default_rng(seed)
    best = None
    total = 0
    for _ in range(n_init):
        index = np.sort(rng.choice(n, size=h, replace=False))
        singular = False
        for _ in range(100):
            sub = X[index]
            location = sub.mean(axis=0)
            covariance = np.atleast_2d(np.cov(sub, rowvar=False, bias=True))
            try:
                distance = _quadratic(X, location, covariance)
            except np.linalg.LinAlgError:
                singular = True
                break
            total += 1
            # the h points closest to the current ellipse are the next support
            updated = np.sort(np.argpartition(distance, h - 1)[:h])
            if np.array_equal(updated, index):
                break
            index = updated
        if singular:
            continue
        sub = X[index]
        location = sub.mean(axis=0)
        covariance = np.atleast_2d(np.cov(sub, rowvar=False, bias=True))
        sign, logdet = np.linalg.slogdet(covariance)
        if sign > 0 and (best is None or logdet < best[3]):
            support = np.zeros(n, dtype=bool)
            support[index] = True
            best = (location, covariance, support, float(logdet))
    if best is None:
        raise ValueError("all MCD starts produced a singular covariance")
    return (*best, total)


def _reweight(X: np.ndarray, location: np.ndarray, covariance: np.ndarray,
              p: int, contamination: float) -> tuple:
    """Keep the points inside the envelope and refit classically.

    The cut is the same 1 - contamination as the envelope itself, not the
    97.5 percent that some implementations hard-code. A looser cut lets points
    that sit between the two thresholds back into the covariance, and a cluster
    of them inflates the very ellipse that is supposed to exclude it.
    """
    distance = _quadratic(X, location, covariance)
    keep = distance <= _chi2_ppf(1.0 - contamination, p)
    if keep.sum() < p + 2:
        return location, covariance, np.ones(len(X), dtype=bool), 1.0
    sub = X[keep]
    new_location = sub.mean(axis=0)
    new_covariance = np.atleast_2d(np.cov(sub, rowvar=False, bias=True))
    scale = _correction(1.0 - contamination, p)
    return new_location, new_covariance * scale, keep, scale


def _ledoit_wolf(X: np.ndarray) -> tuple:
    """Ledoit-Wolf shrinkage of the sample covariance toward a scaled identity.

    lambda = b2 / d2 where d2 is the squared Frobenius distance from the target
    and b2 is the variance of the entries of the sample covariance, both divided
    by p so that they are mean squared entry errors.
    """
    n, p = X.shape
    location = X.mean(axis=0)
    centred = X - location
    sample = (centred.T @ centred) / n
    target = (float(np.trace(sample)) / p) * np.eye(p)
    d2 = float(np.sum((sample - target) ** 2)) / p
    # sum_k || x_k x_k^T - S ||_F^2 without materialising the n x p x p tensor
    squared = np.sum(centred * centred, axis=1)
    quadratic = np.einsum("ij,ij->i", centred @ sample, centred)
    frobenius = float(np.sum(sample * sample))
    b2 = float(np.sum(squared * squared - 2.0 * quadratic + frobenius))
    b2 /= (n * n) * p
    lam = 1.0 if d2 <= 0.0 else min(1.0, max(0.0, b2 / d2))
    covariance = (1.0 - lam) * sample + lam * target
    return location, covariance, lam


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class EllipticEnvelope:
    """An ellipsoidal inlier region around a robust centre.

    Parameters
    ----------
    variant : {"mcd_ellipse", "min_cov_det", "shrinkage", "chi2_threshold"}
        Which estimator and which boundary to use.
    support_fraction : float or None
        Fraction h/n of the sample the MCD covariance is computed from. `None`
        uses (n + p + 1) / (2n), capped at 1, which keeps h > p and leaves the
        rest of the sample for the reweighting step.
    contamination : float
        Fraction of the training data allowed outside the envelope. Sets the
        empirical quantile, or the chi-square level for the two variants that
        use it.
    n_init : int
        Random h-subsets for the C-steps. Each start converges to a local
        minimum of the determinant; more starts mean a better chance at the
        global one.
    seed : int
        Seed for the subset draws. The estimator is deterministic given it.

    Attributes
    ----------
    location_ : ndarray of shape (p,)
        Centre of the ellipsoid.
    covariance_ : ndarray of shape (p, p)
        Shape of the ellipsoid.
    distances_ : ndarray of shape (n,)
        Squared Mahalanobis distance of every training point, equation (1).
    threshold_ : float
        The squared distance at which a point leaves the envelope.
    support_ : ndarray of bool
        Points the MCD estimator was computed from, after reweighting where the
        variant reweights.
    scale_ : float
        The consistency factor applied to the covariance, 1.0 for `shrinkage`.
    axes_ : ndarray of shape (p,)
        Semi-axis lengths of the envelope, sqrt(eigenvalue * threshold).
    volume_ : float
        Volume of the envelope, equation (3).
    lam_ : float
        `shrinkage` only: the weight given to the target.
    n_steps_ : int
        C-steps performed, summed over the random starts.
    """

    def __init__(self, *, variant: str = "mcd_ellipse",
                 support_fraction: float | None = None,
                 contamination: float = 0.1, n_init: int = 10,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must lie in (0, 1)")
        if support_fraction is not None and not 0.0 < support_fraction <= 1.0:
            raise ValueError("support_fraction must lie in (0, 1]")
        if n_init < 1:
            raise ValueError("n_init must be positive")
        self.variant = variant
        self.support_fraction = support_fraction
        self.contamination = float(contamination)
        self.n_init = int(n_init)
        self.seed = int(seed)

    def fit(self, X: np.ndarray) -> EllipticEnvelope:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        n, p = X.shape
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        if n < p + 2:
            raise ValueError(f"need at least p + 2 = {p + 2} rows, got {n}")
        if p < 1:
            raise ValueError("X must have at least one column")
        self.n_features_ = p
        self.n_samples_ = n

        if self.variant == "shrinkage":
            location, covariance, lam = _ledoit_wolf(X)
            self.location_ = location
            self.covariance_ = covariance
            self.support_ = np.ones(n, dtype=bool)
            self.scale_ = 1.0
            self.lam_ = lam
            self.n_steps_ = 0
        else:
            h = self._support_size()
            if h <= p:
                raise ValueError(
                    f"support of {h} points cannot estimate a {p}x{p} "
                    f"covariance; raise support_fraction or collect more data")
            (location, covariance, support,
             self.logdet_, self.n_steps_) = _mcd(X, h, self.n_init, self.seed)
            if self.variant == "min_cov_det":
                self.location_ = location
                self.covariance_ = covariance
                self.support_ = support
                self.scale_ = 1.0
            else:
                scale = _correction(h / n, p)
                covariance = covariance * scale
                if self.variant == "mcd_ellipse":
                    location, covariance, support, scale = _reweight(
                        X, location, covariance, p, self.contamination)
                self.location_ = location
                self.covariance_ = covariance
                self.support_ = support
                self.scale_ = scale
            self.lam_ = 0.0

        self.distances_ = _quadratic(X, self.location_, self.covariance_)
        self.threshold_ = self._threshold()
        eigenvalues = np.linalg.eigvalsh(self.covariance_)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("covariance is singular; use the shrinkage variant")
        self.axes_ = np.sqrt(eigenvalues * self.threshold_)
        self.volume_ = _ellipsoid_volume(self.threshold_, self.covariance_)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance; larger means further outside."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features_:
            raise ValueError("X has the wrong number of columns")
        return _quadratic(X, self.location_, self.covariance_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """True where the point is inside the envelope."""
        return self.score_samples(X) <= self.threshold_

    def _support_size(self) -> int:
        n, p = self.n_samples_, self.n_features_
        fraction = self.support_fraction
        if fraction is None:
            fraction = min(1.0, (n + p + 1.0) / (2.0 * n))
        return int(np.floor(fraction * n))

    def _threshold(self) -> float:
        # `mcd_ellipse` is the distribution-free one: its size is the empirical
        # quantile of the training distances. The other three are Gaussian
        # models, so their size is the chi-square quantile -- including
        # `shrinkage`, whose in-sample distances are far too small when p is a
        # large fraction of n.
        if self.variant == "mcd_ellipse":
            return float(np.quantile(self.distances_, 1.0 - self.contamination))
        return float(_chi2_ppf(1.0 - self.contamination, self.n_features_))


# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _gaussian(rng: np.random.Generator, n: int,
              covariance: np.ndarray) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(len(covariance)), covariance, size=n)


def _classical(X: np.ndarray, contamination: float) -> tuple:
    """The textbook ellipse: sample mean, sample covariance, chi-square size."""
    location = X.mean(axis=0)
    covariance = np.atleast_2d(np.cov(X, rowvar=False, bias=True))
    return location, covariance, _chi2_ppf(1.0 - contamination, X.shape[1])


def _demo_geometry() -> None:
    """On clean Gaussian data every calibrated variant is right, and the volume
    formula is exact."""
    rng = np.random.default_rng(0)
    covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = _gaussian(rng, 400, covariance)
    fresh = _gaussian(rng, 4000, covariance)

    # the chi-square quantile has two closed forms, and the volume of a
    # two-dimensional ellipse is pi t sqrt(det)
    assert abs(_chi2_ppf(0.95, 2) + 2.0 * np.log(0.05)) < 1e-9
    assert abs(_chi2_ppf(0.95, 1) - 1.959963984540054 ** 2) < 1e-9
    t = _chi2_ppf(0.9, 2)
    assert abs(_ellipsoid_volume(t, covariance)
               - np.pi * t * np.sqrt(np.linalg.det(covariance))) < 1e-12

    rows = {}
    for variant in VARIANTS:
        model = EllipticEnvelope(variant=variant).fit(X)
        rows[variant] = (float(np.linalg.norm(model.location_)),
                         model.predict(fresh).mean(), model.threshold_,
                         model.volume_, model.scale_, model.n_steps_)
    for variant, (_error, coverage, _, _, _, _) in rows.items():
        if variant != "min_cov_det":
            assert abs(coverage - 0.9) < 0.05, variant

    print("two-dimensional Gaussian, n = 400, correlation 0.6, nominal "
          "coverage 0.90:")
    print("  variant          centre error   fresh coverage   threshold   "
          "volume   scale   steps")
    for variant, (error, coverage, threshold, volume, scale, steps) in rows.items():
        print(f"  {variant:<15s} {error:12.3f} {coverage:16.3f} "
              f"{threshold:11.3f} {volume:8.3f} {scale:7.3f} {steps:7d}")
    print(f"  The analytic volume of the 90 percent ellipse of the true "
          f"covariance is {np.pi * t * np.sqrt(np.linalg.det(covariance)):.3f}, "
          f"which the corrected estimators reproduce. `min_cov_det` is a third "
          f"of that and its coverage is nowhere near nominal, for the reason in "
          f"the next block.")


def _demo_contamination() -> None:
    """A tenth of the points moved away: the classical ellipse follows them."""
    rng = np.random.default_rng(1)
    covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    clean = _gaussian(rng, 360, covariance)
    bad = rng.normal(loc=[2.5, 0.0], scale=0.4, size=(40, 2))
    X = np.vstack([clean, bad])
    fresh_clean = _gaussian(rng, 4000, covariance)
    fresh_bad = rng.normal(loc=[2.5, 0.0], scale=0.4, size=(2000, 2))

    rows = {}
    location, cov, threshold = _classical(X, 0.1)
    rows["classical"] = (location, cov, threshold)
    for variant in VARIANTS:
        model = EllipticEnvelope(variant=variant).fit(X)
        rows[variant] = (model.location_, model.covariance_, model.threshold_)

    print("360 points from a correlated Gaussian plus 40 points in a tight "
          "cluster at distance 2.5, budget 10 percent:")
    print("  method           centre error   volume   clean covered   "
          "outliers flagged")
    table = {}
    for name, (loc, cov, threshold) in rows.items():
        centre_error = float(np.linalg.norm(loc))
        covered = float(np.mean(_quadratic(fresh_clean, loc, cov) <= threshold))
        caught = float(np.mean(_quadratic(fresh_bad, loc, cov) > threshold))
        volume = _ellipsoid_volume(threshold, cov)
        table[name] = (centre_error, volume, covered, caught)
        print(f"  {name:<16s} {centre_error:12.3f} {volume:8.3f} "
              f"{covered:15.3f} {caught:17.3f}")

    roomy = EllipticEnvelope(variant="mcd_ellipse", contamination=0.2).fit(X)
    roomy_cover = float(np.mean(roomy.score_samples(fresh_clean)
                                <= roomy.threshold_))
    roomy_caught = float(np.mean(roomy.score_samples(fresh_bad)
                                 > roomy.threshold_))

    assert table["classical"][0] > 0.3
    assert table["classical"][3] < 0.55
    assert table["shrinkage"][3] < 0.55
    assert table["min_cov_det"][2] < 0.6
    assert table["chi2_threshold"][3] > 0.9
    assert 0.85 < table["chi2_threshold"][2] < 0.95
    assert 0.85 < roomy_cover < 0.95
    assert roomy_caught > 0.9

    print("  The classical covariance is inflated by the cluster -- the variance "
          "grows by 1 + 0.1 * 2.5^2 -- so its envelope swells and swallows the "
          "very points that inflated it. Masking: the anomalies are inside the "
          "envelope built to find them. Shrinkage has the same mean and the same "
          "problem; it fixes conditioning, not contamination.")
    print(f"  The empirical boundary needs headroom. With a 10 percent budget "
          f"and 10 percent contamination the quantile lands on the largest "
          f"clean distance, which is why `mcd_ellipse` over-covers at "
          f"{table['mcd_ellipse'][2]:.3f} and misses a quarter of the cluster. "
          f"At a 20 percent budget the same fit covers {roomy_cover:.3f} and "
          f"catches {roomy_caught:.3f}: the budget must exceed the "
          f"contamination you actually have.")


def _demo_consistency() -> None:
    """The raw determinant minimiser is biased low, and the bias is not small."""
    print("nominal coverage 0.90, fresh data, h/n forced by support_fraction:")
    print("  p   h/n   raw coverage   corrected coverage   raw volume   "
          "corrected volume")
    for p in (2, 8):
        rng = np.random.default_rng(2 + p)
        covariance = np.eye(p)
        X = _gaussian(rng, 400, covariance)
        fresh = _gaussian(rng, 4000, covariance)
        for fraction in (0.5, 0.75, 0.9):
            raw = EllipticEnvelope(variant="min_cov_det",
                                   support_fraction=fraction).fit(X)
            fixed = EllipticEnvelope(variant="chi2_threshold",
                                     support_fraction=fraction).fit(X)
            raw_cov = raw.predict(fresh).mean()
            fixed_cov = fixed.predict(fresh).mean()
            assert raw_cov < fixed_cov
            print(f"  {p}  {fraction:4.2f}   {raw_cov:12.3f} "
                  f"{fixed_cov:20.3f} {raw.volume_:12.3f} "
                  f"{fixed.volume_:16.3f}")
    print("  The raw covariance of the h most concentrated points is too small "
          "by the factor (6), which is 3.26 at h/n = 0.5 in two dimensions and "
          "1.30 at h/n = 0.75 in eight. Skipping the correction costs most of "
          "the coverage, and the loss grows with the dimension because the "
          "volume carries the factor to the power p/2.")


def _demo_boundary() -> None:
    """The chi-square boundary is a Gaussian assumption, not a fact."""
    rng = np.random.default_rng(9)
    p = 3
    covariance = np.eye(p)

    def student(n: int, df: float) -> np.ndarray:
        z = _gaussian(rng, n, covariance)
        return z / np.sqrt(rng.chisquare(df, size=(n, 1)) / df)

    print("nominal coverage 0.90, n = 600, fresh sample of 4000:")
    print("  data       chi-square boundary   empirical boundary   "
          "chi-square in-sample")
    for label, make in (("gaussian", lambda n: _gaussian(rng, n, covariance)),
                        ("t3", lambda n: student(n, 3.0))):
        X = make(600)
        fresh = make(4000)
        theory = EllipticEnvelope(variant="chi2_threshold").fit(X)
        empirical = EllipticEnvelope(variant="mcd_ellipse").fit(X)
        print(f"  {label:<10s} {theory.predict(fresh).mean():19.3f} "
              f"{empirical.predict(fresh).mean():21.3f} "
              f"{theory.predict(X).mean():21.3f}")
        if label == "gaussian":
            assert abs(theory.predict(fresh).mean() - 0.9) < 0.04
        else:
            assert theory.predict(fresh).mean() < 0.85
            assert abs(empirical.predict(fresh).mean() - 0.9) < 0.05
    print("  On Gaussian data the two boundaries agree, because they are the "
          "same boundary. On t3 the chi-square quantile is the wrong law and "
          "the envelope covers less than it promises; the empirical quantile "
          "does not care what the law is.")


def _demo_high_dimension() -> None:
    """When p approaches n, the robust estimator has no support left."""
    rng = np.random.default_rng(11)
    p = 40
    refusals = []
    for call in (lambda: EllipticEnvelope(variant="mcd_ellipse").fit(
                     _gaussian(rng, 25, np.eye(p))),
                 lambda: EllipticEnvelope(support_fraction=0.85).fit(
                     _gaussian(rng, 45, np.eye(p)))):
        try:
            call()
        except ValueError as error:
            refusals.append(str(error))
        else:
            raise AssertionError("MCD should refuse a support it cannot use")

    n = 50
    X = _gaussian(rng, n, np.eye(p))
    fresh = _gaussian(rng, 2000, np.eye(p))
    sample = np.atleast_2d(np.cov(X, rowvar=False, bias=True))
    shrunk = EllipticEnvelope(variant="shrinkage").fit(X)
    mcd = EllipticEnvelope(variant="mcd_ellipse").fit(X)
    cond_sample = np.linalg.cond(sample)
    cond_shrunk = np.linalg.cond(shrunk.covariance_)
    cond_mcd = np.linalg.cond(mcd.covariance_)

    print(f"p = {p}, n = 25: {refusals[0]}")
    print(f"p = {p}, n = 45, support 85 percent: {refusals[1]}")
    distance = _quadratic(fresh, X.mean(axis=0), sample)
    sample_cover = float(np.mean(distance <= _chi2_ppf(0.9, p)))
    # what the empirical quantile would have done instead of the chi-square one
    training = _quadratic(X, shrunk.location_, shrunk.covariance_)
    naive = float(np.mean(_quadratic(fresh, shrunk.location_, shrunk.covariance_)
                          <= np.quantile(training, 0.9)))
    print(f"p = {p}, n = {n}:")
    print("  estimator    condition number   fresh coverage   shrinkage weight")
    print(f"  sample       {cond_sample:16.1f} {sample_cover:16.3f}   {'-':>16s}")
    print(f"  mcd          {cond_mcd:16.1f} {mcd.predict(fresh).mean():16.3f}"
          f"   {'-':>16s}")
    print(f"  shrinkage    {cond_shrunk:16.1f} "
          f"{shrunk.predict(fresh).mean():16.3f} {shrunk.lam_:16.3f}")
    assert shrunk.lam_ > 0.5
    assert cond_shrunk < 0.01 * cond_sample
    assert 0.7 < shrunk.predict(fresh).mean() < 0.9
    assert naive < shrunk.predict(fresh).mean() - 0.15
    print(f"  With the in-sample empirical quantile instead of the chi-square "
          f"one, the same shrinkage fit covers {naive:.3f} of fresh points: "
          f"in-sample distances are systematically too small when p/n = 0.8.")
    print("  The sample covariance is invertible but barely: its smallest "
          "eigenvalue is noise, so the ellipse is a sliver in some directions "
          "and unbounded in others. Shrinkage gives up the directions the data "
          "cannot resolve and recovers most of the coverage -- not all of it, "
          "because pulling every direction toward a common variance is itself "
          "a bias, and at p/n = 0.8 it is a large one.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    rng = np.random.default_rng(3)
    X = _gaussian(rng, 40, np.eye(3))
    calls = [
        lambda: EllipticEnvelope(variant="nope"),
        lambda: EllipticEnvelope(contamination=0.0),
        lambda: EllipticEnvelope(contamination=1.0),
        lambda: EllipticEnvelope(support_fraction=0.0),
        lambda: EllipticEnvelope(support_fraction=1.5),
        lambda: EllipticEnvelope(n_init=0),
        lambda: EllipticEnvelope().fit(X[:, 0]),
        lambda: EllipticEnvelope().fit(np.r_[X, np.full((1, 3), np.nan)]),
        lambda: EllipticEnvelope().fit(X[:4]),
        lambda: EllipticEnvelope(support_fraction=0.05).fit(X),
        lambda: EllipticEnvelope().fit(X[:, :0]),
        lambda: _chi2_ppf(0.0, 3),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, "
          f"out-of-range fractions, a one-dimensional input, a non-finite "
          f"value, too few rows, a support smaller than the dimension, no "
          f"columns, and a quantile outside (0, 1).")


def main() -> None:
    _demo_geometry()
    _demo_contamination()
    _demo_consistency()
    _demo_boundary()
    _demo_high_dimension()
    _demo_broken()
    print("elliptic envelope demo passed")


if __name__ == "__main__":
    main()
