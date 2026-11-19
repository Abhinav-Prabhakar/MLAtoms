r"""Mahalanobis distance and robust covariance — the distance that knows the shape.

Chapter:   09-density-and-anomaly
Companion: mahalanobis.md
Tier 2 · 4 variants: hotelling_t2, mcd, ogk, shrinkage

    d^2(x) = (x - mu)' Sigma^{-1} (x - mu)                                 (1)
    T^2(x) = n/(n+1) * d^2(x)                                              (2)
    chi^2 threshold:  d^2 > chi^2_p(1 - alpha)                             (3)
    MCD:    arg min_{|H| = h} det( cov(X_H) )                              (4)
    OGK:    s_jk = (scale(x_j + x_k)^2 - scale(x_j - x_k)^2) / 4           (5)
    LW:     Sigma_hat = lambda * (tr(S)/p) I + (1 - lambda) S              (6)

Euclidean distance counts every direction the same. Mahalanobis distance measures
how many standard deviations out a point is *along the direction it lies in*: it
subtracts the mean, rotates to the covariance's eigenvectors, divides by the
standard deviation on each axis, and measures the length there. For a Gaussian it
is the right thing by construction — (1) is the exponent of the normal density —
and its square is chi-square distributed with $p$ degrees of freedom, which is
what makes (3) a calibrated threshold rather than an arbitrary number.

The whole difficulty is $\Sigma$. Plugging in the sample covariance gives the
classical estimate, and it has breakdown point zero: a single point moved far
enough shifts $\mu$ and inflates $\Sigma$ in its own direction until its own
distance is finite again. This is *masking*, and it is not a numerical accident —
it is the maximum-likelihood estimate doing exactly what it is asked, since a
Gaussian that has widened to cover the outlier now explains it. The variants are
four answers to estimating $\Sigma$ when some of the data are not from the model:

- `hotelling_t2` is the classical estimate with the exact small-sample correction
  (2), and the $F$ threshold that goes with it.
- `mcd` is the minimum covariance determinant: the ellipsoid that covers half the
  data and has the smallest volume, refitted on the points it covers.
- `ogk` builds $\Sigma$ from pairwise robust scales (5), which does not need a
  subset search and survives contamination that is not a tight cluster.
- `shrinkage` does not aim at robustness at all; it fixes the other failure —
  singularity — by pulling $\Sigma$ toward a scaled identity (6).

The demo measures both failures. With 20% of the points packed into a tight
sphere far from the blob, the classical fit's AUC falls to 0.34 -- the outliers
score *below* the data -- while OGK stays at 1.00. With $n = 30$ and $p = 60$ the
classical covariance is singular and `hotelling_t2` refuses to fit; shrinkage
returns a positive-definite matrix with condition number 5.1.

The small-sample law in (2) is worth spelling out, because it is the difference
between a statistic and a threshold. If $x$ is a *new* observation and $\bar x, S$
come from $n$ others, then $x - \bar x$ has covariance $\Sigma(1 + 1/n)$, so the
plain distance is inflated by that factor; dividing it out and scaling by the
degrees of freedom gives (2), which follows $\frac{p(n-1)}{n-p}F_{p,n-p}$
exactly. The demo checks the calibration by replication: at a nominal 5%, the
exact $F$ threshold flags 4.4% of held-out points, while the asymptotic
chi-square threshold on the uncorrected distance flags 55.1% — a difference that
matters at $n = 15, p = 8$ and has shrunk to 5.0% against 6.8% by $n = 200$.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("hotelling_t2", "mcd", "ogk", "shrinkage")
TINY = 1e-12


# --------------------------------------------------------------------------- #
# Distributions, without scipy
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Inverse standard normal by bisection on math.erf."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series or continued fraction."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for i in range(1, 1000):
            term *= x / (a + i)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1.0 / TINY
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = TINY if abs(d) < TINY else d
        c = b + an / c
        c = TINY if abs(c) < TINY else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_ppf(q: float, df: int) -> float:
    """Chi-square quantile: P(df/2, x/2) = q, by bisection."""
    lo, hi = 0.0, max(1.0, float(df))
    while _gammainc(df / 2.0, hi / 2.0) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gammainc(df / 2.0, mid / 2.0) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = a * math.log(x) + b * math.log1p(-x) - math.lgamma(a) - math.lgamma(b) \
        + math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(front) * _betacf(b, a, 1.0 - x) / b


def _f_cdf(x: float, p: int, q: int) -> float:
    if x <= 0.0:
        return 0.0
    return _betainc(p / 2.0, q / 2.0, p * x / (p * x + q))


def _f_ppf(q: float, p: int, df2: int) -> float:
    """F quantile by bisection on the incomplete beta."""
    lo, hi = 0.0, 1.0
    while _f_cdf(hi, p, df2) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _f_cdf(mid, p, df2) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Robust pieces
# --------------------------------------------------------------------------- #
def _qn_constant(n: int) -> float:
    """Consistency factor for Q_n, derived from the normal plotting position.

    The pairwise absolute differences are half-normal-ish: for Gaussian x the
    difference of two draws is N(0, 2 sigma^2), so |x_i - x_j| <= t with
    probability 2 Phi(t / (sqrt(2) sigma)) - 1. Q_n takes the k-th smallest of
    the C(n, 2) differences with k = C(h, 2) and h = floor(n/2) + 1; placing k
    at its expected rank and inverting gives the constant, so no table is needed.
    """
    h = n // 2 + 1
    k = h * (h - 1) // 2
    p = k / (n * (n - 1) / 2.0 + 1.0)
    return 1.0 / (math.sqrt(2.0) * _norm_ppf(0.5 * (1.0 + p)))


def _qn(x: np.ndarray) -> float:
    """Rousseeuw-Croux Q_n: a 50%-breakdown scale, O(n^2) and deterministic."""
    n = len(x)
    if n < 2:
        return 0.0
    diffs = np.abs(x[:, None] - x[None, :])[np.triu_indices(n, 1)]
    h = n // 2 + 1
    k = h * (h - 1) // 2
    return float(_qn_constant(n) * np.partition(diffs, k - 1)[k - 1])


def _classical(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and unbiased covariance."""
    mean = X.mean(axis=0)
    centred = X - mean
    return mean, centred.T @ centred / (len(X) - 1)


def _sq_mahal(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Equation (1), via a solve rather than an explicit inverse."""
    diff = X - mean
    solved = np.linalg.solve(cov, diff.T).T
    return np.einsum("ij,ij->i", diff, solved)


def _nearest_pd(S: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues; a pairwise covariance need not be PSD."""
    values, vectors = np.linalg.eigh(0.5 * (S + S.T))
    top = max(float(values.max()), TINY)
    values = np.maximum(values, floor * top)
    return (vectors * values) @ vectors.T


def _mcd(X: np.ndarray, h: int, n_trials: int, n_csteps: int,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fast-MCD: random h-subsets, concentration steps, keep the smallest det."""
    n = len(X)
    best_logdet = np.inf
    best = None
    for _ in range(n_trials):
        idx = rng.choice(n, h, replace=False)
        for _ in range(n_csteps + 1):
            mean, cov = _classical(X[idx])
            d2 = _sq_mahal(X, mean, cov)
            idx = np.argpartition(d2, h - 1)[:h]
        sign, logdet = np.linalg.slogdet(cov)
        if sign > 0 and logdet < best_logdet:
            best_logdet = logdet
            best = (mean, cov)
    if best is None:
        raise ValueError("every MCD subset was singular; reduce the dimension")
    return best


def _ogk(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonalised Gnanadesikan-Kettenring covariance, equation (5)."""
    n, p = X.shape
    S = np.empty((p, p))
    for j in range(p):
        S[j, j] = _qn(X[:, j]) ** 2
        for k in range(j + 1, p):
            plus = _qn(X[:, j] + X[:, k]) ** 2
            minus = _qn(X[:, j] - X[:, k]) ** 2
            S[j, k] = S[k, j] = 0.25 * (plus - minus)
    S = _nearest_pd(S)
    values, vectors = np.linalg.eigh(S)
    root = (vectors * np.sqrt(np.maximum(values, TINY))) @ vectors.T
    inverse_root = (vectors * (1.0 / np.sqrt(np.maximum(values, TINY)))) @ vectors.T
    # Each column of X @ inverse_root should have unit robust scale; the
    # correction V = diag(tau_j^2) is what makes OGK work when the pairwise
    # matrix is a bad estimate of the shape but a good one of the axes.
    tau = np.array([_qn((X @ inverse_root)[:, j]) for j in range(p)])
    return np.median(X, axis=0), root @ np.diag(tau ** 2) @ root


def _shrink(X: np.ndarray, fixed: float | None = None
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity, equation (6)."""
    n, p = X.shape
    mean = X.mean(axis=0)
    centred = X - mean
    S = centred.T @ centred / n
    target = np.trace(S) / p
    if fixed is not None:
        return mean, (1.0 - fixed) * S + fixed * target * np.eye(p), float(fixed)
    d2 = float(np.sum((S - target * np.eye(p)) ** 2) / p)
    if d2 <= 0.0:
        return mean, S, 0.0
    outer = centred[:, :, None] * centred[:, None, :]
    # b^2 is the total variance of the sample covariance entries: sum_ij
    # Var(s_ij) / p. Var(s_ij) = Var(x_i x_j)/n, and the outer product gives
    # the per-observation term, so the double division by n is not a typo.
    b2 = float(np.sum((outer - S) ** 2) / (n * n * p))
    lam = min(max(b2 / d2, 0.0), 1.0)
    return mean, (1.0 - lam) * S + lam * target * np.eye(p), lam


class MahalanobisOutlier:
    """Mahalanobis distances with four covariance estimators.

    Parameters
    ----------
    variant : {"hotelling_t2", "mcd", "ogk", "shrinkage"}
    contamination : float or None
        Quantile used by `predict`, as a flag budget. `None` uses the
        distributional threshold instead: the exact F law for `hotelling_t2`,
        the chi-square quantile for the others.
    alpha : float
        Nominal tail probability for the distributional threshold, used only
        when `contamination` is None.
    support_fraction : float
        MCD subset size as a fraction of n, in (0.5, 1]. The breakdown point of
        the raw estimate is `1 - support_fraction`.
    n_trials, n_csteps : int
        Random starts and concentration steps for the MCD subset search.
    shrinkage : float or None
        Fix lambda instead of estimating it. Mostly for experiments.
    seed : int
        Seed for the MCD start points; the estimate is otherwise deterministic.

    Attributes
    ----------
    mean_, cov_ : ndarray
        Fitted location and scatter, shape (p,) and (p, p).
    scores_ : ndarray
        Training score per row: T^2 for `hotelling_t2`, d^2 otherwise.
    threshold_ : float
        Value above which `predict` returns -1.
    lambda_ : float or None
        Shrinkage intensity, for the `shrinkage` variant only.
    n_, n_features_ : int
    """

    def __init__(self, *, variant: str = "hotelling_t2",
                 contamination: float | None = 0.1, alpha: float = 0.05,
                 support_fraction: float = 0.75, n_trials: int = 10,
                 n_csteps: int = 2, shrinkage: float | None = None,
                 seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if contamination is not None and not 0.0 < contamination < 1.0:
            raise ValueError("contamination must lie in (0, 1) or be None")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.5 < support_fraction <= 1.0:
            raise ValueError("support_fraction must lie in (0.5, 1]")
        if n_trials < 1 or n_csteps < 0:
            raise ValueError("n_trials must be positive and n_csteps non-negative")
        if shrinkage is not None and not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1] or be None")
        self.variant = variant
        self.contamination = contamination
        self.alpha = float(alpha)
        self.support_fraction = float(support_fraction)
        self.n_trials = int(n_trials)
        self.n_csteps = int(n_csteps)
        self.shrinkage = shrinkage
        self.seed = int(seed)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, X: np.ndarray) -> MahalanobisOutlier:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")
        n, p = X.shape
        if n < 3 or p < 1:
            raise ValueError("need at least three points")
        if self.variant == "hotelling_t2" and n <= p:
            raise ValueError("the sample covariance is singular when n <= p; "
                             "use the shrinkage or mcd variant")
        self.n_, self.n_features_ = n, p
        rng = np.random.default_rng(self.seed)
        self.lambda_ = None

        if self.variant == "hotelling_t2":
            mean, cov = _classical(X)
            if np.linalg.matrix_rank(cov) < p:
                raise ValueError("the sample covariance is singular; the "
                                 "columns are collinear or n <= p -- use the "
                                 "shrinkage or mcd variant")
        elif self.variant == "shrinkage":
            mean, cov, lam = _shrink(X, self.shrinkage)
            self.lambda_ = lam
        elif self.variant == "mcd":
            h = max(2, int(math.ceil(self.support_fraction * n)))
            mean, cov = _mcd(X, h, self.n_trials, self.n_csteps, rng)
            mean, cov = self._reweight(X, mean, cov)
        else:  # ogk
            mean, cov = _ogk(X)
            mean, cov = self._reweight(X, mean, cov)

        self.mean_ = mean
        self.cov_ = cov if self.variant == "hotelling_t2" else _nearest_pd(cov)
        self.scores_ = self.score_samples(X)
        self.threshold_ = self._threshold()
        return self

    def _reweight(self, X: np.ndarray, mean: np.ndarray,
                  cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Refit on the points inside the 97.5% ellipsoid; restores efficiency."""
        cutoff = _chi2_ppf(0.975, X.shape[1])
        keep = _sq_mahal(X, mean, cov) <= cutoff
        if keep.sum() <= X.shape[1] + 1:
            return mean, cov
        return _classical(X[keep])

    def _threshold(self) -> float:
        p, n = self.n_features_, self.n_
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        if self.variant == "hotelling_t2":
            return p * (n - 1) / (n - p) * _f_ppf(1.0 - self.alpha, p, n - p)
        return _chi2_ppf(1.0 - self.alpha, p)

    # -- scoring ---------------------------------------------------------- #
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance, scaled by (2) for the Hotelling variant."""
        X = self._check(X)
        d2 = _sq_mahal(X, self.mean_, self.cov_)
        if self.variant == "hotelling_t2":
            return d2 * self.n_ / (self.n_ + 1.0)
        return d2

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def _check(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("fit before scoring")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (m, {self.n_features_})")
        return X


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based area under the ROC curve, ties averaged."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    positive = labels.astype(bool)
    n1 = int(positive.sum())
    n0 = len(labels) - n1
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _calibration(n: int, p: int, reps: int = 200, n_test: int = 300) -> tuple:
    """Average flag rate of three nominal-5% rules over independent samples."""
    rng = np.random.default_rng(11)
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T + p * np.eye(p)
    exact = p * (n - 1) / (n - p) * _f_ppf(0.95, p, n - p)
    chi = _chi2_ppf(0.95, p)
    rates = np.zeros(3)
    for _ in range(reps):
        train = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        mean, cov = _classical(train)
        test = rng.multivariate_normal(np.zeros(p), Sigma, size=n_test)
        d2 = _sq_mahal(test, mean, cov)
        t2 = n / (n + 1.0) * d2
        rates += [np.mean(t2 > exact), np.mean(t2 > chi), np.mean(d2 > chi)]
    return tuple(rates / reps)


def _relative(cov: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(cov - truth) / np.linalg.norm(truth))


def _demo_definition() -> None:
    """The identity, the invariances, and the calibration of the threshold."""
    rng = np.random.default_rng(0)
    p, n = 3, 400
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T + 0.5 * np.eye(p)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

    model = MahalanobisOutlier(variant="hotelling_t2", contamination=None).fit(X)
    d2 = _sq_mahal(X, model.mean_, model.cov_)
    identity = p * (n - 1) / n
    assert abs(float(d2.mean()) - identity) < 1e-10

    # An invertible affine map rebuilds the same distances: (1) is invariant
    # because both the mean and the covariance transform with the data.
    B = rng.normal(size=(p, p)) + 2.0 * np.eye(p)
    moved = X @ B.T + np.array([1.0, -2.0, 3.0])
    other = MahalanobisOutlier(variant="hotelling_t2", contamination=None).fit(moved)
    drift = float(np.abs(other.score_samples(moved) - model.score_samples(X)).max())
    assert drift < 1e-10

    small = _calibration(15, 8)
    large = _calibration(200, 8)
    assert abs(large[0] - 0.05) < 0.02
    assert small[2] > 5.0 * small[0]
    print(f"definition: on {n} points in {p} dimensions the mean squared "
          f"distance is {d2.mean():.4f}; the identity sum_i d_i^2 = (n-1)p "
          f"forces it to be exactly p(n-1)/n = {identity:.4f}, which checks the "
          f"covariance convention rather than the data. A random invertible "
          f"linear map and a translation leave every score unchanged to "
          f"{drift:.0e}. Calibration at a nominal 5%, over 200 independent "
          f"samples: at n = 15, p = 8 the exact F threshold of (2) flags "
          f"{small[0]:.1%} of held-out points, chi-square on the corrected "
          f"statistic flags {small[1]:.1%}, and chi-square on the raw distance "
          f"flags {small[2]:.1%}. At n = 200 the three are {large[0]:.1%}, "
          f"{large[1]:.1%} and {large[2]:.1%}: the correction is what makes a "
          f"threshold mean something on a small sample, and it stops mattering "
          f"as the sample grows.")


def _demo_masking() -> None:
    """A compact far cluster, scored by four covariances."""
    rng = np.random.default_rng(5)
    p = 6
    blob = np.eye(p) * 0.7 + 0.3
    good = rng.multivariate_normal(np.zeros(p), blob, size=400)
    direction = np.zeros(p)
    direction[0] = 1.0
    bad = direction * 8.0 + rng.normal(scale=0.3, size=(100, p))
    X = np.vstack([good, bad])
    labels = np.zeros(len(X), dtype=bool)
    labels[-100:] = True

    rows = {}
    for variant in VARIANTS:
        model = MahalanobisOutlier(variant=variant, contamination=0.1).fit(X)
        scores = model.score_samples(X)
        flags = model.predict(X)
        rows[variant] = (_auc(scores, labels), float(scores[labels].mean()),
                         float(scores[~labels].mean()),
                         float(np.mean(flags[~labels] == -1)),
                         float(np.mean(flags[labels] == -1)),
                         model.lambda_)
    assert rows["hotelling_t2"][0] < 0.4
    assert rows["mcd"][0] < 0.25
    assert rows["ogk"][0] > 0.99
    assert rows["shrinkage"][0] < 0.4

    print("masking: 400 points from a 6-dimensional Gaussian and 100 packed "
          "into a ball of radius 0.3 at distance 8 along one axis:")
    for variant, (area, outlier, normal, fn, fo, _) in rows.items():
        print(f"  {variant:13s} AUC {area:.3f}  outlier {outlier:6.2f}  "
              f"normal {normal:5.2f}  flags {fo:4.0%} of outliers, {fn:4.0%} of "
              f"normals")
    print(f"  The classical fit is not merely fooled, it inverts: outliers score "
          f"{rows['hotelling_t2'][1]:.2f} and normals {rows['hotelling_t2'][2]:.2f}, "
          f"so `predict` flags {rows['hotelling_t2'][3]:.0%} of the good points "
          f"and none of the bad ones. The compact cluster inflates Sigma along "
          f"its own axis until its distance is finite, and in the other "
          f"{p - 1} directions it adds nothing while a normal point still pays "
          f"p - 1, so the outliers come out *below* the data. MCD does no "
          f"better (AUC {rows['mcd'][0]:.3f}) for a different reason: its "
          f"criterion (4) is volume, and a compact cluster has less of it than "
          f"the good data, so the determinant sends it to the wrong subset. OGK "
          f"separates the two populations at AUC {rows['ogk'][0]:.3f}, because "
          f"a pairwise scale never has to decide which half of the data is "
          f"real; its {rows['ogk'][4]:.0%} flag rate is the 10% budget against "
          f"20% contamination. Shrinkage is no help: lambda is "
          f"{rows['shrinkage'][5]:.3f} at n = 500, p = 6, so it is the sample "
          f"covariance with a rounding error.")


def _demo_estimators() -> None:
    """Recovery of a known covariance under gross errors, and under n < p."""
    rng = np.random.default_rng(2)
    p = 6
    B = rng.normal(size=(p, p))
    Sigma = B @ B.T * 0.3 + np.eye(p)
    root = np.linalg.cholesky(Sigma)

    rows = {}
    for fraction in (0.0, 0.05, 0.1, 0.2, 0.3):
        clean = rng.standard_normal((600, p)) @ root.T
        gross = rng.standard_normal((int(fraction * 600), p)) * 4.0
        data = np.vstack([clean, gross])
        rows[fraction] = [_relative(
            MahalanobisOutlier(variant=v, contamination=0.1).fit(data).cov_,
            Sigma) for v in VARIANTS]
    assert rows[0.3][1] < 0.5 and rows[0.3][2] < 0.5
    assert rows[0.3][0] > 5.0 * rows[0.0][0]

    print("estimators: relative Frobenius error against the true covariance, "
          "600 points in 6 dimensions, gross errors from a wide Gaussian:")
    for fraction, errors in rows.items():
        cells = "  ".join(f"{v[:9]:9s} {e:5.3f}"
                          for v, e in zip(VARIANTS, errors, strict=True))
        print(f"  {fraction:4.0%} contamination  {cells}")
    print(f"  The classical fit goes from {rows[0.0][0]:.3f} to "
          f"{rows[0.3][0]:.3f} -- a covariance that is mostly a description of "
          f"the tails. Shrinkage tracks it almost exactly: a scaled identity "
          f"target is a regulariser, not a robust estimator. MCD and OGK stay "
          f"under 0.2 throughout, MCD because a half-space of points has the "
          f"smallest determinant and the reweighting step drops the rest, OGK "
          f"because Q_n measures the bulk of each pairwise difference. OGK is "
          f"also the more accurate on clean data ({rows[0.0][2]:.3f} against "
          f"{rows[0.0][1]:.3f}), which is the usual robust-efficiency trade "
          f"paid the other way.")

    features = 60
    C = rng.normal(size=(features, features))
    Wide = C @ C.T / features + 0.1 * np.eye(features)
    factor = np.linalg.cholesky(Wide)
    grid = {}
    for n in (30, 120, 600, 3000):
        sample = rng.standard_normal((n, features)) @ factor.T
        model = MahalanobisOutlier(variant="shrinkage", contamination=0.1).fit(sample)
        values = np.linalg.eigvalsh(model.cov_)
        grid[n] = (model.lambda_, float(values.min()),
                   float(values.max() / values.min()),
                   _relative(model.cov_, Wide))
    sample = rng.standard_normal((30, features)) @ factor.T
    singular = False
    try:
        MahalanobisOutlier(variant="hotelling_t2").fit(sample)
    except ValueError:
        singular = True
    assert singular
    mean = sample.mean(axis=0)
    empirical = (sample - mean).T @ (sample - mean) / 29
    pseudo = _relative(np.linalg.pinv(empirical), Wide)
    assert grid[30][1] > 0.0 and grid[3000][0] < grid[30][0]
    assert grid[30][3] < pseudo

    print(f"shrinkage: {features} features, and n from 30 to 3000. At n = 30 "
          f"the sample covariance has rank 29 and `hotelling_t2` raises; its "
          f"pseudo-inverse is off by {pseudo:.3f}. The shrunk estimate is "
          f"positive definite at every n, with error {grid[30][3]:.3f}:")
    for n, (lam, low, cond, err) in grid.items():
        print(f"  n = {n:4d}  lambda {lam:6.4f}  p/n {features / n:5.3f}  "
              f"min eig {low:.2e}  condition {cond:5.1f}  error {err:.3f}")
    print(f"  lambda tracks p/n -- {grid[30][0]:.2f}, {grid[600][0]:.2f}, "
          f"{grid[3000][0]:.2f} -- because that ratio decides whether the "
          f"sample covariance has enough independent directions to estimate the "
          f"rest. Shrinkage is the price of the missing ones and it goes to "
          f"zero as they arrive. The condition number at n = 30 is "
          f"{grid[30][2]:.1f} where the sample covariance's is infinite, and "
          f"that finite number is the whole product.")


def _demo_broken() -> None:
    """Bad inputs, and the two singular cases."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    cases = [
        ("an unknown variant", lambda: MahalanobisOutlier(variant="mcd_ellipse")),
        ("contamination 0", lambda: MahalanobisOutlier(contamination=0.0)),
        ("contamination 1", lambda: MahalanobisOutlier(contamination=1.0)),
        ("alpha 0", lambda: MahalanobisOutlier(alpha=0.0)),
        ("support 0.5", lambda: MahalanobisOutlier(support_fraction=0.5)),
        ("zero trials", lambda: MahalanobisOutlier(n_trials=0)),
        ("shrinkage 2", lambda: MahalanobisOutlier(shrinkage=2.0)),
        ("two points", lambda: MahalanobisOutlier().fit(X[:2])),
        ("singular n <= p", lambda: MahalanobisOutlier().fit(rng.normal(size=(3, 5)))),
        ("collinear columns", lambda: MahalanobisOutlier().fit(
            np.column_stack([X[:, 0], X[:, 0], X[:, 1]]))),
        ("scoring before fitting", lambda: MahalanobisOutlier().score_samples(X)),
        ("the wrong width", lambda: MahalanobisOutlier().fit(X).score_samples(X[:, :2])),
    ]
    for name, call in cases:
        try:
            call()
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        raise AssertionError(f"{name} did not raise")

    tied = np.repeat(rng.normal(size=(5, 2)), 4, axis=0)
    model = MahalanobisOutlier(variant="mcd", contamination=0.1,
                               n_trials=5).fit(tied)
    assert np.isfinite(model.score_samples(tied)).all()
    print(f"broken: {len(cases)} bad inputs raise -- "
          f"{', '.join(name for name, _ in cases)}. The two singularities are "
          f"handled in opposite ways: `hotelling_t2` refuses when n <= p, "
          f"because (1) needs an inverse that does not exist and a silent "
          f"pseudo-inverse would report distances from a rank-deficient "
          f"covariance as if they meant something, while `mcd` and `shrinkage` "
          f"return a finite answer because a regularisation is what they are. "
          f"Duplicate rows are not an error -- {len(tied)} copies of 5 points "
          f"fit, and the MCD search finds a usable determinant -- but they do "
          f"mean the effective sample size is 5, not {len(tied)}.")


def main() -> None:
    _demo_definition()
    _demo_masking()
    _demo_estimators()
    _demo_broken()
    print("mahalanobis demo passed")


if __name__ == "__main__":
    main()
