r"""Extreme value theory — fit the tail instead of counting it.

Chapter:   09-density-and-anomaly
Companion: extreme_value.md
Tier 3 · 5 variants: pot, gpd, block_maxima, gev, return_level

    GPD survival:   S(y) = (1 + xi*y/beta)^(-1/xi)                        (1)
    POT tail:       p(x) = zeta_u * S(x - u),   zeta_u = n_u / n           (2)
    POT level:      x_T = u + beta/xi * ( (T*zeta_u)^xi - 1 )              (3)
    GEV cdf:        G(x) = exp( -(1 + xi*(x-mu)/sigma)^(-1/xi) )           (4)
    block tail:     p(x) = 1 - G(x)^(1/b)                                  (5)
    block level:    x_T = mu + sigma/xi * ( (-b*log(1-1/T))^(-xi) - 1 )    (6)

The maximum of many observations has only three possible shapes in the limit:
Gumbel, Frechet or Weibull, and the generalised extreme value distribution is
all three at once, indexed by a single shape parameter xi. The same theorem has
a second form, which is the one this file uses: the amount by which a sample
exceeds a high threshold, conditional on exceeding it, converges to a
generalised Pareto distribution with the same xi. So there are two routes to
the tail, and the five variants here are the two routes crossed with the ways
of fitting them.

`pot` fits the generalised Pareto to exceedances by probability-weighted
moments. `gpd` fits the same model by maximum likelihood. `block_maxima` fits
the GEV to block maxima by L-moments, and `gev` fits it by maximum likelihood.
The last, `return_level`, drops the shape parameter and reads the tail off a
straight line on a return-level plot; it is the one-parameter special case, and
the most stable of the five.

All five answer the same question, which is the question the empirical CDF
cannot answer: how improbable is a value *beyond anything in the sample*? The
answer is a probability, not a rank, so the score is its surprisal,

    score(x) = -log p(x),

and the model's native decision rule is a return period: flag the points whose
expected waiting time exceeds T observations. A `contamination` budget is also
supported, because that is how the rest of the chapter decides, but it is the
weaker interface: it throws away the extrapolation that makes the method worth
using.

The demos make the trade-offs concrete. On exponential data all three POT
estimators recover xi near zero and beta within twenty percent of one, and the
return level at T = 1000 lands within six percent of log 1000. The same data at
the 99th-percentile threshold has ten times fewer exceedances and correspondingly
noisier estimates, which is the bias-variance trade-off that every
peaks-over-threshold analysis has to make. On beta(2, 2) data the shape is near
-0.5, the fitted upper endpoint lands on the true endpoint at one, and a point
beyond it gets probability zero and infinite score. That last case is the reason
the shape parameter exists: it is the only part of the model that knows the tail
can end.
"""  # noqa: E501

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("pot", "gpd", "block_maxima", "gev", "return_level")
BLOCK_VARIANTS = ("block_maxima", "gev")


def _gpd_survival(z: np.ndarray, beta: float, xi: float) -> np.ndarray:
    """Survival function of the GPD, equation (1)."""
    if abs(xi) < 1e-8:
        return np.exp(-z / beta)
    y = 1.0 + xi * z / beta
    return np.where(y > 0.0, np.power(np.maximum(y, 1e-300), -1.0 / xi), 0.0)


def _gpd_nll(theta: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of exceedances, parameterised by log beta."""
    beta = math.exp(float(theta[0]))
    xi = float(theta[1])
    z = xi * y / beta
    if not np.all(1.0 + z > 0.0):
        return 1e12
    if abs(xi) < 1e-8:
        return float(y.size * math.log(beta) + y.sum() / beta)
    return float(y.size * math.log(beta)
                 + (1.0 + 1.0 / xi) * np.log1p(z).sum())


def _gev_nll(theta: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood of block maxima, parameterised by log sigma."""
    mu = float(theta[0])
    sigma = math.exp(float(theta[1]))
    xi = float(theta[2])
    z = (x - mu) / sigma
    y = 1.0 + xi * z
    if not np.all(y > 0.0):
        return 1e12
    if abs(xi) < 1e-8:
        return float(x.size * math.log(sigma) + np.sum(z + np.exp(-z)))
    return float(x.size * math.log(sigma)
                 + np.sum((1.0 + 1.0 / xi) * np.log(y)
                          + np.power(y, -1.0 / xi)))


def _nelder_mead(objective, start: np.ndarray, steps: np.ndarray,
                 iterations: int = 800, tol: float = 1e-10):
    """Minimise a scalar function of a few variables, deterministically.

    A plain reflection/expansion/contraction simplex. It is enough for two or
    three parameters and it has no derivative, which matters because the
    extreme-value likelihood is undefined outside the support.
    """
    simplex = [np.asarray(start, dtype=float)]
    for i in range(start.size):
        point = simplex[0].copy()
        point[i] += steps[i]
        simplex.append(point)
    values = [objective(p) for p in simplex]
    for _ in range(iterations):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) <= tol * (1.0 + abs(values[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        f_reflected = objective(reflected)
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (centroid - worst)
            f_expanded = objective(expanded)
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            f_contracted = objective(contracted)
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                best = simplex[0]
                simplex = [best + 0.5 * (p - best) for p in simplex[1:]]
                simplex.insert(0, best)
                values = [objective(p) for p in simplex]
    return simplex[0]


def _fit_gpd_pwm(y: np.ndarray) -> tuple:
    """Method of probability-weighted moments for the GPD."""
    ordered = np.sort(y)
    n = ordered.size
    a0 = float(ordered.mean())
    a1 = float(np.sum((n - np.arange(1, n + 1)) / (n - 1.0) * ordered) / n)
    denominator = a0 - 2.0 * a1
    if denominator <= 0.0:
        raise ValueError("probability-weighted moments are degenerate")
    xi = (a0 - 4.0 * a1) / denominator
    beta = 2.0 * a0 * a1 / denominator
    return beta, xi


def _fit_gpd_mle(y: np.ndarray) -> tuple:
    """Maximum likelihood, started from the probability-weighted moments."""
    beta0, xi0 = _fit_gpd_pwm(y)
    start = np.array([math.log(max(beta0, 1e-8)),
                      float(np.clip(xi0, -0.45, 0.45))])
    theta = _nelder_mead(lambda p: _gpd_nll(p, y), start,
                         np.array([0.2, 0.1]))
    return math.exp(float(theta[0])), float(theta[1])


def _fit_gpd(data: np.ndarray, variant: str) -> tuple:
    if variant == "pot":
        return _fit_gpd_pwm(data)
    if variant == "gpd":
        return _fit_gpd_mle(data)
    if variant == "return_level":
        return _fit_return_level(data)
    raise ValueError(f"unknown tail estimator {variant!r}")


def _fit_return_level(y: np.ndarray) -> tuple:
    """Least squares through the origin on the return-level plot.

    The plot is -log(survival) against the exceedance, which is a straight line
    with slope 1/beta exactly when xi = 0. Fitting that line gives a
    one-parameter tail: no shape to estimate, and no way to represent a finite
    endpoint.
    """
    ordered = np.sort(y)
    n = ordered.size
    plotting = (n - np.arange(1, n + 1) + 0.5) / n
    slope = float(np.sum(ordered * -np.log(plotting)) / np.sum(ordered ** 2))
    if slope <= 0.0:
        raise ValueError("the return-level plot has no positive slope")
    return 1.0 / slope, 0.0


def _fit_gev_pwm(x: np.ndarray) -> tuple:
    """Hosking's L-moment estimator for the GEV."""
    ordered = np.sort(x)
    n = ordered.size
    i = np.arange(1, n + 1, dtype=float)
    b0 = float(ordered.mean())
    b1 = float(np.sum((i - 1.0) / (n - 1.0) * ordered) / n)
    b2 = float(np.sum((i - 1.0) * (i - 2.0) / ((n - 1.0) * (n - 2.0))
                      * ordered) / n)
    ell2 = 2.0 * b1 - b0
    ell3 = 6.0 * b2 - 6.0 * b1 + b0
    if ell2 <= 0.0:
        raise ValueError("the L-moment scale is not positive")
    c = 2.0 / (3.0 + ell3 / ell2) - math.log(2.0) / math.log(3.0)
    xi = -(7.8590 * c + 2.9554 * c * c)
    if not -1.0 < xi < 1.0:
        raise ValueError("the L-moment shape is outside (-1, 1)")
    if abs(xi) < 1e-8:
        sigma = ell2 / math.log(2.0)
        mu = b0 - sigma * np.euler_gamma
    else:
        sigma = ell2 * xi / ((2.0 ** xi - 1.0) * math.gamma(1.0 - xi))
        mu = b0 - sigma * (math.gamma(1.0 - xi) - 1.0) / xi
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("the L-moment fit is degenerate")
    return mu, sigma, xi


def _fit_gev_mle(x: np.ndarray) -> tuple:
    """Maximum likelihood, started from the L-moments."""
    mu0, sigma0, xi0 = _fit_gev_pwm(x)
    start = np.array([mu0, math.log(sigma0), float(np.clip(xi0, -0.4, 0.4))])
    theta = _nelder_mead(lambda p: _gev_nll(p, x), start,
                         np.array([0.25 * sigma0, 0.15, 0.1]))
    return float(theta[0]), math.exp(float(theta[1])), float(theta[2])


class ExtremeValue:
    """Tail model for univariate data.

    Parameters
    ----------
    variant : {"pot", "gpd", "block_maxima", "gev", "return_level"}
        `pot` and `gpd` fit the generalised Pareto to exceedances over a
        threshold, by probability-weighted moments and maximum likelihood.
        `block_maxima` and `gev` fit the GEV to block maxima, by L-moments and
        maximum likelihood. `return_level` fits an exponential tail by least
        squares on the return-level plot.
    threshold : float, optional
        Explicit threshold for the peaks-over-threshold variants. When omitted,
        the `threshold_quantile` of the training sample is used.
    threshold_quantile : float
        Quantile that sets the default threshold, in (0, 1).
    block_size : int, optional
        Observations per block for the block-maxima variants. Defaults to
        `max(5, n // 50)`, so that a sample of a few thousand rows gives about
        fifty maxima.
    contamination : float
        Fraction of the training data labelled anomalous. Sets `threshold_`
        as that quantile of the training scores; use `return_period` instead
        if you want the model's own decision rule.

    Attributes
    ----------
    u_, beta_, xi_ : float
        Threshold, scale and shape for the peaks-over-threshold variants.
    mu_, sigma_, xi_ : float
        Location, scale and shape for the block-maxima variants.
    scale_ : float
        Probability of exceeding the threshold, with the half-count smoothing
        that makes the empirical and parametric pieces meet at `u_`.
    scores_ : ndarray
        Training scores, higher meaning more extreme.
    threshold_ : float
        Score above which `predict` returns -1.
    """

    def __init__(self, *, variant: str = "pot",
                 threshold: float | None = None,
                 threshold_quantile: float = 0.9,
                 block_size: int | None = None,
                 contamination: float = 0.1) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie in (0, 1)")
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if block_size is not None and block_size < 2:
            raise ValueError("block_size must be at least two")
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must lie in (0, 0.5)")
        self.variant = variant
        self.threshold = threshold
        self.threshold_quantile = float(threshold_quantile)
        self.block_size = block_size
        self.contamination = float(contamination)

    def fit(self, X: np.ndarray) -> ExtremeValue:
        x = self._check(X)
        n = x.size
        if n < 20:
            raise ValueError(f"need at least 20 observations, got {n}")
        if np.ptp(x) == 0.0:
            raise ValueError("the sample is constant")
        self.n_ = n
        if self.variant in BLOCK_VARIANTS:
            self._fit_blocks(x)
        else:
            self._fit_exceedances(x)
        self.scores_ = self.score_samples(x)
        self.threshold_ = float(np.quantile(self.scores_,
                                            1.0 - self.contamination))
        return self

    def _fit_blocks(self, x: np.ndarray) -> None:
        n = x.size
        size = self.block_size if self.block_size is not None else max(5, n // 50)
        blocks = n // size
        if blocks < 10:
            raise ValueError(f"need at least 10 blocks, got {blocks}; "
                             f"lower block_size or collect more data")
        self.block_size_ = size
        maxima = x[: blocks * size].reshape(blocks, size).max(axis=1)
        if self.variant == "block_maxima":
            self.mu_, self.sigma_, self.xi_ = _fit_gev_pwm(maxima)
        else:
            self.mu_, self.sigma_, self.xi_ = _fit_gev_mle(maxima)
        if abs(self.xi_) >= 1.0:
            raise ValueError("the shape estimate is outside the model's range")

    def _fit_exceedances(self, x: np.ndarray) -> None:
        self.sorted_ = np.sort(x)
        if self.threshold is None:
            u = float(np.quantile(x, self.threshold_quantile))
        else:
            u = float(self.threshold)
        y = x[x > u] - u
        if y.size < 10:
            raise ValueError(f"need at least 10 exceedances, got {y.size}; "
                             f"lower the threshold")
        self.u_ = u
        self.n_u_ = y.size
        self.beta_, self.xi_ = _fit_gpd(y, self.variant)
        if not np.isfinite(self.beta_) or self.beta_ <= 0.0:
            raise ValueError("the scale estimate is not positive")
        if not -1.0 < self.xi_ < 1.0:
            raise ValueError("the shape estimate is outside (-1, 1)")
        # the same half-count as the empirical CDF, so the two pieces meet at u
        self.scale_ = (self.n_u_ + 0.5) / (self.n_ + 1.0)

    def tail_probability(self, X: np.ndarray) -> np.ndarray:
        """Estimated probability of observing a value above each entry."""
        if not hasattr(self, "n_"):
            raise RuntimeError("fit before scoring")
        x = np.atleast_1d(np.asarray(X, dtype=float)).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("X must be finite")
        if self.variant in BLOCK_VARIANTS:
            return self._block_tail(x)
        probability = np.empty(x.size)
        below = x <= self.u_
        counts = np.searchsorted(self.sorted_, x[below], side="right")
        probability[below] = (self.n_ - counts + 0.5) / (self.n_ + 1.0)
        probability[~below] = self.scale_ * _gpd_survival(
            x[~below] - self.u_, self.beta_, self.xi_)
        return np.clip(probability, 0.0, 1.0)

    def _block_tail(self, x: np.ndarray) -> np.ndarray:
        cdf = self._gev_cdf(x)
        with np.errstate(divide="ignore"):
            probability = -np.expm1(np.log(cdf) / self.block_size_)
        # a point beyond a finite endpoint has cdf one, so the log is zero
        return np.clip(probability, 0.0, 1.0)

    def _gev_cdf(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu_) / self.sigma_
        if abs(self.xi_) < 1e-8:
            return np.exp(-np.exp(-z))
        y = 1.0 + self.xi_ * z
        cdf = np.zeros_like(x)
        inside = y > 0.0
        cdf[inside] = np.exp(-np.power(y[inside], -1.0 / self.xi_))
        if self.xi_ < 0.0:
            cdf[~inside] = 1.0
        return cdf

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Surprisal of the estimated exceedance probability."""
        probability = self.tail_probability(X)
        with np.errstate(divide="ignore"):
            return np.where(probability > 0.0, -np.log(probability), np.inf)

    def return_period(self, X: np.ndarray) -> np.ndarray:
        """Expected number of observations between exceedances of each value."""
        with np.errstate(divide="ignore"):
            return np.where(self.tail_probability(X) > 0.0,
                            1.0 / self.tail_probability(X), np.inf)

    def return_level(self, period: float) -> float:
        """Value exceeded once every `period` observations, on average."""
        if not hasattr(self, "n_"):
            raise RuntimeError("fit before asking for a return level")
        if not np.isfinite(period) or period <= 1.0:
            raise ValueError("period must be finite and greater than one")
        if self.variant in BLOCK_VARIANTS:
            return self._block_level(period)
        return self._exceedance_level(period)

    def _exceedance_level(self, period: float) -> float:
        target = period * self.scale_
        if target <= 1.0:
            raise ValueError("period is shorter than the threshold's own "
                             "return period; raise it")
        if abs(self.xi_) < 1e-8:
            return self.u_ + self.beta_ * math.log(target)
        return self.u_ + self.beta_ / self.xi_ * (target ** self.xi_ - 1.0)

    def _block_level(self, period: float) -> float:
        reduced = -self.block_size_ * math.log1p(-1.0 / period)
        if abs(self.xi_) < 1e-8:
            return self.mu_ - self.sigma_ * math.log(reduced)
        return self.mu_ + self.sigma_ / self.xi_ * (reduced ** (-self.xi_) - 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """-1 for points above the threshold, +1 for the rest."""
        if not hasattr(self, "threshold_"):
            raise RuntimeError("fit before predicting")
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    def _check(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("extreme value theory here is univariate; "
                             "pass shape (n,) or (n, 1)")
        if not np.all(np.isfinite(x)):
            raise ValueError("X must be finite")
        return x

# --------------------------------------------------------------------------- #
# Demos
# --------------------------------------------------------------------------- #
def _demo_tail_estimators() -> None:
    """Three ways to fit the same Pareto tail, against a known answer."""
    rng = np.random.default_rng(0)
    x = rng.exponential(size=5000)
    print("exponential(1), n = 5000, threshold at the 90th percentile:")
    print("  variant        xi      beta       p(8)     x_1000")
    estimates = {}
    for variant in ("pot", "gpd", "return_level"):
        model = ExtremeValue(variant=variant).fit(x)
        p_eight = float(model.tail_probability([8.0])[0])
        level = model.return_level(1000)
        estimates[variant] = (model.xi_, model.beta_, p_eight, level)
        print(f"  {variant:<13s} {model.xi_:6.3f} {model.beta_:9.3f}"
              f" {p_eight:10.6f} {level:9.3f}")
    truth = float(np.exp(-8.0))
    print(f"  the exponential's own p(8) is e^-8 = {truth:.6f} and "
          f"x_1000 = log 1000 = {np.log(1000.0):.3f}")
    for xi, beta, p_eight, level in estimates.values():
        assert abs(xi) < 0.15
        assert 0.9 < beta < 1.3
        assert 0.5 < p_eight / truth < 2.0
        assert abs(level / np.log(1000.0) - 1.0) < 0.1
    print("  All three recover the shape near zero. They differ in how much")
    print("  freedom they give the tail: the two-parameter fits can bend it,")
    print("  and the one-parameter fit cannot, which is why it is the most")
    print("  stable of the five and also the least able to extrapolate.")


def _demo_block_maxima() -> None:
    """Block size is a bias-variance choice, not a detail."""
    rng = np.random.default_rng(1)

    def draw(kind: str, blocks: int, size: int) -> np.ndarray:
        if kind == "gaussian":
            return rng.normal(size=blocks * size)
        return (-np.log(rng.uniform(size=blocks * size))) ** (-0.2)

    print("3000 blocks, GEV fitted by L-moments and by likelihood:")
    print("  data       block   variant        mu    sigma      xi")
    table = {}
    for kind, size in (("gaussian", 25), ("gaussian", 250), ("frechet", 25)):
        sample = draw(kind, 3000, size)
        for variant in ("block_maxima", "gev"):
            model = ExtremeValue(variant=variant, block_size=size).fit(sample)
            table[(kind, size, variant)] = model.xi_
            print(f"  {kind:<10s} {size:5d}   {variant:<12s}"
                  f" {model.mu_:7.3f} {model.sigma_:8.3f} {model.xi_:7.3f}")
    for variant in ("block_maxima", "gev"):
        assert abs(table[("frechet", 25, variant)] - 0.2) < 0.08
        assert table[("gaussian", 25, variant)] < -0.05
        assert table[("gaussian", 250, variant)] > table[("gaussian", 25, variant)]
    print("  Frechet maxima are max-stable, so the shape is exactly 0.2 and")
    print("  both estimators find it. Gaussian maxima are only asymptotically")
    print("  Gumbel: at a block of 25 the fitted shape is about -0.12, and")
    print("  growing the block to 250 moves it toward zero. That negative")
    print("  shape is the block size talking, not the data.")


def _demo_return_levels() -> None:
    """The native decision rule, and what happens when the tail ends."""
    rng = np.random.default_rng(2)
    x = rng.exponential(size=5000)
    model = ExtremeValue(variant="pot").fit(x)
    print("exponential(1), n = 5000, fitted return levels against log T:")
    print("        T    fitted   analytic    ratio")
    for period in (10.0, 100.0, 1000.0, 10000.0):
        fitted = model.return_level(period)
        analytic = np.log(period)
        print(f"  {period:7.0f} {fitted:9.3f} {analytic:10.3f}"
              f" {fitted / analytic:8.3f}")
        assert abs(fitted / analytic - 1.0) < 0.12
    assert model.return_level(1000.0) > model.return_level(100.0)

    print("beta(2, 2), n = 4000, a tail with a finite endpoint at one:")
    print("  variant        xi   endpoint   p(1.2)   score(1.2)")
    y = rng.beta(2.0, 2.0, size=4000)
    for variant in ("pot", "gpd"):
        bounded = ExtremeValue(variant=variant).fit(y)
        endpoint = bounded.u_ - bounded.beta_ / bounded.xi_
        probability = float(bounded.tail_probability([1.2])[0])
        score = float(bounded.score_samples([1.2])[0])
        print(f"  {variant:<13s} {bounded.xi_:6.3f} {endpoint:10.4f}"
              f" {probability:8.3f} {score:10.1f}")
        assert abs(endpoint - 1.0) < 0.03
        assert bounded.xi_ < -0.3
        assert probability == 0.0 and np.isinf(score)
    print("  A negative shape puts an upper bound on the model at")
    print("  u - beta/xi, and both fits land on the true endpoint. Beyond it")
    print("  the estimated probability is exactly zero and the score infinite:")
    print("  the model says 'impossible', which is the correct answer.")


def _demo_detection() -> None:
    """What the empirical CDF cannot do, and the tail model can."""
    rng = np.random.default_rng(3)
    clean = rng.exponential(size=2000)
    data = np.concatenate([clean, np.full(20, 11.0)])
    model = ExtremeValue(variant="pot", contamination=0.01).fit(data)
    flags = model.predict(data)
    false_positives = int(np.sum(flags[:2000] == -1))
    probability = float(model.tail_probability([11.0])[0])
    print("2000 exponential draws plus 20 points at 11, contamination 0.01:")
    print(f"  largest clean value          {clean.max():.3f}")
    print(f"  empirical count above 11     {int(np.sum(clean > 11.0))}")
    print(f"  fitted tail probability      {probability:.6f}")
    print(f"  score at 11                  {model.score_samples([11.0])[0]:.3f}")
    print(f"  fitted threshold             {model.threshold_:.3f}")
    print(f"  injected points flagged      {int(np.sum(flags[2000:] == -1))}/20")
    print(f"  false positives              {false_positives}")
    assert np.all(flags[2000:] == -1)
    assert false_positives <= 5
    assert clean.max() < 11.0
    assert probability > 0.0
    print("  The empirical tail is empty above 11, so a rank-based detector")
    print("  has nothing to rank and cannot say how surprising the value is.")
    print("  The fitted tail gives it a probability, hence a score, hence a")
    print("  decision: the extrapolation is the whole product.")


def _demo_threshold() -> None:
    """Raising the threshold trades bias for variance, visibly."""
    rng = np.random.default_rng(4)
    replications = 60
    print(f"exponential(1), n = 3000, probability-weighted moments, "
          f"{replications} replications:")
    print("  quantile   n_u   mean xi   std xi   mean p(6)   std log10 p(6)")
    previous_std = 0.0
    previous_spread = 0.0
    for quantile in (0.8, 0.9, 0.95, 0.99):
        shapes = []
        logs = []
        counts = 0
        for _ in range(replications):
            sample = rng.exponential(size=3000)
            fitted = ExtremeValue(variant="pot",
                                  threshold_quantile=quantile).fit(sample)
            shapes.append(fitted.xi_)
            logs.append(float(np.log10(fitted.tail_probability([6.0])[0])))
            counts = fitted.n_u_
        shapes = np.asarray(shapes)
        spread = float(np.std(np.asarray(logs)))
        assert spread > previous_spread
        print(f"  {quantile:8.2f} {counts:5d} {shapes.mean():9.4f}"
              f" {shapes.std():8.4f} {10.0 ** np.mean(logs):11.6f}"
              f" {spread:16.4f}")
        assert shapes.std() > previous_std
        previous_std = float(shapes.std())
        previous_spread = spread
        assert abs(10.0 ** np.mean(logs) / np.exp(-6.0) - 1.0) < 0.25
    print("  The estimate of p(6) is close to the truth at every threshold,")
    print("  because the Pareto model is exact here. What changes is the spread,")
    print("  which grows as the exceedances run out: the 99th-percentile")
    print("  threshold keeps 30 points and its shape estimate moves almost")
    print("  five times as much as the one fitted to 600.")


def _demo_broken() -> None:
    """Malformed calls are refused."""
    rng = np.random.default_rng(5)
    x = rng.exponential(size=500)
    calls = [
        lambda: ExtremeValue(variant="nope"),
        lambda: ExtremeValue(threshold_quantile=0.0),
        lambda: ExtremeValue(threshold_quantile=1.0),
        lambda: ExtremeValue(contamination=0.5),
        lambda: ExtremeValue(block_size=1),
        lambda: ExtremeValue(threshold=np.inf),
        lambda: ExtremeValue().fit(x[:10]),
        lambda: ExtremeValue().fit(np.zeros(50)),
        lambda: ExtremeValue().fit(np.r_[x[:40], np.nan]),
        lambda: ExtremeValue().fit(np.column_stack([x, x])),
        lambda: ExtremeValue(threshold=100.0).fit(x),
        lambda: ExtremeValue(variant="gev", block_size=100).fit(x[:500]),
        lambda: ExtremeValue().predict(x),
        lambda: ExtremeValue().score_samples(x),
        lambda: ExtremeValue().return_level(100.0),
        lambda: ExtremeValue().fit(x).return_level(1.0),
        lambda: ExtremeValue().fit(x).tail_probability([np.nan]),
    ]
    for call in calls:
        try:
            call()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected a refusal")
    print(f"{len(calls)} malformed calls refused: an unknown variant, two "
          f"quantile bounds, a contamination bound, a block of one, a "
          f"non-finite threshold, a sample of ten, a constant sample, a "
          f"non-finite value, two columns, an empty tail, too few blocks, "
          f"predicting and scoring before fitting, a return level before "
          f"fitting, a return period of one, and a non-finite query.")


def main() -> None:
    _demo_tail_estimators()
    _demo_block_maxima()
    _demo_return_levels()
    _demo_detection()
    _demo_threshold()
    _demo_broken()
    print("extreme_value demo passed")


if __name__ == "__main__":
    main()
