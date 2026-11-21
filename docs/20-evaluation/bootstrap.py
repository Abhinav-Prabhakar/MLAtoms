"""Bootstrap — confidence intervals from the data's own resampling distribution.

Chapter:   20-evaluation
Companion: bootstrap.md
Tier 1 · 5 variants: percentile, bca, paired, block_bootstrap, bayesian_bootstrap

The bootstrap replaces "what would happen if we drew a new sample" with "what
happens if we redraw our sample". Resample `n` rows with replacement, recompute
the statistic, repeat `B` times, and read the interval off the resulting spread:

    percentile         quantiles of the bootstrap distribution
    bca                bias-corrected and accelerated quantiles
    paired             resample rows of two paired vectors together
    block_bootstrap    resample blocks, for dependent (time-series) data
    bayesian_bootstrap Dirichlet weights instead of a multinomial draw

Only the last two change the independence assumption. `percentile` and `bca`
differ in which quantiles they read; `paired` differs in what a row is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

VARIANTS = frozenset(
    ["percentile", "bca", "paired", "block_bootstrap", "bayesian_bootstrap"]
)

EPS = 1e-12


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation, one Newton step from `erf`."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    if p < 0.02425:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > 1.0 - 0.02425:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return float(x)


class Bootstrap:
    """Bootstrap confidence intervals for a scalar statistic.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    n_resamples : int
        Number of bootstrap replications (`B`).
    level : float
        Confidence level, e.g. `0.95`.
    random_state : int
        Seed for the resampling stream.
    block_size : int
        Block length of `block_bootstrap`; ignored elsewhere.

    Notes
    -----
    `fit(data, statistic)` accepts an array, or a sequence of equally long
    arrays for `paired`. Except for `bayesian_bootstrap`, `statistic` receives
    what `fit` received, restricted to the resampled rows, and returns a
    scalar. The Bayesian variant passes `(data, weights)` instead, because it
    reweights the sample rather than resampling it.
    """

    def __init__(
        self,
        variant: str = "percentile",
        n_resamples: int = 2000,
        *,
        level: float = 0.95,
        random_state: int = 0,
        block_size: int = 20,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        self.variant = variant
        self.n_resamples = int(n_resamples)
        self.level = float(level)
        self.random_state = int(random_state)
        self.block_size = int(block_size)

    # ------------------------------------------------------------------ #
    # Data plumbing
    # ------------------------------------------------------------------ #
    def _setup(self, data: Any, statistic: Callable[..., float]) -> None:
        if self.variant == "paired":
            self._arrays = [np.asarray(a) for a in data]
            if not self._arrays:
                raise ValueError("paired needs a sequence of arrays")
            n = self._arrays[0].shape[0]
            if any(a.shape[0] != n for a in self._arrays):
                raise ValueError("paired arrays must have equal length")
            self._data = tuple(self._arrays)
        else:
            if isinstance(data, (list, tuple)):
                raise ValueError(f"{self.variant} takes a single array")
            self._data = np.asarray(data)
            n = self._data.shape[0]
        self.n_ = int(n)
        self._statistic = statistic
        if self.variant == "bayesian_bootstrap":
            self.theta_ = float(statistic(self._data, np.full(n, 1.0 / n)))
        else:
            self.theta_ = float(statistic(self._data))

    def _take(self, idx: np.ndarray) -> float:
        if self.variant == "paired":
            return float(self._statistic(tuple(a[idx] for a in self._arrays)))
        return float(self._statistic(self._data[idx]))

    def _indices(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n_
        if self.variant == "block_bootstrap":
            b = min(self.block_size, n)
            n_blocks = -(-n // b)
            starts = rng.integers(0, n, size=n_blocks)
            return np.concatenate([(s + np.arange(b)) % n for s in starts])[:n]
        return rng.integers(0, n, size=n)

    # ------------------------------------------------------------------ #
    # Interval
    # ------------------------------------------------------------------ #
    def _jackknife(self) -> np.ndarray:
        n = self.n_
        out = np.empty(n)
        for i in range(n):
            out[i] = self._take(np.delete(np.arange(n), i))
        return out

    def _bca_quantiles(self) -> np.ndarray:
        dist = self.distribution_
        below = np.sum(dist < self.theta_)
        ties = np.sum(dist == self.theta_)
        z0 = _norm_ppf((below + 0.5 * ties) / dist.size)
        jack = self._jackknife()
        dev = jack.mean() - jack
        scale = 6.0 * float(np.sum(dev ** 2)) ** 1.5
        accel = float(np.sum(dev ** 3)) / max(scale, EPS)
        alpha = (1.0 - self.level) / 2.0
        out = np.empty(2)
        for j, p in enumerate((alpha, 1.0 - alpha)):
            z = _norm_ppf(p)
            denom = 1.0 - accel * (z0 + z)
            out[j] = 0.5 if abs(denom) < EPS else _norm_cdf(z0 + (z0 + z) / denom)
        return np.quantile(dist, np.clip(out, 0.0, 1.0))

    def fit(self, data: Any, statistic: Callable[..., float]) -> Bootstrap:
        """Resample `data`, storing `ci_`, `se_` and the whole `distribution_`."""
        self._setup(data, statistic)
        rng = np.random.default_rng(self.random_state)
        dist = np.empty(self.n_resamples)
        for b in range(self.n_resamples):
            if self.variant == "bayesian_bootstrap":
                dist[b] = float(statistic(self._data, rng.dirichlet(np.ones(self.n_))))
            else:
                dist[b] = self._take(self._indices(rng))
        self.distribution_ = dist
        self.estimate_ = self.theta_
        self.se_ = float(np.std(dist, ddof=1))

        if self.variant == "bca":
            self.ci_ = self._bca_quantiles()
        else:
            alpha = (1.0 - self.level) / 2.0
            self.ci_ = np.quantile(dist, [alpha, 1.0 - alpha])
        self.width_ = float(self.ci_[1] - self.ci_[0])
        return self

    def covers(self, truth: float) -> bool:
        return bool(self.ci_[0] <= truth <= self.ci_[1])


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- percentile on a skewed sample: compare with the CLT interval ---
    x = np.random.default_rng(11).exponential(1.0, size=200)   # true mean 1
    pct = Bootstrap("percentile", 2000, random_state=0).fit(x, np.mean)
    clt = 1.96 * x.std(ddof=1) / np.sqrt(x.size)
    assert pct.covers(1.0), pct.ci_
    assert abs(pct.width_ - 2 * clt) / (2 * clt) < 0.15, (pct.width_, 2 * clt)
    print(f"bootstrap: percentile mean {pct.estimate_:.3f}, ci "
          f"[{pct.ci_[0]:.3f}, {pct.ci_[1]:.3f}] width {pct.width_:.3f} vs CLT "
          f"{2 * clt:.3f}")

    # --- BCa: asymmetric correction for the skewed statistic ---
    bca = Bootstrap("bca", 2000, random_state=0).fit(x, np.mean)
    assert bca.covers(1.0), bca.ci_
    assert bca.width_ > 0.0
    shift = float(bca.ci_[0] - pct.ci_[0])
    assert abs(shift) > 1e-4, shift
    print(f"bootstrap: bca ci [{bca.ci_[0]:.3f}, {bca.ci_[1]:.3f}] shifts the "
          f"lower bound by {shift:+.3f}")

    # --- paired: comparing two predictors on the same rows ---
    n = 300
    y_true = rng.normal(size=n)
    err_a = rng.normal(scale=1.0, size=n)
    pred_a = y_true + err_a
    pred_b = y_true + err_a + 0.20           # same noise, small upward bias
    pred_b = pred_b + rng.normal(scale=0.10, size=n)

    def gap(sample: Sequence[np.ndarray]) -> float:
        yt, pa, pb = sample
        return float(np.mean((pb - yt) ** 2) - np.mean((pa - yt) ** 2))

    pair = Bootstrap("paired", 2000, random_state=0).fit(
        (y_true, pred_a, pred_b), gap)
    pa_err = (pred_a - y_true) ** 2
    pb_err = (pred_b - y_true) ** 2
    draws = np.empty(2000)
    free = np.random.default_rng(0)
    for b in range(2000):
        ia = free.integers(0, n, n)
        ib = free.integers(0, n, n)
        draws[b] = pb_err[ib].mean() - pa_err[ia].mean()
    unpaired = np.quantile(draws, [0.025, 0.975])
    width_unpaired = float(unpaired[1] - unpaired[0])
    assert pair.ci_[0] > 0.0, pair.ci_
    assert pair.width_ < 0.6 * width_unpaired, (pair.width_, width_unpaired)
    print(f"bootstrap: paired gap {pair.estimate_:.3f} ci "
          f"[{pair.ci_[0]:.3f}, {pair.ci_[1]:.3f}] width {pair.width_:.3f} vs "
          f"unpaired {width_unpaired:.3f}")

    # --- block bootstrap: dependent data needs blocks, not rows ---
    m = 500
    rho = 0.85
    noise = rng.normal(size=m)
    series = np.empty(m)
    series[0] = noise[0]
    for t in range(1, m):
        series[t] = rho * series[t - 1] + noise[t]
    iid = Bootstrap("percentile", 1000, random_state=0).fit(series, np.mean)
    block = Bootstrap("block_bootstrap", 1000, random_state=0,
                      block_size=25).fit(series, np.mean)
    ratio = block.width_ / iid.width_
    assert ratio > 2.0, ratio
    print(f"bootstrap: ar(1) mean ci width iid {iid.width_:.3f} vs block "
          f"{block.width_:.3f} (x{ratio:.1f})")

    # --- bayesian bootstrap: Dirichlet weights, no row ever gets zero mass ---
    def weighted_mean(s: np.ndarray, w: np.ndarray) -> float:
        return float(np.sum(w * s))

    bayes = Bootstrap("bayesian_bootstrap", 2000, random_state=0).fit(
        x, weighted_mean)
    assert bayes.covers(1.0), bayes.ci_
    rel = abs(bayes.width_ - pct.width_) / pct.width_
    assert rel < 0.2, (bayes.width_, pct.width_)
    print(f"bootstrap: bayesian ci [{bayes.ci_[0]:.3f}, {bayes.ci_[1]:.3f}] "
          f"width {bayes.width_:.3f} vs percentile {pct.width_:.3f}")

    # --- coverage: percentile vs bca on repeated samples ---
    reps = 200
    hits = {"percentile": 0, "bca": 0}
    for r in range(reps):
        sample = np.random.default_rng(1000 + r).exponential(1.0, size=100)
        for name in hits:
            fit = Bootstrap(name, 400, random_state=r).fit(sample, np.mean)
            hits[name] += int(fit.covers(1.0))
    coverage = {k: v / reps for k, v in hits.items()}
    for name, value in coverage.items():
        assert abs(value - 0.95) <= 0.04, (name, value)
    print(f"bootstrap: nominal 95% coverage over {reps} samples — percentile "
          f"{coverage['percentile']:.3f}, bca {coverage['bca']:.3f}")

    print("bootstrap demo passed")


if __name__ == "__main__":
    main()
