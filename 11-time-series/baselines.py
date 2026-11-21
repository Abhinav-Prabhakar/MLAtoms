"""Naive baselines — the forecasts your model has to beat.

Chapter:   11-time-series
Companion: baselines.md
Tier 1 · 6 variants: naive, seasonal_naive, drift, mean, random_walk, snaive
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

VARIANTS = ("naive", "seasonal_naive", "drift", "mean", "random_walk", "snaive")
SEASONAL = ("seasonal_naive", "snaive")


# --------------------------------------------------------------------------- #
# The forecaster
# --------------------------------------------------------------------------- #
class NaiveForecaster:
    """Last-value, drift, mean and seasonal-naive forecasts with intervals.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`. `snaive` is the R name for `seasonal_naive` and
        returns the same forecasts; `random_walk` shares the point forecast of
        `naive` and differs in its predictive variance.
    season : int
        Seasonal period `m` for the seasonal variants, in observations.

    Notes
    -----
    `fit` stores `sigma_`, the in-sample one-step innovation scale, and the
    statistics each rule needs. `predict` returns `(horizon,)` forecasts;
    `interval` returns a Gaussian band whose width is the one place these
    baselines genuinely differ.
    """

    def __init__(self, variant: str = "naive", *, season: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if season < 1:
            raise ValueError("season must be at least 1")
        self.variant = variant
        self.season = int(season)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> NaiveForecaster:
        """Record the last value, the drift and the innovation scale."""
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        self.n_ = y.size
        self.last_ = float(y[-1])
        self.mean_ = float(y.mean())
        self.slope_ = 0.0
        self.y_ = y.copy()
        if self.variant in SEASONAL:
            if self.season > self.n_ - 3:
                raise ValueError("season leaves too few seasonal differences")
            gap = y[self.season:] - y[:-self.season]
            self.sigma_ = float(np.std(gap, ddof=1))
            self.seasonal_ = y[-self.season:].copy()
        elif self.variant == "drift":
            if self.n_ < 3:
                raise ValueError("the drift method needs at least three points")
            self.slope_ = (y[-1] - y[0]) / (self.n_ - 1)
            residual = np.diff(y) - self.slope_
            self.sigma_ = float(np.std(residual, ddof=1))
        elif self.variant == "mean":
            if self.n_ < 2:
                raise ValueError("the mean method needs at least two points")
            self.sigma_ = float(np.std(y, ddof=1))
        else:
            if self.n_ < 3:
                raise ValueError("this method needs at least three points")
            self.sigma_ = float(np.std(np.diff(y), ddof=1))
        return self

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int) -> np.ndarray:
        """Point forecasts, shape `(horizon,)`."""
        self._check_fitted()
        horizon = self._check_horizon(horizon)
        steps = np.arange(1, horizon + 1, dtype=float)
        if self.variant in ("naive", "random_walk"):
            return np.full(horizon, self.last_)
        if self.variant == "mean":
            return np.full(horizon, self.mean_)
        if self.variant == "drift":
            return self.last_ + steps * self.slope_
        index = (np.arange(horizon) % self.season)
        return self.seasonal_[index]

    def interval(self, horizon: int, level: float = 0.95
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian prediction band, two arrays of shape `(horizon,)`."""
        self._check_fitted()
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        forecast = self.predict(horizon)
        steps = np.arange(1, horizon + 1, dtype=float)
        if self.variant == "random_walk":
            sd = self.sigma_ * np.sqrt(steps)
        elif self.variant == "drift":
            sd = self.sigma_ * np.sqrt(steps * (1.0 + steps / (self.n_ - 1)))
        elif self.variant == "mean":
            sd = np.full(horizon, self.sigma_ * np.sqrt(1.0 + 1.0 / self.n_))
        else:
            sd = np.full(horizon, self.sigma_)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return forecast - z * sd, forecast + z * sd

    # -- checks ----------------------------------------------------------- #
    def _check_horizon(self, horizon) -> int:
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return int(horizon)

    def _check_fitted(self):
        if not hasattr(self, "sigma_"):
            raise RuntimeError("call fit before predict or interval")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _seasonal_series(n: int, rng: np.random.Generator,
                     trend: float = 0.02) -> np.ndarray:
    t = np.arange(n)
    season = 2.0 * np.sin(2.0 * np.pi * t / 12.0)
    return trend * t + season + rng.normal(0.0, 0.3, n)


def _mae(forecast: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - truth)))


def demo() -> None:
    rng = np.random.default_rng(0)
    y = _seasonal_series(144, rng)
    train, holdout = y[:-24], y[-24:]

    print("144 points of a 12-period season on a slow trend, 24 held out:")
    scores = {}
    for variant in ("mean", "naive", "drift", "seasonal_naive"):
        model = NaiveForecaster(variant, season=12).fit(train)
        scores[variant] = _mae(model.predict(24), holdout)
        print(f"  {variant:14s} MAE {scores[variant]:.4f}")
    assert scores["drift"] < scores["naive"] < scores["mean"]
    assert scores["seasonal_naive"] < scores["naive"]
    print("the seasonal naive beats the naive on a seasonal series, and the")
    print("drift beats it on a trend; neither beats the other everywhere")

    flat = 2.0 * np.sin(2.0 * np.pi * np.arange(120) / 12.0)
    flat += rng.normal(0.0, 0.1, 120)
    seasonal = NaiveForecaster("seasonal_naive", season=12).fit(flat[:-24])
    flat_seasonal = _mae(seasonal.predict(24), flat[-24:])
    flat_naive = _mae(NaiveForecaster("naive").fit(flat[:-24]).predict(24),
                      flat[-24:])
    assert flat_seasonal < 0.2
    assert flat_naive > 1.0
    print(f"without a trend the season alone is enough: seasonal naive "
          f"{flat_seasonal:.3f}")
    print(f"against {flat_naive:.3f} for last-value, on the same 24 points")

    model = NaiveForecaster("naive").fit(train)
    horizon = 6
    naive_forecast = model.predict(horizon)
    assert np.all(naive_forecast == train[-1])
    drift = NaiveForecaster("drift").fit(train)
    steps = np.arange(1, horizon + 1)
    slope = (train[-1] - train[0]) / (train.size - 1)
    assert np.allclose(drift.predict(horizon), train[-1] + steps * slope)
    assert np.isclose(drift.slope_, slope)
    seasonal = NaiveForecaster("seasonal_naive", season=12).fit(train)
    expected = train[-12:][(np.arange(horizon)) % 12]
    assert np.allclose(seasonal.predict(horizon), expected)
    average = NaiveForecaster("mean").fit(train)
    assert np.allclose(average.predict(horizon), train.mean())
    print("each rule is exact: last value, last value plus h times the mean")
    print("difference, the last seasonal cycle recycled, and the sample mean")

    alias = NaiveForecaster("snaive", season=12).fit(train)
    assert np.allclose(alias.predict(24), seasonal.predict(24))
    walk = NaiveForecaster("random_walk").fit(train)
    assert np.allclose(walk.predict(24), model.predict(24))
    print("snaive and seasonal_naive coincide, as do random_walk and naive:")
    print("the names differ, the point forecasts do not")

    low, high = walk.interval(10, 0.95)
    width = high - low
    ratio = float(width[-1] / width[0])
    assert np.isclose(ratio, np.sqrt(10.0), rtol=1e-9)
    flat_low, flat_high = model.interval(10, 0.95)
    assert np.allclose(flat_high - flat_low, (flat_high - flat_low)[0])
    print(f"their intervals do: random-walk width grows by sqrt(h) "
          f"({width[0]:.3f} to {width[-1]:.3f}),")
    print("naive's stays constant, because it assumes no accumulation")

    print("a random walk has a known predictive law, so the band is testable:")
    sigma = 1.0
    coverage = {}
    for step in (1, 10):
        hits = 0
        walks = 400
        for _ in range(walks):
            path = np.cumsum(rng.normal(0.0, sigma, 120))
            model = NaiveForecaster("random_walk").fit(path)
            lower, upper = model.interval(step, 0.95)
            future = path[-1] + np.cumsum(rng.normal(0.0, sigma, step))[-1]
            hits += int(lower[step - 1] <= future <= upper[step - 1])
        coverage[step] = hits / walks
    print(f"  nominal 0.95, empirical {coverage[1]:.3f} at h=1 and "
          f"{coverage[10]:.3f} at h=10")
    assert 0.90 < coverage[1] < 0.99
    assert 0.90 < coverage[10] < 0.99

    bad = [
        lambda: NaiveForecaster("holt"),
        lambda: NaiveForecaster(season=0),
        lambda: NaiveForecaster("naive").predict(1),
        lambda: NaiveForecaster("naive").fit(y).predict(0),
        lambda: NaiveForecaster("naive").fit(y).predict(2.5),
        lambda: NaiveForecaster("naive").fit(y).interval(3, 0.0),
        lambda: NaiveForecaster("naive").fit(y).interval(3, 1.0),
        lambda: NaiveForecaster("naive").fit(y.reshape(-1, 1)),
        lambda: NaiveForecaster("naive").fit(np.array([1.0, np.nan, 2.0])),
        lambda: NaiveForecaster("naive").fit(np.array([1.0, 2.0])),
        lambda: NaiveForecaster("drift").fit(np.array([1.0, 2.0])),
        lambda: NaiveForecaster("mean").fit(np.array([1.0])),
        lambda: NaiveForecaster("seasonal_naive", season=12).fit(y[:10]),
        lambda: NaiveForecaster("snaive", season=200).fit(y),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a seasonal period below one, use before fit, a zero or")
    print("  fractional horizon, a level outside (0, 1), a two-dimensional")
    print("  series, a non-finite value, series too short for a variance, a")
    print("  season that leaves too few seasonal differences, and a season")
    print("  longer than the data")
    print("baselines demo passed")


if __name__ == "__main__":
    demo()
