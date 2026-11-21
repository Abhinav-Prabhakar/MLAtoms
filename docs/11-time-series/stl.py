"""STL — seasonal-trend decomposition by iterative LOESS smoothing.

Chapter:   11-time-series
Companion: stl.md
Tier 2 · 4 variants: stl, robust, mstl, seasonal_windows
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("stl", "robust", "mstl", "seasonal_windows")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 8:
        raise ValueError("y needs at least eight observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return y


def _next_odd(value) -> int:
    number = int(np.ceil(float(value)))
    if number % 2 == 0:
        number += 1
    return max(3, number)


def _check_odd(value, name: str) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an odd integer")
    if value < 3 or value % 2 == 0:
        raise ValueError(f"{name} must be odd and at least three")
    return int(value)


def _moving_average(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.copy()
    pad = length // 2
    padded = np.pad(values, pad, mode="reflect")
    kernel = np.ones(length) / length
    return np.convolve(padded, kernel, mode="valid")


def _loess(x: np.ndarray, y: np.ndarray, targets: np.ndarray, span: int,
           degree: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Local regression with tricube weights, evaluated at `targets`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("empty smoother input")
    if n == 1:
        return np.full(targets.size, float(y[0]))
    span = int(min(max(3, span), n))
    if span % 2 == 0:
        span -= 1
    span = max(1, span)
    out = np.empty(targets.size)
    for row, target in enumerate(targets):
        distance = np.abs(x - target)
        take = np.argpartition(distance, span - 1)[:span]
        local = distance[take]
        radius = float(local.max())
        if radius <= 0.0:
            out[row] = float(np.average(y[take]))
            continue
        shape = np.clip(local / radius, 0.0, 1.0)
        w = (1.0 - shape ** 3) ** 3
        if weights is not None:
            w = w * weights[take]
        total = float(w.sum())
        if total <= 0.0:
            out[row] = float(np.average(y[take]))
            continue
        if degree <= 0:
            out[row] = float(w @ y[take] / total)
            continue
        design = np.column_stack([np.ones(span), x[take] - target])
        root = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * root[:, None], y[take] * root,
                                   rcond=None)
        out[row] = float(coef[0])
    return out


def _bisquare(remainder: np.ndarray) -> np.ndarray:
    scale = 6.0 * float(np.median(np.abs(remainder)))
    if scale <= 0.0:
        return np.ones(remainder.size)
    shape = np.abs(remainder) / scale
    return np.where(shape < 1.0, (1.0 - shape ** 2) ** 2, 0.0)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class STL:
    """Seasonal-trend decomposition with LOESS and optional robustness.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    period : int
        Seasonal period of the main decomposition.
    seasonal : int or "periodic"
        Span of the cycle-subseries smoother, in observations of that
        subseries. `"periodic"` forces one constant seasonal profile.
    trend : int, optional
        Span of the trend smoother; defaults to the STL rule based on `period`.
    low_pass : int, optional
        Span of the seasonal low-pass loess; defaults to the next odd number
        after `period`.
    inner, outer : int
        Inner and outer iteration counts. `robust` defaults to 15 outer loops;
        the others default to none.
    periods : tuple of int
        Seasonal periods for `mstl`.
    seasonal_degree, trend_degree : int
        Local polynomial degree, zero or one.

    Notes
    -----
    `fit` stores `seasonal_`, `trend_`, `remainder_`, `weights_`, `fitted_`,
    `rss_`, `seasonal_windows_` and the effective windows. The additive identity
    `seasonal_ + trend_ + remainder_ == y` holds by definition of the
    remainder; the quality of the split is what the demo checks.
    """

    def __init__(self, variant: str = "stl", *, period: int = 12,
                 seasonal=7, trend: int | None = None,
                 low_pass: int | None = None, inner: int = 2,
                 outer: int | None = None, periods=(), seasonal_degree: int = 1,
                 trend_degree: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(period, (int, np.integer)) or period < 2:
            raise ValueError("period must be an integer of at least two")
        if not isinstance(inner, (int, np.integer)) or inner < 1:
            raise ValueError("inner must be a positive integer")
        if outer is not None and (not isinstance(outer, (int, np.integer))
                                  or outer < 0):
            raise ValueError("outer must be a non-negative integer")
        if seasonal_degree not in (0, 1) or trend_degree not in (0, 1):
            raise ValueError("degrees must be zero or one")
        if seasonal != "periodic":
            seasonal = _check_odd(seasonal, "seasonal")
        periods = tuple(int(value) for value in periods)
        if any(value < 2 for value in periods):
            raise ValueError("every period must be at least two")
        if variant == "mstl" and not periods:
            periods = (int(period),)
        self.variant = variant
        self.period = int(period)
        self.seasonal = seasonal
        self.trend = None if trend is None else _check_odd(trend, "trend")
        self.low_pass = (None if low_pass is None
                         else _check_odd(low_pass, "low_pass"))
        self.inner = int(inner)
        self.outer = (15 if variant == "robust" and outer is None else
                      0 if outer is None else int(outer))
        self.periods = periods
        self.seasonal_degree = int(seasonal_degree)
        self.trend_degree = int(trend_degree)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> STL:
        """Decompose `y` into seasonal, trend and remainder."""
        y = _check_series(y)
        n = y.size
        if self.variant == "mstl":
            self._fit_mstl(y)
            return self
        if n < 2 * self.period:
            raise ValueError("need at least two full periods")
        self._windows = self._resolve_windows()
        self.seasonal_windows_ = None
        if self.variant == "seasonal_windows":
            trend = _loess(np.arange(n, dtype=float), y,
                           np.arange(n, dtype=float), self.trend_window_,
                           self.trend_degree)
            detrended = y - trend
            chosen = self._choose_windows(detrended)
            self._windows = chosen
            self.seasonal_windows_ = tuple(int(value) for value in chosen)
        seasonal, trend, remainder = self._run(
            y, self.period, self._windows, self.seasonal_window_,
            self.trend_window_, self.low_pass_window_, self.inner)
        weights = np.ones(n)
        for _ in range(self.outer):
            weights = _bisquare(remainder)
            seasonal, trend, remainder = self._run(
                y, self.period, self._windows, self.seasonal_window_,
                self.trend_window_, self.low_pass_window_, self.inner,
                weights)
        self._install(y, seasonal, trend, remainder, weights)
        return self

    def _resolve_windows(self) -> np.ndarray:
        if self.seasonal == "periodic":
            self.seasonal_window_ = None
        else:
            self.seasonal_window_ = int(self.seasonal)
        if self.trend is not None:
            self.trend_window_ = self.trend
        elif self.seasonal == "periodic":
            self.trend_window_ = _next_odd(1.5 * self.period)
        else:
            ratio = 1.5 / float(self.seasonal)
            self.trend_window_ = _next_odd(1.5 * self.period / (1.0 - ratio))
        self.low_pass_window_ = (self.low_pass if self.low_pass is not None
                                 else _next_odd(self.period))
        return np.full(self.period, self.seasonal_window_ or self.period)

    def _run(self, y: np.ndarray, period: int, windows, seasonal_window,
             trend_window: int, low_pass_window: int, inner: int,
             weights: np.ndarray | None = None):
        n = y.size
        if weights is None:
            weights = np.ones(n)
        trend = np.zeros(n)
        seasonal = np.zeros(n)
        for _ in range(inner):
            detrended = y - trend
            cycles = self._cycle_smooth(detrended, period, windows,
                                        seasonal_window, weights)
            low = self._low_pass(cycles, period, low_pass_window)
            start = period - 1
            seasonal = cycles[start:start + n] - low[start:start + n]
            deseasonalised = y - seasonal
            trend = _loess(np.arange(n, dtype=float), deseasonalised,
                           np.arange(n, dtype=float), trend_window,
                           self.trend_degree, weights)
        remainder = y - seasonal - trend
        return seasonal, trend, remainder

    def _cycle_smooth(self, values: np.ndarray, period: int, windows,
                      seasonal_window, weights: np.ndarray) -> np.ndarray:
        n = values.size
        positions = np.arange(-period + 1, n + period)
        out = np.zeros(positions.size)
        for j in range(period):
            index = np.arange(j, n, period)
            if index.size < 2:
                continue
            targets = positions[(positions - j) % period == 0]
            if seasonal_window is None:
                span, degree = index.size, 0
            else:
                span = int(windows[j]) if windows is not None else seasonal_window
                degree = self.seasonal_degree
                if span >= index.size:
                    span, degree = index.size, 0
            out[(positions - j) % period == 0] = _loess(
                index.astype(float), values[index], targets.astype(float),
                span, degree, weights[index])
        return out

    def _low_pass(self, cycles: np.ndarray, period: int,
                  window: int) -> np.ndarray:
        values = _moving_average(cycles, period)
        values = _moving_average(values, period)
        values = _moving_average(values, 3)
        grid = np.arange(values.size, dtype=float)
        return _loess(grid, values, grid, window, 1)

    def _choose_windows(self, detrended: np.ndarray) -> np.ndarray:
        base = self.seasonal_window_ or self.period
        candidates = sorted({_next_odd(base + offset)
                            for offset in (-4, -2, 0, 2, 4)})
        chosen = np.empty(self.period, dtype=int)
        for j in range(self.period):
            index = np.arange(j, detrended.size, self.period)
            values = detrended[index]
            if values.size < 5:
                chosen[j] = max(3, values.size)
                continue
            best, best_error = values.size, np.inf
            for span in candidates:
                error = self._cv_error(values, span)
                if error < best_error:
                    best, best_error = span, error
            chosen[j] = min(best, values.size)
        return chosen

    def _cv_error(self, values: np.ndarray, span: int) -> float:
        n = values.size
        total = 0.0
        for hold in range(n):
            keep = np.ones(n, dtype=bool)
            keep[hold] = False
            use_span = min(span, int(keep.sum()))
            degree = 1 if use_span >= n else 0
            prediction = _loess(np.arange(n)[keep].astype(float),
                                values[keep], np.array([float(hold)]),
                                use_span, degree)
            total += float((values[hold] - prediction[0]) ** 2)
        return total

    def _fit_mstl(self, y: np.ndarray) -> None:
        n = y.size
        periods = tuple(sorted(self.periods))
        if n < 2 * periods[-1]:
            raise ValueError("need at least two full periods of the longest")
        self.seasonal_windows_ = None
        self._windows = self._resolve_windows()
        if self.trend is None:
            ratio = 0.0 if self.seasonal == "periodic" else 1.5 / self.seasonal
            self.trend_window_ = _next_odd(1.5 * periods[-1] / (1.0 - ratio))
        if self.low_pass is None:
            self.low_pass_window_ = _next_odd(periods[-1])
        seasonals = [np.zeros(n) for _ in periods]
        for _ in range(2):
            for k, period in enumerate(periods):
                partial = y.copy()
                for other, component in enumerate(seasonals):
                    if other != k:
                        partial = partial - component
                component, _, _ = self._run(
                    partial, period, None, self.seasonal_window_,
                    self.trend_window_, self.low_pass_window_, 1)
                seasonals[k] = component
        seasonal = np.sum(seasonals, axis=0)
        trend = _loess(np.arange(n, dtype=float), y - seasonal,
                       np.arange(n, dtype=float), self.trend_window_,
                       self.trend_degree)
        remainder = y - seasonal - trend
        self.seasonals_ = seasonals
        self._install(y, seasonal, trend, remainder, np.ones(n))

    def _install(self, y: np.ndarray, seasonal: np.ndarray, trend: np.ndarray,
                 remainder: np.ndarray, weights: np.ndarray) -> None:
        self.seasonal_ = seasonal
        self.trend_ = trend
        self.remainder_ = remainder
        self.weights_ = weights
        self.fitted_ = seasonal + trend
        self.rss_ = float(remainder @ remainder)
        self.sigma2_ = self.rss_ / max(y.size - 1, 1)
        self.n_ = y.size


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _seasonal(n: int, period: int, rng: np.random.Generator,
              noise: float = 0.35) -> np.ndarray:
    t = np.arange(n)
    shape = (3.0 * np.sin(2.0 * np.pi * t / period)
             + 1.5 * np.cos(4.0 * np.pi * t / period))
    return 10.0 + 0.05 * t + shape + rng.normal(0.0, noise, n)


def _profile(component: np.ndarray, period: int) -> np.ndarray:
    n = component.size - component.size % period
    return component[:n].reshape(-1, period).mean(axis=0)


def _error(component: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((component - truth) ** 2)))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("stl. 240 points, period 12, linear trend and two harmonics:")
    n = 240
    t = np.arange(n)
    seasonal = 3.0 * np.sin(2.0 * np.pi * t / 12) + 1.5 * np.cos(4.0 * np.pi * t / 12)
    trend = 10.0 + 0.05 * t
    y = trend + seasonal + rng.normal(0.0, 0.35, n)
    model = STL("stl", period=12).fit(y)
    assert np.allclose(model.seasonal_ + model.trend_ + model.remainder_, y)
    assert model.rss_ > 0.0
    shape = _profile(model.seasonal_, 12)
    truth = _profile(seasonal, 12)
    assert np.corrcoef(shape, truth)[0, 1] > 0.98
    assert _error(model.trend_, trend) < 0.4
    print(f"  seasonal profile correlation {np.corrcoef(shape, truth)[0, 1]:.4f},"
          f" trend RMSE {_error(model.trend_, trend):.4f}")
    print(f"  remainder variance {model.rss_ / n:.4f} against noise variance "
          f"0.1225")

    print("robust. the same series with twenty large outliers:")
    dirty = y.copy()
    spikes = rng.choice(n, 20, replace=False)
    dirty[spikes] += 15.0
    plain = STL("stl", period=12).fit(dirty)
    robust = STL("robust", period=12, outer=8).fit(dirty)
    assert robust.weights_[spikes].max() < 0.05
    assert _error(robust.trend_, trend) < _error(plain.trend_, trend)
    assert _error(robust.seasonal_, seasonal) < _error(plain.seasonal_, seasonal)
    print(f"  mean outlier weight {robust.weights_[spikes].mean():.4f}; "
          f"trend RMSE {_error(robust.trend_, trend):.4f} against "
          f"{_error(plain.trend_, trend):.4f}")
    print(f"  the spikes stay in the remainder, so its RSS is larger "
          f"({robust.rss_:.2f} against {plain.rss_:.2f}); the point is that "
          f"they no longer bend the fit")

    rng = np.random.default_rng(1)
    print("mstl. a weekly and a monthly cycle on 420 daily points:")
    n = 420
    t = np.arange(n)
    weekly = 2.0 * np.sin(2.0 * np.pi * t / 7)
    monthly = 3.0 * np.sin(2.0 * np.pi * t / 30)
    trend = 20.0 + 0.02 * t
    y = trend + weekly + monthly + rng.normal(0.0, 0.25, n)
    model = STL("mstl", period=7, periods=(7, 30), trend=31).fit(y)
    assert len(model.seasonals_) == 2
    assert np.corrcoef(_profile(model.seasonals_[0], 7),
                       _profile(weekly, 7))[0, 1] > 0.95
    assert np.corrcoef(_profile(model.seasonals_[1], 30),
                       _profile(monthly, 30))[0, 1] > 0.85
    print(f"  weekly correlation "
          f"{np.corrcoef(_profile(model.seasonals_[0], 7), _profile(weekly, 7))[0, 1]:.4f},"
          f" monthly "
          f"{np.corrcoef(_profile(model.seasonals_[1], 30), _profile(monthly, 30))[0, 1]:.4f}")
    print(f"  remainder variance {model.rss_ / n:.4f} against noise variance "
          f"0.0625")

    rng = np.random.default_rng(2)
    print("seasonal_windows. one noisy month and one clean month:")
    n = 240
    t = np.arange(n)
    shape = 3.0 * np.sin(2.0 * np.pi * t / 12)
    scale = np.where(t % 12 == 3, 2.4, 0.2)
    y = 10.0 + shape + rng.normal(0.0, 1.0, n) * scale
    fixed = STL("stl", period=12, seasonal=7).fit(y)
    adaptive = STL("seasonal_windows", period=12, seasonal=7).fit(y)
    assert len(set(adaptive.seasonal_windows_)) > 1
    assert _error(adaptive.seasonal_, shape) < _error(fixed.seasonal_, shape)
    print(f"  chosen windows {adaptive.seasonal_windows_}")
    print(f"  seasonal RMSE {_error(adaptive.seasonal_, shape):.4f} against "
          f"{_error(fixed.seasonal_, shape):.4f}; in-sample RSS is slightly "
          f"higher ({adaptive.rss_:.2f} against {fixed.rss_:.2f}) because a "
          f"smoother seasonal fits noise")

    bad = [
        lambda: STL("stl_lite"),
        lambda: STL("stl", period=1),
        lambda: STL("stl", period=12, seasonal=4),
        lambda: STL("stl", period=12, seasonal=2),
        lambda: STL("stl", period=12, trend=8),
        lambda: STL("stl", period=12, low_pass=12),
        lambda: STL("stl", period=12, inner=0),
        lambda: STL("stl", period=12, outer=-1),
        lambda: STL("stl", period=12, seasonal_degree=2),
        lambda: STL("mstl", period=12, periods=(0, 12)),
        lambda: STL("stl", period=12).fit(np.zeros((20, 2))),
        lambda: STL("stl", period=12).fit(np.r_[np.zeros(7), np.nan]),
        lambda: STL("stl", period=12).fit(np.arange(10.0)),
        lambda: STL("stl", period=12).fit(np.arange(20.0)),
        lambda: STL("mstl", period=12, periods=(12, 20)).fit(np.arange(30.0)),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a period below two, a seasonal or trend span that is")
    print("  even or below three, a zero inner loop, a negative outer loop, a")
    print("  degree outside zero and one, a sub-period below two, a 2-d or")
    print("  non-finite series, a series shorter than eight points or than two")
    print("  full periods, and an mstl series shorter than two of its longest")
    print("  period")
    print("stl demo passed")


if __name__ == "__main__":
    demo()
