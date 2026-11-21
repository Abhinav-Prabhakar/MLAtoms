"""Theta — exponential smoothing with a drift built from a theta line.

Chapter:   11-time-series
Companion: theta.md
Tier 3 · 4 variants: theta, optimised_theta, dstm, dynamic_theta
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

VARIANTS = ("theta", "optimised_theta", "dstm", "dynamic_theta")
THETA_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
DYNAMIC_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 20:
        raise ValueError("y needs at least twenty observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return y


def _check_alpha(value) -> float:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError("alpha must be a number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    return value


def _check_theta(value) -> float:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError("theta must be a number")
    value = float(value)
    if not np.isfinite(value) or value < 1.0:
        raise ValueError("theta must be finite and at least one")
    return value


def _line(y: np.ndarray) -> tuple[float, float]:
    """Ordinary least-squares intercept and slope against t = 1..n."""
    n = y.size
    grid = np.arange(1, n + 1, dtype=float)
    centred = grid - grid.mean()
    slope = float(centred @ (y - y.mean()) / (centred @ centred))
    intercept = float(y.mean() - slope * grid.mean())
    return intercept, slope


# --------------------------------------------------------------------------- #
# The two engines
# --------------------------------------------------------------------------- #
def _static(y: np.ndarray, theta: float, alpha: float,
            ell0: float | None = None) -> dict:
    """Theta-line decomposition with a fixed OLS line and SES on the line."""
    n = y.size
    grid = np.arange(1, n + 1, dtype=float)
    intercept, slope = _line(y)
    line = intercept + slope * grid
    z = theta * y + (1.0 - theta) * line
    weight = 1.0 - 1.0 / theta
    powers = (1.0 - alpha) ** np.arange(n)
    mean = np.empty(n)
    coefficient = np.empty(n)
    level = 0.0
    for index in range(n):
        mean[index] = level / theta + weight * line[index]
        coefficient[index] = powers[index] / theta
        level = alpha * z[index] + (1.0 - alpha) * level
    if ell0 is None:
        ell0 = float(np.sum(coefficient * (y - mean))
                     / np.sum(coefficient * coefficient))
    fitted = mean + coefficient * ell0
    residuals = y - fitted
    final = ell0 * powers[-1] * (1.0 - alpha) + level
    return {"theta": theta, "alpha": alpha, "ell0": ell0, "level": final,
            "line": (intercept, slope), "n": n, "fitted": fitted,
            "residuals": residuals, "burn": 0,
            "sse": float(residuals @ residuals)}


def _dynamic(y: np.ndarray, theta: float, alpha: float,
             ell0: float | None = None) -> dict:
    """Fiorucci's dynamic theta state space with a free theta."""
    n = y.size
    weight = 1.0 - 1.0 / theta
    powers = (1.0 - alpha) ** np.arange(n)
    mean = np.empty(n)
    coefficient = np.empty(n)
    level = 0.0
    slope_state = 0.0
    intercept_state = 0.0
    average = 0.0
    for index in range(n):
        t = index + 1
        drift = weight * (
            powers[index] * intercept_state
            + (1.0 - powers[index] * (1.0 - alpha)) / alpha * slope_state)
        mean[index] = level + drift
        coefficient[index] = powers[index]
        level = alpha * y[index] + (1.0 - alpha) * level
        average_previous = average
        average = ((t - 1) * average_previous + y[index]) / t
        slope_state = ((t - 2) * slope_state
                       + 6.0 / t * (y[index] - average_previous)) / (t + 1)
        intercept_state = average - 0.5 * (t + 1) * slope_state
    if ell0 is None:
        ell0 = float(np.sum(coefficient * (y - mean))
                     / np.sum(coefficient * coefficient))
    fitted = mean + coefficient * ell0
    residuals = y - fitted
    final = ell0 * powers[-1] * (1.0 - alpha) + level
    burn = min(n - 1, max(3, int(np.ceil(2.0 / alpha))))
    used = residuals[burn:]
    return {"theta": theta, "alpha": alpha, "ell0": ell0, "level": final,
            "states": (intercept_state, slope_state, average), "fitted": fitted,
            "residuals": residuals, "burn": burn,
            "sse": float(used @ used)}


def _search_static(y: np.ndarray, theta: float,
                   ell0: float | None = None) -> dict:
    best = None
    for alpha in np.arange(0.02, 0.99, 0.02):
        result = _static(y, theta, float(alpha), ell0)
        if best is None or result["sse"] < best["sse"]:
            best = result
    assert best is not None
    centre = best["alpha"]
    for step in (0.01, 0.005, 0.002):
        for alpha in (centre - step, centre + step):
            if 0.0 < alpha <= 1.0:
                result = _static(y, theta, float(alpha), ell0)
                if result["sse"] < best["sse"]:
                    best = result
        centre = best["alpha"]
    return best


def _search_dynamic(y: np.ndarray, theta: float,
                    ell0: float | None = None) -> dict:
    best = None
    for alpha in np.arange(0.02, 0.99, 0.02):
        result = _dynamic(y, theta, float(alpha), ell0)
        if best is None or result["sse"] < best["sse"]:
            best = result
    assert best is not None
    centre = best["alpha"]
    for step in (0.01, 0.005, 0.002):
        for alpha in (centre - step, centre + step):
            if 0.0 < alpha <= 1.0:
                result = _dynamic(y, theta, float(alpha), ell0)
                if result["sse"] < best["sse"]:
                    best = result
        centre = best["alpha"]
    return best


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class ThetaForecaster:
    """Theta method, optimised theta and the two dynamic state-space forms.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    theta : float, optional
        Theta parameter. `theta` and `dstm` fix it at two; passing another
        value is an error. `optimised_theta` and `dynamic_theta` search
        `THETA_GRID` or `DYNAMIC_GRID` when this is `None`, and use the given
        value when it is not.
    alpha : float, optional
        SES smoothing parameter in `(0, 1]`; `None` fits it by minimising the
        one-step sum of squares.

    Notes
    -----
    `fit` stores `alpha_`, `theta_`, `ell0_`, `level_`, `fitted_`,
    `residuals_`, `sse_`, `sigma_`, `line_` for the static variants and
    `states_` for the dynamic ones, plus `candidates_`, a list of
    `(theta, alpha, sse)` triples visited by the search.
    """

    def __init__(self, variant: str = "theta", *, theta: float | None = None,
                 alpha: float | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        if theta is not None:
            theta = _check_theta(theta)
            if variant in ("theta", "dstm") and theta != 2.0:
                raise ValueError(f"{variant} fixes theta at two")
        self.theta = theta
        self.alpha = None if alpha is None else _check_alpha(alpha)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> ThetaForecaster:
        """Estimate the smoothing and theta parameters."""
        y = _check_series(y)
        dynamic = self.variant in ("dstm", "dynamic_theta")
        if self.variant in ("theta", "dstm"):
            grid = (2.0,)
        elif self.variant == "optimised_theta":
            grid = THETA_GRID
        else:
            grid = DYNAMIC_GRID
        if self.theta is not None:
            grid = (self.theta,)
        search = _search_dynamic if dynamic else _search_static
        best = None
        candidates = []
        for theta in grid:
            if self.alpha is not None:
                engine = _dynamic if dynamic else _static
                result = engine(y, theta, self.alpha)
            else:
                result = search(y, theta)
            candidates.append((float(theta), result["alpha"], result["sse"]))
            if best is None or result["sse"] < best["sse"]:
                best = result
        assert best is not None
        self.theta_ = float(best["theta"])
        self.alpha_ = float(best["alpha"])
        self.ell0_ = float(best["ell0"])
        self.level_ = float(best["level"])
        self.fitted_ = best["fitted"]
        self.residuals_ = best["residuals"]
        self.sse_ = float(best["sse"])
        self.burn_ = int(best["burn"])
        self.sigma_ = float(np.sqrt(self.sse_ / (y.size - self.burn_)))
        self.candidates_ = candidates
        self.n_ = y.size
        if dynamic:
            self.states_ = best["states"]
        else:
            self.line_ = best["line"]
        return self

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int = 12) -> np.ndarray:
        """Forecast `horizon` steps beyond the fitted sample."""
        if not hasattr(self, "theta_"):
            raise RuntimeError("fit the model before predicting")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if self.variant in ("theta", "optimised_theta"):
            intercept, slope = self.line_
            weight = 1.0 - 1.0 / self.theta_
            steps = np.arange(1, horizon + 1)
            line = intercept + slope * (self.n_ + steps)
            return self.level_ / self.theta_ + weight * line
        weight = 1.0 - 1.0 / self.theta_
        intercept, slope, average = self.states_
        level = self.level_
        out = np.empty(horizon)
        for step in range(1, horizon + 1):
            t = self.n_ + step
            decay = (1.0 - self.alpha_) ** (t - 1)
            drift = weight * (
                decay * intercept
                + (1.0 - (1.0 - self.alpha_) ** t) / self.alpha_ * slope)
            mean = level + drift
            out[step - 1] = mean
            level = self.alpha_ * mean + (1.0 - self.alpha_) * level
            average_previous = average
            average = ((t - 1) * average_previous + mean) / t
            slope = ((t - 2) * slope
                     + 6.0 / t * (mean - average_previous)) / (t + 1)
            intercept = average - 0.5 * (t + 1) * slope
        return out

    def interval(self, horizon: int = 12, level: float = 0.95
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian band with the dynamic-theta variance recursion."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between zero and one")
        forecast = self.predict(horizon)
        steps = np.arange(horizon)
        scale = np.sqrt(1.0 + steps * self.alpha_ ** 2) * self.sigma_
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return forecast - z * scale, forecast + z * scale


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _trend_series(n: int, rng: np.random.Generator, *, slope: float = 0.05,
                  curve: float = 0.0, wave: float = 0.0,
                  noise: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=float)
    trend = 10.0 + slope * t + curve * t * t
    if wave:
        trend = trend + wave * np.sin(2.0 * np.pi * t / 120.0)
    return trend + rng.normal(0.0, noise, n)


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def _one_step_mse(model: ThetaForecaster) -> float:
    used = model.residuals_[model.burn_:]
    return float(np.mean(used ** 2))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("theta. 264 points, linear trend and noise, 24 held out:")
    values = _trend_series(264, rng)
    split = 240
    model = ThetaForecaster("theta").fit(values[:split])
    forecast = model.predict(24)
    naive = np.repeat(values[split - 1], 24)
    error = _mae(values[split:], forecast)
    baseline = _mae(values[split:], naive)
    assert error < baseline
    assert 0.0 < model.alpha_ < 1.0
    print(f"  alpha {model.alpha_:.4f}, initial level {model.ell0_:.4f}, "
          f"one-step sigma {model.sigma_:.4f}")
    print(f"  held-out MAE {error:.4f} against {baseline:.4f} for naive")

    print("  theta equals SES with a half-slope drift when the initial")
    print("  level sits on the OLS line:")
    alpha = 0.3
    intercept, slope = _line(values)
    first_line = intercept + slope
    line_model = _static(values, 2.0, alpha, ell0=first_line)
    level = first_line
    for value in values:
        level = alpha * value + (1.0 - alpha) * level
    closed = level + 0.5 * slope * (
        np.arange(1, 25) - 1.0 + 1.0 / alpha)
    difference = float(np.max(np.abs(_forecast_static(line_model, 24) - closed)))
    assert difference < 1e-9
    print(f"  maximum difference from the closed form {difference:.2e}")

    rng = np.random.default_rng(1)
    print("optimised_theta. a quadratic trend, 24 held out:")
    values = _trend_series(264, rng, curve=0.0004, noise=0.6)
    plain = ThetaForecaster("theta").fit(values[:split])
    optimised = ThetaForecaster("optimised_theta").fit(values[:split])
    assert optimised.sse_ <= plain.sse_ + 1e-9
    plain_error = _mae(values[split:], plain.predict(24))
    optimised_error = _mae(values[split:], optimised.predict(24))
    assert optimised_error <= plain_error
    print(f"  theta 2 SSE {plain.sse_:.2f}, chosen theta {optimised.theta_:.2f} "
          f"with SSE {optimised.sse_:.2f}")
    print(f"  held-out MAE {optimised_error:.4f} against {plain_error:.4f} "
          f"at theta two")

    rng = np.random.default_rng(2)
    print("dstm. a slow wave in the trend, 24 held out:")
    values = _trend_series(264, rng, slope=0.03, wave=3.0, noise=0.5)
    static = ThetaForecaster("theta").fit(values[:split])
    dynamic = ThetaForecaster("dstm").fit(values[:split])
    static_error = _mae(values[split:], static.predict(24))
    dynamic_error = _mae(values[split:], dynamic.predict(24))
    assert _one_step_mse(dynamic) < _one_step_mse(static)
    print(f"  alpha {dynamic.alpha_:.4f}, one-step MSE "
          f"{_one_step_mse(dynamic):.4f} against "
          f"{_one_step_mse(static):.4f} for the static model")
    print(f"  held-out 24-step MAE {dynamic_error:.4f} against "
          f"{static_error:.4f}")

    rng = np.random.default_rng(3)
    print("dynamic_theta. a slope change halfway through the sample:")
    t = np.arange(264, dtype=float)
    trend = np.where(t < 140, 10.0 + 0.02 * t, 12.8 + 0.09 * (t - 140))
    values = trend + rng.normal(0.0, 0.5, t.size)
    standard = ThetaForecaster("dstm").fit(values[:split])
    free = ThetaForecaster("dynamic_theta").fit(values[:split])
    assert free.sse_ <= standard.sse_ + 1e-9
    print(f"  chosen theta {free.theta_:.2f} at alpha {free.alpha_:.4f}, "
          f"SSE {free.sse_:.2f} against {standard.sse_:.2f}")
    print(f"  held-out MAE {_mae(values[split:], free.predict(24)):.4f} "
          f"against {_mae(values[split:], standard.predict(24)):.4f}")

    print("  an 80% interval widens with the horizon:")
    lower, upper = model.interval(24, 0.8)
    assert np.all(upper > lower)
    assert (upper - lower)[-1] > (upper - lower)[0]
    print(f"  width at h=1 {(upper - lower)[0]:.4f}, at h=24 "
          f"{(upper - lower)[-1]:.4f}")

    bad = [
        lambda: ThetaForecaster("theta_lite"),
        lambda: ThetaForecaster("theta", theta=0.5),
        lambda: ThetaForecaster("theta", theta=np.nan),
        lambda: ThetaForecaster("theta", theta=3.0),
        lambda: ThetaForecaster("dstm", theta=1.5),
        lambda: ThetaForecaster("theta", alpha=0.0),
        lambda: ThetaForecaster("theta", alpha=1.5),
        lambda: ThetaForecaster("theta", alpha=np.nan),
        lambda: ThetaForecaster("theta").fit(np.zeros((40, 2))),
        lambda: ThetaForecaster("theta").fit(np.arange(10.0)),
        lambda: ThetaForecaster("theta").fit(np.r_[np.arange(20.0), np.nan]),
        lambda: ThetaForecaster("theta").predict(12),
        lambda: ThetaForecaster("theta").fit(np.arange(40.0)).predict(0),
        lambda: ThetaForecaster("theta").fit(np.arange(40.0)).predict(1.5),
        lambda: ThetaForecaster("theta").fit(np.arange(40.0)).interval(
            12, 1.0),
        lambda: ThetaForecaster("theta").fit(np.arange(40.0)).interval(
            12, 0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a theta below one or non-finite, a theta other than two")
    print("  for the fixed variants, an alpha outside (0, 1] or non-finite, a")
    print("  2-d or short series, a non-finite series, prediction before")
    print("  fitting, a zero or fractional horizon, and an interval level")
    print("  outside the open unit interval")
    print("theta demo passed")


def _forecast_static(result: dict, horizon: int) -> np.ndarray:
    """Forecast from a `_static` result, used by the equivalence check."""
    intercept, slope = result["line"]
    steps = np.arange(1, horizon + 1)
    weight = 1.0 - 1.0 / result["theta"]
    return (result["level"] / result["theta"]
            + weight * (intercept + slope * (result["n"] + steps)))


if __name__ == "__main__":
    demo()
