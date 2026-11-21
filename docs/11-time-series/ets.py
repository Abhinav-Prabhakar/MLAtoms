"""ETS exponential smoothing — level, trend and season in one recursion.

Chapter:   11-time-series
Companion: ets.md
Tier 1 · 7 variants: simple_es, holt, damped_trend, holt_winters_add,
                     holt_winters_mul, ets_state_space, auto_ets
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

VARIANTS = ("simple_es", "holt", "damped_trend", "holt_winters_add",
            "holt_winters_mul", "ets_state_space", "auto_ets")

# (trend, season) letters of the ETS taxonomy; the error is always additive.
SPECS = {
    "simple_es": ("N", "N"),
    "holt": ("A", "N"),
    "damped_trend": ("Ad", "N"),
    "holt_winters_add": ("A", "A"),
    "holt_winters_mul": ("A", "M"),
    "ets_state_space": ("Ad", "A"),
}

GRID = (("N", "N"), ("A", "N"), ("Ad", "N"),
        ("N", "A"), ("A", "A"), ("Ad", "A"),
        ("N", "M"), ("A", "M"), ("Ad", "M"))

LIMITS = {"alpha": (0.01, 0.999), "beta": (0.01, 0.999),
          "gamma": (0.01, 0.999), "phi": (0.80, 0.995)}

STARTS = {"alpha": 0.30, "beta": 0.10, "gamma": 0.10, "phi": 0.95}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 4:
        raise ValueError("y needs at least four observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    return y


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class ExponentialSmoothing:
    """Holt-Winters recursions, their state-space form and model selection.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    season : int
        Seasonal period `m`. Only the seasonal variants, `ets_state_space`
        and `auto_ets` may use a value above one.
    alpha, beta, gamma, phi : float or None
        Smoothing weights and the damping factor. `None` estimates the
        parameter by minimising the one-step squared error; an explicit value
        is used as given, which is how the demo checks closed forms.
    Notes
    -----
    `fit` stores `fitted_`, `residuals_`, `params_`, `state_`, `sse_`,
    `sigma2_`, `loglik_`, `aicc_` and `model_` such as `"ETS(A,Ad,A)"`.
    For multiplicative seasonality `sigma2_` is the variance of the relative
    error `e_t / yhat_t`, not of the absolute one.
    `predict` returns point forecasts, `interval` a Gaussian band from the
    forecast-variance recursion, and `state_space` the matrices `(F, w, g, x)`
    of the linear form, which exists only for additive seasonality.
    """

    def __init__(self, variant: str = "holt_winters_add", *, season: int = 1,
                 alpha: float | None = None, beta: float | None = None,
                 gamma: float | None = None, phi: float | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(season, (int, np.integer)) or season < 1:
            raise ValueError("season must be a positive integer")
        if variant in ("simple_es", "holt", "damped_trend") and season != 1:
            raise ValueError("this variant has no seasonal component")
        for name, value in (("alpha", alpha), ("beta", beta),
                            ("gamma", gamma), ("phi", phi)):
            bad = value is not None and not (
                np.isfinite(value) and 0.0 < value <= 1.0)
            if bad:
                raise ValueError(f"{name} must lie in (0, 1]")
        self.variant = variant
        self.season = int(season)
        self.alpha = None if alpha is None else float(alpha)
        self.beta = None if beta is None else float(beta)
        self.gamma = None if gamma is None else float(gamma)
        self.phi = None if phi is None else float(phi)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> ExponentialSmoothing:
        """Estimate the states and weights, then record the fitted model."""
        y = _check_series(y)
        candidates = self._candidates(y)
        best, table = None, []
        for trend, season in candidates:
            fitted = self._fit_spec(y, trend, season)
            table.append((f"ETS(A,{trend},{season})", fitted["aicc"]))
            if best is None or fitted["aicc"] < best["aicc"]:
                best = fitted
        self.model_ = f"ETS(A,{best['trend']},{best['season']})"
        self.trend_ = best["trend"]
        self.season_ = best["season"]
        self.season_period_ = best["m"]
        self.params_ = best["params"]
        self.state_ = best["state"]
        self.fitted_ = best["fitted"]
        self.residuals_ = best["errors"]
        self.n_ = y.size
        self.start_ = best["start"]
        self.sse_ = best["sse"]
        self.sigma2_ = best["sigma2"]
        self.loglik_ = best["loglik"]
        self.aic_ = best["aic"]
        self.aicc_ = best["aicc"]
        self.candidates_ = table
        return self

    def _candidates(self, y: np.ndarray) -> list[tuple[str, str]]:
        if self.variant != "auto_ets":
            trend, season = SPECS[self.variant]
            return [(trend, season if self.season > 1 else "N")]
        grid = []
        for trend, season in GRID:
            if season != "N" and (self.season < 2
                                  or y.size < 2 * self.season + 2):
                continue
            grid.append((trend, season))
        if not grid:
            raise ValueError("series is too short for any candidate model")
        return grid

    def _fit_spec(self, y: np.ndarray, trend: str, season: str) -> dict:
        m = self.season if season != "N" else 1
        n = y.size
        if trend != "N" and n < 6:
            raise ValueError("a trend needs at least six observations")
        if season != "N" and n < 2 * m + 2:
            raise ValueError("a season needs two full cycles plus two points")
        if season == "M" and np.any(y <= 0.0):
            raise ValueError("multiplicative seasonality needs positive y")
        init = self._init_states(y, trend, season, m)
        names = ["alpha"]
        if trend != "N":
            names.append("beta")
        if season != "N":
            names.append("gamma")
        if trend == "Ad":
            names.append("phi")
        given = {"alpha": self.alpha, "beta": self.beta,
                 "gamma": self.gamma, "phi": self.phi}
        for name in ("beta", "gamma", "phi"):
            if name not in names and given[name] is not None:
                raise ValueError(f"{name} is not used by ETS(A,{trend},{season})")
        params = {name: (STARTS[name] if given[name] is None else given[name])
                  for name in names}
        free = [name for name in names if given[name] is None]
        if free:
            params = self._optimise(y, trend, season, m, init, params, free)
        run = self._run(y, trend, season, m, init, params)
        if run is None:
            raise ValueError("the recursion produced a non-positive level")
        fitted, errors, state, start = run
        n_eff = n - start
        if n_eff < 2:
            raise ValueError("too few usable residuals for this model")
        sse = float(np.sum(errors[start:] ** 2))
        if season == "M":
            relative = errors[start:] / fitted[start:]
            sigma2 = float(np.mean(relative ** 2))
            loglik = (-0.5 * n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0)
                      - float(np.sum(np.log(np.abs(fitted[start:])))))
        else:
            sigma2 = sse / n_eff
            loglik = -0.5 * n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0)
        k = (len(names) + 1 + (1 + (trend != "N") + (m if season != "N" else 0)))
        aic = -2.0 * loglik + 2.0 * k
        aicc = aic + 2.0 * k * (k + 1) / max(n_eff - k - 1, 1)
        return {"trend": trend, "season": season, "m": m, "params": params,
                "state": state, "fitted": fitted, "errors": errors,
                "start": start, "sse": sse, "sigma2": sigma2,
                "loglik": float(loglik), "aic": float(aic),
                "aicc": float(aicc)}

    def _init_states(self, y, trend, season, m) -> dict:
        n = y.size
        if season != "N":
            base = float(y[:m].mean())
            seasonal = (y[:m] / base if season == "M" else y[:m] - base)
            level = base
            if trend != "N" and n >= 2 * m:
                slope = (float(y[m:2 * m].mean()) - base) / m
            elif trend != "N":
                slope = (float(y[-1]) - float(y[0])) / (n - 1)
            else:
                slope = 0.0
        else:
            level = float(y[0])
            slope = ((float(y[-1]) - float(y[0])) / (n - 1)
                     if trend != "N" else 0.0)
            seasonal = np.zeros(1)
        return {"level": float(level), "trend": float(slope),
                "season": np.asarray(seasonal, dtype=float)}

    def _optimise(self, y, trend, season, m, init, params, free) -> dict:
        def loss(values):
            run = self._run(y, trend, season, m, init, dict(params, **values))
            if run is None:
                return np.inf
            return float(np.sum(run[1][run[3]:] ** 2))

        values = {name: float(params[name]) for name in free}
        steps = {name: 0.2 for name in free}
        best = loss(values)
        for _ in range(90):
            improved = False
            for name in free:
                low, high = LIMITS[name]
                for delta in (steps[name], -steps[name]):
                    trial = dict(values)
                    trial[name] = float(np.clip(values[name] + delta,
                                                low, high))
                    if trial[name] == values[name]:
                        continue
                    score = loss(trial)
                    if score < best - 1e-12:
                        values, best, improved = trial, score, True
            if not improved:
                for name in free:
                    steps[name] *= 0.5
                if max(steps.values()) < 1e-3:
                    break
        return dict(params, **values)

    def _run(self, y, trend, season, m, init, params):
        alpha = params["alpha"]
        beta = params.get("beta", 0.0)
        gamma = params.get("gamma", 0.0)
        phi = params.get("phi", 1.0)
        level = init["level"]
        slope = init["trend"]
        seasons = init["season"].copy()
        fitted = np.zeros(y.size)
        errors = np.zeros(y.size)
        with_season = season != "N"
        for t in range(y.size):
            season_now = float(seasons[0]) if with_season else 0.0
            base = level + phi * slope
            if season == "A":
                forecast = base + season_now
            elif season == "M":
                if base <= 1e-9 or season_now <= 1e-9:
                    return None
                forecast = base * season_now
            else:
                forecast = base
            error = float(y[t]) - forecast
            level = base + alpha * (error if season != "M"
                                    else error / season_now)
            slope = phi * slope + (beta * (error if season != "M"
                                           else error / season_now))
            if season == "A":
                seasons = np.concatenate([seasons[1:], [season_now + gamma * error]])
            elif season == "M":
                seasons = np.concatenate([seasons[1:],
                                          [season_now + gamma * error / base]])
            fitted[t] = forecast
            errors[t] = error
        if not (np.all(np.isfinite(fitted)) and np.all(np.isfinite(errors))):
            return None
        start = m if with_season else 1
        state = {"level": float(level), "trend": float(slope),
                 "season": seasons}
        return fitted, errors, state, start

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int) -> np.ndarray:
        """Point forecasts, shape `(horizon,)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        level = self.state_["level"]
        slope = self.state_["trend"]
        seasons = self.state_["season"].copy()
        phi = self.params_.get("phi", 1.0)
        out = np.empty(horizon)
        for h in range(horizon):
            base = level + phi * slope
            if self.season_ == "A":
                out[h] = base + seasons[h % self.season_period_]
            elif self.season_ == "M":
                out[h] = base * seasons[h % self.season_period_]
            else:
                out[h] = base
            level = base
            slope = phi * slope
        return out

    def interval(self, horizon: int, level: float = 0.95
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian band from the forecast-variance recursion."""
        self._check_fitted()
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        forecast = self.predict(horizon)
        steps = np.arange(1, horizon + 1, dtype=float)
        if self.season_ == "M":
            sd = forecast * np.sqrt(self.sigma2_) * np.sqrt(steps)
        else:
            matrix, w, g, _ = self.state_space()
            covariance = np.zeros_like(matrix)
            variance = np.empty(horizon)
            for h in range(horizon):
                variance[h] = float(w @ covariance @ w) + self.sigma2_
                covariance = (matrix @ covariance @ matrix.T
                              + self.sigma2_ * np.outer(g, g))
            sd = np.sqrt(variance)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return forecast - z * sd, forecast + z * sd

    # -- state space ------------------------------------------------------ #
    def state_space(self):
        """Matrices `(F, w, g, x)` with `x_t = F x_{t-1} + g e_t`."""
        self._check_fitted()
        if self.season_ == "M":
            raise ValueError("multiplicative seasonality has no linear form")
        m = self.season_period_ if self.season_ != "N" else 0
        has_trend = self.trend_ != "N"
        dim = 1 + int(has_trend) + m
        phi = self.params_.get("phi", 1.0)
        matrix = np.zeros((dim, dim))
        matrix[0, 0] = 1.0
        if has_trend:
            matrix[0, 1] = phi
            matrix[1, 1] = phi
        w = np.zeros(dim)
        w[0] = 1.0
        if has_trend:
            w[1] = phi
        g = np.zeros(dim)
        g[0] = self.params_["alpha"]
        if has_trend:
            g[1] = self.params_["beta"]
        if self.season_ == "A":
            matrix[0, 2] = 1.0
            for j in range(m - 1):
                matrix[2 + j, 3 + j] = 1.0
            w[2] = 1.0
            g[-1] = self.params_["gamma"]
        x = np.zeros(dim)
        x[0] = self.state_["level"]
        if has_trend:
            x[1] = self.state_["trend"]
        if self.season_ == "A":
            x[2:] = self.state_["season"]
        return matrix, w, g, x

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("call fit before predict or interval")


def _check_horizon(horizon) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _additive(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    season = 3.0 * np.sin(2.0 * np.pi * t / 12.0)
    season += 1.5 * np.cos(4.0 * np.pi * t / 12.0)
    return 50.0 + 0.08 * t + season + rng.normal(0.0, 0.8, n)


def _multiplicative(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    level = 20.0 + 0.15 * t
    season = 1.0 + 0.25 * np.sin(2.0 * np.pi * t / 12.0)
    return level * season + rng.normal(0.0, 0.6, n)


def _mae(forecast: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - truth)))


def demo() -> None:
    rng = np.random.default_rng(0)
    series = _additive(132, rng)
    train, holdout = series[:-12], series[-12:]
    print("132 points: level 50 + 0.08 t, a period-12 season of amplitude 4.5,")
    print("noise sd 0.8, and the last 12 points held out")
    cases = (("simple_es", 1), ("holt", 1), ("damped_trend", 1),
             ("holt_winters_add", 12), ("holt_winters_mul", 12),
             ("ets_state_space", 12), ("auto_ets", 12))
    fits, scores = {}, {}
    for variant, season in cases:
        model = ExponentialSmoothing(variant, season=season).fit(train)
        fits[variant] = model
        scores[variant] = _mae(model.predict(12), holdout)
        params = " ".join(f"{k}={v:.3f}" for k, v in model.params_.items())
        print(f"  {variant:17s} {model.model_:12s} MAE {scores[variant]:.4f}  "
              f"{params}")
    assert scores["holt_winters_add"] < 0.5 * scores["simple_es"]
    assert scores["holt_winters_add"] < scores["holt"]
    assert scores["holt_winters_add"] < scores["holt_winters_mul"]
    assert scores["ets_state_space"] < 1.1 * scores["holt_winters_add"]
    print("the seasonal model halves the error, and on additive data the")
    print("additive season also beats the multiplicative one")
    print("simple_es pushes alpha to its ceiling and becomes the naive rule:")
    assert fits["simple_es"].params_["alpha"] > 0.99
    print("a one-step smoother cannot see a period-12 pattern, so tracking the")
    print("last value is the best it can do")

    chosen = fits["auto_ets"]
    assert chosen.model_ == "ETS(A,A,A)"
    assert np.allclose(chosen.predict(12),
                       fits["holt_winters_add"].predict(12))
    print("auto_ets searched nine models by AICc and chose the same fit:")
    for name, aicc in sorted(chosen.candidates_, key=lambda row: row[1]):
        print(f"  {name:12s} {aicc:8.2f}")
    assert np.isclose(chosen.aicc_, min(a for _, a in chosen.candidates_))
    assert len(chosen.candidates_) == 9

    rng = np.random.default_rng(0)
    series = _multiplicative(132, rng)
    train_m, holdout_m = series[:-12], series[-12:]
    print("now a season whose amplitude is a fixed share of a tripling level:")
    multiplicative = {}
    for variant in ("holt_winters_add", "holt_winters_mul"):
        model = ExponentialSmoothing(variant, season=12).fit(train_m)
        multiplicative[variant] = model
        print(f"  {variant:17s} {model.model_:12s} "
              f"MAE {_mae(model.predict(12), holdout_m):.4f} "
              f"in-sample SSE {model.sse_:8.3f}")
    additive_fit = multiplicative["holt_winters_add"]
    multiple_fit = multiplicative["holt_winters_mul"]
    assert multiple_fit.sse_ < additive_fit.sse_
    assert _mae(multiple_fit.predict(12), holdout_m) < _mae(
        additive_fit.predict(12), holdout_m)
    assert ExponentialSmoothing("auto_ets", season=12).fit(
        train_m).model_ == "ETS(A,A,M)"
    print("the multiplicative season fits and forecasts better, and auto_ets")
    print("switches to it: the additive model cannot let the amplitude grow")

    print("mechanics. simple_es is a geometric weighted average:")
    alpha = 0.3
    model = ExponentialSmoothing("simple_es", alpha=alpha).fit(train)
    level = train[0]
    for value in train[1:]:
        level = alpha * value + (1.0 - alpha) * level
    decays = (1.0 - alpha) ** np.arange(train.size)
    weighted = alpha * float(decays[:-1] @ train[:0:-1])
    weighted += decays[-1] * train[0]
    assert np.isclose(model.state_["level"], level)
    assert np.isclose(model.predict(1)[0], level)
    assert np.isclose(level, weighted)
    print(f"  alpha={alpha}: the recursion, the forecast and the weighted sum")
    print(f"  all give {level:.9f}")

    model = ExponentialSmoothing("holt", alpha=0.3, beta=0.1).fit(train)
    steps = np.arange(1, 7)
    straight = model.state_["level"] + steps * model.state_["trend"]
    assert np.allclose(model.predict(6), straight)
    print("  holt forecasts lie on a straight line through the last level")

    model = ExponentialSmoothing("damped_trend", alpha=0.3, beta=0.1,
                                 phi=0.9).fit(train)
    limit = model.state_["level"] + 0.9 * model.state_["trend"] / 0.1
    long = model.predict(400)
    assert abs(long[-1] - limit) < 1e-6
    gaps = np.abs(np.diff(long))
    assert np.all(gaps[1:] <= gaps[:-1] + 1e-12)
    print("  with phi=0.9 the trend decays and the path converges to")
    print(f"  l + phi b / (1 - phi) = {limit:.6f}")

    seasonal = ExponentialSmoothing("holt_winters_add", season=12, alpha=0.2,
                                    beta=0.05, gamma=0.1).fit(train)
    steps = np.arange(1, 25)
    expected = (seasonal.state_["level"] + steps * seasonal.state_["trend"]
                + seasonal.state_["season"][(steps - 1) % 12])
    assert np.allclose(seasonal.predict(24), expected)
    state = ExponentialSmoothing("ets_state_space", season=12, alpha=0.2,
                                 beta=0.05, gamma=0.1, phi=1.0).fit(train)
    assert np.allclose(state.fitted_, seasonal.fitted_)
    assert np.allclose(state.predict(24), seasonal.predict(24))
    assert np.isclose(state.sse_, seasonal.sse_)
    print("  with phi=1 the state-space model and the recursion produce")
    print("  identical fitted values and forecasts, to machine precision")

    print("the interval is a variance recursion, and ses has a closed form:")
    model = ExponentialSmoothing("simple_es", alpha=0.4).fit(train)
    lower, upper = model.interval(6, 0.95)
    z = NormalDist().inv_cdf(0.975)
    analytic = np.sqrt(model.sigma2_ * (1.0 + 0.4 ** 2 * np.arange(6)))
    assert np.allclose((upper - lower) / (2.0 * z), analytic)
    assert np.isclose(model.sigma2_, model.sse_ / (model.n_ - model.start_))
    print(f"  sigma_h^2 = sigma^2 (1 + (h-1) alpha^2): {analytic[-1]:.4f} at h=6")

    model = ExponentialSmoothing("ets_state_space", season=12).fit(train)
    matrix, w, g, x = model.state_space()
    noise = np.random.default_rng(7)
    paths, horizon = 2000, 18
    simulated = np.empty((paths, horizon))
    for row in range(paths):
        state = x.copy()
        for h in range(horizon):
            shock = noise.normal(0.0, np.sqrt(model.sigma2_))
            held = state
            state = matrix @ state + g * shock
            simulated[row, h] = w @ held + shock
    empirical = simulated.std(axis=0, ddof=1)
    lower, upper = model.interval(horizon, 0.95)
    analytic = (upper - lower) / (2.0 * z)
    deviation = float(np.max(np.abs(empirical / analytic - 1.0)))
    print("  2000 paths simulated from the fitted state space deviate from")
    print(f"  the analytic band by at most {deviation:.4f} over 18 steps")
    assert deviation < 0.06

    bad = [
        lambda: ExponentialSmoothing("theta"),
        lambda: ExponentialSmoothing(season=0),
        lambda: ExponentialSmoothing("holt", season=12),
        lambda: ExponentialSmoothing("simple_es", alpha=0.0),
        lambda: ExponentialSmoothing("simple_es", alpha=1.5),
        lambda: ExponentialSmoothing("damped_trend", phi=0.0),
        lambda: ExponentialSmoothing("holt", beta=1.2),
        lambda: ExponentialSmoothing().predict(3),
        lambda: ExponentialSmoothing().interval(3),
        lambda: ExponentialSmoothing("simple_es").fit(train).predict(0),
        lambda: ExponentialSmoothing("simple_es").fit(train).predict(2.5),
        lambda: ExponentialSmoothing("simple_es").fit(train).interval(3, 0.0),
        lambda: ExponentialSmoothing("simple_es").fit(train).interval(3, 1.0),
        lambda: ExponentialSmoothing("simple_es").fit(train.reshape(-1, 1)),
        lambda: ExponentialSmoothing("simple_es").fit(
            np.array([1.0, np.nan, 2.0, 3.0])),
        lambda: ExponentialSmoothing("simple_es").fit(np.array([1.0, 2.0, 3.0])),
        lambda: ExponentialSmoothing("holt").fit(np.array([1.0, 2.0, 3.0, 4.0])),
        lambda: ExponentialSmoothing("holt_winters_mul", season=12).fit(
            _multiplicative(40, rng) - 25.0),
        lambda: ExponentialSmoothing("holt_winters_add", season=200).fit(train),
        lambda: ExponentialSmoothing("simple_es", beta=0.2).fit(train),
        lambda: ExponentialSmoothing("holt_winters_mul", season=12).fit(
            train).state_space(),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a zero or non-integer season, a season on a variant")
    print("  without one, weights outside (0, 1], use before fit, a zero or")
    print("  fractional horizon, a level outside (0, 1), a 2-d or non-finite")
    print("  series, series too short for a trend or a season, a")
    print("  multiplicative season on non-positive data, a weight for a")
    print("  term the model does not have, and a linear form requested for")
    print("  multiplicative seasonality")
    print("ets demo passed")


if __name__ == "__main__":
    demo()
