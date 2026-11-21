"""ARIMA — AR, MA, integration, seasonality and exogenous regressors.

Chapter:   11-time-series
Companion: arima.md
Tier 1 · 7 variants: ar, ma, arma, arima, sarima, arimax, auto_arima
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

VARIANTS = ("ar", "ma", "arma", "arima", "sarima", "arimax", "auto_arima")

DEFAULT_ORDER = {"ar": (1, 0, 0), "ma": (0, 0, 1), "arma": (1, 0, 1),
                 "arima": (1, 1, 0), "sarima": (1, 1, 1), "arimax": (1, 0, 0),
                 "auto_arima": (0, 0, 0)}

DEFAULT_SEASONAL = {"sarima": (1, 1, 1)}

LIMIT = 0.98


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray, name: str = "y") -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if y.size < 8:
        raise ValueError(f"{name} needs at least eight observations")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} must be finite")
    return y


def _difference(y: np.ndarray, d: int, D: int, m: int) -> list[np.ndarray]:
    """Every stage of regular then seasonal differencing, starting with `y`."""
    stages = [y]
    current = y
    for _ in range(d):
        current = np.diff(current)
        stages.append(current)
    for _ in range(D):
        current = current[m:] - current[:-m]
        stages.append(current)
    return stages


def _poly(regular, seasonal, m: int, sign: float) -> np.ndarray:
    """Dense coefficients of `(1 + sign*r(B))(1 + sign*S(B^m))`."""
    first = np.zeros(len(regular) + 1)
    first[0] = 1.0
    first[1:] = sign * np.asarray(regular, dtype=float)
    second = np.zeros(len(seasonal) * m + 1)
    second[0] = 1.0
    second[m::m] = sign * np.asarray(seasonal, dtype=float)
    return np.convolve(first, second)


def _filter(series: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residuals of `u_t = sum a_i u_{t-i} + e_t + sum b_j e_{t-j}`."""
    n = series.size
    out = np.zeros(n)
    la, lb = a.size, b.size
    start = max(la, lb)
    for t in range(start, n):
        value = float(series[t])
        if la:
            value -= float(a @ series[t - la:t][::-1])
        if lb:
            value -= float(b @ out[t - lb:t][::-1])
        out[t] = value
    return out


def _kpss(series: np.ndarray) -> float:
    """KPSS statistic for level stationarity; above 0.463 rejects at 5%."""
    n = series.size
    design = np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(design, series, rcond=None)
    errors = series - design @ coef
    partial = np.cumsum(errors)
    lag = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lag = max(0, min(lag, n - 2))
    gamma = [float(errors @ errors) / n]
    gamma += [float(errors[k:] @ errors[:-k]) / n for k in range(1, lag + 1)]
    weights = 1.0 - np.arange(1, lag + 1) / (lag + 1.0)
    long_run = gamma[0] + 2.0 * float(weights @ np.asarray(gamma[1:]))
    if long_run <= 0.0:
        return 0.0
    return float(partial @ partial) / (n * n * long_run)


def _choose_d(y: np.ndarray, limit: int = 2) -> int:
    """Smallest number of differences that passes the KPSS test."""
    current = y
    for d in range(limit + 1):
        if current.size < 20:
            break
        if _kpss(current) <= 0.463:
            return d
        current = np.diff(current)
    return min(limit, y.size // 20)


def _triple(value, name: str) -> tuple[int, int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a tuple of three integers") from exc
    whole = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                for v in items)
    if len(items) != 3 or not whole:
        raise ValueError(f"{name} must be a tuple of three integers")
    return tuple(int(v) for v in items)


def _check_horizon(horizon) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class ARIMA:
    """ARIMA with optional seasonal orders and exogenous regressors.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`; each restricts the admissible orders.
    order : tuple of three ints, optional
        `(p, d, q)`. `None` uses the variant's default.
    seasonal : tuple of three ints, optional
        `(P, D, Q)` seasonal orders at period `season`. `None` uses `(1, 1, 1)`
        for `sarima` and `(0, 0, 0)` otherwise.
    season : int
        Seasonal period `m`.
    include_mean : bool
        Whether the differenced series has a constant.

    Notes
    -----
    `fit` estimates by conditional sum of squares. For fixed ARMA
    coefficients the constant and the regression coefficients solve a linear
    least-squares problem, so only the short ARMA coefficients are searched;
    an AR-only model with no regressors is one least-squares call.

    `fit` stores `order_`, `seasonal_order_`, `ar_coef_`, `ma_coef_`, `mean_`,
    `exog_coef_`, `sigma2_`, `loglik_`, `aicc_`, `residuals_` and `stages_`.
    `predict` returns forecasts on the original scale; `interval` uses the
    psi-weights of the integrated model.
    """

    def __init__(self, variant: str = "arima", *, order=None, seasonal=None,
                 season: int = 1, include_mean: bool = True):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(season, (int, np.integer)) or season < 1:
            raise ValueError("season must be a positive integer")
        self.variant = variant
        self.order = None if order is None else _triple(order, "order")
        self.seasonal = (None if seasonal is None
                         else _triple(seasonal, "seasonal"))
        self.season = int(season)
        self.include_mean = bool(include_mean)
        if variant != "auto_arima":
            self._spec()

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray, exog: np.ndarray | None = None) -> ARIMA:
        """Estimate the model, choosing among orders for `auto_arima`."""
        y = _check_series(y)
        exog = self._check_exog(exog, y.size)
        if self.variant == "arimax" and exog is None:
            raise ValueError("arimax needs an exog matrix at fit time")
        specs = self._candidates(y)
        best, table = None, []
        for spec in specs:
            try:
                fitted = self._fit_one(y, spec, exog)
            except (ValueError, np.linalg.LinAlgError):
                if self.variant != "auto_arima":
                    raise
                continue
            table.append((spec[:3], fitted["aicc"]))
            if best is None or fitted["aicc"] < best["aicc"]:
                best = fitted
        if best is None:
            raise ValueError("no candidate model could be fitted")
        self._install(best)
        self.candidates_ = table
        return self

    def _candidates(self, y):
        if self.variant != "auto_arima":
            return [self._spec()]
        d = _choose_d(y)
        return [(p, d, q, 0, 0, 0) for p in range(3) for q in range(3)]

    def _spec(self):
        variant = self.variant
        p, d, q = self.order or DEFAULT_ORDER[variant]
        seasonal = self.seasonal
        if seasonal is None:
            seasonal = DEFAULT_SEASONAL.get(variant, (0, 0, 0))
        P, D, Q = seasonal
        if min(p, d, q, P, D, Q) < 0:
            raise ValueError("orders must be non-negative")
        if variant == "sarima":
            if self.season < 2:
                raise ValueError("sarima needs a seasonal period of at least 2")
            if not (P or D or Q):
                raise ValueError("sarima needs at least one seasonal order")
        elif P or D or Q:
            raise ValueError("seasonal orders need the sarima variant")
        if variant == "ar" and (d or q or p < 1):
            raise ValueError("ar allows only a non-seasonal AR order")
        if variant == "ma" and (d or p or q < 1):
            raise ValueError("ma allows only a non-seasonal MA order")
        if variant == "arma" and (d or p + q < 1):
            raise ValueError("arma needs p and q with no differencing")
        if variant == "arima" and d < 1:
            raise ValueError("arima needs d of at least 1")
        if variant == "arimax" and (d or D):
            raise ValueError("arimax keeps the regressors undifferenced")
        return (p, d, q, P, D, Q)

    def _check_exog(self, exog, n: int):
        if exog is None:
            return None
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog[:, None]
        if exog.ndim != 2 or exog.shape[0] != n:
            raise ValueError("exog must be (n,) or (n, k) matching y")
        if exog.shape[1] < 1:
            raise ValueError("exog needs at least one column")
        if not np.all(np.isfinite(exog)):
            raise ValueError("exog must be finite")
        return exog

    def _fit_one(self, y, spec, exog) -> dict:
        p, d, q, P, D, Q = spec
        m = self.season if (P or D or Q) else 1
        stages = _difference(y, d, D, m)
        w = stages[-1]
        if w.size < 8:
            raise ValueError("too few observations after differencing")
        columns = []
        if self.include_mean:
            columns.append(np.ones(w.size))
        exog_w = None
        if exog is not None:
            exog_w = np.column_stack([
                _difference(exog[:, k], d, D, m)[-1]
                for k in range(exog.shape[1])])
            columns.append(exog_w)
        design = (np.column_stack(columns) if columns
                  else np.empty((w.size, 0)))
        ar_lags = list(range(1, p + 1)) + [m * k for k in range(1, P + 1)]
        ma_lags = list(range(1, q + 1)) + [m * k for k in range(1, Q + 1)]
        length = max(ar_lags + ma_lags, default=0)
        rows = np.arange(length, w.size)
        if rows.size < 8:
            raise ValueError("not enough residuals for this order")
        if not ma_lags and not P and design.shape[1] <= 1:
            a, gamma = self._fit_ar(w, rows, length, design)
            b = np.zeros(0)
            parts = (a.copy(), np.zeros(0), np.zeros(0), np.zeros(0))
        else:
            a, b, gamma, parts = self._fit_search(w, design, p, P, q, Q, m)
        start = max(a.size, b.size)
        u = w - design @ gamma
        errors = _filter(u, a, b)
        used = errors[start:]
        sse = float(used @ used)
        if sse <= 0.0:
            raise ValueError("the model fits the data exactly")
        parameters = p + P + q + Q + design.shape[1]
        k = parameters + 1
        sigma2 = sse / max(used.size - parameters, 1)
        loglik = -0.5 * used.size * (np.log(2.0 * np.pi * sse / used.size) + 1.0)
        aic = -2.0 * loglik + 2.0 * k
        aicc = aic + 2.0 * k * (k + 1) / max(used.size - k - 1, 1)
        return {"spec": spec, "m": m, "a": a, "b": b, "gamma": gamma,
                "parts": parts, "sse": sse, "sigma2": sigma2,
                "loglik": float(loglik), "aic": float(aic),
                "aicc": float(aicc), "start": start, "residuals": errors,
                "fitted": w - errors, "stages": stages, "design": design,
                "w": w, "n_used": int(used.size), "k": int(k)}

    def _fit_ar(self, w, rows, length, design):
        blocks = [w[rows - lag] for lag in range(1, length + 1)]
        if design.shape[1]:
            blocks.append(design[rows])
        if not blocks:
            return np.zeros(0), np.zeros(0)
        matrix = np.column_stack(blocks)
        coef, *_ = np.linalg.lstsq(matrix, w[rows], rcond=None)
        a = coef[:length]
        constant = coef[length:]
        if design.shape[1] and self.include_mean:
            total = float(a.sum())
            mean = (constant[0] / (1.0 - total)
                    if abs(1.0 - total) > 1e-9 else 0.0)
            gamma = np.concatenate([[mean], constant[1:]])
        else:
            gamma = constant
        return a, gamma

    def _fit_search(self, w, design, p, P, q, Q, m):
        size = p + P + q + Q

        def split(values):
            phi, Phi = values[:p], values[p:p + P]
            theta, Theta = values[p + P:p + P + q], values[p + P + q:]
            a = -_poly(phi, Phi, m, -1.0)[1:]
            b = _poly(theta, Theta, m, 1.0)[1:]
            return a, b

        def profile(values):
            a, b = split(values)
            start = max(a.size, b.size)
            filtered = _filter(w, a, b)
            if design.shape[1]:
                blocks = np.column_stack([_filter(design[:, j], a, b)
                                          for j in range(design.shape[1])])
                coef, *_ = np.linalg.lstsq(blocks[start:], filtered[start:],
                                           rcond=None)
                residuals = filtered[start:] - blocks[start:] @ coef
            else:
                coef = np.zeros(0)
                residuals = filtered[start:]
            return float(residuals @ residuals), a, b, coef

        values = np.zeros(size)
        steps = np.full(size, 0.2)
        best = profile(values)[0]
        if size:
            for _ in range(80):
                improved = False
                for k in range(size):
                    for delta in (steps[k], -steps[k]):
                        trial = values.copy()
                        trial[k] = np.clip(values[k] + delta, -LIMIT, LIMIT)
                        if trial[k] == values[k]:
                            continue
                        score = profile(trial)[0]
                        if score < best - 1e-10:
                            values, best, improved = trial, score, True
                if not improved:
                    steps *= 0.5
                    if steps.max() < 1e-3:
                        break
        _, a, b, gamma = profile(values)
        phi, Phi = values[:p], values[p:p + P]
        theta, Theta = values[p + P:p + P + q], values[p + P + q:]
        return a, b, gamma, (phi, Phi, theta, Theta)

    def _install(self, best):
        p, d, q, P, D, Q = best["spec"]
        self.order_ = (p, d, q)
        self.seasonal_order_ = ((P, D, Q, best["m"]) if (P or D or Q)
                                else None)
        self.d_, self.D_, self.season_ = d, D, best["m"]
        self.a_, self.b_ = best["a"], best["b"]
        phi, Phi, theta, Theta = best["parts"]
        self.ar_coef_, self.seasonal_ar_coef_ = phi, Phi
        self.ma_coef_, self.seasonal_ma_coef_ = theta, Theta
        self.include_mean_ = self.include_mean
        self.mean_ = float(best["gamma"][0]) if self.include_mean else 0.0
        self.exog_coef_ = (best["gamma"][1:] if self.include_mean
                           else best["gamma"])
        self.w_ = best["w"]
        self.design_ = best["design"]
        self.residuals_ = best["residuals"]
        self.fitted_ = best["fitted"]
        self.stages_ = best["stages"]
        self.start_ = best["start"]
        self.n_used_ = best["n_used"]
        self.sse_ = best["sse"]
        self.sigma2_ = best["sigma2"]
        self.loglik_ = best["loglik"]
        self.aic_ = best["aic"]
        self.aicc_ = best["aicc"]
        self.n_ = self.stages_[0].size
        self.n_exog_ = best["design"].shape[1] - int(self.include_mean)

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int, exog: np.ndarray | None = None
                ) -> np.ndarray:
        """Forecasts on the original scale, shape `(horizon,)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        future = self._future_design(horizon, exog)
        gamma = self._gamma()
        u = self.w_ - self.design_ @ gamma
        n = u.size
        extended = np.concatenate([u, np.zeros(horizon)])
        innovations = np.concatenate([self.residuals_, np.zeros(horizon)])
        for j in range(horizon):
            value = 0.0
            for i, coef in enumerate(self.a_, 1):
                value += coef * extended[n + j - i]
            for k, coef in enumerate(self.b_, 1):
                value += coef * innovations[n + j - k]
            extended[n + j] = value
        differenced = extended[n:] + future @ gamma
        return self._integrate(differenced)

    def _gamma(self) -> np.ndarray:
        if self.include_mean:
            return np.concatenate([[self.mean_], self.exog_coef_])
        return self.exog_coef_

    def _future_design(self, horizon: int, exog) -> np.ndarray:
        columns = []
        if self.include_mean:
            columns.append(np.ones(horizon))
        if self.n_exog_:
            if exog is None:
                raise ValueError("this model needs exog for forecasting")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog[:, None]
            if exog.shape != (horizon, self.n_exog_):
                raise ValueError("exog must be (horizon, n_exog)")
            if not np.all(np.isfinite(exog)):
                raise ValueError("exog must be finite")
            columns.append(exog)
        if not columns:
            return np.empty((horizon, 0))
        return np.column_stack(columns)

    def _integrate(self, forecast: np.ndarray) -> np.ndarray:
        path = forecast
        index = len(self.stages_) - 1
        for _ in range(self.D_):
            parent = self.stages_[index - 1]
            n = parent.size
            extended = np.concatenate([parent, np.zeros(path.size)])
            for j in range(path.size):
                extended[n + j] = path[j] + extended[n + j - self.season_]
            path = extended[n:]
            index -= 1
        for _ in range(self.d_):
            parent = self.stages_[index - 1]
            n = parent.size
            extended = np.concatenate([parent, np.zeros(path.size)])
            for j in range(path.size):
                extended[n + j] = path[j] + extended[n + j - 1]
            path = extended[n:]
            index -= 1
        return path

    def simulate(self, horizon: int, *, paths: int = 1, seed: int = 0,
                 exog: np.ndarray | None = None) -> np.ndarray:
        """Gaussian future paths from the fitted model, `(paths, horizon)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        if not isinstance(paths, (int, np.integer)) or paths < 1:
            raise ValueError("paths must be a positive integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        future = self._future_design(horizon, exog)
        gamma = self._gamma()
        u = self.w_ - self.design_ @ gamma
        rng = np.random.default_rng(seed)
        n = u.size
        out = np.empty((int(paths), horizon))
        for row in range(int(paths)):
            extended = np.concatenate([u, np.zeros(horizon)])
            innovations = np.concatenate([self.residuals_, np.zeros(horizon)])
            shock = rng.normal(0.0, np.sqrt(self.sigma2_), horizon)
            for j in range(horizon):
                value = float(shock[j])
                for i, coef in enumerate(self.a_, 1):
                    value += coef * extended[n + j - i]
                for k, coef in enumerate(self.b_, 1):
                    value += coef * innovations[n + j - k]
                extended[n + j] = value
                innovations[n + j] = shock[j]
            out[row] = self._integrate(extended[n:] + future @ gamma)
        return out

    def psi(self, horizon: int) -> np.ndarray:
        """Psi-weights of the integrated model, length `horizon`."""
        self._check_fitted()
        weights = np.zeros(horizon)
        weights[0] = 1.0
        for j in range(1, horizon):
            value = float(self.b_[j - 1]) if j - 1 < self.b_.size else 0.0
            for i, coef in enumerate(self.a_, 1):
                if j - i >= 0:
                    value += coef * weights[j - i]
            weights[j] = value
        for _ in range(self.d_):
            weights = np.cumsum(weights)
        for _ in range(self.D_):
            out = np.zeros_like(weights)
            for j in range(horizon):
                out[j] = weights[j::-self.season_].sum()
            weights = out
        return weights

    def interval(self, horizon: int, level: float = 0.95,
                 exog: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian band from the psi-weight variance."""
        self._check_fitted()
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        forecast = self.predict(horizon, exog)
        variance = self.sigma2_ * np.cumsum(self.psi(horizon) ** 2)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        spread = z * np.sqrt(variance)
        return forecast - spread, forecast + spread

    def _check_fitted(self):
        if not hasattr(self, "order_"):
            raise RuntimeError("call fit before predict or interval")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _arma(n: int, rng: np.random.Generator, phi=(), theta=(),
          burn: int = 300) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, n + burn)
    path = np.zeros(n + burn)
    for t in range(burn, n + burn):
        value = noise[t]
        for lag, coef in enumerate(phi, 1):
            value += coef * path[t - lag]
        for lag, coef in enumerate(theta, 1):
            value += coef * noise[t - lag]
        path[t] = value
    return path[burn:]


def _seasonal(n: int, rng: np.random.Generator, m: int = 12) -> np.ndarray:
    t = np.arange(n)
    shape = 5.0 * np.sin(2.0 * np.pi * t / m) + 2.0 * np.cos(4.0 * np.pi * t / m)
    return 50.0 + 0.1 * t + shape + _arma(n, rng, phi=(0.5,))


def _acf(series: np.ndarray, lag: int) -> float:
    centred = series - series.mean()
    return float(centred[lag:] @ centred[:-lag] / (centred @ centred))


def _mae(forecast: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - truth)))


def demo() -> None:
    rng = np.random.default_rng(0)
    print("ar. 300 points of an AR(2) with phi = (0.6, -0.3):")
    y = _arma(300, rng, phi=(0.6, -0.3))
    model = ARIMA("ar", order=(2, 0, 0)).fit(y)
    rows = np.arange(2, y.size)
    design = np.column_stack([y[rows - 1], y[rows - 2], np.ones(rows.size)])
    ols, *_ = np.linalg.lstsq(design, y[rows], rcond=None)
    assert np.allclose(model.ar_coef_, ols[:2])
    assert np.isclose(model.sse_, float(np.sum((y[rows] - design @ ols) ** 2)))
    implied = model.mean_ * (1.0 - model.ar_coef_.sum())
    assert np.isclose(implied, ols[2])
    print(f"  phi {np.round(model.ar_coef_, 4)} equals the OLS slopes to")
    print(f"  machine precision, and the implied constant {implied:.6f}")
    print(f"  equals the OLS intercept {ols[2]:.6f}")

    rng = np.random.default_rng(7)
    print("ma. 400 points of an MA(1) with theta = 0.6:")
    y = _arma(400, rng, theta=(0.6,))
    model = ARIMA("ma", order=(0, 0, 1)).fit(y)
    variance = float(y @ y) / y.size
    rho = float(y[1:] @ y[:-1]) / y.size / variance
    moments = (1.0 - np.sqrt(1.0 - 4.0 * rho ** 2)) / (2.0 * rho)
    print(f"  conditional least squares gives {model.ma_coef_[0]:.4f}, and")
    print(f"  the autocovariance estimator gives {moments:.4f} from "
          f"rho = {rho:.4f}")
    assert abs(model.ma_coef_[0] - 0.6) < 0.1
    assert abs(moments - 0.6) < 0.15

    rng = np.random.default_rng(3)
    print("arima. a random walk, and the band that grows with sqrt(h):")
    walk = np.cumsum(rng.normal(0.0, 1.0, 150))
    model = ARIMA("arima", order=(0, 1, 0), include_mean=False).fit(walk)
    assert np.allclose(model.predict(5), walk[-1])
    lower, upper = model.interval(5)
    width = upper - lower
    assert np.allclose(width / width[0], np.sqrt(np.arange(1, 6)))
    print(f"  forecast {walk[-1]:.4f} for all five steps, width "
          f"{width[0]:.3f} to {width[-1]:.3f}")

    rng = np.random.default_rng(4)
    print("arima(1,1,1). psi-weights and a 3000-path check of the band:")
    y = np.cumsum(_arma(300, rng, phi=(0.6,), theta=(0.3,)))
    model = ARIMA("arima", order=(1, 1, 1)).fit(y)
    lower, upper = model.interval(12)
    z = NormalDist().inv_cdf(0.975)
    analytic = (upper - lower) / (2.0 * z)
    empirical = model.simulate(12, paths=3000, seed=9).std(axis=0, ddof=1)
    worst = float(np.max(np.abs(empirical / analytic - 1.0)))
    print(f"  fitted phi {model.ar_coef_[0]:.3f}, theta "
          f"{model.ma_coef_[0]:.3f}, psi {np.round(model.psi(4), 3)}")
    print(f"  simulated sd deviates from the analytic band by at most "
          f"{worst:.4f}")
    assert worst < 0.06

    rng = np.random.default_rng(5)
    print("sarima. a period-12 season with an AR(1) noise:")
    y = _seasonal(180, rng)
    model = ARIMA("sarima", order=(0, 0, 0), seasonal=(1, 1, 0),
                  season=12).fit(y)
    residuals = model.residuals_[model.start_:]
    print(f"  lag-12 autocorrelation falls from {_acf(y, 12):.3f} in the")
    print(f"  series to {_acf(residuals, 12):.3f} in the residuals")
    assert _acf(y, 12) > 0.7
    assert abs(_acf(residuals, 12)) < 0.3

    print("  a seasonal random walk has an exact forecast: repeat the cycle")
    rng = np.random.default_rng(6)
    noise = rng.normal(0.0, 1.0, 120)
    seasonal_walk = noise.copy()
    for t in range(12, 120):
        seasonal_walk[t] = seasonal_walk[t - 12] + noise[t]
    model = ARIMA("sarima", order=(0, 0, 0), seasonal=(0, 1, 0), season=12,
                  include_mean=False).fit(seasonal_walk)
    assert np.allclose(model.predict(12), seasonal_walk[-12:])

    rng = np.random.default_rng(0)
    print("arimax. y = 1.5 x plus AR(1) noise, 300 points fitted, 100 held:")
    size = 400
    x = rng.normal(0.0, 1.0, size)
    y = 1.5 * x + _arma(size, rng, phi=(0.6,))
    model = ARIMA("arimax", order=(1, 0, 0)).fit(y[:300], exog=x[:300])
    plain = ARIMA("ar", order=(1, 0, 0)).fit(y[:300])
    with_x = model.predict(100, exog=x[300:])
    without = plain.predict(100)
    print(f"  beta {model.exog_coef_[0]:.4f} against 1.5; MAE "
          f"{_mae(with_x, y[300:]):.4f} against {_mae(without, y[300:]):.4f}")
    assert abs(model.exog_coef_[0] - 1.5) < 0.2
    assert _mae(with_x, y[300:]) < _mae(without, y[300:])

    rng = np.random.default_rng(2)
    print("auto_arima. KPSS picks d, then AICc picks p and q:")
    y = _arma(400, rng, phi=(0.5, 0.3))
    model = ARIMA("auto_arima").fit(y)
    print(f"  on a stationary AR(2) draw it returns {model.order_}; the")
    print("  three best of its nine candidates are")
    for spec, aicc in sorted(model.candidates_, key=lambda row: row[1])[:3]:
        print(f"    {spec}  {aicc:9.2f}")
    assert model.order_ == (2, 0, 0)
    assert np.isclose(model.aicc_, min(a for _, a in model.candidates_))
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0.0, 1.0, 200))
    assert ARIMA("auto_arima").fit(walk).order_[1] == 1
    print("  on a random walk it differences once")

    bad = [
        lambda: ARIMA("arima_lite"),
        lambda: ARIMA("ar", order=(2,)),
        lambda: ARIMA("ar", order=(1.5, 0, 0)),
        lambda: ARIMA("ar", order=(-1, 0, 0)),
        lambda: ARIMA("ar", order=(1, 0, 1)),
        lambda: ARIMA("ma", order=(1, 0, 1)),
        lambda: ARIMA("arma", order=(1, 1, 1)),
        lambda: ARIMA("arima", order=(1, 0, 0)),
        lambda: ARIMA("sarima", order=(1, 1, 0), seasonal=(0, 0, 0)),
        lambda: ARIMA("sarima", order=(0, 1, 0), seasonal=(0, 1, 0),
                      season=1),
        lambda: ARIMA("arima", order=(1, 1, 0)).predict(3),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y).predict(0),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y).predict(2.5),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y).interval(3, 0.0),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y.reshape(-1, 1)),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(np.r_[y[:7], np.nan]),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(np.arange(6.0)),
        lambda: ARIMA("arimax", order=(1, 0, 0)).fit(y),
        lambda: ARIMA("arimax", order=(1, 0, 0)).fit(y, exog=x[:10]),
        lambda: ARIMA("arimax", order=(1, 0, 0)).fit(
            y, exog=np.column_stack([x, np.full(size, np.nan)])),
        lambda: ARIMA("arima", order=(0, 20, 0)).fit(walk[:20]),
        lambda: ARIMA("arimax", order=(1, 0, 0)).fit(y, exog=x).predict(3),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y).simulate(3, paths=0),
        lambda: ARIMA("ar", order=(1, 0, 0)).fit(y).simulate(3, seed=-1),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a short, fractional or negative order, a variant used")
    print("  outside its order restrictions, a seasonal variant without a")
    print("  seasonal order or period, use before fit, a zero or fractional")
    print("  horizon, a level outside (0, 1), a 2-d or non-finite series, a")
    print("  series below eight points, arimax without exog or with the")
    print("  wrong number of rows or non-finite columns, too much")
    print("  differencing, a forecast without exog, and a simulation with")
    print("  zero paths or a negative seed")
    print("arima demo passed")


if __name__ == "__main__":
    demo()
