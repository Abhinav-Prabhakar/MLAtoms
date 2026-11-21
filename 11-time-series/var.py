"""Vector autoregression — interactions, causality and impulse responses.

Chapter:   11-time-series
Companion: var.md
Tier 2 · 6 variants: var, varx, granger, irf, fevd, var_ridge
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = ("var", "varx", "granger", "irf", "fevd", "var_ridge")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_matrix(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (n, k)")
    if Y.shape[0] < 8 or Y.shape[1] < 1:
        raise ValueError("Y needs at least eight rows and one column")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must be finite")
    return Y


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_survival(f: float, d1: float, d2: float) -> float:
    if f <= 0.0:
        return 1.0
    if not np.isfinite(f):
        return 0.0
    return _betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class VAR:
    """Vector autoregression with optional exogenous regressors.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    lags : int
        Order `p` of the autoregression.
    ridge : float
        L2 penalty for `var_ridge`; ignored by the other variants.
    horizon : int
        Horizon stored by the `irf` and `fevd` variants at fit time.

    Notes
    -----
    Each equation has the same regressors, so equation-by-equation least
    squares is efficient. `fit` stores `coef_` `(p, k, k)`, `intercept_` `(k,)`,
    `sigma_` `(k, k)`, `residuals_`, `is_stable_` and, depending on the variant,
    `granger_pvalues_`, `irf_` or `fevd_`.
    """

    def __init__(self, variant: str = "var", *, lags: int = 1,
                 ridge: float = 1.0, horizon: int = 12):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(lags, (int, np.integer)) or lags < 1:
            raise ValueError("lags must be a positive integer")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be non-negative and finite")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.variant = variant
        self.lags = int(lags)
        self.ridge = float(ridge)
        self.horizon = int(horizon)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, Y: np.ndarray, exog: np.ndarray | None = None) -> VAR:
        """Fit the system and the variant's headline artefact."""
        Y = _check_matrix(Y)
        exog = self._check_exog(exog, Y.shape[0])
        if self.variant == "varx" and exog is None:
            raise ValueError("varx needs an exog matrix at fit time")
        n, k = Y.shape
        p = self.lags
        if n < p + 8:
            raise ValueError("not enough rows for this lag order")
        rows = np.arange(p, n)
        blocks = [np.ones((rows.size, 1))]
        for lag in range(1, p + 1):
            blocks.append(Y[rows - lag])
        if exog is not None:
            blocks.append(exog[rows])
        design = np.column_stack(blocks)
        target = Y[rows]
        penalty = 0.0 if self.variant != "var_ridge" else self.ridge
        if penalty > 0.0:
            gram = design.T @ design
            scale = np.trace(gram) / gram.shape[0]
            shrink = np.eye(gram.shape[0]) * penalty * scale
            shrink[0, 0] = 0.0
            beta = np.linalg.solve(gram + shrink, design.T @ target)
        else:
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ beta
        residuals = target - fitted
        self.k_ = k
        self.p_ = p
        self.n_ = n
        self.n_used_ = int(rows.size)
        self.intercept_ = beta[0].copy()
        self.exog_coef_ = (beta[1 + k * p:].copy()
                           if exog is not None else None)
        self.coef_ = np.empty((p, k, k))
        for lag in range(p):
            self.coef_[lag] = beta[1 + lag * k:1 + (lag + 1) * k].T
        self.fitted_ = fitted
        self.residuals_ = residuals
        self.history_ = Y[::-1].copy()
        dof = max(self.n_used_ - design.shape[1], 1)
        self.sigma_ = residuals.T @ residuals / dof
        self.design_ = design
        self.beta_ = beta
        self.companion_ = self._companion()
        eigenvalues = np.linalg.eigvals(self.companion_)
        self.is_stable_ = bool(np.max(np.abs(eigenvalues)) < 1.0)
        self.eigenvalues_ = eigenvalues
        mle = residuals.T @ residuals / self.n_used_
        sign, logdet = np.linalg.slogdet(mle)
        if sign <= 0.0:
            raise ValueError("residual covariance is singular")
        loglik = (-0.5 * self.n_used_ * k * (math.log(2.0 * math.pi) + 1.0)
                  - 0.5 * self.n_used_ * logdet)
        parameters = k * design.shape[1]
        self.loglik_ = float(loglik)
        self.aic_ = float(-2.0 * loglik + 2.0 * parameters)
        self.bic_ = float(-2.0 * loglik + math.log(self.n_used_) * parameters)
        if self.variant == "granger":
            self.granger_pvalues_ = np.ones((k, k))
            for effect in range(k):
                for cause in range(k):
                    if cause != effect:
                        test = self.granger_cause(cause, effect)
                        self.granger_pvalues_[effect, cause] = test["pvalue"]
        if self.variant == "irf":
            self.irf_ = self.irf(self.horizon)
        if self.variant == "fevd":
            self.fevd_ = self.fevd(self.horizon)
        return self

    def _check_exog(self, exog, n: int):
        if exog is None:
            return None
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog[:, None]
        if exog.ndim != 2 or exog.shape[0] != n or exog.shape[1] < 1:
            raise ValueError("exog must be (n, m) matching Y")
        if not np.all(np.isfinite(exog)):
            raise ValueError("exog must be finite")
        return exog

    def _companion(self) -> np.ndarray:
        k, p = self.k_, self.p_
        top = np.column_stack([self.coef_[lag] for lag in range(p)])
        if p == 1:
            return top
        lower = np.zeros((k * (p - 1), k * p))
        for lag in range(p - 1):
            lower[lag * k:(lag + 1) * k, lag * k:(lag + 1) * k] = np.eye(k)
        return np.vstack([top, lower])

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int, exog: np.ndarray | None = None
                ) -> np.ndarray:
        """Forecasts of the whole system, shape `(horizon, k)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        if self.exog_coef_ is not None:
            if exog is None:
                raise ValueError("this model needs exog for forecasting")
            exog = np.asarray(exog, dtype=float)
            if exog.ndim == 1:
                exog = exog[:, None]
            if exog.shape != (horizon, self.exog_coef_.shape[0]):
                raise ValueError("exog must be (horizon, m)")
            if not np.all(np.isfinite(exog)):
                raise ValueError("exog must be finite")
        history = [self.history_[lag - 1] for lag in range(1, self.p_ + 1)]
        out = np.empty((horizon, self.k_))
        for h in range(horizon):
            value = self.intercept_.copy()
            for lag, past in enumerate(history):
                value = value + self.coef_[lag] @ past
            if self.exog_coef_ is not None:
                value = value + self.exog_coef_.T @ exog[h]
            out[h] = value
            history = [value] + history[:-1]
        return out

    # -- impulse responses ------------------------------------------------ #
    def ma_coefficients(self, horizon: int) -> np.ndarray:
        """Psi-weights of the reduced form, shape `(horizon, k, k)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        k = self.k_
        weights = np.zeros((horizon, k, k))
        weights[0] = np.eye(k)
        for h in range(1, horizon):
            value = np.zeros((k, k))
            for lag in range(1, min(self.p_, h) + 1):
                value += self.coef_[lag - 1] @ weights[h - lag]
            weights[h] = value
        return weights

    def irf(self, horizon: int) -> np.ndarray:
        """Orthogonalised impulse responses, shape `(horizon, k, k)`."""
        self._check_fitted()
        weights = self.ma_coefficients(horizon)
        shock = np.linalg.cholesky(self.sigma_)
        return weights @ shock

    def fevd(self, horizon: int) -> np.ndarray:
        """Forecast error variance shares, shape `(horizon, k, k)`."""
        self._check_fitted()
        responses = self.irf(horizon)
        cumulative = np.cumsum(responses ** 2, axis=0)
        total = cumulative.sum(axis=2, keepdims=True)
        return cumulative / total

    # -- causality -------------------------------------------------------- #
    def granger_cause(self, cause: int, effect: int) -> dict:
        """F-test that lags of `cause` do not improve the `effect` equation."""
        self._check_fitted()
        k, p = self.k_, self.p_
        if not 0 <= cause < k or not 0 <= effect < k:
            raise ValueError("variable index out of range")
        if cause == effect:
            raise ValueError("a variable cannot be tested against itself")
        dropped = [1 + (lag - 1) * k + cause for lag in range(1, p + 1)]
        keep = [c for c in range(self.design_.shape[1]) if c not in dropped]
        target = self.fitted_ + self.residuals_
        subset = self.design_[:, keep]
        restricted, *_ = np.linalg.lstsq(subset, target[:, effect], rcond=None)
        residual = target[:, effect] - subset @ restricted
        sse_r = float(residual @ residual)
        sse_u = float(self.residuals_[:, effect] @ self.residuals_[:, effect])
        dof = self.n_used_ - self.design_.shape[1]
        if dof <= 0 or sse_u <= 0.0:
            raise ValueError("not enough degrees of freedom for the test")
        statistic = ((sse_r - sse_u) / p) / (sse_u / dof)
        statistic = max(statistic, 0.0)
        return {"F": float(statistic),
                "pvalue": float(_f_survival(statistic, p, dof)),
                "dof": int(dof)}

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("call fit before predict or the diagnostics")


def _check_horizon(horizon) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _simulate(coefs, n: int, rng: np.random.Generator, intercept,
              shock, burn: int = 200) -> np.ndarray:
    k = intercept.size
    p = len(coefs)
    path = np.zeros((n + burn, k))
    noise = rng.multivariate_normal(np.zeros(k), shock, n + burn)
    for t in range(p, n + burn):
        value = intercept.copy()
        for lag, coef in enumerate(coefs, 1):
            value = value + coef @ path[t - lag]
        path[t] = value + noise[t]
    return path[burn:]


def demo() -> None:
    rng = np.random.default_rng(0)
    print("var. 400 points of a bivariate VAR(1), true A = [[.6, 0], [.35, .45]]:")
    A = np.array([[0.6, 0.0], [0.35, 0.45]])
    Y = _simulate([A], 400, rng, np.zeros(2), np.eye(2))
    model = VAR("var", lags=1).fit(Y)
    rows = np.arange(1, Y.shape[0])
    design = np.column_stack([np.ones(rows.size), Y[rows - 1]])
    ols, *_ = np.linalg.lstsq(design, Y[rows], rcond=None)
    assert np.allclose(model.coef_[0], ols[1:].T)
    assert np.allclose(model.intercept_, ols[0])
    assert model.is_stable_
    print(f"  fitted A {np.round(model.coef_[0], 4).tolist()}, intercept "
          f"{np.round(model.intercept_, 4).tolist()}")
    print(f"  equation-by-equation least squares reproduces it exactly; "
          f"|eigenvalues| {np.round(np.abs(model.eigenvalues_), 4).tolist()}")

    rng = np.random.default_rng(5)
    print("  a VAR(2), where the companion matrix has four roots:")
    A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
    A2 = np.array([[0.2, 0.0], [0.1, 0.1]])
    Y2 = _simulate([A1, A2], 500, rng, np.zeros(2), np.eye(2))
    model2 = VAR("var", lags=2).fit(Y2)
    manual = (model2.intercept_ + model2.coef_[0] @ Y2[-1]
              + model2.coef_[1] @ Y2[-2])
    assert np.allclose(model2.predict(1)[0], manual)
    assert model2.is_stable_
    print(f"  one-step forecast {np.round(model2.predict(1)[0], 4).tolist()} "
          f"equals A0 y_T + A1 y_(T-1) + c")
    print(f"  |eigenvalues| {np.round(np.abs(model2.eigenvalues_), 4).tolist()}, "
          f"AIC {model2.aic_:.2f}")

    print("granger. in the VAR(1) draw y1 has no lagged effect on y0:")
    model = VAR("granger", lags=1).fit(Y)
    strong = model.granger_cause(0, 1)
    weak = model.granger_cause(1, 0)
    assert strong["pvalue"] < 1e-10
    assert weak["pvalue"] > 0.05
    print(f"  y0 -> y1: F {strong['F']:.2f}, p {strong['pvalue']:.2e}; "
          f"y1 -> y0: F {weak['F']:.2f}, p {weak['pvalue']:.3f}")
    print(f"  p-value matrix (row effect, column cause) "
          f"{np.round(model.granger_pvalues_, 4).tolist()}")

    print("irf. the psi-weights are matrix powers of the fitted A:")
    model = VAR("irf", lags=1, horizon=8).fit(Y)
    psi = model.ma_coefficients(4)
    assert np.allclose(psi[1], model.coef_[0])
    assert np.allclose(psi[2], model.coef_[0] @ model.coef_[0])
    assert np.allclose(model.irf_[0] @ model.irf_[0].T, model.sigma_)
    print(f"  psi1 {np.round(psi[1], 3).tolist()}, psi2 "
          f"{np.round(psi[2], 3).tolist()}")
    print("  the orthogonalised h=0 response reproduces sigma_ to machine "
          "precision")

    rng = np.random.default_rng(2)
    print("fevd. correlated shocks, Cholesky ordering (y0 first):")
    A = np.array([[0.6, 0.0], [0.35, 0.45]])
    factor = np.array([[1.0, 0.0], [0.8, 0.6]])
    Y3 = _simulate([A], 400, rng, np.zeros(2), factor @ factor.T)
    model = VAR("fevd", horizon=12).fit(Y3)
    shares = model.fevd_[-1]
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares[0, 0] > 0.9 and shares[1, 1] < 0.5
    print(f"  at h=12, y0 attribution {np.round(shares[0], 4).tolist()}, "
          f"y1 attribution {np.round(shares[1], 4).tolist()}")
    print("  y1 inherits most of its variance from the first shock, which is")
    print("  what the Cholesky ordering imposes, not a causal discovery")

    rng = np.random.default_rng(0)
    Y = _simulate([np.array([[0.6, 0.0], [0.35, 0.45]])], 400, rng,
                  np.zeros(2), np.eye(2))
    print("var_ridge. an L2 penalty on every coefficient except the intercept:")
    ols = VAR("var", lags=1).fit(Y)
    ridge = VAR("var_ridge", lags=1, ridge=1.0).fit(Y)
    plain = VAR("var_ridge", lags=1, ridge=0.0).fit(Y)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)
    assert np.allclose(plain.coef_, ols.coef_)
    print(f"  coefficient norm {np.linalg.norm(ridge.coef_):.4f} against "
          f"{np.linalg.norm(ols.coef_):.4f} unpenalised; ridge=0 is OLS")
    print(f"  fitted A {np.round(ridge.coef_[0], 4).tolist()}")

    rng = np.random.default_rng(1)
    print("varx. y = A y_(t-1) + B x_t + noise, with B = diag(0.8, -0.5):")
    size = 400
    X = rng.normal(0.0, 1.0, (size, 2))
    Y4 = np.zeros((size, 2))
    noise = rng.normal(0.0, 1.0, (size, 2))
    for t in range(1, size):
        Y4[t] = (A @ Y4[t - 1]
                 + np.array([0.8 * X[t, 0], -0.5 * X[t, 1]]) + noise[t])
    model = VAR("varx", lags=1).fit(Y4, exog=X)
    assert abs(model.exog_coef_[0, 0] - 0.8) < 0.12
    assert abs(model.exog_coef_[1, 1] + 0.5) < 0.12
    first = model.predict(1, exog=X[-1:])[0]
    manual = model.intercept_ + model.coef_[0] @ Y4[-1] + model.exog_coef_.T @ X[-1]
    assert np.allclose(first, manual)
    print(f"  fitted B {np.round(model.exog_coef_, 4).tolist()}; one-step "
          f"forecast {np.round(first, 4).tolist()}")

    bad = [
        lambda: VAR("var_lite"),
        lambda: VAR("var", lags=0),
        lambda: VAR("var", lags=1.5),
        lambda: VAR("var_ridge", ridge=-1.0),
        lambda: VAR("var_ridge", ridge=np.nan),
        lambda: VAR("irf", horizon=0),
        lambda: VAR("var").fit(np.arange(20.0)),
        lambda: VAR("var").fit(np.zeros((20, 0))),
        lambda: VAR("var").fit(np.zeros((6, 2))),
        lambda: VAR("var").fit(np.full((20, 2), np.nan)),
        lambda: VAR("var", lags=3).predict(2),
        lambda: VAR("var").fit(Y).predict(0),
        lambda: VAR("var").fit(Y).predict(1.5),
        lambda: VAR("varx").fit(Y),
        lambda: VAR("varx").fit(Y, exog=X[:10]),
        lambda: VAR("varx").fit(Y, exog=np.full((400, 1), np.nan)),
        lambda: VAR("varx").fit(Y, exog=np.zeros((400, 0))),
        lambda: VAR("varx").fit(Y, exog=X[:400]).predict(2),
        lambda: VAR("granger").fit(Y).granger_cause(2, 0),
        lambda: VAR("granger").fit(Y).granger_cause(0, 0),
        lambda: VAR("fevd", horizon=3).fit(Y).fevd(0),
        lambda: VAR("irf").fit(Y).ma_coefficients(0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a zero or fractional lag order, a negative or non-finite")
    print("  ridge, a zero fit or forecast horizon, a 1-d matrix, an empty or")
    print("  too-short system, a non-finite entry, use before fit, a")
    print("  fractional horizon, varx without exog or with the wrong rows, a")
    print("  non-finite exog, an empty exog, a forecast without future exog, an")
    print("  out-of-range or self Granger test, and a zero diagnostic horizon")
    print("var demo passed")


if __name__ == "__main__":
    demo()
