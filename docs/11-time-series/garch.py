"""GARCH — conditional variance that remembers the last shock.

Chapter:   11-time-series
Companion: garch.md
Tier 2 · 6 variants: arch, garch, egarch, gjr_garch, igarch, garch_t
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

VARIANTS = ("arch", "garch", "egarch", "gjr_garch", "igarch", "garch_t")

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_returns(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if r.size < 50:
        raise ValueError("need at least fifty returns")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    if float(np.var(r)) <= 0.0:
        raise ValueError("returns must not be constant")
    return r


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


def _t_cdf(value: float, nu: float) -> float:
    """CDF of a Student-t with `nu` degrees of freedom."""
    tail = 0.5 * _betainc(nu / 2.0, 0.5, nu / (nu + value * value))
    return 1.0 - tail if value >= 0.0 else tail


def _t_quantile(level: float, nu: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if level == 0.5:
        return 0.0
    lo, hi = -100.0, 100.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class GARCH:
    """GARCH-family volatility model fitted by Gaussian or Student-t MLE.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    p, q : int
        Autoregressive and moving-average orders of the variance recursion;
        `arch` uses only `q`, and the asymmetric variants require `p = q = 1`.
    nu : float
        Degrees of freedom for `garch_t`; ignored elsewhere.

    Notes
    -----
    The mean is a constant `mu`, so `r_t = mu + e_t` and `e_t = sigma_t z_t`.
    `fit` stores `mu_`, `omega_`, `alpha_`, `beta_`, `gamma_`, `nu_`,
    `conditional_variance_`, `standardised_`, `loglik_`, `aic_`, `bic_` and
    `persistence_`. `predict` returns variance forecasts; `interval` returns
    return forecasts and bands on the original scale.
    """

    def __init__(self, variant: str = "garch", *, p: int = 1, q: int = 1,
                 nu: float = 6.0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if not isinstance(p, (int, np.integer)) or not isinstance(
                q, (int, np.integer)) or p < 0 or q < 1:
            raise ValueError("p must be non-negative and q positive")
        if variant == "arch" and p > 1:
            raise ValueError("arch has no autoregressive variance terms")
        if variant != "arch" and p < 1:
            raise ValueError("this variant needs p of at least one")
        if (variant in ("egarch", "gjr_garch", "igarch", "garch_t")
                and (p != 1 or q != 1)):
            raise ValueError(f"{variant} is defined for p = q = 1")
        if not np.isfinite(nu) or nu <= 2.0:
            raise ValueError("nu must be finite and greater than two")
        self.variant = variant
        self.p = 0 if variant == "arch" else int(p)
        self.q = int(q)
        self.nu = float(nu)

    # -- fitting ---------------------------------------------------------- #
    def fit(self, r: np.ndarray) -> GARCH:
        """Estimate the parameters and the conditional variance path."""
        r = _check_returns(r)
        self.n_ = r.size
        self.start_ = min(max(self.p, self.q, 10), r.size // 4)
        theta, bounds, steps = self._initial(r)
        best = theta.copy()
        score = self._objective(best, r)
        if not np.isfinite(score):
            raise ValueError("the initial variance recursion is invalid")
        for _ in range(70):
            improved = False
            for index in range(best.size):
                for delta in (steps[index], -steps[index]):
                    trial = best.copy()
                    low, high = bounds[index]
                    trial[index] = min(max(trial[index] + delta, low), high)
                    if trial[index] == best[index]:
                        continue
                    value = self._objective(trial, r)
                    if value < score - 1e-10:
                        best, score, improved = trial, value, True
            if not improved:
                steps *= 0.5
                if steps.max() < 1e-5:
                    break
        params = self._unpack(best)
        variance = self._variance_path(r, params)
        if variance is None:
            raise ValueError("the fitted variance path is invalid")
        self._install(params, variance, r)
        return self

    def _initial(self, r: np.ndarray):
        variance = float(np.var(r))
        mu = float(np.mean(r))
        omega = 0.1 * variance
        if self.variant == "egarch":
            omega = variance ** 0.1
        theta = [mu, math.log(omega)]
        bounds = [(mu - 10.0 * math.sqrt(variance), mu + 10.0 * math.sqrt(variance)),
                  (math.log(variance) - 8.0, math.log(variance) + 4.0)]
        steps = [0.2 * math.sqrt(variance), 0.5]
        if self.variant == "arch":
            for _ in range(self.q):
                theta.append(0.5 / self.q)
                bounds.append((0.0, 0.98))
                steps.append(0.05)
        else:
            for _ in range(self.p):
                theta.append(0.05)
                bounds.append((0.0, 0.98))
                steps.append(0.05)
            if self.variant != "igarch":
                for _ in range(self.q):
                    theta.append(0.9 / self.q)
                    bounds.append((0.0, 0.98))
                    steps.append(0.05)
        if self.variant == "gjr_garch":
            theta.append(0.05)
            bounds.append((0.0, 0.98))
            steps.append(0.05)
        if self.variant == "egarch":
            theta.append(-0.05)
            bounds.append((-2.0, 2.0))
            steps.append(0.05)
        if self.variant == "garch_t":
            theta.append(math.log(self.nu - 2.0))
            bounds.append((math.log(0.05), math.log(200.0)))
            steps.append(0.3)
        return np.asarray(theta, dtype=float), bounds, np.asarray(steps)

    def _unpack(self, theta: np.ndarray) -> dict:
        index = 2
        if self.variant == "arch":
            alpha = theta[index:index + self.q].copy()
            beta = np.zeros(0)
            index += self.q
        else:
            alpha = theta[index:index + self.p].copy()
            index += self.p
            if self.variant == "igarch":
                beta = np.asarray([1.0 - float(alpha[0])])
            else:
                beta = theta[index:index + self.q].copy()
                index += self.q
        gamma = 0.0
        nu = math.inf
        if self.variant in ("gjr_garch", "egarch"):
            gamma = float(theta[index])
            index += 1
        if self.variant == "garch_t":
            nu = 2.0 + math.exp(float(theta[index]))
        return {"mu": float(theta[0]), "omega": math.exp(float(theta[1])),
                "alpha": alpha, "beta": beta, "gamma": gamma, "nu": nu}

    def _variance_path(self, r: np.ndarray, params: dict):
        n = r.size
        mu = params["mu"]
        omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
        gamma = params["gamma"]
        errors = r - mu
        scale = float(np.var(r))
        variance = np.full(n, scale)
        if self.variant == "egarch":
            variance[0] = scale
            log_variance = math.log(scale)
            for t in range(1, n):
                z = errors[t - 1] / math.sqrt(variance[t - 1])
                log_variance = (math.log(omega) + alpha[0] * (abs(z) - SQRT_2_OVER_PI)
                                + gamma * z + beta[0] * log_variance)
                if not np.isfinite(log_variance) or log_variance > 50.0:
                    return None
                variance[t] = math.exp(log_variance)
        else:
            arch_lags = self.q if self.variant == "arch" else self.p
            for t in range(1, n):
                value = omega
                for lag in range(1, arch_lags + 1):
                    past = errors[t - lag] ** 2 if t >= lag else scale
                    value += alpha[lag - 1] * past
                if beta.size:
                    for lag in range(1, self.q + 1):
                        past = variance[t - lag] if t >= lag else scale
                        value += beta[lag - 1] * past
                if gamma:
                    value += gamma * errors[t - 1] ** 2 * (errors[t - 1] < 0.0)
                if not np.isfinite(value) or value <= 0.0:
                    return None
                variance[t] = value
        return variance

    def _objective(self, theta: np.ndarray, r: np.ndarray) -> float:
        params = self._unpack(theta)
        if (self.variant in ("garch", "garch_t")
                and params["alpha"].sum() + params["beta"].sum() >= 0.999):
            return math.inf
        if self.variant == "gjr_garch":
            total = (params["alpha"].sum() + params["beta"].sum()
                     + 0.5 * params["gamma"])
            if total >= 0.999:
                return math.inf
        if self.variant == "arch" and params["alpha"].sum() >= 0.999:
            return math.inf
        if self.variant == "garch_t" and params["nu"] <= 2.01:
            return math.inf
        variance = self._variance_path(r, params)
        if variance is None:
            return math.inf
        errors = r - params["mu"]
        used = errors[self.start_:]
        var = variance[self.start_:]
        ratio = used ** 2 / var
        if self.variant == "garch_t":
            nu = params["nu"]
            constant = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                        - 0.5 * math.log(math.pi * (nu - 2.0)))
            loglik = float(np.sum(constant - 0.5 * np.log(var)
                                  - 0.5 * (nu + 1.0) * np.log1p(ratio / (nu - 2.0))))
        else:
            loglik = float(-0.5 * np.sum(np.log(2.0 * math.pi) + np.log(var)
                                         + ratio))
        return -loglik

    def _install(self, params: dict, variance: np.ndarray,
                 r: np.ndarray) -> None:
        self.mu_ = params["mu"]
        self.omega_ = params["omega"]
        self.alpha_ = params["alpha"]
        self.beta_ = params["beta"]
        self.gamma_ = params["gamma"]
        self.nu_ = params["nu"]
        self.params_ = params
        self.conditional_variance_ = variance
        self.conditional_volatility_ = np.sqrt(variance)
        self.residuals_ = r - self.mu_
        self.standardised_ = self.residuals_ / self.conditional_volatility_
        if self.variant == "egarch":
            self.persistence_ = float(self.beta_[0])
        elif self.variant == "igarch":
            self.persistence_ = 1.0
        else:
            self.persistence_ = float(self.alpha_.sum() + self.beta_.sum()
                                      + 0.5 * self.gamma_)
        used = self.residuals_[self.start_:]
        var = variance[self.start_:]
        if self.variant == "garch_t":
            nu = self.nu_
            constant = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                        - 0.5 * math.log(math.pi * (nu - 2.0)))
            loglik = float(np.sum(constant - 0.5 * np.log(var)
                                  - 0.5 * (nu + 1.0)
                                  * np.log1p(used ** 2 / (var * (nu - 2.0)))))
        else:
            loglik = float(-0.5 * np.sum(np.log(2.0 * math.pi) + np.log(var)
                                         + used ** 2 / var))
        self.loglik_ = loglik
        self.n_used_ = int(used.size)
        self.k_ = 2 + self.p + self.q
        if self.variant in ("gjr_garch", "egarch"):
            self.k_ += 1
        if self.variant == "garch_t":
            self.k_ += 1
        if self.variant == "igarch":
            self.k_ -= 1
        self.aic_ = float(-2.0 * loglik + 2.0 * self.k_)
        self.bic_ = float(-2.0 * loglik + self.k_ * math.log(self.n_used_))

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int) -> np.ndarray:
        """Variance forecasts, shape `(horizon,)`."""
        self._check_fitted()
        horizon = _check_horizon(horizon)
        errors = self.residuals_
        variance = self.conditional_variance_
        if self.variant == "egarch":
            path = np.empty(horizon)
            log_variance = math.log(variance[-1])
            z = errors[-1] / math.sqrt(variance[-1])
            log_variance = (math.log(self.omega_) + self.alpha_[0]
                            * (abs(z) - SQRT_2_OVER_PI) + self.gamma_ * z
                            + self.beta_[0] * log_variance)
            path[0] = math.exp(log_variance)
            for h in range(1, horizon):
                log_variance = math.log(self.omega_) + self.beta_[0] * log_variance
                path[h] = math.exp(log_variance)
            return path
        out = np.empty(horizon)
        observed = errors ** 2
        future = []
        for h in range(horizon):
            value = self.omega_
            arch_lags = self.q if self.variant == "arch" else self.p
            for lag in range(1, arch_lags + 1):
                past = future[h - lag] if lag <= h else observed[h - lag]
                value += self.alpha_[lag - 1] * past
            if self.beta_.size:
                for lag in range(1, self.q + 1):
                    past = out[h - lag] if lag <= h else variance[h - lag]
                    value += self.beta_[lag - 1] * past
            if self.gamma_:
                if h == 0:
                    value += (self.gamma_ * observed[-1]
                              * (errors[-1] < 0.0))
                else:
                    value += self.gamma_ * 0.5 * future[h - 1]
            out[h] = value
            future.append(value)
        return out

    def interval(self, horizon: int, level: float = 0.95
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Return band from the variance forecast and the innovation law."""
        self._check_fitted()
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        variance = self.predict(horizon)
        if self.variant == "garch_t":
            z = _t_quantile(0.5 + level / 2.0, self.nu_
                            ) * math.sqrt(self.nu_ / (self.nu_ - 2.0))
        else:
            z = NormalDist().inv_cdf(0.5 + level / 2.0)
        spread = z * np.sqrt(variance)
        return self.mu_ - spread, self.mu_ + spread

    def _check_fitted(self):
        if not hasattr(self, "omega_"):
            raise RuntimeError("call fit before predict or interval")


def _check_horizon(horizon) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _simulate(n: int, rng: np.random.Generator, *, omega: float, alpha: float,
              beta: float = 0.0, gamma: float = 0.0, nu: float | None = None,
              arch: bool = False, integrated: bool = False) -> np.ndarray:
    errors = np.zeros(n)
    if integrated:
        variance = np.ones(n)
    elif arch:
        variance = np.full(n, omega / (1.0 - alpha))
    else:
        variance = np.full(n, omega / (1.0 - alpha - beta - 0.5 * gamma))
    if nu is None:
        shocks = rng.normal(0.0, 1.0, n)
    else:
        shocks = rng.standard_t(nu, n) / math.sqrt(nu / (nu - 2.0))
    for t in range(1, n):
        if arch:
            variance[t] = omega + alpha * errors[t - 1] ** 2
        else:
            variance[t] = (omega + alpha * errors[t - 1] ** 2
                           + beta * variance[t - 1]
                           + gamma * errors[t - 1] ** 2 * (errors[t - 1] < 0.0))
        errors[t] = math.sqrt(variance[t]) * shocks[t]
    return errors


def demo() -> None:
    rng = np.random.default_rng(0)
    print("arch. 1000 points of an ARCH(1) with omega = 0.5, alpha = 0.5:")
    r = _simulate(1000, rng, omega=0.5, alpha=0.5, arch=True)
    model = GARCH("arch").fit(r)
    manual = (model.omega_ + model.alpha_[0] * model.residuals_[-1] ** 2)
    assert model.beta_.size == 0
    assert np.isclose(model.predict(1)[0], manual)
    assert abs(model.omega_ - 0.5) < 0.15
    assert abs(model.alpha_[0] - 0.5) < 0.12
    print(f"  omega {model.omega_:.4f}, alpha {model.alpha_[0]:.4f}; no beta "
          f"term, and the one-step forecast matches the recursion")

    rng = np.random.default_rng(1)
    print("garch. 1000 points with omega = 0.05, alpha = 0.08, beta = 0.9:")
    r = _simulate(1000, rng, omega=0.05, alpha=0.08, beta=0.9)
    model = GARCH("garch").fit(r)
    plain = GARCH("arch").fit(r)
    long_run = model.omega_ / (1.0 - model.alpha_.sum() - model.beta_.sum())
    predicted = model.predict(2000)[-1]
    assert model.persistence_ < 1.0
    assert np.isclose(predicted, long_run, rtol=1e-6)
    assert model.aic_ < plain.aic_
    assert abs(float(np.var(model.standardised_)) - 1.0) < 0.1
    print(f"  alpha {model.alpha_[0]:.4f}, beta {model.beta_[0]:.4f}, "
          f"persistence {model.persistence_:.4f}")
    print(f"  variance forecast converges to {predicted:.4f}, the formula "
          f"omega/(1-alpha-beta) = {long_run:.4f}")
    print(f"  AIC {model.aic_:.2f} against {plain.aic_:.2f} for ARCH, and the "
          f"standardised variance is {np.var(model.standardised_):.3f}")

    rng = np.random.default_rng(2)
    print("gjr_garch. 1000 points with a true leverage term gamma = 0.1:")
    r = _simulate(1000, rng, omega=0.05, alpha=0.08, beta=0.8, gamma=0.1)
    model = GARCH("gjr_garch").fit(r)
    symmetric = GARCH("garch").fit(r)
    assert model.gamma_ > 0.05
    assert model.aic_ < symmetric.aic_
    print(f"  gamma {model.gamma_:.4f}, alpha {model.alpha_[0]:.4f}, beta "
          f"{model.beta_[0]:.4f}, persistence {model.persistence_:.4f}")
    print(f"  AIC {model.aic_:.2f} against {symmetric.aic_:.2f} without the "
          f"asymmetry: negative shocks raise variance more")

    rng = np.random.default_rng(3)
    print("egarch. 1000 points with gamma = -0.08 in the log-variance:")
    shocks = rng.normal(0.0, 1.0, 1000)
    r = np.zeros(1000)
    log_variance = np.zeros(1000)
    for t in range(1, 1000):
        log_variance[t] = (0.02
                           + 0.1 * (abs(shocks[t - 1]) - SQRT_2_OVER_PI)
                           - 0.08 * shocks[t - 1]
                           + 0.9 * log_variance[t - 1])
        r[t] = math.exp(0.5 * log_variance[t]) * shocks[t]
    model = GARCH("egarch").fit(r)
    z = model.residuals_[-1] / math.sqrt(model.conditional_variance_[-1])
    manual = math.exp(math.log(model.omega_)
                      + model.alpha_[0] * (abs(z) - SQRT_2_OVER_PI)
                      + model.gamma_ * z
                      + model.beta_[0] * math.log(model.conditional_variance_[-1]))
    assert model.gamma_ < 0.0
    assert np.isclose(model.predict(1)[0], manual)
    print(f"  omega {model.omega_:.4f}, alpha {model.alpha_[0]:.4f}, gamma "
          f"{model.gamma_:.4f}, beta {model.beta_[0]:.4f}")
    print("  the first-step forecast matches the log-variance recursion "
          "written by hand")

    rng = np.random.default_rng(4)
    print("igarch. 1000 points with alpha = 0.2 and beta = 0.8, so persistence 1:")
    r = _simulate(1000, rng, omega=0.05, alpha=0.2, beta=0.8, integrated=True)
    model = GARCH("igarch").fit(r)
    path = model.predict(5)
    assert np.isclose(model.alpha_[0] + model.beta_[0], 1.0)
    assert np.allclose(np.diff(path), model.omega_, rtol=1e-10)
    print(f"  alpha {model.alpha_[0]:.4f}, beta {model.beta_[0]:.4f}, so "
          f"alpha + beta is exactly one; each forecast step adds omega = "
          f"{model.omega_:.4f}")

    rng = np.random.default_rng(5)
    print("garch_t. 1000 points with Student-t shocks, nu = 5:")
    r = _simulate(1000, rng, omega=0.05, alpha=0.08, beta=0.9, nu=5.0)
    model = GARCH("garch_t").fit(r)
    gaussian = GARCH("garch").fit(r)
    lower, upper = model.interval(3)
    normal = NormalDist().inv_cdf(0.975)
    ratio = (upper[0] - lower[0]) / (2.0 * normal
                                     * math.sqrt(model.predict(1)[0]))
    assert 4.0 < model.nu_ < 7.0
    assert model.loglik_ > gaussian.loglik_
    assert ratio > 1.3
    print(f"  nu {model.nu_:.4f}, log-likelihood {model.loglik_:.2f} against "
          f"{gaussian.loglik_:.2f} for a Gaussian")
    print(f"  the 95% band is {ratio:.2f} times the Gaussian width at h=1, "
          f"which is the tail the t likelihood pays for")

    bad = [
        lambda: GARCH("garch_lite"),
        lambda: GARCH("arch", p=2),
        lambda: GARCH("garch", p=0),
        lambda: GARCH("garch", q=0),
        lambda: GARCH("garch", q=1.5),
        lambda: GARCH("egarch", p=2),
        lambda: GARCH("gjr_garch", q=2),
        lambda: GARCH("igarch", p=2),
        lambda: GARCH("garch_t", p=2),
        lambda: GARCH("garch_t", nu=2.0),
        lambda: GARCH("garch_t", nu=np.nan),
        lambda: GARCH("garch").fit(np.ones((200, 1))),
        lambda: GARCH("garch").fit(np.r_[r[:49], np.nan]),
        lambda: GARCH("garch").fit(np.ones(100)),
        lambda: GARCH("garch").fit(np.zeros(100)),
        lambda: GARCH("garch").predict(3),
        lambda: GARCH("garch").fit(r).predict(0),
        lambda: GARCH("garch").fit(r).predict(2.5),
        lambda: GARCH("garch").fit(r).interval(3, 0.0),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, an ARCH order with a p term, a GARCH order with p or q")
    print("  below one, a fractional order, an asymmetric variant with p or q")
    print("  other than one, a t model with nu at or below two or non-finite, a")
    print("  2-d series, a non-finite or constant series, too few observations,")
    print("  use before fit, a zero or fractional horizon, and a level outside")
    print("  (0, 1)")
    print("garch demo passed")


if __name__ == "__main__":
    demo()
