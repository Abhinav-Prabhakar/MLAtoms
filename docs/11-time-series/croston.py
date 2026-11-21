"""Croston — forecasting intermittent demand one arrival at a time.

Chapter:   11-time-series
Companion: croston.md
Tier 3 · 4 variants: croston, sba, tsb, optimized_croston
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("croston", "sba", "tsb", "optimized_croston")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _check_series(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.size < 10:
        raise ValueError("y needs at least ten observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite")
    if np.any(y < 0.0):
        raise ValueError("y must be non-negative")
    if not np.any(y > 0.0):
        raise ValueError("y needs at least one positive demand")
    return y


def _check_rate(value, name: str) -> float:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return value


def _poisson_quantile(mean: float, probability: float) -> int:
    """Smallest k with P(Poisson(mean) <= k) >= probability."""
    if mean <= 0.0:
        return 0
    term = float(np.exp(-mean))
    total = term
    value = 0
    while total < probability and value < 100000:
        value += 1
        term *= mean / value
        total += term
    return value


# --------------------------------------------------------------------------- #
# The two engines
# --------------------------------------------------------------------------- #
def _croston(y: np.ndarray, alpha: float, correction: float = 1.0) -> dict:
    n = y.size
    positive = np.flatnonzero(y > 0.0)
    first = int(positive[0])
    size = float(y[first])
    interval = 1.0
    gap = 1.0
    fitted = np.empty(n)
    fitted[:first + 1] = size / interval
    for t in range(first + 1, n):
        fitted[t] = size / interval
        if y[t] > 0.0:
            size = alpha * y[t] + (1.0 - alpha) * size
            interval = alpha * gap + (1.0 - alpha) * interval
            gap = 1.0
        else:
            gap += 1.0
    residuals = y - fitted
    burn = first + 1
    used = residuals[burn:]
    return {"fitted": fitted, "residuals": residuals, "burn": burn,
            "sse": float(used @ used), "rate": correction * size / interval,
            "size": float(size), "interval": float(interval),
            "correction": float(correction)}


def _tsb(y: np.ndarray, alpha: float, beta: float) -> dict:
    n = y.size
    positive = np.flatnonzero(y > 0.0)
    first = int(positive[0])
    size = float(y[first])
    probability = 0.5
    fitted = np.empty(n)
    fitted[:first + 1] = probability * size
    for t in range(first + 1, n):
        fitted[t] = probability * size
        if y[t] > 0.0:
            probability = alpha + (1.0 - alpha) * probability
            size = beta * y[t] + (1.0 - beta) * size
        else:
            probability = (1.0 - alpha) * probability
    residuals = y - fitted
    burn = first + 1
    used = residuals[burn:]
    return {"fitted": fitted, "residuals": residuals, "burn": burn,
            "sse": float(used @ used), "rate": probability * size,
            "size": float(size), "probability": float(probability),
            "correction": 1.0}


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class Croston:
    """Croston, SBA, TSB and an alpha-optimised Croston.

    Parameters
    ----------
    variant : str
        One of `VARIANTS`.
    alpha : float
        Smoothing parameter for the demand size and interval (`croston`, `sba`)
        or for the demand occurrence probability (`tsb`). `optimized_croston`
        ignores it and searches instead.
    beta : float, optional
        Smoothing parameter for the demand size in `tsb`; defaults to `alpha`.

    Notes
    -----
    `fit` stores `rate_`, the constant per-period demand forecast, plus
    `fitted_`, `residuals_`, `sse_`, `sigma_`, `burn_`, and the components
    `size_`, `interval_` or `probability_` depending on the variant.
    `predict` repeats the rate; `interval` returns Poisson quantiles for the
    cumulative demand over the requested horizon.
    """

    def __init__(self, variant: str = "croston", *, alpha: float = 0.1,
                 beta: float | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.alpha = _check_rate(alpha, "alpha")
        self.beta = None if beta is None else _check_rate(beta, "beta")

    # -- fitting ---------------------------------------------------------- #
    def fit(self, y: np.ndarray) -> Croston:
        """Estimate the demand rate from an intermittent series."""
        y = _check_series(y)
        self.n_ = y.size
        self.demand_count_ = int(np.count_nonzero(y))
        if self.variant == "optimized_croston":
            candidates = []
            best = None
            for alpha in np.arange(0.02, 0.99, 0.02):
                result = _croston(y, float(alpha))
                candidates.append((float(alpha), result["sse"]))
                if best is None or result["sse"] < best["sse"]:
                    best = result
                    best_alpha = float(alpha)
            assert best is not None
            self.candidates_ = candidates
            self.alpha_ = best_alpha
        elif self.variant == "tsb":
            beta = self.alpha if self.beta is None else self.beta
            best = _tsb(y, self.alpha, beta)
            self.alpha_ = self.alpha
            self.beta_ = beta
            self.candidates_ = [(self.alpha, best["sse"])]
        else:
            correction = 1.0 if self.variant == "croston" else 1.0 - 0.5 * self.alpha
            best = _croston(y, self.alpha, correction)
            self.alpha_ = self.alpha
            self.candidates_ = [(self.alpha, best["sse"])]
        self.rate_ = float(best["rate"])
        self.fitted_ = best["fitted"]
        self.residuals_ = best["residuals"]
        self.burn_ = int(best["burn"])
        self.sse_ = float(best["sse"])
        self.sigma_ = float(np.sqrt(self.sse_ / max(y.size - self.burn_, 1)))
        self.size_ = float(best["size"])
        if self.variant == "tsb":
            self.probability_ = float(best["probability"])
        else:
            self.interval_ = float(best["interval"])
            self.correction_ = float(best["correction"])
        return self

    # -- forecasting ------------------------------------------------------ #
    def predict(self, horizon: int = 12) -> np.ndarray:
        """Return the constant per-period rate for each step."""
        if not hasattr(self, "rate_"):
            raise RuntimeError("fit the model before predicting")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return np.full(int(horizon), self.rate_)

    def interval(self, horizon: int = 12, level: float = 0.9
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Poisson quantiles for cumulative demand over each horizon."""
        if not np.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly between zero and one")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        tail = 0.5 * (1.0 - level)
        means = self.rate_ * np.arange(1, int(horizon) + 1)
        lower = np.array([_poisson_quantile(mean, tail) for mean in means])
        upper = np.array([_poisson_quantile(mean, 1.0 - tail) for mean in means])
        return lower, upper


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _intermittent(n: int, rng: np.random.Generator, probability: float = 0.2,
                  size: float = 5.0) -> np.ndarray:
    occurrence = rng.random(n) < probability
    demand = rng.poisson(size, n).astype(float)
    return np.where(occurrence, demand, 0.0)


def demo() -> None:
    rng = np.random.default_rng(0)
    print("croston. 600 periods, 20% demand probability, mean size five:")
    values = _intermittent(600, rng)
    model = Croston("croston", alpha=0.1).fit(values)
    empirical = float(values.mean())
    truth = 1.0
    assert abs(model.rate_ - truth) < 0.25
    assert model.demand_count_ == int(np.count_nonzero(values))
    print(f"  {model.demand_count_} demands, size {model.size_:.4f}, "
          f"interval {model.interval_:.4f}")
    print(f"  rate {model.rate_:.4f} against an empirical mean of "
          f"{empirical:.4f} and a true rate of {truth:.2f}")

    print("sba. the same fit with the finite-sample bias correction:")
    sba = Croston("sba", alpha=0.1).fit(values)
    factor = 1.0 - 0.5 * sba.alpha_
    assert abs(sba.rate_ - model.rate_ * factor) < 1e-12
    assert sba.rate_ < model.rate_
    print(f"  correction factor {factor:.4f}, rate {sba.rate_:.4f} against "
          f"{model.rate_:.4f} for plain Croston")
    print(f"  distance to the empirical mean {abs(sba.rate_ - empirical):.4f} "
          f"against {abs(model.rate_ - empirical):.4f}")

    rng = np.random.default_rng(1)
    print("tsb. demand probability falls from 35% to 10% halfway through:")
    first = _intermittent(150, rng, probability=0.35, size=4.0)
    second = _intermittent(150, rng, probability=0.10, size=4.0)
    values = np.concatenate([first, second])
    croston = Croston("croston", alpha=0.1).fit(values)
    tsb = Croston("tsb", alpha=0.15, beta=0.15).fit(values)
    local = float(values[-100:].mean())
    tsb_error = float(np.mean(np.abs(tsb.residuals_[-100:])))
    croston_error = float(np.mean(np.abs(croston.residuals_[-100:])))
    assert tsb_error < croston_error
    print(f"  local rate over the last 100 periods {local:.4f}; terminal "
          f"rates TSB {tsb.rate_:.4f}, Croston {croston.rate_:.4f}")
    print(f"  one-step MAE over the last 100 periods {tsb_error:.4f} against "
          f"{croston_error:.4f}; TSB probability {tsb.probability_:.4f}")

    print("optimized_croston. the alpha grid on the changing series:")
    optimised = Croston("optimized_croston").fit(values)
    fixed = Croston("croston", alpha=0.1).fit(values)
    assert optimised.sse_ <= fixed.sse_ + 1e-9
    print(f"  chosen alpha {optimised.alpha_:.2f}, SSE {optimised.sse_:.2f} "
          f"against {fixed.sse_:.2f} at alpha 0.1")
    print(f"  rate {optimised.rate_:.4f} against {fixed.rate_:.4f}")

    print("  Poisson intervals for cumulative demand:")
    lower, upper = model.interval(12, 0.9)
    assert np.all(upper >= lower)
    assert upper[-1] >= upper[0]
    assert np.all(np.diff(lower) >= 0)
    assert np.all(np.diff(upper) >= 0)
    print(f"  interval at h=1 [{lower[0]}, {upper[0]}], at h=12 "
          f"[{lower[-1]}, {upper[-1]}]")
    simulated = rng.poisson(model.rate_, size=(4000, 12)).cumsum(axis=1)
    coverage = np.mean((simulated >= lower) & (simulated <= upper), axis=0)
    assert np.all(coverage >= 0.90)
    assert np.all(coverage <= 0.99)
    print(f"  simulated coverage from {coverage[0]:.4f} at h=1 to "
          f"{coverage[-1]:.4f} at h=12 for a nominal 0.90; the discrete")
    print("  Poisson quantiles are conservative when the mean is near one")

    bad = [
        lambda: Croston("croston_lite"),
        lambda: Croston("croston", alpha=0.0),
        lambda: Croston("croston", alpha=1.5),
        lambda: Croston("croston", alpha=np.nan),
        lambda: Croston("tsb", beta=0.0),
        lambda: Croston("tsb", beta=1.5),
        lambda: Croston("croston").fit(np.zeros((30, 2))),
        lambda: Croston("croston").fit(np.arange(5.0)),
        lambda: Croston("croston").fit(np.zeros(20)),
        lambda: Croston("croston").fit(-np.ones(20)),
        lambda: Croston("croston").fit(np.r_[np.arange(20.0), np.nan]),
        lambda: Croston("croston").predict(12),
        lambda: Croston("croston").fit(values).predict(0),
        lambda: Croston("croston").fit(values).interval(12, 1.0),
        lambda: Croston("croston").fit(values).interval(0, 0.9),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, an alpha or beta outside (0, 1] or non-finite, a 2-d or")
    print("  short series, an all-zero or negative series, a non-finite")
    print("  series, prediction before fitting, a zero horizon, and an")
    print("  interval level outside the open unit interval")
    print("croston demo passed")


if __name__ == "__main__":
    demo()
