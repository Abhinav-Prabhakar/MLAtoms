"""Particle filter — sequential Monte Carlo when the posterior leaves Gaussian.

Chapter:   05-probabilistic
Companion: particle_filter.md
Tier 2 · 6 variants: sir, sis, auxiliary_pf, systematic_resample,
stratified_resample, regularized_pf

    w_t^i ∝ w_{t-1}^i · g(z_t | x_t^i),     p(z_{1..T}) ≈ Π_t Σ_i w_{t-1}^i g(z_t|x_t^i)

A cloud of weighted samples replaces the Gaussian. Each particle is a hypothesis
about the state; the weights say how well each hypothesis predicted the
observation; and resampling kills the hypotheses the data has ruled out. Nothing
is linear and nothing is Gaussian — the price is Monte Carlo error that decays as
1/√N, and a sample set that can lose the answer entirely if the weights collapse.

    sir                  bootstrap: propagate, weight, resample when ESS is low
    sis                  propagate and weight only — the weights degenerate, by design
    auxiliary_pf         resample *before* propagating, using one-step predictive fit
    systematic_resample  one uniform draw, N evenly spaced positions
    stratified_resample  one uniform draw per stratum
    regularized_pf       resample, then jitter with a kernel to restore diversity

The first three differ in *when* they resample; the next two in *how* they draw
the ancestors; the last in what to do after. Every one of them is a different
answer to the same question: where do you spend the variance?
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

VARIANTS = frozenset(
    ["sir", "sis", "auxiliary_pf", "systematic_resample", "stratified_resample",
     "regularized_pf"]
)

LOG2PI = math.log(2.0 * math.pi)

Init = Callable[[int, np.random.Generator], np.ndarray]
Transition = Callable[[np.ndarray, np.random.Generator], np.ndarray]
LogLik = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class PFResult:
    """Weighted moments over `T` steps.

    `mean` is `(T, d)`, `cov` is `(T, d, d)`, `ess` is the effective sample size
    `(T,)`, and `loglik` is `log` of the evidence estimate `Π_t Σ_i w g`. The
    particles themselves are not returned: they are `N` vectors per step, and
    anyone who needs them can re-run with `return_particles=True`.
    """

    mean: np.ndarray
    cov: np.ndarray
    loglik: float
    ess: np.ndarray
    n_resample: int
    particles: np.ndarray | None = None
    weights: np.ndarray | None = None


def particle_filter(z: np.ndarray, variant: str = "sir", *, init: Init,
                    transition: Transition, log_likelihood: LogLik,
                    n_particles: int = 2000, seed: int = 0,
                    ess_threshold: float = 0.5,
                    log_predictive: LogLik | None = None,
                    jitter: float = 1.0,
                    return_particles: bool = False) -> PFResult:
    """Run one variant over `z` of shape `(T, m)`.

    Parameters
    ----------
    init : callable
        `init(N, rng) -> (N, d)`, the prior draw.
    transition : callable
        `transition(X, rng) -> (N, d)`, the state noise included. Particles move
        independently, so the whole cloud is propagated in one vectorised call.
    log_likelihood : callable
        `log_likelihood(z_t, X) -> (N,)`, the log of `p(z_t | x)` for every
        particle, up to a constant that does not depend on `x`.
    log_predictive : callable, optional
        `log_predictive(z_t, X) -> (N,)`, `p(z_t | x_{t-1})` with the transition
        integrated out. Required by `auxiliary_pf`; `0` is what the bootstrap
        filter implicitly uses.
    ess_threshold : float
        Resample when `ESS < ess_threshold * N`. Only `sir`, the two resampling
        variants and `regularized_pf` resample at all.
    jitter : float
        Multiplier on the Silverman bandwidth for `regularized_pf`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    if variant == "auxiliary_pf" and log_predictive is None:
        raise ValueError(
            "auxiliary_pf needs log_predictive(z_t, X_prev); without it the method "
            "degenerates to sir — pass it, or use variant='sir'"
        )
    z = np.atleast_2d(np.asarray(z, dtype=float))
    T = z.shape[0]
    rng = np.random.default_rng(seed)

    X = np.asarray(init(n_particles, rng), dtype=float)
    N, d = X.shape
    lw = np.full(N, -math.log(N))  # normalised log weights
    mean = np.empty((T, d))
    cov = np.empty((T, d, d))
    ess = np.empty(T)
    loglik = 0.0
    n_resample = 0

    for t in range(T):
        if variant == "auxiliary_pf":
            # First stage: weight each *ancestor* by how well it would have
            # predicted z_t with the transition integrated out, then move only
            # the ancestors that survive. The normaliser of this stage is the
            # evidence increment; the second stage divides it back out.
            lpred = log_predictive(z[t], X)
            lw = lw + lpred
            loglik += _logsumexp(lw)
            lw -= _logsumexp(lw)
            idx = _resample(lw, "systematic", rng)
            X = np.asarray(transition(X[idx], rng), dtype=float)
            n_resample += 1
            lw = log_likelihood(z[t], X) - lpred[idx]
            lw -= _logsumexp(lw)
        else:
            X = np.asarray(transition(X, rng), dtype=float)
            lw = lw + log_likelihood(z[t], X)
            loglik += _logsumexp(lw)
            lw -= _logsumexp(lw)

        w = np.exp(lw)
        mean[t] = w @ X
        cov[t] = _weighted_cov(X, w)
        ess[t] = 1.0 / np.sum(w**2)

        if variant == "sis":
            continue
        if ess[t] < ess_threshold * N:
            scheme = {"systematic_resample": "systematic",
                      "stratified_resample": "stratified"}.get(variant, "multinomial")
            idx = _resample(lw, scheme, rng)
            X = X[idx]
            if variant == "regularized_pf":
                X = _regularize(X, jitter, rng)
            lw = np.full(N, -math.log(N))
            n_resample += 1

    out = PFResult(mean, cov, float(loglik), ess, n_resample)
    if return_particles:
        out.particles, out.weights = X, np.exp(lw)
    return out


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def _resample(lw: np.ndarray, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """Ancestor indices drawn with probability `exp(lw)`.

    Multinomial draws `N` independent uniforms; systematic draws one and steps
    by exactly `1/N`, so the counts are deterministic given the offset;
    stratified draws one per stratum. Systematatic and stratified both have
    lower variance than multinomial at the same cost, which is why they are the
    default in every serious implementation.
    """
    N = lw.size
    w = np.exp(lw)
    cdf = np.cumsum(w)
    cdf[-1] = 1.0  # guard against the sum landing a ulp below 1
    if scheme == "multinomial":
        return np.searchsorted(cdf, rng.random(N))
    if scheme == "systematic":
        u = (rng.random() + np.arange(N)) / N
        return np.searchsorted(cdf, u)
    if scheme == "stratified":
        u = (np.arange(N) + rng.random(N)) / N
        return np.searchsorted(cdf, u)
    raise ValueError(f"unknown resampling scheme {scheme!r}")


def _regularize(X: np.ndarray, factor: float,
                rng: np.random.Generator) -> np.ndarray:
    """Jitter resampled particles with a Gaussian kernel, Silverman bandwidth.

    Resampling copies particles, so duplicates are the same number repeated and
    the cloud loses the diversity the next step needs. The bandwidth is the
    rule-of-thumb for kernel density estimation, scaled down by `factor`:
    `h = factor · (4/(N(d+2)))^{1/(d+4)}`, applied to the empirical covariance.
    """
    N, d = X.shape
    cov = np.cov(X, rowvar=False) if N > 1 else np.zeros((d, d))
    cov = np.atleast_2d(cov)
    h = factor * (4.0 / (N * (d + 2.0))) ** (1.0 / (d + 4.0))
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(d))
    except np.linalg.LinAlgError:
        return X
    return X + h * rng.standard_normal((N, d)) @ L.T


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _logsumexp(a: np.ndarray) -> float:
    """`log Σ exp(a)`, with the max factored out so weights cannot underflow."""
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _weighted_cov(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`Σ_i w_i (x_i − m)(x_i − m)ᵀ` for a weighted cloud, shape `(d, d)`."""
    m = w @ X
    Z = X - m
    return (Z * w[:, None]).T @ Z


def _gauss_logpdf_1d(resid: np.ndarray, var: float) -> np.ndarray:
    """`log N(resid; 0, var)` for a vector of residuals — the demo's likelihood."""
    return -0.5 * (LOG2PI + math.log(var) + resid**2 / var)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- a model where the exact posterior is available ---------------------
    # Scalar autoregression, observed with noise. The Kalman filter is
    # the exact answer, so every particle-filter claim becomes a comparison
    # against a number rather than against intuition.
    T, q, r = 40, 0.5, 0.5
    phi = 0.9
    A, C = np.array([[phi]]), np.array([[1.0]])
    Q, R = np.array([[q]]), np.array([[r]])
    m0, P0 = np.array([0.0]), np.array([[1.0]])

    x = np.zeros(T)
    z = np.empty((T, 1))
    for t in range(T):
        if t:
            x[t] = phi * x[t - 1] + rng.normal(0.0, math.sqrt(q))
        z[t] = x[t] + rng.normal(0.0, math.sqrt(r))

    from kalman import kalman_filter  # sibling module: same folder, same sys.path

    kf = kalman_filter(z, "kf", transition=A, observation=C, Q=Q, R=R, m0=m0, P0=P0)

    def init(n, g):
        return m0 + math.sqrt(P0[0, 0]) * g.standard_normal((n, 1))

    def transition(X, g):
        return phi * X + math.sqrt(q) * g.standard_normal(X.shape)

    def log_lik(z_t, X):
        return _gauss_logpdf_1d(z_t[0] - X[:, 0], r)

    def log_pred(z_t, X_prev):
        # p(z_t | x_{t-1}) with the transition integrated out: the noise adds
        return _gauss_logpdf_1d(z_t[0] - phi * X_prev[:, 0], q + r)

    common = dict(init=init, transition=transition, log_likelihood=log_lik,
                  n_particles=4000, seed=1)
    sir = particle_filter(z, "sir", **common)
    sis = particle_filter(z, "sis", **common)
    apf = particle_filter(z, "auxiliary_pf", log_predictive=log_pred, **common)

    # the cloud reproduces the exact posterior mean and variance
    assert np.abs(sir.mean[:, 0] - kf.mean[:, 0]).max() < 0.2
    assert np.abs(sir.cov[:, 0, 0] - kf.cov[:, 0, 0]).max() < 0.15
    # and the evidence estimate is the exact likelihood, up to Monte Carlo error
    assert abs(sir.loglik - kf.loglik) < 1.0, (sir.loglik, kf.loglik)
    # the auxiliary filter targets the same posterior, by a different route
    assert np.abs(apf.mean[:, 0] - sir.mean[:, 0]).max() < 0.2
    assert abs(apf.loglik - kf.loglik) < 1.0, (apf.loglik, kf.loglik)
    print(
        f"particle_filter: sir — 4000 particles track the exact Kalman posterior "
        f"(max mean error {np.abs(sir.mean[:, 0] - kf.mean[:, 0]).max():.3f}); "
        f"log p(z) = {sir.loglik:.2f} against {kf.loglik:.2f}"
    )

    # --- why resampling is not optional ------------------------------------
    # Without it the weights concentrate on one particle and the cloud stops
    # representing anything: the effective sample size falls off a cliff.
    assert sis.n_resample == 0 and sir.n_resample > 0
    assert sis.ess[-1] < 0.1 * 4000 < sir.ess[-1]
    print(
        f"particle_filter: sis — no resampling: ESS falls from {sis.ess[0]:.0f} to "
        f"{sis.ess[-1]:.1f} of 4000 particles; sir resampled {sir.n_resample} times "
        f"and ends at ESS {sir.ess[-1]:.0f}"
    )

    # --- the auxiliary filter buys variance, not accuracy -------------------
    # Same target, same particle count: the evidence estimate should be less
    # variable across seeds, because the first stage spends particles where the
    # next observation is plausible instead of uniformly.
    runs = 40
    ll_sir, ll_apf = [], []
    for s in range(runs):
        kw = dict(common, seed=100 + s, n_particles=1000)
        ll_sir.append(particle_filter(z, "sir", **kw).loglik)
        ll_apf.append(particle_filter(z, "auxiliary_pf", log_predictive=log_pred, **kw).loglik)
    spread_sir = float(np.std(ll_sir))
    spread_apf = float(np.std(ll_apf))
    assert spread_apf < spread_sir, (spread_apf, spread_sir)
    print(
        f"particle_filter: auxiliary_pf — evidence spread over {runs} seeds "
        f"{spread_apf:.3f} nats against {spread_sir:.3f} for sir, at the same cost"
    )

    # --- the resampling schemes -------------------------------------------
    # With uniform weights every scheme must return the particles unchanged in
    # expectation; systematic and stratified do it exactly, one copy each,
    # because their positions tile the unit interval.
    uniform = np.full(400, -math.log(400))
    probe = np.random.default_rng(7)
    for scheme in ("systematic", "stratified"):
        counts = np.bincount(_resample(uniform, scheme, probe), minlength=400)
        assert np.all(counts == 1), (scheme, counts.min(), counts.max())
    multi = np.bincount(_resample(uniform, "multinomial", probe), minlength=400)
    assert abs(multi.mean() - 1.0) < 0.1 and multi.max() > 1
    # and the two low-variance schemes are the ones wired to the variants
    sysr = particle_filter(z, "systematic_resample", **common)
    strr = particle_filter(z, "stratified_resample", **common)
    assert np.abs(sysr.mean - sir.mean).max() < 0.2
    assert abs(sysr.loglik - kf.loglik) < 1.0
    print(
        f"particle_filter: systematic_resample — uniform weights give exactly one "
        f"copy per particle, where multinomial gives 0..{multi.max()}; evidence "
        f"{sysr.loglik:.2f} vs {strr.loglik:.2f} for stratified"
    )

    # --- regularisation against sample impoverishment ----------------------
    # A deterministic transition and repeated resampling: every copy of the
    # surviving particle is identical, and the transition keeps it that way, so
    # the cloud ends as N copies of one number. Jitter restores the spread that
    # the next likelihood evaluation needs.
    def trans_det(X, g):
        return phi * X

    def lik_det(z_t, X):
        return _gauss_logpdf_1d(z_t[0] - X[:, 0], r)

    z3 = 1.0e-3 * rng.standard_normal((T, 1))
    plain = particle_filter(z3, "sir", init=init, transition=trans_det,
                            log_likelihood=lik_det, n_particles=500, seed=3,
                            ess_threshold=1.0, return_particles=True)
    reg = particle_filter(z3, "regularized_pf", init=init, transition=trans_det,
                          log_likelihood=lik_det, n_particles=500, seed=3,
                          ess_threshold=1.0, return_particles=True)
    uniq_plain = np.unique(plain.particles[:, 0]).size
    uniq_reg = np.unique(reg.particles[:, 0]).size
    assert uniq_plain < 50 < uniq_reg, (uniq_plain, uniq_reg)
    assert abs(reg.mean[-1, 0] - plain.mean[-1, 0]) < 0.2
    print(
        f"particle_filter: regularized_pf — a deterministic transition leaves "
        f"{uniq_plain} distinct particles out of 500 after resampling; jitter keeps "
        f"all {uniq_reg}, at the same mean to {abs(reg.mean[-1, 0] - plain.mean[-1, 0]):.3f}"
    )

    print("particle_filter demo passed")


if __name__ == "__main__":
    main()
