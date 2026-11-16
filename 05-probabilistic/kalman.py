"""Kalman filter — exact Bayesian filtering for a linear-Gaussian chain.

Chapter:   05-probabilistic
Companion: kalman.md
Tier 2 · 6 variants: kf, ekf, ukf, information_filter, rts_smoother, missing_obs

    m_t = A m_{t-1},              P_t = A P_{t-1} Aᵀ + Q          (predict)
    m_t = m_t + K (z_t − C m_t),  P_t = P_t − K C P_t,  K = P Cᵀ S⁻¹   (update)

The posterior over the state is Gaussian at every step, and those two lines are
Bayes' rule written in moments: prediction is the law of total expectation,
update is conditioning on one observation. `K` is the fraction of the surprise
you accept, and `S = C P Cᵀ + R` is how large a surprise you expect.

    kf                  linear transition and observation
    ekf                 first-order linearisation at the current mean
    ukf                 unscented transform: 2n+1 sigma points, no Jacobian
    information_filter  the same recursion in precision form, additive updates
    rts_smoother        backward pass over the filtered moments
    missing_obs         partial observations; the update shrinks to the observed rows

The filter is optimal — the exact posterior, not an approximation — only for the
first and fourth variants. EKF and UKF are approximations whose error is second
order in the curvature of `f` and `h`; the demo shows both collapsing onto `kf`
to machine precision when the model really is linear, which is the cheapest
possible test that a Jacobian or a sigma-point rule is coded correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

VARIANTS = frozenset(
    ["kf", "ekf", "ukf", "information_filter", "rts_smoother", "missing_obs"]
)

LOG2PI = math.log(2.0 * math.pi)


@dataclass
class FilterResult:
    """Filtered (and optionally smoothed) moments over `T` steps.

    `mean` is `(T, n)` and `cov` is `(T, n, n)`; `loglik` sums the predictive
    log-densities, i.e. it is `log p(z_{1..T})`. `info` and `info_vector` hold
    the precision form `(J, y)`, filled only by `information_filter`;
    `smooth_mean` / `smooth_cov` only by `rts_smoother`.
    """

    mean: np.ndarray
    cov: np.ndarray
    loglik: float
    info: np.ndarray | None = None
    info_vector: np.ndarray | None = None
    smooth_mean: np.ndarray | None = None
    smooth_cov: np.ndarray | None = None
    n_skipped: int = field(default=0)


def kalman_filter(z: np.ndarray, variant: str = "kf", *, transition, observation,
                  Q: np.ndarray, R: np.ndarray, m0: np.ndarray,
                  P0: np.ndarray, alpha: float = 1.0, beta: float = 2.0,
                  kappa: float = 0.0) -> FilterResult:
    """Run one variant over `z` of shape `(T, m)`.

    `transition` and `observation` are matrices for `kf`, `information_filter`,
    `rts_smoother` and `missing_obs`, and callables — or `(callable, jacobian)`
    pairs — for `ekf` and `ukf`. `Q` is `(n, n)`, `R` is `(m, m)`, `m0` is
    `(n,)`, `P0` is `(n, n)`. A NaN in `z` means "not observed at this step".
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    z = np.atleast_2d(np.asarray(z, dtype=float))
    m0 = np.asarray(m0, dtype=float)
    n = m0.size
    for name, mat in (("Q", Q), ("P0", P0)):
        if np.shape(mat) != (n, n):
            raise ValueError(f"{name} must have shape {(n, n)}, got {np.shape(mat)}")
    R = np.atleast_2d(np.asarray(R, dtype=float))
    if z.shape[1] != R.shape[0]:
        raise ValueError("R must match the number of observation columns")
    if variant in ("kf", "information_filter", "rts_smoother", "missing_obs"):
        A, C = _as_matrix(transition, n), _as_matrix(observation, n)
        if variant == "information_filter":
            return _information_filter(z, A, C, Q, R, m0, P0)
        out = _kf(z, A, C, Q, R, m0, P0, skip=variant == "missing_obs")
        if variant == "rts_smoother":
            out.smooth_mean, out.smooth_cov = _rts(out, A, Q)
        return out
    f, F = _as_pair(transition)
    h, H = _as_pair(observation)
    if variant == "ekf":
        return _ekf(z, f, F, h, H, Q, R, m0, P0)
    return _ukf(z, f, h, Q, R, m0, P0, alpha, beta, kappa)


# --------------------------------------------------------------------------- #
# The exact recursion
# --------------------------------------------------------------------------- #
def _kf(z, A, C, Q, R, m0, P0, *, skip: bool) -> FilterResult:
    """Predict, then condition; `skip` keeps only the observed rows of `C`."""
    T = z.shape[0]
    mean = np.empty((T, m0.size))
    cov = np.empty((T, m0.size, m0.size))
    mu, P = m0.copy(), P0.copy()
    loglik = 0.0
    n_skipped = 0
    for t in range(T):
        mu = A @ mu
        P = A @ P @ A.T + Q
        if skip:
            obs = ~np.isnan(z[t])
        else:
            if np.isnan(z[t]).any():
                raise ValueError(
                    "z has a missing observation; use variant='missing_obs', which "
                    "restricts the update to the observed rows"
                )
            obs = np.ones(z.shape[1], dtype=bool)
        if not obs.any():
            # Nothing to condition on: the posterior *is* the prediction. The
            # update still happens, it is just the identity.
            n_skipped += 1
            mean[t], cov[t] = mu, P
            continue
        Ct = C[obs]
        St = Ct @ P @ Ct.T + R[np.ix_(obs, obs)]
        innov = z[t, obs] - Ct @ mu
        loglik += _gauss_logpdf(innov, St)
        K = np.linalg.solve(St, Ct @ P).T  # P Cᵀ S⁻¹, without an explicit inverse
        mu = mu + K @ innov
        P = P - K @ Ct @ P
        P = 0.5 * (P + P.T)
        mean[t], cov[t] = mu, P
    return FilterResult(mean, cov, float(loglik), n_skipped=n_skipped)


def _information_filter(z, A, C, Q, R, m0, P0) -> FilterResult:
    """The same posterior, carried as `(y, J) = (P⁻¹m, P⁻¹)`.

    The update is why this form exists: `J += Cᵀ R⁻¹ C` and `y += Cᵀ R⁻¹ z` are
    sums, so fusing a hundred sensors is a hundred additions and no inversion.
    The prediction is why it is not used everywhere: it needs two inversions,
    `J⁺ = (A J⁻¹ Aᵀ + Q)⁻¹` and `y⁺ = J⁺ A J⁻¹ y`.
    """
    T = z.shape[0]
    n = m0.size
    J = np.linalg.inv(P0)
    y = J @ m0
    mean = np.empty((T, n))
    cov = np.empty((T, n, n))
    info = np.empty((T, n, n))
    info_vector = np.empty((T, n))
    loglik = 0.0
    n_skipped = 0
    for t in range(T):
        P = np.linalg.inv(J)
        J_pred = np.linalg.inv(A @ P @ A.T + Q)
        y_pred = J_pred @ (A @ (P @ y))
        obs = ~np.isnan(z[t])
        if obs.any():
            Ct, Rt, zt = C[obs], R[np.ix_(obs, obs)], z[t, obs]
            P_pred = np.linalg.inv(J_pred)
            loglik += _gauss_logpdf(zt - Ct @ (P_pred @ y_pred),
                                    Ct @ P_pred @ Ct.T + Rt)
            J = J_pred + Ct.T @ np.linalg.solve(Rt, Ct)
            y = y_pred + Ct.T @ np.linalg.solve(Rt, zt)
        else:
            J, y = J_pred, y_pred
            n_skipped += 1
        P = np.linalg.inv(J)
        mean[t], cov[t], info[t], info_vector[t] = P @ y, P, J, y
    return FilterResult(mean, cov, float(loglik), info=info,
                        info_vector=info_vector, n_skipped=n_skipped)


def _rts(out: FilterResult, A, Q) -> tuple[np.ndarray, np.ndarray]:
    """Rauch–Tung–Striebel: condition step `t` on all of `z`, not just `z_{1..t}`.

    The gain `G_t = P_t Aᵀ (A P_t Aᵀ + Q)⁻¹` is the matrix the forward prediction
    already built, reused in the opposite direction — which is why the filter has
    to hand over its filtered covariances rather than just the means.
    """
    T, n = out.mean.shape
    sm = out.mean.copy()
    sc = out.cov.copy()
    for t in range(T - 2, -1, -1):
        P_pred = A @ out.cov[t] @ A.T + Q
        G = np.linalg.solve(P_pred, A @ out.cov[t]).T
        sm[t] = out.mean[t] + G @ (sm[t + 1] - A @ out.mean[t])
        sc[t] = out.cov[t] + G @ (sc[t + 1] - P_pred) @ G.T
        sc[t] = 0.5 * (sc[t] + sc[t].T)
    return sm, sc


# --------------------------------------------------------------------------- #
# Nonlinear variants
# --------------------------------------------------------------------------- #
def _ekf(z, f, F_fn, h, H_fn, Q, R, m0, P0) -> FilterResult:
    """Linearise at the mean: push `m` through `f`, and `P` through `F(m)`.

    One Taylor term, so the covariance is correct to first order and the mean is
    biased by `½ tr(F'' P)`. Invisible on a linear model, decisive on a curved one.
    """
    T = z.shape[0]
    mean = np.empty((T, m0.size))
    cov = np.empty((T, m0.size, m0.size))
    mu, P = m0.copy(), P0.copy()
    loglik = 0.0
    for t in range(T):
        F = F_fn(mu) if F_fn is not None else _fd_jacobian(f, mu)
        if F.shape != (mu.size, mu.size):
            raise ValueError(
                f"transition Jacobian must be {(mu.size, mu.size)}, got {F.shape}"
            )
        mu = f(mu)
        P = F @ P @ F.T + Q
        obs = ~np.isnan(z[t])
        if obs.any():
            C = H_fn(mu) if H_fn is not None else _fd_jacobian(h, mu)
            hz = h(mu)
            if C.shape != (hz.size, mu.size):
                raise ValueError(
                    f"observation Jacobian must be {(hz.size, mu.size)}, got {C.shape}"
                )
            Ct, Rt = C[obs], R[np.ix_(obs, obs)]
            hz = hz[obs]
            S = Ct @ P @ Ct.T + Rt
            innov = z[t, obs] - hz
            loglik += _gauss_logpdf(innov, S)
            K = np.linalg.solve(S, Ct @ P).T
            mu = mu + K @ innov
            P = P - K @ Ct @ P
            P = 0.5 * (P + P.T)
        mean[t], cov[t] = mu, P
    return FilterResult(mean, cov, float(loglik))


def _ukf(z, f, h, Q, R, m0, P0, alpha, beta, kappa) -> FilterResult:
    """Unscented transform: push `2n+1` deterministic points through `f` and `h`.

    The points are chosen so their weighted moments reproduce `(m, P)` exactly;
    after the nonlinearity they reproduce its first two moments to second order.
    No Jacobian, and no derivative of the model — only `2n+1` evaluations a step.
    """
    T = z.shape[0]
    n = m0.size
    lam = alpha**2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1.0 - alpha**2 + beta)
    mean = np.empty((T, n))
    cov = np.empty((T, n, n))
    mu, P = m0.copy(), P0.copy()
    loglik = 0.0
    for t in range(T):
        X = _sigma_points(mu, P, n + lam)
        Y = np.array([f(x) for x in X])
        mu = wm @ Y
        P = _weighted_cov(Y, Y, mu, mu, wc) + Q
        obs = ~np.isnan(z[t])
        if obs.any():
            # Fresh sigma points from the *predicted* moments: the points that
            # went through `f` describe the previous state, not this one, and
            # reusing them would condition on the wrong distribution.
            Xs = _sigma_points(mu, P, n + lam)
            Z = np.array([h(x) for x in Xs])
            zhat = wm @ Z
            Zt = Z[:, obs]
            S = _weighted_cov(Zt, Zt, zhat[obs], zhat[obs], wc) + R[np.ix_(obs, obs)]
            cross = _weighted_cov(Xs, Zt, mu, zhat[obs], wc)
            innov = z[t, obs] - zhat[obs]
            loglik += _gauss_logpdf(innov, S)
            K = np.linalg.solve(S, cross.T).T
            mu = mu + K @ innov
            P = P - K @ S @ K.T
            P = 0.5 * (P + P.T)
        mean[t], cov[t] = mu, P
    return FilterResult(mean, cov, float(loglik))


def _sigma_points(mu: np.ndarray, P: np.ndarray, scale: float) -> np.ndarray:
    """`mu` plus/minus the columns of `√((n+λ)P)`; their moments are `(mu, P)`.

    The square root is a Cholesky factor, so a non-positive-definite `P` fails
    loudly here instead of quietly producing complex sigma points.
    """
    L = np.linalg.cholesky(scale * P)
    return np.vstack([mu, mu + L.T, mu - L.T])


def _weighted_cov(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
    """`Σ_i w_i (a_i − ma)(b_i − mb)ᵀ` — the cross-covariance the gain needs.

    Computing the cross term directly, rather than reading it off a scatter
    matrix, is what lets the UKF update a state it never observed.
    """
    return (a - ma).T @ (w[:, None] * (b - mb))


def _fd_jacobian(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences, one column per input; the fallback when no `F` is given."""
    x = np.asarray(x, dtype=float)
    out = np.empty((np.size(fn(x)), x.size))
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps * max(1.0, abs(x[i]))
        out[:, i] = (fn(x + step) - fn(x - step)) / (2.0 * step[i])
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _gauss_logpdf(x: np.ndarray, cov: np.ndarray) -> float:
    """`log N(x; 0, cov)` by Cholesky: no determinant, no explicit inverse."""
    L = np.linalg.cholesky(cov)
    y = np.linalg.solve(L, x)
    return float(-0.5 * (x.size * LOG2PI + 2.0 * np.sum(np.log(np.diag(L))) + y @ y))


def _as_matrix(obj, n: int) -> np.ndarray:
    A = np.asarray(obj, dtype=float)
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"expected a matrix with {n} columns, got shape {A.shape}")
    return A


def _as_pair(obj):
    """Accept `f` or `(f, F)`; the second form skips finite differences."""
    if callable(obj):
        return obj, None
    f, F = obj
    return f, F


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _constant_velocity(dt: float) -> np.ndarray:
    return np.array([[1.0, dt], [0.0, 1.0]])


def _joint_cov(A, C, Q, R, P0, T: int) -> np.ndarray:
    """`Cov(z_1, …, z_T)` of the linear model, built forward from `P0`.

    This is the exact likelihood of the whole sequence, so the filter's sum of
    predictive log-densities has something to be checked against.
    """
    m = C.shape[0]
    states, P = [], P0.copy()
    for _ in range(T):
        P = A @ P @ A.T + Q
        states.append(P)
    S = np.zeros((T * m, T * m))
    for i in range(T):
        for j in range(i + 1):
            Pij = states[j]
            for _ in range(i - j):
                Pij = A @ Pij
            block = C @ Pij @ C.T
            S[i * m:(i + 1) * m, j * m:(j + 1) * m] = block
            S[j * m:(j + 1) * m, i * m:(i + 1) * m] = block.T
    S[np.diag_indices_from(S)] += R[0, 0] - 0.0
    return S


def main() -> None:
    rng = np.random.default_rng(0)

    # --- a linear model: five variants, one answer --------------------------
    dt, T = 1.0, 60
    A = _constant_velocity(dt)
    C = np.array([[1.0, 0.0]])
    Q = 0.05 * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[0.8]])
    m0, P0 = np.array([0.0, 1.0]), np.eye(2)

    truth = np.zeros((T, 2))
    truth[0] = [0.0, 1.0]
    z = np.empty((T, 1))
    for t in range(T):
        if t:
            truth[t] = A @ truth[t - 1] + rng.multivariate_normal(np.zeros(2), Q)
        z[t] = C @ truth[t] + rng.normal(0.0, math.sqrt(R[0, 0]))

    kw = dict(transition=A, observation=C, Q=Q, R=R, m0=m0, P0=P0)
    kf = kalman_filter(z, "kf", **kw)
    info = kalman_filter(z, "information_filter", **kw)
    rts = kalman_filter(z, "rts_smoother", **kw)
    f_lin, F_lin = (lambda s: A @ s), (lambda s: A)
    h_lin, H_lin = (lambda s: C @ s), (lambda s: C)
    ekf = kalman_filter(z, "ekf", transition=(f_lin, F_lin),
                        observation=(h_lin, H_lin), Q=Q, R=R, m0=m0, P0=P0)
    ukf = kalman_filter(z, "ukf", transition=f_lin, observation=h_lin,
                        Q=Q, R=R, m0=m0, P0=P0)

    # the precision form is the same posterior, not an approximation
    assert np.allclose(info.mean, kf.mean, atol=1e-9)
    assert np.allclose(info.cov, kf.cov, atol=1e-9)
    assert np.allclose(info.info[-1], np.linalg.inv(kf.cov[-1]), atol=1e-9)
    # on a linear model the linearisation is exact and the sigma points are
    # exact, so both nonlinear filters collapse onto the Kalman filter
    assert np.allclose(ekf.mean, kf.mean, atol=1e-9)
    assert np.allclose(ekf.cov, kf.cov, atol=1e-9)
    assert np.allclose(ukf.mean, kf.mean, atol=1e-9)
    assert np.allclose(ukf.cov, kf.cov, atol=1e-9)

    # the smoothed variance is never larger than the filtered variance, and it
    # equals it at the last step — there is no future left to condition on
    assert np.all(np.linalg.eigvalsh(rts.smooth_cov - kf.cov) <= 1e-9)
    assert np.allclose(rts.smooth_mean[-1], kf.mean[-1], atol=1e-12)
    assert np.allclose(rts.smooth_cov[-1], kf.cov[-1], atol=1e-12)

    rmse_f = float(np.sqrt(np.mean((kf.mean[:, 0] - truth[:, 0]) ** 2)))
    rmse_s = float(np.sqrt(np.mean((rts.smooth_mean[:, 0] - truth[:, 0]) ** 2)))
    assert rmse_s < rmse_f

    # the filter's log p(z) is the exact Gaussian log-likelihood of the sequence
    mz = np.empty((T, 1))
    mcur = m0
    for t in range(T):
        mcur = A @ mcur
        mz[t] = C @ mcur
    S_joint = _joint_cov(A, C, Q, R, P0, T)
    exact = _gauss_logpdf((z - mz).ravel(), S_joint)
    assert np.isclose(kf.loglik, exact, atol=1e-8), (kf.loglik, exact)
    print(
        f"kalman: kf — log p(z) = {kf.loglik:.4f} against the exact joint "
        f"Gaussian {exact:.4f}; information_filter, ekf and ukf all agree to 1e-9"
    )
    print(
        f"kalman: rts_smoother — position RMSE {rmse_f:.3f} filtered, "
        f"{rmse_s:.3f} smoothed; the variance shrinks at every step but the last"
    )

    # --- missing observations ----------------------------------------------
    zg = z.copy()
    gaps = np.array([5, 6, 25, 26, 27])
    zg[gaps] = np.nan
    miss = kalman_filter(zg, "missing_obs", **kw)
    assert miss.n_skipped == gaps.size
    # at a gap the posterior is the prediction, exactly
    for t in gaps:
        assert np.allclose(miss.mean[t], A @ miss.mean[t - 1], atol=1e-12)
        assert np.allclose(miss.cov[t], A @ miss.cov[t - 1] @ A.T + Q, atol=1e-12)
    # skipping observations costs information: the trace grows at every gap
    for t in gaps:
        assert np.trace(miss.cov[t]) > np.trace(kf.cov[t])
    # one coordinate of a two-dimensional observation: the update uses the other
    # one, so it is *not* a skipped step, and the result must equal the same
    # filter observing only that coordinate
    C2, R2 = np.eye(2), 0.3 * np.eye(2)
    z2 = np.column_stack([z[:, 0], np.full(T, np.nan)])
    both = kalman_filter(z2, "missing_obs", transition=A, observation=C2,
                         Q=Q, R=R2, m0=m0, P0=P0)
    one = kalman_filter(z, "kf", transition=A, observation=np.array([[1.0, 0.0]]),
                        Q=Q, R=np.array([[0.3]]), m0=m0, P0=P0)
    assert both.n_skipped == 0
    assert np.allclose(both.mean, one.mean, atol=1e-12)
    assert np.allclose(both.cov, one.cov, atol=1e-12)
    assert np.isclose(both.loglik, one.loglik, atol=1e-12)
    # the plain filter has no rule for a gap, and says so
    try:
        kalman_filter(zg, "kf", **kw)
    except ValueError as err:
        assert "missing_obs" in str(err)
    else:
        raise AssertionError("kf should refuse a NaN observation")
    print(
        f"kalman: missing_obs — {gaps.size} unobserved steps skipped, each cost the "
        f"prediction and nothing else; a half-missing observation still updates"
    )

    # --- a symmetric quadratic observation: where the ekf's likelihood lies ---
    # h(x) = x² has H = 2x, which is zero at the prior mean. The EKF concludes
    # the observation says nothing about x and leaves both moments untouched;
    # the UKF's sigma points sit at ±1 and see the curvature. The *mean* cannot
    # move in either case — the posterior is symmetric — so the difference shows
    # up in the number the filter is really for: log p(z).
    def h_sq(s):
        return np.array([s[0] ** 2])

    def H_sq(s):
        return np.array([[2.0 * s[0]]])

    one_step = dict(Q=np.zeros((1, 1)), R=np.array([[0.1]]),
                    m0=np.array([0.0]), P0=np.array([[1.0]]))
    z1 = np.array([[1.0]])
    ekf_sq = kalman_filter(z1, "ekf", transition=(lambda s: s, lambda s: np.array([[1.0]])),
                           observation=(h_sq, H_sq), **one_step)
    ukf_sq = kalman_filter(z1, "ukf", transition=lambda s: s, observation=h_sq,
                           **one_step)
    grid = np.linspace(-8.0, 8.0, 40001)
    prior = np.exp(-0.5 * grid**2) / math.sqrt(2.0 * math.pi)
    lik = np.exp(-0.5 * (1.0 - grid**2) ** 2 / 0.1) / math.sqrt(0.2 * math.pi)
    exact_ll = math.log(float(np.trapezoid(prior * lik, grid)))
    assert abs(ukf_sq.mean[0, 0]) < 1e-12 and abs(ekf_sq.mean[0, 0]) < 1e-12
    assert abs(ukf_sq.loglik - exact_ll) < 0.1, (ukf_sq.loglik, exact_ll)
    assert abs(ekf_sq.loglik - exact_ll) > 3.0, (ekf_sq.loglik, exact_ll)
    print(
        f"kalman: ekf vs ukf — one observation through x²: log p(z) is "
        f"{exact_ll:.3f} exactly, {ukf_sq.loglik:.3f} unscented, "
        f"{ekf_sq.loglik:.3f} linearised; the EKF's H = 0 makes it blind"
    )

    # --- many steps with curved dynamics ------------------------------------
    # Logistic growth observed linearly: both approximations track, and the
    # difference is now second order in the curvature of `f`, so the two agree.
    def f_gr(s):
        return s * np.exp(0.6 * (1.0 - s / 10.0))

    def F_gr(s):
        return np.array([[(1.0 - 0.6 * s[0] / 10.0) * np.exp(0.6 * (1.0 - s[0] / 10.0))]])

    Tn = 100
    xg = np.zeros(Tn)
    zn = np.empty((Tn, 1))
    xg[0] = 2.0
    for t in range(Tn):
        if t:
            xg[t] = f_gr(xg[t - 1]) + rng.normal(0.0, math.sqrt(0.1))
        zn[t] = xg[t] + rng.normal(0.0, math.sqrt(0.5))

    common = dict(Q=np.array([[0.1]]), R=np.array([[0.5]]),
                  m0=np.array([1.0]), P0=np.array([[1.0]]))
    ekf_gr = kalman_filter(zn, "ekf", transition=(f_gr, F_gr),
                           observation=(lambda s: s, lambda s: np.array([[1.0]])),
                           **common)
    ukf_gr = kalman_filter(zn, "ukf", transition=f_gr, observation=lambda s: s,
                           **common)
    # the same EKF with finite-difference Jacobians: 1e-6 steps, 1e-8 agreement
    ekf_fd = kalman_filter(zn, "ekf", transition=f_gr, observation=lambda s: s,
                           **common)
    assert np.allclose(ekf_fd.mean, ekf_gr.mean, atol=1e-5)
    assert np.allclose(ekf_fd.cov, ekf_gr.cov, atol=1e-5)
    rmse_ekf = float(np.sqrt(np.mean((ekf_gr.mean[:, 0] - xg) ** 2)))
    rmse_ukf = float(np.sqrt(np.mean((ukf_gr.mean[:, 0] - xg) ** 2)))
    assert rmse_ekf < 0.5 and rmse_ukf < 0.5
    assert np.max(np.abs(ekf_gr.mean - ukf_gr.mean)) < 0.1
    print(
        f"kalman: ekf vs ukf — {Tn} steps of logistic growth: RMSE {rmse_ekf:.3f} "
        f"linearised, {rmse_ukf:.3f} unscented, and finite-difference Jacobians "
        f"reproduce the analytic ones"
    )

    print("kalman demo passed")


if __name__ == "__main__":
    main()
