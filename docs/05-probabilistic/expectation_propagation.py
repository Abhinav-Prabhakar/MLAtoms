"""Expectation propagation — Gaussian sites in front of any likelihood.

Chapter:   05-probabilistic
Companion: expectation_propagation.md
Tier 3 · 3 variants: ep, damped_ep, ep_for_gp

    p(θ) ∝ N(θ; 0, Σ₀) Π_i f_i(a_iᵀθ)

One Gaussian prior, and one scalar likelihood per factor. EP replaces every
`f_i` by a Gaussian in the projection `a_iᵀθ`, chosen so the resulting Gaussian
posterior reproduces the *marginal* mean and variance of the tilted
distribution. Everything else is bookkeeping.

    p(θ) ∝ N(0, Σ₀) Π f_i(a_iᵀθ)   →   q(θ) = N(μ, Σ)
    Σ = (Σ₀⁻¹ + Σ_i τ_i a_i a_iᵀ)⁻¹,   μ = Σ Σ_i τ_i ν_i a_i

The two coordinate systems are the same code with different `a_i`:

    parameter space   a_i = x_i (a data row)        Bayesian logistic regression
    function space    a_i = e_i (a basis vector)    GP with a robust likelihood

The variants are about two things: the schedule, which is where EP actually
fails, and the factor, which decides whether a Gaussian site is even a
legitimate thing to ask for. `ep` takes full steps, `damped_ep` interpolates,
and `ep_for_gp` is the function-space use with a Student-t likelihood, where
the full step is fine but the tilt can be bimodal.

The demo checks EP against exact answers, never against another approximation:
a 2-D grid for the logistic posterior and its evidence, the closed-form GP
regression posterior when every factor is Gaussian (EP must be exact, and is,
to 2e−13), and a contaminated series where the Student-t fit must beat the
Gaussian one — and where the outlier's site precision goes *negative*, because
a bimodal tilt has no moment-matched Gaussian site.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["ep", "damped_ep", "ep_for_gp"])

LOG2PI = math.log(2.0 * math.pi)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

_PRESET = {
    "ep": {"damping": 1.0, "likelihood": "logistic"},
    "damped_ep": {"damping": 0.7, "likelihood": "logistic"},
    "ep_for_gp": {"damping": 1.0, "likelihood": "student_t"},
}


def sigmoid(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    pos = a >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class EP:
    """Expectation propagation over a Gaussian prior and scalar sites.

    `prior_cov` is `Σ₀`, `(n, n)`. `A` is `(n, d)` with one row per factor; the
    `i`-th likelihood sees `a_iᵀθ`. Omitting `A` puts each factor on its own
    coordinate, which is the function-space (GP) reading.
    """

    def __init__(self, prior_cov, A=None, *, variant: str = "ep",
                 likelihood: str | None = None, damping: float | None = None,
                 n_iter: int = 200, tol: float = 1e-9, noise: float = 1.0,
                 dof: float = 4.0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        preset = _PRESET[variant]
        self.variant = variant
        self.likelihood = likelihood or preset["likelihood"]
        if self.likelihood not in ("logistic", "probit", "student_t", "gaussian"):
            raise ValueError(f"unknown likelihood {self.likelihood!r}")
        self.damping = float(preset["damping"] if damping is None else damping)
        self.prior_cov = np.atleast_2d(np.asarray(prior_cov, dtype=float))
        A = None if A is None else np.atleast_2d(np.asarray(A, dtype=float))
        self.n = self.prior_cov.shape[0] if A is None else A.shape[0]
        if self.prior_cov.shape != (A.shape[1] if A is not None else self.n,) * 2:
            raise ValueError("prior_cov must be d×d, with A of shape (n, d)")
        self.A = A
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.noise = float(noise)
        self.dof = float(dof)

    # -- the tilted distribution -------------------------------------------- #
    def _tilted(self, m: float, s2: float, y: float):
        """Mean, variance and normaliser of `N(η; m, s2)·f(y|η)`."""
        if self.likelihood == "probit":
            z = y * m / math.sqrt(1.0 + s2)
            lz = math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))
            r = math.exp(-0.5 * z * z - 0.5 * LOG2PI - lz)
            mean = m + y * s2 * r / math.sqrt(1.0 + s2)
            var = s2 - s2 * s2 * r * (z + r) / (1.0 + s2)
            return mean, var, lz
        if self.likelihood == "gaussian":
            # the one case with a closed form in every direction, and the
            # reason EP must reproduce GP regression exactly
            prec = 1.0 / s2 + 1.0 / self.noise**2
            mean = (m / s2 + y / self.noise**2) / prec
            lz = (-0.5 * LOG2PI - 0.5 * math.log(s2 + self.noise**2)
                  - 0.5 * (y - m) ** 2 / (s2 + self.noise**2))
            return mean, 1.0 / prec, lz

        x, w = np.polynomial.hermite.hermgauss(40)
        eta = m + math.sqrt(2.0 * s2) * x
        if self.likelihood == "logistic":
            lik = sigmoid(y * eta)
        else:
            # Student-t on the residual: p(y|η) ∝ (1 + z²/ν)^(−(ν+1)/2)
            z = (y - eta) / self.noise
            lik = (1.0 + z * z / self.dof) ** (-0.5 * (self.dof + 1.0))
        den = float((w * lik).sum()) * INV_SQRT_PI
        mean = float((w * lik * eta).sum()) * INV_SQRT_PI / den
        var = float((w * lik * (eta - mean) ** 2).sum()) * INV_SQRT_PI / den
        return mean, var, math.log(den)

    # -- fitting -------------------------------------------------------------- #
    def fit(self, y) -> EP:
        y = np.asarray(y, dtype=float)
        if y.shape != (self.n,):
            raise ValueError(f"expected {self.n} observations, got {y.shape}")
        A = np.eye(self.n) if self.A is None else self.A
        d = A.shape[1]

        logdet0 = float(np.linalg.slogdet(self.prior_cov)[1])

        tau = np.zeros(self.n)
        nu = np.zeros(self.n)
        Sigma = self.prior_cov.copy()
        b = np.zeros(d)
        self.history_ = []

        for _it in range(1, self.n_iter + 1):
            delta = 0.0
            for i in range(self.n):
                a = A[i]
                s = float(a @ Sigma @ a)                    # Var(aᵀθ) under q
                denom = 1.0 - tau[i] * s
                if denom <= 1e-12:                          # cavity would be improper
                    delta = max(delta, 1.0)
                    continue
                s2 = s / denom
                m = s2 * (float(a @ (Sigma @ b)) / s - tau[i] * nu[i])
                m_t, s2_t, _ = self._tilted(m, s2, y[i])
                tau_new = 1.0 / s2_t - 1.0 / s2
                eta_new = m_t / s2_t - m / s2
                nu_new = eta_new / tau_new

                d_ = self.damping
                tau_i = (1.0 - d_) * tau[i] + d_ * tau_new
                nu_i = (1.0 - d_) * nu[i] + d_ * nu_new
                delta = max(delta, abs(tau_i - tau[i]), abs(nu_i - nu[i]))

                # rank-one update of Σ in O(d²), Sherman–Morrison with a
                # negative semi-definite step; b = Σ τ_iν_i a_i tracks the mean
                dt = tau_i - tau[i]
                scale = 1.0 + dt * s
                if abs(dt) > 0.0 and scale > 1e-12:
                    Sig_a = Sigma @ a
                    Sigma = Sigma - (dt / scale) * np.outer(Sig_a, Sig_a)
                b = b + (tau_i * nu_i - tau[i] * nu[i]) * a
                tau[i], nu[i] = tau_i, nu_i
            self.history_.append(delta)
            if delta < self.tol:
                break

        self.n_iter_ = _it
        self.converged_ = bool(self.history_[-1] < self.tol)
        self.tau_, self.nu_ = tau, nu
        self.Sigma_ = Sigma
        self.mu_ = Sigma @ b

        # Evidence, (18). The Gaussian integral over θ contributes `|Σ|`, and
        # the `−½Στ_iν_i²` term is the part of the site normalisation that the
        # product of Gaussians leaves behind; dropping it breaks the identity
        # even for a Gaussian likelihood.
        #
        # The derivation assumes every site is a Gaussian, which needs `τ_i > 0`.
        # A non-log-concave factor can break that -- a Student-t likelihood on a
        # large residual has a *bimodal* tilted distribution, and the Gaussian
        # that matches its first two moments can be wider than the cavity. Then
        # the posterior `Σ` still exists and is still the robust fit, but the
        # site is not a Gaussian and there is no evidence to report.
        self.logZ_valid_ = bool(np.all(tau > 0.0))
        if self.logZ_valid_:
            # |Σ| = |Σ₀⁻¹ + Σ τ a aᵀ|⁻¹, so the term below is +½log|Σ|; the sign
            # is the easiest thing in the file to get wrong
            logdet = float(np.linalg.slogdet(Sigma)[1])
            z_hat = 0.0
            for i in range(self.n):
                a = A[i]
                s = float(a @ Sigma @ a)
                denom = 1.0 - tau[i] * s
                s2 = s / denom if denom > 1e-12 else 1e-12
                m = s2 * (float(a @ (Sigma @ b)) / s - tau[i] * nu[i])
                lz = self._tilted(m, s2, y[i])[2]
                # amplitude: match the site's mass under the cavity to the
                # tilted mass, which moment matching alone leaves free
                v = s2 + 1.0 / tau[i]
                log_nu = -0.5 * LOG2PI - 0.5 * math.log(v) - 0.5 * (nu[i] - m) ** 2 / v
                z_hat += lz - log_nu
            # the `(2π)^{-d/2}` of the prior and the `(2π)^{d/2}` of the Gaussian
            # integral cancel, leaving one `−(n/2)log2π` for the n sites
            self.logZ_ = (z_hat - 0.5 * self.n * LOG2PI
                          + 0.5 * float(np.log(tau).sum())
                          + 0.5 * logdet - 0.5 * logdet0
                          + 0.5 * float(b @ self.mu_)
                          - 0.5 * float((tau * nu**2).sum()))
        else:
            self.logZ_ = float("nan")
        return self

    # -- prediction ------------------------------------------------------------ #
    def latent(self, A_new) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of `aᵀθ` under `q`, for rows `a` of `A_new`."""
        A_new = np.atleast_2d(np.asarray(A_new, dtype=float))
        mean = A_new @ self.mu_
        var = np.einsum("ij,jk,ik->i", A_new, self.Sigma_, A_new)
        return mean, np.maximum(var, 0.0)

    def predict_proba(self, A_new, *, integrate: bool = True) -> np.ndarray:
        mean, var = self.latent(A_new)
        if not integrate:
            return sigmoid(mean)
        x, w = np.polynomial.hermite.hermgauss(20)
        nodes = mean[:, None] + np.sqrt(2.0 * var)[:, None] * x
        return (w * sigmoid(nodes)).sum(axis=1) * INV_SQRT_PI


# --------------------------------------------------------------------------- #
# Exact references
# --------------------------------------------------------------------------- #
def _exact_logistic_grid(X, y, lo=-4.0, hi=4.0, n=241):
    """Exact posterior mean, covariance and `log Z` for two weights, on a grid."""
    g = np.linspace(lo, hi, n)
    W0, W1 = np.meshgrid(g, g, indexing="ij")
    W = np.stack([W0, W1], axis=-1)
    a = W @ X.T
    logp = -0.5 * np.sum(W**2, axis=-1) + np.sum(-np.logaddexp(0.0, -y * a), axis=-1)
    top = logp.max()
    w = np.exp(logp - top)
    cell = (g[1] - g[0]) ** 2
    mean = np.array([(w * W0).sum(), (w * W1).sum()]) / w.sum()
    dev = np.stack([W0 - mean[0], W1 - mean[1]], axis=-1)
    cov = np.einsum("ij,ijk,ijl->kl", w, dev, dev) / w.sum()
    return mean, cov, top + math.log(w.sum() * cell) - math.log(2.0 * math.pi)


def _rbf(A, B, ell=1.0, sf=1.0):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return sf**2 * np.exp(-0.5 * np.maximum(d, 0.0) / ell**2)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- Bayesian logistic regression, against an exact 2-D grid ------------
    # Prior N(0, I), so the grid integral gives the posterior and the evidence
    # with no approximation in it, and both EP numbers are checked against it.
    n, d = 40, 2
    X = rng.normal(size=(n, d))
    w_true = np.array([1.4, -1.1])
    y = np.where(rng.random(n) < sigmoid(X @ w_true), 1.0, -1.0)
    mean_g, cov_g, logz_g = _exact_logistic_grid(X, y)

    ep = EP(np.eye(d), X, variant="ep").fit(y)
    assert ep.converged_ and np.all(ep.tau_ > 0.0)
    assert np.max(np.abs(ep.mu_ - mean_g)) < 0.05
    assert np.max(np.abs(ep.Sigma_ - cov_g)) < 0.01
    print(
        f"expectation_propagation: ep — {n} points, 2 weights: posterior mean "
        f"{np.round(ep.mu_, 3)} against the grid's {np.round(mean_g, 3)}, log Z "
        f"{ep.logZ_:.4f} against {logz_g:.4f}, in {ep.n_iter_} sweeps"
    )

    # --- the same fit, one step at a time ----------------------------------
    # A sweep is not a gradient step, so there is no guarantee the error
    # decreases -- but here it does, geometrically, until it hits the fixed
    # point and stops moving at all.
    path = [EP(np.eye(d), X, variant="ep", n_iter=t).fit(y).mu_ for t in (1, 2, 3, 5)]
    dist = [float(np.linalg.norm(p - mean_g)) for p in path]
    assert dist[0] > dist[1] > dist[2] >= dist[3]
    print(
        f"expectation_propagation: ep — distance to the exact mean after 1, 2, 3, 5 "
        f"sweeps: {', '.join(f'{v:.3f}' for v in dist)}; it is a fixed-point "
        f"iteration, not a descent method"
    )

    # --- damping: the same fixed point, paid for in sweeps -----------------
    # A damped step is a shorter step along the same direction, so it changes
    # the path and not the destination. What it buys is robustness on the
    # problems where the full step oscillates; what it costs is measurable, and
    # on this stiff problem the cost is about 1/damping.
    stiff = np.array([[2.2, 1.9], [-1.9, -2.1], [2.4, -1.8], [-2.0, 1.7]])
    y_s = np.array([1.0, -1.0, 1.0, -1.0])
    sched = (1.0, 0.7, 0.5, 0.3)
    runs = [EP(np.eye(d), stiff, variant="ep", damping=g, n_iter=500).fit(y_s)
            for g in sched]
    assert all(r.converged_ for r in runs)
    # every schedule iterates the same map to a fixed point, so every one lands
    # on the same q -- sites, evidence and mean, not just the prediction
    assert max(float(np.max(np.abs(r.mu_ - runs[0].mu_))) for r in runs) < 1e-8
    assert max(float(np.max(np.abs(r.tau_ - runs[0].tau_))) for r in runs) < 1e-8
    assert max(abs(r.logZ_ - runs[0].logZ_) for r in runs) < 1e-8
    assert all(a.n_iter_ < b.n_iter_ for a, b in zip(runs[:-1], runs[1:], strict=True))
    print(
        f"expectation_propagation: damped_ep — on four nearly collinear points "
        f"damping 1.0, 0.7, 0.5, 0.3 all reach the same mean "
        f"{np.round(runs[0].mu_, 3)} and log Z {runs[0].logZ_:.4f}, in "
        f"{', '.join(str(r.n_iter_) for r in runs)} sweeps: a shorter step "
        f"changes the path, never the answer"
    )

    # --- every factor Gaussian: EP has to be exact ------------------------
    # This is the invariant that catches an error in the site algebra or in the
    # evidence constant, and it is cheap enough to run on every change.
    n_g = 8
    Xg = np.linspace(-2.0, 2.0, n_g)[:, None]
    sigma_n = 0.35
    f_clean = np.sin(1.5 * Xg[:, 0])
    yg = f_clean + sigma_n * rng.normal(size=n_g)
    K = _rbf(Xg, Xg) + 1e-10 * np.eye(n_g)
    exact = EP(K, variant="ep_for_gp", likelihood="gaussian",
               noise=sigma_n).fit(yg)
    A_exact = K + sigma_n**2 * np.eye(n_g)
    mean_exact = K @ np.linalg.solve(A_exact, yg)
    cov_exact = K - K @ np.linalg.solve(A_exact, K)
    logz_exact = (-0.5 * n_g * LOG2PI - 0.5 * float(np.linalg.slogdet(A_exact)[1])
                  - 0.5 * float(yg @ np.linalg.solve(A_exact, yg)))
    # a Gaussian likelihood has an exact Gaussian site, so EP lands on it in
    # one update per site and never moves again
    assert np.allclose(exact.tau_, 1.0 / sigma_n**2, atol=1e-10)
    assert np.max(np.abs(exact.mu_ - mean_exact)) < 1e-9
    assert np.max(np.abs(exact.Sigma_ - cov_exact)) < 1e-9
    assert abs(exact.logZ_ - logz_exact) < 1e-9
    print(
        f"expectation_propagation: ep_for_gp — a Gaussian likelihood makes the fit "
        f"exact GP regression: mean, covariance and log Z agree to "
        f"{max(np.max(np.abs(exact.Sigma_ - cov_exact)), abs(exact.logZ_ - logz_exact)):.0e}"
    )

    # --- Student-t likelihood: one contaminated point, two posteriors ------
    # A Student-t likelihood bounds each point's influence, so a single outlier
    # cannot drag the fit the way it drags a Gaussian one. It also shows the
    # edge of the method: the tilted distribution at the outlier is bimodal, so
    # its moment-matched site is wider than the cavity and `τ` goes negative.
    y_out = yg.copy()
    y_out[5] += 2.6
    robust = EP(K, variant="ep_for_gp", noise=0.3, dof=4.0).fit(y_out)
    gauss = EP(K, variant="ep_for_gp", likelihood="gaussian", noise=sigma_n).fit(y_out)
    Xs = np.linspace(-2.0, 2.0, 41)[:, None]
    ks = _rbf(Xs, Xg)
    Ki = np.linalg.inv(K)

    def predict(model):
        m = ks @ (Ki @ model.mu_)
        v = (1.0 - np.einsum("ij,jk,ik->i", ks, Ki, ks)
             + np.einsum("ij,jk,ik->i", ks @ Ki, model.Sigma_, ks @ Ki))
        return m, np.maximum(v, 1e-12)

    truth = np.sin(1.5 * Xs[:, 0])
    m_r, _ = predict(robust)
    m_g, _ = predict(gauss)
    err_r = float(np.sqrt(np.mean((m_r - truth) ** 2)))
    err_g = float(np.sqrt(np.mean((m_g - truth) ** 2)))
    assert err_r < err_g, (err_r, err_g)
    # the outlier is the only site that leaves the Gaussian regime
    assert robust.tau_[5] < 0.0
    assert np.all(np.delete(robust.tau_, 5) > 0.0)
    assert robust.logZ_valid_ is False and gauss.logZ_valid_ is True
    assert np.isfinite(robust.mu_).all() and not np.isfinite(robust.logZ_)
    print(
        f"expectation_propagation: ep_for_gp — one outlier of +2.6 moves the "
        f"Gaussian fit to RMSE {err_g:.3f} against {err_r:.3f} for the Student-t; "
        f"its site precision at that point is {robust.tau_[5]:+.2f}, the only "
        f"negative one, so the tilted distribution there is bimodal and no "
        f"evidence is reported (posterior only)"
    )

    print("expectation_propagation demo passed")


if __name__ == "__main__":
    main()
