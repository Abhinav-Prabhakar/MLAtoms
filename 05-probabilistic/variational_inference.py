"""Variational inference — turn integration into optimisation.

Chapter:   05-probabilistic
Companion: variational_inference.md
Tier 2 · 5 variants: mean_field, cavi, reparameterization, structured_vi,
black_box_vi

    ELBO(q) = E_q[log p(z, x)] − E_q[log q(z)]  ≤  log p(x)

You cannot integrate the posterior, but you can maximise a lower bound on its
evidence over a family you *can* sample. The gap is exactly KL(q ‖ p(·|x)), so a
step that raises the ELBO moves q towards the posterior, and the bound is tight
only when q *is* the posterior. That single sentence is the whole method; the
variants differ in the family and in how the expectation is differentiated.

    cavi                closed-form coordinate ascent, conjugate model
    mean_field          diagonal Gaussian, exact coordinate updates
    reparameterization  diagonal Gaussian, pathwise gradients
    structured_vi       full-covariance Gaussian, pathwise gradients
    black_box_vi        diagonal Gaussian, score-function gradients

`mean_field` is the shortcut available when the target is Gaussian: the optimal
diagonal q has variances `1 / P_jj` and means solve one linear fixed point.
`reparameterization` and `black_box_vi` optimise the same family when the target
is a black box, and differ only in whether a gradient is available.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

VARIANTS = frozenset(
    ["mean_field", "cavi", "reparameterization", "structured_vi", "black_box_vi"]
)

LOG2PI = math.log(2.0 * math.pi)

LogProb = Callable[[np.ndarray], float]
Grad = Callable[[np.ndarray], np.ndarray]


def _digamma(x: float) -> float:
    """Digamma by recurrence and asymptotic series; no scipy dependency."""
    out = 0.0
    while x < 6.0:
        out -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return out + math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0)
    )


def _pack(m: np.ndarray, log_s: np.ndarray | None,
          L: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Parameters → `(mean, covariance)`, the two things every caller wants.

    Diagonal families carry `log σ` so the step stays unconstrained; the
    structured family carries a Cholesky factor, which is the only triangular
    parameterisation that makes `L Lᵀ` always positive definite.
    """
    if L is not None:
        return m, L @ L.T
    return m, np.diag(np.exp(2.0 * log_s))


class GaussianTarget:
    """A Gaussian log-density in information form.

    `log p(z) = −½ zᵀ P z + bᵀ z`, left unnormalised on purpose: the ELBO differs
    from the log evidence by a constant, so whoever cares about the bound adds
    `½ bᵀ P⁻¹ b + ½ (d log 2π − log det P)` back.

    Parameters
    ----------
    precision : (d, d) array
        Symmetric positive definite `P`.
    linear : (d,) array, optional
        `b`; zero when omitted.
    """

    def __init__(self, precision: np.ndarray, linear: np.ndarray | None = None):
        P = np.atleast_2d(np.asarray(precision, dtype=float))
        if P.shape[0] != P.shape[1]:
            raise ValueError("precision must be square")
        if not np.allclose(P, P.T):
            raise ValueError("precision must be symmetric")
        if np.any(np.linalg.eigvalsh(P) <= 0.0):
            raise ValueError("precision must be positive definite")
        self.precision = P
        self.linear = np.zeros(P.shape[0]) if linear is None else np.asarray(linear, dtype=float)
        if self.linear.shape != (P.shape[0],):
            raise ValueError("linear must have one entry per dimension")

    @property
    def dim(self) -> int:
        return self.precision.shape[0]

    def log_prob(self, z: np.ndarray) -> float:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return float(-0.5 * z @ self.precision @ z + self.linear @ z)

    def grad(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return self.linear - self.precision @ z

    def log_evidence(self) -> float:
        """`log ∫ exp(log_prob)`; the ceiling every ELBO in this file must respect."""
        solve = np.linalg.solve(self.precision, self.linear)
        sign, logdet = np.linalg.slogdet(self.precision)
        d = self.dim
        return float(0.5 * self.linear @ solve + 0.5 * (d * LOG2PI - logdet))


class _Adam:
    """Adam on a list of arrays, in place. Six lines of state per parameter."""

    def __init__(self, params: list[np.ndarray], lr: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p += self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class VariationalInference:
    """Fit a Gaussian `q` to a posterior by maximising the ELBO.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    n_iter : int
        Sweeps for `cavi`/`mean_field`, gradient steps for the rest.
    lr : float
        Adam step size for the gradient variants. Step sizes are in the units of
        the *parameters* (`m`, `log σ`, `L`), not of the ELBO, so a target whose
        curvature is O(100) needs a smaller one.
    n_samples : int
        Draws per gradient step. The pathwise estimator is unbiased at 1; the
        score-function estimator needs more, and a baseline, to be usable.
    seed : int
        Every draw in the file comes from `default_rng(seed)`.
    mu0, kappa0, a0, b0 : float
        Normal-Gamma prior for `cavi`: `μ | τ ~ N(mu0, (kappa0 τ)⁻¹)` and
        `τ ~ Gamma(a0, b0)` — shape and rate, as everywhere in this repo.
    tol : float
        Convergence threshold on the parameter update for the deterministic
        variants.

    Notes
    -----
    `fit` takes different targets per variant, because the variants genuinely
    consume different information:

    - `cavi` takes the 1-D array of observations of a Normal-Gamma model.
    - `mean_field` takes a `GaussianTarget`; it uses the closed-form updates.
    - the other three take a log-density `LogProb` and, except for
      `black_box_vi`, its gradient `Grad`. They also need `mean0`, whose length
      is the only way an arbitrary callable reveals its dimension.

    Fitted attributes: `mean_` and `std_` for the Gaussian families, `cov_` as
    well for `structured_vi`, `mu_`, `mu_var_`, `a_`, `b_` for `cavi`, and
    `elbo_` / `elbo_history_` / `n_iter_` for everything.
    """

    def __init__(self, variant: str = "mean_field", *, n_iter: int = 500,
                 lr: float = 0.05, n_samples: int = 1, seed: int = 0,
                 mu0: float = 0.0, kappa0: float = 1.0, a0: float = 1.0,
                 b0: float = 1.0, tol: float = 1e-10):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        self.variant = variant
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.mu0, self.kappa0 = float(mu0), float(kappa0)
        self.a0, self.b0 = float(a0), float(b0)
        self.tol = float(tol)

    # ------------------------------------------------------------------ #
    def fit(self, target: object, grad: Grad | None = None, *,
            mean0: np.ndarray | None = None) -> VariationalInference:
        """Run the variant. See the class docstring for what `target` must be."""
        if self.variant == "cavi":
            return self._fit_cavi(np.asarray(target, dtype=float))
        if self.variant == "mean_field":
            return self._fit_mean_field(target)
        if self.variant == "black_box_vi":
            return self._fit_gradient(target, None, mean0)
        if grad is None:
            raise ValueError(
                f"variant {self.variant!r} needs grad=...; use 'black_box_vi' when only "
                "the log density is available"
            )
        return self._fit_gradient(target, grad, mean0)

    # ------------------------------------------------------------------ #
    def _fit_mean_field(self, target: object) -> VariationalInference:
        """Coordinate ascent in the diagonal family, in closed form.

        For `log p(z) = −½ zᵀPz + bᵀz` the optimal factor is
        `q_j = N((b_j − Σ_{k≠j} P_jk m_k) / P_jj, 1/P_jj)`: the variance is fixed
        by the diagonal of `P`, and only the means move.
        """
        if not isinstance(target, GaussianTarget):
            raise TypeError(
                "mean_field uses the closed-form Gaussian updates; pass a GaussianTarget "
                "or use a gradient variant"
            )
        P, b = target.precision, target.linear
        lam = np.diag(P).copy()
        m = np.zeros(b.size)
        self.elbo_history_ = []
        n_done = 0
        for _ in range(self.n_iter):
            n_done += 1
            previous = m.copy()
            for j in range(b.size):
                off = P[j] @ m - P[j, j] * m[j]
                m[j] = (b[j] - off) / lam[j]
            self.elbo_history_.append(self._elbo_exact(target, m, 1.0 / lam))
            if np.max(np.abs(m - previous)) < self.tol:
                break
        self.mean_ = m
        self.var_ = 1.0 / lam
        self.std_ = np.sqrt(self.var_)
        self.cov_ = np.diag(self.var_)
        self.elbo_ = self.elbo_history_[-1]
        self.n_iter_ = n_done
        return self

    @staticmethod
    def _elbo_exact(target: GaussianTarget, m: np.ndarray, var: np.ndarray) -> float:
        """`E_q[log p] + H(q)` in closed form for a diagonal q."""
        P = target.precision
        e_logp = -0.5 * (m @ P @ m + float(np.sum(np.diag(P) * var))) + target.linear @ m
        entropy = 0.5 * (var.size * (LOG2PI + 1.0) - float(np.sum(np.log(var))))
        return float(e_logp + entropy)

    # ------------------------------------------------------------------ #
    def _fit_gradient(self, log_prob: object, grad: Grad | None,
                      mean0: np.ndarray | None) -> VariationalInference:
        if mean0 is None:
            raise ValueError("gradient variants need mean0=...; its length sets the dimension")
        m = np.asarray(mean0, dtype=float).copy()
        d = m.size
        rng = np.random.default_rng(self.seed)
        structured = self.variant == "structured_vi"
        L = np.eye(d) if structured else None
        log_s = None if structured else np.zeros(d)
        params = [m, L] if structured else [m, log_s]
        opt = _Adam(params, self.lr)
        baseline = None
        self.elbo_history_ = []
        record = max(1, self.n_iter // 100)
        n_done = 0
        for it in range(self.n_iter):
            n_done += 1
            if self.variant == "black_box_vi":
                grads, baseline = self._score_grads(log_prob, m, log_s, rng, baseline)
            elif structured:
                grads = self._pathwise_grads(grad, m, rng, L=L)
            else:
                grads = self._pathwise_grads(grad, m, rng, log_s=log_s)
            opt.step(grads)
            if it % record == 0:
                mean, cov = _pack(m, log_s, L)
                self.elbo_history_.append(self._elbo_at(log_prob, mean, cov, 256, rng))
        self.n_iter_ = n_done
        self.mean_, self.cov_ = _pack(m, log_s, L)
        self.var_ = np.diag(self.cov_).copy()
        self.std_ = np.sqrt(self.var_)
        self.elbo_ = self._elbo_at(log_prob, self.mean_, self.cov_, 1024, rng)
        self.elbo_history_.append(self.elbo_)
        return self

    def _pathwise_grads(self, grad: Grad, m: np.ndarray, rng: np.random.Generator,
                        *, log_s: np.ndarray | None = None,
                        L: np.ndarray | None = None) -> list[np.ndarray]:
        """`∇_φ E_q[log p] + ∇_φ H(q)` with `z = m + L ε` and no score term.

        The gradient of the sample is the gradient of the parameter, so the
        estimator has the variance of `log p`, not of `log p` times a score.
        """
        eps = rng.standard_normal((self.n_samples, m.size))
        if L is not None:
            z = m + eps @ L.T
            g = np.array([grad(row) for row in z])
            g_m = g.mean(axis=0)
            g_L = np.tril((g.T @ eps) / self.n_samples)
            g_L[np.diag_indices_from(g_L)] += 1.0 / np.diag(L)
            return [g_m, g_L]
        s = np.exp(log_s)
        z = m + s * eps
        g = np.array([grad(row) for row in z])
        g_m = g.mean(axis=0)
        g_ls = (g * (s * eps)).mean(axis=0) + 1.0
        return [g_m, g_ls]

    def _score_grads(self, log_prob: LogProb, m: np.ndarray, log_s: np.ndarray,
                     rng: np.random.Generator,
                     baseline: float | None) -> tuple[list[np.ndarray], float]:
        """`E[(log p(z) − b) ∇ log q(z)] + ∇H`, the estimator that needs no grad.

        Subtracting a baseline leaves the expectation unchanged because
        `E[∇ log q] = 0`, and cuts the variance by roughly the squared
        correlation between `log p` and its own score.
        """
        s = np.exp(log_s)
        eps = rng.standard_normal((self.n_samples, m.size))
        z = m + s * eps
        lp = np.array([log_prob(row) for row in z])
        level = float(lp.mean()) if baseline is None else baseline
        adv = lp - level
        g_m = (adv[:, None] * (z - m) / s**2).mean(axis=0)
        g_ls = (adv[:, None] * (eps**2 - 1.0)).mean(axis=0) + 1.0
        return [g_m, g_ls], 0.9 * level + 0.1 * float(lp.mean())

    # ------------------------------------------------------------------ #
    def elbo(self, log_prob: LogProb, *, n_samples: int = 256,
             rng: np.random.Generator | None = None) -> float:
        """Monte Carlo estimate of `E_q[log p] − E_q[log q]`.

        `log q` is analytic, so only `log p` is sampled. The entropy half of the
        estimator is exact even at one draw, which is why a single-sample
        estimate is noisy but never biased.
        """
        rng = np.random.default_rng(self.seed + 1) if rng is None else rng
        return self._elbo_at(log_prob, self.mean_, self.cov_, n_samples, rng)

    @staticmethod
    def _elbo_at(log_prob: LogProb, mean: np.ndarray, cov: np.ndarray,
                 n_samples: int, rng: np.random.Generator) -> float:
        d = mean.size
        L = np.linalg.cholesky(cov)
        eps = rng.standard_normal((n_samples, d))
        z = mean + eps @ L.T
        lp = np.array([log_prob(row) for row in z])
        solve = np.linalg.solve(L, (z - mean).T).T
        log_q = -0.5 * (d * LOG2PI + 2.0 * np.sum(np.log(np.diag(L))) + np.sum(solve**2, 1))
        return float(np.mean(lp - log_q))

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        """`n` draws from the fitted `q`, shape `(n, d)`.

        For `cavi` the draw is over the two blocks of the Normal-Gamma model and
        returns `(n, 2)` columns `(μ, τ)`.
        """
        rng = np.random.default_rng(self.seed + 2) if rng is None else rng
        if self.variant == "cavi":
            tau = rng.gamma(self.a_, 1.0 / self.b_, size=n)
            mu = rng.normal(self.mu_, np.sqrt(self.mu_var_), size=n)
            return np.column_stack([mu, tau])
        return self.mean_ + rng.standard_normal((n, self.mean_.size)) @ np.linalg.cholesky(
            self.cov_
        ).T

    # ------------------------------------------------------------------ #
    def _fit_cavi(self, x: np.ndarray) -> VariationalInference:
        """Mean-field coordinate ascent for the Normal-Gamma model.

        `x ~ N(μ, τ⁻¹)`, `μ | τ ~ N(mu0, (kappa0 τ)⁻¹)`, `τ ~ Gamma(a0, b0)`.
        The two conditionals are Gaussian and Gamma, so the updates are exact
        expectations and the ELBO is monotone by construction.
        """
        x = np.ravel(x)
        if x.size == 0:
            raise ValueError("cavi needs at least one observation")
        n = x.size
        xbar = float(x.mean())
        mu0, kappa0, a0, b0 = self.mu0, self.kappa0, self.a0, self.b0
        a = a0 + 0.5 * (n + 1.0)
        mu = (kappa0 * mu0 + n * xbar) / (kappa0 + n)
        var_mu = 1.0 / (kappa0 + n)
        b = b0 + 0.5 * float(np.sum((x - xbar) ** 2))
        self.elbo_history_ = []
        n_done = 0
        for _ in range(self.n_iter):
            n_done += 1
            previous = b
            var_mu = 1.0 / ((kappa0 + n) * (a / b))
            b = (
                b0
                + 0.5 * kappa0 * (var_mu + (mu - mu0) ** 2)
                + 0.5 * (float(np.sum((x - mu) ** 2)) + n * var_mu)
            )
            self.elbo_history_.append(self._cavi_elbo(x, mu, var_mu, a, b))
            # Stop when the *parameter* stops moving, not when the ELBO does:
            # near a fixed point the bound is flat (its gap is quadratic in the
            # error), so an ELBO threshold retires the loop while b is still
            # wrong in the fifth digit. This fixed point contracts by 1/(2a).
            if abs(b - previous) < self.tol * max(1.0, abs(b)):
                break
        self.mu_, self.mu_var_, self.a_, self.b_ = mu, var_mu, a, b
        self.mean_ = np.array([mu, a / b])
        self.std_ = np.array([math.sqrt(var_mu), math.sqrt(a) / b])
        self.elbo_ = self.elbo_history_[-1]
        self.n_iter_ = n_done
        return self

    def _cavi_elbo(self, x: np.ndarray, mu: float, var_mu: float, a: float,
                   b: float) -> float:
        n = x.size
        e_log_tau = _digamma(a) - math.log(b)
        e_tau = a / b
        e_log_p = (
            (self.a0 - 1.0) * e_log_tau
            - self.b0 * e_tau
            + self.a0 * math.log(self.b0)
            - math.lgamma(self.a0)
            + 0.5 * math.log(self.kappa0)
            + 0.5 * e_log_tau
            - 0.5 * LOG2PI
            + 0.5 * n * e_log_tau
            - 0.5 * n * LOG2PI
            - 0.5 * self.kappa0 * e_tau * (var_mu + (mu - self.mu0) ** 2)
            - 0.5 * e_tau * (float(np.sum((x - mu) ** 2)) + n * var_mu)
        )
        e_log_q_mu = -0.5 * (LOG2PI + 1.0 + math.log(var_mu))
        e_log_q_tau = (a - 1.0) * _digamma(a) + math.log(b) - a - math.lgamma(a)
        return float(e_log_p - e_log_q_mu - e_log_q_tau)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _sigmoid(a: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def _logistic_posterior(X: np.ndarray, y: np.ndarray,
                        prior_sd: float) -> tuple[LogProb, Grad]:
    """`log p(w | X, y)` for a Gaussian prior, and its gradient."""
    precision = 1.0 / prior_sd**2

    def log_prob(w: np.ndarray) -> float:
        a = X @ w
        return float(np.sum(y * a - np.logaddexp(0.0, a)) - 0.5 * precision * (w @ w))

    def grad(w: np.ndarray) -> np.ndarray:
        return X.T @ (y - _sigmoid(X @ w)) - precision * w

    return log_prob, grad


def _normal_gamma_evidence(x: np.ndarray, mu0: float, kappa0: float, a0: float,
                           b0: float) -> float:
    """`log p(x)` under the Normal-Gamma prior, in closed form.

    The shape here is `a0 + n/2`, *not* the `a0 + (n+1)/2` that the variational
    factor picks up: marginalising `μ` out of the joint is not the same as
    averaging it out of `log p`, and the extra half-degree is exactly the
    mean-field distortion in `q(τ)`.
    """
    n = x.size
    kappa_n = kappa0 + n
    a_n = a0 + 0.5 * n
    b_n = b0 + 0.5 * (
        float(np.sum((x - x.mean()) ** 2)) + kappa0 * n / kappa_n * (x.mean() - mu0) ** 2
    )
    return float(
        -0.5 * n * LOG2PI
        + 0.5 * math.log(kappa0 / kappa_n)
        + a0 * math.log(b0)
        - a_n * math.log(b_n)
        + math.lgamma(a_n)
        - math.lgamma(a0)
    )


def main() -> None:
    rng = np.random.default_rng(0)

    # --- mean_field and structured_vi on a posterior with a known answer ---
    # A correlated Gaussian is the only target where every quantity in the
    # method can be checked against an exact one.
    rho = 0.8
    P = np.array([[1.0, rho], [rho, 1.0]])
    b = np.array([0.5, -0.25])
    target = GaussianTarget(P, b)
    mu = np.linalg.solve(P, b)
    cov = np.linalg.inv(P)
    log_z = target.log_evidence()

    mf = VariationalInference("mean_field").fit(target)
    sv = VariationalInference(
        "structured_vi", n_iter=1000, lr=0.06, n_samples=256, seed=1
    ).fit(target.log_prob, target.grad, mean0=np.zeros(target.dim))

    # the mean-field mean is exact; only the variances are wrong
    assert np.allclose(mf.mean_, mu, atol=1e-8)
    assert np.allclose(mf.var_, 1.0 / np.diag(P), atol=1e-12)
    assert np.all(mf.var_ < np.diag(cov))
    assert mf.elbo_ < log_z - 0.05
    # the full-covariance family contains the exact posterior, so it reaches it —
    # up to the noise of a stochastic gradient, which is what a Monte Carlo ELBO
    # costs and why the bound is stated to two decimals
    assert np.allclose(sv.mean_, mu, atol=0.05), (sv.mean_, mu)
    assert np.allclose(sv.cov_, cov, atol=0.12), (sv.cov_, cov)
    assert abs(sv.elbo_ - log_z) < 0.02, (sv.elbo_, log_z)
    mf_gap = log_z - mf.elbo_
    sv_gap = log_z - sv.elbo_
    assert mf_gap > 100.0 * max(sv_gap, 1e-3)
    print(
        f"variational_inference: mean_field — means exact, variances "
        f"{np.round(mf.var_, 3).tolist()} vs exact {np.round(np.diag(cov), 3).tolist()}, "
        f"ELBO {mf.elbo_:.3f} below the evidence {log_z:.3f}: the bound is loose"
    )
    print(
        f"variational_inference: structured_vi — covariance recovered to "
        f"{np.abs(sv.cov_ - cov).max():.3f}, ELBO {sv.elbo_:.3f} vs evidence "
        f"{log_z:.3f} in {sv.n_iter_} steps: the family, not the optimiser, was the gap"
    )

    # --- reparameterization and black_box_vi on a target with no algebra ---
    # Bayesian logistic regression: the posterior is not Gaussian and not
    # conjugate, which is exactly when the ELBO has to be differentiated.
    n, prior_sd = 200, 2.0
    X = rng.standard_normal((n, 2))
    y = (X @ np.array([-1.5, 2.0]) + 0.6 * rng.standard_normal(n) > 0).astype(float)
    log_prob, grad = _logistic_posterior(X, y, prior_sd)

    w = np.zeros(2)
    for _ in range(50):  # Newton ascent, for a mode to compare the mean against
        p = _sigmoid(X @ w)
        H = X.T @ (X * (p * (1.0 - p))[:, None]) + np.eye(2) / prior_sd**2
        w = w + np.linalg.solve(H, X.T @ (y - p) - w / prior_sd**2)

    rp = VariationalInference(
        "reparameterization", n_iter=2000, lr=0.06, n_samples=8, seed=2
    ).fit(log_prob, grad, mean0=np.zeros(2))
    bb = VariationalInference(
        "black_box_vi", n_iter=6000, lr=0.01, n_samples=24, seed=3
    ).fit(log_prob, mean0=np.zeros(2))

    acc_rp = float(np.mean((_sigmoid(X @ rp.mean_) > 0.5) == y))
    acc_bb = float(np.mean((_sigmoid(X @ bb.mean_) > 0.5) == y))
    # the Laplace approximation at the mode: the curvature-based Gaussian,
    # which is full-covariance and therefore a lower bound on how wide a
    # diagonal q could be
    lap_sd = np.sqrt(np.diag(np.linalg.inv(H)))
    lap_corr = np.linalg.inv(H)[0, 1] / np.prod(lap_sd)
    assert acc_rp > 0.85 and acc_bb > 0.85, (acc_rp, acc_bb)
    assert np.max(np.abs(rp.mean_ - w)) < 0.2, (rp.mean_, w)
    assert np.max(np.abs(bb.mean_ - rp.mean_)) < 0.4, (bb.mean_, rp.mean_)
    # the theorem the mean-field demo showed in two dimensions: a diagonal q is
    # narrower than the posterior in every coordinate
    assert np.all(rp.std_ < lap_sd), (rp.std_, lap_sd)
    print(
        f"variational_inference: reparameterization — logistic posterior, mean "
        f"{np.round(rp.mean_, 3).tolist()} against the mode {np.round(w, 3).tolist()}, "
        f"accuracy {acc_rp:.3f}"
    )
    print(
        f"variational_inference: black_box_vi — score-function mean "
        f"{np.round(bb.mean_, 3).tolist()}, accuracy {acc_bb:.3f}, ELBO {bb.elbo_:.3f}"
    )
    print(
        f"variational_inference: reparameterization — the posterior is correlated "
        f"{lap_corr:.2f}, so the diagonal q is narrower than Laplace in both "
        f"coordinates, {np.round(rp.std_, 3).tolist()} vs {np.round(lap_sd, 3).tolist()}"
    )

    # the pathwise estimator is the reason the gradient methods are usable:
    # at the same q, its variance is a fraction of the score-function's
    probe = np.random.default_rng(11)
    eps = probe.standard_normal((4000, 2))
    z = rp.mean_ + rp.std_ * eps
    g_path = np.array([grad(row) for row in z])
    lp = np.array([log_prob(row) for row in z])
    g_score = lp[:, None] * (z - rp.mean_) / rp.std_**2
    reduction = g_score.var(axis=0) / g_path.var(axis=0)
    assert np.all(reduction > 3.0), reduction
    print(
        f"variational_inference: reparameterization — gradient variance at the same q "
        f"falls {np.round(reduction, 1).tolist()}× versus the score function"
    )

    # --- cavi against the exact Normal-Gamma posterior ---
    x = rng.normal(1.5, 2.0, size=40)
    vi = VariationalInference("cavi", mu0=0.0, kappa0=0.5, a0=2.0, b0=1.0).fit(x)
    n = x.size
    kappa_n = 0.5 + n
    a_q = 2.0 + 0.5 * (n + 1.0)      # the variational shape
    a_ev = 2.0 + 0.5 * n             # the exact posterior shape
    b_n = 1.0 + 0.5 * (
        float(np.sum((x - x.mean()) ** 2)) + 0.5 * n / kappa_n * x.mean() ** 2
    )
    mu_n = (0.5 * 0.0 + n * x.mean()) / kappa_n
    exact_var = b_n / (kappa_n * (a_ev - 1.0))
    evidence = _normal_gamma_evidence(x, 0.0, 0.5, 2.0, 1.0)

    assert np.isclose(vi.mu_, mu_n, atol=1e-12)
    assert np.isclose(vi.a_, a_q, atol=1e-12)
    assert vi.a_ > a_ev  # the half-degree the mean-field factor steals from τ
    # q(τ) keeps a Gamma shape and inflates the rate by the factor the Gaussian
    # block forces on it: b = b_N / (1 − 1/(2a))
    assert np.isclose(vi.b_, b_n / (1.0 - 1.0 / (2.0 * a_q)), atol=1e-5)
    assert vi.mu_var_ < exact_var  # the same underestimate as mean_field
    assert np.all(np.diff(vi.elbo_history_) > -1e-9)
    assert vi.elbo_ <= evidence + 1e-9
    assert vi.elbo_ > evidence - 0.2
    print(
        f"variational_inference: cavi — Normal-Gamma fixed point a={vi.a_:.3f} "
        f"(exact {a_ev:.3f}), b={vi.b_:.3f}, μ={vi.mu_:.3f}; q(μ) variance "
        f"{vi.mu_var_:.4f} below the exact {exact_var:.4f}; ELBO {vi.elbo_:.3f} ≤ "
        f"evidence {evidence:.3f}, monotone over {vi.n_iter_} sweeps"
    )

    print("variational_inference demo passed")


if __name__ == "__main__":
    main()
