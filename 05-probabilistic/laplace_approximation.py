"""Laplace approximation — a Gaussian around the mode.

Chapter:   05-probabilistic
Companion: laplace_approximation.md
Tier 3 · 4 variants: diag_hessian, full_hessian, kfac_laplace, predictive

    p(θ | y) ≈ N(θ̂, P⁻¹),   P = −∇² log p(θ̂ | y),   ∇ log p(θ̂ | y) = 0

Two steps, and only the second is the approximation. Find the mode exactly — it
is a root of the gradient, so Newton gets there in a handful of steps. Then
pretend the log-density is quadratic there. The error is in the curvature away
from the mode, and every variant here is a different answer to "how much
curvature can you afford to know":

    full_hessian   P⁻¹, all d² entries                     O(d³)
    diag_hessian   1/ diag(P), correlations ignored        O(d)
    kfac_laplace   per-layer Kronecker factors            O(k³ + d³)
    predictive     full Hessian, then integrate, not plug in

The demonstration model is Bayesian logistic regression, because its exact
posterior is computable on a 2-D grid and every claim below is checked against
that grid rather than against another approximation.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["diag_hessian", "full_hessian", "kfac_laplace", "predictive"])

LOG2PI = math.log(2.0 * math.pi)


def sigmoid(a: np.ndarray) -> np.ndarray:
    """Logistic function, evaluated without overflowing on either tail."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    pos = a >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def kfac_precision(X: np.ndarray, factors, prior_prec: float = 0.0) -> np.ndarray:
    """`Ā ⊗ (XᵀX) + prior`, the per-layer Kronecker form.

    `factors` is `(n,)` for a scalar output (each `A_n` is a 1×1 matrix) or
    `(n, k, k)` for `k` classes. Pulling the mean outside the Kronecker product
    is the approximation: the exact precision is `Σ_n A_n ⊗ x_n x_nᵀ`, and the
    two agree iff `A_n` does not vary with `n`. The factor is a mean, not a sum,
    because `Σ_n a ⊗ x_n x_nᵀ = a ⊗ (XᵀX)` when every `A_n = a`.
    """
    f = np.asarray(factors, dtype=float)
    if f.ndim not in (1, 3):
        raise ValueError("factors must be (n,) or (n, k, k)")
    A = np.atleast_2d(f.mean(axis=0) if f.ndim == 3 else f.mean())
    k = A.shape[0]
    return np.kron(A, X.T @ X) + prior_prec * np.eye(k * X.shape[1])


def multinomial_precision(X: np.ndarray, probs: np.ndarray,
                          prior_prec: float = 0.0) -> np.ndarray:
    """The exact precision of a multinomial-logistic likelihood, `Σ_n A_n ⊗ x xᵀ`."""
    d = X.shape[1]
    P = prior_prec * np.eye(probs.shape[1] * d)
    for x, p in zip(X, probs, strict=True):
        P += np.kron(np.diag(p) - np.outer(p, p), np.outer(x, x))
    return P


class LaplaceLogistic:
    """Binary logistic regression with a Gaussian prior, Laplace-approximated.

    Parameters are `w` of shape `(d,)`; `X` is `(n, d)` and `y` is `(n,)` in
    `{0, 1}`. The prior is `N(0, prior_prec⁻¹ I)`, so `prior_prec` is the
    precision, not the variance.
    """

    def __init__(self, X, y, *, variant: str = "full_hessian",
                 prior_prec: float = 1.0, n_iter: int = 100,
                 tol: float = 1e-10):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.X.ndim != 2 or self.y.shape != (self.X.shape[0],):
            raise ValueError("expected X of shape (n, d) and y of shape (n,)")
        if not np.all((self.y == 0) | (self.y == 1)):
            raise ValueError("labels must be 0 or 1; use a one-vs-rest wrapper")
        self.prior_prec = float(prior_prec)
        self.n_iter = int(n_iter)
        self.tol = float(tol)

    # -- the density and its first two derivatives -------------------------- #
    def _p(self, w: np.ndarray) -> np.ndarray:
        return sigmoid(self.X @ w)

    def _log_post(self, w: np.ndarray) -> float:
        a = self.X @ w
        # log σ(a) = −log(1+e^{−a}) and log(1−σ(a)) = −log(1+e^{a}); the
        # second form is what logaddexp computes without overflow
        return float(np.sum(self.y * a - np.logaddexp(0.0, a))
                     - 0.5 * self.prior_prec * float(w @ w))

    def _grad(self, w: np.ndarray) -> np.ndarray:
        return self.X.T @ (self.y - self._p(w)) - self.prior_prec * w

    def _precision(self, w: np.ndarray) -> np.ndarray:
        """`−∇² log p(w | y)`: positive definite by construction."""
        p = self._p(w)
        return (self.X.T * (p * (1.0 - p))) @ self.X + self.prior_prec * np.eye(w.size)

    # -- step one: find the mode -------------------------------------------- #
    def fit(self, w0: np.ndarray | None = None) -> LaplaceLogistic:
        w = np.zeros(self.X.shape[1]) if w0 is None else np.asarray(w0, float).copy()
        obj = self._log_post(w)
        self.history_ = [obj]
        for _ in range(self.n_iter):
            step = np.linalg.solve(self._precision(w), self._grad(w))
            # Newton on a logistic posterior can overshoot into a flat region
            # where the quadratic model is worthless; halving until the
            # objective does not decrease costs one extra evaluation per step
            t = 1.0
            while t > 1e-12:
                cand = w + t * step
                new = self._log_post(cand)
                if new >= obj - 1e-12:
                    break
                t *= 0.5
            w, obj = cand, new
            self.history_.append(obj)
            if np.max(np.abs(t * step)) < self.tol:
                break
        self.w_, self.n_iter_ = w, len(self.history_) - 1
        self.objective_ = obj
        return self

    # -- step two: the Gaussian --------------------------------------------- #
    def covariance(self) -> np.ndarray:
        """The approximate posterior covariance, `(d, d)`, per variant."""
        P = self._precision(self.w_)
        if self.variant == "diag_hessian":
            # for a positive-definite P, (P⁻¹)_ii ≥ 1/P_ii with equality iff the
            # row is uncorrelated: dropping the off-diagonals treats the other
            # parameters as known, so it can only *narrow* a marginal
            return np.diag(1.0 / np.diag(P))
        if self.variant == "kfac_laplace":
            p = self._p(self.w_)
            return np.linalg.inv(kfac_precision(self.X, p * (1.0 - p), self.prior_prec))
        return np.linalg.inv(P)

    def log_evidence(self) -> float:
        """Laplace's approximation to `log p(y)`.

        `log p(y) ≈ log p(w, y) + d/2 log 2π − ½ log|P|`: the volume of the
        posterior replaces the integral, which is the whole content of the
        method.
        """
        P = self._precision(self.w_)
        _, logdet = np.linalg.slogdet(P)
        return self.objective_ + 0.5 * self.w_.size * LOG2PI - 0.5 * logdet

    def predict_proba(self, X_new, *, integrate: bool | None = None) -> np.ndarray:
        """`p(y=1 | x)`, plugged in at the mode or integrated over the posterior.

        The two differ because `σ` is nonlinear: averaging over the posterior
        pulls the answer toward ½, which is the honest answer when the weight
        vector is uncertain.
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        mu = X_new @ self.w_
        if integrate is None:
            integrate = self.variant == "predictive"
        if not integrate:
            return sigmoid(mu)
        var = np.einsum("nd,de,ne->n", X_new, self.covariance(), X_new)
        x, w = np.polynomial.hermite.hermgauss(20)
        # ∫ σ(μ + √v z) φ(z) dz = π^{-1/2} Σ w_i σ(μ + √(2v) x_i)
        out = np.sqrt(2.0 * np.maximum(var, 0.0))[:, None] * x + mu[:, None]
        return (w * sigmoid(out)).sum(axis=1) / math.sqrt(math.pi)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _posterior_grid(X, y, prior_prec, lo=-3.0, hi=3.0, n=241):
    """The exact posterior on an `n × n` grid over the two weights."""
    g = np.linspace(lo, hi, n)
    W0, W1 = np.meshgrid(g, g, indexing="ij")
    W = np.stack([W0, W1], axis=-1)                      # (n, n, 2)
    a = W @ X.T                                          # (n, n, m)
    logp = np.sum(y * a - np.logaddexp(0.0, a), axis=-1)
    logp -= 0.5 * prior_prec * np.sum(W**2, axis=-1)
    w = np.exp(logp - logp.max())
    w /= w.sum()
    mean = np.array([(w * W0).sum(), (w * W1).sum()])
    dev = np.stack([W0 - mean[0], W1 - mean[1]], axis=-1)
    cov = np.einsum("ij,ijk,ijl->kl", w, dev, dev)
    # the grid is a Riemann sum: each cell has area (hi − lo)²/(n − 1)²
    cell = (g[1] - g[0]) ** 2
    logz = float(np.log(np.exp(logp - logp.max()).sum() * cell) + logp.max())
    return w, mean, cov, g, logz


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 60, 2
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -1.0])
    y = (rng.random(n) < sigmoid(X @ w_true)).astype(float)
    prior_prec = 1.0

    full = LaplaceLogistic(X, y, variant="full_hessian", prior_prec=prior_prec).fit()
    assert np.max(np.abs(full._grad(full.w_))) < 1e-9, "Newton did not reach the mode"
    assert np.all(np.diff(full.history_) > -1e-9), "the objective decreased"
    print(
        f"laplace_approximation: full_hessian — mode {np.round(full.w_, 3)} reached in "
        f"{full.n_iter_} Newton steps, gradient {np.max(np.abs(full._grad(full.w_))):.1e}, "
        f"true weights {w_true}"
    )

    # --- against the exact posterior on a grid -----------------------------
    wg, mean_g, cov_g, grid, logz_g = _posterior_grid(X, y, prior_prec)
    err_mean = float(np.max(np.abs(full.w_ - mean_g)))
    sd_g = np.sqrt(np.diag(cov_g))
    err_cov = float(np.max(np.abs(full.covariance() - cov_g) / np.outer(sd_g, sd_g)))
    assert err_mean < 0.1 and err_cov < 0.2, (err_mean, err_cov)
    print(
        f"laplace_approximation: full_hessian — the posterior is skewed: its mode and "
        f"mean differ by {err_mean:.4f}, and the Gaussian covariance is within "
        f"{100 * err_cov:.1f}% of the exact one entry by entry"
    )

    # --- the evidence, which is where Laplace earns its keep ---------------
    logz_lap = full.log_evidence()
    err_ev = abs(logz_lap - logz_g)
    assert err_ev < 0.05, (logz_lap, logz_g)
    print(
        f"laplace_approximation: full_hessian — log evidence {logz_lap:.4f} against "
        f"the grid's {logz_g:.4f}, off by {err_ev:.4f} nats"
    )

    # --- the diagonal approximation can only narrow a marginal -------------
    diag = LaplaceLogistic(X, y, variant="diag_hessian", prior_prec=prior_prec).fit()
    v_full = np.diag(full.covariance())
    v_diag = np.diag(diag.covariance())
    assert np.all(v_diag < v_full), (v_diag, v_full)
    # for two parameters the shrink factor is exactly 1 − ρ²
    rho = full.covariance()[0, 1] / np.sqrt(v_full[0] * v_full[1])
    assert np.allclose(v_diag / v_full, 1.0 - rho**2, atol=1e-12)
    print(
        f"laplace_approximation: diag_hessian — variances {np.round(v_diag, 4)} against "
        f"full {np.round(v_full, 4)}, a shrink of 1−ρ² with ρ = {rho:.3f}: treating the "
        f"other weight as known makes each marginal overconfident"
    )

    # --- KFAC: exact where the output factor is shared, and only there -----
    a0 = np.full(n, 0.25)                   # p = ½ at w = 0, so p(1−p) is constant
    kfac_at_zero = np.linalg.inv(kfac_precision(X, a0, prior_prec))
    full_at_zero = np.linalg.inv(full._precision(np.zeros(d)))
    assert np.allclose(kfac_at_zero, full_at_zero, atol=1e-12)
    p = full._p(full.w_)
    P_exact = full._precision(full.w_)
    P_kfac = kfac_precision(X, p * (1.0 - p), prior_prec)
    rel = float(np.linalg.norm(P_kfac - P_exact) / np.linalg.norm(P_exact))
    assert rel > 1e-3, rel
    kfac = LaplaceLogistic(X, y, variant="kfac_laplace", prior_prec=prior_prec).fit()
    ratio = np.diag(kfac.covariance()) / np.diag(full.covariance())
    print(
        f"laplace_approximation: kfac_laplace — exact at the symmetric point (<1e−12) "
        f"and {100 * rel:.1f}% off the precision at the mode: with one output unit the "
        f"factorisation has only the input factor left, so it is a mean curvature; the "
        f"marginal variances move by up to {100 * float(np.max(np.abs(ratio - 1.0))):.0f}%"
    )

    # the factorisation is what makes it cheap: inverting a Kronecker product is
    # inverting each factor, exactly, in O(k³ + d³) instead of O((kd)³)
    A = np.array([[2.0, 0.3], [0.3, 1.0]])
    B = np.array([[3.0, -1.0], [-1.0, 2.0]])
    assert np.allclose(np.linalg.inv(np.kron(A, B)),
                       np.kron(np.linalg.inv(A), np.linalg.inv(B)), atol=1e-12)
    # and the multinomial case shows the same algebra with a real k×k factor
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    X3 = rng.normal(size=(3, d))
    shared = np.diag(np.full(3, 1.0 / 3.0)) - np.full((3, 3), 1.0 / 9.0)
    assert np.allclose(multinomial_precision(X3, np.full((3, 3), 1.0 / 3.0), prior_prec),
                       kfac_precision(X3, np.tile(shared, (3, 1, 1)), prior_prec),
                       atol=1e-12)
    mats = np.stack([np.diag(p) - np.outer(p, p) for p in probs])
    assert not np.allclose(multinomial_precision(X3, probs, prior_prec),
                           kfac_precision(X3, mats, prior_prec), atol=1e-6)
    print(
        f"laplace_approximation: kfac_laplace — for 3 classes over {d} features the "
        f"factors are 3×3 and {d}×{d} instead of a {3 * d}×{3 * d} precision; "
        f"identical when the class factor is shared, otherwise not"
    )

    # --- predictive: integrate, do not plug in -----------------------------
    # The gap is the second-order term the delta method drops:
    # E[σ(μ+ε)] ≈ σ(μ) + ½σ''(μ)v, and σ'' has the sign of −(σ(μ) − ½).
    pred = LaplaceLogistic(X, y, variant="predictive", prior_prec=prior_prec).fit()
    direction = pred.w_ / np.linalg.norm(pred.w_)
    cov = pred.covariance()
    gaps = []
    for sign in (1.0, -1.0):
        x_test = sign * 0.9 * direction
        mu = float(x_test @ pred.w_)
        v = float(x_test @ cov @ x_test)
        plug = float(pred.predict_proba(x_test, integrate=False)[0])
        integ = float(pred.predict_proba(x_test)[0])
        p = float(sigmoid(mu))
        second = -0.5 * p * (1.0 - p) * (1.0 - 2.0 * p) * v
        assert np.isclose(plug - integ, second, rtol=0.05, atol=1e-12), (plug - integ, second)
        gaps.append((mu, plug, integ, second))

    # the exact posterior, for scale: both Laplace numbers land near it, and
    # the sign of the correction is what survives the approximation
    flat = wg.ravel()
    picks = rng.choice(flat.size, size=40_000, p=flat / flat.sum())
    i0, i1 = np.unravel_index(picks, wg.shape)
    samples = np.stack([grid[i0], grid[i1]], axis=1)
    mc = float(np.mean(sigmoid(samples @ (0.9 * direction))))
    assert abs(gaps[0][1] - mc) < 0.02 and abs(gaps[0][2] - mc) < 0.02
    print(
        f"laplace_approximation: predictive — at μ = {gaps[0][0]:+.2f} the plug-in gives "
        f"{gaps[0][1]:.4f} and the integral {gaps[0][2]:.4f}: the difference is "
        f"{gaps[0][3]:+.4f}, which is −½σ''(μ)v, and it flips at −μ. The exact "
        f"posterior says {mc:.4f}"
    )
    print("laplace_approximation demo passed")


if __name__ == "__main__":
    main()
