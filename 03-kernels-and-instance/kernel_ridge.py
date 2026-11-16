"""Kernel ridge — the representer theorem in closed form.

Chapter:   03-kernels-and-instance
Companion: kernel_ridge.md
Tier 1 · 6 variants: linear, rbf, laplacian, polynomial, nystrom, leave_one_out

Least squares in a reproducing-kernel Hilbert space:

    min_f  ½ Σ (y_i − f(x_i))² + λ/2 ‖f‖²_H

The representer theorem says the minimiser is a finite expansion over the
training points, `f = Σ α_i k(x_i, ·)`, so the infinite-dimensional problem
collapses to one `n × n` linear system:

    (K + λI) α = y        K_ij = k(x_i, x_j)

    linear / rbf / laplacian / polynomial   the kernel
    nystrom                                 rank-m approximation, m ≪ n
    leave_one_out                           the exact LOO residual, in closed form

Kernel ridge is the SVM with a squared loss instead of a hinge: no support
vectors, no QP, one linear solve. That makes it the atom to reach for when you
want to see what a kernel *does* without an optimiser in the way.
"""

from __future__ import annotations

import numpy as np
from svm import kernel_matrix as _gram

VARIANTS = frozenset(
    ["linear", "rbf", "laplacian", "polynomial", "nystrom", "leave_one_out"]
)
KERNELS = frozenset(["linear", "rbf", "laplacian", "polynomial"])


def kernel_matrix(X: np.ndarray, Z: np.ndarray | None = None, *,
                  kind: str = "rbf", degree: int = 3,
                  gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    """`svm.kernel_matrix` plus the Laplacian kernel.

    The Laplacian kernel `exp(−γ‖x − z‖₁)` is the RBF kernel's `L1` twin: its
    sample paths are steeper and less smooth, which suits piecewise-constant
    targets. It is conditionally positive definite (not PSD) on `R^d`, so the
    Gram matrix can have small negative eigenvalues — harmless with `λ > 0`.
    """
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}")
    if kind != "laplacian":
        return _gram(X, Z, kind=kind, degree=degree, gamma=gamma, coef0=coef0)
    Z = X if Z is None else Z
    gamma = 1.0 / X.shape[1] if gamma is None else float(gamma)
    d1 = np.abs(X[:, None, :] - Z[None, :, :]).sum(axis=2)
    return np.exp(-gamma * d1)


class KernelRidge:
    """Kernel least squares with an optional Nyström approximation.

    Parameters
    ----------
    variant : str
        One of VARIANTS. A kernel name (`linear`, `rbf`, …) selects that kernel
        and otherwise behaves like the plain solver.
    lam : float
        Ridge penalty `λ` on the RKHS norm. `λ → 0` interpolates the data.
    degree, gamma, coef0 : float
        Kernel parameters. `gamma=None` means `1/d`.
    n_landmarks : int
        Landmark count `m` for `nystrom`; must exceed the kernel's effective
        dimension for the approximation to be close.
    loo_grid : array-like of float
        Candidate `λ` values for `leave_one_out`; the one with the smallest
        closed-form LOO error is kept and the model is refitted at it.
    fit_intercept : bool
        Centre `X` and `y` first, which is what adding an unpenalised bias to
        the feature map would do.

    Attributes
    ----------
    alpha_ : (n,) or (m,) dual weights, in training-point or landmark space
    landmark_ : (m, d) landmarks when `variant="nystrom"`
    loo_scores_ : (n,) leave-one-out residuals when `variant="leave_one_out"`
    """

    def __init__(self, variant: str = "rbf", lam: float = 1e-3, *,
                 kernel: str = "rbf", degree: int = 3, gamma: float | None = None,
                 coef0: float = 1.0, n_landmarks: int = 50,
                 loo_grid=None, fit_intercept: bool = True,
                 random_state: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        self.kernel = variant if variant in KERNELS else kernel
        self.lam = float(lam)
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.n_landmarks = int(n_landmarks)
        self.loo_grid = None if loo_grid is None else np.asarray(
            loo_grid, dtype=float)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------ #
    def _k(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return kernel_matrix(X, Z, kind=self.kernel, degree=self.degree,
                             gamma=self.gamma, coef0=self.coef0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelRidge:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = float(y.mean())
            Xc, yc = X - self.x_mean_, y - self.y_mean_
        else:
            self.x_mean_ = np.zeros(X.shape[1])
            self.y_mean_ = 0.0
            Xc, yc = X, y

        if self.variant == "leave_one_out":
            grid = self.loo_grid if self.loo_grid is not None else np.logspace(
                -6, 2, 25)
            scores = [float(np.sqrt(np.mean(self._loo(Xc, yc, lam) ** 2)))
                      for lam in grid]
            self.lam = float(grid[int(np.argmin(scores))])
            self.loo_path_ = (grid, np.asarray(scores))
            self.loo_scores_ = self._loo(Xc, yc, self.lam)

        if self.variant == "nystrom":
            self._fit_nystrom(Xc, yc)
        else:
            K = self._k(Xc)
            self.X_ = Xc
            self.alpha_ = np.linalg.solve(
                K + self.lam * np.eye(K.shape[0]), yc)
        return self

    # ------------------------------------------------------------------ #
    def _loo(self, Xc: np.ndarray, yc: np.ndarray, lam: float) -> np.ndarray:
        """Exact leave-one-out residuals `e_i / (1 − H_ii)`.

        `H = K(K + λI)^{-1}` is the smoother of the centred problem, so it
        scales the residual of row `i` by `1 − H_ii`. No refitting: one solve
        per `λ`, not `n`.

        The intercept is the full-data mean, held fixed across folds — the
        standard kernel-ridge convention, since the RKHS of an RBF kernel does
        not contain the constant function. Re-estimating it in every fold is a
        slightly different estimator and moves these numbers by `O(1/n)`.
        """
        n = yc.size
        K = self._k(Xc)
        A = K + lam * np.eye(n)
        alpha = np.linalg.solve(A, yc)
        h = np.diag(np.linalg.solve(A, K))     # symmetric in exact arithmetic
        return (yc - K @ alpha) / (1.0 - h)

    def _fit_nystrom(self, Xc: np.ndarray, yc: np.ndarray) -> None:
        """Rank-`m` approximation `K ≈ Z W⁻¹ Zᵀ` solved in `m` dimensions.

        Woodbury turns the `n × n` system into an `m × m` one:

            (Z W⁻¹ Zᵀ + λI)⁻¹ y = (y − Z c)/λ,   c = (λW + ZᵀZ)⁻¹ Zᵀ y

        so both fitting and prediction cost `O(nm²)` instead of `O(n³)`.
        """
        n = Xc.shape[0]
        m = min(self.n_landmarks, n)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)[:m]
        self.landmark_ = Xc[idx]
        Z = self._k(Xc, self.landmark_)                  # (n, m)
        W = self._k(self.landmark_)                      # (m, m)
        W = 0.5 * (W + W.T) + 1e-10 * np.eye(m)          # symmetrise, guard
        c = np.linalg.solve(self.lam * W + Z.T @ Z, Z.T @ yc)
        self.alpha_ = np.linalg.solve(W, Z.T @ ((yc - Z @ c) / self.lam))

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float) - self.x_mean_
        if self.variant == "nystrom":
            out = self._k(X, self.landmark_) @ self.alpha_
        else:
            out = self._k(X, self.X_) @ self.alpha_
        return out + self.y_mean_


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _circles(rng, n, noise=0.15):
    """Concentric rings of radius 0.7 and 1.8 — no line separates them."""
    theta = rng.uniform(0, 2 * np.pi, 2 * n)
    radius = np.where(np.arange(2 * n) < n, 0.7, 1.8)
    X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    X = X + rng.normal(0, noise, size=X.shape)
    return X, (radius > 1.2).astype(int)


def main() -> None:
    rng = np.random.default_rng(0)

    # --- linear kernel ridge *is* primal ridge, exactly ---
    X = rng.normal(size=(60, 5))
    w_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w_true + 0.1 * rng.normal(size=60)
    lam = 0.5
    kr = KernelRidge("linear", lam=lam).fit(X, y)
    Xc, yc = X - X.mean(0), y - y.mean()
    w_primal = np.linalg.solve(Xc.T @ Xc + lam * np.eye(5), Xc.T @ yc)
    w_dual = Xc.T @ kr.alpha_
    assert np.allclose(w_dual, w_primal, atol=1e-9), np.max(
        np.abs(w_dual - w_primal))
    assert np.allclose(kr.alpha_, np.linalg.solve(
        Xc @ Xc.T + lam * np.eye(60), yc), atol=1e-9)
    assert np.allclose(kr.predict(X), X @ w_dual + (y.mean()
                                                    - X.mean(0) @ w_dual),
                       atol=1e-9)
    print(f"kernel_ridge: linear kernel — dual weights reproduce the primal "
          f"ridge solution, max |Δw| {np.max(np.abs(w_dual - w_primal)):.1e}")

    # --- the representer identity: primal optimum = λ/2 · yᵀα ---
    Xm, ym = _circles(rng, 60)
    sign = np.where(ym == 1, 1.0, -1.0)
    norms = []
    for lam_r in (1e-3, 1e-1):
        kr = KernelRidge("rbf", lam=lam_r, gamma=2.0).fit(Xm, sign)
        Xc, yc = Xm - kr.x_mean_, sign - kr.y_mean_
        K = kr._k(Xc)
        alpha = kr.alpha_
        rkhs = float(alpha @ K @ alpha)                  # ‖f‖²_H
        resid = float(np.sum((yc - K @ alpha) ** 2))
        primal = 0.5 * resid + 0.5 * lam_r * rkhs
        assert np.isclose(primal, 0.5 * lam_r * float(yc @ alpha), atol=1e-9)
        # residuals are exactly λ·α, which is why LOO needs no refitting
        assert np.allclose(yc - K @ alpha, lam_r * alpha, atol=1e-9)
        norms.append(rkhs)
    # more smoothing, smaller RKHS norm: the penalty is what is being bought
    assert norms[1] < norms[0], norms
    print(f"kernel_ridge: rbf — primal optimum equals ½λ yᵀα, residuals are "
          f"exactly λα; ‖f‖²_H {norms[0]:.1f} at λ=1e-3 vs {norms[1]:.1f} at 1e-1")

    # --- λ → 0 recovers interpolation: (K + λI)α = y has no slack left ---
    Xg = rng.uniform(-3, 3, size=(40, 1))
    yg = np.sin(3 * Xg[:, 0]) + 0.5 * Xg[:, 0]
    tight = KernelRidge("rbf", lam=1e-8, gamma=0.5).fit(Xg, yg)
    resid_t = np.max(np.abs(tight.predict(Xg) - yg))
    assert resid_t < 1e-3, resid_t
    assert np.max(np.abs(tight.predict(Xg))) < 3.0
    print(f"kernel_ridge: rbf with λ=1e-8 interpolates 40 points, max |residual| "
          f"{resid_t:.1e}")

    # --- kernels: the crescents need a curved one ---
    acc = {}
    for name in ("linear", "polynomial", "rbf", "laplacian"):
        model = KernelRidge(name, lam=1e-2, gamma=2.0, degree=2).fit(Xm, sign)
        acc[name] = float(np.mean(np.sign(model.predict(Xm)) == sign))
    assert acc["linear"] < 0.75, acc
    for name in ("polynomial", "rbf", "laplacian"):
        assert acc[name] > 0.95, acc
    assert np.linalg.eigvalsh(kernel_matrix(Xm, kind="rbf", gamma=2.0)).min() > 0
    print("kernel_ridge: rings — " + ", ".join(
        f"{k} {v:.3f}" for k, v in acc.items()))

    # --- nystrom: exact for a low-rank kernel, and error falls with m ---
    Xn = rng.normal(size=(400, 5))
    yn = np.sin(Xn[:, 0]) + 0.3 * Xn[:, 1] * Xn[:, 2] + 0.05 * rng.normal(size=400)
    full = KernelRidge("linear", lam=1e-3).fit(Xn, yn)
    low = KernelRidge("nystrom", lam=1e-3, n_landmarks=5,
                      kernel="linear").fit(Xn, yn)
    # a linear kernel has rank ≤ d = 5, so m = d reproduces it exactly
    exact = np.max(np.abs(full.predict(Xn) - low.predict(Xn)))
    assert exact < 1e-6, exact
    errors = []
    for m in (10, 40, 120):
        approx = KernelRidge("nystrom", lam=1e-3, n_landmarks=m,
                             gamma=0.5).fit(Xn, yn)
        tru = KernelRidge("rbf", lam=1e-3, gamma=0.5).fit(Xn, yn)
        errors.append(float(np.sqrt(np.mean(
            (approx.predict(Xn) - tru.predict(Xn)) ** 2))))
    assert errors[0] > errors[1] > errors[2], errors
    print(f"kernel_ridge: nystrom — rank-5 linear kernel reproduced exactly "
          f"(max |Δ| {exact:.1e}); rbf rmse vs full {errors[0]:.4f}, "
          f"{errors[1]:.4f}, {errors[2]:.4f} at m=10/40/120")

    # --- leave_one_out: closed form == refit-without-i, and it picks λ ---
    loo = KernelRidge("leave_one_out", gamma=2.0,
                      loo_grid=np.logspace(-5, 1, 13)).fit(Xm, sign)
    picked = loo.lam
    Xc = Xm - loo.x_mean_
    yc = sign - loo.y_mean_
    worst = 0.0
    for i in (0, 17, 55, 119):
        kept = np.arange(Xm.shape[0]) != i
        refit = KernelRidge("rbf", lam=picked, gamma=2.0,
                            fit_intercept=False).fit(Xc[kept], yc[kept])
        err = float(yc[i] - refit.predict(Xc[i:i + 1])[0])
        worst = max(worst, abs(err - loo.loo_scores_[i]))
    assert worst < 1e-9, worst
    grid, path = loo.loo_path_
    assert np.all(np.isfinite(path)) and picked == grid[int(np.argmin(path))]
    assert grid[0] < picked < grid[-1]                  # an interior minimum
    print(f"kernel_ridge: leave_one_out — e_i/(1−H_ii) matches the refitted "
          f"model to {worst:.0e}; log λ picked {np.log10(picked):+.2f} "
          f"(LOO rmse {path.min():.3f} vs {path[0]:.3f} at the grid edge)")

    print("kernel_ridge demo passed")


if __name__ == "__main__":
    main()
