"""Elastic net — lasso and ridge in one objective, and the four ways to solve it.

Chapter:   02-linear-models
Companion: elastic_net.md
Tier 1 · 4 variants: naive, coordinate_descent, path, group_elastic_net

    min  ½‖y − Xβ‖² + λα‖β‖₁ + λ(1−α)/2 ‖β‖²

`α = 1` is the lasso, `α = 0` is ridge, and anything between is a strictly convex
problem that keeps lasso's exact zeros while sharing weight between correlated
columns. That last property is the reason it exists.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _centre(X, y, fit_intercept=True):
    if not fit_intercept:
        return X, y, np.zeros(X.shape[1]), 0.0
    xm = X.mean(axis=0)
    ym = y.mean()
    return X - xm, y - ym, xm, ym


def lam_max(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """Smallest λ that zeroes every coefficient: ‖Xᵀy‖∞ / α.

    The `1/α` is the whole reason an elastic-net λ grid is not comparable to a
    lasso grid at the same nominal value.
    """
    Xc, yc, _, _ = _centre(X, y)
    return float(np.abs(Xc.T @ yc).max()) / max(alpha, 1e-12)


def coordinate_descent(
    X: np.ndarray, y: np.ndarray, lam: float, alpha: float = 0.5, *,
    fit_intercept: bool = True, iters: int = 5000, tol: float = 1e-10,
    beta0: np.ndarray | None = None,
) -> dict:
    """One-coordinate exact updates.

    Each subproblem is ½(a + λ(1−α))β² − zβ + λα|β|, so the update is a soft
    threshold with the ridge term folded into the denominator:

        β_j ← S_{λα}(z_j) / (‖x_j‖² + λ(1−α))
    """
    Xc, yc, xm, ym = _centre(X, y, fit_intercept)
    n, d = Xc.shape
    beta = np.zeros(d) if beta0 is None else np.asarray(beta0, float).copy()
    sq = (Xc * Xc).sum(axis=0) + lam * (1.0 - alpha)
    sq = np.where(sq <= 0, 1.0, sq)
    thr = lam * alpha
    r = yc - Xc @ beta
    for _ in range(iters):
        delta = 0.0
        for j in range(d):
            old = beta[j]
            rho = Xc[:, j] @ r + (Xc[:, j] @ Xc[:, j]) * old
            # rho uses ‖x_j‖², not sq: sq includes the ridge term, which is not
            # part of the correlation.
            new = soft_threshold(rho, thr) / (sq[j])
            if new != old:
                beta[j] = new
                r -= Xc[:, j] * (new - old)
                delta = max(delta, abs(new - old))
        if delta < tol:
            break
    intercept = float(ym - xm @ beta) if fit_intercept else None
    return {"coef": beta, "intercept": intercept}


def naive(
    X: np.ndarray, y: np.ndarray, lam: float, alpha: float = 0.5, *,
    fit_intercept: bool = True, iters: int = 5000,
) -> dict:
    """Zou & Hastie's data augmentation: the elastic net *is* a lasso on

        X̃ = [X; √(λ(1−α))·I],   ỹ = [y; 0]

    because ½‖ỹ − X̃β‖² = ½‖y − Xβ‖² + λ(1−α)/2·‖β‖². Elegant, and it costs
    `n + d` rows, which is why nobody ships it.
    """
    Xc, yc, _, _ = _centre(X, y, fit_intercept)
    Xc = Xc if fit_intercept else np.asarray(X, dtype=float)
    yc = yc if fit_intercept else np.asarray(y, dtype=float).ravel()
    d = Xc.shape[1]
    Xa = np.vstack([Xc, np.sqrt(lam * (1.0 - alpha)) * np.eye(d)])
    ya = np.concatenate([yc, np.zeros(d)])
    beta = coordinate_descent(Xa, ya, lam * alpha, alpha=1.0,
                              fit_intercept=False, iters=iters)["coef"]
    xm = X.mean(axis=0) if fit_intercept else np.zeros(d)
    intercept = float(y.mean() - xm @ beta) if fit_intercept else None
    return {"coef": beta, "intercept": intercept, "augmented_shape": Xa.shape}


def path(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.5, *, n_lams: int = 25,
    lams: np.ndarray | None = None, fit_intercept: bool = True,
) -> dict:
    """Warm-started λ path at fixed α. Equals cold solves, at roughly the cost of
    one fit."""
    if lams is None:
        lams = np.geomspace(lam_max(X, y, alpha), 1e-3 * lam_max(X, y, alpha), n_lams)
    lams = np.asarray(lams, float)
    coefs = np.zeros((lams.size, X.shape[1]))
    beta = None
    for i, lam in enumerate(lams):
        beta = coordinate_descent(X, y, lam, alpha, fit_intercept=fit_intercept,
                                  beta0=beta)["coef"]
        coefs[i] = beta
    return {"lams": lams, "coefs": coefs, "alpha": alpha}


def group_elastic_net(
    X: np.ndarray, y: np.ndarray, lam: float, groups: list[np.ndarray], *,
    alpha: float = 0.5, iters: int = 5000, tol: float = 1e-10,
    fit_intercept: bool = True,
) -> dict:
    """Proximal gradient with the sparse-group prox. For a block `z`:

        prox(z) = max(1 − t·λα/‖z‖₂, 0) · z / (1 + t·λ(1−α))

    i.e. an L2 shrink followed by an L2 scaling. `α = 1` is the group lasso,
    `α = 0` is block ridge.
    """
    Xc, yc, xm, ym = _centre(X, y, fit_intercept)
    d = Xc.shape[1]
    L = float(np.linalg.eigvalsh(Xc.T @ Xc).max()) + 1e-12
    step = 1.0 / L
    beta = np.zeros(d)
    for _ in range(iters):
        z = beta - step * (Xc.T @ (Xc @ beta - yc))
        new = z.copy()
        for g in groups:
            g = np.asarray(g)
            nrm = np.linalg.norm(z[g])
            shrink = max(1.0 - step * lam * alpha / nrm, 0.0) if nrm > 0 else 0.0
            new[g] = shrink * z[g] / (1.0 + step * lam * (1.0 - alpha))
        if np.linalg.norm(new - beta) < tol:
            beta = new
            break
        beta = new
    intercept = float(ym - xm @ beta) if fit_intercept else None
    return {"coef": beta, "intercept": intercept}


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 100, 8
    X = rng.normal(size=(n, d))
    beta_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta_true + rng.normal(0, 0.5, size=n)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    lmax = float(np.abs(Xc.T @ yc).max())

    # 1. The augmentation trick and the closed-form update are the same estimator.
    diffs = []
    for alpha in (0.2, 0.5, 0.9):
        for frac in (0.5, 0.2, 0.05):
            lam = frac * lmax / alpha
            b1 = coordinate_descent(X, y, lam, alpha)["coef"]
            b2 = naive(X, y, lam, alpha)["coef"]
            diffs.append(np.abs(b1 - b2).max())
            assert np.allclose(b1, b2, atol=1e-8), (alpha, frac, diffs[-1])
    print(f"naive augmentation == coordinate descent, 9 (α, λ) pairs "
          f"(max diff {max(diffs):.1e})")

    # 2. α = 1 is the lasso and α = 0 is ridge: check against the closed form.
    lam = 0.05 * lmax
    b_ridge = coordinate_descent(X, y, lam, alpha=0.0)["coef"]
    b_closed = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    assert np.allclose(b_ridge, b_closed, atol=1e-9)
    b_lasso = coordinate_descent(X, y, lam, alpha=1.0)["coef"]
    assert (np.abs(b_lasso) < 1e-12).sum() >= 3          # the null features
    assert np.abs(b_ridge).min() > 1e-8                  # ridge never zeros
    print(f"α=0 == ridge closed form; α=1 zeros {(np.abs(b_lasso) < 1e-12).sum()} "
          f"of {d} coefficients, α=0 zeros none")

    # 3. The reason the elastic net exists: with two near-duplicate columns, the
    #    lasso keeps one and drops the other; the elastic net splits the weight.
    dup = X[:, [0]].copy()
    Xd = np.column_stack([dup, dup + 0.01 * rng.normal(size=(n, 1)), X[:, 2:]])
    yd = Xd[:, 0] * 2.5 + rng.normal(0, 0.3, size=n)
    lam_d = 0.08 * float(np.abs((Xd - Xd.mean(0)).T @ (yd - yd.mean())).max())
    b_l = coordinate_descent(Xd, yd, lam_d, alpha=1.0)["coef"]
    b_e = coordinate_descent(Xd, yd, lam_d, alpha=0.5)["coef"]
    pair_lasso = np.abs(b_l[:2])
    pair_enet = np.abs(b_e[:2])
    assert (pair_lasso < 1e-10).sum() == 1, pair_lasso
    assert np.all(pair_enet > 1e-3), pair_enet
    assert abs(pair_enet[0] - pair_enet[1]) < 0.3 * pair_enet.max(), pair_enet
    print(f"duplicate columns: lasso |β₁:₂| = {np.round(pair_lasso, 3)}, "
          f"elastic net {np.round(pair_enet, 3)}")

    # 4. Above λ_max = ‖Xᵀy‖∞/α every coefficient is zero, for every α.
    for alpha in (0.25, 0.5, 1.0):
        lm = lam_max(X, y, alpha)
        assert np.allclose(coordinate_descent(X, y, 1.01 * lm, alpha)["coef"], 0.0)
        assert np.abs(coordinate_descent(X, y, 0.99 * lm, alpha)["coef"]).max() > 0
    print("λ_max = ‖Xᵀy‖∞/α: one coefficient enters exactly at λ_max")

    # 5. Warm-started path equals cold solves, and the support only grows.
    P = path(X, y, alpha=0.5, n_lams=15)
    for i, lam in enumerate(P["lams"]):
        cold = coordinate_descent(X, y, lam, 0.5)["coef"]
        assert np.allclose(P["coefs"][i], cold, atol=1e-9)
    nnz = (np.abs(P["coefs"]) > 1e-8).sum(1)
    assert nnz[0] <= nnz[-1] and np.all(np.diff(nnz) >= 0)
    print(f"path: {nnz.size} λ values, nnz {nnz[0]} -> {nnz[-1]}, monotone")

    # 6. Sparse-group prox: whole groups switch off, and the mixture changes how
    #    much the survivors shrink without changing which ones survive.
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
    lam_g = 0.3 * lmax
    gl = group_elastic_net(X, y, lam_g, groups, alpha=1.0)["coef"]
    ge = group_elastic_net(X, y, lam_g, groups, alpha=0.5)["coef"]
    gl_norms = np.array([np.linalg.norm(gl[g]) for g in groups])
    ge_norms = np.array([np.linalg.norm(ge[g]) for g in groups])
    assert (gl_norms == 0).sum() >= 2 and (ge_norms == 0).sum() >= 2
    assert np.all(np.argsort(-gl_norms)[:2] == np.argsort(-ge_norms)[:2])
    print(f"group_elastic_net: α=1 norms {np.round(gl_norms, 2)}, "
          f"α=0.5 {np.round(ge_norms, 2)} (same support)")
    print("elastic_net demo passed")


if __name__ == "__main__":
    main()
