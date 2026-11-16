"""Ridge — 6 ways to add a quadratic penalty, and one place where it is a matrix.

Chapter:   02-linear-models
Companion: ridge.md
Tier 1 · 6 variants: closed_form, svd, generalized_ridge, loo_cv, group_ridge,
nuclear_norm

Ridge solves min ||y - Xβ||² + λ||β||². The variants differ in *how* they solve it
(SVD exposes the shrinkage, Cholesky/the normal equations are fast), *what* is
penalised (all coefficients, per-group, or the singular values of a matrix), and
*how* λ is chosen (closed-form leave-one-out).

Intercepts are never penalised: shrink toward zero assumes the features are
centred. This implementation centres whenever `fit_intercept=True`.
"""

from __future__ import annotations

import numpy as np


def _design(X: np.ndarray, fit_intercept: bool):
    if not fit_intercept:
        return X, np.zeros(X.shape[1])
    xm = X.mean(axis=0)
    return X - xm, xm


def ridge(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    variant: str = "closed_form",
    *,
    fit_intercept: bool = True,
    penalty: np.ndarray | None = None,
    groups: list[np.ndarray] | None = None,
    lams: np.ndarray | None = None,
) -> dict:
    """Ridge regression.

    penalty: per-coefficient penalty (generalized_ridge), defaults to lam * ones.
    groups:  list of index arrays; each group gets its own lam (group_ridge).
    lams:    grid for loo_cv.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    Xc, xm = _design(X, fit_intercept)
    ym = y.mean() if fit_intercept else 0.0
    yc = y - ym
    diag = np.full(d, lam) if penalty is None else np.asarray(penalty, dtype=float)

    if variant == "closed_form":
        beta = np.linalg.solve(Xc.T @ Xc + np.diag(diag), Xc.T @ yc)
        chosen = lam
    elif variant == "svd":
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        shrink = s / (s * s + lam)              # component-wise 1/(s + λ/s)
        beta = Vt.T @ (shrink * (U.T @ yc))
        chosen = lam
    elif variant == "generalized_ridge":
        beta = np.linalg.solve(Xc.T @ Xc + np.diag(diag), Xc.T @ yc)
        chosen = float(diag.mean())
    elif variant == "group_ridge":
        if groups is None:
            raise ValueError("group_ridge needs groups")
        pen = np.zeros(d)
        for j, g in enumerate(groups):
            pen[np.asarray(g)] = lam if penalty is None else np.asarray(penalty)[j]
        beta = np.linalg.solve(Xc.T @ Xc + np.diag(pen), Xc.T @ yc)
        chosen = float(pen.mean())
    elif variant == "loo_cv":
        grid = np.logspace(-2, 3, 40) if lams is None else np.asarray(lams)
        errors = [loo_error(X, y, ll, fit_intercept) for ll in grid]
        chosen = float(grid[int(np.argmin(errors))])
        beta = np.linalg.solve(Xc.T @ Xc + chosen * np.eye(d), Xc.T @ yc)
        return {
            "coef": beta,
            "intercept": float(ym - xm @ beta) if fit_intercept else None,
            "lam": chosen,
            "cv": np.array(errors),
            "grid": grid,
        }
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "coef": beta,
        "intercept": float(ym - xm @ beta) if fit_intercept else None,
        "lam": chosen,
    }


def _augmented(X: np.ndarray, fit_intercept: bool):
    """Design with an unpenalised intercept column, and its penalty mask."""
    if fit_intercept:
        return np.column_stack([np.ones(len(X)), X]), np.diag([0.0] + [1.0] * X.shape[1])
    return X, np.eye(X.shape[1])


def hat_matrix(X: np.ndarray, lam: float, fit_intercept: bool = True) -> np.ndarray:
    """H = X̃ (X̃ᵀX̃ + λD)⁻¹ X̃ᵀ, with the intercept column unpenalised."""
    Xa, D = _augmented(X, fit_intercept)
    return Xa @ np.linalg.solve(Xa.T @ Xa + lam * D, Xa.T)


def loo_error(X: np.ndarray, y: np.ndarray, lam: float, fit_intercept: bool = True) -> float:
    """Closed-form leave-one-out error: r_i/(1-h_ii), no refits.

    This is the reason ridge can tune λ at the cost of one fit, not n fits.
    """
    H = hat_matrix(X, lam, fit_intercept)
    Xa, D = _augmented(X, fit_intercept)
    coef = np.linalg.solve(Xa.T @ Xa + lam * D, Xa.T @ y)
    resid = y - Xa @ coef
    return float(np.mean((resid / (1.0 - np.diag(H))) ** 2))


def nuclear_norm(
    M: np.ndarray, lam: float = 1.0, iters: int = 300, tol: float = 1e-8
) -> np.ndarray:
    """Matrix completion by singular-value soft thresholding (soft-impute).

    Solves min ½||P_Ω(A) - P_Ω(X)||² + λ||X||_*. Missing entries are unconstrained,
    which is exactly what makes the nuclear norm a low-rank prior rather than a
    penalty on the observed values.
    """
    obs = ~np.isnan(M)
    X = np.where(obs, M, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_new = np.maximum(s - lam, 0.0)
        X_new = (U * s_new) @ Vt
        X_new[obs] = M[obs]
        if np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-12) < tol:
            X = X_new
            break
        X = X_new
    return X


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 120, 6
    X = rng.normal(size=(n, d))
    beta_true = np.array([3.0, -2.0, 1.0, 0.0, 0.0, 0.5])
    y = X @ beta_true + rng.normal(0, 0.5, size=n)
    lam = 2.0

    # 1. The two closed forms agree.
    b1 = ridge(X, y, lam, "closed_form")["coef"]
    b2 = ridge(X, y, lam, "svd")["coef"]
    b3 = ridge(X, y, lam, "generalized_ridge")["coef"]
    assert np.allclose(b1, b2, atol=1e-10) and np.allclose(b1, b3, atol=1e-10)
    print(f"closed_form == svd == generalized ({np.abs(b1 - b2).max():.1e})")

    # 2. Shrinkage is explicit in the SVD form: each singular direction is scaled
    #    by s²/(s²+λ). At λ=0 it is OLS; as λ→∞ every coefficient goes to zero.
    b_ols = np.linalg.lstsq(X - X.mean(0), y - y.mean(), rcond=None)[0]
    b_zero = ridge(X, y, 0.0, "svd")["coef"]
    b_huge = ridge(X, y, 1e8, "svd")["coef"]
    assert np.allclose(b_zero, b_ols, atol=1e-9)
    assert np.linalg.norm(b_huge) < 1e-4
    U, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    shrink = s * s / (s * s + lam)
    assert np.all(shrink < 1.0) and shrink[0] > shrink[-1]
    print(f"svd shrinkage: {np.round(shrink, 3)} (λ={lam}), λ→∞ gives ||β||→0")

    # 3. λ trades variance for bias: on collinear features OLS has huge variance,
    #    and a small penalty wins on average over 200 noise draws.
    nc = 60
    common = rng.normal(size=(nc, 1))
    Xcol = common + 0.1 * rng.normal(size=(nc, d))
    err_ols, err_ridge = [], []
    for _ in range(200):
        yb = Xcol @ beta_true + rng.normal(0, 1.0, size=nc)
        err_ols.append(np.linalg.norm(np.linalg.lstsq(Xcol, yb, rcond=None)[0] - beta_true))
        err_ridge.append(np.linalg.norm(ridge(Xcol, yb, 0.3)["coef"] - beta_true))
    assert np.mean(err_ridge) < np.mean(err_ols)
    print(f"mean ||β̂-β||: ols {np.mean(err_ols):.3f}, ridge {np.mean(err_ridge):.3f}")

    # 4. LOOCV matches brute-force leave-one-out and picks a sensible λ.
    Xs, ys = X[:30], y[:30]
    grid = np.logspace(-2, 3, 25)
    for ll in (0.1, 1.0, 10.0):
        brute = []
        for i in range(len(ys)):
            keep = np.arange(len(ys)) != i
            b = ridge(Xs[keep], ys[keep], ll, "closed_form")["coef"]
            b0 = ridge(Xs[keep], ys[keep], ll, "closed_form")["intercept"]
            brute.append((ys[i] - (Xs[i] @ b + b0)) ** 2)
        assert abs(loo_error(Xs, ys, ll) - np.mean(brute)) < 1e-8 * max(1.0, np.mean(brute))
    picked = ridge(Xs, ys, variant="loo_cv", lams=grid)["lam"]
    assert grid[0] <= picked <= grid[-1]
    print(f"loo_cv: closed form == brute force; picked λ = {picked:.3f}")

    # 5. Group ridge with singleton groups is plain ridge; with a pair it shrinks
    #    the pair toward each other (both coefficients get the same penalty).
    single = [[i] for i in range(d)]
    bg = ridge(X, y, lam, "group_ridge", groups=single)["coef"]
    assert np.allclose(bg, b1, atol=1e-10)
    pair = [[0, 1]] + [[i] for i in range(2, d)]
    lams = [10.0] + [1.0] * (d - 2)
    bpair = ridge(X, y, variant="group_ridge", groups=pair, penalty=lams)["coef"]
    manual = np.linalg.solve(
        (X - X.mean(0)).T @ (X - X.mean(0)) + np.diag([10.0, 10.0] + [1.0] * (d - 2)),
        (X - X.mean(0)).T @ (y - y.mean()),
    )
    assert np.allclose(bpair, manual)
    print("group_ridge: singleton groups == ridge, block penalty matches manual solve")

    # 6. Nuclear norm recovers a low-rank matrix from half its entries. The
    #    matrix is built with orthogonal factors so the conditioning is explicit:
    #    two singular values, 10 and 5.
    r2 = np.random.default_rng(7)
    m, ncol = 20, 15
    Uo, _ = np.linalg.qr(r2.normal(size=(m, 2)))
    Vo, _ = np.linalg.qr(r2.normal(size=(ncol, 2)))
    A = 10.0 * Uo[:, [0]] @ Vo[:, [0]].T + 5.0 * Uo[:, [1]] @ Vo[:, [1]].T
    mask = r2.random(A.shape) < 0.5
    mask[:, 0] = True                     # guarantee every row/column is seen
    mask[0, :] = True
    M = np.where(mask, A, np.nan)
    Xhat = nuclear_norm(M, lam=0.05, iters=2000)
    err = np.linalg.norm((Xhat - A)[~mask]) / np.linalg.norm(A[~mask])
    mean_fill = np.where(mask, A, np.nanmean(M))
    err_mean = np.linalg.norm((mean_fill - A)[~mask]) / np.linalg.norm(A[~mask])
    assert err < 0.15 and err < 0.2 * err_mean, (err, err_mean)
    print(f"nuclear_norm: held-out error {err:.3f} vs mean-fill {err_mean:.3f}")
    print("ridge demo passed")


if __name__ == "__main__":
    main()
