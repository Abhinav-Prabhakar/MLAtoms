"""OLS — 7 ways to solve the same least-squares problem.

Chapter:   02-linear-models
Companion: ols.md
Tier 1 · 7 variants: normal_equations, qr, svd_pinv, cholesky, recursive,
with_intercept, standard_errors

The model is always y = Xβ + ε. The variants are numerical paths to β̂ and the two
things people actually need afterwards: an intercept that is not biased by the
data's offset, and standard errors. Everything returns a dict so the extra output
is optional.

    normal_equations  solve(X^T X, X^T y)      — fastest, worst conditioning
    qr                R β = Q^T y              — stable, O(nd^2)
    svd_pinv          V S^+ U^T y              — stable, gives rank/pseudoinverse
    cholesky          X^T X = LL^T, two solves — stable only if X is well scaled
    recursive         rank-1 updates (RLS)     — streaming, same answer
    with_intercept    centre X and y, then fit — intercept unpenalised later
    standard_errors   QR fit + σ̂²(X^T X)^-1   — inference, not just prediction
"""

from __future__ import annotations

import numpy as np


def _fit_intercept(X: np.ndarray, y: np.ndarray):
    xm = X.mean(axis=0)
    ym = y.mean()
    beta = _solve(X - xm, y - ym, "qr")
    return beta, float(ym - xm @ beta), X - xm, y - ym


def _solve(X: np.ndarray, y: np.ndarray, variant: str) -> np.ndarray:
    if variant == "normal_equations":
        return np.linalg.solve(X.T @ X, X.T @ y)
    if variant == "cholesky":
        G = X.T @ X
        L = np.linalg.cholesky(G)
        z = np.linalg.solve(L, X.T @ y)
        return np.linalg.solve(L.T, z)
    if variant == "svd_pinv":
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        tol = max(X.shape) * np.finfo(float).eps * s[0]
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        return Vt.T @ (s_inv * (U.T @ y))
    if variant == "qr":
        Q, R = np.linalg.qr(X)
        return np.linalg.solve(R, Q.T @ y)
    raise ValueError(f"unknown solve variant {variant!r}")


def ols(
    X: np.ndarray, y: np.ndarray, variant: str = "qr", *, se: bool = False
) -> dict:
    """Ordinary least squares.

    Returns {"coef": (d,), "intercept": float or None, "se": (d,) or None,
             "cov": (d, d) or None, "resid": (n,)}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape

    if variant == "recursive":
        coef = recursive_least_squares(X, y)
        intercept = None
        resid = y - X @ coef
    elif variant == "with_intercept":
        coef, intercept, Xc, yc = _fit_intercept(X, y)
        resid = y - (X @ coef + intercept)
    elif variant in ("normal_equations", "qr", "svd_pinv", "cholesky", "standard_errors"):
        path = "qr" if variant == "standard_errors" else variant
        coef = _solve(X, y, path)
        intercept = None
        resid = y - X @ coef
        if variant == "standard_errors" or se:
            rss = float(resid @ resid)
            sigma2 = rss / max(n - d, 1)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            return {
                "coef": coef,
                "intercept": intercept,
                "se": np.sqrt(np.diag(cov)),
                "cov": cov,
                "resid": resid,
            }
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {"coef": coef, "intercept": intercept, "se": None, "cov": None, "resid": resid}


def recursive_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Recursive least squares: the exact OLS estimate after each row, updated in
    O(d^2) per observation with the Sherman-Morrison identity.

    P is the inverse of the information matrix; it shrinks as data arrives, which
    is why later observations move the estimate less.
    """
    n, d = X.shape
    warm = min(max(d, 1), n)
    beta = np.linalg.lstsq(X[:warm], y[:warm], rcond=None)[0]
    P = np.linalg.pinv(X[:warm].T @ X[:warm])
    for xi, yi in zip(X[warm:], y[warm:], strict=True):
        Px = P @ xi
        denom = 1.0 + xi @ Px
        gain = Px / denom
        beta = beta + gain * (yi - xi @ beta)
        P = P - np.outer(gain, Px)
    return beta


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    beta_true = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ beta_true + rng.normal(0, 0.3, size=n)

    # 1. All four direct solvers agree to machine precision on a benign problem.
    fits = {v: ols(X, y, v)["coef"] for v in
            ("normal_equations", "qr", "svd_pinv", "cholesky")}
    ref = fits["svd_pinv"]
    for v, b in fits.items():
        assert np.allclose(b, ref, atol=1e-10), (v, np.abs(b - ref).max())
    print("4 direct solvers agree to 1e-10")

    # 2. Recursive least squares reproduces the batch fit exactly, one row at a time.
    b_rls = ols(X, y, "recursive")["coef"]
    assert np.allclose(b_rls, ref, atol=1e-9), np.abs(b_rls - ref).max()
    print(f"recursive == batch (max diff {np.abs(b_rls - ref).max():.1e})")

    # 3. Conditioning is what separates them: on a Hilbert-like design the normal
    #    equations lose digits that QR and SVD keep.
    t = np.linspace(0, 1, 60)
    V = np.vander(t, 12, increasing=True)         # polynomial regression
    b_true = rng.normal(size=12)
    yv = V @ b_true
    err_ne = np.linalg.norm(ols(V, yv, "normal_equations")["coef"] - b_true)
    err_qr = np.linalg.norm(ols(V, yv, "qr")["coef"] - b_true)
    err_svd = np.linalg.norm(ols(V, yv, "svd_pinv")["coef"] - b_true)
    assert err_qr < err_ne and err_svd < err_ne, (err_ne, err_qr, err_svd)
    print(f"cond(V) = {np.linalg.cond(V):.1e}: normal eq err {err_ne:.1e}, "
          f"qr {err_qr:.1e}, svd {err_svd:.1e}")

    # 4. with_intercept recovers the offset; fitting without one absorbs it into
    #    the slope whenever X has a non-zero mean.
    x1 = rng.normal(10.0, 1.0, size=400)
    y1 = 5.0 + 2.0 * x1 + rng.normal(0, 0.2, size=400)
    fit = ols(x1.reshape(-1, 1), y1, "with_intercept")
    assert abs(fit["intercept"] - 5.0) < 0.2, fit["intercept"]
    assert abs(fit["coef"][0] - 2.0) < 0.02, fit["coef"]
    no_int = ols(x1.reshape(-1, 1), y1, "qr")["coef"][0]
    assert abs(no_int - 2.0) > 0.3, no_int        # biased by y's mean
    print(f"intercept {fit['intercept']:.3f} (truth 5), "
          f"slope {fit['coef'][0]:.3f}; no-intercept slope {no_int:.3f}")

    # 5. Standard errors match σ̂²(X^T X)^-1 and shrink as 1/sqrt(n).
    fit = ols(X, y, "standard_errors")
    sigma2 = float(fit["resid"] @ fit["resid"]) / (n - d)
    se_manual = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    assert np.allclose(fit["se"], se_manual, rtol=1e-8)
    se_half = ols(X[:50], y[:50], "standard_errors")["se"]
    assert np.all(se_half > fit["se"])
    print(f"standard errors ok (mean {fit['se'].mean():.4f}, "
          f"n=50 gives {se_half.mean():.4f})")

    # 6. The residual is orthogonal to every column — the first-order condition.
    assert np.allclose(X.T @ fit["resid"], 0.0, atol=1e-10)
    print("normal equations: X^T r = 0")
    print("ols demo passed")


if __name__ == "__main__":
    main()
