"""Lasso — 6 solvers and structured variants of the L1 problem.

Chapter:   02-linear-models
Companion: lasso.md
Tier 1 · 6 variants: coordinate_descent, lars, homotopy, path, positive_lasso,
group_lasso

The lasso is min ½||y - Xβ||² + λ||β||₁. It has no closed form, so everything here
is an algorithm: coordinate descent (the workhorse), LARS (the exact piecewise
linear path), homotopy (warm-started continuation), the path view, a sign
constraint, and the group penalty.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(z, t):
    """The lasso prox: sign(z)·max(|z| - t, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def _centre(X, y, fit_intercept=True):
    if not fit_intercept:
        return X, y, np.zeros(X.shape[1]), 0.0
    xm = X.mean(axis=0)
    ym = y.mean()
    return X - xm, y - ym, xm, ym


def coordinate_descent(
    X: np.ndarray, y: np.ndarray, lam: float, *, positive: bool = False,
    fit_intercept: bool = True, iters: int = 5000, tol: float = 1e-10,
    beta0: np.ndarray | None = None,
) -> dict:
    """Cyclic coordinate descent with soft-thresholding updates.

    Each coordinate update is exact, so the objective decreases monotonically and
    the KKT conditions hold at convergence: |x_jᵀr| ≤ λ, with equality on the
    support.
    """
    Xc, yc, xm, ym = _centre(X, y, fit_intercept)
    n, d = Xc.shape
    beta = np.zeros(d) if beta0 is None else np.asarray(beta0, float).copy()
    sq = (Xc * Xc).sum(axis=0)
    sq = np.where(sq == 0, 1.0, sq)
    r = yc - Xc @ beta
    for _ in range(iters):
        delta = 0.0
        for j in range(d):
            old = beta[j]
            rho = Xc[:, j] @ r + sq[j] * old          # partial residual correlation
            new = soft_threshold(rho, lam) / sq[j]
            if positive:
                new = max(new, 0.0)
            if new != old:
                beta[j] = new
                r -= Xc[:, j] * (new - old)
                delta = max(delta, abs(new - old))
        if delta < tol:
            break
    intercept = float(ym - xm @ beta) if fit_intercept else None
    return {"coef": beta, "intercept": intercept, "resid": yc - Xc @ beta}


def lars(
    X: np.ndarray, y: np.ndarray, lam: float, *, iters: int = 500,
    fit_intercept: bool = True,
) -> dict:
    """LARS-lasso: follow the exact piecewise-linear path from λ_max down to
    `lam`. Along each segment every active correlation falls at unit rate, so the
    step length that reaches the next event is closed-form; the path bends when a
    new variable reaches the corridor, or an active coefficient crosses zero (the
    lasso modification).

    The entering variable is carried explicitly (`pending`) rather than detected
    with a tolerance, which is what keeps the path from stalling one ulp short of
    a tie.
    """
    Xc, yc, xm, ym = _centre(X, y, fit_intercept)
    d = Xc.shape[1]
    beta = np.zeros(d)
    knots = []
    pending: int | None = None
    dropped: set[int] = set()
    for _ in range(iters):
        r = yc - Xc @ beta
        c = Xc.T @ r
        cmax = float(np.abs(c).max())
        if cmax <= lam + 1e-12:
            break
        knots.append(cmax)

        top = int(np.argmax(np.abs(c)))
        candidates = {j for j in range(d) if abs(beta[j]) > 1e-12} | {top}
        if pending is not None:
            candidates.add(pending)
        A = np.array(sorted(j for j in candidates if j not in dropped), dtype=int)
        if A.size == 0:
            break

        s = np.where(np.sign(c[A]) == 0.0, 1.0, np.sign(c[A]))
        w = np.linalg.solve(Xc[:, A].T @ Xc[:, A] + 1e-12 * np.eye(A.size), s)
        # w = (X_A^T X_A)^-1 s_A makes every active correlation fall at unit rate:
        # dc_k/dt = -x_k^T X_A w = -s_k, so |c_k| drops by exactly dt along the
        # segment and cmax(t) = cmax - t. No column standardisation is needed for this
        # identity to hold; scaling only improves the conditioning of the solve.
        a = Xc.T @ (Xc[:, A] @ w)          # dc_k/dt = -a_k for every column

        gamma, kind, arg = cmax - lam, "lambda", None      # stop exactly at λ
        for k in range(d):
            if k in A:
                continue
            # c_k - t*a_k meets +cmax(t) = cmax - t, or -cmax(t) = -(cmax - t).
            g_plus = (cmax - c[k]) / (1.0 - a[k]) if 1.0 - a[k] > 1e-12 else np.inf
            g_minus = (cmax + c[k]) / (1.0 + a[k]) if 1.0 + a[k] > 1e-12 else np.inf
            g = min(g_plus, g_minus)
            if g < gamma - 1e-14:
                gamma, kind, arg = g, "join", k
        for pos, j in enumerate(A):
            if beta[j] * w[pos] < 0:                    # coefficient crosses zero
                g = -beta[j] / w[pos]
                if g < gamma - 1e-14:
                    gamma, kind, arg = g, "drop", int(j)

        beta[A] += gamma * w
        beta[np.abs(beta) < 1e-14] = 0.0
        if kind == "drop":
            dropped, pending = {arg}, None
        elif kind == "join":
            dropped, pending = set(), arg
        else:
            dropped, pending = set(), None

    intercept = float(ym - xm @ beta) if fit_intercept else None
    return {"coef": beta, "intercept": intercept, "knots": np.array(knots)}


def path(
    X: np.ndarray, y: np.ndarray, lams: np.ndarray | None = None, *, n_lams: int = 20,
    positive: bool = False, fit_intercept: bool = True,
) -> dict:
    """The whole regularisation path with warm starts: β(λ_k) seeded from
    β(λ_{k+1}). Total cost is close to a single fit, which is why glmnet is fast."""
    Xc, _, _, _ = _centre(X, y, fit_intercept)
    if lams is None:
        lam_max = float(np.abs(Xc.T @ (y - y.mean())).max()) if fit_intercept else \
            float(np.abs(Xc.T @ y).max())
        lams = np.geomspace(lam_max, lam_max * 1e-3, n_lams)
    lams = np.asarray(lams, float)
    coefs = np.zeros((lams.size, X.shape[1]))
    beta = None
    for i, lam in enumerate(lams):
        fit = coordinate_descent(X, y, lam, positive=positive,
                                 fit_intercept=fit_intercept, beta0=beta)
        beta = fit["coef"]
        coefs[i] = beta
    return {"lams": lams, "coefs": coefs}


def homotopy(
    X: np.ndarray, y: np.ndarray, lam: float, *, n_steps: int = 40,
    fit_intercept: bool = True,
) -> dict:
    """Continuation: solve on a geometric grid from λ_max to `lam`, warm-starting
    each solve. Records where the support changes — the discrete events of the
    exact path."""
    Xc, _, _, _ = _centre(X, y, fit_intercept)
    lam_max = float(np.abs(Xc.T @ (y - y.mean())).max()) if fit_intercept else \
        float(np.abs(Xc.T @ y).max())
    grid = np.geomspace(max(lam_max, lam), lam, n_steps)
    coefs, supports = [], []
    beta = None
    for g in grid:
        fit = coordinate_descent(X, y, g, fit_intercept=fit_intercept, beta0=beta)
        beta = fit["coef"]
        coefs.append(beta.copy())
        supports.append(tuple(np.flatnonzero(np.abs(beta) > 1e-10)))
    return {"coef": beta, "lams": grid, "coefs": np.array(coefs), "supports": supports}


def group_lasso(
    X: np.ndarray, y: np.ndarray, lam: float, groups: list[np.ndarray], *,
    iters: int = 3000, tol: float = 1e-10, fit_intercept: bool = True,
) -> dict:
    """Group lasso by proximal gradient: the prox of λ Σ_g ||β_g||₂ is a block
    soft-threshold, which sets whole groups to zero at once."""
    Xc, yc, xm, ym = _centre(X, y, fit_intercept)
    n, d = Xc.shape
    L = float(np.linalg.eigvalsh(Xc.T @ Xc).max()) + 1e-12
    beta = np.zeros(d)
    step = 1.0 / L
    for _ in range(iters):
        grad = Xc.T @ (Xc @ beta - yc)
        z = beta - step * grad
        new = z.copy()
        for g in groups:
            g = np.asarray(g)
            nrm = np.linalg.norm(z[g])
            new[g] = z[g] * max(1.0 - step * lam / nrm, 0.0) if nrm > 0 else 0.0
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
    lam_max = float(np.abs(Xc.T @ yc).max())     # smallest λ with β = 0 exactly

    # 1. KKT conditions at the CD solution: |x_j^T r| = λ on the support and ≤ λ
    #    off it. At λ > λ_max the empty model is already optimal.
    lam = 0.25 * lam_max
    fit = coordinate_descent(X, y, lam)
    corr = Xc.T @ (yc - Xc @ fit["coef"])
    active = np.abs(fit["coef"]) > 1e-8
    assert np.allclose(np.abs(corr[active]), lam, atol=1e-6), corr[active]
    assert np.abs(corr[~active]).max() <= lam + 1e-6
    assert np.allclose(coordinate_descent(X, y, 1.01 * lam_max)["coef"], 0.0)
    print(f"coordinate descent: {active.sum()} active at λ = 0.25·λ_max, KKT holds; "
          f"λ > λ_max = {lam_max:.1f} gives β = 0")

    # 2. LARS reaches exactly the same solution, but along the exact path.
    errs = []
    for frac in (0.6, 0.4, 0.25, 0.1, 0.02):
        lam = frac * lam_max
        cd = coordinate_descent(X, y, lam)["coef"]
        la = lars(X, y, lam)["coef"]
        errs.append(np.abs(cd - la).max())
        assert np.allclose(cd, la, atol=1e-6), (frac, errs[-1])
    print(f"lars == coordinate descent at 5 values down to 0.02·λ_max "
          f"(max diff {max(errs):.1e})")

    # 3. The path solver equals cold solves at every λ, at a fraction of the cost.
    grid = np.geomspace(0.6, 0.02, 12) * lam_max
    P = path(X, y, grid)
    for i, lam in enumerate(grid):
        cold = coordinate_descent(X, y, lam)["coef"]
        assert np.allclose(P["coefs"][i], cold, atol=1e-8)
    nnz = (np.abs(P["coefs"]) > 1e-8).sum(1)
    assert nnz[0] <= nnz[-1]                       # more variables as λ shrinks
    print(f"path: {grid.size} λ values, nnz grows {nnz[0]} -> {nnz[-1]}")

    # 4. Homotopy records where the support changes; the number of active
    #    variables is monotone along a decreasing λ sequence.
    H = homotopy(X, y, 0.02 * lam_max, n_steps=25)
    sizes = np.array([len(s) for s in H["supports"]])
    assert sizes[0] <= sizes[-1] and sizes.max() >= 3
    assert np.allclose(H["coef"], coordinate_descent(X, y, 0.02 * lam_max)["coef"], atol=1e-7)
    print(f"homotopy: support size {sizes[0]} -> {sizes[-1]} over {sizes.size} steps")

    # 5. Positive lasso: the constraint bites when a feature is genuinely
    #    negative. β₃ = -1.5 is clamped to exactly 0 and the rest of the fit
    #    absorbs the difference — the constrained solution is a different point,
    #    not the unconstrained one with a coordinate deleted.
    b_pos = np.array([3.0, 2.0, 1.0, -1.5, 0, 0, 0, 0])
    yp = X @ b_pos + rng.normal(0, 0.5, size=n)
    lam_p = 0.03 * float(np.abs((X - X.mean(0)).T @ (yp - yp.mean())).max())
    pos = coordinate_descent(X, yp, lam_p, positive=True)["coef"]
    unpos = coordinate_descent(X, yp, lam_p)["coef"]
    assert np.all(pos >= 0) and pos[3] == 0.0, pos
    assert unpos[3] < -1.0, unpos
    assert np.all(pos[:3] > 0.5), pos
    print(f"positive_lasso: β₃ {unpos[3]:.3f} -> {pos[3]:.1e}; "
          f"β[:3] = {np.round(pos[:3], 2)} (truth 3, 2, 1)")

    # 6. Group lasso zeroes whole groups: the prox is an L2 shrink per group, so a
    #    group is either fully in or fully out, never partly.
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
    gl = group_lasso(X, y, lam=0.12 * lam_max, groups=groups)["coef"]
    group_norms = np.array([np.linalg.norm(gl[g]) for g in groups])
    assert (group_norms == 0).sum() >= 2, group_norms
    assert np.all(group_norms[:2] > 0), group_norms
    print(f"group_lasso: group norms {np.round(group_norms, 2)} (whole groups are exactly 0)")
    print("lasso demo passed")


if __name__ == "__main__":
    main()
