"""Support vector machines — the max-margin classifier, from the primal to SMO.

Chapter:   03-kernels-and-instance
Companion: svm.md
Tier 1 · 12 variants: hard_margin, soft_margin, dual, smo, linear, polynomial,
rbf, sigmoid, nu_svm, class_weight, ovo, ovr

The primal asks for the separating hyperplane with the widest street:

    min  ½‖w‖² + C Σ ξ_i    s.t.  y_i (wᵀφ(x_i) + b) ≥ 1 − ξ_i,  ξ_i ≥ 0

Its dual asks for weights on the training points, and every point with a
non-zero weight is a support vector — the model *is* a subset of the data:

    max  Σ α_i − ½ ΣΣ α_i α_j y_i y_j K(x_i, x_j)
    s.t. 0 ≤ α_i ≤ C,  Σ α_i y_i = 0

    hard_margin   C = ∞            only separable data, margin is the model
    soft_margin   C finite         slack absorbs the impossible points
    linear / polynomial / rbf / sigmoid    the kernel K
    dual          pair coordinate ascent, first-order working set
    smo           the same, second-order working set (Fan/Chen/Lin)
    nu_svm        parameterise by the support-vector fraction instead of C
    class_weight  per-class C, so the minority gets a wider street
    ovo / ovr     multiclass by k(k−1)/2 duels or k one-against-rest machines

The whole file is one dual solver plus a kernel table plus two multiclass
wrappers. Everything else is a parameter choice.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["hard_margin", "soft_margin", "dual", "smo", "linear", "polynomial",
     "rbf", "sigmoid", "nu_svm", "class_weight", "ovo", "ovr"]
)

KERNELS = frozenset(["linear", "polynomial", "rbf", "sigmoid"])
STRATEGIES = frozenset(["ovo", "ovr"])
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def _sqdist(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, (n, m), without materialising (n, m, d)."""
    d2 = (X * X).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2.0 * X @ Z.T
    return np.maximum(d2, 0.0)


def kernel_matrix(X: np.ndarray, Z: np.ndarray | None = None, *,
                  kind: str = "rbf", degree: int = 3,
                  gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    """Gram matrix `K[i, j] = k(X[i], Z[j])`, shape `(n, m)`.

    `gamma` defaults to `1/d`: on standardised data that puts the typical
    squared distance `2d` at `exp(−2)`, which is neither a step function nor a
    straight line. It is the one hyperparameter that always needs tuning.
    """
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}")
    Z = X if Z is None else Z
    gamma = 1.0 / X.shape[1] if gamma is None else float(gamma)
    if kind == "linear":
        return X @ Z.T
    if kind == "polynomial":
        return (gamma * (X @ Z.T) + coef0) ** degree
    if kind == "rbf":
        return np.exp(-gamma * _sqdist(X, Z))
    return np.tanh(gamma * (X @ Z.T) + coef0)


# --------------------------------------------------------------------------- #
# The dual solver
# --------------------------------------------------------------------------- #
def _solve_dual(K: np.ndarray, y: np.ndarray, C: np.ndarray, *,
                second_order: bool = True, tol: float = 1e-6,
                max_iter: int = 40000) -> tuple[np.ndarray, float, int]:
    """Maximise the SVM dual by pairwise coordinate ascent.

    `K` is the kernel Gram matrix `(n, n)`; `y` is `(n,)` of ±1; `C` is the
    per-sample upper bound on `α` (`np.inf` for the hard margin).

    Returns `(alpha, b, iterations)`. The KKT conditions this converges to are

        α_i = 0     →  y_i f_i ≥ 1
        0 < α_i < C →  y_i f_i = 1
        α_i = C     →  y_i f_i ≤ 1
    """
    n = y.size
    alpha = np.zeros(n)
    f = np.zeros(n)                       # f = K (α ⊙ y), without b
    G = -np.ones(n)                       # gradient of ½αᵀQα − Σα at α = 0
    diag = np.diag(K)
    it = 0
    for _ in range(max_iter):
        up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
        low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
        if not up.any() or not low.any():
            break
        viol = -y * G
        gi = np.where(up, viol, -np.inf)
        i = int(np.argmax(gi))
        if gi[i] - np.where(low, viol, np.inf).min() < tol:
            break
        if second_order:                  # Fan/Chen/Lin: minimise −b²/a
            # only pairs with a feasible positive step: `viol[j] < viol[i]`
            ok = low & (viol < gi[i])
            if not ok.any():
                break
            a_ij = np.maximum(diag[i] + diag - 2.0 * K[i], EPS)
            b_ij = viol - gi[i]
            crit = np.where(ok, -(b_ij * b_ij) / a_ij, np.inf)
        else:                             # first-order maximal violating pair
            crit = np.where(low, viol, np.inf)
        j = int(np.argmin(crit))
        a = max(diag[i] + diag[j] - 2.0 * K[i, j], EPS)
        t = ((y[i] - y[j]) - (f[i] - f[j])) / a

        # clip so both coordinates stay inside their boxes
        lo, hi = -np.inf, np.inf
        for m, sign in ((i, y[i]), (j, -y[j])):
            room = C[m] - alpha[m]
            if sign > 0:
                lo = max(lo, -alpha[m])
                hi = min(hi, room)
            else:
                lo = max(lo, -room)
                hi = min(hi, alpha[m])
        t = min(max(t, lo), hi)
        if abs(t) < EPS:
            break

        alpha[i] += y[i] * t
        alpha[j] -= y[j] * t
        # f = Σ_l α_l y_l K_l, so Δf = Δα_i y_i K_i + Δα_j y_j K_j = t(K_i − K_j)
        f += t * (K[i] - K[j])
        G = y * f - 1.0
        it += 1

    free = (alpha > 1e-8) & (alpha < C - 1e-8)
    if free.any():
        b = float(np.mean(y[free] - f[free]))
    else:                                  # degenerate: clip between the bounds
        low_side = (y > 0) & (alpha < C)
        b = float(np.mean(1.0 - f[low_side])) if low_side.any() else 0.0
    return alpha, b, it


def _solve_nu_dual(K: np.ndarray, y: np.ndarray, nu: float, *,
                  tol: float = 1e-6, max_iter: int = 20000
                  ) -> tuple[np.ndarray, float, float, int]:
    """Minimise `½αᵀQα` over the ν-SVM dual.

        min  ½ αᵀQα    s.t.  0 ≤ α_i ≤ 1/n,  Σ α_i y_i = 0,  Σ α_i = ν

    `Q = (y yᵀ) ⊙ K`. The two equalities force `Σ_{y=+1} α = Σ_{y=−1} α = ν/2`,
    so the feasible directions that keep both invariants are *same-class* pair
    moves `(α_i, α_j) → (α_i + t, α_j − t)` with `y_i = y_j`. That is the whole
    solver: unlike the C-SVM, the working set never crosses classes.

    Returns `(alpha, b, rho, iterations)` with the KKT condition

        α_i = 0     →  y_i f_i ≥ ρ
        0 < α_i < U →  y_i f_i = ρ
        α_i = U     →  y_i f_i ≤ ρ

    so `ρ` replaces the C-SVM's constant margin of 1.
    """
    n = y.size
    U = 1.0 / n
    alpha = np.zeros(n)
    for sign in (1.0, -1.0):
        idx = np.flatnonzero(y == sign)
        if idx.size == 0:
            raise ValueError("nu_svm needs both classes")
        if 0.5 * nu > U * idx.size:
            raise ValueError("nu too large for the class balance")
        alpha[idx] = 0.5 * nu / idx.size        # a feasible interior start

    Q = np.outer(y, y) * K
    g = Q @ alpha                              # g_i = y_i · Σ_l α_l y_l K_il
    blocks = [np.flatnonzero(y == s) for s in (1.0, -1.0)]
    it = 0
    for _ in range(max_iter):
        # best same-class pair: move weight from the largest gradient to the
        # smallest, which is the steepest feasible descent direction
        best, pair = 0.0, None
        for idx in blocks:
            up = idx[alpha[idx] < U - EPS]
            dn = idx[alpha[idx] > EPS]
            if up.size == 0 or dn.size == 0:
                continue
            i = int(up[np.argmin(g[up])])
            j = int(dn[np.argmax(g[dn])])
            if g[j] - g[i] > best:
                best, pair = g[j] - g[i], (i, j)
        if pair is None or best <= tol:
            break
        it += 1
        i, j = pair
        a = max(K[i, i] + K[j, j] - 2.0 * K[i, j], EPS)
        t = min(best / a, U - alpha[i], alpha[j])
        if t < EPS:
            break
        alpha[i] += t
        alpha[j] -= t
        g += t * (Q[i] - Q[j])

    free_p = (y > 0) & (alpha > 1e-8) & (alpha < U - 1e-8)
    free_n = (y < 0) & (alpha > 1e-8) & (alpha < U - 1e-8)
    if free_p.any() and free_n.any():
        gp, gn = float(g[free_p].mean()), float(g[free_n].mean())
        b, rho = 0.5 * (gn - gp), 0.5 * (gn + gp)
    else:                                     # no bracketing pair: fall back
        b, rho = 0.0, 1.0
    return alpha, b, rho, it


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
class SVM:
    """Maximum-margin classifier.

    Parameters
    ----------
    variant : str
        One of VARIANTS. A kernel name (`linear`, `rbf`, …) selects that kernel
        and otherwise behaves like `soft_margin`.
    C : float
        Box bound of the dual. Larger is a harder margin; `hard_margin` ignores
        it and uses an unbounded box.
    kernel : str
        One of KERNELS; overridden when `variant` is itself a kernel name.
    degree, gamma, coef0 : float
        Kernel parameters. `gamma=None` means `1/d`.
    nu : float
        Target support-vector fraction for `nu_svm`, in `(0, 1)`.
    class_weight : str or dict
        `"balanced"` scales `C` by `n / (k · count)`; a dict maps class → weight.
    tol, max_iter : float, int
        KKT tolerance and iteration cap of the dual solver.

    Notes
    -----
    `fit` stores `support_`, `dual_coef_` and `intercept_`. For the multiclass
    variants those are lists, one entry per binary machine; `predict` handles
    the voting.
    """

    def __init__(self, variant: str = "soft_margin", C: float = 1.0, *,
                 kernel: str = "rbf", degree: int = 3, gamma: float | None = None,
                 coef0: float = 1.0, nu: float = 0.5, class_weight=None,
                 tol: float = 1e-6, max_iter: int = 40000):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}")
        if variant == "nu_svm" and not 0.0 < nu < 1.0:
            raise ValueError("nu must be in (0, 1)")
        self.variant = variant
        self.C = float(C)
        self.kernel = variant if variant in KERNELS else kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.nu = float(nu)
        self.class_weight = class_weight
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    # ------------------------------------------------------------------ #
    def _gram(self, X: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        return kernel_matrix(X, Z, kind=self.kernel, degree=self.degree,
                             gamma=self.gamma, coef0=self.coef0)

    def _bounds(self, n: int, weight: np.ndarray | None) -> np.ndarray:
        """Per-sample box bound: `C` scaled by the class weight."""
        if self.variant == "hard_margin":
            return np.full(n, np.inf)
        return self.C * (np.ones(n) if weight is None else weight)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVM:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("need at least two classes")
        self.X_ = X
        weights = None
        if isinstance(self.class_weight, dict):
            weights = {c: float(self.class_weight.get(c, 1.0))
                       for c in self.classes_}
        elif self.class_weight == "balanced":
            k = self.classes_.size
            weights = {c: y.size / (k * float(np.sum(y == c)))
                       for c in self.classes_}

        if self.variant in STRATEGIES and self.classes_.size > 2:
            self._fit_multiclass(X, y, weights)
            return self

        signed = np.where(y == self.classes_[-1], 1.0, -1.0)
        K = self._gram(X)
        weight = None
        if weights:
            weight = np.where(signed > 0, weights[self.classes_[-1]],
                              weights[self.classes_[0]])
        C = self._bounds(y.size, weight)
        if self.variant == "nu_svm":
            self.alpha_, self.intercept_, _ = self._fit_nu(K, signed, C)
        else:
            so = self.variant != "dual"
            self.alpha_, self.intercept_, self.n_iter_ = _solve_dual(
                K, signed, C, second_order=so, tol=self.tol,
                max_iter=self.max_iter)
        self.y_signed_ = signed
        self.support_ = np.flatnonzero(self.alpha_ > 1e-8)
        self.dual_coef_ = self.alpha_[self.support_] * signed[self.support_]
        return self

    def _fit_nu(self, K: np.ndarray, y: np.ndarray,
                C: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Solve the ν dual exactly; `C` is ignored (the box is `1/n`)."""
        alpha, b, rho, it = _solve_nu_dual(K, y, self.nu, tol=self.tol,
                                           max_iter=self.max_iter)
        self.rho_ = rho
        self.n_iter_ = it
        return alpha, b, rho

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray,
                        weights: dict | None) -> None:
        K = self._gram(X)
        self.machines_ = []
        if self.variant == "ovo":
            pairs = [(a, b) for a in range(self.classes_.size)
                     for b in range(a + 1, self.classes_.size)]
            for a, b in pairs:
                idx = np.flatnonzero((y == self.classes_[a]) | (y == self.classes_[b]))
                sub = np.where(y[idx] == self.classes_[b], 1.0, -1.0)
                weight = None if not weights else np.array(
                    [weights[v] for v in y[idx]])
                C = self._bounds(idx.size, weight)
                alpha, bias, _ = _solve_dual(K[np.ix_(idx, idx)], sub, C,
                                             tol=self.tol, max_iter=self.max_iter)
                keep = alpha > 1e-8
                self.machines_.append((idx[keep], alpha[keep] * sub[keep], bias,
                                       a, b))
        else:
            for a in range(self.classes_.size):
                sub = np.where(y == self.classes_[a], 1.0, -1.0)
                weight = None if not weights else np.array(
                    [weights[v] for v in y])
                C = self._bounds(y.size, weight)
                alpha, bias, _ = _solve_dual(K, sub, C, tol=self.tol,
                                             max_iter=self.max_iter)
                keep = alpha > 1e-8
                self.machines_.append((np.flatnonzero(keep),
                                       alpha[keep] * sub[keep], bias, a, -1))

    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the hyperplane.

        `(n,)` for binary problems, `(n, k)` for the multiclass variants: votes
        for `ovo`, per-class decision values for `ovr`.
        """
        X = np.asarray(X, dtype=float)
        if hasattr(self, "machines_"):
            out = np.zeros((X.shape[0], self.classes_.size))
            rows = np.arange(X.shape[0])
            if self.variant == "ovo":
                for idx, coef, bias, a, b in self.machines_:
                    d = self._gram(X, self.X_[idx]) @ coef + bias
                    np.add.at(out, (rows, np.where(d > 0, b, a)), 1.0)
            else:
                for idx, coef, bias, a, _ in self.machines_:
                    out[:, a] = self._gram(X, self.X_[idx]) @ coef + bias
            return out
        return self._gram(X, self.X_[self.support_]) @ self.dual_coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _blobs(rng, centres, n_per, scale=1.0):
    X = np.vstack([c + rng.normal(scale=scale, size=(n_per, len(c)))
                   for c in centres])
    y = np.repeat(np.arange(len(centres)), n_per)
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)

    # --- hard margin: separable data, and the margin is 1/‖w‖ ---
    Xs, ys = _blobs(rng, [(-2.0, 0.0), (2.0, 0.0)], 30, scale=0.6)
    hard = SVM("hard_margin", kernel="linear", tol=1e-7).fit(Xs, ys)
    sv = hard.support_
    w = (hard.alpha_ * hard.y_signed_)[:, None] * Xs
    w = w.sum(axis=0)
    margins = hard.y_signed_ * (
        kernel_matrix(Xs, Xs[sv], kind="linear") @ hard.dual_coef_ + hard.intercept_)
    assert np.isclose(np.min(margins), 1.0, atol=1e-6), margins.min()
    assert np.all(margins >= 1.0 - 1e-6), margins.min()
    gap = float((Xs[ys == 1] @ w).min() - (Xs[ys == 0] @ w).max())
    assert np.isclose(gap, 2.0, atol=1e-6), gap
    dual_obj = float(hard.alpha_.sum() - 0.5 * w @ w)
    assert np.isclose(dual_obj, 0.5 * w @ w, atol=1e-6), dual_obj
    assert np.mean(hard.predict(Xs) == ys) == 1.0
    print(f"svm: hard margin on separable data — {sv.size} support vectors, "
          f"margin width {2.0 / np.linalg.norm(w):.4f}, class gap {gap:.6f}, "
          f"duality gap {abs(dual_obj - 0.5 * w @ w):.1e}")

    # --- soft margin: overlapping data, C controls the number of SVs ---
    Xo, yo = _blobs(rng, [(-1.2, 0.0), (1.2, 0.0)], 40, scale=0.8)
    small = SVM("soft_margin", C=0.05, kernel="linear").fit(Xo, yo)
    large = SVM("soft_margin", C=100.0, kernel="linear").fit(Xo, yo)
    acc = np.mean(large.predict(Xo) == yo)
    assert acc > 0.85, acc
    assert small.support_.size > large.support_.size, (small.support_.size,
                                                       large.support_.size)
    print(f"svm: soft margin — accuracy {acc:.3f}, support vectors "
          f"{small.support_.size} at C=0.05 vs {large.support_.size} at C=100")

    # --- dual vs smo: same solution, different working-set rule ---
    K = kernel_matrix(Xo, kind="rbf")
    signed = np.where(yo == 1, 1.0, -1.0)
    Cvec = np.full(yo.size, 1.0)
    a_first, b_first, it_first = _solve_dual(K, signed, Cvec, second_order=False,
                                             tol=1e-6, max_iter=60000)
    a_second, b_second, it_second = _solve_dual(K, signed, Cvec, second_order=True,
                                                tol=1e-6, max_iter=60000)
    def obj(a: np.ndarray) -> float:
        return float(a.sum() - 0.5 * (a * signed) @ K @ (a * signed))

    assert abs(obj(a_first) - obj(a_second)) < 1e-5, (obj(a_first), obj(a_second))
    assert it_second < it_first, (it_second, it_first)
    gap = float(np.max(np.abs(a_first - a_second)))
    assert gap < 1e-3, gap
    print(f"svm: dual {it_first} iterations vs smo {it_second}, dual objective "
          f"{obj(a_first):.6f} == {obj(a_second):.6f}, max |Δα| {gap:.1e}")

    # --- kernels: rbf solves a problem no linear machine can ---
    theta = rng.uniform(0, 2 * np.pi, 80)
    radius = np.where(np.arange(80) < 40, 0.6, 2.0)
    Xc = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    yc = (radius > 1.0).astype(int)
    lin = SVM("linear", C=10.0).fit(Xc, yc)
    rbf = SVM("rbf", C=10.0, gamma=1.0).fit(Xc, yc)
    poly = SVM("polynomial", C=10.0, degree=2, gamma=1.0, coef0=1.0).fit(Xc, yc)
    sig = SVM("sigmoid", C=1.0, gamma=0.5, coef0=0.0).fit(Xc, yc)
    acc = {name: float(np.mean(mod.predict(Xc) == yc))
           for name, mod in (("linear", lin), ("poly", poly),
                             ("rbf", rbf), ("sigmoid", sig))}
    assert acc["linear"] < 0.75, acc
    assert acc["rbf"] > 0.98 and acc["poly"] > 0.95, acc
    # the sigmoid kernel is famously not positive semi-definite; rbf is
    assert np.linalg.eigvalsh(kernel_matrix(Xc, kind="rbf")).min() > -1e-10
    assert np.linalg.eigvalsh(
        kernel_matrix(Xc, kind="sigmoid", gamma=1.0, coef0=0.0)).min() < -1e-3
    print(f"svm: concentric circles — linear {acc['linear']:.3f}, polynomial "
          f"{acc['poly']:.3f}, rbf {acc['rbf']:.3f}, sigmoid {acc['sigmoid']:.3f} "
          f"(sigmoid Gram has a negative eigenvalue)")

    # --- nu_svm: nu *is* the SV fraction, up to slack ---
    nus = SVM("nu_svm", nu=0.3, kernel="rbf").fit(Xo, yo)
    U = 1.0 / yo.size
    signed_nu = nus.y_signed_
    g = (np.outer(signed_nu, signed_nu) * nus._gram(Xo)) @ nus.alpha_
    # optimality certificate: no feasible same-class pair reduces the objective
    residual = 0.0
    for sign in (1.0, -1.0):
        idx = np.flatnonzero(signed_nu == sign)
        up = idx[nus.alpha_[idx] < U - 1e-12]
        dn = idx[nus.alpha_[idx] > 1e-12]
        residual = max(residual, float(g[dn].max() - g[up].min()))
    assert residual < 1e-5, residual
    assert np.isclose(nus.alpha_.sum(), nus.nu, atol=1e-9)
    assert abs(float(signed_nu @ nus.alpha_)) < 1e-9
    score = signed_nu * nus.decision_function(Xo)
    frac_err = float(np.mean(score < nus.rho_ - 1e-6))
    bounded = float(np.mean(nus.alpha_ >= U - 1e-9))
    frac_sv = nus.support_.size / yo.size
    assert frac_err <= nus.nu + 1e-9, (frac_err, nus.nu)
    assert frac_sv >= nus.nu - 1e-9, (frac_sv, nus.nu)
    assert frac_err <= bounded + 1e-9, (frac_err, bounded)
    sizes = [SVM("nu_svm", nu=v, kernel="rbf").fit(Xo, yo).support_.size
             for v in (0.1, 0.3, 0.6)]
    assert sizes[0] < sizes[1] < sizes[2], sizes
    print(f"svm: nu_svm nu=0.3 -> {frac_err:.3f} margin errors <= "
          f"{bounded:.3f} bounded <= nu, {frac_sv:.3f} support vectors; "
          f"SV counts {sizes} at nu=0.1/0.3/0.6")

    # --- class_weight: the minority class gets its own street ---
    # 90 majority points sitting on top of 12 minority ones: the unweighted
    # machine buys accuracy by calling every minority point an outlier.
    imb = np.random.default_rng(2)
    Xb = np.vstack([imb.normal(0.0, 1.4, size=(90, 2)),
                    imb.normal(1.1, 0.8, size=(12, 2))])
    yb = np.r_[np.zeros(90, dtype=int), np.ones(12, dtype=int)]
    plain = SVM("class_weight", C=1.0, kernel="rbf").fit(Xb, yb)
    bal = SVM("class_weight", C=1.0, kernel="rbf",
              class_weight="balanced").fit(Xb, yb)

    def rec(model) -> float:
        return float(np.mean(model.predict(Xb[yb == 1]) == 1))

    up = int(np.sum(bal.predict(Xb) == 1) - np.sum(plain.predict(Xb) == 1))
    assert rec(bal) > rec(plain) and up > 0, (rec(bal), rec(plain), up)
    print(f"svm: {np.mean(yb == 1):.2f} positive — minority recall "
          f"{rec(plain):.2f} plain vs {rec(bal):.2f} balanced, {up} more "
          f"points called positive")

    # --- ovo and ovr: three classes, two ways to vote ---
    Xm, ym = _blobs(rng, [(-2.0, -1.0), (2.0, -1.0), (0.0, 2.0)], 25, scale=0.7)
    ovo = SVM("ovo", C=10.0, kernel="rbf", gamma=1.0).fit(Xm, ym)
    ovr = SVM("ovr", C=10.0, kernel="rbf", gamma=1.0).fit(Xm, ym)
    assert np.mean(ovo.predict(Xm) == ym) > 0.97
    assert np.mean(ovr.predict(Xm) == ym) > 0.97
    votes = ovo.decision_function(Xm)
    k = np.unique(ym).size
    assert len(ovo.machines_) == k * (k - 1) // 2 and len(ovr.machines_) == k
    assert np.allclose(votes.sum(axis=1), len(ovo.machines_)), votes.sum(axis=1)
    assert np.all(votes == np.round(votes))          # ovo scores are vote counts
    both = (np.mean(ovo.predict(Xm) == ym), np.mean(ovr.predict(Xm) == ym))
    print(f"svm: {k} classes — ovo {both[0]:.3f} ({len(ovo.machines_)} duels, "
          f"{votes.sum(axis=1)[0]:.0f} votes each), ovr {both[1]:.3f} "
          f"({len(ovr.machines_)} machines)")
    # cost grows as k(k−1)/2 for ovo but only k for ovr
    X4 = np.vstack([Xm, np.array([[0.0, -2.5]]) + 0.3 * rng.normal(size=(25, 2))])
    y4 = np.r_[ym, np.full(25, 3)]
    assert len(SVM("ovo", kernel="rbf").fit(X4, y4).machines_) == 6
    assert len(SVM("ovr", kernel="rbf").fit(X4, y4).machines_) == 4

    print("svm demo passed")


if __name__ == "__main__":
    main()
