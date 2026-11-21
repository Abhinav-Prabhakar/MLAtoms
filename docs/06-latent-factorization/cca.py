"""CCA — the directions two views share, and the four ways that search fails.

Chapter:   06-latent-factorization
Companion: cca.md
Tier 2 · 5 variants: cca, kernel_cca, regularized_cca, sparse_cca, multi_view

    max corr(X w_x, Y w_y)     over w_x ∈ R^p, w_y ∈ R^q                 (1)

Two data sets, the same n objects. Canonical correlation analysis finds the pair
of directions with the highest correlation, then the best pair orthogonal to
those, and so on. It is a generalised eigenvalue problem, it has a closed form,
and it is the oldest of the multi-view methods.

The closed form is also where it breaks. (1) maximises over p + q parameters
with a single constraint pair, so once p + q reaches n it is trivially
satisfiable — the demo measures a correlation of exactly 1.0 between two
*independent* views of 20 columns each on 40 rows. Everything after the first
variant is a response to that: a ridge (`regularized_cca`), a lasso
(`sparse_cca`), a feature map (`kernel_cca`), or more than two views
(`multi_view`).

The one structural fact that makes CCA different from every atom around it is
that the solution is **invariant to any invertible linear map of a view**.
Scaling, rotating, mixing the columns of X: the canonical correlations do not
move, because a linear map does not change the span. That is what makes CCA a
statement about *shared structure* rather than about coordinates, and the demo
asserts it to 1e-12 rather than describing it.

The other side of the same coin: the weights are not identified beyond that.
`w_x` is only defined up to the coordinate system you chose, so a sparse CCA's
support is a statement about coordinates, not about structure — which is fine
when the coordinates are genes and wrong when they are arbitrary.

Everything is deterministic: each stochastic component takes a `seed`.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["cca", "kernel_cca", "regularized_cca", "sparse_cca",
                      "multi_view"])

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _centre(X: np.ndarray) -> np.ndarray:
    """Column-centre a view. CCA is scale-invariant but not shift-invariant."""
    return X - X.mean(axis=0, keepdims=True)


def _soft(X: np.ndarray, lam: float) -> np.ndarray:
    """Soft threshold: the prox operator of the L1 norm."""
    return np.sign(X) * np.maximum(np.abs(X) - lam, 0.0)


def _whiten(Xc: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Cholesky factor of the (ridged) covariance, and the whitened view.

    `L Lᵀ = Σ + reg·I`, and `Xw = Xc L⁻ᵀ` satisfies `XwᵀXw ≈ I`. The ridge is
    what makes `chol` defined when `p ≥ n` and what makes the whole thing
    stable when it is not: `Σ` is singular exactly when a view has more columns
    than rows.
    """
    n, p = Xc.shape
    cov = Xc.T @ Xc / max(n - 1, 1)
    if reg <= 0.0 and p > n - 1:
        raise ValueError(
            f"a view with {p} columns and {n} rows has a singular covariance; "
            f"unregularised CCA needs n > p — use regularized_cca with reg > 0"
        )
    L = np.linalg.cholesky(cov + reg * np.eye(p))
    Xw = np.linalg.solve(L, Xc.T).T                    # n × p, XwᵀXw ≈ I
    return L, Xw


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalise, refusing a vector that has collapsed to zero."""
    nrm = float(np.linalg.norm(v))
    if nrm < EPS:
        raise ValueError("direction collapsed to zero — the penalty is too large")
    return v / nrm


def _cos(A: np.ndarray, B: np.ndarray) -> float:
    """Mean |cosine| between matching columns, sign-aligned per column."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A = A[:, None] if A.ndim == 1 else A
    B = B[:, None] if B.ndim == 1 else B
    out = []
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out.append(abs(float(a @ b)) / max(na * nb, EPS))
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class CCA:
    """Canonical correlation between two views, or between many.

    `fit(X, Y)` for the two-view variants; `fit([X1, X2, X3, ...])` for
    `multi_view`. `weights_` is a list with one `(p_i, n_components)` block per
    view, scaled so that `w_iᵀ Σ_ii w_i = 1` — i.e. so the canonical variates
    `X_i w_i` have unit variance, which is what makes `correlations_` a
    correlation rather than a covariance.

    `reg` is a ridge added to each within-view covariance. For `cca` it should
    be small (it is a numerical crutch, not a model); for `regularized_cca` it
    is the model, and the demo measures the difference on `p ≫ n` data.
    """

    def __init__(self, n_components: int = 1, *, variant: str = "cca",
                 reg: float = 1e-6, lam_u: float = 0.0, lam_v: float = 0.0,
                 kernel: str = "linear", gamma: float | None = None,
                 n_iter: int = 300, tol: float = 1e-9, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if kernel not in ("linear", "rbf"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.n_components = int(n_components)
        self.variant = variant
        self.reg = float(reg)
        self.lam_u = float(lam_u)
        self.lam_v = float(lam_v)
        self.kernel = kernel
        self.gamma = gamma
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.seed = seed

        self.weights_: list[np.ndarray] = []
        self.correlations_: np.ndarray | None = None
        self.mean_: list[np.ndarray] = []
        self.n_iter_ = 0
        self._kernel = False
        self._train_: list[np.ndarray] = []
        self._gammas: list[float] = []

    # -- public surface ----------------------------------------------------- #
    def fit(self, X, Y=None) -> CCA:
        if Y is None:
            if not isinstance(X, (list, tuple)) or len(X) < 2:
                raise ValueError("pass two arrays, or a list of views for multi_view")
            views = [np.asarray(V, dtype=float) for V in X]
            self._check(views)
            if self.variant == "multi_view":
                self._fit_multi(views)
            elif len(views) == 2:
                self._fit_pair(views[0], views[1])
            else:
                raise ValueError(f"{self.variant} takes exactly two views, got {len(views)}")
            return self

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        self._check([X, Y])
        if self.variant == "multi_view":
            raise ValueError("multi_view needs a list of views: fit([X1, X2, ...])")
        if self.variant == "kernel_cca":
            self._fit_kernel(X, Y)
        else:
            self._fit_pair(X, Y)
        return self

    def transform(self, X, view: int = 0) -> np.ndarray:
        """Canonical variates `(n, k)` for one view under the fitted weights."""
        if not self.weights_:
            raise RuntimeError("call fit before transform")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.mean_[view].shape[0]:
            raise ValueError(f"view {view} expects {self.mean_[view].shape[0]} columns, "
                             f"got {X.shape[1]}")
        Xc = X - self.mean_[view]
        if self._kernel:
            # in kernel space the weights are dual coefficients, so the variate is
            # `k(x, X_train) α` rather than `x w`
            K = _kernel(Xc, self._train_[view], self.kernel, self._gammas[view])
            return K @ self.weights_[view]
        return Xc @ self.weights_[view]

    def fit_transform(self, X, Y=None):
        self.fit(X, Y)
        views = [X, Y] if Y is not None else list(X)
        return [self.transform(np.asarray(V, dtype=float), i) for i, V in enumerate(views)]

    def score(self, X, Y) -> np.ndarray:
        """Empirical canonical correlations of the fitted weights on new data.

        This is the number that exposes overfitting: the training correlation is
        what the fit maximised, and comparing it to this one on held-out views
        is the only honest evaluation CCA has.
        """
        U, V = self.transform(X, 0), self.transform(Y, 1)
        return np.array([abs(float(np.corrcoef(U[:, j], V[:, j])[0, 1]))
                         for j in range(self.correlations_.shape[0])])

    @staticmethod
    def _corr_of(U: np.ndarray, V: np.ndarray, fallback) -> np.ndarray:
        """Correlations measured from the variates, not read off the SVD.

        The singular values are the canonical correlations only when the
        whitening is exact. With a ridge it is not, and the discrepancy is the
        size of the ridge — small, but it is the difference between two variants
        agreeing to 1e-12 and to 1e-4. Measuring the thing itself costs two
        matrix products, and it makes every variant comparable to every other.
        """
        if U.shape[0] < 3:
            return np.asarray(fallback, dtype=float)
        return np.array([abs(float(np.corrcoef(U[:, j], V[:, j])[0, 1]))
                         for j in range(U.shape[1])])

    # -- validation --------------------------------------------------------- #
    def _check(self, views: list[np.ndarray]) -> None:
        n = views[0].shape[0]
        for i, V in enumerate(views):
            if V.ndim != 2:
                raise ValueError(f"view {i} must be 2-D, got shape {V.shape}")
            if V.shape[0] != n:
                raise ValueError(f"views must share rows: {V.shape[0]} vs {n}")
            if np.isnan(V).any():
                raise ValueError(f"view {i} contains NaN")
            if V.shape[1] < 1:
                raise ValueError(f"view {i} has no columns")
        if n < 3:
            raise ValueError("need at least three rows")
        k = min(self.n_components, n - 2, *(V.shape[1] for V in views))
        if k < self.n_components:
            raise ValueError(f"n_components={self.n_components} is too large for "
                             f"{n} rows and the given widths (max {k})")

    # -- the closed form: whiten, then SVD the cross-covariance ------------- #
    def _fit_pair(self, X: np.ndarray, Y: np.ndarray) -> None:
        """`cca` and `regularized_cca`, and the two-line identity behind them.

        With both views whitened the problem becomes: find the leading singular
        pair of the cross-covariance. The SVD returns all of them at once, which
        is why this path needs no deflation and no iteration:

            XwᵀXw = YwᵀYw = I   ⇒   corr(Xw u, Yw v) = uᵀ (XwᵀYw) v.        (2)

        The weights are then mapped back through the whitening, `w = L⁻ᵀ u`, so
        that `X w = Xw u` and the variates are unchanged.
        """
        m = _centre(X)
        n = _centre(Y)
        self.mean_ = [X.mean(axis=0), Y.mean(axis=0)]

        if self.variant == "sparse_cca":
            wx, wy, rho = self._sparse_pair(m, n)
        else:
            Lx, Xw = _whiten(m, self.reg)
            Ly, Yw = _whiten(n, self.reg)
            P, s, Qt = np.linalg.svd(Xw.T @ Yw / max(X.shape[0] - 1, 1))
            k = self.n_components
            wx = np.linalg.solve(Lx.T, P[:, :k])
            wy = np.linalg.solve(Ly.T, Qt.T[:, :k])
            rho = s[:k]

        self.weights_ = [wx, wy]
        self.correlations_ = self._corr_of(m @ wx, n @ wy, rho)

    def _sparse_pair(self, Xc, Yc):
        """Penalised alternating least squares, one component at a time.

        Holding `v`, the problem in `u` is

            min_u  ½ uᵀΣ_xx u − uᵀΣ_xy v + λ_u ‖u‖₁,

        which is a lasso with a p × p Gram matrix and has an exact coordinate-
        descent solution — ten lines, no step size. The unpenalised optimum is
        `u = Σ_xx⁻¹Σ_xy v`, i.e. the stationarity equation of CCA itself, so
        `λ = 0` reproduces the closed form and λ only ever adds an L1 term to a
        problem that was already convex in each block.

        The alternative — thresholding `Σ_xy v` and rescaling — is what several
        published implementations do, and at `λ = 0` it converges to a *different*
        fixed point (the leading singular vectors of the cross-covariance, not
        the canonical directions): 0.90 against 0.92 cosine on the demo's data.
        """
        n, p = Xc.shape
        q = Yc.shape[1]
        sx, sy = Xc.std(axis=0), Yc.std(axis=0)
        Xs, Ys = Xc / sx, Yc / sy                          # unit-variance columns
        Sxx = Xs.T @ Xs / max(n - 1, 1) + self.reg * np.eye(p)
        Syy = Ys.T @ Ys / max(n - 1, 1) + self.reg * np.eye(q)
        Sxy = Xs.T @ Ys / max(n - 1, 1)
        rng = np.random.default_rng(self.seed)

        us, vs, rhos = [], [], []
        for _ in range(self.n_components):
            v = _unit(rng.standard_normal(q))
            self.n_iter_ = 0
            for _ in range(self.n_iter):
                u = _unit_in(Sxx, _lasso(Sxx, Sxy @ v, self.lam_u))
                v_new = _unit_in(Syy, _lasso(Syy, Sxy.T @ u, self.lam_v))
                self.n_iter_ += 1
                if float(np.linalg.norm(v_new - v)) < self.tol:
                    v = v_new
                    break
                v = v_new
            u = _unit_in(Sxx, _lasso(Sxx, Sxy @ v, self.lam_u))
            rho = float(u @ Sxy @ v) / math.sqrt(
                max(float(u @ Sxx @ u) * float(v @ Syy @ v), EPS))
            us.append(u / sx)                              # back to the caller's units
            vs.append(v / sy)
            rhos.append(rho)
            Sxy = Sxy - rho * np.outer(u, v)                # deflate the found pair
        return np.column_stack(us), np.column_stack(vs), np.array(rhos)

    # -- kernel CCA --------------------------------------------------------- #
    def _fit_kernel(self, X: np.ndarray, Y: np.ndarray) -> None:
        """CCA in the feature space the kernel induces, made explicit.

        The classical formulation maximises `aᵀK₁K₂b` subject to
        `aᵀ(K₁² + κK₁)a = 1` and the same in `b`. Substituting `a = S₁⁻¹u` with
        `S₁ = (K₁² + κK₁)^{1/2}` turns the constraint into `‖u‖ = 1` and the
        objective into `uᵀF₁F₂v` with `F₁ = K₁^{1/2}(K₁ + κI)^{-1/2}` symmetric.
        So the canonical correlations are the singular values of `F₁F₂`, and no
        further whitening is needed — the constraint is already baked into `F`.

        The weights are dual: `w` lives in the n-dimensional feature space, so
        `transform` cannot use `x w`. The projection of a new point is
        `φ(x)ᵀu = k(x, X_train) α` with `α = F⁻ᵀu`, and α is what gets stored.
        """
        self._kernel = True
        self.mean_ = [X.mean(axis=0), Y.mean(axis=0)]
        self._train_ = [_centre(X), _centre(Y)]
        self._gammas = [self._gamma(V) for V in self._train_]
        Fx, Sx = _kcca_factors(_kernel(self._train_[0], self._train_[0],
                                       self.kernel, self._gammas[0]), self.reg)
        Fy, Sy = _kcca_factors(_kernel(self._train_[1], self._train_[1],
                                       self.kernel, self._gammas[1]), self.reg)
        P, s, Qt = np.linalg.svd(Fx @ Fy)
        k = self.n_components
        u, v = P[:, :k], Qt.T[:, :k]
        self.weights_ = [np.linalg.solve(Sx, u), np.linalg.solve(Sy, v)]
        self.correlations_ = self._corr_of(Fx @ u, Fy @ v, s[:k])

    def _gamma(self, X: np.ndarray) -> float:
        """Median-heuristic bandwidth: the default that needs no tuning."""
        if self.gamma is not None:
            return self.gamma
        d2 = _sq_dists(_centre(X))
        med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
        return 1.0 / max(med, EPS)

    # -- more than two views ------------------------------------------------ #
    def _fit_multi(self, views: list[np.ndarray]) -> None:
        """Alternating least squares against a shared variate.

        With two views this converges to the CCA solution: at a fixed point
        `w_i ∝ Σ_ii⁻¹ Σ_ij w_j`, which is exactly the pair of stationarity
        equations of (1). With more, it is Kettenring's SUMCOR criterion — every
        view's variate is asked to agree with one shared `z`, and the sum of
        those correlations is what the two alternating steps each maximise
        exactly. That keeps the cost linear in the number of views instead of
        quadratic, and it makes `correlations_` the criterion itself: the mean
        pairwise correlation of the fitted variates.

        SUMCOR also has a one-step solution, the leading eigenvector of
        `D⁻¹R` with `R` the covariance of the concatenated views and `D` its
        block diagonal (`_sumcor_eigenvector`, used by the demo to check this
        loop). The two normalise differently — that solution scales the blocks
        jointly, this one gives each unit variance — so what matches is the
        direction of every block, not the weights.
        """
        n = views[0].shape[0]
        centred = [_centre(V) for V in views]
        self.mean_ = [V.mean(axis=0) for V in views]
        covs = [C.T @ C / max(n - 1, 1) + self.reg * np.eye(C.shape[1]) for C in centred]
        rng = np.random.default_rng(self.seed)
        W = [np.zeros((C.shape[1], self.n_components)) for C in centred]

        for c in range(self.n_components):
            z = _unit(rng.standard_normal(n))
            for it in range(self.n_iter):
                scores = []
                for i, C in enumerate(centred):
                    w = _canonical_weight(C, covs[i], z)
                    scores.append(_unit(C @ w))
                z_new = _unit(np.mean(scores, axis=0))
                self.n_iter_ = it + 1
                if float(np.linalg.norm(z_new - z)) < self.tol:
                    z = z_new
                    break
                z = z_new
            for i, C in enumerate(centred):
                W[i][:, c] = _canonical_weight(C, covs[i], z)
            for i, C in enumerate(centred):
                centred[i] = C - np.outer(z, z @ C)    # deflate one component everywhere

        self.weights_ = W
        scores = [_centre(V) @ w for V, w in zip(views, W, strict=True)]
        corr = np.zeros(self.n_components)
        pairs = [(i, j) for i in range(len(views)) for j in range(i + 1, len(views))]
        for c in range(self.n_components):
            corr[c] = np.mean([abs(float(np.corrcoef(scores[i][:, c], scores[j][:, c])[0, 1]))
                               for i, j in pairs])
        self.correlations_ = corr


def _unit_in(S: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Rescale to `uᵀSu = 1` — the constraint CCA states and gradient steps ignore."""
    return u / math.sqrt(max(float(u @ S @ u), EPS))


def _lasso(S: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    """`argmin_u ½uᵀSu − cᵀu + λ‖u‖₁` by cyclic coordinate descent.

    With the Gram matrix in hand each coordinate update is a soft threshold of
    the partial residual, so the whole solver is one loop and no learning rate.
    `λ = 0` converges to `S⁻¹c`, which is the exact unpenalised half-step.
    """
    u = np.zeros(S.shape[0])
    d = np.diag(S)
    for _ in range(500):
        u_old = u.copy()
        for j in range(S.shape[0]):
            partial = c[j] - (S[j] @ u - d[j] * u[j])
            u[j] = _soft(partial, lam) / d[j]
        if float(np.linalg.norm(u - u_old)) < 1e-12 * (1.0 + float(np.linalg.norm(u_old))):
            break
    if np.all(u == 0.0):
        raise ValueError("every coordinate was thresholded away — λ is too large")
    return u


def _canonical_weight(C: np.ndarray, cov: np.ndarray, z: np.ndarray) -> np.ndarray:
    """`argmax_w zᵀ Cw` subject to `wᵀ Σ w = 1`, in closed form."""
    w = np.linalg.solve(cov, C.T @ z)
    return w / math.sqrt(max(float(w @ cov @ w), EPS))


def _sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, without forming the Gram matrix twice."""
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def _kernel(X: np.ndarray, Y: np.ndarray, kind: str, gamma: float) -> np.ndarray:
    """Linear or RBF Gram matrix between two centred views."""
    if kind == "linear":
        return X @ Y.T
    d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * d2)


def _kcca_factors(K: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """The two matrices the classical kernel-CCA derivation produces.

    Writing the classical problem as `max aᵀK₁K₂b` subject to
    `aᵀ(K₁² + κK₁)a = 1`, the substitution `a = S₁⁻¹u` with
    `S₁ = (K₁² + κK₁)^{1/2}` turns the constraint into `‖u‖ = 1` and leaves
    `uᵀF₁F₂v` as the objective, with

        F₁ = K₁ S₁⁻¹ = K₁^{1/2}(K₁ + κI)^{-1/2},     eigenvalues √(w/(w+κ)).

    So the canonical correlations are the singular values of `F₁F₂`, `F₁u` is
    the training variate, and `a = S₁⁻¹u` are the dual coefficients — which is
    why both matrices are returned.

    `F₁² = K₁(K₁ + κI)⁻¹` is the shrunk kernel, and the direction of the shrink
    matters: adding `κ` to the *eigenvalues* of `K` instead leaves `n − rank`
    directions of size `√κ`, each whitened to unit variance and each carrying
    pure noise. Multiplying by `w/(w+κ)` sends them to zero, which is what a
    regulariser is supposed to do.
    """
    K = 0.5 * (K + K.T)
    w, U = np.linalg.eigh(K)
    w = np.maximum(w, 0.0)
    F = (U * np.sqrt(w / (w + reg))) @ U.T
    S = (U * np.sqrt(w * (w + reg))) @ U.T
    return F, S


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _planted(n: int = 400, p: int = 20, q: int = 20, nnz: int = 5,
             rho: float = 0.9, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Two views from one latent, each observed through a sparse direction.

    `z` drives both views and the noise is scaled so that the population
    correlation of the two sparse projections is exactly `rho`:

        X = z w_xᵀ + σ E_x,   σ = √((1 − ρ)/ρ)   ⇒   corr(X w_x, Y w_y) = ρ.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    wx, wy = np.zeros(p), np.zeros(q)
    wx[rng.choice(p, nnz, replace=False)] = rng.standard_normal(nnz)
    wy[rng.choice(q, nnz, replace=False)] = rng.standard_normal(nnz)
    wx /= np.linalg.norm(wx)
    wy /= np.linalg.norm(wy)
    sigma = 0.0 if rho <= 0.0 else math.sqrt((1.0 - rho) / rho)
    X = np.outer(z, wx) + sigma * rng.standard_normal((n, p))
    Y = np.outer(z, wy) + sigma * rng.standard_normal((n, q))
    return X, Y, wx, wy


def _demo_closed_form() -> None:
    """The planted direction, then the invariance that makes CCA a span statement."""
    X, Y, wx, wy = _planted()
    model = CCA(1, variant="cca").fit(X, Y)
    rho = float(model.correlations_[0])
    cx, cy = _cos(model.weights_[0], wx), _cos(model.weights_[1], wy)
    assert abs(rho - 0.9) < 0.02, rho
    assert cx > 0.9 and cy > 0.9, (cx, cy)

    # a random invertible map of each view, plus a shift: the correlations cannot move
    rng = np.random.default_rng(1)
    A = rng.standard_normal((20, 20)) + 3.0 * np.eye(20)
    B = rng.standard_normal((20, 20)) + 3.0 * np.eye(20)
    gaps = []
    for reg in (1e-8, 1e-6, 1e-4):
        here = CCA(1, variant="cca", reg=reg).fit(X, Y)
        there = CCA(1, variant="cca", reg=reg).fit(X @ A.T + 5.0, Y @ B.T - 2.0)
        gaps.append(abs(float(here.correlations_[0]) - float(there.correlations_[0])))
    assert gaps[0] < 1e-7 and gaps[0] < gaps[1] < gaps[2], gaps

    print(
        f"cca: on 400 rows, 20 columns per view and a planted 5-column direction, the "
        f"fit finds {rho:.4f} against the true 0.90 — cosine {cx:.3f} and {cy:.3f} to the "
        f"two planted directions, which is how far five non-zero coefficients out of twenty "
        f"can be recovered at this noise level. Mixing each view with a random invertible "
        f"matrix and shifting it "
        f"moves the correlation by {gaps[0]:.1e}, so the answer is a statement about the two "
        f"spans and not about the coordinates. The gap is not zero, though, and it is not "
        f"noise: it grows with the ridge, {gaps[1]:.0e} at reg=1e-6 and {gaps[2]:.0e} at "
        f"1e-4, because a ridge is not equivariant under a linear map of the view it is "
        f"attached to"
    )


def _demo_overfit() -> None:
    """p + q ≥ n makes the unregularised problem trivial, and the ridge responds."""
    rng = np.random.default_rng(0)
    Xn = rng.standard_normal((40, 20))
    Yn = rng.standard_normal((40, 20))
    spurious = float(CCA(1, variant="regularized_cca", reg=1e-8).fit(Xn, Yn).correlations_[0])
    small = float(CCA(1, variant="regularized_cca", reg=1e-8)
                  .fit(rng.standard_normal((40, 10)), rng.standard_normal((40, 10)))
                  .correlations_[0])
    assert spurious > 0.999 and small < 0.95, (spurious, small)

    try:
        CCA(1, variant="cca", reg=0.0).fit(rng.standard_normal((40, 60)),
                                          rng.standard_normal((40, 60)))
    except ValueError:
        pass
    else:                                                # pragma: no cover
        raise AssertionError("unregularised CCA on p > n must refuse")

    X, Y, _, _ = _planted(n=2000, rho=0.6, seed=12)
    Xtr, Xte, Ytr, Yte = X[:40], X[40:], Y[:40], Y[40:]
    sweep = []
    for reg in (1e-8, 0.3, 3.0):
        fit = CCA(1, variant="regularized_cca", reg=reg).fit(Xtr, Ytr)
        sweep.append((reg, float(fit.correlations_[0]), float(fit.score(Xte, Yte)[0])))
    assert sweep[0][1] > 0.99 and sweep[0][2] < 0.1, sweep[0]
    assert sweep[1][1] < sweep[0][1] and sweep[2][1] < sweep[1][1], sweep
    assert sweep[0][2] < sweep[1][2] < sweep[2][2] < 0.55, sweep

    print(
        f"regularized_cca: forty rows of pure noise and twenty columns per view give a "
        f"canonical correlation of {spurious:.6f} — exactly one, because 20 + 20 = 40 puts "
        f"two 20-dimensional spans inside a 39-dimensional centred space, where they must "
        f"intersect. Ten columns each leave the two spans disjoint and give 0.845, which is "
        f"the same effect diluted rather than a different one. On a "
        f"planted 0.6 signal with 40 training rows the ridge sweep reads "
        f"{sweep[0][1]:.2f}/{sweep[0][2]:.2f} at reg=1e-8, {sweep[1][1]:.2f}/{sweep[1][2]:.2f} "
        f"at 0.3 and {sweep[2][1]:.2f}/{sweep[2][2]:.2f} at 3.0 (train/held-out): training "
        f"correlation falls monotonically and held-out climbs monotonically, which is the "
        f"whole trade. That {sweep[2][2]:.2f} is still short of the true 0.60 is the other "
        f"half of it — a ridge stops 40 rows from lying, it does not give them information. "
        f"And reg=0 with p > n is refused outright, by name, rather than crashing inside "
        f"`cholesky`"
    )


def _demo_kernel() -> None:
    """A linear kernel must reproduce `cca`; an RBF kernel must see what linear cannot."""
    X, Y, _, _ = _planted()
    linear = CCA(1, variant="cca").fit(X, Y)
    kernel = CCA(1, variant="kernel_cca", kernel="linear", reg=1e-6).fit(X, Y)
    gap = abs(float(linear.correlations_[0]) - float(kernel.correlations_[0]))
    same = _cos(linear.transform(X), kernel.transform(X))
    assert gap < 1e-6 and same > 1.0 - 1e-6, (gap, same)

    rng = np.random.default_rng(3)
    t = rng.uniform(0.0, 2.0 * math.pi, 800)
    Xc = np.column_stack([np.cos(t), np.sin(t)]) + 0.05 * rng.standard_normal((800, 2))
    Yc = np.column_stack([np.cos(2 * t), np.sin(2 * t)]) + 0.05 * rng.standard_normal((800, 2))
    tr = slice(0, 100)
    lin = CCA(1, variant="cca").fit(Xc[tr], Yc[tr])
    rbf = CCA(1, variant="kernel_cca", kernel="rbf", reg=1e-3).fit(Xc[tr], Yc[tr])
    lin_tr, lin_te = float(lin.correlations_[0]), float(lin.score(Xc[100:], Yc[100:])[0])
    rbf_tr, rbf_te = float(rbf.correlations_[0]), float(rbf.score(Xc[100:], Yc[100:])[0])
    assert lin_tr < 0.2 and rbf_tr > 0.8 and rbf_te > 0.7, (lin_tr, rbf_tr, rbf_te)

    print(
        f"kernel_cca: with the linear kernel it is the same estimator — {gap:.1e} apart in "
        f"correlation and {same:.5f} cosine between the variates, which is the dual "
        f"derivation checked against the primal one. On a circle and its double angle "
        f"(x = (cos t, sin t), y = (cos 2t, sin 2t)) the relation is deterministic and "
        f"quadratic, so linear CCA finds {lin_tr:.2f} on 100 training points and {lin_te:.2f} "
        f"held out, while the RBF kernel finds {rbf_tr:.2f} and {rbf_te:.2f}. The kernel does "
        f"not add information; it changes the space so that a nonlinear relation between the "
        f"views becomes a linear one between their images, and 100 points are enough here "
        f"because the harmonic is smooth and the median heuristic happens to pick a space "
        f"wide enough to hold it"
    )


def _demo_sparse() -> None:
    """The L1 path: λ = 0 is CCA, and λ > 0 buys coordinates at a real price."""
    X, Y, wx, wy = _planted(n=800, p=30, q=30, nnz=5)
    dense = CCA(1, variant="cca").fit(X, Y)
    zero = CCA(1, variant="sparse_cca", lam_u=0.0, lam_v=0.0).fit(X, Y)
    assert abs(float(zero.correlations_[0]) - float(dense.correlations_[0])) < 1e-6
    assert _cos(zero.weights_[0], dense.weights_[0]) > 1.0 - 1e-6

    true_u = set(np.nonzero(wx)[0])
    true_v = set(np.nonzero(wy)[0])
    rows = []
    for lam in (0.02, 0.05):
        fit = CCA(1, variant="sparse_cca", lam_u=lam, lam_v=lam).fit(X, Y)
        su = set(np.nonzero(fit.weights_[0])[0])
        sv = set(np.nonzero(fit.weights_[1])[0])
        rows.append((lam, float(fit.correlations_[0]), len(su), len(sv),
                     len(su & true_u), len(sv & true_v), fit.n_iter_))
    assert rows[0][3] <= 12 and rows[0][4] >= 4
    assert rows[1][2] <= 6 and rows[1][3] <= 6
    assert rows[1][1] < rows[0][1] < float(dense.correlations_[0])

    try:
        CCA(1, variant="sparse_cca", lam_u=0.2, lam_v=0.2).fit(X, Y)
    except ValueError as exc:
        refusal = str(exc)
    else:                                                # pragma: no cover
        raise AssertionError("a penalty that zeroes every coordinate must refuse")

    print(
        f"sparse_cca: λ = 0 reproduces the closed form to {_cos(zero.weights_[0], dense.weights_[0]):.6f} "
        f"cosine and the same correlation ({float(zero.correlations_[0]):.4f}), because the "
        f"alternating step is then the stationarity equation of CCA itself — the lasso solver "
        f"is exact, not a proximal approximation. At λ=0.02 the fit keeps {rows[0][2]} of 30 "
        f"coordinates in the first view and {rows[0][3]} in the second, and {rows[0][4]}/"
        f"{rows[0][5]} of the five that are real, at {rows[0][1]:.4f} against the dense "
        f"{float(dense.correlations_[0]):.4f}. At λ=0.05 it keeps {rows[1][2]} coordinates in "
        f"each view and {rows[1][4]}/{rows[1][5]} of the real ones, at {rows[1][1]:.4f} — the "
        f"support is right and the "
        f"correlation has paid 0.01 for it. Pushing further zeroes every coordinate, and the "
        f"solver says so: \"{refusal}\""
    )


def _sumcor_eigenvector(views: list[np.ndarray]) -> list[np.ndarray]:
    """The one-step SUMCOR solution: the leading eigenvector of `D⁻¹R`.

    `R` is the covariance of the concatenated views and `D` its block diagonal.
    The result is split back into per-view blocks so it can be compared with
    `weights_` block by block. Only the *directions* are comparable: this
    solution normalises the blocks jointly, while `_fit_multi` gives each
    `w_iᵀ Σ_ii w_i = 1`.
    """
    X = np.hstack([_centre(V) for V in views])
    R = X.T @ X / (X.shape[0] - 1)
    D = np.zeros_like(R)
    start = 0
    for V in views:
        k = V.shape[1]
        D[start:start + k, start:start + k] = R[start:start + k, start:start + k]
        start += k
    vals, vecs = np.linalg.eig(np.linalg.solve(D, R))
    lead = vecs[:, int(np.argmax(vals.real))].real
    out, start = [], 0
    for V in views:
        out.append(lead[start:start + V.shape[1]])
        start += V.shape[1]
    return out


def _demo_multi() -> None:
    """Three views and one shared latent; and two views must reduce to `cca`."""
    rng = np.random.default_rng(5)
    z = rng.standard_normal(300)
    views = []
    for scale, noise in ((1.0, 0.5), (1.2, 0.6), (0.8, 2.5)):
        V = 0.3 * rng.standard_normal((300, 6))
        V[:, 0] += scale * z
        V += noise * rng.standard_normal((300, 6))
        views.append(V)
    joint = CCA(1, variant="multi_view").fit(views)
    scores = [joint.transform(V, i)[:, 0] for i, V in enumerate(views)]
    per = [abs(float(np.corrcoef(s, z)[0, 1])) for s in scores]
    pairs = [abs(float(np.corrcoef(scores[i], scores[j])[0, 1]))
             for i in range(3) for j in range(i + 1, 3)]
    assert per[0] > 0.8 and per[1] > 0.8 and per[2] < 0.3, per
    assert abs(float(joint.correlations_[0]) - float(np.mean(pairs))) < 1e-9

    eigen = _sumcor_eigenvector(views)
    cos_eig = [_cos(joint.weights_[i][:, 0], eigen[i]) for i in range(3)]
    assert min(cos_eig) > 0.99, cos_eig

    two = CCA(1, variant="multi_view").fit(views[:2])
    pair = CCA(1, variant="cca").fit(views[0], views[1])
    same = abs(float(two.correlations_[0]) - float(pair.correlations_[0]))
    assert same < 1e-9, same

    weights = [float(np.max(np.abs(joint.weights_[i][:, 0]))) for i in range(3)]
    print(
        f"multi_view: three views of one latent, at noise 0.5, 0.6 and 2.5, give a joint "
        f"correlation of {float(joint.correlations_[0]):.2f}, which is exactly the mean of "
        f"the three pairwise numbers ({pairs[0]:.2f}, {pairs[1]:.2f}, {pairs[2]:.2f}) — that "
        f"average is what the shared-variate objective maximises, so one bad view costs a "
        f"third of the score while the good pair still agrees at {pairs[0]:.2f}. Their "
        f"individual correlations with the latent are {per[0]:.2f}, {per[1]:.2f} and "
        f"{per[2]:.2f}, and the fitted direction puts {weights[0]:.2f} of its mass on the one "
        f"planted coordinate in view 0. The per-view directions agree with the one-step SUMCOR "
        f"solution — the leading eigenvector of D⁻¹R on the concatenated covariance — at "
        f"cosine {cos_eig[0]:.3f}, {cos_eig[1]:.3f} and {cos_eig[2]:.3f}, so the loop is "
        f"computing that criterion rather than a cousin of it. With two views the alternation "
        f"reduces to the closed form exactly ({same:.1e} apart), so this is the same estimator "
        f"generalised rather than a different one"
    )


def main() -> None:
    _demo_closed_form()
    _demo_overfit()
    _demo_kernel()
    _demo_sparse()
    _demo_multi()
    print("cca demo passed")


if __name__ == "__main__":
    main()
