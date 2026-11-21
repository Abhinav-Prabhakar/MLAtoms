"""ICA — separating sources, not just decorrelating them.

Chapter:   06-latent-factorization
Companion: ica.md
Tier 2 · 6 variants: fastica_deflation, fastica_symmetric, infomax, jade,
                     kernel_ica, whitening

    x = A s,    s_i independent, at most one of them Gaussian                (1)

PCA and factor analysis find directions that are uncorrelated. ICA finds
directions along which the projections are *independent*, which is a much
stronger statement and, remarkably, a solvable one: under (1) the mixing matrix
is identifiable up to a permutation and a scale of its columns.

Uncorrelated is not enough because independence is a statement about the whole
distribution and covariance only sees second moments. Whitening makes every
rotation look identical to second moments — cov(Z) = I whatever you do — so the
information ICA uses lives in the higher cumulants, and the algorithm is a search
for the rotation that arranges them correctly.

    whitening          Z = W_w (x − μ),  cov(Z) = I     necessary, not sufficient
    fastica_deflation  one component at a time, then Gram–Schmidt
    fastica_symmetric  all components at once, decorrelated together
    infomax            maximum likelihood with a logistic density, natural gradient
    jade               joint diagonalisation of the fourth-order cumulant
    kernel_ica         dependence measured by kernel alignment, not cumulants

The first five are the same fixed point in different clothes, and the demo shows
them landing on the same unmixing. `kernel_ica` is not: it replaces kurtosis with
a kernel-based dependence measure, which is what lets it work when the
non-Gaussianity is too weak for a cumulant.

The identifiability theorem has a sharp edge and the demo walks up to it: with
two Gaussian sources **no** method can separate them, because every rotation of a
spherical Gaussian is again a spherical Gaussian. The contrast is exactly flat,
and that is a property of the problem, not a bug in any of the six variants.

Everything is deterministic: each random component takes a `seed`, and the demo
prints identical output on every run.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["fastica_deflation", "fastica_symmetric", "infomax", "jade",
                      "kernel_ica", "whitening"])

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sym_inv_sqrt(A: np.ndarray) -> np.ndarray:
    """`A^{-1/2}` for a symmetric positive-definite `A`, via one eigendecomposition."""
    w, V = np.linalg.eigh(A)
    return (V / np.sqrt(np.maximum(w, EPS))) @ V.T


def _whiten(X: np.ndarray):
    """`Z = (X − μ) W_wᵀ` with `cov(Z) = I`, and the `W_w` that does it.

    Whitening is a change of coordinates, not a projection: no dimension is
    discarded unless its variance is numerically zero. The component count is
    decided later, by the estimator.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    S = Xc.T @ Xc / X.shape[0]
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    keep = w > EPS * max(float(w[0]), 1.0)
    W_w = (V[:, keep] / np.sqrt(w[keep])).T          # (r, d), rows are directions
    return mean, W_w, Xc @ W_w.T


def _kurtosis(y: np.ndarray) -> float:
    """Excess kurtosis of a centred, unit-variance sample."""
    return float(np.mean(y ** 4) - 3.0)


def _disagreement(W1: np.ndarray, W2: np.ndarray) -> float:
    """Worst-row angle (radians) between two sets of unmixing rows, best pairing.

    Rows are directions, so the comparison has to be invariant to sign, to scale
    and to the order the components came out in: `arccos` of the largest absolute
    cosine, maximised over the rows of `W1`.
    """
    A = W1 / np.maximum(np.linalg.norm(W1, axis=1, keepdims=True), EPS)
    B = W2 / np.maximum(np.linalg.norm(W2, axis=1, keepdims=True), EPS)
    return float(np.max(np.arccos(np.clip(np.max(np.abs(A @ B.T), axis=1), -1.0, 1.0))))


def _align_error(W: np.ndarray, A: np.ndarray) -> float:
    """Worst-row angle (radians) from `W` to the rows of `A^{-1}`, best pairing."""
    return _disagreement(W, np.linalg.inv(A))


def _negentropy(Y: np.ndarray) -> float:
    """Kurtosis-based negentropy proxy, summed over the columns of `Y`.

    `J(y) ≈ Σ_i (E[y_i⁴] − 3)² / 48` for unit-variance `y_i`: 0 for a Gaussian,
    positive away from one. The nonlinearity has to be *even* — a proxy built on
    an odd function such as `tanh` is not invariant to flipping a component's
    sign, which is not a property the underlying negentropy has.
    """
    k = np.mean(Y ** 4, axis=0) - 3.0
    return float(np.sum(k * k) / 48.0)


def _givens(d: int, angles: np.ndarray) -> np.ndarray:
    """A `d × d` orthogonal matrix from `d(d−1)/2` angles, one per pair (p, q)."""
    R = np.eye(d)
    i = 0
    for p in range(d - 1):
        for q in range(p + 1, d):
            c, s = math.cos(angles[i]), math.sin(angles[i])
            G = np.eye(d)
            G[p, p], G[q, q] = c, c
            G[p, q], G[q, p] = -s, s
            R = R @ G
            i += 1
    return R


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class ICA:
    """Independent component analysis by one of six variants.

    `components_` is `(k, d)` — the unmixing matrix, so that
    `transform(X) = (X − mean_) @ components_.T` — and `mixing_` is its
    pseudo-inverse. `rotation_` is the `(k, d)` map from the whitened space, so
    `components_ = rotation_ @ whitening_` always holds; keeping it exposed is
    what makes the variants comparable, since they all whiten identically and
    differ only in `rotation_`.
    """

    def __init__(self, n_components: int | None = None, *,
                 variant: str = "fastica_symmetric", seed: int = 0, extend: bool = True,
                 max_iter: int = 4000, tol: float = 1e-8, lr: float = 0.2):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        self.n_components = n_components
        self.variant = variant
        self.seed = seed
        self.extend = bool(extend)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lr = float(lr)

        self.mean_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.unmixing_: np.ndarray | None = None
        self.rotation_: np.ndarray | None = None      # sources = Z @ rotation_.T
        self.bandwidth_: float = 1.0                  # kernel_ica only
        self.kernel_sample_: np.ndarray | None = None  # kernel_ica only
        self.kurtosis_signs_: np.ndarray | None = None   # infomax only
        self.negentropy_: float = 0.0                    # infomax only
        self.score_residual_: float = 0.0                # infomax only
        self.contrast_curve_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.n_features_in_: int | None = None

    # -- public surface ----------------------------------------------------- #
    @property
    def components_(self) -> np.ndarray:
        """`(k, d)` unmixing matrix, sklearn's orientation."""
        if self.unmixing_ is None:
            raise RuntimeError("call fit first")
        return self.unmixing_

    @property
    def mixing_(self) -> np.ndarray:
        """`(d, k)` mixing matrix: the columns that generated the data."""
        return np.linalg.pinv(self.components_)

    def fit(self, X) -> ICA:
        X = self._check(X)
        d = X.shape[1]
        k = d if self.n_components is None else int(self.n_components)
        if not 1 <= k <= d:
            raise ValueError(f"n_components={k} is outside 1..{d}")
        if k < d and self.variant != "fastica_deflation":
            raise ValueError(
                f"{self.variant} extracts all {d} components at once; use "
                f"variant='fastica_deflation' to pull out k < d of them one at a time")
        self.mean_, self.whitening_, Z = _whiten(X)
        self.n_iter_ = 0
        self.converged_ = False
        self.contrast_curve_ = []
        if self.variant == "whitening":
            self.rotation_ = np.eye(Z.shape[1])
        elif self.variant in ("fastica_deflation", "fastica_symmetric"):
            self.rotation_ = self._fit_fastica(Z, k)
        elif self.variant == "infomax":
            self.rotation_ = self._fit_infomax(Z)
        elif self.variant == "jade":
            self.rotation_ = self._fit_jade(Z)
        else:
            self.rotation_ = self._fit_kernel(Z)
        self.unmixing_ = self.rotation_ @ self.whitening_
        return self

    def transform(self, X) -> np.ndarray:
        """Unmixed sources, shape `(m, k)` — the estimate of `s`."""
        if self.unmixing_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X, fit=False)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    # -- validation --------------------------------------------------------- #
    def _check(self, X, fit: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if np.isnan(X).any():
            raise ValueError("ICA needs a complete matrix")
        if fit and X.shape[0] < 2 * X.shape[1]:
            raise ValueError(
                f"{X.shape[0]} rows for {X.shape[1]} columns: whitening and the "
                f"higher-order statistics both need more data than that")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        return X

    # -- FastICA: the tanh fixed point -------------------------------------- #
    def _fastica_step(self, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
        """`E[z g(wᵀz)] − E[g'(wᵀz)] w` for every row at once."""
        U = Z @ W.T
        G = np.tanh(U)                                # g = tanh, logcosh's derivative
        Gp = 1.0 - G * G                              # g'
        return (G.T @ Z) / Z.shape[0] - Gp.mean(axis=0)[:, None] * W

    def _fit_fastica(self, Z: np.ndarray, k: int) -> np.ndarray:
        n, d = Z.shape
        rng = np.random.default_rng(self.seed)
        if self.variant == "fastica_symmetric":
            W = np.linalg.qr(rng.standard_normal((d, d)))[0]
            for it in range(1, self.max_iter + 1):
                W_new = self._fastica_step(Z, W)
                # symmetric decorrelation: (W⁺W⁺ᵀ)^{-1/2}W⁺ makes the rows
                # orthonormal without singling any of them out
                W_new = _sym_inv_sqrt(W_new @ W_new.T) @ W_new
                change = float(np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)))
                W = W_new
                self.contrast_curve_.append(_negentropy(Z @ W.T))
                self.n_iter_ = it
                if change < self.tol:
                    self.converged_ = True
                    break
            return W

        # deflation: one direction at a time, each orthogonalised against the rest
        W = np.zeros((k, d))
        for i in range(k):
            w = rng.standard_normal(d)
            w /= np.linalg.norm(w)
            for it in range(1, self.max_iter + 1):
                u = Z @ w
                g = np.tanh(u)
                gp = 1.0 - g * g
                w_new = Z.T @ g / n - gp.mean() * w
                # two Gram–Schmidt passes: one leaves a residue of the size of
                # the projection error, which is enough to spoil convergence
                for _ in range(2):
                    if i:
                        w_new -= (W[:i] @ w_new) @ W[:i]
                w_new /= max(np.linalg.norm(w_new), EPS)
                done = abs(float(w_new @ w)) > 1.0 - self.tol
                w = w_new
                self.n_iter_ = max(self.n_iter_, it)
                if done:
                    self.converged_ = True
                    break
            W[i] = w
            self.contrast_curve_.append(_negentropy(Z @ W[:i + 1].T))
        return W

    # -- Infomax: natural gradient, with a switching score function ---------- #
    def _fit_infomax(self, Z: np.ndarray) -> np.ndarray:
        """ML by natural gradient, with a score function chosen per component.

        For a super-Gaussian source the score is `ψ(u) = −tanh(u) − u`; for a
        sub-Gaussian one it is `ψ(u) = +tanh(u) − u`. The sign is read off the
        kurtosis of the initial projections and then **held fixed**: switching it
        every iteration, as the online extended-infomax papers do, makes this
        batch iteration oscillate between two score functions instead of
        converging.

        A wrong sign locks the run into the wrong stationary point — the uniform
        source in the demo's data has negative kurtosis, and a run that assumes
        `+1` everywhere lands 48° away — so the fit makes three starts and keeps
        the most non-Gaussian one. That selection is also what
        `fastica_symmetric` optimises, which is why the two agree; the algorithm
        is still a different one, ascended by a different rule.
        """
        n, d = Z.shape
        rng = np.random.default_rng(self.seed)
        best = None
        total = 0
        for _ in range(3 if self.extend else 1):
            W = np.linalg.qr(rng.standard_normal((d, d)))[0]
            # Bell–Sejnowski's original model assumes every source is
            # super-Gaussian; "extended" reads the sign off the data instead
            K = (np.sign(np.mean((Z @ W.T) ** 4, axis=0) - 3.0) if self.extend
                 else np.ones(d))
            for it in range(1, self.max_iter + 1):
                U = Z @ W.T
                psi = -K * np.tanh(U) - U
                # ΔW = lr (I + E[ψ(u)uᵀ]) W: the natural gradient, whose identity
                # part cancels the −u in ψ and leaves −K E[tanh(u)uᵀ]
                W_new = W + self.lr * ((np.eye(d) + psi.T @ U / n) @ W)
                delta = float(np.linalg.norm(W_new - W) / max(np.linalg.norm(W), EPS))
                W = W_new
                total += 1
                if it > 20 and delta < self.tol:
                    break
            # the fixed-point condition, measured on the W it was attained at:
            # the step vanishes exactly when E[ψ(u)uᵀ] = −I. That is a stronger
            # check than the step size, because row scaling changes both u and
            # the score, so a small step and a small residual are not the same.
            U = Z @ W.T
            psi = -K * np.tanh(U) - U
            resid = float(np.max(np.abs(np.eye(d) + psi.T @ U / n)))
            R = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), EPS)
            neg = _negentropy(Z @ R.T)
            if best is None or neg > best[0]:
                best = (neg, R, K, delta < self.tol, it, resid)
        self.n_iter_ = total
        self.negentropy_ = best[0]
        self.kurtosis_signs_ = best[2]
        self.converged_ = bool(best[3])
        self.score_residual_ = best[5]
        return best[1]

    # -- JADE: diagonalise the cumulant tensor ------------------------------ #
    @staticmethod
    def _cumulant_apply(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
        """`Q(M) = E[(zᵀMz) z zᵀ] − tr(M) I − 2M`, valid for whitened `z`."""
        q = np.einsum("ij,jk,ik->i", Z, M, Z)
        return (Z * q[:, None]).T @ Z / Z.shape[0] - np.trace(M) * np.eye(len(M)) - 2.0 * M

    def _fit_jade(self, Z: np.ndarray) -> np.ndarray:
        d = Z.shape[1]
        pairs = [(i, j) for i in range(d) for j in range(i, d)]
        # the operator on symmetric matrices, in an orthonormal basis of them: its
        # eigenvectors are the matrices the rotation has to diagonalise
        basis = []
        for i, j in pairs:
            E = np.zeros((d, d))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / math.sqrt(2.0)
            basis.append(E)
        m = len(pairs)
        Op = np.zeros((m, m))
        for col, E in enumerate(basis):
            Q = self._cumulant_apply(Z, E)
            for row, F in enumerate(basis):
                Op[row, col] = float(np.sum(Q * F))
        lam, vec = np.linalg.eigh(Op)
        order = np.argsort(np.abs(lam))[::-1][:d]     # the d significant eigenmatrices
        Ms = [sum(vec[i, c] * basis[i] for i in range(m)) for c in order]

        # joint approximate diagonalisation: exact Jacobi steps on each pair
        V = np.eye(d)
        for _ in range(60):
            before = sum(float(np.sum(np.diag(M) ** 2)) for M in Ms)
            for p in range(d - 1):
                for q in range(p + 1, d):
                    u = np.array([(M[p, p] - M[q, q]) / 2.0 for M in Ms])
                    b = np.array([M[p, q] for M in Ms])
                    # under a rotation of the (p, q) plane by θ, M'_{pq} =
                    # b cos2θ − u sin2θ, so Σ(M'_{pq})² is
                    # ½(Σu² + Σb²) − ½(Σu² − Σb²)cos4θ − Σub sin4θ and is
                    # minimised at 4θ = atan2(2Σub, Σu² − Σb²). The ¼ is the
                    # subtle part: getting it wrong still decreases the
                    # criterion at every step, and still stalls short of the
                    # answer -- monotonically, and at the wrong rotation.
                    theta = 0.25 * math.atan2(2.0 * float(u @ b), float(u @ u - b @ b))
                    c, s = math.cos(theta), math.sin(theta)
                    R = np.eye(d)
                    R[p, p] = R[q, q] = c
                    R[p, q], R[q, p] = -s, s
                    Ms = [R.T @ M @ R for M in Ms]
                    V = V @ R
            self.n_iter_ += 1
            after = sum(float(np.sum(np.diag(M) ** 2)) for M in Ms)
            if after - before < self.tol * max(abs(after), 1.0):
                self.converged_ = True
                break
        return V.T                                    # sources = Z @ V

    # -- kernel ICA: dependence that is not a cumulant ---------------------- #
    def _kernel_contrast(self, Z: np.ndarray, U: np.ndarray) -> float:
        """Sum of normalised HSIC over all pairs of components under `Y = Z Uᵀ`.

        Each component gets a Gaussian kernel matrix, centred in feature space
        (`H K H`) and scaled to unit Frobenius norm; a pair is scored by the
        Frobenius inner product of the two, which is 0 for independent
        components and positive when they share structure. Unlike a cumulant
        this sees *all* the dependence, not only the fourth-order part, and
        unlike the kernel generalised variance it is a bounded correlation with
        no determinant to go singular.
        """
        Y = Z @ U.T
        n = Y.shape[0]
        H = np.eye(n) - 1.0 / n
        Ks = []
        for i in range(Y.shape[1]):
            D = (Y[:, i][:, None] - Y[:, i][None, :]) ** 2
            K = H @ np.exp(-0.5 * D / self.bandwidth_) @ H
            Ks.append(K / max(float(np.linalg.norm(K)), EPS))
        return float(sum(np.sum(Ks[i] * Ks[j])
                         for i in range(len(Ks)) for j in range(i + 1, len(Ks))))

    def _fit_kernel(self, Z: np.ndarray) -> np.ndarray:
        n, d = Z.shape
        rng = np.random.default_rng(self.seed)
        sub = Z[rng.choice(n, size=min(n, 300), replace=False)]
        self.kernel_sample_ = sub
        # one bandwidth, from the median off-diagonal pairwise distance of the
        # whitened sample: whitening has already fixed the scale, so a single
        # knob is the right number, and including the zero diagonal in that
        # median would set the bandwidth to 0 and the contrast to noise
        D = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=2)
        self.bandwidth_ = max(float(np.median(D[np.triu_indices(len(sub), 1)])), 1e-6)

        m = d * (d - 1) // 2
        angles = np.zeros(m)
        best = self._kernel_contrast(sub, _givens(d, angles).T)
        self.contrast_curve_.append(best)
        # coordinate descent, each coordinate by a coarse sweep of the circle
        # followed by two refinements around the winner. For d = 2 there is one
        # angle and no coupling, so this minimises the contrast exhaustively up
        # to the final grid step -- the search is not iterative in the sense the
        # gradient methods are.
        for _ in range(2):
            for a in range(m):
                lo, hi = -0.5 * math.pi, 0.5 * math.pi
                for level in range(3):
                    grid = np.linspace(lo, hi, 17 if level == 0 else 9)
                    vals = []
                    for g in grid:
                        trial = angles.copy()
                        trial[a] = float(g)
                        vals.append(self._kernel_contrast(sub, _givens(d, trial).T))
                    k = int(np.argmin(vals))
                    angles[a] = float(grid[k])
                    best = min(best, vals[k])
                    self.contrast_curve_.append(best)
                    self.n_iter_ += len(grid)          # contrast evaluations
                    step = float(grid[1] - grid[0])
                    lo, hi = angles[a] - step, angles[a] + step
        self.converged_ = True
        return _givens(d, angles).T


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _sources(n: int, d: int = 3, seed: int = 0):
    """Independent sources with different shapes, and a well-conditioned mixing."""
    rng = np.random.default_rng(seed)
    s = np.empty((n, d))
    s[:, 0] = rng.laplace(size=n)                     # heavy tails, kurtosis 3
    s[:, 1] = rng.uniform(-1.0, 1.0, size=n)          # light tails, kurtosis -1.2
    if d > 2:
        s[:, 2] = rng.standard_normal(n) * (1.0 + 0.35 * rng.laplace(size=n))
    s = (s - s.mean(axis=0)) / s.std(axis=0)
    if d > 3:
        s[:, 3] = np.sign(rng.standard_normal(n))     # bimodal, kurtosis −2
        if d > 4:
            s[:, 4:] = rng.standard_normal((n, d - 4))
    A = np.array([[1.0, 0.5, 0.2, 0.1], [0.3, 1.2, 0.4, 0.2],
                  [0.6, -0.3, 1.1, 0.3], [0.1, 0.4, -0.2, 1.0]])[:d, :d]
    return s, A, s @ A.T                              # the observed mixture


def _gaussians(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, 2))
    A = np.array([[1.0, 0.6], [0.4, 1.1]])
    return s, A, s @ A.T


def main() -> None:
    n, d = 4000, 3
    S_true, A, X = _sources(n, d, seed=0)

    # --- 1. whitening decorrelates, and that is all it does -----------------
    # cov(Z) = I to machine precision, and the projections are still dependent:
    # the correlations of the *squares* are large, and independence would make
    # them zero. That gap is what the other five variants fill.
    _, Ww, Z = _whiten(X)
    coup = float(np.max(np.abs(np.corrcoef((Z ** 2).T) - np.eye(d))))
    assert np.max(np.abs(np.cov(Z, rowvar=False, bias=True) - np.eye(d))) < 1e-12
    assert coup > 0.15, coup
    wh = ICA(variant="whitening").fit(X)
    assert np.max(np.abs(wh.components_ - Ww)) < 1e-12
    print(
        f"ica: whitening gives cov(Z) = I to "
        f"{np.max(np.abs(np.cov(Z, rowvar=False, bias=True) - np.eye(d))):.1e} while the "
        f"squared projections stay correlated at {coup:.2f} — uncorrelated, still "
        f"dependent"
    )

    # --- 2. the two FastICA forms agree, and beat whitening -----------------
    # Deflation and symmetric decorrelation are different iterations for the same
    # fixed point, so their unmixings have to coincide; and both have to be near
    # A^{-1}, which whitening is not.
    fd = ICA(d, variant="fastica_deflation", seed=1).fit(X)
    fs = ICA(d, variant="fastica_symmetric", seed=1).fit(X)
    err_fd = _align_error(fd.components_, A)
    err_fs = _align_error(fs.components_, A)
    err_wh = _align_error(wh.components_, A)
    agree = _disagreement(fd.components_, fs.components_)
    assert agree < 0.05, agree
    assert err_fs < 0.05 and err_fd < 0.05, (err_fs, err_fd)
    assert err_wh > 10 * max(err_fs, err_fd), (err_wh, err_fs)
    # and the fit attains essentially all the negentropy the true sources have:
    # the rotation it found is the one that makes the projections maximally
    # non-Gaussian, which is the whole criterion
    j_fit = _negentropy(fs.transform(X))
    j_true = _negentropy(S_true)
    assert j_fit > 0.95 * j_true, (j_fit, j_true)
    print(
        f"ica: fastica_deflation and fastica_symmetric land on the same unmixing to "
        f"{math.degrees(agree):.1f}° ({fd.n_iter_} and {fs.n_iter_} iterations), recover "
        f"the true directions to {math.degrees(err_fd):.2f}° and {math.degrees(err_fs):.2f}° "
        f"where whitening alone is {math.degrees(err_wh):.0f}° off, and reach {j_fit:.3f} "
        f"of the {j_true:.3f} negentropy the true sources carry"
    )

    # --- 3. infomax: the same answer, and why the switch matters ------------
    # A different objective (a likelihood, ascended by natural gradient), so
    # agreement is a real check on both. And the reason `extend=True` exists:
    # without the kurtosis switch, the logistic model forces every source to be
    # super-Gaussian, and the one uniform source is fitted as a mixture instead.
    inf = ICA(d, variant="infomax", seed=1).fit(X)
    err_inf = _align_error(inf.components_, A)
    dis = _disagreement(inf.components_, fs.components_)
    plain = ICA(d, variant="infomax", seed=1, extend=False).fit(X)
    err_plain = _align_error(plain.components_, A)
    k_inf = np.sort([_kurtosis(inf.transform(X)[:, i]) for i in range(d)])
    k_plain = np.sort([_kurtosis(plain.transform(X)[:, i]) for i in range(d)])
    assert err_inf < 0.05, err_inf
    assert dis < 0.05, dis
    assert inf.converged_ and inf.score_residual_ < 1e-6, inf.score_residual_
    assert float(np.max(np.abs(k_inf - np.sort([_kurtosis(S_true[:, i])
                                                for i in range(d)])))) < 0.2
    assert err_plain > 0.5, err_plain
    print(
        f"ica: infomax with the kurtosis switch reaches the FastICA unmixing to "
        f"{math.degrees(dis):.1f}° and the truth to {math.degrees(err_inf):.1f}°, "
        f"recovering kurtoses ({', '.join(f'{v:.1f}' for v in k_inf)}) in {inf.n_iter_} "
        f"natural-gradient steps, where the fixed-point condition E[ψ(u)uᵀ] = −I holds to "
        f"{inf.score_residual_:.0e}. Forcing every source super-Gaussian instead gives "
        f"({', '.join(f'{v:.1f}' for v in k_plain)}) and lands "
        f"{math.degrees(err_plain):.0f}° away"
    )

    # --- 4. JADE: diagonalising the fourth-order cumulant -------------------
    # Four sources with distinct kurtosis -- two of them strongly non-Gaussian,
    # with opposite sign, which is exactly the case that matters: a Gaussian
    # source has a zero cumulant and is invisible to this method. In the source
    # frame the cumulant operator is diagonal on every symmetric matrix, so a
    # rotation that diagonalises all of them at once is the unmixing.
    S4, A4, X4 = _sources(n, 4, seed=3)
    jd = ICA(4, variant="jade", seed=0).fit(X4)
    err_jd = _align_error(jd.components_, A4)
    Zs = (X4 - jd.mean_) @ jd.whitening_.T
    R = jd.rotation_
    off = tot = 0.0
    for i in range(4):
        for j in range(i, 4):
            E = np.zeros((4, 4))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / math.sqrt(2.0)   # orthonormal basis
            Qs = R @ jd._cumulant_apply(Zs, E) @ R.T        # the source frame
            off += float(np.linalg.norm(Qs - np.diag(np.diag(Qs))))
            tot += float(np.linalg.norm(Qs))
    assert err_jd < 0.1, err_jd
    assert off / tot < 0.15, off / tot
    kurt = np.sort([_kurtosis(S4[:, i]) for i in range(4)])
    kurt_hat = np.sort([_kurtosis(jd.transform(X4)[:, i]) for i in range(4)])
    assert np.max(np.abs(kurt - kurt_hat)) < 0.15, (kurt, kurt_hat)
    print(
        f"ica: jade recovers four sources with kurtoses "
        f"({', '.join(f'{v:.2f}' for v in kurt_hat)}) against the true "
        f"({', '.join(f'{v:.2f}' for v in kurt)}) to {math.degrees(err_jd):.1f}° in "
        f"{jd.n_iter_} Jacobi sweeps, leaving {off / tot:.1%} of the cumulant mass off "
        f"the diagonal"
    )

    # --- 5. kernel ICA: dependence that is not a cumulant -------------------
    # Two sources, the second nearly Gaussian (excess kurtosis −0.08). The pair
    # is still identifiable, because the first source is strongly non-Gaussian
    # and the rotation has to satisfy both -- FastICA handles this. JADE does
    # not: its whole criterion is one cumulant eigenvalue per source, and −0.08
    # is indistinguishable from the exact zero that whitening puts in the last
    # direction, so it picks the noise eigenvector and reports it as a source.
    # The kernel contrast never forms a cumulant; it measures dependence.
    rng = np.random.default_rng(11)
    S2 = np.empty((1500, 2))
    S2[:, 0] = rng.laplace(size=1500)
    S2[:, 1] = 0.55 * rng.uniform(-1.0, 1.0, size=1500) + 0.83 * rng.standard_normal(1500)
    S2 = (S2 - S2.mean(axis=0)) / S2.std(axis=0)
    A2 = np.array([[1.0, 0.5], [0.4, 1.1]])
    X2 = S2 @ A2.T
    kurt2 = _kurtosis(S2[:, 1])
    kk = ICA(2, variant="kernel_ica", seed=0).fit(X2)
    err_kk = _align_error(kk.components_, A2)
    err_j2 = _align_error(ICA(2, variant="jade", seed=0).fit(X2).components_, A2)
    # the fitted rotation really is the minimum of its own objective: an
    # independent 721-point scan of the circle on the same subsample finds
    # nothing lower. For two components there is one angle, so this is exact.
    sub = kk.kernel_sample_
    scan = [kk._kernel_contrast(sub, _givens(2, np.array([t])).T)
            for t in np.linspace(-0.5 * math.pi, 0.5 * math.pi, 721)]
    ref = float(np.min(scan))
    at_fit = kk._kernel_contrast(sub, kk.rotation_)
    assert err_kk < 0.1, err_kk
    assert at_fit <= ref * 1.01 + 1e-12, (at_fit, ref)
    assert err_j2 > 0.5, err_j2
    print(
        f"ica: on a pair whose second source has excess kurtosis {kurt2:.2f}, kernel_ica "
        f"recovers the mixing to {math.degrees(err_kk):.1f}° in {kk.n_iter_} contrast "
        f"evaluations, and its objective ({at_fit:.5f}) is an exhaustive 721-point scan's "
        f"minimum ({ref:.5f}); jade is {math.degrees(err_j2):.0f}° off on the same data — a "
        f"kurtosis of {kurt2:.2f} cannot be told from the exact zero whitening puts in the "
        f"last direction"
    )

    # --- 6. the boundary: two Gaussians cannot be separated -----------------
    # Every rotation of a spherical Gaussian is again a spherical Gaussian, so
    # the contrast is flat and the mixing matrix is not identifiable. More data
    # does not help, because there is nothing to estimate: the objective's
    # variation over the whole circle is smaller than its sampling noise.
    _, Ag, Xg = _gaussians(4000, seed=2)
    _, _, Zg = _whiten(Xg)
    vals = [_negentropy(Zg @ _givens(2, np.array([t])).T)
            for t in np.linspace(0.0, math.pi, 72, endpoint=False)]
    spread = float(max(vals) - min(vals))
    angs = set()
    for s in range(6):
        f = ICA(2, variant="fastica_symmetric", seed=s).fit(Xg)
        # modulo the 90° symmetry that swapping the two components induces
        angs.add(round(math.degrees(math.atan2(f.rotation_[1, 0], f.rotation_[0, 0])) % 90.0, 1))
    angs = sorted(angs)
    assert spread < 1e-3, spread
    assert len(angs) > 1 and angs[-1] - angs[0] > 1.0, angs
    print(
        f"ica: with two Gaussian sources the negentropy contrast moves by only {spread:.0e} "
        f"over the whole circle — less than the sampling noise of the fit — so six FastICA "
        f"seeds stop at {', '.join(f'{a:.1f}°' for a in angs)} (mod 90°). Each is a "
        f"stationary point of a flat objective, not an estimate of anything"
    )

    print("ica demo passed")


if __name__ == "__main__":
    main()
