"""PCA — the projection, six ways to compute it, and what each one is for.

Chapter:   06-latent-factorization
Companion: pca.md
Tier 1 · 7 variants: full_svd, randomized, kernel_pca, incremental,
                     robust_pca, sparse_pca, whitening

One objective, several algorithms:

    max_{V^T V = I}  tr(V^T Σ V),      Σ = (1/(n−1)) X_c^T X_c

The maximiser is the top-k eigenvectors of the covariance, and the answer is
unique up to rotation *within* a repeated eigenvalue and sign. Every variant
below returns the same object — except where the variant changes the objective
or the space it lives in:

    full_svd      the definition: centre, SVD, truncate
    randomized    the same answer from a random sketch, O(ndk) not O(nd²)
    kernel_pca    the same idea in a feature space where Φ(x) replaces x
    incremental   the same answer, updated one block of rows at a time
    robust_pca    low-rank + sparse, so a few corrupted cells cannot rotate V
    sparse_pca    a relaxed objective with an L1 penalty: loadings with zeros
    whitening     a post-processing: rescale so the covariance is exactly I

The demo is built on identities rather than on pictures: orthonormality of V,
the reconstruction error equal to the sum of the discarded eigenvalues, the
randomized sketch agreeing with the exact SVD to 1e-10, kernel PCA reproducing
the eigendecomposition of the centred Gram matrix, the incremental factor
agreeing with a full refactorisation, robust PCA separating a low-rank matrix
from sparse corruption that moves plain PCA, sparse loadings with exact zeros
that still explain the variance, and a whitened transform whose covariance is
the identity to 1e-10.

Ties are broken deterministically: each component is sign-flipped so that its
largest-magnitude entry is positive. Without that, two runs of the same SVD can
differ by a sign per component depending on the LAPACK build.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["full_svd", "randomized", "kernel_pca", "incremental",
                      "robust_pca", "sparse_pca", "whitening"])


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _flip(V: np.ndarray) -> np.ndarray:
    """Sign-flip rows so the largest-magnitude entry of each is positive."""
    idx = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[np.arange(V.shape[0]), idx])
    signs[signs == 0.0] = 1.0
    return V * signs[:, None]


def _svt(A: np.ndarray, tau: float) -> np.ndarray:
    """Singular-value soft threshold: the proximal operator of the nuclear norm."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vt


def _soft(A: np.ndarray, tau: float) -> np.ndarray:
    """Entrywise soft threshold: the proximal operator of the L1 norm."""
    return np.sign(A) * np.maximum(np.abs(A) - tau, 0.0)


def _reduce(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, SVD, truncate; return `(mean, components, singular_values)`."""
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, s.size)
    return mean, _flip(Vt[:k]), s[:k]


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class PCA:
    """Principal components, fitted by one of seven algorithms.

    `X` is `(n, d)`; `n_components=k` gives `components_` of shape `(k, d)`,
    except for `kernel_pca`, where it is `(k, n_fit)` dual coefficients —
    there is no finite vector in input space to store.
    """

    def __init__(self, n_components: int = 2, *, variant: str = "full_svd", seed: int = 0,
                 gamma: float | None = None, lam: float = 0.1, oversample: int = 10,
                 power_iter: int = 2, block_size: int | None = None,
                 mean: np.ndarray | None = None, n_iter: int = 200, tol: float = 1e-7):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = int(n_components)
        self.variant = variant
        self.seed = int(seed)
        self.gamma = gamma
        self.lam = float(lam)
        self.oversample = int(oversample)
        self.power_iter = int(power_iter)
        self.block_size = block_size
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.n_iter = int(n_iter)
        self.tol = float(tol)

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.noise_variance_: float = 0.0
        self.n_samples_seen_: int = 0
        self.n_features_in_: int | None = None

    # -- the public surface ------------------------------------------------ #
    def fit(self, X) -> PCA:
        X = self._check(X)
        v = self.variant
        if v in ("full_svd", "whitening"):
            self._fit_full(X)
        elif v == "randomized":
            self._fit_randomized(X)
        elif v == "kernel_pca":
            self._fit_kernel(X)
        elif v == "incremental":
            self.n_samples_seen_ = 0
            self.mean_ = None if self.mean is None else self.mean.copy()
            block = self.block_size or max(1, X.shape[0])
            for start in range(0, X.shape[0], block):
                self.partial_fit(X[start:start + block])
        elif v == "robust_pca":
            self._fit_robust(X)
        else:
            self._fit_sparse(X)
        self._set_variance_ratios()
        return self

    def partial_fit(self, X) -> PCA:
        """Add a block of rows. Only `incremental` supports this."""
        if self.variant != "incremental":
            raise ValueError(f"partial_fit is only meaningful for variant='incremental', "
                             f"not {self.variant!r}")
        X = self._check(X)
        if self.mean_ is None:
            self.mean_ = self.mean.copy() if self.mean is not None else X.mean(axis=0)
        Xc = X - self.mean_
        k = self.n_components
        if self.n_samples_seen_ == 0:
            # keep the whole range of the first block, not just k: an
            # incremental fit that truncates on block 1 can never recover a
            # direction it discarded, and the update is only exact otherwise
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._V, self._S = _flip(Vt).T, S
        else:
            V, S = self._V, self._S                 # d×r, r
            kc = V.shape[1]
            P = Xc @ V                              # b×r: what the old basis sees
            R = Xc - P @ V.T                        # b×d: what it does not
            Q, Rq = np.linalg.qr(R.T)               # Q: d×l, Rq: l×b
            lq = Q.shape[1]
            M = np.zeros((kc + Xc.shape[0], kc + lq))
            M[:kc, :kc] = np.diag(S)
            M[kc:, :kc] = P
            M[kc:, kc:] = Rq.T
            _, S2, V2t = np.linalg.svd(M, full_matrices=False)
            k2 = min(S2.size, Xc.shape[1])
            big = np.zeros((Xc.shape[1], kc + lq))
            big[:, :kc] = V
            big[:, kc:] = Q
            self._V, self._S = big @ V2t[:k2].T, S2[:k2]
        self.components_ = _flip(self._V[:, :k].T)
        self.singular_values_ = self._S[:k]
        self.n_samples_seen_ += X.shape[0]
        self._set_variance_ratios()
        return self

    def transform(self, X) -> np.ndarray:
        """Project `(m, d)` onto the fitted components: `(m, k)`."""
        if self.components_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X)
        if self.variant == "kernel_pca":
            K = self._rbf(X, self.X_fit_)
            Kc = (K - K.mean(axis=1, keepdims=True) - self._col_mean_
                  + self._grand_mean_)
            return Kc @ self.components_.T
        Z = (X - self.mean_) @ self.components_.T
        if self.variant == "whitening":
            Z = Z / np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        """Map `(m, k)` back to input space. Kernel PCA has no exact preimage."""
        if self.variant == "kernel_pca":
            raise NotImplementedError(
                "kernel PCA has no exact preimage in input space; a preimage can be "
                "approximated by gradient descent on ||Φ(x) − z V^T||², which is not "
                "the same object and not implemented here"
            )
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.components_.shape[0]:
            raise ValueError(f"expected (m, {self.components_.shape[0]}), got {Z.shape}")
        if self.variant == "whitening":
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_

    # -- checks and bookkeeping -------------------------------------------- #
    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        if X.shape[0] < 2:
            raise ValueError("need at least two rows")
        return X

    def _set_variance_ratios(self) -> None:
        if self.singular_values_ is None or self.variant == "kernel_pca":
            return
        n = max(self.n_samples_seen_ or 0, 2)
        var = self.singular_values_ ** 2 / (n - 1)
        self.explained_variance_ = var
        total = var.sum()
        self.explained_variance_ratio_ = var / total if total > 0 else var

    # -- full SVD ----------------------------------------------------------- #
    def _fit_full(self, X) -> None:
        self.mean_, self.components_, self.singular_values_ = _reduce(X, self.n_components)
        self.n_samples_seen_ = X.shape[0]
        self.noise_variance_ = self._noise(X)

    def _noise(self, X) -> float:
        """Mean of the discarded variances: the isotropic part of the residual."""
        n, d = X.shape
        total = np.linalg.norm(X - self.mean_, "fro") ** 2 / max(n - 1, 1)
        kept = float(np.sum(self.singular_values_ ** 2) / max(n - 1, 1))
        rank = min(n - 1, d)
        return max(total - kept, 0.0) / max(rank - len(self.singular_values_), 1)

    # -- randomized range finder ------------------------------------------- #
    def _fit_randomized(self, X) -> None:
        """Halko–Martinsson–Tropp: sketch the range, then exactly SVD the sketch."""
        n, d = X.shape
        k = self.n_components
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        rng = np.random.default_rng(self.seed)
        width = min(k + self.oversample, d, n)
        Q, _ = np.linalg.qr(Xc @ rng.standard_normal((d, width)))
        for _ in range(self.power_iter):            # sharpen the spectrum
            Q, _ = np.linalg.qr(Xc.T @ Q)
            Q, _ = np.linalg.qr(Xc @ Q)
        B = Q.T @ Xc
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        k = min(k, s.size)
        self.components_ = _flip(Vt[:k])
        self.singular_values_ = s[:k]
        self.n_samples_seen_ = n
        self.noise_variance_ = self._noise(X)

    # -- kernel PCA --------------------------------------------------------- #
    def _fit_kernel(self, X) -> None:
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        self.n_samples_seen_ = n
        self.X_fit_ = X
        self.gamma = 1.0 / d if self.gamma is None else float(self.gamma)
        K = self._rbf(X, X)
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K @ H
        w, A = np.linalg.eigh(Kc)
        order = np.argsort(w)[::-1][: self.n_components]
        w, A = np.maximum(w[order], 0.0), A[:, order]
        keep = w > 1e-10 * max(w.max(), 1.0)
        self.eigenvalues_ = w[keep]
        self.components_ = _flip((A[:, keep] / np.sqrt(self.eigenvalues_)).T)
        self.singular_values_ = np.sqrt(self.eigenvalues_ * max(n - 1, 1))
        self._col_mean_ = K.mean(axis=0)[None, :]
        self._grand_mean_ = float(K.mean())

    def _rbf(self, A, B) -> np.ndarray:
        d2 = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
              - 2.0 * A @ B.T)
        return np.exp(-self.gamma * np.maximum(d2, 0.0))

    # -- robust PCA --------------------------------------------------------- #
    def _fit_robust(self, X) -> None:
        """IALM for X = L + S: nuclear norm on L, L1 on S, exact on X."""
        n, d = X.shape
        self.mean_ = np.zeros(d)                     # the model absorbs the offset into L
        lam = 1.0 / math.sqrt(max(n, d))
        mu = 1.25 / max(np.abs(X).max(), 1e-12)
        Y = np.zeros_like(X)
        S = np.zeros_like(X)
        L = X.copy()
        iters = 0
        for _ in range(self.n_iter):
            iters += 1
            L = _svt(X - S + Y / mu, 1.0 / mu)
            S = _soft(X - L + Y / mu, lam / mu)
            residual = X - L - S
            Y += mu * residual
            mu = min(mu * 1.5, 1e12)
            if np.linalg.norm(residual) <= self.tol * np.linalg.norm(X):
                break
        self.low_rank_ = L
        self.sparse_ = S
        _, self.components_, self.singular_values_ = _reduce(L, self.n_components)
        self.n_samples_seen_ = n
        self.n_iter_used_ = iters

    # -- sparse PCA --------------------------------------------------------- #
    def _fit_sparse(self, X) -> None:
        """Soft-thresholded power iteration with deflation (Witten et al. style)."""
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        rng = np.random.default_rng(self.seed)
        comps = np.zeros((min(self.n_components, d), d))
        for c in range(len(comps)):
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            for _ in range(self.n_iter):
                u = Xc.T @ (Xc @ v)
                u = _soft(u, self.lam * np.abs(u).max())
                nrm = np.linalg.norm(u)
                if nrm < 1e-12:
                    break
                v_new = u / nrm
                if np.linalg.norm(v_new - v) < self.tol:
                    v = v_new
                    break
                v = v_new
            comps[c] = v
            Xc = Xc - np.outer(Xc @ v, v)           # deflate one rank-one term
        self.components_ = _flip(comps)
        self.singular_values_ = np.linalg.norm(
            (X - self.mean_) @ self.components_.T, axis=0)
        self.n_samples_seen_ = n


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _low_rank(n: int = 400, d: int = 10, k: int = 3, noise: float = 0.05, seed: int = 0):
    """A rank-`k` signal in `d` dimensions, plus small isotropic noise."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, d))
    Z = rng.standard_normal((n, k))
    return Z @ W + noise * rng.standard_normal((n, d)), W, Z


def _angle(U: np.ndarray, V: np.ndarray) -> float:
    """Smallest principal angle between two row spaces, via the cosines."""
    Q1, _ = np.linalg.qr(U.T)
    Q2, _ = np.linalg.qr(V.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


def main() -> None:
    X, W, Z = _low_rank()

    # --- 1. the definition, and the identities that pin it down ----------
    # The reconstruction error is exactly the sum of the eigenvalues that were
    # thrown away, which is the statement that PCA is a projection and not a
    # compression heuristic.
    p = PCA(3, variant="full_svd").fit(X)
    assert p.components_.shape == (3, X.shape[1])
    assert abs(np.linalg.norm(p.components_ @ p.components_.T - np.eye(3))) < 1e-12
    Xc = X - p.mean_
    ev = np.linalg.eigvalsh(Xc.T @ Xc / (len(X) - 1))[::-1]
    assert np.allclose(p.explained_variance_, ev[:3], rtol=1e-10), (p.explained_variance_, ev[:3])
    Zp = p.transform(X)
    err = np.linalg.norm(X - p.inverse_transform(Zp)) ** 2
    assert abs(err / (len(X) - 1) - ev[3:].sum()) < 1e-8 * ev.sum()
    # and the projection is orthogonal: the residual is orthogonal to the basis
    residual = Xc - Zp @ p.components_
    assert np.max(np.abs(residual @ p.components_.T)) < 1e-10
    # a second fit gives the identical answer, signs included
    again = PCA(3, variant="full_svd").fit(X)
    assert np.array_equal(p.components_, again.components_)
    print(
        f"pca: 3 components explain {p.explained_variance_ratio_.sum():.4f} of the "
        f"variance, V is orthonormal to 1e-12, the residual is orthogonal to V to "
        f"1e-10, and the discarded variance is {err / (len(X) - 1):.6f}"
    )

    # --- 2. randomized: the same subspace from a sketch -------------------
    # Two power iterations are what makes the sketch see the top of the
    # spectrum rather than the bulk; without them the angle is ~1e-2.
    full = PCA(3, variant="full_svd").fit(X)
    rnd = PCA(3, variant="randomized", seed=0, power_iter=2).fit(X)
    assert _angle(full.components_, rnd.components_) < 1e-7
    assert np.allclose(full.singular_values_, rnd.singular_values_, rtol=1e-9)
    assert np.allclose(full.explained_variance_ratio_, rnd.explained_variance_ratio_, rtol=1e-8)
    # on a wide, low-rank matrix the sketch is exact once the range is captured
    assert np.linalg.matrix_rank(Xc @ rnd.components_.T) == 3
    print(
        f"pca: randomized sketch reproduces the exact singular values to "
        f"{np.max(np.abs(rnd.singular_values_ / full.singular_values_ - 1)):.1e} and the "
        f"subspace to {_angle(full.components_, rnd.components_):.1e} rad"
    )

    # --- 3. kernel PCA: a rotation cannot separate two moons --------------
    # Linear PCA is an orthogonal rotation, so a linear classifier on its scores
    # is a linear classifier on the raw data -- hopeless on interleaving moons.
    # The RBF kernel's first component is a nonlinear function of x, and one
    # threshold on it separates the classes.
    rng = np.random.default_rng(1)
    m = 200
    t1 = np.linspace(0.0, np.pi, m)
    moons = np.vstack([np.column_stack([np.cos(t1), np.sin(t1)]),
                       np.column_stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)])])
    y = np.r_[np.zeros(m), np.ones(m)]
    Xm = moons + 0.06 * rng.standard_normal(moons.shape)

    def fisher(z: np.ndarray) -> float:
        return float((z[y == 0].mean() - z[y == 1].mean()) ** 2
                     / (z[y == 0].var() + z[y == 1].var()))

    kp = PCA(2, variant="kernel_pca", gamma=10.0).fit(Xm)
    lin = PCA(2, variant="full_svd").fit(Xm)
    f_k = fisher(kp.transform(Xm)[:, 0])
    f_l = max(fisher(lin.transform(Xm)[:, 0]), fisher(lin.transform(Xm)[:, 1]))
    assert f_k > 5.0 and f_k > 4.0 * f_l, (f_k, f_l)
    # the algebra: transform on the training set is eigenvectors × sqrt(eigenvalues)
    gram = kp._rbf(Xm, Xm)
    H = np.eye(len(Xm)) - np.ones((len(Xm), len(Xm))) / len(Xm)
    w, A = np.linalg.eigh(H @ gram @ H)
    order = np.argsort(w)[::-1][:2]
    assert np.allclose(kp.eigenvalues_, np.maximum(w[order], 0.0), rtol=1e-10)
    got = kp.transform(Xm)
    assert np.allclose(np.abs(got), np.abs(A[:, order] * np.sqrt(np.maximum(w[order], 0.0))),
                       atol=1e-9)
    try:
        kp.inverse_transform(got)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("kernel PCA must refuse an exact preimage")
    print(
        f"pca: kernel PCA separates the two moons with a Fisher ratio of {f_k:.2f} on "
        f"one component, where the best linear component reaches {f_l:.2f}; its scores "
        f"reproduce eigh of the centred Gram matrix to 1e-9"
    )

    # --- 4. incremental: the same factorisation, block by block -----------
    # The update is exact, not a gradient step: [X; B] is factored through a
    # small matrix built from the old factors and the new block's residual.
    inc = PCA(3, variant="incremental", block_size=50, mean=X.mean(axis=0)).fit(X)
    assert _angle(full.components_, inc.components_) < 1e-7
    assert np.allclose(full.singular_values_, inc.singular_values_, rtol=1e-9)
    # and its transform agrees with a refactorisation of the same data
    ref = PCA(3, variant="full_svd").fit(X)
    assert np.allclose(np.abs(inc.transform(X)), np.abs(ref.transform(X)), atol=1e-7)
    # the mean is the one thing a block update cannot improve for free: with no
    # mean given it freezes the first block's, and 50 rows are not 400
    drift = PCA(3, variant="incremental", block_size=50).fit(X)
    assert _angle(full.components_, drift.components_) > 1e-4
    print(
        f"pca: eight incremental blocks reproduce the full SVD to "
        f"{np.max(np.abs(inc.singular_values_ / full.singular_values_ - 1)):.1e} in the "
        f"singular values, in {inc.n_samples_seen_} rows"
    )

    # --- 5. robust PCA: separate the low-rank part from corruption -------
    # PCA minimises squared error, so one cell of magnitude 20 can tilt a
    # component. The nuclear-norm + L1 decomposition puts that cell in S.
    rng = np.random.default_rng(2)
    L0 = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 40))
    S0 = np.zeros_like(L0)
    idx = rng.choice(L0.size, size=L0.size // 50, replace=False)
    S0.flat[idx] = rng.choice([-10.0, 10.0], size=idx.size)
    Xr = L0 + S0
    rob = PCA(5, variant="robust_pca").fit(Xr)
    plain = PCA(5, variant="full_svd").fit(Xr)
    e_rob = np.linalg.norm(rob.low_rank_ - L0) / np.linalg.norm(L0)
    e_plain = np.linalg.norm(plain.inverse_transform(plain.transform(Xr)) - L0) / np.linalg.norm(L0)
    support = np.abs(rob.sparse_) > 0.5
    hit = np.mean(support.ravel()[idx])
    assert e_rob < 0.05 and e_plain > 0.1, (e_rob, e_plain)
    assert hit > 0.95
    # the decomposition is exact to the tolerance it stopped at
    assert np.linalg.norm(Xr - rob.low_rank_ - rob.sparse_) / np.linalg.norm(Xr) < 1e-6
    print(
        f"pca: robust PCA recovers the low-rank matrix to {e_rob:.1e} where PCA's "
        f"best rank-5 fit is off by {e_plain:.1e}, finds {hit:.0%} of the corrupted "
        f"cells, and leaves a residual of "
        f"{np.linalg.norm(Xr - rob.low_rank_ - rob.sparse_) / np.linalg.norm(Xr):.1e}"
    )

    # --- 6. sparse PCA: zeros in the loadings, variance still explained ---
    # The data is generated from a basis with four non-zeros per component, so
    # a method that finds zeros is finding the truth, not a constraint.
    rng = np.random.default_rng(3)
    d, k = 12, 3
    Wt = np.zeros((k, d))
    for j in range(k):
        Wt[j, 4 * j:4 * j + 4] = rng.standard_normal(4) + 2.0
    Xs = rng.standard_normal((600, k)) @ Wt + 0.01 * rng.standard_normal((600, d))
    sp = PCA(k, variant="sparse_pca", lam=0.12, seed=0).fit(Xs)
    dense = PCA(k, variant="full_svd").fit(Xs)
    zeros = np.mean(np.abs(sp.components_) < 1e-12)
    # every true component is matched by one estimate with cosine > 0.98
    cos = np.abs(sp.components_ @ Wt.T) / (
        np.linalg.norm(sp.components_, axis=1)[:, None] * np.linalg.norm(Wt, axis=1)[None, :])
    assert cos.max(axis=1).min() > 0.98, cos
    assert zeros > 0.5
    # deflation keeps the explained variance close to dense PCA's
    assert (sp.singular_values_ ** 2).sum() > 0.9 * (dense.singular_values_ ** 2).sum()
    print(
        f"pca: sparse PCA zeroes {zeros:.0%} of its loadings, matches every true "
        f"component at cosine > {cos.max(axis=1).min():.3f}, and keeps "
        f"{(sp.singular_values_ ** 2).sum() / (dense.singular_values_ ** 2).sum():.0%} "
        f"of the dense explained variance"
    )

    # --- 7. whitening: the covariance becomes the identity ----------------
    # Whitening is PCA plus a diagonal rescale. It is the only variant whose
    # transform is not a projection: it is invertible on the k-subspace.
    wh = PCA(3, variant="whitening").fit(X)
    Zw = wh.transform(X)
    cov = np.cov(Zw, rowvar=False)
    assert np.max(np.abs(cov - np.eye(3))) < 1e-10, cov
    back = wh.inverse_transform(Zw)
    rank3 = full.inverse_transform(full.transform(X))
    assert np.max(np.abs(back - rank3)) < 1e-9
    # the components themselves are the same as plain PCA's
    assert np.max(np.abs(wh.components_ - full.components_)) < 1e-12
    print(
        f"pca: whitened scores have covariance I to "
        f"{np.max(np.abs(cov - np.eye(3))):.1e}, the transform is invertible to "
        f"{np.max(np.abs(back - rank3)):.1e} on the 3-subspace, and the components "
        f"are unchanged"
    )

    print("pca demo passed")


if __name__ == "__main__":
    main()
