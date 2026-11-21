"""PPCA — PCA as a latent-variable model, which is what gives it a likelihood.

Chapter:   06-latent-factorization
Companion: probabilistic_pca.md
Tier 2 · 4 variants: ppca_em, ppca_closed, missing_data, mixture_ppca

    x = W z + μ + ε,    z ~ N(0, I_k),    ε ~ N(0, σ² I_d)          (1)
    x ~ N(μ, C),        C = W Wᵀ + σ² I                             (2)

PCA is (2) with σ² → 0 and W's columns constrained to be orthogonal. Dropping
both constraints costs nothing in the mean and buys three things a projection
cannot offer:

    a likelihood      so models with different k are comparable
    a noise estimate  σ² is fitted, not assumed away
    a conditional     E[z | x_obs] is well defined when x is partly missing

Three of the variants are one model reached three ways; the fourth is several
copies of it:

    ppca_em         EM on the observed likelihood
    ppca_closed     the closed form: top-k eigenvectors minus σ²
    missing_data    EM where the E-step conditions on the observed block only
    mixture_ppca    a mixture of PPCA components, fitted by EM

The demo is built on exact statements rather than pictures: the EM and closed
forms agree in the covariance (W is only identified up to a rotation, C is
identified), the fitted spectrum is exactly (λ₁…λ_k, σ²…σ²), σ² is the mean of
the discarded eigenvalues, the column space of W is PCA's subspace, missing
entries are recovered better than by mean imputation, the held-out likelihood
peaks at the true rank, a mixture separates two subspaces that one PPCA cannot,
the posterior mean reconstructs to (d−k)σ² -- a denoiser, not a copy -- and
20000 draws from the fitted model reproduce C on its entrywise scale.

The MLE uses the biased covariance S = Xcᵀ Xc / n, not the unbiased one: the
likelihood is a function of the second moment about the fitted mean, and mixing
in n−1 makes the closed form disagree with EM at the third decimal.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["ppca_em", "ppca_closed", "missing_data", "mixture_ppca"])


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _top_eigs(S: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix."""
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), V[:, order]


def _logpdf(Xc: np.ndarray, W: np.ndarray, sigma2: float) -> np.ndarray:
    """Log density of `N(0, W Wᵀ + σ² I)` at the rows of the centred `Xc`.

    Woodbury, so that a d×d determinant and inverse never appear:

        log|C|      = (d−k) log σ² + log|WᵀW + σ² I|
        xᵀC⁻¹x      = (‖x‖² − ‖(WᵀW + σ² I)⁻¹ Wᵀx‖²_{WᵀW+σ² I}) / σ²
    """
    n, d = Xc.shape
    k = W.shape[1]
    M = W.T @ W + sigma2 * np.eye(k)
    _, logdet_M = np.linalg.slogdet(M)
    logdet = (d - k) * math.log(sigma2) + logdet_M
    P = Xc @ W                                   # (n, k)
    quad = (np.sum(Xc * Xc, axis=1) - np.sum((P @ np.linalg.inv(M)) * P, axis=1)) / sigma2
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class PPCA:
    """Probabilistic PCA, fitted by one of four algorithms.

    `W_` is `(d, k)` — it maps the latent to the observed space, so it is the
    transpose of the `components_` used in `pca.py`; both are exposed.
    """

    def __init__(self, n_components: int = 1, *, variant: str = "ppca_em",
                 n_mixtures: int = 1, max_iter: int = 400, tol: float = 1e-9,
                 seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if variant == "mixture_ppca" and n_mixtures < 1:
            raise ValueError("a mixture needs at least one component")
        self.n_components = int(n_components)
        self.variant = variant
        self.n_mixtures = int(n_mixtures)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)

        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.sigma2_: float = 0.0
        self.loglik_: float = -math.inf
        self.loglik_curve_: list[float] = []
        self.n_iter_: int = 0
        self.n_features_in_: int | None = None
        # mixture parameters, set only by mixture_ppca
        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.Ws_: list[np.ndarray] = []
        self.sigma2s_: list[float] = []

    # -- public surface ----------------------------------------------------- #
    @property
    def components_(self) -> np.ndarray:
        """`(k, d)` loadings, the same orientation as `pca.PCA.components_`."""
        if self.variant == "mixture_ppca":
            raise NotImplementedError("a mixture has one loading per component; see Ws_")
        if self.W_ is None:
            raise RuntimeError("call fit first")
        return self.W_.T

    @property
    def covariance_(self) -> np.ndarray:
        """The fitted `C = W Wᵀ + σ² I` of the single-component model."""
        if self.variant == "mixture_ppca":
            raise NotImplementedError("a mixture has one covariance per component")
        if self.W_ is None:
            raise RuntimeError("call fit first")
        return self.W_ @ self.W_.T + self.sigma2_ * np.eye(self.W_.shape[0])

    def fit(self, X) -> PPCA:
        X = self._check(X)
        if self.variant == "ppca_closed":
            self._fit_closed(X)
        elif self.variant == "missing_data":
            self._fit_missing(X)
        elif self.variant == "mixture_ppca":
            self._fit_mixture(X)
        else:
            self._fit_em(X)
        return self

    def transform(self, X) -> np.ndarray:
        """Posterior mean `E[z | x]`, shape `(m, k)`. NaNs are missing coordinates."""
        if self.variant == "mixture_ppca":
            raise NotImplementedError(
                "a mixture has one latent per component; use predict_proba for the "
                "responsibilities, or transform with a single-component model"
            )
        if self.W_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X, fit=False)
        Ez, _ = self._posterior(X, self.W_, self.sigma2_, self.mean_)
        return Ez

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def reconstruct(self, X) -> np.ndarray:
        """`E[x | x_obs]` — the denoised rows, with missing coordinates filled in."""
        X = self._check(X, fit=False)
        if self.variant == "mixture_ppca":
            raise NotImplementedError("reconstruct is implemented for a single component")
        Ez, _ = self._posterior(X, self.W_, self.sigma2_, self.mean_)
        return self.mean_ + Ez @ self.W_.T

    def score_samples(self, X) -> np.ndarray:
        """Per-row log likelihood under the fitted model."""
        X = self._check(X, fit=False)
        if self.variant == "mixture_ppca":
            logp = np.column_stack([
                np.log(self.weights_[c]) + _logpdf(X - self.means_[c], self.Ws_[c],
                                                   self.sigma2s_[c])
                for c in range(self.n_mixtures)])
            mx = logp.max(axis=1, keepdims=True)
            return (mx + np.log(np.exp(logp - mx).sum(axis=1, keepdims=True)))[:, 0]
        return _logpdf(X - self.mean_, self.W_, self.sigma2_)

    def predict_proba(self, X) -> np.ndarray:
        """Responsibilities `(m, n_mixtures)`."""
        if self.variant != "mixture_ppca":
            raise ValueError("predict_proba is only defined for variant='mixture_ppca'")
        logp = np.column_stack([
            np.log(self.weights_[c]) + _logpdf(X - self.means_[c], self.Ws_[c],
                                               self.sigma2s_[c])
            for c in range(self.n_mixtures)])
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw `n` rows from the fitted model — a check that it is generative."""
        if self.variant == "mixture_ppca":
            raise NotImplementedError("sample from a mixture needs the component labels")
        if self.W_ is None:
            raise RuntimeError("call fit first")
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, self.n_components))
        return (self.mean_ + Z @ self.W_.T
                + math.sqrt(self.sigma2_) * rng.standard_normal((n, self.W_.shape[0])))

    # -- validation --------------------------------------------------------- #
    def _check(self, X, fit: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        has_nan = bool(np.isnan(X).any())
        if has_nan and self.variant != "missing_data":
            raise ValueError(
                f"{self.variant} needs a complete matrix; variant='missing_data' is the "
                f"one that conditions on the observed block, and it treats NaN as missing")
        if fit and X.shape[0] < 2:
            raise ValueError("need at least two rows to fit")
        return X

    # -- EM on a complete matrix -------------------------------------------- #
    def _fit_em(self, X) -> None:
        n, d = X.shape
        k = min(self.n_components, d)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        S = Xc.T @ Xc / n
        lam, V = _top_eigs(S, k)
        # a sensible start: the closed form with a shrunk noise estimate
        sigma2 = max(float(lam[k:].mean()) if d > k else 0.1 * float(lam[0]), 1e-8)
        W = V[:, :k] * np.sqrt(np.maximum(lam[:k] - sigma2, 1e-6))
        self.loglik_curve_ = []
        for it in range(1, self.max_iter + 1):
            M = W.T @ W + sigma2 * np.eye(k)
            Minv = np.linalg.inv(M)
            Ez = Xc @ W @ Minv                          # (n, k)
            Covz = sigma2 * Minv                        # same for every row
            Ezz = Ez.T @ Ez + n * Covz                  # Σ_i E[z_i z_iᵀ]
            W_new = np.linalg.solve(Ezz, Ez.T @ Xc).T   # (d, k)
            resid = Xc - Ez @ W_new.T
            # Covz is already σ²M⁻¹, so the posterior spread enters once
            ssr = (float(np.sum(resid * resid))
                   + n * float(np.trace(Covz @ (W_new.T @ W_new))))
            sigma2_new = max(ssr / (n * d), 1e-12)
            W, sigma2 = W_new, sigma2_new
            ll = float(np.sum(_logpdf(Xc, W, sigma2)))
            self.loglik_curve_.append(ll)
            self.n_iter_ = it
            if it > 1 and abs(ll - self.loglik_curve_[-2]) <= self.tol * abs(ll):
                break
        self.W_, self.sigma2_ = W, sigma2
        self.loglik_ = self.loglik_curve_[-1]

    # -- the closed form ---------------------------------------------------- #
    def _fit_closed(self, X) -> None:
        n, d = X.shape
        k = min(self.n_components, d)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        lam, V = _top_eigs(Xc.T @ Xc / n, k)
        sigma2 = float(lam[k:].mean()) if d > k else 0.0
        self.W_ = V[:, :k] * np.sqrt(np.maximum(lam[:k] - sigma2, 0.0))
        self.sigma2_ = max(sigma2, 1e-12)
        self.loglik_ = float(np.sum(_logpdf(Xc, self.W_, self.sigma2_)))
        self.loglik_curve_ = [self.loglik_]
        self.n_iter_ = 0

    # -- missing data: condition on the observed block ---------------------- #
    def _condition(self, X, W, sigma2, mean):
        """Per-row `z | x_obs` moments and observed log density, batched.

        With a per-row observed mask `m_i`, the k×k system is
        `M_i = Wᵀ diag(m_i) W + σ² I`, so all n of them are one `einsum` and one
        batched inverse instead of a Python loop. NaN is missing.
        """
        k = W.shape[1]
        m = (~np.isnan(X)).astype(float)
        Xf = np.where(m > 0.0, X - mean, 0.0)          # missing coords contribute 0
        M = sigma2 * np.eye(k) + np.einsum("dj,id,dk->ijk", W, m, W)
        Minv = np.linalg.inv(M)                        # (n, k, k)
        P = Xf @ W                                     # Wᵀ x_i, on observed coords
        Ez = np.einsum("ijk,ik->ij", Minv, P)
        Cov = sigma2 * Minv
        # the same Woodbury identity gives the density of the observed block
        nobs = m.sum(axis=1)
        _, logdet_M = np.linalg.slogdet(M)
        logdet = np.log(sigma2) * (nobs - k) + logdet_M
        quad = (np.sum(Xf * Xf, axis=1) - np.einsum("ij,ik,ijk->i", P, P, Minv)) / sigma2
        logp = -0.5 * (nobs * math.log(2.0 * math.pi) + logdet + quad)
        return Ez, Cov, logp

    def _posterior(self, X, W, sigma2, mean) -> tuple[np.ndarray, np.ndarray]:
        """Per-row mean and covariance of `z | x_obs`, with NaN as missing."""
        Ez, Cov, _ = self._condition(X, W, sigma2, mean)
        return Ez, Cov

    def _loglik_missing(self, X, W, sigma2, mean) -> np.ndarray:
        """Per-row log likelihood of the *observed* coordinates only."""
        return self._condition(X, W, sigma2, mean)[2]

    def _fit_missing(self, X) -> None:
        n, d = X.shape
        k = min(self.n_components, d)
        mask = ~np.isnan(X)
        self.mean_ = np.where(mask.any(axis=0), np.nanmean(X, axis=0), 0.0)
        mean = self.mean_
        n_obs = int(mask.sum())
        # start from the closed form on the mean-imputed matrix
        filled = np.where(mask, X, mean)
        lam, V = _top_eigs((filled - mean).T @ (filled - mean) / n, k)
        sigma2 = float(lam[k:].mean()) if d > k else 0.1 * float(lam[0])
        W = V[:, :k] * np.sqrt(np.maximum(lam[:k] - sigma2, 1e-6))
        self.loglik_curve_ = []
        for it in range(1, self.max_iter + 1):
            Ez, Cov = self._posterior(X, W, sigma2, mean)      # (n,k), (n,k,k)
            second = Ez[:, :, None] * Ez[:, None, :] + Cov     # E[z_i z_iᵀ]
            # Stationarity, per row j of W, over the rows where j is observed:
            #     W_j = [Σ_{i∈O_j} (x_ij − μ_j) E[z_i]ᵀ] [Σ_{i∈O_j} E[z_i z_iᵀ]]⁻¹
            # The observed sets differ per row, so this one cannot be shared --
            # and the second moment, not the covariance, is what enters.
            W_new = W.copy()
            ssr = 0.0
            for j in range(d):
                rows = mask[:, j]
                if not rows.any():
                    continue                     # never observed: W_j stays put
                Aj = second[rows].sum(axis=0)
                bj = ((X[rows, j] - mean[j])[:, None] * Ez[rows]).sum(axis=0)
                W_new[j] = np.linalg.solve(Aj, bj)
                resid = X[rows, j] - mean[j] - Ez[rows] @ W_new[j]
                ssr += float(resid @ resid) + float(np.sum(W_new[j] @ Cov[rows] @ W_new[j]))
            # a missing cell contributes E[ε²] = σ², which does not depend on the
            # parameters and so drops out; what remains normalises by observed cells
            W, sigma2 = W_new, max(ssr / max(n_obs, 1), 1e-12)
            ll = float(np.sum(self._loglik_missing(X, W, sigma2, mean)))
            self.loglik_curve_.append(ll)
            self.n_iter_ = it
            if it > 1 and abs(ll - self.loglik_curve_[-2]) <= self.tol * abs(ll):
                break
        self.W_, self.sigma2_ = W, sigma2
        self.loglik_ = self.loglik_curve_[-1]

    # -- mixture of PPCA ---------------------------------------------------- #
    def _fit_mixture(self, X) -> None:
        n, d = X.shape
        k = min(self.n_components, d)
        m = self.n_mixtures
        rng = np.random.default_rng(self.seed)
        # k-means++ seeding, then a few Lloyd steps. Two centres drawn uniformly
        # can land in the same cluster; a mixture started that way splits the data
        # in half, both components stay identical, and EM has no reason to break
        # the symmetry -- it converges to 50/50 and calls it a day.
        centres = [X[rng.integers(n)]]
        for _ in range(m - 1):
            gap = np.min([((X - c) ** 2).sum(axis=1) for c in centres], axis=0)
            centres.append(X[rng.choice(n, p=gap / gap.sum())])
        centres = np.array(centres)
        for _ in range(10):
            dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(dist, axis=1)
            for c in range(m):
                if np.any(assign == c):
                    centres[c] = X[assign == c].mean(axis=0)
        dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        resp = np.zeros((n, m))
        resp[np.arange(n), np.argmin(dist, axis=1)] = 1.0
        self.loglik_curve_ = []
        for it in range(1, self.max_iter + 1):
            # --- M-step from the responsibilities -------------------------
            Nc = np.maximum(resp.sum(axis=0), 1e-8)
            weights = Nc / n
            means, Ws, sig2s = [], [], []
            for c in range(m):
                mu = resp[:, c] @ X / Nc[c]
                xc = X - mu
                # weighted scatter, then the closed form for this component
                lam, V = _top_eigs((xc * resp[:, c][:, None]).T @ xc / Nc[c], k)
                s2 = float(lam[k:].mean()) if d > k else 1e-6
                s2 = max(s2, 1e-6)
                Wc = V[:, :k] * np.sqrt(np.maximum(lam[:k] - s2, 1e-6))
                means.append(mu)
                Ws.append(Wc)
                sig2s.append(s2)
            # --- one weighted EM refinement of each component -------------
            for c in range(m):
                xc = X - means[c]
                M = Ws[c].T @ Ws[c] + sig2s[c] * np.eye(k)
                Minv = np.linalg.inv(M)
                Ez = xc @ Ws[c] @ Minv
                Cov = sig2s[c] * Minv
                A = (Ez.T * resp[:, c]) @ Ez + Nc[c] * Cov
                Wc = np.linalg.solve(A, ((xc * resp[:, c][:, None]).T @ Ez).T).T
                resid = xc - Ez @ Wc.T
                ssr = (float(np.sum(resp[:, c] * np.sum(resid * resid, axis=1)))
                       + Nc[c] * float(np.trace(Cov @ (Wc.T @ Wc))))
                Ws[c] = Wc
                sig2s[c] = max(ssr / (Nc[c] * d), 1e-8)
            # --- E-step ---------------------------------------------------
            logp = np.column_stack([
                np.log(max(weights[c], 1e-12))
                + _logpdf(X - means[c], Ws[c], sig2s[c]) for c in range(m)])
            mx = logp.max(axis=1, keepdims=True)
            resp = np.exp(logp - mx)
            tot = resp.sum(axis=1, keepdims=True)
            resp /= tot
            ll = float(np.sum(mx[:, 0] + np.log(tot[:, 0])))
            self.loglik_curve_.append(ll)
            self.n_iter_ = it
            if it > 1 and abs(ll - self.loglik_curve_[-2]) <= self.tol * abs(ll):
                break
        self.weights_ = weights
        self.means_ = np.array(means)
        self.Ws_, self.sigma2s_ = Ws, sig2s
        # A mixture has no single (μ, W, σ²). The per-component tuples are the
        # model, so the single-component attributes stay None: asking for one
        # covariance is then an error instead of a plausible-looking lie.
        self.mean_ = X.mean(axis=0)
        self.loglik_ = self.loglik_curve_[-1]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _latent(n: int = 600, d: int = 8, k: int = 2, noise: float = 0.4, seed: int = 0):
    """Rank-`k` data in `d` dimensions: `x = W z + ε` with `z ~ N(0, I)`."""
    rng = np.random.default_rng(seed)
    W = 1.5 * rng.standard_normal((d, k))
    X = rng.standard_normal((n, k)) @ W.T + noise * rng.standard_normal((n, d))
    return X, W


def main() -> None:
    X, Wt = _latent()
    n, d = X.shape
    k = 2
    Xc = X - X.mean(axis=0)
    lam_true = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / n))[::-1]

    # --- 1. two routes, one maximum --------------------------------------
    # EM and the closed form solve the same maximisation. W is not identified
    # (any rotation of its columns gives the same C) but C is, and so is σ².
    em = PPCA(k, variant="ppca_em").fit(X)
    cl = PPCA(k, variant="ppca_closed").fit(X)
    assert abs(em.loglik_ - cl.loglik_) < 1e-6 * abs(cl.loglik_), (em.loglik_, cl.loglik_)
    assert np.max(np.abs(em.covariance_ - cl.covariance_)) < 1e-6
    assert abs(em.sigma2_ / cl.sigma2_ - 1.0) < 1e-6
    # the spectrum of the fit is the sample spectrum with the tail flattened
    lam_fit = np.sort(np.linalg.eigvalsh(em.covariance_))[::-1]
    assert np.allclose(lam_fit[:k], lam_true[:k], rtol=1e-6)
    assert np.allclose(lam_fit[k:], em.sigma2_, rtol=1e-6)
    assert abs(em.sigma2_ - lam_true[k:].mean()) < 1e-6
    # and the curve never falls: EM on this model is monotone
    curve = np.array(em.loglik_curve_)
    assert np.all(np.diff(curve) >= -1e-9 * abs(curve[0]))
    print(
        f"ppca: EM and the closed form give the same likelihood to "
        f"{abs(em.loglik_ - cl.loglik_):.1e} and the same C to "
        f"{np.max(np.abs(em.covariance_ - cl.covariance_)):.1e}; the fitted spectrum is "
        f"({', '.join(f'{v:.3f}' for v in lam_fit[:k])}) then σ² = {em.sigma2_:.4f} six "
        f"times, and σ² is the mean of the discarded eigenvalues"
    )

    # --- 2. what is identified is the subspace, not the loading -----------
    rng = np.random.default_rng(1)
    R, _ = np.linalg.qr(rng.standard_normal((k, k)))
    assert np.max(np.abs((em.W_ @ R) @ (em.W_ @ R).T - em.covariance_ + em.sigma2_ * np.eye(d))) < 1e-10
    # the column space of W is PCA's: W = V_k (Λ_k − σ²I)^{1/2} R for any R
    lam_pca, V = np.linalg.eigh(Xc.T @ Xc / n)
    order = np.argsort(lam_pca)[::-1][:k]
    Qp, _ = np.linalg.qr(V[:, order])
    Qw, _ = np.linalg.qr(em.W_)
    angle = np.arccos(np.clip(np.linalg.svd(Qp.T @ Qw, compute_uv=False).min(), -1.0, 1.0))
    assert angle < 1e-6, angle          # arccos near 1: 1e-8 costs 1e-4 in rad
    print(
        f"ppca: rotating the loadings leaves C unchanged to "
        f"1e-10, and the subspace they span is PCA's to {angle:.1e} rad — the "
        f"projection was the identified part all along"
    )

    # --- 3. missing data: condition on what is observed -------------------
    # 15% of the entries are hidden. Mean imputation fills them with a column
    # mean; PPCA uses the conditional E[x_miss | x_obs] under the fitted model.
    rng = np.random.default_rng(2)
    Xm = X.copy()
    hid = rng.choice(Xm.size, size=Xm.size // 7, replace=False)
    Xm.flat[hid] = np.nan
    hidden = np.zeros(X.shape, dtype=bool)
    hidden.flat[hid] = True
    miss = PPCA(k, variant="missing_data").fit(Xm)
    rec = miss.reconstruct(Xm)
    rmse_miss = float(np.sqrt(np.mean((rec[hidden] - X[hidden]) ** 2)))
    rmse_base = float(np.sqrt(np.mean(
        (np.broadcast_to(np.nanmean(Xm, axis=0), X.shape)[hidden] - X[hidden]) ** 2)))
    assert rmse_miss < 0.6 * rmse_base, (rmse_miss, rmse_base)
    # with nothing hidden, this variant must reduce to the complete-data fit
    same = PPCA(k, variant="missing_data").fit(X)
    assert np.max(np.abs(same.covariance_ - em.covariance_)) < 1e-5
    print(
        f"ppca: with 14% of the entries hidden, the conditional reconstruction misses "
        f"them by {rmse_miss:.3f} where filling them with the column mean misses by "
        f"{rmse_base:.3f}; with nothing hidden the variant returns the complete-data fit"
    )

    # --- 4. the likelihood picks the rank; a projection cannot ------------
    # Fit on 2000 rows and score 2000 held out. Training likelihood rises with k
    # for ever, so the curve that peaks is the held-out one -- and it needs data
    # to peak: on 400 training rows k = 3 wins by 1e-3, because the sample
    # covariance's noise tail is not yet spherical enough to ignore.
    Xl, _ = _latent(n=4000, noise=0.4, seed=5)
    tr, te = Xl[:2000], Xl[2000:]
    held = [PPCA(kk, variant="ppca_closed").fit(tr).score_samples(te).mean()
            for kk in (1, 2, 3, 4)]
    assert int(np.argmax(held)) == 1, held          # index 1 is k = 2
    # and the model really is generative: samples have covariance C. A sample
    # covariance entry has Monte Carlo error sqrt((C_ij² + C_ii C_jj)/n), so the
    # comparison has to be on that scale -- a raw tolerance would pass the small
    # entries and fail the 25.0 one for reasons that have nothing to do with fit.
    S = PPCA(k, variant="ppca_em").fit(X)
    Xs = S.sample(20000, seed=3)
    C_fit = S.covariance_
    scale = np.sqrt(np.outer(np.diag(C_fit), np.diag(C_fit)))
    dev = float(np.max(np.abs(np.cov(Xs, rowvar=False, bias=True) - C_fit) / scale))
    assert dev < 0.05, dev
    print(
        f"ppca: held-out log likelihood peaks at k = 2 ({', '.join(f'{v:.3f}' for v in held)} "
        f"for k = 1..4), and 20000 draws from the fitted model reproduce C to "
        f"{dev:.1%} of its entrywise scale"
    )

    # --- 5. a mixture where one PPCA cannot fit both ----------------------
    # Two clusters, each a rank-1 subspace with its own offset and noise.
    rng = np.random.default_rng(4)
    d2, n1, n2 = 6, 400, 250
    u1, u2 = rng.standard_normal((d2, 1)), rng.standard_normal((d2, 1))
    A = np.vstack([np.array([4.0, 0, 0, 0, 0, 0]) + rng.standard_normal((n1, 1)) @ u1.T * 0.7,
                   np.array([-3.0, 0, 0, 0, 0, 0]) + rng.standard_normal((n2, 1)) @ u2.T * 0.7])
    A += 0.15 * rng.standard_normal(A.shape)
    lab = np.r_[np.zeros(n1), np.ones(n2)]
    mix = PPCA(1, variant="mixture_ppca", n_mixtures=2, seed=0).fit(A)
    resp = mix.predict_proba(A)
    # every row's responsibility is nearly a 0/1 indicator of its own cluster;
    # which component gets called 0 is arbitrary, so match the labelling first
    pred = resp.argmax(axis=1)
    purity = float(max(np.mean(pred == lab), np.mean(pred != lab)))
    assert purity > 0.98, purity
    single = PPCA(2, variant="ppca_em").fit(A)
    assert mix.loglik_ > single.loglik_ + 1.0, (mix.loglik_, single.loglik_)
    assert abs(mix.weights_.sum() - 1.0) < 1e-12
    w = np.sort(mix.weights_)
    assert abs(w[0] - n2 / (n1 + n2)) < 0.03 and abs(w[1] - n1 / (n1 + n2)) < 0.03, w
    print(
        f"ppca: a two-component mixture splits the two subspaces at {purity:.0%} purity "
        f"with weights {w[0]:.2f}/{w[1]:.2f} against the true "
        f"{n2 / (n1 + n2):.2f}/{n1 / (n1 + n2):.2f}, and beats a single PPCA with the "
        f"same latent dimension by {mix.loglik_ - single.loglik_:.0f} nats"
    )

    # --- 6. the posterior, and the density it implies ---------------------
    # E[z|x] = (WᵀW + σ²I)⁻¹Wᵀ(x−μ). Reconstruction is a denoiser, not a copy:
    # the residual is (I − P)(x − μ) with P = W(WᵀW + σ²I)⁻¹Wᵀ, which is 1 on
    # the discarded directions and ~0 on the retained ones, so the error is
    # (d − k)σ² -- the noise the model never claimed to explain.
    Z = em.transform(X)
    M = em.W_.T @ em.W_ + em.sigma2_ * np.eye(k)
    assert np.max(np.abs(Z - (X - em.mean_) @ em.W_ @ np.linalg.inv(M))) < 1e-10
    back = em.reconstruct(X)
    resid = float(np.mean(np.sum((X - back) ** 2, axis=1)))
    P = em.W_ @ np.linalg.inv(M) @ em.W_.T
    Xc = X - em.mean_
    assert abs(resid - float(np.trace((np.eye(d) - P) @ (np.eye(d) - P) @ (Xc.T @ Xc / n)))) < 1e-9
    assert abs(resid - (d - k) * em.sigma2_) < 0.05 * (d - k) * em.sigma2_, resid
    # the analytic log density agrees with a direct multivariate normal
    C = em.covariance_
    diff = X[0] - em.mean_
    direct = -0.5 * (d * math.log(2 * math.pi) + np.linalg.slogdet(C)[1]
                     + float(diff @ np.linalg.solve(C, diff)))
    assert abs(direct - em.score_samples(X[:1])[0]) < 1e-9
    print(
        f"ppca: the posterior mean is the Woodbury projection to 1e-10, the mean "
        f"reconstruction error is exactly tr((I−P)²S) and equals (d−k)·σ² = "
        f"{(d - k) * em.sigma2_:.3f}: the retained subspace is rebuilt, the six "
        f"discarded directions keep their noise; and score_samples matches a "
        f"direct N(μ, C) density to 1e-9"
    )

    print("probabilistic pca demo passed")


if __name__ == "__main__":
    main()
