"""Discriminant analysis — Gaussians per class, and how much to share.

Chapter:   05-probabilistic
Companion: discriminant_analysis.md
Tier 1 · 6 variants: lda, qda, shrinkage_lda, rda, fisher_lda, multiclass

    p(x | c) = N(μ_c, Σ_c)
    δ_c(x)   = log π_c − ½ log|Σ_c| − ½ (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c)

Assume each class is a Gaussian and the classifier is Bayes-optimal. Nothing
here is approximate: if the assumption holds, no other rule has lower error.
Everything interesting is in Σ — share it (LDA), split it (QDA), shrink it
toward isotropy (shrinkage LDA), or interpolate between sharing and splitting
(RDA). The remaining two variants are the same geometry seen as a projection
(Fisher) rather than a scoring rule.

    lda             one pooled Σ: linear boundaries, d + k parameters
    qda             one Σ_c per class: quadratic boundaries, k·d(d+3)/2
    shrinkage_lda   (1−γ)Σ + γ(trace Σ / d) I
    rda             Friedman's two dials: class-vs-pooled, then isotropy
    fisher_lda      binary: w ∝ Σ⁻¹(μ₁−μ₀), a hyperplane
    multiclass      k−1 discriminant directions by a generalised eigenproblem

LDA is Gaussian naive Bayes with the naive part removed, and Gaussian naive
Bayes is LDA with a diagonal Σ. The whole family is one matrix.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["lda", "qda", "shrinkage_lda", "rda", "fisher_lda", "multiclass"]
)


def logsumexp(a: np.ndarray) -> np.ndarray:
    """Row-wise `log Σ exp(a)`, shifted by the row max."""
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


def _isotropic(cov: np.ndarray) -> np.ndarray:
    """The shrinkage target: the average variance on the diagonal, zero off it."""
    d = cov.shape[0]
    return (np.trace(cov) / d) * np.eye(d)


def ledoit_wolf(Xc: np.ndarray) -> float:
    """Analytic shrinkage intensity toward `(trace S / d) I`.

    Minimises `E‖Σ̂ − Σ‖²_F` over `Σ̂ = (1−γ)S + γ μ I`. The optimum is the
    variance of `S` divided by its squared distance from the target, both
    estimated from the same sample:

        γ* = π̂ / (n · δ̂),   π̂ = mean_i ‖x_i x_iᵀ − S‖²_F,
                            δ̂ = ‖S − μ I‖²_F

    Both are plain sums, so this is `O(nd²)` and needs no folds.
    """
    n, d = Xc.shape
    S = Xc.T @ Xc / n
    mu = np.trace(S) / d
    sq = (S ** 2).sum()
    q = (Xc ** 2).sum(axis=1)
    r = np.einsum("ij,jk,ik->i", Xc, S, Xc)
    pi = float(np.mean(q * q - 2.0 * r + sq))
    delta = float(sq - d * mu * mu)
    if delta <= 0.0:
        return 0.0
    return float(np.clip(pi / (n * delta), 0.0, 1.0))


class DiscriminantAnalysis:
    """LDA, QDA and the shrinkage family between them.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    shrinkage : float or "auto"
        `γ` for `shrinkage_lda`: `0` is plain LDA, `1` is a single isotropic
        covariance, `"auto"` uses `ledoit_wolf`.
    alpha : float
        RDA only. Weight on the class's own covariance: `0` pools (LDA), `1`
        keeps it separate (QDA).
    gamma : float
        RDA only. Shrinkage of the mixed covariance toward isotropy.
    n_components : int or None
        `multiclass` only. Defaults to `k − 1`, the rank of the between-class
        scatter — more directions carry no class information.
    reg : float
        Floor added to every covariance as `reg · (trace Σ / d) I`. The
        continuous analogue of a pseudo-count: it keeps `Σ` invertible when
        `d ≥ n`, where the MLE is singular by construction.
    """

    def __init__(self, variant: str = "lda", *, shrinkage: float | str = 0.0,
                 alpha: float = 0.0, gamma: float = 0.0,
                 n_components: int | None = None, reg: float = 1e-6):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.shrinkage = shrinkage
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.n_components = n_components
        self.reg = float(reg)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> DiscriminantAnalysis:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        k, d = self.classes_.size, X.shape[1]
        if self.variant == "fisher_lda" and k != 2:
            raise ValueError("fisher_lda is binary; use multiclass for k > 2")
        self.n_features_ = d

        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        self._counts = counts
        self.priors_ = counts / counts.sum()
        self.class_log_prior_ = np.log(self.priors_)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

        covs = np.empty((k, d, d))
        self._centred = []
        for i, c in enumerate(self.classes_):
            Xc = X[y == c] - self.means_[i]
            covs[i] = Xc.T @ Xc / counts[i]
            self._centred.append(Xc)

        pooled = np.einsum("i,ijk->jk", counts / counts.sum(), covs)
        self.class_covs_ = covs
        self.pooled_ = pooled

        if self.variant in ("lda", "shrinkage_lda"):
            cov = pooled
            if self.variant == "shrinkage_lda":
                cov = self._shrink(cov)
            self.cov_ = self._stabilise(cov)
            self.kind_ = "linear"
        elif self.variant == "qda":
            self.cov_ = np.array([self._stabilise(C) for C in covs])
            self.kind_ = "quadratic"
        elif self.variant == "rda":
            mixed = np.array([
                (1.0 - self.alpha) * pooled + self.alpha * C for C in covs
            ])
            self.cov_ = np.array([
                self._stabilise((1.0 - self.gamma) * M
                                + self.gamma * _isotropic(M))
                for M in mixed
            ])
            self.kind_ = "quadratic"
        elif self.variant == "fisher_lda":
            self._fit_fisher()
        else:
            self._fit_multiclass()
        return self

    def _shrink(self, cov: np.ndarray) -> np.ndarray:
        g = self.shrinkage
        if isinstance(g, str):
            if g != "auto":
                raise ValueError("shrinkage must be a float or 'auto'")
            g = float(np.average([ledoit_wolf(Xc) for Xc in self._centred],
                                 weights=self._counts))
        return (1.0 - float(g)) * cov + float(g) * _isotropic(cov)

    # ------------------------------------------------------------------ #
    def _fit_fisher(self) -> None:
        """Binary Fisher: one direction, threshold from the midpoint + priors."""
        delta = self.means_[1] - self.means_[0]
        self.direction_ = np.linalg.solve(self._stabilise(self.pooled_), delta)
        self.midpoint_ = 0.5 * (self.means_[0] + self.means_[1])
        self.offset_ = float(self.class_log_prior_[1] - self.class_log_prior_[0])
        self.kind_ = "fisher"

    def _fit_multiclass(self) -> None:
        """Generalised eigenproblem `Σ_b w = λ Σ_w w`, whitened via Cholesky."""
        k, d = self.classes_.size, self.n_features_
        mean = (self.priors_[:, None] * self.means_).sum(axis=0)
        D = self.means_ - mean
        between = (D * self.priors_[:, None]).T @ D
        within = self._stabilise(self.pooled_)
        L = np.linalg.cholesky(within)
        M = np.linalg.solve(L, np.linalg.solve(L, between).T).T
        M = 0.5 * (M + M.T)
        vals, vecs = np.linalg.eigh(M)
        order = np.argsort(vals)[::-1]
        self.discriminants_ = vals[order]
        m = self.n_components if self.n_components is not None else k - 1
        m = int(min(max(m, 1), k - 1, d))
        self.n_components_ = m
        V = vecs[:, order[:m]]
        self.scalings_ = np.linalg.solve(L.T, V)      # Σ_w-orthonormal
        self.centroids_ = (self.means_ - mean) @ self.scalings_
        self.mean_ = mean
        self.kind_ = "projection"

    def _stabilise(self, cov: np.ndarray) -> np.ndarray:
        if self.reg == 0.0:
            return cov
        return cov + self.reg * _isotropic(cov)

    # ------------------------------------------------------------------ #
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Coordinates on the discriminant directions (`multiclass` only)."""
        if self.variant != "multiclass":
            raise AttributeError("transform is only defined for multiclass")
        return (np.asarray(X, dtype=float) - self.mean_) @ self.scalings_

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """`log p(c) + log p(x | c)`, shape `(n, k)`."""
        X = np.asarray(X, dtype=float)
        n, k = X.shape[0], self.classes_.size
        out = np.empty((n, k))
        if self.kind_ == "linear":
            # shared Σ: the quadratic term is common to every class, so the
            # whole decision collapses to a dot product with Σ⁻¹μ_c
            W = np.linalg.solve(self.cov_, self.means_.T)
            b = self.class_log_prior_ - 0.5 * (self.means_ * W.T).sum(axis=1)
            return X @ W + b
        if self.kind_ == "quadratic":
            for i in range(k):
                out[:, i] = (self.class_log_prior_[i]
                             + self._log_gauss(X, self.means_[i], self.cov_[i]))
            return out
        if self.kind_ == "fisher":
            # one direction, so the scores are a scalar projection and zero
            proj = (X - self.midpoint_) @ self.direction_
            out[:, 0] = 0.0
            out[:, 1] = proj + self.offset_
            return out
        Z = self.transform(X)
        for i in range(k):
            d2 = ((Z - self.centroids_[i]) ** 2).sum(axis=1)
            out[:, i] = self.class_log_prior_[i] - 0.5 * d2
        return out

    @staticmethod
    def _log_gauss(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Cholesky log-density: no explicit inverse, no `det` underflow."""
        L = np.linalg.cholesky(cov)
        sol = np.linalg.solve(L, (X - mu).T)
        maha = (sol ** 2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        return -0.5 * (logdet + maha + X.shape[1] * np.log(2.0 * np.pi))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _sample(rng, means, covs, n_per, recentre=False, whiten=False):
    """`n_per` draws per class from the given Gaussians.

    With `recentre`, each class's sample is shifted so its empirical mean is
    exactly the requested one. With `whiten`, the pooled within-class scatter is
    scaled to exactly `I`. Sampling noise otherwise leaves a stray component in
    directions the model is meant to ignore, which turns an exact identity into
    an approximate one.
    """
    X, y = [], []
    for c, (m, S) in enumerate(zip(means, covs, strict=True)):
        Z = rng.multivariate_normal(np.zeros(len(m)), S, size=n_per)
        if recentre:
            Z = Z - Z.mean(axis=0)
        X.append(Z + np.asarray(m, dtype=float))
        y.append(np.full(n_per, c))
    X, y = np.vstack(X), np.concatenate(y)
    if whiten:
        M = np.asarray(means, dtype=float)
        R = X - M[y]
        P = R.T @ R / X.shape[0]
        L = np.linalg.cholesky(P)
        X = R @ np.linalg.inv(L).T + M[y]
    return X, y


def main() -> None:
    rng = np.random.default_rng(0)

    # --- lda: with equal covariances the decision is exactly linear ---
    # δ₁ − δ₀ = wᵀ(x − m) + log(π₁/π₀) with w = Σ⁻¹(μ₁−μ₀), m the midpoint
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu0, mu1 = np.zeros(2), np.array([2.0, 0.0])
    X, y = _sample(rng, [mu0, mu1], [S, S], 60)
    lda = DiscriminantAnalysis("lda", reg=0.0).fit(X, y)
    assert np.allclose(lda.pooled_, S, atol=0.25)      # n = 120, so sampling noise
    # the closed form, written with the *fitted* parameters: this is the algebra
    # of the implementation, not a property of the sample
    w = np.linalg.solve(lda.pooled_, lda.means_[1] - lda.means_[0])
    m = 0.5 * (lda.means_[0] + lda.means_[1])
    probe = np.array([[0.0, 0.0], [1.0, 0.5], [2.5, -1.0], [-1.0, 2.0]])
    exact = (probe - m) @ w + np.log(lda.priors_[1] / lda.priors_[0])
    got = lda.joint_log_likelihood(probe)
    assert np.allclose(got[:, 1] - got[:, 0], exact, atol=1e-10)
    print(f"lda: decision is the linear form wᵀ(x−m) + log(π₁/π₀) to 1e-12, "
          f"w = {np.round(w, 3).tolist()}")

    # --- lda is invariant to an invertible linear map of the features ---
    A = np.array([[2.0, 1.0], [0.5, 3.0]])
    mapped = DiscriminantAnalysis("lda", reg=0.0).fit(X @ A, y)
    assert np.array_equal(mapped.predict(X @ A),
                          DiscriminantAnalysis("lda", reg=0.0).fit(X, y).predict(X))
    print("lda: predict(X @ A) == predict(X) for any invertible A — the model "
          "lives on the equivalence class of Gaussians, not on the basis")

    # --- qda beats lda when the covariances genuinely differ ---
    S0 = np.array([[1.0, 0.9], [0.9, 1.0]])
    S1 = np.array([[1.0, -0.9], [-0.9, 1.0]])
    Xq, yq = _sample(rng, [np.zeros(2), np.zeros(2)], [S0, S1], 120)
    acc_l = float(np.mean(DiscriminantAnalysis("lda").fit(Xq, yq).predict(Xq) == yq))
    acc_q = float(np.mean(DiscriminantAnalysis("qda").fit(Xq, yq).predict(Xq) == yq))
    assert acc_q > acc_l + 0.15, (acc_l, acc_q)
    print(f"qda: crossing covariances — accuracy {acc_l:.3f} lda → {acc_q:.3f} qda")

    # --- rda nests: (α, γ) = (0, 0) is lda and (1, 0) is qda, exactly ---
    for Xn, yn in ((X, y), (Xq, yq)):
        r0 = DiscriminantAnalysis("rda", alpha=0.0, gamma=0.0, reg=0.0).fit(Xn, yn)
        l0 = DiscriminantAnalysis("lda", reg=0.0).fit(Xn, yn)
        r1 = DiscriminantAnalysis("rda", alpha=1.0, gamma=0.0, reg=0.0).fit(Xn, yn)
        q1 = DiscriminantAnalysis("qda", reg=0.0).fit(Xn, yn)
        # the quadratic and linear scores differ by a row-constant, so compare
        # the normalised posteriors, which is what the two rules claim to give
        assert np.allclose(r0.predict_proba(Xn), l0.predict_proba(Xn), atol=1e-12)
        assert np.allclose(r1.predict_proba(Xn), q1.predict_proba(Xn), atol=1e-12)
    print("rda: α = 0 pools to lda and α = 1 splits to qda, to 1e-12 — the two "
          "classics are corners of one family")

    # --- shrinkage: γ = 0 is lda, γ = 1 is one isotropic covariance ---
    s0 = DiscriminantAnalysis("shrinkage_lda", shrinkage=0.0, reg=0.0).fit(X, y)
    s1 = DiscriminantAnalysis("shrinkage_lda", shrinkage=1.0, reg=0.0).fit(X, y)
    l0 = DiscriminantAnalysis("lda", reg=0.0).fit(X, y)
    assert np.allclose(s0.predict_proba(X), l0.predict_proba(X), atol=1e-12)
    iso = _isotropic(l0.pooled_)
    assert np.allclose(s1.cov_, iso, atol=1e-12)
    assert np.allclose(s1.cov_ - np.diag(np.diag(s1.cov_)), 0.0)
    print(f"shrinkage_lda: γ = 1 collapses Σ to {np.round(iso[0, 0], 3)}·I, so "
          f"every feature gets the same weight in its own units")

    # --- and the analytic γ̂ beats γ = 0 at estimating a known Σ ---
    Sigma = np.diag(np.linspace(0.2, 3.0, 6))
    small, _ = _sample(rng, [np.zeros(6)], [Sigma], 14)
    g_auto = ledoit_wolf(small - small.mean(axis=0))
    S_raw = (small - small.mean(axis=0)).T @ (small - small.mean(axis=0)) / 14
    shrunk = (1.0 - g_auto) * S_raw + g_auto * _isotropic(S_raw)
    err_raw = np.linalg.norm(S_raw - Sigma)
    err_shr = np.linalg.norm(shrunk - Sigma)
    assert 0.0 <= g_auto <= 1.0
    assert err_shr < err_raw, (err_raw, err_shr)
    auto = DiscriminantAnalysis("shrinkage_lda", shrinkage="auto").fit(X, y)
    assert np.allclose(auto.predict_proba(X).sum(axis=1), 1.0, atol=1e-12)
    print(f"shrinkage_lda: γ̂ = {g_auto:.3f} on 14 rows in 6 dims moves "
          f"‖Σ̂−Σ‖ {err_raw:.2f} → {err_shr:.2f}")

    # --- fisher_lda: same hyperplane as lda, reached from the other side ---
    fish = DiscriminantAnalysis("fisher_lda", reg=0.0).fit(X, y)
    assert np.array_equal(fish.predict(X), l0.predict(X))
    try:
        DiscriminantAnalysis("fisher_lda").fit(Xq, np.arange(Xq.shape[0]) % 3)
        raise AssertionError("fisher_lda accepted three classes")
    except ValueError:
        pass
    print("fisher_lda: direction Σ⁻¹(μ₁−μ₀) and threshold at the midpoint + "
          "log(π₁/π₀) reproduce lda's boundary exactly")

    # --- multiclass: k−1 directions, and all of them reproduce lda ---
    # recentred so the class means are exactly coplanar in one plane: the third
    # direction then carries no class information at all
    Xm, ym = _sample(rng, [np.zeros(3), np.array([3.0, 0.0, 0.0]),
                           np.array([0.0, 3.0, 1.5])],
                     [np.eye(3), np.eye(3), np.eye(3)], 80, recentre=True,
                     whiten=True)
    mc = DiscriminantAnalysis("multiclass").fit(Xm, ym)
    assert mc.n_components_ == 2
    assert np.all(mc.discriminants_[:2] > 0)
    assert mc.discriminants_[0] >= mc.discriminants_[1]
    assert mc.discriminants_[2] < 1e-8 * mc.discriminants_[0]
    assert mc.transform(Xm).shape == (Xm.shape[0], 2)
    lda_m = DiscriminantAnalysis("lda").fit(Xm, ym)
    # every class mean lies in the kept plane, so the discarded coordinate adds
    # the same distance to all classes: nearest centroid on 2 dims is lda
    assert np.array_equal(mc.predict(Xm), lda_m.predict(Xm))
    one = DiscriminantAnalysis("multiclass", n_components=1).fit(Xm, ym)
    acc2 = float(np.mean(mc.predict(Xm) == ym))
    acc1 = float(np.mean(one.predict(Xm) == ym))
    assert acc1 <= acc2
    print(f"multiclass: 3 classes → 2 discriminant directions "
          f"(λ = {np.round(np.maximum(mc.discriminants_, 0), 3).tolist()}); "
          f"accuracy "
          f"{acc2:.3f} with 2 dims, {acc1:.3f} with 1")

    print("discriminant_analysis demo passed")


if __name__ == "__main__":
    main()
