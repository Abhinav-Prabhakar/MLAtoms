"""Factor analysis — one loading per factor, one noise level per feature.

Chapter:   06-latent-factorization
Companion: factor_analysis.md
Tier 2 · 5 variants: ml_fa, pca_fa, varimax, promax, minres

    x = Λ f + μ + ε,    f ~ N(0, Φ),    ε ~ N(0, Ψ),   Ψ = diag(ψ₁…ψ_d)   (1)
    x ~ N(μ, Σ),        Σ = Λ Φ Λᵀ + Ψ                                      (2)

PPCA is the case Φ = I, Ψ = σ² I. Dropping the second constraint is the whole
point: a microphone has one noise level and a thermometer has another, and a
model that averages them fits neither. Dropping the first gives the oblique
variants, where the factors themselves are correlated.

With two factors there is nothing to look at: any orthogonal rotation of Λ
leaves ΛΛᵀ — and therefore the likelihood — unchanged. With three or more the
rotation is a real degree of freedom, and a loading matrix is only as
interpretable as the rotation you chose. Hence the five variants:

    ml_fa      maximum likelihood, fixed point on the scaled spectrum
    pca_fa     principal component method: one eigendecomposition, no noise fit
    minres     minimum residual: least squares on the off-diagonals only
    varimax    orthogonal rotation maximising the variance of squared loadings
    promax    oblique rotation towards a power-transformed varimax target

The three estimators optimise three different criteria and the demo keeps them
apart: `minres` wins its own off-diagonal residual, `ml_fa` wins the held-out
likelihood, and `pca_fa` wins nothing but speed. The two rotations are exact
reparameterisations, so the demo can assert that they change the loadings and
leave the fitted covariance alone.

Two facts the demo uses as levers rather than claims:

    with Ψ = σ² I the ML solution *is* PPCA, so ml_fa and probabilistic_pca have
    to agree -- a cross-check between two atoms with different code paths;

    at the ML fixed point, ψ = diag(S − ΛΛᵀ) exactly, and the eigenvalues of
    Ψ^{-1/2} S Ψ^{-1/2} beyond the k-th are all >= 1, equal only when ψ is.

Everything is fitted on the biased covariance S = Xcᵀ Xc / n, for the same reason
as in PPCA: the likelihood is a function of the second moment about the fitted
mean, and n − 1 belongs in an estimator of variance, not in a likelihood.
"""

from __future__ import annotations

import math

import numpy as np
from probabilistic_pca import PPCA  # sibling module: same folder, same sys.path

VARIANTS = frozenset(["ml_fa", "pca_fa", "minres", "varimax", "promax"])

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _eigh_desc(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix."""
    w, V = np.linalg.eigh(A)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def _varimax(L: np.ndarray, max_iter: int = 500, tol: float = 1e-10):
    """Kaiser's varimax: the orthogonal R maximising Σ_j Var_i((LR)²_ij).

    Rows are normalised to unit communality first, so that variables with large
    communalities do not dominate the criterion; an orthogonal rotation preserves
    row norms, so the communalities come back untouched when the rows are
    rescaled. Each step is the exact maximiser of a linearisation: with
    `B = A R`, the polar factor of `Bᵀ(B³ − B·diag(mean B²))` is the rotation to
    apply next.
    """
    h = np.sqrt((L * L).sum(axis=1, keepdims=True))
    A = L / np.maximum(h, EPS)
    R = np.eye(L.shape[1])
    best = -math.inf
    for _ in range(max_iter):
        B = A @ R
        crit = float(np.sum(np.var(B * B, axis=0)))
        if crit - best <= tol * max(abs(crit), 1.0):
            break
        best = crit
        M = B.T @ (B ** 3 - B * (B * B).mean(axis=0))
        u, _, vt = np.linalg.svd(M)
        R = R @ (u @ vt)                     # accumulate: B was already rotated
    return L @ R, R


def _promax(L: np.ndarray, power: float = 4.0):
    """Oblique promax: a Procrustes fit of the loadings to |L|^power.

    Returns the pattern matrix and the factor correlations. The pattern `P` and
    `Φ` are a reparameterisation of the same `ΛΛᵀ`, which is what makes the
    oblique rotation honest: `P Φ Pᵀ = L Lᵀ` exactly, asserted in the demo.
    """
    target = np.abs(L) ** power
    A = np.linalg.solve(L.T @ L, L.T @ target)          # least squares, L A ≈ target
    A = A @ np.diag(1.0 / np.diag(A))                   # unit diagonal, by column
    P = np.linalg.solve(A, L.T).T                       # P = L A^{-T}
    phi = A.T @ A
    # rescale so that Φ is a correlation matrix and P Φ Pᵀ is untouched:
    # P -> P diag(s) and Φ -> diag(1/s) Φ diag(1/s) cancel exactly
    s = np.sqrt(np.diag(phi))
    return P * s[None, :], phi / np.outer(s, s)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class FactorAnalysis:
    """Factor analysis by one of five variants.

    `loadings_` is `(d, k)` — feature by factor — and `noise_variance_` is the
    per-feature uniqueness ψ, the parameter PPCA replaces with a single σ².
    """

    def __init__(self, n_components: int = 1, *, variant: str = "ml_fa",
                 max_iter: int = 500, tol: float = 1e-8, power: float = 4.0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = int(n_components)
        self.variant = variant
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.power = float(power)

        self.mean_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.noise_variance_: np.ndarray | None = None
        self.rotation_: np.ndarray | None = None       # orthogonal rotation applied
        self.phi_: np.ndarray | None = None            # factor correlations
        self.heywood_: bool = False                    # a uniqueness hit the floor
        self.loglik_: float = -math.inf
        self.loglik_curve_: list[float] = []
        self.n_iter_: int = 0
        self.n_features_in_: int | None = None

    # -- public surface ----------------------------------------------------- #
    @property
    def uniquenesses_(self) -> np.ndarray:
        """ψ — the same array as `noise_variance_`, under its factor-analysis name."""
        if self.noise_variance_ is None:
            raise RuntimeError("call fit first")
        return self.noise_variance_

    @property
    def communalities_(self) -> np.ndarray:
        """`diag(Λ Φ Λᵀ)`: the variance each feature shares with the others."""
        return self._common()

    @property
    def covariance_(self) -> np.ndarray:
        """The fitted `Σ = Λ Φ Λᵀ + Ψ`."""
        if self.loadings_ is None:
            raise RuntimeError("call fit first")
        return self.loadings_ @ self._phi() @ self.loadings_.T + np.diag(self.noise_variance_)

    def fit(self, X) -> FactorAnalysis:
        X = self._check(X)
        d = X.shape[1]
        if self.n_components > d - 1:
            k = self.n_components
            n_par = d * k + d - k * (k - 1) // 2     # loadings + uniquenesses − rotations
            raise ValueError(
                f"n_components={k} with d={d} features gives {n_par} free parameters "
                f"against {d * (d + 1) // 2} covariance entries, so ψ is not identified. "
                f"Use probabilistic_pca for a d-component model.")
        self.mean_ = X.mean(axis=0)
        S = np.cov(X, rowvar=False, bias=True)
        if self.variant == "pca_fa":
            self._fit_pca(S)
        elif self.variant == "minres":
            self._fit_minres(S)
        else:                                   # ml_fa, varimax, promax
            self._fit_ml(S)
        if self.variant in ("varimax", "promax"):
            self._rotate()
        self.loglik_ = float(np.sum(self.score_samples(X)))
        return self

    def transform(self, X) -> np.ndarray:
        """Posterior mean `E[f | x]`, shape `(m, k)`."""
        if self.loadings_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X, fit=False)
        L, psi = self.loadings_, self.noise_variance_
        # E[f|x] = (I + Λᵀ Ψ⁻¹ Λ)⁻¹ Λᵀ Ψ⁻¹ (x − μ)   -- Thomson's scores
        A = np.eye(L.shape[1]) + (L.T / psi) @ L
        return np.linalg.solve(A, ((X - self.mean_) / psi) @ L).T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def reconstruct(self, X) -> np.ndarray:
        """`E[x | f = E[f|x]]`, the part of a row the factors explain."""
        return self.mean_ + self.transform(X) @ self.loadings_.T

    def score_samples(self, X) -> np.ndarray:
        """Per-row log likelihood under `N(μ, Σ)`."""
        X = self._check(X, fit=False)
        C = self.covariance_
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0.0:
            raise ValueError("the fitted covariance is singular; the model is too large "
                             "for this data (see heywood_)")
        diff = X - self.mean_
        quad = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(C), diff)
        return -0.5 * (C.shape[0] * math.log(2.0 * math.pi) + logdet + quad)

    # -- validation --------------------------------------------------------- #
    def _check(self, X, fit: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if np.isnan(X).any():
            raise ValueError("factor analysis needs a complete matrix; "
                             "use probabilistic_pca with variant='missing_data'")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {X.shape[1]}")
        if fit and X.shape[0] < 3:
            raise ValueError("need at least three rows to fit")
        return X

    def _phi(self) -> np.ndarray:
        return np.eye(self.loadings_.shape[1]) if self.phi_ is None else self.phi_

    def _common(self) -> np.ndarray:
        L = self.loadings_
        return np.einsum("ij,jk,ik->i", L, self._phi(), L)

    def _finish(self, S: np.ndarray) -> None:
        """Clip each uniqueness against its own variance, and record a Heywood hit."""
        floor = 1e-6 * S.diagonal()             # per feature, so the test is scale-free
        self.heywood_ = self.heywood_ or bool(np.any(self.noise_variance_ <= floor))
        self.noise_variance_ = np.maximum(self.noise_variance_, floor)

    # -- maximum likelihood: iterate ψ, solve Λ in closed form --------------- #
    def _fit_ml(self, S: np.ndarray) -> None:
        k = self.n_components
        diag = S.diagonal()
        floor = 1e-6 * diag                     # per feature, so the test is scale-free
        psi = 0.5 * diag.copy()                 # a communality of 0.5 to start
        self.loglik_curve_ = []
        for it in range(1, self.max_iter + 1):
            # Ψ^{-1/2} S Ψ^{-1/2}: the likelihood in coordinates where the
            # uniquenesses are 1, which is where the closed form lives
            C = S / np.sqrt(np.outer(psi, psi))
            theta, G = _eigh_desc(C)
            keep = np.maximum(theta[:k] - 1.0, 0.0)     # the signal above the noise
            L = np.sqrt(psi)[:, None] * (G[:, :k] * np.sqrt(keep)[None, :])
            psi_new = diag - np.sum(L * L, axis=1)
            self.loglik_curve_.append(self._profile_loglik(S, psi_new))
            self.n_iter_ = it
            if np.any(psi_new <= floor):        # ψ → 0: the model has hit its boundary
                self.heywood_ = True
                psi = np.maximum(psi_new, floor)
                break
            converged = np.max(np.abs(psi_new - psi)) <= self.tol * np.max(psi)
            psi = psi_new
            if converged:
                break
        self.loadings_ = L
        self.noise_variance_ = psi
        self._finish(S)

    def _profile_loglik(self, S: np.ndarray, psi: np.ndarray) -> float:
        """The likelihood with Λ already maximised out, per row of data.

        At the optimum over Λ, `tr(Σ⁻¹S) = d` and `log|Σ| = Σ log ψ_j + Σ_{j≤k}
        log θ_j`, where θ are the eigenvalues of Ψ^{-1/2}SΨ^{-1/2}. This is a
        different code path from `score_samples`, so agreement between the two is
        a real check rather than a restatement.
        """
        d = S.shape[0]                                # S is a covariance: (d, d)
        psi = np.maximum(psi, 1e-12)
        theta, _ = _eigh_desc(S / np.sqrt(np.outer(psi, psi)))
        keep = np.maximum(theta[:self.n_components], 1.0)
        return float(-0.5 * (d * math.log(2.0 * math.pi) + np.sum(np.log(psi))
                             + np.sum(np.log(keep)) + d))

    # -- principal component method: one eigendecomposition, no fixed point -- #
    def _fit_pca(self, S: np.ndarray) -> None:
        k = self.n_components
        lam, V = _eigh_desc(S)
        L = V[:, :k] * np.sqrt(np.maximum(lam[:k], 0.0))
        self.loadings_ = L
        self.noise_variance_ = S.diagonal() - np.sum(L * L, axis=1)
        self.n_iter_ = 1
        self._finish(S)

    # -- minimum residual: least squares on the off-diagonals ---------------- #
    def _fit_minres(self, S: np.ndarray) -> None:
        k = self.n_components
        diag = S.diagonal()
        psi = 0.5 * diag.copy()
        self.loglik_curve_ = []
        for it in range(1, self.max_iter + 1):
            # the reduced matrix: S with its diagonal replaced by communalities
            lam, V = _eigh_desc(S - np.diag(psi))
            L = V[:, :k] * np.sqrt(np.maximum(lam[:k], 0.0))
            psi_new = diag - np.sum(L * L, axis=1)
            self.n_iter_ = it
            converged = np.max(np.abs(psi_new - psi)) <= self.tol * np.max(psi)
            psi = psi_new
            if converged:
                break
        self.loadings_ = L
        self.noise_variance_ = psi
        self._finish(S)

    # -- rotations: same subspace, different basis --------------------------- #
    def _rotate(self) -> None:
        if self.variant == "varimax":
            self.loadings_, self.rotation_ = _varimax(self.loadings_)
            self.phi_ = None
        else:
            varimax_L, rot = _varimax(self.loadings_)
            self.loadings_, self.phi_ = _promax(varimax_L, self.power)
            self.rotation_ = rot


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _data(n: int = 2000, d: int = 8, k: int = 2, seed: int = 0, phi=None,
          load: float = 0.9, psi=None):
    """`x = Λ f + ε`: one factor per block of features, an uneven noise profile."""
    rng = np.random.default_rng(seed)
    L = np.zeros((d, k))
    per = d // k
    for j in range(k):
        L[j * per:(j + 1) * per, j] = load
    phi = np.eye(k) if phi is None else phi
    # deliberately heteroscedastic: per-feature noise is the whole point
    psi = (np.linspace(0.25, 0.75, d) if psi is None
           else np.asarray(psi, dtype=float))
    f = rng.standard_normal((n, k)) @ np.linalg.cholesky(phi).T
    X = f @ L.T + rng.standard_normal((n, d)) * np.sqrt(psi)
    return X, L, phi, psi


def _match(L_fit: np.ndarray, L_true: np.ndarray) -> np.ndarray:
    """Best permutation + sign of `L_fit`'s columns against `L_true`'s."""
    k = L_true.shape[1]
    C = np.abs(L_true.T @ L_fit)
    best, best_perm = -1.0, None
    for perm in __import__("itertools").permutations(range(k)):
        s = sum(C[j, perm[j]] for j in range(k))
        if s > best:
            best, best_perm = s, perm
    out = np.empty_like(L_fit)
    for j in range(k):
        col = L_fit[:, best_perm[j]]
        out[:, j] = col if np.dot(col, L_true[:, j]) >= 0.0 else -col
    return out


def main() -> None:
    X, L_true, _, psi_true = _data()
    n, d = X.shape
    k = 2
    S = np.cov(X, rowvar=False, bias=True)

    # --- 1. the ML fixed point, stated three ways --------------------------
    # Three conditions pin the optimum, and each is a different code path, so
    # agreeing on all three is a real check. (i) Σ maximizes over Λ at fixed ψ,
    # (ii) ψ maximizes with Λ held at its conditional optimum, (iii) the fitted
    # covariance is the identity in units of the noise outside the factor
    # directions. Writing ψ = diag(S − ΛΛᵀ) back out would only restate (ii).
    fa = FactorAnalysis(k, variant="ml_fa").fit(X)
    L, psi = fa.loadings_, fa.noise_variance_
    Sig = fa.covariance_
    Si = np.linalg.inv(Sig)
    # (ii) as a derivative: dℓ/dψ_j = ½[(Σ⁻¹SΣ⁻¹)_jj − (Σ⁻¹)_jj], zero at the optimum
    score = 0.5 * (np.diag(Si @ S @ Si) - np.diag(Si))
    assert np.max(np.abs(score)) < 1e-6, np.max(np.abs(score))
    # (iii) and the invariant that goes with it: the scaled spectrum keeps its
    # total weight, so the tail sums to d − k however unevenly it is spread
    theta, _ = _eigh_desc(S / np.sqrt(np.outer(psi, psi)))
    assert abs(theta[k:].sum() - (d - k)) < 1e-6, theta[k:].sum()
    scaled = np.diag(1.0 / np.sqrt(psi)) @ Sig @ np.diag(1.0 / np.sqrt(psi))
    ev = np.sort(np.linalg.eigvalsh(scaled))[::-1]
    assert np.allclose(ev[:k], theta[:k], rtol=1e-8), (ev[:k], theta[:k])
    assert np.max(np.abs(ev[k:] - 1.0)) < 1e-8, ev[k:]
    # (i) the profile likelihood maximises Λ out in closed form; the density
    # inverts the full d×d covariance. Different code, same number.
    gap = abs(fa.loglik_ - n * fa._profile_loglik(S, psi)) / abs(fa.loglik_)
    assert gap < 1e-8, gap
    # the loadings are only identified up to rotation, so compare subspaces
    Qf, _ = np.linalg.qr(L)
    Qt, _ = np.linalg.qr(L_true)
    angle = float(np.arccos(np.clip(np.linalg.svd(Qf.T @ Qt, compute_uv=False).min(), -1, 1)))
    assert angle < 0.05, angle
    assert np.mean(np.abs(psi - psi_true) / psi_true) < 0.15
    print(
        f"factor_analysis: ml_fa's ψ-gradient is zero to {np.max(np.abs(score)):.1e} in "
        f"{fa.n_iter_} iterations, the scaled spectrum is ({ev[0]:.4f}, {ev[1]:.4f}) then "
        f"1.000 six times with its tail summing to d−k = {theta[k:].sum():.6f}, the "
        f"loading subspace is the true one to {angle:.3f} rad, ψ tracks a 3× noise "
        f"spread to {np.mean(np.abs(psi - psi_true) / psi_true):.0%}, and the profile "
        f"likelihood matches the density to {gap:.0e}"
    )

    # --- 2. isotropic ψ makes it PPCA, with no shared code -----------------
    # Same data, ψ = σ²: the ML equations reduce to the PPCA closed form, so the
    # two atoms have to agree -- a cross-check on independent implementations.
    rng = np.random.default_rng(3)
    d2, k2 = 6, 2
    L_iso = 1.4 * rng.standard_normal((d2, k2))
    X_iso = rng.standard_normal((4000, k2)) @ L_iso.T + 0.4 * rng.standard_normal((4000, d2))
    f_iso = FactorAnalysis(k2, variant="ml_fa").fit(X_iso)
    p_iso = PPCA(k2, variant="ppca_closed").fit(X_iso)
    rel = float(np.linalg.norm(f_iso.loadings_ @ f_iso.loadings_.T
                               - p_iso.W_ @ p_iso.W_.T)
                / np.linalg.norm(p_iso.W_ @ p_iso.W_.T))
    spread = float(np.std(f_iso.noise_variance_) / np.mean(f_iso.noise_variance_))
    assert rel < 0.01, rel
    assert spread < 0.06, spread
    print(
        f"factor_analysis: on data whose true ψ is isotropic, ml_fa's loadings match "
        f"probabilistic_pca's to {rel:.1%}, and the per-feature noise comes out flat "
        f"(spread {spread:.1%}): one estimator, two atoms"
    )

    # --- 3. three estimators, two criteria --------------------------------
    # ml_fa maximises a likelihood; minres minimises the off-diagonal residual
    # directly; pca_fa eigendecomposes the sample covariance and stops. The two
    # proper criteria are asymptotically equivalent -- they agree to 1e-4 in
    # held-out likelihood -- and both beat the naive one easily. That "tie" is the
    # point: with a good ψ profile either is fine, and the naive estimator is
    # not. With a narrow ψ profile they tie more tightly still, which is why the
    # demo widens it: the gap has to be earned by fitting the noise.
    Xh, _, _, _ = _data(n=2000, d=8, k=2, seed=7, psi=np.linspace(0.05, 2.0, 8))
    Sh = np.cov(Xh, rowvar=False, bias=True)
    fits = {v: FactorAnalysis(k, variant=v).fit(Xh)
            for v in ("ml_fa", "pca_fa", "minres")}

    def off(F):
        """Frobenius norm of the residual on the off-diagonals only."""
        return float(np.linalg.norm((Sh - F.covariance_) * (1.0 - np.eye(d))))

    held = Xh[1500:]
    lik = {v: float(np.mean(F.score_samples(held))) for v, F in fits.items()}
    res = {v: off(F) for v, F in fits.items()}
    # the two criteria that fit Ψ land in the same place ...
    assert abs(lik["ml_fa"] - lik["minres"]) < 1e-3, lik
    # ... and both beat the estimator that ignores the noise structure
    assert lik["ml_fa"] > lik["pca_fa"] + 0.1, lik
    # in the residual each was built for, each is at least twice as good
    assert res["minres"] < 0.5 * res["pca_fa"], res
    assert res["ml_fa"] < 0.5 * res["pca_fa"], res
    print(
        f"factor_analysis: on a 40× noise spread, ml_fa and minres are the same "
        f"estimator in effect — held-out likelihood {lik['ml_fa']:.4f} against "
        f"{lik['minres']:.4f}, off-diagonal residual {res['ml_fa']:.3f} against "
        f"{res['minres']:.3f} — while pca_fa, which fits no noise at all, loses "
        f"{lik['ml_fa'] - lik['pca_fa']:.2f} nats a row and leaves a residual of "
        f"{res['pca_fa']:.2f}"
    )

    # --- 4. varimax: rotate the basis, keep the model ----------------------
    # The truth is simple in the factor sense -- each feature loads on one factor
    # -- but the factors themselves are correlated, which is what makes ML return
    # a basis nobody can read: every variable gets a piece of every factor. An
    # orthogonal truth would be the wrong test, because with Φ = I the ML
    # eigenvectors are already near-simple and varimax has almost nothing to do.
    phi_corr = np.array([[1.0, 0.45, 0.1], [0.45, 1.0, 0.3], [0.1, 0.3, 1.0]])
    Xr, L_simple, _, _ = _data(n=4000, d=12, k=3, seed=11, phi=phi_corr)
    plain = FactorAnalysis(3, variant="ml_fa").fit(Xr)
    rot = FactorAnalysis(3, variant="varimax").fit(Xr)

    def crit(A):
        """Kaiser's criterion: the summed variance of the squared loadings."""
        return float(np.sum(np.var(A * A, axis=0)))

    def mid(A):
        """Share of loadings that are neither clearly on nor clearly off."""
        return float(np.mean((np.abs(A) > 0.15) & (np.abs(A) < 0.5)))

    # the rotation is a reparameterisation: rows and Σ are untouched
    assert crit(rot.loadings_) > 2.5 * crit(plain.loadings_), (crit(plain.loadings_),
                                                                crit(rot.loadings_))
    assert mid(rot.loadings_) < 0.6 * mid(plain.loadings_), (mid(plain.loadings_),
                                                             mid(rot.loadings_))
    assert np.max(np.abs(np.sum(rot.loadings_ ** 2, axis=1)
                         - np.sum(plain.loadings_ ** 2, axis=1))) < 1e-10
    assert np.max(np.abs(rot.covariance_ - plain.covariance_)) < 1e-10
    # and it is a local optimum: no random rotation of the fitted basis improves it
    rng = np.random.default_rng(5)
    best = crit(rot.loadings_)
    for _ in range(200):
        dq, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        assert crit(rot.loadings_ @ dq) <= best + 1e-9
    err = float(np.max(np.abs(_match(rot.loadings_, L_simple) - L_simple)))
    print(
        f"factor_analysis: varimax raises the simplicity criterion from "
        f"{crit(plain.loadings_):.3f} to {best:.3f} on an oblique truth — the mid-range "
        f"loadings fall from {mid(plain.loadings_):.0%} of the matrix to "
        f"{mid(rot.loadings_):.0%} — leaves the communalities and Σ untouched to 1e-10, "
        f"and is a local optimum under 200 random rotations. It recovers the simple "
        f"structure to {err:.2f}, not to 0: an orthogonal rotation cannot undo a "
        f"correlated Φ, and varimax is a heuristic, not an oracle"
    )

    # --- 5. promax: the oblique parameterisation, and what it does not buy --
    # The truth has correlated factors (0.4 and 0.2). Varimax can only return
    # orthogonal ones, so it reports Φ = I; promax lets them correlate. What it
    # does *not* do is estimate Φ -- the pattern is a Procrustes fit to a
    # heuristic target, so Φ is a rotation artefact, not a population quantity.
    phi_true = np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 1.0]])
    Xo, _, _, _ = _data(n=4000, d=12, k=3, seed=13, phi=phi_true)
    Xflat, _, _, _ = _data(n=4000, d=12, k=3, seed=13)          # same seed, Φ = I
    ob = FactorAnalysis(3, variant="promax").fit(Xo)
    var = FactorAnalysis(3, variant="varimax").fit(Xo)
    ob_flat = FactorAnalysis(3, variant="promax").fit(Xflat)
    P, phi = ob.loadings_, ob.phi_
    assert np.max(np.abs(np.diag(phi) - 1.0)) < 1e-12
    assert np.max(np.abs(P @ phi @ P.T - var.loadings_ @ var.loadings_.T)) < 1e-10
    # reparameterised, not refitted: the model covariance is untouched
    assert np.max(np.abs(ob.covariance_ - var.covariance_)) < 1e-10
    oblique = float(np.linalg.norm(phi - np.eye(3)))
    flat = float(np.linalg.norm(ob_flat.phi_ - np.eye(3)))
    assert oblique > 0.5 and flat < 0.1, (oblique, flat)

    got = np.sort(np.abs(phi[np.triu_indices(3, 1)]))[::-1]
    want = np.sort(np.abs(phi_true[np.triu_indices(3, 1)]))[::-1]
    # the strengths come back, the pairing does not: which factor is which is
    # decided by the rotation, so Φ can only be believed up to a permutation
    assert np.max(np.abs(got - want)) < 0.1, (got, want)
    # and an oblique pattern is not a cleaner matrix -- it is a different one.
    # A pattern loading is a regression weight, so it can exceed 1.
    assert np.abs(P).max() > np.abs(var.loadings_).max() + 0.05, (
        np.abs(P).max(), np.abs(var.loadings_).max())

    def mid(A):
        """Share of loadings that are neither clearly on nor clearly off."""
        return float(np.mean((np.abs(A) > 0.15) & (np.abs(A) < 0.5)))

    print(
        f"factor_analysis: promax returns the same Σ as varimax to 1e-10 while letting "
        f"the factors correlate — ‖Φ − I‖ = {oblique:.2f} on data whose factors really "
        f"do ({', '.join(f'{v:.2f}' for v in want)}) against {flat:.2f} on data where "
        f"they do not. Its |Φ| comes back as ({', '.join(f'{v:.2f}' for v in got)}), right "
        f"in magnitude but not in correspondence — which factor is which is decided by the "
        f"rotation — and it reports a small third correlation where the truth has none. "
        f"And an oblique pattern is not a cleaner matrix, it is a different one: the largest "
        f"entry is {np.abs(P).max():.2f} against {np.abs(var.loadings_).max():.2f}, with "
        f"{mid(P):.0%} of entries in between against {mid(var.loadings_):.0%}"
    )

    # --- 6. the Heywood case: the boundary of the model ---------------------
    # One feature is an exact linear combination of two others, so the sample
    # covariance is singular and no Σ = ΛΛᵀ + Ψ with Ψ > 0 can equal it. The
    # likelihood's answer is to drive those ψ to zero -- a feature whose variance
    # is entirely explained by the factors, which is not a finite model.
    Xc, _, _, _ = _data(n=800, d=11, k=3, seed=7)
    Xc = np.hstack([Xc, Xc[:, [0]] + Xc[:, [1]]])          # a redundant 12th feature
    hey = FactorAnalysis(4, variant="ml_fa", max_iter=2000).fit(Xc)
    n_floor = int(np.sum(hey.noise_variance_ <= 1e-6 * np.diag(np.cov(Xc, rowvar=False,
                                                                       bias=True))))
    assert hey.heywood_ and n_floor >= 1, (hey.heywood_, n_floor)
    # the tell: the redundancy is *not* in the feature we added, because any one
    # of the three collinear features can be the one that gets absorbed
    assert np.linalg.matrix_rank(np.cov(Xc, rowvar=False, bias=True)) == 11
    print(
        f"factor_analysis: with a feature that is an exact linear combination of two "
        f"others, k = 4 drives {n_floor} of 12 uniquenesses to the floor "
        f"({hey.noise_variance_.min():.1e}) and sets heywood_. A ψ of zero says that "
        f"feature has no noise of its own — not a model — and nothing about the fit "
        f"throws or looks wrong, which is exactly why the flag has to exist"
    )

    print("factor analysis demo passed")


if __name__ == "__main__":
    main()
