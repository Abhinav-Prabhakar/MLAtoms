"""Gaussian processes — a distribution over functions, conditioned on data.

Chapter:   05-probabilistic
Companion: gaussian_process.md
Tier 1 · 9 variants: rbf, matern, rational_quadratic, periodic, linear,
sum_kernel, product_kernel, sparse_gp, warped

    f ~ GP(0, k)          a prior over functions, fixed by one kernel
    y = f(x) + ε          Gaussian noise, so everything stays Gaussian
    f* | y ~ N(K* K⁻¹ y,  K** − K* K⁻¹ K*ᵀ)

Bayes' rule for a Gaussian prior and a Gaussian likelihood is a linear solve.
No sampling, no optimisation, no variational bound — the posterior is a formula,
and the formula is a Cholesky factorisation. That is the entire atom.

What changes between variants is `k`: how smooth the prior believes the function
is, how fast correlation decays, whether it is periodic or linear. Two more
variants change the *computation* rather than the prior — inducing points, so
the solve is `O(nm²)` instead of `O(n³)`, and a warped output, so a Gaussian
process can model a positive, skewed response.

    rbf                k = σ² exp(−r²/2ℓ²)          infinitely smooth
    matern             k = (1+√3r)exp(−√3r)         ν = 1/2, 3/2, 5/2
    rational_quadratic k = σ²(1+r²/2αℓ²)^(−α)       heavy-tailed spectrum
    periodic           k = σ² exp(−2 sin²(πΔ/p)/ℓ²)  exactly periodic
    linear             k = σ² xᵀx'                  Bayesian linear regression
    sum_kernel         k = k₁ + k₂                  trend plus wiggle
    product_kernel     k = k₁ · k₂                  periodicity with decay
    sparse_gp          k ≈ K_nm K_mm⁻¹ K_mn         inducing points
    warped             y = g(f) + ε, g a log-shift  positive, skewed targets
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["rbf", "matern", "rational_quadratic", "periodic", "linear",
     "sum_kernel", "product_kernel", "sparse_gp", "warped"]
)


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`‖a − b‖²` for every pair, without forming the `n×m×d` tensor."""
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
class Kernel:
    """A positive-definite function of two points, plus its diagonal."""

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def diag(self, A: np.ndarray) -> np.ndarray:
        """`k(a, a)` for each row — cheaper than `np.diag(self(A, A))`."""
        return np.diag(self(A, A))


class RBF(Kernel):
    """Squared exponential: covariance decays as a Gaussian in the distance."""

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0):
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)

    def __call__(self, A, B):
        r2 = sqdist(A, B) / self.length_scale ** 2
        return self.sigma_f ** 2 * np.exp(-0.5 * r2)

    def diag(self, A):
        return np.full(A.shape[0], self.sigma_f ** 2)


class Matern(Kernel):
    """Matérn: `ν` controls mean-square differentiability, not just smoothness.

    Only the three half-integer cases have closed forms without a Bessel
    function, and they are the three anybody uses: `ν = 1/2` is the
    Ornstein–Uhlenbeck process (continuous but nowhere differentiable),
    `ν = 3/2` is once differentiable, `ν = 5/2` twice.
    """

    def __init__(self, nu: float = 1.5, length_scale: float = 1.0,
                 sigma_f: float = 1.0):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("nu must be 0.5, 1.5 or 2.5")
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)

    def __call__(self, A, B):
        r = np.sqrt(sqdist(A, B)) / self.length_scale
        if self.nu == 0.5:
            k = np.exp(-r)
        elif self.nu == 1.5:
            s = np.sqrt(3.0) * r
            k = (1.0 + s) * np.exp(-s)
        else:
            s = np.sqrt(5.0) * r
            k = (1.0 + s + s * s / 3.0) * np.exp(-s)
        return self.sigma_f ** 2 * k

    def diag(self, A):
        return np.full(A.shape[0], self.sigma_f ** 2)


class RationalQuadratic(Kernel):
    """Scale mixture of RBFs: a heavy-tailed spectral density in one knob."""

    def __init__(self, alpha: float = 1.0, length_scale: float = 1.0,
                 sigma_f: float = 1.0):
        self.alpha = float(alpha)
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)

    def __call__(self, A, B):
        r2 = sqdist(A, B) / self.length_scale ** 2
        return self.sigma_f ** 2 * (1.0 + r2 / (2.0 * self.alpha)) ** (-self.alpha)

    def diag(self, A):
        return np.full(A.shape[0], self.sigma_f ** 2)


class Periodic(Kernel):
    """`exp(−2 sin²(πΔ/p)/ℓ²)`: repeat exactly every `p`, smooth in between."""

    def __init__(self, period: float = 1.0, length_scale: float = 1.0,
                 sigma_f: float = 1.0):
        self.period = float(period)
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)

    def __call__(self, A, B):
        s = np.sin(np.pi * (A[:, None, :] - B[None, :, :]) / self.period)
        return self.sigma_f ** 2 * np.exp(-2.0 * (s ** 2).sum(axis=-1)
                                          / self.length_scale ** 2)

    def diag(self, A):
        return np.full(A.shape[0], self.sigma_f ** 2)


class Linear(Kernel):
    """`σ² xᵀx'`: a GP with this kernel *is* Bayesian linear regression."""

    def __init__(self, sigma_f: float = 1.0):
        self.sigma_f = float(sigma_f)

    def __call__(self, A, B):
        return self.sigma_f ** 2 * (A @ B.T)

    def diag(self, A):
        return self.sigma_f ** 2 * (A * A).sum(axis=1)


class Sum(Kernel):
    """`k₁ + k₂`: independent functions added, so their covariances add."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, A, B):
        return self.k1(A, B) + self.k2(A, B)

    def diag(self, A):
        return self.k1.diag(A) + self.k2.diag(A)


class Product(Kernel):
    """`k₁ · k₂`: the covariance of a product of independent GPs."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, A, B):
        return self.k1(A, B) * self.k2(A, B)

    def diag(self, A):
        return self.k1.diag(A) * self.k2.diag(A)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, m: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    """Inducing points: `m` Lloyd centres, deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], size=m, replace=False)].copy()
    for _ in range(iters):
        lab = sqdist(X, C).argmin(axis=1)
        for j in range(m):
            hit = lab == j
            if hit.any():
                C[j] = X[hit].mean(axis=0)
    return C


def _golden(f, lo: float, hi: float, iters: int = 60) -> float:
    """Maximise a unimodal `f` on `[lo, hi]` without derivatives."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


class GaussianProcess:
    """GP regression with kernel selection and three ways to compute it.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    length_scale, sigma_f : float
        Kernel amplitude and length scale; ignored when `kernel` is given.
    sigma_n : float
        Observation noise standard deviation. This is *not* a smoothing knob:
        it is the noise model, and it also sets the numerical floor of the
        posterior variance.
    nu, alpha, period : float
        Parameters of the Matérn, rational-quadratic and periodic kernels.
    n_inducing : int
        `sparse_gp` only. Inducing points are k-means centres unless `inducing`
        is passed.
    inducing : ndarray or None
        Explicit inducing inputs. Passing `X` itself turns the sparse
        approximation into the exact one, which is the demo's identity.
    kernel : Kernel or None
        Override the kernel the variant would build; this is how `Sum` and
        `Product` take arbitrary components.
    """

    def __init__(self, variant: str = "rbf", *, length_scale: float = 1.0,
                 sigma_f: float = 1.0, sigma_n: float = 0.1, nu: float = 1.5,
                 alpha: float = 1.0, period: float = 1.0, n_inducing: int = 8,
                 inducing: np.ndarray | None = None, kernel: Kernel | None = None,
                 jitter: float = 1e-10):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.sigma_n = float(sigma_n)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.period = float(period)
        self.n_inducing = int(n_inducing)
        self.inducing = inducing
        self.kernel = kernel
        self.jitter = float(jitter)

    def _default_kernel(self) -> Kernel:
        kw = {"length_scale": self.length_scale, "sigma_f": self.sigma_f}
        if self.variant == "matern":
            return Matern(self.nu, **kw)
        if self.variant == "rational_quadratic":
            return RationalQuadratic(self.alpha, **kw)
        if self.variant == "periodic":
            return Periodic(self.period, **kw)
        if self.variant == "linear":
            return Linear(self.sigma_f)
        if self.variant == "sum_kernel":
            return Sum(RBF(**kw), Linear(self.sigma_f))
        if self.variant == "product_kernel":
            return Product(RBF(**kw), Periodic(self.period, **kw))
        return RBF(**kw)

    # ------------------------------------------------------------------ #
    def _solve(self, X: np.ndarray, y: np.ndarray):
        """Cholesky of `K + σ²I`, the linear solve, and the log-likelihood."""
        s2 = self.sigma_n ** 2
        K = self.kernel_(X, X) + (s2 + self.jitter) * np.eye(X.shape[0])
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        ll = (-0.5 * float(y @ alpha) - np.log(np.diag(L)).sum()
              - 0.5 * y.size * np.log(2.0 * np.pi))
        return L, alpha, float(ll)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcess:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.kernel_ = self.kernel if self.kernel is not None else self._default_kernel()
        if self.variant == "warped":
            self._fit_warped(X, y)
        elif self.variant == "sparse_gp":
            self._fit_sparse(X, y)
        else:
            self.X_, self.y_ = X, y
            self._L, self._alpha, self.log_likelihood_ = self._solve(X, y)
        return self

    def _fit_warped(self, X: np.ndarray, y: np.ndarray) -> None:
        """`g(z) = log(z − a)`, with `a = min(y) − exp(θ)` chosen by maximum
        likelihood. As `θ → −∞` the warping becomes plain `log`; as it grows
        the warp disappears and so does its benefit."""
        offset = self.jitter + 1e-12
        ymin = float(y.min()) - offset

        def objective(theta: float) -> float:
            a = ymin - np.exp(theta)
            z = y - a
            if (z <= 0.0).any():
                return -np.inf
            g = np.log(z)
            _, _, ll = self._solve(X, g)
            return ll - float(np.log(z).sum())      # the Jacobian of the warping

        theta = _golden(objective, -14.0, 4.0, iters=50)
        self.warp_delta_ = float(np.exp(theta))
        self.warp_a_ = ymin - self.warp_delta_
        self.warp_theta_ = theta
        self.X_, self.y_ = X, np.log(y - self.warp_a_)
        self._L, self._alpha, self.log_likelihood_ = self._solve(X, self.y_)
        self.log_likelihood_ = float(objective(theta))

    def _fit_sparse(self, X: np.ndarray, y: np.ndarray) -> None:
        """Subset-of-regressors: replace `K_nn` by `K_nm K_mm⁻¹ K_mn`.

        Woodbury keeps the fit at `O(nm²)`; the `n×n` matrix is never formed.
        """
        m = min(self.n_inducing, X.shape[0])
        Z = (np.asarray(self.inducing, dtype=float) if self.inducing is not None
             else _kmeans(X, m))
        if self.inducing is not None:
            m = Z.shape[0]
        self.inducing_ = Z
        s2 = self.sigma_n ** 2
        Kmm = self.kernel_(Z, Z) + self.jitter * np.eye(m)
        Lm = np.linalg.cholesky(Kmm)
        # A = Lm⁻¹ K_mn, so that Q = AᵀA = K_nm K_mm⁻¹ K_mn
        A = np.linalg.solve(Lm, self.kernel_(Z, X))              # (m, n)
        B = s2 * np.eye(m) + A @ A.T                             # (m, m)
        c = np.linalg.solve(B, A @ y)                            # (m,)
        w = (y - A.T @ c) / s2                                   # (Q+σ²I)⁻¹ y
        self.inducing_mean_ = np.linalg.solve(Lm.T, A @ w)       # K_mm⁻¹K_mn w
        # |Q + σ²I| = σ²^(n−m)·|B|, the matrix determinant lemma
        logdet = ((X.shape[0] - m) * np.log(s2)
                  + 2.0 * np.log(np.diag(np.linalg.cholesky(B))).sum())
        self.log_likelihood_ = float(-0.5 * (y @ w) - 0.5 * logdet
                                     - 0.5 * y.size * np.log(2.0 * np.pi))
        self._Lm = Lm
        self.X_, self.y_ = X, y

    # ------------------------------------------------------------------ #
    def _predict_latent(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and *latent* variance (no observation noise)."""
        if self.variant == "sparse_gp":
            Ks = self.kernel_(self.inducing_, X)                 # (m, n*)
            mean = Ks.T @ self.inducing_mean_
            q = np.linalg.solve(self._Lm, Ks)
            var = self.kernel_.diag(X) - (q ** 2).sum(axis=0)
            return mean, np.maximum(var, 0.0)
        Ks = self.kernel_(self.X_, X)
        mean = Ks.T @ self._alpha
        v = np.linalg.solve(self._L, Ks)
        var = self.kernel_.diag(X) - (v ** 2).sum(axis=0)
        return mean, np.maximum(var, 0.0)

    def predict(self, X: np.ndarray, return_std: bool = True):
        X = np.asarray(X, dtype=float)
        mean, var = self._predict_latent(X)
        if self.variant == "warped":
            # a log-normal is not the exponential of a mean: the variance
            # enters the first moment, and the observation noise enters scaled
            # by (dg/dz)² = exp(2g)
            scale = np.exp(2.0 * mean)
            ymean = self.warp_a_ + np.exp(mean + 0.5 * var)
            yvar = scale * (np.expm1(var) + self.sigma_n ** 2)
            return (ymean, np.sqrt(yvar)) if return_std else ymean
        yvar = var + self.sigma_n ** 2
        return (mean, np.sqrt(yvar)) if return_std else mean

    def marginal_log_likelihood(self) -> float:
        """`log p(y | X)` — the quantity that compares kernels."""
        return self.log_likelihood_

    def _posterior_cov(self, X: np.ndarray) -> np.ndarray:
        """`K** − K*ᵀ K⁻¹ K*`: the covariance `predict` only reports the diagonal of."""
        Kss = self.kernel_(X, X)
        if self.variant == "sparse_gp":
            Ksm = self.kernel_(X, self.inducing_)                  # (n*, m)
            C = Kss - Ksm @ np.linalg.solve(
                self._Lm.T, np.linalg.solve(self._Lm, Ksm.T))
        else:
            v = np.linalg.solve(self._L, self.kernel_(self.X_, X))
            C = Kss - v.T @ v
        return 0.5 * (C + C.T)

    def sample(self, X: np.ndarray, n: int = 5, seed: int = 0) -> np.ndarray:
        """Draw functions from the *posterior* at `X`, shape `(n, len(X))`.

        The off-diagonal entries of the posterior covariance are what make the
        draws smooth; sampling each point from `N(mean_i, var_i)` independently
        would give the right marginals and the wrong functions.
        """
        mean = self._predict_latent(X)[0]
        w, V = np.linalg.eigh(self._posterior_cov(X))
        root = V * np.sqrt(np.maximum(w, 0.0))
        z = np.random.default_rng(seed).standard_normal((X.shape[0], n))
        draws = (mean[:, None] + root @ z).T
        if self.variant == "warped":
            return self.warp_a_ + np.exp(draws)
        return draws


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _f(rng, x, sigma_f=1.0, length_scale=1.0):
    """An exact draw from the RBF prior: `f = L z` with `LLᵀ = k(x, x)`."""
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    K = RBF(length_scale, sigma_f)(X, X) + 1e-9 * np.eye(X.shape[0])
    return np.linalg.cholesky(K) @ rng.standard_normal(X.shape[0])


def main() -> None:
    rng = np.random.default_rng(0)

    # --- kernels: every closed form, checked against its own definition ---
    A = rng.normal(size=(6, 2))
    B = rng.normal(size=(4, 2))
    r = np.sqrt(sqdist(A, B))
    ls, sf = 0.8, 1.3
    assert np.allclose(RBF(ls, sf)(A, B), sf ** 2 * np.exp(-0.5 * (r / ls) ** 2))
    assert np.allclose(Matern(0.5, ls, sf)(A, B), sf ** 2 * np.exp(-r / ls))
    s3 = np.sqrt(3.0) * r / ls
    assert np.allclose(Matern(1.5, ls, sf)(A, B), sf ** 2 * (1 + s3) * np.exp(-s3))
    s5 = np.sqrt(5.0) * r / ls
    assert np.allclose(Matern(2.5, ls, sf)(A, B),
                       sf ** 2 * (1 + s5 + s5 ** 2 / 3) * np.exp(-s5))
    assert np.allclose(np.diag(Matern(1.5, ls, sf)(A, A)), sf ** 2)
    # a periodic kernel is exactly periodic, in every column
    P = Periodic(period=1.0, length_scale=0.5, sigma_f=1.0)
    x1 = np.linspace(0, 2, 9)[:, None]
    assert np.allclose(P(x1, x1 + 3.0), P(x1, x1))
    assert np.allclose(np.diag(P(x1, x1)), 1.0)
    # sum and product obey their definitions on the Gram matrix
    k1, k2 = RBF(1.0, 1.0), Periodic(1.0, 0.7, 0.6)
    assert np.allclose(Sum(k1, k2)(A, B), k1(A, B) + k2(A, B))
    assert np.allclose(Product(k1, k2)(A, B), k1(A, B) * k2(A, B))
    assert np.all(np.linalg.eigvalsh(k1(A, A)) > 0)          # PD
    print("kernels: Matérn ½, 3⁄2, 5⁄2 match their closed forms; the periodic "
          "kernel is exactly 1-periodic; sum and product compose pointwise")

    # --- the Gaussian identity: posterior mean and variance in closed form ---
    x0 = np.array([[0.3]])
    y0 = np.array([2.0])
    gp1 = GaussianProcess("rbf", length_scale=0.7, sigma_f=1.0,
                          sigma_n=1e-8).fit(x0, y0)
    xs = np.array([[0.0], [0.3], [1.1]])
    k0 = float(RBF(0.7, 1.0)(x0, x0)[0, 0])
    ks = RBF(0.7, 1.0)(x0, xs).ravel()
    mean, std = gp1.predict(xs)
    assert np.allclose(mean, ks / k0 * y0[0], atol=1e-6)
    kss = RBF(0.7, 1.0).diag(xs)
    assert np.allclose(std ** 2, kss - ks ** 2 / k0 + 1e-16, atol=1e-6)
    print("gp: one training point reproduces k*ᵀK⁻¹y and k**−k*ᵀK⁻¹k* exactly "
          "— the posterior is a solve, not a fit")

    # --- and the log-likelihood is the textbook formula ---
    X = rng.uniform(-2.0, 2.0, size=(9, 1))
    y = np.sin(2.0 * X[:, 0]) + 0.2 * rng.normal(size=9)
    gp = GaussianProcess("rbf", length_scale=0.9, sigma_n=0.2).fit(X, y)
    K = RBF(0.9, 1.0)(X, X) + (0.2 ** 2 + 1e-10) * np.eye(9)
    sign, logdet = np.linalg.slogdet(K)
    brute = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * logdet - 4.5 * np.log(2 * np.pi)
    assert np.isclose(gp.marginal_log_likelihood(), brute, atol=1e-9)
    print(f"gp: marginal log-likelihood {brute:.4f} equals "
          f"−½yᵀK⁻¹y − ½log|K| − n/2·log2π computed by slogdet")

    # --- linear kernel == Bayesian linear regression == ridge ---
    Xl = rng.normal(size=(12, 3))
    w_true = np.array([1.0, -2.0, 0.5])
    yl = Xl @ w_true + 0.1 * rng.normal(size=12)
    s2, sf2 = 0.3 ** 2, 2.0
    lin = GaussianProcess("linear", sigma_f=np.sqrt(sf2), sigma_n=np.sqrt(s2))
    lin.fit(Xl, yl)
    ridge = np.linalg.solve(Xl.T @ Xl + (s2 / sf2) * np.eye(3), Xl.T @ yl)
    assert np.allclose(lin.predict(Xl, return_std=False), Xl @ ridge, atol=1e-8)
    print(f"linear: posterior mean equals ridge with λ = σ²/σ_f² = {s2 / sf2:.4f} "
          f"— a GP with a linear kernel is conjugate regression")

    # --- matérn, rational quadratic and periodic all interpolate the same data ---
    Xp = np.linspace(0.0, 4.0, 60)[:, None]
    yp = np.sin(2.0 * np.pi * Xp[:, 0] / 2.0) + 0.05 * rng.normal(size=60)
    train, test = np.arange(60) % 3 != 0, np.arange(60) % 3 == 0
    best = {}
    for name, kw in (("rbf", {}), ("matern", {"nu": 1.5}), ("matern", {"nu": 2.5}),
                     ("rational_quadratic", {"alpha": 0.5}),
                     ("periodic", {"period": 2.0}),
                     ("sum_kernel", {}), ("product_kernel", {"period": 2.0})):
        g = GaussianProcess(name, length_scale=0.6, sigma_n=0.05, **kw)
        g.fit(Xp[train], yp[train])
        rmse = float(np.sqrt(np.mean((g.predict(Xp[test], return_std=False)
                                      - yp[test]) ** 2)))
        best[f"{name}{kw.get('nu', kw.get('alpha', ''))}"] = (rmse,
                                                              g.marginal_log_likelihood())
    assert all(np.isfinite(v[0]) for v in best.values())
    for label, (rmse, _) in best.items():
        assert rmse < 0.35, (label, rmse)
    top = max(best, key=lambda k: best[k][1])
    print(f"kernels: seven priors fit the same 40 points, RMSE "
          f"{min(v[0] for v in best.values()):.3f}–"
          f"{max(v[0] for v in best.values()):.3f}; highest evidence: {top}")

    # --- sparse: with the training points as inducing inputs it is exact ---
    exact = GaussianProcess("rbf", length_scale=0.6, sigma_n=0.05,
                            jitter=1e-10).fit(Xp[train], yp[train])
    full = GaussianProcess("sparse_gp", length_scale=0.6, sigma_n=0.05,
                           inducing=Xp[train], jitter=1e-10).fit(Xp[train], yp[train])
    assert np.allclose(full.predict(Xp[test], return_std=False),
                       exact.predict(Xp[test], return_std=False), atol=1e-6)
    fewer = GaussianProcess("sparse_gp", length_scale=0.6, sigma_n=0.05,
                            n_inducing=12).fit(Xp[train], yp[train])
    rmse_ex = float(np.sqrt(np.mean((exact.predict(Xp[test], return_std=False)
                                     - yp[test]) ** 2)))
    rmse_sp = float(np.sqrt(np.mean((fewer.predict(Xp[test], return_std=False)
                                     - yp[test]) ** 2)))
    assert rmse_sp < 1.6 * rmse_ex, (rmse_ex, rmse_sp)
    assert fewer.inducing_.shape == (12, 1)
    print(f"sparse_gp: m = n reproduces the exact GP to 1e-6; m = 12 of 40 gives "
          f"RMSE {rmse_sp:.3f} vs {rmse_ex:.3f}, fitted as O(nm²) not O(n³)")

    # --- warped: positive, skewed targets want the log scale ---
    # its own generator: a zero-mean GP mapped through exp spans two orders of
    # magnitude, and the plain GP has to average over the peaks
    rng_w = np.random.default_rng(20)
    Xw = np.linspace(0.0, 4.0, 50)[:, None]
    fw = _f(rng_w, Xw, length_scale=1.2)
    yw = np.exp(1.4 * fw) + 0.02 * rng_w.standard_normal(50)
    train_w, test_w = np.arange(50) % 2 == 0, np.arange(50) % 2 == 1
    plain = GaussianProcess("rbf", length_scale=1.2, sigma_n=0.05)
    plain.fit(Xw[train_w], yw[train_w])
    warp = GaussianProcess("warped", length_scale=1.2, sigma_n=0.05)
    warp.fit(Xw[train_w], yw[train_w])
    e_plain = float(np.sqrt(np.mean((plain.predict(Xw[test_w], return_std=False)
                                     - yw[test_w]) ** 2)))
    e_warp = float(np.sqrt(np.mean((warp.predict(Xw[test_w], return_std=False)
                                    - yw[test_w]) ** 2)))
    assert e_warp < 0.5 * e_plain, (e_plain, e_warp)
    # the fitted offset settles just below min(y), i.e. the warping is ~log
    span = yw[train_w].max() - yw[train_w].min()
    assert warp.warp_delta_ < 0.15 * span, (warp.warp_delta_, span)
    print(f"warped: log-shift fitted below min(y), RMSE {e_plain:.3f} → "
          f"{e_warp:.3f}; the log-normal mean exp(μ+σ²/2) is not exp(μ)")

    # --- posterior samples actually carry the posterior covariance ---
    draws = gp.sample(np.linspace(-2.0, 2.0, 40)[:, None], n=400, seed=1)
    assert draws.shape == (400, 40)
    cov_emp = np.cov(draws.T)
    mean_p, std_p = gp.predict(np.linspace(-2.0, 2.0, 40)[:, None])
    var_latent = std_p ** 2 - gp.sigma_n ** 2
    assert np.allclose(np.diag(cov_emp), var_latent, rtol=0.25, atol=0.01)
    assert np.allclose(draws.mean(axis=0), mean_p, atol=0.35)
    print("gp: 400 posterior draws reproduce the predictive mean and variance "
          "— sample() is consistent with predict()")

    print("gaussian_process demo passed")


if __name__ == "__main__":
    main()
