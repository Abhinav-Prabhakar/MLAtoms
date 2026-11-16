"""Gaussian process classification — a non-Gaussian likelihood, approximated.

Chapter:   05-probabilistic
Companion: gp_classification.md
Tier 3 · 5 variants: laplace, logistic, probit, expectation_propagation,
one_vs_rest

    f ~ GP(0, k),    y_i | f_i ~ p(y_i | f_i),    p(y_i | f_i) ≠ Gaussian

Regression is a linear solve because a Gaussian prior times a Gaussian
likelihood is Gaussian. Swap in a logistic or probit likelihood and the product
stops being anything you can write down: the posterior over `f` is a
`n`-dimensional distribution with no closed form, and the predictive is an
integral nobody can do.

Two approximations, both Gaussian:

    laplace    find the mode, keep the curvature there         O(n³)
    ep         match each site's first two moments, iterate     O(Tn³)

and one extension: `one_vs_rest` builds k binary classifiers and normalises
their log-odds, because neither approximation turns into a real multiclass
model on its own.

The demo measures every claim against exact answers: a 1-D quadrature for one
data point, a 2-D grid for two. No approximation is checked against another
approximation except to say how far apart they are.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(
    ["laplace", "logistic", "probit", "expectation_propagation", "one_vs_rest"]
)

LOG2PI = math.log(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

# variant -> (likelihood, inference, how the predictive is formed)
_SPEC = {
    "laplace": ("logistic", "laplace", "plug"),
    "logistic": ("logistic", "laplace", "integrate"),
    "probit": ("probit", "ep", "analytic"),
    "expectation_propagation": ("logistic", "ep", "integrate"),
    "one_vs_rest": ("logistic", "laplace", "integrate"),
}


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def sigmoid(a: np.ndarray) -> np.ndarray:
    """Logistic function without overflowing on either tail."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    pos = a >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    e = np.exp(a[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def ndtr(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF. numpy has no `erf`, so this goes through `math`."""
    z = np.asarray(z, dtype=float)
    flat = np.array([0.5 * math.erfc(-v / SQRT2) for v in z.ravel()], dtype=float)
    return flat.reshape(z.shape)


def log_ndtr(z: np.ndarray) -> np.ndarray:
    """`log Φ(z)`, accurate into the far left tail where `Φ` underflows."""
    z = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore"):
        out = np.log(np.maximum(ndtr(z), 1e-320))
    tail = z < -30.0
    if np.any(tail):
        t = z[tail]
        # Φ(z) = φ(z)/|z| (1 − 1/z² + 3/z⁴ − ...) as z → −∞
        out[tail] = (-0.5 * t * t - 0.5 * LOG2PI - np.log(-t)
                     + np.log1p(-1.0 / t**2 + 3.0 / t**4))
    return out


def rbf(A, B, length_scale: float, sigma_f: float) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return sigma_f**2 * np.exp(-0.5 * np.maximum(d, 0.0) / length_scale**2)


def _logistic_site(y: np.ndarray, f: np.ndarray):
    """`Σ log p`, `∂ log p/∂f`, and `−∂² log p/∂f²` for `p(y|f) = σ(y f)`."""
    z = y * f
    ll = -np.logaddexp(0.0, -z)
    t = y * np.exp(-np.logaddexp(0.0, z))                # y σ(−z)
    w = np.exp(-np.logaddexp(0.0, -f)) * np.exp(-np.logaddexp(0.0, f))  # σ(f)σ(−f)
    return ll, t, w


def _probit_site(y: np.ndarray, f: np.ndarray):
    """Same three quantities for `p(y|f) = Φ(y f)`."""
    z = y * f
    ll = log_ndtr(z)
    r = np.exp(-0.5 * z * z - 0.5 * LOG2PI - ll)          # φ(z)/Φ(z)
    return ll, y * r, r * (z + r)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class GPClassifier:
    """Binary GP classification with a logistic or probit likelihood.

    `variant` fixes the likelihood *and* the approximation, because the two are
    chosen together in practice. `likelihood`, `inference` and `predictive`
    override one axis at a time, which is how the demo isolates them.
    """

    def __init__(self, variant: str = "laplace", *, likelihood: str | None = None,
                 inference: str | None = None, predictive: str | None = None,
                 length_scale: float = 1.0, sigma_f: float = 1.0,
                 jitter: float = 1e-8, n_iter: int = 100, tol: float = 1e-9,
                 damping: float = 0.7):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        like, infer, pred = _SPEC[variant]
        self.variant = variant
        self.likelihood = likelihood or like
        self.inference = inference or infer
        self.predictive = predictive or pred
        if self.likelihood not in ("logistic", "probit"):
            raise ValueError("likelihood must be 'logistic' or 'probit'")
        if self.inference not in ("laplace", "ep"):
            raise ValueError("inference must be 'laplace' or 'ep'")
        if self.predictive not in ("plug", "integrate", "analytic"):
            raise ValueError("predictive must be 'plug', 'integrate' or 'analytic'")
        self.length_scale = float(length_scale)
        self.sigma_f = float(sigma_f)
        self.jitter = float(jitter)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.damping = float(damping)

    # -- likelihood --------------------------------------------------------- #
    def _site(self, f: np.ndarray):
        fn = _logistic_site if self.likelihood == "logistic" else _probit_site
        return fn(self.y_, f)

    # -- fitting ------------------------------------------------------------ #
    def fit(self, X, y) -> GPClassifier:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y disagree on the number of points")
        if self.variant == "one_vs_rest":
            self.classes_ = np.unique(y)
            if self.classes_.size < 3:
                raise ValueError("one_vs_rest needs at least three classes")
            self.models_ = []
            for c in self.classes_:
                m = GPClassifier("logistic", length_scale=self.length_scale,
                                 sigma_f=self.sigma_f, jitter=self.jitter,
                                 n_iter=self.n_iter, tol=self.tol,
                                 damping=self.damping)
                self.models_.append(m._fit_binary(X, np.where(y == c, 1.0, -1.0)))
            return self
        return self._fit_binary(X, y)

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> GPClassifier:
        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("labels must be +1/−1; use one_vs_rest beyond two classes")
        self.X_, self.y_ = X, y
        n = X.shape[0]
        self.K_ = rbf(X, X, self.length_scale, self.sigma_f) + self.jitter * np.eye(n)
        return self._laplace() if self.inference == "laplace" else self._ep()

    # -- Laplace ------------------------------------------------------------ #
    def _laplace(self) -> GPClassifier:
        """Newton on `log p(f|y)`; each step is one Cholesky of `I + √W K √W`."""
        n = self.X_.shape[0]
        f = np.zeros(n)
        for _it in range(1, self.n_iter + 1):
            _, t, W = self._site(f)
            sW = np.sqrt(W)
            L = np.linalg.cholesky(np.eye(n) + sW[:, None] * self.K_ * sW[None, :])
            b = W * f + t
            # a = (K + W⁻¹)⁻¹ b, and the mode is K a. Doing it this way keeps
            # every solve triangular; forming K + W⁻¹ would need an inverse.
            v = np.linalg.solve(L, sW * (self.K_ @ b))
            a = b - sW * np.linalg.solve(L.T, v)
            new = self.K_ @ a
            step = float(np.max(np.abs(new - f)))
            f = new
            if step < self.tol:
                break

        _, self.t_, W = self._site(f)
        self.f_ = f
        self.n_iter_ = _it
        self.sqrtW_ = np.sqrt(W)
        self.L_ = np.linalg.cholesky(
            np.eye(n) + self.sqrtW_[:, None] * self.K_ * self.sqrtW_[None, :]
        )
        # a = K⁻¹f is the residual `f − Kt`, which is what makes the evidence
        # formula collapse to a few matrix-vector products
        self.a_ = np.linalg.solve(self.K_, f)
        self.logZ_ = (-0.5 * float(f @ self.a_) + float(self._site(f)[0].sum())
                      - float(np.log(np.diag(self.L_)).sum()))
        # predictive: mean = k*ᵀ t, var = k** − ‖L⁻¹ √W k*‖²
        self._coef, self._chol, self._scale = self.t_, self.L_, self.sqrtW_
        return self

    # -- expectation propagation -------------------------------------------- #
    def _ep_posterior(self, nu: np.ndarray, tau: np.ndarray):
        """`q(f) = N(μ, Σ)` from site means `ν` and precisions `τ`."""
        n = nu.size
        sT = np.sqrt(np.maximum(tau, 0.0))
        L = np.linalg.cholesky(np.eye(n) + sT[:, None] * self.K_ * sT[None, :])
        V = np.linalg.solve(L, sT[:, None] * self.K_)
        Sigma = self.K_ - V.T @ V                      # (K⁻¹ + diag τ)⁻¹
        return Sigma, Sigma @ (tau * nu)

    def _tilted(self, m: float, s2: float, y: float):
        """Mean, variance and normaliser of `q(f)·p(y|f)` for one site."""
        if self.likelihood == "probit":
            z = y * m / math.sqrt(1.0 + s2)
            lz = float(log_ndtr(np.array([z]))[0])
            r = math.exp(-0.5 * z * z - 0.5 * LOG2PI - lz)
            mean = m + y * s2 * r / math.sqrt(1.0 + s2)
            var = s2 - s2 * s2 * r * (z + r) / (1.0 + s2)
            return mean, var, lz
        x, w = np.polynomial.hermite.hermgauss(40)
        f = m + math.sqrt(2.0 * s2) * x
        p = sigmoid(y * f)
        den = float((w * p).sum()) * INV_SQRT_PI
        mean = float((w * p * f).sum()) * INV_SQRT_PI / den
        var = float((w * p * (f - mean) ** 2).sum()) * INV_SQRT_PI / den
        return mean, var, math.log(den)

    def _ep(self) -> GPClassifier:
        """Iterate tilting and moment matching; `damping` decides how far to move."""
        n = self.X_.shape[0]
        nu, tau = np.zeros(n), np.zeros(n)
        d = self.damping
        for _it in range(1, self.n_iter + 1):
            Sigma, mu = self._ep_posterior(nu, tau)
            delta = 0.0
            for i in range(n):
                # cavity: remove site i from q, which is one subtraction of
                # precisions and one of means in natural form
                s2 = 1.0 / max(1.0 / Sigma[i, i] - tau[i], 1e-12)
                m = s2 * (mu[i] / Sigma[i, i] - nu[i] * tau[i])
                m_t, s2_t, _ = self._tilted(m, s2, self.y_[i])
                tau_i = 1.0 / max(s2_t, 1e-300) - 1.0 / s2
                # natural-form site: precision `tau_i`, linear term `eta_i`.
                # `nu` is stored as the *mean*, eta/tau, which is what makes
                # `mu = Sigma @ (tau * nu)` one matrix-vector product.
                nu_i = (m_t / max(s2_t, 1e-300) - m / s2) / tau_i
                delta = max(delta, abs(tau_i - tau[i]), abs(nu_i - nu[i]))
                tau[i] = (1.0 - d) * tau[i] + d * tau_i
                nu[i] = (1.0 - d) * nu[i] + d * nu_i
            if delta < self.tol:
                break

        Sigma, mu = self._ep_posterior(nu, tau)
        S = 1.0 / tau
        self.f_, self.Sigma_ = mu, Sigma
        self.nu_, self.tau_ = nu, tau
        self.n_iter_ = _it

        # Evidence. The site is `Z̃_i N(f; ν_i, 1/τ_i)` where Z̃_i is fixed by
        # matching its mass under the cavity to the tilted mass Ẑ_i: moment
        # matching alone leaves the amplitude free, and without this the
        # estimate is not even exact for a Gaussian likelihood.
        L = np.linalg.cholesky(self.K_ + np.diag(S))
        B = self.K_ + np.diag(S)
        coef = np.linalg.solve(B, nu)
        log_zq = (-0.5 * n * LOG2PI - float(np.log(np.diag(L)).sum())
                  - 0.5 * float(nu @ coef))
        log_amp = 0.0
        for i in range(n):
            s2 = 1.0 / max(1.0 / Sigma[i, i] - tau[i], 1e-12)
            m = s2 * (mu[i] / Sigma[i, i] - nu[i] * tau[i])
            lz = self._tilted(m, s2, self.y_[i])[2]
            v = s2 + S[i]                                  # variance of q_{-i} * site
            log_nu = -0.5 * LOG2PI - 0.5 * math.log(v) - 0.5 * (nu[i] - m) ** 2 / v
            log_amp += lz - log_nu
        self.logZ_ = log_amp + log_zq
        self._coef, self._chol, self._scale = coef, L, np.ones(n)
        return self

    # -- prediction --------------------------------------------------------- #
    def _latent(self, X: np.ndarray):
        Ks = rbf(X, self.X_, self.length_scale, self.sigma_f)
        mean = Ks @ self._coef
        Z = np.linalg.solve(self._chol, self._scale[:, None] * Ks.T)
        var = self.sigma_f**2 - (Z * Z).sum(axis=0)
        return mean, np.maximum(var, 1e-12)

    def predict_proba(self, X) -> np.ndarray:
        """`p(y=1|x)`, or `(m, k)` for `one_vs_rest`."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.variant == "one_vs_rest":
            P = np.stack([m.predict_proba(X) for m in self.models_], axis=1)
            logit = np.log(np.clip(P, 1e-12, 1.0)) - np.log(np.clip(1.0 - P, 1e-12, 1.0))
            logit -= logit.max(axis=1, keepdims=True)
            E = np.exp(logit)
            return E / E.sum(axis=1, keepdims=True)

        mean, var = self._latent(X)
        if self.predictive == "plug":
            return sigmoid(mean)
        if self.predictive == "analytic":       # probit: exact integral, no quadrature
            return ndtr(mean / np.sqrt(1.0 + var))
        x, w = np.polynomial.hermite.hermgauss(20)
        nodes = mean[:, None] + np.sqrt(2.0 * var)[:, None] * x
        return (w * sigmoid(nodes)).sum(axis=1) * INV_SQRT_PI

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[P.argmax(axis=1)] if P.ndim == 2 else np.where(P > 0.5, 1, -1)


# --------------------------------------------------------------------------- #
# Exact references
# --------------------------------------------------------------------------- #
def _log_joint(K: np.ndarray, y: np.ndarray, F: np.ndarray, lik: str) -> np.ndarray:
    """`log p(f, y)` for each row of `F`, up to the `|K|` constant."""
    Ki = np.linalg.inv(K)
    quad = np.einsum("ij,jk,ik->i", F, Ki, F)
    z = y * F
    ll = (log_ndtr(z) if lik == "probit" else -np.logaddexp(0.0, -z)).sum(axis=1)
    return -0.5 * quad + ll


def _exact_1d(k: float, y: float, lik: str, n: int = 40_001):
    """Exact posterior moments and `log Z` for a single latent variable."""
    f = np.linspace(-8.0, 8.0, n)
    logw = -0.5 * f**2 / k + (log_ndtr(y * f) if lik == "probit"
                              else -np.logaddexp(0.0, -y * f))
    top = logw.max()
    w = np.exp(logw - top)
    cell = f[1] - f[0]
    mean = float((w * f).sum() / w.sum())
    var = float((w * (f - mean) ** 2).sum() / w.sum())
    logz = top + math.log(w.sum() * cell) - 0.5 * math.log(2.0 * math.pi * k)
    return mean, var, logz


def _exact_2d(K: np.ndarray, y: np.ndarray, lik: str, lo: float = -6.0,
              hi: float = 6.0, n: int = 121):
    """Exact posterior over two latents on a grid: moments, `log Z`, samples."""
    g = np.linspace(lo, hi, n)
    F = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    logw = _log_joint(K, y, F, lik)
    top = logw.max()
    w = np.exp(logw - top)
    cell = (g[1] - g[0]) ** 2
    mean = (w[:, None] * F).sum(0) / w.sum()
    cov = (F - mean).T @ (w[:, None] * (F - mean)) / w.sum()
    logz = (top + math.log(w.sum() * cell) - math.log(2.0 * math.pi)
            - 0.5 * float(np.linalg.slogdet(K)[1]))
    return F, w / w.sum(), mean, cov, logz


def _exact_predictive(F, w, K, k_s, k_ss, lik: str) -> float:
    """`∫ p(y*=1|f*) p(f*|f) p(f|y) df* df`, the last integral by quadrature."""
    Ki = np.linalg.inv(K)
    mu = F @ (Ki @ k_s)
    v = max(k_ss - float(k_s @ Ki @ k_s), 1e-12)
    if lik == "probit":
        p = ndtr(mu / math.sqrt(1.0 + v))
    else:
        x, wq = np.polynomial.hermite.hermgauss(20)
        p = (wq * sigmoid(mu[:, None] + math.sqrt(2.0 * v) * x)).sum(1) * INV_SQRT_PI
    return float((w * p).sum())


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- one data point: EP is moment-exact, and that is a theorem ---------
    # With a single site the tilted distribution *is* the exact posterior, so
    # moment matching reproduces its mean and variance exactly. Laplace cannot:
    # it reports the mode, which for a skewed posterior is not the mean.
    k = 1.0
    mean1, var1, logz1 = _exact_1d(k, 1.0, "probit")
    assert np.isclose(logz1, -math.log(2.0), atol=1e-9)   # by symmetry
    ep1 = GPClassifier("probit", sigma_f=math.sqrt(k), jitter=0.0)
    ep1.fit(np.zeros((1, 1)), np.ones(1))
    lap1 = GPClassifier("probit", sigma_f=math.sqrt(k), jitter=0.0,
                        inference="laplace")
    lap1.fit(np.zeros((1, 1)), np.ones(1))
    assert abs(float(ep1.f_[0]) - mean1) < 1e-8, (ep1.f_[0], mean1)
    assert abs(float(ep1.Sigma_[0, 0]) - var1) < 1e-8, (ep1.Sigma_[0, 0], var1)
    print(
        f"gp_classification: probit — one point, exact posterior mean {mean1:.4f} "
        f"and variance {var1:.4f}; EP reproduces both to 1e−8 by construction, "
        f"Laplace reports the mode {float(lap1.f_[0]):.4f} and variance "
        f"{float(1.0 / (1.0 / k + lap1.sqrtW_[0] ** 2)):.4f}"
    )

    # --- two points: the grid is exact, so the two errors are comparable ----
    X2 = np.array([[-0.8], [0.9]])
    y2 = np.array([1.0, -1.0])
    K2 = rbf(X2, X2, 1.0, 1.0)
    F2, w2, mean2, cov2, logz2 = _exact_2d(K2, y2, "probit")
    res = {}
    for name, kwargs in (("laplace", {"inference": "laplace"}),
                         ("ep", {"inference": "ep"})):
        m = GPClassifier("probit", **kwargs).fit(X2, y2)
        res[name] = (m, float(np.max(np.abs(m.f_ - mean2))), abs(m.logZ_ - logz2))
    assert res["ep"][1] < res["laplace"][1]
    assert res["ep"][2] < res["laplace"][2]
    print(
        f"gp_classification: probit — two points against a 121×121 grid: EP is "
        f"{res['ep'][1]:.1e} off the posterior mean and {res['ep'][2]:.1e} nats off "
        f"the evidence, Laplace {res['laplace'][1]:.1e} and {res['laplace'][2]:.1e}"
    )

    # --- the predictive: plugging in the mode is not the same as integrating -
    # With a logistic likelihood the integral has no closed form, so the
    # question "which of these three is closest to the truth" is worth asking.
    ylog = np.array([1.0, -1.0])
    Fl, wl, _, _, _ = _exact_2d(K2, ylog, "logistic")
    x_test = np.array([[-1.8]])
    ks = rbf(x_test, X2, 1.0, 1.0)[0]
    exact_p = _exact_predictive(Fl, wl, K2, ks, 1.0, "logistic")
    plug = GPClassifier("laplace").fit(X2, ylog).predict_proba(x_test)[0]
    integ = GPClassifier("logistic").fit(X2, ylog).predict_proba(x_test)[0]
    ep_p = GPClassifier("expectation_propagation").fit(X2, ylog).predict_proba(x_test)[0]
    assert abs(ep_p - exact_p) < abs(integ - exact_p) < abs(plug - exact_p)
    # the plug-in is not merely wrong, it is overconfident: the logit at the mode
    # is further from zero than the averaged probability, always
    assert abs(plug - 0.5) > abs(exact_p - 0.5)
    print(
        f"gp_classification: logistic — exact predictive {exact_p:.4f} at the test "
        f"point; plugging the mode in gives {plug:.4f} (overconfident), integrating "
        f"gives {integ:.4f}, EP {ep_p:.4f}"
    )

    # --- and on real-sized data the two approximations agree where it counts -
    n, d = 60, 2
    X = rng.normal(size=(n, d))
    w_true = np.array([1.2, -0.8])
    f_true = X @ w_true + 0.4 * np.sin(3.0 * X[:, 0])
    ylog = np.where(f_true + 0.3 * rng.normal(size=n) > 0.0, 1.0, -1.0)
    Xt = rng.normal(size=(400, d))
    yt = np.where(Xt @ w_true + 0.4 * np.sin(3.0 * Xt[:, 0]) > 0.0, 1.0, -1.0)
    big = {name: GPClassifier(name, length_scale=1.2).fit(X, ylog)
           for name in ("laplace", "expectation_propagation")}
    acc = {name: float(np.mean(m.predict(Xt) == yt)) for name, m in big.items()}
    gap = float(np.max(np.abs(big["laplace"].predict_proba(Xt)
                              - big["expectation_propagation"].predict_proba(Xt))))
    assert min(acc.values()) > 0.85 and gap < 0.05, (acc, gap)
    assert np.all(big["expectation_propagation"].tau_ > 0.0)
    print(
        f"gp_classification: laplace — {n} points, {d} features: Laplace scores "
        f"{acc['laplace']:.3f} and EP {acc['expectation_propagation']:.3f} on 400 "
        f"held-out points, and the two predictives differ by at most {gap:.4f}; "
        f"EP needs {big['expectation_propagation'].n_iter_} sweeps"
    )

    # --- more than two classes: k binary problems and a softmax over logits --
    centres = np.array([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
    y3 = np.repeat([0, 1, 2], 25)
    X3 = centres[y3] + 0.45 * rng.normal(size=(75, 2))
    ovr = GPClassifier("one_vs_rest", length_scale=1.0).fit(X3, y3)
    y3t = np.repeat([0, 1, 2], 100)
    X3t = centres[y3t] + 0.45 * rng.normal(size=(300, 2))
    P = ovr.predict_proba(X3t)
    assert P.shape == (300, 3) and np.allclose(P.sum(axis=1), 1.0)
    acc3 = float(np.mean(ovr.predict(X3t) == y3t))
    assert acc3 > 0.9, acc3
    print(
        f"gp_classification: one_vs_rest — three classes, {acc3:.3f} accuracy on 300 "
        f"held-out points from k = {len(ovr.models_)} binary models, combined by a "
        f"softmax over per-class log-odds"
    )

    print("gp_classification demo passed")


if __name__ == "__main__":
    main()
