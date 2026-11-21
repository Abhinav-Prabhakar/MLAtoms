"""Gaussian mixture models — EM with four covariance shapes and three priors.

Chapter:   07-clustering
Companion: gmm.md
Tier 1 · 7 variants: full, diag, spherical, tied, variational, bic_selection,
map_mixture

A GMM is a latent-variable model: each point is drawn from one of K Gaussians,
and the component is never observed. EM maximises the observed-data likelihood

    log p(X) = sum_i log sum_k pi_k N(x_i | mu_k, Sigma_k)

by alternating two closed-form steps:

    E: r_ik = pi_k N(x_i|mu_k,Sigma_k) / sum_j pi_j N(x_i|mu_j,Sigma_j)
    M: pi_k = N_k/n,  mu_k = sum_i r_ik x_i / N_k,
       Sigma_k = sum_i r_ik (x_i-mu_k)(x_i-mu_k)^T / N_k

The variants are the covariance parameterisation (full, diag, spherical, tied),
model selection by BIC, and two Bayesian regularisations (variational and MAP)
that keep the fit finite when a component collapses onto a few points.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(
    ["full", "diag", "spherical", "tied", "variational", "bic_selection",
     "map_mixture"]
)

EPS = 1e-12


def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


def _log_gauss(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a multivariate normal, via Cholesky."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = (X - mean).T
    sol = np.linalg.solve(L, diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(L)).sum()
                   + (sol * sol).sum(axis=0))


class GaussianMixture:
    """EM for a Gaussian mixture.

    `covariance_type` follows the variant unless overridden. `weight_prior` is the
    Dirichlet concentration for `variational`/`map_mixture`; `cov_prior` is the
    pseudo-count of the inverse-Wishart prior for `map_mixture`.
    """

    def __init__(
        self,
        variant: str = "full",
        *,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        n_init: int = 3,
        k_max: int = 6,
        weight_prior: float = 1.0,
        cov_prior: float = 1.0,
        random_state: int = 0,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.k_max = k_max
        self.alpha0 = weight_prior
        self.nu0 = float(cov_prior)
        self.rng = np.random.default_rng(random_state)
        if variant == "diag":
            self.covariance_type = "diag"
        elif variant == "spherical":
            self.covariance_type = "spherical"
        elif variant == "tied":
            self.covariance_type = "tied"
        else:
            self.covariance_type = "full"

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _init(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, d = X.shape
        # k-means++ seeding, then global-covariance components.
        idx = [int(self.rng.integers(n))]
        dist = ((X - X[idx[0]]) ** 2).sum(axis=1)
        for _ in range(1, k):
            total = float(dist.sum())
            if total <= EPS:
                idx.append(int(self.rng.integers(n)))
            else:
                idx.append(int(self.rng.choice(n, p=dist / total)))
            dist = np.minimum(dist, ((X - X[idx[-1]]) ** 2).sum(axis=1))
        means = X[idx].copy()
        global_cov = np.atleast_2d(np.cov(X.T, bias=True))
        global_cov = global_cov + self.reg_covar * np.eye(d)
        covs = np.repeat(global_cov[None], k, axis=0)
        weights = np.full(k, 1.0 / k)
        return weights, means, covs

    # ------------------------------------------------------------------ #
    # M-step
    # ------------------------------------------------------------------ #
    def _m_step(self, X, resp, weights, means, covs):
        n, d = X.shape
        k = weights.size
        Nk = resp.sum(axis=0)

        if self.variant == "map_mixture":
            prior = self.alpha0
            weights = (Nk + prior - 1.0) / (n + k * (prior - 1.0))
            weights = np.maximum(weights, EPS)
            weights /= weights.sum()
        else:
            weights = Nk / n

        prior_mean = X.mean(axis=0)
        prior_cov = np.cov(X.T, bias=True) + self.reg_covar * np.eye(d)
        for j in range(k):
            r = resp[:, j]
            nk = float(Nk[j])
            if self.variant == "map_mixture":
                kappa = self.nu0
                means[j] = (r @ X + kappa * prior_mean) / (nk + kappa)
            else:
                means[j] = (r @ X) / max(nk, EPS)

        for j in range(k):
            r = resp[:, j]
            diff = X - means[j]
            S = (r[:, None] * diff).T @ diff
            nk = float(Nk[j])
            if self.variant == "map_mixture":
                cov = (S + self.nu0 * prior_cov) / (nk + self.nu0 + d + 1.0)
            else:
                cov = S / max(nk, EPS)
            cov = self._project_cov(cov)
            covs[j] = cov

        if self.covariance_type == "tied":
            total = np.zeros((d, d))
            for j in range(k):
                r = resp[:, j]
                diff = X - means[j]
                total += (r[:, None] * diff).T @ diff
            shared = total / max(float(Nk.sum()), EPS)
            shared = self._project_cov(shared)
            covs[:] = shared
        return weights, means, covs

    def _project_cov(self, cov: np.ndarray) -> np.ndarray:
        cov = cov + self.reg_covar * np.eye(cov.shape[0])
        if self.covariance_type == "diag":
            return np.diag(np.diag(cov))
        if self.covariance_type == "spherical":
            return np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0])
        return cov

    # ------------------------------------------------------------------ #
    # E-step
    # ------------------------------------------------------------------ #
    def _e_step(self, X, weights, means, covs):
        k = weights.size
        log_prob = np.empty((X.shape[0], k))
        for j in range(k):
            log_prob[:, j] = _log_gauss(X, means[j], covs[j])
        if self.variant == "variational":
            alpha = getattr(self, "_alpha", None)
            if alpha is None or alpha.size != k:
                alpha = np.full(k, self.alpha0)
            Elogpi = _digamma(alpha) - _digamma(alpha.sum())
            log_prob = log_prob + Elogpi[None, :]
        else:
            log_prob = log_prob + np.log(np.maximum(weights, EPS))[None, :]
        lse = _logsumexp(log_prob)
        resp = np.exp(log_prob - lse[:, None])
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), EPS)
        return resp, float(lse.sum()), log_prob

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def _fit_once(self, X: np.ndarray, k: int):
        weights, means, covs = self._init(X, k)
        self._alpha = np.full(k, self.alpha0)
        prev = -np.inf
        history: list[float] = []
        for _it in range(self.max_iter):
            resp, loglik, _ = self._e_step(X, weights, means, covs)
            if self.variant == "variational":
                # Mean-field update: responsibilities use E[log pi] from the
                # posterior implied by the previous responsibilities.
                alpha = self.alpha0 + resp.sum(axis=0)
                self._alpha = alpha
                Elogpi = _digamma(alpha) - _digamma(alpha.sum())
                log_joint = _log_gauss_matrix(X, means, covs) + Elogpi[None, :]
                resp = np.exp(log_joint - _logsumexp(log_joint)[:, None])
                resp /= np.maximum(resp.sum(axis=1, keepdims=True), EPS)
                weights, means, covs = self._m_step(X, resp, weights, means, covs)
                weights = alpha / alpha.sum()
                bound = _elbo(X, resp, means, covs, alpha, self.alpha0)
            else:
                weights, means, covs = self._m_step(X, resp, weights, means, covs)
                bound = loglik
            history.append(float(bound))
            if bound - prev < self.tol * max(1.0, abs(prev)):
                prev = bound
                break
            prev = bound
        if self.variant == "variational":
            resp, _, _ = self._e_step(X, weights, means, covs)
            bound = _elbo(X, resp, means, covs, self._alpha, self.alpha0)
        else:
            resp, loglik, _ = self._e_step(X, weights, means, covs)
            bound = loglik
        n_params = (k - 1) + k * X.shape[1] + k * _cov_params(
            self.covariance_type, X.shape[1])
        bic = -2.0 * loglik + n_params * np.log(X.shape[0])
        return {
            "weights": weights, "means": means, "covs": covs,
            "resp": resp, "loglik": loglik, "bound": bound, "bic": bic,
            "n_iter": _it + 1, "alpha": getattr(self, "_alpha", None),
            "history": history,
        }

    def fit(self, X: np.ndarray) -> GaussianMixture:
        X = np.asarray(X, dtype=float)
        if self.variant == "bic_selection":
            best = None
            scores = {}
            for k in range(1, self.k_max + 1):
                res = max(
                    (self._fit_once(X, k) for _ in range(self.n_init)),
                    key=lambda r: r["bound"],
                )
                scores[k] = res["bic"]
                if best is None or res["bic"] < best[1]["bic"]:
                    best = (k, res)
            k, res = best
            self.n_components_ = k
            self.bic_ = res["bic"]
            self.bic_scores_ = [scores[j] for j in range(1, self.k_max + 1)]
        else:
            k = self.k
            res = max(
                (self._fit_once(X, k) for _ in range(self.n_init)),
                key=lambda r: r["bound"],
            )
            self.n_components_ = k
            self.bic_ = res["bic"]
        if res["alpha"] is not None:
            self._alpha = res["alpha"]
        self.weights_ = res["weights"] / res["weights"].sum()
        self.means_ = res["means"]
        self.covariances_ = res["covs"]
        self.lower_bound_ = float(res["bound"])
        self.bound_history_ = res["history"]
        self.loglik_ = float(res["loglik"])
        self.n_iter_ = int(res["n_iter"])
        # Fill in the final posterior for the chosen parameters.
        resp, loglik, _ = self._e_step(X, self.weights_, self.means_,
                                       self.covariances_)
        self.resp_ = resp
        self.labels_ = np.argmax(resp, axis=1)
        self.loglik_ = float(loglik)
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        resp, _, _ = self._e_step(np.asarray(X, dtype=float), self.weights_,
                                  self.means_, self.covariances_)
        return resp

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=float)
        log_prob = _log_gauss_matrix(X, self.means_, self.covariances_)
        log_prob = log_prob + np.log(np.maximum(self.weights_, EPS))[None, :]
        return float(_logsumexp(log_prob).mean())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _log_gauss_matrix(X: np.ndarray, means: np.ndarray,
                      covs: np.ndarray) -> np.ndarray:
    out = np.empty((X.shape[0], means.shape[0]))
    for j in range(means.shape[0]):
        out[:, j] = _log_gauss(X, means[j], covs[j])
    return out


def _digamma(x: np.ndarray) -> np.ndarray:
    """Digamma by recurrence + asymptotic series; no scipy dependency."""
    x = np.asarray(x, dtype=float).copy()
    out = np.zeros_like(x)
    while np.any(x < 6.0):
        small = x < 6.0
        out[small] -= 1.0 / x[small]
        x[small] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return out + np.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))


def _lgamma(x: float) -> float:
    return float(math.lgamma(x))


def _elbo(X, resp, means, covs, alpha, alpha0) -> float:
    """Dirichlet-prior variational bound for the mixing weights."""
    k = alpha.size
    Elogpi = _digamma(alpha) - _digamma(alpha.sum())
    log_joint = _log_gauss_matrix(X, means, covs) + Elogpi[None, :]
    expected_ll = float((resp * log_joint).sum())
    log_b_alpha = sum(_lgamma(a) for a in alpha) - _lgamma(float(alpha.sum()))
    log_b_prior = k * _lgamma(alpha0) - _lgamma(k * alpha0)
    kl = (log_b_prior - log_b_alpha
          + float(((alpha - alpha0) * Elogpi).sum()))
    return expected_ll - kl


def _cov_params(kind: str, d: int) -> float:
    if kind == "full":
        return d * (d + 1) / 2.0
    if kind == "diag":
        return float(d)
    if kind == "spherical":
        return 1.0
    return d * (d + 1) / 2.0      # tied: one shared covariance


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _match_accuracy(labels: np.ndarray, truth: np.ndarray, k: int) -> float:
    import itertools
    best = 0.0
    for perm in itertools.permutations(range(k)):
        best = max(best, float(np.mean(np.asarray(perm)[labels] == truth)))
    return best


def main() -> None:
    rng = np.random.default_rng(0)
    n = 600
    truth = rng.integers(0, 3, size=n)
    centres = np.array([[-3.0, 0.0], [3.0, 0.0], [0.0, 3.5]])
    X = centres[truth] + rng.normal(0.0, 0.9, size=(n, 2))

    # --- full covariance: the reference EM ---
    gm = GaussianMixture("full", n_components=3, n_init=5, random_state=1).fit(X)
    acc = _match_accuracy(gm.labels_, truth, 3)
    assert acc > 0.95, acc
    assert np.allclose(gm.weights_.sum(), 1.0)
    assert abs(np.linalg.det(gm.covariances_[0])) > 0.0
    print(f"gmm: full acc {acc:.3f}, loglik {gm.loglik_:.1f}, "
          f"BIC {gm.bic_:.1f}, iters {gm.n_iter_}")

    # --- diagonal and spherical: the covariance structure is enforced ---
    diag = GaussianMixture("diag", n_components=3, n_init=3,
                           random_state=2).fit(X)
    off = max(np.abs(c - np.diag(np.diag(c))).max()
              for c in diag.covariances_)
    assert off < 1e-12, off
    assert diag.score(X) > gm.score(X) - 0.5

    sph = GaussianMixture("spherical", n_components=3, n_init=3,
                          random_state=3).fit(X)
    for c in sph.covariances_:
        assert np.allclose(c, np.eye(2) * c[0, 0])
    acc_s = _match_accuracy(sph.labels_, truth, 3)
    assert acc_s > 0.9, acc_s
    print(f"gmm: diag off-diagonal {off:.1e}, spherical acc {acc_s:.3f}")

    # --- tied: one shared covariance, fewer parameters ---
    cov_shared = np.array([[1.2, 0.4], [0.4, 0.6]])
    Xt = np.vstack([
        rng.multivariate_normal(c, cov_shared, size=200) for c in centres
    ])
    tied = GaussianMixture("tied", n_components=3, n_init=3,
                           random_state=4).fit(Xt)
    for c in tied.covariances_[1:]:
        assert np.allclose(c, tied.covariances_[0])
    full_t = GaussianMixture("full", n_components=3, n_init=3,
                             random_state=4).fit(Xt)
    gap = full_t.loglik_ - tied.loglik_
    assert gap < 30.0, gap
    print(f"gmm: tied loglik {tied.loglik_:.1f} vs full {full_t.loglik_:.1f} "
          f"(gap {gap:.1f}) on shared-covariance data")

    # --- variational: a Dirichlet prior, ELBO monotone, sparse weights ---
    var = GaussianMixture("variational", n_components=5, n_init=1,
                          weight_prior=0.3, random_state=5).fit(X)
    assert np.allclose(var.weights_.sum(), 1.0)
    assert np.isfinite(var.lower_bound_)
    assert np.all(np.diff(var.bound_history_) >= -1e-6), var.bound_history_[:5]
    assert var.weights_.min() < 1.0 / 5.0
    print(f"gmm: variational weights {np.round(var.weights_, 3)}, "
          f"ELBO {var.lower_bound_:.1f}")

    # --- BIC selection recovers three components ---
    sel = GaussianMixture("bic_selection", k_max=6, n_init=2,
                          random_state=6).fit(X)
    assert sel.n_components_ == 3, sel.n_components_
    assert len(sel.bic_scores_) == 6
    k_best = int(np.argmin(sel.bic_scores_)) + 1
    assert k_best == sel.n_components_
    print(f"gmm: BIC selects k={sel.n_components_} "
          f"(scores {np.round(sel.bic_scores_, 1)})")

    # --- MAP: priors keep a collapsing component's covariance finite ---
    mapm = GaussianMixture("map_mixture", n_components=5, n_init=1,
                           weight_prior=1.5, cov_prior=2.0,
                           random_state=7).fit(X)
    plain = GaussianMixture("full", n_components=5, n_init=1,
                            random_state=7).fit(X)
    min_map = min(np.linalg.eigvalsh(c).min() for c in mapm.covariances_)
    min_plain = min(np.linalg.eigvalsh(c).min() for c in plain.covariances_)
    assert min_map > min_plain, (min_map, min_plain)
    assert min_map > 0.0
    print(f"gmm: MAP min covariance eigenvalue {min_map:.3f} "
          f"vs unregularised {min_plain:.4f}")

    # --- posterior responsibilities are a proper distribution ---
    proba = gm.predict_proba(X[:5])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)
    assert np.array_equal(gm.predict(X), gm.labels_)
    print("gmm demo passed")


if __name__ == "__main__":
    main()
