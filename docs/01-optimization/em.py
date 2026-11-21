"""EM and friends — 6 latent-variable optimisers.

Chapter:   01-optimization
Companion: em.md
Tier 1 · 6 variants: em, mm, map_em, gmm_em, incremental_em, hard_em

EM is a template, not an algorithm: any model with a latent variable and a
tractable complete-data M-step gets a monotone likelihood increase for free.
The variants here are the template (`em`), its generalisation (`mm`), its Bayesian
form (`map_em`), the canonical instance (`gmm_em`), its online form
(`incremental_em`) and its hard-assignment limit (`hard_em`, which is k-means).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Generic driver
# --------------------------------------------------------------------------- #
def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))


def run_em(estep, mstep, init, iters: int = 200, tol: float = 1e-10, loglik=None):
    """The EM loop, with the model supplied as two callbacks.

    estep(params) -> responsibilities (or any complete-data sufficient statistics)
    mstep(resp, params) -> new params
    loglik(params) -> observed-data log-likelihood, for the monotonicity check
    """
    params = init
    hist = []
    for _ in range(iters):
        resp = estep(params)
        params = mstep(resp, params)
        if loglik is not None:
            hist.append(loglik(params))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol:
            break
    return params, hist


# --------------------------------------------------------------------------- #
# em — Bernoulli mixture, the smallest model where EM is non-trivial
# --------------------------------------------------------------------------- #
def bernoulli_em(
    X: np.ndarray, k: int = 2, iters: int = 300, prior=None,
    rng: np.random.Generator | None = None,
):
    """Mixture of k Bernoulli distributions: E-step responsibilities, M-step
    weighted frequencies. `prior=(a, b)` on the mixing weights and `(c, d)` on the
    emissions turns this into MAP-EM (see `map_em`)."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    def log_prob(params):
        pi, theta = params
        theta = np.clip(theta, 1e-12, 1 - 1e-12)
        return (np.log(pi)
                + X @ np.log(theta).T
                + (1.0 - X) @ np.log1p(-theta).T)

    def estep(params):
        lp = log_prob(params)
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        return r / r.sum(axis=1, keepdims=True)

    def mstep(r, params):
        nk = r.sum(axis=0)
        if prior is None:
            pi = (nk + 1e-12) / n
            theta = (r.T @ X + 1e-12) / (nk[:, None] + 1e-12)
        else:
            (a, b), (c, e) = prior
            pi = (nk + a - 1.0) / (n + a + b - 2.0)
            theta = (r.T @ X + c - 1.0) / (nk[:, None] + c + e - 2.0)
        return pi, theta

    init = (np.full(k, 1.0 / k),
            rng.uniform(0.25, 0.75, size=(k, d)))
    params, hist = run_em(
        estep, mstep, init, iters,
        loglik=lambda p: float(_logsumexp(log_prob(p)).sum()),
    )
    return params, hist


# --------------------------------------------------------------------------- #
# mm — minorise-maximise, demonstrated on a Student-t location
# --------------------------------------------------------------------------- #
def student_t_mm(x: np.ndarray, nu: float = 4.0, iters: int = 300):
    """MM for the location of a Student-t. The majoriser of
    log(1 + r²/ν) at r₀ is quadratic, so each step is a weighted mean:

        w_i = (ν+1)/(ν + r_i²),   μ ← Σ w_i x_i / Σ w_i

    Every iteration decreases the negative log-likelihood — not approximately:
    the surrogate is tangent at the current point, which is the MM guarantee.
    """
    x = np.asarray(x, dtype=float)
    mu = float(np.median(x))
    hist = []
    for _ in range(iters):
        r2 = (x - mu) ** 2
        w = (nu + 1.0) / (nu + r2)
        mu_new = float((w * x).sum() / w.sum())
        hist.append(0.5 * (nu + 1.0) * float(np.log1p(r2 / nu).sum()))
        if abs(mu_new - mu) < 1e-12:
            break
        mu = mu_new
    return mu, hist


# --------------------------------------------------------------------------- #
# gmm_em — the canonical instance, full or spherical covariances
# --------------------------------------------------------------------------- #
def _log_gaussian(X, mu, cov):
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = X - mu
    sol = np.linalg.solve(L, diff.T).T
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + logdet + (sol * sol).sum(axis=1))


def gmm_em(
    X: np.ndarray, k: int = 3, iters: int = 300, reg: float = 1e-5,
    cov: str = "full", hard: bool = False, rng: np.random.Generator | None = None,
):
    """Gaussian mixture by EM. `cov='spherical'` + `hard=True` is exactly Lloyd's
    k-means; `hard=True` alone is classification EM."""
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    means = X[idx].copy()
    covs = np.array([np.cov(X.T) + reg * np.eye(d) for _ in range(k)])
    pi = np.full(k, 1.0 / k)

    def e_step(means, covs, pi, hard):
        if covs.ndim == 2:  # shared spherical covariance
            lp = np.column_stack([_log_gaussian(X, means[j], covs) for j in range(k)])
        else:
            lp = np.column_stack([_log_gaussian(X, means[j], covs[j]) for j in range(k)])
        lp = lp + np.log(pi)
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        r /= r.sum(axis=1, keepdims=True)
        if hard:
            r = np.eye(k)[r.argmax(axis=1)]
        return r

    def loglik(means, covs, pi):
        if covs.ndim == 2:
            lp = np.column_stack([_log_gaussian(X, means[j], covs) for j in range(k)])
        else:
            lp = np.column_stack([_log_gaussian(X, means[j], covs[j]) for j in range(k)])
        lp = lp + np.log(pi)
        m = lp.max(axis=1, keepdims=True)
        return float((m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True))).sum())

    hist = []
    for _ in range(iters):
        r = e_step(means, covs, pi, hard)
        nk = r.sum(axis=0) + 1e-12
        pi = nk / n
        means = (r.T @ X) / nk[:, None]
        if cov == "spherical":
            var = float((r * ((X[:, None, :] - means[None]) ** 2).sum(-1)).sum()
                        / (n * d))
            covs = np.eye(d) * max(var, reg)
        else:
            covs = np.empty((k, d, d))
            for j in range(k):
                diff = X - means[j]
                covs[j] = (r[:, j, None] * diff).T @ diff / nk[j] + reg * np.eye(d)
        hist.append(loglik(means, covs, pi))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < 1e-8:
            break
    return {"means": means, "covs": covs, "pi": pi, "resp": r, "hist": hist}


# --------------------------------------------------------------------------- #
# incremental_em — one observation at a time, step size 1/t
# --------------------------------------------------------------------------- #
def incremental_em(
    x: np.ndarray, k: int = 2, sigma2: float = 1.0, epochs: int = 30,
    rng: np.random.Generator | None = None,
):
    """Online EM for a location mixture with known variance: sufficient statistics
    are updated with one point at a time and averaged with weight 1/t."""
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    n = x.size
    means = rng.choice(x, size=k, replace=False).astype(float)
    pi = np.full(k, 1.0 / k)
    nk = np.zeros(k)
    sx = np.zeros(k)
    t = 0

    def loglik():
        lp = -0.5 * (x[:, None] - means[None]) ** 2 / sigma2 + np.log(pi)[None]
        m = lp.max(axis=1, keepdims=True)
        return float((m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True))).sum())

    hist = [loglik()]
    for _ in range(epochs):
        for i in rng.permutation(n):
            t += 1
            lp = -0.5 * (x[i] - means) ** 2 / sigma2 + np.log(pi)
            lp -= lp.max()
            r = np.exp(lp)
            r /= r.sum()
            gamma = 1.0 / max(t / k, 1.0)
            nk = (1 - gamma) * nk + gamma * r
            sx = (1 - gamma) * sx + gamma * r * x[i]
            means = sx / np.maximum(nk, 1e-12)
            pi = nk / nk.sum()
        hist.append(loglik())
    return means, pi, hist


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # 1. Bernoulli mixture: EM recovers the two coins and the mixing weight, with
    #    a monotone log-likelihood.
    true_pi = np.array([0.6, 0.4])
    true_theta = np.array([[0.9, 0.8, 0.7, 0.2], [0.2, 0.3, 0.1, 0.8]])
    z = rng.choice(2, size=3000, p=true_pi)
    X = (rng.random((3000, 4)) < true_theta[z]).astype(float)
    (pi, theta), hist = bernoulli_em(X, k=2, rng=rng)
    order = np.argsort(-pi)
    assert hist[-1] >= hist[0]
    assert np.allclose(np.sort(pi)[::-1], np.sort(true_pi)[::-1], atol=0.05), pi
    assert np.allclose(theta[order], true_theta, atol=0.1), theta[order]
    print(f"em: pi = {np.round(pi, 3)}, LL {hist[0]:.1f} -> {hist[-1]:.1f}")

    # 2. MAP-EM: the same model with Beta(a, b) priors. A strongly informative
    #    prior around 0.5 shrinks the mixing weights toward it.
    prior = ((1000.0, 1000.0), (2.0, 2.0))
    (pi_map, _), _ = bernoulli_em(X, k=2, prior=prior, rng=rng)
    assert np.abs(pi_map - 0.5).max() < np.abs(pi - 0.5).max(), (pi_map, pi)
    print(f"map_em: max|pi-0.5| {np.abs(pi - 0.5).max():.3f} -> "
          f"{np.abs(pi_map - 0.5).max():.3f}")

    # 3. MM on a Student-t location: monotone NLL, and it ignores outliers that
    #    move the sample mean.
    clean = rng.normal(0.0, 1.0, size=200)
    noisy = np.r_[clean, rng.normal(15.0, 1.0, size=40)]   # 17% contamination
    mu, mm_hist = student_t_mm(noisy, nu=3.0)
    assert all(mm_hist[i] >= mm_hist[i + 1] - 1e-9 for i in range(len(mm_hist) - 1))
    assert abs(mu) < 0.25 and abs(noisy.mean() - 2.5) < 2.5, (mu, noisy.mean())
    print(f"mm: t-location {mu:.3f} vs mean {noisy.mean():.3f} (truth 0)")

    # 4. GMM: three blobs, recovered up to a permutation, monotone likelihood.
    mus = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    Xg = np.vstack([rng.normal(m, 0.6, size=(400, 2)) for m in mus])
    out = gmm_em(Xg, k=3, rng=rng)
    hist = out["hist"]
    assert all(hist[i] <= hist[i + 1] + 1e-6 for i in range(len(hist) - 1))
    d = np.linalg.norm(out["means"][:, None] - mus[None], axis=-1)
    assert d.min(axis=1).max() < 0.2, d
    for c in out["covs"]:
        assert np.all(np.linalg.eigvalsh(c) > 0)
    print(f"gmm_em: means matched within {d.min(axis=1).max():.3f}, LL {hist[0]:.1f} -> {hist[-1]:.1f}")

    # 5. Incremental EM reaches (almost) the batch solution with a single pass per
    #    epoch and O(k) memory, not an n x k responsibility matrix.
    x1 = np.r_[rng.normal(-2.0, 1.0, 500), rng.normal(2.0, 1.0, 500)]
    batch = gmm_em(x1.reshape(-1, 1), k=2, cov="spherical", rng=rng)
    inc_means, inc_pi, inc_hist = incremental_em(x1, k=2, rng=rng)
    assert inc_hist[-1] >= inc_hist[0] - 1e-9
    gap = np.abs(np.sort(inc_means) - np.sort(batch["means"].ravel())).max()
    assert gap < 0.1, (inc_means, batch["means"].ravel())
    print(f"incremental_em: means {np.round(inc_means, 3)} vs batch "
          f"{np.round(batch['means'].ravel(), 3)}")

    # 6. Hard EM with a shared spherical covariance *is* Lloyd's k-means: the
    #    assignment rule reduces to nearest-mean.
    out = gmm_em(Xg, k=3, cov="spherical", hard=True, rng=rng)
    means = out["means"]
    lloyd = means.copy()
    for _ in range(100):  # plain Lloyd, implemented here to compare
        assign = ((Xg[:, None] - lloyd[None]) ** 2).sum(-1).argmin(1)
        lloyd = np.array([Xg[assign == j].mean(0) for j in range(3)])
    a1 = ((Xg[:, None] - means[None]) ** 2).sum(-1).argmin(1)
    a2 = ((Xg[:, None] - lloyd[None]) ** 2).sum(-1).argmin(1)
    same_partition = np.array_equal(a1[:, None] == a1[None], a2[:, None] == a2[None])
    assert same_partition, "hard EM != Lloyd"
    print("hard_em: identical partition to Lloyd's k-means")
    print("em demo passed")


if __name__ == "__main__":
    main()
