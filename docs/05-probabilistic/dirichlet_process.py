"""Dirichlet processes — a prior over partitions, and the mixtures it builds.

Chapter:   05-probabilistic
Companion: dirichlet_process.md
Tier 3 · 5 variants: crp, stick_breaking, dpmm, truncated_dp, dp_means

    G ~ DP(α, G₀)         a distribution over distributions
    x_i | G ~ G           the data are draws from a draw

Two ways to write the same object, and they are both used below:

    stick-breaking   G = Σ_k w_k δ_{θ_k},  w_k = v_k Π_{j<k}(1−v_j)
    Chinese restaurant  p(z_i = k) ∝ n_k,  p(z_i = new) ∝ α

The first says what a DP *is*; the second says what it *does* to a sequence of
data, and it is the one that turns into a sampling algorithm.

    crp            the partition prior: EPPF, predictive rule, sampler
    stick_breaking the weights: GEM, truncation error, decay
    dpmm           the infinite mixture, fitted by collapsed Gibbs
    truncated_dp   the same mixture with K components, so it fits in a matrix
    dp_means       the small-variance limit: k-means that chooses its own k

The demo checks exact statements where they exist — the EPPF sums to one over
all partitions, E[K_n] = Σ α/(α+i−1), E[w_k] = α^{k−1}/(1+α)^k, the truncated
mixing measure is exactly the residual mass away from the full one, and the
NIW posterior predictive is a Student-t that a numerical integral confirms.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["crp", "stick_breaking", "dpmm", "truncated_dp", "dp_means"])


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def _log_t(x, df: float, loc: float, scale: float) -> np.ndarray:
    """`log t_df(x; loc, scale)`, elementwise — the NIW posterior predictive."""
    z = (np.asarray(x, dtype=float) - loc) / scale
    return (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
            - 0.5 * math.log(df * math.pi) - math.log(scale)
            - 0.5 * (df + 1.0) * np.log1p(z * z / df))


def _niw_update(X: np.ndarray, mu0: float, kappa0: float, a0: float, b0: float):
    """Normal–inverse-χ² posterior for one coordinate, given its data.

    `X` is `(n,)` and may be empty, which is how the prior predictive is
    obtained: the same formula with `n = 0`.
    """
    X = np.asarray(X, dtype=float)
    n = X.size
    xbar = X.mean() if n else 0.0
    ss = float(((X - xbar) ** 2).sum()) if n else 0.0
    kappa = kappa0 + n
    mu = (kappa0 * mu0 + n * xbar) / kappa
    a = a0 + 0.5 * n
    b = b0 + 0.5 * ss + 0.5 * kappa0 * n / kappa * (xbar - mu0) ** 2
    return mu, kappa, a, b


def _predictive_logpdf(x: float, params) -> float:
    """`p(x | cluster)` after integrating `(μ, σ²)` out of the NIW."""
    mu, kappa, a, b = params
    scale = math.sqrt(b * (kappa + 1.0) / (a * kappa))
    return float(_log_t(x, 2.0 * a, mu, scale))


def _partitions(n: int):
    """Every set partition of `{1..n}`, as label tuples in restricted growth form."""
    out: list[tuple[int, ...]] = []

    def rec(i: int, labels: list[int], k: int) -> None:
        if i == n:
            out.append(tuple(labels))
            return
        for j in range(k + 1):
            labels.append(j)
            rec(i + 1, labels, max(k, j + 1))
            labels.pop()

    rec(0, [], 0)
    return out


# --------------------------------------------------------------------------- #
# The process
# --------------------------------------------------------------------------- #
class DirichletProcess:
    """One Dirichlet process, five finite faces.

    `crp` and `stick_breaking` are the prior itself: they expose
    `sample_partition`, `eppf` and `gem`, and `fit` refuses, because a DP has no
    parameters to fit. The other three variants take data and set `labels_`.
    """

    def __init__(self, variant: str = "crp", alpha: float = 1.0, *, K: int = 10,
                 lam: float = 10.0, mu0: float = 0.0, kappa0: float = 1.0,
                 a0: float = 2.0, b0: float = 1.0, n_iter: int = 50, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.variant = variant
        self.alpha = float(alpha)
        self.K = int(K)
        self.lam = float(lam)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

    # -- the partition prior (crp) ------------------------------------------ #
    def eppf(self, counts) -> float:
        """Exchangeable partition probability of cluster sizes `counts`."""
        counts = np.asarray(counts, dtype=float)
        n, k = counts.sum(), counts.size
        logp = (k * math.log(self.alpha)
                + float(sum(math.lgamma(c) for c in counts))
                + math.lgamma(self.alpha) - math.lgamma(self.alpha + n))
        return math.exp(logp)

    def expected_clusters(self, n: int) -> float:
        """`E[K_n] = Σ_{i=1}^n α/(α+i−1)`, exact for a CRP."""
        return float(sum(self.alpha / (self.alpha + i - 1.0) for i in range(1, n + 1)))

    def sample_partition(self, n: int, rng=None) -> np.ndarray:
        """Draw labels from the CRP by seating tables one at a time."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        labels = np.zeros(n, dtype=int)
        counts = [1]
        for i in range(1, n):
            p = np.array([c / (self.alpha + i) for c in counts]
                         + [self.alpha / (self.alpha + i)])
            j = int(rng.choice(p.size, p=p))
            if j == len(counts):
                counts.append(1)
            else:
                counts[j] += 1
            labels[i] = j
        return labels

    # -- the weights (stick_breaking) --------------------------------------- #
    def gem(self, K: int | None = None, rng=None) -> tuple[np.ndarray, float]:
        """`K` stick-breaking weights and the mass they leave behind."""
        K = self.K if K is None else int(K)
        rng = np.random.default_rng(self.seed) if rng is None else rng
        # V ~ Beta(1, α) by inverse CDF: 1 − V ~ Beta(α, 1)
        v = 1.0 - rng.random(K) ** (1.0 / self.alpha)
        w = v * np.concatenate([[1.0], np.cumprod(1.0 - v)[:-1]])
        return w, float(np.prod(1.0 - v))

    def posterior_weights(self, counts) -> tuple[np.ndarray, float]:
        """`E[w_k | counts]` under the `K`-component stick-breaking posterior.

        With `v_j | counts ~ Beta(1+n_j, α+Σ_{l>j}n_l)` independently, the mean
        weight is a product of beta means — no sampling, and it is the exact
        conditional a Gibbs sweep needs.
        """
        counts = np.asarray(counts, dtype=float)
        tail = np.concatenate([np.cumsum(counts[::-1])[::-1][1:], [0.0]])
        ev = (1.0 + counts) / (1.0 + counts + self.alpha + tail)
        w = ev * np.concatenate([[1.0], np.cumprod(1.0 - ev)[:-1]])
        return w, float(np.prod(1.0 - ev))

    # -- fitting ------------------------------------------------------------- #
    def fit(self, X) -> DirichletProcess:
        """Fit a DP mixture, a truncated DP mixture, or DP-means."""
        if self.variant in ("crp", "stick_breaking"):
            raise ValueError(
                f"the {self.variant} variant is the prior, not a fitted model: "
                "use sample_partition() / eppf() / gem() instead"
            )
        X = np.asarray(X, dtype=float)
        X = X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
        self.d_ = X.shape[1]
        if self.variant == "dpmm":
            return self._dpmm(X)
        if self.variant == "truncated_dp":
            return self._truncated_dp(X)
        return self._dp_means(X)

    def _cluster_params(self, X: np.ndarray, labels: np.ndarray, K: int | None = None):
        """Per-coordinate NIW posteriors, one per cluster; empty gives the prior."""
        K = (int(labels.max()) + 1 if labels.size else 0) if K is None else int(K)
        params = []
        for k in range(K):
            cols = []
            for j in range(X.shape[1]):
                cols.append(_niw_update(X[labels == k, j], self.mu0, self.kappa0,
                                        self.a0, self.b0))
            params.append(cols)
        return params

    def _log_predictive(self, x: np.ndarray, params) -> float:
        """`log p(x | cluster)`, a product of independent Student-t over dims."""
        return float(sum(_predictive_logpdf(float(x[j]), params[j])
                         for j in range(x.size)))

    def _dpmm(self, X: np.ndarray) -> DirichletProcess:
        """Collapsed Gibbs: the prior on assignments is the CRP."""
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        z = self.sample_partition(n)
        self.history_ = []
        prior = [_niw_update(np.empty(0), self.mu0, self.kappa0, self.a0, self.b0)
                 for _ in range(self.d_)]
        for _it in range(self.n_iter):
            moved = 0
            for i in range(n):
                old = z[i]
                z[i] = -1                                   # leave the table
                labels = z[z >= 0]
                params = self._cluster_params(X[z >= 0], labels)
                logp = []
                for k in np.unique(labels):
                    c = int((labels == k).sum())
                    logp.append(math.log(c) + self._log_predictive(X[i], params[k]))
                # a new table: the prior predictive, weighted by alpha
                logp.append(math.log(self.alpha) + self._log_predictive(X[i], prior))
                p = np.exp(np.array(logp) - max(logp))
                p /= p.sum()
                pick = int(rng.choice(p.size, p=p))
                if pick == len(logp) - 1:
                    z[i] = int(labels.max()) + 1 if labels.size else 0
                else:
                    z[i] = int(np.unique(labels)[pick])
                moved += int(z[i] != old)
            self.history_.append(moved)
        _, z = np.unique(z, return_inverse=True)
        self.labels_ = z
        self.K_ = int(z.max()) + 1 if z.size else 0
        self.params_ = self._cluster_params(X, z)
        self.counts_ = np.bincount(z, minlength=self.K_)
        self.centroids_ = np.array([X[z == k].mean(0) for k in range(self.K_)])
        return self

    def _truncated_dp(self, X: np.ndarray) -> DirichletProcess:
        """The same sampler with `K` sticks: the prior is finite, so `w_k` is."""
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        z = rng.integers(0, self.K, size=n)
        self.history_ = []
        for _it in range(self.n_iter):
            moved = 0
            for i in range(n):
                counts = np.bincount(z, minlength=self.K).astype(float)
                counts[z[i]] -= 1.0
                w, _ = self.posterior_weights(counts)
                # leave point i out of its own cluster's posterior, or the
                # density at x_i is inflated by its own contribution
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                params = self._cluster_params(X[mask], z[mask], self.K)
                logp = np.log(np.maximum(w, 1e-300)) + np.array(
                    [self._log_predictive(X[i], params[k]) for k in range(self.K)]
                )
                p = np.exp(logp - logp.max())
                p /= p.sum()
                new = int(rng.choice(self.K, p=p))
                moved += int(new != z[i])
                z[i] = new
            self.history_.append(moved)
        _, z = np.unique(z, return_inverse=True)            # drop the empty sticks
        self.labels_ = z
        self.K_ = int(z.max()) + 1 if z.size else 0
        self.params_ = self._cluster_params(X, z)
        self.counts_ = np.bincount(z, minlength=self.K_)
        # weights of the K_ occupied components, and the mass beyond them: that
        # residual *is* the truncation error of the fitted mixture
        self.weights_, self.residual_ = self.posterior_weights(self.counts_)
        self.centroids_ = np.array([X[z == k].mean(0) for k in range(self.K_)])
        return self

    def _dp_means(self, X: np.ndarray) -> DirichletProcess:
        """Small-variance limit: assign to the nearest mean, split if too far."""
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        centroids = X.mean(axis=0, keepdims=True)
        labels = np.zeros(n, dtype=int)
        self.history_ = []
        for _it in range(self.n_iter):
            before = labels.copy()
            for i in rng.permutation(n):
                d2 = ((centroids - X[i]) ** 2).sum(axis=1)
                j = int(d2.argmin())
                if d2[j] > self.lam:            # cheaper to pay λ than to fit
                    centroids = np.vstack([centroids, X[i]])
                    j = centroids.shape[0] - 1
                labels[i] = j
            keep = np.unique(labels)
            remap = {int(k): j for j, k in enumerate(keep)}
            labels = np.array([remap[int(k)] for k in labels])
            centroids = np.array([X[labels == k].mean(axis=0) for k in range(keep.size)])
            sse = float(((X - centroids[labels]) ** 2).sum())
            self.history_.append(sse + self.lam * keep.size)
            if np.array_equal(labels, before):
                break
        self.labels_ = labels
        self.K_ = int(keep.size)
        self.centroids_ = centroids
        self.objective_ = self.history_[-1]
        return self

    # -- prediction ---------------------------------------------------------- #
    def _weights(self) -> tuple[np.ndarray, float]:
        if self.variant == "truncated_dp":
            return self.weights_, self.residual_
        n = float(self.counts_.sum())
        w = self.counts_ / (self.alpha + n)
        return w, self.alpha / (self.alpha + n)

    def predict_log_density(self, X) -> np.ndarray:
        """`log p(x*)`, including the new-cluster term for `dpmm`."""
        X = np.asarray(X, dtype=float)
        X = X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
        w, new = self._weights()
        prior = [_niw_update(np.empty(0), self.mu0, self.kappa0, self.a0, self.b0)
                 for _ in range(self.d_)]
        out = np.empty(X.shape[0])
        for i, x in enumerate(X):
            terms = [math.log(w[k]) + self._log_predictive(x, self.params_[k])
                     for k in range(self.K_)]
            terms.append(math.log(new) + self._log_predictive(x, prior))
            top = max(terms)
            out[i] = top + math.log(sum(math.exp(t - top) for t in terms))
        return out

    def predict(self, X) -> np.ndarray:
        """MAP component: nearest centroid for `dp_means`, else the density."""
        X = np.asarray(X, dtype=float)
        X = X.reshape(-1, 1) if X.ndim == 1 else np.atleast_2d(X)
        if self.variant == "dp_means":
            d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
            return d2.argmin(axis=1)
        w, _ = self._weights()
        D = np.array([[self._log_predictive(x, self.params_[k]) for k in range(self.K_)]
                      for x in X])
        return (D + np.log(w)[None, :]).argmax(axis=1)


# --------------------------------------------------------------------------- #
# Demo helpers
# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 100):
    """Textbook Lloyd, used only as the reference DP-means has to match."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(X.shape[0])]]                        # k-means++ seeding
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C.append(X[rng.choice(X.shape[0], p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(n_iter):
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        newC = np.array([X[lab == j].mean(0) for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff the two labellings induce the same partition."""
    ca = {}
    for x, y in zip(a, b, strict=True):
        if ca.setdefault(int(x), int(y)) != int(y):
            return False
    return len(set(ca.values())) == len(set(map(int, b)))


def main() -> None:
    rng = np.random.default_rng(0)

    # --- crp: the EPPF is a probability distribution over partitions -------
    # Summing over every set partition of five items is the cleanest test that
    # the EPPF is normalised; Bell(5) = 52 terms, and they must total one.
    dp = DirichletProcess("crp", alpha=1.2)
    parts = _partitions(5)
    total = sum(dp.eppf(np.bincount(p)) for p in parts)
    assert abs(total - 1.0) < 1e-12, total

    # and it is normalised at six items too, where Bell(6) = 203
    total6 = sum(dp.eppf(np.bincount(p)) for p in _partitions(6))
    assert abs(total6 - 1.0) < 1e-12, total6

    # E[K_n] has a closed form, so the sampler can be checked against it
    n_crp, reps = 12, 20_000
    seat = np.random.default_rng(1)
    draws = np.array([dp.sample_partition(n_crp, rng=seat).max() + 1 for _ in range(reps)])
    exact = dp.expected_clusters(n_crp)
    assert abs(draws.mean() - exact) < 0.1, (draws.mean(), exact)
    print(
        f"dirichlet_process: crp — {len(parts)} partitions of five items sum to "
        f"{total:.12f} and 203 partitions of six to {total6:.12f}; E[K_{n_crp}] = "
        f"{exact:.3f} closed form against {draws.mean():.3f} over {reps} draws"
    )

    # --- stick_breaking: the weights are a size-biased permutation ----------
    sb = DirichletProcess("stick_breaking", alpha=1.5)
    K = 8
    stick = np.random.default_rng(1)
    ws, resid = [], []
    for _ in range(20_000):
        w, r = sb.gem(K, rng=stick)
        ws.append(w)
        resid.append(r)
    ws, resid = np.array(ws), np.array(resid)
    # E[w_k] = alpha^(k-1) / (1+alpha)^k exactly, since the V_k are independent
    exact_w = np.array([sb.alpha ** k / (1.0 + sb.alpha) ** (k + 1) for k in range(K)])
    assert np.allclose(ws.mean(axis=0), exact_w, atol=0.004)
    # every draw's K sticks plus the residual is the whole distribution
    assert np.max(np.abs(ws.sum(axis=1) + resid - 1.0)) < 1e-12
    assert abs(float(resid.mean()) - (sb.alpha / (1.0 + sb.alpha)) ** K) < 0.002
    print(
        f"dirichlet_process: stick_breaking — E[w_1..w_{K}] decays as "
        f"α^(k−1)/(1+α)^k: {np.round(exact_w, 4)}; the K sticks carry "
        f"{1.0 - resid.mean():.4f} of the mass and leave {resid.mean():.4f} "
        f"behind, against the exact (α/(1+α))^{K} = "
        f"{(sb.alpha / (1.0 + sb.alpha)) ** K:.4f}"
    )

    # --- dpmm: the infinite mixture, on three separated clusters ------------
    centres = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    truth = np.repeat(np.arange(3), 40)
    X = centres[truth] + 0.4 * rng.normal(size=(120, 2))
    hyper = {"kappa0": 0.5, "a0": 5.0, "b0": 1.0}
    mm = DirichletProcess("dpmm", alpha=1.0, n_iter=40, seed=2, **hyper).fit(X)
    assert mm.K_ == 3, mm.K_
    assert _same_partition(mm.labels_, truth)
    # the predictive is a Student-t, so a numerical integral over the
    # inverse-gamma posterior has to reproduce it
    x1 = X[mm.labels_ == 0, 0]
    mu, kappa, a, b = _niw_update(x1, mm.mu0, mm.kappa0, mm.a0, mm.b0)
    # σ² ~ InvGamma(a, b) means τ = b/σ² ~ Gamma(a, rate 1), and the
    # Jacobian kills the b: the weight below is τ^(a−1)e^(−τ)/Γ(a)
    grid = np.linspace(1e-8, 80.0, 400_000)
    var = (1.0 + 1.0 / kappa) * b / grid
    xq = float(x1.mean()) + 0.5
    dens = np.exp(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (xq - mu) ** 2 / var)
    dens *= np.exp((a - 1.0) * np.log(grid) - grid - math.lgamma(a))
    integral = float(np.trapezoid(dens, grid))
    analytic = math.exp(_predictive_logpdf(xq, (mu, kappa, a, b)))
    assert abs(integral - analytic) < 1e-10, (integral, analytic)
    print(
        f"dirichlet_process: dpmm — collapsed Gibbs finds {mm.K_} clusters "
        f"(sizes {sorted(mm.counts_.tolist())}) in {len(mm.history_)} sweeps; "
        f"its Student-t predictive at x={xq:.2f}, {analytic:.6f}, is the exact "
        f"inverse-gamma integral {integral:.6f}"
    )

    # --- truncated_dp: K sticks, and the truncation is exactly residual mass -
    td = DirichletProcess("truncated_dp", alpha=1.0, K=8, n_iter=40, seed=2,
                          **hyper).fit(X)
    assert td.K_ == 3, td.K_
    # the fit uses 3 of the 8 sticks it was offered; the mass still sitting on
    # the unoccupied ones is the truncation error, not a rounding artefact
    assert abs(td.weights_.sum() + td.residual_ - 1.0) < 1e-12
    assert 0.0 < td.residual_ < 0.02
    w_full, _ = td.posterior_weights(np.concatenate([td.counts_, np.zeros(200, int)]))
    l1 = float(np.abs(np.concatenate([td.weights_, np.zeros(200)]) - w_full).sum())
    assert np.allclose(w_full[:td.K_], td.weights_)
    assert abs(l1 - td.residual_) < 1e-12, (l1, td.residual_)
    gap = float(np.max(np.abs(td.predict_log_density(X) - mm.predict_log_density(X))))
    assert gap < 0.06, gap
    print(
        f"dirichlet_process: truncated_dp — {td.K_} of the {td.K} sticks it was "
        f"offered carry {td.weights_.sum():.4f} of the mass; the residual "
        f"{td.residual_:.4f} is the truncation error, the L1 gap between the two "
        f"measures, and its log predictive is within {gap:.4f} of the infinite fit"
    )

    # --- dp_means: k-means that chooses its own k ---------------------------
    dm = DirichletProcess("dp_means", lam=9.0, n_iter=50, seed=3).fit(X)
    assert dm.K_ == 3, dm.K_
    assert _same_partition(dm.labels_, truth)
    km = _kmeans(X, 3)
    assert _same_partition(dm.labels_, km)
    assert all(a >= b - 1e-9 for a, b in zip(dm.history_, dm.history_[1:], strict=False))
    sse = float(((X - dm.centroids_[dm.labels_]) ** 2).sum())
    assert abs(dm.objective_ - (sse + dm.lam * dm.K_)) < 1e-9
    one = DirichletProcess("dp_means", lam=1e6, n_iter=20, seed=3).fit(X)
    many = DirichletProcess("dp_means", lam=1e-6, n_iter=20, seed=3).fit(X)
    assert one.K_ == 1 and X.shape[0] == many.K_, (one.K_, many.K_)
    print(
        f"dirichlet_process: dp_means — λ = 9 recovers the same three clusters as "
        f"Lloyd, with objective {dm.objective_:.2f} = SSE {sse:.2f} + λ·k; "
        f"λ = 1e6 gives k = {one.K_} and λ = 1e−6 gives k = {many.K_}"
    )

    print("dirichlet_process demo passed")


if __name__ == "__main__":
    main()
