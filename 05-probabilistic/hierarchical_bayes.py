"""Hierarchical Bayes — partial pooling, shrinkage and empirical Bayes.

Chapter:   05-probabilistic
Companion: hierarchical_bayes.md
Tier 2 · 5 variants: partial_pooling, full_pooling, no_pooling, empirical_bayes,
noncentered

    y_ij = μ + a_j + ε_ij,   a_j ~ N(0, τ²),   ε_ij ~ N(0, σ²)
    E[a_j | y] = λ_j μ + (1 − λ_j) ȳ_j,   λ_j = (σ²/n_j) / (τ² + σ²/n_j)

Three ways to answer "what is group j's mean". Ignore the groups and pool
everything (`full_pooling`). Ignore the other groups and use this one's data
(`no_pooling`). Or put a prior on the group effects, which is what makes the
estimate a weighted average of the two — `partial_pooling`, where the weight is
the ratio of within-group noise to between-group spread.

That shrinkage factor is the whole atom. It is not a heuristic: it is the exact
posterior mean under the model, and it is why the estimator beats both extremes
on squared error. `empirical_bayes` fills in τ² and σ² from the data by marginal
maximum likelihood; `noncentered` is the parameterisation that lets a sampler
explore the same posterior when τ is small and the funnel is narrow.
"""

from __future__ import annotations

import math

import numpy as np
from mcmc import MCMC, ess  # sibling module: same folder, same sys.path

VARIANTS = frozenset(
    ["partial_pooling", "full_pooling", "no_pooling", "empirical_bayes", "noncentered"]
)

LOG2PI = math.log(2.0 * math.pi)


def _exp(x: float) -> float:
    """`exp` that saturates instead of raising, so a diverging leapfrog returns
    a very negative log density rather than a Python exception."""
    return math.exp(min(x, 700.0))


class HierarchicalBayes:
    """Fit a one-way random-effects model, one variant at a time.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    tau2, sigma2 : float
        Between- and within-group variance. They are the *known* quantities for
        `partial_pooling`; `empirical_bayes` and `noncentered` estimate them and
        report the fitted values in `tau2_` / `sigma2_`.
    n_iter : int
        EM iterations for `empirical_bayes`.
    tol : float
        EM convergence threshold on the marginal log-likelihood.

    Notes
    -----
    `fit(y, groups)` takes a flat array and a label per observation. Fitted
    attributes are `mu_` (the fixed effect), `group_mean_` `(J,)`, `group_var_`
    `(J,)` (the posterior variance of each group's mean), `lambda_` `(J,)` (the
    shrinkage weight on the grand mean), `tau2_`, `sigma2_`, `loglik_` (the
    marginal log-likelihood of `y`), and for `noncentered` the posterior draws
    in `samples_` plus the ESS of each parameter in `ess_`.
    """

    def __init__(self, variant: str = "partial_pooling", *, tau2: float = 1.0,
                 sigma2: float = 1.0, n_iter: int = 500, tol: float = 1e-10,
                 n_samples: int = 2000, n_warmup: int = 500, seed: int = 0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if tau2 <= 0.0 or sigma2 <= 0.0:
            raise ValueError("variances must be positive")
        self.variant = variant
        self.tau2, self.sigma2 = float(tau2), float(sigma2)
        self.n_iter, self.tol = int(n_iter), float(tol)
        self.n_samples, self.n_warmup, self.seed = int(n_samples), int(n_warmup), int(seed)

    # ------------------------------------------------------------------ #
    def fit(self, y: np.ndarray, groups: np.ndarray) -> HierarchicalBayes:
        """Estimate `μ`, the group means and their uncertainties.

        `y` is `(N,)`, `groups` is `(N,)` of hashable labels. Groups with a
        single observation are allowed everywhere except `no_pooling`, where
        their variance is undefined.
        """
        y = np.asarray(y, dtype=float).ravel()
        groups = np.asarray(groups).ravel()
        if y.size != groups.size:
            raise ValueError("y and groups must have the same length")
        labels = np.unique(groups)
        ys = [y[groups == g] for g in labels]
        n = np.array([g.size for g in ys], dtype=float)
        ybar = np.array([g.mean() for g in ys])
        ss = np.array([np.sum((g - g.mean()) ** 2) for g in ys])
        self.labels_, self.n_ = labels, n
        self.grand_ = float(y.mean())

        if self.variant == "no_pooling":
            if np.any(n < 2):
                raise ValueError(
                    "no_pooling needs at least two observations per group; use "
                    "partial_pooling, which borrows strength instead of dividing by n−1"
                )
            self.tau2_, self.sigma2_ = 0.0, self.sigma2
            self.mu_ = self.grand_
            self.group_mean_ = ybar
            self.group_var_ = ss / (n - 1.0) / n
            self.lambda_ = np.zeros_like(n)
        elif self.variant == "full_pooling":
            self.tau2_, self.sigma2_ = 0.0, self.sigma2
            self.mu_ = self.grand_
            self.group_mean_ = np.full_like(ybar, self.grand_)
            self.group_var_ = np.full_like(ybar, self.sigma2 / n.sum())
            self.lambda_ = np.ones_like(n)
        elif self.variant == "empirical_bayes":
            self._em(ybar, n, ss)
            self._shrink(ybar, n, ss, mu=self.mu_)
        elif self.variant == "noncentered":
            self._hmc(ys, centered=False)
        else:
            self.tau2_, self.sigma2_ = self.tau2, self.sigma2
            self._shrink(ybar, n, ss)
        return self

    # ------------------------------------------------------------------ #
    def _shrink(self, ybar: np.ndarray, n: np.ndarray, ss: np.ndarray, *,
                mu: float | None = None) -> None:
        """The posterior mean and variance of each `a_j`, given `(τ², σ²)`.

        `mu` defaults to the grand mean, which is the fixed effect the model
        implies; `empirical_bayes` passes the μ that EM converged to instead,
        which is not quite the same number when the groups are unbalanced.
        """
        self.mu_ = self.grand_ if mu is None else float(mu)
        # τ² → 0 is not an error: it is the full-pooling limit, λ → 1. Clamp so
        # the arithmetic takes that limit instead of dividing by zero.
        tau2 = max(self.tau2_, 1e-12)
        prec = n / self.sigma2_ + 1.0 / tau2
        self.group_mean_ = (n * ybar / self.sigma2_ + self.mu_ / tau2) / prec
        self.group_var_ = 1.0 / prec
        self.lambda_ = (self.sigma2_ / n) / (tau2 + self.sigma2_ / n)
        self.loglik_ = _marginal_loglik(ybar, n, self.mu_, self.tau2_, self.sigma2_, ss)

    def _em(self, ybar: np.ndarray, n: np.ndarray, ss: np.ndarray) -> None:
        """Empirical Bayes by EM on the marginal likelihood.

        The E-step is the shrinkage formula — the "missing" group effects are
        replaced by their posterior means and variances. The M-step is their
        second moment: `τ² = mean(a² + var(a))`, which is why the variance term
        has to be inside the update or the estimate collapses.
        """
        tau2, sigma2 = self.tau2, self.sigma2
        mu = float(ybar.mean())
        N = float(np.sum(n))
        self.loglik_history_ = []
        n_done = 0
        for _ in range(self.n_iter):
            n_done += 1
            # E-step: the posterior of each a_j given the current variances
            prec = n / sigma2 + 1.0 / tau2
            var = 1.0 / prec
            theta = (n * (ybar - mu) / sigma2) / prec
            # M-step: maximise the expected complete-data log-likelihood, which
            # keeps the squared residual (ȳ_j − μ − θ_j)² in the σ² update —
            # dropping it is the classic bug, and it biases σ² downward
            mu = float(np.sum(n * (ybar - theta)) / N)
            tau2 = float(np.mean((theta - mu) ** 2 + var))
            sigma2 = float(np.sum(ss + n * ((ybar - mu - theta) ** 2 + var)) / N)
            ll = _marginal_loglik(ybar, n, mu, tau2, sigma2, ss)
            self.loglik_history_.append(ll)
            if len(self.loglik_history_) > 1 and ll - self.loglik_history_[-2] < self.tol:
                break
        self.mu_, self.tau2_, self.sigma2_ = mu, tau2, sigma2
        self.n_iter_ = n_done
        self.loglik_ = float(self.loglik_history_[-1])

    # ------------------------------------------------------------------ #
    def _hmc(self, ys: list[np.ndarray], *, centered: bool) -> None:
        """Sample the posterior by HMC, centered or not.

        The centered parameterisation carries `a_j` and must learn `τ` from the
        spread of `a`; the non-centered one carries `z_j = a_j / τ`, which is
        `N(0, 1)` regardless of `τ`. Same posterior, different geometry — and
        when `τ` is small the centered one has a funnel that a sampler cannot
        traverse.
        """
        J = len(ys)
        n = np.array([g.size for g in ys], dtype=float)
        grand = float(np.concatenate(ys).mean())
        log_prob, grad = _model(ys, centered=centered)

        # Start at the stated (τ², σ²), with zero group effects.
        tau2_start = max(self.tau2, 1e-3)
        sigma2_start = max(self.sigma2, 1e-3)
        x0 = np.concatenate([[grand], np.zeros(J),
                             [0.5 * math.log(tau2_start), 0.5 * math.log(sigma2_start)]])
        sampler = MCMC(log_prob, variant="hmc", grad=grad, step_size=0.15,
                       n_leapfrog=16, n_warmup=self.n_warmup)
        chain = sampler.sample(self.n_samples, x0, rng=np.random.default_rng(self.seed))
        self.samples_ = chain
        self.ess_ = np.array([ess(chain[None, :, k]) for k in range(chain.shape[1])])
        eff = (chain[:, 1:1 + J] if centered
               else np.exp(np.clip(chain[:, -2:-1], -700.0, 700.0)) * chain[:, 1:1 + J])
        self.mu_ = float(chain[:, 0].mean())
        self.tau2_ = float(np.mean(np.exp(np.clip(2.0 * chain[:, -2], -700.0, 700.0))))
        self.sigma2_ = float(np.mean(np.exp(np.clip(2.0 * chain[:, -1], -700.0, 700.0))))
        self.group_mean_ = self.mu_ + eff.mean(axis=0)
        self.group_var_ = eff.var(axis=0)
        self.lambda_ = (self.sigma2_ / n) / (self.tau2_ + self.sigma2_ / n)
        self.loglik_ = float(np.mean([log_prob(row) for row in chain]))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _marginal_loglik(ybar: np.ndarray, n: np.ndarray, mu: float, tau2: float,
                     sigma2: float, ss: np.ndarray | None) -> float:
    """`log p(y)`: group means plus the within-group sums of squares.

    The decomposition is exact — the group means are independent of the
    residuals — so this is the likelihood the EM should be increasing, not a
    proxy for it.
    """
    var = tau2 + sigma2 / n
    out = float(np.sum(-0.5 * (LOG2PI + np.log(var)) - 0.5 * (ybar - mu) ** 2 / var))
    if ss is not None:
        out += float(np.sum(-0.5 * (n - 1.0) * (LOG2PI + math.log(sigma2))
                            - 0.5 * ss / sigma2))
    return out


def _model(ys: list[np.ndarray], *, centered: bool):
    """`(log_prob, grad)` for the random-effects model, in one parameterisation.

    The state is `x = [μ, a_1..a_J, log τ, log σ]`, where `a_j` is the effect
    itself when `centered` and the standardised `z_j = a_j/τ` when not. The two
    forms have the same target but very different geometry; the returned
    functions are what makes the comparison in the demo fair, since both are
    handed to the same HMC kernel with the same step size.

    `ȳ_j` and `ss_j` are sufficient statistics: the full likelihood factorises
    into a between-group part in `(ȳ_j, σ²/n_j)` and a within-group part in
    `ss_j`, which is why the sampler never needs the observations themselves.
    """
    J = len(ys)
    ybar = np.array([g.mean() for g in ys])
    ss = np.array([np.sum((g - g.mean()) ** 2) for g in ys])
    n = np.array([g.size for g in ys], dtype=float)
    N = float(n.sum())
    ss_total = float(np.sum(ss))

    def log_prob(x: np.ndarray) -> float:
        mu, a, lt, ls = x[0], x[1:1 + J], x[-2], x[-1]
        tau2, sigma2 = _exp(2.0 * lt), _exp(2.0 * ls)
        eff = a if centered else _exp(lt) * a
        resid = ybar - mu - eff
        # -N·log σ, not -N/2·log σ: the likelihood carries one log σ per
        # observation. The group means are not the whole data — `ss` is the
        # within-group sum of squares, and dropping it makes σ → 0 a free
        # lunch, leaving an improper posterior that HMC will happily run down.
        ll = (-0.5 * N * LOG2PI - N * ls
              - 0.5 * (np.sum(n * resid**2) + ss_total) / sigma2)
        ll += -0.5 * J * LOG2PI - 0.5 * np.sum(eff**2) / tau2
        if centered:
            ll -= J * lt
        # Weakly informative priors: N(0, 5²) on μ and N(0, 2²) on the two
        # log-scales. Tight scale priors would decide the variance components.
        return float(ll - 0.5 * (mu / 5.0) ** 2
                     - 0.5 * (lt / 2.0) ** 2 - 0.5 * (ls / 2.0) ** 2)

    def grad(x: np.ndarray) -> np.ndarray:
        mu, a, lt, ls = x[0], x[1:1 + J], x[-2], x[-1]
        tau2, sigma2 = _exp(2.0 * lt), _exp(2.0 * ls)
        tau = _exp(lt)
        eff = a if centered else tau * a
        resid = ybar - mu - eff
        g = np.empty_like(x)
        g[0] = -mu / 25.0 + np.sum(n * resid) / sigma2
        if centered:
            # `a` is the effect itself; the prior pulls it to 0 with precision
            # 1/τ², so a small τ makes this coordinate stiff — the funnel.
            g[1:1 + J] = n * resid / sigma2 - a / tau2
            g[-2] = -0.25 * lt - J + np.sum(a**2) / tau2
        else:
            # z = a/τ is N(0, 1), so its prior term has no τ in it at all: the
            # log τ from the change of variables cancels the -log τ in N(0, τ²)
            g[1:1 + J] = tau * n * resid / sigma2 - a
            g[-2] = -0.25 * lt + tau * np.sum(n * resid * a) / sigma2
        g[-1] = -0.25 * ls - N + (np.sum(n * resid**2) + ss_total) / sigma2
        return g

    return log_prob, grad


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _simulate(rng, J: int, n_j: int, mu: float, tau: float, sigma: float):
    """`J` groups of `n_j` observations from the model in the header."""
    a = rng.normal(0.0, tau, size=J)
    groups, y = [], []
    for j in range(J):
        groups.append(np.full(n_j, j))
        y.append(mu + a[j] + rng.normal(0.0, sigma, size=n_j))
    return np.concatenate(y), np.concatenate(groups), a


def main() -> None:
    rng = np.random.default_rng(0)

    # --- the shrinkage estimator, against both extremes --------------------
    J, n_j, mu, tau, sigma = 12, 4, 0.0, 0.6, 1.0
    y, groups, a_true = _simulate(rng, J, n_j, mu, tau, sigma)
    truth = mu + a_true

    no_pool = HierarchicalBayes("no_pooling").fit(y, groups)
    full_pool = HierarchicalBayes("full_pooling", sigma2=sigma**2).fit(y, groups)
    part = HierarchicalBayes("partial_pooling", tau2=tau**2, sigma2=sigma**2).fit(y, groups)
    eb = HierarchicalBayes("empirical_bayes").fit(y, groups)

    # (5) exactly: the estimate is the posterior mean, a convex combination
    lam = (sigma**2 / n_j) / (tau**2 + sigma**2 / n_j)
    assert np.allclose(part.lambda_, lam, atol=1e-12)
    assert np.allclose(
        part.group_mean_, part.lambda_ * part.mu_ + (1.0 - part.lambda_) * no_pool.group_mean_
    )
    # so it sits strictly between the two extremes, coordinate by coordinate
    assert np.all(part.group_mean_ > np.minimum(no_pool.group_mean_, part.mu_))
    assert np.all(part.group_mean_ < np.maximum(no_pool.group_mean_, part.mu_))
    # Borrowing strength has a precise meaning: the posterior variance of each
    # group mean is below the *sampling* variance σ²/n_j of that group's mean.
    # It can still exceed a group's own s_j²/n_j when s_j² came out low — that
    # is the trade, and it is why the pooled σ² wins on average (the MSE below).
    assert np.all(part.group_var_ < sigma**2 / n_j)
    n_low = int(np.sum(no_pool.group_var_ < part.group_var_))

    def mse(est):
        return float(np.mean((est - truth) ** 2))

    mse_no, mse_full, mse_part = mse(no_pool.group_mean_), mse(full_pool.group_mean_), mse(part.group_mean_)
    assert mse_part < mse_no and mse_part < mse_full
    print(
        f"hierarchical_bayes: partial_pooling — shrinkage λ = {lam:.3f} turns "
        f"{n_j} observations per group into MSE {mse_part:.4f}, against "
        f"{mse_no:.4f} unpooled and {mse_full:.4f} fully pooled; {n_low} of "
        f"{J} groups had s² below σ², where no pooling is the overconfident one"
    )

    # --- empirical Bayes recovers the variance components ------------------
    ybar = np.array([g.mean() for g in (y[groups == j] for j in range(J))])
    ss = np.array([np.sum((g - g.mean()) ** 2) for g in (y[groups == j] for j in range(J))])
    n = np.full(J, float(n_j))
    assert abs(eb.tau2_ - tau**2) < 0.5 * tau**2
    assert abs(eb.sigma2_ - sigma**2) < 0.25 * sigma**2
    # the reported evidence is the marginal likelihood at the reported
    # parameters — the same formula, evaluated the same way
    assert np.isclose(eb.loglik_,
                      _marginal_loglik(ybar, n, eb.mu_, eb.tau2_, eb.sigma2_, ss),
                      atol=1e-12)
    print(
        f"hierarchical_bayes: empirical_bayes — τ² {eb.tau2_:.3f} against a true "
        f"{tau**2:.3f}, σ² {eb.sigma2_:.3f} against {sigma**2:.3f}, "
        f"marginal log p(y) = {eb.loglik_:.2f}"
    )

    # --- EM is monotone in the likelihood it optimises ---------------------
    # Re-run the iterations by hand so the trajectory can be checked: each EM
    # step must increase the *marginal* likelihood, not just the surrogate.
    tau2, sigma2, hist = 1.0, 1.0, []
    mu_hat = float(ybar.mean())
    N = float(np.sum(n))
    for _ in range(60):
        prec = n / sigma2 + 1.0 / tau2
        var = 1.0 / prec
        theta = (n * (ybar - mu_hat) / sigma2) / prec
        mu_hat = float(np.sum(n * (ybar - theta)) / N)
        tau2 = float(np.mean((theta - mu_hat) ** 2 + var))
        sigma2 = float(np.sum(ss + n * ((ybar - mu_hat - theta) ** 2 + var)) / N)
        hist.append(_marginal_loglik(ybar, n, mu_hat, tau2, sigma2, ss))
    assert np.all(np.diff(hist) > -1e-9), np.min(np.diff(hist))
    assert np.isclose(hist[-1], eb.loglik_, atol=1e-6)
    print(
        f"hierarchical_bayes: empirical_bayes — marginal likelihood rises "
        f"{hist[0]:.3f} → {hist[-1]:.3f} over {len(hist)} EM steps, monotone"
    )

    # --- centered vs non-centered: the funnel ------------------------------
    # Ten groups of two observations, and a true τ small against σ/√n. The
    # centered sampler has to find a needle (a_j ≈ 0 to within τ) inside a
    # funnel whose width is τ; the non-centered one samples z_j ~ N(0, 1) and
    # is told the width afterwards.
    Jf, n_f, tau_f = 10, 2, 0.15
    yf, gf, _ = _simulate(np.random.default_rng(4), Jf, n_f, 0.0, tau_f, 1.0)
    ys_f = [yf[gf == j] for j in range(Jf)]
    ncf = HierarchicalBayes("noncentered", tau2=0.25, sigma2=1.0,
                            n_samples=3000, n_warmup=500, seed=1).fit(yf, gf)
    cen = HierarchicalBayes("noncentered", tau2=0.25, sigma2=1.0,
                            n_samples=3000, n_warmup=500, seed=1)
    cen._hmc(ys_f, centered=True)
    ess_c = float(cen.ess_[-2])
    ess_n = float(ncf.ess_[-2])
    assert ess_n > 2.0 * ess_c, (ess_n, ess_c)
    print(
        f"hierarchical_bayes: noncentered — ESS of log τ over 3000 draws is "
        f"{ess_n:.0f} non-centered against {ess_c:.0f} centered, at identical cost"
    )

    # --- both gradients are the derivative of the density they are given ----
    # A finite-difference check at one point, in each parameterisation. This is
    # what catches a half-power of τ: the log-density and its gradient are
    # written out by hand, so `lp` and `grad` can disagree while each looks
    # plausible.
    ybar_f = np.array([g.mean() for g in ys_f])
    probe = np.concatenate([[ybar_f.mean()], np.full(Jf, 0.1),
                            [math.log(0.5), math.log(0.8)]])
    for centered in (True, False):
        lp, gr = _model(ys_f, centered=centered)
        step = 1e-6
        numeric = np.array([
            (lp(probe + step * e) - lp(probe - step * e)) / (2.0 * step)
            for e in np.eye(probe.size)
        ])
        assert np.max(np.abs(gr(probe) - numeric)) < 1e-5, centered
    print(
        "hierarchical_bayes: noncentered — hand-written gradients of both "
        "parameterisations match central differences to 1e-5"
    )

    # --- the sampler, against the exact marginal posterior -----------------
    # Marginalise the group effects out analytically (they are Gaussian) and
    # integrate the resulting three-parameter posterior on a grid. The grid
    # never touches the sampler, so agreement is evidence and not a tautology.
    mus = np.linspace(-1.5, 1.5, 61)
    lts = np.linspace(-4.0, 1.5, 111)
    lss = np.linspace(-2.5, 1.5, 81)
    MU, LT, LS = np.meshgrid(mus, lts, lss, indexing="ij")
    tau2g, sig2g = np.exp(2.0 * LT), np.exp(2.0 * LS)
    n_fv = np.full(Jf, float(n_f))
    var = tau2g[None] + sig2g[None] / n_fv[:, None, None, None]
    ll = np.sum(-0.5 * np.log(2.0 * np.pi * var)
                - 0.5 * (ybar_f[:, None, None, None] - MU[None]) ** 2 / var, axis=0)
    ll += -0.5 * np.sum(n_fv - 1.0) * np.log(2.0 * np.pi * sig2g)
    ll += -0.5 * np.sum([
        np.sum((g - g.mean()) ** 2) for g in ys_f
    ]) / sig2g
    ll += -0.5 * (MU / 5.0) ** 2 - 0.5 * (LT / 2.0) ** 2 - 0.5 * (LS / 2.0) ** 2
    W = np.exp(ll - ll.max())
    W /= W.sum()
    grid_tau2 = float((W * tau2g).sum())
    grid_sigma2 = float((W * sig2g).sum())
    lam_g = (sig2g / n_f) / (tau2g + sig2g / n_f)
    theta_g = lam_g[None] * MU[None] + (1.0 - lam_g[None]) * ybar_f[:, None, None, None]
    grid_group = (W[None] * theta_g).sum(axis=(1, 2, 3))
    assert abs(ncf.tau2_ - grid_tau2) < 0.05, (ncf.tau2_, grid_tau2)
    assert abs(ncf.sigma2_ - grid_sigma2) < 0.1, (ncf.sigma2_, grid_sigma2)
    assert np.max(np.abs(ncf.group_mean_ - grid_group)) < 0.05
    print(
        f"hierarchical_bayes: noncentered — posterior τ² {ncf.tau2_:.3f} against "
        f"the exact {grid_tau2:.3f}, σ² {ncf.sigma2_:.3f} against {grid_sigma2:.3f}, "
        f"and group means within "
        f"{np.max(np.abs(ncf.group_mean_ - grid_group)):.3f}"
    )
    print(
        f"hierarchical_bayes: noncentered — the data put τ² near {grid_tau2:.2f}, "
        f"not the true {tau_f**2:.3f}: with 10 groups of 2, τ is barely identified, "
        f"which is exactly why the funnel matters"
    )

    print("hierarchical_bayes demo passed")


if __name__ == "__main__":
    main()
