"""Bayesian inference — the posterior, exactly, when the algebra allows it.

Chapter:   05-probabilistic
Companion: bayesian_inference.md
Tier 1 · 7 variants: mle, map, conjugate_normal, beta_binomial,
dirichlet_multinomial, credible_intervals, posterior_predictive

    prior × likelihood ∝ posterior

For three families the product closes: a Gaussian prior with a Gaussian
likelihood is Gaussian, a Beta with a Bernoulli is Beta, a Dirichlet with a
categorical is Dirichlet. Conjugacy means the posterior is a formula in the
sufficient statistics — no sampler, no optimiser, and the *entire* posterior,
not a point.

    mle                     the likelihood alone: x̄, k/n
    map                     prior × likelihood, maximised
    conjugate_normal        N(μ₀, σ₀²) + n draws → N(μₙ, σₙ²)
    beta_binomial           Beta(a, b) + Bernoulli → Beta(a+k, b+n−k)
    dirichlet_multinomial   Dirichlet(α) + counts → Dirichlet(α + c)
    credible_intervals      where the posterior mass actually is
    posterior_predictive    the distribution of the *next* draw

The point estimates people quote are summaries of these posteriors; this file
keeps the summary as a method, so the difference between the MLE and the
posterior is one call apart rather than one paper apart.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

VARIANTS = frozenset(
    ["mle", "map", "conjugate_normal", "beta_binomial",
     "dirichlet_multinomial", "credible_intervals", "posterior_predictive"]
)

_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Special functions, without scipy
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta `I_x(a, b)`, the CDF of `Beta(a, b)`."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """`F⁻¹(p)` by bisection — the CDF is monotone, so this always converges."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - lbeta)


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class BayesianInference:
    """Conjugate posteriors, their point estimates and their predictives.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    family : str or None
        `"normal"`, `"beta"` or `"dirichlet"`. Inferred from the variant when
        omitted; pass it explicitly to use e.g. `credible_intervals` on a Beta
        posterior.
    prior : tuple, array or None
        `(μ₀, σ₀)` for the normal family, `(a, b)` for Beta, a vector for
        Dirichlet. Defaults: a very flat normal prior, `Beta(1, 1)`,
        `Dirichlet(ones)`.
    sigma : float
        Known observation standard deviation for the normal family. Conjugacy
        needs the variance known; an unknown variance needs a normal-inverse-
        gamma prior and a Student-t marginal, which is `hierarchical_bayes.py`.
    """

    def __init__(self, variant: str = "conjugate_normal", *, family: str | None = None,
                 prior=None, sigma: float = 1.0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.family = family or {
            "beta_binomial": "beta",
            "dirichlet_multinomial": "dirichlet",
        }.get(variant, "normal")
        self.prior = prior
        self.sigma = float(sigma)

    # ------------------------------------------------------------------ #
    def fit(self, data) -> BayesianInference:
        d = np.asarray(data, dtype=float)
        if self.family == "normal":
            self.n_ = d.size
            self.mean_ = float(d.mean()) if d.size else 0.0
            self.ss_ = float(((d - self.mean_) ** 2).sum()) if d.size else 0.0
            if self.prior is None:
                mu0, sigma0 = 0.0, 1e6
            else:
                mu0, sigma0 = self.prior
            self.prior_mean_, self.prior_sd_ = float(mu0), float(sigma0)
            prec = 1.0 / sigma0 ** 2 + self.n_ / self.sigma ** 2
            self.post_var_ = 1.0 / prec
            self.post_mean_ = self.post_var_ * (
                mu0 / sigma0 ** 2 + self.n_ * self.mean_ / self.sigma ** 2)
            self.post_sd_ = math.sqrt(self.post_var_)
        elif self.family == "beta":
            if not np.isin(d, (0.0, 1.0)).all():
                raise ValueError("the beta family takes 0/1 data")
            self.n_ = int(d.size)
            self.k_ = int(d.sum())
            a, b = self.prior if self.prior is not None else (1.0, 1.0)
            self.prior_a_, self.prior_b_ = float(a), float(b)
            self.post_a_ = a + self.k_
            self.post_b_ = b + self.n_ - self.k_
        elif self.family == "dirichlet":
            self.counts_ = d
            self.n_ = int(d.sum())
            alpha = (np.asarray(self.prior, dtype=float) if self.prior is not None
                     else np.ones(d.size))
            self.prior_alpha_ = alpha
            self.post_alpha_ = alpha + d
        else:
            raise ValueError(f"unknown family {self.family!r}")
        return self

    # ------------------------------------------------------------------ #
    def estimate(self):
        """The point this variant is about: MLE, MAP or posterior mean.

        A scalar for the normal and beta families; the probability vector for
        the Dirichlet, where no single number is the estimate.
        """
        if self.variant == "mle":
            if self.family == "normal":
                return self.mean_
            if self.family == "beta":
                return self.k_ / self.n_
            return self.counts_ / self.counts_.sum()
        if self.variant == "map":
            if self.family == "normal":
                return self.post_mean_          # symmetric: mode == mean
            if self.family == "beta":
                a, b = self.post_a_, self.post_b_
                if a <= 1.0:
                    return 0.0
                if b <= 1.0:
                    return 1.0
                return (a - 1.0) / (a + b - 2.0)
            alpha = self.post_alpha_
            if (alpha <= 1.0).any():
                # the mode sits on a face of the simplex; there is no interior
                # maximum, so report the posterior mean rather than a boundary
                return alpha / alpha.sum()
            return (alpha - 1.0) / (alpha.sum() - alpha.size)
        if self.family == "normal":
            return self.post_mean_
        if self.family == "beta":
            return self.post_a_ / (self.post_a_ + self.post_b_)
        return self.post_alpha_ / self.post_alpha_.sum()

    def posterior(self):
        """The posterior's parameters, as a tuple or vector."""
        if self.family == "normal":
            return self.post_mean_, self.post_var_
        if self.family == "beta":
            return self.post_a_, self.post_b_
        return self.post_alpha_

    # ------------------------------------------------------------------ #
    def credible_interval(self, level: float = 0.95, hpd: bool = False):
        """Equal-tailed by default; `hpd` gives the shortest interval."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        if self.family == "normal":
            z = _NORMAL.inv_cdf(1.0 - tail)
            lo, hi = self.post_mean_ - z * self.post_sd_, self.post_mean_ + z * self.post_sd_
            return (lo, hi)                     # symmetric: HPD is the same
        if self.family == "dirichlet":
            i = int(np.argmax(self.post_alpha_))
            sub = BayesianInference("credible_intervals", family="beta",
                                    prior=(self.post_alpha_[i],
                                           self.post_alpha_.sum() - self.post_alpha_[i]))
            sub.fit(np.empty(0))
            return sub.credible_interval(level, hpd)
        a, b = self.post_a_, self.post_b_
        if not hpd:
            return (beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b))
        # shortest interval: move `lo` until the two densities match
        def gap(lo: float) -> float:
            hi = beta_ppf(betainc(a, b, lo) + level, a, b)
            return beta_pdf(hi, a, b) - beta_pdf(lo, a, b)

        left, right = 1e-12, beta_ppf(1.0 - level, a, b)
        for _ in range(200):
            mid = 0.5 * (left + right)
            if gap(mid) > 0.0:          # gap decreases in `lo`: root is right
                left = mid
            else:
                right = mid
            if right - left < 1e-12:
                break
        lo = 0.5 * (left + right)
        return (lo, beta_ppf(betainc(a, b, lo) + level, a, b))

    # ------------------------------------------------------------------ #
    def predictive_mean_var(self, n: int = 1):
        """Mean and variance of the *next* `n` observations, not of the mean."""
        if self.family == "normal":
            return self.post_mean_, self.sigma ** 2 + self.post_var_
        if self.family == "beta":
            a, b = self.post_a_, self.post_b_
            tot = a + b
            mean = n * a / tot
            # beta-binomial: the binomial variance inflated by the Beta spread
            var = n * a * b * (tot + n) / (tot ** 2 * (tot + 1.0))
            return mean, var
        p = self.post_alpha_ / self.post_alpha_.sum()
        return p, p * (1.0 - p) / (self.post_alpha_.sum() + 1.0)

    def predictive_pmf(self, x, n: int | None = None):
        """Density (normal), pmf over `0..n` (beta) or pmf of a count vector."""
        if self.family == "normal":
            s = math.sqrt(self.sigma ** 2 + self.post_var_)
            return np.exp(-0.5 * ((np.asarray(x) - self.post_mean_) / s) ** 2) \
                / (s * math.sqrt(2.0 * math.pi))
        if self.family == "beta":
            n = self.n_ if n is None else int(n)
            a, b = self.post_a_, self.post_b_
            k = np.arange(n + 1)
            logpmf = (np.array([math.lgamma(n + 1) - math.lgamma(i + 1)
                                - math.lgamma(n - i + 1) for i in k])
                      + np.array([log_beta(a + i, b + n - i) for i in k])
                      - log_beta(a, b))
            return np.exp(logpmf)
        c = np.asarray(x, dtype=float)
        al = self.prior_alpha_
        logp = (math.lgamma(c.sum() + 1) - sum(math.lgamma(v + 1) for v in c)
                + math.lgamma(al.sum()) - math.lgamma(al.sum() + c.sum())
                + sum(math.lgamma(ai + ci) - math.lgamma(ai)
                      for ai, ci in zip(al, c, strict=True)))
        return float(math.exp(logp))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # --- mle and map differ by exactly the prior's precision ---
    data = np.array([1.4, 2.2, 1.9, 2.4, 1.6, 2.1, 2.0, 1.8, 2.6])
    mle = BayesianInference("mle").fit(data)
    assert np.isclose(mle.estimate(), data.mean())
    # a flat prior reproduces the MLE exactly
    flat = BayesianInference("map", prior=(0.0, 1e6), sigma=1.0).fit(data)
    assert np.isclose(flat.estimate(), data.mean(), atol=1e-6)
    tight = BayesianInference("map", prior=(0.0, 1.0), sigma=1.0).fit(data)
    w = (9 / 1.0) / (1 / 1.0 + 9 / 1.0)          # data weight = n/σ² over total prec
    assert np.isclose(tight.estimate(), w * data.mean(), atol=1e-12)
    assert 0.0 < tight.estimate() < data.mean() < 3.0
    print(f"mle/map: x̄ = {data.mean():.4f}; with N(0, 1) the MAP is the "
          f"precision-weighted {tight.estimate():.4f}, pulled toward the prior")

    # --- conjugate normal: the posterior is exactly Gaussian ---
    prior = (0.0, 2.0)
    cn = BayesianInference("conjugate_normal", prior=prior, sigma=1.5).fit(data)
    n = data.size
    prec = 1 / prior[1] ** 2 + n / 1.5 ** 2
    mu_n = (prior[0] / prior[1] ** 2 + n * data.mean() / 1.5 ** 2) / prec
    assert np.isclose(cn.posterior()[0], mu_n, atol=1e-12)
    assert np.isclose(cn.posterior()[1], 1 / prec, atol=1e-12)
    assert cn.post_sd_ < prior[1] and cn.post_sd_ < 1.5 / math.sqrt(n)
    print(f"conjugate_normal: posterior N({mu_n:.4f}, {cn.post_sd_:.4f}²) is "
          f"sharper than both the prior (σ=2) and the likelihood (σ/√n="
          f"{1.5 / math.sqrt(n):.4f})")

    # --- beta-binomial: the prior is a pseudo-count, not a penalty ---
    yy = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=float)   # 8 of 10
    bb = BayesianInference("beta_binomial", prior=(2.0, 2.0)).fit(yy)
    assert bb.posterior() == (10.0, 4.0)                  # (2+8, 2+2)
    assert np.isclose(bb.estimate(), 10 / 14)
    pmf = bb.predictive_pmf(None, n=10)
    assert np.isclose(pmf.sum(), 1.0, atol=1e-12)
    k = 7
    # the predictive is the beta-binomial at the *posterior*, not the prior
    exact = (math.comb(10, k) * math.exp(log_beta(10 + k, 4 + 10 - k)
                                        - log_beta(10, 4)))
    assert np.isclose(pmf[k], exact, atol=1e-12)
    # over-dispersed relative to a binomial with the same mean
    _, var_bb = bb.predictive_mean_var(n=10)
    p_hat = bb.estimate()
    assert var_bb > 10 * p_hat * (1 - p_hat)
    print(f"beta_binomial: Beta(2,2) + 8/10 → Beta(10,4); P(next = 1) = "
          f"{bb.estimate():.4f}, next-10 variance {var_bb:.3f} > binomial "
          f"{10 * p_hat * (1 - p_hat):.3f}")

    # --- beta cumulative distribution, checked against its finite sum ---
    for a, b in ((1, 3), (2, 5), (3, 2)):
        for x in (0.1, 0.4, 0.85):
            finite = sum(math.comb(a + b - 1, j) * x ** j * (1 - x) ** (a + b - 1 - j)
                         for j in range(a, a + b))
            assert np.isclose(betainc(a, b, x), finite, atol=1e-12)
    assert np.isclose(betainc(1.0, 3.0, 0.5), 1 - 0.5 ** 3, atol=1e-12)
    print("betainc: the continued fraction matches the exact binomial sum "
          "Σⱼ C(a+b−1, j)xʲ(1−x)^(a+b−1−j) to 1e-12")

    # --- dirichlet-multinomial: one vector, updated by addition ---
    counts = np.array([3.0, 0.0, 2.0])
    dm = BayesianInference("dirichlet_multinomial", prior=np.ones(3)).fit(counts)
    assert np.allclose(dm.posterior(), [4.0, 1.0, 3.0])
    assert np.allclose(dm.post_alpha_ / dm.post_alpha_.sum(), [0.5, 0.125, 0.375])
    total = sum(dm.predictive_pmf(np.array([i, j, 5 - i - j]))
                for i in range(6) for j in range(6 - i))
    assert np.isclose(total, 1.0, atol=1e-12)
    print(f"dirichlet_multinomial: α = (1,1,1) + (3,0,2) → (4,1,3); the pmf "
          f"over all 21 count vectors sums to {total:.6f}")

    # --- credible intervals: width, shape, and the HPD shortcut ---
    ci = BayesianInference("credible_intervals", prior=(0.0, 2.0), sigma=1.5).fit(data)
    lo, hi = ci.credible_interval(0.95)
    z = _NORMAL.inv_cdf(0.975)
    assert np.isclose(hi - lo, 2 * z * ci.post_sd_, atol=1e-12)
    assert np.allclose(ci.credible_interval(0.95, hpd=True), (lo, hi))
    beta_ci = BayesianInference("credible_intervals", family="beta",
                                prior=(2.0, 2.0)).fit(yy)
    blo, bhi = beta_ci.credible_interval(0.95)
    hlo, hhi = beta_ci.credible_interval(0.95, hpd=True)
    assert blo < beta_ci.estimate() < bhi
    assert (hhi - hlo) < (bhi - blo)            # HPD is never wider
    assert np.allclose(beta_pdf(hlo, 10, 4), beta_pdf(hhi, 10, 4), atol=1e-9)
    print(f"credible_intervals: normal 95% = ±{z:.3f}σ (HPD identical); "
          f"Beta(10,4) equal-tailed [{blo:.3f}, {bhi:.3f}], HPD "
          f"[{hlo:.3f}, {hhi:.3f}]")

    # --- coverage: with a flat prior the credible interval is a confidence one ---
    hits, reps = 0, 500
    truth, sd = 0.7, 1.0
    for _ in range(reps):
        sample = rng.normal(truth, sd, size=8)
        m = BayesianInference("credible_intervals", prior=(0.0, 1e6),
                              sigma=sd).fit(sample)
        lo_, hi_ = m.credible_interval(0.95)
        hits += lo_ <= truth <= hi_
    rate = hits / reps
    assert 0.91 < rate < 0.99, rate
    print(f"credible_intervals: the 95% interval contains the true mean in "
          f"{rate:.1%} of 500 replications — flat prior, so it is also a CI")

    # --- posterior predictive: the next draw is noisier than the mean ---
    pp = BayesianInference("posterior_predictive", prior=(0.0, 2.0), sigma=1.5).fit(data)
    mean_p, var_p = pp.predictive_mean_var()
    assert np.isclose(mean_p, pp.post_mean_)
    assert np.isclose(var_p, 1.5 ** 2 + pp.post_var_, atol=1e-12)
    assert var_p > pp.post_var_ * 5                    # noise dominates
    scale = math.sqrt(var_p)
    xs = np.linspace(mean_p - 6.0 * scale, mean_p + 6.0 * scale, 12001)
    dens = pp.predictive_pmf(xs)
    assert np.isclose(dens.sum() * (xs[1] - xs[0]), 1.0, atol=1e-4)
    assert np.isclose(scale, math.sqrt(var_p))
    print(f"posterior_predictive: N({mean_p:.3f}, {var_p:.3f}) — the spread of "
          f"the next observation, not of the parameter")

    # --- and the discrete predictive is the beta-binomial, not the binomial ---
    bp = BayesianInference("posterior_predictive", family="beta",
                           prior=(2.0, 2.0)).fit(yy)
    m_p, v_p = bp.predictive_mean_var(n=3)
    assert np.isclose(m_p, 3 * 10 / 14, atol=1e-12)
    assert np.isclose(bp.predictive_pmf(None, n=3).sum(), 1.0, atol=1e-12)
    print(f"posterior_predictive: 3 future Bernoulli draws → mean {m_p:.4f}, "
          f"variance {v_p:.4f} > 3p(1−p) = {3 * (10 / 14) * (4 / 14):.4f}")

    print("bayesian_inference demo passed")


if __name__ == "__main__":
    main()
