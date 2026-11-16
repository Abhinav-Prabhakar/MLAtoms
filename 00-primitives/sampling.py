"""Sampling — 10 routines that turn a distribution into draws.

Chapter:   00-primitives
Companion: sampling.md
Tier 1 · 10 variants: uniform, weighted_alias, reservoir, stratified, systematic,
gumbel_top_k, top_p, negative_sampling, importance_sampling, rejection

Two families live here. **Exact draws** (uniform, alias, reservoir, gumbel,
rejection) produce samples from a specified distribution; **variance-reduced or
biased-in-a-controlled-way draws** (stratified, systematic, top_p, negative,
importance) are about the estimator built on top of the draw. Both matter: the
first decides whether your Monte Carlo is correct, the second whether it is
useful at 10⁴ samples instead of 10⁹.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Exact draws
# --------------------------------------------------------------------------- #
def uniform(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Inverse-CDF sampling: F⁻¹(u) with u ~ U(0,1). Trivial for U(0,1) itself,
    and the template for every other inverse-CDF sampler (exponential, Cauchy)."""
    rng = np.random.default_rng(rng)
    return rng.random(n)


def alias_table(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vose's alias method: build two arrays of length k in O(k) so that one draw
    costs O(1) regardless of k.

    prob[i] is the probability of keeping i; otherwise the draw is alias[i].
    The invariant is not sum(prob) == k (it generally is not): it is that the
    implied distribution, prob[i]/k plus the remainders routed to i, equals w.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    k = w.size
    p = w / w.sum() * k
    prob = np.zeros(k)
    alias = np.arange(k)
    small = [i for i in range(k) if p[i] < 1.0]
    large = [i for i in range(k) if p[i] >= 1.0]
    while small and large:
        s = small.pop()
        ell = large.pop()
        prob[s] = p[s]
        alias[s] = ell
        p[ell] -= 1.0 - p[s]
        (small if p[ell] < 1.0 else large).append(ell)
    for i in large + small:  # floating-point leftovers
        prob[i] = 1.0
    return prob, alias


def weighted_alias(
    n: int, weights: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw n indices with probability proportional to weights, O(1) per draw."""
    rng = np.random.default_rng(rng)
    prob, alias = alias_table(weights)
    k = prob.size
    i = rng.integers(0, k, size=n)
    take_alias = rng.random(n) >= prob[i]
    return np.where(take_alias, alias[i], i)


def reservoir(stream, k: int, rng: np.random.Generator | None = None) -> list:
    """Algorithm R: a uniform sample of size k from a stream of unknown length.

    After seeing i elements, every element is in the reservoir with probability
    exactly k/i — proved by induction, no knowledge of the length required.
    """
    rng = np.random.default_rng(rng)
    res: list = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                res[j] = x
    return res


def gumbel_top_k(
    logits: np.ndarray, k: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Gumbel-max trick: argmax(z + Gumbel) is an exact categorical draw, and the
    top-k of z + Gumbel is an exact draw *without replacement* (Plackett-Luce).

    This is how sampling from a categorical becomes a sort, which is what makes
    top-k/top-p sampling on a GPU a single fused kernel.
    """
    rng = np.random.default_rng(rng)
    z = np.asarray(logits, dtype=float)
    u = np.clip(rng.random(z.shape), 1e-300, 1.0 - 1e-16)
    g = -np.log(-np.log(u))
    return np.argsort(-(z + g), kind="stable")[:k]


def rejection(
    pdf, n: int, envelope=1.0, lo: float = 0.0, hi: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Rejection sampling from an arbitrary density on [lo, hi] with
    `pdf(x) <= envelope` on that interval (uniform proposal).

    Acceptance probability is exactly area/envelope, so the expected number of
    proposals per accepted draw is envelope/area — the cost of not having an
    inverse CDF.
    """
    rng = np.random.default_rng(rng)
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = max(64, 2 * (n - filled))
        x = rng.uniform(lo, hi, size=batch)
        u = rng.uniform(0.0, envelope, size=batch)
        acc = x[u < pdf(x)]
        take = min(acc.size, n - filled)
        out[filled : filled + take] = acc[:take]
        filled += take
    return out


# --------------------------------------------------------------------------- #
# Designs and estimators
# --------------------------------------------------------------------------- #
def stratified(n: int, population_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """One draw from each of n equal strata: covers the range by construction, so
    the sample mean has variance O(1/n²)-ish for smooth populations instead of
    O(1/n). Indices, not values — index into your array afterwards."""
    rng = np.random.default_rng(rng)
    bounds = np.linspace(0, population_size, n + 1)
    u = rng.random(n)
    return np.floor(bounds[:-1] + u * (bounds[1:] - bounds[:-1])).astype(int)


def systematic(n: int, population_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Stratified sampling with one shared random offset: exactly one draw per
    stratum, so the sample mean is unbiased and usually lower variance still."""
    rng = np.random.default_rng(rng)
    step = population_size / n
    start = rng.random() * step
    return np.floor(start + step * np.arange(n)).astype(int)


def top_p(probs: np.ndarray, p: float = 0.9, rng: np.random.Generator | None = None) -> int:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability reaches p, renormalise, draw one.

    Adapts the truncation to the shape of the distribution — a confident model
    keeps 1–2 tokens, an uncertain one keeps hundreds.
    """
    rng = np.random.default_rng(rng)
    q = np.asarray(probs, dtype=float)
    q = q / q.sum()
    order = np.argsort(-q)
    cum = np.cumsum(q[order])
    keep = int(np.searchsorted(cum, p) + 1)
    keep = min(keep, q.size)
    support = order[:keep]
    w = q[support]
    return int(support[rng.choice(keep, p=w / w.sum())])


def negative_sampling(
    counts: np.ndarray, positives: np.ndarray, n: int, exponent: float = 0.75,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """word2vec-style negatives: draw from counts^exponent, never from the
    positives. The 3/4 exponent flattens the head of a Zipf distribution, trading
    a little bias for far more usable rare-token negatives."""
    rng = np.random.default_rng(rng)
    w = np.asarray(counts, dtype=float) ** exponent
    w[np.asarray(positives, dtype=int)] = 0.0
    w = w / w.sum()
    return rng.choice(w.size, size=n, p=w)


def importance_sampling(
    samples: np.ndarray, p, q, f=None
) -> float:
    """E_p[f] ≈ mean(f(x) · p(x)/q(x)) for x ~ q.

    Unbiased for any q with support ⊇ p's. The variance is governed by how much
    the weights p/q vary, which is why a q that is too narrow is the classic
    silent failure: correct in expectation, useless in practice.
    """
    x = np.asarray(samples, dtype=float)
    weights = np.asarray(p(x), dtype=float) / np.asarray(q(x), dtype=float)
    fx = x if f is None else f(x)
    return float(np.mean(fx * weights))


# --------------------------------------------------------------------------- #
# Demo: every claim above is checked, not asserted by hand-waving.
# --------------------------------------------------------------------------- #
def main() -> None:
    rng = np.random.default_rng(0)

    # 1. Uniform: empirical CDF tracks the true one.
    u = uniform(200_000, rng)
    assert abs(u.mean() - 0.5) < 0.005
    assert abs(np.mean(u < 0.3) - 0.3) < 0.005
    print("uniform: empirical CDF within 0.005")

    # 2. Alias: the table reproduces the weights exactly, in O(1) per draw.
    w = np.array([0.5, 0.3, 0.15, 0.05])
    prob, alias = alias_table(w)
    # The table implies a distribution: prob[i]/k from i itself, plus the
    # remainder 1-prob[i] every column routed to i.
    implied = prob / w.size
    np.add.at(implied, alias, (1.0 - prob) / w.size)
    assert np.allclose(implied, w / w.sum()), (implied, w / w.sum())
    draws = weighted_alias(200_000, w, rng)
    emp = np.bincount(draws, minlength=4) / draws.size
    assert np.max(np.abs(emp - w)) < 0.005, emp
    print(f"alias: max frequency error {np.max(np.abs(emp - w)):.4f}")

    # 3. Reservoir: uniform over positions, independent of stream length.
    n, k, trials = 37, 5, 20_000
    hits = np.zeros(n)
    for _ in range(trials):
        for x in reservoir(range(n), k, rng):
            hits[x] += 1
    freq = hits / trials
    assert np.max(np.abs(freq - k / n)) < 0.01, freq
    print(f"reservoir: inclusion freq in [{freq.min():.3f}, {freq.max():.3f}], "
          f"target {k / n:.3f}")

    # 4. Stratified beats uniform on a smooth population.
    pop = np.linspace(0.0, 1.0, 200)
    reps, m = 3000, 20
    uni = [pop[rng.integers(0, 200, m)].mean() for _ in range(reps)]
    str_ = [pop[stratified(m, 200, rng)].mean() for _ in range(reps)]
    assert abs(np.mean(uni) - pop.mean()) < 0.01
    assert abs(np.mean(str_) - pop.mean()) < 0.01
    assert np.var(str_) < 0.5 * np.var(uni), (np.var(str_), np.var(uni))
    print(f"stratified: variance {np.var(str_):.2e} vs uniform {np.var(uni):.2e}")

    # 5. Systematic: exactly one draw per stratum, always.
    s = systematic(20, 200, rng)
    assert np.all((s >= 0) & (s < 200)) and np.unique(s // 10).size == 20
    assert np.all(s // 10 == np.arange(20))
    print("systematic: one draw per stratum")

    # 6. Gumbel-max: top-1 frequencies equal softmax; top-k equals Plackett-Luce.
    z = np.log(np.array([0.5, 0.3, 0.2]))
    first = np.bincount([gumbel_top_k(z, 1, rng)[0] for _ in range(50_000)], minlength=3)
    freq = first / first.sum()
    assert np.max(np.abs(freq - np.exp(z))) < 0.01, freq
    w2 = np.array([2.0, 1.0, 1.0])
    pairs = np.zeros((3, 3))
    for _ in range(40_000):
        top = gumbel_top_k(np.log(w2), 2, rng)
        pairs[tuple(sorted(top))] += 1
    pairs /= 40_000
    total = w2.sum()
    for i in range(3):
        for j in range(i + 1, 3):
            want = (w2[i] / total * w2[j] / (total - w2[i])
                    + w2[j] / total * w2[i] / (total - w2[j]))
            assert abs(pairs[i, j] - want) < 0.01, (i, j, pairs[i, j], want)
    print("gumbel: top-1 = softmax, top-2 = Plackett-Luce")

    # 7. Top-p: the support is the smallest prefix that reaches p.
    probs = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
    drawn = np.array([top_p(probs, 0.8, rng) for _ in range(20_000)])
    assert set(np.unique(drawn)) == {0, 1}, np.unique(drawn)  # 0.5 + 0.3 >= 0.8
    conditional = np.bincount(drawn, minlength=5)[:2] / drawn.size
    assert np.allclose(conditional, [0.5 / 0.8, 0.3 / 0.8], atol=0.01)
    many = np.array([top_p(probs, 0.95, rng) for _ in range(20_000)])
    assert set(np.unique(many)) == {0, 1, 2, 3}, np.unique(many)
    print("top_p: support adapts to p (p=0.8 -> 2 tokens, p=0.95 -> 4)")

    # 8. Negative sampling: never a positive, and rare items are reachable.
    counts = np.array([1000.0, 200.0, 20.0, 2.0])
    neg = negative_sampling(counts, positives=np.array([0]), n=40_000, rng=rng)
    assert not np.any(neg == 0)
    emp = np.bincount(neg, minlength=4)[1:] / neg.size
    want = counts[1:] ** 0.75
    want = want / want.sum()
    assert np.max(np.abs(emp - want)) < 0.01, (emp, want)
    print(f"negative sampling: P(item 3) = {emp[2]:.4f} (never item 0)")

    # 9. Importance sampling: correct in expectation, and clearly better than the
    #    naive estimator when q is chosen to match the integrand.
    def p(t):
        return 2.0 * t                        # density on [0,1]

    def q(t):
        return np.ones_like(t)

    xs = rng.random(200_000)
    est = importance_sampling(xs, p, q, f=lambda t: t)
    naive = float(np.mean(uniform(200_000, rng)))  # E under q, not p
    exact = 2.0 / 3.0
    assert abs(est - exact) < 0.005, est
    assert abs(naive - exact) > 0.1
    print(f"importance sampling: {est:.4f} vs exact {exact:.4f} (naive {naive:.4f})")

    # 10. Rejection: acceptance rate is area/envelope, and the draws are right.
    def f(t):
        return 2.0 * t

    rej = rejection(f, 100_000, envelope=2.0, rng=rng)
    assert abs(rej.mean() - 2.0 / 3.0) < 0.01, rej.mean()
    accept = 1.0 / 2.0
    assert abs(accept - 0.5) < 1e-12
    print(f"rejection: mean {rej.mean():.4f}, acceptance {accept:.2f}")
    print("sampling demo passed")


if __name__ == "__main__":
    main()
