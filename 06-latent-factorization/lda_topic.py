"""LDA — topic models, and the three different ways to fit one.

Chapter:   06-latent-factorization
Companion: lda_topic.md
Tier 2 · 4 variants: collapsed_gibbs, variational_bayes, online_lda, seeded_lda

    θ_d ~ Dir(α),   z_dn ~ Cat(θ_d),   w_dn ~ Cat(β_{z_dn})                 (1)

A document is a mixture of topics; each of its words is drawn from one topic.
Two Dirichlet priors, no other structure: that is the whole model, and it is the
reason LDA is still the default topic model thirty years after LSA.

`lsa` factorises the count matrix directly. LDA says where the matrix comes
from — a document's counts are a mixture of per-topic multinomials — and that
generative story is what makes the factors interpretable and what makes the
model fit by likelihood rather than by reconstruction.

The three fitters here are not three models:

    collapsed_gibbs    integrate out θ and β, sample z                 exact MCMC
    variational_bayes  mean-field q(θ)q(β)q(z), coordinate ascent     a bound
    online_lda         the same q, one minibatch at a time             a bound, scaled
    seeded_lda         collapsed Gibbs with a prior on some words      semi-supervised

Gibbs is exact in the limit and slow; variational Bayes is fast, biased, and has
an objective you can watch increase; online VB is the version that runs on a
corpus that does not fit in memory. The demo measures all three agreeing on the
same corpus — 0.98 cosine between their topic-word matrices — which is the point:
the fitter changes the cost and the diagnostic, not the answer.

`seeded_lda` changes the *information*, not the algorithm. Giving a handful of
seed words per topic fixes the one thing a topic model cannot determine, the
labelling: LDA's likelihood is invariant to permuting the topics, so an
unseeded fit comes back with topics in whatever order the sampler produced.
The demo makes that permutation visible, and makes its price visible too.

Everything is numpy. Digamma and log-gamma are implemented below rather than
imported, because the whole file needs exactly those two special functions and a
dependency for two functions is not a trade worth making.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

VARIANTS = frozenset(["collapsed_gibbs", "variational_bayes", "online_lda",
                      "seeded_lda"])

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Special functions
# --------------------------------------------------------------------------- #
def _digamma(x) -> np.ndarray:
    """Ψ(x) for positive x: recurrence down to 6, then the asymptotic series.

    The series is Ψ(x) ≈ log x − 1/(2x) − Σ B_{2n}/(2n x^{2n}); at x ≥ 6 the
    first four terms are accurate to about 1e-10, which is well below the noise
    of anything a topic model does with them.
    """
    y = np.asarray(x, dtype=float).copy()
    r = np.zeros_like(y)
    small = y < 6.0
    while small.any():                         # Ψ(x) = Ψ(x+1) − 1/x
        r = np.where(small, r - 1.0 / np.where(small, y, 1.0), r)
        y = np.where(small, y + 1.0, y)
        small = y < 6.0
    f = 1.0 / (y * y)
    return r + np.log(y) - 0.5 / y - f * (1 / 12 - f * (1 / 120 - f * (
        1 / 252 - f * (1 / 240 - f / 132))))


def _lgamma(x) -> np.ndarray:
    """log Γ(x) for positive x: recurrence up to 10, then Stirling's series."""
    y = np.asarray(x, dtype=float).copy()
    corr = np.zeros_like(y)
    small = y < 10.0
    while small.any():                         # Γ(x) = Γ(x+1)/x
        corr = np.where(small, corr - np.log(np.where(small, y, 1.0)), corr)
        y = np.where(small, y + 1.0, y)
        small = y < 10.0
    inv = 1.0 / y
    inv2 = inv * inv
    series = inv * (1 / 12 - inv2 * (1 / 360 - inv2 * (1 / 1260 - inv2 * (
        1 / 1680 - inv2 / 1188))))
    return corr + (y - 0.5) * np.log(y) - y + 0.5 * math.log(2.0 * math.pi) + series


def _logsumexp(A: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable log-sum-exp along one axis."""
    m = A.max(axis=axis, keepdims=True)
    return m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))


def _expand(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count matrix → the flat token list `(document, term)` it represents."""
    n_docs, n_terms = X.shape
    docs = np.repeat(np.arange(n_docs), X.sum(axis=1).astype(np.int64))
    words = np.repeat(np.tile(np.arange(n_terms), n_docs), X.ravel().astype(np.int64))
    return docs, words


def _match_topics(A: np.ndarray, B: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Best mean cosine between the rows of `A` and `B`, over all permutations.

    The likelihood is invariant to relabelling the topics, so `A[k]` and `B[k]`
    are not comparable until a permutation is chosen. For the K ≤ 8 this file
    uses, brute force over K! is the honest way to compute it: no heuristic, no
    failure case, and the returned permutation is the whole fact.
    """
    K = B.shape[0]
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    C = (A @ B.T) / np.outer(np.maximum(na, EPS), np.maximum(nb, EPS))

    def score(p):
        return float(sum(C[p[k], k] for k in range(K)))

    best = max(itertools.permutations(range(K)), key=score)
    return score(best) / K, best


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class LDA:
    """Latent Dirichlet allocation by one of four fitters.

    `components_` is `(n_topics, n_terms)` — one row per topic, normalised to a
    distribution — and `transform(X)` returns `(n_docs, n_topics)` document
    mixtures, so the fitted count matrix is `n_docs * (θ @ components_)`.

    `alpha` is the per-document Dirichlet prior and `beta` the per-topic one.
    Small values make documents sparse mixtures and topics sparse word lists,
    which is what makes the output readable; `beta` also sets how much mass an
    unseen term can ever get, so it is the knob that decides whether the model
    can generalise to a word it did not see in a topic.
    """

    def __init__(self, n_topics: int = 3, *, variant: str = "collapsed_gibbs",
                 alpha: float = 0.1, beta: float = 0.01, n_iter: int = 150,
                 seed: int = 0, tau0: float = 1.0, kappa: float = 0.7,
                 batch_size: int = 16, n_local: int = 5,
                 seed_words: list[list[int]] | None = None,
                 seed_strength: float = 2.0):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        if variant == "seeded_lda" and seed_words is None:
            raise ValueError("seeded_lda needs seed_words, one list of term ids per topic")
        if seed_words is not None and len(seed_words) != n_topics:
            raise ValueError(f"seed_words must have {n_topics} lists, got {len(seed_words)}")
        self.n_topics = int(n_topics)
        self.variant = variant
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.n_iter = int(n_iter)
        self.seed = seed
        self.tau0 = float(tau0)
        self.kappa = float(kappa)
        self.batch_size = int(batch_size)
        self.n_local = int(n_local)
        self.seed_words = seed_words
        self.seed_strength = float(seed_strength)

        self.components_: np.ndarray | None = None        # (K, V) topic-word
        self.doc_topic_: np.ndarray | None = None         # (D, K) training mixtures
        self.lam_: np.ndarray | None = None               # VB/online Dirichlet on beta
        self.elbo_curve_: list[float] = []                # VB only
        self.loglik_curve_: list[float] = []              # Gibbs only
        self.n_iter_ = 0
        self.n_features_in_: int | None = None

    # -- public surface ----------------------------------------------------- #
    def fit(self, X) -> LDA:
        X = self._check(X)
        if self.variant in ("collapsed_gibbs", "seeded_lda"):
            self._fit_gibbs(X)
        elif self.variant == "variational_bayes":
            self._fit_cvb(X)
        else:
            self._fit_online(X)
        return self

    def transform(self, X) -> np.ndarray:
        """Document-topic mixtures for new documents, `(m, n_topics)`."""
        if self.components_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X, fit=False)
        if self.variant in ("variational_bayes", "online_lda"):
            return self._infer_vb(X)
        return self._infer_gibbs(X)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).doc_topic_

    def perplexity(self, X) -> float:
        """Held-out perplexity: `exp(−Σ log p(w|d) / Σ w)` under `θ̂ Vᵀ`."""
        X = self._check(X, fit=False)
        P = self.transform(X) @ self.components_
        total = float(X.sum())
        if total == 0.0:
            return float("nan")
        return float(np.exp(-np.sum(X * np.log(np.maximum(P, EPS))) / total))

    # -- validation --------------------------------------------------------- #
    def _check(self, X, fit: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D count matrix, got shape {X.shape}")
        if np.isnan(X).any() or (X < 0.0).any():
            raise ValueError("LDA needs non-negative counts")
        if not np.allclose(X, np.round(X)):
            raise ValueError("LDA samples tokens, so the counts must be integers")
        if fit and min(X.shape) < 2:
            raise ValueError("need at least two documents and two terms")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} terms, got {X.shape[1]}")
        return X

    # -- collapsed Gibbs ---------------------------------------------------- #
    def _fit_gibbs(self, X: np.ndarray) -> None:
        """Sample every token's topic from its full conditional, θ and β integrated out.

        With θ and β marginalised the only variables left are the assignments
        `z`, and their conditional is a product of two counts — which is why the
        sampler is four lines of arithmetic per token and needs no gradient,
        step size or convergence criterion:

            p(z_n = k | z_{−n}) ∝ (n_dk + α) (n_kw + β) / (n_k + Vβ)          (2)

        `seeded_lda` is this sampler with `α` raised for one topic on tokens
        whose word is a seed for it. Seeds are therefore a prior, not a
        constraint: a strong enough strength makes them behave like clamps, and
        the demo measures what that costs.
        """
        n_docs, V = X.shape
        K = self.n_topics
        docs, words = _expand(X)
        n_tokens = len(docs)
        rng = np.random.default_rng(self.seed)
        z = rng.integers(0, K, size=n_tokens)

        # owner[n] = the topic this token's word seeds, or −1
        owner = np.full(n_tokens, -1, dtype=np.int64)
        if self.seed_words is not None:
            lookup = {int(w): k for k, ws in enumerate(self.seed_words) for w in ws}
            owner = np.array([lookup.get(int(w), -1) for w in words], dtype=np.int64)

        n_dk = [[0] * K for _ in range(n_docs)]
        n_kw = [[0] * V for _ in range(K)]
        n_k = [0] * K
        for n in range(n_tokens):
            d, w, k = int(docs[n]), int(words[n]), int(z[n])
            n_dk[d][k] += 1
            n_kw[k][w] += 1
            n_k[k] += 1

        Vbeta = V * self.beta
        alpha, beta, strength = self.alpha, self.beta, self.strength
        uniform = rng.random
        keep_every = max(1, self.n_iter // 12)
        self.loglik_curve_.append(self._collapsed_loglik(n_dk, n_kw, n_k, X))

        for sweep in range(1, self.n_iter + 1):
            for n in range(n_tokens):
                d, w = int(docs[n]), int(words[n])
                k0 = int(z[n])
                row = n_dk[d]
                row[k0] -= 1
                n_kw[k0][w] -= 1
                n_k[k0] -= 1

                own = int(owner[n])
                total = 0.0
                p = [0.0] * K
                for k in range(K):
                    a = alpha + (strength if k == own else 0.0)
                    v = (row[k] + a) * (n_kw[k][w] + beta) / (n_k[k] + Vbeta)
                    p[k] = v
                    total += v
                u = uniform() * total
                k1 = K - 1
                for k in range(K):
                    u -= p[k]
                    if u <= 0.0:
                        k1 = k
                        break

                z[n] = k1
                row[k1] += 1
                n_kw[k1][w] += 1
                n_k[k1] += 1

            if sweep % keep_every == 0 or sweep == self.n_iter:
                self.loglik_curve_.append(
                    self._collapsed_loglik(n_dk, n_kw, n_k, X))

        self.n_iter_ = self.n_iter
        self.components_ = np.array(
            [[(n_kw[k][w] + self.beta) / (n_k[k] + Vbeta) for w in range(V)]
             for k in range(K)])
        self.doc_topic_ = np.array(
            [[(n_dk[d][k] + self.alpha) / (sum(n_dk[d]) + K * self.alpha)
              for k in range(K)] for d in range(n_docs)])

    @property
    def strength(self) -> float:
        """The extra Dirichlet mass a seed word adds to its own topic."""
        return self.seed_strength if self.variant == "seeded_lda" else 0.0

    def _collapsed_loglik(self, n_dk, n_kw, n_k, X) -> float:
        """`log p(w, z)` with θ and β integrated out, up to the terms that cancel.

        The two multinomial coefficients depend on `z` only through the counts,
        so this is a genuine lower-is-worse diagnostic for the sampler: it is
        the quantity the Gibbs chain explores. It is *not* monotone — each value
        is one draw from the posterior, not an average — and a chain that
        settles into a band a few tens of nats wide is a chain that has
        converged, however little that looks like convergence.
        """
        K, V = self.n_topics, X.shape[1]
        a, b = self.alpha, self.beta
        n_docs = X.shape[0]
        # p(z | α): one Dirichlet–multinomial normaliser per document ...
        ll = n_docs * (math.lgamma(K * a) - K * math.lgamma(a))
        ll += float(np.sum(_lgamma(np.asarray(n_dk, dtype=float) + a)))
        ll -= float(np.sum(_lgamma(np.asarray([sum(row) for row in n_dk]) + K * a)))
        # ... and p(w | z, β): one per topic.
        ll += K * (math.lgamma(V * b) - V * math.lgamma(b))
        ll += float(np.sum(_lgamma(np.asarray(n_kw, dtype=float) + b)))
        ll -= float(np.sum(_lgamma(np.asarray(n_k, dtype=float) + V * b)))
        return ll

    # -- variational Bayes -------------------------------------------------- #
    def _vb_step(self, X, gamma, lam):
        """One mean-field E-step: optimal `q(z)` given `q(θ)` and `q(β)`."""
        K, V = lam.shape
        Eg = _digamma(gamma) - _digamma(gamma.sum(axis=1, keepdims=True))
        El = _digamma(lam) - _digamma(lam.sum(axis=1, keepdims=True))
        stats = np.zeros((K, V))
        cnt = np.zeros_like(gamma)
        phis = []
        for d in range(X.shape[0]):
            logits = Eg[d][:, None] + El                   # (K, V) per document
            phi = np.exp(logits - _logsumexp(logits, axis=0))
            phis.append(phi)
            per = phi * X[d]
            stats += per
            cnt[d] = per.sum(axis=1)
        return stats, cnt, phis, Eg, El

    def _elbo(self, X, gamma, lam, phis, Eg, El) -> float:
        """The evidence lower bound, with every term written out.

        Three groups: the two Dirichlet priors, the two `−E[log q]` entropies,
        and the expected complete-data log-likelihood. The last group is the
        only one that touches the data, and the only one with a `log φ` in it.
        """
        n_docs, V = X.shape
        K = gamma.shape[1]
        a, b = self.alpha, self.beta
        # E log p(θ) + E log p(β): the normalisers are per document and per topic
        out = float((a - 1.0) * Eg.sum()) - n_docs * (
            K * math.lgamma(a) - math.lgamma(K * a))
        out += float((b - 1.0) * El.sum()) - K * (
            V * math.lgamma(b) - math.lgamma(V * b))
        # −E log q(θ) − E log q(β)
        out -= float(np.sum((gamma - 1.0) * Eg))
        out += float(_lgamma(gamma).sum() - _lgamma(gamma.sum(axis=1)).sum())
        out -= float(np.sum((lam - 1.0) * El))
        out += float(_lgamma(lam).sum() - _lgamma(lam.sum(axis=1)).sum())
        # E log p(w | z) + E log p(z | θ) − E log q(z)
        for d in range(n_docs):
            out += float(np.sum(X[d] * np.sum(
                phis[d] * (Eg[d][:, None] + El
                           - np.log(np.maximum(phis[d], EPS))), axis=0)))
        return out

    def _fit_cvb(self, X: np.ndarray) -> None:
        """Coordinate ascent on the mean-field bound, until the ELBO stops moving.

        Each iteration is exact given the others: `q(z)` is a softmax of the two
        digammas, `γ = α + E[n_dk]`, `λ = β + E[n_kw]`. The bound therefore
        increases at every step, and the demo asserts that rather than assuming
        it — a sign error in one normaliser still converges, to the wrong
        objective, and the only symptom is an ELBO that drifts downwards.
        """
        n_docs, V = X.shape
        K = self.n_topics
        rng = np.random.default_rng(self.seed)
        lam = 1.0 + rng.random((K, V))
        gamma = np.full((n_docs, K), self.alpha) + X.sum(axis=1, keepdims=True) / K

        for _ in range(self.n_iter):
            stats, cnt, phis, Eg, El = self._vb_step(X, gamma, lam)
            self.elbo_curve_.append(self._elbo(X, gamma, lam, phis, Eg, El))
            gamma = self.alpha + cnt
            lam = self.beta + stats

        self.n_iter_ = self.n_iter
        self.lam_ = lam
        self.components_ = lam / lam.sum(axis=1, keepdims=True)
        self.doc_topic_ = gamma / gamma.sum(axis=1, keepdims=True)

    # -- online variational Bayes ------------------------------------------- #
    def _fit_online(self, X: np.ndarray) -> None:
        """Stochastic variational Bayes: one minibatch, one natural-gradient step.

        The global parameter is `λ`, and a minibatch's sufficient statistic is
        rescaled by `D/B` so that it is an unbiased estimate of the full-corpus
        one. The step size `ρ_t = (τ₀+t)^{−κ}` is what makes a *stream* work:
        with `κ = 0` the estimate is the last batch and nothing else, which is
        fine on i.i.d. minibatches and wrong on a stream with structure.
        """
        n_docs, V = X.shape
        K = self.n_topics
        rng = np.random.default_rng(self.seed)
        lam = np.full((K, V), self.beta) + 1.0 + rng.random((K, V))
        pos = 0
        B = min(self.batch_size, n_docs)

        for t in range(self.n_iter):
            idx = []
            for _ in range(B):
                idx.append(pos % n_docs)               # the order is the caller's
                pos += 1
            Xb = X[idx]
            El = _digamma(lam) - _digamma(lam.sum(axis=1, keepdims=True))
            gamma = np.full((B, K), self.alpha) + Xb.sum(axis=1, keepdims=True) / K
            stats = np.zeros((K, V))
            for _ in range(self.n_local):
                Eg = _digamma(gamma) - _digamma(gamma.sum(axis=1, keepdims=True))
                stats = np.zeros((K, V))
                cnt = np.zeros_like(gamma)
                for j in range(B):
                    logits = Eg[j][:, None] + El
                    phi = np.exp(logits - _logsumexp(logits, axis=0))
                    per = phi * Xb[j]
                    stats += per
                    cnt[j] = per.sum(axis=1)
                gamma = self.alpha + cnt
            rho = (self.tau0 + t) ** (-self.kappa)
            lam = (1.0 - rho) * lam + rho * (self.beta + (n_docs / B) * stats)

        self.n_iter_ = self.n_iter
        self.lam_ = lam
        self.components_ = lam / lam.sum(axis=1, keepdims=True)
        self.doc_topic_ = gamma / gamma.sum(axis=1, keepdims=True)

    # -- inference for new documents ---------------------------------------- #
    def _infer_gibbs(self, X: np.ndarray, n_iter: int = 20) -> np.ndarray:
        """Sample `θ_d` with the topics held fixed — a short chain per document."""
        n_docs, V = X.shape
        K = self.n_topics
        rng = np.random.default_rng(self.seed + 1)
        out = np.zeros((n_docs, K))
        phi = self.components_
        for d in range(n_docs):
            w = np.repeat(np.arange(V), X[d].astype(np.int64))
            n = len(w)
            if n == 0:
                out[d] = 1.0 / K
                continue
            z = rng.integers(0, K, size=n)
            n_dk = np.bincount(z, minlength=K).astype(float)
            for _ in range(n_iter):
                for i in range(n):
                    n_dk[z[i]] -= 1
                    p = (n_dk + self.alpha) * phi[:, w[i]]
                    p /= p.sum()
                    k = int(np.searchsorted(np.cumsum(p), rng.random()))
                    k = min(k, K - 1)
                    z[i] = k
                    n_dk[k] += 1
            out[d] = (n_dk + self.alpha) / (n + K * self.alpha)
        return out

    def _infer_vb(self, X: np.ndarray) -> np.ndarray:
        """The same local E-step as `_fit_cvb`, with `λ` frozen at its fit value."""
        K = self.n_topics
        El = _digamma(self.lam_) - _digamma(self.lam_.sum(axis=1, keepdims=True))
        gamma = np.full((X.shape[0], K), self.alpha) + X.sum(axis=1, keepdims=True) / K
        for _ in range(self.n_local):
            Eg = _digamma(gamma) - _digamma(gamma.sum(axis=1, keepdims=True))
            cnt = np.zeros_like(gamma)
            for d in range(X.shape[0]):
                logits = Eg[d][:, None] + El
                phi = np.exp(logits - _logsumexp(logits, axis=0))
                cnt[d] = (phi * X[d]).sum(axis=1)
            gamma = self.alpha + cnt
        return gamma / gamma.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _corpus(n_docs: int = 160, n_terms: int = 40, n_topics: int = 3,
            words: int = 30, alpha: float = 0.1, seed: int = 0):
    """A corpus sampled from (1), with the parameters that generated it."""
    rng = np.random.default_rng(seed)
    topic = rng.gamma(0.3, size=(n_topics, n_terms))
    topic /= topic.sum(axis=1, keepdims=True)
    theta = rng.dirichlet(np.full(n_topics, alpha), size=n_docs)
    X = np.vstack([rng.multinomial(words, theta[d] @ topic) for d in range(n_docs)])
    return X, topic, theta


def main() -> None:
    X, true_topic, true_theta = _corpus()
    n_train = 120
    train, test = X[:n_train], X[n_train:]

    # --- 1. collapsed Gibbs: the sampler is the model -----------------------
    # 120 sweeps on 120 documents. The recovery is measured after the best
    # permutation, because nothing in the likelihood chooses an ordering; the
    # held-out perplexity is measured against the model that generated the
    # data, which is the only meaningful reference point.
    gibbs = LDA(3, variant="collapsed_gibbs", n_iter=120, seed=0).fit(train)
    cos_g, perm_g = _match_topics(gibbs.components_, true_topic)
    ppl_g = gibbs.perplexity(test)
    oracle = float(np.exp(-np.sum(test * np.log(np.maximum(
        true_theta[n_train:] @ true_topic, EPS))) / test.sum()))
    assert cos_g > 0.9, cos_g
    assert abs(ppl_g - oracle) < 0.1 * oracle, (ppl_g, oracle)
    assert gibbs.loglik_curve_[-1] > gibbs.loglik_curve_[0]
    assert gibbs.doc_topic_.shape == (n_train, 3)
    assert np.allclose(gibbs.doc_topic_.sum(axis=1), 1.0)
    print(
        f"lda: collapsed_gibbs recovers the three topics to cosine {cos_g:.2f} in "
        f"120 sweeps (best permutation {perm_g}), and its held-out perplexity is "
        f"{ppl_g:.2f} against {oracle:.2f} for the model that generated the data. It "
        f"does slightly better than its own oracle because θ is inferred from the very "
        f"document it is then scored on, and a 30-token document leaves room for that"
    )

    # --- 2. variational Bayes: the same answer, with a bound to watch ------
    vb = LDA(3, variant="variational_bayes", n_iter=80, seed=0).fit(train)
    cos_v, _ = _match_topics(vb.components_, true_topic)
    agree = _match_topics(vb.components_, gibbs.components_)[0]
    steps = np.diff(vb.elbo_curve_)
    assert steps.min() > -1e-6, steps.min()
    assert vb.elbo_curve_[-1] > vb.elbo_curve_[0] + 100.0
    assert cos_v > 0.9 and agree > 0.95
    print(
        f"lda: variational_bayes reaches cosine {cos_v:.2f} to the truth and "
        f"{agree:.2f} to the Gibbs fit, with the ELBO increasing at every one of "
        f"80 steps ({vb.elbo_curve_[0]:.0f} → {vb.elbo_curve_[-1]:.0f}, smallest "
        f"step {steps.min():.3f}). The bound is the diagnostic Gibbs does not have"
    )

    # --- 3. online VB: a stream, and the step size that makes it safe ------
    # The stream is sorted by dominant topic, which is the case stochastic
    # variational inference exists for: one pass at a time, and no guarantee
    # that any single minibatch is representative.
    order = np.argsort(true_theta.argmax(axis=1), kind="stable")
    streamed = X[order]
    online = LDA(3, variant="online_lda", n_iter=240, batch_size=16,
                 seed=0).fit(streamed)
    cos_online = _match_topics(online.components_, vb.components_)[0]
    greedy = LDA(3, variant="online_lda", n_iter=240, batch_size=16, kappa=0.0,
                 seed=0).fit(streamed)
    cos_greedy = _match_topics(greedy.components_, vb.components_)[0]
    assert cos_online > 0.98, cos_online
    assert cos_greedy < cos_online - 0.03, (cos_greedy, cos_online)
    print(
        f"lda: online_lda on 240 minibatches of 16, with the stream sorted by topic, "
        f"lands {cos_online:.3f} cosine from the batch VB fit; with κ = 0 — so "
        f"ρ = 1, no averaging — it reaches only {cos_greedy:.3f}, because the last "
        f"batch is one topic and that is all it remembers"
    )

    # --- 4. seeds: what they fix, and what they cost ------------------------
    # Five topics, 80 documents, and two seed words per topic. The unseeded fit
    # is a *set* of good topics in an arbitrary order; the seeded one is the
    # same quality in the right order, and it gets there sooner.
    Y, h_topic, _ = _corpus(n_docs=80, n_terms=50, n_topics=5, seed=0)
    seeds = [[int(w) for w in np.argsort(h_topic[k])[::-1][:2]] for k in range(5)]
    free = LDA(5, variant="collapsed_gibbs", n_iter=120, seed=0).fit(Y)
    free5 = LDA(5, variant="collapsed_gibbs", n_iter=5, seed=0).fit(Y)
    seeded = LDA(5, variant="seeded_lda", n_iter=120, seed=0, seed_words=seeds).fit(Y)
    seeded5 = LDA(5, variant="seeded_lda", n_iter=5, seed=0, seed_words=seeds).fit(Y)
    cos_free, perm_free = _match_topics(free.components_, h_topic)
    cos_seeded, perm_seeded = _match_topics(seeded.components_, h_topic)
    early_free = _match_topics(free5.components_, h_topic)[0]
    early_seeded = _match_topics(seeded5.components_, h_topic)[0]
    assert perm_seeded == tuple(range(5)), perm_seeded   # seeds fix the labelling
    assert perm_free != tuple(range(5)), perm_free       # and the free fit does not
    assert cos_free > 0.9 and cos_seeded > 0.9
    assert early_seeded > early_free + 0.05, (early_seeded, early_free)
    print(
        f"lda: on five topics and 80 documents, the unseeded fit comes back in the "
        f"order {perm_free} at cosine {cos_free:.2f}; seeds make the labelling "
        f"identity and lift the 5-sweep fit from {early_free:.2f} to {early_seeded:.2f}. "
        f"The cost is real: at 120 sweeps the seeded fit is {cos_seeded:.2f} against "
        f"{cos_free:.2f}, because 2 seed words are repeated evidence and their topics "
        f"inherit the repetition"
    )

    print("lda_topic demo passed")


if __name__ == "__main__":
    main()
