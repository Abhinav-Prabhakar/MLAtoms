"""Probabilistic LSA — an aspect model, and three ways to fit it.

Chapter:   06-latent-factorization
Companion: plsa.md
Tier 3 · 3 variants: em, tempered_em, map_plsa

    p(w|d) = Σ_z p(w|z) p(z|d),    p(z|d) ≥ 0,  Σ_z p(z|d) = 1                (1)

Every document is a mixture of topics, every topic is a distribution over words,
and the word distribution of a document is the mixture. That is the entire model:
`X`, the document × term count matrix, is the only observed thing, and the two
factors are `doc_topic_` — how much of each topic each document uses — and
`topic_word_` — what each topic talks about.

It is a latent-variable model, fitted by EM with a closed form at every step. The
E-step computes the responsibility of topic `z` for one occurrence of word `w` in
document `d`; the M-step is a weighted average of the counts. Maximum likelihood
is the objective, and maximum likelihood is also the problem: nothing in it
charges for parameters, so on the demo's corpus twenty topics reach a higher
likelihood per token than the three that generated the data (-3.47257 against
-3.58817) while recovering 0.5978 of the planted topics against 0.9971. Held-out
perplexity does not save you either — with fold-in it prefers twenty as well,
34.20 against 36.24. That missing complexity control is why LDA exists.

The three variants answer two different failures. `em` is plain EM.
`tempered_em` anneals the responsibilities: with `beta = 0.8` it beats plain EM
on all four starts at six topics, and with `beta = 0.4` it collapses to the
symmetric point where every topic is identical, in 23 iterations, because
identical rows stay identical under every further M-step. `map_plsa` adds a
Dirichlet prior to the topic-word counts, which is the only thing here that gives
a term no training document contains any mass at all: under plain EM a document
of such terms has infinite perplexity, and with `alpha = 5` it has 768.2.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
VARIANTS = ("em", "map_plsa", "tempered_em")


def _normalise_rows(P: np.ndarray) -> np.ndarray:
    """Scale every row to sum 1; an all-zero row becomes uniform rather than nan.

    The uniform fallback is not cosmetic. With a Dirichlet prior below 1 the
    update can zero a row outright, and a dead topic then produces `0/0` in every
    later step and a `nan` three functions away from the cause.
    """
    s = P.sum(axis=1, keepdims=True)
    dead = s[:, 0] < EPS
    if np.any(dead):
        return np.where(dead[:, None], 1.0 / P.shape[1], P / np.maximum(s, EPS))
    return P / s


def _e_step(X: np.ndarray, topic_word: np.ndarray, doc_topic: np.ndarray,
            beta: float = 1.0) -> np.ndarray:
    """`p(z|d,w)`, `(D, V, Z)` — the responsibilities, optionally tempered.

    `beta = 1` is the model's own posterior. `beta < 1` flattens it, which is what
    deterministic annealing wants: early iterations see a surface with fewer and
    wider basins, and the temperature is raised to 1 as the fit settles.
    """
    log_joint = (np.log(np.maximum(doc_topic, EPS))[:, None, :]
                 + np.log(np.maximum(topic_word, EPS)).T[None, :, :])
    log_joint *= beta
    top = log_joint.max(axis=2, keepdims=True)
    q = np.exp(log_joint - top)
    return q / np.maximum(q.sum(axis=2, keepdims=True), EPS)


def _log_likelihood(X: np.ndarray, topic_word: np.ndarray,
                    doc_topic: np.ndarray) -> float:
    """`Σ_{d,w} n(d,w) log Σ_z p(z|d)p(w|z)`, exactly, including `-inf`.

    A word that no topic assigns mass to contributes `-inf` rather than a very
    negative number, so a completely unseen word makes the held-out likelihood
    infinite. Clipping the logs would turn that into `exp(25)` and hide the
    failure behind a large finite perplexity.
    """
    with np.errstate(divide="ignore"):
        log_joint = (np.log(doc_topic)[:, None, :]
                     + np.log(topic_word).T[None, :, :])
    top = log_joint.max(axis=2, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        log_sum = top[:, :, 0] + np.log(np.exp(log_joint - top).sum(axis=2))
    nz = X > 0
    return float(np.sum(X[nz] * log_sum[nz]))


def _perplexity(X: np.ndarray, topic_word: np.ndarray,
                doc_topic: np.ndarray) -> float:
    """`exp(−log p(X) / n_tokens)`; `inf` when the model assigns a token zero."""
    ll = _log_likelihood(X, topic_word, doc_topic)
    return float(np.exp(-ll / max(float(X.sum()), EPS)))


class PLSA:
    """Aspect model for a document × term count matrix.

    `PLSA(n_topics=3, *, variant="em", n_iter=200, tol=1e-7, beta=0.5,
    alpha=2.0, seed=0)`. `fit(X)` takes counts `(D, V)` and produces `topic_word_`
    `(Z, V)`, `doc_topic_` `(D, Z)`, `log_likelihood_` after each iteration and
    `n_iter_`. `transform(X)` folds new documents in with the topics held fixed.

    `beta` is the temperature `tempered_em` starts from and `alpha` the symmetric
    Dirichlet concentration `map_plsa` adds to every count; both are ignored by
    the variants that do not use them.
    """

    def __init__(self, n_topics: int = 3, *, variant: str = "em",
                 n_iter: int = 200, tol: float = 1e-7, beta: float = 0.5,
                 alpha: float = 2.0, seed: int = 0) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if n_topics < 1:
            raise ValueError("n_topics must be at least 1")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta is a temperature; it must be in (0, 1]")
        if alpha <= 0.0:
            raise ValueError("alpha is a Dirichlet concentration; it must be positive")
        self.n_topics = int(n_topics)
        self.variant = variant
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.seed = int(seed)

    def _temperature(self, it: int) -> float:
        """Beta schedule: climb linearly from `beta` to 1 over the first half.

        Deterministic and monotone, which is all annealing needs. The half-way
        point is a choice, not a result: anneal for too long and the fit never
        reaches the true posterior; too short and the early flattending is wasted.
        """
        if self.variant != "tempered_em":
            return 1.0
        span = max(1, self.n_iter // 2)
        return self.beta + (1.0 - self.beta) * min(1.0, it / span)

    def fit(self, X: np.ndarray) -> PLSA:
        """Run the selected variant from a random start; returns `self`.

        `X` is `(D, V)` of non-negative counts. The initialisation is a random
        row-stochastic pair, which is the standard choice: EM has no way to
        escape a bad basin, so the start is the algorithm's only exploration.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (documents, terms), got shape {X.shape}")
        if float(X.min()) < 0.0:
            raise ValueError("X holds counts; negative entries are not counts")
        rng = np.random.default_rng(self.seed)
        self.topic_word_ = _normalise_rows(rng.random((self.n_topics, X.shape[1])))
        self.doc_topic_ = _normalise_rows(rng.random((X.shape[0], self.n_topics)))

        self.log_likelihood_ = []
        self.n_iter_ = self.n_iter
        for it in range(self.n_iter):
            q = _e_step(X, self.topic_word_, self.doc_topic_,
                        self._temperature(it))
            counts_word = np.einsum("dw,dwz->zw", X, q)
            if self.variant == "map_plsa":
                # Dirichlet(alpha) prior on each p(·|z): the MAP solution adds
                # alpha − 1 to every count, which is smoothing above 1 and
                # sparsification below it
                counts_word = np.maximum(counts_word + self.alpha - 1.0, 0.0)
            self.topic_word_ = _normalise_rows(counts_word)
            self.doc_topic_ = _normalise_rows(np.einsum("dw,dwz->dz", X, q))

            self.log_likelihood_.append(
                _log_likelihood(X, self.topic_word_, self.doc_topic_))
            if (len(self.log_likelihood_) > 1
                    and abs(self.log_likelihood_[-1] - self.log_likelihood_[-2])
                    < self.tol * max(abs(self.log_likelihood_[-1]), 1.0)):
                self.n_iter_ = it + 1
                break
        return self

    def transform(self, X: np.ndarray, n_iter: int = 100) -> np.ndarray:
        """`p(z|d)` for new documents, with `topic_word_` held fixed.

        Folding in is EM on one factor only: the topics are the model and a new
        document is a mixture over them. Its held-out likelihood is the honest
        measure of the fit, because the topics were not allowed to see it.
        """
        if not hasattr(self, "topic_word_"):
            raise RuntimeError("call fit before transform")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rng = np.random.default_rng(self.seed)
        doc_topic = _normalise_rows(rng.random((X.shape[0], self.n_topics)))
        for _ in range(n_iter):
            q = _e_step(X, self.topic_word_, doc_topic)
            doc_topic = _normalise_rows(np.einsum("dw,dwz->dz", X, q))
        return doc_topic

    def perplexity(self, X: np.ndarray) -> float:
        """Held-out perplexity of `X` under the fitted model, folding in first."""
        return _perplexity(np.asarray(X, dtype=float), self.topic_word_,
                           self.transform(X))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _corpus(n_docs: int = 120, n_terms: int = 60, n_topics: int = 3,
            per_topic: int = 10, doc_len: int = 80, seed: int = 0,
            drop: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counts from a known aspect model; `drop` terms never appear in the counts.

    The truth is returned so the demo can score recovery. `drop` removes the
    highest-numbered terms from every document, which is how a corpus with an
    out-of-vocabulary problem is made: those terms exist in the world and not in
    the training data.
    """
    rng = np.random.default_rng(seed)
    topic_word = np.zeros((n_topics, n_terms))
    for z in range(n_topics):
        words = np.arange(z * per_topic, (z + 1) * per_topic) % n_terms
        topic_word[z, words] = rng.uniform(0.5, 1.5, words.size)
        topic_word[z] += 0.02 * rng.random(n_terms)
    if drop:
        topic_word[:, n_terms - drop:] = 0.0
    topic_word = _normalise_rows(topic_word)
    doc_topic = rng.dirichlet(np.full(n_topics, 0.4), size=n_docs)
    word_p = doc_topic @ topic_word
    X = np.stack([rng.multinomial(doc_len, p / p.sum()) for p in word_p])
    return X, topic_word, doc_topic


def _topic_recovery(learned: np.ndarray, truth: np.ndarray) -> float:
    """Mean best cosine per learned topic — topics are identified up to order."""
    ln = learned / np.maximum(np.linalg.norm(learned, axis=1, keepdims=True), EPS)
    tn = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), EPS)
    return float(np.mean(np.max(ln @ tn.T, axis=1)))


def _demo_em() -> None:
    """Monotone likelihood, recovered topics, and an objective that cannot count."""
    X, truth, _ = _corpus(per_topic=20)
    fit = PLSA(3, variant="em", n_iter=600, tol=1e-9).fit(X)
    rec = _topic_recovery(fit.topic_word_, truth)
    ll = np.array(fit.log_likelihood_)
    rises = int(np.sum(np.diff(ll) < -1e-9))
    assert rec > 0.99 and rises == 0 and fit.n_iter_ < 600, (rec, rises, fit.n_iter_)

    wide = PLSA(20, variant="em", n_iter=150, tol=1e-9).fit(X)
    rec_wide = _topic_recovery(wide.topic_word_, truth)
    per_token = [float(f.log_likelihood_[-1]) / float(X.sum()) for f in (fit, wide)]
    narrow = [float(np.mean(f.topic_word_ > 1e-3)) for f in (fit, wide)]
    assert per_token[1] > per_token[0] and rec_wide < 0.7, (per_token, rec_wide)

    train, held = X[:60], X[60:]
    ppl = [PLSA(z, variant="em", n_iter=150, tol=1e-9).fit(train).perplexity(held)
           for z in (3, 20)]
    assert ppl[1] < ppl[0], ppl

    print(
        f"em: 120 documents of 80 tokens over 60 terms, three planted disjoint "
        f"aspects. Three topics recover them at mean cosine {rec:.4f} in "
        f"{fit.n_iter_} iterations, and the log-likelihood rises on every one of "
        f"the {len(ll) - 1} steps — that is EM's guarantee, and it is the whole of "
        f"it. Ask for twenty topics on the same data and the likelihood per token "
        f"goes *up*, {per_token[0]:.5f} → {per_token[1]:.5f}: the objective has no "
        f"term that charges for parameters, so it cannot tell that three is the "
        f"answer. Recovery falls to {rec_wide:.4f} and each topic narrows from "
        f"{narrow[0]:.2f} to {narrow[1]:.2f} of the vocabulary above 1e-3. "
        f"Held-out perplexity, the usual defence, sides with the wrong model: "
        f"{ppl[0]:.2f} with three topics against {ppl[1]:.2f} with twenty, because "
        f"folding a held-out document in gives it its own twenty-vector and that is "
        f"enough freedom to fit the damage"
    )


def _demo_tempered() -> None:
    """A schedule, not an estimator: what annealing buys, and how it collapses."""
    X, truth, _ = _corpus(per_topic=20)
    plain = PLSA(6, variant="em", n_iter=300, tol=1e-9).fit(X)
    beta_one = PLSA(6, variant="tempered_em", n_iter=300, tol=1e-9, beta=1.0).fit(X)
    gap = abs(beta_one.log_likelihood_[-1] - plain.log_likelihood_[-1])
    assert gap < 1e-12, gap

    rows = []
    for seed in range(4):
        pair = []
        for variant in ("em", "tempered_em"):
            f = PLSA(6, variant=variant, n_iter=300, tol=1e-9, beta=0.8,
                     seed=seed).fit(X)
            pair.append(f.log_likelihood_[-1])
        rows.append(pair)
    wins = sum(t > p + 1e-9 for p, t in rows)
    gains = [t - p for p, t in rows]
    assert wins == len(rows) and float(np.mean(gains)) > 10.0, (wins, gains)

    plain3 = PLSA(3, variant="em", n_iter=300, tol=1e-9).fit(X)
    cold = PLSA(3, variant="tempered_em", n_iter=300, tol=1e-9, beta=0.4).fit(X)
    spread = float(np.max(np.abs(cold.topic_word_[:, None, :]
                                 - cold.topic_word_[None, :, :])))
    flat = float(np.mean(cold.doc_topic_.max(axis=1)))
    drops = int(np.sum(np.diff(np.array(cold.log_likelihood_)) < -1e-9))
    assert spread < 1e-4 and drops == len(cold.log_likelihood_) - 1, (spread, drops)
    assert cold.log_likelihood_[-1] < plain3.log_likelihood_[-1] - 4000.0

    print(
        f"tempered_em: β = 1 has to reproduce em and does, to {gap:.1e} — the "
        f"variant is a schedule wrapped around the same M-step, not a different "
        f"estimator. With β = 0.8 climbing to 1 over the first 150 iterations and "
        f"six topics, it beats plain em on all {len(rows)} starts, by "
        f"{float(np.mean(gains)):.1f} nats on average and {max(gains):.1f} at best, "
        f"from the same initialisation: the flatter early surface keeps the fit out "
        f"of the basin plain EM settles into. Push the temperature to β = 0.4 and "
        f"the same code dies in {len(cold.log_likelihood_)} iterations. "
        f"Responsibilities that flat hand every topic the same counts; identical "
        f"rows stay identical under every further M-step, so the run stops on a "
        f"symmetric point where the topics agree to {spread:.1e} in every entry and "
        f"every document is {flat:.4f} of each. The likelihood falls on all "
        f"{drops} steps, to {cold.log_likelihood_[-1]:.1f} against "
        f"{plain3.log_likelihood_[-1]:.1f} for em. Annealing decreases the free "
        f"energy, not the likelihood, and the distance between those two objectives "
        f"is the entire risk"
    )


def _demo_map() -> None:
    """The prior is the only thing that gives an unseen word any mass at all."""
    X, truth, _ = _corpus(per_topic=20, drop=5)
    assert X[:, 55:].sum() == 0.0          # five terms exist in the world, not here
    plain = PLSA(3, variant="em", n_iter=300, tol=1e-9).fit(X)
    grid = [PLSA(3, variant="map_plsa", n_iter=300, tol=1e-9, alpha=a).fit(X)
            for a in (1.0, 5.0, 20.0)]
    gap = abs(grid[0].log_likelihood_[-1] - plain.log_likelihood_[-1])
    assert gap < 1e-12, gap

    tiny = [float(f.topic_word_.min()) for f in grid]
    ent = [float(np.mean(-np.sum(f.topic_word_
                                 * np.log(np.maximum(f.topic_word_, EPS)), axis=1)))
           for f in grid]
    per_token = [float(f.log_likelihood_[-1]) / float(X.sum()) for f in grid]
    assert tiny[0] == 0.0 and tiny[-1] > 1e-3, tiny

    probe = np.zeros((1, X.shape[1]))
    probe[0, 57] = 10.0                    # ten tokens of a term no document used
    ppl_plain = plain.perplexity(probe)
    ppl_map = grid[1].perplexity(probe)
    assert np.isinf(ppl_plain) and np.isfinite(ppl_map), (ppl_plain, ppl_map)

    print(
        f"map_plsa: α = 1 reproduces em to {gap:.1e} — the prior is additive, so "
        f"concentration 1 is no prior at all. Raising it to 5 and 20 pulls every "
        f"topic towards uniform: mean entropy climbs "
        f"{ent[0]:.3f} → {ent[1]:.3f} → {ent[2]:.3f} nats, the smallest entry of "
        f"topic_word_ goes from exactly {tiny[0]:.1f} to {tiny[1]:.1e} and "
        f"{tiny[2]:.1e}, and the training likelihood per token pays for it, "
        f"{per_token[0]:.4f} → {per_token[1]:.4f} → {per_token[2]:.4f}. That is the "
        f"bargain, and it is not cosmetic: hand the model ten tokens of a term that "
        f"appears in no training document and em's perplexity is {ppl_plain}, "
        f"because one unseen word multiplies the likelihood by zero and no "
        f"arithmetic recovers it. With α = 5 the perplexity is {ppl_map:.1f}. "
        f"A prior is what lets a model say 'unlikely' instead of 'impossible', and "
        f"the likelihood it gives up is the price of that sentence"
    )


def main() -> None:
    _demo_em()
    _demo_tempered()
    _demo_map()
    print("plsa demo passed")


if __name__ == "__main__":
    main()
