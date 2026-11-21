"""LSA — the SVD of a term-document matrix, and the weighting that matters more.

Chapter:   06-latent-factorization
Companion: lsa.md
Tier 2 · 4 variants: truncated_svd, randomized, log_entropy, tfidf_lsa

    X  ∈ R^{n_docs × n_terms}_{≥0},     X ≈ U_k S_k V_kᵀ                (1)

Latent semantic analysis is a truncated SVD, and that is the least interesting
thing about it. The SVD is exact, it is the unique best rank-k approximation in
the Frobenius norm, and there is nothing to tune. What there *is* to tune is the
matrix you hand it: raw counts, log counts, tf-idf, log-entropy. Two variants
here are solvers for the same problem and two are weightings for it, and the demo
measures which choice matters — the solver reproduces the exact answer to 1e-8,
while the weighting moves retrieval accuracy by tens of percent.

The reason a weighting is needed at all is that raw counts measure length as much
as content. A long document mentions every term more often; a term that appears in
every document (`the`, `and`) separates nothing. tf-idf divides by document
frequency, log-entropy divides by how *evenly* a term is spread, and both then
put a long and a short document on the same footing.

What the SVD adds on top is that terms which never co-occur in the same document
still end up close in the k-dimensional space, because they co-occur with the
same third terms. That is the "latent" in latent semantic analysis, and it is the
same mechanism as factor analysis's shared factors — the low-rank model is what
generalises from seen pairs to unseen ones.

Everything is numpy: no sparse matrices, so `X` is dense and a term-document
matrix with a million documents is out of scope. The variants are here because
the weighting and the solver are the two decisions that matter, not because this
is a production retrieval system.
"""

from __future__ import annotations

import math

import numpy as np

VARIANTS = frozenset(["truncated_svd", "randomized", "log_entropy", "tfidf_lsa"])

# variant -> weighting, so that the two decisions are visible separately: the
# solver varies first and third, the weighting varies third and fourth
WEIGHTING = {"truncated_svd": "tf", "randomized": "tf",
             "log_entropy": "log_entropy", "tfidf_lsa": "tfidf"}

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Weightings
# --------------------------------------------------------------------------- #
def _local_weight(X: np.ndarray, kind: str) -> np.ndarray:
    """The term-frequency part: raw counts, or `log(1 + tf)` for log-entropy.

    The log is there to stop one term that a document happens to repeat from
    dominating its vector; it is a within-document decision and needs no corpus.
    """
    return np.log1p(X) if kind == "log_entropy" else X


def _term_weights(X: np.ndarray, kind: str) -> np.ndarray:
    """The corpus-level weight per term: `1`, `log(n / df)`, or `1 − H/log n`.

    These are **fitted** quantities. They are read off the training corpus, stored,
    and reused verbatim for every later query — which is the difference between a
    model of the corpus and a normalisation of whatever batch you happen to pass.
    """
    n, d = X.shape
    if kind == "tf":
        return np.ones(d)
    if kind == "tfidf":
        df = np.count_nonzero(X > 0.0, axis=0)
        return np.log(n / np.maximum(df, 1))
    total = X.sum(axis=0)
    p = X / np.maximum(total, EPS)                    # term j's mass over documents
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(X > 0.0, p * np.log(np.maximum(p, EPS)), 0.0)
    entropy = -plogp.sum(axis=0)                      # in nats, in [0, log n]
    # a one-document corpus has entropy 0 by definition and nothing to weigh
    return 1.0 - entropy / (math.log(n) if n > 1 else 1.0)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def _svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The top `k` singular triplets of `X`, descending, by one full SVD."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k]


def _randomized_svd(X: np.ndarray, k: int, n_iter: int, oversample: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko–Martinsson–Tropp: sketch, power-iterate, then solve a small problem.

    `Y = X Ω` with `Ω` Gaussian gives a `(n × (k + p))` sketch whose columns span
    the dominant subspace *plus* whatever the top of the neglected spectrum leaks
    in. Each power step `Y ← X (Xᵀ Y)` multiplies the ratio between the wanted and
    unwanted singular values by its square, so a handful of them buys a subspace
    that is accurate to machine precision; `n_iter = 0` is the honest no-iteration
    version, and the demo measures the difference.
    """
    n, d = X.shape
    width = min(k + oversample, min(n, d))
    omega = rng.standard_normal((d, width))
    Y = X @ omega
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)                            # orthonormal basis of the sketch
    Ub, s, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
class LSA:
    """Latent semantic analysis by one of four (weighting, solver) pairs.

    `components_` is `(k, n_terms)` — one row per latent dimension, sklearn's
    orientation — and `documents_` is `(n_docs, k)`, so a document vector is
    `X_w @ components_.T`. `weights_` is the per-term weighting, kept because it
    is the part of the model a user can inspect and argue with.
    """

    def __init__(self, n_components: int = 100, *, variant: str = "tfidf_lsa",
                 seed: int = 0, n_iter: int = 4, oversample: int = 10,
                 norm: bool = True):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
        self.n_components = int(n_components)
        self.variant = variant
        self.weighting = WEIGHTING[variant]
        self.seed = seed
        self.n_iter = int(n_iter)
        self.oversample = int(oversample)
        self.norm = bool(norm)

        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.documents_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.explained_variance_ratio_: float = 0.0
        self.reconstruction_error_: float = 0.0
        self.n_features_in_: int | None = None

    # -- public surface ----------------------------------------------------- #
    def fit(self, X) -> LSA:
        X = self._check(X)
        n, d = X.shape
        k = min(self.n_components, n, d)
        self.weights_ = _term_weights(X, self.weighting)
        Xw = self._weighted(X)
        if self.variant == "randomized":
            rng = np.random.default_rng(self.seed)
            U, s, Vt = _randomized_svd(Xw, k, self.n_iter, self.oversample, rng)
        else:
            U, s, Vt = _svd(Xw, k)
        self.components_ = Vt
        self.singular_values_ = s
        self.documents_ = U * s
        total = float(np.sum(Xw ** 2))                # ‖X_w‖_F² — not another SVD
        self.explained_variance_ratio_ = float(np.sum(s ** 2) / max(total, EPS))
        self.reconstruction_error_ = float(
            np.sqrt(max(total - np.sum(s ** 2), 0.0)))
        return self

    def transform(self, X) -> np.ndarray:
        """Document coordinates in the latent space, `(m, k)`."""
        if self.components_ is None:
            raise RuntimeError("call fit before transform")
        X = self._check(X, fit=False)
        return self._weighted(X) @ self.components_.T

    def _weighted(self, X: np.ndarray) -> np.ndarray:
        """Apply the *fitted* weights, then optionally L2-normalise each row."""
        Xw = _local_weight(X, self.weighting) * self.weights_
        if self.norm:
            # rows on the same scale, which makes cosine similarity a dot product
            nrm = np.linalg.norm(Xw, axis=1, keepdims=True)
            Xw = Xw / np.maximum(nrm, EPS)
        return Xw

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).documents_

    def inverse_transform(self, Z) -> np.ndarray:
        """`Z V_k`: back to term space. With `Z = fit_transform(X)` this is `U S Vᵀ`,
        the best rank-k approximation of the *weighted* matrix -- of the
        row-normalised one when `norm=True`, not of the raw counts."""
        if self.components_ is None:
            raise RuntimeError("call fit before inverse_transform")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.components_.shape[0]:
            raise ValueError(f"expected (m, {self.components_.shape[0]}), got {Z.shape}")
        return Z @ self.components_

    # -- validation --------------------------------------------------------- #
    def _check(self, X, fit: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if np.isnan(X).any() or (X < 0.0).any():
            raise ValueError("LSA needs a non-negative, complete count matrix")
        if fit and min(X.shape) < 2:
            raise ValueError("need at least two documents and two terms")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} terms, got {X.shape[1]}")
        return X


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _corpus(n_docs: int = 240, n_terms: int = 60, n_topics: int = 4, seed: int = 0,
            words: int = 40):
    """A topic model's output: each document draws `words` terms from its topic."""
    rng = np.random.default_rng(seed)
    topic = rng.gamma(0.3, size=(n_topics, n_terms))
    topic /= topic.sum(axis=1, keepdims=True)
    labels = np.arange(n_docs) % n_topics
    X = np.vstack([rng.multinomial(words, topic[j]) for j in labels])
    return X, labels, topic


def _tiny():
    """A 3 × 4 count matrix small enough to check the weightings by hand."""
    return np.array([[2.0, 1.0, 0.0, 3.0],
                     [1.0, 1.0, 2.0, 3.0],
                     [0.0, 3.0, 1.0, 3.0]])


def main() -> None:
    X, labels, _ = _corpus()
    n, d = X.shape
    k = 4

    # --- 1. the truncated SVD is the best rank-k approximation --------------
    # Eckart–Young, stated as an identity rather than as a tolerance: the energy
    # discarded is the tail of the spectrum, exactly, and no rank-k matrix does
    # better -- which is checked against 200 random ones. Note there is no
    # centring. PCA subtracts the mean; here the origin is meaningful (a document
    # with no words), so subtracting it would be a modelling choice, and it is
    # not taken.
    lsa = LSA(k, variant="truncated_svd").fit(X)
    Xw = lsa._weighted(X)                                    # tf weighting, L2 rows
    U, s, _ = np.linalg.svd(Xw, full_matrices=False)
    recon = U[:, :k] * s[:k] @ lsa.components_
    tail = float(np.sum(s[k:] ** 2))
    resid = float(np.linalg.norm(Xw - recon))
    assert abs(resid ** 2 - tail) < 1e-10, (resid ** 2, tail)
    assert np.all(np.diff(lsa.singular_values_) <= 0.0)
    assert np.max(np.abs(lsa.components_ @ lsa.components_.T - np.eye(k))) < 1e-12
    # The comparison that means something is against the best rank-k approximation
    # *in a random k-dimensional subspace* -- a Gaussian matrix has a random scale
    # and beating it says nothing. By symmetry E[QQᵀ] = (k/d) I over Haar draws, so
    # a typical random subspace keeps exactly k/d of the energy.
    rng = np.random.default_rng(3)
    rand_res = []
    for _ in range(200):
        Q = np.linalg.qr(rng.standard_normal((d, k)))[0]
        rand_res.append(float(np.linalg.norm(Xw - Xw @ Q @ Q.T)))
    mean_rand = float(np.mean(rand_res))
    predicted = math.sqrt((1.0 - k / d) * float(np.sum(s ** 2)))
    assert abs(mean_rand - predicted) < 0.05 * predicted, (mean_rand, predicted)
    assert min(rand_res) > resid, (min(rand_res), resid)
    assert X.mean() > 0.5                                    # the origin is not centred
    # documents_ = U S, and projecting a row of X is a row of U S: the two agree
    agree = float(np.max(np.abs(lsa.transform(X) - lsa.documents_)))
    assert agree < 1e-12, agree
    print(
        f"lsa: the rank-{k} truncation of {n}×{d} counts leaves a residual of {resid:.4f}, "
        f"exactly the tail √(Σσ²) = {math.sqrt(tail):.4f}, and keeps "
        f"{lsa.explained_variance_ratio_:.0%} of the energy; the best of 200 random rank-"
        f"{k} subspaces averages {mean_rand:.2f}, exactly the √((1−k/d)Σσ²) = "
        f"{predicted:.2f} symmetry predicts, and none beats it — Eckart–Young. Not PCA: the "
        f"mean count is {X.mean():.2f}, and it is deliberately not subtracted. And "
        f"`transform(X)` reproduces `documents_` to {agree:.0e}, because both are `X_w Vᵀ`"
    )

    # --- 2. the randomized solver: what power iteration buys ----------------
    # The sketch is n × (k + p) and the small SVD is (k + p) × d, so one pass over
    # X is the cost. Without power iteration the sketch is biased towards the top
    # of the spectrum and the subspace is wrong at the first decimal; each step
    # squares the ratio between wanted and unwanted singular values.
    zero = LSA(k, variant="randomized", n_iter=0, seed=0).fit(X)
    mid = LSA(k, variant="randomized", n_iter=2, seed=0).fit(X)
    rnd = LSA(k, variant="randomized", n_iter=8, seed=0).fit(X)
    P = lsa.components_.T @ lsa.components_                  # exact projector
    def gap(F):
        return float(np.linalg.norm(F.components_.T @ F.components_ - P))
    assert gap(zero) > gap(mid) > gap(rnd) > 0.0
    assert gap(rnd) < 1e-8, gap(rnd)
    assert np.max(np.abs(rnd.singular_values_ - s[:k])) < 1e-10
    # documents_ is U S, whose columns are only defined up to sign; the
    # reconstruction is not, so that is what gets compared
    assert np.max(np.abs(rnd.inverse_transform(rnd.documents_) - recon)) < 1e-7
    print(
        f"lsa: the randomized solver on a {n}×{k + rnd.oversample} sketch converges to the "
        f"exact subspace by power iteration alone — {gap(zero):.0e} with none, "
        f"{gap(mid):.0e} with two, {gap(rnd):.0e} with eight — and then matches the "
        f"singular values to {np.max(np.abs(rnd.singular_values_ - s[:k])):.0e}. The cost "
        f"is a handful of products against a narrow matrix; the answer is the same one"
    )

    # --- 3. the weightings, checked by hand on a 3 × 4 matrix ---------------
    # Two of the four terms appear in all three documents: column 1 as (1, 1, 3)
    # and column 3 as (3, 3, 3). Both have df = n = 3, so tf-idf deletes both --
    # it only asks whether a term appears, never how. log-entropy deletes only
    # column 3: its distribution over documents is flat, maximum entropy, so it
    # cannot tell you which document you are reading, while (1, 1, 3) is
    # concentrated and does. That difference is the whole reason the variant
    # exists.
    T = _tiny()
    tfidf = LSA(2, variant="tfidf_lsa", norm=False).fit(T)
    ent = LSA(2, variant="log_entropy", norm=False).fit(T)
    idf, g = tfidf.weights_, ent.weights_
    assert abs(idf[1]) < 1e-12 and abs(idf[3]) < 1e-12
    assert abs(idf[0] - math.log(3 / 2)) < 1e-12
    assert abs(g[3]) < 1e-12, g
    H1 = -2.0 * 0.2 * math.log(0.2) - 0.6 * math.log(0.6)      # term 1's (1,1,3)
    assert abs(g[1] - (1.0 - H1 / math.log(3))) < 1e-12, g[1]
    # both are zero up to rounding, not exactly: the entropy path goes through logs
    assert np.count_nonzero(idf < 1e-12) == 2 and np.count_nonzero(g < 1e-12) == 1
    # and `transform` re-derives the weighting from the input rather than reusing
    # a stored matrix: the hand-computed weighted matrix must reproduce it
    Wn = np.log1p(T) * g                       # norm=False, so no row normalisation
    Wt = T * idf
    assert np.allclose(ent.transform(T), Wn @ ent.components_.T)
    assert np.allclose(tfidf.transform(T), Wt @ tfidf.components_.T)
    print(
        f"lsa: on a 3×4 matrix whose fourth column is (3, 3, 3), tf-idf zeroes two of four "
        f"terms — the second and fourth, the ones with df = n — while log-entropy zeroes "
        f"only the fourth (g = {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}, {g[3]:.3f}); a term in "
        f"every document can still be informative, and only the entropy weighting tells "
        f"the two cases apart"
    )

    # --- 4. what the weighting is worth, measured --------------------------
    # 240 documents, four topics, forty words each: at the natural rank everything
    # works, so cut below it. At k = 2 the raw-count model gets the topic right for
    # just over half the held-out documents, because a long document's counts
    # dominate the cosine; tf-idf lifts that to 90%. log-entropy lands between the
    # two here -- it keeps terms tf-idf throws away, which is not free.
    def accuracy(F, n_train: int = 200) -> float:
        Q = F.transform(X[n_train:])
        D = F.documents_
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), EPS)
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), EPS)
        hit = np.argmax(Q @ D.T, axis=1)
        return float(np.mean(labels[:n_train][hit] == labels[n_train:]))

    acc = {(k_, v): accuracy(LSA(k_, variant=v).fit(X[:200]))
           for k_ in (2, 4)
           for v in ("truncated_svd", "tfidf_lsa", "log_entropy", "randomized")}
    assert acc[4, "truncated_svd"] > 0.95                 # at the natural rank, saturated
    assert acc[2, "truncated_svd"] < 0.65
    assert acc[2, "tfidf_lsa"] > 0.85 and acc[2, "log_entropy"] > 0.65
    assert abs(acc[2, "randomized"] - acc[2, "truncated_svd"]) < 1e-12
    # idf is the *fitted* weight, so a one-document query still has a direction.
    # Recomputing it on the query gives log(1/1) = 0 for every term it contains,
    # hence the zero vector, hence no cosine to rank with.
    q = LSA(k, variant="tfidf_lsa").fit(X[:200]).transform(X[205:206])
    assert float(np.linalg.norm(q)) > 0.5, q
    assert np.all(_term_weights(X[205:206], "tfidf") == 0.0)
    print(
        f"lsa: cut below the natural rank and the weighting is the whole model — at k = 2 "
        f"raw counts put the right topic first for {acc[2, 'truncated_svd']:.0%} of 40 "
        f"held-out documents, tf-idf for {acc[2, 'tfidf_lsa']:.0%} and log-entropy for "
        f"{acc[2, 'log_entropy']:.0%}; at k = 4 all three are perfect and nothing is left "
        f"to choose. The randomized solver reproduces the raw-count number exactly "
        f"({acc[2, 'randomized']:.0%}), because it is the same model — the solver was "
        f"never the interesting decision. And a single-document query still embeds "
        f"(‖z‖ = {np.linalg.norm(q):.2f}), because idf is the fitted one: recomputed on "
        f"that one document every term would get log(1/1) = 0 and the query would be the "
        f"zero vector"
    )

    # --- 5. what the decomposition does not identify -----------------------
    # U S Vᵀ is unique when the singular values are distinct, but the pair is
    # not: for any signature D, (U S D)(D V)ᵀ = U S Vᵀ. So the sign of every
    # component is arbitrary, two fits can differ by one, and the reconstruction
    # cannot. Cosine similarity is sign-invariant, which is why retrieval does not
    # care and why interpreting a component does.
    first = LSA(k, variant="randomized", n_iter=8, seed=0).fit(X[:200])
    other = LSA(k, variant="randomized", n_iter=8, seed=1).fit(X[:200])
    flips = np.sign(np.sum(first.components_ * other.components_, axis=1))
    assert np.all(flips != 0.0)
    signed = float(np.max(np.abs(other.components_ * flips[:, None] - first.components_)))
    assert signed < 1e-6, signed

    # a component and its negation satisfy every equation the component satisfies:
    # both are exact eigenvectors of XᵀX, so nothing in the model chooses between them
    gram = first._weighted(X[:200]).T @ first._weighted(X[:200])
    lam, V = first.singular_values_ ** 2, first.components_
    resid = max(float(np.max(np.abs(V @ gram - lam[:, None] * V))),
                float(np.max(np.abs(-V @ gram - lam[:, None] * -V))))
    assert resid < 1e-7, resid        # the randomized fit's own accuracy

    # and the ambiguity cancels: flip the two sides together and nothing observable moves
    sig = np.array([-1.0, 1.0, 1.0, -1.0])
    rec = first.inverse_transform(first.documents_)
    flipped = (first.documents_ * sig) @ (first.components_ * sig[:, None])
    assert np.max(np.abs(flipped - rec)) < 1e-12
    assert accuracy(first) == accuracy(other)
    print(
        f"lsa: the singular values are distinct ({lsa.singular_values_[0]:.1f} against "
        f"{s[4]:.1f} for the first discarded one), so the decomposition is unique up to one "
        f"sign per component — and a sign is still an exact eigenvector, both residuals "
        f"{resid:.0e}. Flipping two of four signs in `components_` with the matching columns "
        f"of `documents_` leaves the reconstruction identical to 1e-12 and the retrieval "
        f"accuracy unchanged, so the sign is a convention; two seeds agree to {signed:.0e} "
        f"once theirs are aligned"
    )

    print("lsa demo passed")


if __name__ == "__main__":
    main()
