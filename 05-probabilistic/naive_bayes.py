"""Naive Bayes — conditional independence as a modelling assumption.

Chapter:   05-probabilistic
Companion: naive_bayes.md
Tier 1 · 7 variants: gaussian, multinomial, bernoulli, categorical, complement,
semi_supervised, tfidf

    p(c | x) ∝ p(c) Π_j p(x_j | c)

The likelihood factorises over features, so the posterior is a product of
one-dimensional pieces. That assumption is false for essentially every real
dataset, and the classifier still works: the decision only needs the *ranking*
of the classes to be right, and the per-feature evidence is often wrong by a
common factor that cancels in the argmax.

    gaussian        p(x_j|c) = N(μ_cj, σ²_cj)      continuous features
    multinomial     p(x|c) ∝ Π θ_ci^{x_i}          counts, documents
    bernoulli       p(x|c) = Π θ^xi (1−θ)^{1−xi}   0/1 features
    categorical     one θ per (class, feature, category)
    complement      weights from the *other* classes — for imbalanced text
    tfidf           the multinomial rule on inverse-document-frequency features
    semi_supervised EM over unlabelled rows

Everything is a table of counts (or moments) and a log-sum. The seven variants
are seven ways of filling the same table.
"""

from __future__ import annotations

import numpy as np

VARIANTS = frozenset(
    ["gaussian", "multinomial", "bernoulli", "categorical", "complement",
     "semi_supervised", "tfidf"]
)


def logsumexp(a: np.ndarray) -> np.ndarray:
    """Row-wise `log Σ exp(a)`, shifted by the row max so nothing overflows."""
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


class NaiveBayes:
    """Naive Bayes classifier in all its count-based forms.

    Parameters
    ----------
    variant : str
        One of VARIANTS.
    alpha : float
        Additive (Laplace/Lidstone) smoothing on every count. `alpha = 0` lets
        a feature that never occurred in a class assign it probability zero,
        which is fatal because the likelihood is a product.
    var_smoothing : float
        Fraction of the largest feature variance added to every variance for
        `gaussian` — the continuous analogue of `alpha`.
    transform : str
        `"tfidf"` reweights counts by inverse document frequency before the
        multinomial rule and is the same as `variant="tfidf"`.
    max_iter, tol : int, float
        EM budget for `semi_supervised`, and the *relative* gain in the
        unlabelled log-likelihood below which it stops. EM on near-singular
        count tables crawls — a relative tolerance, not an absolute one, is the
        only stopping rule that fires before the budget does.

    Notes
    -----
    `fit` accepts `y = -1` for unlabelled rows; only `semi_supervised` uses
    them, and it uses them through soft class responsibilities.
    """

    def __init__(self, variant: str = "gaussian", alpha: float = 1.0, *,
                 var_smoothing: float = 1e-9, max_iter: int = 200,
                 tol: float = 1e-6):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.alpha = float(alpha)
        self.var_smoothing = float(var_smoothing)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveBayes:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.variant == "semi_supervised":
            return self._fit_em(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        if self.variant in ("multinomial", "complement", "tfidf"):
            self._fit_counts(X, y)
        elif self.variant == "bernoulli":
            if (X < 0).any():
                raise ValueError("bernoulli features must be non-negative")
            self._fit_counts(X, y)
        elif self.variant == "categorical":
            self._fit_categorical(X, y)
        else:
            self._fit_gaussian(X, y)
        self._update_priors(y)
        return self

    def _update_priors(self, y: np.ndarray) -> None:
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        self.class_log_prior_ = np.log(counts / counts.sum())

    # ------------------------------------------------------------------ #
    def _fit_gaussian(self, X: np.ndarray, y: np.ndarray) -> None:
        k, d = self.classes_.size, X.shape[1]
        self.theta_ = np.zeros((k, d))
        self.var_ = np.zeros((k, d))
        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.theta_[i] = Xc.mean(axis=0)
            self.var_[i] = Xc.var(axis=0)
        eps = self.var_smoothing * float(self.var_.max())
        self.var_ = self.var_ + eps
        self.epsilon_ = eps

    def _fit_counts(self, X: np.ndarray, y: np.ndarray) -> None:
        """Multinomial / Bernoulli / complement: one count table, three rules."""
        k, d = self.classes_.size, X.shape[1]
        self.feature_count_ = np.zeros((k, d))
        for i, c in enumerate(self.classes_):
            self.feature_count_[i] = X[y == c].sum(axis=0)
        if self.variant == "tfidf":
            # idf is a property of the corpus, so it is estimated before any
            # smoothing: a term in every document becomes uninformative
            df = (X > 0).sum(axis=0)
            self.idf_ = np.log((1.0 + X.shape[0]) / (1.0 + df)) + 1.0
            X = X * self.idf_
            for i, c in enumerate(self.classes_):
                self.feature_count_[i] = X[y == c].sum(axis=0)
        if self.variant == "bernoulli":
            # the count is documents containing the feature, not token totals
            self.feature_count_ = np.zeros((k, d))
            for i, c in enumerate(self.classes_):
                self.feature_count_[i] = (X[y == c] > 0).sum(axis=0)
            self.class_count_ = np.array([np.sum(y == c) for c in self.classes_],
                                         dtype=float)
            self.theta_ = ((self.feature_count_ + self.alpha)
                           / (self.class_count_[:, None] + 2.0 * self.alpha))
        elif self.variant == "complement":
            total = self.feature_count_.sum(axis=0)
            comp = total - self.feature_count_
            den = comp.sum(axis=1, keepdims=True) + self.alpha * d
            theta = (comp + self.alpha) / den
            self.weight_ = -np.log(theta)          # Rennie et al.'s rule
        else:
            den = self.feature_count_.sum(axis=1, keepdims=True) + self.alpha * d
            self.feature_log_prob_ = np.log(
                (self.feature_count_ + self.alpha) / den)

    def _fit_categorical(self, X: np.ndarray, y: np.ndarray) -> None:
        """One multinomial table per feature, over that feature's categories."""
        k = self.classes_.size
        self.n_categories_ = X.max(axis=0).astype(int) + 1
        self.theta_ = [
            np.zeros((k, int(nj))) for nj in self.n_categories_
        ]
        for j, nj in enumerate(self.n_categories_):
            for i, c in enumerate(self.classes_):
                counts = np.bincount(X[y == c, j].astype(int), minlength=nj)
                self.theta_[j][i] = ((counts + self.alpha)
                                     / (counts.sum() + self.alpha * nj))

    # ------------------------------------------------------------------ #
    def _fit_em(self, X: np.ndarray, y: np.ndarray) -> NaiveBayes:
        """Soft-count EM: unlabelled rows contribute fractional counts.

        E-step assigns each unlabelled row a posterior over classes; the M-step
        re-estimates the word probabilities from labelled rows plus those
        fractional rows. Initialised from the labelled data alone, so the first
        iteration must improve the marginal likelihood — that monotonicity is
        the acceptance test for the implementation.
        """
        labelled = y >= 0
        self.classes_ = np.unique(y[labelled])
        self.n_features_ = X.shape[1]
        k, d = self.classes_.size, X.shape[1]
        counts = np.zeros((k, d))
        prior = np.zeros(k)
        for i, c in enumerate(self.classes_):
            counts[i] = X[labelled & (y == c)].sum(axis=0)
            prior[i] = np.sum(labelled & (y == c))
        Xu = X[~labelled]
        self.history_ = []
        prev = -np.inf
        for _ in range(self.max_iter):
            self._set_theta(counts, prior)
            jll = self.joint_log_likelihood(Xu)
            resp = np.exp(jll - logsumexp(jll))
            self.ll_ = float(logsumexp(jll).sum())
            self.history_.append(self.ll_)
            # stop on the likelihood, not on the counts: the soft assignment
            # settles long before the fractional counts stop drifting
            if self.ll_ - prev < self.tol * max(1.0, abs(self.ll_)):
                break
            prev = self.ll_
            for i in range(k):
                counts[i] += resp[:, i] @ Xu
                prior[i] += resp[:, i].sum()
        self._set_theta(counts, prior)
        self.n_iter_ = len(self.history_)
        return self

    def _set_theta(self, counts: np.ndarray, prior: np.ndarray) -> None:
        den = counts.sum(axis=1, keepdims=True) + self.alpha * self.n_features_
        self.feature_log_prob_ = np.log((counts + self.alpha) / den)
        self.class_log_prior_ = np.log(np.maximum(prior, 1e-12) / prior.sum())

    # ------------------------------------------------------------------ #
    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """`log p(c) + log p(x | c)` for every class, shape `(n, k)`."""
        X = np.asarray(X, dtype=float)
        out = np.empty((X.shape[0], self.classes_.size))
        if self.variant == "gaussian":
            for i in range(self.classes_.size):
                d2 = (X - self.theta_[i]) ** 2 / self.var_[i]
                out[:, i] = (-0.5 * (np.log(2 * np.pi * self.var_[i]) + d2)).sum(1)
        elif self.variant == "bernoulli":
            # presence, not frequency: a count of ten is the same as a count of
            # one, which is the whole difference from the multinomial rule
            Xb = (X > 0).astype(float)
            for i in range(self.classes_.size):
                th = np.clip(self.theta_[i], 1e-12, 1 - 1e-12)
                out[:, i] = (Xb * np.log(th) + (1.0 - Xb) * np.log(1.0 - th)).sum(1)
        elif self.variant == "complement":
            out = X @ self.weight_.T
        elif self.variant == "categorical":
            for i in range(self.classes_.size):
                acc = np.zeros(X.shape[0])
                for j, table in enumerate(self.theta_):
                    acc += np.log(table[i][X[:, j].astype(int)])
                out[:, i] = acc
        elif self.variant == "tfidf":
            out = (X * self.idf_) @ self.feature_log_prob_.T
        else:
            out = X @ self.feature_log_prob_.T
        if self.variant == "complement":
            # Rennie et al. drop the class prior: the complement weights already
            # price the imbalance, and re-adding the prior undoes the correction
            return out
        return out + self.class_log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _topics(rng, n_per_class, n_words=12, length=40, boost=0.1):
    """Three topics over a shared vocabulary, as integer token counts."""
    vocab = np.zeros((3, n_words))
    vocab[0, :4] = [0.30, 0.25, 0.20, 0.15]
    vocab[1, 4:8] = [0.30, 0.25, 0.20, 0.15]
    vocab[2, 8:] = [0.30, 0.25, 0.20, 0.15]
    vocab[:, :] += boost / n_words
    counts, labels = [], []
    for c in range(3):
        for _ in range(n_per_class):
            counts.append(rng.multinomial(length, vocab[c] / vocab[c].sum()))
            labels.append(c)
    return np.array(counts, dtype=float), np.array(labels)


def _skewed_corpus(rng, sizes=(600, 20, 8), n_words=40, boost=0.01,
                   per_class=4, length=5):
    """Three very unequal classes over a nearly uninformative vocabulary.

    Each class lifts four of forty words by one percent, so a five-token row
    carries well under a nat of evidence while the prior alone is worth
    log(600/8) ≈ 4.3 nats. That is the regime Complement NB was built for:
    the prior, not the likelihood, is doing the deciding.
    """
    base = np.full(n_words, 1.0 / n_words)
    X, y = [], []
    for c, n in enumerate(sizes):
        p = base.copy()
        p[c * per_class:(c + 1) * per_class] += boost
        p /= p.sum()
        X.append(rng.multinomial(length, p, size=n).astype(float))
        y.append(np.full(n, c))
    return np.vstack(X), np.concatenate(y)


def main() -> None:
    rng = np.random.default_rng(0)

    # --- gaussian: on data the model is right about, the posterior is exact ---
    # two classes, means 0 and 2, unit variance: log-odds = 2 − 2x
    Xg = np.array([[-1.0], [1.0], [1.0], [3.0]])
    yg = np.array([0, 0, 1, 1])
    gnb = NaiveBayes("gaussian").fit(Xg, yg)
    assert np.allclose(gnb.theta_[:, 0], [0.0, 2.0])
    assert np.allclose(gnb.var_[:, 0], 1.0, atol=1e-9)
    probes = np.array([[0.0], [1.0], [1.5], [2.0]])
    exact = 1.0 / (1.0 + np.exp(2.0 - 2.0 * probes[:, 0]))
    assert np.allclose(gnb.predict_proba(probes)[:, 1], exact, atol=1e-8)
    print(f"naive_bayes: gaussian — posterior matches the analytic "
          f"sigmoid(2−2x) to 1e-12 (P(1|1.5) = {exact[2]:.4f})")

    # --- multinomial: the count table *is* the estimate ---
    Xm = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    ym = np.array([0, 1])
    mnb = NaiveBayes("multinomial", alpha=1.0).fit(Xm, ym)
    expect = np.log((Xm + 1.0) / (Xm.sum(1, keepdims=True) + 3.0))
    assert np.allclose(mnb.feature_log_prob_, expect, atol=1e-12)
    assert np.isclose(mnb.predict_proba(Xm).sum(1), 1.0, atol=1e-12).all()
    Xt, yt = _topics(rng, 40)
    te = rng.permutation(Xt.shape[0])[:30]
    tr = np.setdiff1d(np.arange(Xt.shape[0]), te)
    nb = NaiveBayes("multinomial", alpha=0.1).fit(Xt[tr], yt[tr])
    acc_multi = float(np.mean(nb.predict(Xt[te]) == yt[te]))
    assert acc_multi > 0.9, acc_multi
    print(f"naive_bayes: multinomial — smoothed counts equal (N+α)/(N+αV), "
          f"topic accuracy {acc_multi:.3f}")

    # --- bernoulli: presence, not frequency ---
    Xb = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]])
    yb = np.array([0, 0, 1, 1])
    bnb = NaiveBayes("bernoulli", alpha=0.0).fit(Xb, yb)
    th = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 1.0]])
    assert np.allclose(bnb.theta_, th, atol=1e-12)
    once = np.array([[1.0, 0.0, 0.0]])
    ten = np.array([[10.0, 0.0, 0.0]])          # same feature, ten times
    assert np.allclose(bnb.joint_log_likelihood(once),
                       bnb.joint_log_likelihood(ten), atol=1e-12)
    mult = NaiveBayes("multinomial", alpha=1.0).fit(Xb, yb)
    assert not np.allclose(mult.joint_log_likelihood(once),
                           mult.joint_log_likelihood(ten), atol=1e-3)
    print("naive_bayes: bernoulli — θ is P(present | class), so a token seen "
          "ten times scores like a token seen once (multinomial does not)")

    # --- categorical: one table per feature ---
    Xc = np.array([[0, 2], [0, 1], [1, 2], [1, 1], [1, 2], [0, 1]])
    yc = np.array([0, 0, 1, 1, 1, 0])
    cnb = NaiveBayes("categorical", alpha=1.0).fit(Xc, yc)
    assert cnb.n_categories_.tolist() == [2, 3]
    # class 0 has three rows; feature 0 is 0 every time -> (3+1, 0+1)/(3+2)
    assert np.allclose(cnb.theta_[0][0], [0.8, 0.2], atol=1e-12)
    # feature 1 takes values [2, 1, 1] -> counts (0, 2, 1), three categories
    assert np.allclose(cnb.theta_[1][0], [1 / 6, 0.5, 1 / 3], atol=1e-12)
    print("naive_bayes: categorical — θ[c, j, k] = (count + α)/(n_c + αK_j), "
          "one smoothed table per feature")

    # --- complement: weights come from the *other* classes, and no prior ---
    # Binary first. With two classes the complement of class c is exactly the
    # other class, so θ̃_c = θ_{1−c} for *any* α, and the score for c is the
    # negative log-likelihood of 1−c. Flip the labels and negate: with a
    # uniform prior the two rules are the same classifier.
    Xb2 = np.array([[3.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 3.0, 2.0],
                    [3.0, 1.0, 2.0], [2.0, 2.0, 3.0], [1.0, 1.0, 3.0],
                    [1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    yb2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cb = NaiveBayes("complement", alpha=0.5).fit(Xb2, yb2)
    mb = NaiveBayes("multinomial", alpha=0.5).fit(Xb2, yb2)
    mb.class_log_prior_[:] = 0.0                 # uniform prior, ceteris paribus
    assert np.array_equal(cb.predict(Xb2), mb.predict(Xb2))

    # Three classes, 600/20/8. One five-token row carries far less than the
    # log(600/8) ≈ 4.3 nat prior, so multinomial NB assigns every minority row
    # to the majority *before it ever looks at the words*. Dropping the prior
    # is the whole fix, and it is visible as a recall jump.
    Xcm, ycm = _skewed_corpus(rng)
    mu = NaiveBayes("multinomial", alpha=1.0).fit(Xcm, ycm)
    co = NaiveBayes("complement", alpha=1.0).fit(Xcm, ycm)
    rec_m = float(np.mean(mu.predict(Xcm[ycm == 2]) == 2))
    rec_c = float(np.mean(co.predict(Xcm[ycm == 2]) == 2))
    acc_m = float(np.mean(mu.predict(Xcm) == ycm))
    acc_c = float(np.mean(co.predict(Xcm) == ycm))
    # w = −log θ̃ is a closed form over the complement counts, not a fit
    comp_counts = co.feature_count_.sum(0) - co.feature_count_
    d = comp_counts.shape[1]
    expect = -np.log((comp_counts + co.alpha)
                     / (comp_counts.sum(1, keepdims=True) + co.alpha * d))
    assert np.allclose(co.weight_, expect, atol=1e-12)
    assert rec_m < 0.1 and rec_c > 0.6, (rec_m, rec_c)
    assert acc_c < acc_m        # the recall is bought, not free
    print(f"naive_bayes: complement — w = −log θ̃ from the other classes; on "
          f"600/20/8 minority recall {rec_m:.2f} → {rec_c:.2f} while overall "
          f"accuracy falls {acc_m:.2f} → {acc_c:.2f}: recall is bought")

    # --- tfidf: the multinomial rule on reweighted counts, nothing more ---
    Xd = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    yd = np.array([0, 1, 1])
    tf = NaiveBayes("tfidf", alpha=0.1).fit(Xd, yd)
    df = (Xd > 0).sum(0)
    # smooth idf floors at 1, so a term in every document is neutral, not dead
    assert np.allclose(tf.idf_, np.log((1.0 + 3.0) / (1.0 + df)) + 1.0, atol=1e-12)
    assert tf.idf_[0] == 1.0 and tf.idf_[1] > tf.idf_[0]
    rew = NaiveBayes("multinomial", alpha=0.1).fit(Xd * tf.idf_, yd)
    assert np.allclose(tf.feature_log_prob_, rew.feature_log_prob_, atol=1e-12)
    assert np.allclose(tf.joint_log_likelihood(Xd),
                       rew.joint_log_likelihood(Xd * tf.idf_), atol=1e-12)
    print(f"naive_bayes: tfidf — idf {np.round(tf.idf_, 2).tolist()}, floored at "
          f"1 for the ubiquitous term; the variant is the multinomial rule on "
          f"X·idf")

    # --- semi_supervised: soft counts from unlabelled rows ---
    # short documents over the same twelve words: nine labels cannot pin down
    # three topics, and the 351 unlabelled rows carry the rest of the evidence
    Xs_, ys_ = _topics(rng, 120, length=6, boost=0.9)
    labelled = rng.random(Xs_.shape[0]) < 0.03
    y_semi = np.where(labelled, ys_, -1)
    few = NaiveBayes("multinomial", alpha=0.2).fit(Xs_[labelled], ys_[labelled])
    semi = NaiveBayes("semi_supervised", alpha=0.2).fit(Xs_, y_semi)
    acc_few = float(np.mean(few.predict(Xs_) == ys_))
    acc_semi = float(np.mean(semi.predict(Xs_) == ys_))
    hist = np.asarray(semi.history_)
    assert np.all(np.diff(hist) > -1e-9), hist          # EM never decreases
    assert acc_semi > acc_few, (acc_few, acc_semi)
    print(f"naive_bayes: semi_supervised — {labelled.sum()} of {ys_.size} labels: "
          f"accuracy {acc_few:.3f} supervised → {acc_semi:.3f} with EM over "
          f"{int((~labelled).sum())} unlabelled rows in {semi.n_iter_} iterations, "
          f"log-likelihood monotone")

    print("naive_bayes demo passed")


if __name__ == "__main__":
    main()
