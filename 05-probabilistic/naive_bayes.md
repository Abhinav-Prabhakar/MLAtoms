# Naive Bayes

`naive_bayes.py` · Probabilistic models · Tier 1 · 7 variants

## Intuition

Assume the features are independent once you know the class. Then Bayes'
theorem collapses from an integral over the joint density into a sum over
one-dimensional pieces, and a classifier becomes a table of counts.

$$p(c \mid x) \;\propto\; p(c)\,\prod_{j=1}^{d} p(x_j \mid c) \qquad (1)$$

The assumption is false for almost every dataset — words in a document are
correlated, pixels in an image are correlated, symptoms in a diagnosis are
correlated. The classifier works anyway, and understanding *why* is the whole
atom: the decision only reads the **ranking** of the classes, and ranking
survives perturbations that would destroy a calibrated posterior.

The practical consequence is a model with `O(kd)` parameters estimated by
average-and-divide, fitting in one pass over the data, that beats far more
sophisticated models on small, high-dimensional problems. The price is
confidence: `predict_proba` on correlated features is overconfident by orders
of magnitude, so the numbers it prints are ranks pretending to be
probabilities.

## Variants

| Variant | Likelihood `p(x_j | c)` | Use when |
|:--|:--|:--|
| `gaussian` | `N(μ_cj, σ²_cj)` | continuous features, roughly bell-shaped per class |
| `multinomial` | `θ_cj^{x_j}` on counts | bag-of-words, any non-negative count matrix |
| `bernoulli` | `θ_cj^{x_j}(1−θ_cj)^{1−x_j}` | presence/absence matters, frequency does not |
| `categorical` | one table per feature over its own categories | ordinal-free discrete columns (colour, region, token id) |
| `complement` | weights from the **other** classes | text with severely skewed class sizes |
| `tfidf` | multinomial on `x·idf` | long documents, common words dominating |
| `semi_supervised` | multinomial fitted by EM on `y = -1` rows | a handful of labels, a pile of unlabelled rows |

The four count variants differ only in *what is counted*: token totals
(`multinomial`), document frequencies (`bernoulli`), per-feature category
histograms (`categorical`). `complement` and `tfidf` reuse the multinomial
counts with a different table or a different weighting — they are not new
estimators, and the code makes that literal.

## Derivation

### From Bayes to a log-linear score

The evidence `p(x)` in (1) does not depend on `c`, so it cancels in the argmax.
Taking logs turns the product into the score

$$s_c(x) \;=\; \log \pi_c \;+\; \sum_{j=1}^{d} \log p(x_j \mid c),
\qquad \pi_c = p(c). \qquad (2)$$

For the multinomial likelihood `p(x \mid c) = \prod_j \theta_{cj}^{x_j}` this is

$$s_c(x) \;=\; \log \pi_c \;+\; \sum_j x_j \log \theta_{cj}. \qquad (3)$$

Two classes, subtract: `s_1 − s_0 = \log(\pi_1/\pi_0) + \sum_j x_j \log(\theta_{1j}/\theta_{0j})`.
The model is a **linear** classifier in `x` with weights `w_j = \log(\theta_{1j}/\theta_{0j})`.
Everything that follows is a story about how those weights are estimated.

### Why a wrong model still wins

Three separate arguments, in increasing strength.

*The bias is common-mode.* If a feature is duplicated, the model doubles its
evidence — but it doubles it for every class, so if the duplication is roughly
balanced the argmax does not move. Zero-one loss is insensitive to any error
that shifts all classes together.

*The ranks, not the values, are estimated.* The argmax reads only the sign of
the pairwise margin. An estimate of `w_j` can be off by 40% and still give the
right sign; a calibrated posterior is a much stronger requirement than a
correct ordering.

*Bias can help.* Domingos and Pazzani (1997) show that naive Bayes is optimal
under zero-one loss whenever the bias it introduces pushes the estimate in the
same direction as the truth. In the `n \ll d` regime, a high-bias,
low-variance estimator beats a low-bias, high-variance one, and naive Bayes
has the lowest variance of any classifier that estimates one parameter per
(class, feature).

The same argument explains the failure mode: with `n \gg d` and genuinely
correlated features, logistic regression recovers the right weights and naive
Bayes does not, and the gap is real.

### Smoothing is a Dirichlet prior

The multinomial M-step is `\theta_{cj} = N_{cj} / N_{c\cdot}` with
`N_{cj} = \sum_{i: y_i = c} x_{ij}`. If a feature never occurs in a class, its
probability is exactly zero, and one zero is enough to make the whole product
zero and the log-likelihood `-\infty` — the class can then never win, no
matter the evidence. The fix is to put a Dirichlet prior on each row and take
the posterior mean:

$$\theta_{cj} \;=\; \frac{N_{cj} + \alpha}{N_{c\cdot} + \alpha V},
\qquad \alpha = 1 \text{ is Laplace, } \alpha < 1 \text{ Lidstone.} \qquad (4)$$

`α` is a pseudo-count, not a regularisation knob bolted on: the posterior is
exact for the conjugate model, and `α` is the number of imaginary documents in
which every word appears equally often. The denominator uses the number of
*distinct values* `V` for that feature — `d` for a shared vocabulary, `K_j` for
a categorical column.

### Gaussian: one mean and one variance per (class, feature)

$$\log p(x_j \mid c) = -\tfrac12\log\bigl(2\pi\sigma^2_{cj}\bigr)
- \frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}. \qquad (5)$$

The MLEs are the per-class column mean and variance (`ddof = 0`). A feature
that is constant within a class gives `σ² = 0`, hence an infinite density on a
measure-zero set and `-\infty` everywhere else. `var_smoothing` adds
`ε = \varepsilon \cdot \max \sigma^2` to every variance, exactly as `α` adds a
pseudo-count: the continuous version of the same degeneracy.

When every class shares one variance matrix, the quadratic terms cancel and
(5) reduces to a linear discriminant — naive Bayes with shared variances *is*
LDA, which is the bridge to `discriminant_analysis.py`.

### Bernoulli: documents, not tokens

The Bernoulli likelihood treats `x` as a bit vector, `x_j \in \{0,1\}`, and
models presence explicitly:

$$\log p(x \mid c) = \sum_j \Bigl[x_j \log \theta_{cj}
+ (1 - x_j)\log(1 - \theta_{cj})\Bigr],
\qquad \theta_{cj} = \frac{\#\{i : y_i = c,\ x_{ij} > 0\}}{N_c}. \qquad (6)$$

The second term is the part everyone forgets. It penalises a class whose
vocabulary does *not* contain the observed words, and it is the reason
Bernoulli NB can differ sharply from multinomial on short documents: ten
occurrences of "free" count once. The denominator carries `+2α` instead of
`+αV` because each feature has exactly two outcomes.

### Complement: estimate the class you are not predicting

Rennie et al. (2003) observed that multinomial NB is dominated by the prior on
skewed text corpora, and proposed estimating each class's weights from the
data of *all the other classes*. With `C_{cj} = \sum_{c' \neq c} N_{c'j}`:

$$\tilde\theta_{cj} = \frac{C_{cj} + \alpha}{C_{c\cdot} + \alpha d},
\qquad w_{cj} = -\log \tilde\theta_{cj},
\qquad s_c(x) = \sum_j x_j w_{cj}. \qquad (7)$$

Three things are happening at once:

1. **Weights are large where the class is *absent*.** `-\log` of a small
   complement probability is a big positive number, so a word used by every
   class *except* `c` is strong evidence *for* `c`. That is the complement
   trick: sifting the evidence for the others and negating it.
2. **The prior is dropped.** Re-adding `\log \pi_c` would restore exactly the
   term the complement weights were built to cancel; the demo asserts the
   drop, and reports the trade it buys (minority recall up, accuracy down).
3. **With two classes it is not a new classifier.** The complement of `c` is
   the single class `1−c`, so `\tilde\theta_{cj} = \theta_{1-c,j}` for any
   `α`, and

$$s_c(x) = -\sum_j x_j \log \theta_{1-c,j} = -\,s^{\text{mult}}_{1-c}(x)\big|_{\text{no prior}}. \qquad (8)$$

   The argmax of a negated, reversed score is the same argmax: for `k = 2`,
   complement NB *is* multinomial NB with a uniform prior. It only becomes a
   different classifier from three classes up, which is why the demo proves
   the identity and then measures the difference at `k = 3`.

### tfidf: a reweighting, not an estimator

`\mathrm{idf}_j = \log\frac{1+n}{1+\mathrm{df}_j} + 1` is estimated on the
training corpus, then the multinomial rule runs on `x \odot \mathrm{idf}`.
The smooth form is deliberate: the unsmoothed `\log(n/\mathrm{df})` is exactly
zero for a term in every document, which silently deletes it, while the `+1`
floor keeps it at weight 1 — neutral rather than dead. Because idf is a column
scaling, `\mathrm{tfidf}` changes the *features*, not the model, and the demo
asserts equality with the multinomial fit on the scaled matrix.

### Semi-supervised: EM over soft counts

With labelled rows `L` and unlabelled rows `U`, EM treats the missing labels as
latent. The E-step computes responsibilities `r_{ic} = p(c \mid x_i)` for
`i \in U`; the M-step re-estimates the tables with fractional rows:

$$\hat N_{cj} = \sum_{i \in L} \mathbb{1}[y_i = c]\,x_{ij}
+ \sum_{i \in U} r_{ic}\, x_{ij}. \qquad (9)$$

Each iteration increases the marginal likelihood of the unlabelled data:

$$\mathcal{L} = \sum_{i \in U} \log \sum_c \pi_c \prod_j \theta_{cj}^{x_{ij}}. \qquad (10)$$

Monotonicity is a theorem, not a heuristic — which is why the demo asserts it
rather than merely printing it. The catch is that EM finds *a* local optimum:
if the labelled seed splits the data badly, the unlabelled rows harden that
split, and the accuracy can end up below the supervised-only fit.

## Code walkthrough

`logsumexp` is the numerical backbone. Every posterior in the file is computed
as `exp(jll − logsumexp(jll))`, because `jll` for a 2000-word document is a
sum of 2000 terms of order `log(10^{-4})` and exponentiating it directly
underflows to zero:

```python
def logsumexp(a: np.ndarray) -> np.ndarray:
    """Row-wise `log Σ exp(a)`, shifted by the row max so nothing overflows."""
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))
```

`_fit_counts` builds one table and then hands it to three different rules.
The docstring is the design: *one count table, three rules*:

```python
    def _fit_counts(self, X: np.ndarray, y: np.ndarray) -> None:
        """Multinomial / Bernoulli / complement: one count table, three rules."""
        k, d = self.classes_.size, X.shape[1]
        self.feature_count_ = np.zeros((k, d))
        for i, c in enumerate(self.classes_):
            self.feature_count_[i] = X[y == c].sum(axis=0)
```

`tfidf` is the only variant that touches `X` before counting, and it must, or
the counts and the idf would be computed from different matrices:

```python
        if self.variant == "tfidf":
            # idf is a property of the corpus, so it is estimated before any
            # smoothing: a term in every document becomes uninformative
            df = (X > 0).sum(axis=0)
            self.idf_ = np.log((1.0 + X.shape[0]) / (1.0 + df)) + 1.0
            X = X * self.idf_
            for i, c in enumerate(self.classes_):
                self.feature_count_[i] = X[y == c].sum(axis=0)
```

Bernoulli rebuilds the table as **document frequencies** — the line that a
naive implementation gets wrong by leaving the token sums in place:

```python
        if self.variant == "bernoulli":
            # the count is documents containing the feature, not token totals
            self.feature_count_ = np.zeros((k, d))
            for i, c in enumerate(self.classes_):
                self.feature_count_[i] = (X[y == c] > 0).sum(axis=0)
            self.class_count_ = np.array([np.sum(y == c) for c in self.classes_],
                                         dtype=float)
            self.theta_ = ((self.feature_count_ + self.alpha)
                           / (self.class_count_[:, None] + 2.0 * self.alpha))
```

Complement subtracts the class's own counts from the column totals, so one
subtraction produces the whole complement table. `total` is computed once
outside the loop; recomputing it per class is the usual `O(k)` mistake:

```python
        elif self.variant == "complement":
            total = self.feature_count_.sum(axis=0)
            comp = total - self.feature_count_
            den = comp.sum(axis=1, keepdims=True) + self.alpha * d
            theta = (comp + self.alpha) / den
            self.weight_ = -np.log(theta)          # Rennie et al.'s rule
```

The multinomial fallthrough is equation (4) line for line, with `α·d` where a
per-feature category count would be if this were the categorical variant:

```python
        else:
            den = self.feature_count_.sum(axis=1, keepdims=True) + self.alpha * d
            self.feature_log_prob_ = np.log(
                (self.feature_count_ + self.alpha) / den)
```

Scoring is a single switch on the same variant string. Gaussian is the only
branch that needs a log-determinant term — dropping it is harmless for
*classification* (it is constant across `x`) but wrong for `predict_proba`
when the class variances differ:

```python
        if self.variant == "gaussian":
            for i in range(self.classes_.size):
                d2 = (X - self.theta_[i]) ** 2 / self.var_[i]
                out[:, i] = (-0.5 * (np.log(2 * np.pi * self.var_[i]) + d2)).sum(1)
```

Bernoulli binarises **at predict time**, not at fit time. That one line is the
entire semantic difference from multinomial, and it is asserted in the demo by
feeding a count of 1 and a count of 10 and checking the scores match:

```python
        elif self.variant == "bernoulli":
            # presence, not frequency: a count of ten is the same as a count of
            # one, which is the whole difference from the multinomial rule
            Xb = (X > 0).astype(float)
            for i in range(self.classes_.size):
                th = np.clip(self.theta_[i], 1e-12, 1 - 1e-12)
                out[:, i] = (Xb * np.log(th) + (1.0 - Xb) * np.log(1.0 - th)).sum(1)
```

The `np.clip` is not defensive noise: with `α = 0` a feature never seen in a
class yields `θ = 0`, and `log(0)` would poison the whole row with `-inf`.

Complement scores with the weight matrix and then **returns before the prior is
added**. Every other variant falls through to the `+ class_log_prior_`:

```python
        elif self.variant == "complement":
            out = X @ self.weight_.T
        ...
        if self.variant == "complement":
            # Rennie et al. drop the class prior: the complement weights already
            # price the imbalance, and re-adding the prior undoes the correction
            return out
        return out + self.class_log_prior_
```

EM is the same `joint_log_likelihood` doing double duty. The E-step is the
responsibility matrix; the M-step is `resp.T @ Xu`, one matmul that adds the
fractional rows to the counts. The stopping rule reads the likelihood, not the
counts, because the counts never stop creeping:

```python
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
```

`_set_theta` is shared by EM init, the M-step and the final refit, so there is
exactly one place where `α` enters the emission table. `np.maximum(prior, ...)`
guards the case where a class ends with zero soft mass:

```python
    def _set_theta(self, counts: np.ndarray, prior: np.ndarray) -> None:
        den = counts.sum(axis=1, keepdims=True) + self.alpha * self.n_features_
        self.feature_log_prob_ = np.log((counts + self.alpha) / den)
        self.class_log_prior_ = np.log(np.maximum(prior, 1e-12) / prior.sum())
```

`predict_proba` is normalised in log space; `predict` is the argmax of the same
matrix, so the two can never disagree:

```python
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return np.exp(jll - logsumexp(jll))
```

## Complexity

Let `n` be rows, `d` features, `k` classes, `m` non-zeros per document.

| Stage | Gaussian | Counts | EM |
|:--|:--|:--|:--|
| `fit` | `O(nkd)` | `O(nm)` | `O(T·n_u·k·d)` |
| `predict` | `O(nkd)` | `O(nmk)` | `O(nmk + nk)` |
| memory | `O(kd)` | `O(kd)` | `O(kd + n_u k)` |

The `fit` cost is the reason this atom exists: for text `m \ll d` and the whole
training pass is one sparse column sum. Prediction is a single `k × d` matmul
per row — the same asymptotics as a linear model, at a fraction of the
constants, because there is no optimisation loop. EM multiplies `fit` by the
iteration count and is the only variant where `T` appears.

## Failure modes

**A zero count survives smoothing.** With `α = 0` any feature absent from a
class yields `log 0 = -\infty`, and the class is eliminated forever. The
diagnostic is `np.isneginf(variant.feature_log_prob_).any()`; the fix is
`α ≥ 1e-9`.

**Correlated features make the posterior a lie.** Duplicating a column doubles
its log-odds, so `predict_proba` reports `0.9999` on data where the true
posterior is `0.6`. The predictions stay good, which is exactly what makes the
bug invisible. Diagnose with a reliability curve and a log-loss comparison; the
rank is trustworthy, the number is not.

**The prior vetoes the minority before the evidence is read.** With `π_0 = 0.97`
a minority row needs `\log(0.97/0.01) \approx 4.6` nats of evidence just to
tie, and a five-token document cannot supply it — all minority rows are called
majority and accuracy still looks like 97%. Diagnose with per-class recall;
the fix is `complement`, or setting the priors by hand.

**EM hardens a bad seed.** Semi-supervised EM maximises a non-convex
likelihood; if the labelled rows split the clusters wrongly, the unlabelled
responsibilities saturate and the labels never recover. Diagnose by printing
`max_c r_{ic}` — values at `1.0 − 10^{-12}` for most rows mean the
responsibilities have collapsed; compare against the supervised-only accuracy,
which the demo does.

## Applications

Text classification is the natural home: spam, topic tags, sentiment, language
identification — anything where `n` is small, `d` is 10⁴ and the classes are
roughly balanced. Complement NB for the imbalanced version of the same task
(one rare category against many common ones) and for long-tailed tag
prediction.

Outside text it is the baseline every tabular model should beat: Gaussian NB
fits in microseconds and its confusion matrix tells you whether the classes are
separable by means. In streaming settings the count tables are `O(kd)`
sufficient statistics, so `fit` can be made incremental with a running sum and
no loss of exactness.

Where it is a trap: any application that needs *calibrated* probabilities
(medical risk scores, pricing), any dataset with strongly correlated features
where a discriminative model is affordable, and any regression-shaped problem —
naive Bayes is a classifier, and the Gaussian variant regresses nothing.

## References

- P. Domingos and M. Pazzani, *On the optimality of the simple Bayesian
  classifier under zero-one loss*, Machine Learning 29 (1997) — the argument
  that a wrong posterior can still give the right decision.
- H. Zhang, *The optimality of naive Bayes*, FLAIRS 2003 — the same result
  stated as a ranking theorem, with the conditions made precise.
- J. D. M. Rennie, L. Shih, J. Teevan and D. R. Karger, *Tackling the poor
  assumptions of naive Bayes text classifiers*, ICML 2003 — complement NB,
  and the empirical study of why the prior dominates skewed corpora.
- K. Nigam, A. McCallum, S. Thrun and T. Mitchell, *Text classification from
  labeled and unlabeled documents using EM*, Machine Learning 39 (2000) — the
  semi-supervised variant, including the failure mode.
- C. D. Manning, P. Raghavan and H. Schütze, *Introduction to Information
  Retrieval*, chapters 13 and 11 — the multinomial and Bernoulli models with
  the numbers.
- `sklearn.naive_bayes` — the reference implementation this file mirrors,
  including the smooth idf and the single-class prior rule.
