# Probabilistic LSA

`plsa.py` · Latent structure · Tier 3 · 3 variants

## Intuition

A document × term matrix of counts is the most common object in text analysis, and
it has two obvious low-rank stories. The first is linear algebra: truncate an SVD
and call the result latent semantic analysis. That works, and it produces factors
with negative entries that are not probabilities of anything. The second is
probabilistic: say that each document is a **mixture over topics**, each topic is
a **distribution over words**, and the word distribution of a document is the
mixture.

```
                 terms (V = 60)
              ┌───────┬───────┬───────┐
   d  ┌───────┼───────┼───────┼───────┤
   o  │       │  ···  │       │       │   each row: counts of 80 tokens
   c  │       │       │  ···  │       │   each row is a mixture of 3 blocks
   s  │       │       │       │  ···  │
      └───────┴───────┴───────┴───────┘
                 z=1     z=2     z=3
```

The generative story is two lines. To produce one word of document $d$: draw a
topic $z$ with probability $p(z\mid d)$, then draw a word with probability
$p(w\mid z)$. Marginalise the topic you did not observe and the document's word
distribution is

$$p(w\mid d) \;=\; \sum_{z=1}^{Z} p(w\mid z)\,p(z\mid d). \qquad (1)$$

Everything the model knows is in two stochastic matrices: `doc_topic_` $(D\times
Z)$, how much of each topic each document uses, and `topic_word_` $(Z\times V)$,
what each topic says. The counts are the only observed thing, and the likelihood
is the only objective.

That objective is where the trouble is. Maximum likelihood has no term that
charges for parameters, so on the demo's corpus — 120 documents, 80 tokens each,
60 terms in three disjoint blocks, generated from exactly three topics — asking
for **twenty** topics produces a *higher* likelihood per token than the three that
generated the data ($-3.47257$ against $-3.58817$) while recovering the planted
topics at cosine $0.5978$ instead of $0.9971$. The extra topics are not
describing anything; they are absorbing sampling noise, and the likelihood cannot
tell the difference. That is the whole content of pLSA: a clean, identifiable,
closed-form latent-variable model with no complexity control whatsoever.

## Variants

| Variant | What changes | Update | Cost per iteration |
|:--|:--|:--|:--|
| `em` | nothing — the model of (1) | exact E and M steps | $O(DVZ)$ |
| `tempered_em` | responsibilities raised to $\beta < 1$, annealed to 1 | same M-step, tempered E-step | $O(DVZ)$ |
| `map_plsa` | Dirichlet prior on each topic's word distribution | $\alpha-1$ added to every count | $O(DVZ)$ |

**`em`** is the reference: alternate the exact posterior over topics with the
closed-form maximisation. On the demo it recovers the three planted topics at
mean cosine $0.9971$ in 388 iterations, with the log-likelihood rising on every
one of the 387 steps. That is EM's guarantee and the demo measures it rather than
assuming it.

**`tempered_em`** is a *schedule*, not a different estimator: with $\beta = 1$ it
must reproduce `em` exactly, and the demo checks that the two final likelihoods
agree to $0.0$. The reason to anneal is that the likelihood surface has local
optima and EM walks downhill from wherever it starts; flattening the
responsibilities early widens the basins, then raising the temperature sharpens
them back to the true posterior. The demo's measurement is unambiguous at six
topics: with $\beta = 0.8$ climbing linearly to 1 over the first 150 iterations it
beats plain `em` on **all four** starts, by $15.6$ nats on average and $22.0$ at
best, from the same initialisation.

The same code at $\beta = 0.4$ is a catastrophe, and the demo shows exactly why.
Responsibilities that flat hand every topic nearly the same counts; identical rows
produce identical responsibilities; the map preserves that symmetry exactly, so
the run stops in **23 iterations** at a fixed point where all three topics agree
to $6.5\times10^{-6}$ in every entry and every document is $0.3334$ of each. The
likelihood falls on all 22 steps, to $-38896.3$ against $-34446.5$ for `em`. The
temperature is a hyper-parameter with a cliff on the low side.

**`map_plsa`** adds a symmetric $\mathrm{Dirichlet}(\alpha)$ prior to every
topic's word distribution, which adds $\alpha-1$ to each count before
normalising. At $\alpha = 1$ that is exactly `em` (measured: $0.0$ difference in
likelihood), because concentration 1 is the uniform prior and the uniform prior is
no prior. Above 1 it smooths: the demo's mean topic entropy climbs
$2.928 \to 3.367 \to 3.749$ nats for $\alpha = 1, 5, 20$, the smallest entry of
`topic_word_` goes from exactly $0$ to $1.1\times10^{-3}$ and $4.2\times10^{-3}$,
and the training likelihood per token pays for it, $-3.4771 \to -3.5103 \to
-3.6041$. That price buys the one thing `em` cannot do: when the demo hands the
model ten tokens of a term that appears in **no** training document, `em`'s
perplexity is `inf`, because one unseen word multiplies the likelihood by zero and
no arithmetic recovers it, while $\alpha = 5$ gives $768.2$.

## Derivation

### 1. The likelihood

Documents are independent, words within a document are independent given the
topics, so with $n(d,w)$ the count of term $w$ in document $d$ the likelihood is

$$\mathcal{L} \;=\; \prod_{d=1}^{D}\prod_{w=1}^{V}
\Big[\sum_{z} p(w\mid z)\,p(z\mid d)\Big]^{n(d,w)}, \qquad (2)$$

a product over observed tokens, not over cells. The constraint set is two
simplices per document, $\sum_z p(z\mid d) = 1$, and one per topic, $\sum_w
p(w\mid z) = 1$, and (2) is not concave on it: the sum inside the log is where the
non-convexity lives.

### 2. The complete-data view

Introduce a distribution $q_{dwz}$ over which topic produced word $w$ in document
$d$. Multiplying and dividing inside the log by $q$ and applying Jensen gives, for
**any** $q$ on the simplex,

$$\log \mathcal{L} \;\ge\; \sum_{d,w} n(d,w)\sum_z q_{dwz}
\Big[\log p(z\mid d) + \log p(w\mid z) - \log q_{dwz}\Big], \qquad (3)$$

with equality exactly when $q$ is the posterior. Writing $H(q)$ for the entropy
term, the bound is the usual decomposition

$$\log \mathcal{L} \;=\; \underbrace{\mathbb{E}_q\big[\log p(d,w,z)\big] + H(q)}_{\text{ELBO}}
\;+\; \mathrm{KL}\big(q \,\Vert\, p(z\mid d,w)\big), \qquad (4)$$

and (4) is the entire algorithm in one line: the E-step sets the KL term to zero
by choosing the posterior, the M-step maximises the ELBO, and neither step can
decrease the left-hand side. EM is the coordinate ascent this identity forces.

### 3. The E-step

The posterior of a topic given one observed word is Bayes applied to (1):

$$q_{dwz} \;=\; \frac{p(z\mid d)\,p(w\mid z)}
{\sum_{z'} p(z'\mid d)\,p(w\mid z')}, \qquad (5)$$

and it depends on the cell $(d,w)$ only through the word — every occurrence of
the same word in the same document has the same responsibility. That is why the
whole E-step is $D\times V\times Z$ numbers rather than a loop over 9600 tokens.

### 4. The M-step

Maximise the ELBO part of (3) subject to the two simplex constraints. With
multipliers $\lambda_d$ and $\mu_z$,

$$\frac{\partial}{\partial p(w\mid z)}\Big[\sum_{d,w} n(d,w) q_{dwz}\log p(w\mid z)
- \mu_z\Big(\sum_w p(w\mid z)-1\Big)\Big] = 0, \qquad (6)$$

so $p(w\mid z) \propto \sum_d n(d,w)q_{dwz} = N_{zw}$, and normalising,

$$p(w\mid z) \;=\; \frac{N_{zw}}{\sum_{w'} N_{zw'}},
\qquad N_{zw} = \sum_{d} n(d,w)\, q_{dwz}. \qquad (7)$$

The document side is the same computation with the roles of $d$ and $w$
exchanged, and since $\sum_z q_{dwz} = 1$ it simplifies to

$$p(z\mid d) \;=\; \frac{\sum_w n(d,w) q_{dwz}}{n(d,\cdot)}. \qquad (8)$$

Both are weighted averages of the data: (7) is "a topic's word distribution is
the counts it takes responsibility for", (8) is "a document's mixture is the
fraction of its tokens each topic explains". Nothing is solved iteratively inside
a step; the only iteration is the alternation.

### 5. What monotonicity does not say

(4) guarantees that the likelihood never falls, and that is the only guarantee.
It says nothing about which stationary point is reached, and it says nothing
about the number of topics. The demo's 20-topic fit is a better optimum of a
bigger model on the same data, so the guarantee is satisfied and the answer is
worse. Model selection has to come from somewhere else — a prior, a held-out
criterion that actually penalises parameters, or a hypothesis about the data.

### 6. Tempering

The tempered E-step raises the posterior to a power,

$$q^{(\beta)}_{dwz} \;=\; \frac{\big[p(z\mid d)\,p(w\mid z)\big]^{\beta}}
{\sum_{z'}\big[p(z'\mid d)\,p(w\mid z')\big]^{\beta}}, \qquad (9)$$

which is the stationary point of the tilted free energy

$$F_\beta(\theta) \;=\; -\frac{1}{\beta}\sum_{d,w} n(d,w)
\log \sum_{z}\big[p(z\mid d)\,p(w\mid z)\big]^{\beta}. \qquad (10)$$

At $\beta = 1$, (10) is the negative log-likelihood and (9) is (5). At $\beta \to
0$ the expression inside the log flattens towards the uniform average, the
responsibilities stop discriminating, and the M-step sees every topic receiving
the same counts. The M-step in the code is unchanged — only the $q$ that feeds it
is tempered — which is why $\beta = 1$ is an exact identity rather than an
approximation.

There is a fixed point at the bottom of that road. If `topic_word_` ever has
identical rows, then $q_{dwz} = 1/Z$ for every $(d,w,z)$ at any $\beta$, the
M-step returns identical rows, and the symmetry is preserved forever. The demo's
$\beta = 0.4$ run reaches it to $6.5\times10^{-6}$ and stops, because there is
nothing left to move.

### 7. The MAP variant

Put an independent $\mathrm{Dirichlet}(\alpha)$ prior on each row of
`topic_word_`. The penalised objective is the ELBO plus

$$\log p\big(p(\cdot\mid z)\big) \;=\; (\alpha-1)\sum_w \log p(w\mid z) + \text{const},
\qquad (11)$$

so the same Lagrangian as (6) gives

$$p(w\mid z) \;=\; \frac{N_{zw} + \alpha - 1}{\sum_{w'}\big(N_{zw'} + \alpha - 1\big)},
\qquad (12)$$

which is (7) with every count incremented. For $\alpha > 1$ that is add-$\alpha-1$
smoothing: no entry can be zero, because a zero would cost infinite prior
density. For $\alpha < 1$ the increment is negative and the closed form can go
below zero, so implementations clamp at zero — the code does, and the clamp is a
projection onto the simplex rather than the exact MAP.

The consequence that matters is visible only out of sample. Under (7), a term
with $N_{zw} = 0$ for every $z$ gets probability exactly zero, so any document
containing it has likelihood zero and perplexity infinite. Under (12) with
$\alpha > 1$ every term gets a floor, and "unlikely" becomes expressible.

### 8. Perplexity, and folding in

The held-out measure is per-token perplexity,

$$\mathrm{ppl}(X) \;=\; \exp\!\left(-\frac{\log p(X)}{\sum_{d,w} n(d,w)}\right),
\qquad (13)$$

the geometric mean inverse probability the model assigns to a token. For a new
document the topics are fixed and only the mixture is free, so folding in solves

$$\hat\theta \;=\; \arg\min_{\theta \in \Delta_Z}
-\sum_w n(d^*,w)\log \sum_z \theta_z\, p(w\mid z), \qquad (14)$$

by EM on that one vector. Note what (14) is: a **maximum-likelihood** fit of $Z$
free parameters to one document, not a posterior. It has enough freedom to
compensate for a bad topic matrix, which is why the demo's held-out perplexity
also prefers twenty topics — $34.20$ against $36.24$ for three — even though
recovery has collapsed. Fold-in does not measure the model; it measures the model
plus a per-document refit.

### 9. Symmetry, and what is identified

Relabel the topics and nothing observable changes:

$$p(z\mid d) \mapsto p(\sigma(z)\mid d), \qquad p(w\mid z) \mapsto p(w\mid \sigma(z)),
\qquad (15)$$

for any permutation $\sigma$. That is the only exact ambiguity in the model: pLSA
is a mixture of conditionally independent unigrams, and generically its parameters
are determined up to (15) — which is why a recovered topic can be compared to a
planted one at all, and why the demo's cosine is a meaningful score rather than a
score of one representative of a rotation family, as in `factor_analysis` or
Tucker. What is *not* unique is the optimisation: many local optima exist, and
which one you reach depends on the start, which is the practical cost of the
non-convexity in (2).

## Code walkthrough

`PLSA(n_topics=3, *, variant="em", n_iter=200, tol=1e-7, beta=0.5, alpha=2.0,
seed=0)` with `fit(X)`, `transform(X, n_iter=100)`, `perplexity(X)` and the
attributes `topic_word_`, `doc_topic_`, `log_likelihood_`, `n_iter_`. `X` is a
`(documents, terms)` array of non-negative counts.

### 1. A row-stochastic matrix with a dead-row guard

```python
    s = P.sum(axis=1, keepdims=True)
    dead = s[:, 0] < EPS
    if np.any(dead):
        return np.where(dead[:, None], 1.0 / P.shape[1], P / np.maximum(s, EPS))
    return P / s
```

Every update in the file ends in this function, because both factors are
stochastic matrices. The guard is not decoration: with $\alpha < 1$ in (12) the
clamped update can zero a whole row, and a topic that has died then produces
`0/0` in the next E-step and a `nan` two functions away from the cause. Replacing
the dead row by the uniform distribution keeps the arithmetic finite and makes
the death visible in the output instead of in a traceback.

### 2. The E-step, in logs

```python
    log_joint = (np.log(np.maximum(doc_topic, EPS))[:, None, :]
                 + np.log(np.maximum(topic_word, EPS)).T[None, :, :])
    log_joint *= beta
    top = log_joint.max(axis=2, keepdims=True)
    q = np.exp(log_joint - top)
    return q / np.maximum(q.sum(axis=2, keepdims=True), EPS)
```

(5) and (9) in one expression. The broadcast produces the full $(D,V,Z)$ array of
log joints; the `max` subtraction is the log-sum-exp shift, without which a
probability of $10^{-300}$ underflows to zero and a document of rare words gets
`nan` responsibilities. The `beta` line is the entire difference between the
plain and tempered variants, applied before the shift so the shift stays exact.

### 3. The M-step, and the prior

```python
            counts_word = np.einsum("dw,dwz->zw", X, q)
            if self.variant == "map_plsa":
                # Dirichlet(alpha) prior on each p(·|z): the MAP solution adds
                # alpha − 1 to every count, which is smoothing above 1 and
                # sparsification below it
                counts_word = np.maximum(counts_word + self.alpha - 1.0, 0.0)
            self.topic_word_ = _normalise_rows(counts_word)
            self.doc_topic_ = _normalise_rows(np.einsum("dw,dwz->dz", X, q))
```

(7), (8) and (12). Both M-steps are the same contraction of the same $(D,V,Z)$
responsibility array against the counts, one summing over documents and one over
terms — that symmetry is why pLSA is cheap, and it is also why the array itself is
the cost. The clamp is where the code stops being exact MAP: for $\alpha < 1$ the
true MAP has zeros in it, the clamp forces that, and the projection is not the
maximiser of (11) but the nearest point in the simplex.

### 4. The temperature schedule

```python
        if self.variant != "tempered_em":
            return 1.0
        span = max(1, self.n_iter // 2)
        return self.beta + (1.0 - self.beta) * min(1.0, it / span)
```

(10)'s schedule: linear from `beta` to 1 over the first half of the iteration
budget, then exactly 1. Deterministic, which is what makes the demo reproducible,
and monotone, which is what makes $\beta = 1$ reachable in finite time. The half
is a choice: too long and the run spends its budget at a temperature that is not
the model, too short and the flattened phase is over before the basins have
merged. The demo's $\beta = 0.4$ failure is not a schedule bug — it is the
temperature itself, and a slower ramp does not fix it.

### 5. The likelihood, exactly

```python
    with np.errstate(divide="ignore"):
        log_joint = (np.log(doc_topic)[:, None, :]
                     + np.log(topic_word).T[None, :, :])
    top = log_joint.max(axis=2, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        log_sum = top[:, :, 0] + np.log(np.exp(log_joint - top).sum(axis=2))
    nz = X > 0
    return float(np.sum(X[nz] * log_sum[nz]))
```

(2) without a floor. `topic_word` may legitimately contain zeros under `em`, and
the whole point of the `map_plsa` demo is that those zeros make the held-out
likelihood infinite: a simulated $-\infty$ would turn the headline into a finite
number. `top = np.where(np.isfinite(top), top, 0.0)` handles the all-$-\infty$
column (a word no topic can produce) so that the shift does not produce `nan`,
and `log_sum` is then $-\infty$ as it should be. Only the nonzero cells
contribute, which is both correct and faster on a sparse count matrix.

### 6. Folding in

```python
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rng = np.random.default_rng(self.seed)
        doc_topic = _normalise_rows(rng.random((X.shape[0], self.n_topics)))
        for _ in range(n_iter):
            q = _e_step(X, self.topic_word_, doc_topic)
            doc_topic = _normalise_rows(np.einsum("dw,dwz->dz", X, q))
        return doc_topic
```

(14) by the same two steps as `fit`, with `topic_word_` frozen. `n_iter=100` and
no convergence test is honest rather than lazy: the objective in (14) is a
$Z$-parameter problem for each document, 100 sweeps is far past what it needs,
and a per-document stopping rule would make the reported perplexity depend on a
tolerance. Note that `atleast_2d` is what lets `perplexity` accept a single
document, which the demo uses to score one probe.

### 7. The stopping rule

```python
            if (len(self.log_likelihood_) > 1
                    and abs(self.log_likelihood_[-1] - self.log_likelihood_[-2])
                    < self.tol * max(abs(self.log_likelihood_[-1]), 1.0)):
                self.n_iter_ = it + 1
                break
```

A **relative** change, because the likelihood scales with corpus size and an
absolute threshold would mean a different stopping point on every dataset. The
`max(..., 1.0)` guards the scale when the likelihood is small. This rule is also
the reason the $\beta = 0.4$ run ends at 23 iterations: a symmetric fixed point
has a flat likelihood, so the relative change falls below tolerance while the fit
is still terrible — convergence of the objective is not convergence to the
answer.

### 8. Scoring a recovery that is only identified up to a permutation

```python
    ln = learned / np.maximum(np.linalg.norm(learned, axis=1, keepdims=True), EPS)
    tn = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), EPS)
    return float(np.mean(np.max(ln @ tn.T, axis=1)))
```

(15) is the only ambiguity, so the score is: normalise both sides, take the best
cosine for each learned topic, average. Max over the truth handles the
permutation; mean over the learned topics means one perfect topic out of three
scores $1/3$, not $1$. The absolute value is deliberately absent — pLSA's topics
are non-negative distributions, so a negative cosine is a genuinely bad match and
should not be folded back.

## Complexity

$D$ documents, $V$ terms, $Z$ topics, $T$ iterations, $N = \sum_{d,w} n(d,w)$
tokens.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| E-step | $O(DVZ)$ | $O(DVZ)$ | the responsibility array dominates |
| M-step | $O(DVZ)$ | $O(VZ + DZ)$ | two contractions of the same array |
| one iteration | $O(DVZ)$ | $O(DVZ)$ | independent of $N$ |
| full `fit` | $O(T\,DVZ)$ | $O(DVZ)$ | $T$ decided by the stopping rule |
| fold-in, one document | $O(VZ)$ per sweep | $O(VZ)$ | 100 sweeps |
| parameters | $DZ$ (document mixtures) + $ZV$ (topics) | — | see the failure modes |
| full demo | — | — | 20 fits, 2.7 s |

Two facts decide whether this runs on your data. The first is the memory of the
responsibility array: $D V Z$ floats, which is 43 thousand for the demo and $10^9$
for a million documents over a thousand-term vocabulary at ten topics. Every
serious implementation either streams over documents or keeps the responsibilities
implicit — the M-step only needs their contractions, so a chunked
$O(VZ)$-memory version is a few lines away, and the demo's version is the clear
one because the demo's matrices are small.

The second is the parameter count in the table's last row: $DZ + ZV$, which grows
with the number of documents. That is not an accounting detail, it is the reason
pLSA overfits small corpora and the reason LDA replaces the per-document
parameters with a distribution over them. A pLSA model of a corpus is not a
generative model of the corpus's documents; it is a model with one free vector per
document, fitted jointly.

Convergence is linear, with a rate that degrades as the topics overlap — the
fraction of missing information in the E-step is large exactly when a word is
compatible with several topics. The demo's three disjoint blocks are the easy
case and still take 388 iterations at a relative tolerance of $10^{-9}$; real
text with overlapping topics takes several times that, which is why practical
fits use loose tolerances and compare restarts rather than waiting for the last
digit.

## Failure modes

**1. The likelihood cannot count parameters.** The demo's 20-topic fit is better
by every number the objective can see — likelihood per token $-3.47257$ against
$-3.58817$ — and recovers the three planted topics at $0.5978$ instead of
$0.9971$. The extra topics split the noise, and the model is not penalised for
it. Choose $Z$ by a prior, by a held-out criterion that does not refit per
document, or by what you claim the data is.

**2. Held-out perplexity is not that criterion.** On 60 training documents and 60
held out, the demo gets $36.24$ with three topics and $34.20$ with twenty: the
usual defence prefers the wrong model. Fold-in (14) fits a fresh $Z$-vector to
each held-out document, and that freedom compensates for topic estimates that
have nothing to do with the truth. This is a known weakness of the metric, not a
bug in this implementation — and it is measured here rather than assumed.

**3. Zero is absorbing.** Under `em`, a term that no topic takes responsibility
for keeps probability exactly zero forever: the E-step multiplies by it, the
M-step gets no counts back, and the state reproduces itself. Out of sample this
is a single word turning perplexity into `inf`. The prior in `map_plsa` is the
fix, and it is a real cost — $-3.4771 \to -3.5103 \to -3.6041$ per token as
$\alpha$ rises through 5 and 20.

**4. The wrong temperature collapses the model to one topic.** At $\beta = 0.4$
the demo's run dies in 23 iterations with all three topics agreeing to
$6.5\times10^{-6}$ and every document at $0.3334$ of each. This is a fixed point,
not a transient: identical rows give uniform responsibilities, which give
identical rows. If a tempered run converges suspiciously fast to a suspiciously
bad likelihood, check the rows for agreement before blaming the data.

**5. Local optima are the norm, and the start is the only exploration.** Two
restarts from different seeds land on different likelihoods; the demo's
`tempered_em` gain of $15.6$ nats on average is precisely the size of that
spread. Report the best of several runs *and* the spread, and compare recovered
topics across runs — the likelihood alone will not tell you whether you are in
the same basin.

**6. The mixture is per document, so "the model of the corpus" is a fiction.**
`doc_topic_` has $DZ$ entries, one vector per document, all fitted. pLSA does not
define a distribution over documents, so you cannot sample a new one from it;
`transform` is a fit, not a draw. This is the structural difference from LDA and
the reason pLSA's perplexity numbers are not comparable to LDA's.

**7. The responsibilities are $D\times V\times Z$.** Memory, not time, is the
constraint at scale, and it fails as an allocation far from the line that chose
the algorithm. Chunk over documents; the contractions do not need the whole array
at once.

**8. Overlapping topics make the cosine score look worse than the model is.** The
demo's planted topics are disjoint blocks, so recovery is a clean number. Real
topics overlap, a fitted topic can be a blend of two true ones, and the mean-best
cosine then understates the fit. Compare topic *co-occurrence* structure or
downstream task performance when the ground truth is not known.

## Applications

- **Information retrieval.** pLSA was introduced as probabilistic latent semantic
  indexing: query and document are projected into the same topic simplex and
  matched there, which handles synonymy and polysemy that a term-matching
  baseline cannot. The original paper's retrieval results are still the cleanest
  demonstration of why the probabilistic formulation is worth the trouble.
- **Text classification and clustering with few labels.** Topics as features,
  `doc_topic_` as the representation. The per-document mixture is a compact
  summary that a small classifier can use, and it is fitted unsupervised.
- **Image and audio annotation.** Any co-occurrence matrix of "documents" ×
  "terms" works: images × visual words, tracks × audio tokens, users × items. The
  model does not know what a document is, which is a strength and a trap.
- **A baseline for topic models.** Every LDA paper compares against pLSA, and
  reproducing it is the fastest way to check a new inference routine: the
  likelihood is exact in closed form at every iteration, so a wrong E-step shows
  up immediately as a non-monotone likelihood.
- **Teaching the ELBO.** Equations (3)–(4) are the cleanest example of the
  variational decomposition in the field: a discrete latent variable, an exact
  posterior, and a coordinate ascent whose two steps both have closed forms.
- **Segmentation and change detection.** Fit topics to windows of text and track
  `doc_topic_` across the windows; the mixture vector is a low-dimensional signal
  and a shift in it is a topic change.

Where it is a trap: choosing $Z$ by likelihood; trusting held-out perplexity with
fold-in; reporting topics when the model has $DZ$ parameters and the corpus is
small; using a tempered schedule with a low starting temperature; and expecting
pLSA to generate documents, which it structurally cannot.

## References

- **Hofmann (1999),** *Probabilistic latent semantic indexing* — the aspect model,
  the EM algorithm, and the retrieval results that made the probabilistic reading
  of LSA credible.
- **Hofmann (2001),** *Unsupervised learning by probabilistic latent semantic
  analysis* — the journal treatment, including the tempered variant and the
  discussion of local optima that the annealing is meant to address.
- **Dempster, Laird & Rubin (1977),** *Maximum likelihood from incomplete data via
  the EM algorithm* — the monotonicity result behind equation (4), and the
  reason a rising likelihood is a guarantee about the objective only.
- **Neal & Hinton (1998),** *A view of the EM algorithm that justifies incremental,
  sparse, and other variants* — the ELBO reading of (3)–(4), and the framework
  that makes MAP-EM and tempered EM look like the same algorithm with a different
  bound.
- **Ueda & Nakano (1998),** *Deterministic annealing EM algorithm* — the tempered
  E-step of (9), with the free-energy schedule of (10) and the conditions under
  which it escapes local optima.
- **Blei, Ng & Jordan (2003),** *Latent Dirichlet allocation* — the model that
  replaces pLSA's per-document parameters with a Dirichlet prior, which is the
  fix for failure modes 1, 2 and 6.
- **Lee & Seung (1999),** *Learning the parts of objects by non-negative matrix
  factorization* — the multiplicative-update view of the same factorisation, and
  the comparison that makes pLSA's probabilistic constraints visible as
  normalisation rather than non-negativity.
- **Gaussier & Goutte (2005),** *Relation between PLSA and NMF and implications*
  — the exact correspondence, including the equivalence of the PLSA likelihood
  and a specific NMF divergence up to a constant.
- **Wallach, Murray, Salakhutdinov & Mimno (2009),** *Evaluation methods for topic
  models* — the systematic account of why held-out perplexity with fold-in
  rewards the wrong models, which is failure mode 2 measured on real corpora.
- **Arora, Ge, Halpern, Mimno, Moitra, Sontag, Wu & Zhu (2013),** *A practical
  algorithm for topic modeling with provable guarantees* — identifiability of the
  topic-word matrix from moments, which is what makes equation (15) the *only*
  ambiguity rather than one of many.
- **Blei & Lafferty (2009),** *Topic models* — the survey that places pLSA, LDA
  and their descendants in one framework, and the honest account of what each
  buys and costs.
