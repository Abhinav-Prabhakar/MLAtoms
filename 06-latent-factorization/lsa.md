# Latent semantic analysis

`lsa.py` · Latent structure · Tier 2 · 4 variants

## Intuition

A corpus is a matrix. Rows are documents, columns are terms, entries are counts.
It is the least exotic data structure in machine learning, and it has one
awkward property: it is **sparse and literal**. Two documents about the same
subject can share no words at all.

```
   X (240 documents × 60 terms, 31% non-zero)

   "the car needs a new engine"    →  the 3, car 1, need 1, new 1, engine 1
   "my automobile requires a motor" →  my 1, automobile 1, require 1, motor 1

   XᵀX for those two rows: 0.     No shared term, no similarity, no match.
```

LSA's answer is that the two sentences *do* share terms indirectly — `car` and
`automobile` both co-occur with `drive`, `engine`, `motorway` — and that
indirect sharing shows up as soon as the matrix is approximated by a low-rank
one. Terms that never appear together get mapped to nearby points because they
appear with the same third terms.

$$\underbrace{X}_{n \times d} \;\approx\; \underbrace{U_k S_k}_{n \times k}\, \underbrace{V_k^\top}_{k \times d}, \qquad k \ll \min(n, d). \qquad (1)$$

Each of the $k$ rows of $V_k^\top$ is a **topic**: a weighted list of terms. Each
row of $U_k S_k$ says how much of each topic a document contains. A document is
then a point in $\mathbb{R}^k$, and similarity is a dot product there.

The important thing about this atom is that (1) is the easy part. The SVD is
exact, unique, and available in one library call; there is nothing to tune and
nothing to get wrong. What *is* interesting is which matrix you hand it. Raw
counts, $\log(1 + \mathrm{tf})$, tf-idf and log-entropy all produce a different
$X$ and therefore a different decomposition, and the demo measures the
difference at the only place it shows up — retrieval accuracy on documents the
model never saw:

| $k$ | raw counts | tf-idf | log-entropy |
|:--|:--|:--|:--|
| 1 | 25% | 25% | 25% |
| 2 | **55%** | **90%** | **72%** |
| 3 | 88% | 100% | 100% |
| 4 | 100% | 100% | 100% |

At the natural rank every model is perfect and the choice is invisible. Cut one
step below it and the weighting is the entire model: raw counts get the topic
right for barely half the held-out documents, tf-idf for nine in ten. Meanwhile
the two *solvers* — exact SVD and a randomized sketch — give the same retrieval
number to the last digit, because they compute the same decomposition. That
asymmetry is the thesis of this file.

Two further things are worth knowing before the derivation, because they are
where implementations go wrong.

**It is not PCA.** PCA centres the columns before decomposing; LSA does not.
The demo's mean count is $0.67$, and subtracting it would be a modelling
decision, not a cleanup — a row of zeros means "empty document", not "average
document". Centring would make the first component the mean document, which
for counts is mostly *document length*, precisely the nuisance the weighting
exists to remove.

**The weights are part of the model.** `idf` and the entropy weight $g$ are
fitted on the training corpus, stored, and reused verbatim for every query.
Recomputing them on a batch of one gives $\log(1/1) = 0$ for every term in it,
so the query vector is zero and it has no cosine with anything. The demo
measures $\lVert z \rVert = 0.87$ for a single-document query with the fitted
weights, and $0$ without — a failure that is silent, because a zero vector
still returns *some* argmax.

## Variants

| Variant | Solver | Weighting | What it is for | Measured |
|:--|:--|:--|:--|:--|
| `truncated_svd` | full SVD | raw $\mathrm{tf}$ | the reference, and the exact answer | residual $= \sqrt{\Sigma_{i>k}\sigma_i^2}$ |
| `randomized` | sketch + power iteration | raw $\mathrm{tf}$ | same answer, fewer flops on a large $X$ | subspace gap $10^{-9}$ in 8 steps |
| `tfidf_lsa` | full SVD | $\mathrm{tf}\cdot\log(n/\mathrm{df})$ | downweight ubiquitous terms | 90% at $k=2$ |
| `log_entropy` | full SVD | $\log(1+\mathrm{tf})\cdot g$ | downweight evenly spread terms | 72% at $k=2$ |

The four ids are not four algorithms. There is one solver decision (exact or
sketched) and one weighting decision (three choices), and the catalog exposes
them as a **T** rather than a full cross:

|  | raw `tf` | tf-idf | log-entropy |
|:--|:--|:--|:--|
| full SVD | `truncated_svd` | `tfidf_lsa` | `log_entropy` |
| randomized | `randomized` | — | — |

Any two variants on the top row differ *only* in the weighting, and the left
column differs *only* in the solver. That is enough to separate the two effects — the
demo's randomized fit reproduces the raw-count accuracy to the last digit, and
the weighting moves it by 35 points — but it does mean the solver comparison is
always made under raw counts, and nobody has measured whether the sketch is as
accurate on tf-idf'd data. It should be: the sketch approximates the range of
whatever matrix it is given, and the weighting is applied before the solver.
The table is encoded once so the two decisions cannot drift apart:

```python
WEIGHTING = {"truncated_svd": "tf", "randomized": "tf",
             "log_entropy": "log_entropy", "tfidf_lsa": "tfidf"}
```

**`truncated_svd`** is `numpy.linalg.svd` followed by a slice. It is the
definition of the atom: the exact best rank-$k$ approximation in the Frobenius
norm, computed directly. It is also the honest baseline every other variant has
to be compared against, and the reason the demo can assert an identity
(residual$^2$ equals the discarded spectrum) rather than a tolerance.

**`randomized`** is the Halko–Martinsson–Tropp range finder: draw a Gaussian
sketch of the range, sharpen it with power iterations, orthonormalise, and
solve the SVD on a matrix that is only $(k+p) \times d$. On a $240 \times 60$
matrix this is not faster than the exact SVD — the constant dominates — but it
is the algorithm that makes LSA possible at $n = 10^6$, where a full SVD is
$O(nd\min(n,d))$ and out of reach. Its accuracy is a *choice* (`n_iter`), not a
hope, and the demo measures the choice.

**`tfidf_lsa`** is what most people mean by "LSA". The weighting is the
model: it rescales each column by $\log(n/\mathrm{df}_j)$, so a term that
appears in every document gets weight zero and a term unique to one document
gets $\log n$. That is a statement about *informativeness*, and it is
deliberately crude — it only asks whether a term appears, never how often or how
the appearances are distributed.

**`log-entropy`** is Deerwester et al.'s original weighting, and it asks the
second question. A term's distribution over documents has an entropy; if the
entropy is maximal the term is spread perfectly evenly and cannot distinguish
documents, so its weight is zero; if the term is concentrated, it can. The
difference from tf-idf is visible on a term that appears in *every* document but
mostly in one: tf-idf cannot see it, log-entropy can. The demo's tiny $3 \times
4$ matrix is constructed to make that difference exact.

## Derivation

### 1. Eckart–Young, with the inequality written out

Let $X = U S V^\top$ with $\sigma_1 \ge \cdots \ge \sigma_r > 0$. For any matrix
$B$ with $\mathrm{rank}(B) \le k$,

$$\lVert X - B \rVert_F^2 \;\ge\; \sum_{i>k} \sigma_i^2, \qquad (2)$$

with equality at $B = U_k S_k V_k^\top$. Expanding the left side,

$$\lVert X - B \rVert_F^2 = \lVert X \rVert_F^2 - 2\langle X, B\rangle + \lVert B \rVert_F^2, \qquad (3)$$

and both inner products are controlled by the singular values. Von Neumann's
trace inequality gives

$$\langle X, B \rangle = \mathrm{tr}\!\left(X^\top B\right) \;\le\; \sum_i \sigma_i(X)\,\sigma_i(B) \;=\; \sum_{i \le k} \sigma_i\,\tau_i, \qquad (4)$$

where $\tau_i = \sigma_i(B)$ and the last equality is $\mathrm{rank}(B) \le k$.
Also $\lVert B \rVert_F^2 = \sum_{i\le k}\tau_i^2$. Substituting into (3),

$$\lVert X - B \rVert_F^2 \;\ge\; \sum_i \sigma_i^2 - 2\sum_{i\le k}\sigma_i\tau_i + \sum_{i\le k}\tau_i^2 \;=\; \sum_{i>k}\sigma_i^2 + \sum_{i\le k}(\sigma_i - \tau_i)^2. \qquad (5)$$

The right side is a sum of squares, so it is minimised by $\tau_i = \sigma_i$
and equals $\sum_{i>k}\sigma_i^2$ there. That is the whole theorem, and it is why
the demo can write

```python
    assert abs(resid ** 2 - tail) < 1e-10, (resid ** 2, tail)
```

rather than comparing two approximations: the residual of the truncated SVD is
not approximately the discarded spectrum, it *is* the discarded spectrum. On the
demo's matrix, $7.5624^2 = 57.19$, and the 56 discarded singular values account
for exactly that.

Making "best" non-vacuous needs the right comparison, and the obvious one is
wrong. A random Gaussian $\tilde{U}\tilde{V}^\top$ has a random scale, so
beating it says nothing about the subspace; what matters is the best rank-$k$
approximation *in a random $k$-dimensional subspace*. Over Haar-random $Q$,
$\mathbb{E}[QQ^\top] = \frac{k}{d}I$, so a typical subspace keeps exactly
$k/d$ of the energy and its residual is

$$\sqrt{\left(1 - \frac{k}{d}\right)\sum_i \sigma_i^2}, \qquad (6)$$

which is $14.97$ for the demo's $k=4$, $d=60$. The mean of 200 draws is $14.96$
and the smallest is still larger than $7.56$. That is the theorem in the form
worth remembering: the truncation is not slightly better than a typical
subspace, it is better by exactly the amount the spectrum says it should be.

One consequence that is easy to miss: (2) is a statement about *the matrix you
passed in*. Change the weighting and you change the objective. The tf-idf model
has a **larger** reconstruction error than raw counts ($8.84$ against $7.56$ at
$k=4$) and retrieves better. The SVD is optimal for its own criterion; if that
criterion is not the one you care about, optimality is worth nothing.

### 2. tf-idf, and what $\log$ is doing there

Raw counts encode two things at once: what a document is about, and how long it
is. A 2 000-word document has larger counts than a 200-word one for every term
it shares, so its vector is longer and it wins every cosine by default. The
term side has the mirror problem: `the` occurs in every document, so it has a
large count everywhere and contributes a constant to every similarity.

tf-idf addresses the second one. With $\mathrm{df}_j = \lvert\{i : x_{ij} > 0\}\rvert$,

$$w_j = \log\frac{n}{\mathrm{df}_j}, \qquad \tilde{x}_{ij} = x_{ij}\, w_j. \qquad (7)$$

The $\log$ is not derived from anything; it is chosen so that the weight is
**bounded and zero at the useless end**. Without it, $w_j = n/\mathrm{df}_j$
grows without bound as $\mathrm{df}_j \to 1$ and a single rare term can dominate
a document's vector. With it, $w_j = 0$ exactly when a term is in every
document, $w_j = \log n$ when it is in one, and the scale is the log of the
informativeness ratio. Smoothed variants ($\log(1 + n/\mathrm{df})$,
$\log((1+n)/(1+\mathrm{df})) + 1$) exist to keep $w_j > 0$; the unsmoothed form
is what this file uses because the zero is the feature.

The important implementation point is that (7) is a **corpus statistic**.
$\mathrm{df}_j$ is a property of the training set. Fit it once, store it, reuse
it. `_term_weights` is a separate function from `_local_weight` for exactly this
reason: one is a per-document transform, the other is a fitted parameter.

### 3. Log-entropy: presence is not the whole story

Let $p_{ij} = x_{ij} / \sum_{i'} x_{i'j}$ be the fraction of term $j$'s
occurrences that fall in document $i$. Its entropy over documents,

$$H_j = -\sum_{i=1}^{n} p_{ij}\log p_{ij} \;\in\; [0, \log n], \qquad (8)$$

is maximal when the term is spread perfectly evenly and zero when it lives in
one document. The global weight is the normalised complement,

$$g_j = 1 - \frac{H_j}{\log n} \;\in\; [0, 1], \qquad (9)$$

and the weighted matrix is $\tilde{x}_{ij} = \log(1 + x_{ij})\, g_j$.

Two design choices are bundled here, and they are independent:

- the **global** weight $g_j$ replaces $\log(n/\mathrm{df}_j)$, and it is a
  strictly finer statistic: $\mathrm{df}$ is the support of $p_{\cdot j}$, $H_j$
  is its whole shape;
- the **local** weight is $\log(1+\mathrm{tf})$ rather than $\mathrm{tf}$, which
  is a within-document decision and needs no corpus: the twentieth occurrence of
  a term says much less than the second.

The difference matters on a term that appears in every document. Consider the
demo's tiny matrix,

$$T = \begin{pmatrix} 2 & 1 & 0 & 3\\ 1 & 1 & 2 & 3\\ 0 & 3 & 1 & 3\end{pmatrix}, \qquad (10)$$

whose second column is $(1,1,3)$ and fourth is $(3,3,3)$. Both have
$\mathrm{df} = n = 3$, so tf-idf gives both weight zero and deletes them. But
the fourth is flat — it says nothing about which document you are reading —
while the second is concentrated in document 3, which is a real signal. The demo
measures $g = (0.421,\,0.135,\,0.421,\,0.000)$: the fourth is deleted, the second
is kept, and the two weightings disagree about a term both of them can see.

One edge case is worth stating because it is the degenerate-corpus failure mode
in this file. For $n = 1$, $\log n = 0$ and (9) is $0/0$. The guard divides by $1$
instead, which makes a one-document corpus have all weights zero — correct in
the sense that there is nothing to distinguish, and a reminder that these
weights are corpus statistics, not per-document ones.

### 4. Row normalisation, and why L2

Cosine similarity between documents $a$ and $b$ is

$$\cos(a, b) = \frac{a^\top b}{\lVert a \rVert\, \lVert b \rVert}, \qquad (11)$$

so if every row is scaled to unit norm, similarity *is* the dot product and the
retrieval step is one matrix multiply. That is the only reason `norm=True`
exists, and it is the second half of the length normalisation: tf-idf stops
ubiquitous terms from dominating, row normalisation stops long documents from
doing it.

L2 rather than L1 because the SVD's geometry is Euclidean. (2) minimises a
Frobenius norm, so "close" in the latent space means "close in $\ell_2$"; if the
rows were L1-normalised, the space in which similarity was measured and the
space in which the approximation was optimal would not be the same one, and the
low-rank model's guarantee would not apply to the quantity being used.

With unit rows, the leading singular direction is the axis of largest scatter
of the document cloud. When the corpus shares a background vocabulary — which
almost every corpus does — that axis is close to the mean document direction,
which is why the first component reads as "general English" and is often
discarded. This is the uncentred analogue of PCA's first component landing on
the mean: PCA removes the mean and does not have the problem, LSA keeps it and
has to throw the component away instead.

### 5. The randomized SVD, and what power iteration buys

Draw $\Omega \in \mathbb{R}^{d \times (k+p)}$ with i.i.d. Gaussians and form

$$Y = X\Omega \in \mathbb{R}^{n \times (k+p)}. \qquad (12)$$

With probability at least $1 - 3p^{-p}$, the range of $Y$ captures the top $k$
singular directions up to an error of order $\sigma_{k+1}$. A Gaussian sketch
is not the cheapest choice — a subsampled random Fourier transform or a sparse
sign matrix works — but it is the one with the cleanest bound and no structural
assumption on $X$, which for a count matrix with a handful of very large entries
is worth the constant.

The sketch is sharpened by alternating products:

$$Y_q = \left(XX^\top\right)^q X\Omega, \qquad (13)$$

and in the SVD basis this multiplies the $i$-th direction's amplitude by
$\sigma_i^{2q+1}$. The ratio between the last wanted and the first unwanted
direction goes from $\sigma_k/\sigma_{k+1}$ to

$$\left(\frac{\sigma_k}{\sigma_{k+1}}\right)^{2q+1}, \qquad (14)$$

which is the entire reason a few iterations are enough. On the demo's spectrum
$(\sigma_4, \sigma_5) = (4.41, 1.91)$, a ratio of $2.3$: after eight power
steps the gap is $2.3^{17} \approx 1.4 \times 10^6$, and the measured subspace
error drops from $3 \times 10^{-1}$ to $10^{-9}$.

The rest is bookkeeping. Orthonormalise the sketch, $Y = QR$; project the data,
$B = Q^\top X$, which is $(k+p) \times d$; decompose $B = U_b S V^\top$; then
$X \approx Q U_b S V^\top$ and the factors are $(QU_b, S, V^\top)$. `Q @ Ub` is
exactly the orthogonal factor the exact SVD would have produced in the range of
the sketch, which is why the two agree to machine precision once the sketch is
good.

Oversampling $p$ is not cosmetic. The sketch has $k+p$ columns, so even a
perfect rank-$(k+p)$ approximation has room for $p$ directions of leakage. The
standard bound has the shape
$\left[1 + c\sqrt{k+p}\,(p-1)^{-1}\sqrt{\min(n,d)}\right]^{1/(2q+1)}$ for an
absolute constant $c$ of order one, and the two appearances of $p$ pull in
opposite directions — a wider sketch is a better
subspace, but the concentration constant degrades as $p$ shrinks. $p = 10$ is
the usual compromise for dense data, and it is what this file's default is.

### 6. What the decomposition does and does not identify

$X = U S V^\top$ is unique when the singular values are distinct, but the pair
of factors is not. For any diagonal $D$ with entries $\pm 1$,

$$(U S D)(D V)^\top = U S D D V^\top = U S V^\top, \qquad (15)$$

so the sign of each component is arbitrary — a convention, not an estimate.
When $\sigma_i = \sigma_j$ the ambiguity is larger: any rotation in that
$2 \times 2$ block gives the same $X$, which is the same non-identifiability
factor analysis has, in a setting where it is harmless. The demo's singular
values are distinct ($9.1$ against $1.9$ for the first discarded one), so the
only freedom is sign.

Sign is invisible to everything the model is used for. Cosine similarity is
invariant under $a \mapsto -a$, so retrieval does not see it; the reconstruction
is invariant under the *paired* flip of (15), so `inverse_transform` does not see
it either. What *does* see it is interpretation: a topic component and its
negation have the same set of large-magnitude terms, but opposite signs, so
"topic 3 is about finance" is a statement about the row's absolute values. The
sign is the one bit of an LSA component you have to fix by convention before
writing the row down.

The split between `documents_` and `components_` is a second convention.
$U S$ and $V^\top$ together reconstruct $X$, but the distribution of the
singular values between them is a choice. This file puts $S$ on the documents,
which is the standard LSA convention and the one that makes `transform` and
`documents_` agree: $X_w V_k = U_k S_k$.

### 7. Why this is not PCA

PCA decomposes the centred matrix: $X_c = X - \mathbf{1}\mu^\top$, then
$X_c = U S V^\top$. LSA decomposes $X$ itself. Three reasons, in increasing
order of importance:

1. The origin is meaningful. A row of zeros is a document with no words in it,
   not a document of average length. Centring moves that row to $-\mu$, which is
   not a document.
2. The mean is an artefact of corpus composition. Add a thousand documents about
   sport to a corpus of recipes and the mean moves; LSA's factors move with it
   only through the shared terms, PCA's first component moves to follow the new
   mean.
3. Centring and weighting do not commute. tf-idf is already a nonlinear
   transform of the counts; centring after it produces negative entries, which
   is fine for PCA's algebra and wrong for the model — a negative count is not a
   count. PCA on tf-idf vectors is a legitimate technique, but it is a different
   atom, and it is not in this file.

The demo asserts the mean is not zero ($0.67$), which is the numerical form of
the same statement.

## Code walkthrough

`LSA(n_components=100, *, variant="tfidf_lsa", seed=0, n_iter=4,
oversample=10, norm=True)` with `fit`, `transform`, `fit_transform` and
`inverse_transform`, and the attributes `components_` $(k, n_{\text{terms}})$,
`documents_` $(n_{\text{docs}}, k)$, `singular_values_`, `weights_`,
`explained_variance_ratio_` and `reconstruction_error_`.

Two conventions are load-bearing and consistent with the rest of the repo.
`components_` is `(k, n_terms)` — one row per latent dimension — so a document
vector is `Xw @ components_.T`, and `documents_` is `U S`, not `U`, so that the
two live in the same coordinates. `weights_` is exposed because it is the part
of the model a reader can inspect and argue with: the topic matrix is opaque,
the fact that `idf` gave a term weight zero is not.

`norm=True` is the default because raw counts plus no normalisation is the
model that gets 55% in the table above. `n_iter=4` is the randomized solver's
power-iteration count and is ignored by the exact one.

### 1. The two halves of a weighting

```python
def _local_weight(X: np.ndarray, kind: str) -> np.ndarray:
    """The term-frequency part: raw counts, or `log(1 + tf)` for log-entropy.

    The log is there to stop one term that a document happens to repeat from
    dominating its vector; it is a within-document decision and needs no corpus.
    """
    return np.log1p(X) if kind == "log_entropy" else X
```

The split between `_local_weight` and `_term_weights` is the design decision
that makes the atom honest. A weighting has a per-document factor and a
per-corpus factor; convolving them into one function is how implementations end
up recomputing the corpus statistic at query time. Here the local half takes
only `X` and a string, and the corpus half is a separate function that returns
a vector to be stored.

### 2. The corpus half: idf

```python
    n, d = X.shape
    if kind == "tf":
        return np.ones(d)
    if kind == "tfidf":
        df = np.count_nonzero(X > 0.0, axis=0)
        return np.log(n / np.maximum(df, 1))
```

(7). `np.count_nonzero(X > 0.0, axis=0)` is $\mathrm{df}_j$ and the
`np.maximum(df, 1)` guards the all-zero column, which a term that appears in no
document at all would produce — `log(n/0)` is $+\infty$, and an infinite weight
on a column of zeros is harmless in exact arithmetic and a NaN factory in
floating point. The `axis=0` is the one that matters: documents are rows, so
"how many documents contain term $j$" is a count down a column.

### 3. The corpus half: entropy

```python
    total = X.sum(axis=0)
    p = X / np.maximum(total, EPS)                    # term j's mass over documents
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(X > 0.0, p * np.log(np.maximum(p, EPS)), 0.0)
    entropy = -plogp.sum(axis=0)                      # in nats, in [0, log n]
    # a one-document corpus has entropy 0 by definition and nothing to weigh
    return 1.0 - entropy / (math.log(n) if n > 1 else 1.0)
```

(8) and (9). The axis is the same as above and the roles are swapped: `total` is
per **term**, and `p` distributes each term over the documents. `np.where(X >
0.0, ...)` implements the $0\log 0 = 0$ convention explicitly rather than
relying on a masked `nan`; `np.errstate` silences the division for the zero
entries, which would otherwise warn on every call. The `np.maximum(p, EPS)`
*inside* the log is what stops those zero entries from taking $\log 0 = -\infty$
before `where` gets a chance to discard them; the resulting entropy is not
clipped, because $0 \log 0$ is taken as zero by the `where` and $p = 1$ gives
$\log 1 = 0$.

The single-document branch is the $n = 1$ problem from the derivation's §3, and
it is in the code rather than in a comment because the alternative is a NaN that propagates
into every weight and turns the whole fit into `nan`.

### 4. Fitting: two decisions and their composition

```python
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
```

The order is the point: weights first, then the weighted matrix, then the
solver. `self.weights_` is stored **before** `_weighted` runs, so a caller can
recover the corpus statistic even though it is no longer separable from the
product. `k = min(...)` is a silent truncation: asking for 100 components from a
$240 \times 60$ matrix gives 60, the smaller dimension, and `components_` is
honest about its own shape.

`U * s` rather than `U` is the LSA convention and it makes the two coordinate
systems agree: the reconstruction is `(U S) Vᵀ`, and a query is projected as
`Xw Vᵀ`, which is the same map. The demo asserts the two agree on the training
matrix to $2 \times 10^{-15}$.

### 5. The weighting at transform time

```python
        Xw = _local_weight(X, self.weighting) * self.weights_
        if self.norm:
            # rows on the same scale, which makes cosine similarity a dot product
            nrm = np.linalg.norm(Xw, axis=1, keepdims=True)
            Xw = Xw / np.maximum(nrm, EPS)
        return Xw
```

`self.weights_`, not `_term_weights(X, ...)`. Two characters separate a model
from a normalisation, and the difference is measurable: the demo's
single-document query has norm $0.87$ with the stored weights and would have
norm $0$ with recomputed ones. The `np.maximum(nrm, EPS)` is the empty-document
guard; a row of zeros stays a row of zeros rather than becoming a NaN.

### 6. The sketch, the power loop, and the small SVD

```python
    n, d = X.shape
    width = min(k + oversample, min(n, d))
    omega = rng.standard_normal((d, width))
    Y = X @ omega
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)                            # orthonormal basis of the sketch
    Ub, s, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]
```

(12) and (13). `width = min(k + oversample, min(n, d))` is the only line that is
not straight from the paper: a sketch wider than the matrix cannot have full
rank, so the extra columns are wasted work and the small SVD they produce has
guaranteed zero singular values at the bottom. Clamping is cheaper than
explaining them. The power loop
is written as `X @ (X.T @ Y)` and not `(X @ X.T) @ Y` because the former is two
matrix-vector-block products and the latter forms an $n \times n$ matrix — at
$n = 10^6$ that is the difference between 8 GB and a few megabytes.

`Q.T @ X` is $(k+p) \times d$, so the second SVD is a small one; the returned
`Q @ Ub` is the left factor the exact solver would have given, restricted to the
sketch's range. The demo checks that this converges to the exact subspace rather
than merely claiming it: $3 \times 10^{-1}$ with no power iteration,
$2 \times 10^{-3}$ with two, $10^{-9}$ with eight.

### 7. Eckart–Young, as an identity

```python
    recon = U[:, :k] * s[:k] @ lsa.components_
    tail = float(np.sum(s[k:] ** 2))
    resid = float(np.linalg.norm(Xw - recon))
    assert abs(resid ** 2 - tail) < 1e-10, (resid ** 2, tail)
```

The demo does not compare the truncated SVD against a random matrix and declare
victory; it checks (5)'s equality case, which is a much stronger statement. The
random comparison is there too, but against the right thing: 200 random
rank-4 **subspaces**, each projected exactly, average $14.96$ against the
predicted $\sqrt{(1-k/d)\sum\sigma^2} = 14.97$, and none of them reaches
$7.5624$. A Gaussian random matrix would have been a meaningless baseline —
its scale is arbitrary, so beating it proves nothing.

`U[:, :k] * s[:k] @ lsa.components_` is `(U S) Vᵀ` written with the same
orientation as the model, so the check is comparing the model's own
reconstruction with the one computed from a fresh SVD. If `components_` were
transposed, or if `documents_` were `U` instead of `U S`, this line would fail.

### 8. The single-document query

```python
    q = LSA(k, variant="tfidf_lsa").fit(X[:200]).transform(X[205:206])
    assert float(np.linalg.norm(q)) > 0.5, q
    assert np.all(_term_weights(X[205:206], "tfidf") == 0.0)
```

Two assertions that are the same fact from both sides. The first says the fitted
model embeds a one-document query with a real direction; the second says the
*unfitted* weighting gives every term in that document weight zero, because
$\mathrm{df}_j = n = 1$ makes $\log(n/\mathrm{df}_j) = \log 1 = 0$. If
`transform` called `_term_weights` instead of using `self.weights_`, the first
assertion would be `0.0 > 0.5` and the second would still pass — which is
precisely the kind of bug that survives a test suite that only checks shapes.

### 9. The sign, and the ambiguity that cancels

```python
    sig = np.array([-1.0, 1.0, 1.0, -1.0])
    rec = first.inverse_transform(first.documents_)
    flipped = (first.documents_ * sig) @ (first.components_ * sig[:, None])
    assert np.max(np.abs(flipped - rec)) < 1e-12
```

(15), as an experiment. `sig` is a diagonal of $\pm 1$; multiplying the columns
of `documents_` by it and the matching rows of `components_` by it cancels in
the product, so the reconstruction is unchanged to $10^{-12}$ while the factors
differ by a sign. The demo also checks the stronger statement that a negated
component is still an *exact* eigenvector — both signs satisfy
$V X^\top X = \lambda V$ to $3 \times 10^{-9}$ — so there is no equation in the
model that could pick between them.

### 10. Back to term space, and what the check refuses

```python
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.components_.shape[0]:
            raise ValueError(f"expected (m, {self.components_.shape[0]}), got {Z.shape}")
        return Z @ self.components_
```

`inverse_transform` is one product and one shape check, and it is worth having
because it makes the model's claim testable: `inverse_transform(fit_transform(X))`
is the rank-$k$ approximation of the weighted matrix, so the residual can be
compared against (2). The shape check is not decoration — it catches the
transposed-`components_` convention error at the boundary rather than three
functions later.

```python
        if np.isnan(X).any() or (X < 0.0).any():
            raise ValueError("LSA needs a non-negative, complete count matrix")
```

The refusal is the model, not a limitation. Log weights are undefined for
negative entries, `np.log1p` is not monotone below $-1$, and idf's `df` is a
count of *presence*, which a negative entry is not. A centred matrix — the
thing PCA would happily take — is rejected here, and the error says why.

## Complexity

$n$ documents, $d$ terms, $k$ components, $p$ oversampling, $q$ power
iterations.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_term_weights` | $O(nd)$ | $O(nd)$ | one pass for `df`, one for the entropy |
| `_weighted` | $O(nd)$ | $O(nd)$ | materialises the weighted matrix |
| `_svd` | $O(nd\min(n,d))$ | $O(nd)$ | one full decomposition |
| `_randomized_svd` | $O(nd(k+p)q)$ | $O(n(k+p) + nd)$ | $q$ passes over $X$ |
| `transform` | $O(mdk)$ | $O(mk)$ | one product per query batch |
| `inverse_transform` | $O(mdk)$ | $O(md)$ | back to term space |

The interesting comparison is the first two rows against the third and fourth.
Weighting is linear in the size of the data and cannot be avoided: you have to
touch every entry once to know how often each term appears. The solver is where
the asymptotics differ, and the randomized one wins only when $k+p \ll
\min(n,d)$ *and* the matrix is too large for a full SVD — for $240 \times 60$
the exact decomposition is microseconds and the sketch is pure overhead.

Memory is the harder constraint in practice. `_weighted` materialises a dense
$n \times d$ matrix, which for a million documents and a $10^5$-term vocabulary
is $8 \times 10^{11}$ bytes. This file is deliberately dense; the sparse version
of LSA is a different piece of engineering (sparse `df`, sparse matvec, Lanczos
on the normal equations), and the algebra is unchanged.

The whole five-section demo runs in $0.08$ s, most of it in the SVDs and the
randomized fits. That is the point of a $240 \times 60$ example: the algorithms
are not the bottleneck, the modelling choices are.

One line in `fit` is worth calling out as a small piece of the same lesson.
`explained_variance_ratio_` needs $\lVert X_w\rVert_F^2$, and the obvious way to
get it is a second SVD; `np.sum(Xw ** 2)` is the same number in $O(nd)$ with no
decomposition at all. The reference implementation had the SVD, which would
have made the randomized variant pointless at scale — it would have paid the
full cost to measure how much it had saved.

## Failure modes

**1. Recomputing the weights at query time silently zeroes the query.** The
demo measures $\lVert z \rVert = 0.87$ with the fitted `idf` and exactly $0$
with the recomputed one, because a single document has $\mathrm{df} = n$ for
every term it contains. Nothing raises; a zero vector has an argmax, so a
retrieval pipeline built this way returns document 1 for every query and looks
like a bad model rather than a bug. The fix is structural: fit the weights,
store them, apply them.

**2. There is no natural $k$, and the spectrum's cliff is a property of the
data.** The demo's singular values are $9.1, 6.9, 5.8, 4.4, 1.9, 1.8, 1.7, 1.6,
\ldots$: a factor of $2.3$ between the fourth and fifth, then ratios near $1.06$.
The cliff is real — the corpus was generated from four topics — but a real
corpus has no such guarantee, and a flat tail gives no defensible $k$. Choosing
$k$ by explained variance is choosing it by *reconstruction*, which is not the
task: the demo's table shows the same $k$ giving 55% and 90% under different
weightings. Choose $k$ on a held-out retrieval or classification metric, or fix
it at a budget.

**3. Dense only.** Every operation here is $O(nd)$ in memory. A term-document
matrix from a real corpus is 99.9% zeros, and the dense representation is the
wrong data structure by four orders of magnitude. This is not a tuning issue; it
is the difference between a demo and a system.

**4. tf-idf deletes terms that are informative.** $\mathrm{df}$ is binary, so a
term in every document gets weight zero regardless of how it is distributed. The
$3 \times 4$ demo's second column, $(1,1,3)$, is exactly that case: tf-idf removes it,
log-entropy keeps it, and there is no way to fix it inside tf-idf without
switching to the entropy form. On a corpus where the distinction matters, the
two weightings give materially different topics.

**5. A one-document corpus is degenerate.** $\log n = 0$ makes (9) undefined;
the guard returns all-zero weights, which is the correct limit and a warning
that entropy weighting needs a corpus. It is also the shape of a more common
mistake: fitting on a sample of one class from each category and then applying
the weights to the full data.

**6. LSA is linear and bag-of-words.** It cannot represent word order,
negation, or two senses of one word. `bank` gets one vector, which is a mixture
of the river and the finance senses, and every document containing it is pulled
towards both. This is the limitation that dense contextual embeddings exist to
address, and no amount of rank tuning removes it.

**7. Randomized without power iteration is a different algorithm.** With
`n_iter=0` the subspace gap is $3 \times 10^{-1}$ and the singular values are
wrong in the first decimal; the fit still returns a plausible-looking topic
matrix and a plausible-looking accuracy. The iteration is not an optimisation —
it is what makes the sketch an approximation of the *top* of the spectrum rather
than of a random subspace.

**8. Sign and scale are not identified.** §6: a component and its negation are
both exact solutions, and two fits can differ by one. Anything that compares
components across fits — "topic 1 in model A is topic 2 in model B" — has to
align signs first, and anything that treats a loading's sign as meaningful has
to fix a convention. Cosine-based retrieval is immune; interpretation is not.

**9. The reconstruction is not a document.** `inverse_transform` returns real
numbers, including negative ones, because a low-rank approximation of a
non-negative matrix is not non-negative. Rounding it to counts is not
"denoising"; it is a different model (non-negative matrix factorisation, the
neighbouring atom).

**10. Explained variance and retrieval quality disagree.** The raw-count model
reconstructs its own matrix better ($7.56$ against $8.84$ at $k=4$) and retrieves
worse ($55\%$ against $90\%$ at $k=2$). Those are different matrices, so the
numbers are not a like-for-like comparison — which is exactly the point: the SVD
is optimal for whichever matrix it is given, and the matrix is a modelling
choice. A pipeline that tunes the solver and not the weighting has optimised the
wrong thing.

**11. Truncating at the smaller dimension is silent.** `n_components=100` on a
$240 \times 60$ matrix yields 60, and every downstream index that assumed 100 is
off by 40. The attribute is right; the constructor argument is not a promise.

## Applications

- **Information retrieval.** The original application (Deerwester et al. 1990):
  match a query against documents that share no query terms, by projecting both
  into the latent space. It matched lexical search on some TREC collections and
  lost on others, which is the honest summary of the evaluations; it is the
  textbook example of a retrieval method that works because of a modelling
  insight rather than a better index.
- **Cross-language retrieval.** Build the term-document matrix over a
  *parallel* corpus with both languages' terms as columns; the latent dimensions
  align translations, and a query in one language retrieves documents in the
  other. The alignment is a consequence of shared context, and it is the
  cleanest demonstration that the low-rank structure is semantic rather than
  lexical.
- **Document clustering and classification.** The $k$-dimensional coordinates
  are a feature map: cheap, dense, and often as good as a bag-of-words model with
  a hundred times the dimension. Standard preprocessing before a linear
  classifier.
- **Topic segmentation and summarisation.** A document's row in $U_k S_k$ is a
  point in topic space, and consecutive documents trace a trajectory through it; a change point in that trajectory is a
  section boundary. Crude, and it works.
- **Recommendation.** Treat users as documents and items as terms, or the
  reverse; the SVD of a rating or interaction matrix is the same algebra, and
  the "latent factors" are the same latent factors. The difference from a
  proper recommender is that the missing entries are zeros here, which biases
  everything towards popular items.
- **Duplicate and near-duplicate detection.** Cosine in the latent space is
  robust to paraphrase in a way that lexical overlap is not, and the sketch is
  cheap enough to run over a large corpus.
- **As the ancestor of embeddings.** Levy & Goldberg (2014) showed that word2vec's
  skip-gram with negative sampling is implicitly factorising a shifted
  pointwise-mutual-information matrix. LSA is the explicit version of that with a
  different matrix: the modern method learns the matrix and the factorisation
  together, and inherits the same geometry.

Where it is a trap: tuning $k$ on reconstruction error; fitting the weighting on
a query batch; comparing topics across two fits without aligning signs;
reading a component's sign as meaningful; and using LSA on data that is not
counts at all — the idf and entropy weights are statistics of a count matrix,
and applying them to, say, tf-idf'd embeddings is a category error.

## References

- **Deerwester, Dumais, Furnas, Landauer & Harshman (1990),** *Indexing by
  latent semantic analysis* — the paper that named the method, the log-entropy
  weighting of §3, and the retrieval experiments that motivated all of it.
- **Landauer & Dumais (1997),** *A solution to Plato's problem: the latent
  semantic analysis theory of acquisition, induction, and representation of
  knowledge* — the psychological claim, and the demonstration that a
  $300$-dimensional LSA passes synonym tests at a human level on the TOEFL
  synonym section.
- **Dumais (1991),** *Improving the retrieval of information from external
  sources* — the empirical comparison of weighting schemes that made
  log-entropy the default in the LSA literature.
- **Eckart & Young (1936),** *The approximation of one matrix by another of
  lower rank* — the theorem of §1, in the form it is still quoted.
- **Halko, Martinsson & Tropp (2011),** *Finding structure with randomness:
  probabilistic algorithms for constructing approximate matrix decompositions*
  — the sketch and power-iteration scheme of §5, including the error bounds and
  the oversampling constant.
- **Berry, Dumais & O'Brien (1995),** *Using linear algebra for intelligent
  information retrieval* — the practical guide to the sparse implementation this
  file deliberately does not attempt.
- **Manning, Raghavan & Schütze (2008),** *Introduction to Information
  Retrieval*, ch. 18 — the term-document matrix, tf-idf, and the vector-space
  model, with the retrieval evaluation conventions used by the demo's accuracy
  numbers.
- **Levy & Goldberg (2014),** *Neural word embedding as implicit matrix
  factorization* — the result that connects this atom to the embeddings that
  replaced it.
