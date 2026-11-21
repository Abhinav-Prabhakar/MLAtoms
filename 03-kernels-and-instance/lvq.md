# Learning vector quantisation — prototypes that move under supervision

`lvq.py` · Kernels and instance · Tier 3 · 5 variants

## Intuition

k-nearest neighbours stores the training set and calls it a model. Learning
vector quantisation stores a dozen labelled points and calls it a model: the
prediction is

$$
\hat y(x)=y_{\,c},\qquad c=\arg\min_k\ \lVert x-w_k\rVert^2,
\qquad (1)
$$

and the whole algorithm is a rule for moving the $w_k$. The labels are attached
to prototypes and never change, so each prototype is a small, human-readable
statement — "this is what class 2 looks like here" — and the decision boundary
is the Voronoi diagram of a few dozen points.

Training is competitive: for each sample, the nearest prototype wins the right to
move, and supervision decides the direction. A winner whose label matches the
sample is pulled towards it; a winner whose label is wrong is pushed away. That
single sentence is LVQ1, and every other variant in this file is a refinement of
which prototypes move and how far.

There are two eras here and the file covers both. Kohonen's LVQ1, LVQ2.1 and
LVQ3 are *heuristics*: plausible local updates with no objective, no convergence
proof, and a window parameter. GLVQ defines a cost per sample and derives the
update as its gradient, which turns the heuristic into an optimisation; GRLVQ
then attaches a relevance weight to every axis and learns the metric as well.
The difference is measurable. In the demo the first four variants all score
0.62 on a linear boundary buried under six axes with twice the spread — they are
Euclidean and cannot know which axes matter — while GRLVQ scores 0.98 with a
relevance vector of $(0.73, 0.69, 0, \dots, 0)$.

## Variants

| variant | which prototypes move | ingredient | metric |
|:--|:--|:--|:--|
| `lvq1` | the nearest one | its label | fixed |
| `lvq2` | the two nearest | distance window | fixed |
| `lvq3` | the two nearest, plus a shrink | window and $\epsilon$ | fixed |
| `glvq` | nearest correct and nearest wrong | gradient of $\mu$ | fixed |
| `grlvq` | same two | gradient of $\mu$ | learned diagonal |

**`lvq1`** is one prototype per class in the simplest case and the default to
start from. It is fast, it never diverges for $\alpha<1$, and it inherits every
weakness of Euclidean distance.

**`lvq2`** (properly LVQ2.1) only updates when the sample falls in a *window*
between the nearest correct and nearest wrong prototype. The idea is that a point
deep inside a class carries no boundary information, so only near-boundary
samples should move anything. On easy data the window almost never opens: the
demo records 3 updates out of 7 200 samples.

**`lvq3`** adds the missing case. When the two nearest prototypes are both of the
correct class, both are shrunk towards the sample by a small factor
$\epsilon\in[0.1,0.5]$, which stops prototypes from drifting apart in regions
where they agree. It is the version Kohonen recommended for multi-prototype
classes.

**`glvq`** replaces the winner-take-all rule with the cost of (7) and its exact
gradient. Every update now has a sign and a magnitude that come from
differentiating something, and the epoch cost is a diagnostic rather than a
guess.

**`grlvq`** puts a relevance $\lambda_j$ on each axis and learns it by the same
gradient. It is the variant to use when the features have unequal importance and
you want the model to say which ones matter: the relevance vector is the answer,
and it is directly interpretable.

## Derivation

### Setup

Prototypes $w_1,\dots,w_P\in\mathbb R^d$ carry labels $\ell_k\in\{1,\dots,K\}$,
seeded from the training data (one or more rows per class). Define
$d_k(x)=\lVert x-w_k\rVert^2$ for the Euclidean variants and the weighted form
of (8) for GRLVQ. Prediction is (1). The learning rate is annealed linearly,

$$
\alpha(t)=\alpha_0\Big(1-\frac{t}{T}\Big),
\qquad (2)
$$

so the last epoch takes steps of $\alpha_0/T$ and the prototypes settle instead
of oscillating.

### LVQ1

For a sample $(x,y)$ let $c=\arg\min_k d_k(x)$ and
$s=+1$ if $\ell_c=y$, $-1$ otherwise. Then

$$
w_c\ \leftarrow\ w_c+\alpha\,s\,(x-w_c),
\qquad (3)
$$

which is $(1-\alpha)w_c+\alpha x$ on a match and $(1+\alpha)w_c-\alpha x$ on a
miss. For a match the prototype is an exponential moving average of the samples
it wins; for a miss it steps away from $x$ by the same fraction. The demo
asserts both closed forms: $(0.5,1.0)$ from $(0,0)$ with $\alpha=0.5$ and
$x=(1,2)$, and $(3.25,3.25)$ from $(3,3)$ with a mismatched sample at
$(2.5,2.5)$.

Because only the winner moves, the prototypes partition the space between them
and each one specialises. The cost is that a single outlier can drag a prototype
across a boundary and there is no objective to say so.

### The window: LVQ2.1 and LVQ3

Let $c$ be the nearest prototype of the correct class and $r$ the nearest of any
other class. LVQ2.1 updates only when the sample lies in the window

$$
s<\frac{\min(d_c,d_r)}{\max(d_c,d_r)}\le 1,
\qquad s\approx 0.7,
\qquad (4)
$$

and then moves both, in opposite directions:

$$
w_c\leftarrow w_c+\alpha(x-w_c),\qquad
w_r\leftarrow w_r-\alpha(x-w_r).
\qquad (5)
$$

The ratio test is a cheap proxy for "the sample is near the boundary". A point
with $d_c/d_r=0.06$ is deep inside class $c$; moving $w_r$ towards it would be
noise. The demo's constructed pair gives 0.0625 (no move) and 0.818 (inside,
total movement 1.00), which is exactly the window doing its job.

LVQ3 keeps (5) and adds the case where the two nearest prototypes are *both*
correct:

$$
w_k\leftarrow w_k+\epsilon\,\alpha\,(x-w_k),
\qquad k\in\{\text{two nearest}\},\ \ \epsilon\in[0.1,0.5].
\qquad (6)
$$

Without it, two prototypes of one class can drift apart and leave a hole that a
wrong prototype fills. With it, they cohere.

### GLVQ

Define the per-sample cost from the two distances

$$
\mu(x)=\frac{d_+(x)-d_-(x)}{d_+(x)+d_-(x)}\in[-1,1],
\qquad (7)
$$

where $d_+$ is the distance to the nearest correct prototype and $d_-$ the
distance to the nearest wrong one. Minimising $\mu$ pulls the correct prototype
closer than the wrong one; the floor $-1$ is reached when the sample sits exactly
on its correct prototype. Differentiating (7) with $d_k=\lVert x-w_k\rVert^2$,

$$
\frac{\partial\mu}{\partial w_+}=-\frac{4d_-}{(d_++d_-)^2}(x-w_+),\qquad
\frac{\partial\mu}{\partial w_-}=+\frac{4d_+}{(d_++d_-)^2}(x-w_-),
\qquad (8)
$$

so a gradient step is

$$
w_+\leftarrow w_++\eta\frac{4d_-}{s^2}(x-w_+),\qquad
w_-\leftarrow w_--\eta\frac{4d_+}{s^2}(x-w_-),
\qquad s=d_++d_-.
\qquad (9)
$$

The factor 4 is not cosmetic: the derivative of the squared distance brings a 2,
and the quotient brings another 2. The demo checks (8) against a finite
difference and gets $-0.369822$ from both. It also checks the floor of the cost
path: on the blobs GLVQ drives the epoch mean from $-0.9090$ to $-0.9232$.

### GRLVQ

Replace the Euclidean distance by a diagonal metric with relevance vector
$\lambda\in\mathbb R^d_{\ge0}$, $\lVert\lambda\rVert_2=1$:

$$
d^\lambda(x,w)=\sum_{j=1}^{d}\lambda_j^2\,(x_j-w_j)^2 .
\qquad (10)
$$

The squared relevance is the weight, so $\lambda_j$ is a scale and $\lambda_j^2$
a share of the metric. The prototype update is (9) with $d^\lambda$ and with each
coordinate additionally multiplied by $\lambda_j^2$; differentiating (7) in
$\lambda$ gives

$$
\frac{\partial\mu}{\partial\lambda_j}
=\frac{4\lambda_j}{s^2}\Big(d_-^\lambda(x_j-w_{+j})^2
-d_+^\lambda(x_j-w_{-j})^2\Big),
\qquad (11)
$$

and the relevance takes a descent step followed by projection onto the
non-negative unit sphere:

$$
\lambda\leftarrow
\frac{\max\big(\lambda-\eta_\lambda\nabla_\lambda\mu,\ 0\big)}
{\big\lVert\max(\lambda-\eta_\lambda\nabla_\lambda\mu,\ 0)\big\rVert_2}.
\qquad (12)
$$

The bracket in (11) is the whole mechanism: an axis on which the wrong prototype
is *closer* than the correct one contributes a positive gradient and loses
relevance, and an axis on which the correct prototype wins gains it. On the
demo's buried linear boundary the six noise axes end at exactly zero and the two
signal axes at $0.73$ and $0.69$.

One check makes the relationship between the two variants exact: at uniform
$\lambda_j=1/\sqrt d$ the weighted distance is $1/d$ times the Euclidean one, so
$s^2$ gains $1/d^2$ and $\lambda_j^2$ gains $1/d$; in (9) the factors cancel and
the GRLVQ step equals the GLVQ step. The demo asserts that to machine precision,
which is why GRLVQ can be read as "GLVQ plus directions" rather than as a
different algorithm.

## Code walkthrough

Prototypes are seeded per class from the training rows, using the same seeded
generator that later shuffles the epochs:

```python
        for code in range(classes.size):
            members = np.flatnonzero(codes == code)
            if members.size < self.prototypes_per_class:
                raise ValueError("a class has fewer rows than prototypes requested")
            chosen = rng.choice(members, self.prototypes_per_class, replace=False)
```

Choosing rows rather than class means matters: a prototype is then an actual
example, which is what makes the model readable, and GRLVQ's relevance is learned
from distances between real points instead of a synthetic centroid.

The distance function is where the metric enters. Two branches, one `einsum`
each, and no materialised $(P,n,d)$ cube beyond the temporary:

```python
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        if self.relevance_ is None:
            return np.einsum("npd,npd->np", diff, diff)
        return np.einsum("d,npd,npd->np", self.relevance_ ** 2, diff, diff)
```

Squaring the relevance in the second branch is the convention of (10); using
$\lambda_j$ directly would make a negative relevance indistinguishable from a
positive one of the same size.

LVQ1 is (3) in five lines, including the sign:

```python
        d = self.distances(x[None])[0]
        k = int(np.argmin(d))
        delta = rate * (x - self.prototypes_[k])
        if self.prototype_codes_[k] == label:
            self.prototypes_[k] += delta
        else:
            self.prototypes_[k] -= delta
```

Computing `delta` once and flipping the sign is the difference between (3) and
two separate code paths that can drift apart. Note that `d` is recomputed for
every sample: the prototypes change after each update, so caching distances
across samples is wrong, not merely slower.

LVQ2.1's window test is a ratio of the two smallest distances, computed by
sorting rather than by two `argmin` passes:

```python
        first, second = np.argsort(d)[:2]
        near, far = d[first], d[second]
...
        if self.prototype_codes_[first] == self.prototype_codes_[second]:
            return 0.0
        if near / far <= self.window:
            return 0.0
        correct = first if self.prototype_codes_[first] == label else second
        if self.prototype_codes_[correct] != label:
            return 0.0
        wrong = second if correct == first else first
        self.prototypes_[correct] += rate * (x - self.prototypes_[correct])
        self.prototypes_[wrong] -= rate * (x - self.prototypes_[wrong])
```

The `near + far <= 0` guard above these lines is not paranoia: a sample sitting
exactly on a prototype makes the ratio `0/0`. The `correct`/`wrong` resolution
through the labels, rather than assuming `first` is right, is what lets the same
block serve LVQ2.1 and LVQ3.

The $\epsilon$ term of (6) is the only difference between those two variants, and
it lives in the branch where both nearest prototypes are correct:

```python
        if both_correct:
            if self.variant == "lvq3" and self.epsilon > 0.0:
                for k in (first, second):
                    self.prototypes_[k] += self.epsilon * rate * (
                        x - self.prototypes_[k])
```

GLVQ and GRLVQ share the search for the two operative prototypes, which is worth
reading closely: `np.where(same, d, inf)` and `np.where(same, inf, d)` select
the nearest correct and nearest wrong in the same pass, with no Python loop and
no risk of picking the same prototype twice.

```python
        plus = int(np.argmin(np.where(same, d, np.inf)))
        minus = int(np.argmin(np.where(same, np.inf, d)))
        total = d[plus] + d[minus]
...
        toward = x - self.prototypes_[plus]
        away = x - self.prototypes_[minus]
        weight = 4.0 * rate / total ** 2
```

`toward` and `away` are computed *before* either prototype is mutated, and that
ordering is the one subtle line in the GRLVQ branch. The relevance gradient needs
the pre-update differences; computing it after the prototype step would
differentiate the wrong point.

```python
            lam_sq = self.relevance_ ** 2
            self.prototypes_[plus] += weight * d[minus] * lam_sq * toward
            self.prototypes_[minus] -= weight * d[plus] * lam_sq * away
            gradient = (4.0 * self.relevance_ / total ** 2) * (
                d[minus] * toward ** 2 - d[plus] * away ** 2)
            proposal = np.maximum(self.relevance_ - relevance_rate * gradient,
                                  0.0)
            norm = float(np.linalg.norm(proposal))
            if norm > 0.0:
                self.relevance_ = proposal / norm
```

The projection is (12): clip at zero, then divide by the norm. Skipping the
renormalisation lets the relevance vector grow without bound — the gradient is
homogeneous, so nothing else constrains its scale — and the metric then collapses
onto whichever axis accumulated the most mass first.

Prediction is a nearest-prototype lookup and nothing else, which is the point of
the atom:

```python
        d = self.distances(X)
        return self.classes_[self.prototype_codes_[np.argmin(d, axis=1)]]
```

`prototype_codes_` holds integer class indices and `classes_` maps them back, so
the labels can be strings without any per-sample conversion.

## Complexity

With $n$ samples, $d$ features, $P$ prototypes, $K$ classes, $T$ epochs:

- **One update:** $O(Pd)$ to find the nearest prototype, plus $O(d)$ to move it.
  GRLVQ adds $O(d)$ for the relevance step.
- **One epoch:** $O(nPd)$; the whole fit is $O(TnPd)$.
- **Memory:** $O(Pd)$ for the prototypes, $O(1)$ per sample. Seeds and shuffles
  are $O(n)$.
- **Against the chapter:** this is the cheapest atom in it. Kernel ridge is
  $O(n^3)$, metric learning forms an $n\times n$ matrix, and even kNN pays
  $O(nd)$ memory. LVQ pays for that with a fixed-capacity model: $P$ prototypes
  cannot represent more than $P$ prototypes' worth of boundary.

The demo's largest fit is 300 samples with 4 prototypes over 30 epochs, which is
36 000 distance rows — less than one $200\times200$ kernel matrix.

## Failure modes

**The window closes and the model stops learning.** On well-separated classes
LVQ2.1 and LVQ3 update on almost nothing: 3 of 7 200 samples in the demo, with
the accuracy essentially LVQ1's. If `n_updates_` is tiny, the window variants are
LVQ1 with two extra hyperparameters. Diagnose by counting updates; fix by
seeding more prototypes per class or by accepting that the data has no boundary
to refine.

**Prototype initialisation decides the outcome.** LVQ1/2/3 have no objective, and
even GLVQ's cost has local minima. A prototype seeded on an outlier can stay
there for the whole fit, because it only moves when it wins and it wins only
nearby samples. A control run outside the demo — a 2-d XOR embedded in six loud
axes — recovered the two signal axes on three training seeds out of five and
stalled at chance on the other two, with relevance mass spread over the noise.
Diagnose with several seeds and a held-out score; the fix is more prototypes or
a relevance warm-up, not a smaller learning rate.

**Euclidean blindness, and the diagonal metric that only partly fixes it.**
LVQ1 and GLVQ score 0.62 on the demo's buried boundary because six axes with
twice the spread dominate the distance. GRLVQ fixes that case, but its metric is
diagonal: it can down-weight an axis, not rotate the space, so two informative
axes that are strongly correlated are still seen as two axes. When the useful
directions are oblique, the full matrix of `metric_learning` is the right tool.

**A learning rate that does not decay.** With $\alpha=1$ a prototype lands
exactly on the sample and forgets everything else; with a constant $\alpha$ the
last epoch undoes part of the previous one. The schedule in (2) is not a
formality — it is what makes the final prototypes an average rather than a
snapshot.

## Applications

LVQ was built for phoneme recognition and handwritten digit classification, and
it remains a strong choice when the model must be **small and inspectable**: a
few dozen prototypes can be printed, plotted and argued about with a domain
expert, which no kernel machine or forest offers. In clinical and industrial
settings that readability is often the deciding feature.

GRLVQ's relevance vector is a feature-ranking device. Training it on a
high-dimensional panel and reading off the nonzero $\lambda_j$ is a supervised
screening step that accounts for interactions through the prototypes, and it
costs a fraction of a wrapper search. The same idea underlies divergence-based
LVQ, where the metric is chosen to match the data's statistics rather than the
Euclidean prior.

As a baseline, one prototype per class is the cheapest classifier worth reporting
— it is nearest-centroid with a learned instead of an averaged position — and it
often beats logistic regression on small, well-separated problems.

It is a trap when the classes are not convex and $P$ is small: prototypes tile
the space with Voronoi cells, so a class that wraps around another needs many
prototypes. It is also a trap for high-stakes probability estimates: LVQ outputs
a label and a distance, not a calibrated posterior, and the distance is not a
confidence.

## References

- Kohonen (1990). *The self-organizing map.* Proceedings of the IEEE 78(9).
  The paper that introduced LVQ1, LVQ2.1 and LVQ3 alongside the map.
- Sato, Yamada (1996). *Generalized learning vector quantization.* NIPS.
  The cost (7) and the gradient that turns LVQ into an optimisation.
- Hammer, Villmann (2002). *Generalized relevance learning vector
  quantization.* Neural Networks 15. The relevance vector and (11)–(12).
- Nova, Estévez (2014). *A review of learning vector quantization
  classifiers.* Neural Computing and Applications 25. The survey that puts the
  variants, and the many later ones, in one place.
- Biehl, Hammer, Villmann (2013). *Distance measures for prototype based
  classification.* Lecture Notes in Computer Science. Why the metric, not the
  update rule, is usually what decides LVQ's accuracy.
