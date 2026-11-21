# Active learning — eight ways to choose the next label to buy

`active_learning.py` · Training · Tier 2 · 8 variants

## Intuition

Labels cost money, so the interesting question is not "how do I train on the
data I have" but "which data should I pay for next". Active learning answers it
with a scoring rule: fit the model, score every unlabelled point, buy the top
`batch`, repeat. The variants differ only in the score.

Three scores read the same probability vector. Uncertainty takes one minus the
largest class probability, margin takes the gap between the top two, and
entropy sums the whole distribution. Two scores need more than one model: query
by committee asks where the members' votes disagree, and BALD measures the
mutual information between the prediction and the parameters. Expected model
change asks how far one gradient step would move the weights. The last two
ignore the model entirely and choose a batch that covers the input space.

On the demo — 600 points in four quadrants, the four-way XOR — a random budget
of 48 labels ends at 0.883 test accuracy. Uncertainty sampling reaches 0.957
with the same 48 labels, margin 0.962, and expected model change 0.955. The
coverage methods reach 0.942 and 0.952 without ever reading the model. The
honest counterexample is BALD: 0.888 at this budget, below random's noise
floor, because seven bootstrap members are not enough for the disagreement
term to mean anything; at 128 labels it climbs to 0.965 while uncertainty
reaches 0.992. Active learning is a large win here, and it is still a bet.

## Variants

| variant | score | needs | reads |
|:--|:--|:--|:--|
| `uncertainty` | $1-\max_c p_c$ | one model | probabilities |
| `margin` | $-(p_{(1)}-p_{(2)})$ | one model | probabilities |
| `entropy` | $-\sum_c p_c\log p_c$ | one model | probabilities |
| `diversity` | distance to the k-means centres | one batch | inputs |
| `qbc` | entropy of the vote shares | $M$ models | labels |
| `bald` | mutual information | $M$ models | probabilities |
| `core_set` | distance to the labelled set | nothing | inputs |
| `expected_model_change` | expected gradient norm | one model | probabilities |

One learner class, one pool, one budget. `variant=` selects the score; the
training loop, the seed set and the budget accounting are shared, so the table
at the end compares like with like.

## Derivation

### The budget

Let $\mathcal U$ be the unlabelled pool, $\mathcal L$ the labelled set, and $B$
the number of labels you can afford. Active learning chooses an ordering of
$\mathcal U$ and trains on the first $B$ elements:

$$
S^\star=\arg\max_{S\subseteq\mathcal U,\ |S|\le B}
\mathbb E\left[\mathbf{1}\{\hat y_S(x)=y\}\right],
\qquad (1)
$$

where $\hat y_S$ is trained on $\mathcal L\cup S$. Equation (1) is not
differentiable, $\hat y_S$ changes discontinuously with $S$, and the
expectation is over the same distribution you are sampling from. Every method
below is a proxy, and the demo's table is the only honest test of a proxy.

### Three ways to read one distribution

With $p_c(x)$ the predicted probability of class $c$:

$$
\alpha_{\mathrm{unc}}(x)=1-\max_c p_c(x),\qquad
\alpha_{\mathrm{mar}}(x)=-\left(p_{(1)}(x)-p_{(2)}(x)\right),
\qquad (2)
$$

$$
\alpha_{\mathrm{ent}}(x)=-\sum_{c=1}^{C}p_c(x)\log p_c(x).
\qquad (3)
$$

For $C=4$ all three peak at the uniform vector $p=(1/4,\dots,1/4)$, and all
three are near zero when one class dominates. They differ in the middle: the
margin prefers two plausible classes over four, so it hugs the actual decision
boundary, while entropy is happy with a four-way tie in a region no boundary
touches. On the demo they are within 0.005 of each other (0.957, 0.962,
0.962), which is the usual outcome when the model is reasonably calibrated.

The demo's boundaries are the two axes, so the highest-scoring points sit along
$x=0$ and $y=0$ and the interior of each quadrant is never queried. That is the
whole mechanism: random sampling spends half its budget far from a boundary,
uncertainty sampling spends almost none.

### Query by committee

Train $M$ models on bootstrap resamples and turn their predictions into a
distribution over labels:

$$
\bar v_c(x)=\frac{1}{M}\sum_{m=1}^{M}
\mathbf{1}\{\hat y_m(x)=c\},\qquad
\alpha_{\mathrm{qbc}}(x)=-\sum_c \bar v_c(x)\log \bar v_c(x).
\qquad (4)
$$

The score is high where the committee splits, which is where the training data
does not yet determine the answer. It costs $M$ fits per round, and on the
demo it reaches 0.943 against uncertainty's 0.957. Committee methods are worth
their cost when the models are genuinely diverse — different seeds, subsets,
or architectures — and a bootstrap of the same model is a weak substitute.

### BALD

BALD scores the mutual information between the label and the parameters:

$$
\alpha_{\mathrm{bald}}(x)=
H\!\left[y\mid x,\mathcal L\right]
-\mathbb E_{\theta\sim p(\theta\mid\mathcal L)}
H\!\left[y\mid x,\theta\right].
\qquad (5)
$$

The first term is the entropy of the committee's mean prediction; the second
is the average entropy of the members. Subtracting removes the irreducible
noise — a point that every member is unsure about for the same reason scores
zero — and leaves only the part of the uncertainty that more labels could
remove. The demo computes both terms from the same seven bootstrap models and
reaches 0.888 at 48 labels: the epistemic estimate is too noisy to beat plain
entropy. At 128 labels it is 0.965, still behind uncertainty's 0.992, which is
the standard finding that BALD's advantage is asymptotic and modest.

### Expected model change

A gradient step on $(x,y)$ moves the weights by $\eta\nabla_\theta\ell$. Take
the expected squared length of that step under the model's own predictive
distribution. For a softmax linear model with features $\phi(x)$,

$$
\mathbb E\!\left[\lVert\nabla_W\ell\rVert^2\right]
=\lVert\phi(x)\rVert^2
\sum_{c=1}^{C}p_c(x)\left\lVert p(x)-e_c\right\rVert^2,
\qquad (6)
$$

because the gradient of the cross-entropy is $\phi(x)\,(p-e_c)^\top$. The
factor $\lVert\phi(x)\rVert^2$ grows with distance from the feature centres, so
the score prefers points that are both uncertain and far from anything the
model has seen. The code takes the square root, which is monotone and keeps the
numbers readable. It reaches 0.955 on the demo, essentially tied with entropy.

### Coverage: core-set and k-means

Core-set ignores the model and solves a k-center problem over the labelled set:

$$
S^\star=\arg\min_{|S|\le b}\ \max_{x\in\mathcal U}\
\min_{s\in S}\lVert\phi(x)-\phi(s)\rVert .
\qquad (7)
$$

Greedy farthest-point selection — repeatedly add the point furthest from
everything already chosen — is a 2-approximation to (7), which is what the code
does. It never looks at a probability, so it cannot be fooled by a
mis-calibrated model, and it cannot exploit the boundary either: the demo's
0.952 says that covering the square is almost as good as covering the axes.

Batch diversity runs k-means on the available pool and queries the point
nearest each centroid:

$$
\min_{\{\mu_k\}}\sum_{x\in\mathcal U}\min_k\lVert x-\mu_k\rVert^2 .
\qquad (8)
$$

The purpose is different from (7): core-set spreads labels over the whole
history, k-means spreads one batch. The demo measures that directly. The mean
distance between the four points bought in a round is 1.28 for uncertainty —
four labels piled onto the same stretch of boundary — against 2.34 for k-means
and 2.43 for core-set, which is what "diverse batch" means in numbers.

### The model

The learner is multinomial logistic regression on Gaussian bumps:

$$
\phi_j(x)=\exp\!\left(-\frac{\lVert x-c_j\rVert^2}{2\sigma^2}\right),
\qquad j=1,\dots,81,
\qquad (9)
$$

with the $c_j$ on a fixed $9\times9$ grid over $[-2.2,2.2]^2$, $\sigma=0.42$,
and one extra feature equal to one. The bumps make the decision boundary
flexible enough for the four quadrants, and the fixed grid means the features
are identical at every round, which keeps the fit deterministic. Training is
full-batch gradient descent from zero with weight decay:

$$
W\leftarrow W-\eta\left(\frac{1}{n}\Phi^\top(P-Y)+\lambda W\right),
\qquad (10)
$$

for 500 steps at $\eta=0.5$ and $\lambda=10^{-3}$. Starting from zero matters:
it makes each fit a pure function of the labelled set, so a curious reader can
reproduce every number in the table.

### Label efficiency

The quantity that matters is not the final accuracy but the budget needed to
reach a target:

$$
B_\tau=\min\{b:\ \mathrm{acc}(b)\ge\tau\}.
\qquad (11)
$$

Equivalently, the area under the learning curve is the average accuracy over
the budget. The demo prints five points of the curve for three methods. At 8
labels every method is at 0.882 because the seed set is the same; uncertainty
pulls ahead at 16 (0.927 against 0.870 and 0.872) and stays ahead to 48. The
curves are noisy and non-monotone — uncertainty dips from 0.927 to 0.907 — and
a single seed is a single sample.

### Why the seed set is stratified

The demo labels two points per class before the loop starts. That is a choice,
not a law: a purely random seed set can miss a class entirely, and then nothing
downstream can recover it. Stratifying makes the comparison about the
acquisition function rather than about luck, and it is what the demo's
`initial` argument is for.

## Code walkthrough

The features are a fixed RBF grid, so every acquisition sees the same
representation and the model can bend around the quadrants:

```python
_AXIS = np.linspace(-2.2, 2.2, 9)
_CENTRES = np.stack(np.meshgrid(_AXIS, _AXIS, indexing="ij"), axis=-1).reshape(
    -1, 2)
_WIDTH = 0.42


def _features(x: np.ndarray) -> np.ndarray:
    """RBF features on a fixed 9x9 grid, plus an intercept."""
    values = np.asarray(x, dtype=float)
    gap = ((values[:, None, :] - _CENTRES[None, :, :]) ** 2).sum(axis=-1)
    return np.column_stack([np.exp(-gap / (2.0 * _WIDTH ** 2)),
                            np.ones(len(values))])
```

Training is (10) written out, with the gradient computed on the whole labelled
set at once:

```python
    weights = np.zeros((features.shape[1], classes))
    target = np.zeros((len(labels), classes))
    target[np.arange(len(labels)), labels] = 1.0
    for _ in range(steps):
        probs = _softmax(features @ weights)
        gradient = features.T @ (probs - target) / len(labels)
        gradient += penalty * weights
        weights -= lr * gradient
    return weights
```

The three single-model scores are one-liners on the same probability matrix:

```python
        if self.variant == "uncertainty":
            return 1.0 - probs.max(axis=1)
        if self.variant == "margin":
            ordered = np.sort(probs, axis=1)
            return -(ordered[:, -1] - ordered[:, -2])
        if self.variant == "entropy":
            return _entropy(probs)
```

The committee methods reuse the ensemble that `_fit_models` has already
trained: QBC builds the vote shares, BALD subtracts the average member
entropy from the entropy of the mean:

```python
        if self.variant == "qbc":
            votes = np.stack([_predict(features, weights).argmax(axis=1)
                              for weights in self.models])
            shares = np.stack([(votes == label).mean(axis=0)
                               for label in range(self.classes)], axis=1)
            return _entropy(shares)
        if self.variant == "bald":
            members = np.array([_predict(features, weights)
                                for weights in self.models])
            return (_entropy(probs)
                    - np.mean([_entropy(member) for member in members],
                              axis=0))
```

Expected model change is the same weighted residual as (6), multiplied by the
feature norm and square-rooted:

```python
        if self.variant == "expected_model_change":
            expected = np.zeros(len(pool))
            for label in range(self.classes):
                residual = probs.copy()
                residual[:, label] -= 1.0
                expected += probs[:, label] * np.sum(
                    residual ** 2, axis=1)
            return np.sqrt(expected * np.sum(features ** 2, axis=1))
```

Core-set is greedy farthest-point: score every candidate by its distance to
the nearest labelled point, take the maximum, then update the distances with
the point just bought. The `available` copy keeps the loop from reconsidering
a point inside the same batch:

```python
        if self.variant == "core_set":
            picked = []
            scores = nearest.copy()
            for _ in range(min(self.batch, len(available))):
                candidates = np.where(available)[0]
                index = candidates[int(np.argmax(scores[candidates]))]
                picked.append(index)
                available = available.copy()
                available[index] = False
                gap = pool - pool[index]
                scores = np.minimum(scores,
                                    np.sqrt(np.sum(gap ** 2, axis=1)))
            return np.array(picked)
```

Diversity runs twenty Lloyd iterations from a random initialisation, then snaps
each centroid to the nearest actual pool point, because a centroid is not a
point you can label:

```python
        if self.variant == "diversity":
            candidates = np.where(available)[0]
            width = min(self.batch, len(candidates))
            centres = candidates[self.rng.choice(len(candidates), width,
                                                 replace=False)]
            for _ in range(20):
                gap = pool[candidates, None, :] - pool[centres][None, :, :]
                near = np.argmin(np.sum(gap ** 2, axis=-1), axis=1)
                for cluster in range(width):
                    members = candidates[near == cluster]
                    if len(members):
                        centroid = pool[members].mean(axis=0)
                        centres[cluster] = members[int(np.argmin(
                            np.sum((pool[members] - centroid) ** 2, axis=1)))]
```

The loop is the whole experiment: fit, score the test set, buy a batch, record
the spread of that batch. The last fit after the loop is what makes the final
entry of `history` the accuracy at the full budget:

```python
        for _ in range(self.rounds):
            index = np.array(labelled)
            self._fit_models(_features(pool[index]), pool_labels[index])
            probs = self._probs(_features(test))
            self.history.append(float(np.mean(
                np.argmax(probs, axis=1) == test_labels)))
            chosen = np.array(labelled)
            query = self._query(pool, available, chosen)
            gap = pool[query][:, None, :] - pool[query][None, :, :]
            upper = np.triu_indices(len(query), 1)
            self.spread.append(float(np.mean(
                np.sqrt(np.sum(gap ** 2, axis=-1))[upper])))
            self.queried.extend(int(index) for index in query)
            available[query] = False
            labelled.extend(int(index) for index in query)
```

## Complexity

With $n$ pool points, $d$ features (82 here), $C$ classes, $M$ ensemble
members and $R$ rounds:

- **Uncertainty, margin, entropy:** one predict pass, $O(ndC)$, plus the fit
  $O(ndC)$ per round. These are the cheapest methods and the ones that usually
  win.
- **Expected model change:** the same predict pass plus a loop over classes,
  still $O(ndC^2)$ at worst and $O(ndC)$ in practice.
- **QBC, BALD:** $M$ fits and $M$ predict passes per round, so $O(MndC)$ for
  the ensemble and $O(MndC)$ for scoring. The demo uses seven members; the
  cost is linear in the committee size.
- **Core-set:** the greedy pass keeps a distance vector and updates it once per
  selected point, $O(nb)$ for a batch of $b$, with no model at all.
- **Diversity:** twenty Lloyd iterations over the available pool, $O(20\,nb)$,
  plus the same $O(nb)$ snapping step.
- **The demo:** eight methods at 48 labels, one extra section at 128 labels,
  five learning-curve points. 500 gradient steps on at most 600 rows is the
  unit of work, and the whole run takes about seven seconds on one core. The
  BALD section is more than half of that, which is itself the point.

## Failure modes

**Uncertainty sampling can be confidently wrong.** The score is computed from
the model's own probabilities. A model that is confidently wrong assigns a low
score to exactly the region where it needs labels. The demo avoids this because
the RBF model is smooth, but it is the standard reason to mix in coverage or
diversity.

**The batch collapses onto one boundary.** Four independent argmax queries in
one round usually land within a few tenths of each other. The demo measures it:
mean within-batch distance 1.28 for uncertainty against 2.34 for k-means.
Sequential (one-at-a-time) selection with a re-fit after each label is the
other cure, at four times the training cost.

**A weak committee makes QBC and BALD noisy.** Bootstrap resamples of the same
model on a small labelled set produce members that agree almost everywhere, so
the disagreement signal is a handful of points plus noise. BALD at 48 labels
(0.888) is worse than random, and only reaches 0.965 at 128. Ensembles need
genuine diversity — different seeds, subsets, or dropout masks.

**BALD's two entropies are estimated from the same $M$ samples.** With seven
members the average member entropy is a noisy estimate, and the subtraction can
go negative. The code does not clip, so an over-confident committee can produce
small negative scores; they simply rank last, which is the right behaviour but
not a meaningful number to report.

**Core-set ignores the labels.** It will happily spend the budget covering a
region where the class is constant. On the demo it is only 0.005 behind
uncertainty, because the quadrants are balanced; on an imbalanced or
low-density-boundary problem the gap can be much larger.

**The metric is the pool's metric.** Core-set and k-means use the raw $(x,y)$
distance; with features on different scales or a non-Euclidean input space the
notion of "coverage" changes. The demo's inputs are both on $[-2,2]$, so the
default is harmless, and a real application should standardise first.

**The pool is finite and iid.** Active learning's guarantees are about sampling
from a distribution; with a fixed pool the best possible acquisition is a
combinatorial problem, and pool bias becomes model bias. The demo's pool and
test set are drawn from the same uniform square, which is the easy case.

**The seed set is stratified.** It is a convenience that removes the most
dramatic failure mode (missing a class) and therefore makes every method look
better than it would from a random start. A realistic budget must include the
cost of finding each class at least once.

**Learning curves are noisy and non-monotone.** Uncertainty dips from 0.927 to
0.907 between rounds two and three. A single run of a single seed is an
anecdote; the ranking in the table is stable across the seeds tried here, but
that is evidence, not proof.

**A label is not always available.** The demo assumes that any pool point can
be labelled on demand. In practice labelling has latency and cost per item, and
the batch size is a compromise between query quality and the number of fits.
The same acquisition functions work, but the schedule matters.

**Test accuracy is not the deployment metric.** If one class is rare, overall
accuracy hides the errors that matter; the acquisition function should score
the same quantity you will be judged on. The demo's four balanced quadrants are
the simplest possible case.

## Applications

Uncertainty sampling is the default in production labelling tools — Prodigy,
Label Studio, and most annotation platforms ship a variant of it. Margin
sampling is common in text classification, where the top two classes are often
genuinely close, and entropy is used when the number of classes is large and a
two-way tie is not the only interesting case.

Committee and BALD methods are standard in Bayesian deep learning and in drug
discovery, where the model is expensive, the labels are expensive, and an
ensemble is affordable. Batch diversity, core-set and BADGE are used for image
and video annotation, where one round means a human session and the batch must
span the visual space rather than cluster on one object.

The same machinery appears outside labelling: Bayesian optimisation's
acquisition functions are active learning over hyperparameters, experiment
design in the natural sciences is active learning over measurements, and
pool-based uncertainty sampling is how recommender systems decide which item to
show next. The names change; the loop — fit, score, buy, refit — does not.

## References

- Lewis, D. D., Gale, W. A. (1994). *A sequential algorithm for training text
  classifiers.* SIGIR. Uncertainty sampling, (2).
- Settles, B. (2009). *Active learning literature survey.* University of
  Wisconsin–Madison. The taxonomy this atom follows.
- Seung, H. S., Opper, M., Sompolinsky, H. (1992). *Query by committee.*
  COLT. Vote entropy, (4).
- Houlsby, N., Huszár, F., Ghahramani, Z., Lengyel, M. (2011). *Bayesian
  active learning for classification and preference learning.*
  arXiv:1112.5745. BALD, (5).
- Roy, N., McCallum, A. (2001). *Toward optimal active learning through
  sampling estimation of error reduction.* ICML. Expected model change, (6).
- Sener, O., Savarese, S. (2018). *Active learning for convolutional neural
  networks: a core-set approach.* ICLR. Core-set, (7).
- Agarwal, S., Arora, H., Anand, A., Arora, C. (2020). *Contextual diversity
  for efficient active learning on deep networks.* CVPR. Batch diversity and
  the k-means objective, (8).
- Ash, J. T., Zhang, C., Krishnamurthy, A., Langford, J., Agarwal, A. (2020).
  *Deep batch active learning by diverse, uncertain gradient lower bounds.*
  ICLR. BADGE, the combination this atom keeps separate.
- Kirsch, A., van Amersfoort, J., Gal, Y. (2019). *BatchBALD: efficient and
  diverse batch acquisition for deep Bayesian active learning.* NeurIPS. The
  batch version of (5).
- Dasgupta, S. (2011). *Two faces of active learning.* Theoretical Computer
  Science. The coverage-versus-uncertainty split as a formal dichotomy.
