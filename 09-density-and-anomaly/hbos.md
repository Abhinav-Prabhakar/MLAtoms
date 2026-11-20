# HBOS — one histogram per feature

`hbos.py` · Density & anomaly · Tier 2 · 3 variants

## Intuition

Fit a histogram to each feature separately. A point's score is how surprising each
of its coordinates is under its own marginal, added up:

```
  feature 1                    feature 2                  score
 ┌────────────────┐           ┌────────────────┐
 │   ▁▃▇█▇▃▁      │           │  ▁▂▅█▅▂▁       │
 └────────────────┘           └────────────────┘
     ↓ x₁ = 0.4                   ↓ x₂ = −1.7
  −log p̂₁ = 0.9                −log p̂₂ = 3.1      →  4.0
```

Two assumptions do all the work. The first is that the joint density is the
product of the marginals, so the negative log density is their sum — that is what
makes the score cheap. The second is that a histogram is an adequate summary of a
marginal. Both are false in general, and the first is false in an interesting way,
which is why the method is worth a chapter slot: it is the fastest multivariate
detector that is not a distance, it is linear in both $n$ and $d$, and its
failures are exactly the failures of the product model.

The demo plants the failure. Two features correlated at 0.971, and forty points on
the anti-diagonal at radius 0.8 to 1.5 — every coordinate inside the data's range
and near its mode, but jointly impossible. Each variant scores them *below* the
ordinary points:

| variant | bins | AUC | planted | normals | flagged |
|:--|--:|--:|--:|--:|--:|
| `static_bins` | 20 | 0.365 | 2.54 | 2.93 | 0 of 40 |
| `dynamic_bins` | 32 | 0.474 | 2.41 | 2.90 | 0 of 40 |
| `adaptive` | 45 | 0.420 | 2.31 | 2.89 | 0 of 40 |

The score of $(1.2, 1.2)$ and $(1.2, -1.2)$ differs by 1.13 nats — the sampling
noise in two bin counts — while their true bivariate log-density differs by 46.9.
A product model has no representation of "ordinary in each coordinate, impossible
together", so it cannot rank such a point at all. Everything HBOS is good at comes
from the same place: it notices when *one coordinate* is in a region no training
point occupied, and it notices it cheaply.

The three variants are three answers to the only real question, which is how many
bins:

- `static_bins` uses a fixed count for every feature, ten by default, which is the
  original paper's choice. It is indifferent to $n$ and to the shape of the data:
  ten bins over a million rows throw away almost all the resolution, and ten bins
  over a peaky feature put the entire mode in one bin.
- `dynamic_bins` sets a count per feature from the Freedman–Diaconis width, so
  the count scales like $n^{1/3}$ and is measured in units of the feature's own
  robust spread. A flat feature gets few bins, a narrow one gets many — sometimes
  so many that `max_bins` is what actually decides.
- `adaptive` chooses the count by maximising a leave-one-out log-likelihood and
  places the edges at quantiles rather than at equal widths, so resolution follows
  the density instead of the range.

Neither data-driven rule is uniformly better, and the demo measures that too. On a
sample that is 85% a narrow Gaussian and 15% a wide one, ten equal-width bins cost
3.02 nats per held-out point and quantile bins 0.94; on a plain Gaussian the same
comparison is 2.89 against 2.96. A fixed count is a fixed *resolution*, and
resolution is adequate only when the data's scale agrees with it.

## Variants

| variant | bin count | edges | fit cost | measured |
|:--|:--|:--|:--|:--|
| `static_bins` | `n_bins`, 10 | equal width | $O(nd)$ | 3.02 nats on peaky data |
| `dynamic_bins` | Freedman–Diaconis, capped | equal width | $O(nd)$ | 1.01 nats on peaky data |
| `adaptive` | leave-one-out maximiser | quantiles | $O(Knd)$ | 0.94 nats on peaky data |

The measured column is held-out mean negative log density on a 2-feature sample of
600 training points scored on 3000 fresh ones — lower is a better density model.
On that sample `static_bins` chooses 20 bins in total, `dynamic_bins` 128 (both
features hit the cap of 64) and `adaptive` 84. On a plain Gaussian the same three
give 2.89, 2.88 and 2.96 over 20, 38 and 53 bins: the ordering reverses, `adaptive`
spending its extra 33 bins on variance to model a shape 20 bins already cover.

`static_bins` is the right choice when the data are smooth, roughly unimodal and
already on a comparable scale, and when you want a number that does not depend on
$n$ — a detector fitted on a 200-row calibration window and then applied to live
traffic should not change its notion of resolution because the window grew.
`dynamic_bins` is the sane default: it adapts to $n$ and to each feature's robust
spread at no measurable cost, and it degrades gracefully. `adaptive` is for data
whose density varies by orders of magnitude across its range — a mixture, a
lognormal, a count — where equal-width edges are structurally wrong.

On the lognormal feature in the demo the three rules choose 10, 45 and 22 bins, and
the fixed count puts 68% of the sample in its largest bin against 5% for quantile
bins. The leave-one-out criterion behind `adaptive` is $-1.4572$ at $k = 16$,
$-1.4571$ at $k = 24$ and $-1.4638$ at $k = 48$: the optimum is flat, which is why
the variant can land anywhere inside it and still be right.

## Derivation

### 1. From the product model to the score

Under independence the joint density factorises, so

$$
-\log p(x) = \sum_{j=1}^{d} -\log p_j(x_j)
$$

and replacing each marginal by its histogram estimate gives the score. The
estimator itself is the usual one, with two floors that matter:

$$
\hat p_j(v) = \frac{\max(N_b, 1)}{n\, w_b},
\quad v \in \text{bin } b
\qquad (1)$$

$$
\mathrm{HBOS}(x) = \sum_{j=1}^{d} -\log \hat p_j(x_j)
\qquad (2)$$

The $\max(N_b, 1)$ turns an empty bin from an infinite score into a finite one. It
is a modelling decision rather than a numerical guard: a region with no training
points is not infinitely improbable, only unobserved, and one pseudo-count says so.
The demo shows the consequence in the degenerate case — a constant column gets one
bin holding everything, so its density is exactly 1 and its contribution exactly 0,
which is the right answer for a feature that cannot distinguish anything.

Values outside the fitted range have no bin at all. They get half a point in the
adjacent edge bin,

$$
\hat p_j(v) = \frac{1}{2\,n\,w_{\text{edge}}}, \qquad v \notin [\min x_j, \max x_j]
\qquad (3)$$

which is "less plausible than anything observed, still finite". The number is
arbitrary and the arbitrariness is visible: with `n_bins=1` every training point
scores 3.76 and a point outside scores 16.91, exactly $3\log 80 = 13.15$ more — the
same whether it is one unit or a thousand beyond the edge. That is what a
histogram knows about the tail: nothing.

### 2. Why the chi-square law fails

The paper's calibration argument is that under the null each $-\log \hat p_j(x_j)$
is exponential with rate one, so twice the score is a sum of $d$ chi-squares with
two degrees of freedom each:

$$
2\,\mathrm{HBOS}(x) \sim \chi^2_{2d}
\qquad (4)$$

This is not a theorem, and the reason is dimensional. The mean of $-\log \hat p_j$
is the differential entropy of the marginal plus a discretisation term:

$$
\mathbb{E}\left[-\log \hat p_j(X_j)\right] = H_{\text{bin}} + \log w
= h(p_j) + O(w^2)
\qquad (5)$$

with $H_{\text{bin}} = -\sum_b p_b \log p_b$ the entropy of the binned variable.
For a standard normal $h = \tfrac12 \log(2\pi e) = 1.4189$, and ten bins over the
sample range measure 1.443 — the $O(w^2)$ term from $\hat p$ being a step function.
So each term has mean 1.443, not the 1 that a unit-rate exponential has, and the
score is centred $0.443\,d$ above where (4) puts it. The threshold allows only
$1.645\sqrt{d}$ of tail room above a mean of $d$, so that offset eats a growing
share of the budget as $d$ rises — and the variance is not chi-square's either,
since each term is a handful of atoms rather than an exponential. The result is
measured: at a nominal 5%, `static_bins` flags 5.9% of held-out points in two
dimensions, 10.5% in four and 20.7% in eight.

There is a stronger objection, and the demo makes it in one line of arithmetic.
Density has units, so negative log density does too:

$$
\hat p_j(v/s) = s\,\hat p_j(v)
\;\Longrightarrow\;
\mathrm{HBOS}(x \text{ after } x_j \mapsto x_j/s) = \mathrm{HBOS}(x) + \log s
\qquad (6)$$

Multiply one feature by 1000 and every score rises by exactly $\log 1000 = 6.91$
nats, measured to $10^{-6}$, while the threshold in (4) is a constant. The demo's
detector then flags 100% of the same data. No threshold that does not move with the
score can be calibrated for all data, and a constant cannot move. This is the real
content of (4): it is a rule of thumb that happens to land near the right answer
when the marginals have entropy near one *in the units you happen to use*.

The fix costs twenty lines and is exact where (4) is approximate. Under the fitted
model a point lands in bin $b$ of feature $j$ with probability $N_b/n$ and
contributes a known quantity there, so the null distribution of (2) can be sampled
without any data:

```python
        value = -np.log(np.maximum(model.counts_[j], 1) / (model.n_ * widths))
        mass = model.counts_[j] / model.n_
        total += value[rng.choice(len(mass), size=reps, p=mass)]
```

That threshold is calibrated by construction — 4.9%, 7.1% and 9.3% at $d$ = 2, 4,
8 instead of 5.9%, 10.5% and 20.7% — and it shifts with the units because the score
it is derived from does. The residue is real and does not vanish with $n$: at $d=8$
it falls from 7.1% at $n = 500$ to 5.3% at $n = 2000$ and then sits at 5.5% at
$n = 8000$, because what is left is the gap between a ten-bin histogram and the
Gaussian it is modelling, not estimation error. The chi-square rate is 19.4% at
every $n$: it is not measuring the model at all.

### 3. The bin counts

`dynamic_bins` is the Freedman–Diaconis rule, which is the same rule the histogram
atom uses. The width that minimises integrated squared error for a smooth density
is of order $\mathrm{IQR}\, n^{-1/3}$, so

$$
k_j = \left\lceil \frac{\max x_j - \min x_j}{2\,\mathrm{IQR}_j\, n^{-1/3}} \right\rceil
\qquad (7)$$

clipped to $[1, \texttt{max\_bins}]$. The rule is invariant to the units of the
feature, which is the point: it expresses a width in units of the data's own robust
spread. It is also blind to shape beyond the IQR, which is why a peaky sample gets
a count that hits the cap and a flat one gets a handful.

`adaptive` maximises a leave-one-out likelihood. For a histogram this is free of
resampling, because removing point $i$ from its own bin changes that bin from $N$
to $N-1$ and nothing else:

$$
\hat p_j^{(-i)}(x_i) = \frac{N_{b(i)} - 1}{(n-1)\, w_{b(i)}}
\qquad (8)$$

so the criterion is

$$
k_j = \arg\max_k \; \frac{1}{n}\sum_{i=1}^{n} \log \hat p_j^{(-i)}(x_i)
\qquad (9)$$

evaluated over every $k$ from 2 to $\min(\texttt{max\_bins}, \lfloor n/2 \rfloor)$
(floored at three), with the edges placed at quantiles so each bin carries the same
mass. Two things follow. The count is chosen by the data rather than by a formula,
and the placement follows the density rather than the range — on a lognormal, quantile edges put 5% of the sample in the
largest bin against 68% for equal widths.

The criterion has a subtlety that is easy to miss. (8) is exact, but the edges were
chosen using every point including $i$, so the criterion is not a held-out estimate
of the model's performance; it removes the count's dependence on the point and not
the edges'. That is why the optimum is flat and why `adaptive` can lose to a fixed
count on data where a fixed count is already right.

## Code walkthrough

### 1. Scoring one feature

```python
            index = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                            0, k - 1)
            # Equation (1): never a zero density, so one point in a bin is
            # enough to make it plausible and an empty bin is not a wall.
            density = np.maximum(counts[index], 1) / (self.n_ * widths[index])
            outside = (X[:, j] < edges[0]) | (X[:, j] > edges[-1])
            # A value beyond the fitted range is worth half a point in the edge
            # bin: less plausible than anything observed, still finite.
            edge = np.where(X[:, j] < edges[0], 0, k - 1)
            density = np.where(outside, 0.5 / (self.n_ * widths[edge]), density)
            out[:, j] = -np.log(np.maximum(density, FLOOR))
```

The whole method is these seven lines, repeated per feature. `searchsorted` with
`side="right"` reproduces `np.histogram`'s convention that the last bin includes
its right edge, which is why the counts stored at fit time and the indices
computed at score time agree; getting the side wrong silently sends every value
that equals an internal edge into the bin above and skews the score. The clip is
needed because a value exactly equal to the largest edge sorts past the last bin.
`FLOOR` is a guard rather than part of the method: no reachable density is zero,
since (1) floors the count, so it cannot fire unless a bin width has collapsed —
and one `np.maximum` is cheaper than an argument about whether that is impossible
after a future edit.

### 2. The leave-one-out criterion

```python
    counts, _ = np.histogram(x, edges)
    widths = np.diff(edges)
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    held = counts[index] - 1
    if np.any(held <= 0):
        return -np.inf
    density = held / ((n - 1.0) * widths[index])
    return float(np.mean(np.log(density)))
```

Equation (8), one line per term. The `held <= 0` guard is doing real work: a bin
with a single point gives its occupant zero held-out density and a log of $-\infty$,
so any partition that isolates a point is rejected outright. That is what stops the
criterion from running to $k = n/2$ and declaring a perfect fit, and it is why the
optimum in the demo is an interior $k$ rather than the largest allowed one. Note
that the denominator is $n-1$, not $n$: every point is scored against a histogram
built from the other $n-1$.

### 3. Choosing the count

```python
        best_score, best = -np.inf, _frequency_edges(column, 2)
        for k in range(2, min(self.max_bins, max(3, len(column) // 2)) + 1):
            edges = _frequency_edges(column, k)
            score = _loo_score(column, edges)
            if score > best_score:
                best_score, best = score, edges
        return best
```

A linear scan over every allowed count, which is affordable because each evaluation
is one `np.histogram` and one vectorised logarithm. The initial `best` matters when
every count scores $-\infty$ — with three distinct values no $k$ can satisfy the
guard — in which case the two-bin quantile split is returned rather than an
unbound variable. The upper limit $n/2$ is a sanity bound: beyond it bins are
guaranteed to be nearly empty. Strict improvement (`>`) rather than `>=` makes the
choice deterministic at the flat part of the optimum, and it picks the smallest $k$
among ties, which is the conservative direction.

### 4. The threshold

```python
    def _threshold(self) -> float:
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        dof = 2 * self.n_features_
        return float(_chi2_ppf(1.0 - self.alpha, dof) / 2.0)
```

Two regimes. `contamination` is a quantile of the training scores and therefore
flags at most the requested share of any sample, clean or not — 4.8% here, where
ties at the quantile resolve outward. `contamination=None` asks for the
distribution-free threshold of (4), which the demo shows is neither
distribution-free nor a threshold. Both are supplied because the honest empirical
option makes no claim that can be wrong, and the analytic option is what the
literature uses.

## Complexity

For $n$ points, $d$ features, $m$ queries, $k$ bins and $K$ = `max_bins`:

| stage | time | memory |
|:--|:--|:--|
| `static_bins` / `dynamic_bins` fit | $O(nd)$ | $O(dk)$ |
| `adaptive` fit | $O(Knd)$ | $O(dK)$ |
| score $m$ points | $O(md\log k)$ | $O(m)$ |

Fitting is a pass over the data per feature, plus one sort for quantile edges. The
`adaptive` factor of $K$ is the whole extra cost, and it is why the default
`max_bins` is 64 rather than $n/2$: the criterion is flat near its optimum, so
searching further buys nothing and costs linearly. Scoring is a binary search per
feature per point, so it is logarithmic in the bin count and linear in the
dimension: the same order as a Euclidean distance, and a factor of $n$ cheaper than
a kernel density estimate, which is the entire reason the method exists. Returning
the per-feature terms rather than the sum costs no extra time and $O(md)$ instead
of $O(m)$ memory — usually the better trade, since the breakdown is the only
explanation a product model can offer.

The memory figure for `adaptive` counts the stored edges and counts; the search
itself reuses one histogram at a time.

## Failure modes

### Correlation blindness

The demo's headline. Two features correlated at 0.971, anomalies on the
anti-diagonal at radius 0.8 to 1.5, and the AUC is 0.365 to 0.474 — the planted
points score *lower* than the ordinary ones and none of the forty is flagged at a
10% budget. This is not a tuning problem, it is the product model working exactly
as specified: the score never sees the pair. Any dataset with real dependence
between features — which is most of them — will contain anomalies of this shape,
and they are precisely the anomalies that a naive detector misses. If the
dependence matters, transform first (PCA or a whitening step), or use a joint
method.

### The threshold is not the one in the paper

`contamination=None` uses (4), and the demo shows what that is worth: 5.9% at
$d=2$, 20.7% at $d=8$ on clean standard normal data at a nominal 5%, and 100% after
rescaling one feature by 1000. The score's mean is the marginal's differential
entropy, not 1, and the score has units. Both defects are structural and neither
shrinks with $n$. Use `contamination`, or resample the null from the fitted model.

### Resolution versus shape

A fixed count is a fixed width, and a width that suits a Gaussian is wrong for a
mixture. On the demo's peaky sample `static_bins` costs 3.02 nats per held-out
point against 0.94 for quantile bins, a factor of eight in likelihood. The symptom
is a score that is nearly constant over a large region — the mode is one bin, so
every point near it is tied — followed by a cliff between adjacent bins. Ties are
also the mechanism by which a fine histogram overfits: with $n$ points and $k$
bins, $k \gtrsim n/10$ leaves about ten counts per bin on average and one or two in
the tail bins, so the density estimate there is a single observation.

### Contamination stretches the range

Fitting on data that contains a few far outliers widens every edge, so every
density falls and the resolution available to the bulk collapses: the good data is
compressed into a handful of bins while empty space gets bins of its own. HBOS has
no notion of robustness — the fit is a set of counts, and a count is an average.
The practical remedy is to fit on a window known to be clean, or to clip the
feature at robust quantiles before fitting, and to remember that the score is
therefore conditional on that window.

### Saturation outside the range

Every point beyond the fitted range gets the same score, (3), whatever its
distance. That is one of the reasons HBOS is a triage tool rather than a
measurement: it tells you a coordinate went somewhere unseen, not how far. It also
means a drifted feature — one whose whole distribution has moved — produces a
constant, undifferentiated alarm rather than a graded one.

### Degenerate and discrete features

A constant column contributes exactly zero, which is correct and also means it
silently disappears from the score. A feature with few distinct values gets few
quantile bins, because duplicate edges are removed — four distinct values repeated
ten times each occupy four bins — and `adaptive` narrowing itself is the right
response rather than a division by a zero-width bin. But it does mean the effective
dimension can be smaller than `X.shape[1]`, and a score threshold calibrated on
`n_features_` may be wrong about the number of terms that actually vary.

## Applications

- **Network intrusion detection.** HBOS was proposed for exactly this: many
  features, a need for speed, and a stream where the model must be refitted often.
  Per-feature histograms are cheap to update and the score is additive, so a stale
  fit degrades one coordinate at a time.
- **Credit and fraud triage.** Each feature's contribution is interpretable —
  "this transaction is unusual in amount and ordinary in everything else" — which
  matters more in a regulated setting than a slightly better AUC.
- **Manufacturing and sensor monitoring.** Independent channels, each with its own
  scale, is the setting the method was designed for; `dynamic_bins` handles the
  differing spreads without any per-channel tuning.
- **A first pass over high-volume logs.** Aggregate counts, latencies and payload
  sizes per entity, score every entity, and send the top fraction to something
  expensive. Linear cost is the whole point.
- **Feature attribution inside a bigger detector.** `feature_scores` returns the
  per-feature terms, so HBOS is often the explainable layer wrapped around, or
  underneath, a joint model — the terms say which coordinate moved.
- **Benchmark baseline.** Goldstein and Uchida score it against much slower
  multivariate detectors, and it is the reference anything more elaborate has to
  beat on independent features — which is the profile this atom derives.
- **Where it is a trap.** Correlated or redundant features, strong interaction
  effects, anomalies defined by combination rather than by magnitude, and any
  threshold taken from (4) without checking the units.

## References

- M. Goldstein and A. Dengel, *Histogram-based outlier score (HBOS): a fast
  unsupervised anomaly detection algorithm*, KI-2012 Poster and Demo Track (2012).
  The method, the fixed ten-bin default, and the chi-square calibration of (4).
- D. Freedman and P. Diaconis, *On the histogram as a density estimator: $L^2$
  theory*, Zeitschrift für Wahrscheinlichkeitstheorie 57:453–476 (1981). The width
  behind (7).
- D. W. Scott, *Multivariate Density Estimation: Theory, Practice, and
  Visualization*, 2nd ed., Wiley (2015). Bin-count rules, and why the count grows
  like $n^{1/3}$.
- L. Birgé and Y. Rozenholc, *How many bins should be put in a regular histogram*,
  ESAIM: Probability and Statistics 10:24–45 (2006). Penalised-likelihood counts,
  and the degeneracy that the guard in (8) is standing in for.
- L. Wasserman, *All of Nonparametric Statistics*, Springer (2006). Histogram
  entropy and the discretisation term in (5).
- F. T. Liu, K. M. Ting and Z.-H. Zhou, *Isolation forest*, ICDM (2008). The
  random-split alternative, and the atom in this chapter with the opposite failure
  profile.
- M. Goldstein and S. Uchida, *A comparative evaluation of unsupervised anomaly
  detection algorithms for multivariate data*, PLoS ONE 11(4):e0152173 (2016).
  Benchmarks HBOS against considerably slower multivariate detectors, and measures
  where the independence assumption costs it.
- C. C. Aggarwal, *Outlier Analysis*, 2nd ed., Springer (2017). The product-model
  assumption in the wider taxonomy of detectors.
- Z. Li, Y. Zhao, X. Hu, N. Botta, C. Ionescu and G. Chen, *ECOD: unsupervised
  outlier detection using empirical cumulative distribution functions*, TKDE
  (2023). The successor in this chapter: same per-feature decomposition, no bins.
