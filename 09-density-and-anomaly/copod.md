# COPOD — the empirical copula, one marginal at a time

`copod.py` · Density & anomaly · Tier 3 · 3 variants

## Intuition

Sklar's theorem says that a joint distribution is its marginals plus a copula,
and that the copula is the part which carries the dependence. Estimating a
copula is hard. COPOD makes the bravest possible assumption about it — the
copula is the independence copula — and then everything becomes counting:

$$
p(x_1,\dots,x_p) \approx \prod_{j=1}^{p} p_j(x_j),
\qquad
-\log p(x) \approx \sum_{j=1}^{p} -\log \hat F_j(x_j).
$$

There is no bandwidth, no tree depth, no distance metric, no learning rate. The
model is the empirical CDF of each column and the score is how improbable a
point is under those CDFs. It is the simplest multivariate density estimator in
this chapter, and on tabular data it is embarrassingly hard to beat.

The picture is one histogram per feature, with the score read off the tail:

```
  density
    │        ▁▁▁
    │     ▁▁▁   ▁▁▁
    │  ▁▁▁         ▁▁▁
    └──┬───────┬───────┬──────► x
       q01     med     q99
       ↑               ↑
    score 4.6       score 4.6
```

A point in either tail is surprising, and the score says how surprising. The
symmetry in that picture is the whole design: the 3rd smallest value and the
3rd largest get the same score, because both are three counts from an end.

That symmetry is wrong when the distribution is skewed, because the two tails
do not carry the same kind of surprise: a lognormal is bounded below and
unbounded above, so a point just under the minimum is impossible while a point
far above the maximum is merely rare. The skewness correction tries to encode
that. The demo shows both what it buys and what it costs, on a beta$(2, 5)$
sample with a probe beyond each end:

| variant | score at $-0.5$ | score at $1.5$ | clean max |
|:--|--:|--:|--:|
| `copod` | 3.546 | 3.546 | 1.621 |
| `copod_skew` | 3.546 | 7.092 | 3.200 |
| `ecod` | 7.092 | 7.092 | 3.200 |

The correction does not suppress the short tail; it *doubles* the long one.
That is a prior about where anomalies live, and here the prior is arbitrary:
both probes lie outside the support of the beta sample, so $-0.5$ is exactly as
impossible as $1.5$ and gets half the weight. On a lognormal the same prior
would be right, because the left probe is impossible and the right one merely
rare. ECOD keeps the maximum of both tails at full weight and is the safer
default.

The structural limitation is independence, and it is not a small one. On a
Gaussian with correlation $0.95$, the points $(2.5, 2.5)$ and $(2.5, -2.5)$ have
identical marginal coordinates, so every variant gives them the *same* score to
machine precision:

| variant | score $(2.5,2.5)$ | score $(2.5,-2.5)$ | clean 99th pct |
|:--|--:|--:|--:|
| `copod` | 5.087710 | 5.087710 | 5.642 |
| `copod_skew` | 10.175420 | 10.175420 | 11.283 |
| `ecod` | 10.175420 | 10.175420 | 11.283 |

The true squared Mahalanobis distance is $6.410$ for the first and $250$ for the
second. Neither is flagged, because along the ridge a 2.5-sigma marginal is
ordinary: the clean 99th percentile is higher than both. One probe is a normal
draw from the joint and the other is impossible, and the independence copula
cannot tell which is which.

## Variants

| variant | per-feature score | idea |
|:--|:--|:--|
| `copod` | $\tfrac{1}{2}(u_L + u_R)$ | charge both tails equally |
| `copod_skew` | $\max(\text{skew term},\ \tfrac{1}{2}(u_L+u_R))$ | amplify the long tail |
| `ecod` | $\max(u_L, u_R, \text{skew term})$ | keep both tails at full weight |

**`copod`** is the symmetric estimator: the average of the two negative
log-tail probabilities. It is the one to reach for when the marginals are
roughly symmetric, and it is the baseline that any correlation-aware method has
to beat.

**`copod_skew`** adds the skewness term. The sign of the sample skewness picks
the tail — right-skewed features score on the right, left-skewed on the left —
and the score is the maximum of that term and the symmetric one. It therefore
never lowers a score; it only amplifies points in the long tail. That is a
prior, and priors are sometimes wrong.

**`ecod`** is the same three terms combined differently: the maximum of the two
tail scores and the skew term, per feature. It keeps the short tail at full
weight, which is what saves it in the skewed demo. On perfectly symmetric data
`ecod` and `copod_skew` coincide, because the zero-skew branch sums both tails
and dominates both. ECOD is the recommended default of the two published
methods; COPOD is kept for the historical record and because its symmetric
score is easier to reason about.

## Derivation

### The empirical marginal

For feature $j$ with training values $x_{1j},\dots,x_{nj}$, the empirical CDF
is the fraction of the sample at or below a value. COPOD needs the two tails
separately, and it smooths both by half a count so that neither is ever zero:

$$
L_j(x) = \frac{\#\{i : x_{ij} < x\} + \tfrac{1}{2}}{n+1},
\qquad (1)
$$

$$
R_j(x) = \frac{\#\{i : x_{ij} > x\} + \tfrac{1}{2}}{n+1}.
\qquad (2)
$$

The denominators are $n+1$, not $n$, and that is what makes the smoothing
coherent: the three pieces — strictly below, exactly equal, strictly above —
partition the sample with weights $(\text{below}+\tfrac12)$, the atom at the
point, and $(\text{above}+\tfrac12)$, which sum to $n+1$. For a value that
occurs once, the atom has weight $1/(n+1)$, exactly the spacing between
consecutive plotting positions. In code the two counts are two binary searches:

```python
    n = sorted_values.size
    below = np.searchsorted(sorted_values, x, side="left")
    above = n - np.searchsorted(sorted_values, x, side="right")
    return (below + 0.5) / (n + 1.0), (above + 0.5) / (n + 1.0)
```

`side="left"` counts strictly less; `side="right"` counts at most, so $n$ minus
that is strictly greater. The half-count is the difference between a score of
infinity at the sample maximum and a score of $\log(2(n+1))$, and it costs one
constant per feature.

### The symmetric score

The independence assumption turns the joint tail probability into a product, so
the negative log is a sum. COPOD's symmetric version averages the two tails:

$$
s(x) = \sum_{j=1}^{p} -\frac{1}{2}\Big[\log L_j(x_j) + \log R_j(x_j)\Big].
\qquad (3)
$$

The average, rather than the sum, is what makes the score comparable between a
point that is extreme on the left and one that is extreme on the right: a point
at the sample minimum has $L = \tfrac{1}{2}(n+1)^{-1}$ and $R \approx 1$, and
the average is half the surprisal of the tail it is in.

The expectation has a closed form. Under any continuous marginal the
probability integral transform makes $L_j(X_j)$ uniform on $(0,1)$, and
$\mathbb{E}[-\log U] = \int_0^1 -\log u\,du = 1$. So the mean score is exactly
the number of features, whatever the marginals are. The demo checks it:

| $p$ | mean `copod` | mean `copod_skew` | mean `ecod` | `copod` 99th pct |
|--:|--:|--:|--:|--:|
| 1 | 1.000 | 1.347 | 1.693 | 2.628 |
| 2 | 2.001 | 2.694 | 3.387 | 4.023 |
| 4 | 4.001 | 5.387 | 6.773 | 6.492 |
| 8 | 8.003 | 10.775 | 13.546 | 11.379 |
| 16 | 16.005 | 21.549 | 27.093 | 20.849 |

The `copod` column is the theorem. The other two are proportional to it: the
maximum of two or three terms exceeds their average by a factor that depends
only on the variant, not on the dimension, so over this table `copod_skew` is
$1.347\,p$ and `ecod` is $1.693\,p$ to three decimals. The constants are the
expected maximum of those terms after the probability integral transform, and
they are what a threshold fitted on one variant has to account for.

### Skewness and the two corrections

The sample skewness of feature $j$ is the third standardised moment,

$$
b_j = \frac{m_{3j}}{m_{2j}^{3/2}},
\qquad m_{kj} = \frac{1}{n}\sum_i (x_{ij} - \bar x_j)^k,
\qquad (4)
$$

with $b_j = 0$ for a constant feature. A positive $b_j$ means a long right tail,
so the tail where anomalies are expected is the right one, and the skew term is
$u_R$; a negative $b_j$ gives $u_L$. When $b_j$ is exactly zero the term sums
both tails, which is the published convention and the reason `copod_skew` and
`ecod` agree on symmetric data.

The two methods then differ only in how they combine the three numbers:

$$
s_{\text{copod-skew}}(x) = \sum_j \max\Big(u_{T_j j},\; \tfrac{1}{2}(u_{Lj}+u_{Rj})\Big),
\qquad T_j = R \text{ if } b_j > 0, \quad L \text{ if } b_j < 0.
\qquad (5)
$$

$$
s_{\text{ecod}}(x) = \sum_j \max\big(u_{Lj},\; u_{Rj},\; \text{skew term}_j\big).
\qquad (6)
$$

When $b_j = 0$ the chosen term is the sum $u_{Lj}+u_{Rj}$, which is why the two
variants agree exactly on symmetric data. Both are sums of per-feature maxima,
so both are at least the symmetric score: the short tail is never ignored, only
charged at the symmetric rate. What the skew sign changes is the long tail, and
the demo's sample makes the difference visible — the right-tail probe scores
$7.092$ and the left-tail probe $3.546$ under `copod_skew`, while `ecod` gives
both $7.092$.

The sign test is fragile in one specific way. A skewness of $10^{-17}$ is
rounding noise, and its sign is meaningless, but it is still a sign. The
implementation treats $|b_j| < 10^{-12}$ as zero:

```python
    safe = np.where(m2 > 0.0, m2, 1.0)
    b = np.where(m2 > 0.0, m3 / safe ** 1.5, 0.0)
    # the sign of a skewness of 1e-17 is rounding noise, not a shape
    return np.where(np.abs(b) < 1e-12, 0.0, b)
```

Without that guard, the antithetic training sample in the correlation demo —
which is exactly symmetric by construction — gets one column labelled "right"
and the other "left" from floating-point dust, and the two probes stop having
equal scores.

### What the score is, and what it is not

The score is a sum of surprisals, not a log-density. Under independence the two
would coincide, but the marginals are empirical, so the score is a *rank*:
it says how many sample points are less extreme, not how much probability mass
is out there. Three consequences follow, and all three are failure modes below.

1. The score has no units. A score of 6 means something different at $p=2$ and
   at $p=16$, and the threshold is a quantile of the training scores, not a
   probability.
2. The score saturates outside the training range, because the empirical CDF
   has no data there.
3. The score cannot see dependence, because it never looks at two features at
   once.

## Code walkthrough

### Per-feature contributions

The whole estimator is one loop over columns, and the loop is written so that
the per-feature matrix is the return value. Summing it is scoring; reading a row
is explaining a point.

```python
            left, right = _tail_probabilities(self.sorted_[j], X[:, j])
            u_left = -np.log(left)
            u_right = -np.log(right)
            symmetric = 0.5 * (u_left + u_right)
            if self.variant == "copod":
                per_dimension[:, j] = symmetric
                continue
```

`self.sorted_` is a list of sorted columns built once at fit time. Scoring is
therefore $p$ binary searches of $m$ queries each, with no re-sorting and no
state that depends on the batch — the same point gets the same score whether it
arrives alone or in a thousand-row batch. That sounds obvious, and it is not:
the reference implementation concatenates the training set with the test batch
before computing the ECDF, which makes a point's score depend on which other
points were scored with it.

### Choosing the tail

```python
            skew = self.skew_[j]
            if skew < 0.0:
                skew_term = u_left
            elif skew > 0.0:
                skew_term = u_right
            else:
                skew_term = u_left + u_right
```

Three branches, one per sign. The zero branch is the surprising one: it sums
rather than averages, so a symmetric feature contributes twice its symmetric
score to `copod_skew` and to `ecod`. That is faithful to the published
algorithm and it is why the two variants coincide on symmetric data, but it
also means the score jumps discontinuously as the skewness crosses zero. In
practice the crossing happens only for data that are symmetric by construction;
for real data $b_j$ is never exactly zero.

### Combining

```python
            if self.variant == "copod_skew":
                per_dimension[:, j] = np.maximum(skew_term, symmetric)
            else:
                per_dimension[:, j] = np.maximum(
                    np.maximum(u_left, u_right), skew_term)
```

Two lines separate the published methods. `copod_skew` takes the better of the
skew term and the symmetric average; `ecod` takes the best of all three. Since
the skew term is one of the two tails (their sum when $b_j = 0$), `ecod`'s
maximum is the larger tail score, and the skew sign becomes irrelevant to the
result — which is exactly the robustness ECOD was designed for.

### Fitting and the threshold

Fitting is a sort followed by a self-score, and the threshold is a quantile of
the scores the model gives its own training data:

```python
        self.scores_ = self.score_samples(X)
        self.threshold_ = float(np.quantile(self.scores_,
                                            1.0 - self.contamination))
```

Scoring the training set through the same code path as a test batch is what
makes the score a pure function of the point and the sorted columns. It also
fixes the meaning of `contamination`: the threshold is the $1-c$ empirical
quantile, so exactly a fraction $c$ of the training set is labelled `-1`, by
construction, whether or not any of it is anomalous. That is why
`contamination=0.1` on a clean training set gives ten percent false positives,
and why the parameter is a budget rather than a test.

`predict` compares with `>` rather than `>=`, so points exactly at the quantile
stay normal. With continuous scores that is a measure-zero tie; with discrete
data it decides how a whole tie group is labelled, and the strict comparison
keeps the flagged fraction at or below the budget.

## Complexity

Let $n$ be the training size, $m$ the number of points scored and $p$ the
number of features.

| operation | time | memory |
|:--|:--|:--|
| fit (sort each column) | $O(np\log n)$ | $O(np)$ |
| score | $O(mp\log n)$ | $O(mp)$ |
| per-feature explanation | $O(mp\log n)$ | $O(mp)$ |

Sorting dominates the fit and the binary searches dominate scoring. There is no
iterative step, no random seed and no hyperparameter except the threshold, so
the estimator is deterministic by construction — a property that is rare in
this chapter and worth stating explicitly. The $O(mp)$ memory for the
per-feature matrix can be avoided by accumulating the row sums, at the cost of
losing the explanation; the reference keeps it because $p$ is usually small.

## Failure modes

**Dependence is invisible.** The demo's two probes have the same marginals and
therefore the same score, while their true squared Mahalanobis distances are
$6.4$ and $250$. Worse, neither is flagged, because the clean 99th percentile
along the ridge is higher than a 2.5-sigma marginal point. If the features are
strongly correlated — sensor arrays, prices, pixels — COPOD is the wrong
estimator, and the fix is a method that models the joint: `mahalanobis`,
`isolation_forest`, or a copula that is not the independence one.

**The score saturates outside the range.** Every point beyond the training
maximum scores $3.200960$ in the demo, and the maximum itself scores $3.202627$.
"Largest value ever seen" and "a thousand times larger" are the same event. If
the anomalies you care about are the ones just past the edge, this is fatal;
tail extrapolation is the `extreme_value` atom, and a held-out calibration set
is the cheaper fix.

**The score scales with the feature count.** The mean is exactly $p$ on uniform
data, and the 99th percentile grows from $2.628$ at $p=1$ to $20.849$ at $p=16$.
A threshold fitted on a ten-feature model is meaningless on a twenty-feature
one, and adding a noisy feature raises every score, including the clean ones.

**The skew sign is a guess.** `copod_skew` doubles the long tail and leaves the
short tail at the symmetric value. On a bounded-below distribution the short
tail is where the impossible points are, so the correction gives them half the
weight of points that are merely far out. ECOD's max-of-both is the safer
default; neither is right for every distribution.

**Constant features contribute a constant.** A column with no variance has
$b_j = 0$ and $L = R = \tfrac{1}{2}(n+1)^{-1}$ for every training point, so it
adds $\log(2(n+1))$ to every score and changes nothing. A *test* point that
differs from the constant gets $u_L = 0$ and $u_R = \log(2(n+1))$, so the
feature suddenly becomes a perfect detector. Drop constant columns before
fitting.

**The threshold is a quantile, not a probability.** `contamination=0.1` puts
the cut at the 90th percentile of the training scores, so by construction ten
percent of the training set is labelled anomalous. On a clean training set that
is ten percent false positives. The number is a budget, and it has to be chosen
from the cost of a false alarm, not from the data.

**Ties inside the range collapse ranks.** The half-count smoothing gives every
point with the same value the same tail probability, which is correct but means
a large tied group — a discretised feature, a sensor with few levels — cannot be
ranked within itself. The score is still finite and monotone in the value; it
just stops discriminating inside the tie.

## Applications

**Tabular anomaly detection.** This is the default first model for a table of
mixed features, precisely because it has nothing to tune. Fit it, look at the
per-feature scores for the top rows, and you have an explanation as well as a
ranking.

**Interpretable alerting.** `feature_scores` returns the contribution of each
column, so a flagged row can be reported as "flagged because feature 7 is at the
0.1st percentile and feature 12 at the 0.3rd" rather than as an opaque score.
That is the property that gets a model deployed in an operations setting.

**Baselines for density estimators.** Any multivariate density model — `kde`,
`gaussian_mixture`, a normalising flow — should be compared against the
independence copula. If it does not beat COPOD on your data, the dependence it
models is not worth the parameters.

**Feature screening.** The per-feature score matrix is a cheap measure of which
columns carry the anomaly signal. A column whose contribution is flat across the
flagged rows is not doing any work, and can be dropped before a heavier model is
fitted.

**Streaming and edge deployment.** Fitting is a sort and scoring is a binary
search, so the model is a few arrays of sorted values. There is no iterative
solver to time out and no floating-point accumulation to drift, which makes it a
reasonable choice where the model has to run on the device that produces the
data.

## References

- Li, Z., Zhao, Y., Botta, N., Ionescu, C. & Hu, X. (2020). COPOD: copula-based
  outlier detection. *IEEE International Conference on Data Mining (ICDM)*,
  1118–1123. The symmetric score and the skewness correction.
- Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. & Chen, G. (2022). ECOD:
  unsupervised outlier detection using empirical cumulative distribution
  functions. *IEEE Transactions on Knowledge and Data Engineering* 35(12),
  12181–12193. The max-of-three formulation and the argument for it.
- Sklar, A. (1959). Fonctions de répartition à $n$ dimensions et leurs marges.
  *Publications de l'Institut de Statistique de l'Université de Paris* 8,
  229–231. The theorem the method is named after.
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd edition. Springer. The
  independence copula, and why it is the simplest one.
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press. What a real
  copula costs to estimate, which is the context for COPOD's assumption.
- Aggarwal, C. C. & Yu, P. S. (2001). Outlier detection for high dimensional
  data. *ACM SIGMOD*, 37–46. The dimension-scaling problem that the additive
  score inherits.
- Kriegel, H.-P., Kroger, P., Schubert, E. & Zimek, A. (2009). Outlier detection
  in axis-parallel subspaces. *PVLDB* 2(1), 253–264. Why marginal scores and
  joint structure disagree, which is the correlation failure mode.
