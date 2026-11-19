# Mahalanobis distance and robust covariance

`mahalanobis.py` · Density & anomaly · Tier 2 · 4 variants

## Intuition

Euclidean distance treats every direction as equally surprising. Mahalanobis
distance asks how far out a point is *along the direction it lies in*, measured
in the standard deviations the data actually have:

```
       x  ← far in x, but the data are wide here,
      /        so this is ordinary
     /
    ·  ·  · ·  ·        the ellipse is one standard
   ·  · · · ·  · ·      deviation of the sample
   · · ·  · · ·  ·
    ·  ·  · ·  ·
        \  y  ← close in y, but the data are thin
               here, so this is anomalous
```

Concretely: subtract the mean, rotate onto the covariance's eigenvectors, divide
by the standard deviation on each new axis, then measure length. Those four steps
are one quadratic form,

$$
d^2(x) = (x - \mu)^\top \Sigma^{-1} (x - \mu)
\qquad (1)$$

and for a Gaussian it is not an approximation — (1) is the squared exponent of the
density, so a level set of $d^2$ is an ellipsoid of constant probability. That is
also why the threshold is a real number rather than a tuning knob: under the model
$d^2 \sim \chi^2_p$, and the $95\%$ quantile is a fact about the dimensions, not
about the data.

The whole difficulty is $\Sigma$. The obvious estimator, the sample covariance,
has breakdown point zero. Move a point far enough and it stops looking far: the
mean travels toward it and the covariance widens along its direction until its
own distance is finite again. This is **masking**, and it is not a bug in the
arithmetic — it is the maximum-likelihood Gaussian doing exactly what it is asked,
since a distribution that has widened to cover the outlier now explains it. The
demo shows the extreme form. Four hundred points from a 6-dimensional Gaussian
plus a hundred packed into a ball of radius $0.3$ at distance 8 along one axis:

| variant | AUC | mean $d^2$, outliers | normals | flagged |
|:--|--:|--:|--:|:--|
| `hotelling_t2` | 0.340 | 4.43 | 6.36 | 0% of outliers, 12% of normals |
| `mcd` | 0.174 | 4.00 | 8.34 | 0% of outliers, 12% of normals |
| `ogk` | 1.000 | 82.23 | 6.11 | 50% of outliers, 0% of normals |
| `shrinkage` | 0.350 | 4.45 | 6.27 | 0% of outliers, 12% of normals |

The classical fit does not merely fail to find the outliers; it ranks them
*below* the data and then flags the data. The variants are four answers to
estimating $\Sigma$ when part of the sample is not from the model:

- `hotelling_t2` is the classical estimate with the exact small-sample
  correction, and the $F$ threshold that goes with it.
- `mcd` is the minimum covariance determinant: the smallest-volume ellipsoid
  covering a fixed fraction of the data, refitted on what it covers.
- `ogk` assembles $\Sigma$ from pairwise robust scales, with no subset search.
- `shrinkage` does not aim at robustness at all; it fixes singularity by pulling
  $\Sigma$ toward a scaled identity.

## Variants

| variant | $\hat\Sigma$ | breakdown | fixes | measured |
|:--|:--|--:|:--|:--|
| `hotelling_t2` | sample covariance | 0% | nothing; it is the reference | error 0.078 clean → 1.010 at 30% |
| `mcd` | min-det subset, reweighted | $1 - h/n$ | scattered contamination | error 0.141 → 0.134 |
| `ogk` | pairwise $Q_n$, reweighted | 50% | compact clusters too | error 0.073 → 0.104 |
| `shrinkage` | Ledoit–Wolf toward $\mu I$ | 0% | $n \le p$ singularity | $\lambda = 0.59$ at $p/n = 2$ |

The measured column is the relative Frobenius error of the fitted covariance
against a known truth, on 600 points in 6 dimensions with gross errors drawn from
a wide Gaussian: `hotelling_t2` 0.078 at no contamination and 1.010 at 30%, `mcd`
0.141 and 0.134, `ogk` 0.073 and 0.104, `shrinkage` 0.070 and 1.014. Read it as
two separate claims. `mcd` and `ogk` are insensitive to the contamination; the
other two are not, and shrinkage tracks the classical estimate so closely that it
is clearly not a robust estimator wearing a different hat.

`hotelling_t2` is the one to be precise about, because it is the variant with a
distributional result rather than a heuristic. If $x$ is a new observation and
$\bar x, S$ come from $n$ others, then $x - \bar x$ has covariance
$\Sigma(1 + 1/n)$, so the plain distance is inflated by that factor; dividing it
out gives Hotelling's statistic

$$
T^2(x) = \frac{n}{n+1}\, (x - \bar x)^\top S^{-1} (x - \bar x)
\qquad (2)$$

which follows an $F$ distribution exactly,

$$
T^2 \sim \frac{p(n-1)}{n-p}\, F_{p,\,n-p}
\qquad (3)$$

and the demo checks it. At a nominal 5%, averaged over 200 independent samples of
$n = 15$, $p = 8$, the exact threshold (3) flags 4.4% of held-out points, the
asymptotic $\chi^2_8$ threshold on the same statistic flags 51.6%, and the
$\chi^2_8$ threshold on the uncorrected $d^2$ flags 55.1%. At $n = 200$ the three
are 5.0%, 6.6% and 6.8%. The correction is not cosmetic: at $n = 15$, $p = 8$ the
asymptotic threshold is wrong by a factor of ten, because $\chi^2$ is the limit
in which $\Sigma$ is known, and here it is estimated from 15 points.

`mcd` replaces the mean and covariance with the ones from the half of the data
that is most tightly packed. Formally, over subsets $H$ of size $h$,

$$
(\hat\mu, \hat\Sigma) = \arg\min_{|H| = h} \det \hat\Sigma_H
\qquad (4)$$

with $h = \lceil \alpha n \rceil$ and $\alpha$ the `support_fraction`. The
determinant is a volume, so (4) is asking for the smallest ellipsoid that still
covers an $\alpha$ fraction of the sample; $1 - \alpha$ is the fraction of
contamination it can survive, and the default $\alpha = 0.75$ gives a nominal
breakdown of 25%. The demo's 20% compact cluster is inside that budget and MCD
still fails — see the failure modes — because the determinant is minimised by
*volume*, and a compact cluster of outliers has less volume than the good data.

`ogk` avoids the search. The identity
$\mathrm{Var}(u+v) - \mathrm{Var}(u-v) = 4\,\mathrm{Cov}(u,v)$ lets it estimate
every off-diagonal entry from two robust scales of one-dimensional projections,

$$
s_{jk} = \tfrac{1}{4}\left(\hat\sigma^2(x_j + x_k) - \hat\sigma^2(x_j - x_k)\right)
\qquad (5)$$

which costs $p(p-1)/2$ one-dimensional problems instead of a combinatorial
search, and has the 50% breakdown of the scale it uses. The pairwise matrix is
not guaranteed positive semidefinite (the $s_{jk}$ need not be a covariance), so
it is repaired by clipping negative eigenvalues, and the shape is then corrected
by the transformation in the walkthrough.

`shrinkage` is the regulariser. Ledoit–Wolf pulls the sample covariance toward a
scaled identity,

$$
\hat\Sigma = (1 - \lambda) S + \lambda \frac{\mathrm{tr}(S)}{p} I
\qquad (6)$$

with $\lambda$ estimated from the data rather than chosen. It never claims to be
robust — the demo's $\lambda$ on the masking data is 0.007, which is the sample
covariance plus a rounding error — and it is the only variant that returns an
answer when $n \le p$.

## Derivation

### 1. Why (1) and not some other distance

Let $S = U\Lambda U^\top$ and $W = \Lambda^{-1/2}U^\top$, so that rows of $XW^\top$
are the data whitened: their covariance is $I$. Then

$$
(x - \mu)^\top \Sigma^{-1} (x-\mu) = \lVert W(x - \mu) \rVert^2
$$

so Mahalanobis distance is Euclidean distance after whitening. Every property
follows from that. It is invariant to invertible linear maps — whitening either
version of the data gives the same $I$ — and the demo confirms it: a random
invertible matrix and a translation leave every score unchanged to $3 \times
10^{-14}$. It is *not* invariant to a change of units that is not a linear map of
the whole vector, which is why a column's scaling still matters through $\Sigma$.
And it degenerates to Euclidean when $\Sigma = I$, so nothing here contradicts the
simple case; it generalises it.

The identity that the demo checks to $10^{-10}$ is a useful sanity test for any
implementation. With the unbiased covariance and the sample mean,

$$
\sum_i (x_i - \bar x)^\top S^{-1}(x_i - \bar x)
= \mathrm{tr}\left(S^{-1} \sum_i (x_i-\bar x)(x_i-\bar x)^\top\right)
= \mathrm{tr}\left(S^{-1}(n-1)S\right) = (n-1)p
$$

so the mean training distance is exactly $p(n-1)/n$ — 2.9925 for the demo's
$n = 400$, $p = 3$ — whatever the data are. If a fit does not reproduce that
number, the covariance convention (divisor $n$ or $n-1$) is wrong, not the data.

### 2. The exact threshold

For a *new* point, $x - \bar x \sim N_p(0, \Sigma(1+1/n))$ independently of
$(n-1)S \sim W_p(\Sigma, n-1)$. The standard Hotelling result for a normal vector
independent of a Wishart matrix is

$$
z^\top S^{-1} z \cdot \frac{n-p}{p(n-1)} \sim F_{p,\,n-p}
$$

and applying it to $z = x - \bar x$, whose covariance is $\Sigma(1+1/n)$, gives
(2) and (3). The $\chi^2$ threshold

$$
d^2 > \chi^2_p(1 - \alpha)
\qquad (7)$$

is the limit of (3) as $n \to \infty$: $F_{p,\,q}$ converges in distribution to
$\chi^2_p / p$, so the product in (3) tends to $\chi^2_p$, and $n/(n+1) \to 1$. The
gap between them is the price of estimating $\Sigma$, and it is large exactly when
$p$ is a substantial fraction of $n$ — the demo's $n = 15, p = 8$ is more than
half, and 5% becomes 55%.

The calibration is also a warning about how to measure it. The $F$ law holds when
averaged over the estimation of $\Sigma$: for one fixed fit every test point shares
the same $\hat\Sigma$, so the flag rate of that fit depends on how it happened to
land, and a single fit measured on 200 000 points is one draw from the distribution
of flag rates rather than the nominal 5%. That is why the demo averages over 200
independent samples instead.

### 3. Why masking inverts the ranking

Take good data $N(0, I_p)$, and put a fraction $\epsilon$ in a tight cluster at
distance $r$ along $e_1$. The mean moves to $\epsilon r e_1$; the variance along
$e_1$ becomes $(1-\epsilon) + \epsilon r^2$, and along the other $p-1$ axes it is
$1-\epsilon$. An outlier at $r e_1$ is $(1-\epsilon)r$ from the mean, so

$$
d^2_{\text{out}}
\approx \frac{(1-\epsilon)^2 r^2}{(1-\epsilon) + \epsilon r^2}
\;\xrightarrow[\epsilon r^2 \gg 1]{}\;
\frac{(1-\epsilon)^2}{\epsilon}
\qquad (8)$$

while a normal point near the origin pays almost nothing along $e_1$ and a full
standard deviation along the rest:

$$
d^2_{\text{in}} \approx \frac{p-1}{1-\epsilon}
\qquad (9)$$

At $\epsilon = 0.2$, $p = 6$ these are 3.2 and 6.25, against the demo's measured
4.43 and 6.36 — the approximations ignore the cluster's own radius and the
sampling noise, but the ordering is the point. Two consequences follow. First,
(8) has a limit in $r$: moving the cluster *further away* makes it look *more*
normal, because the variance grows like $r^2$ while the distance grows like $r$.
Second, the gap widens with $p$, because the outlier gains nothing in the other
$p-1$ directions while every normal point pays for them. In one dimension there is
no masking by this mechanism; in ten it is catastrophic. The demo runs at $p = 6$
because that is where the inversion is unambiguous.

### 4. MCD, and why volume is the wrong thing to trust

Equation (4) has two moving parts: the subset search and the refit. The search is
combinatorial — $\binom{n}{h}$ subsets — so Fast-MCD takes random starts and
applies *concentration steps*: fit on the current $h$ points, sort all points by
distance, keep the $h$ closest, repeat. Each step cannot increase the determinant
of the retained set, so the procedure converges, and a handful of starts find the
global optimum with high probability for $n$ in the thousands. The refit is the
step that makes it usable: after the raw subset, points inside the 97.5%
ellipsoid are kept and the ordinary covariance is computed on them. That restores
consistency — the raw MCD covariance is biased low because it is computed on a
truncated sample — and it is what makes the errors in the table comparable to the
classical ones at zero contamination.

The failure is structural. The determinant of a subset is a product of variances,
and a compact cluster contributes a *small* variance in every direction but one.
At $p = 6$, shrinking five variances by a factor of ten more than pays for
inflating one, so the determinant prefers a subset that contains the cluster — the
demo's MCD ends up with a covariance that describes neither population, collapsed
in most directions and stretched along the cluster's axis. Volume is the right
criterion when the outliers are scattered, which is the assumption the method was
designed under, and the wrong one when they are clustered, which is what masks the
classical estimator in the first place. `ogk` has no such preference because a
pairwise scale over all pairs is dominated by whichever group holds the majority
of each projection.

### 5. OGK's correction step

Equation (5) gives a matrix that is symmetric but not necessarily a covariance.
Two repairs follow. The matrix is projected to the nearest positive semidefinite
one by clipping eigenvalues, then decomposed as $S = U\Lambda U^\top$ and used to
decorrelate the data: $Y = X S^{-1/2}$ has identity covariance under the pairwise
estimate. If the shape were right, every column of $Y$ would have robust scale
one; whatever scale it actually has is the correction. With $V =
\mathrm{diag}(\tau_j^2)$,

$$
\hat\Sigma = S^{1/2} \, V \, S^{1/2}
\qquad (10)$$

which leaves the pairwise *directions* alone and fixes the *lengths*. This is what
makes OGK work in the demo's masking case: the pairwise estimates recover the
orientation of the good data, and the column scales then measure the good data's
extent rather than a mixture. The final reweighting step is the same one MCD
uses, and for the same reason.

### 6. Why shrinkage tracks p/n

Ledoit–Wolf chooses $\lambda$ to minimise expected squared error against the true
$\Sigma$, with the target $F = (\mathrm{tr}S/p)I$. Writing
$d^2 = \lVert S - F \rVert_F^2/p$ for how far the sample covariance is from the
target, and $b^2$ for the total variance of the entries of $S$ around their
expectation, the optimum is

$$
\lambda = \frac{b^2}{d^2}, \qquad
b^2 = \frac{1}{p}\sum_{ij} \mathrm{Var}(s_{ij})
\qquad (11)$$

and the estimate is (6). The important scaling is in $b^2$: $s_{ij}$ is an average
of $n$ terms, so its variance falls like $1/n$, and the sum over $p^2$ entries
then gives $b^2 = O(p/n)$ while $d^2 = O(1)$. Hence $\lambda \approx p/n$ — the
demo measures 0.59 at $p/n = 2$, 0.11 at $0.1$ and 0.02 at $0.02$. Shrinkage is
needed exactly when there are not enough observations per dimension to pin the
covariance down, which is the same condition as singularity, so the same knob
fixes both. At $n = 30$, $p = 60$ the demo's fitted covariance has condition
number 7.7 and is positive definite, where the sample covariance has rank 29 and
`hotelling_t2` refuses to fit.

The direction of the bias matters. Shrinking toward $\mu I$ makes every
eigenvalue at least $\lambda\mu$, which is why the condition number is bounded,
and it also *understates* the large eigenvalues and *overstates* the small ones.
The demo's error at $n = 30$ is 0.600 against the pseudo-inverse's 1.025, but
that comparison overstates the case: the pseudo-inverse is not an estimate of
$\Sigma$ at all, it inverts a rank-29 matrix as if the 31 directions the data never
spanned had zero variance. The honest comparison is with what an unbiased
estimator would achieve given enough data, and that estimator does not exist here.
That is the trade: a biased estimator whose variance is small enough to be useful,
against an unbiased one that cannot be computed.

## Code walkthrough

### 1. The distance

```python
    diff = X - mean
    solved = np.linalg.solve(cov, diff.T).T
    return np.einsum("ij,ij->i", diff, solved)
```

One `solve` rather than an explicit inverse, and then a row-wise dot product.
`np.linalg.solve(cov, diff.T)` is $(p \times p)^{-1}(p \times m)$, which is
$O(p^3 + mp^2)$ and numerically better behaved than forming $\Sigma^{-1}$; the
transpose is because numpy's `solve` wants the matrix on the left. The `einsum`
sums $ij$ over $j$ — the quadratic form, without materialising the
$m \times p \times p$ outer product. If `cov` were singular this raises
`LinAlgError`, and `fit` checks the rank first so that the failure arrives as a
clear refusal rather than as a division by a zero eigenvalue.

### 2. The MCD search

```python
        idx = rng.choice(n, h, replace=False)
        for _ in range(n_csteps + 1):
            mean, cov = _classical(X[idx])
            d2 = _sq_mahal(X, mean, cov)
            idx = np.argpartition(d2, h - 1)[:h]
```

A random start, then concentration steps. `n_csteps + 1` is deliberate: the loop
body fits first and trims second, so $c$ concentration steps need $c+1$ fits to
end on a fitted subset. `np.argpartition` is $O(n)$ and returns the $h$ smallest
distances without sorting the rest — the selection is all that is needed, and the
order inside the subset does not matter. The determinant is compared in log space,
via `slogdet`, because determinants of covariance matrices overflow quickly; the
sign is checked because a subset of $h$ points in $p$ dimensions with $h < p$
gives a singular matrix, and that start must be discarded rather than ranked.

### 3. OGK's pairwise step

```python
        for k in range(j + 1, p):
            plus = _qn(X[:, j] + X[:, k]) ** 2
            minus = _qn(X[:, j] - X[:, k]) ** 2
            S[j, k] = S[k, j] = 0.25 * (plus - minus)
```

Equation (5) literally, with $Q_n$ as the scale. The diagonal is filled
separately with $Q_n^2$ of the column, since (5) is an identity about
covariances and the diagonal is a variance. The loop is $O(p^2)$ projections, each
$O(n^2)$ because $Q_n$ needs the pairwise differences — which makes OGK
quadratic in the sample, unlike MCD, and the reason it is the choice when $p$ is
large and $n$ is not.

### 4. The shrinkage intensity

```python
    outer = centred[:, :, None] * centred[:, None, :]
    # b^2 is the total variance of the sample covariance entries: sum_ij
    # Var(s_ij) / p. Var(s_ij) = Var(x_i x_j)/n, and the outer product gives
    # the per-observation term, so the double division by n is not a typo.
    b2 = float(np.sum((outer - S) ** 2) / (n * n * p))
    lam = min(max(b2 / d2, 0.0), 1.0)
    return mean, (1.0 - lam) * S + lam * target * np.eye(p), lam
```

The outer product is $n \times p \times p$, one $x_t x_t^\top$ per observation,
and `S` broadcasts across the first axis so the subtraction is
$x_{ti}x_{tj} - s_{ij}$ for every $t$. The double division is where
implementations go wrong: `np.mean` over $t$ would give the per-entry variance of
$x_ix_j$, and the variance of the *average* $s_{ij}$ is that divided by another
$n$. Getting this wrong makes $b^2$ independent of $n$, and then $\lambda \to 1$
for every sample however large — the estimator collapses to the identity target
and the error stops improving. That is why the demo prints $\lambda$ next to
$p/n$ rather than only an error. The clip
to $[0, 1]$ is not cosmetic either: $b^2$ is estimated, and an estimate above
$d^2$ would ask for negative weight on $S$.

### 5. The threshold

```python
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        if self.variant == "hotelling_t2":
            return p * (n - 1) / (n - p) * _f_ppf(1.0 - self.alpha, p, n - p)
        return _chi2_ppf(1.0 - self.alpha, p)
```

Three regimes in five lines. A `contamination` budget is a quantile of the
training scores and makes no distributional claim — it flags that fraction of any
sample, including a clean one. `contamination=None` asks for the calibrated
threshold instead: (3) for the Hotelling statistic, (7) for the others. The $F$
quantile is computed by bisection on the incomplete beta function, which is why
the file carries `_betainc` and `_betacf`; the chi-square quantile is bisection on
the incomplete gamma. Both are twenty lines of standard numerical code and remove
the only reason this atom would otherwise need scipy.

## Complexity

For $n$ points, $p$ features, $m$ queries, $t$ MCD starts and $c$ concentration
steps:

| stage | time | memory |
|:--|:--|:--|
| classical fit | $O(np^2 + p^3)$ | $O(p^2)$ |
| score $m$ points | $O(mp^2)$ | $O(p^2)$ |
| MCD | $O(t\,c\,n p^2 + t\,p^3)$ | $O(np)$ |
| OGK | $O(p^2 n^2 + p^3)$ | $O(np)$ |
| shrinkage | $O(np^2)$ | $O(np^2)$ |
| all four, demo size | under a second | tens of MB |

MCD is linear in $n$ per start and quadratic in $p$ per fit, which is why it is
the practical robust estimator at moderate dimension; with $p$ in the hundreds
the subset search is replaced by a projection-pursuit or a cell-wise method. OGK's
$Q_n$ is the awkward term: $O(n^2)$ per projection and $p(p-1)/2$ projections. For
large $n$ the scale is replaced by a faster high-breakdown estimator, or the
pairwise step is run on a subsample; the structure of (5) is unchanged. The
shrinkage estimate's memory is the only one that grows with $np^2$, because of the
outer-product array — a loop over observations would remove it at the cost of
speed.

Scoring is where these methods pay for their fitting cost: once $\hat\mu$ and
$\hat\Sigma$ exist, a new point costs one solve, and the same covariance can be
reused for a stream of points. That is why the classical and shrinkage variants
are the default in monitoring systems, and why the robust ones are fitted on a
window and then applied cheaply.

## Failure modes

### Masking, in the strong sense

The demo's numbers are the definition: classical AUC 0.340 with outliers scoring
4.43 and normals 6.36, and 12% of the normals flagged at a 10% budget. In high
dimensions the inversion is not a rare coincidence but the expected behaviour of a
tight far cluster, per (8)–(9). If a Mahalanobis detector reports that the
anomalies are *unusually close* to the mean, that is this, not a sign flip.

### A compact cluster defeats MCD

The demo's MCD scores AUC 0.174 on the same data, worse than the classical fit it
was meant to repair, because the determinant criterion prefers a subset whose
volume is smaller — and a tight cluster has less volume than the bulk. The
breakdown point is about the *fraction* of contamination, not its geometry, and
these are different failure modes. Remedies: increase `support_fraction` so the
cluster cannot dominate the subset, run MCD per cluster or on a projection, or use
OGK, which is insensitive here.

### n <= p is not a small-sample problem

With fewer observations than features, or with collinear columns, the sample
covariance is singular and (1) is undefined. `hotelling_t2` raises rather than
returning a pseudo-inverse distance, because a distance in the null space is
arbitrary — any value can be produced by moving within the subspace the data never
spanned. `shrinkage` and `mcd` return finite answers by construction. In between,
$n$ slightly above $p$, the distances are finite and wildly unstable, because the
smallest eigenvalue is essentially noise; the condition number is the diagnostic,
and the demo's table shows it climbing from 7.7 to 37.1 as $n/p$ falls.

### The chi-square threshold assumes the model

(7) is a statement about Gaussian data. Heavy tails inflate $d^2$ for everyone,
so a nominal 5% threshold on a $t_3$ sample flags far more; skew makes the
ellipsoid the wrong shape, so the flag set is concentrated in one tail. The
symptom is a detector that fires constantly on clean data. Fix the threshold with
an empirical quantile (`contamination`), or transform the marginals first — but
note that a marginal transform destroys the correlations, so it must be a
copula-style or rank-based transform, not a per-column log.

### Qn assumes the bulk is inliers

`ogk`'s 50% breakdown is real, but the guarantee is about the *number* of
contaminated points, and $Q_n$ is a single order statistic — a bimodal
contamination that shifts the 25th percentile of differences will move it. It is
also $O(n^2)$, which is the reason production code uses faster scales at large
$n$. The demo's OGK is the most accurate estimator on clean data (0.073 against
MCD's 0.141), so the method's cost is not efficiency but computation.

### Shrinkage is not robustness

The demo makes this explicit: on the masking data shrinkage's $\lambda$ is 0.007,
so it is the sample covariance, and its AUC is 0.350 against the classical 0.340.
On the contamination sweep its error tracks the classical one almost exactly
(0.278 against 0.273 at 5%). A scaled-identity target absorbs singular directions,
not outlying ones — the two problems look similar and are not.

## Applications

- **Statistical process control.** Hotelling's $T^2$ chart is the multivariate
  version of a Shewhart chart: fit $\mu, \Sigma$ on a calibration window, monitor
  (2) on incoming batches, and use (3) for the control limit. The small-sample
  correction matters because calibration windows are short.
- **Novelty detection on features.** A Gaussian fit on embeddings, with (1) as
  the score, is one of the strongest baselines for out-of-distribution detection
  in vision and language models — precisely because it uses the covariance that a
  distance-on-features detector throws away.
- **Quality control with contamination.** A production line's calibration sample
  usually has some bad parts in it; MCD is the standard way to fit the process
  covariance without them, and the reweighted distances are what get plotted.
- **Anomaly detection under masking.** `ogk` is the robust default when
  anomalies come in clusters — a bad sensor batch, a botnet's traffic — where the
  classical estimator provably inverts the ranking.
- **Regularising a downstream model.** Shrinkage covariances are the standard
  input to Gaussian mixture models, linear discriminant analysis and portfolio
  optimisation, where the sample covariance is singular or nearly so.
- **Metric learning and matching.** (1) is the distance inside a learned metric;
  re-ID, face verification and tracking all use a Mahalanobis form, and the
  covariance's conditioning is the practical issue at embedding dimension.
- **Where it is a trap.** Clustered anomalies, heavy tails, $n \lesssim p$, and
  any setting where "the mean and covariance of normal data" is not a meaningful
  object because normal behaviour is multimodal. The local methods earlier in
  this chapter exist for that last case.

## References

- P. C. Mahalanobis, *On the generalised distance in statistics*, Proceedings of
  the National Institute of Sciences of India 2(1):49–55 (1936). Equation (1).
- H. Hotelling, *The generalisation of Student's ratio*, Annals of Mathematical
  Statistics 2(3):360–378 (1931). The $T^2$ statistic and its $F$ distribution,
  (2)–(3).
- P. J. Rousseeuw, *Multivariate estimation with high breakdown point*, in
  Mathematical Statistics and Applications (1985). The minimum covariance
  determinant and the breakdown-point argument.
- P. J. Rousseeuw and K. Van Driessen, *A fast algorithm for the minimum
  covariance determinant estimator*, Technometrics 41(3):212–223 (1999).
  Concentration steps, random starts, and the reweighting that makes (4) usable.
- P. J. Rousseeuw and B. C. van Zomeren, *Unmasking multivariate outliers and
  leverage points*, Journal of the American Statistical Association
  85(411):633–639 (1990). Masking and swamping, and the robust-distance plot.
- R. A. Maronna and R. H. Zamar, *Robust estimates of location and dispersion for
  high-dimensional datasets*, Technometrics 44(4):307–317 (2002). OGK and
  equation (5).
- P. J. Rousseeuw and C. Croux, *Alternatives to the median absolute deviation*,
  Journal of the American Statistical Association 88(424):1273–1283 (1993). $Q_n$
  and the $2.2219$ constant whose finite-sample form is derived here.
- O. Ledoit and M. Wolf, *A well-conditioned estimator for large-dimensional
  covariance matrices*, Journal of Multivariate Analysis 88(2):365–411 (2004).
  Equations (6) and (11).
- C. Stein, *Inadmissibility of the usual estimator for the mean of a
  multivariate normal distribution*, Berkeley Symposium (1956). Why a biased
  estimate can dominate an unbiased one, which is the logic of shrinkage.
- M. Hubert, M. Debruyne and P. J. Rousseeuw, *Minimum covariance determinant and
  extensions*, WIREs Computational Statistics 10(3):e1421 (2018). The compact-
  cluster failure and the modern fixes.
- M. Goldstein and S. Uchida, *A comparative evaluation of unsupervised anomaly
  detection algorithms for multivariate data*, PLoS ONE 11(4):e0152173 (2016).
  Where Mahalanobis-family scores win and where the local methods take over.
