# Elliptic envelope — the robust ellipsoid as a decision region

`elliptic_envelope.py` · Density & anomaly · Tier 2 · 4 variants

## Intuition

An anomaly score says how far a point is. A decision region says whether it is
inside. The second is the one you can draw on a chart, ship to a plant, or
compare across dimensions, and for a Gaussian model the region is an ellipsoid:

```
            ·    ·    ·
        ·                 ·          ✕
      ·      ·     ·        ·
     ·    ·     ·     ·       ·
      ·      ·     ·        ·
        ·                 ·
            ·    ·    ·
```

The points are the sample, the ring is the boundary, and the cross is outside.
Everything in this atom is a choice about that ring: where its centre is, what
shape it has, and how big it is. The Mahalanobis atom computes the score that
the ring is a level set of; this atom is about fitting the ring itself.

The obvious fit is the sample mean and covariance. It is also the one that
fails, because the outliers take part in it. Move a tenth of the data into a
tight cluster at distance $2.5$ and the variance along that direction grows by
$1 + 0.1 \cdot 2.5^2 = 1.63$; the ellipse becomes 28 percent wider in every
direction, and the cluster that caused the inflation is now comfortably inside
the ellipse built to exclude it. This is **masking**, and it is the reason the
minimum covariance determinant exists.

On clean two-dimensional Gaussian data the four variants agree, which is the
right way to check that a robust estimator is not simply a broken one:

| variant | centre error | fresh coverage | volume | scale | C-steps |
|:--|--:|--:|--:|--:|--:|
| `mcd_ellipse` | 0.047 | 0.897 | 11.656 | 1.344 | 85 |
| `min_cov_det` | 0.046 | 0.488 | 3.568 | 1.000 | 85 |
| `shrinkage` | 0.091 | 0.899 | 11.714 | 1.000 | 0 |
| `chi2_threshold` | 0.046 | 0.876 | 11.555 | 3.238 | 85 |

The data are 400 points from a correlation-$0.6$ Gaussian and the nominal
coverage is 90 percent. The analytic volume of the 90 percent ellipse of the
true covariance is $11.574$, and the three corrected estimators land on it to
within one percent. The raw determinant minimiser does not: its volume is
$3.568$ and it covers 49 percent of fresh points. That gap is a factor of
$3.26$ — the small-sample correction (6) — and it is not a detail you can skip.

## Variants

| variant | centre and shape | boundary | works when |
|:--|:--|:--|:--|
| `min_cov_det` | raw MCD | $\chi^2_p$ | never on its own |
| `mcd_ellipse` | corrected MCD, reweighted | empirical quantile | default |
| `chi2_threshold` | corrected MCD | $\chi^2_p$ | the data are plausibly Gaussian |
| `shrinkage` | Ledoit-Wolf | $\chi^2_p$ | $p$ approaches $n$ |

**`min_cov_det`** is the estimator of (5) with the Gaussian boundary and no
correction. It exists because it is the shortest path from the paper to running
code, and because seeing it fail is the fastest way to understand why (6) is
not optional. It recovers the centre and finds every outlier in the contaminated
demo, and it still covers only half the clean points.

**`mcd_ellipse`** is the complete detector: corrected covariance, a reweighting
step that refits on the points inside the envelope, and a size taken from the
empirical quantile of the training distances. It is distribution-free in the
sense that its in-sample coverage is exactly $1-\alpha$ whatever the data look
like. The price is that the empirical quantile is only as good as the sample:
with a 10 percent budget and 10 percent contamination it lands on the largest
clean distance and over-covers.

**`chi2_threshold`** keeps the corrected estimator and uses the theoretical
boundary, which is what scikit-learn's `EllipticEnvelope` does. It is the right
choice when the Gaussian model is believable, and it is the one that stays
calibrated in high dimensions, where the in-sample quantile is optimistic.

**`shrinkage`** replaces the robust estimator with Ledoit-Wolf shrinkage toward
a scaled identity. It is not robust — the demo shows it inheriting the classical
mean and the classical failure — but it is the only variant that produces an
invertible, well-conditioned covariance when the dimension is a large fraction
of the sample size. At $p=40$ and $n=50$ the sample covariance has condition
number $356$ and covers none of a fresh sample; shrinkage has condition number
$1.4$ and covers 81 percent.

## Derivation

### The score, the region, and the name

For a location $\mu$ and a positive-definite covariance $\Sigma$, the squared
Mahalanobis distance is

$$
d^2(x) = (x-\mu)^\top \Sigma^{-1} (x-\mu)
\qquad (1)$$

and the envelope at level $t$ is the set of points whose distance does not
exceed it:

$$
R_t = \{\, x : d^2(x) \le t \,\}
\qquad (2)$$

Write $\Sigma = U\Lambda U^\top$ with
$\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_p)$. Then
$d^2(x) = \sum_i u_i^\top(x-\mu)^2/\lambda_i$, so $R_t$ is the ellipsoid
whose $i$-th semi-axis is $\sqrt{t\lambda_i}$ along $u_i$. The name is literal:
the region is an ellipsoid, and its orientation is the eigenbasis of the
covariance.

### The volume of the region

The linear map $x \mapsto \Lambda^{-1/2}U^\top(x-\mu)/\sqrt{t}$ sends $R_t$ to
the unit ball. Its determinant is $t^{-p/2}\det(\Sigma)^{-1/2}$, so

$$
\mathrm{vol}(R_t) = V_p\, t^{p/2} \sqrt{\det \Sigma},
\qquad V_p = \frac{\pi^{p/2}}{\Gamma(p/2+1)}
\qquad (3)$$

with $V_p$ the volume of the unit ball. For $p=2$, $V_2 = \pi$ and the volume is
$\pi t\sqrt{\det\Sigma}$; the demo checks that identity against (3) to machine
precision. This is why the determinant is the right thing for (5) to minimise:
it is the volume of the ellipse up to the factor $V_p t^{p/2}$, which does not
depend on the support.

### Where the boundary comes from

If $x \sim \mathcal{N}(\mu,\Sigma)$ then $\Sigma^{-1/2}(x-\mu) \sim
\mathcal{N}(0,I)$, so $d^2(x)$ is a sum of $p$ squared standard normals:

$$
d^2(x) \sim \chi^2_p
\qquad (4)$$

and the level set at the $1-\alpha$ quantile contains $1-\alpha$ of the mass.
Two cases have closed forms and the demo asserts both: for $p=2$,
$P(\chi^2_2 > t) = e^{-t/2}$, so the 95 percent quantile is $-2\log 0.05 =
5.991$; for $p=1$ it is the square of the normal quantile, $1.96^2 = 3.841$.

The catch is that (4) is exact only when $\mu$ and $\Sigma$ are the true
parameters. With the plug-in estimates the in-sample distances are smaller than
the law says — the demo measures 87.2 percent coverage at a nominal 90 — because
the ellipse was fitted to those points. That in-sample optimism is the whole
difference between `chi2_threshold` and `mcd_ellipse`.

### The minimum covariance determinant

The classical estimate is not robust because every point contributes to it. The
minimum covariance determinant estimator replaces the average with a search:

$$
(\hat\mu, \hat\Sigma) = \arg\min_{H \subset \{1..n\},\ |H| = h}
\det \mathrm{cov}(H)
\qquad (5)$$

Since the determinant is a volume, this is "the tightest ellipsoid that
contains $h$ points". A far-away point cannot make a set tighter by being far
away, which is exactly the property the sample covariance lacks.

The naive version of this idea — trim the points with the largest distance —
is circular: you need a centre to measure distance and a covariance to scale it,
and the outliers are the ones that corrupt both. MCD breaks the circle by
searching over subsets rather than over residuals. The price is a combinatorial
problem, and the algorithm that makes it practical is the C-step:

1. Start from a random $h$-subset.
2. Compute its mean and covariance.
3. Replace the subset with the $h$ points of smallest distance under them.
4. Repeat until the subset stops changing.

Each step cannot increase the determinant, so the loop terminates. It
terminates at a *local* minimum, so the implementation runs several random
starts and keeps the smallest determinant; in the demo ten starts take 85
C-steps in total and agree on the same support.

The choice of $h$ is the robustness/efficiency dial. With $h = (n+p+1)/2$ the
estimator tolerates just under 50 percent contamination, which is the maximum
any affine-equivariant estimator can achieve; larger $h$ uses more data and is
more efficient, and tolerates less contamination. The default here is that
maximal-breakdown value, capped at $n$.

### The correction

The covariance of the $h$ most concentrated points of a Gaussian is not the
covariance of the Gaussian. It is the covariance of a *truncated* Gaussian, and
truncation removes variance. The exact factor comes from one identity. If
$Z \sim \mathcal{N}(0,I_p)$ and $q = \chi^2_p(\alpha)$, then

$$
t\,f_{\chi^2_p}(t) = p\,f_{\chi^2_{p+2}}(t)
$$

so the first moment of the truncated squared radius is

$$
\mathbb{E}\big[\|Z\|^2 \mathbf{1}\{\|Z\|^2 \le q\}\big]
= \int_0^q t f_{\chi^2_p}(t)\,dt
= p\,F_{\chi^2_{p+2}}(q),
$$

and dividing by $\alpha = P(\|Z\|^2 \le q)$ gives
$\mathbb{E}[\|Z\|^2 \mid \text{inside}] = p F_{\chi^2_{p+2}}(q)/\alpha$.
By symmetry the truncated second moment matrix is a multiple of the identity, so
the truncated covariance is $[F_{\chi^2_{p+2}}(q)/\alpha]\,\Sigma$, and
multiplying by

$$
c(\alpha,p) = \frac{\alpha}{F_{\chi^2_{p+2}}\big(\chi^2_p(\alpha)\big)}
\qquad (6)$$

undoes it. The factor is large exactly where the estimator is most robust:
$c = 3.259$ at $\alpha = 0.5$ in two dimensions, $1.344$ at $\alpha = 0.9$, and
$1.131$ at $\alpha = 0.9$ in eight dimensions. Skipping it turns a 90 percent
envelope into a 49 percent one, and even after the correction the coverage falls
further short of nominal as the dimension grows — 0.878 at $p=2$ against 0.823
at $p=8$ at $h/n = 0.5$ — because the volume carries the factor to the power
$p/2$ and the sampling error in the support covariance grows with it.

### Reweighting

The corrected MCD covariance is unbiased but noisy: it is computed from $h$
points, which at the default $h \approx n/2$ throws away half the sample. The
fix is to use the robust fit only to decide *which* points are clean, and then
estimate the covariance classically from all of them:

1. Correct the MCD covariance by (6).
2. Keep the points inside the $1-\alpha$ ellipse.
3. Refit the mean and covariance on the kept points.
4. Correct again — the kept set is itself a truncation, now at a much larger
   $\alpha$, where $c$ is close to one.

The demo's `mcd_ellipse` reports `scale_ = 1.344`, which is exactly $c(0.9,2)$:
the reweighted set is the central 90 percent of the data, and its covariance is
corrected by the corresponding factor. The step buys back the efficiency the
$h$-subset gave up: `chi2_threshold`, which skips it, covers 87.6 percent where
`mcd_ellipse` covers 89.7.

The cut has to be the same $1-\alpha$ as the envelope. Some implementations
hard-code 97.5 percent, which lets points between the two thresholds back into
the covariance; when those points are a contaminating cluster, the ellipse
inflates to include the very points it was meant to exclude.

### Shrinkage

When $p$ is a large fraction of $n$ the sample covariance is invertible but
useless: its smallest eigenvalues are noise, so the ellipse is a sliver in some
directions and unbounded in others. Ledoit-Wolf replaces it with a convex
combination of the sample covariance and a scaled identity,

$$
\hat\Sigma = (1-\lambda) S + \lambda\,\mu I, \qquad \mu = \frac{\mathrm{tr}(S)}{p},
$$

choosing $\lambda$ to minimise the expected squared Frobenius error. With
$d^2 = \|S - \mu I\|_F^2/p$ the squared distance to the target and
$b^2 = \frac{1}{n^2p}\sum_k \|x_kx_k^\top - S\|_F^2$ the variance of the sample
covariance's entries, the optimal weight is $\lambda = \min(1, b^2/d^2)$.

The estimator is not robust — a cluster still moves the mean — but it is the
only one here that survives $p/n \to 1$. In the demo at $p=40$, $n=50$ it
chooses $\lambda = 0.893$, collapses the condition number from $356$ to $1.4$,
and covers 81 percent of a fresh sample where both the sample covariance and the
MCD cover none. The shortfall from 90 is the bias of pulling every direction
toward a common variance, and at $p/n = 0.8$ it is large.

## Code walkthrough

### The quadratic form

```python
    centred = X - location
    solved = np.linalg.solve(covariance, centred.T)
    return np.einsum("ij,ij->j", centred.T, solved)
```

One `solve` for the whole matrix rather than one inverse and a matrix product:
the solve is $O(p^3)$ once and $O(np^2)$ for the right-hand side, and it is more
accurate than forming $\Sigma^{-1}$. The `einsum` is the row-wise dot product
$\sum_i c_{ij} s_{ij}$, which is (1) for every point at once.

### Chi-square without scipy

The whole atom needs $F_{\chi^2}$ and its inverse, and neither is in numpy. The
CDF is the regularised lower incomplete gamma function, evaluated by its power
series below $a+1$ and by a continued fraction above it; the quantile is then
twenty lines of bisection:

```python
    lo, hi = 0.0, 1.0
    while _chi2_cdf(hi, k) < q:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("chi-square quantile did not converge")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, k) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
```

Doubling to bracket, then halving to converge. The bracket starts at
$[0,1]$ and doubles, which costs at most 40 extra CDF evaluations and removes
any need for a good initial guess. The tolerance is relative, so it holds for
the $p=40$ quantile as well as the $p=2$ one, and the demo checks the result
against the two closed forms.

### The correction

```python
    return alpha / _chi2_cdf(_chi2_ppf(alpha, p), p + 2)
```

Equation (6) in one line. The inner quantile is the truncation point; the outer
CDF is evaluated with $p+2$ degrees of freedom, which is the identity
$t f_{\chi^2_p}(t) = p f_{\chi^2_{p+2}}(t)$ doing its work.

### The concentration step

```python
            sub = X[index]
            location = sub.mean(axis=0)
            covariance = np.atleast_2d(np.cov(sub, rowvar=False, bias=True))
            try:
                distance = _quadratic(X, location, covariance)
            except np.linalg.LinAlgError:
                singular = True
                break
            total += 1
            # the h points closest to the current ellipse are the next support
            updated = np.sort(np.argpartition(distance, h - 1)[:h])
            if np.array_equal(updated, index):
                break
            index = updated
```

`argpartition` selects the $h$ smallest distances in $O(n)$ without sorting the
rest, which matters when $n$ is large and $h$ is not. The `try` is not
decoration: a random $h$-subset can be degenerate (duplicate rows, or $h$ points
on a hyperplane), and a singular covariance means the start is abandoned rather
than crashing the whole fit. `np.sort` keeps the index set canonical so that the
equality test detects a genuine fixed point rather than a permutation of one.

### Reweighting

```python
    distance = _quadratic(X, location, covariance)
    keep = distance <= _chi2_ppf(1.0 - contamination, p)
    if keep.sum() < p + 2:
        return location, covariance, np.ones(len(X), dtype=bool), 1.0
    sub = X[keep]
    new_location = sub.mean(axis=0)
    new_covariance = np.atleast_2d(np.cov(sub, rowvar=False, bias=True))
    scale = _correction(1.0 - contamination, p)
    return new_location, new_covariance * scale, keep, scale
```

The guard is the interesting line. If the envelope is so small that fewer than
$p+2$ points survive, the refit would be singular, so the function returns the
robust fit unchanged rather than a broken one. The returned `keep` mask becomes
`support_`, so the caller can see which points the final estimate came from —
useful when the next question is "why is this point inside".

### Shrinkage without the tensor

```python
    sample = (centred.T @ centred) / n
    target = (float(np.trace(sample)) / p) * np.eye(p)
    d2 = float(np.sum((sample - target) ** 2)) / p
    # sum_k || x_k x_k^T - S ||_F^2 without materialising the n x p x p tensor
    squared = np.sum(centred * centred, axis=1)
    quadratic = np.einsum("ij,ij->i", centred @ sample, centred)
    frobenius = float(np.sum(sample * sample))
    b2 = float(np.sum(squared * squared - 2.0 * quadratic + frobenius))
    b2 /= (n * n) * p
```

The textbook definition of $b^2$ is a sum of $n$ Frobenius norms of $p\times p$
matrices. Expanding the square turns it into three scalar sums per point:
$\|x\|^4$, $-2x^\top S x$, and $\|S\|_F^2$. That is the difference between
$O(np^2)$ memory and $O(n)$ memory, and it is why this atom can run a $p=40$
demo at all.

### Choosing the size

```python
        if self.variant == "mcd_ellipse":
            return float(np.quantile(self.distances_, 1.0 - self.contamination))
        return float(_chi2_ppf(1.0 - self.contamination, self.n_features_))
```

Two lines, two philosophies. The empirical quantile is exact in-sample and
needs no distributional assumption; the chi-square quantile is exact
out-of-sample under the Gaussian model and is the only one that survives
$p/n \to 1$, because in-sample distances are systematically too small there.
The demo measures that directly: at $p=40$, $n=50$ the empirical quantile would
cover 56.5 percent of fresh points against a nominal 90.

## Complexity

Let $n$ be the sample size, $p$ the dimension, $S$ the number of C-steps per
start and $K$ the number of starts.

| operation | time | memory |
|:--|:--|:--|
| score $n$ points | $O(p^3 + np^2)$ | $O(np)$ |
| one C-step | $O(np^2)$ | $O(np)$ |
| MCD | $O(KSp^2n)$ | $O(np)$ |
| reweight | $O(np^2)$ | $O(np)$ |
| Ledoit-Wolf | $O(np^2 + p^3)$ | $O(np)$ |
| volume | $O(p^3)$ | $O(p^2)$ |

The MCD dominates, and the constant that matters is $S$, not $K$: each start
converges in a handful of steps, so ten starts cost about ten times one start,
not a hundred. The demo's ten starts take 85 C-steps in total on 400 points in
two dimensions, which is a few milliseconds. In high dimensions the $p^2$ inside
each step and the $p^3$ in the solve are what hurt, and $h$ must exceed $p$ for
the covariance to be non-singular at all.

## Failure modes

**Masking.** The classical covariance is dragged toward the contamination and
inflates until the outliers are inside. In the demo the volume goes from 11.6 to
14.6 and the classical envelope flags only 49 percent of the cluster while
covering 93 percent of the clean points. Shrinkage has the same mean and the
same failure: it fixes conditioning, not contamination.

**Skipping the correction.** `min_cov_det` covers 0.488 at a nominal 0.9, and
its volume is a third of the truth. The factor is 3.26 in two dimensions at
$h/n=0.5$; it is not a rounding error, and it grows with the dimension because
the volume carries it to the power $p/2$.

**The empirical quantile needs headroom.** With a 10 percent budget and 10
percent contamination, the 90th percentile of the training distances lands on
the largest clean distance, so `mcd_ellipse` over-covers at 0.976 and misses a
quarter of the cluster. At a 20 percent budget the same fit covers 0.866 and
catches 0.989. The budget must exceed the contamination you actually have, and
you rarely know it.

**The chi-square boundary assumes a Gaussian.** On $t_3$ data the theoretical
envelope covers 0.736 at a nominal 0.9, because the squared radius is not
chi-square when the tails are heavy. The empirical boundary covers 0.910 on the
same data: it does not care what the law is, only that the training sample
resembles the test sample.

**In high dimensions the in-sample quantile is optimistic.** At $p=40$,
$n=50$ the same shrinkage fit covers 0.807 with the chi-square boundary and
0.565 with the empirical one. Distances computed on the training points are
smaller than distances on new points, and the gap grows with $p/n$. Use the
chi-square boundary there, or a held-out calibration set.

**Contamination above 50 percent breaks the estimator.** The default support is
the maximal-breakdown one, and it assumes the majority of the data is clean. If
more than half the sample is bad, the "tightest half" is the bad cluster. There
is no affine-equivariant estimator that survives that, so the answer is a
different model, not a different $h$.

**The result depends on the seed.** C-steps converge to local minima. Ten starts
are enough for the demo and are not a guarantee; if two runs disagree on the
support, the determinant surface has more than one deep basin and you should
raise `n_init`. The estimator is deterministic given `seed`, which is what makes
the demo reproducible, not what makes it correct.

**$p \ge n$ has no answer here.** The fit refuses with "need at least $p+2$
rows", and even when $n > p+2$ the MCD support can be too small to estimate a
$p \times p$ covariance. The refusal is deliberate: the alternative is a
singular matrix and a silent `inf` score.

## Applications

**Analytical chemistry.** MCD comes from chemometrics, where a calibration set
contains a few samples that are not what the label says. The ellipse is fitted
on the clean majority and the suspect samples are re-run; the method is
Rousseeuw's, and it is still the standard robust covariance in that field.

**Manufacturing and process control.** A two- or three-dimensional envelope over
a few measured quality variables is a control chart that respects their
correlation. Points outside are inspected; the volume of the envelope is a
summary of how much variation the process is allowed.

**Fraud and transaction monitoring.** An elliptic envelope over a handful of
scaled features is a cheap first-pass detector, and its main virtue is that the
decision is explainable: a flagged point is outside a described ellipsoid, not
"the 4,217th leaf of a tree".

**Astronomy and survey pipelines.** Outlier rejection before fitting a model is
the same problem, and MCD is used to define the clean subset without letting the
outliers set the scale. The robust covariance then feeds a Mahalanobis cut on
the full catalogue.

**As a baseline for `lof` and `isolation_forest`.** The envelope is the
multivariate Gaussian baseline: if a non-parametric detector does not beat it on
your data, the extra machinery is not paying for itself.

**As a preprocessing step for supervised models.** Fit the envelope on the
training set, drop or down-weight the points outside, and fit the model on the
rest. The risk is the same as everywhere here: the envelope defines "normal",
and a rare but valid subpopulation will be thrown away with the errors.

## References

- Rousseeuw, P. J. (1985). Multivariate estimation with high breakdown point.
  In *Mathematical Statistics and Applications*, 283–297. The minimum
  covariance determinant estimator.
- Rousseeuw, P. J. & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator. *Technometrics* 41(3), 212–223. FAST-MCD,
  the C-steps and the multiple random starts.
- Croux, C. & Haesbroeck, G. (1999). Influence function and efficiency of the
  minimum covariance determinant scatter matrix estimator. *Journal of
  Multivariate Analysis* 71(2), 161–190. The consistency correction (6).
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*
  88(2), 365–411. The shrinkage estimator and its optimal weight.
- Hubert, M. & Debruyne, M. (2010). Minimum covariance determinant. *WIREs
  Computational Statistics* 2(1), 36–43. A short survey with the breakdown
  point and the reweighting step.
- Hardin, J. & Rocke, D. M. (2005). The distribution of robust distances.
  *Journal of Computational and Graphical Statistics* 14(4), 928–946. Why the
  chi-square approximation to MCD distances is off, and the correction that
  fixes it.
- Rousseeuw, P. J. & Leroy, A. M. (1987). *Robust Regression and Outlier
  Detection*. Wiley. The book that made masking and swamping standard
  vocabulary.
- Barnett, V. & Lewis, T. (1994). *Outliers in Statistical Data*, 3rd edition.
  Wiley. The classical taxonomy of outlier problems this atom is a special case
  of.
