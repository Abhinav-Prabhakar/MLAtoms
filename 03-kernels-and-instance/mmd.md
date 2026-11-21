# MMD and HSIC — kernel two-sample and independence tests

`mmd.py` · Kernels and instance · Tier 3 · 5 variants

## Intuition

Given two samples, is the difference between them real? A t-test answers the
question for means, a KS test for whole distributions in one dimension, and
neither generalises to objects with no natural coordinates — images, strings,
graphs. The kernel answer is one line: embed each distribution as a mean in an
RKHS and measure the distance between the two means.

$$
\mathrm{MMD}^2(P,Q)=\lVert\mu_P-\mu_Q\rVert_{\mathcal H}^2
=\mathbb E\,k(x,x')+\mathbb E\,k(y,y')-2\,\mathbb E\,k(x,y),
\qquad (1)
$$

with $x,x'\sim P$ and $y,y'\sim Q$ independent. If $k$ is **characteristic** —
the RBF and Laplacian kernels are, a polynomial of finite degree is not — then
$\mathrm{MMD}(P,Q)=0$ if and only if $P=Q$. So a single scalar decides equality
of distributions in any dimension, and its estimate is a sum of kernel values
with no binning, no density estimate and no curse of dimensionality.

The estimator is not the hard part. The hard part is that its null distribution
has no closed form: under $P=Q$ the statistic is a weighted sum of $\chi^2$
variables with weights equal to the kernel's eigenvalues. The tractable answer is
the same one used for every other statistic: permute the labels, recompute, and
compare. That calibration is variant five of this atom and the reason the file is
a test rather than a distance.

The same machinery answers a second question — are two variables independent? —
by measuring the distance between the joint distribution and the product of the
marginals. That is HSIC, and it is equation (7) below: a trace of two centred
Gram matrices, computed by the same code path.

## Variants

| variant | statistic | time | null |
|:--|:--|:--|:--|
| `quadratic` | V-statistic or U-statistic | $O(n^2)$ | pooled permutation |
| `linear_time` | mean of block U-statistics | $O(nb)$ | pooled permutation |
| `mmd_rbf` | quadratic plus the witness function | $O(n^2)$ | pooled permutation |
| `hsic` | normalised HSIC | $O(n^2)$ | row permutation |
| `permutation_test` | any of the above plus the null | $B\cdot O(n^2)$ | — |

**`quadratic`** is the estimator of (2) or (3): one $n\times n$ Gram matrix on
the pooled sample, three block means, a subtraction. Use it unless $n$ is past a
few thousand.

**`linear_time`** splits both samples into contiguous blocks, computes the
unbiased estimate inside each block, and averages. The default block is
$\lfloor\sqrt n\rfloor$, giving $O(n^{1.5})$; a fixed block size makes the
estimator strictly linear in $n$. The price is variance — the demo's gap to the
quadratic estimate is 0.017 at $n=160$ and 0.0002 at $n=2560$.

**`mmd_rbf`** is the RBF instantiation with the median heuristic bandwidth, plus
the witness function: the actual function $f$ whose RKHS norm is the statistic,
evaluated on the pooled points. The witness is what makes an MMD result
diagnosable — it says *where* the two samples differ, not only that they do.

**`hsic`** turns the same kernel sums into an independence test for paired
samples. It is the only variant that requires equal lengths, because the pairing
is the data.

**`permutation_test`** is not a different statistic; it is the calibration,
exposed as a variant so that `mmd(..., variant="permutation_test")` is the full
test (statistic, null, p-value) and the other variants can be called without
paying for 200 permutations. It refuses to run with `n_perm=0`.

## Derivation

### From the embedding to a sum of kernels

Expanding the squared norm in (1) gives the population functional. Its plug-in
estimate on samples of sizes $m$ and $n$ replaces each expectation by the
corresponding average over all pairs, diagonal included:

$$
\widehat{\mathrm{MMD}}^2_b=
\frac{1}{m^2}\sum_{i,j}K^{xx}_{ij}+
\frac{1}{n^2}\sum_{i,j}K^{yy}_{ij}-
\frac{2}{mn}\sum_{i,j}K^{xy}_{ij}.
\qquad (2)
$$

This is the V-statistic, and it is a squared norm, so it is non-negative for a
PSD kernel. Its cost is three Gram blocks — one $m\times m$, one $n\times n$, one
$m\times n$ — and no other work.

The diagonal terms $k(x_i,x_i)$ are the problem: they are in (2) but not in the
population quantity of (1). Removing them gives the U-statistic,

$$
\widehat{\mathrm{MMD}}^2_u=
\frac{\sum_{i\ne j}K^{xx}_{ij}}{m(m-1)}+
\frac{\sum_{i\ne j}K^{yy}_{ij}}{n(n-1)}-
\frac{2}{mn}\sum_{i,j}K^{xy}_{ij},
\qquad (3)
$$

which is unbiased but can be negative. The bias of (2) is exactly the diagonal
mass, and for a kernel with $k(x,x)=1$ it is

$$
\mathbb E\big[\widehat{\mathrm{MMD}}^2_b\big]-\mathrm{MMD}^2
=\frac{1-\rho_x}{m}+\frac{1-\rho_y}{n},
\qquad \rho_x=\mathbb E\,k(x,x'),
\qquad (4)
$$

so it vanishes as $1/m+1/n$ and it is positive whenever the kernel is not
degenerate. That is why the biased estimator reports 0.00778 for two samples from
the *same* distribution while the unbiased one centres at zero. The demo measures
the difference over 200 replicates: $+0.02289$ for the biased estimator against
$+0.00200\pm0.00109$ for the unbiased one at $m=n=40$.

### Blocks

For large $n$, cut both samples into $B$ contiguous blocks of size $b$ and
average the unbiased estimate inside each:

$$
\widehat{\mathrm{MMD}}^2_{\text{block}}=
\frac1B\sum_{b=1}^{B}\widehat{\mathrm{MMD}}^2_u(X_b,Y_b).
\qquad (5)
$$

Each block is a valid (if noisy) unbiased estimate, so the average is unbiased
for the same quantity, with variance falling as $1/B$. Its cost is
$B\cdot b^2=nb$. With $b=\lfloor\sqrt n\rfloor$ that is $O(n^{1.5})$, the bias
/variance choice of Gretton et al.; with a fixed $b$ it is $O(n)$ but the
constant bias of a $b$-point estimate never goes away. The range of validity is
$2\le b\le\min(m,n)$, and the implementation refuses to leave it.

One condition is easy to miss: the bandwidth must be fixed **before** blocking.
If each block re-estimates $\gamma$ from its own six points, the block estimates
are no longer estimates of the same functional, and the average is biased by an
amount that has nothing to do with the sample size.

### The bandwidth

The median heuristic sets

$$
\gamma=\frac{1}{2\,\mathrm{median}_{i<j}\lVert z_i-z_j\rVert^2},
\qquad z\in X\cup Y,
\qquad (6)
$$

so the kernel value at the median squared distance is $\exp(-1/2)=0.6065$
regardless of the data's scale. That is the property the demo asserts: at
$\gamma=0.004960$ and median $d^2=100.8$, $k=0.6065$. The factor two matters;
without it the kernel sits at $e^{-1}$, which over-smooths. The heuristic is not
optimal, but it removes the units, which is the failure mode that actually
happens.

### Independence

HSIC measures the distance between the joint distribution $P_{xy}$ and the
product $P_xP_y$ in the RKHS of the tensor kernel $k\otimes l$. With Gram
matrices $K$ ($n\times n$ on $x$) and $L$ ($n\times n$ on $y$) and the centring
matrix $H=I-\mathbf{1}\mathbf{1}^{\top}/n$,

$$
\mathrm{HSIC}=\frac{1}{n^2}\,\mathrm{tr}\big(KHLH\big)
=\frac{1}{n^2}\sum_{i,j}\tilde K_{ij}\tilde L_{ij},
\qquad \tilde K=HKH,
\qquad (7)
$$

which is zero in the population if and only if $x$ and $y$ are independent, for
characteristic $k$ and $l$. Because both $\tilde K$ and $\tilde L$ are PSD, the
sum is a Frobenius inner product of two PSD matrices, so it is non-negative and
has a natural scale:

$$
\mathrm{nHSIC}=
\frac{\langle\tilde K,\tilde L\rangle_F}
{\lVert\tilde K\rVert_F\,\lVert\tilde L\rVert_F}\in[0,1].
\qquad (8)
$$

That cosine is what the `hsic` variant returns: 1 when $y$ is a deterministic
function of $x$, near 0 when they are independent, and invariant to the units of
either. In the demo, independent variables give 0.008 with $p=0.95$, and
$y=x^2+\text{noise}$ gives 0.378 with $p=0.005$ — a dependence no correlation
would find, since the relationship is symmetric around zero.

### The witness

For the RBF kernel the statistic has a representer, the witness function

$$
f(z)=\frac1m\sum_i k(x_i,z)-\frac1n\sum_j k(y_j,z),
\qquad \lVert f\rVert_{\mathcal H}=\mathrm{MMD},
\qquad (9)
$$

and its class means recover the biased statistic exactly:

$$
\frac1m\sum_i f(x_i)-\frac1n\sum_j f(y_j)
=\widehat{\mathrm{MMD}}^2_b .
\qquad (10)
$$

The demo asserts (10) to $10^{-12}$ and prints the value 0.012063 twice. It is a
better diagnostic than the scalar: a witness that is large on a few points names
the observations that differ.

### Calibration

Under $P=Q$ the labels are exchangeable, so every split of the pooled sample
into groups of sizes $m$ and $n$ is equally likely, and the statistic computed on
each split is a draw from the null. With $B$ permutations,

$$
p=\frac{1+\#\{b:\ T_b\ge T_{\text{obs}}\}}{1+B},
\qquad (11)
$$

the $+1$ in numerator and denominator counting the observed split itself. The
test is exact at level $\alpha$ whenever $\alpha$ is a multiple of $1/(1+B)$, and
the smallest attainable p-value is $1/(1+B)$ — 0.005 at $B=200$. Reporting
"$p<0.001$" from 200 permutations is a lie about the resolution.

For HSIC the null is different: $x$ stays fixed and the rows of $y$ are permuted,
which breaks the pairing while preserving each marginal. Pooling would destroy
the marginal distributions and test the wrong hypothesis. That is the `rows` mode,
and it is why the router picks the mode per variant rather than exposing it.

## Code walkthrough

Every statistic in the file is built from one function, so the kernel policy is
in one place. The Laplacian kernel uses the $L_1$ distance, not the square root
of the squared Euclidean one:

```python
    if kernel == "linear":
        return A @ B.T
    if gamma is None:
        raise ValueError("gamma is required for a non-linear kernel")
    if kernel == "rbf":
        return np.exp(-gamma * _sqdist(A, B))
    distances = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    return np.exp(-gamma * distances)
```

`gamma is None` raising rather than defaulting is deliberate: a silent default
bandwidth is the single most common way an MMD result becomes meaningless.

The median heuristic is (6), and the `upper` triangle is what makes it an
average over *pairs* rather than over an $n^2$ matrix that counts each pair
twice:

```python
    pooled = np.vstack([A, B])
    squared = _sqdist(pooled, pooled)
    upper = squared[np.triu_indices(pooled.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("all pooled points coincide; no bandwidth exists")
    return 1.0 / (2.0 * float(np.median(positive)))
```

The `positive` filter matters for duplicated rows; a zero distance would drag
the median to zero and the bandwidth to infinity. If *every* distance is zero,
there is no kernel at all, and the guard says so.

`mmd2` is (2) and (3). The unbiased branch removes the diagonal by subtracting
the trace, which is cheaper and less error-prone than masking the matrix:

```python
    if m < 2 or n < 2:
        raise ValueError("the unbiased estimator needs two rows per sample")
    within_x = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    within_y = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(within_x + within_y - 2.0 * kxy.mean())
```

The denominators are $m(m-1)$, not $m^2$: after removing $m$ diagonal entries
from an $m^2$ sum, $m(m-1)$ terms remain. Using $m^2$ is the classic off-by-one
here and makes the estimator biased towards zero.

The block estimator passes one resolved bandwidth into every block, which is the
whole correctness argument of (5):

```python
    gamma = _resolve_gamma(X, Y, kernel, gamma)
    m, n = X.shape[0], Y.shape[0]
    size = _block_size(m, n, block)
    count = min(m, n) // size
    if count < 1:
        raise ValueError("samples are shorter than one block")
    total = 0.0
    for start in range(0, count * size, size):
        xb = X[start:start + size]
        yb = Y[start:start + size]
        total += mmd2(xb, yb, kernel=kernel, gamma=gamma, unbiased=True)
    return float(total / count)
```

`count = min(m, n) // size` drops the tail of both samples past the last full
block, and that is the right trade: a ragged final block would have a different
variance from the others, so the average would no longer be an average of
identically distributed estimates.

HSIC is (7) and (8), and `_centre` is the double centring without forming $H$:

```python
    kc = _centre(_gram(X, X, kernel, gx))
    lc = _centre(_gram(Y, Y, kernel, gy))
    value = float(np.mean(kc * lc))
    if not normalize:
        return value
    denominator = float(np.sqrt(np.mean(kc * kc) * np.mean(lc * lc)))
    return value / denominator if denominator > EPS else 0.0
```

`np.mean(kc * lc)` is `tr(KHLH)/n²` because the trace of an elementwise product
is the sum of all entries. The normaliser is the product of Frobenius norms over
$n^2$, which is exactly the cosine of (8) — the `n²` cancels, and the guard
handles a constant variable whose centred block is zero.

The permutation loop builds the null in the two modes, and the p-value counts
ties as evidence against the null:

```python
    if mode == "pool":
        pooled = np.vstack([first, second])
        cut = first.shape[0]
        for i in range(n_perm):
            order = rng.permutation(pooled.shape[0])
            null[i] = statistic(pooled[order[:cut]], pooled[order[cut:]])
    else:
        if first.shape[0] != second.shape[0]:
            raise ValueError("mode='rows' needs samples of equal length")
        for i in range(n_perm):
            null[i] = statistic(first, second[rng.permutation(second.shape[0])])
    exceed = int(np.sum(null >= observed))
```

`>=` rather than `>` is what makes the test exact for discrete statistics, where
ties have positive probability. The `cut` is fixed at the original group size,
so every permutation is a split the original experiment could have produced.

The non-obvious line in the router is the one that resolves the bandwidth
*outside* the statistic closure:

```python
        gamma_value = _resolve_gamma(X, Y, kernel, gamma)
        gamma_y = None
        if variant == "linear_time":
            statistic = partial(linear_time_mmd2, kernel=kernel,
                                gamma=gamma_value, block=block)
```

If `gamma_value` were left as `None` and resolved inside `mmd2`, every
permutation would use the bandwidth of its own permuted split. The null would
then compare statistics computed under different kernels, which is not a null
distribution of anything. This is the bug that survives casual testing: p-values
still come out, they are just wrong.

The witness is (9), one matrix product against each sample:

```python
    pooled = np.vstack([X, Y])
    return (_gram(pooled, X, kernel, gamma).mean(axis=1)
            - _gram(pooled, Y, kernel, gamma).mean(axis=1))
```

Both blocks are $(m+n)\times m$ and $(m+n)\times n$, and the row means are the
two averages in (9). The identity (10) then follows from summing rows of
$K^{xx}$ and $K^{xy}$ — no separate code path is needed for the check.

## Complexity

With pooled size $N=m+n$, block size $b$ and $B$ permutations:

- **`quadratic`:** $O(N^2)$ time and memory per statistic, so $O(BN^2)$ for the
  test. At $N=2000$ one statistic is 4M kernel entries and the test is 200 of
  them.
- **`linear_time`:** $O(nb)$ time and $O(b^2)$ memory per block. With
  $b=\lfloor\sqrt n\rfloor$ that is $O(n^{1.5})$, and with $b$ fixed, $O(n)$.
- **`mmd_rbf`:** the quadratic cost plus one $(m+n)\times(m+n)$ pass for the
  witness, which is the same order.
- **`hsic`:** two $n\times n$ Gram matrices and three centrings, $O(n^2)$.
- **The bandwidth:** the median heuristic in (6) is $O(N^2)$ by itself. This is
  the uncomfortable line in the complexity table: a linear-time estimator with a
  median-heuristic bandwidth is not linear. Pass an explicit `gamma` estimated
  on a pilot sample if the $O(N^2)$ term matters.

Memory is the binding constraint everywhere. The permutation test's null array is
$O(B)$ and trivial; the Gram matrices are not.

## Failure modes

**The bandwidth is recomputed inside the null.** p-values still appear and are
slightly too small or too large, with no error message. Diagnose by checking that
every permutation used the same $\gamma$; the router's `gamma` key reports the
value it fixed, and `mmd2` never resolves a `None` silently.

**Reading a p-value finer than the permutation grid.** With 200 permutations the
smallest p-value is 0.005 and the test has essentially no power below
$\alpha=0.005$. If a result needs $p<0.001$, run 2 000 permutations or use the
asymptotic spectral estimate, not more decimal places from the same null.

**Significance is not distance.** MMD is consistent — with enough data it detects
any difference at all — so a large $n$ makes a trivial distributional difference
significant. Report $\widehat{\mathrm{MMD}}^2$ alongside $p$; the demo's shifted
pair has 0.09557 against 0.00778 for the null pair, a factor of twelve, which is
the effect size the p-value hides. The witness function says where the difference
lives.

**A bad kernel makes the test blind, not conservative.** A bandwidth far below
the data's scale turns every kernel entry into zero except the diagonal, and the
statistic loses all power while remaining valid. This is why (6) exists and why
the guard refuses a zero bandwidth rather than clipping it.

**HSIC is not a conditional independence test.** It detects any dependence
between two variables, including the dependence induced by a common cause, so it
cannot orient an edge in a causal graph on its own. $y=x^2$ scores 0.378 while
$\mathrm{corr}(x,y)\approx0$; the converse error — assuming a strong HSIC implies
a causal link — is the more expensive one.

## Applications

**Evaluating generative models.** Two-sample testing between generated and real
samples is the standard quantitative check on a GAN or VAE, and MMD is the
default because it needs no density estimate and no shared support. The same
statistic, uncalibrated, is a training loss in MMD-GAN.

**Detecting distribution shift.** Compare a live feature distribution against
the training one per batch; a permutation test with 200 draws is cheap enough to
run on every batch, and the witness names the drifting observations. For
covariate-shift correction the same statistic supplies the sample weights.

**Feature selection and independence screening.** HSIC between each feature and
the target ranks features by dependence rather than by linear correlation; HSIC
Lasso turns that ranking into a sparse selection. It finds $x^2$ effects and
periodic effects that a correlation filter misses.

**Domain adaptation and causal discovery.** MMD measures the distance between
source and target feature distributions, which is the quantity most adaptation
methods minimise; HSIC is a building block of kernel conditional-independence
tests used for skeleton discovery in causal graphs.

It is a trap when the samples are dependent (time series, paired measurements)
and the permutation null assumes exchangeability — block permutations or a
wild bootstrap are needed; and when $N$ is large enough that the quadratic
statistic does not fit, where the block estimator or a random Fourier feature
approximation of the kernel is the honest substitute.

## References

- Gretton, Borgwardt, Rasch, Schölkopf, Smola (2012). *A kernel two-sample
  test.* JMLR 13. The quadratic and linear-time statistics, the median
  heuristic, and the spectral null this file avoids by permuting.
- Gretton, Borgwardt, Rasch, Schölkopf, Smola (2006). *A kernel method for the
  two-sample problem.* NIPS. The original MMD test.
- Gretton, Fukumizu, Teo, Song, Schölkopf, Smola (2008). *A kernel statistical
  test of independence.* NIPS. HSIC as a test, with the gamma approximation.
- Song, Smola, Gretton, Bedo, Borgwardt (2012). *Feature selection via
  dependence maximization.* JMLR 13. HSIC Lasso, and the reason HSIC is in a
  feature-selection pipeline at all.
- Muandet, Fukumizu, Sriperumbudur, Schölkopf (2017). *Kernel mean embedding of
  distributions: a review.* Foundations and Trends in ML. The survey that makes
  (1) and (7) one idea.
