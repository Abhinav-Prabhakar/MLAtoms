# Kernel density estimation

`kde.py` · Density & anomaly · Tier 1 · 8 variants

## Intuition

Put the same small bump on every data point and add them up. That is the whole
estimator:

$$
\hat{p}(x) = \frac{1}{n h^d} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)
\qquad (1)$$

It has no parameters in the sense that a mixture model has parameters. Nothing is
fitted, nothing is iterated, and the sample is kept verbatim. The only decision
is the width $h$ of the bump — and that decision *is* the model:

```
  sample     |         |      |  |              |

  h small      /\     /\     /\/\        /\    /\
              /  \   /  \   /    \      /  \  /  \     spikes, gaps, no mass

  h right    _/    \_/    \_/      \____/    \/    \_   the shape of the data

  h large    \____________________.__________________/  one blob, all structure gone
```

Every method in this chapter is built on this estimate. A histogram is a KDE
with a box kernel and fixed bins; a copula is a KDE after reordering each
coordinate; LOF, isolation forest and the autoencoder in the atoms below are
answers to "how small is $\hat{p}$ here", with different definitions of small
and different tolerances for computing it. Getting the bandwidth wrong is the
one error that propagates into all of them.

The bias–variance structure is exact, and it is why the estimator has a rule of
thumb at all rather than a fitted parameter:

- **Bias** grows as $h^2 \nabla^2 p$: a wide kernel averages the density over a
  region, so it can only see curvature, and it flattens peaks by an amount that
  depends on the local second derivative.
- **Variance** falls as $1/(n h^d)$: a wide kernel averages more points.

Their sum is minimised at $h \propto n^{-1/(d+4)}$, which is the whole content
of the Silverman and Scott rules. Two consequences are unavoidable and both are
measured below. The constant in that optimum contains $\nabla^2 p$, which is
what you were trying to estimate, so every rule of thumb is a guess about shape
made from scale. And the exponent degrades with dimension: in $d$ dimensions,
halving the error needs $2^{d+4}$ times as many points, which is 32 in one
dimension and 16 384 in ten.

## Variants

`variant` is a triple — kernel, bandwidth rule, evaluator — and each row changes
one of the three.

| variant | kernel | bandwidth | evaluator | the trade |
|:--|:--|:--|:--|:--|
| `gaussian` | Gaussian (2) | Scott (6) | dense | smooth everywhere, mass in the tails |
| `epanechnikov` | Epanechnikov (3) | Scott | dense | minimum-variance order-2 kernel |
| `tophat` | top-hat (4) | Scott | dense | fastest, discontinuous estimate, 0.930 efficiency |
| `silverman` | Gaussian | Silverman (5) | dense | survives a heavy tail |
| `scott` | Gaussian | Scott (6) | dense | the default, and fooled by one outlier |
| `cv_bandwidth` | Gaussian | leave-one-out (7) | dense | shape-aware, and it undersmooths |
| `tree_kde` | Gaussian | scalar or fixed | ball tree | the same estimate for less work |
| `product_kernel` | Gaussian | per dimension (6) | dense | a different scale on each axis |

The kernel choice is the cheap one, and the demo measures how cheap. Two
hundred points from a bimodal mixture, all three kernels at Scott's bandwidth
$h = 0.415$, against the true density on 901 grid points:

| kernel | integrated squared error |
|:--|--:|
| Gaussian | 8.76e-03 |
| Epanechnikov | 1.05e-02 |
| top-hat | 8.22e-03 |

A factor of 1.3 between the best and the worst kernel — and the top-hat wins,
which is noise, since the three asymptotic efficiencies are 1.000, 0.951 and
0.930. Changing $h$ instead, to the cross-validated 0.119, costs a factor of
2.4. One number decides the estimate; the kernel only decides the constant in
front of the error.

The two rules disagree by more than that on a skewed sample. Two hundred
lognormal points, where the standard deviation is inflated by the tail:

| rule | bandwidth | leave-one-out log-likelihood |
|:--|--:|--:|
| Scott (6) | 0.976 | −2.0984 |
| Silverman (5) | 0.420 | −8.6035 |
| cross-validated (7) | 1.266 | −2.0616 |

Silverman's robust scale is 1.347 against a standard deviation of 2.816, so the
two formulas differ by a factor of 2.3 on the same data, and the likelihood
prefers the wider of the two — on this sample the tail is signal, not
contamination. That is the honest summary of bandwidth selection: a formula
encodes an assumption about shape, cross-validation encodes the data, and
neither is free. The formulas cost one pass over the sample. Cross-validation
costs sixty likelihoods of an $n \times n$ kernel, which is why it is not the
default.

## Derivation

### 1. The estimator as a convolution

$\hat{p}$ is the empirical measure convolved with the scaled kernel,
$\hat{p} = \mathbb{P}_n * K_h$. Taking expectations under the true density,

$$
\mathbb{E}\,\hat{p}(x) = \int p(y)\, K_h(x - y)\, dy = (p * K_h)(x)
$$

so the bias is the error of a convolution approximation, which Taylor expands to

$$
\mathbb{E}\,\hat{p}(x) - p(x) = \tfrac{1}{2} h^2 \sigma_K^2 \nabla^2 p(x) + O(h^4),
\qquad \sigma_K^2 = \int \lVert u \rVert^2 K(u)\,du
$$

and the variance follows from independence of the summands:

$$
\mathrm{Var}\,\hat{p}(x) = \frac{1}{n h^d} R(K)\, p(x) + O(n^{-1}),
\qquad R(K) = \int K(u)^2\,du
$$

Integrating squared bias plus variance gives the asymptotic mean integrated
squared error, and differentiating it in $h$ gives the optimum:

$$
\mathrm{AMISE}(h) = \tfrac{1}{4} h^4 \sigma_K^4 R(\nabla^2 p)
+ \frac{R(K)}{n h^d},
\qquad
h^\star = \left[\frac{d\, R(K)}{n\, \sigma_K^4\, R(\nabla^2 p)}\right]^{1/(d+4)}
$$

with the minimum AMISE falling like $n^{-4/(d+4)}$. That pair is why every rule
in the file has the same $n^{-1/(d+4)}$ shape and only the constant differs: the
constant is $R(\nabla^2 p)^{-1/(d+4)}$, and no rule can know it. Scott's rule
replaces it with the value it takes for a Gaussian, which is where the $1.06$ of
the one-dimensional version comes from; Silverman's replaces it with a robust
scale so that one far outlier cannot set the width of the whole sample.

The exponent is the part with teeth. To halve the error you multiply $n$ by
$2^{d+4}$:

| dimension $d$ | 1 | 2 | 3 | 5 | 10 |
|:--|--:|--:|--:|--:|--:|
| points needed to halve the error | 32 | 64 | 128 | 512 | 16 384 |

### 2. The three kernels

With $V_d = \pi^{d/2}/\Gamma(d/2 + 1)$ the volume of the unit ball,

$$
K(u) = (2\pi)^{-d/2} \exp\!\left(-\tfrac{1}{2}\lVert u\rVert^2\right)
\qquad (2)$$

$$
K(u) = \frac{d+2}{2 V_d}\left(1 - \lVert u\rVert^2\right)_+
\qquad (3)$$

$$
K(u) = \frac{1}{V_d}\,\mathbf{1}\!\left[\lVert u\rVert \le 1\right]
\qquad (4)$$

The constants are not decoration. For (3), integrate in spherical shells with
surface area $S_{d-1} = d V_d$:

$$
\int K = \frac{d+2}{2 V_d} \cdot d V_d \int_0^1 (1 - r^2) r^{d-1} dr
= \frac{d+2}{2}\left(\frac{1}{d} - \frac{1}{d+2}\right) = 1
$$

which is the one line a from-scratch implementation usually gets wrong, in the
same way that (4) is wrong without $V_d$: a unit cube's worth of intuition in
three dimensions is off by $4\pi/3$. The demo checks all three kernels by
radial quadrature in $d = 1, 2, 3$ and the worst error is $2.1 \times 10^{-10}$,
which is the quadrature's error and not the constants'.

All three are non-negative, integrate to one, have their second moment fixed by
the bandwidth, and are of order two — they reproduce linear functions exactly,
so the bias starts at $h^2$ and no kernel is better than another asymptotically
except through the constant in the AMISE. Epanechnikov's minimises $R(K)$ subject
to that constraint, which is the sense in which it is optimal; the Gaussian's
advantage is that it is smooth, so the estimate is differentiable everywhere and
the constants in the next order of the bias expansion are finite.

### 3. Rules of thumb

Silverman's rule is the one-dimensional rule applied per column:

$$
h_S = 0.9 \min\!\left(\hat{\sigma},\ \frac{\mathrm{IQR}}{1.349}\right)\, n^{-1/(d+4)}
\qquad (5)$$

and Scott's is its non-robust sibling,

$$
h_{\text{scott}} = \hat{\sigma}\, n^{-1/(d+4)}
\qquad (6)$$

The $1.349$ is $2\Phi^{-1}(0.75)$, so $\mathrm{IQR}/1.349$ is a standard
deviation estimated from the middle half of the sample, and it is what makes
(5) survive a heavy tail. The $1.06$ that appears in some textbook versions of
Scott's rule is the one-dimensional Gaussian constant; here the coefficient is
$1$ and the exponent already contains the dimension, which is the multivariate
form.

Both are scale-equivariant: multiplying the data by $c$ multiplies $h$ by $c$
and leaves the estimate unchanged. Neither is shape-equivariant, and that is
the failure the demo measures — a bimodal sample with modes at $-1.2$ and $1.0$
has $\hat\sigma = 1.197$, which is mostly the gap between the modes, so Scott's
rule returns $h = 0.415$ and smooths two peaks into one hump. Silverman's rule
does the same thing whenever the modes are further apart than a standard
deviation.

### 4. Bandwidth by cross-validation

The alternative to a formula is to choose $h$ by the criterion you will be
judged on, and the natural one is the likelihood of the data under the model.
The in-sample likelihood is useless — it increases without bound as $h \to 0$,
since the estimate becomes a sum of delta functions — so each point is scored
against the estimate built from the *other* points:

$$
\hat{h} = \arg\max_h \sum_i \log\!\left(\frac{1}{n-1}
\sum_{j \ne i} K_h(x_i - x_j)\right)
\qquad (7)$$

Computing the inner sum for all $i$ at once is one $n \times n$ kernel matrix
with its diagonal removed, which is why (7) is a matrix expression rather than
$n$ separate estimates. The optimisation is one-dimensional, and the objective
is smooth and unimodal in $\log h$ on any sensible range, so sixty golden-section
steps find the argmax to machine precision without a grid.

Equation (7) is not a good estimator of the ISE-optimal bandwidth, and the demo
shows it failing by a factor of 2.4. The reason is in the expression: each term
contains $-\log h$ from the normalisation, so the objective rewards shrinking
$h$ until the nearest neighbour falls outside the kernel. On the bimodal sample
it chooses $h = 0.119$ — less than a third of Scott's — where the leave-one-out
likelihood really is higher, and the integrated squared error is
$2.11 \times 10^{-2}$ against $8.76 \times 10^{-3}$. The likelihood is a local
criterion: it weights relative error where the data actually is, so it chases
sharp modes and does not care that the gaps between them are wrong.
Least-squares cross-validation, which compares $\hat{p}$ with a held-out
estimate instead of evaluating it at the sample, tracks ISE and is what a
production selector should use; it is not implemented here because the point of
this atom is the estimator, not the selector zoo.

### 5. Pruning a tree

The kernel is a decreasing function of distance, so a ball gives a bound for
free. For a node with centre $c$, radius $r$ and $m$ points, no point in the
node is closer to a query $x$ than $\max(\lVert x - c\rVert - r, 0)$, so

$$
\sum_{i \in \text{node}} K_h(x - x_i) \;\le\;
\frac{m}{h^d}\, K\!\left(\left(\frac{\max(\lVert x - c\rVert - r, 0)}{h}\right)^2\right)
\qquad (8)$$

If that bound is negligible against the sum accumulated so far, the whole
subtree is skipped. Two details make the difference between a bound that helps
and one that does not. The comparison is against the *running* sum, so a node's
worth is judged relative to what is already known rather than to an absolute
threshold. And the bound can be exactly zero: for the Gaussian, radii beyond
about $38h$ underflow, and a far query then prunes everything. Both cases are in
the demo, and the second one is a bug in the obvious implementation — a purely
relative test never fires while the running sum is still zero, which is exactly
the far field.

The guarantee is per node: each skipped node contributes less than the tolerance
times the sum at the moment it was tested, so the total relative error is
bounded by the tolerance times the number of skipped nodes, not by the tolerance
itself. With `tol = 1e-8` and a few hundred nodes the worst case is $10^{-5}$;
measured, the tree agrees with the brute-force sum to $4.1 \times 10^{-11}$ in
log density.

### 6. The product kernel

A per-dimension bandwidth replaces the isotropic scaling with a diagonal one:

$$
K_h(x - x_i) = \prod_{j=1}^{d} \frac{1}{h_j}\,
k\!\left(\frac{x_j - x_{ij}}{h_j}\right)
\qquad (9)$$

For the Gaussian kernel this is a Mahalanobis distance with
$\Sigma = \mathrm{diag}(h_1^2, \dots, h_d^2)$: dividing the difference by $h$
before squaring is all the code needs, and (9) then falls out of the same three
lines. With all $h_j$ equal it is the isotropic estimator exactly — the demo
asserts bit-for-bit agreement — so `product_kernel` is not a different
algorithm, it is the same algorithm with a scale per axis. What it cannot do is
follow a correlation: a ridge at 45° in two dimensions needs an off-diagonal
term, and the diagonal family can only widen both axes until the ridge is a
blob. Whiten the data first, or use a full-bandwidth estimator.

## Code walkthrough

### 1. The kernels and their constants

```python
    if name == "gaussian":
        return np.exp(-0.5 * u2) / (2.0 * math.pi) ** (n_features / 2.0)
    volume = ball_volume(n_features)
    if name == "tophat":
        return np.where(u2 <= 1.0, 1.0 / volume, 0.0)
    if name == "epanechnikov":
        return np.where(u2 <= 1.0, (n_features + 2.0) / (2.0 * volume) * (1.0 - u2), 0.0)
```

The function takes a *squared* radius because that is what the distance code
produces, and it saves a square root per pair. `ball_volume` is
$\pi^{d/2}/\Gamma(d/2+1)$, so the tophat's constant is $1/V_d$ and the
Epanechnikov's is $(d+2)/(2V_d)$ — the normalisation integral above, read
backwards. `np.where` evaluates both branches and then selects: for the compact
kernels the discarded branch is finite, so this is safe, and it lets the caller
pass a whole distance matrix instead of looping. A Python branch would be
marginally faster and would force that loop, which is the design constraint that
matters.

### 2. Distances in bandwidth units

```python
    offset = (A[:, None, :] - B[None, :, :]) / bandwidth
    return np.einsum("ijk,ijk->ij", offset, offset)
```

Three lines, and the order of operations is the point: divide by the bandwidth,
*then* square. Squaring first and dividing by $h^2$ is the same number for a
scalar bandwidth and a different algorithm for a vector one — the product kernel
in (9) is exactly the decision to divide per dimension. The broadcast produces
an $(m, n, d)$ difference array, and `einsum` contracts the last axis without
materialising a second array; `(offset ** 2).sum(axis=2)` is the readable
version and allocates one more $m n d$ block.

### 3. The rules of thumb

```python
    return X.std(axis=0, ddof=1) * n ** (-1.0 / (d + 4.0))
```

```python
    q75, q25 = np.percentile(X, [75.0, 25.0], axis=0)
    robust = np.minimum(X.std(axis=0, ddof=1), (q75 - q25) / 1.349)
    return 0.9 * robust * n ** (-1.0 / (d + 4.0))
```

`ddof=1` because this is a sample standard deviation; the difference between $n$
and $n-1$ is not the point, but the convention should match every other
statistic in the chapter. The exponent is written with float literals so the
expression does not depend on the caller's integer type. The `np.minimum` in the
robust version is the entire content of Silverman's rule: on the lognormal
sample the two candidate scales are 2.816 and 1.347, so the rule takes 1.347 and
the bandwidth drops from 0.976 to 0.420.

### 4. Choosing, with a precedence order

```python
        if self.bandwidth is not None:
            given = np.asarray(self.bandwidth, dtype=float)
            if given.ndim == 0:
                return np.full(d, float(given))
            if given.shape != (d,):
                raise ValueError(f"bandwidth must be a scalar or shape ({d},)")
            return given.copy()
        if self.variant == "silverman":
            return silverman_factor(X)
```

An explicit bandwidth wins over the variant, always. That is what makes the
variant list usable as a set of defaults rather than a set of algorithms: the
demo fits `gaussian`, `epanechnikov` and `tophat` at the same $h$ to isolate the
kernel, and fits `tree_kde` at a fixed 0.15 because Scott's rule reads the
spread of all three blobs and returns a bandwidth wider than a blob. The
`copy()` is not paranoia — the returned array is stored and later used to
divide, so an aliased caller array would be a silent mutation bug.

```python
        if self.variant == "tree_kde":
            # One bandwidth for every dimension is what the node bound in
            # `_tree_sum` assumes; the median is the least surprising summary.
            return np.full(d, float(np.median(base)))
```

The tree's bound is only valid for an isotropic kernel, so the variant reduces
Scott's vector to its median rather than pretending a per-axis bandwidth is
fine. Passing a genuine vector raises instead, which the failure-mode demo
checks: the alternative is a tree that is fast and quietly wrong.

### 5. The leave-one-out likelihood

```python
        sq = _sq_distances(X, X, np.full(X.shape[1], h))
        kernel = kernel_value(sq, X.shape[1], "gaussian") / h ** X.shape[1]
        held_out = kernel.sum(axis=1) - np.diag(kernel)
        return float(np.mean(np.log(np.maximum(held_out / (len(X) - 1), 1e-300))))
```

Equation (7) as one matrix expression. `kernel.sum(axis=1)` includes $x_i$'s own
bump, and subtracting the diagonal removes the only term that must not be there —
cheaper than building the matrix with a hole in it, and exact, since the diagonal
entries are exactly $K(0)/h^d$ and not a floating-point approximation of it. The
`np.maximum` guards the log: a point far from every other point has a held-out
density of zero, and `log 0` would make the objective $-\infty$ and break the
golden section. Dividing by $n-1$ rather than $n$ keeps the number comparable to
a log density.

```python
        return math.exp(_golden_max(
            lambda log_h: self._loo_loglik(X, math.exp(log_h)),
            math.log(h0 / 4.0), math.log(h0 * 4.0)))
```

The search is in $\log h$ because the objective is asymptotically a function of
$\log h$ and its curvature is far more uniform there, which is what lets sixty
steps of golden section be enough. The bracket is a factor of four either side
of the rule of thumb, which is a *prior*: if the optimum lay outside it the
search would return the boundary and the model would silently use a bandwidth
nobody chose. That is a real risk with multimodal data and the reason the
returned bandwidth should be inspected rather than trusted.

### 6. Scoring

```python
        scale = float(np.prod(self.bandwidth_))
        if self.variant == "tree_kde":
            self.evaluations_ = 0
            sums = np.array([self._tree_sum(row) for row in X])
        else:
            sq = _sq_distances(X, self.data_, self.bandwidth_)
            kernel = kernel_value(sq, self.n_features_, self.kernel_)
            sums = kernel.sum(axis=1)
        density = sums / (len(self.data_) * scale)
```

The normaliser is $\prod_j h_j$, not $h^d$: with a per-dimension bandwidth the
Jacobian of the scaling is the product, and using a mean instead would silently
produce something that does not integrate to one. The tree branch and the dense
branch compute the *same* quantity — that is the assertion in the demo — and
differ only in how many kernel values they evaluate. The `1e-300` floor is the
only place the code gives up on exactness: it turns a genuine zero density into
$-690.8$, which is finite and therefore comparable, and it is also why a
far-away query reports the same number however far away it is.

### 7. The pruning test

```python
            gap = max(float(np.linalg.norm(row - node.centre)) - node.radius, 0.0) / h
            # Every kernel above is a decreasing function of radius, so the
            # nearest point in the node is the best case for the node's whole
            # contribution -- and if even that is negligible, so is the node.
            bound = node.count * kernel_value(gap * gap, d, self.kernel_) / h ** d
            # Exact when the bound underflows, relative once the running sum is
            # non-zero. Both are needed: in the far field every node evaluates to
            # zero and a relative test alone would never fire.
            if bound == 0.0 or (total > 0.0 and bound < self.tol * total):
                continue
```

Equation (8), and then the two clauses that make it work. `max(..., 0.0)` is
the case where the query is inside the node: the closest point could be the
query itself, so the gap is zero and the bound is the kernel's maximum. The
`bound == 0.0` clause looks like a micro-optimisation and is not: for a query
twenty units from a cloud with $h = 0.15$, every radius is beyond $100h$, the
Gaussian underflows, and the bound is exactly zero. Without that clause the
condition is `0.0 < 0.0`, which is false, so nothing is pruned and the tree
evaluates all 45 000 kernels — the demo measured that before the clause existed.

### 8. Growing the tree

```python
    spread = np.ptp(points, axis=0)
    axis = int(np.argmax(spread))
    if len(indices) <= leaf_size or spread[axis] == 0.0:
        return node
    order = indices[np.argsort(points[:, axis], kind="stable")]
    middle = len(order) // 2
    node.indices = None  # interior nodes are summarised by centre and radius
```

A median split on the widest axis: $O(n \log n)$ to build, balanced by
construction, and — the part that matters for (8) — each node's centre and
radius are the tightest ball around its own points, computed once. `stable`
sorting makes the build deterministic for tied coordinates, which is what lets
the demo be reproducible. Setting `indices = None` on interior nodes is a
memory decision with a diagnostic value: a leaf is the only kind of node that
can be evaluated, so a bug that tries to read `indices` off an interior node
fails loudly instead of summing a subset.

## Complexity

For $n$ points in $d$ dimensions, $m$ queries, and a cross-validation run of $T$
objective evaluations:

| stage | time | memory |
|:--|:--|:--|
| Scott or Silverman bandwidth | $O(nd)$ | $O(d)$ |
| cross-validated bandwidth | $O(T n^2 d)$ | $O(n^2)$ |
| tree build | $O(n \log n)$ | $O(n \log n)$ |
| score, dense | $O(nmd)$ | $O(nmd)$ |
| score, tree | between $O(m \log n)$ and $O(nmd)$ | $O(n \log n)$ |

The dense path allocates the whole $m \times n \times d$ difference block, which
is why the demo is sized in the thousands and why every production KDE loops
over chunks of queries. The tree's $\log n$ term is the traversal; the range is
the honest one, because how much of the tree is visited is a property of the
query's position, not of $n$ — the demo measured 37%, 65% and 100% of the kernel
evaluations skipped for queries in a blob, between blobs, and far away. A
dual-tree implementation, which prunes on node *pairs* instead of
point-to-node, is what turns this into $O(n)$ for a whole data set and is the
version in the published libraries.

## Failure modes

### Likelihood cross-validation undersmooths

This is the one to know, because it looks like a success. On the bimodal sample
the leave-one-out likelihood peaks at $h = 0.119$ and every step of the
optimisation is correct: the objective really is higher there. The integrated
squared error is 2.4 times the rule-of-thumb value. The mechanism is the
$-\log h$ in each term: narrow kernels are rewarded for being narrow, and the
reward is only cancelled once the nearest neighbour falls outside the kernel,
which on this sample is three orders of magnitude below the chosen $h$. If a
selector based on (7) returns a bandwidth much smaller than a rule of thumb,
that is not a discovery about the data; it is the objective doing what it is
defined to do. Use least-squares cross-validation, or compare the two estimates
by eye.

### Boundaries

A kernel centred near the edge of the support puts half its mass where there is
no data, so the estimate is biased down by roughly the missing mass. Five
hundred uniform points on $[0, 1]$, $h = 0.1$: the estimate is $0.542$ at $x = 0$
and $1.039$ at $x = 0.5$, against a truth of $1.000$. That is a 46% deficit at
exactly the place where an anomaly detector is most likely to be asked a
question — "is this value unusually small" — and no bandwidth fixes it: widening
$h$ spreads the deficit and shrinking it sharpens it. The fixes are structural:
reflect the data about the boundary, fit on a transformed scale, or use a local
likelihood estimator that knows where the support ends.

### Empty space in high dimensions

$h^\star \propto n^{-1/(d+4)}$ is not a curiosity; it is the reason the next ten
atoms exist. At $d = 10$ the bandwidth shrinks as $n^{-1/14}$, so the number of
points inside a kernel of width $h$ around a query grows like
$n h^d = n^{4/(d+4)}$ — that is $n^{0.29}$ — and the estimate's relative error
falls at that rate instead of the $n^{-1/2}$ a parametric model gets. With
$n = 10^4$ that is a relative error around 7%; with $n = 10^6$, 2%. The
diagnostic is not the error, it is the count: if the 20 nearest neighbours are
all further than $h$, the local density estimate is an interpolation between
points that are nowhere near, and the number it returns is arithmetic rather
than evidence.

### A zero density and the floor

`score_samples` returns $\log \max(\hat p, 10^{-300})$, so a query with no
neighbours within reach reports $-690.8$ — and so does every other such query,
however far away it is. Two points a hundred units and a million units from the
data are indistinguishable in the output, and so are "extremely unlikely" and
"impossible". That is fine for ranking and for thresholding at a quantile; it is
not fine for the likelihood of a model comparison or for anything that averages
log densities, where the floor contributes a constant that has nothing to do
with the data. The tree demo shows the mechanism from the other side: the 75
far queries evaluate no kernels at all and return the floor exactly.

### Collinearity and duplicated columns

Nothing checks that the columns are independent, and a duplicated column is
silently a different estimator: it turns the density of $x_1$ into the density
of the diagonal $x_1 = x_2$, and the same point scores 0.259 nats differently
with no information added. A product kernel on correlated data has the same
problem in a milder form — the diagonal bandwidth family cannot follow a ridge,
so it widens both axes until the ridge is isotropic. Whiten first, then fit.

## Applications

- **Anomaly detection.** Score $-\log \hat{p}(x_i)$ and threshold at a quantile
  of the training scores; this is the baseline every detector in this chapter is
  compared against, and on low-dimensional data it is hard to beat. The threshold
  is where KDE and the rest of the chapter meet: the estimator gives a number,
  the quantile gives a decision, and the false-positive rate is the quantile by
  construction.
- **Class-conditional densities.** A generative classifier fits one KDE per
  class and applies Bayes' rule; it handles multimodal classes that a Gaussian
  naive Bayes cannot, at the cost of $n_c$ points per class rather than $d^2$
  parameters.
- **Mode seeking.** The gradient of (1) points uphill, and following it is mean
  shift — which is why bandwidth selection and the number of clusters in mean
  shift are the same question.
- **Mutual information and entropy.** The Kozachenko–Leonenko estimator is a
  KDE in disguise: $-\log \hat p(x_i)$ averaged over the sample estimates the
  differential entropy, and the nearest-neighbour form drops the kernel entirely.
  Every "estimate MI by binning" pipeline is a histogram version of the same
  identity.
- **Smoothing as an operation.** Convolution with $K_h$ is a low-pass filter, so
  the same code smooths a spatial field, a histogram or a spectral estimate. The
  bandwidth is a scale, and choosing it is a choice about which scales are signal.
- **Where it is a trap.** Above roughly five dimensions the estimator's
  convergence rate collapses, the boundary bias is everywhere, and the sample
  size needed is not available. That is the regime of the scorers: LOF compares
  local densities instead of estimating them, isolation forest measures isolation
  instead of density, and COPOD uses a parametric copula. They are not better
  estimators; they are better questions.

## References

- M. Rosenblatt, *Remarks on some nonparametric estimates of a density
  function*, Annals of Mathematical Statistics 27(3):832–837 (1956). The
  estimator, one year before Parzen.
- E. Parzen, *On estimation of a probability density function and mode*, Annals
  of Mathematical Statistics 33(3):1065–1076 (1962). The name, the consistency
  conditions and the mode-seeking reading.
- V. A. Epanechnikov, *Non-parametric estimation of a multivariate probability
  density*, Theory of Probability and Its Applications 14(1):153–158 (1969).
  The kernel that minimises $R(K)$, i.e. equation (3).
- B. W. Silverman, *Density Estimation for Statistics and Data Analysis*,
  Chapman and Hall (1986). The rule in (5), the bias–variance expansion, and the
  caution about likelihood cross-validation.
- D. W. Scott, *Multivariate Density Estimation: Theory, Practice, and
  Visualization*, Wiley (1992). The multivariate rule in (6) and the dimension
  dependence in the optimal bandwidth.
- M. P. Wand and M. C. Jones, *Kernel Smoothing*, Chapman and Hall (1995). The
  AMISE calculus and every variant of the kernel family.
- A. W. Bowman, *An alternative method of cross-validation for the smoothing of
  density estimates*, Biometrika 71(2):353–360 (1984). Least-squares
  cross-validation, the selector that does not undersmooth.
- P. Hall, *On Kullback-Leibler loss and density estimation*, Annals of
  Statistics 15(4):1491–1519 (1987). Why likelihood cross-validation is
  inconsistent, in the form the failure-mode section describes.
- A. G. Gray and A. W. Moore, *Nonparametric density estimation: toward
  computational tractability*, SIAM International Conference on Data Mining
  (2003). The dual-tree algorithm and the pruning bound in (8).
- C. R. Loader, *Local Regression and Likelihood*, Springer (1999). Local
  likelihood, the structural fix for the boundary bias above.
- L. F. Kozachenko and N. N. Leonenko, *Sample estimate of the entropy of a
  random vector*, Problems of Information Transmission 23(2):95–101 (1987).
  The nearest-neighbour entropy estimator, i.e. KDE without the kernel.
