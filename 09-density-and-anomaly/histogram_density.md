# Histogram density

`histogram_density.py` · Density & anomaly · Tier 1 · 4 variants

## Intuition

Cut the line into bins, count the points in each, divide by the bin's width. That
is a density estimate, and it is the one every plotting library draws when you
ask for a histogram:

```
   density
      │        ┌──┐
      │   ┌──┐ │  │ ┌──┐
      │   │  │ │  │ │  │
      │───┘  └─┘  └─┘  └───
      └───┬────┬────┬────┬──→ x
        bin  bin  bin  bin
```

Each bar's height is a count over a width, so the *area* is a probability and the
whole picture integrates to one. The estimate is piecewise constant, and that is
the entire difference from the KDE of the previous atom: a KDE puts a smooth bump
on every point and adds, a histogram puts every point in a box and divides.

$$
\hat{p}(x) = \frac{\text{count of points in } \mathrm{bin}(x)}
{n \cdot \text{width of } \mathrm{bin}(x)}
\qquad (1)$$

The only decision is the binning, and there are three classical answers plus one
modern one:

$$
\text{Freedman--Diaconis:}\quad h = 2\,\mathrm{IQR}\,n^{-1/3}
\qquad (2)$$

$$
\text{Scott:}\quad h = 3.49\,\hat{\sigma}\,n^{-1/3}
\qquad (3)$$

$$
\text{Sturges:}\quad k = \lceil \log_2 n \rceil + 1
\qquad (4)$$

$$
\text{Bayesian blocks:}\quad
\max_{\text{partitions}} \sum_{b} N_b \log\frac{N_b}{W_b} - \text{prior}\cdot B
\qquad (5)$$

The first three choose a constant width from a summary statistic; the last
chooses a partition and lets the width vary from block to block. All three rules
have the same $n^{-1/3}$, which is the bias–variance optimum for a
piecewise-constant estimate — a slower rate than the KDE's $n^{-1/5}$ in width
and $n^{-4/5}$ in error, and the reason a histogram is a fast baseline rather
than a competitor. Equations (2) and (3) differ only in how they estimate the
scale: Scott uses the standard deviation and is exact under a Gaussian
assumption, Freedman–Diaconis uses the interquartile range and survives a heavy
tail. Sturges does not look at the data's spread at all, which is why it is the
default in old software and the worst of the three on anything skewed.

The whole estimator is `np.histogram(x, bins="auto")` in numpy, with the rule in
place of `"auto"` and the counts divided by $n$ times the widths. What the rest
of this atom adds is the choice of rule, the product across features, and the
arithmetic for scoring new points.

The demo measures the whole story: on a lognormal sample, (2) asks for 29 bins of
width 0.214, (3) asks for 19 of width 0.326, and (4) asks for 10. All three
integrate to one to $10^{-12}$. Quadrupling the sample cuts the integrated squared
error by 0.53, against the $n^{-2/3}$ prediction of 0.40 and the KDE's 0.33 — the
histogram learns, just more slowly. And translating the sample by half a bin
moves 12.8% of the probability mass to a different bin, because the bin edges
move with the data: a histogram is not translation invariant, and that is a
property of the estimator, not a bug in this one.

## Variants

| variant | rule | width | measured on 500 lognormal points |
|:--|:--|:--|:--|
| `fd_rule` | (2) | constant, robust scale | 29 bins, $h = 0.214$ |
| `sturges` | (4) | constant, no data | 10 bins |
| `scott` | (3) | constant, Gaussian scale | 19 bins, $h = 0.326$ |
| `bayesian_blocks` | (5) | varies per block | 2 blocks on a two-level sample, prior 4 |

The first three are one line apart in the code — they all end in
`np.linspace(min, max, k + 1)` — and the demo makes them comparable on the same
data by reporting both the target width and the width actually used: the rule
gives a target $h$, and the implementation then uses the smallest number of
equal bins that achieves it, so the realised width is at most the target (0.2137
against 0.2152 for Freedman–Diaconis).

`sturges` is the odd one: ten bins for any sample between 257 and 512 points,
regardless of whether the data are a spike or a uniform. It was derived for the
binomial coefficients of a normal and for samples of a few hundred; at $n =
10^6$ it still asks for about 21 bins, where (2) asks for hundreds. The demo's
skewed sample makes the difference visible: ten bins hide the tail completely,
twenty-nine resolve it.

`bayesian_blocks` is a different kind of answer. On 800 points uniform on
$[0,1]$ and 400 uniform on $[1,3]$ — a density that drops by a factor of four at
$x = 1$ — the dynamic program with prior 1 finds 84 blocks, prior 2 gives 21, and
prior 4 gives 2. The two-block answer is the truth, and its boundary is at
0.997, one grid step from 1. The prior is not estimated from the data; it is a
statement about how many change points you expect. That is the honest trade: the
partition is exact for a given prior and the prior is a judgement.

## Derivation

### 1. Why the rate is n^{-1/3}

Write the point's position inside its bin as $x = a + uh$ with $u \in [0, 1)$.
The expected estimate is the average of $p$ over the bin,

$$
\mathbb{E}\,\hat{p}(x) = \frac{1}{h}\int_a^{a+h} p(t)\,dt
= p(x) + h\left(\tfrac{1}{2} - u\right)p'(x) + O(h^2)
$$

so the bias is *first order* in $h$, not second: it is the sawtooth in $u$
visible when you compare a point near a bin edge with one at its centre. Averaging
$(\tfrac{1}{2}-u)^2$ over the bin gives $1/12$, and the variance of a Poisson
count is $p(x)/(nh)$, so

$$
\mathrm{AMISE}(h) = \frac{h^2}{12}\int p'(x)^2\,dx + \frac{1}{nh}
\qquad (6)$$

Differentiating and solving gives

$$
h^\star = \left(\frac{6}{n\int p'^2}\right)^{1/3}
\propto n^{-1/3},
\qquad
\mathrm{MISE}(h^\star) \propto n^{-2/3}
\qquad (7)$$

The constant is $\int p'^2$, which is exactly as unknown as the $\int p''^2$ in
the KDE's optimum. For a normal density $\int p'^2 = 1/(4\sqrt{\pi}\sigma^3)$, so

$$
h^\star = (24\sqrt{\pi})^{1/3}\,\sigma\, n^{-1/3}
= 3.49\,\sigma\, n^{-1/3}
\qquad (8)$$

which is Scott's rule (3). It is not a heuristic: it is the normal-reference
optimum for a histogram, and the constant $3.49$ is $(24\sqrt{\pi})^{1/3}$ rather
than a fitted number. Freedman–Diaconis keeps the same exponent and replaces
$\sigma$ with $\mathrm{IQR}/1.349$, which is the normal-consistent scale
estimated from the middle half of the sample; on a Gaussian the two differ by a
factor of $3.49/2.70 = 1.29$ in width. On the demo's skewed sample the robust
scale is smaller still, so the gap widens: 29 bins from Freedman–Diaconis against
19 from Scott.

The rate is the part worth remembering. The histogram's width shrinks like
$n^{-1/3}$ and its error like $n^{-2/3}$; the KDE's width shrinks like
$n^{-1/5}$ and its error like $n^{-4/5}$. The histogram is paying for the
discontinuity at every bin edge — the bias in (6) is first order because the
estimate jumps, where a kernel's smoothness makes its bias second order. The demo
checks the rate on a standard normal: 2000 points give ISE $2.56\times10^{-3}$,
8000 points give $1.35\times10^{-3}$, and the ratio 0.53 brackets the predicted
0.40 from above, as a finite-sample measurement should.

### 2. The product form, and what integrates to one

For $d$ features, this implementation fits one histogram per column and takes the
product of the per-axis densities. Each axis integrates to one because each
density is counts over $n$ times width, so the product integrates to one over
$\mathbb{R}^d$. This is the same decision as `product_kernel` in the KDE atom and
it has the same limitation: it cannot represent a correlation between columns.
A joint histogram would capture the correlation and would need $k^d$ cells, which
is the emptiness of §3 measured directly.

### 3. Why high dimensions kill it

In $d$ dimensions a grid with $k$ bins per axis has $k^d$ cells. The number of
points is $n$ and the number of occupied cells is at most $n$, so as $d$ grows
the cells become singletons and the estimate at a cell is $1/(n \prod_j h_j)$ —
a number determined by the grid, not by the density. The demo's table is the
measurement: 1000 standard normal points, Freedman–Diaconis per axis, 500 fresh
points scored —

| $d$ | mean log density | points at the floor | occupied cells |
|--:|--:|--:|--:|
| 1 | −2.81 | 0.2% | 24 |
| 2 | −5.57 | 0.4% | 265 |
| 5 | −19.46 | 1.8% | 1000 |
| 10 | −41.78 | 4.0% | 1000 |

39 nats between one and ten dimensions at the same $n$. The occupied-cell column
is the mechanism: by $d = 5$ every point is alone in its cell, so the density is
one over a volume and the volume shrinks with $d$. A KDE degrades more gracefully
because its bandwidth is a length, not a grid; a histogram has no such knob.

### 4. Bayesian blocks as an exact partition

For a partition into blocks $b$ with counts $N_b$ and widths $W_b$, the
multinomial log-likelihood of the binned sample is

$$
\log L = \sum_b N_b \log\frac{N_b}{n\,W_b}
= \sum_b N_b \log\frac{N_b}{W_b} - n\log n
\qquad (9)$$

and the second term is constant across partitions, so maximising the first is
maximising the likelihood of a piecewise-constant density. Adding a penalty per
block gives (5); subtracting a constant penalty for each cut turns the problem
into a shortest path, and the dynamic program

$$
\mathrm{best}(j) = \max_{i<j}\left[\mathrm{best}(i) + N_{i:j}\log\frac{N_{i:j}}{W_{i:j}}
- \text{prior}\right]
\qquad (10)$$

solves it exactly in $O(K^2)$ for $K$ candidate cut points. This implementation
puts the candidates on a regular grid of `n_grid = 512` cells, which is what
makes a block of one point impossible: without a minimum width the likelihood is
unbounded and the optimum is a partition into singletons. The grid is a
resolution limit, not a smoothing parameter — the demo's boundary is at 0.997
rather than 1.000 because 0.997 is the nearest grid point.

The penalty is the model selection. It is not estimated; it is the expected
number of blocks, expressed in log-likelihood units. The demo sweeps it and gets
84, 21 and 2 blocks on the same data, which is the whole range from "follow every
fluctuation" to "two pieces, which happen to be the truth".

## Code walkthrough

### 1. The three rules

```python
    bins = _sturges_bins(n)
    if variant == "fd_rule":
        q75, q25 = np.percentile(x, [75.0, 25.0])
        width = 2.0 * (q75 - q25) * n ** (-1.0 / 3.0)
        if width > 0.0:
            bins = max(1, int(math.ceil(span / width)))
    elif variant == "scott":
        width = 3.49 * float(x.std(ddof=1)) * n ** (-1.0 / 3.0)
        if width > 0.0:
            bins = max(1, int(math.ceil(span / width)))
    return np.linspace(low, high, bins + 1)
```

The shared structure is the point: each rule produces a target width, and the
code converts it to a bin count over the data's range. `ceil` means the realised
width is never larger than the target, which matters for (2) and (3) — a rule
that asked for 0.215 and got 0.25 would be a different estimator — and it means
the last bin absorbs the rounding, so the bins stay equal. `bins = _sturges_bins(n)`
is the fallback for a zero IQR or zero standard deviation: a constant column has
no scale, and the alternative to a fallback is a division by zero. `ddof=1` is the
sample standard deviation, matching every other scale estimate in the chapter.

### 2. The dynamic program

```python
    for j in range(1, n_grid + 1):
        i = np.arange(j)
        mass = cumulative[j] - cumulative[i]
        length = (j - i) * width
        # A block with no points has zero likelihood contribution and is never
        # useful; -inf keeps it out of the argmax rather than letting 0 log 0
        # stand for a density of zero.
        fitness = np.where(mass > 0.0, mass * np.log(np.maximum(mass, 1.0) / length),
                           -np.inf)
        candidate = best[:j] + fitness - prior
        pick = int(np.argmax(candidate))
        best[j] = candidate[pick]
        cut[j] = pick
```

Equation (10), vectorised over the cut point $i$ for each right endpoint $j$.
`cumulative` is the prefix sum of the grid counts, so `mass` is the count in
$[i, j)$ for every $i$ at once, and `length` is the block's width in data units.
Subtracting `prior` per candidate is the per-block penalty, and because it is
subtracted once per cut it is equivalent to subtracting it from every block —
the constant for the whole partition does not affect the argmax. The
`mass > 0` guard is what keeps empty blocks out of the solution; without it,
`0 log 0` would be `nan` and the argmax would return the first candidate
silently. Backtracking through `cut` reconstructs the partition, and `grid` maps
the cell indices back to data coordinates.

### 3. Fitting one axis

```python
        for j in range(d):
            column = X[:, j]
            if self.bins is not None:
                low, high = float(column.min()), float(column.max())
                if high <= low:
                    edges = np.array([low - 0.5, high + 0.5])
                else:
                    edges = np.linspace(low, high, self.bins + 1)
            elif self.variant == "bayesian_blocks":
                edges = _bayesian_blocks(column, self.prior, self.n_grid)
            else:
                edges = _rule_edges(column, self.variant)
            counts, _ = np.histogram(column, edges)
            widths = np.diff(edges)
            self.edges_.append(edges)
            self.counts_.append(counts)
            self.densities_.append(counts / (n * widths))
```

A loop rather than a single vectorised call because the rules need percentiles
per column and the blocks need a dynamic program per column; a joint grid would
be the wrong model anyway, per §2. The `bins` branch comes first, so an explicit
bin count overrides the variant — that is how the demo puts two rules on the same
grid and how a caller can reproduce a plot. The constant-column branch invents a
unit-width bin around the value rather than returning a zero-width edge; the
density is then $1$ over that width, which is a statement about a point mass and
is the least surprising finite answer. Storing `counts_` as well as `densities_`
is deliberate: the counts are what the blocks objective is written in, and they
are what a reader checks when the density looks wrong.

### 4. Scoring, and the floor

```python
            index = np.searchsorted(edges, column, side="right") - 1
            index = np.clip(index, 0, len(density) - 1)
            outside = (column < edges[0]) | (column > edges[-1])
            value = np.where(outside, 0.0, density[index])
            total += np.log(np.maximum(value, FLOOR))
```

`searchsorted` is the whole lookup: the bins are sorted, so a point's index is a
binary search, and `side="right" - 1` puts a point exactly on an edge in the bin
to its right — the same convention `np.histogram` uses, which is what makes
scoring the training data reproduce the fitted counts. `np.clip` handles the two
endpoints, and the `outside` mask undoes the clip's optimism: a query beyond the
last edge is not in the last bin, it is outside the support, and a histogram has
no tail to put it in. The floor then turns that zero into $-690.8$, the same
number for every out-of-range query. `total` accumulates the log density across
axes rather than multiplying densities, so a point outside on one axis adds the
floor once and keeps the other axes' contributions — the log is the right space
for the product form.

## Complexity

For $n$ points, $d$ features, $m$ queries, $k_j$ bins on axis $j$ and a blocks
grid of $K$ cells:

| stage | time | memory |
|:--|:--|:--|
| rule edges, per axis | $O(n)$ plus a sort for percentiles | $O(1)$ |
| bin counts, per axis | $O(n + k_j)$ | $O(k_j)$ |
| bayesian blocks, one axis | $O(K^2 + n)$ | $O(K)$ |
| fit, all axes | $O(nd + n\log n + K^2)$ | $O(\sum_j k_j)$ |
| score one query | $O(d \log \max_j k_j)$ | $O(1)$ |
| score $m$ queries | $O(m d \log \max_j k_j)$ | $O(m)$ |

The binary searches are why scoring is fast: a fitted histogram is a lookup
table, and a million queries cost a million small binary searches rather than a
million distance computations. The blocks step is the only superlinear one, and
it is quadratic in the *grid*, not in the sample: $K = 512$ is about 260 000
candidate intervals, each one a vector operation. The per-axis memory is the bin
count, not the sample — the estimate is a summary, which is the other reason it
is used as a baseline.

## Failure modes

### The bin origin is arbitrary

Move the sample half a bin to the right and 12.8% of the probability mass lands in
a different bin. The estimate at a fixed $x$ is a step function of the bin edges,
so it is discontinuous in the data's location: `x` and `x + h/2` can be assigned
densities that differ by a factor of two. No rule in (2)-(4) fixes this, because
each one computes edges from the sample's own minimum and maximum. The fixes are
structural — average over several offsets, or use a kernel — and the practical
consequence is that a histogram's density numbers should not be compared across
batches whose ranges differ slightly.

### No tails, and the floor

Outside $[\min x, \max x]$ the estimate is exactly zero, and `score_samples`
returns the floor $-690.8$ for every such point however far away it is. On the
demo's held-out normal sample that affects 0.2% of points in one dimension and
4.0% in ten; in a tail-heavy application it can be most of the interesting
queries. A histogram has no model for what happens beyond its last edge, and a
"density" of zero is not a density — it is an admission. Smoothing the edges,
adding a parametric tail, or using a KDE are the alternatives.

### The curse, measured

39 nats between one and ten dimensions at $n = 1000$, and at $d = 5$ every point
is alone in its cell. This is not a failure of the rule; it is the product grid.
A joint histogram makes it worse, not better: the cell count is $k^d$ and the
sample is $n$, so the number of points per occupied cell falls like $n/k^d$. The
diagnostic is the occupied-cell count in the demo's table. When it approaches the
sample size, the estimate has stopped summarising and started enumerating.

### The rules encode a shape

Scott's constant is exact for a normal and wrong for anything else; on the
lognormal sample it asks for 19 bins where the robust rule asks for 29, and
Sturges asks for 10 regardless. A rule of thumb is a prior about the density's
shape, and a histogram is more sensitive to it than a KDE because the width is
the whole model. The symptom is a plot that is clearly too smooth (a wide
lognormal tail collapsed into the last bin) or too spiky (empty cells in the
middle of the data). Look at the plot, not just at the rule's number.

### The blocks prior is a model choice

The same data give 84, 21 or 2 blocks depending on the penalty, and all three are
the exact optimum for their prior. There is no cross-validation in (5) and no
asymptotic guarantee that the penalty is identifiable; the method is a way to
explore how many change points the data support, not a way to decide. The grid
adds a second resolution limit: no block can be narrower than `range / n_grid`,
so a genuine spike narrower than that is invisible.

## Applications

- **Fast baselines and plots.** One pass, no parameters beyond the rule, and an
  estimate that is exact to compute. Every anomaly-detection pipeline should have
  a histogram number to compare against before a fancier estimator is believed.
- **Discretisation for information theory.** Mutual information, entropy and
  transfer entropy estimators are usually histogram-based: bin each variable,
  count the joint table, and plug in. The bin rule matters more than the
  estimator, and (2)-(4) are the standard choices.
- **Change-point detection and astronomy.** Bayesian blocks was developed for
  photon-counting light curves, where the rate changes abruptly and the blocks
  are the events. The same dynamic program segments any piecewise-constant
  signal: counts per window, click-through rates, binned financial data.
- **Feature construction.** Histogram counts, bin indices and the distance to
  the nearest occupied bin are cheap features for downstream models, and a
  histogram of a residual is often the fastest way to see that a model is
  misspecified.
- **Streaming summaries.** A histogram is a fixed-size sketch: update counts as
  points arrive, and the density is a lookup. That is why it is the default in
  databases and monitoring systems, where the sample cannot be kept.
- **Where it is a trap.** Above a handful of dimensions, or whenever the tail
  matters, or when two batches have different ranges, per the failure modes. The
  KDE is the drop-in replacement at small $d$; the local scorers later in the
  chapter are the drop-in replacement when only ranking matters.

## References

- H. A. Sturges, *The choice of a class interval*, Journal of the American
  Statistical Association 21(153):65–66 (1926). Rule (4), derived for the
  binomial coefficients of a normal sample.
- D. W. Scott, *On optimal and data-based histograms*, Biometrika
  66(3):605–610 (1979). The normal-reference optimum (8) and the constant 3.49.
- D. Freedman and P. Diaconis, *On the histogram as a density estimator: $L_2$
  theory*, Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete
  57(4):453–476 (1981). Rule (2), the robust scale, and the $n^{-2/3}$ rate.
- M. P. Wand, *Data-based choice of histogram bin width*, The American
  Statistician 51(1):59–64 (1997). Why the rules disagree and how to choose
  between them.
- L. Birgé and Y. Rozenholc, *How many bins should be put in a regular
  histogram*, ESAIM: Probability and Statistics 10:24–45 (2006). A penalised
  likelihood selector with a non-asymptotic guarantee.
- J. D. Scargle, J. P. Norris, B. Jackson and J. Chiang, *Studies in
  astronomical time series analysis. VI. Bayesian blocks, a new method to analyze
  structure in photon counting data*, The Astrophysical Journal 764(2):167
  (2013). Equation (5), the dynamic program in (10) and the prior.
- K. H. Knuth, *Optimal data-based binning for histograms*, arXiv
  physics/0605197 (2006; revised 2019). The Bayesian reading of the bin count,
  and why the number of bins is a model.
- B. W. Silverman, *Density Estimation for Statistics and Data Analysis*,
  Chapman and Hall (1986), Chapter 2. The histogram as a kernel estimate with a
  box kernel, and the comparison with the smooth case.
