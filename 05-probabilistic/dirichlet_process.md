# Dirichlet processes

`dirichlet_process.py` · Probabilistic models · Tier 3 · 5 variants

## Intuition

Every mixture model this far needed $k$ up front. The DP removes that
requirement by putting a prior on the distribution itself, and letting the data
decide how many components to occupy:

$$G \sim \mathrm{DP}(\alpha, G_0), \qquad \theta_i \mid G \sim G, \qquad x_i \mid \theta_i \sim F(\theta_i) \qquad (1)$$

$G$ is a random probability measure. Draw it once, then draw the parameters from
it — and because a DP draw is discrete, the $\theta_i$ collide, and the
collisions are the clusters. The number of distinct values is not a parameter
and not fixed: it is a random variable whose distribution the concentration
$\alpha$ controls.

The same object has two faces, and the code uses both.

```
  stick-breaking                          Chinese restaurant
  ───────────────                         ──────────────────
  v_k ~ Beta(1, α)                        table 1: ● ● ●      n₁ = 3
  w_k = v_k ∏_{j<k}(1−v_j)                table 2: ●          n₂ = 1
  G = Σ_k w_k δ_{θ_k}                     table 3: new?
  ├──●───────●───●────●────────► w        p(join k) ∝ n_k
    0.4  0.24  .14  .09   ...             p(new)    ∝ α
```

Left: break a unit stick at $v_1$, keep the piece, break the rest at $v_2$, and
so on. The pieces are the mixture weights, and they are a size-biased
permutation — the first is the largest in expectation, which is what makes a
DP mixture look like a finite mixture with a decaying tail.

Right: seat customers at tables one at a time, each preferring a crowded table
in proportion to its current occupancy, or opening a new one with probability
proportional to $\alpha$. This is the view that turns into a sampler, because it
gives the conditional distribution of one label given all the others — which is
all a Gibbs sweep needs.

The consequence that matters for modelling is that $\alpha$ controls $k$
**logarithmically**:

$$\mathbb{E}[K_n] = \sum_{i=1}^{n}\frac{\alpha}{\alpha+i-1} \approx \alpha\log\!\left(1+\frac{n}{\alpha}\right) \qquad (2)$$

Four times the data does not give four times the clusters; it gives a constant
number more. That is the whole reason to use a DP instead of growing $k$ by
cross-validation: it is a prior on model size, and it is a weak one.

Three ideas in this chapter do the same job differently, and the differences are
worth stating once:

| | what is random | how $k$ is chosen | cost |
|:--|:--|:--|:--|
| finite mixture + BIC | $k$ by search | penalised likelihood | one fit per $k$ |
| **DP mixture** | the measure | posterior over $k$ | Gibbs, $O(TnK)$ |
| **DP-means** | nothing | threshold $\lambda$ | $O(nKd)$ per sweep |

The last one is the small-variance limit of the first: as the cluster variance
goes to zero, the DP mixture's MAP estimate becomes $k$-means with a penalty
$\lambda$ per cluster, and "open a new cluster" becomes "this point is more than
$\sqrt{\lambda}$ from every centroid". The demo's $\lambda = 9$ is a distance of
3, and on data whose cluster centres are 5.7 to 8 apart it recovers them exactly.

## Variants

| Variant | Object | Fitted by | Exact statement the demo checks |
|:--|:--|:--|:--|
| `crp` | the partition prior | seating, one draw at a time | EPPF sums to 1 over all 52 partitions of 5 |
| `stick_breaking` | the weights | GEM, one stick at a time | $\mathbb{E}[w_k]=\alpha^{k-1}/(1+\alpha)^k$ |
| `dpmm` | an infinite mixture | collapsed Gibbs | predictive = NIW Student-t integral |
| `truncated_dp` | a $K$-component mixture | collapsed Gibbs over $K$ sticks | truncation error = residual mass |
| `dp_means` | hard clustering | assignment + split | same partition as Lloyd at $k=3$ |

`crp` and `stick_breaking` are the prior, not a model, so `fit` refuses them and
says which method to use instead. They are variants because they are the two
ways a reader meets a DP, and because every assertion about them is exact:
partitions of five items are enumerable, and $\mathbb{E}[w_k]$ is a product of
beta means.

The other three are fitters, and they differ in how much of the DP they keep:

- `dpmm` keeps all of it. Collapsed Gibbs over a CRP prior, with the cluster
  parameters integrated out. The predictive is a Student-t, exactly.
- `truncated_dp` keeps $K$ sticks. Same sampler, finite prior, and a measurable
  error: the mass left on the sticks it did not use. On the demo's data it uses
  3 of the 8 it was offered, carries 0.9921 of the mass, and leaves 0.0079.
- `dp_means` keeps none of it. No posterior, no $\alpha$, one threshold.

The two Gibbs samplers share the predictive and differ in one line — the prior
on the assignment:

```
dpmm           p(z_i = k) ∝ n_k           p(z_i = new) ∝ α
truncated_dp   p(z_i = k) ∝ E[w_k | n]    no new component beyond K
```

That is the entire difference between an infinite mixture and its truncation,
and it is one substitution in the code.

## Derivation

### 1. What a Dirichlet process is

For every finite measurable partition $A_1,\dots,A_k$ of the space,

$$\left(G(A_1),\dots,G(A_k)\right) \sim \mathrm{Dirichlet}\!\left(\alpha G_0(A_1),\dots,\alpha G_0(A_k)\right) \qquad (3)$$

That is the definition, and it has two immediate consequences. First,
$G(A)\sim\mathrm{Beta}(\alpha G_0(A),\alpha(1-G_0(A)))$, so $G$ is a probability
measure with a prior mean $G_0$ and a prior "sample size" $\alpha$. Second, $G$
is almost surely discrete: draw two independent $\theta,\theta'$ from $G$ and
they are equal with positive probability, because a discrete measure puts mass
on atoms.

Ferguson (1973) defined it this way; Blackwell and MacQueen (1973) gave the
Pólya urn that makes it usable, and Sethuraman (1994) gave the construction that
makes it computable.

### 2. Stick-breaking

Let $v_1,v_2,\dots$ be independent $\mathrm{Beta}(1,\alpha)$ and $\theta_k$
independent $G_0$. Then

$$w_k = v_k\prod_{j<k}(1-v_j), \qquad G=\sum_{k=1}^{\infty}w_k\,\delta_{\theta_k} \qquad (4)$$

is a draw from $\mathrm{DP}(\alpha,G_0)$. The weights sum to one because the
residual after $K$ sticks is

$$R_K=\prod_{k=1}^{K}(1-v_k), \qquad \mathbb{E}[R_K]=\left(\frac{\alpha}{1+\alpha}\right)^{K} \qquad (5)$$

and $R_K\to0$ almost surely. Because the $v_k$ are independent,

$$\mathbb{E}[w_k]=\mathbb{E}[v]\left(\mathbb{E}[1-v]\right)^{k-1}=\frac{\alpha^{k-1}}{(1+\alpha)^{k}} \qquad (6)$$

which is a geometric decay: the mixture weights of a DP mixture are not
exchangeable across components, they are ordered by expected size. That is why
truncating at $K$ is not the same as a $K$-component mixture with free weights —
the truncation has to break the last stick somewhere.

### 3. The partition, and the restaurant

Integrating $G$ out turns the DP into a distribution over partitions. For a
partition into clusters of sizes $n_1,\dots,n_K$ summing to $n$, the exchangeable
partition probability function is

$$p(n_1,\dots,n_K)=\frac{\alpha^{K}\prod_{k=1}^{K}(n_k-1)!}{\alpha(\alpha+1)\cdots(\alpha+n-1)} \qquad (7)$$

which is what `eppf` computes in log form with $\Gamma$ functions. It depends
only on the sizes, not on the labels or the order — that is exchangeability, and
it is the reason a DP is a sensible prior for clustering: the data's order
carries no information.

Summing (7) over all set partitions gives one. That is not obvious from the
formula, which is why the demo checks it by brute force: 52 terms for $n=5$,
203 for $n=6$, both exactly 1.

The conditional of one label given the rest is the seating rule:

$$p(z_i=k\mid z_{-i})=\frac{n_k^{(-i)}}{\alpha+n-1}, \qquad p(z_i=\text{new}\mid z_{-i})=\frac{\alpha}{\alpha+n-1} \qquad (8)$$

and (2) follows by summing the probability of a new table over $i=1,\dots,n$.

### 4. The mixture, and why the predictive is a Student-t

Attach a likelihood and the DP becomes a mixture with infinitely many components
and a prior on their weights. For a Gaussian likelihood with a normal–inverse-$\chi^2$
prior on $(\mu_k,\sigma_k^2)$,

$$\sigma^2\sim\mathrm{InvGamma}(a_0,b_0), \qquad \mu\mid\sigma^2\sim N\!\left(\mu_0,\frac{\sigma^2}{\kappa_0}\right) \qquad (9)$$

the posterior given $m$ points is available in closed form:

$$\kappa_m=\kappa_0+m, \quad \mu_m=\frac{\kappa_0\mu_0+m\bar x}{\kappa_m}, \quad a_m=a_0+\frac{m}{2}, \quad b_m=b_0+\frac{1}{2}\sum_i(x_i-\bar x)^2+\frac{\kappa_0 m}{2\kappa_m}(\bar x-\mu_0)^2 \qquad (10)$$

and the posterior predictive — what a Gibbs sweep needs — is

$$p(x_*\mid\text{cluster})=t_{2a_m}\!\left(\mu_m,\ \frac{b_m(\kappa_m+1)}{a_m\kappa_m}\right) \qquad (11)$$

a Student-t with $2a_m$ degrees of freedom. Two things follow. The predictive has
tails heavier than a Gaussian's, which is correct: the variance is itself
unknown. And with $m=0$ the same formula is the prior predictive, which is why
the code can use one function for "join cluster $k$" and "open a new table".

The demo checks (11) against a numerical integral over the posterior of
$\sigma^2$: substituting $\tau=b_m/\sigma^2\sim\mathrm{Gamma}(a_m,1)$, the
integral is

$$p(x_*)=\int_0^\infty N\!\left(x_*;\mu_m,\frac{b_m}{\tau}\left(1+\frac{1}{\kappa_m}\right)\right)\frac{\tau^{a_m-1}e^{-\tau}}{\Gamma(a_m)}\,d\tau \qquad (12)$$

and 400k trapezoid points agree with the closed form to $10^{-13}$.

### 5. Collapsed Gibbs, in one line

Because the cluster parameters are integrated out, a sweep only touches the
labels. For point $i$, remove it from its cluster and sample

$$p(z_i=k)\propto\begin{cases}n_k^{(-i)}\,p(x_i\mid\text{cluster }k) & k \text{ occupied} \\ \alpha\,p(x_i\mid\text{prior}) & \text{new table}\end{cases} \qquad (13)$$

Everything is in log space, because $n_k$ can be zero-sized and because a
Student-t density underflows fast. The subtlety is the leave-one-out: cluster
$k$'s predictive must be computed *without* $x_i$, or $x_i$ contributes to its
own likelihood and every cluster looks better than it is. This is not a
cosmetic detail — with the point left in, the sampler in this file still
converged, and on the truncated variant it over-split.

### 6. Truncation, and the size of the error

Replace the infinite sum (4) with $K$ terms and normalise, or keep the last
stick's residual as a point mass. The cost is a bound: for any measurable set
$A$,

$$\left\lvert \sum_{k\le K}w_k\,\delta_{\theta_k}(A)-\sum_{k}w_k\,\delta_{\theta_k}(A)\right\rvert\le R_K \qquad (14)$$

so the $L_1$ distance between the truncated and full weight vectors is at most
$\mathbb{E}[R_K]$, and for a single draw it is exactly $R_K$ — the
total-variation distance between the two probability measures is half that. The
demo asserts the exact version: the $L_1$ distance between its 3-component
weight vector and the 203-component one is 0.0079, the residual, to $10^{-12}$.

For sampling, the truncated model has a closed-form conditional too. With counts
$n=(n_1,\dots,n_K)$,

$$v_j\mid n\sim\mathrm{Beta}\!\left(1+n_j,\ \alpha+\sum_{l>j}n_l\right) \qquad (15)$$

independently, so the mean weight of component $k$ is a product of beta means,
which is `posterior_weights`. No sampling of the sticks is needed for the
assignment step, and the leftover mass $R_K$ enters prediction as one extra
component with the prior predictive — which is exactly what a component beyond
the truncation would predict.

The practical rule is that $K$ should be large enough that $R_K$ is small on the
fitted solution, and no larger, because every extra stick redistributes mass.
The demo offers 8 sticks, uses 3, and reports the 0.0079 it left behind.

### 7. The small-variance limit

Put a penalty $\lambda$ per cluster into the DP mixture's MAP objective, scale it
as $\lambda/2\sigma^2$ so that it survives the limit, and let the cluster
variance go to zero:

$$\sum_i\log N\!\left(x_i;\mu_{z_i},\sigma^2\right)-\frac{\lambda}{2\sigma^2}K\;\xrightarrow[\sigma^2\to0]{}\;-\frac{1}{2\sigma^2}\left[\sum_i\left\lVert x_i-\mu_{z_i}\right\rVert^2+\lambda K\right] \qquad (16)$$

so the surviving objective is

$$\min_{\{z,\mu\}}\ \sum_i\left\lVert x_i-\mu_{z_i}\right\rVert^2+\lambda K \qquad (17)$$

$k$-means with a price per cluster. The algorithm is what the objective suggests:
assign each point to its nearest centroid, and if the squared distance exceeds
$\lambda$, pay $\lambda$ and open a new cluster there. Both operations decrease
(17), so the loop terminates, and at $\lambda$ between the within-cluster and
between-cluster squared distances it finds exactly the clusters $k$-means with
the right $k$ would find — without being told $k$.

$\lambda$ is not $\alpha$. It has units of squared distance, it has no posterior,
and it is the price at which the model prefers a new cluster to a bad fit. That
is a smaller, cruder model than the DP mixture, and the demo is explicit that on
this data it happens to agree with the Bayesian answer.

## Code walkthrough

The EPPF is (7), and the only care needed is to stay in logs: the factorials are
$\Gamma(n_k)$ and the normaliser is $\Gamma(\alpha+n)/\Gamma(\alpha)$.

```python
    def eppf(self, counts) -> float:
        """Exchangeable partition probability of cluster sizes `counts`."""
        counts = np.asarray(counts, dtype=float)
        n, k = counts.sum(), counts.size
        logp = (k * math.log(self.alpha)
                + float(sum(math.lgamma(c) for c in counts))
                + math.lgamma(self.alpha) - math.lgamma(self.alpha + n))
        return math.exp(logp)
```

The GEM weights are (4). `1 − U^(1/α)` is a $\mathrm{Beta}(1,\alpha)$ draw by
inverse CDF, which avoids needing a beta sampler, and the residual is returned
rather than dropped — it is the truncation error and the demo uses it.

```python
        # V ~ Beta(1, α) by inverse CDF: 1 − V ~ Beta(α, 1)
        v = 1.0 - rng.random(K) ** (1.0 / self.alpha)
        w = v * np.concatenate([[1.0], np.cumprod(1.0 - v)[:-1]])
        return w, float(np.prod(1.0 - v))
```

`posterior_weights` is (15) plus the product of means. `tail` is the suffix sum
$\sum_{l>k}n_l$, which is what makes the sticks dependent: the mass available to
stick $k$ is what the earlier sticks left.

```python
        counts = np.asarray(counts, dtype=float)
        tail = np.concatenate([np.cumsum(counts[::-1])[::-1][1:], [0.0]])
        ev = (1.0 + counts) / (1.0 + counts + self.alpha + tail)
        w = ev * np.concatenate([[1.0], np.cumprod(1.0 - ev)[:-1]])
        return w, float(np.prod(1.0 - ev))
```

The NIW posterior is (10). `n = 0` returning the prior is not a special case to
be handled elsewhere: it is the new-table prediction, and the same function
serves both.

```python
    X = np.asarray(X, dtype=float)
    n = X.size
    xbar = X.mean() if n else 0.0
    ss = float(((X - xbar) ** 2).sum()) if n else 0.0
    kappa = kappa0 + n
    mu = (kappa0 * mu0 + n * xbar) / kappa
    a = a0 + 0.5 * n
    b = b0 + 0.5 * ss + 0.5 * kappa0 * n / kappa * (xbar - mu0) ** 2
    return mu, kappa, a, b
```

The predictive is (11). The scale is the part people get wrong: it is
$b(\kappa+1)/(a\kappa)$, not $b/a$, because the predictive has to account for
the uncertainty in $\mu$ as well as in $\sigma^2$.

```python
def _predictive_logpdf(x: float, params) -> float:
    """`p(x | cluster)` after integrating `(μ, σ²)` out of the NIW."""
    mu, kappa, a, b = params
    scale = math.sqrt(b * (kappa + 1.0) / (a * kappa))
    return float(_log_t(x, 2.0 * a, mu, scale))
```

The collapsed Gibbs sweep is (13). The `z[i] = -1` line is the removal, and the
`logp` list is built in log space over the occupied tables plus one new table.

```python
                old = z[i]
                z[i] = -1                                   # leave the table
                labels = z[z >= 0]
                params = self._cluster_params(X[z >= 0], labels)
                logp = []
                for k in np.unique(labels):
                    c = int((labels == k).sum())
                    logp.append(math.log(c) + self._log_predictive(X[i], params[k]))
                # a new table: the prior predictive, weighted by alpha
                logp.append(math.log(self.alpha) + self._log_predictive(X[i], prior))
```

The truncated sweep differs in exactly the two lines that implement (15): the
weights replace the counts, and the leave-one-out mask keeps $x_i$ out of its
own cluster's posterior.

```python
                counts = np.bincount(z, minlength=self.K).astype(float)
                counts[z[i]] -= 1.0
                w, _ = self.posterior_weights(counts)
                # leave point i out of its own cluster's posterior, or the
                # density at x_i is inflated by its own contribution
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                params = self._cluster_params(X[mask], z[mask], self.K)
```

DP-means is (17) as an algorithm. The split is a `>` against $\lambda$ on the
*squared* distance, and the new centroid is the offending point itself — which
makes the objective fall by exactly $d^2-\lambda$.

```python
            for i in rng.permutation(n):
                d2 = ((centroids - X[i]) ** 2).sum(axis=1)
                j = int(d2.argmin())
                if d2[j] > self.lam:            # cheaper to pay λ than to fit
                    centroids = np.vstack([centroids, X[i]])
                    j = centroids.shape[0] - 1
                labels[i] = j
```

Prediction mixes the components and adds the leftover mass as one more term. For
`dpmm` that term is $\alpha/(\alpha+n)$ with the prior predictive; for
`truncated_dp` it is the residual. One code path, two meanings, and both are
correct.

```python
            terms = [math.log(w[k]) + self._log_predictive(x, self.params_[k])
                     for k in range(self.K_)]
            terms.append(math.log(new) + self._log_predictive(x, prior))
            top = max(terms)
            out[i] = top + math.log(sum(math.exp(t - top) for t in terms))
```

The demo's exact check of (11) is (12), and the Jacobian is the line that is
easy to get wrong: with $\tau=b/\sigma^2$ the weight is
$\tau^{a-1}e^{-\tau}/\Gamma(a)$, with no $b^a$ in it.

```python
    grid = np.linspace(1e-8, 80.0, 400_000)
    var = (1.0 + 1.0 / kappa) * b / grid
    xq = float(x1.mean()) + 0.5
    dens = np.exp(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (xq - mu) ** 2 / var)
    dens *= np.exp((a - 1.0) * np.log(grid) - grid - math.lgamma(a))
```

## Complexity

$n$ points, $d$ dimensions, $K$ occupied clusters, $T$ sweeps:

| Step | Time | Memory |
|:--|:--|:--|
| EPPF of one partition | $O(K)$ | $O(1)$ |
| one CRP draw of $n$ labels | $O(nK)$ | $O(n)$ |
| one GEM draw of $K$ sticks | $O(K)$ | $O(K)$ |
| one Gibbs sweep (naive, as written) | $O(n^2Kd)$ | $O(nd+nK)$ |
| one Gibbs sweep (cached statistics) | $O(nKd)$ | $O(nd+nK)$ |
| one DP-means sweep | $O(nKd)$ | $O(nK)$ |

The code is the naive version: `_cluster_params` recomputes every cluster's
sufficient statistics before every point's update, which is $O(nKd)$ per point
and $O(n^2Kd)$ per sweep. That is fine at $n=120$ and wrong at $n=10^5$; the fix is to
keep $(\text{count},\sum x,\sum x^2)$ per cluster and update them when a point
moves, which is what every production sampler does and why the second row exists.
With cached statistics the cost is the same as DP-means.

The demo runs all five variants in about 2.2 s, of which most is the 20,000
partition draws that exist to make the $\mathbb{E}[K_n]$ check tight; the two
Gibbs fits take 0.3 s and 0.5 s, and the 400k-point quadrature is under a
millisecond.

## Failure modes

**Over-splitting, and the prior on the variance.** The most common surprise: a
DP mixture puts four clusters on three. This is not a bug and not a sampler
problem — with a vague prior on $\sigma^2$ and a weak prior on $\mu$, splitting
one Gaussian cluster into two buys enough marginal likelihood to pay for the
extra component. It happened here during development: with $\kappa_0=0.05$,
$a_0=2$, $b_0=0.3$ the count wandered between three and five across five seeds
of both samplers, on clusters whose centres are six units apart with noise 0.4.
Tightening to $\kappa_0=0.5$, $a_0=5$, $b_0=1$ made all five seeds agree on
three, which is the configuration in the file. *Diagnostic:* if the count
is unstable across seeds, the count is not identified; report the co-clustering
matrix, not $K$.

**Label switching.** Cluster labels are exchangeable, so two chains can agree
perfectly on the partition and disagree on the labels. Any average of
per-cluster parameters across chains is meaningless. *Diagnostic:* summarise
with $p(z_i=z_j)$, an $n\times n$ matrix that is invariant to relabelling.

**$\alpha$ is not a knob for $k$.** Equation (2) is logarithmic: to double the
expected number of clusters you square the data. Setting $\alpha$ to hit a
target $K$ is possible and standard (Teh et al.'s VIP does it properly), but
treating $\alpha$ as "the number of clusters" produces a model that is not the
one intended. *Diagnostic:* print `expected_clusters(n)` before fitting, and be
surprised if the posterior is far from it.

**Truncation that truncates too much.** $K$ too small forces real clusters to
share a stick, and the fitted residual $R_K$ cannot reveal it because the
sampler will never leave mass unused. *Diagnostic:* fit at $K$ and $2K$ and
compare the predictive log-likelihood; if it moves, $K$ was the constraint.

**The residual as a component.** In `truncated_dp` the leftover mass predicts
like the prior, which is right and also why the predictive has heavier tails
than the fitted mixture suggests. A single extrapolated point can therefore be
predicted better by the residual than by the nearest cluster. That is a feature
of the model, not an artefact.

**The leave-one-out omission.** Including $x_i$ in its own cluster's statistics
inflates that cluster's density at $x_i$, which biases the sampler toward
keeping points where they are and, in the truncated variant, toward splitting
clusters that should be merged. *Diagnostic:* the sampler converges to a stable
answer with the wrong $K$; check the mask.

**Gibbs is a local method.** Two seeds, two modes, and no guarantee that either
is the global one. The posterior over partitions is enormous, and $T$ sweeps of
a Gibbs chain explore a tiny neighbourhood. *Diagnostic:* run several chains
from CRP-prior initialisations, and compare the partitions rather than the
parameters.

**DP-means is not Bayesian.** $\lambda$ has units, no posterior, and no
uncertainty. Two data sets scaled by 10 need different $\lambda$ for the same
answer. And the algorithm's answer depends on the order of the points within a
sweep, through which clusters exist when a point is visited. *Diagnostic:*
rescale the data to unit variance and re-run; if the number of clusters changes,
$\lambda$ was doing the work rather than the data.

**Diagonal covariance.** The code uses one NIW per coordinate, so each cluster
is axis-aligned. Correlated clusters are split, because two axis-aligned
components fit a tilted one better than one. The full-covariance version needs a
normal–inverse-Wishart prior and a matrix inverse per cluster; the sampler's
structure is identical.

## Applications

**Density estimation and clustering without $k$.** The core use. Whenever the
number of groups is a finding rather than an input — customer segments,
astronomical object classes, cell populations in flow cytometry — the DP is the
default Bayesian answer, and its posterior over $K$ is the deliverable.

**Topic models.** Latent Dirichlet allocation fixes the number of topics; its
DP version lets the corpus decide, and the hierarchical DP (Teh et al., 2006)
does it per document while sharing topics across the corpus. It is the reason
"how many topics" is not a hyperparameter in the modern literature.

**Genetics and species sampling.** The EPPF (7) is the Ewens sampling formula,
which is the null model for allele frequencies and for discovery counts in
population genetics. The same algebra appears as the prior in Dirichlet process
mixtures for genotype clustering.

**Entity resolution and record linkage.** Each entity is a cluster of records;
the number of entities is unknown and grows with the data. The DP's
exchangeability assumption is exactly right for records arriving in arbitrary
order, and the predictive (11) gives a calibrated "does this record belong to an
existing entity" score.

**Spatial and image segmentation.** With a spatial base measure, DP mixtures
segment images and cluster geo-referenced counts without a chosen number of
regions. The truncated version is what fits on a GPU.

**Scalable clustering.** DP-means is the practical answer when $n$ is large and
a Gibbs chain is not: $O(nKd)$ per sweep, no posterior, and the number of
clusters emerges. It is used as a fast alternative to $k$-means when $k$ is
unknown and to hierarchical clustering when $n$ is too large for a dendrogram.

**Where it is a trap.** Data with heavy-tailed clusters (the Gaussian likelihood
fights back), problems where the number of clusters is genuinely fixed by the
design, and any application that reports $K$ without its posterior. Also:
treating DP-means' $\lambda$ as if it had a Bayesian interpretation, which it
does not.

## References

- Ferguson (1973). *A Bayesian analysis of some nonparametric problems.* Annals
  of Statistics 1, 209. The definition (3), and the first conjugacy results.
- Blackwell & MacQueen (1973). *Ferguson distributions via Pólya urn schemes.*
  Annals of Statistics 1, 353. The seating rule (8) and its exchangeability.
- Sethuraman (1994). *A constructive definition of Dirichlet priors.* Statistica
  Sinica 4, 639. Stick-breaking (4), and the proof that it gives a DP.
- Antoniak (1974). *Mixtures of Dirichlet processes with applications to Bayesian
  nonparametric problems.* Annals of Statistics 2, 1152. The EPPF (7) and the
  first DP mixture.
- Escobar & West (1995). *Bayesian density estimation and inference using
  mixtures.* JASA 90, 577. The collapsed Gibbs sampler and the NIW predictive.
- Neal (2000). *Markov chain sampling methods for Dirichlet process mixture
  models.* JCGS 9, 249. The definitive treatment of the sampler, including the
  cases where the base measure is not conjugate.
- Ishwaran & James (2001). *Gibbs sampling methods for stick-breaking priors.*
  JASA 96, 161. The truncated sampler (15), and how to choose $K$.
- Blei & Jordan (2006). *Variational inference for Dirichlet process mixtures.*
  Bayesian Analysis 1, 121. The mean-field alternative, and the truncated
  variational bound this file's `truncated_dp` is the sampling counterpart of.
- Teh, Jordan, Beal & Blei (2006). *Hierarchical Dirichlet processes.* JASA 101,
  1566. The HDP, and the reason DP topic models share topics across documents.
- Kulis & Jordan (2012). *Revisiting k-means: new algorithms via Bayesian
  nonparametrics.* ICML. DP-means, the limit (16)–(17), and the guarantee that
  both steps decrease the objective.
- Gershman & Blei (2012). *A tutorial on Bayesian nonparametric models.* Journal
  of Mathematical Psychology 56, 1. The clearest short derivation of everything
  in the first three sections.
- Miller & Harrison (2018). *Mixture models with a prior on the number of
  components.* JASA 113, 340. Why the posterior on $K$ concentrates on the wrong
  answer when the model is misspecified, and the over-splitting in the failure
  modes above.
