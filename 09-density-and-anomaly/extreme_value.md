# Extreme value theory — fit the tail instead of counting it

`extreme_value.py` · Density & anomaly · Tier 3 · 5 variants

## Intuition

A rank-based detector answers one question and refuses the next. How unusual is
a value of 11 when the largest of two thousand observations is 7.8? The
empirical CDF says *never*, because it has never seen 11. That is not an answer;
it is the absence of one. Extreme value theory replaces the empty count with a
fitted probability, and a fitted probability can be turned into a score, a
threshold and a return period.

The theory comes in two shapes, and they are the same theory seen from two
sides:

- **Block maxima.** Cut the series into blocks, keep the maximum of each, and
  model those maxima. The Fisher–Tippett–Gnedenko theorem says their
  distribution converges to the generalised extreme value (GEV) family, which
  contains Gumbel, Fréchet and Weibull as the three signs of one shape
  parameter.
- **Peaks over threshold.** Keep everything above a high threshold and model the
  excesses. The Pickands–Balkema–de Haan theorem says those excesses converge to
  the generalised Pareto distribution (GPD), with the *same* shape parameter.

Two routes, five estimators: the GPD by probability-weighted moments (`pot`) or
by maximum likelihood (`gpd`); the GEV by L-moments (`block_maxima`) or by
maximum likelihood (`gev`); and a one-parameter straight-line fit on the
return-level plot (`return_level`).

```
  x_T
   │                        ╱
   │                    ╱
   │                ╱          each point is a value
   │            ╱              exceeded once every T
   │        ╱                  observations, on average
   │    ╱
   └──┬─────┬──────┬───────┬──────► log T
     10   100   1000   10000
```

That line is the product. On the demo's exponential sample it is fitted to the
top 10% of 5000 observations and used to predict a level exceeded once every
thousand observations, and it lands within 6% of the truth.

The other half of the story is what the tail does when it ends. A distribution
with a finite upper endpoint has a negative shape parameter, and the fit places
that endpoint. On a beta$(2,2)$ sample the true endpoint is one and the two GPD
fits land at $0.9964$ and $0.9910$; a query at $1.2$ then gets probability zero
and infinite score. The model can say "impossible", which no rank-based method
can.

The demos also make the choices visible. On two thousand exponential draws
with twenty injected points at 11, the fitted tail flags all twenty and one
false positive, while the empirical count above 11 is zero. That is the trade:
a rank method is assumption-free and cannot extrapolate; this one extrapolates
and therefore has to be right about the shape.

## Variants

| variant | model | fit | shape |
|:--|:--|:--|:--|
| `pot` | GPD on exceedances | probability-weighted moments | estimated |
| `gpd` | GPD on exceedances | maximum likelihood | estimated |
| `block_maxima` | GEV on block maxima | L-moments | estimated |
| `gev` | GEV on block maxima | maximum likelihood | estimated |
| `return_level` | exponential tail | least squares on the return-level plot | fixed at zero |

**`pot`** is the default and the workhorse. Probability-weighted moments are
closed-form, need no optimiser and are stable down to a few dozen exceedances.
It is the estimator to reach for when the threshold has already been chosen.

**`gpd`** fits the same model by likelihood. It is asymptotically efficient and
more sensitive to the largest exceedances, which is either the point or the
problem depending on how much you trust them. It is started from the
probability-weighted moments, which is what makes it reliable on the samples
where it works at all.

**`block_maxima`** is the classical route: L-moments of the block maxima, which
are themselves closed-form and far more stable than likelihood in small
samples. It throws away everything in a block except the largest value, which
is a real cost when the data are expensive.

**`gev`** is the likelihood version of the same model, again started from the
L-moments. It uses the maxima more efficiently than the moment fit but inherits
the GEV likelihood's small-sample instability.

**`return_level`** drops the shape parameter entirely and fits
$-\log p = (x-u)/\beta$ as a straight line through the origin. One parameter,
no optimiser, no finite endpoint, and no ability to bend: it is the model behind
a return-level plot that looks straight, and the most stable of the five.

## Derivation

### Peaks over threshold

Let $X$ have distribution $F$ and fix a threshold $u$. The excess
$Y = X-u$ conditional on $X>u$ has, in the limit of large $u$, a generalised
Pareto distribution with scale $\beta>0$ and shape $\xi$:

$$
S(y) = \Pr(Y > y \mid Y > 0) = \Big(1 + \frac{\xi y}{\beta}\Big)^{-1/\xi},
\qquad (1)
$$

with the exponential $e^{-y/\beta}$ at $\xi=0$ and the convention that
$S(y)=0$ once $1+\xi y/\beta \le 0$. The support is $y\ge 0$ for $\xi\ge 0$ and
$0 \le y \le -\beta/\xi$ for $\xi<0$, so a negative shape is a finite endpoint
at $u-\beta/\xi$.

The unconditional tail follows from the chain rule for conditional probability:

$$
p(x) = \Pr(X>x) = \zeta_u\, S(x-u),
\qquad \zeta_u = \frac{n_u + \tfrac{1}{2}}{n+1},
\qquad (2)
$$

for $x>u$, where $n_u$ is the number of exceedances. The half-count is the same
smoothing the empirical CDF uses in the `copod` atom, and it is chosen so that
(2) at $x=u$ equals the empirical survival $\Pr(X>u)$ exactly. Below the
threshold the implementation switches to the empirical survival, because the
Pareto model says nothing about the body and its extrapolation downwards is
nonsense.

The code is a single guarded evaluation of (1), with the exponential branch
kept separate so that no $0/0$ ever appears:

```python
    y = 1.0 + xi * z / beta
    return np.where(y > 0.0, np.power(np.maximum(y, 1e-300), -1.0 / xi), 0.0)
```

The `np.maximum` is not cosmetic. Without it, `np.power` of a negative base with
a fractional exponent returns NaN, and one NaN poisons the whole tail
probability — including the training scores, which is how the failure would
first be noticed.

### Return levels

The return level $x_T$ is the value exceeded once every $T$ observations on
average, which is the same as solving $p(x_T)=1/T$. Inverting (2) and (1):

$$
x_T = u + \frac{\beta}{\xi}\Big((T\zeta_u)^{\xi} - 1\Big),
\qquad (3)
$$

and $x_T = u + \beta\log(T\zeta_u)$ at $\xi=0$. The factor $T\zeta_u$ is why
the threshold's own return period, $1/\zeta_u$, is a floor: asking for a period
shorter than that is asking for a quantile below the threshold, where the model
is not defined, and the implementation refuses.

```python
        if abs(self.xi_) < 1e-8:
            return self.u_ + self.beta_ * math.log(target)
        return self.u_ + self.beta_ / self.xi_ * (target ** self.xi_ - 1.0)
```

For $\xi<0$ the bracket $(T\zeta_u)^\xi$ tends to zero as $T$ grows, so the
return level tends to the endpoint $u-\beta/\xi$ and never crosses it. That
limit is the finite-endpoint behaviour the beta demo checks.

### Block maxima

For a block of $b$ observations with maximum $M$, independence gives
$\Pr(M\le x)=F(x)^b$. The GEV is the limit law of $M$:

$$
G(x) = \exp\!\Big(-\big(1+\xi\tfrac{x-\mu}{\sigma}\big)^{-1/\xi}\Big),
\qquad (4)
$$

again with the Gumbel limit $\exp(-e^{-(x-\mu)/\sigma})$ at $\xi=0$. Inverting
$G(x)=F(x)^b$ turns a statement about maxima into a statement about a single
observation:

$$
p(x) = 1 - G(x)^{1/b}.
\qquad (5)
$$

That inversion is exact under the model, not a large-$x$ approximation, which
is why the block variants can be scored on the same surprisal scale as the
threshold variants. The implementation computes it with `log` and `expm1` to
avoid cancelling $1-G^{1/b}$ when $G$ is close to one:

```python
        cdf = self._gev_cdf(x)
        with np.errstate(divide="ignore"):
            probability = -np.expm1(np.log(cdf) / self.block_size_)
```

A point beyond a finite endpoint has $G=1$, so $\log G=0$ and the probability
is zero: the point is impossible, and the score is infinite. A point below the
lower support has $G=0$, so the probability is one: it is not anomalous, it is
merely small.

### Block return levels

Setting (5) equal to $1/T$ and solving gives the block version of (3):

$$
x_T = \mu + \frac{\sigma}{\xi}\Big(\big(-b\log(1-1/T)\big)^{-\xi} - 1\Big),
\qquad (6)
$$

with the Gumbel limit $\mu-\sigma\log(-b\log(1-1/T))$ at $\xi=0$. The quantity
$-b\log(1-1/T)$ is $-\log G(x_T)$, the GEV's own reduced variable at the return
level, and for large $T$ it is about $b/T$. It is the bridge between the two
routes: (4) turns it back into a cdf, and (5) turns the cdf into a probability.

### Estimating the parameters

**GPD by probability-weighted moments.** With $Y$ the exceedances and
$a_s=\mathbb{E}[Y(1-F(Y))^s]$, the GPD gives
$a_s=\beta/((s+1)(s+1-\xi))$. For $s=0,1$ that is two equations in two
unknowns, and solving them is closed-form:

$$
\hat\xi = \frac{a_0-4a_1}{a_0-2a_1},
\qquad
\hat\beta = \frac{2a_0a_1}{a_0-2a_1}.
\qquad (7)
$$

The sample versions replace $a_s$ by its order-statistic average. It is
consistent, needs no iteration, and the denominator is the only thing that can
go wrong — which is why the implementation checks it:

```python
    a0 = float(ordered.mean())
    a1 = float(np.sum((n - np.arange(1, n + 1)) / (n - 1.0) * ordered) / n)
    denominator = a0 - 2.0 * a1
```

**GEV by L-moments.** The same idea applied to maxima: form $b_0,b_1,b_2$ from
the sorted maxima with the weights $(i-1)/(n-1)$ and
$(i-1)(i-2)/((n-1)(n-2))$, turn them into the L-moments
$\ell_2=2b_1-b_0$ and $\ell_3=6b_2-6b_1+b_0$, and read the shape off the
L-skewness $\tau_3=\ell_3/\ell_2$ through Hosking's approximation

$$
\xi = -\big(7.8590\,c + 2.9554\,c^2\big),
\qquad
c = \frac{2}{3+\tau_3} - \frac{\log 2}{\log 3}.
\qquad (8)
$$

The scale and location then follow from
$\sigma=\ell_2\xi/((2^\xi-1)\Gamma(1-\xi))$ and
$\mu=b_0-\sigma(\Gamma(1-\xi)-1)/\xi$, with the Gumbel limits
$\ell_2/\log 2$ and $b_0-\sigma\gamma$ at $\xi=0$. The signs here are a genuine
trap: several references write the shape with the opposite convention, and a
wrong sign reproduces every Gumbel sample correctly while turning a Fréchet
sample inside out. This implementation's first version did exactly that, and
the max-stable demo — where the answer is known to be $0.2$ — is what caught
it:

```python
    c = 2.0 / (3.0 + ell3 / ell2) - math.log(2.0) / math.log(3.0)
    xi = -(7.8590 * c + 2.9554 * c * c)
```

**Both by likelihood.** The GPD and GEV log-likelihoods are two or three
parameters, and the only awkward part is the support: the likelihood is
undefined, not merely small, outside it. The implementation parameterises the
scale by its logarithm, returns a large constant outside the support, and runs
a deterministic Nelder–Mead simplex started from the moment fit:

```python
    z = (x - mu) / sigma
    y = 1.0 + xi * z
    if not np.all(y > 0.0):
        return 1e12
```

A few dozen lines of reflection, expansion and contraction buy two consistent
estimators and no dependency. The returned point is the best vertex of the
simplex, so the result is deterministic and reproducible.

## Code walkthrough

### Scoring a point

Everything reduces to a tail probability, and every variant shares the same
public surface: `tail_probability`, `score_samples`, `return_period`,
`return_level`, `predict`. The POT branch shows the hybrid that (2) describes:

```python
        probability = np.empty(x.size)
        below = x <= self.u_
        counts = np.searchsorted(self.sorted_, x[below], side="right")
        probability[below] = (self.n_ - counts + 0.5) / (self.n_ + 1.0)
        probability[~below] = self.scale_ * _gpd_survival(
            x[~below] - self.u_, self.beta_, self.xi_)
```

`searchsorted` on the stored sorted sample is what keeps scoring independent of
the batch: a point's probability depends on the training data and on the point,
not on the other points scored with it. The split at `u_` also makes the model
honest about its own domain — the parametric part is only ever evaluated in the
region where the theorem applies.

### The GEV cdf, with the support handled

```python
        y = 1.0 + self.xi_ * z
        cdf = np.zeros_like(x)
        inside = y > 0.0
        cdf[inside] = np.exp(-np.power(y[inside], -1.0 / self.xi_))
        if self.xi_ < 0.0:
            cdf[~inside] = 1.0
```

Masking rather than `np.where` is deliberate: the power is only evaluated on
the inside of the support, so no warning is raised and no NaN is created. The
`xi_ < 0` branch sets the cdf to one above the finite endpoint; for `xi_ > 0`
the zeros below the lower support are already correct.

### The decision threshold

```python
        self.scores_ = self.score_samples(x)
        self.threshold_ = float(np.quantile(self.scores_,
                                            1.0 - self.contamination))
```

The default decision rule is the same contamination quantile as the rest of the
chapter, but it is the weaker interface: it fixes the false-alarm rate to
whatever the training set contains and gives up the return period. The model's
own rule is `return_period(x) > T`, which is stable under changes to the sample
size and has an interpretation a domain expert can argue with. Both are
available; only one of them is a budget.

## Complexity

Let $n$ be the number of observations, $n_u$ the exceedances, $n_b$ the blocks,
$m$ the number of points scored and $I$ the number of simplex iterations.

| operation | time | memory |
|:--|:--|:--|
| fit, moments (`pot`, `block_maxima`, `return_level`) | $O(n\log n)$ | $O(n)$ |
| fit, likelihood (`gpd`, `gev`) | $O(I\,n_u)$ after the sort | $O(n_u)$ |
| score, threshold variants | $O(m\log n)$ | $O(m)$ |
| score, block variants | $O(m)$ | $O(m)$ |
| return level | $O(1)$ | $O(1)$ |

The sort dominates every fit, and the exceedances are a small fraction of the
sample, so the likelihood fits are cheap in practice: the exponential demos fit
five hundred exceedances in a few milliseconds. Scoring is a binary search for
the threshold variants because of the empirical body, and a closed form for the
block variants, which never look at the sample again. For the threshold
variants the state is one sorted copy of the data; the block variants keep
three numbers.

## Failure modes

**A high threshold buys honesty at the price of variance.** The sweep in the
demo keeps 600, 300, 150 and 30 exceedances from the same exponential sample,
and the standard deviation of the shape estimate grows from $0.043$ to $0.206$:

| threshold quantile | exceedances | mean $\hat\xi$ | std $\hat\xi$ | mean $\hat p(6)$ | std $\log_{10}\hat p(6)$ |
|--:|--:|--:|--:|--:|--:|
| 0.80 | 600 | $-0.0072$ | $0.0429$ | $0.002314$ | $0.124$ |
| 0.90 | 300 | $0.0042$ | $0.0714$ | $0.002467$ | $0.155$ |
| 0.95 | 150 | $-0.0072$ | $0.1050$ | $0.002347$ | $0.162$ |
| 0.99 | 30 | $-0.0400$ | $0.2060$ | $0.002264$ | $0.177$ |

The probability estimate stays near the truth because the Pareto model is exact
here, but the spread does not: the highest threshold's shape estimate moves
almost five times as much as the lowest one's. A threshold is a bias-variance
choice, not a formality.

**A low threshold buys data at the price of bias.** The GPD is a limit as
$u\to\infty$. Below the point where the limit has bitten, the fitted shape is
describing the body of the distribution, and the return levels it produces are
confidently wrong. The diagnostics are a mean-residual-life plot (should be
linear in $u$), a shape-stability plot (should be flat in $u$), and the
sweep table above.

**Block size is a bias, too.** Gaussian maxima are only asymptotically Gumbel,
and at a block of 25 the fitted shape is $-0.12$; at a block of 250 it is
$-0.08$. Both are negative because the Gaussian's tail converges slowly, not
because the data have a finite endpoint. A Fréchet sample, which is max-stable,
recovers its $0.2$ even at the small block:

| data | block | `block_maxima` $\hat\xi$ | `gev` $\hat\xi$ |
|:--|--:|--:|--:|
| Gaussian | 25 | $-0.120$ | $-0.118$ |
| Gaussian | 250 | $-0.083$ | $-0.090$ |
| Fréchet$(5)$ | 25 | $0.162$ | $0.171$ |

**Extrapolation error grows with distance.** The fitted return levels are within
$1.5\%$ at $T=100$ but $10\%$ low at $T=10000$. That is the honest error bar on
the plot above: the further the extrapolation, the more the answer depends on
the shape, and the shape is the least well estimated parameter. Report an
interval, not a point.

**The endpoint can be missed, and the shape can be degenerate.** A bounded
distribution whose density does not vanish at the endpoint has $\xi=-1$, where
the GPD is a point mass and the likelihood has no interior maximum. Uniform
data are the canonical case: both fits in this file return a shape below $-1$
and are refused with `the shape estimate is outside (-1, 1)`. Beta$(2,2)$,
whose density vanishes linearly, has $\xi=-0.5$ and fits cleanly. "The tail has
an endpoint" is not enough information; the behaviour of the density at the
endpoint decides whether the model can represent it.

**Non-stationarity breaks everything.** Every equation above assumes the tail
is the same at the start and the end of the sample. A trend, a season or a
regime change makes the fitted shape an average of incompatible tails, and the
return level a statement about a distribution that never existed. Detrend or
condition first; the `change_point` atom is the diagnostic.

**It is univariate, by construction.** The theory describes the tail of one
random variable. Correlated features must be reduced first — a norm, a
projection, a residual — and the reduction is a modelling choice that the tail
fit cannot make for you. For a multivariate score, use `copod` or
`mahalanobis` and keep this atom for the univariate statistic they produce.

**The contamination interface hides the extrapolation.** With
`contamination=0.01` the threshold is the 99th percentile of the training
scores, so one percent of the training set is flagged by construction, exactly
as in every other atom in the chapter. The whole point of the tail model is the
part beyond the data; the quantile rule throws it away. Use
`return_period` when the question is "how rare is this", and the quantile when
the question is "flag the worst one percent".

## Applications

**Hydrology and civil engineering.** The hundred-year flood is a return level
with $T$ in years, fitted to annual maxima or to peaks over a threshold. The
confidence interval matters more than the point estimate, and the block-size
and threshold diagnostics above are standard practice.

**Finance and insurance.** Value-at-risk and expected shortfall are tail
quantiles and tail means; the GPD is the standard model for operational-loss
severity above a high threshold, and the shape parameter is the number that
decides how much capital the far tail needs.

**Reliability and materials.** Component failure times, fatigue lives and
extreme loads are block maxima or exceedances; the Weibull case ($\xi<0$)
gives the finite life that engineers design against.

**Network and system monitoring.** The SPOT and DSPOT detectors apply exactly
this file's pipeline — fit a GPD to the largest observations of a stream and
flag the points whose fitted probability crosses a target — to network traffic
and sensor streams, with a drifting threshold for non-stationary data.

**Environmental compliance.** Air-quality and temperature extremes are regulated
as return levels ("not to be exceeded more than once in ten years"), and the
shape parameter is what distinguishes a bounded pollutant from an unbounded one.

**A second opinion for any anomaly detector.** Fit the tail to the scores of
another model and check whether the flagged points are rare under the fitted
tail. If a point is flagged by a rank rule but has a return period of two
observations under the tail model, the flag is a tie, not an extreme.

## References

- Fisher, R. A. & Tippett, L. H. C. (1928). Limiting forms of the frequency
  distribution of the largest or smallest member of a sample. *Proceedings of
  the Cambridge Philosophical Society* 24(2), 180–190. The three limit laws.
- Gnedenko, B. (1943). Sur la distribution limite du terme maximum d'une série
  aléatoire. *Annals of Mathematics* 44(3), 423–453. The convergence theorem.
- Balkema, A. A. & de Haan, L. (1974). Residual life time at great age. *Annals
  of Probability* 2(5), 792–804. Exceedances converge to the GPD.
- Pickands, J. (1975). Statistical inference using extreme order statistics.
  *Annals of Statistics* 3(1), 119–131. The same result, independently.
- Hosking, J. R. M., Wallis, J. R. & Wood, E. F. (1985). Estimation of the
  generalized extreme-value distribution by the method of probability-weighted
  moments. *Technometrics* 27(3), 251–261. Equation (8) and its constants.
- Hosking, J. R. M. & Wallis, J. R. (1987). Parameter and quantile estimation
  for the generalized Pareto distribution. *Technometrics* 29(3), 339–349.
  Equation (7) and the GPD moments.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*.
  Springer. Threshold choice, return levels and the diagnostics.
- Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal
  Events for Insurance and Finance*. Springer. The risk-management reading.
- Siffer, A., Fouque, P.-A., Termier, A. & Largouët, C. (2017). Anomaly
  detection in streams with extreme value theory. *KDD*, 1067–1075. SPOT and
  DSPOT, the streaming version of this pipeline.
- Goix, N., Trinquier, J. & Lanz, A. (2016). Sparse representation of
  multivariate extremes with applications to anomaly ranking. *AISTATS*. The
  multivariate extension this file deliberately does not attempt.
