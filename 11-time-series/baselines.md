# Naive baselines — the forecasts your model has to beat

`baselines.py` · Time series · Tier 1 · 6 variants

## Intuition

A forecast is a claim about the future, and the cheapest possible claim is that
nothing changes:

$$
\hat y_{T+h}=y_T\qquad\text{for every }h\ge1.
\qquad (1)
$$

That is the naive forecast, and on the M4 and M5 competitions it is embarrassing
how much of the field it beats. The reason is not that the world is static; it is
that a random walk is a good approximation for prices, inventories, click counts
and demand, and for a random walk the last value is not merely a baseline — it is
the conditional mean. Any model that loses to it has explained nothing.

This file contains the five baselines that every forecasting result should be
reported against, plus the two aliases the literature uses for them, and the
predictive intervals that separate them. The interesting part is not the point
forecasts, which are one line each, but the *variance*: the naive forecast and
the random-walk forecast are the same number, yet one assumes the forecast error
is constant in $h$ and the other assumes it grows like $\sqrt h$. Over a horizon
of ten, one of those is wrong by a factor of three.

## Variants

| variant | point forecast | error scale | distribution |
|:--|:--|:--|:--|
| `naive` | $y_T$ | $\hat\sigma$ constant | none |
| `random_walk` | $y_T$ | $\hat\sigma\sqrt h$ | Gaussian |
| `drift` | $y_T+h\hat b$ | $\hat\sigma\sqrt{h(1+h/(T-1))}$ | Gaussian |
| `mean` | $\bar y$ | $\hat\sigma\sqrt{1+1/T}$ | Gaussian |
| `seasonal_naive` | $y_{T+h-m\lceil h/m\rceil}$ | $\hat\sigma$ constant | none |
| `snaive` | same as `seasonal_naive` | same | same |

**`naive`** is (1). Use it as the denominator of every accuracy ratio, and use
its *in-sample* residuals to see whether the series has structure left.

**`random_walk`** is the same point rule with the model behind it made explicit:
$y_t=y_{t-1}+\varepsilon_t$ with $\varepsilon_t$ i.i.d. Then the $h$-step error is
a sum of $h$ innovations, so its variance is $h\sigma^2$ and the interval fans
out. The demo's band is 3.243 wide at $h=1$ and 10.254 at $h=10$, exactly
$\sqrt{10}$ times wider.

**`drift`** adds the average historical change, $\hat b=(y_T-y_1)/(T-1)$, which
is the maximum-likelihood slope of a random walk with drift. It wins on the
demo's trending series (MAE 1.3616 against 1.3939) and loses badly on anything
that mean-reverts.

**`mean`** is the forecast for a stationary series: the sample average repeated.
It is the only variant whose interval includes the error in estimating the mean,
the $1/T$ term, and it is usually the worst of the five in practice because
economic series are rarely stationary.

**`seasonal_naive`** recycles the last observed cycle. With a period of 12 on the
demo's seasonal series its MAE is 0.5068 against 1.3939 for the naive, and on the
seasonal series with no trend the gap is 0.145 against 1.515. Getting $m$ right is
the entire method.

**`snaive`** is the name the R ecosystem uses for the same estimator, so the
implementation returns the same forecasts and the demo asserts it. It is listed
separately because readers search for it, not because it computes anything else.

## Derivation

With observations $y_1,\dots,y_T$ and horizon $h=1,2,\dots$, the four distinct
point rules are

$$
\hat y_{T+h}=y_{T+h-m\lceil h/m\rceil}
\qquad (2)
$$

for the seasonal naive (the observation one cycle back, $m$ being the period),

$$
\hat y_{T+h}=y_T+h\hat b,\qquad
\hat b=\frac{y_T-y_1}{T-1}
\qquad (3)
$$

for drift, and

$$
\hat y_{T+h}=\bar y=\frac1T\sum_{t=1}^{T}y_t
\qquad (4)
$$

for the mean. The naive rule is (1). All four are linear in the data with
coefficients that do not depend on $h$ except through (3), which is why they are
the right thing to compute before anything expensive.

### The error scale

Each rule has a one-step residual it implies, and $\hat\sigma$ is the sample
standard deviation of that residual:

$$
\hat\sigma^2=
\frac{1}{T-1}\sum_t \hat e_t^2,\qquad
\hat e_t=
\begin{cases}
y_t-y_{t-1}, & \text{naive, random walk}\\
y_t-y_{t-1}-\hat b, & \text{drift}\\
y_t-y_{t-m}, & \text{seasonal}\\
y_t-\bar y, & \text{mean}
\end{cases}
\qquad (5)
$$

The `ddof=1` matters: with $T=20$ the difference between dividing by 19 and 20 is
5% of the variance and 2.5% of the interval width, which is the same order as
the effect being measured.

### Predictive variance

For the random walk, the $h$-step error is $\sum_{j=1}^{h}\varepsilon_{T+j}$, so

$$
\mathrm{Var}(\hat y_{T+h}-y_{T+h})=h\sigma^2 .
\qquad (6)
$$

For the drift method the slope is estimated, and the estimation error accumulates
with the forecast, giving the standard approximation

$$
\mathrm{Var}=h\sigma^2\Big(1+\frac{h}{T-1}\Big).
\qquad (7)
$$

For the mean, the forecast error is the deviation of one new observation from an
estimated mean:

$$
\mathrm{Var}=\sigma^2\Big(1+\frac1T\Big).
\qquad (8)
$$

For naive and seasonal naive the implementation uses the constant $\hat\sigma^2$
of (5): the naive method has no model that would justify a growing variance, and
pretending otherwise would be an assumption, not a baseline. That is the honest
asymmetry between `naive` and `random_walk` in the table — same number, different
claim.

The band is Gaussian,

$$
\hat y_{T+h}\pm z_{1-\alpha/2}\,\hat\sigma_h,
\qquad (9)
$$

with $z$ from the standard normal quantile. For the random walk that is not an
approximation but the exact predictive law when the innovations are Gaussian,
which is why the demo can test it: 400 simulated walks give 0.948 coverage at
$h=1$ and 0.965 at $h=10$ against a nominal 0.95.

## Code walkthrough

`fit` computes every statistic any variant could need and then the one scale that
variant implies. The seasonal branch is where the period enters:

```python
        if self.variant in SEASONAL:
            if self.season > self.n_ - 3:
                raise ValueError("season leaves too few seasonal differences")
            gap = y[self.season:] - y[:-self.season]
            self.sigma_ = float(np.std(gap, ddof=1))
            self.seasonal_ = y[-self.season:].copy()
```

`gap` is the vector of $y_t-y_{t-m}$, computed with two slices rather than a
loop; `self.seasonal_` keeps the last cycle so that `predict` can recycle it
without knowing anything else. The guard is not cosmetic: with `season > n_ - 3`
there are fewer than three seasonal differences and `ddof=1` is undefined.

Drift estimates its slope from the endpoints, which is the least-squares slope
only because the regressor is time:

```python
        elif self.variant == "drift":
            if self.n_ < 3:
                raise ValueError("the drift method needs at least three points")
            self.slope_ = (y[-1] - y[0]) / (self.n_ - 1)
            residual = np.diff(y) - self.slope_
            self.sigma_ = float(np.std(residual, ddof=1))
```

The residual is the *difference* minus the slope, not the level minus a fitted
line. That is the one-step error the method actually makes, and it is the right
scale for (7).

`predict` is a dispatch on the variant, with the seasonal branch expressed as an
index into the stored cycle:

```python
        steps = np.arange(1, horizon + 1, dtype=float)
        if self.variant in ("naive", "random_walk"):
            return np.full(horizon, self.last_)
        if self.variant == "mean":
            return np.full(horizon, self.mean_)
        if self.variant == "drift":
            return self.last_ + steps * self.slope_
        index = (np.arange(horizon) % self.season)
        return self.seasonal_[index]
```

`np.arange(horizon) % self.season` is the whole of (2) once the cycle is stored
in order: $h=1$ reads `seasonal_[0]`, which is $y_{T-m+1}$, and $h=m$ reads
`seasonal_[m-1]`, which is $y_T$. Writing the index from 1 and forgetting the
shift is the usual off-by-one here, and it shows up as a forecast one period out
of phase.

The interval is the only place the variants diverge after `fit`, and each branch
is one line of (6)–(8):

```python
        if self.variant == "random_walk":
            sd = self.sigma_ * np.sqrt(steps)
        elif self.variant == "drift":
            sd = self.sigma_ * np.sqrt(steps * (1.0 + steps / (self.n_ - 1)))
        elif self.variant == "mean":
            sd = np.full(horizon, self.sigma_ * np.sqrt(1.0 + 1.0 / self.n_))
        else:
            sd = np.full(horizon, self.sigma_)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return forecast - z * sd, forecast + z * sd
```

`NormalDist` is in the standard library and removes any temptation to hardcode
1.96; the quantile is exact for any level. Note that `steps` is float, so
`np.sqrt` broadcasts without an integer division trap on older NumPy.

The guards are two small functions, and the split between them is deliberate:

```python
    def _check_horizon(self, horizon) -> int:
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return int(horizon)
```

A float horizon of `2.5` raises rather than truncating: silently forecasting 2
steps when 2.5 were asked for is a bug that surfaces as a shape mismatch three
layers up. The `isinstance` test also rejects `True`, which is an `int` in Python
and would otherwise be a one-step forecast.

## Complexity

With $T$ observations and horizon $h$:

- **`fit`:** $O(T)$ time and $O(T)$ memory for the copy of the series; the
  seasonal branch stores only $m$ values.
- **`predict`:** $O(h)$.
- **`interval`:** $O(h)$.
- **All variants together:** the whole demo, including 400 simulated random
  walks of length 120 and the coverage check, is a few hundred thousand floating
  point operations.

This is the cheapest atom in the book, and that is the point: any model that
costs more must be justified against a forecast that costs nothing.

## Failure modes

**Naive on a trending series.** If the series has a stable trend, last-value is
biased by exactly the trend per step, and the error grows linearly with $h$
rather than staying constant. The demo shows the drift method winning on its
trending series, but the margin is small (1.3616 against 1.3939) because the
season dominates; on a pure trend the gap is the trend itself. Diagnose by
plotting the residuals of the naive forecast against $h$: a fan shape means
drift, a constant band means a random walk.

**The wrong seasonal period is worse than no season.** A seasonal naive with
$m$ off by one forecasts the wrong phase, and the result can be worse than the
naive. The demo asserts the reverse ordering on data where $m=12$ is correct.
Diagnose by comparing MAE with the naive on held-out data before believing any
seasonal model; if seasonal naive loses, the period or the phase is wrong.

**Intervals that ignore regime change.** Equation (6) assumes the innovations
are stationary. A volatility break makes the band too narrow, and the coverage
check that passes on simulated walks fails on real data. The diagnostic is a
rolling coverage plot, not a single number, and no baseline can fix it: the
interval is only as good as the stationarity assumption behind $\hat\sigma$.

**Drift extrapolates a slope that has already changed.** The estimate uses only
the first and last observation, so a single outlier at either end moves the
whole forecast path. With $T=20$ one bad endpoint can flip the sign. Diagnose by
recomputing $\hat b$ on the second half of the series; if it changes materially,
drift is not a baseline but a bet.

## Applications

Every forecasting benchmark is scored relative to these methods: MASE divides by
the in-sample MAE of the naive (or seasonal naive) forecast, and the M4/M5
leaderboards are read that way. Reporting a model without the naive number is
reporting an unnormalised error.

Operationally, `random_walk` is the null model for prices, inventories and
anything with a unit root, and the interval of (6) is what a "no change"
scenario should look like in a planning tool. `seasonal_naive` is the standard
for weekly and yearly cycles — call volumes, retail, energy — and is often the
production model for a new SKU with no history. `drift` is the honest default for
a slowly moving KPI over a short horizon.

They are a trap as *probabilistic* forecasts on autocorrelated data, where the
residuals are not iid and every interval here is too narrow, and as models for
series with a known driver (price, promotion, weather) where the whole point is
to use it.

## References

- Hyndman, Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd
  ed., ch. 5. The canonical treatment of these five methods, their residuals and
  the interval formulas (6)–(8).
- Makridakis, Spiliotis, Assimakopoulos (2020). *The M4 competition: 100 000
  time series and 61 forecasting methods.* International Journal of Forecasting
  36(1). How the baselines actually placed.
- Hyndman, Koehler (2006). *Another look at measures of forecast accuracy.*
  International Journal of Forecasting 22(4). MASE, the scaled error that uses
  the naive forecast as its unit.
- Kolassa (2016). *Evaluating predictive count distributions.* International
  Journal of Forecasting. Why a point-forecast baseline is not a distribution,
  and how to score the interval versions properly.
