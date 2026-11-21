# Croston — forecasting demand that mostly is not there

`croston.py` · Time series · Tier 3 · 4 variants

## Intuition

Most forecasting methods assume a value arrives every period. Intermittent
demand breaks that assumption: a spare part sells on 8% of days, a repair kit
is ordered a few times a year, a call centre receives nothing for hours and then
three calls at once. Applying exponential smoothing directly to such a series
gives a forecast that is too high on most days and too low on the days that
matter, and the errors are dominated by the zeros.

Croston's insight was to stop forecasting the series and start forecasting its
two ingredients. Demand has a **size** when it occurs and an **interval**
between occurrences. Smooth each one separately, then forecast the rate as
size divided by interval. A long gap lowers the rate through the interval
estimate rather than through a zero that drags an ordinary smoother down.

The variants fix different weaknesses of that recipe:

- **`croston`** is the original: smooth sizes and inter-arrival gaps, forecast
  size over interval.
- **`sba`** applies the Syntetos–Boylan correction, multiplying the rate by
  $1-\alpha/2$ to remove Croston's small positive bias.
- **`tsb`** smooths the *probability* of a demand every period and the size only
  at arrivals. Because the probability decays during a long gap, TSB can
  forecast obsolescence; Croston cannot.
- **`optimized_croston`** fits the smoothing parameter by minimising the
  one-step sum of squares instead of using a fixed value.

The demo measures all four on simulated intermittent demand: a 600-period
series with 115 demands, a regime change in the demand probability, and a
Poisson interval whose coverage is checked by simulation.

## Variants

| variant | state | demo result |
|:--|:--|:--|
| `croston` | size and interval | rate 0.9537 vs mean 0.9217 |
| `sba` | Croston with $1-\alpha/2$ | rate 0.9061, distance 0.0156 |
| `tsb` | probability and size | MAE 0.5219 vs 0.6603 |
| `optimized_croston` | fitted $\alpha$ | $\alpha=0.32$, SSE 1042.56 |

**`croston`** sees 115 demands in 600 periods, with an average size of $4.6951$
and an average smoothed interval of $4.9228$. The implied rate is $0.9537$
against an empirical mean of $0.9217$ and a true rate of $1.00$. The estimate is
noisy because the interval state reacts slowly to a long gap at the end of the
sample; with $\alpha=0.1$ the last gap carries a lot of weight.

**`sba`** multiplies the rate by $1-\alpha/2=0.95$, giving $0.9061$. Its distance
to the empirical mean is $0.0156$ against $0.0321$ for plain Croston. That is
the bias correction working on this sample, but it is not a guarantee: the
correction is derived for a stationary process with a known $\alpha$, and a
single series can go either way.

**`tsb`** is fitted to a series whose demand probability drops from 35% to 10%
halfway through. Its probability state ends at $0.0172$ after a long stretch
without a demand, so its terminal rate ($0.0678$) is far below the trailing
100-period mean ($0.2600$). The terminal rate is not the right comparison: the
one-step MAE over the last 100 periods is $0.5219$ against $0.6603$ for Croston.
TSB tracks the new regime faster and also decays toward zero when demand stops
altogether.

**`optimized_croston`** scans $\alpha$ on a grid and chooses $0.32$, lowering the
one-step SSE from $1176.96$ at $\alpha=0.1$ to $1042.56$. The rate moves from
$0.4177$ to $0.3819$, closer to the post-change local mean. The fitted $\alpha$
is a one-step-optimal value; it need not be optimal for a 12-step forecast.

## Derivation

### The demand process

Write the series as a marked point process:

$$
Y_t=Z_t\,\mathbf 1[A_t],
\qquad (1)
$$

where $A_t$ is the event that a demand arrives at $t$ and $Z_t>0$ is its size.
The forecast target is the rate
$\lambda=\mathbb E[Y_t]=\Pr(A_t)\,\mathbb E[Z_t\mid A_t]$. The two factors are
estimated separately because they carry different information: sizes are
observed only on arrival, and the interval is observed only as elapsed time.

### Croston's recursions

Let $z_t$ be the smoothed size and $p_t$ the smoothed interval. At an arrival,

$$
z_t=\alpha Y_t+(1-\alpha)z_{t-1},
\qquad (2)
$$

and the interval is updated with the gap $q_t$ since the previous arrival:

$$
p_t=\alpha q_t+(1-\alpha)p_{t-1}.
\qquad (3)
$$

On a zero period nothing is updated except the gap counter $q_t\leftarrow
q_t+1$; the size and interval stay where they were. After the last observation
the forecast is flat:

$$
\hat Y_{T+h}=\frac{z_T}{p_T},\qquad h=1,\dots,H.
\qquad (4)
$$

The flat forecast is deliberate. Croston models a rate, not a path; there is no
trend or seasonal term to extrapolate, and allowing the rate to drift would
invent structure the data do not support.

### The bias and the SBA correction

The ratio $z/p$ is a biased estimator of $z/p$'s population counterpart. For
independent size and interval with means $\mu_z,\mu_p$ and interval coefficient
of variation $\mathrm{CV}_p$,

$$
\mathbb E\!\left[\frac{z}{p}\right]
\approx\frac{\mu_z}{\mu_p}
\bigl(1+\mathrm{CV}_p^2\bigr),
\qquad (5)
$$

so the estimate is too high, and the bias grows with the variability of the
intervals. Syntetos and Boylan proposed the correction

$$
\hat Y^{\mathrm{SBA}}_{T+h}
=\Bigl(1-\frac{\alpha}{2}\Bigr)\frac{z_T}{p_T},
\qquad (6)
$$

which is the leading term of the inverse for a smoothing parameter $\alpha$.
It is an approximation, not an exact unbiasedness result: it assumes
stationarity and a particular relationship between $\alpha$ and the interval
variability.

### TSB

Teunter, Syntetos and Babai replace the interval state with a probability that
is updated every period. Let $\pi_t$ be the probability of a demand at $t$ and
$z_t$ the size:

$$
\pi_t=\alpha\,\mathbf 1[Y_t>0]+(1-\alpha)\pi_{t-1},
\qquad (7)
$$

$$
z_t= \begin{cases}
\beta Y_t+(1-\beta)z_{t-1}, & Y_t>0,\\
z_{t-1}, & Y_t=0,
\end{cases}
\qquad (8)
$$

and the forecast is

$$
\hat Y_{T+h}=\pi_T\,z_T .
\qquad (9)
$$

The difference from Croston is visible in (7)–(8): the probability is smoothed
on every observation, so a run of zeros drives $\pi$ toward zero, while the size
is only revised when a demand arrives. That is what makes TSB able to forecast
obsolescence. The price is that the probability reacts to every zero, so a
single unusual gap can push the forecast down quickly.

### Fitting alpha

For `optimized_croston`, $\alpha$ is chosen by minimising the one-step sum of
squares:

$$
\hat\alpha=\arg\min_{\alpha\in(0,1]}
\sum_{t=\text{burn}+1}^{n}
\bigl(Y_t-\hat Y_t(\alpha)\bigr)^2,
\qquad (10)
$$

where $\hat Y_t(\alpha)$ is the forecast made before $Y_t$ is observed. The
first few residuals are excluded because the state is initialised from the first
demand and is not meaningful before it. The search is a grid of 49 values from
$0.02$ to $0.98$; the objective is not smooth in $\alpha$ because the gap
counter is integer-valued, so a grid is more reliable than a gradient.

### Poisson intervals

Conditional on the rate $\lambda$, the number of demands in $h$ periods is

$$
S_h\sim\mathrm{Poisson}(h\lambda).
\qquad (11)
$$

The interval reports the smallest $k$ with $\Pr(S_h\le k)\ge q$, and the
analogous upper quantile, where $q=(1-\text{level})/2$. This treats demand
occurrences as Poisson and ignores the variability of the sizes: a lumpy
series with occasional very large orders has a much wider total-demand
distribution than (11) implies. Use the interval as a count model, not as a
full predictive distribution for revenue.

## Code walkthrough

The Croston engine keeps three pieces of state: the smoothed size, the smoothed
interval, and the gap since the last demand. The forecast is written before the
update, so `fitted_[t]` is a genuine one-step-ahead prediction:

```python
    for t in range(first + 1, n):
        fitted[t] = size / interval
        if y[t] > 0.0:
            size = alpha * y[t] + (1.0 - alpha) * size
            interval = alpha * gap + (1.0 - alpha) * interval
            gap = 1.0
        else:
            gap += 1.0
```

The initialisation sets `size` to the first positive demand and `interval` to
one. That is arbitrary but harmless for the fit, because the first `first + 1`
residuals are excluded from the sum of squares. Using a future mean of the
positive demands would leak information into the forecast and flatter the
in-sample error.

TSB shares the skeleton but updates the probability on every period. Only the
size is updated at arrivals, which is what lets the probability decay during a
gap:

```python
    for t in range(first + 1, n):
        fitted[t] = probability * size
        if y[t] > 0.0:
            probability = alpha + (1.0 - alpha) * probability
            size = beta * y[t] + (1.0 - beta) * size
        else:
            probability = (1.0 - alpha) * probability
```

The initial probability of $0.5$ is a weak prior, not an estimate. With 600
observations its influence is negligible, but on a short series it matters;
in a production system it should be set from the known demand frequency or
estimated from a longer history.

The optimised variant scans alpha and keeps the best sum of squares. The
candidate list is stored so that the demo, or a user, can see how flat the
objective is:

```python
            candidates = []
            best = None
            for alpha in np.arange(0.02, 0.99, 0.02):
                result = _croston(y, float(alpha))
                candidates.append((float(alpha), result["sse"]))
                if best is None or result["sse"] < best["sse"]:
                    best = result
                    best_alpha = float(alpha)
```

The SBA branch is the same engine with a correction factor. Keeping the
uncorrected `size_` and `interval_` on the object means the components can be
inspected even when the reported rate is corrected:

```python
        else:
            correction = 1.0 if self.variant == "croston" else 1.0 - 0.5 * self.alpha
            best = _croston(y, self.alpha, correction)
            self.alpha_ = self.alpha
            self.candidates_ = [(self.alpha, best["sse"])]
```

The correction multiplies only the final rate, not the size or interval states;
applying it inside the recursion would double-correct every update.

Prediction is intentionally trivial. The method forecasts a rate, and the
rate does not depend on the horizon:

```python
    def predict(self, horizon: int = 12) -> np.ndarray:
        """Return the constant per-period rate for each step."""
        if not hasattr(self, "rate_"):
            raise RuntimeError("fit the model before predicting")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return np.full(int(horizon), self.rate_)
```

Returning an array rather than a scalar keeps the interface uniform with the
other forecasters in this chapter, and makes it explicit that every horizon
gets the same number. A user who wants a cumulative forecast should sum the
array, not expect the method to grow.

The Poisson quantile is computed by summing probabilities until the target is
reached. The loop is bounded so that a pathological mean cannot hang the
program:

```python
def _poisson_quantile(mean: float, probability: float) -> int:
    """Smallest k with P(Poisson(mean) <= k) >= probability."""
    if mean <= 0.0:
        return 0
    term = float(np.exp(-mean))
    total = term
    value = 0
    while total < probability and value < 100000:
        value += 1
        term *= mean / value
        total += term
    return value
```

The recurrence `term *= mean / value` computes each Poisson probability from
the previous one, which avoids a factorial that would overflow for even
moderate counts. The cap of 100000 is far above any realistic demand count and
exists only to make the termination property obvious.

The interval applies the quantile to the cumulative mean at each horizon.
Because the mean grows with $h$, both bounds are non-decreasing:

```python
        tail = 0.5 * (1.0 - level)
        means = self.rate_ * np.arange(1, int(horizon) + 1)
        lower = np.array([_poisson_quantile(mean, tail) for mean in means])
        upper = np.array([_poisson_quantile(mean, 1.0 - tail) for mean in means])
        return lower, upper
```

At $h=1$ with a rate near one, the 90% interval is $[0,3]$, which covers more
than 90% because the Poisson is discrete and skewed. That conservatism is
visible in the demo: the simulated coverage starts at $0.9835$ and settles near
$0.9235$ by $h=12$.

## Complexity

With $n$ observations, $G$ alpha candidates and $H$ forecast steps:

- **One Croston or TSB pass:** $O(n)$ time and $O(n)$ memory for the fitted
  values and residuals.
- **Optimised fit:** $O(Gn)$ with $G=49$ in the demo; a grid search over two
  parameters would be $O(G^2n)$, which is why TSB fixes `beta` to `alpha` when
  it is not supplied.
- **Prediction:** $O(H)$ to build the constant array, or $O(1)$ for a scalar.
- **Interval:** $O(H\bar k)$ for the cumulative sums, where $\bar k$ is the
  largest quantile; at a rate near one that is a handful of terms per step.
- **Memory:** $O(n)$ for `fitted_` and `residuals_`; none of the states grow
  with $n$.

The demo's four fits on 600 points run in a few hundredths of a second. The
method is cheap enough to refit per SKU in a nightly batch, which is how it is
usually deployed.

## Failure modes

**A zero is data, not a missing value.** Croston and TSB interpret every zero
as "no demand this period". If a zero actually means "the product was out of
stock" or "the feed was down", the model learns a demand rate that is too low
and the bias propagates into safety stock. Distinguish censored periods from
genuine zeros before fitting, and treat stockouts as missing observations.

**Croston is slow to react to a change in frequency.** The interval state is
only updated at arrivals, so it takes several arrivals to reflect a new regime.
In the demo's regime change, Croston's terminal rate is $0.4177$ against a
trailing mean of $0.2600$. TSB reacts faster because it updates the probability
every period, which is why TSB is preferred when demand may be disappearing.

**TSB overreacts to a long gap.** The mirror image of the previous point: a run
of zeros multiplies the probability by $(1-\alpha)$ each period, so the
terminal rate can be near zero even when the long-run rate is not. In the demo,
TSB ends at $0.0678$ while the last 100 periods average $0.2600$. Judge the
method by one-step accuracy over the recent window, not by the terminal number
alone.

**The SBA correction assumes stationarity.** The factor $1-\alpha/2$ is derived
for a stable demand process with a fixed smoothing parameter. Under a regime
change, or when $\alpha$ was chosen for a different regime, it can move the
forecast away from the truth. It is a small correction, not a substitute for
modelling the change.

**Optimised alpha overfits one-step errors.** The grid chooses the value that
minimises the sum of squared one-step errors. That is not the same as the value
that minimises the error of a 12-step cumulative forecast, and on a short series
the selected $\alpha$ can be unstable. Validate on a rolling origin, and prefer
a value from a neighbourhood of the optimum rather than a sharp minimum.

**The Poisson interval ignores size variability.** Equation (11) models the
number of arrivals, not the total size. A series with occasional orders of
hundreds of units has a heavy-tailed total-demand distribution that a Poisson
count interval understates badly. For inventory, bootstrap the sizes or use a
compound Poisson model; the count interval is only a lower bound on the
uncertainty.

**The flat forecast is an assumption.** Croston and TSB produce the same rate
for every horizon. If the demand rate is trending, or has a seasonal pattern
(e.g. heating spares), the flat forecast is biased at long horizons. Deseasonalise
or detrend first, forecast the adjusted rate, and add the structure back.

## Applications

Croston's method is standard in spare-parts and slow-moving inventory
management. Airlines, utilities and heavy-equipment manufacturers use it to
forecast the demand rate for components that fail rarely but expensively, and
to set reorder points from the resulting rate and interval. The SBA variant is
the default in several commercial inventory packages because the positive bias
of plain Croston translates directly into excess safety stock.

TSB is used where obsolescence matters: end-of-life electronics, fashion
accessories, and any SKU whose demand can stop permanently. Its ability to
drive the probability toward zero is the reason it is preferred to Croston for
product-life-cycle forecasting. The same machinery applies to call arrivals in
a service centre, insurance claims with a low frequency and a heavy size tail,
and failure counts in reliability engineering.

The method is the wrong tool when demand is dense enough for ordinary
exponential smoothing, when the sizes carry a trend, or when the arrivals are
strongly seasonal. It is also not a model of the size distribution: for revenue
forecasting, the size tail matters and a compound model is needed. Use Croston
as the rate forecast and model the size distribution separately.

## References

- Croston, J. D. (1972). *Forecasting and stock control for intermittent
  demands.* Operational Research Quarterly 23(3), 289–303. The original
  size-and-interval decomposition.
- Syntetos, A. A., Boylan, J. E. (2001). *On the bias of intermittent demand
  estimates.* International Journal of Production Economics 71(1–3), 457–466.
  The bias of the ratio estimator behind (5).
- Syntetos, A. A., Boylan, J. E. (2005). *The accuracy of intermittent demand
  estimates.* International Journal of Forecasting 21(2), 303–314. The SBA
  correction of (6).
- Teunter, R. H., Syntetos, A. A., Babai, M. Z. (2011). *Intermittent demand:
  linking forecasting to inventory obsolescence.* European Journal of
  Operational Research 214(3), 606–615. TSB and the probability recursion of
  (7)–(9).
- Hyndman, R. J., Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed., ch. 13. The applied treatment of intermittent demand and
  Croston's method.
