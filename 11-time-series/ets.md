# Exponential smoothing — level, trend and season in one recursion

`ets.py` · Time series · Tier 1 · 7 variants

## Intuition

A forecast is a summary that keeps correcting itself. Simple exponential
smoothing holds one number, the level, and after each observation moves it a
fraction of the way toward the error it just made:

$$
\ell_t=\ell_{t-1}+\alpha\,(y_t-\ell_{t-1}).
\qquad (1)
$$

That is a weighted average with geometrically decaying weights, and $\alpha$ is
the only thing to choose. It is the cheapest adaptive forecaster in existence
and the ancestor of everything in this file.

Holt added a slope, so the summary becomes a line: one state for where the
series is, one for where it is going. Winters added a season, so the summary
becomes a line plus a repeating profile. Damping shrinks the slope so that a
long-horizon forecast flattens instead of shooting off. These are the three
switches of the ETS taxonomy — **E**rror, **T**rend, **S**eason — and every
variant here is the same loop with some states present and some absent.

Two things make this more than a bag of heuristics. First, the recursions are
exactly the error-correction form of a linear state-space model, so the same
recursion that produces point forecasts also produces an exact forecast variance
by propagating a covariance matrix. The demo checks that band against 2 000
simulated paths and the two agree to 2.6% over eighteen steps. Second, the
smoothing weights can be estimated by minimising the one-step squared error,
which for additive errors is maximum likelihood, and the resulting likelihood
supports AICc comparison across models with different numbers of states. On the
demo's seasonal series the best seasonal model beats the best non-seasonal one
by 177 AICc points.

The interesting tension is that all of this is still local: one level, one
slope, one profile. A series with a regime change will be chased slowly, and no
amount of AICc will warn you.

## Variants

| variant | ETS | states | parameters | demo MAE |
|:--|:--|:--|:--|--:|
| `simple_es` | A,N,N | level | $\alpha$ | 1.9850 |
| `holt` | A,A,N | level, trend | $\alpha,\beta$ | 2.0781 |
| `damped_trend` | A,Ad,N | level, damped trend | $\alpha,\beta,\phi$ | 1.9777 |
| `holt_winters_add` | A,A,A | + additive season | $\alpha,\beta,\gamma$ | 0.9615 |
| `holt_winters_mul` | A,A,M | season × level | $\alpha,\beta,\gamma$ | 1.0841 |
| `ets_state_space` | A,Ad,A | damped, seasonal | $\alpha,\beta,\gamma,\phi$ | 0.8984 |
| `auto_ets` | searched | chosen by AICc | chosen | 0.9615 |

**`simple_es`** is (1). On a seasonal series it does the only thing it can: the
demo's optimiser drives $\alpha$ to its 0.999 ceiling and the model degenerates
to the naive forecast. That is not a bug, it is the method telling you it has no
way to represent a period.

**`holt`** adds a slope updated by its own error term. It is the right default
for a series with a local linear trend and no season, and it is what `drift`
becomes when the slope is allowed to adapt.

**`damped_trend`** multiplies the slope by $\phi\in(0,1)$ at every step, so the
forecast is a geometric series that converges to a finite level. On the demo it
edges out `holt` (1.9777 against 2.0781) with a fitted $\phi$ at the 0.8 lower
bound, which means the data wants very aggressive damping.

**`holt_winters_add`** adds a seasonal profile whose entries are added to the
line. This is the model to reach for when the seasonal swing is roughly constant
in absolute size — temperature, staffing, scheduled maintenance.

**`holt_winters_mul`** multiplies the line by the profile, so the swing grows
with the level. On the demo's tripling series it cuts the in-sample SSE from
87.468 to 57.861 and the held-out MAE from 0.8698 to 0.7888.

**`ets_state_space`** is the same additive-season model written as
$x_t=Fx_{t-1}+ge_t$ with a damped trend, which buys two things: the matrices are
exposed, and `interval` uses the exact variance recursion instead of a
heuristic. Its fitted $\phi$ sits at 0.995, the top of the search range.

**`auto_ets`** fits the nine admissible combinations and returns the one with
the smallest AICc. On the additive series it chooses `ETS(A,A,A)` with the same
parameters as `holt_winters_add`; on the multiplicative series it switches to
`ETS(A,A,M)`.

## Derivation

### Setup

Write the level $\ell_t$, the slope $b_t$, and a profile $s_t$ with period $m$.
The one-step forecast is built from the states as they stood before $y_t$ was
seen, and the error is

$$
e_t=y_t-\hat y_t .
\qquad (2)
$$

Every update below is an error-correction step: the state moves in proportion to
$e_t$, and the forecast for $t+1$ is recomputed from the moved states. This is
the form in which the recursions are both easiest to implement and exactly
equivalent to the state-space model of (9).

### Level only

With $\ell_0=y_1$ and $\hat y_t=\ell_{t-1}$,

$$
\ell_t=\ell_{t-1}+\alpha e_t,\qquad \hat y_{T+h}=\ell_T .
\qquad (3)
$$

Unrolling the recursion gives the weighted-average form the demo asserts,

$$
\hat y_{T+1}=\alpha\sum_{j=0}^{T-1}(1-\alpha)^j y_{T-j}
+(1-\alpha)^T y_1 ,
\qquad (4)
$$

which makes the memory explicit: the weight on an observation $j$ steps old is
$\alpha(1-\alpha)^j$, so a half-life of $\log(1/2)/\log(1-\alpha)$ observations.

### Trend, and damping

Holt's model adds a slope that is itself smoothed:

$$
\ell_t=\ell_{t-1}+\phi b_{t-1}+\alpha e_t,\qquad
b_t=\phi b_{t-1}+\beta e_t,\qquad
\hat y_t=\ell_{t-1}+\phi b_{t-1}.
\qquad (5)
$$

With $\phi=1$ this is Holt; with $\phi<1$ the slope decays geometrically and the
forecast is

$$
\hat y_{T+h}=\ell_T+b_T\sum_{j=1}^{h}\phi^j
\;\longrightarrow\;
\ell_T+\frac{\phi b_T}{1-\phi}
\qquad (6)
$$

as $h\to\infty$. The demo's fitted path converges to 54.619977, which is exactly
$\ell_T+0.9b_T/(1-0.9)$, and the gaps between successive forecasts shrink
monotonically. Damping is the difference between a forecast that is still
extrapolating at $h=100$ and one that has settled.

### Seasonality

The additive profile is a fourth state, indexed modulo $m$:

$$
\hat y_t=\ell_{t-1}+\phi b_{t-1}+s_{t-m},\qquad
s_t=s_{t-m}+\gamma e_t .
\qquad (7)
$$

The multiplicative version scales the line by the profile:

$$
\hat y_t=(\ell_{t-1}+\phi b_{t-1})\,s_{t-m},\qquad
s_t=s_{t-m}+\gamma\,\frac{e_t}{\ell_{t-1}+\phi b_{t-1}} .
\qquad (8)
$$

and its level and slope updates divide $e_t$ by $s_{t-m}$, because a relative
error of $\varepsilon_t=e_t/\hat y_t$ is what the multiplicative model assumes.
Equations (7) and (8) are why the code guards `season_now > 0` and `base > 0`:
a zero or negative seasonal factor makes the multiplicative recursion
meaningless, and during a parameter search it is reached routinely.

### The state-space form

Stack the states into $x_t=(\ell_t,b_t,s_t,\dots,s_{t-m+1})^\top$ with the
seasonal entry in use first. Then (5) and (7) are

$$
x_t=Fx_{t-1}+g e_t,\qquad \hat y_t=w^\top x_{t-1},
\qquad (9)
$$

with

$$
F=\begin{pmatrix}
1 & \phi & 1 & 0 & \cdots\\
0 & \phi & 0 & 0 & \cdots\\
0 & 0 & 0 & 1 & \cdots\\
\vdots & & & \ddots & \\
0 & 0 & 0 & \cdots & 0
\end{pmatrix},\quad
w=\begin{pmatrix}1\\ \phi\\ 1\\ 0\\ \vdots\end{pmatrix},\quad
g=\begin{pmatrix}\alpha\\ \beta\\ 0\\ \vdots\\ \gamma\end{pmatrix}.
\qquad (10)
$$

The first row of $F$ rebuilds the forecast from level, damped slope and the
seasonal entry; the last row of the seasonal block is zero because the new
seasonal value arrives through $g e_t$. The shift rows rotate the profile, so
the entry used at $t+1$ moves to the front. For a non-seasonal model the
seasonal rows and columns are simply absent, and $F$ is $1\times1$ or $2\times2$.

### Forecast variance

Because (9) is linear in the state, the forecast covariance propagates exactly.
Let $P_0=0$, conditional on the estimated states, and iterate

$$
\mathrm{Var}(\hat y_{T+h})=w^\top P_{h-1}w+\sigma^2,\qquad
P_h=FP_{h-1}F^\top+\sigma^2 gg^\top .
\qquad (11)
$$

The trailing $\sigma^2$ is the innovation in $y_{T+h}$ itself; the covariance
term carries the uncertainty accumulated in the states. For simple exponential
smoothing $F=1$, $w=1$, $g=\alpha$, so $P_h=h\alpha^2\sigma^2$ and

$$
\mathrm{Var}(\hat y_{T+h})=\sigma^2\bigl(1+(h-1)\alpha^2\bigr),
\qquad (12)
$$

which the demo checks against the band to machine precision: at $h=6$ with
$\alpha=0.4$ the standard deviation is 3.3064. Simulation of the full
state-space model is the independent check, and it agrees to 2.6%.

### Estimation and selection

With additive errors and $\sigma^2$ concentrated out, minus twice the Gaussian
log-likelihood is

$$
-2\log L=n_{\rm eff}\Bigl(\log\frac{2\pi\,\mathrm{SSE}}{n_{\rm eff}}+1\Bigr),
\qquad (13)
$$

a monotone function of the sum of squared one-step errors, so minimising SSE is
maximum likelihood. For multiplicative seasonality the likelihood is written in
the relative errors with the Jacobian $-\sum_t\log\hat y_t$, which is why
`sigma2_` is a relative variance for those models. The information criteria are

$$
\mathrm{AIC}=-2\log L+2k,\qquad
\mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n_{\rm eff}-k-1},
\qquad (14)
$$

with $k$ counting the smoothing parameters, $\sigma^2$, and the initial states.
The correction matters here: a seasonal model carries $m$ extra states, and at
$n_{\rm eff}=108$ and $m=12$ the AICc penalty for those states is tens of points,
not a rounding error.

### Initialisation

The first season sets the level and the profile, and the first two seasons set
the slope:

$$
\ell_0=\frac1m\sum_{j=1}^{m}y_j,\qquad
s_{1-j}=y_j-\ell_0,\qquad
b_0=\frac{\bar y_{m+1:2m}-\bar y_{1:m}}{m}.
\qquad (15)
$$

With a multiplicative profile the subtraction is a division. The first $m$
residuals are near zero by construction, so the fit drops them from the SSE and
from $n_{\rm eff}$; that is the `start_` attribute, and it is why a seasonal fit
on 120 points reports 108 usable residuals.

## Code walkthrough

Everything happens in one loop over the series, with the seasonal entry rotated
so that `seasons[0]` is always the value for the current step:

```python
        for t in range(y.size):
            season_now = float(seasons[0]) if with_season else 0.0
            base = level + phi * slope
            if season == "A":
                forecast = base + season_now
            elif season == "M":
                if base <= 1e-9 or season_now <= 1e-9:
                    return None
                forecast = base * season_now
            else:
                forecast = base
```

`base` is computed once and reused by both the forecast and the multiplicative
updates, which is what keeps (8) consistent. The `return None` is not an error
path for the user: it is how an invalid parameter combination is signalled to
the optimiser, which turns it into an infinite loss.

The correction step is the whole of (5) and (8) in four lines:

```python
            error = float(y[t]) - forecast
            level = base + alpha * (error if season != "M"
                                    else error / season_now)
            slope = phi * slope + (beta * (error if season != "M"
                                           else error / season_now))
```

Reading the two conditional expressions is the fastest way to see why the
multiplicative model needs the seasonal factor twice: the level error is
relative to $s_{t-m}$, and the slope error is too. For an additive model the
same lines are the textbook Holt updates.

The profile is rotated with a slice and a concatenation rather than with an
index variable:

```python
            if season == "A":
                seasons = np.concatenate([seasons[1:], [season_now + gamma * error]])
            elif season == "M":
                seasons = np.concatenate([seasons[1:],
                                          [season_now + gamma * error / base]])
```

After this line `seasons[0]` is the value for $t+1$, so the loop never has to
know where in the cycle it is. The cost is a copy of $m$ floats per step, which
is irrelevant next to the clarity.

The parameters are found by a coordinate pattern search: try a step up and down
each free parameter, keep any improvement, and halve all step sizes when a full
sweep finds nothing.

```python
        for _ in range(90):
            improved = False
            for name in free:
                low, high = LIMITS[name]
                for delta in (steps[name], -steps[name]):
                    trial = dict(values)
                    trial[name] = float(np.clip(values[name] + delta,
                                                low, high))
                    if trial[name] == values[name]:
                        continue
                    score = loss(trial)
                    if score < best - 1e-12:
                        values, best, improved = trial, score, True
```

There is no gradient because propagating one through the recursion gives an
expression that changes shape with the model and is easy to get subtly wrong,
and because $lpha$ and $eta$ trade off along a flat ridge where a
gradient's step size is hard to tune. Pattern search is slower but far harder
to derail. Two limits
are clipped: $\alpha,\beta,\gamma$ to $[0.01,0.999]$ and $\phi$ to
$[0.80,0.995]$, the range in which a damped trend is identified rather than
merely fitted.

Forecasting damps the slope, not just the first step:

```python
        for h in range(horizon):
            base = level + phi * slope
            if self.season_ == "A":
                out[h] = base + seasons[h % self.season_period_]
            elif self.season_ == "M":
                out[h] = base * seasons[h % self.season_period_]
            else:
                out[h] = base
            level = base
            slope = phi * slope
```

The `slope = phi * slope` line is the one that is easy to forget, and forgetting
it turns (6) into an undamped line with a $\phi$-scaled slope that diverges. The
damped-limit assertion in the demo exists precisely to catch that bug.

`state_space` builds the matrices of (10) from the fitted states, which is used
by `interval` and by the demo's simulation:

```python
        if self.season_ == "A":
            matrix[0, 2] = 1.0
            for j in range(m - 1):
                matrix[2 + j, 3 + j] = 1.0
            w[2] = 1.0
            g[-1] = self.params_["gamma"]
```

The shift is `matrix[2 + j, 3 + j] = 1.0`, so row 2 copies column 3, row 3
copies column 4, and so on; the last seasonal row is left at zero because its
new value comes from `g`. Only additive seasonality has this form, so the method
raises for multiplicative models rather than inventing a matrix.

The variance recursion is six lines and no simulation:

```python
            matrix, w, g, _ = self.state_space()
            covariance = np.zeros_like(matrix)
            variance = np.empty(horizon)
            for h in range(horizon):
                variance[h] = float(w @ covariance @ w) + self.sigma2_
                covariance = (matrix @ covariance @ matrix.T
                              + self.sigma2_ * np.outer(g, g))
```

`covariance` starts at zero because the states are treated as known; that
understates the band slightly, since it ignores the error in estimating
$\ell_T$ and $b_T$, but it is the standard conditional ETS variance and it is
what the Monte Carlo check validates.

Model selection is a loop over candidates with AICc as the comparator:

```python
        best, table = None, []
        for trend, season in candidates:
            fitted = self._fit_spec(y, trend, season)
            table.append((f"ETS(A,{trend},{season})", fitted["aicc"]))
            if best is None or fitted["aicc"] < best["aicc"]:
                best = fitted
```

`table` is kept so that the chosen model can be interrogated: the demo prints
all nine AICc values and the gap between the first and second is 3.4 points,
while the gap to the best non-seasonal model is 177.

## Complexity

With $n$ observations, $m$ the period, $p$ free parameters and $h$ the horizon:

- **One pass of the recursion:** $O(n)$ time — the seasonal rotation copies $m$
  values per step, so the constant is $O(m)$ — and $O(n)$ memory for the fitted
  values and residuals.
- **One fit:** $O(n \cdot p \cdot 2 \cdot R)$ where $R\approx 8$ is the number of
  step halvings; the demo's nine-candidate `auto_ets` is about seventy thousand
  recursion steps.
- **`predict`:** $O(h)$.
- **`interval`:** $O(h\,d^3)$ for the $d\times d$ covariance products, with
  $d=1+m$ at most 14. Negligible.
- **The demo's simulation:** 2 000 paths of 18 steps, 36 000 small matrix
  products, is the most expensive thing in the file.

This is $O(n)$ against $O(n^3)$ for a Gaussian process and $O(np)$ per boosting
round, which is why exponential smoothing survived the neural era: it is the
cheapest model that represents a trend and a season at all.

## Failure modes

**A weight stuck at its boundary is a diagnosis, not a nuisance.** `simple_es`
ends at $\alpha=0.999$ on the seasonal series because no value of $\alpha$ can
represent a period-12 pattern; the best it can do is track the last observation.
A grid over 200 values confirms 0.995 as the true optimum, so the boundary is
real. When a weight hits the ceiling, the model is misspecified — add the
missing state rather than narrowing the bound.

**AICc and a short holdout can disagree.** `ets_state_space` has the best
12-point MAE (0.8984 against 0.9615) but a worse AICc (325.25 against 321.87),
because the extra damping parameter does not pay for itself in-sample. With
twelve held-out points the MAE difference is one noise draw; AICc uses 108
residuals. Prefer the criterion that uses more data, and treat a one-cycle
holdout as a sanity check, not a ranking.

**A trend model on a series without a trend.** Holt and the damped variant will
extrapolate whatever slope the last few points imply, and damping only shrinks
it. On a mean-reverting series a differencing or a mean forecast is safer; the
diagnostic is a residual autocorrelation that stays positive at long lags.

**Multiplicative seasonality near zero.** Equations (8) divide by the level and
the seasonal factor, so a series that touches zero makes the recursion explode.
The code refuses non-positive $y$ up front, but a series that is merely small is
still numerically fragile: relative errors become enormous and the SSE surface
is dominated by the smallest values. Use the additive model unless the swing is
demonstrably proportional to the level.

**Discarding the first season.** With $m=12$ and $n=120$, 10% of the sample is
spent on initialisation. On a series of two cycles there are only twelve usable
residuals, $k$ can approach $n_{\rm eff}$, and AICc's denominator
$n_{\rm eff}-k-1$ becomes small enough to dominate the ranking. Require at least
three cycles before trusting `auto_ets`.

**The wrong period.** A seasonal model with $m$ off by one forecasts the wrong
phase every cycle. Unlike the baselines in `baselines.py`, ETS does not warn:
the SSE falls, because a misaligned profile still absorbs variance. Always
compare against the non-seasonal model on held-out data, and inspect the fitted
profile for a plausible shape.

## Applications

Exponential smoothing is the production forecaster for demand planning, call
centre staffing, energy load and inventory replenishment, and it has been for
sixty years. The reasons are practical: the states are interpretable (a planner
can read the current level and the seasonal index), the model is cheap enough to
refit every night for millions of SKUs, and the intervals from (11) are honest
enough for safety stock.

The ETS framework also gives a principled way to answer "which model?" without a
validation set, which matters when each series is short. `auto_ets` is the
workhorse behind the `forecast` package's `ets()`, and the AICc table it produces
is a useful artefact in its own right: it shows how much of the signal each
state is buying.

Hierarchical forecasting uses the additive decomposition directly: forecasts at
the region, category and SKU level must add up, and the level/trend/season states
give a natural space in which to reconcile them.

It is a trap when the series is driven by known covariates (price, promotion,
weather), where a regression with ARIMA errors or a gradient-boosted model will
win, and when the future is a regime change rather than a continuation. It is
also a trap for long horizons with a seasonal model: the profile repeats
forever, and after two or three cycles the band is wide enough that the point
forecast is not the interesting part.

## References

- Brown (1959). *Statistical Forecasting for Inventory Control.* McGraw-Hill.
  The origin of simple exponential smoothing.
- Holt (1957). *Forecasting seasonals and trends by exponentially weighted
  moving averages.* ONR Research Memorandum 52. The level-plus-trend recursion.
- Winters (1960). *Forecasting sales by exponentially weighted moving averages.*
  Management Science 6(3). The seasonal extension, additive and multiplicative.
- Gardner, McKenzie (1985). *Forecasting trends in time series.* Management
  Science 31(10). The damped trend and the case for $\phi<1$.
- Hyndman, Koehler, Snyder, Grose (2008). *Forecasting with Exponential
  Smoothing: The State Space Approach.* Springer. The ETS taxonomy, the
  innovations form and the variance recursions of (11)–(12).
- Ord, Koehler, Snyder (1997). *Estimation and prediction for a class of dynamic
  nonlinear statistical models.* Journal of the American Statistical Association
  92(440). The likelihood and the AICc correction for these models.
- Hyndman, Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd
  ed., ch. 8. The applied treatment, including `auto_ets` and its pitfalls.
