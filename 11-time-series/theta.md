# Theta — a trend you can see through exponential smoothing

`theta.py` · Time series · Tier 3 · 4 variants

## Intuition

Exponential smoothing is a good short-term forecaster and a bad trend
extrapolator: the level lags a moving series, so a flat forecast trails behind a
rising one. The theta method fixes that by adding back half the slope of a
straight line fitted to the whole record. It is one of the most successful
entries in the M3 competition, and its structure is almost embarrassingly
simple: smooth the series, add a drift, done.

The name comes from the *theta line*. Given a series $Y$, define a family
$Z(\theta)$ whose second differences are $\theta$ times those of $Y$. At
$\theta=0$ the line has no curvature and is a straight line; at $\theta=2$ the
local bends are doubled. The original method decomposes the series into the
$\theta=0$ and $\theta=2$ lines, forecasts the first with simple exponential
smoothing and the second with a linear regression, and averages the two. A
little algebra collapses that recipe into SES plus half the OLS slope, which is
what this file implements.

The variants add the pieces the original paper left fixed:

- **`theta`** is the standard method: $\theta=2$, half the OLS drift, and a
  fitted smoothing parameter.
- **`optimised_theta`** searches $\theta$ and the smoothing parameter jointly,
  so a series with more curvature can use a larger $\theta$ than two.
- **`dstm`** is the Dynamic Standard Theta Model: the intercept and slope of
  the line are re-estimated at every observation from data seen so far, so the
  drift can follow a changing trend.
- **`dynamic_theta`** is the same state-space recursion with $\theta$ free,
  which is the Dynamic Optimised Theta Model of Fiorucci et al.

The demo checks the closed-form equivalence between the theta-line
implementation and SES-with-drift to $3.6\times10^{-15}$, shows that optimising
$\theta$ lowers both the in-sample sum of squares and the held-out error on a
curved series, and measures the small but consistent gain from a dynamic drift
when the trend changes slope.

## Variants

| variant | what moves | demo result |
|:--|:--|:--|
| `theta` | $\theta=2$, static drift | MAE 1.0994 vs 1.2859 |
| `optimised_theta` | $\theta$ searched | $\theta=4$, MAE 2.2349 vs 2.7054 |
| `dstm` | dynamic intercept and slope | one-step MSE 0.3959 vs 0.4024 |
| `dynamic_theta` | dynamic states, free $\theta$ | MAE 1.1569 vs 1.3479 |

**`theta`** fits a linear trend with noise and finds $\alpha=0.2120$ and an
initial level of $10.1867$. The one-step residual standard deviation is
$0.5345$, close to the simulated noise scale of $0.5$. On the 24 held-out
points its MAE is $1.0994$ against $1.2859$ for the naive last-value forecast.
The gain is not huge because the series is dominated by noise, but it is the
right sign: smoothing plus half a slope beats repeating the last observation.

**`optimised_theta`** faces a quadratic trend, where a fixed $\theta=2$ cannot
bend enough. The search chooses $\theta=4.00$, the top of the grid, and lowers
the in-sample sum of squares from $109.25$ to $101.51$ and the held-out MAE
from $2.7054$ to $2.2349$. The boundary solution is a signal, not a theorem:
the grid should be extended, or the trend modelled with a quadratic term, if a
larger theta keeps helping.

**`dstm`** updates the line coefficients at every step. On a trend with a slow
wave, its one-step mean squared error is $0.3959$ against $0.4024$ for the
static model, and its 24-step MAE is $2.4905$ against $2.4956$. The improvement
is real but small: a state-space drift can track a slowly changing slope, not a
full seasonal cycle, and most of the remaining error is the wave the model
cannot see.

**`dynamic_theta`** is fitted to a series whose slope jumps from $0.02$ to
$0.09$ halfway through. It chooses $\theta=4.00$ and $\alpha=0.2220$, lowers the
sum of squares from $72.13$ to $69.99$, and improves the held-out MAE from
$1.3479$ to $1.1569$. Freeing theta helps because the post-break trend is
steeper than the pre-break trend, and a fixed $\theta=2$ cannot represent that
asymmetry.

## Derivation

### The theta line

For a series $Y_t$ and a parameter $\theta\ge1$, the theta line is defined by

$$
\nabla^2Z_t(\theta)=\theta\,\nabla^2Y_t,
\qquad (1)
$$

where $\nabla^2$ is the second difference. Any two series with proportional
second differences differ by a straight line, so

$$
Z_t(\theta)=\theta Y_t+(1-\theta)(A+Bt)
\qquad (2)
$$

for some intercept $A$ and slope $B$. The natural choice is the OLS line of
$Y$:

$$
(A,B)=\arg\min_{a,b}\sum_{t=1}^{n}(Y_t-a-bt)^2,
\qquad (3)
$$

which makes $Z(0)$ exactly the long-term linear component and $Z(2)$ the
short-term component with doubled curvature. A larger $\theta$ keeps more of
the original shape; $\theta=1$ returns the series unchanged.

### Simple exponential smoothing

For a series $Z$ and smoothing parameter $\alpha\in(0,1]$, SES is

$$
\ell_t=\alpha Z_t+(1-\alpha)\ell_{t-1},
\qquad (4)
$$

with initial level $\ell_0$. The recursion is linear in $\ell_0$, a fact the
fitting code exploits. For a straight line $a+bt$, the steady-state level lags
the line by $(1-\alpha)b/\alpha$, which is exactly the bias the theta drift is
designed to remove.

### The static theta forecast

Apply (4) to the theta line and forecast it flat, and forecast the linear part
by its own line. Since $Y=(1/\theta)Z+(1-1/\theta)(A+Bt)$,

$$
\hat Y_{n+h}=\frac{1}{\theta}\ell_n^Z
+\Bigl(1-\frac1\theta\Bigr)\bigl(A+B(n+h)\bigr).
\qquad (5)
$$

For $\theta=2$ this is half the smoothed theta line plus half the OLS line. It
is the implementation's entire forecast: two terms, no iteration.

### Equivalence to SES with drift

When the initial level is placed on the line, (5) collapses to the familiar
form. For $\theta=2$,

$$
\ell_n^Z=2\ell_n^Y-\ell_n^L,\qquad
\ell_n^L=A+Bn-\frac{1-\alpha}{\alpha}B,
\qquad (6)
$$

where $\ell_n^Y$ is the SES level of the original series. Substituting into (5)
gives

$$
\hat Y_{n+h}=\ell_n^Y+\frac{B}{2}
\Bigl(h-1+\frac1\alpha\Bigr),
\qquad (7)
$$

which is the Hyndman–Billah result: the theta method is SES with drift $B/2$.
The demo checks (7) numerically and finds a maximum difference of
$3.55\times10^{-15}$. The identity is the reason the method needs no separate
trend model: the drift is just half the slope of a line through the data.

### Fitting the initial level

Both $\ell_0$ and $\alpha$ are estimated by minimising the one-step sum of
squares. For fixed $\alpha$ and $\theta$, every fitted value is affine in
$\ell_0$:

$$
\hat Y_t=\mu_t^0+u_t\ell_0,
\qquad (8)
$$

where $\mu_t^0$ is the one-step mean with $\ell_0=0$ and $u_t$ is the
coefficient of the initial level. For the static model
$u_t=(1-\alpha)^{t-1}/\theta$; for the dynamic model
$u_t=(1-\alpha)^{t-1}$. Minimising $\sum_t(Y_t-\mu_t^0-u_t\ell_0)^2$ gives the
closed form

$$
\ell_0^\star=
\frac{\sum_t u_t\,(Y_t-\mu_t^0)}{\sum_t u_t^2},
\qquad (9)
$$

so only $\alpha$ (and $\theta$, when free) needs a search. The code scans
$\alpha$ on a grid of step $0.02$ and then refines around the winner with steps
$0.01,0.005,0.002$. The resulting $\ell_0$ is exact for the chosen $\alpha$,
not a grid approximation.

### Optimised theta

Freeing $\theta$ gives the optimisation

$$
\min_{\theta\ge1,\ \alpha\in(0,1]}
\sum_{t=1}^{n}\bigl(Y_t-\hat Y_t(\theta,\alpha)\bigr)^2,
\qquad (10)
$$

with $\ell_0$ profiled out by (9). The objective is not convex in $\theta$ and
the search is a grid, not a gradient method. The implementation evaluates
`THETA_GRID`, a fixed set of eleven values from one to four, and keeps the best.
A larger grid is cheap; a finer one is not necessarily better, because the
in-sample optimum can overfit the curvature of the training window.

### The dynamic state space

The dynamic models replace the fixed line with coefficients estimated from
data up to $t$. The states are the level $\ell_t$, the mean
$\bar Y_t$, and the line coefficients $A_t,B_t$. The recursions are

$$
\ell_t=\alpha Y_t+(1-\alpha)\ell_{t-1},
\qquad (11)
$$

$$
\bar Y_t=\frac{(t-1)\bar Y_{t-1}+Y_t}{t},
\qquad (12)
$$

$$
B_t=\frac{1}{t+1}\Bigl((t-2)B_{t-1}
+\frac{6}{t}\bigl(Y_t-\bar Y_{t-1}\bigr)\Bigr),
\qquad (13)
$$

$$
A_t=\bar Y_t-\frac{t+1}{2}B_t .
\qquad (14)
$$

Equation (13) is a recursive least-squares slope: the factor $6/t$ makes
$B_t$ converge to the slope of a local line, and (14) puts the intercept at the
midpoint of the observations seen so far. The one-step mean is

$$
\mu_t=\ell_{t-1}+\Bigl(1-\frac1\theta\Bigr)
\Bigl[(1-\alpha)^{t-1}A_{t-1}
+\frac{1-(1-\alpha)^t}{\alpha}B_{t-1}\Bigr],
\qquad (15)
$$

which is (5) with the fixed line replaced by the state at $t-1$. The coefficient
of $A_{t-1}$ decays geometrically because the influence of the initial
intercept fades; the coefficient of $B_{t-1}$ tends to $1/\alpha$, the same
lag correction as in (7).

For $h\ge2$ the recursion is iterated with the forecast standing in for the
unobserved $Y$: compute $\mu_{n+h}$ from the current states, append it, update
$\ell,\bar Y,A,B$ with $\mu_{n+h}$ in place of $Y$, and repeat. The dynamic
forecast is therefore a deterministic path, not a closed form.

The state space is initialised at $A_0=B_0=\bar Y_0=0$, which makes $B_1=3Y_1$
and produces large one-step errors at the start. The implementation excludes
the first $\max(3,\lceil2/\alpha\rceil)$ residuals from the sum of squares, so
the fit is not dominated by the transient. The forecasts still use the states
after the transient, so only the objective is trimmed.

### Prediction intervals

Fiorucci et al. give the conditional variance

$$
\mathrm{Var}(Y_{n+h}\mid Y_1,\dots,Y_n)
=\bigl[1+(h-1)\alpha^2\bigr]\sigma^2,
\qquad (16)
$$

where $\sigma^2$ is the one-step residual variance. The interval is Gaussian:
$\hat Y_{n+h}\pm z_{1-\gamma/2}\sqrt{1+(h-1)\alpha^2}\,\sigma$. It ignores
parameter uncertainty and the correlation between successive forecast errors,
so it is a lower bound on the true predictive width, not a calibrated interval.

## Code walkthrough

The OLS line is computed once for the static variants. Centring the time grid
avoids forming a design matrix and makes the slope a single dot product:

```python
    n = y.size
    grid = np.arange(1, n + 1, dtype=float)
    centred = grid - grid.mean()
    slope = float(centred @ (y - y.mean()) / (centred @ centred))
    intercept = float(y.mean() - slope * grid.mean())
    return intercept, slope
```

The $1,\dots,n$ indexing matches the state-space equations of Fiorucci et al.,
where $\bar Y_t$ is the mean of the first $t$ observations. Using $0,\dots,n-1$
would shift every intercept by one slope and silently change the forecast.

The static engine is (2), (4) and (5) in one loop. The `coefficient` array holds
$u_t$ so that (9) can be applied after the pass, when the fitted values with
$\ell_0=0$ are known:

```python
    mean = np.empty(n)
    coefficient = np.empty(n)
    level = 0.0
    for index in range(n):
        mean[index] = level / theta + weight * line[index]
        coefficient[index] = powers[index] / theta
        level = alpha * z[index] + (1.0 - alpha) * level
    if ell0 is None:
        ell0 = float(np.sum(coefficient * (y - mean))
                     / np.sum(coefficient * coefficient))
```

The alternative would be to run SES twice, once with $\ell_0=0$ and once with
$\ell_0=1$, and subtract. Carrying `coefficient` is cheaper and makes the
affine dependence explicit.

The dynamic engine adds the state updates (11)–(14). The order matters: `B` uses
the mean *before* the current observation is folded in, so `average_previous` is
saved first:

```python
    for index in range(n):
        t = index + 1
        drift = weight * (
            powers[index] * intercept_state
            + (1.0 - powers[index] * (1.0 - alpha)) / alpha * slope_state)
        mean[index] = level + drift
        coefficient[index] = powers[index]
        level = alpha * y[index] + (1.0 - alpha) * level
        average_previous = average
        average = ((t - 1) * average_previous + y[index]) / t
        slope_state = ((t - 2) * slope_state
                       + 6.0 / t * (y[index] - average_previous)) / (t + 1)
        intercept_state = average - 0.5 * (t + 1) * slope_state
```

`powers[index]` is $(1-\alpha)^{\text{index}}=(1-\alpha)^{t-1}$, and the factor
`(1.0 - powers[index] * (1.0 - alpha)) / alpha` is
$(1-(1-\alpha)^t)/\alpha$. Both appear directly in (15), so the loop is a
transcription rather than a derivation.

The alpha search is a coarse scan followed by local refinement. Refining after
the scan matters because the sum of squares is flat near the optimum and a
$0.02$ grid can miss the best value by more than the improvement being measured:

```python
    best = None
    for alpha in np.arange(0.02, 0.99, 0.02):
        result = _static(y, theta, float(alpha), ell0)
        if best is None or result["sse"] < best["sse"]:
            best = result
    assert best is not None
    centre = best["alpha"]
    for step in (0.01, 0.005, 0.002):
        for alpha in (centre - step, centre + step):
            if 0.0 < alpha <= 1.0:
                result = _static(y, theta, float(alpha), ell0)
                if result["sse"] < best["sse"]:
                    best = result
        centre = best["alpha"]
```

The search is nested inside the theta loop, so `optimised_theta` performs eleven
alpha fits and `dynamic_theta` seven, which is why those variants cost more than
the fixed-theta ones. Each alpha fit is a single $O(n)$ pass, so the total is
still small.

The variant grid is chosen once in `fit`. The two standard models fix
$\theta=2$; the two optimised models scan their grids; an explicit `theta` in
the constructor overrides both:

```python
        if self.variant in ("theta", "dstm"):
            grid = (2.0,)
        elif self.variant == "optimised_theta":
            grid = THETA_GRID
        else:
            grid = DYNAMIC_GRID
        if self.theta is not None:
            grid = (self.theta,)
```

The distinction between `THETA_GRID` and `DYNAMIC_GRID` is deliberate: the
dynamic state space is more expensive per evaluation, so its grid is coarser.
Both grids include $\theta=1$ and $\theta=2$, which means the optimised models
can always fall back to SES or to the standard theta forecast.

Prediction for the static models is a vectorised evaluation of (5). The line is
extrapolated at the absolute future times $n+1,\dots,n+h$, not at
$1,\dots,h$:

```python
        if self.variant in ("theta", "optimised_theta"):
            intercept, slope = self.line_
            weight = 1.0 - 1.0 / self.theta_
            steps = np.arange(1, horizon + 1)
            line = intercept + slope * (self.n_ + steps)
            return self.level_ / self.theta_ + weight * line
```

Using `steps` alone is the classic off-by-one-slope bug: the forecast would be
shifted down by $nB$, which on a series of length 240 with slope $0.05$ is a
constant error of 12. The demo's equivalence check catches exactly this.

The dynamic prediction repeats the state recursion into the future. Each
forecast is appended before the state update, so the update sees the forecast it
just produced:

```python
        for step in range(1, horizon + 1):
            t = self.n_ + step
            decay = (1.0 - self.alpha_) ** (t - 1)
            drift = weight * (
                decay * intercept
                + (1.0 - (1.0 - self.alpha_) ** t) / self.alpha_ * slope)
            mean = level + drift
            out[step - 1] = mean
            level = self.alpha_ * mean + (1.0 - self.alpha_) * level
            average_previous = average
            average = ((t - 1) * average_previous + mean) / t
            slope = ((t - 2) * slope
                     + 6.0 / t * (mean - average_previous)) / (t + 1)
            intercept = average - 0.5 * (t + 1) * slope
```

At `step=1`, `t = n+1` and `decay` is $(1-\alpha)^n$, the same coefficient as in
(15). At later steps the state has been updated with forecasts, so the drift
evolves even though no new data arrive.

The interval is the variance formula (16) evaluated on the forecast path:

```python
        forecast = self.predict(horizon)
        steps = np.arange(horizon)
        scale = np.sqrt(1.0 + steps * self.alpha_ ** 2) * self.sigma_
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        return forecast - z * scale, forecast + z * scale
```

The width grows like $\sqrt{1+(h-1)\alpha^2}$, which is much slower than the
$\sqrt{h}$ of a random walk. That is appropriate for a model whose forecast is
a smooth extrapolation, and misleading if the residuals are autocorrelated;
check the residual autocorrelation before trusting the band.

## Complexity

With $n$ observations, $G$ values of $\theta$, $A$ values of $\alpha$ and $H$
forecast steps:

- **One static pass:** $O(n)$ time and $O(n)$ memory for the arrays.
- **One dynamic pass:** the same, with a slightly larger constant.
- **Alpha search:** about $50+6$ passes per theta, so $O(An)$.
- **Full fit:** $O(GAn)$; with $G=11$ and $A\approx56$ this is a few hundred
  passes, independent of the forecast horizon.
- **Prediction:** $O(H)$ for the static models and $O(H)$ for the dynamic
  recursion, since the state updates are scalar.
- **Interval:** $O(H)$ plus one normal quantile.

The demo's four fits on 264 points take about a quarter of a second in pure
Python. The dominant cost is the nested alpha search, not the state recursion;
if the grid were replaced by a golden-section search the fit would be roughly
ten times faster, at the cost of a unimodality assumption that the sum of
squares does not always satisfy.

## Failure modes

**Theta is not a seasonal model.** It has no seasonal component, so applying it
directly to monthly data with a strong calendar pattern produces a forecast that
ignores the pattern entirely. The standard practice is to deseasonalise first
with STL or seasonal means, forecast the adjusted series, and add the seasonal
component back. The demo uses non-seasonal series for that reason.

**The drift is half a global slope.** Equation (7) uses the OLS slope of the
whole record. If the trend changed direction, the global slope is an average of
two regimes and the forecast is wrong in both. The dynamic variants address
this, but only partially: their slope state is a local linear fit, not a
breakpoint detector. A structural-break test, or a model with a time-varying
trend, is the right tool when the break is the phenomenon of interest.

**A boundary theta is a warning.** `optimised_theta` choosing $\theta$ at the
top of the grid means the objective still wants more curvature. Extending the
grid is legitimate, but a very large $\theta$ makes the theta line close to the
original series and the forecast close to SES, which may be worse out of sample
even when the in-sample sum of squares improves. Check the held-out error, as
the demo does, before accepting a boundary solution.

**The dynamic model has a violent transient.** The state-space initialisation
sets $A_0=B_0=0$, so $B_1=3Y_1$ and the first few one-step forecasts are far
from the data. The implementation trims those residuals from the objective, but
the *states* still carry the transient for a while. If the series starts near
zero, or has a huge first observation, the early forecasts are unusable. Fit on
a longer prefix, or initialise the states from a preliminary OLS line.

**The interval is too narrow.** Equation (16) accounts for the smoothing
recursion but not for parameter uncertainty, model misspecification, or
autocorrelated residuals. On a series with a cycle the residuals are strongly
autocorrelated, and the true predictive interval is much wider than the printed
one. Use the interval as a relative width, not as a coverage guarantee.

**Theta and SES share a blind spot.** Both forecast a level, and both lag a
trend by roughly $(1-\alpha)/\alpha$ times the slope. The theta drift corrects
the average lag, but it cannot correct a lag that changes over time. If the
residuals of the fitted model still show a trend, the drift is mis-specified;
plot the residuals against time before trusting the forecast.

**Alpha near one.** A fitted $\alpha$ close to one means the model is almost
the naive forecast; the drift term then dominates, and the method is effectively
a random walk with a fixed slope. That is sometimes right, but it is worth
checking whether the series is better described as a random walk with no drift
at all.

## Applications

Theta is a workhorse in the M-competitions and in production forecasting stacks
for retail and demand planning, where its accuracy is close to much more
elaborate models at a fraction of the cost. The `forecTheta` package and the
`statsforecast` implementations are widely used, and the method is often the
first non-naive benchmark in a forecasting comparison.

In energy and utilities it is applied to seasonally adjusted load, where the
trend is slow and the seasonal pattern is removed first. In finance it is used
on log prices and on realised volatility, both of which have a strong local
trend and no stable seasonal component. In inventory control it forecasts the
demand rate for slow-moving items, usually after intermittent-demand handling
such as Croston's method.

It is the wrong tool when seasonality is the signal, when the trend has a
turning point inside the forecast horizon, or when the series is intermittent.
It is also not a probabilistic model: the interval in (16) is a rough Gaussian
band around a deterministic path, not a likelihood. For those cases use a
state-space ETS model, a seasonal ARIMA, or a count model, and keep theta as the
baseline that they must beat.

## References

- Assimakopoulos, V., Nikolopoulos, K. (2000). *The theta model: a decomposition
  approach to forecasting.* International Journal of Forecasting 16(4),
  521–530. The original theta line and the two-line combination.
- Hyndman, R. J., Billah, B. (2003). *Unmasking the Theta method.*
  International Journal of Forecasting 19(2), 287–290. The equivalence to SES
  with drift in (7).
- Fiorucci, J. A., Pellegrini, T. R., Louzada, F., Petropoulos, F., Koehler,
  A. B. (2016). *Models for optimising the theta method and their relationship
  to state space models.* International Journal of Forecasting 32(4),
  1151–1161. Optimised theta, the dynamic state space (11)–(15), and the
  variance formula (16).
- Makridakis, S., Hibon, M. (2000). *The M3-Competition: results, conclusions
  and implications.* International Journal of Forecasting 16(4), 451–476. The
  competition that made theta's reputation.
- Hyndman, R. J., Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed., ch. 8. Exponential smoothing and the drift formulation
  used as the baseline.
