# ARIMA — past values, past errors, differences and seasons

`arima.py` · Time series · Tier 1 · 7 variants

## Intuition

ARIMA is the statement that a series is explainable by its own history. Three
mechanisms do the explaining, and the order triple $(p,d,q)$ says how much of
each:

- **AR($p$)** — regress on the last $p$ values. A series that tends to continue
  is captured by positive coefficients; one that oscillates by alternating
  signs.
- **I($d$)** — difference $d$ times before modelling. A random walk is not
  stationary, but its first difference is white noise; a straight line becomes
  a constant.
- **MA($q$)** — regress on the last $q$ forecast errors. This is how the model
  represents shocks that persist for a few steps and then fade.

The seasonal versions multiply each polynomial by a copy in $B^m$, which adds
lags at $m,2m,\dots$ instead of shifting the whole model by a season. Exogenous
regressors add a linear part, turning the model into a regression with
autocorrelated errors.

Estimation is where ARIMA gets its reputation for being fiddly, and the reason
is that the errors in the MA part are unobserved: they are defined recursively
from the data and the parameters. The trick used here is **conditional sum of
squares**: assume the first few errors are zero, then the likelihood is a sum of
squares, and for fixed ARMA coefficients the regression part is a linear
least-squares problem that can be solved exactly. Only the short ARMA
coefficient vector is searched. In the AR-only case there is nothing to search
at all: the whole fit is one `lstsq`, and the demo checks it against OLS to
machine precision.

The second idea that makes the family coherent is that differencing is a linear
filter. Forecasts are produced on the differenced scale and then integrated back
up; forecast variances are the squared **psi-weights** of the integrated model.
That single mechanism handles a random walk (whose band grows like $\sqrt h$), a
seasonal random walk (whose forecast repeats the last cycle exactly) and a
regression with AR errors, and the demo validates it against 3 000 simulated
paths to 1.4%.

## Variants

| variant | model | estimation | demo result |
|:--|:--|:--|:--|
| `ar` | AR($p$), no difference | one least squares | $\phi=(0.5341,-0.3085)$ |
| `ma` | MA($q$) | CSS search | $\theta=0.6094$ |
| `arma` | ARMA($p,q$) | CSS search | — |
| `arima` | ARIMA($p,d,q$) | CSS on $\Delta^d y$ | random walk exact |
| `sarima` | seasonal ARIMA | polynomials in $B^m$ | ACF 0.824 to $-0.158$ |
| `arimax` | regression + ARMA errors | profiled least squares | $\beta=1.5854$ |
| `auto_arima` | order search | KPSS then AICc | $(2,0,0)$ |

**`ar`** is the model that is just a regression on lags. It is the only variant
with a closed-form fit, and it is the right baseline for any autocorrelated
series: if an AR(1) explains the data, nothing more complicated is justified.

**`ma`** models the error process. For an MA(1) the autocovariance gives a
method-of-moments estimate from $\rho_1=\theta/(1+\theta^2)$, which is the
demo's cross-check: on 400 points the CSS estimate is 0.6094 against a true 0.6,
and the moment estimator gives 0.6563 from $\rho_1=0.4587$. The two agree in
magnitude but not exactly, which is the usual story: moments are consistent but
inefficient, least squares is efficient but needs a search.

**`arma`** combines them. It is the workhorse for stationary series with both
persistence and short-lived shocks, and it is where the CSS search earns its
keep.

**`arima`** adds differencing. With $p=q=0$ it is a random walk or a random walk
with drift, and with $p,q>0$ it is the Box-Jenkins model proper. The demo fits
$(0,1,0)$ without a mean and gets exactly the last value, with a band whose
width grows as $\sqrt h$: 4.148 at $h=1$ and 9.274 at $h=5$.

**`sarima`** multiplies the AR and MA polynomials by seasonal copies. On the
demo's period-12 series with AR noise, one seasonal AR term and one seasonal
difference cut the lag-12 autocorrelation from 0.824 to $-0.158$. A seasonal
random walk is exact: the forecast is the last observed cycle, because that is
what $(1-B^{12})y_t=e_t$ says.

**`arimax`** adds regressors. The demo's $\beta$ is 1.5854 against a true 1.5 —
a reminder that 300 points of AR(1) noise with $\phi=0.6$ is an effective sample
of about 75 — but the exogenous model's held-out MAE is 1.0902 against 1.5880
for an AR model that cannot see $x$.

**`auto_arima`** chooses $d$ with a KPSS test and then $p,q$ by AICc over nine
candidates. On a stationary AR(2) draw it returns $(2,0,0)$ with AICc 1118.81,
followed by $(1,0,2)$ at 1119.04 and $(2,0,1)$ at 1120.23; on a random walk it
differences once. Order selection on a near-tie is not a decision worth
defending, which is why the demo asserts only the minimum-AICc property.

## Derivation

### The model

Let $y_t$ be the observed series and

$$
w_t=\Delta^d\Delta_m^D\,y_t
\qquad (1)
$$

its regular and seasonal differences. On the differenced scale the model is

$$
A(B)\,(w_t-\mu-x_t^\top\beta)=B(B)\,e_t,\qquad e_t\sim\mathcal N(0,\sigma^2),
\qquad (2)
$$

where $A$ and $B$ are polynomials in the backshift $B$,

$$
A(B)=\phi(B)\Phi(B^m),\qquad B(B)=\theta(B)\Theta(B^m),
\qquad (3)
$$

$$
\phi(B)=1-\sum_{i=1}^{p}\phi_iB^i,\quad
\Phi(B^m)=1-\sum_{k=1}^{P}\Phi_kB^{km},
\qquad (4)
$$

and similarly $\theta$ and $\Theta$ with plus signs. Expanding the product in
(3) produces cross terms: for $p=P=1$ the AR polynomial is
$1-\phi B-\Phi B^m+\phi\Phi B^{m+1}$, so the recursion needs lag $m+1$ as well
as lags $1$ and $m$. That expansion is why `_poly` exists and why the number of
AR coefficients in the recursion is not $p+P$.

### Conditional sum of squares

Write $u_t=w_t-\mu-x_t^\top\beta$ and expand (2) into a recursion:

$$
u_t=\sum_{l\ge1}a_l\,u_{t-l}+\sum_{l\ge1}b_l\,e_{t-l}+e_t,
\qquad (5)
$$

where $a$ and $b$ are the expanded coefficients of $-A$ and $B$ excluding the
leading one. Conditioning on $e_1=\dots=e_L=0$, with $L$ the largest lag,
gives residuals for $t>L$ and the objective

$$
\mathrm{SSE}(\phi,\Phi,\theta,\Theta,\mu,\beta)
=\sum_{t=L+1}^{n}e_t^2 .
\qquad (6)
$$

The important structural fact is that (5) is **linear in $u$**, so the map from
$u$ to $e$ is a linear filter. Therefore

$$
e(w-Z\gamma)=e(w)-\sum_k\gamma_k\,e(Z_k),
\qquad (7)
$$

and for fixed ARMA coefficients the minimisation over $\gamma=(\mu,\beta)$ is an
ordinary least-squares problem on the filtered data:

$$
\hat\gamma=\arg\min_\gamma\bigl\lVert e(w)-e(Z)\gamma\bigr\rVert^2 .
\qquad (8)
$$

That is the whole estimation strategy: search the short ARMA vector, solve (8)
in closed form at every candidate, and keep the best. It reduces a
$(p+P+q+Q+\text{regressors})$-dimensional problem to a
$(p+P+q+Q)$-dimensional one and makes `arimax` no harder than `arma`.

### The AR case is OLS

When $q=Q=0$ the filter is pure AR, and (5) becomes

$$
w_t=\sum_{l\ge1}a_l\,w_{t-l}+c+e_t,\qquad
c=\mu\Bigl(1-\sum_l a_l\Bigr)
\qquad (9)
$$

for a model with a constant. That is exactly the regression of $w_t$ on its own
lags and an intercept, so one `lstsq` gives both the AR coefficients and $c$;
the mean is recovered as $\mu=c/(1-\sum a_l)$. The demo checks all three
against an independent OLS: the coefficients, the residual sum of squares, and
the implied constant. On the AR(2) draw the implied constant is $-0.017038$,
matching the OLS intercept to every printed digit.

### Forecasting

With future errors set to zero, (5) gives the forecast recursion

$$
\hat u_{T+h}=\sum_{l\ge1}a_l\,\hat u_{T+h-l}
+\sum_{l\ge1}b_l\,\hat e_{T+h-l},
\qquad (10)
$$

where $\hat u$ and $\hat e$ are observed for indices up to $T$ and recursively
defined afterwards. The forecast on the differenced scale is
$\hat w_{T+h}=\hat u_{T+h}+\mu+x_{T+h}^\top\beta$, and the forecasts are then
integrated back through every differencing stage:

$$
\hat y_{T+h}=\hat w_{T+h}+\hat y_{T+h-1}
\quad\text{or}\quad
\hat y_{T+h}=\hat w_{T+h}+\hat y_{T+h-m},
\qquad (11)
$$

depending on whether the stage was a regular or a seasonal difference. The code
keeps all intermediate stages in `stages_` for exactly this reason; undoing them
in reverse order needs the parent series at each level.

### Psi-weights and the forecast variance

The psi-weights are the coefficients of the infinite MA representation of the
integrated model. On the differenced scale they satisfy

$$
\psi_0=1,\qquad
\psi_j=\sum_{l\ge1}a_l\,\psi_{j-l}+b_j .
\qquad (12)
$$

Differencing multiplies the MA polynomial by $(1-B)^{-d}(1-B^m)^{-D}$, so the
psi-weights of the observed series are obtained by cumulative summation $d$
times and by summing every $m$-th term $D$ times:

$$
\psi^{(d)}_j=\sum_{l=0}^{j}\psi^{(d-1)}_l,\qquad
\psi^{(D)}_j=\sum_{k\ge0}\psi^{(D-1)}_{j-km}.
\qquad (13)
$$

The $h$-step forecast error is $\sum_{j=0}^{h-1}\psi_j e_{T+h-j}$, hence

$$
\mathrm{Var}(\hat y_{T+h})=\sigma^2\sum_{j=0}^{h-1}\psi_j^2 .
\qquad (14)
$$

For a random walk, (12) gives $\psi_j=0$ for $j\ge1$ and (13) turns that into
$\psi_j=1$ for all $j$, so (14) is $h\sigma^2$ and the band widens as
$\sqrt h$ — the demo's 4.148 to 9.274 over five steps. For the fitted
ARIMA(1,1,1) the weights are $1,1.877,2.434,2.788$, and the 3 000 simulated
paths match the analytic standard deviations to 1.3%.

### Order selection

AICc compares models on the same differenced scale:

$$
\mathrm{AICc}=-2\log L+2k+\frac{2k(k+1)}{n-k-1},
\qquad (15)
$$

with $k$ counting ARMA coefficients, regression coefficients, and $\sigma^2$.
Comparing across different $d$ is a mistake — each difference removes an
observation and changes the scale — so `auto_arima` fixes $d$ first with the
KPSS test:

$$
\eta=\frac{1}{n^2}\sum_{t=1}^{n}S_t^2\Big/\hat\lambda^2,\qquad
S_t=\sum_{i=1}^{t}\hat e_i,
\qquad (16)
$$

where $\hat e$ are the residuals from a regression on a constant and
$\hat\lambda^2$ is a Newey-West long-run variance. The null is level
stationarity, so a value above 0.463 rejects at 5% and the series is differenced
once more, up to twice. On the AR(2) draw the statistic is 0.232; on a random
walk it is 1.44.

## Code walkthrough

The filter is the heart of the file: one loop, two dot products, and the lag
arrays already expanded.

```python
def _filter(series: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residuals of `u_t = sum a_i u_{t-i} + e_t + sum b_j e_{t-j}`."""
    n = series.size
    out = np.zeros(n)
    la, lb = a.size, b.size
    start = max(la, lb)
    for t in range(start, n):
        value = float(series[t])
        if la:
            value -= float(a @ series[t - la:t][::-1])
        if lb:
            value -= float(b @ out[t - lb:t][::-1])
        out[t] = value
    return out
```

`series[t - la:t][::-1]` is the window $u_{t-1},\dots,u_{t-la}$ in lag order, so
the dot product pairs $a_1$ with $u_{t-1}$ without an index reversal elsewhere.
The output is written in place, which is what lets the MA part read its own past
residuals.

The seasonal polynomials are built by convolution, not by hand:

```python
def _poly(regular, seasonal, m: int, sign: float) -> np.ndarray:
    """Dense coefficients of `(1 + sign*r(B))(1 + sign*S(B^m))`."""
    first = np.zeros(len(regular) + 1)
    first[0] = 1.0
    first[1:] = sign * np.asarray(regular, dtype=float)
    second = np.zeros(len(seasonal) * m + 1)
    second[0] = 1.0
    second[m::m] = sign * np.asarray(seasonal, dtype=float)
    return np.convolve(first, second)
```

The slice assignment `second[m::m]` puts the seasonal coefficients at $m,2m,
\dots$ and `np.convolve` produces the cross terms of (3) automatically. The sign
argument is what distinguishes the AR polynomial from the MA one; in the caller
the AR result is negated once more, because (5) moves the AR terms to the
left-hand side.

The AR fast path is a single least-squares call, with the mean recovered
afterwards:

```python
        matrix = np.column_stack(blocks)
        coef, *_ = np.linalg.lstsq(matrix, w[rows], rcond=None)
        a = coef[:length]
        constant = coef[length:]
        if design.shape[1] and self.include_mean:
            total = float(a.sum())
            mean = (constant[0] / (1.0 - total)
                    if abs(1.0 - total) > 1e-9 else 0.0)
            gamma = np.concatenate([[mean], constant[1:]])
```

The guard on $1-\sum a_l$ matters: a sum of AR coefficients exactly one is a
unit root, and dividing by zero there would produce an infinite mean. It cannot
happen for stationary data, but it can for a differenced series that is still
non-stationary.

The search path profiles the regression coefficients out of the objective, so
the pattern search only ever moves the ARMA vector:

```python
        def profile(values):
            a, b = split(values)
            start = max(a.size, b.size)
            filtered = _filter(w, a, b)
            if design.shape[1]:
                blocks = np.column_stack([_filter(design[:, j], a, b)
                                          for j in range(design.shape[1])])
                coef, *_ = np.linalg.lstsq(blocks[start:], filtered[start:],
                                           rcond=None)
                residuals = filtered[start:] - blocks[start:] @ coef
            else:
                coef = np.zeros(0)
                residuals = filtered[start:]
            return float(residuals @ residuals), a, b, coef
```

This is (8) written out: filter the response and every regressor with the same
filter, then regress one on the others. For `arimax` that means the exog column
is filtered alongside $w$, which is exactly the GLS transformation that makes
the regression coefficients consistent in the presence of AR errors.

Forecasting keeps the observed residuals and extends both arrays by the
horizon, so the same recursion serves the observed and future parts:

```python
        for j in range(horizon):
            value = 0.0
            for i, coef in enumerate(self.a_, 1):
                value += coef * extended[n + j - i]
            for k, coef in enumerate(self.b_, 1):
                value += coef * innovations[n + j - k]
            extended[n + j] = value
        differenced = extended[n:] + future @ gamma
        return self._integrate(differenced)
```

`innovations` beyond $T$ stays zero, which is the conditional expectation of a
future shock; `extended` beyond $T$ is filled by the recursion itself. Note that
the future design is multiplied by $\gamma$ *after* the recursion, because the
recursion is on $u=w-Z\gamma$, not on $w$.

Undoing the differences walks the stages backwards:

```python
        for _ in range(self.D_):
            parent = self.stages_[index - 1]
            n = parent.size
            extended = np.concatenate([parent, np.zeros(path.size)])
            for j in range(path.size):
                extended[n + j] = path[j] + extended[n + j - self.season_]
            path = extended[n:]
            index -= 1
```

Appending zeros to the parent and filling forward is the inverse of
differencing in four lines. The seasonal loop runs before the regular one
because the stages were built in the other order; getting that order wrong
produces a forecast that looks plausible but is integrated from the wrong
level.

The psi-weights are the same recursion as the forecast, with an indicator
instead of an innovation:

```python
        for j in range(1, horizon):
            value = float(self.b_[j - 1]) if j - 1 < self.b_.size else 0.0
            for i, coef in enumerate(self.a_, 1):
                if j - i >= 0:
                    value += coef * weights[j - i]
            weights[j] = value
```

This is (12). The `if j - i >= 0` guard is the difference between computing
$\psi_j$ and indexing before the start of the array; for $j<i$ those terms are
absent because $\psi$ is zero at negative lags.

The order search is two decisions, and the first is a test rather than a
criterion:

```python
def _choose_d(y: np.ndarray, limit: int = 2) -> int:
    """Smallest number of differences that passes the KPSS test."""
    current = y
    for d in range(limit + 1):
        if current.size < 20:
            break
        if _kpss(current) <= 0.463:
            return d
        current = np.diff(current)
    return min(limit, y.size // 20)
```

Differences are added until the statistic falls below the 5% critical value,
with a hard cap at two and a size guard so a short series is not differenced
into nothing. The AICc loop then only has to choose $p$ and $q$ at that fixed
$d$.

## Complexity

With $n$ observations, $L$ the largest expanded lag, $p+P+q+Q$ the number of
searched coefficients and $h$ the horizon:

- **One filter:** $O(nL)$.
- **One AR fit:** one `lstsq` on an $n\times L$ matrix, $O(nL^2)$.
- **One search:** about eight step halvings times two directions times the
  number of coefficients, each evaluation costing one filter per design column.
- **`auto_arima`:** nine fits at the chosen $d$, plus the KPSS passes.
- **`predict`:** $O(hL)$; **`interval`:** adds $O(hL)$ for the psi-weights.
- **Memory:** $O(n)$ for the series and its stages, $O(L)$ for the coefficients.

The demo's largest job is the 3 000-path simulation of an ARIMA(1,1,1), which
runs the forecast recursion 36 000 times — still under a second, because the AR
and MA vectors have one element each.

## Failure modes

**Over-differencing.** Each difference costs an observation, inflates the
variance and pushes the MA polynomial towards a unit root at $-1$, where the
CSS surface is flat. Worse, AICc computed on the differenced series is not
comparable across $d$: in a control run outside the demo, an AICc search over
$d$ chose $(2,1,2)$ for a stationary AR(1) and beat the correct $(1,0,0)$ by
more than 20 points. That is why `auto_arima` fixes $d$ with KPSS instead of
including it in the criterion.

**Roots on the unit circle.** The search bounds every ARMA coefficient at
$\pm0.98$. A true MA root at $-1$ (which is what a first difference of a
stationary series produces) cannot be represented, so the fit parks a
coefficient near the boundary and compensates with extra AR terms. If a fitted
MA coefficient sits at the bound, the model is trying to difference again;
inspect it rather than trusting the forecast.

**Spurious regression in `arimax`.** If $y$ and $x$ are both trending and the
errors are strongly autocorrelated, the profiled $\beta$ can be large and
significant with no causal content. The demo's $\beta$ is 1.5854 against a true
1.5, which is honest sampling error at an effective sample of 75 — but it is
also the size of error that a naive t-test would call a discovery. Difference or
detrend both series before reading anything into $\beta$.

**Seasonal lags eat the sample.** With $m=12$ and $P=Q=1$ the largest lag is 12,
so the first 12 residuals are discarded before estimation begins. On 60 points
that is 20% of the data, and on two cycles it is most of the information. Require
at least three cycles for a seasonal model, and prefer $P=Q=1$ to larger orders.

**A forecast needs a forecast of $x$.** `arimax` requires future exog values,
and the interval conditional on them ignores their uncertainty. A promotion
forecast built on an assumed price path is only as good as that path; the
ARIMA band does not include the risk that the assumption is wrong.

## Applications

ARIMA is the default model in macroeconomics and official statistics: monthly
inflation, industrial production and GDP are still routinely forecast with
SARIMA, and the Box-Jenkins workflow — identify, estimate, diagnose, forecast —
is taught as the entry point to time-series modelling. Its appeal is that the
components mean something: the AR coefficients describe persistence, the MA
coefficients describe shock decay, and the differencing describes what kind of
non-stationarity the series has.

`auto_arima` is the standard baseline on Kaggle-style forecasting problems and in
production demand pipelines, precisely because it is cheap and hard to beat on
short, univariate series. `arimax` is the standard way to adjust a forecast for
a known driver — price, weather, a holiday dummy — while keeping the error
structure honest.

It is a trap when the series has a structural break, when the seasonality is
multiple and evolving (use a state-space or neural model), or when the driver is
itself forecast. It is also a trap at long horizons: the psi-weights of an
integrated model grow without bound, so a 24-step band on a random walk is
almost useless, and that is a property of the model, not a bug.

## References

- Box, Jenkins (1970). *Time Series Analysis: Forecasting and Control.*
  Holden-Day. The original ARIMA workflow and the seasonal multiplicative form.
- Hyndman, Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd
  ed., ch. 9. The applied treatment of ARIMA, ACF/PACF identification and
  `auto.arima`.
- Kwiatkowski, Phillips, Schmidt, Shin (1992). *Testing the null hypothesis of
  stationarity against the alternative of a unit root.* Journal of
  Econometrics 54. The KPSS statistic of (16) and its critical values.
- Burnham, Anderson (2002). *Model Selection and Multimodel Inference*, 2nd ed.
  Springer. The AICc correction and its small-sample motivation.
- Ansley (1979). *An algorithm for the exact likelihood of a mixed
  autoregressive-moving average process.* Biometrika 66(1). What the
  conditional sum of squares here approximates, and why the first $L$ residuals
  are the price of the approximation.
- Hannan, Rissanen (1982). *Recursive estimation of mixed autoregressive-moving
  average order.* Biometrika 69(1). The two-stage initialisation that a
  production implementation uses before the CSS refinement.
