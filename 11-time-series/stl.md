# STL — seasonal-trend decomposition by smoothing, not by a model

`stl.py` · Time series · Tier 2 · 4 variants

## Intuition

STL is the answer to a practical question: given a series that clearly has a
trend and a repeating shape, how do you split it into those two parts without
assuming a parametric model for either? Cleveland's method is iterative
backfitting. Guess the trend, subtract it, smooth each seasonal subseries,
remove the low-frequency part of that seasonal estimate, subtract the result,
and smooth the remainder to update the trend. Repeat.

Two details make it work. The first is **cycle-subseries smoothing**: with
period $m$, split the data into $m$ sequences (all Januaries, all Februaries,
...). Each sequence is short and smooth, so a local regression with a modest
window captures how that month's effect evolves over the years. The second is
the **low-pass filter**: the raw seasonal estimate also contains slow drift,
which must be removed before the seasonal component is subtracted from the
series. STL does that with three moving averages followed by a local regression,
and the result is the part of the seasonal estimate that is genuinely periodic.

The variants are the practical extensions of that skeleton:

- **`robust`** adds an outer loop that downweights outliers, so a handful of
  spikes cannot bend the trend or inflate a month's seasonal effect.
- **`mstl`** repeats the decomposition for several periods, subtracting the
  other seasonal components before estimating each one.
- **`seasonal_windows`** chooses a separate seasonal smoothing window for each
  subseries by leave-one-out cross-validation, instead of forcing all months to
  share one window.

The demo checks the split against the simulation truth, not just the additive
identity: the identity holds by construction, while the correlation and RMSE
figures are what would look wrong if the smoothing were broken.

## Variants

| variant | change | demo result |
|:--|:--|:--|
| `stl` | base iterative decomposition | profile $r=0.9993$ |
| `robust` | bisquare outer weights | outlier weight 0.0000 |
| `mstl` | several periods | weekly $r=0.9999$ |
| `seasonal_windows` | per-subseries span by CV | RMSE 0.4738 vs 0.5024 |

**`stl`** is the default. On 240 points with a linear trend, two harmonics and
noise with variance $0.1225$, it recovers the seasonal profile with correlation
$0.9993$ and the trend with RMSE $0.1189$. The remainder variance is $0.0665$,
below the noise variance because the smoothers absorb some of the noise — a
reminder that a low remainder variance is not by itself evidence of a good
decomposition.

**`robust`** fits the same series after twenty observations are shifted by
$+15$. The final weights at those observations are exactly zero, the trend RMSE
falls from $1.5137$ to $0.1719$, and the seasonal RMSE falls as well. The
remainder sum of squares is *larger* ($4670.79$ against $2491.88$), because the
outliers stay in the remainder; robustness protects the components, it does not
remove the data.

**`mstl`** handles a weekly and a monthly cycle on 420 daily points. The two
extracted components correlate $0.9999$ and $0.9995$ with the truth, and the
remainder variance is $0.0213$ against a noise variance of $0.0625$. It works by
estimating each period on the series with the other seasonal components
subtracted, then repeating the pass once.

**`seasonal_windows`** is for the case where one part of the year is much noisier
than another. The demo makes month 3 four times as noisy as the others; the
variant chooses windows $(11,11,11,11,5,11,9,11,9,5,11,11)$ and lowers the
seasonal RMSE from $0.5024$ to $0.4738$. Its in-sample RSS is slightly higher
($138.84$ against $132.39$) because a smoother seasonal fit is not the same as a
lower residual; the cross-validated window is choosing bias over variance.

## Derivation

### The decomposition

All variants target the additive identity

$$
y_t=T_t+S_t+R_t,
\qquad (1)
$$

where $T$ is the trend, $S$ is seasonal with period $m$, and $R$ is what is
left. The identity is definitional: once $T$ and $S$ are chosen, $R$ is
determined. The modelling content is entirely in the two smoothers.

### Local regression

Both smoothers are LOESS. For a target $z$, fit a local polynomial of degree
$d$ by weighted least squares:

$$
\min_{\beta_0,\beta_1}\sum_{i=1}^{n}w_i(z)
\bigl(y_i-\beta_0-\beta_1(x_i-z)\bigr)^2,
\qquad (2)
$$

and evaluate at $z$ as $\hat\beta_0$. The weights are tricube in the distance to
the $k$-th nearest point:

$$
w_i(z)=\left(1-\left(\frac{\lvert x_i-z\rvert}{h(z)}\right)^3\right)^3
\mathbf 1[\lvert x_i-z\rvert<h(z)],
\qquad (3)
$$

where $h(z)$ is the distance to the farthest of the $k$ selected neighbours and
$k$ is the span. Only $k$ points receive nonzero weight, so the cost per target
is $O(k)$ plus the small least-squares solve. Degree $1$ follows local slopes;
degree $0$ is a weighted mean. The code uses degree $1$ for the trend and for
the seasonal subseries, and degree $0$ when a subseries is treated as periodic
or has fewer points than the requested span.

### Cycle-subseries smoothing

For each phase $j=0,\dots,m-1$, let
$\mathcal I_j=\{t:t\equiv j\pmod m\}$ and let $y^{(j)}$ be that subsequence.
Smooth it with span $s$:

$$
C_t=\mathrm{Loess}\bigl(\{y_i\}_{i\in\mathcal I_j}\bigr)(t),
\qquad t\in\mathcal I_j,
\qquad (4)
$$

and extrapolate the local fit one period before and one period after the
observed range. The extended estimate $\tilde C$ has length $n+2m$; this is why
the code builds `positions = np.arange(-period + 1, n + period)`. The
extrapolation matters at the boundaries, where the seasonal component would
otherwise be undefined for the first and last partial cycle.

### Low-pass filtering

The extended seasonal estimate still contains slow variation. Remove it with

$$
L=\mathrm{Loess}_{l}\Bigl(\mathrm{MA}_{3}
\bigl(\mathrm{MA}_{m}(\mathrm{MA}_{m}(\tilde C))\bigr)\Bigr),
\qquad (5)
$$

where $\mathrm{MA}_{k}$ is a centred moving average of length $k$ and
$\mathrm{Loess}_{l}$ is a local linear smoother with span $l$. The two length-$m$
averages annihilate the period-$m$ component; the length-$3$ average removes
the remaining high-frequency ripple; the final LOESS smooths what is left. The
seasonal component is then

$$
S_t=\tilde C_t-L_t
\qquad (6)
$$

on the observed range. Subtracting $L$ is what prevents the seasonal component
from absorbing the trend, and it is the step most naive implementations omit.

### Trend update

Deseasonalise and smooth:

$$
T_t=\mathrm{Loess}_{t}(y-S)(t).
\qquad (7)
$$

The pair (6)–(7) is one inner iteration. STL runs it `inner` times (two by
default), which is enough because the two smoothers are contractions on the
components they do not own.

### Robust weights

After an inner pass, compute the remainder $R=y-S-T$ and the bisquare weights

$$
w_t=\left(1-\left(\frac{\lvert R_t\rvert}{6\,\mathrm{median}\lvert R\rvert}
\right)^2\right)^2
\mathbf 1\bigl[\lvert R_t\rvert<6\,\mathrm{median}\lvert R\rvert\bigr].
\qquad (8)
$$

These weights multiply the LOESS weights in (2) on the next outer pass. An
observation beyond six median absolute deviations gets weight zero and cannot
influence the seasonal or trend estimate. The median scale is recomputed each
outer iteration, so the procedure adapts to the remaining noise level.

### Multiple periods

For periods $m_1,\dots,m_P$, MSTL estimates one component at a time on the
series with the others removed:

$$
S^{(k)}=\mathrm{STL}_{m_k}\Bigl(y-\sum_{j\ne k}S^{(j)}\Bigr),
\qquad
S=\sum_{k=1}^{P}S^{(k)} .
\qquad (9)
$$

The code makes two passes over the periods, shortest to longest. The first pass
gives each component a rough estimate; the second re-estimates it with the other
components already subtracted. The trend is then smoothed on $y-S$.

### Choosing seasonal windows

For `seasonal_windows`, each subseries gets its own span. For a candidate span
$s$, the leave-one-out error is

$$
\mathrm{CV}(s)=\sum_{i\in\mathcal I_j}
\bigl(y_i-\hat y_i^{(-i)}(s)\bigr)^2,
\qquad (10)
$$

where $\hat y_i^{(-i)}$ is the local fit at $i$ with point $i$ removed. The
variant evaluates five odd spans around the requested one, picks the minimiser
for each phase, and uses that vector in the main decomposition. The candidate
with span at least the subseries length is the periodic option, fitted as a
weighted mean.

## Code walkthrough

The moving average is deliberately simple: reflect at the edges, then convolve.
Reflection keeps the output the same length as the input and avoids the
shrinking-window bias that a truncated average would introduce:

```python
def _moving_average(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return values.copy()
    pad = length // 2
    padded = np.pad(values, pad, mode="reflect")
    kernel = np.ones(length) / length
    return np.convolve(padded, kernel, mode="valid")
```

The low-pass filter calls this three times. The first two calls use the seasonal
period, which is what kills the seasonal signal; the third uses length three,
which is what removes the ripple left by the first two.

The LOESS core selects the nearest `span` points by absolute distance and forms
tricube weights. The design is centred at the target, so the intercept is the
prediction and the slope is only a nuisance parameter:

```python
        distance = np.abs(x - target)
        take = np.argpartition(distance, span - 1)[:span]
        local = distance[take]
        radius = float(local.max())
        if radius <= 0.0:
            out[row] = float(np.average(y[take]))
            continue
        shape = np.clip(local / radius, 0.0, 1.0)
        w = (1.0 - shape ** 3) ** 3
        if weights is not None:
            w = w * weights[take]
```

`np.argpartition` is $O(n)$ per target instead of the $O(n\log n)$ of a full
sort, which matters because the trend smoother evaluates every observation. The
`radius <= 0` branch handles a target that coincides with every selected point,
where dividing by the radius would be undefined.

Cycle-subseries smoothing is the only place that knows about phases. The
modular mask picks the targets that belong to phase $j$, and the branch on
`seasonal_window` implements the periodic option:

```python
        for j in range(period):
            index = np.arange(j, n, period)
            if index.size < 2:
                continue
            targets = positions[(positions - j) % period == 0]
            if seasonal_window is None:
                span, degree = index.size, 0
            else:
                span = int(windows[j]) if windows is not None else seasonal_window
                degree = self.seasonal_degree
                if span >= index.size:
                    span, degree = index.size, 0
```

Passing `windows=None` gives every phase the same span; passing a vector is what
`seasonal_windows` does. The `index.size < 2` guard skips a phase that appears
only once, where a local fit is not identified.

The low-pass filter is a compact transcription of (5), with the final LOESS
evaluated on the same grid:

```python
    def _low_pass(self, cycles: np.ndarray, period: int,
                  window: int) -> np.ndarray:
        values = _moving_average(cycles, period)
        values = _moving_average(values, period)
        values = _moving_average(values, 3)
        grid = np.arange(values.size, dtype=float)
        return _loess(grid, values, grid, window, 1)
```

The input is the extended cycle estimate, so the output is also extended. The
caller trims `start = period - 1` from both the cycle estimate and the low-pass
output, which aligns the first observation with index $m-1$ of the extended
grid.

One inner iteration is four lines of component algebra:

```python
        for _ in range(inner):
            detrended = y - trend
            cycles = self._cycle_smooth(detrended, period, windows,
                                        seasonal_window, weights)
            low = self._low_pass(cycles, period, low_pass_window)
            start = period - 1
            seasonal = cycles[start:start + n] - low[start:start + n]
            deseasonalised = y - seasonal
            trend = _loess(np.arange(n, dtype=float), deseasonalised,
                           np.arange(n, dtype=float), trend_window,
                           self.trend_degree, weights)
```

`detrended` is the input to the seasonal smoother, not the original series;
`deseasonalised` is the input to the trend smoother. Swapping those two lines
produces a decomposition that still sums to $y$ but has the trend and seasonal
roles exchanged, which is why the identity alone is not a test.

The robust outer loop recomputes the scale from the current remainder and
restarts the inner loop with the new weights:

```python
def _bisquare(remainder: np.ndarray) -> np.ndarray:
    scale = 6.0 * float(np.median(np.abs(remainder)))
    if scale <= 0.0:
        return np.ones(remainder.size)
    shape = np.abs(remainder) / scale
    return np.where(shape < 1.0, (1.0 - shape ** 2) ** 2, 0.0)
```

The `scale <= 0` case is not hypothetical: a perfectly periodic series with no
noise has a zero remainder, and the weights should then be all ones rather than
a division by zero.

MSTL is two passes over the periods, each estimating one component on the
partial series:

```python
        for _ in range(2):
            for k, period in enumerate(periods):
                partial = y.copy()
                for other, component in enumerate(seasonals):
                    if other != k:
                        partial = partial - component
                component, _, _ = self._run(
                    partial, period, None, self.seasonal_window_,
                    self.trend_window_, self.low_pass_window_, 1)
                seasonals[k] = component
```

The `None` for `windows` is deliberate: each period has its own subseries
length, so a single window vector of length `self.period` would index the wrong
phase. The two passes are enough to separate cycles that are not exact
multiples; cycles that share a harmonic, such as 7 and 14, are not identifiable
from the sum alone.

The window selection is a leave-one-out loop with five candidates:

```python
            best, best_error = values.size, np.inf
            for span in candidates:
                error = self._cv_error(values, span)
                if error < best_error:
                    best, best_error = span, error
            chosen[j] = min(best, values.size)
```

`_cv_error` refits the subseries with one point held out and predicts it, which
is why the selected window can differ from phase to phase. The `min` against
`values.size` keeps a span from exceeding the number of available points; the
smoother itself also clamps, so this is belt and braces.

## Complexity

With $n$ observations, period $m$, seasonal span $s$, trend span $t$, inner
iterations $i$ and outer iterations $o$:

- **One LOESS evaluation:** $O(k)$ to find neighbours plus $O(k)$ for the
  weighted fit, with $k$ the span.
- **Cycle-subseries smoothing:** $m$ subseries, each with about $n/m$ points;
  one pass costs $O(n s)$.
- **Low-pass:** three moving averages at $O(n)$ plus one LOESS at $O(n l)$.
- **Trend smoothing:** $O(n t)$.
- **One inner pass:** $O(n(s+t+l))$.
- **Full fit:** $O(i(1+o)\,n(s+t+l))$.
- **`mstl`:** $P$ periods, two passes, so $O(2P\,i\,n(s+t+l))$.
- **`seasonal_windows`:** the CV selection adds $O(m\,C\,n_j^2)$ for $C$
  candidates and $n_j\approx n/m$ points per subseries, plus the fit.
- **Memory:** $O(n)$ for the components and $O(n)$ for the extended cycle
  estimate.

The demo's 240-point fit is dominated by the trend LOESS and runs in well under
a second. The cubic term is in the robust outer count, not in $n$: a series with
100 000 points and 15 outer iterations is still a linear scan per smoother, but
the Python loops make it slow enough that a production implementation would
batch the local fits.

## Failure modes

**A wrong period splits one seasonal shape into two.** If the true period is 12
and you use 6, the smoother can represent the fundamental but not the harmonic,
so the leftover harmonic leaks into the trend and the remainder. The diagnostic
is a trend that oscillates with the seasonal period, and a remainder with a
large lag-$m$ autocorrelation. Plot the trend's first difference; a periodic
zig-zag there means the period is wrong.

**A short trend window absorbs the seasonality.** The trend smoother and the
seasonal smoother compete for the same low-frequency variation. If `trend` is
close to `period`, the trend can follow the seasonal cycle and the seasonal
component collapses. Keep the trend window well above the period — the default
rule uses $1.5m/(1-1.5/s)$ — and check that the seasonal profile has the
amplitude you expect.

**Robustness is not outlier removal.** The demo's robust fit has a larger
remainder RSS than the non-robust one because the spikes remain in $R$. If the
goal is a clean series for forecasting, the outliers must be removed or modeled
separately; robust STL only stops them from corrupting $T$ and $S$. The weight
vector is the diagnostic: weights near zero mark the observations the fit
ignored.

**MSTL cannot separate shared harmonics.** Periods 7 and 14, or 12 and 24,
produce seasonal subspaces that overlap. The two-pass subtraction then has an
identifiability problem: many splits of the same signal fit equally well. Use
the smallest set of periods that spans the observed cycles, and check that each
extracted component has the expected shape rather than trusting the sum.

**Boundary extrapolation is an assumption.** The cycle smoother extrapolates
one period beyond each end, and the moving averages reflect at the edges. Both
are reasonable but neither is estimated from data. The first and last few
seasonal estimates are therefore less reliable than the middle, which matters
when STL is used to seasonally adjust the most recent observations for a
nowcast.

## Applications

STL is the standard preprocessing step for seasonal economic data. Statistical
agencies decompose industrial production, retail sales and employment into
trend, seasonal and irregular components; the seasonally adjusted series is
then forecast with ARIMA or an exponential smoothing model on the adjusted
scale. Its appeal is transparency: there is no likelihood, no latent state and
no convergence to a local optimum, only two smoothers and a loop.

In energy and utilities it separates weather-driven seasonal demand from the
trend, and in traffic forecasting it removes the weekly and daily cycles before
a short-horizon model. In sensor networks it is used to remove diurnal drift
from a slowly failing sensor, where the robust variant matters because glitches
are common. MSTL is the natural choice for series with more than one cycle, such
as hourly electricity demand with daily and weekly periods.

It is the wrong tool when you need a probabilistic forecast: STL produces
components, not predictive distributions. It is also weak when the seasonality
changes shape abruptly, because every subseries is smoothed with a fixed window
(unless `seasonal_windows` is used) and the method has no notion of a
breakpoint. For those cases a state-space decomposition or a regression with
time-varying seasonal terms is more appropriate.

## References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., Terpenning, I. (1990).
  *STL: a seasonal-trend decomposition procedure based on loess.* Journal of
  Official Statistics 6(1), 3–73. The original algorithm, including the
  low-pass filter and the outer robustness loop.
- Cleveland, W. S., Devlin, S. J. (1988). *Locally weighted regression: an
  approach to regression analysis by local fitting.* Journal of the American
  Statistical Association 83(403), 596–610. The LOESS smoother and tricube
  weights behind (2)–(3).
- Cleveland, W. S. (1979). *Robust locally weighted regression and smoothing
  scatterplots.* JASA 74(368), 829–836. The bisquare robustness idea of (8).
- Hyndman, R. J., Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed., ch. 3. The applied treatment of STL and its use before
  forecasting.
- Bandara, K., Hyndman, R. J., Bergmeir, C. (2021). *MSTL: a seasonal-trend
  decomposition algorithm for time series with multiple seasonal patterns.*
  International Journal of Operational Research. The multiple-period extension
  behind (9).
