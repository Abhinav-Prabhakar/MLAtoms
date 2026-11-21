# DLinear — decomposition and a linear map, and why that is hard to beat

`dlinear.py` · Time series · Tier 2 · 4 variants

## Intuition

In 2023 a paper asked whether Transformers are actually necessary for
long-horizon forecasting. Its answer was a one-layer linear map applied to a
decomposed look-back window, and that map matched or beat the Transformer
models on the standard benchmarks. The result is uncomfortable but simple:
when the data are a smooth trend plus a repeating seasonal shape, a linear
function of the recent past is already a good forecast, and most of the
modelling effort in a deep architecture is spent relearning that structure.

DLinear writes the assumption down explicitly. Take the last $L$ observations,
split them into a moving-average trend and a residual seasonal part, and learn
two linear maps from those $L$ values to the next $H$. The sum is the forecast.
The model has $2LH$ weights — a few thousand for typical settings — and fitting
it is one ridge regression, not a training loop.

The variants isolate the two ideas inside that recipe:

- **`dlinear`** is the full model: decomposition plus a jointly fitted linear
  head on the concatenated trend and seasonal features.
- **`nlinear`** removes the decomposition and instead subtracts the last
  observation from every input and from the target. The forecast is invariant
  to an additive level shift, which is its whole purpose.
- **`moving_avg_decomp`** keeps the decomposition but replaces the learned
  trend head with an explicit least-squares line through the trend, so the
  trend forecast is extrapolation rather than regression.
- **`channel_independent`** fits one pair of heads per channel instead of
  sharing a single map across channels, which is what the original paper does.

The demo checks accuracy against naive and seasonal-naive baselines, verifies
the decomposition identity, measures the shift-equivariance that separates
`nlinear` from `dlinear`, and compares pooled against per-channel heads on a
two-channel series whose scales differ by an order of magnitude.

## Variants

| variant | forecast | demo result |
|:--|:--|:--|
| `dlinear` | linear map on trend and seasonal | MAE 0.2545 against 1.7795 |
| `nlinear` | linear map on last-value differences | shift error 1.1e-13 |
| `moving_avg_decomp` | line extrapolation and seasonal map | MAE 0.1989 vs 1.1324 |
| `channel_independent` | one map per channel | MAE 0.1468 against 0.1518 |

**`dlinear`** fits 541 windows from a 600-point training segment of a series
with a linear trend, two harmonics and noise. The in-sample MSE is $0.0910$,
and rolling one-step-ahead blocks give a mean absolute error of $0.2545$
against $1.7795$ for the naive forecast and $0.5632$ for seasonal naive with
the correct period of 24. The gain over seasonal naive is the interesting one:
the linear map is not just repeating yesterday, it is learning how the seasonal
shape and the trend combine over the 48-step look-back.

**`nlinear`** is trained on a stationary series and then asked to forecast after
the level jumps by $+1000$. Its forecast is exactly equivariant: shifting the
history by $c$ shifts every forecast by $c$, with a numerical error of
$1.14\times10^{-13}$. `dlinear` is not equivariant; the same shift changes its
forecast by $1.3626$ too little, because its intercept and weights were fitted
to the old level. One full window after the break the rolling MAE is $0.1847$
for `nlinear` and $0.9947$ for `dlinear`. The difference is real but modest on
this synthetic series — a linear model partially follows a level shift — so the
equivariance check, not the MAE, is the honest demonstration of the mechanism.

**`moving_avg_decomp`** handles a trend whose slope changes halfway through the
record. The trend component is smoothed with a 25-point kernel, a line is fitted
to the last 60 trend values and extrapolated 12 steps, and a ridge head learns
the residual. Its rolling MAE is $0.1989$ against $1.1324$ for naive. The
explicit line is more interpretable than a learned trend head and is the right
choice when the trend is locally linear by assumption.

**`channel_independent`** is fitted to two channels with very different scales
and periods: one has amplitude 4 and period 12, the other amplitude 0.4 and
period 7. Separate heads reach a two-channel MAE of $0.1468$ against $0.1518$
for a single shared head. The improvement is only 3%, which is itself a useful
result: on short windows with standardised features, pooling is a mild
regulariser rather than a disaster. On channels with genuinely different
dynamics the gap widens.

## Derivation

### The windowed forecasting problem

Fix a look-back $L$ and a horizon $H$. For every origin $t$ with enough history
and future, form the pair

$$
x_{t-L+1:t}\in\mathbb R^{L},\qquad x_{t+1:t+H}\in\mathbb R^{H},
\qquad (1)
$$

and choose a forecast function $f$ minimising squared error over the training
origins:

$$
\min_{f}\sum_{t=L}^{n-H}
\bigl\lVert f(x_{t-L+1:t})-x_{t+1:t+H}\bigr\rVert_2^2 .
\qquad (2)
$$

The whole family differs only in how $f$ is parameterised. DLinear restricts
$f$ to be affine after a fixed decomposition; NLinear applies an affine map
after a fixed normalisation.

### Moving-average decomposition

For an odd kernel $k=2p+1$, the trend is the centred average

$$
\tau_t=\frac1k\sum_{j=-p}^{p}x_{t+j},
\qquad (3)
$$

with the series extended by edge replication at both ends so that every
observation has a full window. The seasonal residual is whatever is left:

$$
\sigma_t=x_t-\tau_t .
\qquad (4)
$$

The identity (4) is exact by definition, which is why the demo can assert it to
machine precision. The modelling assumption is not the identity but the claim
that $\tau$ is smooth and $\sigma$ is periodic; a kernel much shorter than the
period leaves seasonal variation in the trend, and a kernel much longer than
the window flattens genuine changes.

### DLinear

Apply (3)–(4) to the look-back window, giving vectors $\tau$ and $\sigma$ of
length $L$. The forecast is

$$
\hat x_{t+1:t+H}=W_\tau\tau+W_\sigma\sigma+b,
\qquad (5)
$$

with $W_\tau,W_\sigma\in\mathbb R^{H\times L}$ and $b\in\mathbb R^{H}$.
Equation (5) is a single linear map on the concatenated feature vector
$[\tau;\sigma]$, which is why the implementation fits one ridge regression
rather than two independent heads. Fitting the two heads separately to the raw
target would make each approximate the whole future and their sum double-count
it; the demo's first version did exactly that and produced an in-sample MSE of
$273$ instead of $0.091$.

### Ridge fitting

The features have different scales across lags and across channels, so the code
standardises each column before solving. With $z_i$ the standardised
concatenation and $Y$ the matrix of targets,

$$
\min_{b,W}\sum_{i}\lVert y_i-b-Wz_i\rVert_2^2+\lambda\lVert W\rVert_F^2,
\qquad (6)
$$

where the intercept $b$ is not penalised. Stacking a column of ones in front of
$Z$ and writing $P=\mathrm{diag}(0,1,\dots,1)$ gives the closed form

$$
\begin{bmatrix}b\\ W\end{bmatrix}
=\bigl(Z^\top Z+\lambda P\bigr)^{-1}Z^\top Y .
\qquad (7)
$$

The code solves (7) with `np.linalg.solve` and falls back to `lstsq` if the
Gram matrix is singular, which happens when a channel is constant or when
$L$ exceeds the number of windows. Standardisation is what makes a single
$\lambda$ sensible across columns; without it the penalty would depend on the
units of each lag.

### NLinear

NLinear removes the level of each window before applying the map. Let
$\ell=x_t$ be the last observed value. The normalised input and target are

$$
\tilde x_{i}=x_i-\ell,\qquad \tilde y_{h}=x_{t+h}-\ell,
\qquad (8)
$$

and the model is a single affine map $g$ fitted to
$(\tilde x,\tilde y)$. At prediction time the level is added back:

$$
\hat x_{t+1:t+H}=g(\tilde x)+\ell\,\mathbf 1 .
\qquad (9)
$$

Substituting $x_i+c$ for every input shifts $\tilde x$ by nothing and $\ell$ by
$c$, so the forecast shifts by exactly $c$. That is the equivariance the demo
measures at $10^{-13}$. DLinear lacks it because $b$ is fixed: its response to a
uniform shift is $c$ times the sum of its weights, which is close to but not
exactly one.

NLinear handles an additive level break that happened before the window. It
does not handle a break inside the look-back: the differences then contain the
jump, and the linear map extrapolates it as if it were a trend. In the demo the
transition window is excluded for that reason; a real deployment should detect
the break and wait for a clean window, or model the break explicitly.

### Explicit trend extrapolation

For `moving_avg_decomp`, the trend head is replaced by a line fitted to the
trend values at positions $0,\dots,L-1$:

$$
\hat\tau_{L+h}=\bar\tau+\hat\beta\,(L+h-\bar t),\qquad
\hat\beta=\frac{\sum_{t=0}^{L-1}(t-\bar t)(\tau_t-\bar\tau)}
{\sum_{t=0}^{L-1}(t-\bar t)^2},
\qquad (10)
$$

and the ridge head is fitted to the residual target
$y-\hat\tau$. The trend forecast is now extrapolation, so it degrades
gracefully on a locally linear trend and badly at a turning point. This is the
classical decomposition-plus-extrapolation idea, with the seasonal part still
learned.

### Channel independence

For a series with $C$ channels, the pooled model stacks all windows from all
channels into one design matrix and solves (7) once. The channel-independent
model solves one problem per channel:

$$
\hat x^{(c)}_{t+1:t+H}=W^{(c)}_\tau\tau^{(c)}
+W^{(c)}_\sigma\sigma^{(c)}+b^{(c)},
\qquad c=1,\dots,C .
\qquad (11)
$$

Pooling assumes that the same linear combination of lags is useful for every
channel. That is a strong assumption when scales or periods differ, and a mild
regulariser when they do not. The original DLinear is channel-independent; the
pooled option is useful as a cheaper model and as a diagnostic for how much the
channels actually share.

### Evaluation

All demo numbers are rolling-origin errors. For a set of origins $\mathcal O$,

$$
\mathrm{MAE}=\frac{1}{\lvert\mathcal O\rvert H}
\sum_{o\in\mathcal O}\sum_{h=1}^{H}
\bigl\lvert\hat x_{o+h}-x_{o+h}\bigr\rvert,
\qquad (12)
$$

with the model refitted once on the training prefix and only the forecast origin
moved. A random train/test split would leak future information into the
standardisation and into the window construction, and would flatter every
variant.

## Code walkthrough

The decomposition is a sliding-window mean. Padding by edge replication keeps
the output length equal to the input, which is what lets the trend and residual
be subtracted elementwise:

```python
def _moving_average(values: np.ndarray, kernel: int) -> np.ndarray:
    """Centred moving average along the last axis, edge-padded."""
    pad = kernel // 2
    padding = [(0, 0)] * (values.ndim - 1) + [(pad, pad)]
    padded = np.pad(values, padding, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=-1)
    return view.mean(axis=-1)
```

`sliding_window_view` returns a view, so the average is computed without
materialising `m` copies of each window. Edge replication is a choice: it
biases the trend toward the endpoint at the boundary, which is preferable to
zero padding (which would introduce a fake jump) and simpler than fitting a
local line there.

Window construction is pure index arithmetic, vectorised over all origins:

```python
    count = values.shape[0] - seq_len - horizon + 1
    if count < 2:
        raise ValueError("need at least two training windows")
    starts = np.arange(count)
    left = values[starts[:, None] + np.arange(seq_len)[None, :]]
    right = values[starts[:, None] + seq_len + np.arange(horizon)[None, :]]
    return left, right
```

The result is `(count, seq_len, channels)` and `(count, horizon, channels)`.
Fancy indexing copies, so this is the memory hotspot: about $nL$ doubles for
the design matrix. For very long series, the same windows can be generated in
batches, but the closed-form fit needs the full Gram matrix anyway.

The explicit trend head is a vectorised least-squares line. Centring the index
grid avoids forming a design matrix and makes the slope a single dot product:

```python
    length = values.shape[1]
    grid = np.arange(length, dtype=float)
    centred = grid - grid.mean()
    slope = values @ centred / float(centred @ centred)
    intercept = values.mean(axis=1)
    future = length + np.arange(horizon) - grid.mean()
    return intercept[:, None] + slope[:, None] * future[None, :]
```

Each row of `values` is one window, so the same expression fits every window at
once. The future grid starts at `length`, the first unobserved index, and runs
to `length + horizon - 1`.

The ridge solver is the only numerical machinery in the file. It standardises
features, leaves the intercept out of the penalty, and solves the normal
equations directly:

```python
        self.mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        design = np.column_stack(
            [np.ones(features.shape[0]), (features - self.mean_) / self.scale_])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        moment = design.T @ targets
        try:
            self.coef_ = np.linalg.solve(gram, moment)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(gram, moment, rcond=None)[0]
```

The `scale > 1e-12` guard turns a constant feature into a zero column instead
of dividing by zero; the intercept then absorbs its mean. The fallback is not
decorative: with `ridge=0` and more lags than windows the Gram matrix is
singular, and `solve` raises rather than returning a least-norm solution.

Fitting DLinear is three lines once the windows exist. The target is flattened
in the same window-major, channel-minor order as the features, so row $i$ of the
design always corresponds to row $i$ of the target:

```python
            flat = left.transpose(0, 2, 1).reshape(-1, self.seq_len_)
            target = right.transpose(0, 2, 1).reshape(-1, self.horizon)
            trend, seasonal = _decompose(flat, self.kernel)
            if self.variant == "dlinear":
                design = np.column_stack([trend, seasonal])
                self.model_ = _Ridge(self.ridge).fit(design, target)
            else:
                self.trend_future_train_ = _line_forecast(trend, self.horizon)
                self.seasonal_model_ = _Ridge(self.ridge).fit(
                    seasonal, target - self.trend_future_train_)
```

The `column_stack` is (5) in code: the two heads share one solve, so the fit
allocates the error between them however it likes. The `else` branch is the
only place where the trend is not learned; there the target residual is
`target - trend_future`, which is why the seasonal head is fitted to a
detrended target.

NLinear's normalisation is symmetric between fit and predict. The last column
of each window is the reference, and `[:, None, :]` re-inserts the axis that
broadcasting needs:

```python
        last = left[:, -1, :][:, None, :]
        features = (left - last).transpose(0, 2, 1).reshape(
            -1, self.seq_len_)
        targets = (right - last).transpose(0, 2, 1).reshape(
            -1, self.horizon)
        return features, targets
```

Because the reference is subtracted from both sides, the learned map sees only
within-window differences. The last feature column is identically zero, so the
corresponding weight is unidentifiable; ridge drives it to zero, and the
intercept carries the average change.

Prediction inverts the same operation. The reshape restores the
window/channel order before the level is added back, and the final transpose
puts the horizon before the channels:

```python
        if self.variant == "nlinear":
            last = left[:, -1, :][:, None, :]
            features = (left - last).transpose(0, 2, 1).reshape(
                -1, self.seq_len_)
            forecast = self.nlinear_.predict(features)
            forecast = forecast.reshape(count, channels, self.horizon)
            return forecast.transpose(0, 2, 1) + last
```

The decomposition variants share a single `_predict_windows` path. The pooled
model predicts all windows at once; the per-channel branch loops over channels
and writes into a preallocated array:

```python
        flat = left.transpose(0, 2, 1).reshape(-1, self.seq_len_)
        trend, seasonal = _decompose(flat, self.kernel)
        if self.variant == "dlinear":
            forecast = self.model_.predict(
                np.column_stack([trend, seasonal]))
        else:
            forecast = (_line_forecast(trend, self.horizon)
                        + self.seasonal_model_.predict(seasonal))
        return forecast.reshape(count, channels, self.horizon).transpose(0, 2, 1)
```

Keeping the window axis in the reshape is what makes `predict` and the
in-sample loss use the same function. The `train_loss_` reported by `fit` is
computed by calling `_predict_windows` on the training windows, so it cannot
silently disagree with the forecasts the model actually produces.

## Complexity

With $n$ observations, $C$ channels, look-back $L$, horizon $H$, $m=n-L-H+1$
windows and $k$ the moving-average kernel:

- **Windowing:** $O(mLC)$ time and memory, the dominant cost for long series.
- **Decomposition:** $O(mLC)$ with the sliding-window view, plus $O(mLC)$ for
  the residual.
- **One ridge solve:** $O(m(2L)^2+(2L)^3)$ time for pooled DLinear, with the
  same solve producing all $H$ target columns.
- **`channel_independent`:** $C$ solves, so $O(C(mL^2+L^3))$.
- **`nlinear`:** one solve of the same size as DLinear without the doubled
  feature count.
- **Prediction:** $O(LC)$ to prepare and $O(HLC)$ to apply the maps; a single
  forecast is a handful of dot products.
- **Memory:** $O(mLC)$ for the training windows, $O((2L)H)$ for the weights.

The demo's four fits on 720 points each take a few hundredths of a second. The
model is small enough to run on a microcontroller, which is part of its appeal;
the expensive part of a deployment is usually the windowing and the data
plumbing, not the arithmetic.

## Failure modes

**A linear map cannot extrapolate a nonlinear trend.** DLinear learns weights
that combine lags, so it follows a straight trend well and a curve poorly. If
the trend accelerates, the forecast lags behind it; the residual grows with the
horizon. `moving_avg_decomp` does not fix this in general — it fits a line to
the trend window — but it makes the assumption explicit so you can test it.
Difference the series or model the trend separately when curvature matters.

**The look-back must cover the longest period.** A 48-step window sees only two
cycles of a period-24 seasonal pattern. That is enough for the demo, but the
weights are then estimated from few independent seasonal shapes. If the period
is 168 (hourly data with a weekly cycle), a 48-step window is blind to it, and
the model will fit noise. Set $L$ to at least several periods, or feed the model
seasonal features explicitly.

**Edge padding biases the decomposition.** The moving average replicates the
first and last values, so the trend at the boundary is pulled toward them and
the residual there absorbs the error. The bias is $O(k)$ observations wide. In
the demo the interior of a pure ramp is recovered to $3.6\times10^{-15}$, but
the first and last few trend values are not exact. Do not read the boundary
residual as a change in seasonality.

**NLinear's invariance is additive only.** It handles a level shift, not a
change in scale or in the seasonal amplitude. Multiply the series by three and
the differences triple while the learned map still expects the old scale; the
forecast is then too small by roughly a factor of three in the seasonal
component. Detect scale changes separately, or normalise by a robust scale
rather than by the last value alone.

**A break inside the window breaks both models.** NLinear subtracts the last
value, but if the jump occurred 20 steps ago the differences still contain it
and the linear map extrapolates the jump as a trend. In the demo the transition
window has to be excluded. A real system should detect the break, discard the
contaminated window, and only forecast once the look-back is clean.

**Pooling channels can hide a bad channel.** The pooled model minimises average
error, so a high-variance channel dominates the fit and a low-variance channel
is fitted almost by accident. In the demo the penalty is only 3%, but with
channels whose scales differ by $10^3$ the pooled weights are effectively
chosen for the largest channel. Standardise per channel or use
`channel_independent`.

**Random splits leak the future.** Sliding windows overlap, so a random
train/test split puts near-duplicate windows on both sides and the test error
becomes an in-sample error in disguise. Always split by time and evaluate on
rolling origins, as in (12). This is the single most common reason a DLinear
result looks better than it is.

**The benchmark claim is conditional.** DLinear beat several Transformers on
the long-horizon benchmarks used in the 2023 paper, but later work showed that
those benchmarks reward trend extrapolation and that tuned transformers can
close the gap. DLinear is best read as a strong, cheap baseline that any
sequence model must beat — not as proof that attention is useless.

## Applications

DLinear is used as the default baseline in long-term forecasting benchmarks and
as a sanity check before training a deep model: if a linear map on the
decomposed window matches the candidate, the candidate's extra capacity is not
buying anything on that dataset. Its parameter count makes it attractive for
edge deployment — smart meters, weather stations, industrial sensors — where
the forecast runs on the device and retraining must be cheap.

The decomposition variant is useful in energy load forecasting, where the trend
is driven by slow economic and demographic change and the seasonal component by
daily and weekly routines. In traffic forecasting the same structure appears at
the scale of a road network, and channel-independent heads let each sensor keep
its own scale. In retail demand the series are intermittent and the linear
assumption is weak; there a count model or Croston's method is a better fit, and
DLinear's failure is informative rather than embarrassing.

The model is the wrong choice when the target depends on interactions between
lags that are not additive, when the horizon is much longer than the look-back
and the trend is nonlinear, or when exogenous drivers (weather, promotions,
holidays) carry most of the signal. In those cases add the covariates
explicitly, use a model with a nonlinear link, or move to a sequence model —
but keep DLinear as the baseline that the more complex model has to justify
itself against.

## References

- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). *Are Transformers effective for
  time series forecasting?* Proceedings of the AAAI Conference on Artificial
  Intelligence 37(9), 11121–11129. DLinear, NLinear and the decomposition
  baseline.
- Hoerl, A. E., Kennard, R. W. (1970). *Ridge regression: biased estimation for
  nonorthogonal problems.* Technometrics 12(1), 55–67. The penalty in (6)–(7).
- Hyndman, R. J., Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice*, 3rd ed., ch. 5. Rolling-origin evaluation and the baseline
  forecasts used in the demo.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., Terpenning, I. (1990).
  *STL: a seasonal-trend decomposition procedure based on loess.* Journal of
  Official Statistics 6(1), 3–73. The decomposition idea that DLinear
  simplifies to a moving average.
- Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020). *N-BEATS:
  neural basis expansion analysis for interpretable time series forecasting.*
  International Conference on Learning Representations. The trend-plus-seasonal
  decomposition as a forecasting architecture.
- Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). *A time series is
  worth 64 words: long-term forecasting with Transformers.* International
  Conference on Learning Representations. PatchTST, the patching approach that
  followed the linear-model result.
