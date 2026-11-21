# N-HiTS — sample coarsely, forecast smoothly, refine in layers

`nhits.py` · Time series · Tier 3 · 3 variants

## Intuition

N-BEATS spends most of its parameters on the first weight matrix, which maps
the whole look-back window into the hidden layer. On a long window that matrix
is huge, and most of its columns describe high-frequency wiggles that do not
survive to the horizon. N-HiTS asks a cheaper question at each layer: how much
resolution does this block actually need?

Two mechanisms answer it. **Multi-rate sampling** pools the input by a factor
that grows for the earlier blocks, so a coarse block reads a summary of the
window rather than every point. **Hierarchical interpolation** makes each block
emit only a few forecast coefficients — one per interval of the horizon — and a
fixed linear interpolation stretches them back to full resolution. The stack
then runs coarse to fine: the first block lays down the broad shape, and later
blocks add detail at a higher rate.

The payoff is fewer parameters, a shorter path from input to output, and a
forecast that is smooth by construction at the coarse scales. The cost is that
the coarse blocks cannot represent sharp turning points, and the fine blocks
can overfit if the input resolution is high enough to memorise individual
windows.

The three variants are the two mechanisms and their combination:

- **`nhits`** pools the input and coarsens the forecast grid, both decreasing
  with depth.
- **`multi_rate`** pools the input but keeps a full-resolution forecast head.
- **`hierarchical_interp`** keeps the input at rate one and coarsens only the
  forecast grid.

The demo makes the ablation visible: `nhits` reaches a held-out MAE of
$0.2882$, `multi_rate` $0.3119$, and `hierarchical_interp` $0.6808$ despite
the best in-sample fit ($0.0182$). Interpolation alone smooths the output but
does not stop the blocks from memorising the input.

## Variants

| variant | input rate | forecast grid | train MSE | held-out MAE |
|:--|:--|:--|:--|:--|
| `nhits` | 4, 2, 1 | 3, 6, 12 | 0.0205 | 0.2882 |
| `multi_rate` | 4, 2, 1 | 12, 12, 12 | 0.0214 | 0.3119 |
| `hierarchical_interp` | 1, 1, 1 | 3, 6, 12 | 0.0182 | 0.6808 |

**`nhits`** trains on 289 windows with pooling rates $4,2,1$ and forecast
grids $3,6,12$. The in-sample mean squared error is $0.0205$ in per-window
variance units, and the held-out MAE of $0.2882$ is close to the noise level of
$0.25$ per point. It beats seasonal naive ($0.4627$) by a comfortable margin.

**`multi_rate`** keeps the pooling but gives every block a twelve-point
forecast head, so no interpolation happens. The error rises to $0.3119$: the
coarse input is helpful, but forcing the block to emit every future step itself
costs accuracy relative to letting a low-resolution head be smoothed.

**`hierarchical_interp`** removes the pooling and keeps the grids at $3,6,12$.
Its train MSE of $0.0182$ is the best of the three, and its held-out MAE of
$0.6808$ is by far the worst. The blocks see all 48 inputs and enough capacity
to fit the training windows; the interpolation smooths the *output* but does
nothing to stop the *input* path from memorising. Pooling is the regulariser,
not interpolation.

## Derivation

### The stack

A stack of $K$ blocks with inputs $x_0$ and residual targets $y_0$ computes

$$
\hat y=\sum_{k=1}^{K} f_k(x_{k-1}),\qquad
x_k=x_{k-1}-b_k(x_{k-1}),\qquad
y_k=y_{k-1}-f_k(x_{k-1}),
\qquad (1)
$$

exactly as in N-BEATS. What changes is how $f_k$ and $b_k$ are parameterised.

### Multi-rate sampling

Block $k$ has an integer sampling rate $r_k$. The window of length $L$ is
left-padded with copies of its first value until its length is a multiple of
$r_k$, and then max-pooled:

$$
p_{k,t}=\max_{0\le j<r_k} \tilde x_{t r_k+j},\qquad
t=0,\dots,m_k-1,\quad m_k=\lceil L/r_k\rceil .
\qquad (2)
$$

Padding on the *left* keeps the last pooled group aligned with the most recent
observations, which is the end of the window that matters for forecasting.
Max pooling is used rather than averaging because it preserves local extremes:
a spike that survives the pool is still visible to the block.

### The block network

Each block is a single hidden layer followed by two linear heads:

$$
h=\mathrm{ReLU}(W_kp_k+c_k),
\qquad (3)
$$

$$
\theta^{f}=A_f h+d_f,\qquad \theta^{b}=A_b h+d_b,
\qquad (4)
$$

with $\theta^{f}\in\mathbb R^{c_k}$ and $\theta^{b}\in\mathbb R^{m_k}$. The
forecast head is deliberately smaller than the horizon: $c_k\le H$.

### Hierarchical interpolation

A fixed matrix $U_k\in\mathbb R^{H\times c_k}$ expands the coarse coefficients:

$$
f_k=U_k\theta^{f},\qquad b_k=V_k\theta^{b},
\qquad (5)
$$

with $V_k\in\mathbb R^{L\times m_k}$ doing the same for the backcast. The
interpolation is piecewise linear between evenly spaced nodes. Writing
$s_i=(c_k-1)\,i/(H-1)$ for the position of fine index $i$ in coarse
coordinates, and $\ell=\lfloor s_i\rfloor$,

$$
(U_k)_{i\ell}=1-(s_i-\ell),\qquad
(U_k)_{i,\ell+1}=s_i-\ell,
\qquad (6)
$$

with all other entries zero and the indices clipped to $[0,c_k-1]$. Every row
sums to one, the first row is $e_1$ and the last is $e_{c_k}$, so the endpoints
are reproduced exactly.

Because $U_k$ is fixed, its transpose is all the backward pass needs:

$$
\frac{\partial L}{\partial\theta^{f}}
=\frac{\partial L}{\partial f_k}\,U_k^{\top}.
\qquad (7)
$$

That is the whole hierarchy: the block thinks in a low-dimensional grid and the
matrix does the resampling.

### The rate schedule

For $K$ blocks the implementation uses

$$
r_k=2^{K-1-k},\qquad c_k=\Bigl\lceil \frac{H}{r_k}\Bigr\rceil,
\qquad k=1,\dots,K,
\qquad (8)
$$

so a twelve-step horizon with three blocks gives rates $4,2,1$ and grids
$3,6,12$. The coarsest block describes the horizon with three numbers, which is
roughly a level and a slope; the finest block can place every point. Every
block is supervised against the same target $y_k$, so the coarse blocks are
asked to be accurate at their own resolution rather than merely smooth.

### The block loss

With $\lambda$ the backcast weight and $W$ the number of windows,

$$
L_k=\frac{1}{WH}\lVert f_k-y_k\rVert_F^2
+\lambda\,\frac{1}{WL}\lVert b_k-x_{k-1}\rVert_F^2 .
\qquad (9)
$$

### Gradients

With one hidden layer the backward pass is short. Let
$\delta=\partial L/\partial h$ be the gradient reaching the hidden activation:

$$
\delta=\Bigl(\frac{\partial L}{\partial\theta^{f}}A_f
+\frac{\partial L}{\partial\theta^{b}}A_b\Bigr)\odot\mathbf 1[z>0],
\qquad z=W_kp_k+c_k,
\qquad (10)
$$

$$
\frac{\partial L}{\partial W_k}=\delta^{\top}p_k,\qquad
\frac{\partial L}{\partial c_k}=\sum_{\text{batch}}\delta .
\qquad (11)
$$

Parameters are updated with Adam, the same rule used in N-BEATS:
$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$, and
$\theta\leftarrow\theta-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ with bias
correction.

### Standardisation

Each window is centred and scaled by its own mean and standard deviation, and
the target is scaled by the *input* window's standard deviation. Prediction
repeats that transform on the last $L$ observations. The model therefore learns
a shape, not a level, and a jump in the series does not invalidate it as long
as the shape within the window is familiar.

## Code walkthrough

The interpolation matrix is the only new piece of linear algebra. It is built
once per block and never learned, so it costs nothing in the backward pass:

```python
def _interp_matrix(fine: int, coarse: int) -> np.ndarray:
    """Linear interpolation from `coarse` evenly spaced points to `fine`."""
    if coarse == 1:
        return np.ones((fine, 1))
    position = np.linspace(0.0, coarse - 1.0, fine)
    left = np.floor(position).astype(int)
    frac = position - left
    left = np.clip(left, 0, coarse - 1)
    right = np.clip(left + 1, 0, coarse - 1)
    matrix = np.zeros((fine, coarse))
    rows = np.arange(fine)
    matrix[rows, left] += 1.0 - frac
    matrix[rows, right] += frac
    return matrix
```

Clipping both indices is what makes the endpoints exact: at the last position
`left` and `right` both equal `coarse - 1`, so the two weights `1 - frac` and
`frac` sum to one on the same column. Without the clip, the last row would read
past the end of the coefficient vector.

Pooling left-pads, reshapes and reduces. The padding repeats the first value
rather than zero-filling, because a zero is a real level and would distort a
window that does not contain zero:

```python
def _max_pool(values: np.ndarray, rate: int, pooled: int) -> np.ndarray:
    """Left-pad, reshape and take a maximum per group of `rate` steps."""
    padding = pooled * rate - values.shape[1]
    if padding > 0:
        edge = np.repeat(values[:, :1], padding, axis=1)
        values = np.concatenate([edge, values], axis=1)
    grouped = values.reshape(values.shape[0], pooled, rate)
    return grouped.max(axis=2)
```

The reshape works because numpy splits the last axis contiguously; the groups
are consecutive time steps, not interleaved channels.

The block's forward pass applies (2) and (3). Keeping the pooled input and the
pre-activation is enough to reconstruct every gradient later:

```python
    def _hidden(self, values: np.ndarray):
        pooled = _max_pool(values, self.rate, self.pooled)
        pre = pooled @ self.w1.T + self.b1
        return pooled, pre, np.maximum(pre, 0.0)
```

The heads are ordinary linear layers whose output width is the *coarse* grid,
not the horizon. This is the one line where `hierarchical_interp` differs from
a plain MLP forecaster:

```python
            theta_fore = activation @ self.wf.T + self.cf
            theta_back = activation @ self.wb.T + self.cb
            forecast = theta_fore @ self.fore_matrix.T
            backcast = theta_back @ self.back_matrix.T
```

Both the forecast and the backcast are interpolated. The backcast being coarse
is not a compromise: it only has to carry a smooth summary of the window into
the residual chain, and the next block will see the fine detail anyway.

The gradient at the hidden layer combines both heads and applies the ReLU mask.
The transpose of the interpolation matrix converts the output-space gradient
into coefficient space with no extra work:

```python
            theta_fore_grad = grad_fore @ self.fore_matrix
            theta_back_grad = grad_back @ self.back_matrix
            hidden_grad = (theta_fore_grad @ self.wf
                           + theta_back_grad @ self.wb)
            delta = hidden_grad * (pre > 0.0)
```

Adam is applied to all six parameter arrays in one loop. Because the list of
gradients is built in the same order as `_parameters()`, the `strict=True` zip
fails loudly if the two ever diverge:

```python
            for index, (parameter, gradient) in enumerate(
                    zip(parameters, gradients, strict=True)):
                moment1[index] = beta1 * moment1[index] + (1.0 - beta1) * gradient
                moment2[index] = beta2 * moment2[index] + (1.0 - beta2) * gradient ** 2
                corrected1 = moment1[index] / (1.0 - beta1 ** epoch)
                corrected2 = moment2[index] / (1.0 - beta2 ** epoch)
                parameter -= (learning_rate * corrected1
                              / (np.sqrt(corrected2) + epsilon))
```

The stack keeps two residual streams, one for the input and one for the target,
exactly as (1) requires:

```python
    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> _Stack:
        residual = inputs.copy()
        remaining = targets.copy()
        forecast = np.zeros_like(targets)
        for block in self.blocks:
            block.fit(residual, remaining, epochs=self.epochs,
                      learning_rate=self.learning_rate,
                      backcast_weight=self.backcast_weight)
            contribution = block.forecast(residual)
            forecast = forecast + contribution
            remaining = remaining - contribution
            residual = residual - block.backcast(residual)
        self.loss_ = float(np.mean((forecast - targets) ** 2))
        return self
```

The rate schedule is the variant definition. `hierarchical_interp` sets the
rate to one but keeps the halving grid; `multi_rate` keeps the rate but resets
every grid to the full horizon:

```python
        for index in range(self.blocks):
            power = 2 ** (self.blocks - 1 - index)
            rate = 1 if self.variant == "hierarchical_interp" else power
            rates.append(rate)
            if self.variant == "multi_rate":
                size = self.horizon
            elif self.variant == "hierarchical_interp":
                size = max(1, self.horizon // power)
            else:
                size = max(1, int(round(self.horizon / power)))
            coarse.append(min(size, self.horizon))
```

The three variants therefore share every line of code except this table, which
is the point of an ablation: the difference in accuracy is attributable to the
mechanism, not to a different training loop.

Prediction standardises the window with its own statistics, because the blocks
were trained on windows standardised the same way:

```python
        window = values[-self.seq_len:]
        mean = float(window.mean())
        spread = float(window.std())
        if spread <= 1e-8:
            spread = 1.0
        normalised = (window - mean) / spread
        forecast = self.stack_.predict(normalised[None, :])[0]
        return mean + spread * forecast
```

## Complexity

With $n$ observations, $W$ windows, $L$ look-back, $H$ horizon, $K$ blocks,
width $d$ and $E$ epochs, and writing $m_k=\lceil L/r_k\rceil$:

- **Windowing:** $O(W(L+H))$ time and memory.
- **Pooling a block:** $O(WL)$ comparisons regardless of $r_k$; the rate does
  not change the work, only the width of the weight matrix.
- **Forward pass of block $k$:** $O(W(m_kd+c_kd+c_kH))$; the first matrix
  dominates because $m_k$ is the widest input.
- **Backward pass:** about twice the forward cost for a one-hidden-layer net.
- **One block's training:** $O(E W m_k d)$.
- **Full fit:** $O(E W d\sum_k m_k)$, which is $O(E W d L)$ for rate one and
  about $O(E W d L)$ still for the geometric schedule, since
  $\sum_k L/2^k < 2L$. The saving is in constants and in the head widths.
- **Prediction:** $O(1)$ blocks over one window, independent of $n$.

The demo trains three stacks plus two structural checks in about $0.4$ seconds.
The interpolation matrix is $O(Hc_k)$ to build and $O(Hc_k)$ to apply, which is
negligible next to the hidden layer.

## Failure modes

**Pooling hides a level shift.** A max pool reports the largest value in each
group, so a downward step inside a group can disappear entirely. On series
with abrupt drops — a product going out of stock, a sensor failing — the coarse
blocks see a flat or rising window and the forecast misses the break. Use a
shorter window, a smaller rate, or an explicit break feature.

**Interpolation alone does not regularise.** The ablation is blunt about this:
`hierarchical_interp` has the best training error and the worst held-out error
of the three variants. Smoothing the output does not constrain the input path.
If pooling is undesirable for a series, control capacity some other way —
smaller width, fewer epochs, or weight decay — rather than relying on the
interpolation grid.

**Coarse grids cannot express a turning point.** With a grid of three
coefficients over twelve steps, the block's forecast is piecewise linear with
at most two kinks. A series that reverses direction inside the horizon needs a
finer block, and the schedule has to be tuned so that the fine blocks get
enough of the signal to correct the coarse ones.

**The residual chain is order-dependent.** Blocks are trained greedily, and the
coarsest block trains first on the full target. If it captures the wrong shape,
the later blocks inherit the error and spend their capacity correcting it. N-HiTS
mitigates this with the rate hierarchy, but it does not eliminate the
greediness: a jointly trained stack can reach a different, sometimes better,
decomposition.

**Long windows are still expensive in the fine block.** The last block has rate
one, so it carries the full $L\times d$ matrix. On a window of a thousand
steps the fine block dominates the parameter count and the fit time, which is
exactly the cost the hierarchy was meant to avoid. The practical fix is to cap
the window and let the coarse blocks cover the long range.

**The horizon is fixed at fit time.** As in N-BEATS, the interpolation matrix
has $H$ rows, so a model trained for twelve steps cannot emit twenty-four. Refit
with the desired horizon, or train a separate head per horizon.

## Applications

N-HiTS is used for the same long-horizon forecasting problems as N-BEATS —
electricity load, retail demand, traffic, and financial volatility — where a
long look-back window would otherwise make the first layer enormous. The
multi-rate idea is the reason it scales to windows of several hundred points,
which is where the plain N-BEATS parameter count becomes the bottleneck.

The hierarchical forecast is also useful for reporting: the coarse block's
three numbers are close to a trend and a level, and the finer blocks add the
wiggles. When a planner asks why the forecast bends upward in the last three
steps, the answer is in the finest block's coefficients.

It is the wrong tool for very short series, for intermittent demand, and for
problems where the level shift is the phenomenon of interest. It is also not
probabilistic: it gives a point forecast and, at best, an ensemble spread. For
intervals, pair it with conformal prediction or a probabilistic model.

## References

- Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco,
  M., Dubrawski, A. (2023). *N-HiTS: Neural hierarchical interpolation for time
  series forecasting.* AAAI. Multi-rate sampling, hierarchical interpolation,
  and the rate schedule behind (8).
- Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020). *N-BEATS:
  Neural basis expansion analysis for interpretable time series forecasting.*
  ICLR. The residual stack that N-HiTS inherits.
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). *M5 accuracy
  competition: results, findings and conclusions.* International Journal of
  Forecasting 38(4), 1346–1364. The benchmark where the N-HiTS family was
  competitive on long retail hierarchies.
- Kingma, D. P., Ba, J. (2015). *Adam: A method for stochastic optimization.*
  ICLR. The update shared with N-BEATS.
