# N-BEATS — residual basis expansion with small neural blocks

`nbeats.py` · Time series · Tier 3 · 4 variants

## Intuition

Most neural forecasters win by being large. N-BEATS wins by being small and
stacked. A single block is a two-layer perceptron with a few thousand weights
and two linear heads: one head reconstructs the input window, the other emits
the forecast. Blocks are then chained, and each one sees only the part of the
window that the previous blocks failed to reconstruct. The final forecast is
the sum of every block's contribution.

The trick is that both heads are *linear in a basis*. The perceptron does not
output the forecast directly; it outputs coefficients, and a fixed matrix turns
those coefficients into a backcast and a forecast. With an identity basis the
block can produce anything, which is the `generic` variant. With a polynomial
basis and a Fourier basis the coefficients become a trend slope and a set of
seasonal amplitudes, which is the `interpretable` variant — the same
architecture, but the forecast is now readable.

The residual chain is what makes the stack behave like a decomposition. Block
one removes the trend it can reconstruct; block two sees a detrended window and
removes the seasonality; block three sees what is left. Nothing forces the
blocks to specialise, but the architecture nudges them that way, and when the
bases are interpretable the split is usually obvious.

The four variants cover the paper's main configurations:

- **`generic`** uses identity bases, so every coefficient is free.
- **`interpretable`** alternates polynomial trend blocks and Fourier seasonal
  blocks, producing an additive trend-season forecast.
- **`nbeatsx`** adds a ridge head on exogenous variables that are known for the
  forecast window but are not in the target's own history.
- **`ensemble`** averages several independently initialised stacks.

The demo trains on 348 points of a trend plus two harmonics and holds out 12.
The generic stack reaches a held-out MAE of $0.2466$ against $0.4627$ for
seasonal naive. The interpretable stack reaches $0.3779$ — it pays about
$0.13$ for a forecast you can read.

## Variants

| variant | bases | demo result |
|:--|:--|:--|
| `generic` | identity | MAE 0.2466, train MSE 0.0279 |
| `interpretable` | trend and Fourier | MAE 0.3779, train MSE 0.0642 |
| `nbeatsx` | generic plus ridge head | MAE 1.0332 vs 2.9888 |
| `ensemble` | three generic stacks | MAE 0.2408 vs 0.2466 |

**`generic`** trains two blocks on 301 sliding windows. The in-sample mean
squared error is $0.0279$ in units of the per-window variance, so the stack
explains over 97% of the variance of the standardised windows. The held-out
MAE of $0.2466$ is close to the simulated noise level ($0.25$ per point in
original units), and it beats both naive baselines by a wide margin.

**`interpretable`** uses four alternating blocks with a linear trend basis and
two Fourier harmonics. Its train MSE is $0.0642$ — more than twice the generic
error — and its held-out MAE is $0.3779$. The constrained basis cannot bend to
absorb the second harmonic as flexibly as a free linear map, so the parametric
model is genuinely a compromise: a readable decomposition at a measurable cost.

**`nbeatsx`** is fitted to the same trend-and-season signal plus a driver that
is drawn independently at every step and enters with a coefficient of three.
The driver is invisible in the target's past, so the blocks cannot anticipate
it; only the exogenous head can. Held-out MAE falls from $2.9888$ without the
driver to $1.0332$ with it, and the fitted head is
$[-0.0371,\ 2.7271]$ on the standardised driver against a true effect of three.

**`ensemble`** averages three stacks that differ only in their initial weights.
Member MAEs are $0.2466$, $0.2277$ and $0.2364$; the ensemble reaches $0.2408$
against a member mean of $0.2369$. Averaging beats the worst seed and beats the
first seed, but it does not beat the average member — the members are highly
correlated, so the variance reduction is modest.

## Derivation

### The residual stack

Let $x_0=Y_{1:L}$ be the standardised look-back window and $y$ the target
window. A stack of $K$ blocks produces

$$
\hat y=\sum_{k=1}^{K} f_k(x_{k-1}),
\qquad x_k=x_{k-1}-b_k(x_{k-1}),
\qquad (1)
$$

where $f_k$ is block $k$'s forecast and $b_k$ its backcast. The backcast is a
reconstruction of the block's own input, not of the target: subtracting it
means each block passes forward only the structure it could not explain.

### One block

Hidden layers are plain ReLU perceptrons. With two hidden layers,

$$
h_1=\mathrm{ReLU}(W_1x+b_1),\quad
h_2=\mathrm{ReLU}(W_2h_1+b_2),
\qquad (2)
$$

and the two heads are linear maps of the last hidden layer:

$$
\theta^{f}=W_fh_2+c_f,\qquad \theta^{b}=W_bh_2+c_b .
\qquad (3)
$$

The forecast and backcast are the coefficients times fixed basis matrices:

$$
f=V^{f}\theta^{f},\qquad b=V^{b}\theta^{b},
\qquad (4)
$$

with $V^{f}\in\mathbb R^{H\times d_f}$ and $V^{b}\in\mathbb R^{L\times d_b}$.
Only $\theta$ is learned. Choosing $V$ is the entire difference between the
generic and interpretable variants.

### The block loss

Because the head is linear, the block's objective is least squares in
coefficient space plus a backcast penalty:

$$
L_k=\frac{1}{H}\lVert f_k-y_k\rVert_2^2
+\lambda\,\frac{1}{L}\lVert b_k-x_{k-1}\rVert_2^2,
\qquad (5)
$$

where $y_k$ is the target still unexplained when block $k$ is trained and
$\lambda$ is `backcast_weight`. Both terms are averaged over the batch so that
the gradient scale does not depend on the number of windows.

### Sequential training

The implementation trains the blocks greedily rather than jointly. Each block
is fitted, then the running forecast and the running target are both reduced:

$$
y_{k+1}=y_k-f_k(x_{k-1}),\qquad x_k=x_{k-1}-b_k(x_{k-1}).
\qquad (6)
$$

Reducing the target is the step that is easy to get wrong. If every block is
trained against the original target, the forecasts add up and the stack
over-predicts; the loss rises with every additional block. Training against
$y_k$ makes the contributions complementary by construction.

### The generic basis

For `generic`, both bases are identities:

$$
V^{f}=I_H,\qquad V^{b}=I_L .
\qquad (7)
$$

The coefficients are then the forecast itself, so the linear head is just
another fully connected layer. This is the most flexible setting and the one
the paper uses for its pure accuracy benchmark.

### The trend basis

For a trend block of degree $p$, let $u_t=(t-L)/L$ so that the last observed
step is $u=0$ and the previous steps are negative. Then

$$
V^{b}_{t,j}=u_t^{\,j},\qquad V^{f}_{h,j}=\Bigl(\frac{h}{L}\Bigr)^{j},
\qquad j=0,\dots,p .
\qquad (8)
$$

Both bases live in the same variable, so a linear coefficient vector continues
the line into the future instead of restarting it. An earlier version placed
the backcast at $t/L\in(0,1]$ and the forecast at $(L+h)/L\approx10$; the two
scales were so different that the polynomial coefficients had to cancel almost
exactly, and the fit was numerically useless. Anchoring both at the end of the
window fixes that.

### The seasonal basis

For a seasonal block with period $m$ and $J$ harmonics,

$$
V^{b}_{t,2j-1}=\cos\frac{2\pi jt}{m},\quad
V^{b}_{t,2j}=\sin\frac{2\pi jt}{m},
\qquad (9)
$$

and the forecast basis uses the same formula at $t=L+1,\dots,L+H$. The phase is
therefore continuous across the boundary between the window and the horizon,
which is what lets a block extrapolate a cycle it saw in the past.

### Standardisation

Each window is centred and scaled by its own statistics before it reaches the
network:

$$
\tilde x=\frac{x-\bar x}{s_x},\qquad
\tilde y=\frac{y-\bar y}{s_x},
\qquad (10)
$$

with $s_x$ the window standard deviation, floored at $10^{-8}$. Using the
input window's scale for the target keeps the signal-to-noise ratio of a
window intact: a quiet window and a volatile window both become unit-variance
problems, and the same weights serve both.

### Backpropagation through a ReLU layer

With $\delta_\ell$ the gradient of the loss with respect to the pre-activation
$z_\ell$,

$$
\delta_\ell=\bigl(W_{\ell+1}^{\top}\delta_{\ell+1}\bigr)\odot
\mathbf 1[z_\ell>0],\qquad
\frac{\partial L}{\partial W_\ell}=\delta_\ell a_{\ell-1}^{\top},
\qquad (11)
$$

where $a_{\ell-1}$ is the previous activation. The code first forms the
gradient with respect to the last hidden layer by combining the two heads,
then walks backwards applying (11) once per layer.

### Adam

Every parameter is updated by

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\qquad (12)
$$

$$
\theta_t=\theta_{t-1}-\eta\,
\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon},
\qquad (13)
$$

with $\beta_1=0.9$, $\beta_2=0.999$ and $\epsilon=10^{-8}$. The bias
correction matters here because every block is trained for a few hundred
full-batch steps, not hundreds of thousands; without it the first steps are
far too large.

### The exogenous head

For `nbeatsx`, let $X$ stack the standardised exogenous rows that lie inside
each forecast window and $r$ the residual $y-\hat y$ of the block stack. The
head is

$$
\hat\beta=\arg\min_\beta \lVert r-X\beta\rVert_2^2+\lambda\lVert \beta\rVert_2^2,
\qquad \beta=(X^{\top}X+\lambda P)^{-1}X^{\top}r,
\qquad (14)
$$

with $P$ the identity except for a zero at the intercept. The fit alternates:
fit the head, subtract its contribution from the target, retrain the stack,
refit the head. Three rounds bring the coefficient from $1.44$ (one round) to
$2.73$ against a true effect of three, because later rounds stop asking the
blocks to memorise a driver they cannot generalise.

### The ensemble

The `ensemble` variant trains $S$ stacks that differ only in their seeds and
reports

$$
\hat y^{\mathrm{ens}}=\frac{1}{S}\sum_{s=1}^{S}\hat y^{(s)}.
\qquad (15)
$$

## Code walkthrough

The basis is the only place where the variants differ. The trend branch
anchors the polynomial at the end of the window so that both bases share one
variable:

```python
        scale = float(length)
        back = (np.arange(1, length + 1, dtype=float) - length) / scale
        fore = np.arange(1, horizon + 1, dtype=float) / scale
        powers = range(degree + 1)
        return (np.column_stack([back ** power for power in powers]),
                np.column_stack([fore ** power for power in powers]))
```

The seasonal branch evaluates the same Fourier functions at absolute times on
both sides of the boundary, which is what makes the phase continuous:

```python
    back = 2.0 * np.pi * np.arange(1, length + 1, dtype=float) / period
    fore = 2.0 * np.pi * np.arange(length + 1, length + horizon + 1,
                                   dtype=float) / period
```

The forward pass keeps every pre-activation and activation, because the
backward pass needs both: the activation for the weight gradient and the sign
of the pre-activation for the ReLU mask:

```python
    def _forward(self, values: np.ndarray):
        activations = [values]
        pre = []
        current = values
        for weight, bias in zip(self.weights, self.biases, strict=True):
            current = current @ weight.T + bias
            pre.append(current)
            current = np.maximum(current, 0.0)
            activations.append(current)
        theta_fore = current @ self.head_fore.T + self.offset_fore
        theta_back = current @ self.head_back.T + self.offset_back
        return theta_fore, theta_back, activations, pre
```

Both heads read the same hidden state, which is the point: a block must
reconstruct its input with one head while forecasting the target with the
other, and the shared representation forces those two jobs to agree.

The backward pass starts at the heads and folds the two gradients into one
before entering the stack. The loop is (11) applied in reverse:

```python
            top = activations[-1]
            weight_grads: list[np.ndarray | None] = [None] * self.depth
            bias_grads: list[np.ndarray | None] = [None] * self.depth
            hidden = (theta_fore_grad @ self.head_fore
                      + theta_back_grad @ self.head_back)
            for layer in range(self.depth - 1, -1, -1):
                delta = hidden * (pre[layer] > 0.0)
                weight_grads[layer] = delta.T @ activations[layer]
                bias_grads[layer] = delta.sum(axis=0)
                if layer > 0:
                    hidden = delta @ self.weights[layer]
```

In-place `parameter -= ...` works because the parameters are numpy arrays
stored in a list; the lists of moments are built with `np.zeros_like`, so the
shapes cannot drift out of sync with the weights:

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

The stack is where the residual idea lives. `remaining` tracks the target the
next block still has to explain; `residual` tracks the input it still sees.
Both are reduced by the same block's output, and the forecast accumulates:

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

Prediction is the same loop without training and without the target: only the
backcast chain survives, because there is no future target to compare against:

```python
    def predict(self, values: np.ndarray) -> np.ndarray:
        residual = values.copy()
        output = np.zeros((values.shape[0], self.horizon))
        for block in self.blocks:
            output = output + block.forecast(residual)
            residual = residual - block.backcast(residual)
        return output
```

The exogenous head alternates between the regression and the blocks. The
design matrix and its Gram matrix do not depend on the network, so they are
computed once outside the loop:

```python
        adjusted = targets.copy()
        contribution = np.zeros_like(targets)
        for _ in range(3):
            self.stack_ = self._new_stack(self.seed).fit(inputs, adjusted)
            base = self._normalised_forecasts(inputs) * scale + mean
            residual = targets * scale + mean - base
            coef = np.linalg.solve(gram, design.T @ residual.reshape(-1))
            contribution = (design @ coef).reshape(targets.shape)
            adjusted = targets - contribution / scale
```

The `gram` matrix already contains the ridge penalty, so the solve is a single
linear system per round. Subtracting `contribution / scale` converts the
original-unit correction back into the standardised space the blocks train in.

At prediction time the window is standardised with *its own* statistics, not
with the training statistics, because that is what the blocks were trained on:

```python
        window = values[-self.seq_len:]
        mean = float(window.mean())
        spread = float(window.std())
        if spread <= 1e-8:
            spread = 1.0
        normalised = (window - mean) / spread
        forecast = self._normalised_forecasts(normalised[None, :])[0]
        out = mean + spread * forecast
        if self.variant == "nbeatsx":
            design = self._exog_design(exog)
            out = out + design @ self.exog_coef_
```

Using the training mean and scale here would break on a series with a slow
level shift, and using the global standard deviation would make the forecast
scale with the training period rather than with the recent window.

## Complexity

With $n$ observations, $W$ windows, $L$ look-back, $H$ horizon, $K$ blocks,
$P$ parameters per block and $E$ epochs:

- **Windowing:** $O(W(L+H))$ time and memory to materialise the sliding
  windows.
- **One forward pass:** $O(W(LP+H d_f+L d_b))$, dominated by the first
  weight matrix because $L$ is the largest dimension.
- **One epoch:** roughly three forward passes' worth of work for the backward
  pass, so about $4\times$ the forward cost.
- **One block:** $O(EWLP)$; the demo's 301 windows, $L=36$, width 24 and 150
  epochs take a few milliseconds.
- **Full fit:** $O(KEWLP)$ for a stack, multiplied by $S$ for the ensemble and
  by three for the alternating exogenous rounds of `nbeatsx`.
- **Prediction:** $O(KLP)$, independent of the training size.

The whole demo — ten stack trainings including the deeper one — runs in
about $1.5$ seconds in pure numpy. The cost is linear in the number of blocks
but cubic in the width only through the hidden-to-hidden matrices, which are
small.

## Failure modes

**The residual target is the easy bug.** Training every block against the
original target makes the stack additive in the wrong way: each block predicts
the whole signal, the forecasts are summed, and the error grows with depth.
The symptom is a training loss that increases as blocks are added. The fix is
the `remaining` update in (6); after it, adding a block can only help the
in-sample fit.

**The basis can be numerically catastrophic.** A trend basis evaluated at
$t/L$ on the window and $(L+h)/L$ on the horizon spans a factor of ten in
scale. The coefficients must cancel to four or five digits, so the forecast is
dominated by rounding and by any slight miscalibration. Anchoring both bases at
the end of the window, as in (8), is not cosmetic; without it the interpretable
model is worse than seasonal naive.

**Depth buys in-sample fit, not accuracy.** Four blocks of width 32 reach a
train MSE of $0.0158$ against $0.0279$ for two blocks of width 24, but their
held-out MAE rises from $0.2466$ to $0.3396$. The blocks have enough capacity
to memorise individual windows. Early stopping, a wider validation split, or a
smaller width is cheaper than adding blocks.

**The exogenous head is only as good as the alternation.** Fitting the head
once on the residuals of a fully trained stack gives a coefficient of about
half the truth, because the blocks have already memorised part of the driver
in sample. The three-round alternation in (14) largely fixes this, but the
coefficient is still attenuated ($2.73$ against $3.00$), and a driver that
interacts with the target nonlinearly will not be captured by a linear head at
all.

**Ensembles do not fix a biased member.** Averaging reduces variance, not bias.
If every seed overfits the same windows, the members are correlated and the
ensemble is barely better than one of them: the demo's improvement is
$0.2408$ against a member mean of $0.2369$. Gain comes from diversity in the
initialisation and architecture, not from the number of seeds alone.

**One window's scale is one window's noise.** Standardising each window by its
own standard deviation makes the model robust to level and amplitude changes,
but a window that happens to be unusually quiet gets amplified. On series with
bursty variance this produces unstable forecasts near the end of the sample;
clipping the scale, or standardising by a longer rolling estimate, is the usual
remedy.

**The horizon is baked into the basis.** A model trained with $H=12$ has a
forecast basis of twelve columns and cannot emit a different horizon. Recursive
multi-step forecasting with a one-step model is an alternative, but it
compounds errors, and it discards the direct multi-horizon structure that makes
N-BEATS accurate.

## Applications

N-BEATS placed first in the M4 competition's pure machine-learning track and
is used in production for retail demand, electricity load and financial
volatility, usually as a strong benchmark that classical models must beat. The
`interpretable` configuration is attractive where a forecast has to be
explained to a planner: the trend block gives a slope, the seasonal block gives
amplitudes, and the remainder is visible.

The `nbeatsx` extension fits settings where future covariates are known —
promotions and price plans in retail, calendar and weather forecasts in energy,
scheduled maintenance in industrial telemetry. The covariate path is the reason
the method is used instead of a pure autoregression: the model can react to a
known future shock that never appears in its own history.

The method is the wrong tool when the series is very short (a few dozen
points, where a neural network has nothing to learn), when the data are
intermittent (use Croston or TSB), or when a fully probabilistic forecast is
required. It produces point forecasts and, at best, an ensemble spread; it does
not give a likelihood. For intervals, combine it with conformal prediction or
use a probabilistic model such as DeepAR.

## References

- Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. (2020). *N-BEATS:
  Neural basis expansion analysis for interpretable time series forecasting.*
  ICLR. The block, the residual stack, and the generic and interpretable
  bases.
- Oreshkin, B. N., Dudek, G., Pelka, P., Turkina, E. (2021). *N-BEATS neural
  network for interpretable time series forecasting.* arXiv:2105.12958. The
  extended treatment of the basis functions and the ensemble protocol.
- Olivares, K. G., Challú, C., Marcjasz, G., Weron, R., Dubrawski, A. (2023).
  *Neural basis expansion analysis with exogenous variables: Forecasting
  electricity prices with NBEATSx.* International Journal of Forecasting
  39(3), 1080–1095. The exogenous pathway behind (14).
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). *The M4
  Competition: 100,000 time series and 61 forecasting methods.* International
  Journal of Forecasting 36(1), 54–74. The benchmark where N-BEATS was the
  strongest pure-ML entry.
- Kingma, D. P., Ba, J. (2015). *Adam: A method for stochastic optimization.*
  ICLR. The update in (12)–(13).
