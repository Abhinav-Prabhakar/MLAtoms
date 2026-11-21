# DeepAR — forecast a distribution, then sample from it

`deepar.py` · Time series · Tier 3 · 5 variants

## Intuition

A point forecaster answers "what is $y_{T+1}$?" A distributional forecaster
answers "what is $p(y_{T+1}\mid y_{\le T})$, and what does a plausible future
look like?" DeepAR is the second kind. It runs a recurrent network over the
series, and at every step the network emits the parameters of a probability
distribution for the next observation. Training maximises the likelihood;
forecasting draws from the network and feeds the draw back in.

That feedback is the whole idea. A one-step distribution does not tell you what
a twelve-month path looks like, because errors compound: a high draw today
changes the state tomorrow, which changes the mean and the width of tomorrow's
distribution. Sampling trajectories reproduces that compounding. The resulting
band is wider than the one you get by adding one-step variances, and it has the
right shape — skewed for counts, heavy-tailed for jumps — instead of being
Gaussian by assumption.

The network is deliberately small: one tanh layer of width sixteen. All the
modelling lives in the likelihood, and the likelihood is where the variants
differ.

- **`gaussian_likelihood`** is the textbook version: mean and standard
  deviation per step, Gaussian log-likelihood.
- **`deepar`** uses the same head but forecasts by Monte Carlo, feeding samples
  back into the state.
- **`negbin_likelihood`** replaces the Gaussian with a negative binomial, which
  is the right shape for counts: non-negative, integer, variance growing with
  the mean.
- **`student_t`** replaces the Gaussian with a heavy-tailed $t$, so a spike in
  the training data is treated as a plausible draw rather than an impossible
  one.
- **`quantile_output`** drops the likelihood entirely and trains five pinball
  losses, which is the distribution-free alternative.

On 348 training points and a 12-step hold-out the Gaussian model reaches a mean
absolute error of $0.6048$ against $0.8454$ for seasonal naive. The count model
reaches $1.0421$ against $1.7500$ for the same weekday last week. The $t$ head
on a spike-riddled series reaches $1.7220$ where the Gaussian head reaches
$2.7498$.

## Variants

| variant | likelihood | prediction | held-out MAE |
|:--|:--|:--|:--|
| `gaussian_likelihood` | Gaussian | analytic mean path | 0.6077 |
| `deepar` | Gaussian | Monte Carlo median | 0.6048 |
| `negbin_likelihood` | negative binomial | mean path | 1.0421 |
| `student_t` | Student $t$, unit scale | mean path | 1.7220 |
| `quantile_output` | five pinball losses | median path | 0.9626 |

**`gaussian_likelihood`** and **`deepar`** are the same fitted model. Both
minimise the Gaussian negative log-likelihood, both use the same seed, and the
demo asserts that their training losses are exactly equal. They diverge only at
prediction time: the first recurses on the conditional mean, the second draws
400 trajectories and takes the median. The paths differ by up to a unit on this
series, because the mean recursion ignores the variance that the sampling
accumulates.

**`negbin_likelihood`** is fitted to a weekly count series with a mean of
$3.971$. It reaches $1.0421$ against $1.7500$ for the same weekday last week,
and its 80% band covers $0.917$ of the hold-out while never going below zero —
a property no transformed Gaussian can guarantee.

**`student_t`** is fitted to the seasonal series with a spike of size eight
injected every seventh point. The $t$ head reaches $1.7220$ against $2.7498$
for the Gaussian head on the same data. The spikes stop distorting the fit
because the likelihood can explain them.

**`quantile_output`** trains five pinball losses at levels
$0.1,0.25,0.5,0.75,0.9$. It reaches $0.9626$ — better than the last value but
worse than seasonal naive. Five quantiles per step is a harder estimation
problem than two likelihood parameters, and on 347 transitions the difference
shows.

## Derivation

### The autoregressive factorisation

A series is modelled as a product of one-step conditionals:

$$
p(y_{1:T}\mid y_{1:0})=
\prod_{t=1}^{T}p\bigl(y_t\mid h_t\bigr),
\qquad h_t=f_\theta(h_{t-1},y_{t-1}).
\qquad (1)
$$

The state $h_t$ is a deterministic summary of everything before $t$. Nothing is
assumed about the shape of $p$ beyond the parametric family chosen for the
head. In particular the variance is a function of the state, so the model is
heteroscedastic by construction.

### The recurrence

With standardised observations $z_t=(y_t-\mu)/\sigma$ and a hidden width $d$:

$$
h_t=\tanh\!\bigl(w_x z_{t-1}+W_h h_{t-1}+b_h\bigr),
\qquad h_t\in\mathbb R^{d},
\qquad (2)
$$

$$
o_t=W_o h_t+b_o\in\mathbb R^{k},
\qquad (3)
$$

where $k$ is the number of head outputs: two for the Gaussian and the negative
binomial, one for the $t$, five for the quantile grid. The input at step $t$ is
the previous *observation*, not the previous mean. Teacher forcing during
training and free running during sampling is exactly the difference the two
modes implement.

### Gaussian likelihood

$$
\mu_t=o_{t,1},\qquad \sigma_t=\mathrm{softplus}(o_{t,2}),
\qquad (4)
$$

$$
\ell_t=\tfrac12\Bigl(\frac{z_t-\mu_t}{\sigma_t}\Bigr)^{2}
+\log\sigma_t+\tfrac12\log 2\pi .
\qquad (5)
$$

The softplus keeps $\sigma_t>0$ without a clamp, and its derivative
$\sigma'(a)=\mathrm{sigmoid}(a)$ is what turns the gradient in $\sigma$ into a
gradient in $a$:

$$
\frac{\partial\ell_t}{\partial o_{t,1}}
=-\frac{z_t-\mu_t}{\sigma_t^{2}},\qquad
\frac{\partial\ell_t}{\partial o_{t,2}}
=\frac{1-r_t^{2}}{\sigma_t}\,\mathrm{sigmoid}(o_{t,2}),
\qquad (6)
$$

with $r_t=(z_t-\mu_t)/\sigma_t$. The factor $1/\sigma_t$ in the second
expression is easy to drop. Doing so still trains — the model simply learns a
slightly different objective — which is why it survives code review. The
finite-difference check is the only reliable detector.

### Student-$t$ likelihood

$$
\ell_t=c_\nu+\tfrac{\nu+1}{2}\log\!\Bigl(1+\frac{(z_t-\mu_t)^{2}}{\nu}\Bigr),
\qquad
c_\nu=\log\Gamma\!\bigl(\tfrac{\nu+1}{2}\bigr)
-\log\Gamma\!\bigl(\tfrac{\nu}{2}\bigr)-\tfrac12\log(\nu\pi).
\qquad (7)
$$

$$
\frac{\partial\ell_t}{\partial\mu_t}
=-\frac{(\nu+1)(z_t-\mu_t)}{\nu+(z_t-\mu_t)^{2}} .
\qquad (8)
$$

The scale is fixed at one. That is not a convenience: with a free $\sigma$ the
$t$ likelihood is unbounded below. As $\sigma\to\infty$ the density at any fixed
point grows like $\sigma^{-1}$, so the negative log-likelihood decreases without
limit and the optimiser runs away. The demo's earlier version learned the scale
and produced an 80% band 79 units wide. Standardising the target fixes the
scale at the only value that is identified, and the heavy tail is what remains.

### Negative binomial likelihood

For counts with mean $\mu_t$ and dispersion $r_t$:

$$
p(y_t)=\frac{\Gamma(y_t+r_t)}{\Gamma(r_t)\Gamma(y_t+1)}
\Bigl(\frac{r_t}{r_t+\mu_t}\Bigr)^{r_t}
\Bigl(\frac{\mu_t}{r_t+\mu_t}\Bigr)^{y_t},
\qquad (9)
$$

$$
\mu_t=\exp(o_{t,1}),\qquad r_t=\exp(o_{t,2}),
\qquad (10)
$$

trained by the negative log-likelihood. The gradients need the digamma
function,

$$
\frac{\partial\ell_t}{\partial\mu_t}
=\frac{r_t+y_t}{r_t+\mu_t}-\frac{y_t}{\mu_t},
\qquad (11)
$$

and a similar expression in $r_t$. Both are implemented with a Lanczos
$\log\Gamma$ and a recurrence for $\psi$, so nothing outside numpy is needed.

The counts are **not** standardised. A count likelihood lives on the integers,
and feeding it a $z$-score is meaningless — the demo's first attempt did exactly
that and crashed inside $\log\Gamma$. The cell's input is still rescaled by the
series standard deviation, but the target and the output stay in count units.

### Quantile output

The head emits one number per level and the levels are made monotone by
construction, not by sorting:

$$
q_{t,1}=o_{t,1},\qquad
q_{t,k}=q_{t,k-1}+\mathrm{softplus}(o_{t,k}),
\qquad k=2,\dots,K .
\qquad (12)
$$

Each increment is positive, so the quantiles cannot cross and the gradient is
well defined everywhere. The loss is the pinball loss averaged over levels:

$$
\ell_t=\frac1K\sum_{k=1}^{K}
\rho_{\tau_k}\bigl(z_t-q_{t,k}\bigr),\qquad
\rho_\tau(u)=u\bigl(\tau-\mathbb 1\{u<0\}\bigr).
\qquad (13)
$$

Because $q_{t,k}$ is a cumulative sum, the gradient for $o_{t,k}$ is the sum of
the pinball gradients at all levels above it, scaled by the softplus
derivative:

$$
\frac{\partial\ell_t}{\partial o_{t,k}}
=\mathrm{sigmoid}(o_{t,k})\sum_{j\ge k}
\frac{\partial\ell_t}{\partial q_{t,j}} .
\qquad (14)
$$

### Backpropagation through time

With $g_t=\partial\ell_t/\partial o_t$, the state gradient obeys

$$
\delta_t=(1-h_t^{2})\odot\bigl(W_o^{\top}g_t+W_h^{\top}\delta_{t+1}\bigr),
\qquad \delta_{T+1}=0,
\qquad (15)
$$

and the parameter gradients are outer products accumulated over $t$:

$$
\frac{\partial L}{\partial W_h}=\sum_t h_{t-1}\delta_t^{\top},\qquad
\frac{\partial L}{\partial w_x}=\sum_t z_{t-1}\delta_t^{\top}.
\qquad (16)
$$

One detail matters in practice: $h_t$ enters the recurrence twice, once through
$W_h h_t$ at the next step and once through $W_o h_t$ in the output, and the
second term is what makes $\delta_t$ different from a plain RNN without a head.

### Monte Carlo versus analytic paths

The `deepar` forecast draws

$$
\tilde z_t\sim p\bigl(\cdot\mid h_t\bigr),\qquad
h_{t+1}=f_\theta(h_t,\tilde z_t),
\qquad (17)
$$

repeatedly and reports the empirical median at each horizon. The
`gaussian_likelihood` forecast instead plugs in the conditional mean:

$$
\bar z_t=\mathbb E[z_t\mid h_t],\qquad
h_{t+1}=f_\theta(h_t,\bar z_t),
\qquad (18)
$$

and approximates the predictive variance by accumulating one-step variances
along that mean path:

$$
v_h=\sum_{j=1}^{h}\sigma_j^{2},\qquad
\mathrm{band}_h=\bar z_h\pm z_{1-\alpha/2}\sqrt{v_h}.
\qquad (19)
$$

Equation (19) is not exact — it ignores the fact that the variance changes the
path — and the demo shows the consequence: the analytic band is $0.8535$ wide on
average where the sampled band is $1.4790$. The missing width is the
path-dependence that (17) captures and (18) discards.

### Gradient clipping

Each gradient entry is clipped to $[-5,5]$ before the Adam update. Without it,
a hidden state that saturates at $\pm1$ produces a spike in the backward pass
and the run diverges; with it, the same configuration trains. DeepAR's own
implementation does the same thing.

## Code walkthrough

Every likelihood returns a per-step loss and a per-step gradient in the head
output, which keeps the recurrent code independent of the distribution. The
Gaussian is the template:

```python
    mu = output[:, 0]
    activation = output[:, 1]
    sigma = _softplus(activation) + 1e-6
    standardised = (target - mu) / sigma
    loss = 0.5 * standardised ** 2 + np.log(sigma) + _HALF_LOG_2PI
    gradient = np.empty_like(output)
    gradient[:, 0] = -(target - mu) / sigma ** 2
    gradient[:, 1] = ((1.0 - standardised ** 2) * _sigmoid(activation)
                      / sigma)
    return loss, gradient
```

The `+ 1e-6` prevents a zero standard deviation from producing an infinite
gradient on the first step, when the head is still near zero and
$\sigma=\mathrm{softplus}(0)\approx0.69$.

The $t$ head needs only one output, because the scale is fixed:

```python
    offset = target - output[:, 0]
    constant = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                - 0.5 * math.log(nu * math.pi))
    loss = constant + 0.5 * (nu + 1.0) * np.log1p(offset ** 2 / nu)
    gradient = np.empty_like(output)
    gradient[:, 0] = -(nu + 1.0) * offset / (nu + offset ** 2)
    return loss, gradient
```

`np.log1p` rather than `np.log(1 + ·)` matters when the offset is small, which
is most of the time: it keeps full precision on the typical residuals instead
of losing it to cancellation.

The quantile head builds monotonicity into the parametrisation:

```python
    quantiles = np.empty_like(output)
    quantiles[:, 0] = output[:, 0]
    for index in range(1, quantiles.shape[1]):
        quantiles[:, index] = (quantiles[:, index - 1]
                               + _softplus(output[:, index]))
    error = target[:, None] - quantiles
    loss = np.where(error >= 0.0, levels * error, (levels - 1.0) * error)
```

The first output is the lowest quantile, not the median, and each later one is
the previous plus a positive increment. Sampling from this head is then a
one-dimensional interpolation, which is why the same `sample` method works for
all five variants.

The forward pass is a single loop that stores every state, because the backward
pass needs all of them:

```python
        for step in range(steps):
            states[step + 1] = np.tanh(inputs[step] @ p["w_x"]
                                       + states[step] @ p["W_h"] + p["b_h"])
            outputs[step] = states[step + 1] @ p["W_o"] + p["b_o"]
```

`inputs[step]` is a length-one vector, so the matrix product is a row times a
matrix — no batch axis. That is deliberate: DeepAR trains on one long sequence,
not on shuffled windows, and the whole series is the batch.

The backward loop is (15) and (16) written out:

```python
        carry = np.zeros(p["b_h"].size)
        for step in range(steps - 1, -1, -1):
            total = outputs_grad[step] @ p["W_o"].T + carry
            delta = total * (1.0 - states[step + 1] ** 2)
            g["w_x"] += inputs[step][:, None] * delta[None, :]
            g["W_h"] += states[step][:, None] @ delta[None, :]
            g["b_h"] += delta
            carry = delta @ p["W_h"].T
```

`carry` is $\delta_{t+1}$ propagated through $W_h$. The `states[step + 1]`
indexing is the easiest place to introduce an off-by-one: `states[0]` is the
initial zero state, so the state produced at step `t` lives at `t + 1`, and the
derivative of the tanh must use that same entry.

Sampling is a nested loop over trajectories and horizons, with the draw
assigned back to `current`:

```python
        for index in range(n_samples):
            current_state = state
            current = previous
            for step in range(horizon):
                current_state, output = self._step(current_state, current)
                current = self._draw(output, rng)
                paths[index, step] = current
```

Each trajectory starts from the same encoded state and the same last
observation, and diverges only through the draws. That is what makes the sample
a draw from the predictive distribution rather than a perturbation of a fixed
path.

The single place where model space and input space differ is the count model:

```python
        p = self.cell_.parameters
        feed = previous
        if self.variant == "negbin_likelihood":
            feed = (previous - self.input_mean_) / self.input_scale_
```

Continuous variants live in standardised units, so the previous value is
already what the cell wants. The count model keeps its observations in count
units because the likelihood demands it, and rescales only on the way into the
recurrence.

## Complexity

With sequence length $T$, hidden width $d$, $k$ head outputs, $E$ epochs and
$N$ sampled trajectories of length $H$:

- **One forward step:** $O(d^{2}+kd)$ — the state matrix dominates.
- **One epoch:** $O(T(d^{2}+kd))$ for the forward pass and the same again for
  the backward pass, so $O(Td^{2})$ in the width.
- **Full fit:** the above times $E$. For $T=347$, $d=16$, $E=300$ the demo's
  six fits take about five seconds.
- **Prediction:** $O(Hd^{2})$ for the mean path; $O(NHd^{2})$ for the samples.
- **Memory:** $O(Td)$ for the stored states, which is what makes the backward
  pass possible and what limits the sequence length.

The cost is linear in $T$, unlike attention, and quadratic in $d$, unlike a
wide MLP. For long series that is the right trade: the state carries the
history, so nothing needs to look back over the raw sequence.

Sampling multiplies the prediction cost by $N$. With 400 trajectories and a
12-step horizon the demo's band is computed in a few milliseconds, but a
production system sampling thousands of paths for a thousand series would feel
it. The analytic band in (19) exists precisely for that case.

## Failure modes

**The mean recursion can diverge.** Feeding the conditional mean back into the
network is not the same as feeding a draw, and the map can have a fixed point
far from the data. In development, a learning rate of $10^{-2}$ with 400 epochs
produced a mean path whose MAE was $4.5$ on a series whose seasonal naive error
is $0.85$. The training loss had also risen, so the failure is visible in the
loss — but only if you look. A stable configuration is a small learning rate, a
conservative hidden width, and gradient clipping.

**Long sequences make the gradient fragile.** Backpropagation through 347 steps
multiplies 347 Jacobians. A tanh state with a spectral radius near one keeps
them alive; above one they explode. Nothing in the architecture enforces
stability, so the initialisation and the clip do the work. Truncated
backpropagation through time over windows of 50–100 steps is the standard
remedy.

**A free scale makes the $t$ likelihood improper.** As shown in (7)–(8), the
negative log-likelihood of a $t$ with learned scale decreases as the scale grows,
because the log-density at a fixed point scales like $-\log\sigma$. The
optimiser finds this direction and the predictive band explodes — the demo's
first version produced a band 79 units wide on a series whose standard
deviation is about four. Fixing the scale, or bounding it, is not optional.

**Counts cannot be standardised.** The negative binomial is defined on
non-negative integers. Standardising the target turns counts into real numbers
and the likelihood into nonsense; the demo originally crashed inside
$\log\Gamma$ because a scaled count was negative. The input to the cell may be
rescaled, the target may not.

**Calibration needs more than twelve points.** The demo reports 80% band
coverage of $0.583$ for the Gaussian model, $0.917$ for the count model and
$0.500$ for the quantile model. On twelve points the standard error of a
coverage estimate is about $0.14$, so none of these is a calibration claim.
Coverage has to be measured over hundreds of windows and many series.

**The quantile head is data-hungry.** Five quantiles per step, each with its own
pinball loss, is more parameters to identify than a two-parameter likelihood on
the same 347 transitions. The demo's quantile median reaches $0.9626$ against
$0.8454$ for seasonal naive. It is the right tool when the shape is unknown and
the data are plentiful, not when the data are scarce.

**One series is one sample.** The model is trained on a single sequence, so
every gradient step sees the same $347$ transitions. That is fine for a demo
and wrong for production: DeepAR's advantage comes from training one model
across thousands of related series, where the shared weights learn the shape
and the state adapts per series.

**There is no seasonality prior.** The network must encode a period-twelve
pattern in its hidden state from scratch. A seasonal lag in the input, or a
seasonal naive residual as the target, would make the same model much easier to
fit. The demo's modest error against seasonal naive is the cost of leaving that
structure out.

## Applications

DeepAR's home is large-scale probabilistic forecasting: thousands of SKUs in
retail, thousands of meters in electricity, thousands of series in cloud
telemetry. One model is trained across all of them, with the series identity
carried by the state or by covariates, and the output is a distribution per
series per horizon. Decisions — safety stock, staffing, capacity — need the
distribution, not the mean.

The count head is used for demand that arrives in units: units sold per day,
calls per hour, failures per machine. The negative binomial captures the
over-dispersion that a Poisson misses, and the non-negativity is structural
rather than clipped.

The heavy-tailed head is used where spikes are real: promotional sales,
traffic incidents, sensor glitches. Treating a spike as an outlier to be removed
throws away information; treating it as a $t$ draw keeps it and stops it from
dominating the fit.

The quantile head is used when the shape is unknown or the loss is asymmetric.
Predicting a 0.9 quantile directly optimises the quantity a newsvendor cares
about, and it does not require the distribution to belong to a named family.

It is the wrong tool for a single short series with a strong seasonal pattern
and no covariates, where a state-space model or a seasonal naive is a better
use of the data, and for very long horizons where the recursion's compounding
error dominates.

## References

- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). *DeepAR:
  Probabilistic forecasting with autoregressive recurrent networks.*
  International Journal of Forecasting. The architecture, the likelihood heads
  and the sampling procedure.
- Flunkert, V., Salinas, D., Januschowski, T. (2017). *DeepAR: Probabilistic
  forecasting with autoregressive recurrent networks.* arXiv:1704.04110. The
  original technical report.
- Koenker, R., Bassett, G. (1978). *Regression quantiles.* Econometrica. The
  pinball loss and the quantile regression objective in (13).
- Cameron, A. C., Trivedi, P. K. (2013). *Regression analysis of count data.*
  Cambridge University Press. The negative binomial parametrisation and its
  score in (11).
- Bergmeir, C., Hyndman, R. J., Koo, B. (2018). *A note on the validity of
  cross-validation for evaluating autoregressive time series prediction.*
  Computational Statistics and Data Analysis. Why a single hold-out window is
  not enough to judge a forecaster.
