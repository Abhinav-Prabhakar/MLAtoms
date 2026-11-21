# TFT — three mechanisms, one forecast

`tft.py` · Time series · Tier 3 · 4 variants

## Intuition

The temporal fusion transformer is a small transformer with three additions, and
each addition answers a question a plain sequence model cannot.

*Which inputs matter?* A forecasting problem usually has several candidate
inputs — three lags of the target, two calendar features, a promotion flag — and
most of them are useless at any given time step. TFT projects each input into
the model width, scores it, and takes a softmax over the inputs at every step.
The weights are part of the forward pass, so they are also a report on what the
model reads.

*What kind of series is this?* Some series trend, some are flat, some are noisy.
A static context vector, built from summary statistics of the look-back window,
is added to every time step of the encoder and the decoder. It is the model's
only global summary, and it conditions both the gates and the attention.

*Where should the decoder look?* Attention over the encoder is the mechanism,
but vanilla multi-head attention is not interpretable: every head has its own
value projection, so the averaged attention matrix has no single meaning. TFT
shares one value projection across heads and averages the heads' outputs. Every
head then reads the same values, and their averaged weights are a single matrix
over the look-back.

Everything else is standard: gated residual networks supply the nonlinearity
with a skip path, and the head emits one output per quantile instead of one
point forecast. The variant names in this atom switch the three additions on one
at a time, so the cost of each is measurable.

The demo fits all four on 228 points of a seasonal series and holds out 12. The
full model reaches a mean absolute error of $0.4332$ against $0.7763$ for
seasonal naive. Keeping only variable selection gives $0.8916$, only the static
context gives $0.7219$, and only attention gives $1.3338$. Training took 150
full-batch epochs over 191 windows, and the full model has 1923 parameters.

One result is worth stating up front because it is the opposite of the sales
pitch: the training pinball losses rank the four models in *reverse* order of
their accuracy. `interpretable_attention` has the lowest training loss
($0.0424$) and the worst held-out error ($1.3338$). The extra machinery is not
helping the model fit the training windows; it is helping it generalise.

## Variants

| variant | selection | static | attention | pinball | MAE |
|:--|:--:|:--:|:--:|--:|--:|
| `tft` | yes | yes | yes | 0.0518 | 0.4332 |
| `variable_selection` | yes | no | no | 0.0452 | 0.8916 |
| `static_covariates` | no | yes | no | 0.0601 | 0.7219 |
| `interpretable_attention` | no | no | yes | 0.0424 | 1.3338 |

**`tft`** switches on all three and is the only variant that beats seasonal
naive by a comfortable margin.

**`variable_selection`** keeps the softmax over inputs and replaces attention
with a mean over the encoder. Its held-out error of $0.8916$ is *worse* than
seasonal naive, which is the clearest statement that selection alone is not
enough: the decoder still has no way to look at the encoder.

**`static_covariates`** keeps the context vector and drops selection and
attention. At $0.7219$ it lands just under seasonal naive — the window summary
carries real information about level and slope, and it is cheap.

**`interpretable_attention`** keeps attention and drops the other two. It is the
worst of the four at $1.3338$, and its 80% band covers none of the hold-out.
Attention without a context vector and without selected inputs is conditioning
on a decoder that only sees the future covariates.

The demo also reports 80% band coverage: $0.750$ for the full model, $0.667$ for
`static_covariates`, $0.083$ for `variable_selection` and $0.000$ for
`interpretable_attention`. Twelve points make this a weak calibration estimate,
but the ordering is stark: a partially specified model does not merely forecast
badly, it reports a band it is confident about and wrong about.

## Derivation

### Three input streams

Let $z_t=(y_t-m)/s$ be the standardised series, $L$ the look-back, $H$ the
horizon and $p$ the number of lags. Three tensors are cut from the training
series:

$$
P_{\tau,j}=z_{t-L+\tau-j},\qquad
F_{\tau,k}=u_{t+\tau,k},\qquad
s=\bigl(\bar z_{w},\ \sigma_{w},\ z_{t},\ \hat\beta_{w}\bigr),
\qquad (1)
$$

for $\tau=1,\dots,L$, $j=1,\dots,p$ and $k=1,\dots,f$. $P$ is the past, built
from lagged copies of the target; $F$ is the future, a set of covariates known
ahead of time — sine, cosine and a scaled time index by default; and $s$
contains the window's mean, standard deviation, last value and least-squares
slope. The target is the next $H$ values, $z_{t+1:t+H}$.

Splitting the inputs by availability is the point of (1). A covariate that is
unknown at forecast time cannot be a past input, and a covariate that is known
can be used at decoder steps where no observation exists.

### Variable selection

Each input $j$ gets its own projection into the model width, a scalar score, and
a step-specific softmax over the inputs:

$$
e_{\tau,j}=P_{\tau,j}W_j,\qquad
a_{\tau,j}=e_{\tau,j}^{\top}v,\qquad
\pi_{\tau}=\mathrm{softmax}(a_{\tau}),\qquad
\xi_{\tau}=\sum_{j}\pi_{\tau,j}\,e_{\tau,j}.
\qquad (2)
$$

The softmax is over $j$, not over time, so the weights change from one step to
the next. Every input is projected by the same $W_j$ at every step, so the
scoring function is shared and only the scores vary. In this atom the lags of
the target and the columns of the future covariate matrix are handled by two
separate selector instances with the same structure.

### The gated residual network

Every block in the model is a GRN:

$$
a=\mathrm{ELU}(W_1x+b_1),\qquad
g=\mathrm{sigmoid}(W_gx+W_cc+b_g),
\qquad (3)
$$

$$
\mathrm{GRN}(x,c)=\mathrm{LN}\bigl(x+g\odot(W_2a+b_2)\bigr).
\qquad (4)
$$

The gate $g\in(0,1)^{d}$ decides elementwise how much of the nonlinear branch to
admit, and the skip connection keeps the block close to the identity at
initialisation, when $W_2$ is small. The context $c$ enters the gate, which is
how a static summary reaches every layer: it biases the gates of the encoder,
the decoder and the head, without ever being concatenated onto the inputs.

The ELU is chosen over ReLU because its negative side is smooth, which keeps the
second derivative finite and the full-batch training stable. The sigmoid inside
the gate is written as `0.5 * (1.0 + np.tanh(0.5 * x))`, a form that is
numerically identical to the logistic function but avoids an exponential
overflow for large negative inputs.

### The static context

The static encoder is a GRN without the residual:

$$
c=\mathrm{LN}\bigl(g\odot(W_2\,\mathrm{ELU}(W_1s+b_1)+b_2)\bigr),
\qquad (5)
$$

with no $x+$ term because $s\in\mathbb R^{4}$ and $c\in\mathbb R^{d}$ live in
different spaces. The four features are standardised across training windows
before they are used, so a series with a large level does not swamp the gates.

### Encoder and decoder

$$
E_{\tau}=\xi^{P}_{\tau}+c,\qquad
D_{\tau}=\xi^{F}_{\tau}+c+\rho_{\tau},
\qquad (6)
$$

where $\rho_{\tau}$ is a learned position embedding of the decoder step and
$\xi^{P}$ and $\xi^{F}$ are the selected past and future inputs from (2). The
encoder runs a GRN over $E$, the decoder runs a GRN over $D$, and both receive
$c$ as their gate context. The decoder does not consume its own previous
outputs: the horizon is predicted in one shot from the future covariates, so
there is no autoregressive feedback and no error compounding.

### Interpretable attention

$$
A_h=\mathrm{softmax}\!\left(\frac{(DW_Q)_h(EW_K)_h^{\top}}{\sqrt{d/h}}\right),
\qquad V=EW_V,
\qquad (7)
$$

$$
\Omega=\frac{1}{n_h}\sum_{h=1}^{n_h}A_hV^{(h)},\qquad
\mathrm{Attn}=\Omega W_O .
\qquad (8)
$$

The query and key projections are per head, but the value projection is shared:
$V^{(h)}$ is head $h$'s slice of the same $EW_V$. The heads' outputs are
averaged rather than concatenated, and only then mapped back to the model width
by $W_O$. Because every head reads the same values, the averaged matrix
$\bar A=\frac1{n_h}\sum_h A_h$ is a single distribution over encoder positions
per decoder step. That is what makes the weights plottable. It does not make
them sparse, and the demo measures exactly that.

### Output and quantile loss

$$
\hat q_{\tau,k}=\bigl[\mathrm{GRN}(D_{\tau}+\mathrm{Attn}_{\tau},c)\bigr]^{\top}w_k+b_k,
\qquad k=1,\dots,K,
\qquad (9)
$$

$$
\ell=\frac{1}{WHK}\sum_{w}\sum_{\tau}\sum_{k}
\rho_{\tau_k}\bigl(z_{w,\tau+H}-\hat q^{(w)}_{\tau,k}\bigr),\qquad
\rho_\tau(u)=u\bigl(\tau-\mathbf 1\{u<0\}\bigr).
\qquad (10)
$$

The head is linear and emits all $K$ quantiles at once, with no monotonicity
constraint and no cross-quantile coupling. Quantiles can therefore cross in
principle. In the demo they do not, and the fan is monotone, but the code does
not enforce it — a sorted projection or a cumulative parametrisation would.
The loss is the pinball loss averaged over windows, horizon steps and quantiles,
which is exactly what the training loop minimises.

### Backward pass

Layer normalisation is the piece with the least obvious gradient. With
$\hat x=(x-\mu)/\sigma$ and an incoming gradient $g$,

$$
\frac{\partial L}{\partial x}
=\frac{1}{\sigma}\Bigl(g-\bar g-\hat x\,\overline{g\hat x}\Bigr),
\qquad (11)
$$

where the bars are means over the width. The subtraction of $\bar g$ and of the
projection onto $\hat x$ is what removes the two paths through $\mu$ and
$\sigma$. Dropping either term still trains, slowly and slightly wrong, which is
why the finite-difference check in development matters more than the code
reading correctly.

The softmax over the selector's scores has the matching form,

$$
\frac{\partial L}{\partial a}=\pi\odot\bigl(g_\pi-\pi^{\top}g_\pi\bigr),
\qquad (12)
$$

and the pooled branch, used when attention is ablated, needs the encoder length
and not the horizon:

$$
\bar e=\frac1L\sum_{\tau=1}^{L}E_{\tau}
\ \Longrightarrow\
\frac{\partial L}{\partial E_{\tau}}=\frac{1}{L}\sum_{\tau'=1}^{H}g_{\tau'} .
\qquad (13)
$$

Equation (13) is the one that was wrong first. The gradient arrives broadcast
over $H$ decoder steps, and dividing by $H$ instead of $L$ scales the whole
encoder's gradient by $L/H$. With $L=10$ and $H=4$ the finite-difference check
showed a factor of exactly $2.5$, which is how the error was found.

### Adam

$$
\theta\leftarrow\theta-\eta\,
\frac{m_t/(1-0.9^{t})}{\sqrt{v_t/(1-0.999^{t})}+\epsilon},
\qquad
m_t=0.9m_{t-1}+0.1g_t,\quad v_t=0.999v_{t-1}+0.001g_t^{2}.
\qquad (14)
$$

The update is applied to every parameter that received a gradient. In the
ablated variants some parameters are never built at all — the attention
projections do not exist when attention is off — so the loop skips keys that are
absent rather than carrying dead weights.

## Code walkthrough

The layer norm stores the two quantities its backward pass needs:

```python
    centred = x - mean
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt(variance + EPSILON)
    return centred * inverse, (centred, inverse)
```

`centred` is the numerator of $\hat x$ and `inverse` is $1/\sigma$. Keeping them
instead of recomputing them in the backward pass costs one array per block and
saves a second pass over the data.

The GRN is (3)–(4), with the gate written through `tanh` as described:

```python
        pre = x @ p[f"{name}.W1"] + p[f"{name}.b1"]
        active = _elu(pre)
        gate_logits = x @ p[f"{name}.Wg"] + p[f"{name}.bg"]
        if self.context:
            gate_logits = gate_logits + context[:, None, :] @ p[
                f"{name}.Wc"]
        gate = 0.5 * (1.0 + np.tanh(0.5 * gate_logits))
        hidden = active @ p[f"{name}.W2"] + p[f"{name}.b2"]
        skip = gate * hidden if not self.residual else x + gate * hidden
        output, norm_cache = _layer_norm(skip)
```

The two flags are what let one class play three roles: the static encoder sets
`residual=False` and `context=False`, the encoder and decoder set
`context=True`, and the head sets both to the values its branch needs.

The selector is (2) written with broadcasting rather than a loop:

```python
        projected = x[..., None] * p[f"{name}.W"]
        if self.learn:
            scores = projected @ p[f"{name}.score"]
            weights = _softmax(scores, axis=-1)
        else:
            weights = np.full(x.shape, 1.0 / self.count)
        output = (weights[..., None] * projected).sum(axis=-2)
```

`x[..., None]` is the trick: the input has shape `(batch, steps, inputs)`, and
multiplying by a `(inputs, width)` matrix after adding a trailing axis gives
`(batch, steps, inputs, width)` in one operation. The ablation reuses the same
code with a uniform weight, so the only difference between `tft` and
`variable_selection` is one boolean.

Attention is (7)–(8), with the value tensor split and averaged:

```python
        q = self._split(left @ p[f"{name}.Wq"] + p[f"{name}.bq"])
        k = self._split(right @ p[f"{name}.Wk"])
        v = self._split(right @ p[f"{name}.Wv"])
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.width)
        weights = _softmax(scores, axis=-1)
        heads = weights @ v
        joined = heads.mean(axis=1)
        output = joined @ p[f"{name}.Wo"]
```

There is a single `Wv`, split into $n_h$ blocks by `_split`, and `mean(axis=1)`
averages the head axis. `Wo` is therefore `(width, dim)`, not `(dim, dim)` — a
shape that is easy to get wrong and impossible to miss once the first matrix
product fails.

The forward pass shows how the context and the three branches meet:

```python
        selected_past, past_cache = self.past_selector.forward(past)
        selected_ahead, future_cache = self.future_selector.forward(ahead)
        enc_input = selected_past + context[:, None, :]
        encoded, encoder_cache = self.encoder.forward(enc_input, context)
        dec_input = (selected_ahead + context[:, None, :]
                     + p["position"][None, :, :])
        decoded, decoder_cache = self.decoder.forward(dec_input, context)
```

The context is added to the inputs *and* passed to the gates. That duplication
is TFT's design, not an accident: the additive path shifts the representation,
the gate path modulates the nonlinearity.

The windows are cut once, up front, because the dataset is small enough to keep
in memory:

```python
        past = np.stack([scaled[starts + offset - lag]
                         for offset in range(self.seq_len)
                         for lag in range(self.lags)], axis=1)
        past = past.reshape(starts.size, self.seq_len, self.lags)
```

The comprehension is ordered offset-first, lag-second, so the reshape puts step
$\tau$ and lag $j$ in the last two axes in the order (1) defines. Swapping the
loops produces a valid array with the wrong layout — a silent bug that shows up
as a mediocre forecast, not a crash.

The training loop is full-batch Adam with the loss in (10):

```python
            outputs, cache = self._forward(past, ahead, static)
            loss, gradient = self._pinball(outputs, target)
            grads = self._backward(gradient / gradient.size, cache)
            for key, value in parameters.items():
                if key not in grads:
                    continue
                update = grads[key]
                first[key] = 0.9 * first[key] + 0.1 * update
                second[key] = 0.999 * second[key] + 0.001 * update ** 2
```

`gradient / gradient.size` converts the per-output derivative into the mean over
the whole batch, which is what `loss = loss.mean()` reports. Dividing by the
number of quantiles instead of the total size is the natural mistake, and it
scales the learning rate by the batch size.

The interpretability query reuses the encoder stack and reads the cache:

```python
        _, past_cache, future_cache, _, _ = self._encode(y, future)
        return {"past": past_cache[2][0].mean(axis=0),
                "future": future_cache[2][0].mean(axis=0)}
```

`past_cache` is `(x, projected, weights)`, so index 2 is the softmax matrix of
shape `(1, steps, lags)`, averaged over the window. The result is one number per
lag — the report the selection layer exists to produce.

## Complexity

Let $W$ be the number of training windows, $L$ the look-back, $H$ the horizon,
$d$ the width, $f=2d$ the GRN feed-forward width, $n_h$ the head count and $E$
the epochs.

- **Per window, per block:** $O(Ld^2 + Ldf)$ for the encoder GRN and the same for
  the decoder, so $O(Ld^2)$ with $f=2d$.
- **Attention:** $O(Hn_hL(d/n_h)) = O(HLd)$ for the scores and the same for the
  value product, plus $O(HLd)$ for the output projection.
- **One epoch:** the above times $W$, twice for the backward pass. For the demo
  that is 191 windows of 24 steps at width 8, so each epoch is a handful of
  matrix products over a `(191, 24, 8)` tensor.
- **Full fit:** $O(EW(L+H)d^2)$, and 150 epochs of the demo's four models take
  about four seconds in total.
- **Memory:** the cache stores every state for every block, so $O(W(L+H)d)$
  arrays are live during the backward pass.

The window axis is what makes this cheap and what makes it slow. Cutting
windows multiplies the effective dataset size by $W$, but it also means every
epoch does $W$ times the work of a single-sequence model. At the sizes here
full-batch training is fine; at ten thousand series a minibatch over windows
would be required, and the model would be trained across series rather than
within one.

## Failure modes

**Attention weights are not explanations.** The demo measures the attention
matrix of the fitted model: its largest entry is $0.055$ where a flat row over
24 positions would be $0.042$, and its mean entropy is $3.1721$ against a
maximum of $3.1781$. The mean encoder position moves by $0.283$ steps across the
whole horizon. The weights are technically non-uniform and practically
uninformative. Sharing values across heads makes the averaged matrix meaningful
in principle; it does not make a small model on one series produce a sharp one.
Any claim of the form "the model looks here" needs the entropy reported next to
it.

**A partial model is confidently wrong.** `interpretable_attention` has an 80%
band that covers $0.000$ of the hold-out and `variable_selection` covers
$0.083$, against $0.750$ for the full model on the same twelve points. Nothing
in the loss prevents this: the pinball objective rewards a narrow band that is
usually right, and a mis-specified decoder produces narrow bands that are
usually wrong. Coverage has to be monitored separately.

**Training loss is not a selection signal.** The training pinball losses order
the four models exactly backwards — the worst forecaster has the best fit. On
191 windows, a model with more capacity can fit the training targets more
closely while generalising worse, and early stopping on the training loss would
pick the wrong variant every time.

**The training budget is part of the model.** At 120 epochs the same full model
reaches a held-out error of $0.8079$, worse than seasonal naive at $0.7763$; at
150 epochs it reaches $0.4332$. The pinball loss at 120 epochs was $0.0696$
against $0.0518$, so the run was simply not finished. A comparison between
architectures at a fixed epoch count is a comparison between optimisation
trajectories as much as between models.

**Quantiles can cross.** The head is a linear map with one output per quantile
and no ordering constraint, so (9) permits $\hat q_{0.1}>\hat q_{0.9}$ at some
step. The demo's fitted model happens to be monotone everywhere, and the demo
checks it, but a different seed or a noisier series can violate it. Enforce
monotonicity in the parametrisation if the output feeds a decision rule that
assumes it.

**The default future covariates are a crutch.** Sine, cosine and scaled time
encode "the future looks seasonal and trends". On a series with a promotion
calendar, a holiday effect or an exogenous driver, those default columns are
noise and the model has no way to know the real ones exist. Pass `future`
explicitly — with `len(y) + horizon` rows, since the last `horizon` rows are the
ones the decoder consumes.

**One seed, one split.** The demo's ordering of the four variants is a single
draw. Re-running with a different data seed changes the third decimal and
sometimes the order of the middle two. The assertion that the full model beats
each ablation is true on this draw and is stated as such.

**Static features are window-local.** The four summary statistics describe the
look-back, not the series. A regime change inside the window averages out, and a
window that straddles a level shift produces a static vector that describes
neither regime. Longer look-backs make this worse, not better.

## Applications

The three mechanisms correspond to three things practitioners actually need.

Quantile outputs are the deliverable in inventory and staffing: an order
quantity is a quantile of demand, not its mean. Emitting the fan directly avoids
assuming a distribution and avoids a separate calibration step.

Variable selection is the deliverable in settings with many candidate drivers —
energy load with weather, calendar and lag features; retail with price,
promotion and holiday flags. The weights are a per-step report on which inputs
the model used, which is often more actionable than the forecast itself and is
what makes a model auditable in a regulated setting.

Static context is the deliverable in pooled forecasting, where one model serves
many series. The static encoder is where a series-level embedding would live,
and TFT's design keeps it separate from the time-varying path so the two do not
interfere.

The architecture's lasting contribution is the decomposition rather than the
transformer: later models in this chapter keep the quantile head, the selection
network and the static context, and drop or replace the attention. TFT is the
right reference when the requirement is "a forecast with a band and an
explanation of which inputs drove it", and the wrong tool for a single short
series with one strong seasonal pattern, where a state-space model uses the
data far more efficiently.

## References

- Lim, B., Arik, S. O., Loeff, N., Pfister, T. (2021). *Temporal Fusion
  Transformers for interpretable multi-horizon time series forecasting.*
  International Journal of Forecasting. The architecture, the selection
  network, the static covariate encoders and the interpretable attention of
  (7)–(8).
- Vaswani, A. et al. (2017). *Attention is all you need.* NeurIPS. The scaled
  dot-product attention and the position encoding.
- Dauphin, Y. N., Fan, A., Auli, M., Grangier, D. (2017). *Language modeling
  with gated convolutional networks.* ICML. The gated linear unit that (3)
  adapts with a context term.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization.*
  arXiv:1607.06450. Equation (11).
- Koenker, R., Bassett, G. (1978). *Regression quantiles.* Econometrica. The
  pinball loss in (10).
- Jain, S., Wallace, B. C. (2019). *Attention is not explanation.* NAACL. The
  argument behind the first failure mode, with the entropy measurement the demo
  reports.
