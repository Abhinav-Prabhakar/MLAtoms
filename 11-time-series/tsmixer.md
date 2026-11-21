# TSMixer — replace attention with two alternating perceptrons

`tsmixer.py` · Time series · Tier 3 · 4 variants

## Intuition

Every transformer forecaster spends its parameters on pairwise interactions:
token $i$ against token $j$. TSMixer asks what is actually being mixed, and
answers that there are only two axes. A multivariate window is a matrix
$X\in\mathbb R^{L\times C}$ — time down the rows, channels across the columns.
To forecast it you need to combine observations along time, and you need to
combine channels. Those are two operations, and neither needs a quadratic
matrix of interactions.

So the block is two multi-layer perceptrons applied along the two axes. The
**time-mixing** MLP takes the $L$ values of one channel, projects them to a
hidden width, and projects back — it can learn any linear filter over the
window, and with the ReLU it can learn a nonlinear one. The **feature-mixing**
MLP does the same across the $C$ channels at a single timestep: it learns which
channels predict which. Each is wrapped in a residual connection, and the
stack of blocks alternates time, features, time, features.

The cost is linear in both $L$ and $C$ rather than quadratic, and the model is
a stack of matrix products — the kind of thing that runs in microseconds on a
CPU. On the 289-window demo it reaches a held-out MAE of $0.2282$ against
$1.3784$ for repeating the last value.

The variants make the ablation explicit:

- **`time_mix`** keeps only the time-axis MLP.
- **`feature_mix`** keeps only the channel-axis MLP.
- **`tsmixer`** alternates both.
- **`tsmixer_ext`** also consumes exogenous features, in the feature branch
  and in the head.

The demo's most interesting result is the negative one: `time_mix` alone reaches
$0.1912$ and beats the full model. The channel branch costs parameters and buys
nothing when the channels are redundant. That is the honest headline, and it is
the reason the ablation exists.

## Variants

| variant | time branch | feature branch | exog | held-out MAE |
|:--|:--|:--|:--|:--|
| `time_mix` | yes | no | no | 0.1912 |
| `feature_mix` | no | yes | no | 0.2402 |
| `tsmixer` | yes | yes | no | 0.2282 |
| `tsmixer_ext` | yes | yes | yes | 3.3795 vs 4.0221 |

**`time_mix`** owns 10 arrays and $6892$ numbers, all of them in the two time
matrices per block and the head. It is the best of the three on this data,
which is unsurprising: the three channels are noisy affine copies of one
another, so there is no independent cross-channel signal to learn.

**`feature_mix`** also owns 10 arrays but only $1042$ numbers, because the
channel axis has three columns. It reaches $0.2402$. Remarkably it is close to
the full model — most of the useful fit is a per-timestep linear map plus the
head.

**`tsmixer`** owns 18 arrays and $7346$ numbers: two time branches and two
feature branches, plus one head. It reaches $0.2282$, between the two
ablations. Per block the time branch costs $3152$ numbers and the feature
branch $227$, a ratio of fourteen to one.

**`tsmixer_ext`** is trained on a different series, one with a random driver
that the history cannot predict but the future does reveal. Without the driver
the model reaches $4.0221$ — worse than the last value, because the series is
mostly unpredictable noise around a seasonal mean. With the driver it reaches
$3.3795$. The head widens from $48$ to $60$ columns: $48$ look-back steps plus
$12$ future driver steps.

## Derivation

### Notation

A training window is $X\in\mathbb R^{L\times C}$ with target
$Y\in\mathbb R^{H\times C}$. $L$ is the look-back, $H$ the horizon, $C$ the
number of channels. A batch stacks $B$ windows.

### Standardisation

Each window is standardised channel by channel, using its own look-back
statistics:

$$
\tilde x_{tc}=\frac{x_{tc}-\mu_c}{\sigma_c},\qquad
\mu_c=\frac1L\sum_{t=1}^{L}x_{tc},\qquad
\sigma_c^2=\frac1L\sum_{t=1}^{L}(x_{tc}-\mu_c)^2 .
\qquad (1)
$$

The same $\mu_c,\sigma_c$ standardise the target. This is the "instance
normalisation" that makes a non-stationary series stationary within each
window: the network never sees an absolute level, only a shape. The forecast is
mapped back with $\hat y=\mu_c+\sigma_c\hat z$ at prediction time.

### Layer normalisation

Every mixing MLP is preceded by a normalisation over the axis it is about to
mix:

$$
\mathrm{LN}(z)=\frac{z-\bar z}{\sqrt{\tfrac1d\sum_{j}(z_j-\bar z)^2+\epsilon}},
\qquad \bar z=\frac1d\sum_{j=1}^{d}z_j .
\qquad (2)
$$

After transposing, the time MLP sees a vector of length $L$ and the feature MLP
a vector of length $C$, so (2) is applied with $d=L$ and $d=C$ respectively.
The normalisation is affine-free: the gain and bias are redundant with the
following linear layer's own bias and with the residual scale, and dropping
them removes two arrays per branch.

### Time mixing

For each channel $c$, with $W_t\in\mathbb R^{L\times h}$, $V_t\in\mathbb R^{h\times L}$:

$$
a_c=\mathrm{ReLU}\bigl(\mathrm{LN}(X_{\cdot c})\,W_t+b_t\bigr)\in\mathbb R^{h},
\qquad (3)
$$

$$
Z_{\cdot c}=X_{\cdot c}+a_c\,V_t+c_t .
\qquad (4)
$$

The map $X_{\cdot c}\mapsto a_cV_t$ is a rank-$h$ linear operator on the window,
gated by which of its $h$ hidden units are active. With $h\ge L$ it can
represent any linear filter — moving average, difference, seasonal lag — and it
learns the filter from data rather than from a fixed kernel. This is exactly
what DLinear does with a fixed moving average plus a linear layer; TSMixer
learns both parts jointly.

### Feature mixing

At each timestep $t$, with $W_f\in\mathbb R^{(C+A)\times h}$ and
$V_f\in\mathbb R^{h\times C}$:

$$
g_t=\mathrm{ReLU}\bigl(\mathrm{LN}(Z_{t\cdot})\,W_f+b_f\bigr)\in\mathbb R^{h},
\qquad (5)
$$

$$
X'_{t\cdot}=Z_{t\cdot}+g_t\,V_f+c_f .
\qquad (6)
$$

$A$ is the number of exogenous features; they are concatenated to the
normalised channels before $W_f$, which is why that matrix is
$(C+A)\times h$. Setting $A=0$ recovers the plain model.

Unlike channel independence, this branch gives every channel its own row of
$W_f$ and its own column of $V_f$. The model is therefore **not** equivariant to
permuting the channels: a permuted input meets a different set of weights. That
is the price of letting the channels talk, and it is why the branch is a good
place to attach exogenous variables — each one gets an identifiable coefficient.

### The block

One block is the composition of the two:

$$
X^{(r+1)}=\mathcal F^{(r)}\bigl(\mathcal T^{(r)}(X^{(r)})\bigr),
\qquad r=1,\dots,R,
\qquad (7)
$$

with $X^{(1)}=\tilde X$. Order matters: time mixing sees channels with
independent residuals, feature mixing then sees a representation already
smoothed along time. Reversing the order is a different model; the demo's
`feature_mix` and `time_mix` ablations are the two degenerate cases where one
factor is the identity.

### The head

The final representation is flattened across channels and mapped to the
horizon:

$$
F=\mathrm{reshape}_{C\times H}\!\left(\mathrm{vec}_C(X^{(R)})\,W_o+b_o\right)
\in\mathbb R^{C\times H},
\qquad W_o\in\mathbb R^{L\times H}.
\qquad (8)
$$

`vec_C` puts the $L$ values of one channel contiguously, so the same $W_o$ is
applied to every channel. This is where TSMixer gets its parameter efficiency:
one $48\times12$ matrix forecasts all three channels.

With exogenous features the head receives the concatenation

$$
\mathrm{vec}_C(X^{(R)})\;\|\;\mathbf 1_C\otimes \mathrm{vec}(U),
\qquad U\in\mathbb R^{H\times A},
\qquad (9)
$$

where $U$ is the standardised future driver and $\mathbf 1_C\otimes\cdot$
repeats it for every channel. The driver rows of $W_o$ are then per-channel
regression coefficients on the known future, learned jointly with the rest.

### Objective

Training minimises mean squared error on standardised targets:

$$
L=\frac{1}{BCH}\sum_{b,c}\sum_{h=1}^{H}
\bigl(\hat y_{bch}-y_{bch}\bigr)^2 .
\qquad (10)
$$

Optimisation is full-batch Adam. The backward pass through a residual is the
identity plus the branch Jacobian, which is why the residual gradient in the
code is a sum:

$$
\frac{\partial L}{\partial X}=\frac{\partial L}{\partial X'}
+\mathcal B^{\top}\!\left(\frac{\partial L}{\partial X'}\right).
\qquad (11)
$$

## Code walkthrough

The whole architecture is two flags, so the three variants are not three code
paths:

```python
        self.use_time = variant != "feature_mix"
        self.use_feature = variant != "time_mix"
```

Everything downstream — parameter creation, forward, backward, the optimiser's
parameter dict — reads those flags. A variant that turns a branch off does not
merely skip it; the branch's matrices are never allocated, which is why the
demo can assert an exact array count for each variant.

The time branch transposes so that the last axis is the one being mixed:

```python
            if self.use_time:
                transposed = current.transpose(0, 2, 1)
                norm, norm_cache = _layer_norm(transposed)
                pre = norm @ p[f"T1_{index}"] + p[f"t1_{index}"]
                active = np.maximum(pre, 0.0)
                mixed = active @ p[f"T2_{index}"] + p[f"t2_{index}"]
                current = current + mixed.transpose(0, 2, 1)
```

`current` is always `(batch, time, channel)`. Transposing to
`(batch, channel, time)` lets one matrix product apply the same filter to every
channel without a loop, and the transpose back restores the layout the next
branch expects. Two transposes per block are the entire cost of sharing.

The feature branch needs no transpose because the channel axis is already last:

```python
            if self.use_feature:
                norm, norm_cache = _layer_norm(current)
                joined = (np.concatenate([norm, aux], axis=2)
                          if self.aux else norm)
                pre = joined @ p[f"F1_{index}"] + p[f"f1_{index}"]
                active = np.maximum(pre, 0.0)
                mixed = active @ p[f"F2_{index}"] + p[f"f2_{index}"]
                current = current + mixed
```

Concatenating the auxiliary features into the normalised channels means the
exogenous inputs are normalised by the *same* layer norm, so they enter the
third branch on the same scale as the series. That is a small choice with a
large effect: an unnormalised driver of magnitude $10^3$ would saturate the
ReLU and stall training.

The head folds the channel axis into the batch and then restores it:

```python
        flat = current.transpose(0, 2, 1).reshape(-1, self.seq_len)
        if self.aux:
            repeated = np.repeat(future.reshape(x.shape[0], -1),
                                 self.channels, axis=0)
            head = np.concatenate([flat, repeated], axis=1)
        else:
            head = flat
        forecast = (head @ p["Wo"] + p["bo"]).reshape(
            x.shape[0], self.channels, self.horizon)
        return forecast.transpose(0, 2, 1), {"blocks": cache, "head": head}
```

`reshape(-1, L)` after the transpose gives `(batch * channel, L)`, so the head
is one matrix product for all channels. The reshape back to
`(batch, channel, horizon)` and the final transpose assume the channel order is
preserved — which it is, because `transpose` and `reshape` are applied in the
reverse order on the way out.

The backward pass starts from the same fold. The head's input is cached because
the residual makes it unrecoverable from the gradient alone:

```python
        flat_grad = gradient.transpose(0, 2, 1).reshape(-1, self.horizon)
        g["Wo"] = cache["head"].T @ flat_grad
        g["bo"] = flat_grad.sum(axis=0)
        current_grad = (flat_grad @ p["Wo"][:self.seq_len].T).reshape(
            batch, self.channels, self.seq_len).transpose(0, 2, 1)
```

The slice `[:self.seq_len]` is the part of the wide head that came from the
series; the remaining rows belong to the exogenous features and have no path
back into the mixer.

The feature branch's backward is worth reading in full because it shows the
residual sum:

```python
                    width = active.shape[-1]
                    g[f"F2_{index}"] = (active.reshape(-1, width).T
                                        @ current_grad.reshape(-1, self.channels))
                    g[f"f2_{index}"] = current_grad.sum(axis=(0, 1))
                    delta = (current_grad @ p[f"F2_{index}"].T) * (pre > 0.0)
                    joined = entry["joined"]
                    g[f"F1_{index}"] = (joined.reshape(-1, joined.shape[-1]).T
                                        @ delta.reshape(-1, width))
                    g[f"f1_{index}"] = delta.sum(axis=(0, 1))
                    normalised_grad = (delta @ p[f"F1_{index}"].T)[
                        ..., :self.channels]
                    current_grad = current_grad + _layer_norm_backward(
                        normalised_grad, entry["cache"])
```

`delta` is the gradient after the ReLU, `normalised_grad` is the gradient into
the normalised channels, and the last line adds the branch's contribution to
the identity path. The slice `[..., :self.channels]` drops the exogenous
columns: their gradient exists but there is nothing upstream to receive it.

For `tsmixer_ext`, the windows and the future rows are cut from one standardised
matrix:

```python
            exog_windows = standardised[
                starts[:, None] + np.arange(self.seq_len)[None, :], :]
            future = standardised[
                starts[:, None] + self.seq_len
                + np.arange(self.horizon)[None, :], :]
```

Both are built from `starts`, the same sliding-window indices used for the
series, which guarantees that the driver at time $t$ lines up with the target
at time $t$. Getting that alignment wrong — using `starts + 1` or forgetting
the horizon offset — is the classic exogenous-forecasting bug, and it produces
a model that looks excellent in training and useless out of sample.

## Complexity

With $B$ windows, $L$ look-back, $C$ channels, $A$ auxiliary features, $R$
blocks, hidden width $h$ and horizon $H$:

- **Standardisation:** $O(BLC)$.
- **Time branch:** $O(B C L h)$ time and $O(B C \max(L,h))$ memory for the
  activations; the same matrix serves every channel.
- **Feature branch:** $O(B L (C+A) h)$.
- **One block:** the sum, so linear in $L$, $C$ and $h$.
- **Head:** $O(B C L H)$.
- **Full forward pass:** $O\bigl(BR(C L h + L(C+A)h) + BCLH\bigr)$.
- **One epoch:** three times the forward pass for the backward, plus the
  parameter count.
- **Prediction:** one window, so the same expression with $B=1$.

The comparison that matters is against attention. PatchTST mixes $N=L/P$
patches with an $N\times N$ attention map, so its mixing cost is $O(L^2 h/P^2)$
per channel; TSMixer's is $O(Lh)$ per channel, smaller by a factor of $L/P^2$.
Against channel mixing attention, which is $O(C^2h)$ per timestep, TSMixer pays
$O(Ch)$ — the same saving in $C$.

In absolute terms the demo's largest model has $7346$ parameters and trains in
about a second for 120 full-batch epochs over 289 windows. There is no
attention map to materialise, so memory stays proportional to the activations
and not to $L^2$.

## Failure modes

**Feature mixing overfits when the channels are redundant.** The full model
reaches $0.2282$ and the time-only ablation $0.1912$. The channel branch has
per-channel weights and only three channels to estimate them from, so it fits
noise that the time branch ignores. With three hundred channels the same branch
would be the one carrying the signal; with three, it is a liability. Always run
the ablation before trusting channel mixing.

**Instance normalisation discards the level.** Because every window is
standardised by its own mean, the model never sees an absolute value. A linear
map can still extrapolate a trend, but it has to encode the drift as a fixed
transformation of the shape, which a trending series defeats. If the level
matters, add it back as an explicit covariate or use a model with a level
component.

**No probabilistic output.** The head emits a point forecast. Uncertainty would
require a quantile head, a distributional loss, or a residual bootstrap — none
of which the block computes. For decision-making under uncertainty this is the
main limitation relative to DeepAR or a Gaussian state-space model.

**The head is $L\times H$ per channel-set.** At $L=48$ it is 588 numbers; at
$L=512$ and $H=96$ it would dominate the parameter count. Longer look-backs
want a bottleneck or a patching scheme in front of the head.

**Stacking blocks does not monotonically help.** On the demo configuration,
one block reaches $0.2573$, two reach $0.2282$, three $0.2606$ and four
$0.2401$, for $3967$, $7346$, $10725$ and $14104$ parameters. Depth buys
capacity that a few hundred windows cannot identify, and the held-out error
wanders rather than falling. Tune the block count; do not assume more is
better.

**Reversal of the block order changes the model.** Time-then-feature is not the
same as feature-then-time; the second branch sees a different residual. If you
change the order, retune, and do not assume the ablation results transfer.

**Layer norm over few channels is noisy.** With $C=3$ the normalisation has
three numbers to estimate a variance from, so it is a coarse operation. With
$C=1$ it divides by the standard deviation of a single value — zero — and the
code falls back to a unit scale. The model still trains, but the normalisation
stops doing anything useful.

## Applications

TSMixer is aimed at the regime where attention is overkill: hundreds of related
series, moderate horizons, and a hard latency budget. Retail demand (the M5
dataset in the paper), electricity load, traffic counts and weather station
networks all fit. The model is small enough to run on a CPU at the edge or
inside a database, and the alternating structure maps to two batched matrix
products that vectorise well.

The extended variant is for the case where the future is partly known:
promotions and holidays in retail, temperature forecasts in load forecasting,
scheduled maintenance in industrial telemetry. Its head puts a per-channel
coefficient on each future driver, which makes the exogenous effect
inspectable — a rare property in a deep forecaster.

It is the wrong choice for a single short series with no cross-channel
structure (there the time branch alone is a small MLP and a linear model is a
fair fight), for long horizons on high-frequency data, and for anything that
needs calibrated intervals.

## References

- Chen, S.-A., Li, C.-L., Yoder, N., Arik, S. O., Pfister, T. (2023).
  *TSMixer: An all-MLP architecture for time series forecasting.* TMLR.
  arXiv:2303.06053. The alternating time and feature mixing blocks and the
  instance-normalisation scheme.
- Google Research (2024). *TSMixer-Extended: Towards improved modeling of
  exogenous variables.* The exogenous branch and the widened head.
- Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner,
  T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., Lucic, M., Dosovitskiy,
  A. (2021). *MLP-Mixer: An all-MLP architecture for vision.* NeurIPS. The
  two-axis mixing pattern that TSMixer adapts to time series.
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). *Are transformers effective for
  time series forecasting?* AAAI. The linear-baseline argument behind the
  time-mixing branch.
- Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). *A time series is
  worth one word.* ICLR. Patching and channel independence, the design TSMixer
  trades against.
