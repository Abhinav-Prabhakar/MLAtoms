# Dropout family — noise that is scaled to disappear at inference

`dropout.py` · Neural blocks · Tier 2 · 8 variants

## Intuition

Dropout multiplies activations by random noise during training and does nothing
at inference. The noise forces a unit to work without its neighbours, which
breaks up co-adaptation; the scaling is what makes the two regimes agree. With
*inverted* dropout the surviving units are divided by the keep probability, so
the expected activation is unchanged and inference needs no correction. Every
variant in this file shares that contract:

$$
\text{training: } y=x\odot\text{noise},\qquad
\text{inference: } y=x .
\qquad (1)
$$

What changes is the *shape* of the noise. Independent elementwise masks are the
default; spatial masks drop whole feature maps; variational noise is fixed
across time; dropout1d drops timesteps; drop-path drops a residual branch;
stochastic depth makes the drop rate grow with depth; alpha dropout replaces the
mask with a shift-and-scale that preserves the first two moments of a
standardized input. The demo measures each contract on synthetic data rather
than trusting it.

The demo also shows the distinction that matters most in practice. Inverted
dropout leaves the mean alone but inflates the variance to
$p/(1-p)$ times the activation; alpha dropout leaves both mean and variance
alone, at the cost of assuming the input is already standardized. On the same
$50000\times8$ standard-normal batch, standard dropout has variance 1.3288 and
alpha dropout 0.9965.

## Variants

| variant | mask or noise | shared over | rate |
|:--|:--|:--|:--|
| `standard` | Bernoulli, elementwise | nothing | `rate` |
| `spatial` | Bernoulli per channel | spatial axes, per sample | `rate` |
| `gaussian` | Gaussian, mean one | nothing | `sigma` |
| `variational` | Gaussian, mean one | time, per sample and feature | `sigma` |
| `dropout1d` | Bernoulli per timestep | features, per sample | `rate` |
| `droppath` | Bernoulli per sample | the whole branch | `rate` |
| `stochastic_depth` | Bernoulli per sample | the whole branch | depth schedule |
| `alpha_dropout` | affine shift and scale | nothing | `rate` |

**`standard`** is the original: a fresh Bernoulli mask for every element, with
survivors scaled by $1/(1-r)$.

**`spatial`** draws one mask per sample and channel and broadcasts it over the
spatial axes, so a feature map is either kept whole or zeroed whole. It is the
usual regulariser for convolutional trunks.

**`gaussian`** multiplies by $1+\sigma\varepsilon$ with
$\varepsilon\sim\mathcal N(0,1)$. It never produces an exact zero and has no
keep probability, but its mean is still one.

**`variational`** uses the same Gaussian noise at every time step of a sequence,
one noise per sample and feature. That is the fixed-noise reading of
variational dropout for recurrent nets, and it is what makes the regulariser
consistent along the sequence.

**`dropout1d`** drops entire time steps: one Bernoulli mask per sample and step,
shared across features. It removes a frame rather than a unit, which suits
sequence and audio models. (PyTorch's `Dropout1d` drops channels instead; that
is `spatial` in this file.)

**`droppath`** applies the mask to a whole residual branch: the output is the
residual plus either the scaled branch or nothing. It is the residual-network
regulariser from the stochastic-depth paper.

**`stochastic_depth`** is drop-path with a schedule. At depth $\ell$ of $L$
blocks the rate is $r_\ell=r_0(\ell+1)/L$, so early blocks are almost always
kept and late blocks are regularised hardest.

**`alpha_dropout`** replaces dropped values by a constant and rescales the kept
ones so that a zero-mean unit-variance input keeps mean zero and variance one.
It is designed for SELU networks, whose self-normalizing property needs exactly
those moments.

## Derivation

### Standard dropout

Let $K\sim\mathrm{Bernoulli}(p)$ with $p=1-r$, independent per element. Inverted
dropout is

$$
y=x\,\frac{K}{p},
\qquad
\mathbb E[y]=x,
\qquad
\mathrm{Var}(y)=x^2\Bigl(\frac1p-1\Bigr)=x^2\frac{r}{1-r}.
\qquad (2)
$$

The mean identity is exact. The variance grows without bound as $p\to0$, which
is why a rate near one destroys the signal even though the expectation is
formally preserved. For a random input $X$ independent of $K$,

$$
\mathrm{Var}(y)=\frac{\mathrm{Var}(X)}{p}
+\mathbb E[X]^2\Bigl(\frac1p-1\Bigr),
\qquad (3)
$$

so a mean-zero activation has its variance inflated by $1/p$. In the demo,
$x=1$ and $r=0.25$ give a measured mean of 0.9983 and variance 0.3345 against
the predicted $1/3$.

### Spatial dropout

For an image tensor of shape $(n,h,w,c)$ the mask has shape $(n,1,1,c)$:

$$
m_{n:\,:\,c}\sim\mathrm{Bernoulli}(p),
\qquad
y_{nhwc}=x_{nhwc}\frac{m_{n:\,:\,c}}{p}.
\qquad (4)
$$

Every pixel of a surviving channel is scaled by the same factor, so the
regulariser removes a feature detector rather than a location. The demo keeps
16 of 24 maps and checks that each is either all zero or exactly $x/0.5$
everywhere.

### Gaussian and variational noise

Independent Gaussian dropout uses

$$
y=x\,(1+\sigma\varepsilon),
\qquad
\mathbb E[y]=x,
\qquad
\mathrm{Var}(y)=x^2\sigma^2 .
\qquad (5)
$$

The variational variant draws one $\varepsilon$ per sample and feature and
broadcasts it over the middle axes:

$$
\varepsilon_{n:\,:\,c}\sim\mathcal N(0,1),
\qquad
y_{n t c}=x_{n t c}(1+\sigma\varepsilon_{n:\,:\,c}).
\qquad (6)
$$

The same unit is therefore noisy in the same direction at every step, which is
the behaviour a recurrent model needs; fresh noise per step would average out
along the sequence. The demo uses five sequences of seven steps and six
features and checks that the ratio $y/x$ is constant along time.

### Dropout1d

With one mask per sample and time step,

$$
m_{nt}\sim\mathrm{Bernoulli}(p),
\qquad
y_{ntc}=x_{ntc}\frac{m_{nt}}{p}.
\qquad (7)
$$

This drops a frame from every feature at once. The demo drops 16 of 35 steps and
verifies that the zero pattern is constant along the feature axis.

### Drop path and stochastic depth

A residual branch is either kept or skipped per sample:

$$
y=r+x\,\frac{K}{p},
\qquad
K\sim\mathrm{Bernoulli}(p).
\qquad (8)
$$

Dropping the branch leaves the identity path untouched, which is why the
gradient of a dropped sample is exactly zero and the residual still carries
information. With a schedule $r_\ell=r_0(\ell+1)/L$ at depth
$\ell=0,\dots,L-1$ the rate at the last block is $r_0$ and the first is
$r_0/L$. The demo uses $r_0=0.5$ and $L=6$, giving rates

$$
0.083,\ 0.167,\ 0.250,\ 0.333,\ 0.417,\ 0.500,
\qquad (9)
$$

and measured drop fractions 0.066 to 0.494 over 2000 samples. The measured
value tracks the nominal rate because each sample is one Bernoulli draw.

### Alpha dropout

Let $\alpha'=-\alpha\lambda$ be the SELU dropout constant, with
$\alpha=1.6733$ and $\lambda=1.0507$. The layer is

$$
y=a\bigl(xK+\alpha'(1-K)\bigr)+b,
\qquad K\sim\mathrm{Bernoulli}(p),
\qquad (10)
$$

with

$$
a=\bigl(p(1+r\alpha'^2)\bigr)^{-1/2},
\qquad
b=-a\,\alpha'\,r .
\qquad (11)
$$

For an input with mean zero and variance one, the shift cancels the mean of the
dropped branch:

$$
\mathbb E[y]=a\,p\,\mathbb E[x]+a\,\alpha' r+b=0.
\qquad (12)
$$

For the variance, write $c=\mathbb E[y]=0$. Since
$\mathbb E[(xK+\alpha'(1-K))^2]=p+r\alpha'^2$ and
$\mathbb E[xK+\alpha'(1-K)]=\alpha' r$,

$$
\mathrm{Var}(y)
=a^2\bigl(p+r\alpha'^2\bigr)+2ab\,\alpha'r+b^2
=a^2p\bigl(1+r\alpha'^2\bigr)=1,
\qquad (13)
$$

using $b=-a\alpha'r$. The demo checks (13) numerically and in closed form, and
finds mean $-0.0016$ and variance 0.9965 on a standard-normal batch. If the
input has nonzero mean the identity no longer holds: the output mean is
$ap\mu$, not $\mu$. That is not a bug; SELU networks are standardized before
every layer.

### Inference

Every non-path variant returns $x$ unchanged when `training=False`, and the path
variants return $r+x$. There is no keep-probability correction at inference
because the inverted scaling already put it in training. A rate of zero is the
identity in both regimes, which the demo checks for all eight variants.

## Code walkthrough

Standard and alpha dropout share the mask but not the arithmetic. Alpha's
learning signal is the affine map `kept * a`, so its backward pass differs by
one constant:

```python
            mask = self.rng.random(x.shape) < keep
            kept = mask.astype(float)
            if self.variant == "alpha_dropout":
                alpha_p = -self.alpha * self.scale
                a = (keep * (1.0 + (1.0 - keep) * alpha_p ** 2)) ** -0.5
                b = -a * alpha_p * (1.0 - keep)
                output = x * kept * a + alpha_p * (1.0 - kept) * a + b
                scale = kept * a
                cache.update(scale=scale, a=a, b=b, alpha_p=alpha_p)
                return output, cache
            scale = kept / keep
            cache["scale"] = scale
            return x * scale, cache
```

The broadcast shapes are the whole trick behind the structured variants. One
helper for channel masks and one for time masks cover spatial, variational and
dropout1d without a second implementation:

```python
    def _channel_shape(self, x: np.ndarray) -> tuple[int, ...]:
        """One mask entry per sample and channel, broadcast over space."""
        return (x.shape[0],) + (1,) * (x.ndim - 2) + (x.shape[-1],)
```

Spatial dropout is the channel helper plus a Bernoulli draw. Note that the mask
is generated on the small shape, not on the full tensor, so the memory cost is
one number per feature map:

```python
        if self.variant == "spatial":
            if x.ndim < 2:
                raise ValueError("spatial needs a sample and channel axis")
            shape = self._channel_shape(x)
            mask = self.rng.random(shape) < keep
            scale = mask.astype(float) / keep
            cache["scale"] = scale
            return x * scale, cache
```

Gaussian and variational dropout share the noise formula; only the shape of the
draw differs. The cache keeps the noise because the backward pass is linear in
it:

```python
        if self.variant == "gaussian":
            noise = 1.0 + self.sigma * self.rng.standard_normal(x.shape)
            cache["noise"] = noise
            return x * noise, cache
```

```python
        if self.variant == "variational":
            if x.ndim < 2:
                raise ValueError("variational needs a sample and feature axis")
            shape = self._channel_shape(x)
            noise = 1.0 + self.sigma * self.rng.standard_normal(shape)
            cache["noise"] = noise
            return x * noise, cache
```

Dropout1d uses the other helper and a fresh Bernoulli mask per step:

```python
        if self.variant == "dropout1d":
            if x.ndim < 2:
                raise ValueError("dropout1d needs a sample and time axis")
            shape = self._time_shape(x)
            mask = self.rng.random(shape) < keep
            scale = mask.astype(float) / keep
            cache["scale"] = scale
            return x * scale, cache
```

The path variants are the same mask at sample granularity, added back to the
residual. A dropped sample returns the residual exactly, because multiplying by
a zero mask gives an exact zero before the addition:

```python
        shape = self._sample_shape(x)
        mask = self.rng.random(shape) < keep
        scale = mask.astype(float) / keep
        cache.update(scale=scale, rate=1.0 - keep)
        return residual + x * scale, cache
```

Stochastic depth's only extra machinery is the schedule. It is computed at
forward time from the depth, so the same object can be reused at every block:

```python
    def _rate_at(self, depth: int | None) -> float:
        if self.variant != "stochastic_depth":
            return self.rate
        level = self.depth if depth is None else depth
        if level < 0 or level >= self.max_depth:
            raise ValueError("depth must be in [0, max_depth)")
        return self.rate * (level + 1) / self.max_depth
```

Backward is one multiply by the cached mask or noise, and the identity at
inference:

```python
    def backward(self, gradient: np.ndarray, cache) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float)
        if cache.get("eval"):
            return gradient
        if "noise" in cache:
            return gradient * cache["noise"]
        return gradient * cache["scale"]
```

## Complexity

For $n$ elements:

- **Standard, gaussian, alpha:** one random draw and one or two elementwise
  operations per element, $O(n)$ time and one mask's worth of cache.
- **Spatial:** $O(n/c)$ draws for $c$ channels and an $O(n)$ broadcast multiply.
  The mask is one number per sample and channel.
- **Variational and dropout1d:** the same broadcast pattern over time rather
  than space, with a mask one axis smaller than the tensor.
- **Path variants:** one draw per sample and one multiply of the branch, so the
  cost is $O(n)$ in the branch and free in the residual.
- **Parameters:** zero. The only state is the random generator, which is why
  seeding is part of reproducibility.
- **The demo:** masks over at most 40000 elements and one 50000-row moment
  check; every section is deterministic under its seed and the whole file runs
  in a fraction of a second.

## Failure modes

**Train and inference diverge if the scaling is left out.** Without the $1/p$
factor, training produces activations $p$ times too small, and inference is
then a sudden rescaling. Inverted dropout is the fix, not a detail.

**High rates destroy the signal.** The expectation is preserved for any
$p>0$, but the variance in (3) grows like $1/p$. A rate of 0.9 turns a
well-conditioned layer into noise with ten times the variance; the guard
refuses $r=1$ because the scale would be infinite.

**Dropping before batch norm leaks the training distribution.** If dropout is
applied before a batch-norm layer, the batch statistics are computed on the
noisy tensor while inference sees the clean one. The usual placement is after
the norm and before the activation, or on the residual branch rather than the
identity path.

**Spatial masks are easy to broadcast into nothing.** A mask shaped $(n,c)$ on a
tensor $(n,h,w,c)$ either fails or broadcasts along the wrong axis. The helper
builds $(n,1,1,c)$ explicitly; the demo asserts that the zero pattern is
constant across space.

**Variational noise that is redrawn per step is just Gaussian dropout.** The
point of the variational variant is the shared draw. If the mask changes at
every timestep the regulariser averages out along the sequence and the
recurrent interpretation is lost.

**A path variant without a residual is a zero layer.** Dropping the branch then
leaves nothing, and the gradient of a dropped sample is zero. The constructor
requires `residual`, and the demo checks that the surviving samples are exactly
$r+2x$ at rate 0.5.

**Depth indexing is off by one.** A schedule written as $\ell/L$ instead of
$(\ell+1)/L$ never reaches the maximum rate, and one written as
$\ell/(L-1)$ divides by zero at $L=1$. The guard keeps `depth` inside
`[0, max_depth)` and the demo prints the six rates.

**Alpha dropout assumes standardized inputs.** It preserves zero mean and unit
variance, not an arbitrary mean. Applying it after a layer with a large bias
shifts the output by $ap\mu$ and breaks the self-normalizing argument. It also
uses the SELU constants, so pairing it with ReLU is a different (and
unjustified) regulariser.

**Alpha's dropped value is a constant, not zero.** The dropped branch becomes
$\alpha'a+b$, which is nonzero; code that assumes zeros will misread the
sparsity. The demo checks that every dropped entry equals that constant.

**Gaussian noise has no sparsity.** It regularises by variance, not by removing
units, so it does not give the sparse-activation effect of a Bernoulli mask.
Large $\sigma$ also compounds multiplicatively through depth.

**Dropout in evaluation inflates the variance.** Forgetting to switch to
inference mode leaves the model noisy and, for spatial or path variants,
randomly zeroes whole feature maps. The demo's `training=False` checks are the
contract every caller depends on.

**Randomness must be seeded per run, not per layer.** Two layers seeded with
the same value share a mask, which correlates their noise. The demo seeds each
model independently and relies on that for its printed counts.

## Applications

Standard dropout is still used on transformer attention weights and feed-forward
activations, and on the classifier head of almost every vision model. Spatial
dropout is common in convolutional trunks where whole filters should be
regularised; dropout1d and variational dropout target sequence and recurrent
models.

Drop-path and stochastic depth are standard in residual networks and vision
transformers, where dropping a whole block is cheaper than dropping units and
matches the identity-path argument. Alpha dropout belongs with SELU and
self-normalizing networks, and Gaussian dropout appears in variational
inference and as a smooth alternative to Bernoulli masks.

## References

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.
  (2014). *Dropout: a simple way to prevent neural networks from overfitting.*
  JMLR. Equations (2) and (3).
- Tompson, J., Goroshin, R., Jain, A., LeCun, Y., Bregler, C. (2015).
  *Efficient object localization using convolutional networks.* CVPR. Spatial
  dropout, equation (4).
- Kingma, D. P., Salimans, T., Welling, M. (2015). *Variational dropout and the
  local reparameterization trick.* NeurIPS. The fixed-noise reading of
  equation (6).
- Gal, Y., Ghahramani, Z. (2016). *A theoretically grounded application of
  dropout in recurrent neural networks.* NeurIPS. Shared masks across time.
- Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K. (2016). *Deep networks
  with stochastic depth.* ECCV. Equations (8) and (9).
- Klambauer, G., Unterthiner, T., Mayr, A., Hochreiter, S. (2017).
  *Self-normalizing neural networks.* NeurIPS. Alpha dropout, equations (10)
  to (13).
- PyTorch documentation. *Dropout1d*, *Dropout2d*, *Dropout3d*: the channel
  convention that this file calls `spatial`.
