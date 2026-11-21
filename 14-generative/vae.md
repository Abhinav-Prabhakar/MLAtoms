# Variational autoencoder — the ELBO, and four ways to tighten or break it

`vae.py` · Generative · Tier 1 · 5 variants

## Intuition

A VAE is two models that share a latent space. The generative model says a data
point was produced by drawing a code $z$ from a standard normal and pushing it
through a decoder; the inference model says, given a data point, where its code
probably sits. Training pushes the two together with one objective, and that
objective has two terms that fight.

The demo makes the fight visible because the data is deliberately low
dimensional: eight Gaussian blobs on a circle of radius 1.2, 100 points each,
noise 0.12. The manifold is a circle — two degrees of freedom, eight identities.
A latent of eight dimensions is therefore four times more room than the data
needs, which is exactly the setting where a VAE has to decide how much
information to keep.

```
                  encoder                 decoder
    x (2)  ──▶  2 → 48 ──▶ (mu, log var) ──▶ z (8) ──▶ 48 → 2  ──▶  x hat
                              │
                              └── eps ~ N(0, 1) is added to mu * exp(log var / 2)
```

Running all five variants gives one table. The first two columns are estimates
of the same quantity — the log likelihood of held-out data — computed with one
sample and with eight:

```
variant            elbo    iwae8  active  mean sd
standard         -1.891    -1.605    2/8    0.814
conditional       1.305       --    0/8    0.997
iwae             -7.607    -1.273    7/8    0.879
beta             -5.291    -1.651    2/8    0.867
collapse_fix     -4.671    -2.537    8/8    0.613
```

`active` counts latent dimensions whose posterior mean varies across the data
(var above 0.01), and `mean sd` is the average posterior standard deviation over
all eight.

Three things in that table are the whole atom.

**`standard` uses two of the eight latent dimensions and leaves six at the
prior.** That is correct — the data has two degrees of freedom. The per
dimension KL is `[1.57, 1.36, 0, 0, 0, 0, 0, 0]`: the model discovered the
manifold's dimension from nothing but a bound.

**`conditional` uses none of them.** Hand the decoder the blob's label and the
latent becomes redundant; the KL collapses to 0.002 and every dimension sits at
the prior. Reconstruction is still passable (mean squared error 0.031), because
the label already says where the point is, and that is posterior collapse — not
a bug, an optimum.

**`iwae` is best on the eight-sample bound and worst on the one-sample
estimate.** That is not a contradiction. The importance-weighted bound is a
better estimate of the marginal likelihood even when the posterior it produces
is a worse approximation to the true posterior; optimising a tighter bound
sharpens the estimate and tolerates a diffuse $q$.

The rest of the file is four ways to move along that frontier, and one way to
get the arithmetic wrong.

## Variants

**`standard`** — the original (Kingma & Welling 2014; Rezende et al. 2014).
Gaussian encoder, Gaussian decoder with a fixed variance, one latent sample per
step, exact analytic KL. This is the reference the other four deviate from.

**`conditional`** — encode and decode the label as a one-hot concatenated to the
input and to $z$. The model becomes $p(x\mid z,c)$, so a sample is drawn for a
chosen class: the demo's 200 draws per class land 0.03–0.05 from the right
centre. The cost is that $z$ stops carrying class identity, and here it stops
carrying anything.

**`iwae`** — replace the one-sample bound with the $K$-sample importance
weighted bound of Burda et al. (2016), $K=8$. Each step draws eight latents and
weights them by their own likelihood, so the gradient is dominated by the
samples that explain the point. The bound is tighter (monotonically in $K$ in
expectation) and each step costs eight times the decoding.

**`beta`** — multiply the KL by $\beta=8$, the $\beta$-VAE of Higgins et al.
(2017). The rate is capped, so the model stores less per dimension: the two
active dimensions drop from 1.57 and 1.36 nats to 0.78 and 0.77, mean squared
error rises from 0.0137 to 0.0167, and the reported ELBO falls to $-5.29$
because the objective being optimised is no longer the ELBO.

**`collapse_fix`** — $\beta=16$ plus free bits $\lambda=0.5$ (Kingma et al. 2016;
Chen et al. 2018). Each dimension's KL is floored at $\lambda$: dimensions below
the floor get no KL gradient at all, so the optimiser cannot switch them off for
free. Plain $\beta=16$ keeps two dimensions at 0.52 and 0.49 nats
and reconstructs with mean squared error 0.0655; the floor forces all eight
dimensions open at about 0.48 nats each, to 0.0224.

The five differ in which term of the bound they attack. `conditional` changes
the model, `iwae` changes the estimator, `beta` and `collapse_fix` change the
objective's weighting, and `standard` changes nothing.

## Derivation

### The generative model

Two ingredients. A prior on the code, and a decoder that turns a code into a
distribution over data:

$$
p(z)=\mathcal N(z;0,I),\qquad
p_\theta(x\mid z)=\mathcal N\big(x;\,\hat x_\theta(z),\,\sigma^2 I\big),
\qquad (1)
$$

with $\sigma^2$ fixed at 0.02 in this file. Everything the model can represent
is in $\hat x_\theta$, and everything it can be uncertain about is that fixed
noise.

### The marginal is the thing you want and cannot have

The likelihood of a point is the integral over all codes that could have made
it:

$$
p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz .
\qquad (2)
$$

For a linear decoder this is a Gaussian and the integral is closed form — that
is factor analysis and it is in chapter 06. For the two-layer tanh decoder here
it is not, and the dimension of $z$ makes quadrature hopeless.

### Jensen gives a bound, and the gap has a name

Introduce any distribution $q_\phi(z\mid x)$ that is easy to sample. Then

$$
\log p_\theta(x)
=\log\mathbb E_{q_\phi}\Big[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\Big]
\;\ge\;\mathbb E_{q_\phi}\Big[\log\frac{p_\theta(x,z)}
{q_\phi(z\mid x)}\Big]=\mathcal L,
\qquad (3)
$$

by Jensen, because $\log$ is concave. The gap is exactly the KL between the
approximation and the true posterior,

$$
\log p_\theta(x)-\mathcal L
=\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big),
\qquad (4)
$$

so maximising $\mathcal L$ in $\phi$ does two things at once: it pushes the
likelihood up and it pulls $q_\phi$ toward the posterior. That is why the same
gradient trains a model and an inference network.

### The Gaussian posterior makes the KL closed form

Take

$$
q_\phi(z\mid x)=\mathcal N\big(z;\,\mu_\phi(x),\,
\mathrm{diag}(\sigma^2_\phi(x))\big),
\qquad (5)
$$

written in code as `mu` and `lv = log sigma^2`. Against a standard normal prior
the KL is a sum over dimensions with no sampling in it:

$$
\mathrm{KL}\big(q_\phi\,\Vert\,p\big)
=\tfrac12\sum_d\Big(\mu_d^2+\sigma_d^2-1-\log\sigma_d^2\Big).
\qquad (6)
$$

This is the line `kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - 1.0 - lv, axis=1)`,
and it is worth noticing that (6) is a function of $\mu$ and $\log\sigma^2$
alone. Any gradient that flows into the encoder through the reconstruction uses
samples; the KL gradient does not.

### Reparameterisation is what makes the sample differentiable

The expectation in (3) is over $q_\phi$, whose parameters we want to
differentiate. Move the randomness out:

$$
z_\phi(\epsilon)=\mu_\phi+\sigma_\phi\odot\epsilon,\qquad
\epsilon\sim\mathcal N(0,I),
\qquad (7)
$$

so that

$$
\nabla_\phi\mathcal L
=\mathbb E_{\epsilon}\Big[\nabla_\phi\big(\log p_\theta(x,z_\phi(\epsilon))
-\log q_\phi(z_\phi(\epsilon)\mid x)\big)\Big].
\qquad (8)
$$

Now the sample is a deterministic function of $\phi$ and a fixed noise draw,
and the gradient is an ordinary chain rule through the decoder. The demo checks
both estimators against central differences and lands at $5.7\times10^{-9}$ and
$9.8\times10^{-10}$.

### The same objective has two single-sample gradients

For the Gaussian there are two algebraically different but equally valid
single-sample gradients, and which one you write down decides whether your code
is right.

Write $J=\partial\hat x/\partial z$ for the decoder Jacobian, so that
$\frac{x-\hat x}{\sigma^2}J$ is a row vector of length $\dim z$ — the
backpropagated reconstruction gradient.

**The pathwise form** differentiates everything, including $\log q$:

$$
\nabla_\mu\mathcal L=\frac{x-\hat x}{\sigma^2}J-z,
\qquad
\nabla_{\log\sigma^2}\mathcal L=\Big(\frac{x-\hat x}{\sigma^2}J-z\Big)
\odot\tfrac12\sigma\epsilon+\tfrac12 .
\qquad (9)
$$

The $\tfrac12\sigma\epsilon$ is $\partial z/\partial\log\sigma^2$: a factor of
$\sigma$ from the standard deviation and a factor of $\tfrac12$ from the
logarithm. Both are easy to drop and neither throws — they mis-scale the
gradient on $\log\sigma^2$ by the posterior's own width, which is the quantity
that gradient exists to set, so the run still trains and settles somewhere
wrong.

**The analytic form** uses (6) for the KL and reparameterises only the
reconstruction:

$$
\nabla_\mu\mathcal L=\mathbb E\Big[\frac{x-\hat x}{\sigma^2}J\Big]-\mu,\qquad
\nabla_{\log\sigma^2}\mathcal L=\mathbb E\Big[\Big(\frac{x-\hat x}{\sigma^2}J
\Big)\odot\tfrac12\sigma\epsilon\Big]-\tfrac12(\sigma^2-1).
\qquad (10)
$$

They agree in expectation and not per sample. The analytic form takes the
expectation of the reconstruction part before backpropagation, and its prior
rows are the means of the pathwise ones, $-\mu$ for the first and
$-\tfrac12(\sigma^2-1)$ for the second, because $\mathbb E[-z\odot
\tfrac12\sigma\epsilon]=-\tfrac12\sigma^2$. The analytic form therefore has
lower variance, so it is what `standard`, `conditional`, `beta` and
`collapse_fix` use. `iwae` needs the pathwise form, because an importance
weighted bound has no closed-form KL to differentiate.

The trap is to mix them. If you reparameterise the reconstruction but keep the
prior term in its **pathwise** form with the wrong sign, you get a model that
never converges and never throws: the prior term points away from the origin
along $z$ instead of toward it, the latent scale grows without bound, and the
reconstruction loss is swamped long after the first few hundred steps. The
finite-difference check in the demo exists because this is the mistake that does
not announce itself.

### The importance weighted bound

Draw $K$ latents per point and average their likelihood ratios, then take the
log:

$$
\mathcal L_K=\mathbb E\Big[\log\frac1K\sum_{k=1}^K w_k\Big],\qquad
w_k=\frac{p_\theta(x,z_k)}{q_\phi(z_k\mid x)},
\qquad (11)
$$

with $z_k$ reparameterised as in (7). Jensen on the inner average gives

$$
\log p_\theta(x)\;\ge\;\mathcal L_K\;\ge\;\mathcal L_{K-1}\;\ge\;\mathcal L_1,
\qquad (12)
$$

in expectation — the bound tightens with $K$, which is why the demo's iwae row
wins on the eight-sample column. Its gradient is a self-normalised average over
the samples in the batch:

$$
\tilde w_k=\frac{w_k}{\sum_j w_j},\qquad
\nabla_\phi\mathcal L_K
=\mathbb E\Big[\sum_{k=1}^K\tilde w_k\,\nabla_\phi\log w_k\Big],
\qquad (13)
$$

because $\nabla\log\frac1K\sum_k w_k=\sum_k\tilde w_k\nabla\log w_k$. The
weights are computed with a max shift before exponentiation; without it a
batch of eight samples with log weights around $-36$ underflows to a vector of
zeros and the gradient becomes `nan`.

The estimator is unbiased for the bound but the bound is not the likelihood. At
$K=8$ the demo estimates $-1.27$ nats for the iwae model against $-1.61$ for the
standard one, while the *same* iwae model scores $-7.61$ on the one-sample
estimate against $-1.89$. Measured on the held-out set, the iwae model's weights
put 45% of the mass on a single one of the eight samples and the effective
sample size is 3.5, against 5.6 for the standard model. A few samples carry the
bound; the single-sample estimate is then a draw that usually lands in a region
the weights call unimportant, which is exactly the lottery. A fit that is better
by the metric it optimises can be much worse by the metric you were using to
compare it.

### What beta does to the rate

$\beta$-VAE replaces (3) with

$$
\mathcal L_\beta=\mathbb E_{q}\big[\log p_\theta(x\mid z)\big]
-\beta\,\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big).
\qquad (14)
$$

Since the two terms are a rate and a distortion,

$$
\mathrm{rate}=\mathrm{KL}(q_\phi\,\Vert\,p),\qquad
\mathrm{distortion}=-\mathbb E\big[\log p_\theta(x\mid z)\big],
\qquad (15)
$$

$\beta$ is the exchange rate between them: a larger $\beta$ buys a lower rate
and pays for it in distortion. The demo's sweep is that frontier,
one point per $\beta$:

```
beta  1.0  kl  2.934  rec  1.731  elbo   -1.891  mse  0.0137
beta  2.0  kl  2.527  rec  1.679  elbo   -2.054  mse  0.0158
beta  4.0  kl  2.086  rec  1.667  elbo   -2.870  mse  0.0163
beta  8.0  kl  1.552  rec  1.656  elbo   -5.291  mse  0.0167
```

Rate falls by half and mean squared error rises by a fifth across the sweep, so
this manifold is cheap to encode and $\beta$ has little distortion to buy. Push
to $\beta=16$ and the rate falls to 1.011, spread over the same two dimensions
at 0.52 and 0.49 nats, and the distortion finally breaks: mean squared error
0.0655, five times the unweighted model. The two dimensions are still there;
there is just not enough rate left in them to say where on the circle a point
sits.

### Free bits is a floor, and a floor is a budget

The fix is not to lower $\beta$, it is to forbid the KL from going to zero
dimension by dimension. Replace the KL with a hinge:

$$
\mathcal L_{\text{free}}
=\mathbb E\big[\log p_\theta(x\mid z)\big]
-\sum_d\max\big(\lambda,\,\mathrm{KL}_d\big),
\qquad (16)
$$

where $\mathrm{KL}_d$ is the batch mean of dimension $d$'s contribution. The
gradient of the hinge with respect to $\mathrm{KL}_d$ is one when
$\mathrm{KL}_d>\lambda$ and zero otherwise, which is exactly the switch the code
implements: `(kld > self.free).astype(float)`.

The demo runs $\beta=16$ with $\lambda=0.5$ and finds all eight dimensions parked
at 0.47–0.49 nats — just *under* the floor, where the KL gradient switches off.
The floor is a budget of $8\times0.5=4$ nats and the optimiser fills it one
dimension at a time, because below $\lambda$ a dimension costs nothing; it does
not spend the budget where the data would. The ELBO recovers from $-10.90$ to
$-4.67$ and the reconstruction from 0.0655 to 0.0224, but neither returns to the
$\beta=1$ model's $-1.89$ and 0.0137. Free bits buys information flow, not
quality.

### Why a conditional model throws its latent away

Conditioning adds the label to both networks:

$$
q_\phi(z\mid x,c),\qquad p_\theta(x\mid z,c).
\qquad (17)
$$

Now the decoder can compute the conditional mean of $x$ given $c$ without
consulting $z$, and if it does, the only term in (3) that depends on the
encoder's output is $-\mathrm{KL}$, whose optimum is the prior. For a dimension
that the decoder ignores, the objective is minimised at

$$
\mu_d=0,\qquad \log\sigma_d^2=0,
\qquad (18)
$$

which is what the demo measures: KL 0.001 and zero active dimensions, stable
from 2000 to 8000 steps. Collapse is not the encoder failing; it is the decoder
winning.

## Code walkthrough

The data first, because it sets the difficulty. Eight centres on a circle, each
with isotropic noise, which gives a manifold that needs exactly two latent
dimensions:

```python
def make_data(seed: int, per_mode: int = 100) -> tuple:
    """Return (x, y) with `per_mode` points around each of eight centres."""
    rng = np.random.default_rng(seed)
    angle = 2.0 * np.pi * np.arange(NMODES) / NMODES
    centres = RADIUS * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    x = np.vstack([c + rng.normal(0.0, SPREAD, (per_mode, 2))
                   for c in centres])
```

Both networks are one hidden tanh layer. `_decode` reshapes to whatever leading
batch shape it was handed, so the same function serves a single sample `(n, d)`
and the `(n, k, d)` stack that `iwae` needs:

```python
def _encode(params: dict, xc: np.ndarray) -> tuple:
    h = np.tanh(xc @ params["E1"] + params["b1"])
    return h, h @ params["Em"] + params["bm"], h @ params["El"] + params["bl"]
```

```python
def _decode(params: dict, zc: np.ndarray) -> tuple:
    """Decode any leading batch shape; returns (hidden, reconstruction)."""
    shape = zc.shape[:-1]
    flat = zc.reshape(-1, zc.shape[-1])
    g = np.tanh(flat @ params["D1"] + params["d1"])
    out = g @ params["D2"] + params["d2"]
    return g, out.reshape(*shape, -1)
```

`_terms` computes the bound and hands back everything the backward pass needs.
The `onehot` argument is how the conditional variant reaches into the same
function: it is concatenated to `z` before decoding, so there is one code path
for both models:

```python
def _terms(params: dict, x: np.ndarray, xc: np.ndarray, eps: np.ndarray,
           onehot: np.ndarray | None = None) -> tuple:
    """Return (elbo, rec, kl, cache) per data point for one latent sample."""
    h, mu, lv = _encode(params, xc)
    sd = np.exp(0.5 * lv)
    z = mu + sd * eps
    zc = z if onehot is None else np.concatenate([z, onehot], axis=1)
    g, xr = _decode(params, zc)
    rec = (-0.5 * np.sum((xr - x) ** 2, axis=1) / DEC_VAR
           - 0.5 * x.shape[1] * np.log(2.0 * np.pi * DEC_VAR))
    kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - 1.0 - lv, axis=1)
    return rec - kl, rec, kl, (h, mu, lv, sd, z, zc, g, xr)
```

The backward pass is the analytic form of (10). Note the two halves: the
decoder chain is the usual backpropagation with `dxr` carrying $1/\sigma^2$,
and the encoder gets `dmu` and `dlv` where only the first term comes from the
data and the second is the closed-form KL:

```python
    dxr = (xr - x) / DEC_VAR / n
    grads = {"D2": g.T @ dxr, "d2": dxr.sum(0)}
    dg = (dxr @ params["D2"].T) * (1.0 - g ** 2)
    grads["D1"] = zc.T @ dg
    grads["d1"] = dg.sum(0)
    dz = (dg @ params["D1"].T)[:, :lv.shape[1]]
    dmu = dz + beta * (mu / n) * mask
    dlv = (dz * eps * 0.5 * sd
           + beta * 0.5 * ((np.exp(lv) - 1.0) / n) * mask)
```

`dz` is cut back to the latent width because `zc` may be `z` concatenated with
a one-hot; the label columns must not receive a gradient from the code. The
`beta * ...` terms are (6) differentiated, and `mask` is the free-bits switch of
(16) — an all-ones vector unless the variant is `collapse_fix`.

The importance weights are computed in log space, shifted, and only then
exponentiated:

```python
    n, k, latent = eps.shape
    _, mu, lv = _encode(params, x)
    sd = np.exp(0.5 * lv)
    z = mu[:, None, :] + sd[:, None, :] * eps
    _, xr = _decode(params, z)
    ss = np.sum((xr - x[:, None, :]) ** 2, axis=2)
    log_px = -0.5 * ss / DEC_VAR - np.log(2.0 * np.pi * DEC_VAR)
    log_pz = -0.5 * np.sum(z ** 2, axis=2) - 0.5 * latent * LOG_2PI
    log_qz = -0.5 * np.sum(eps ** 2 + lv[:, None, :], axis=2)
    log_qz -= 0.5 * latent * LOG_2PI
    return log_px + log_pz - log_qz
```

The three densities are the whole of (11): likelihood of the point under that
sample's reconstruction, prior density of the sample, and the proposal density
that produced it. `log_qz` is written in terms of `eps` rather than `z` because
that is the form that stays finite when $\sigma$ is tiny.

The IWAE gradient is the same chain rule with the per-sample weights applied to
each sample's contribution, then summed over $k$ after the decoder-backprop
step:

```python
    n, k, latent = eps.shape
    weight = _softmax_rows(iwae_log_weights(params, x, eps))
    h, mu, lv = _encode(params, x)
    sd = np.exp(0.5 * lv)
    z = mu[:, None, :] + sd[:, None, :] * eps
    g, xr = _decode(params, z)
    dxr = weight[:, :, None] * (xr - x[:, None, :]) / DEC_VAR / n
    dz = weight[:, :, None] * z / n
    flat_dxr = dxr.reshape(n * k, 2)
    flat_g = g.reshape(n * k, h.shape[1])
    grads = {"D2": flat_g.T @ flat_dxr, "d2": flat_dxr.sum(0)}
    dg = (flat_dxr @ params["D2"].T) * (1.0 - flat_g ** 2)
    grads["D1"] = z.reshape(n * k, latent).T @ dg
    grads["d1"] = dg.sum(0)
    dz = dz + (dg @ params["D1"].T).reshape(n, k, latent)
    dmu = dz.sum(axis=1)
    dlv = np.sum(dz * eps * 0.5 * sd[:, None, :], axis=1) - 0.5 / n
```

`dz` starts as the prior term of the pathwise gradient and then receives the
decoder's contribution in-place; `dmu` and `dlv` are sums over the $k$ samples
because the encoder is evaluated once per point, not once per sample. Every
line here is a sign that has to be right, which is why the demo
finite-differences this function too.

`_beta` and `_mask` are the two knobs that define `beta` and `collapse_fix`.
`_mask` recomputes the batch-mean KL from the *current* parameters on every
step, so the set of open dimensions can change during training:

```python
    def _beta(self) -> float:
        if self.variant == "beta":
            return self.beta
        if self.variant == "collapse_fix":
            return HARD_BETA
        return 1.0
```

```python
    def _mask(self, params: dict, xc: np.ndarray) -> np.ndarray | None:
        if self.variant != "collapse_fix":
            return None
        _, mu, lv = _encode(params, xc)
        kld = 0.5 * (np.mean(mu ** 2, axis=0) + np.mean(np.exp(lv), axis=0)
                     - 1.0 - np.mean(lv, axis=0))
        return (kld > self.free).astype(float)
```

Evaluation is deliberately separate from training. `elbo(..., importance=True)`
is the only place the $K$-sample bound is computed, and it refuses the
conditional model outright because the weight ratio is defined against the
unconditional prior:

```python
        if importance and self.conditional:
            raise ValueError("the iwae bound is unconditional only")
        if importance:
            log_w = iwae_log_weights(self.params, x, eps)
            top = log_w.max(axis=1, keepdims=True)
            return float(np.mean(np.log(np.exp(log_w - top).mean(axis=1))
                                 + top[:, 0]))
```

The `top` shift is the same trick as inside the weights, applied to the bound
itself: with 32 samples and log weights near $-36$ the plain mean underflows,
and the max-shifted form is exact up to rounding.

## Complexity

The unconditional network has

$$
(2\cdot48+48)+(2\cdot48\cdot8+2\cdot8)+(48\cdot8+48)+(48\cdot2+2)=1458
\qquad (19)
$$

— encoder weights and bias, the two heads, then the decoder's weights and bias.

parameters — encoder weights and bias, the two heads, then the decoder's. The
conditional model has 2226 because both the input and the decoder input grow by
eight. One forward pass costs about 1.3k multiply-adds per point;
a training step is a forward and a backward over a batch of 128, so roughly
$5\times10^{5}$ multiply-adds, and the 2000-step fit is about $10^{9}$. The demo
is nine fits — five variants, three extra $\beta$ values and the collapsed
model — so a few seconds of numpy.

The interesting cost is `iwae`: the encoder runs once per point but the decoder
runs $K$ times, so a step costs $K$ times the decode and the bound's memory is
$K$ copies of $(n,\text{latent})$. Eight samples is the usual compromise;
past a few dozen the bound improves by less than the sampler's variance.

## Failure modes

**The prior term's sign is the one to get wrong.** The loss gradient of the
prior is $+z$ pathwise and $+\mu$ in the analytic form, because the loss
contains $+\tfrac12 z^\top z$ from $-\log p(z)$. Writing $-z$ (which is what
you get from $\nabla\log w$ if you forget that the objective is $-\log w$)
makes the objective unbounded below along $z$: the prior stops pulling the code
toward the origin and starts pushing it out, while the reconstruction term is
quadratic and flat, so the run does not stall — it runs away. The first version
of this file did exactly that and the bound reached $-10^{170}$ within 500 steps.
Nothing in the code looks wrong, because nothing in the code is wrong except one
character; only a finite-difference check on the objective catches it.

**Conditional models collapse and it is not a transient.** KL $0.001$, zero
active dimensions at 2000, 4000 and 8000 steps, with reconstruction error more
than twice the unconditional model's. The decoder prefers the label because the
label is free information. If you need $z$ to matter — for interpolation, for
style transfer — you have to take something away: drop the label out randomly,
anneal $\beta$ upward, or apply free bits. None of those is in this file, and
that is the honest state of the variant.

**$\beta$ is not scale free, because $\sigma^2$ is fixed.** The reconstruction
term is $1/\sigma^2$ times a squared error while the KL is order one, so the
KL's effective weight is $\beta\sigma^2$. At `DEC_VAR = 0.02` the whole sweep is
shifted by a factor of fifty against a unit-variance decoder: $\beta=8$ here is
as strong as $\beta=0.16$ would be there, and $\beta=1$ is already a heavily
down-weighted KL. That is why this data tolerates $\beta=8$ with almost no
distortion cost, and why a $\beta$ from one codebase never transfers to another
unless $\sigma^2$ matches.

**"Active dimensions" cannot see rate starvation.** The demo's $\beta=1$ and
$\beta=16$ models both report two active dimensions, and $\beta=16$ reconstructs
five times worse. The count measures which dimensions
carry data, not how much; the per-dimension KL is what changes (1.57 and 1.36
against 0.52 and 0.49). Use both, and treat the count as a coarse screen.

**Free bits with a large floor forces capacity you did not ask for.** At
$\lambda=0.5$ and eight dimensions the floor is four nats, more than the
unconstrained model spends in total. The optimiser satisfies it by opening all
eight dimensions shallowly — 0.47 to 0.49 each — and reconstruction is worse
than the two-dimension solution. The floor is a target as well as a bound, and
setting it above what the data needs spends rate on dimensions that cannot
help.

**A tighter bound is not a better model.** The iwae row is best on eight samples
and worst on one. If you report the bound you trained, you cannot compare
variants trained with different $K$; if you report a single-sample estimate, you
are measuring the variance of an estimator as much as the model. The demo's
tables keep both columns for that reason.

**The ELBO is a lower bound on likelihood, not a calibrated likelihood.**
Nothing in the training signal prevents a VAE from assigning high density to
data it has never seen — the reconstruction term rewards confident decoders and
the KL only bounds the code. Anomaly detection by ELBO alone is a known trap
(Nalisnick et al. 2019); the usual repair is a likelihood ratio against a
background model, or a typicality test.

## Applications

**Compression and rate-distortion coding.** (14) is literally a rate-distortion
Lagrangian with $\beta$ as the multiplier, which is why VAE-like models show up
in learned image and audio codecs: the KL is the bit rate you pay for the code
and the reconstruction term is the distortion you accept. The $\beta$ sweep in
this file is the smallest possible version of that curve.

**Density estimation.** Importance weighted bounds are tight enough to rank
models on image benchmarks, at a fraction of the sequential cost of an
autoregressive likelihood, and the estimator is the same one used for marginal
likelihood comparison anywhere a latent variable makes the integral
intractable.

**Structured generation.** A conditional VAE is a generator you can steer: draw
$z$, fix the class, decode. The demo's sampling block is the whole recipe, and
the collapse in its table is why practical conditional generators add KL
annealing, free bits or an auxiliary loss that forces the code to predict
something.

**Latent spaces for downstream work.** Two dimensions recovered from an
eight-dimensional code is the standard outcome for data on a manifold, and the
recovered coordinates are useful for interpolation, nearest-neighbour search
and visualisation. The caveat is the same one (4) makes: the code is only as
faithful to the posterior as the bound is tight.

**As the inference network for something else.** The encoder is a learned
amortised posterior; every latent-variable model with intractable posteriors —
hierarchical models, sequence models, topic models — can use this file's
`_encode`/`_grads` pair as the pattern, with the decoder swapped for the
model's generative part.

## References

- Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes.
  arXiv:1312.6114.
- Rezende, D. J., Mohamed, S. and Wierstra, D. (2014). Stochastic
  backpropagation and approximate inference in deep generative models. ICML.
- Burda, Y., Grosse, R. and Salakhutdinov, R. (2016). Importance weighted
  autoencoders. ICLR. arXiv:1509.00519.
- Higgins, I. et al. (2017). $\beta$-VAE: learning basic visual concepts with a
  constrained variational framework. ICLR.
- Chen, T. Q. et al. (2018). Isolating sources of disentanglement in variational
  autoencoders. NeurIPS. arXiv:1802.04942.
- Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I. and
  Welling, M. (2016). Improved variational inference with inverse autoregressive
  flow. NeurIPS. arXiv:1606.04934.
- Sohn, K., Lee, H. and Yan, X. (2015). Learning structured output representation
  using deep conditional generative models. NeurIPS.
- Bowman, S. R. et al. (2016). Generating sentences from a continuous space.
  CoNLL. arXiv:1511.06349.
- Rainforth, T. et al. (2018). Tighter variational bounds are not necessarily
  better. ICML. arXiv:1802.02550.
- Doersch, C. (2016). Tutorial on variational autoencoders. arXiv:1606.05908.
- Blei, D. M., Kucukelbir, A. and McAuliffe, J. D. (2017). Variational inference:
  a review for statisticians. JASA.
- Lucas, J., Tucker, G., Grosse, R. and Norouzi, M. (2019). Don't blame the
  ELBO for posterior collapse. ICLR. arXiv:1901.05534.
- Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D. and Lakshminarayanan, B.
  (2019). Do deep generative models know what they don't know? ICLR.
