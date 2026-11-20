# Autoencoder anomaly detection — what the network cannot rebuild

`autoencoder_anomaly.py` · Density & anomaly · Tier 2 · 5 variants

## Intuition

An autoencoder is trained to reproduce its own input through a narrow bottleneck.
What it learns is the set of directions the data actually use — the manifold —
and a point that needs directions the bottleneck cannot represent comes back
distorted. The distortion is the score:

$$
e(x) = \frac{1}{d} \| x - g_\phi(f_\theta(x)) \|^2
\qquad (1)$$

Read that as a distance, not a probability, and everything else follows. The
score is large when a point is far from the learned manifold and small when it is
on it. Those are different questions. A point in the middle of a hole in the
data — a place the manifold spans but the data never occupies — sits *on* the
learned surface and reconstructs perfectly:

```
        ·   ·   ·   ·
      ·   ·       ·   ·        · real data
     ·   ·    o    ·   ·       o hole: low score, no data
      ·   ·       ·   ·        × shell: high score, no data
        ·   ×   ·   ·
```

The `o` and the `×` are both anomalies and the model sees only one of them. That
is not a defect of the optimiser; it is what "distance to a surface" means.

The simplest case makes the geometry exact. Take $f$ and $g$ linear, so the
composition is a single matrix $W = W_2 W_1$ of rank at most $k$. The
least-squares optimum over rank-$k$ matrices is the orthogonal projector onto the
top-$k$ principal subspace, so (1) is precisely the squared distance to the
$k$-dimensional PCA subspace and the total score is the discarded variance. An
autoencoder is PCA with a learned, nonlinear projection. Every property of PCA
that makes it a poor density model — it cannot tell a hole from a cluster, it
has one global scale — carries over, and the depth buys a curved manifold rather
than a flat one.

The five variants are four ways to make the manifold harder to fake and one way
to read the residual. The demo's central table is the eight-dimensional version
of the picture above: five hundred points on a circle of radius $0.5$ inside a
random plane in $\mathbb{R}^8$, thirty points in the hole at radius $0.1$, thirty
on a shell at radius $1.1$.

| variant | latent | train score | hole score | AUC hole | AUC shell |
|:--|--:|--:|--:|--:|--:|
| `ae` | 2 | 3.17e-04 | 3.44e-04 | 0.542 | 1.000 |
| `ae` | 8 | 2.46e-06 | 3.74e-06 | 0.585 | 1.000 |
| `denoising_ae` | 2 | 1.69e-03 | 5.48e-04 | 0.026 | 1.000 |
| `denoising_ae` | 8 | 2.01e-03 | 4.56e-04 | 0.005 | 1.000 |

The shell is found perfectly by both. The hole is a coin flip for the plain
autoencoder and *inverted* for the denoiser: the hole scores below the data.
That second number is the interesting one, and it is a theorem rather than a
quirk — see the derivation.

## Variants

| variant | score | what it changes |
|:--|:--|:--|
| `ae` | $\frac{1}{d}\|x - g(f(x))\|^2$ | nothing; the baseline |
| `denoising_ae` | same, trained on corrupted input | removes the identity shortcut |
| `vae` | $\|x - g(\mu)\|^2 + \beta\,\mathrm{KL}$ | puts a law on the latent code |
| `lstm_ae` | mean over $T$ and $d$ | sequences, with weights shared across time |
| `thresholding` | $\sum_j r_j^2 / v_j$ | the residual, per feature scale |

**`ae`** is a two-layer encoder and a two-layer decoder with a `latent`-wide
bottleneck between them. With `latent` equal to the manifold's dimension it is
the right model; with `latent` too small it becomes a prototype detector and
scores the wrong points; with `latent` large enough it can approach the identity
and score nothing. The demo pins all three regimes.

**`denoising_ae`** trains on $x + \sigma\varepsilon$ with the clean $x$ as the
target. Copying is no longer available, so the network is forced to learn the
structure that survives the corruption. The optimum of that objective is the
conditional mean $\mathbb{E}[x \mid \tilde x]$, which is a shrinkage operator:
it maps every input toward the centre of the manifold. That is what makes it
robust and what makes it blind to the hole. Use it when the inputs are genuinely
noisy or when the bottleneck cannot be trusted to be narrow.

**`vae`** replaces the deterministic code with a Gaussian posterior
$q(z \mid x) = \mathcal{N}(\mu(x), \mathrm{diag}\,\sigma^2(x))$ and scores the
negative ELBO: reconstruction plus the KL divergence from the prior. The KL term
is a prior on *where* a point may sit in code space, so an input that would need
an arbitrarily remote code is penalised twice. Use it when you want the score to
mean something in nats, or when you want to sample the learned distribution. The
`beta` argument is the weight; 1.0 is the exact ELBO and on a thin manifold it
collapses (see failure modes).

**`lstm_ae`** is the same idea for a window of $T$ steps: an LSTM reads the
sequence and returns its final state $(h_T, c_T)$, a second LSTM is run for $T$
steps on a zero input starting from that state, and a linear map turns each
output into a reconstruction of the corresponding step. The weights do not depend
on $t$, so the model represents a *phase* rather than a list of values, and a
window with an unseen phase reconstructs as well as a training window. Use it for
telemetry, audio, and anything else where the same pattern can start anywhere.

**`thresholding`** keeps the plain autoencoder and changes only the score. Each
feature's squared residual is divided by that feature's training residual
variance before summing, so a quiet feature's small absolute error counts as much
as a loud one's. The sum is then a quadratic form with a known reference law
under the model, which gives a threshold that does not need a contamination
budget.

## Derivation

### The objective and the linear case

Training minimises the mean squared reconstruction error,

$$
\min_{\theta,\phi}\; \frac{1}{nd}\sum_{i=1}^{n}\|x_i - g_\phi(f_\theta(x_i))\|^2
\qquad (2)$$

with $f_\theta : \mathbb{R}^d \to \mathbb{R}^k$ and $g_\phi : \mathbb{R}^k \to
\mathbb{R}^d$. Nothing in (2) mentions a density, and that is the point: the
objective asks for a low-dimensional description that loses as little squared
error as possible, which is a statement about the data's second moment, not its
distribution.

Let $f(x) = W_1 x$ and $g(z) = W_2 z$. Then $g(f(x)) = Wx$ with
$W = W_2W_1$ and $\mathrm{rank}\,W \le k$. Write $\hat\Sigma$ for the sample
covariance and $\lambda_1 \ge \dots \ge \lambda_d$ for its eigenvalues. The
Eckart–Young theorem gives

$$
\min_{\mathrm{rank}\,W \le k} \frac{1}{n}\|X - XW\|_F^2
= \sum_{j>k}\lambda_j,
\qquad W^\star = U_kU_k^\top
\qquad (3)$$

so the linear autoencoder is the projector onto the top-$k$ principal subspace
and its score is $\|(I - U_kU_k^\top)x\|^2$, the squared distance to that
subspace. Two consequences survive the move to nonlinear $f,g$: the training
error is a *reconstruction* error and therefore bounded by the discarded
variance, and the score is a distance to a set rather than a height of a density.

### Why the hole is invisible

A point $x$ on the manifold with noise $\epsilon \sim \mathcal{N}(0,
\sigma^2 I)$ has residual $x - g(f(x)) \approx (I - J_gJ_f)\epsilon$ if the
manifold is learned exactly, so $e(x) \approx \sigma^2$ and the score measures
noise. A point at distance $\delta$ off the manifold along a direction the
bottleneck cannot represent has $e(x) \approx \sigma^2 + \delta^2$: the score
measures distance. A point *inside* the span of the manifold has $\delta = 0$ by
construction, so it scores like data. There is no $\delta$ to measure.

The demo makes the failure quantitative in two dimensions. A circle of radius
$0.5$ has intrinsic dimension one. With a one-dimensional bottleneck the model
cannot represent the circle at all — it becomes a prototype, the training error
is $5.88\times10^{-2}$, and the hole at radius $0.1$ lands closer to the single
code than the circle does. The hole's AUC is $0.093$: it is ranked *below every
real point*. With a two-dimensional bottleneck the training error falls to
$4.30\times10^{-6}$ and the same hole scores $0.924$. A third dimension buys
nothing ($6.80\times10^{-6}$). The bottleneck width is the manifold dimension,
and it is not something the data tells you.

In eight dimensions, with a bottleneck of two or more, the hole returns to
chance (AUC $0.52$–$0.59$) no matter how wide the bottleneck gets. Capacity does
not rescue it, because the hole is inside the span at every width.

### Denoising estimates a conditional mean

The denoising objective is

$$
\min_{g}\; \mathbb{E}_{x,\tilde x}\|x - g(\tilde x)\|^2 ,
\qquad \tilde x = x + \sigma\varepsilon
\qquad (4)$$

and it separates pointwise: for each $\tilde x$ the minimiser is

$$
g^\star(\tilde x) = \mathbb{E}[x \mid \tilde x].
\qquad (5)$$

That is the whole difference. The plain autoencoder's target is the input, so
the identity is a perfect solution whenever the bottleneck permits it. The
denoiser's target is the *clean* input, and the best it can do is the posterior
mean — a smooth map from the corrupted point to the centre of the mass that could
have produced it.

Equation (5) explains the inverted AUC. Near the hole, the conditional mean of
the data given a noisy observation at the centre is the centre itself, so the
hole reconstructs with error below the data's own noise level. The denoiser does
not merely fail to see the hole; it prefers it. The effect grows with
$\sigma$: at `corruption=0.3` the hole's AUC is $0.026$ at `latent=2` and $0.005$
at `latent=8`.

### The variational bound

For a latent variable model $p(z) = \mathcal{N}(0, I)$ and decoder
$p(x \mid z) = \mathcal{N}(g(z), \tau^2 I)$,

$$
\log p(x) \;\ge\; \mathbb{E}_{q}\Big[-\frac{1}{2\tau^2}\|x - g(z)\|^2\Big]
- \mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big) + \mathrm{const}
\qquad (6)$$

so maximising the bound means minimising reconstruction plus a KL penalty. With
$q = \mathcal{N}(\mu, \mathrm{diag}\,\sigma^2)$ the penalty is closed form:

$$
\mathrm{KL} = \frac{1}{2}\sum_{j=1}^{k}
\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)
\qquad (7)$$

which is zero exactly when $\mu = 0$ and $\sigma^2 = 1$, and the code checks it
to $10^{-12}$: the demo builds an encoder that emits a fixed $(\mu, \log\sigma^2)$
and a zero decoder, so the score is the KL alone. For $\mu = (1, 0)$ and unit
variance the score is $0.500000$; for $\mu = 0$ and $\sigma^2 = 0.25$ it is
$0.318147$.

The gradient needs the sampling to be differentiable. The reparameterisation

$$
z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
\qquad (8)$$

moves the randomness into $\varepsilon$, so $\partial z/\partial\mu = 1$ and
$\partial z/\partial\sigma = \varepsilon$ and backpropagation reaches both heads.
The code's loss is (6) with $\tau^2 = 1/2$; `beta` absorbs the ratio
$1/(2\tau^2)$, so `beta=1` is the ELBO for that decoder variance and any other
value is a $\beta$-VAE.

### Why the VAE collapses

The KL is charged per latent dimension; the reconstruction gain is bounded by the
data variance. For data on a manifold of intrinsic dimension $m$ embedded in
$k \ge m$ code dimensions, encoding the position costs roughly
$\frac{1}{2}\sum_j \mu_j^2$ while the gain is at most the variance the code
explains, and the sampled code carries noise $\sigma$ that the decoder must
average out. Past a small $\beta$ the constant solution — $\mu = 0$,
$\sigma^2 = 1$, KL exactly zero — wins, and no amount of training leaves it
because it is a genuine optimum, not a local trap.

On the circle the transition is sharp and sits between 0.2 and 0.3:

| `beta` | reconstruction mse | mean $\|\mu\|$ |
|--:|--:|--:|
| 0.03 | 1.23e-03 | 1.297 |
| 0.10 | 5.60e-03 | 1.100 |
| 0.30 | 1.25e-01 | 0.004 |

At `beta=0.3` the latent mean has fallen to $0.004$ and the reconstruction error
is $0.125$ — the variance of the circle itself. The model has learned to output
the mean. The default is `beta=0.1` for that reason, and the demo's KL check
runs at `beta=1` where the formula is the textbook one.

### The LSTM cell

The encoder is an LSTM with the standard gates, stacked into one matrix for
speed:

$$
\begin{aligned}
i_t &= \varsigma(W_i x_t + U_i h_{t-1} + b_i) \\
f_t &= \varsigma(W_f x_t + U_f h_{t-1} + b_f) \\
o_t &= \varsigma(W_o x_t + U_o h_{t-1} + b_o) \\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)
\end{aligned}
\qquad (9)$$

The forget gate is what makes the state a summary rather than a queue: $c_t$ can
hold a value across many steps, which is how a phase survives from the start of
the window to the end. The decoder is the same cell run on a zero input with
$(h_T, c_T)$ as its initial state, so all the information about the window has to
pass through $k$ numbers. Backpropagation through time is the chain rule applied
to (9) in reverse; the only subtlety is that the cell gradient has two parents,
$c_t \to c_{t+1}$ and $c_t \to h_t$, and both must be added.

### Standardised residuals

For the `thresholding` variant, let $r_j = x_j - y_j$ and let $v_j$ be the mean
squared training residual of feature $j$. The score is

$$
s(x) = \sum_{j=1}^{d}\frac{r_j^2}{v_j}
\qquad (10)$$

If the $v_j$ were the true residual variances, each term would be $\chi^2_1$ and
the sum $\chi^2_d$, giving a threshold with no tuning: $7.815$ at the $95\%$
level for $d = 3$. The demo's quiet-feature probe is five standard deviations out
on a feature whose spread is $0.1$ while another feature's is $10$. The
unweighted score ranks it at the $37.5$th percentile; (10) puts it at the
$99.8$th, with a score of $26.72$ against a threshold of $8.29$. The training
set's mean standardised score is exactly $d = 3$ by construction, which the demo
asserts.

The reference law is not exact, because $v_j$ is estimated from the same data:
the in-sample score has a heavier tail than $\chi^2_3$ and the chi-square
threshold flags $6.2\%$ of the training set instead of $5\%$. On fresh data from
the same distribution the miss is worse — $7.7\%$ — because the in-sample
residuals are smaller than out-of-sample ones. That is the price of a
distribution-free threshold, and it is the honest way to report it.

## Code walkthrough

The file is a small autodiff engine, an LSTM, and a detector, in that order. No
framework is involved: every gradient below is written out and every one of them
is checked by finite differences during development.

### The MLP and its two backward passes

`_mlp` stores both the pre-activations and the post-activations at every layer,
because the backward pass needs the latter and the input gradient needs both:

```python
    layers = len(params) // 2
    pres, posts = [], [X]
    for i in range(layers):
        z = X @ params[2 * i].T + params[2 * i + 1]
        pres.append(z)
        X = np.tanh(z) if i < layers - 1 else z
        posts.append(X)
    return X, (pres, posts)
```

The output layer is linear: a tanh on the last layer would clip the reconstruction
and put a ceiling on the score. `_mlp_back` turns $\partial L/\partial y$ into
parameter gradients, walking backwards:

```python
    for i in range(layers - 1, -1, -1):
        grads[2 * i] = delta.T @ posts[i]
        grads[2 * i + 1] = delta.sum(axis=0)
        if i:
            delta = (delta @ params[2 * i]) * (1.0 - posts[i] ** 2)
```

`delta.T @ posts[i]` is the outer product summed over the batch; `delta.sum(axis=0)`
is the bias. The `1.0 - posts[i] ** 2` is the tanh derivative written in terms of
its own output — the one line that is wrong if you differentiate `tanh` as
`1 - tanh(z)` on the *pre*-activation, which is a common slip. `_mlp_input` is
the same walk without the parameter accumulation, returning $\partial L/\partial x$:

```python
    posts = cache[1]
    for i in range(len(params) // 2 - 1, 0, -1):
        delta = (delta @ params[2 * i]) * (1.0 - posts[i] ** 2)
    return delta @ params[0]
```

The loop stops at `1`, not `0`, because the input has no activation to
differentiate. That function is how the encoder learns: the decoder's input
gradient is the encoder's output gradient, and without it the encoder would have
no gradient at all.

### Adam on a list

Parameters are a flat list of arrays, so the optimiser is a loop and nothing
else. The bias correction is the part that matters:

```python
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            self.p[i] -= rate * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + 1e-8)
```

Dividing by `c1` and `c2` is what makes the first steps full-sized instead of
$\sqrt{1 - 0.999} \approx 0.03$ times too small. Without it the network takes
thousands of epochs to leave its initialisation, and the demo's 600 would not be
enough.

### The LSTM

`_lstm_forward` runs the gates in one matrix product and slices the result:

```python
        z = X[:, t] @ W.T + h @ U.T + b
        i = _sigmoid(z[:, :width])
        f = _sigmoid(z[:, width:2 * width])
        o = _sigmoid(z[:, 2 * width:3 * width])
        g = np.tanh(z[:, 3 * width:])
        c_prev, h_prev = c, h
        c = f * c_prev + i * g
        tc = np.tanh(c)
        h = o * tc
```

Four gates in one $4h \times (d + h)$ product rather than four separate ones: the
gate order inside the slice is a contract between forward and backward, and the
cache tuple records it. `_lstm_backward` reverses the chain, and the only
non-obvious line is the gate-gradient concatenation:

```python
        dz = np.concatenate([
            dc * g * i * (1.0 - i),
            dc * c_prev * f * (1.0 - f),
            do * o * (1.0 - o),
            dc * i * (1.0 - g ** 2),
        ], axis=1)
```

Each entry is $\partial L/\partial z$ for one gate, in the same order the forward
pass sliced them. The input gate multiplies $g$, the forget gate multiplies
$c_{t-1}$, the output gate multiplies $\tanh(c_t)$ and the cell gate multiplies
$i$. Getting one of those four wrong still trains — it just trains the wrong
model, which is why the development check was a finite-difference comparison on
every parameter, not a loss curve.

### The plain autoencoder

`_fit_ae` is one minibatch loop. The two lines that matter are the gradient
chain and the loss scale:

```python
                delta_out = 2.0 * diff / (len(index) * d)
                d_code = _mlp_input(dec, dec_cache, delta_out)
                grads = _mlp_back(enc, enc_cache, d_code)
                grads += _mlp_back(dec, dec_cache, delta_out)
                opt.step(grads)
```

`2 * diff / (batch * d)` is $\partial/\partial y$ of the mean squared error over
both the batch and the features — the `d` is there because the loss is a mean,
and dropping it multiplies the effective learning rate by $d$. The encoder's
gradient is the decoder's *input* gradient, which is why `_mlp_input` exists.
`seen` is the corrupted copy when `noisy` is set, while `diff` is always against
`clean`:

```python
                clean = X[index]
                seen = clean
                if noisy:
                    seen = clean + self.corruption * rng.standard_normal(clean.shape)
```

That pair of lines is the entire difference between `ae` and `denoising_ae`.

### The VAE

The VAE adds the reparameterisation, the closed-form KL, and the gradient of the
KL with respect to both heads:

```python
                mu = head[:, :k]
                logvar = np.clip(head[:, k:], -8.0, 8.0)
                sd = np.exp(0.5 * logvar)
                eps = rng.standard_normal(mu.shape)
                code = mu + sd * eps
                out, dec_cache = _mlp(dec, code)
                diff = out - clean
                kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar, axis=1)
                total += float(np.sum(diff ** 2)) + self.beta * float(kl.sum())
```

The `clip` on `logvar` is not cosmetic: $\exp(\log\sigma^2)$ overflows to `inf`
within a few epochs if the encoder is allowed to drive the variance to zero, and
one `inf` in the KL poisons every parameter on the next step. The loss is a sum
over features, not a mean, because (6) is a log-likelihood and a mean would
silently change the balance between the two terms by a factor of $d$.

The backward pass differentiates through the sampling:

```python
                d_code = _mlp_input(dec, dec_cache, delta_out)
                d_mu = d_code + self.beta * mu / m
                d_logvar = (d_code * eps * 0.5 * sd
                            + self.beta * 0.5 * (np.exp(logvar) - 1.0) / m)
                delta_head = np.concatenate([d_mu, d_logvar], axis=1)
```

`d_code * eps * 0.5 * sd` is the chain through $z = \mu + \sigma\varepsilon$
with $\partial\sigma/\partial\log\sigma^2 = \sigma/2$; the second term in each
line is the KL's gradient. `m` is the batch size because the loss is a batch
mean. Concatenating the two heads back into one delta is what lets `_mlp_back`
treat the encoder as an ordinary network with $2k$ outputs.

### Scoring and the threshold

`score_samples` dispatches on the variant and each branch is one line:

```python
        if self.variant == "lstm_ae":
            return ((X - recon) ** 2).mean(axis=(1, 2))
        if self.variant == "thresholding":
            return (((X - recon) ** 2) / self.residual_ ** 2).sum(axis=1)
        if self.variant == "vae":
            head, _ = _mlp(self.enc_, X)
            mu = head[:, :self.latent]
            logvar = np.clip(head[:, self.latent:], -8.0, 8.0)
            kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar, axis=1)
            return ((X - recon) ** 2).sum(axis=1) + self.beta * kl
```

The VAE branch scores the *mean* of the posterior, not a sample: a sampled score
would not be deterministic, and the mean is the minimum-expected-error code.
`thresholding` divides before summing, which is (10) exactly. The default branch
is the per-feature mean of (1).

`_threshold` is deliberately narrow about what it will claim:

```python
        if self.contamination is not None:
            return float(np.quantile(self.scores_, 1.0 - self.contamination))
        if self.variant == "thresholding":
            return float(_chi2_ppf(self.alpha, self.n_features_))
        raise ValueError("contamination=None is only calibrated for the "
                         "thresholding variant; pass a contamination budget")
```

Only `thresholding` has a null distribution to appeal to; the others get a
quantile of their own training scores, which is a budget and not a calibration.
Asking for `contamination=None` from `ae` raises instead of inventing a number.
`_chi2_ppf` is the Wilson–Hilferty transform, accurate to $0.5\%$ at $d = 3$
($7.775$ against the exact $7.815$) and to $2.5\%$ at $d = 1$, which is well
inside the error of any threshold anyone will use.

## Complexity

Let $n$ be the sample count, $d$ the features, $h$ the hidden width, $k$ the
latent width, $T$ the window length and $E$ the epochs.

| step | time | memory |
|:--|:--|:--|
| one `ae` epoch | $O(n\,(dh + hk + hd)) = O(nh(d+k))$ | $O(nh)$ for the caches |
| one `vae` epoch | the same, plus $O(nk)$ | the same |
| one `lstm_ae` epoch | $O(nT(h(d+h) + h^2))$ | $O(nTh)$ for the BPTT caches |
| Adam step | $O(dh + hk)$ per layer | three copies of the parameters |
| scoring | one forward pass, $O(nh(d+k))$ | $O(nk)$ |

The plain autoencoder's cost is linear in every parameter and dominated by the
first and last matrices; the bottleneck is never the expensive part. The LSTM is
quadratic in $h$ because of $U$, and linear in $T$ because the recurrence cannot
be parallelised — that is the real cost of the sequence variant, not the extra
parameters.

The demo's largest fits are $500 \times 8$ with $h = 32$ for 600 epochs, six of
them, and the whole file runs in a few seconds. Memory is the training set plus
the caches; for $n$ in the millions you would stream minibatches rather than hold
`data_`, which this implementation does not do.

## Failure modes

**The hole.** A point inside the span of the learned manifold scores like data.
The eight-dimensional demo puts the AUC at $0.542$, $0.524$ and $0.585$ for
bottlenecks of 2, 4 and 8 — no signal at any width. The diagnostic is to compare
against a density model on the same data, or to check the score at points
interpolated between clusters: if they score low, the model is a distance and not
a density. The fix is a different atom (`kde`, `hbos`, `gmm`), not a different
autoencoder.

**The denoiser's inversion.** `denoising_ae` scores the hole *below* the data
(AUC $0.026$ at `latent=2`, $0.005$ at `latent=8`) because the conditional mean
pulls the centre of the manifold toward itself. If you use a denoiser and see a
suspiciously clean score on your most obviously wrong points, this is why. The
diagnostic is to plot score against distance from the data centroid; a denoiser's
score is non-monotone in that distance.

**The bottleneck width.** Too narrow and the model is a prototype: the circle
with `latent=1` has a training error of $5.88\times10^{-2}$ and ranks the hole at
AUC $0.093$, below every real point. Too wide and it can approach the identity;
in eight dimensions `latent=8` reaches a training score of $2.46\times10^{-6}$,
which is the network memorising rather than modelling. The diagnostic is the
training-error curve against `latent`: the elbow is the manifold dimension, and a
score that keeps falling past the elbow is capacity you are spending on noise.

**Calibration.** The default threshold is a quantile of the training scores, and
those are optimistic: the model has seen the data. The `thresholding` variant
makes the size of the miss visible, because it has a reference law — $6.2\%$ of
the training set flagged at a nominal $5\%$, and $7.7\%$ of fresh data. For the
other variants the miss is unmeasurable from inside the model, which is the
argument for holding out a clean set and setting the quantile there.

**Scale.** The loss is on the raw scale, so a feature with variance $100$ is
worth ten thousand times a feature with variance $0.01$. In the demo the network
reproduces the loud feature to $2\%$ of its spread and the quiet one to $97\%$,
and the quiet feature's five-sigma excursion is invisible to the unweighted
score. Standardise before fitting, or use `thresholding`, which reweights after
the fact. Neither fixes a feature that is constant in training and varies in
deployment.

**VAE collapse.** At `beta=1` on a thin manifold the encoder falls into the prior
and reconstructs the mean; the demo shows the latent norm collapsing from $1.30$
at `beta=0.03` to $0.004$ at `beta=0.3`. The diagnostic is the mean latent norm
and the reconstruction error of the training set: if both are flat, the KL has
won. Lower `beta`, or standardise the data so the prior's scale matches.

## Applications

**Tabular anomaly detection** is the honest use: fit on a clean reference set,
score new rows, set the threshold on held-out data. It is the right tool when the
data lies near a low-dimensional curved manifold and the anomalies are off it —
sensor arrays, calibrated instruments, transaction features after a
transformation.

**Sequence monitoring** is where `lstm_ae` earns its place. A window of
telemetry that no longer looks like any window the model has seen reconstructs
badly, and the weights shared across time mean the model does not care where in
the window the pattern starts. The demo's sixty unseen phases score $5.10\times
10^{-4}$ against $4.84\times10^{-4}$ for the training phases, while a spike
scores $2.68\times10^{-2}$ and a doubled frequency $9.95\times10^{-1}$.

**Image and audio defect detection** is the classical denoising application:
train on known-good patches with corruption, score by reconstruction. It works
when defects are local and the good class is broad; it fails on the hole, so it
is a poor choice for "missing part" defects that sit inside the normal span.

**Dimensionality reduction as a by-product.** The latent code is a nonlinear
projection, so the same fit gives you a visualisation and a score. Do not confuse
the two: the code is optimised for reconstruction, not for separating classes.

**Where it is a trap.** As a density model, on data with holes, on data whose
anomalies are new *combinations* of normal features rather than new values, and
on anything where a per-feature model already works — `hbos` will beat it there
and you will be able to explain why. If the anomaly is a single coordinate out of
range, no amount of manifold learning is needed.

## References

- Hinton, G. E. & Salakhutdinov, R. R. (2006). Reducing the dimensionality of
  data with neural networks. *Science* 313(5786), 504–507. The paper that made
  the bottleneck respectable.
- Eckart, C. & Young, G. (1936). The approximation of one matrix by another of
  lower rank. *Psychometrika* 1(3), 211–218. The linear case, exactly.
- Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A. (2008). Extracting
  and composing robust features with denoising autoencoders. *ICML*. The
  conditional-mean argument and the corruption schedule.
- Kingma, D. P. & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
  The ELBO, the reparameterisation trick, and the closed-form KL.
- Rezende, D. J., Mohamed, S. & Wierstra, D. (2014). Stochastic backpropagation
  and approximate inference in deep generative models. *ICML*. The same trick,
  independently.
- Higgins, I. et al. (2017). $\beta$-VAE: learning basic visual concepts with a
  constrained variational framework. *ICLR*. Where the `beta` knob and its
  collapse behaviour come from.
- Sakurada, M. & Yairi, T. (2014). Anomaly detection using autoencoders with
  nonlinear dimensionality reduction. *MLSDA*. The reconstruction-score recipe.
- Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P. & Shroff, G.
  (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection.
  arXiv:1607.00148. The sequence variant.
- Hawkins, S., He, H., Williams, G. & Baxter, R. (2002). Outlier detection using
  replicator neural networks. *DaWaK*. An early statement of the reconstruction
  score, and of its limitations.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation* 9(8), 1735–1780. The cell in (9).
