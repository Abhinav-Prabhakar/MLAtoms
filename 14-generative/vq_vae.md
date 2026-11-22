# VQ-VAE — a codebook instead of a Gaussian

A VAE puts a continuous Gaussian behind every input. VQ-VAE puts a *dictionary*
there instead: the encoder's output is snapped to the nearest of $K$ learned
vectors, and only that vector reaches the decoder. The latent is an integer.

That one change buys three things a Gaussian VAE cannot offer. The posterior
can be a point mass, so the decoder never averages two explanations into a blur.
The latent costs $\log_2 K$ bits instead of a float vector. And an integer is a
token, so a second model — a transformer, a PixelCNN — can learn the
distribution over tokens and generate by sampling them.

The price is a derivative that does not exist. Nearest-neighbour is piecewise
constant, so $\partial z_q/\partial z_e = 0$ almost everywhere and the encoder
receives no gradient at all. Every variant below is an answer to that one
problem, and every failure below is a consequence of it.

## Intuition

Consider the eight blobs this chapter keeps returning to. The trouble with a
Gaussian latent is not that the posterior is unimodal but that the *prior* is:
$\mathcal{N}(0, I)$ puts density everywhere on the plane, including in the gaps
between blobs. The model must explain that density somehow, and the cheapest
explanation a smooth decoder can offer is to put mass between the modes. The
standard normal also pulls the aggregate posterior toward a single blob, so the
gaps fill in and samples drift off the ring.

The fix is to stop asking the decoder to interpolate. Give it a *symbol* that
means "the third blob", and it can answer with a distribution centred there and
nowhere else. Interpolation is impossible because there is nothing between
symbol 3 and symbol 4 — the latent space is a finite set, and the prior lives on
those points only.

```
   VAE                          VQ-VAE
   z ~ N(mu, sigma)             z in {e_1, ..., e_K}
   continuous, ordered          discrete, no neighbours
   decoder must hedge           decoder commits to a cell
```

The dictionary is tiny: $K$ vectors of dimension $D$ describe the whole latent
space, here $16\times 8 = 128$ numbers. Compare that with the 4866 parameters of
the decoder next to it. The model does not store the data in the codebook; it
stores the *vocabulary* the decoder speaks, and the decoder stores the meaning.

The awkward part is training. Write the forward pass as

```
   x ──► encoder ──► z_e ──► argmin_j ‖z_e - e_j‖ ──► e_k ──► decoder ──► x̂
```

and ask what $\partial \hat{x}/\partial z_e$ is. The argmin returns an index.
Perturb $z_e$ by a hair and the index does not change, so the derivative is
exactly zero; perturb it enough to cross a boundary and the output jumps. The
map has no usable derivative anywhere, which is why the encoder would never
train by ordinary backpropagation.

VQ-VAE's answer is to lie about the derivative in a controlled way and to fix up
the lie with two extra loss terms. The reconstruction gradient is copied from
$z_q$ to $z_e$ as if the quantiser were the identity (the *straight-through*
estimator). The codebook is moved toward the encoder's outputs by a second term,
and the encoder is pulled toward the codebook by a third. The three terms
together are the whole algorithm.

## Variants

**`vq_vae`** — the original. One codebook of 16 vectors in 8 dimensions, trained
with Adam on the codebook loss, $\beta = 0.25$, straight-through gradients. This
is the paper's recipe with the paper's hyperparameters, and on this data it is
the weakest model in the table: 0.0225 reconstruction error, 2.3 times the floor,
with only 12 of its 16 codes carrying any mass.

**`vq_vae2`** — the VQ-VAE-2 recipe applied to the same single-level model: if a
code has not won an assignment for 25 consecutive steps, teleport it onto a
random encoder output from the current batch. This is not the full VQ-VAE-2
architecture (which adds a coarse level and a PixelCNN prior over codes); it is
the codebook *maintenance* half, which is the part that fixes collapse. Here it
recovers all 16 codes with 48 re-seeds and cuts the error to 0.0105, a hair above
the floor.

**`ema_codebook`** — replace the codebook's optimizer with an exponential moving
average of the assignments, as in VQ-VAE-2. There is no learning rate, no Adam
state, and no possibility of the dictionary overshooting; each entry is dragged
directly toward the mean of the vectors that chose it. It reaches 0.0118 with 11
of 16 codes alive, and — the interesting part — its entries sit 0.059 of a latent
norm from the mean of their own cell, against 0.388 for the gradient-trained
`vq_vae`.

**`commitment_loss`** — the same architecture with $\beta = 4$ instead of
$0.25$. One number, and the best reconstruction in the discrete half of the
table: 0.0079, below the data's own variance, because the decoder has learned to
denoise. It is also the variant whose codebook actually converges, for a reason
the derivation makes precise: $\beta$ is the coefficient on the term that stops
the encoder from running away from the codes.

**`straight_through`** — replaces the identity surrogate with the exact gradient
of a temperature-relaxed quantiser, evaluated at the hard assignment. No lie:
the relaxed map $z_q = \sum_k w_k(z)e_k$ is differentiable, and its Jacobian turns
out to be a covariance matrix. Here $\tau = 1$ relative to the codebook's own
spread, and the relaxed gradient has cosine $+0.53$ with the straight-through
one — same direction on average, very different per coordinate.

**`residual_vq`** — two codebooks, the second quantising the first one's
residual. The vocabulary becomes $K^2 = 256$ nominal pairs, the latent becomes a
sum of two codes, and the error falls to 0.0026, 3.7 times below the floor. This
is the construction behind SoundStream, EnCodec and DAC: a stack of codebooks,
each fixing what the previous one missed, traded off against bitrate.

## Derivation

### The three terms

Let $z_e(x)$ be the encoder output and $e_1,\dots,e_K$ the codebook. The
quantiser is a hard nearest-neighbour lookup:

$$
z_q(x) = e_k, \qquad k = \arg\min_j \lVert z_e(x) - e_j \rVert^2 .
\qquad (1)
$$

The objective has three pieces, and the placement of the stop-gradient operator
$\mathrm{sg}$ is the whole design:

$$
\mathcal{L} = \log p(x \mid z_q(x))
 + \lVert \mathrm{sg}[z_e(x)] - e \rVert^2
 + \beta \lVert z_e(x) - \mathrm{sg}[e] \rVert^2 .
\qquad (2)
$$

The first term trains the encoder and decoder through the surrogate derivative.
The second trains only the codebook: $\mathrm{sg}$ freezes $z_e$, so
$\nabla_{e}\mathcal{L} = 2(e - z_e)$. The third trains only the encoder:
$\mathrm{sg}$ freezes $e$, so the gradient reaches $z_e$ and stops. Read the
second and third terms as a leash: the codebook chases the encoder, the encoder
is held near the codebook, and the two meet in the middle.

The reconstruction term is a fixed-variance Gaussian,
$\log p(x\mid z_q) = -\lVert x - \hat{x}\rVert^2/(2\sigma^2) + \mathrm{const}$,
with $\sigma$ set to the data's own spread rather than learned. That is why the
demo's `/floor` column is the reconstruction error divided by $2\sigma^2$: a
perfect model scores 1.0, and the number is the excess negative log-likelihood up
to a constant every row shares.

### Why the codebook term is k-means

Sum the second term over a batch and drop the stop-gradient, which does not
affect the minimiser:

$$
\sum_i \lVert z_{e,i} - e_{k_i} \rVert^2 .
$$

Differentiate with respect to one entry $e_j$ and set to zero:

$$
\frac{\partial}{\partial e_j} \sum_i \lVert z_{e,i} - e_{k_i} \rVert^2
 = -2 \sum_{i \,:\, k_i = j} (z_{e,i} - e_j) = 0
\;\Rightarrow\;
e_j = \frac{1}{n_j} \sum_{i \,:\, k_i = j} z_{e,i} .
\qquad (3)
$$

That is the M-step of Lloyd's algorithm. Equation (1) is the E-step. So the
latent half of VQ-VAE is running k-means on the encoder's outputs, jointly with
representation learning — the encoder reshapes the point cloud and k-means tiles
it. The demo checks the fixed point directly: a converged code sits 0.038 to
0.059 of a latent norm from the mean of its own cell, while `vq_vae` at
$\beta = 0.25$ sits 0.388 away because its encoder never stops moving.

This also explains the collapse. If no point chooses $e_j$, then $n_j = 0$, the
sum in (3) is empty, and

$$
n_j = 0 \;\Rightarrow\; \nabla_{e_j}\mathcal{L} = 0 ,
\qquad (4)
$$

so the dead entry is never updated and never will be. Collapse is an absorbing
state, not a slow drift. Fixing it needs an intervention — a re-seed or an EMA
that carries a prior count — not more training.

### The straight-through surrogate

The estimator is one line:

$$
\frac{\partial \mathcal{L}}{\partial z_e}
 := \frac{\partial \mathcal{L}}{\partial z_q} .
\qquad (5)
$$

Forward, $z_q = e$; backward, pretend $z_q = z_e$. It is biased, and the bias is
easy to state: it ignores the change in *which* code is selected. If a step
moves $z_e$ across a Voronoi boundary, the true loss jumps and the surrogate
does not see it. In practice the direction is right often enough to train, which
is why a one-line lie has survived nearly a decade of production codecs.

### The relaxed quantiser and its covariance

There is a smooth alternative. Replace the argmin with a softmax over negative
squared distances,

$$
w_k(z) = \frac{\exp(-d_k(z)/\tau s)}{\sum_j \exp(-d_j(z)/\tau s)},
\qquad d_k = \lVert z - e_k \rVert^2 ,
\qquad (6)
$$

and form $z_q = \sum_k w_k e_k$. Write $\bar{e}_w = \sum_k w_k e_k$ for that
same weighted mean. Differentiating $w_k$ with respect to $z$ gives
$\partial w_k/\partial z = (2/\tau s)\,w_k(e_k - \bar{e}_w)$, so

$$
\frac{\partial z_q}{\partial z}
 = \frac{2}{\tau s} \sum_k w_k (e_k - \bar{e}_w)(e_k - \bar{e}_w)^\top
 = \frac{2}{\tau s}\,\mathrm{Cov}_w(e),
\qquad (7)
$$

the weighted covariance of the codes under the soft assignment. The Jacobian is
symmetric and positive semidefinite, and the chain rule gives

$$
\frac{\partial \mathcal{L}}{\partial z}
 = \frac{2}{\tau s}\,\mathrm{Cov}_w(e)\,
   \frac{\partial \mathcal{L}}{\partial z_q} .
\qquad (8)
$$

Two consequences follow immediately. As $\tau \to 0$ the weights become one-hot,
$\mathrm{Cov}_w(e) \to 0$, and the gradient vanishes *exponentially* while the
prefactor $1/\tau$ grows only polynomially — the relaxed gradient is not a
low-temperature limit of anything useful. And the scale $s$ is the codebook's
own spread, so $\tau$ is dimensionless: the same $\tau$ means the same
relaxation whether the latent norm is 1 or 20.

The file computes (8) without ever forming the matrix, because only the product
with $\partial \mathcal{L}/\partial z_q$ is needed:

$$
\frac{\partial \mathcal{L}}{\partial z}
 = \frac{2}{\tau s}\Bigl[\sum_k w_k (e_k \cdot g)\, e_k
   - (\bar{e}_w \cdot g)\,\bar{e}_w\Bigr],
\qquad g = \frac{\partial \mathcal{L}}{\partial z_q} .
\qquad (9)
$$

The demo checks this against central differences to $5\times 10^{-9}$, and
builds the covariance matrix of (7) and checks it against a finite-difference
Jacobian to $4.8\times 10^{-10}$.

### Exponential moving average

The VQ-VAE-2 codebook update keeps two running sums per entry, a count and a
vector sum, and divides:

$$
N_j \leftarrow \gamma N_j + (1-\gamma) n_j, \qquad
m_j \leftarrow \gamma m_j + (1-\gamma) \sum_{i \in j} z_i, \qquad
e_j \leftarrow \frac{m_j}{N_j} .
\qquad (10)
$$

In the limit $\gamma \to 0$ this is the batch mean; at $\gamma = 0.99$ it is an
exponential average of batch means with effective window $1/(1-\gamma) = 100$
steps. Dividing by the decayed count rather than a constant is what makes it a
mean instead of a scaled sum. The paper also adds a small constant to $N_j$
before the division, so an entry with very few assignments is not divided by
nearly zero and flung across the latent space by one unlucky batch.

The point is that (10) has (3) as its fixed point, so it solves the same problem
as the optimizer without a learning rate, an Adam state, or a step size to tune.
The measured gap is 0.059 of a latent norm, the third tightest in the table.

### Residual stacks

One codebook has a ceiling: the error left over after quantisation is bounded
below by the cell radius, and 16 codes in 8 dimensions cannot tile the encoder's
output finely. Residual VQ spends a second codebook on that leftover:

$$
r_0 = z_e, \qquad
r_l = r_{l-1} - e_{l,k_l}(r_{l-1}), \qquad
z_q = \sum_{l=1}^{L} e_{l,k_l} .
\qquad (11)
$$

Each level's code is chosen to minimise the residual it sees, so the levels are
greedy and the reconstruction error is the residual left after the last one. The
demo's two levels take the error from 0.0098 to 0.0026, a factor of 3.7; more
levels keep helping until the residual is pure noise. The cost is a vocabulary
that is nominally $K^L$ — and actually much smaller, because only the pairs the
data visits are reachable.

### The ELBO reading

It is worth seeing why no KL term appears in (2). Take the posterior to be a
point mass on the chosen code and the prior to be uniform over $K$ codes. Then

$$
\log p(x) \ge \mathbb{E}_{q}[\log p(x \mid z)]
 - \mathrm{KL}(q \Vert p)
 = \mathbb{E}_{q}[\log p(x\mid z)] - \log K ,
\qquad (12)
$$

and $\log K$ is the same for every input, so it contributes no gradient. VQ-VAE
is a VAE with a degenerate posterior and a uniform prior, and the whole
interesting part of the ELBO has moved into the reconstruction term. Posteriors
cannot collapse toward the prior here, because the KL *is* the prior's entropy
and never moves; the failure mode that replaces it is collapse of the
*dictionary* — dead codes — which is what the rest of this file is about.

### Reading the table

Every model is trained for 3000 steps of batch 256 on the same 2048 points.

| variant | mse | /floor | used | ppl | to cell | q. err | params |
|---|---|---|---|---|---|---|---|
| `vq_vae` | 0.02247 | 2.29 | 12/16 | 10.41 | 0.388 | 0.300 | 9866 |
| `vq_vae2` | 0.01047 | 1.07 | 16/16 | 12.67 | 0.319 | 0.062 | 9866 |
| `ema_codebook` | 0.01180 | 1.20 | 11/16 | 9.42 | 0.059 | 0.055 | 9866 |
| `commitment_loss` | 0.00789 | 0.80 | 14/16 | 13.32 | 0.047 | 0.033 | 9866 |
| `straight_through` | 0.01022 | 1.04 | 16/16 | 14.64 | 0.038 | 0.037 | 9866 |
| `residual_vq` | 0.00263 | 0.27 | 69/256 | 60.72 | — | — | 9994 |

`/floor` is the reconstruction error in units of the data's own variance, so 1.0
is the best a model can do without denoising. `used` counts codes that win at
least 0.1% of the test set; `ppl` is the perplexity of that distribution, which
is 16.0 for a perfectly uniform codebook. `to cell` is the distance from each
code to the mean of the vectors that chose it, divided by the mean latent norm —
the direct measure of whether (3) has been reached. `q. err` is the squared
quantisation error divided by the latent energy.

Three things stand out. The paper's $\beta = 0.25$ is the *worst* setting here,
not a good default: it leaves the encoder free to inflate the latent until the
codebook cannot follow, and both `to cell` and `q. err` are an order of magnitude
worse than everything else. Coverage and convergence are not the same axis —
`vq_vae2` has all 16 codes at 0.0105, but its `to cell` is 0.319 because
re-seeding keeps teleporting entries and they never settle, while
`commitment_loss` has 14 codes at 0.0079 and a `to cell` of 0.047. "No dead
codes" is necessary but not sufficient. And `residual_vq` is a different regime,
not a better version of the others: two levels buy a 4x lower error and a 256-way
vocabulary, at which point the code distribution is no longer nearly uniform and
an empirical histogram is no longer a plausible prior.

## Code walkthrough

### The quantiser

```python
    def _quantise(self, ze: np.ndarray) -> tuple:
        """Replace each row by its nearest code; returns (quantised, picks)."""
        resid = ze
        zq = np.zeros_like(ze)
        picks = []
        for level in range(self.levels):
            d2 = ((resid[:, None, :] - self.E[level][None]) ** 2).sum(axis=2)
            k = np.argmin(d2, axis=1)
            zq = zq + self.E[level][k]
            resid = resid - self.E[level][k]
            picks.append(k)
        return zq, picks
```

`resid[:, None, :] - self.E[level][None]` is the $(n, K, D)$ table of
differences, and the
sum over the last axis turns it into $(n, K)$ distances. One level is a lookup;
more levels repeat the lookup on the leftover, which is (11) in four lines. The
same loop is written out twice more — in `reconstruct` (with a stopping level)
and in `distances` — because each needs a different output, and a shared helper
would return a tuple nobody fully uses.

### The relaxed gradient

```python
        w, _ = _soft_weights(ze, self.E[0], self.tau)
        scale = float(((self.E[0] - self.E[0].mean(axis=0)) ** 2)
                      .sum(axis=1).mean()) + 1e-9
        bar = w @ self.E[0]
        proj = w * (self.E[0][None] * dq[:, None, :]).sum(axis=2)
        return (2.0 / (self.tau * scale)) * (
            proj @ self.E[0] - proj.sum(axis=1, keepdims=True) * bar)
```

`proj` is $w_k (e_k \cdot g)$ for every $k$ and every row, so `proj @ E` is the
first sum in (9) and `proj.sum(axis=1, keepdims=True) * bar` is
$(\bar{e}_w\cdot g)\bar{e}_w$. The `1e-9` keeps the scale finite when the
codebook has not spread yet, which happens on step 0.

The scale is recomputed rather than cached because the codebook moves every
step, and the value that matters is the one at the point where the gradient is
taken. It is the same expression `_soft_weights` uses internally, so the forward
and backward passes cannot drift apart.

```python
def _soft_weights(ze: np.ndarray, E: np.ndarray, tau: float) -> tuple:
    """Softmax over negative squared distances, plus those distances.

    `tau` is relative to the codebook's own spread, so the relaxation means the
    same thing wherever the encoder happens to place the latent.
    """
    scale = float(((E - E.mean(axis=0)) ** 2).sum(axis=1).mean()) + 1e-9
    d2 = ((ze[:, None, :] - E[None]) ** 2).sum(axis=2)
    z = d2 - d2.min(axis=1, keepdims=True)
    w = np.exp(-z / (tau * scale))
    return w / w.sum(axis=1, keepdims=True), d2
```

Subtracting the row minimum before the exponential guarantees that the largest
weight is $e^0 = 1$, so the denominator can never be zero and the division is
always defined. It leaves the ratio unchanged and makes the result independent
of where the encoder puts the origin. It does *not* protect the other weights:
when the spread of distances dwarfs $\tau s$ they underflow to zero and the
softmax becomes exactly one-hot, which is the failure described further down.

### The training step

```python
        for step in range(self.train_steps):
            xb = x[rng.integers(0, len(x), self.batch)]
            n = len(xb)
            ze, cache_e = _mlp_forward(self.enc, xb)
            zq, picks = self._quantise(ze)
            xh, cache_d = _mlp_forward(self.dec, zq)

            recon = ((xb - xh) ** 2).sum(axis=1).mean() / (2.0 * SPREAD ** 2)
            dq, gd = _mlp_backward(self.dec, cache_d,
                                   (xh - xb) / (n * SPREAD ** 2))
            g_ze = (self.soft_gradient(ze, dq) if self.grad == "soft"
                    else dq.copy())
            g_E = [np.zeros_like(e) for e in self.E]
            resid, gap = ze, 0.0
            for level in range(self.levels):
                k = picks[level]
                eq = self.E[level][k]
                gap = gap + float(((resid - eq) ** 2).sum(axis=1).mean())
                if not self.ema:
                    np.add.at(g_E[level], k, 2.0 * (eq - resid) / n)
                g_ze = g_ze + 2.0 * self.beta * (resid - eq) / n
                resid = resid - eq
```

Read the three gradient streams separately. `dq` is
$\partial \mathcal{L}/\partial z_q$ from the decoder; under straight-through it
*is* the encoder's gradient, which is `dq.copy()`. `g_E` is the codebook's
gradient, scattered with `np.add.at` because several rows can choose the same
code and a plain fancy-index assignment would keep only the last. `g_ze` then
gathers one commitment gradient per level.

That last part deserves a note for the residual model. The decoder sees
$z_q = \sum_l e_{l,k_l}$ and hands back one gradient with respect to the sum.
Since $\partial z_q/\partial r_l = 1$ for every level's residual, a single
`g_ze = dq` is already correct at every level; the loop adds only the commitment
gradients, one per level. For one level the loop does one iteration and the code
is the textbook version.

`np.add.at` is the slow, correct tool. `g_E[level][k] += 2*(eq - resid)` would
silently drop all but the last contribution of each duplicate index, and the
error is invisible in the loss — the codebook just learns a biased mean.

### The EMA update

```python
    def _ema_step(self, ze: np.ndarray, picks: list, counts: list) -> None:
        """Smoothed exponential moving average, as in VQ-VAE-2."""
        resid = ze
        for level in range(self.levels):
            k = picks[level]
            for j in range(self.codes):
                m = k == j
                nj = float(m.sum())
                if nj == 0.0:
                    continue
                nk = self.decay * counts[level][j] + (1.0 - self.decay) * nj
                tot = (self.decay * counts[level][j] * self.E[level][j]
                       + (1.0 - self.decay) * ze[m].sum(axis=0))
                self.E[level][j] = tot / nk
                counts[level][j] = nk
            resid = resid - self.E[level][k]
```

This is (10) verbatim, with `counts` playing $N_j$ and `tot` playing $m_j$. The
`continue` on `nj == 0` is not an optimisation: with $n_j = 0$ the numerator is
$\gamma m_j$ and the denominator $\gamma N_j$, so the entry would be unchanged
anyway, but the early exit makes the equality explicit.

A vectorised version would group by `k` with `np.bincount` and a matrix product.
It is faster and it is easier to get subtly wrong, because the decayed count has
to be updated for the dead codes too (by the decay alone) and a sparse grouping
naturally skips them.

### Re-seeding

```python
    def _restart(self, ze: np.ndarray, picks: list, idle: list,
                 rng: np.random.Generator) -> None:
        """Re-seed a code after it has gone a while without an assignment."""
        for level in range(self.levels):
            hit = np.zeros(self.codes, dtype=bool)
            hit[picks[level]] = True
            idle[level] = np.where(hit, 0.0, idle[level] + 1.0)
            for j in np.where(idle[level] > IDLE)[0]:
                self.E[level][j] = ze[rng.integers(0, len(ze))]
                idle[level][j] = 0.0
                self.reseeds += 1
```

The patience counter is what makes this safe. Re-seeding every dead code every
step would churn the dictionary, because on any single batch of 256 points some
of 16 codes legitimately win nothing. Requiring 25 consecutive empty batches
means a code has to be unwanted by *every* draw before it moves. In the demo
this fires 48 times over 3000 steps — sixteen codes, three resurrections each on
average — and ends with complete coverage.

The new location is a single encoder output, not a mean of anything. That is the
point: a dead code should land where the data actually is, and one sample is
enough to put it inside a cell that will then pull it toward that cell's mean.

### Reconstruction and sampling

```python
        ze, _ = _mlp_forward(self.enc, x)
        resid = ze
        zq = np.zeros_like(ze)
        for level in range(upto):
            d2 = ((resid[:, None, :] - self.E[level][None]) ** 2).sum(axis=2)
            k = np.argmin(d2, axis=1)
            zq = zq + self.E[level][k]
            resid = resid - self.E[level][k]
        return _mlp_forward(self.dec, zq)[0]
```

The `upto` argument is what lets the demo report "level 1 alone scores 0.0098,
two levels score 0.0026" from one trained model, which is the cleanest statement
of what a residual level is worth: the same encoder, the same decoder, one fewer
codebook.

```python
        rng = np.random.default_rng(seed)
        joint = rng.choice(len(self.hist), size=n, p=self.hist)
        zq = np.zeros((n, self.latent))
        for level in reversed(range(self.levels)):
            zq = zq + self.E[level][joint % self.codes]
            joint = joint // self.codes
        return _mlp_forward(self.dec, zq)[0]
```

Sampling needs a prior over codes, and this one is the empirical histogram of the
training set — a lookup table, not a model. `_joint` packs the levels as digits,
$k_1 K^{L-1} + \cdots + k_L$, so decoding walks the levels from the last to the
first with `% K` and `// K`. `reversed` is doing real work: the levels are stored
most-significant first, and iterating forward would assign each code to the wrong
slot and still produce eight plausible blobs, which is exactly the kind of bug
that survives a visual check.

### The k-means reference

```python
def kmeans(x: np.ndarray, k: int, *, iters: int = 100) -> tuple:
    """Lloyd's algorithm; returns (distortion, centres)."""
    c = x[np.linspace(0, len(x) - 1, k).astype(int)].copy()
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
        new = np.stack([x[lab == j].mean(axis=0) if np.any(lab == j) else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
    return float(((x - c[lab]) ** 2).sum(axis=1).mean()), c
```

This exists to be a baseline, not a component. Running Lloyd's algorithm on the
*trained encoder's* outputs gives the best distortion any codebook of this size
can achieve for that point cloud, so dividing the learned distortion by it says
how far the joint optimisation has drifted from the alternating solution. The
demo prints the ratio for every variant: 1.14 for `straight_through`, 1.33 for
`commitment_loss`, 3.95 for `vq_vae`. That is the `to cell` story measured a
different way. Where the encoder has settled, the codebook is within 15% of the
best tiling of the representation it is given; where the encoder is still
moving, no codebook can be optimal for a target that runs away, and the ratio
stays near 4. The remaining error lives in the representation, not the tiling.

## Complexity

Per point, the encoder and decoder cost $O(dH + H^2)$ each, with $d = 2$ the data
dimension, $D = 8$ the latent and $H = 64$ the width. The quantiser costs
$O(KDL)$ for $L$ levels, which is 128 multiply-adds at these sizes and 33k at
image scale ($K = 512$, $D = 64$, $L = 1$). One training step is
$O(B(KDL + dH + H^2))$ with $B = 256$.

The demo trains eleven models — six variants and five commitment settings — for
3000 steps each, so about 8.4 million point-steps through the encoder and
decoder. At the codebook sizes used here the lookup is under 3% of the
arithmetic; the networks dominate, which is the opposite of the situation at
image resolution.

Memory is the codebook's $O(KDL)$ floats plus the two networks: 9866 parameters
in total, of which the codebook is 128. Sampling stores the histogram, which is
dense and therefore
$O(K^L)$: fine at $16^2 = 256$, impossible at $512^4 = 6.9\times 10^{10}$. That
dense histogram is the atom's one genuinely non-scalable choice, and every
production system replaces it with an autoregressive or masked prior.

Comparison against the Gaussian VAE is a wash on arithmetic and a real win on
latent size: 4 bits instead of 256, which is 64x less to model with a second
network. That ratio is the reason discrete latents won.

## Failure modes

**Codebook collapse.** The headline failure, and the one the demo exists to show.
`vq_vae` leaves 4 of 16 codes unassigned; by (4) those codes have exactly zero
gradient and cannot recover. Re-seeding recovers all 16 in 48 interventions.
EMA recovers only 11, and the reason is structural: with $n_j = 0$ the update
in (10) is a no-op, so a dead entry stays exactly where it is and stays
unwanted forever. EMA buys convergence for the entries that live, not coverage
for the ones that do not. Neither fix is free: re-seeding takes coverage from 12
to 16 but the error only falls from 2.29 to 1.07 floors, and
`commitment_loss` beats both with 14 codes. Coverage is a necessary condition
for a good codebook, not a sufficient one.

**A commitment weight tuned for images.** $\beta = 0.25$ is the original paper's
value, and on this data it is the worst of the five settings tried. The sweep is
unambiguous: $\beta = 0$ lets the latent norm reach $9.5\times10^4$ and the error
sit at 10 floors, because the encoder is free to expand and the codebook can
never catch up; $\beta = 0.25$ still leaves the quantisation error at 30% of the
latent energy; $\beta = 4$ reaches 0.80 floors with a latent norm of 2.65. The
paper's $\beta$ was chosen for 256×256 images with 64-dimensional latents and a
much larger decoder, where the encoder cannot easily inflate the latent; this
data is two-dimensional with an eight-dimensional latent, and the same constant
does not transfer. The number is a prior, not a law.

**Pushing $\beta$ too far.** At $\beta = 64$ the error is 0.0072, marginally the
best, but the latent norm has fallen to 1.11 — close to the data's own radius of
0.8. The encoder has stopped expanding, so 16 codes must cover a ball about the
size of the data cloud and the decoder is doing the work of separating the modes
from a 4-bit index. The model is close to a plain 16-way vector quantiser with
the representation learning removed. The limit is instructive: as
$\beta \to \infty$ the encoder maps every input to one point, all codes chase
that point, and the codebook collapses to a single entry.

**An absolute temperature.** The first version of the soft quantiser used
$\tau = 0.05$ as a raw squared distance. With the encoder placing the latent at
norm 8, squared distances ran to a hundred or so, and after subtracting the row
minimum every other weight underflowed to zero: the softmax was *exactly*
one-hot, so $\mathrm{Cov}_w = 0$ and by (7) the encoder gradient was exactly
zero. Training ran, the loss fell, and the encoder never moved — 0.40 error, 2 of
8 modes covered, worse than doing nothing. Normalising $\tau$ by the codebook's
own spread fixed it in one line and made the parameter dimensionless.

**A temperature that trains is not a temperature that helps.** Sweeping the
normalised $\tau$ from 0.02 to 3 gives final errors between 0.0077 and 0.0388
with no monotone trend; at one seed each most of that spread is noise, and Adam
divides out the gradient's *scale*, so the only fatal setting is a gradient that
is exactly zero. What $\tau$ changes is the direction — large $\tau$ mixes
neighbouring codes into $z_q$, small $\tau$ approaches the hard lookup. The
measured cosine at $\tau = 1$ is $+0.53$: the relaxed and straight-through
surrogates agree on the average direction and disagree coordinate by coordinate.
Neither is the true derivative, and the loss curve cannot tell you which one to
use — only samples can.

**A vocabulary that is nominally larger than it is.** `residual_vq` reports
69 of 256 pairs reachable. Any model trained on the nominal vocabulary would
spend $1 - 69/256 = 73\%$ of its probability on pairs the encoder never produces,
and sampling from it would decode codes the decoder never saw. In production this
is handled by learning the prior on the *observed* codes — the entropy coding in
SoundStream is built on exactly this — but it is worth knowing that "$K^L$ codes"
is an upper bound on capacity, not a description of it.

**Reconstruction error below the data floor is not a bug.** `commitment_loss`
scores 0.80 floors and the reconstruction's mean distance to a mode centre
(0.045) is *below* the data's own (0.088). The decoder outputs conditional means,
so it removes the within-blob noise as well as the quantisation error. This means
that a low MSE can mean either "good model" or "outputs the mode centres", and
the two are indistinguishable by MSE alone. The floor of exactly 1.0 for a
centre-emitting decoder is why the demo prints the ratio rather than the raw
number: 1.0 is the target, not 0.

**Confusing quantisation error with reconstruction error.** They are different
quantities and they move in opposite directions as $\beta$ grows. `q. err` is how
far the latent sits from its code, 0.033 for `commitment_loss`; the mse is how
far the output sits from the data, 0.0079. A model with a large $\beta$ can have
a tiny quantisation error and a poor decoder, and one with a small $\beta$ can
have the reverse. The demo keeps the two columns apart because the most common
debugging mistake in this family is to watch the codebook loss fall and conclude
that the model is learning.

## Applications

- **Neural audio codecs.** SoundStream, EnCodec and DAC all use residual vector
  quantisation over a learned encoder, with the codebooks trained exactly as
  here and entropy-coded for transmission. A 3 kbps speech codec is a VQ-VAE with
  a stack of codebooks and a language model over the codes.

- **Image tokenisation for transformers.** VQ-VAE-2, VQGAN and ViT-VQGAN turn an
  image into a grid of code indices, which a transformer can then model with a
  categorical cross-entropy — the same loss it uses for text. DALL·E's dVAE,
  MaskGIT, Muse and Parti all sit downstream of this idea.

- **Audio and video generation.** Jukebox generates music in a VQ-VAE's discrete
  code space; video models (CogVideo, Phenaki) tokenise frames the same way.

- **Self-supervised speech representations.** wav2vec 2.0 and HuBERT quantise
  continuous features into a codebook to form targets for masked prediction. The
  quantiser is the same nearest-neighbour lookup; the training objective is
  different.

- **Compression with a learned prior.** A VQ-VAE plus an autoregressive model over
  codes is a lossy compressor whose rate is the code entropy, and whose decoder is
  a neural network. This is the architecture of every neural codec shipped since
  2021.

- **Anomaly detection and retrieval.** Because the codebook is a finite
  vocabulary, the reconstruction error of an input that maps to a distant code is
  a natural outlier score, and the code indices themselves are a hash for
  nearest-neighbour search.

## References

- van den Oord, Vinyals, Kavukcuoglu (2017). *Neural Discrete Representation
  Learning.* The VQ-VAE paper: the three-term loss, the straight-through
  estimator, and the claim that discrete latents avoid posterior collapse.
- Razavi, van den Oord, Vinyals (2019). *Generating Diverse High-Fidelity Images
  with VQ-VAE-2.* The hierarchical model, the EMA codebook, and the re-seeding
  of unused codes.
- Bengio, Léonard, Courville (2013). *Estimating or Propagating Gradients Through
  Stochastic Neurons for Conditional Computation.* The straight-through
  estimator, stated and analysed before it was fashionable.
- Jang, Gu, Poole (2017). *Categorical Reparameterization with Gumbel-Softmax.*
  The other relaxation of a discrete choice; compare its gradient with (8).
- Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi (2021). *SoundStream: An
  End-to-End Neural Audio Codec.* Residual VQ in production, with the bitrate
  ladder.
- Défossez, Copet, Synnaeve, Adi (2022). *High Fidelity Neural Audio
  Compression.* EnCodec: residual VQ, entropy coding, and the numbers that made
  it the default.
- Esser, Rombach, Ommer (2021). *Taming Transformers for High-Resolution Image
  Synthesis.* VQGAN: a perceptual and adversarial reconstruction loss on top of
  the VQ-VAE, which is what makes the codebook usable at image resolution.
- Chang, Yan, Liu, Li, Hu, Metzler (2022). *MaskGIT: Masked Generative Image
  Transformer.* Parallel decoding over VQ tokens, and the reason the token
  ordering matters less than it looks.
- Baevski, Zhou, Mohamed, Auli (2020). *wav2vec 2.0.* Quantised targets for
  self-supervised speech, with a codebook trained by a contrastive objective
  rather than a reconstruction one.
