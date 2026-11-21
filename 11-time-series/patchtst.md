# PatchTST — patch the window, attend within it, share across channels

`patchtst.py` · Time series · Tier 3 · 4 variants

## Intuition

A transformer on a long univariate series has a problem: the sequence of
timesteps is short and each token is a single scalar. Attention over a thousand
barely-informative scalars is expensive, and the self-attention pattern has
little to latch onto. PatchTST changes the unit of attention from the timestep
to the **patch**: cut the look-back window into overlapping chunks of twelve
observations, embed each chunk, and attend over a handful of tokens instead of
hundreds.

Two consequences follow. First, a patch carries local shape — level, slope,
curvature — so the token is informative before any attention happens. Second,
the sequence length collapses from $L$ to roughly $L/P$, so the cost of
attention falls by a factor of $P^2$ and the model can afford a longer
look-back.

The second idea is **channel independence**. Given a multivariate series, do not
mix the channels: treat each one as its own univariate problem and share the
weights. The forecast for one channel never depends on the others, which is a
strong prior — it says the relationship between channels is not something the
model should learn. It sounds like a limitation, and it is, but it is also the
regularisation that makes PatchTST hard to beat on standard benchmarks: a
channel-mixing model can overfit spurious cross-correlations, while the
independent one cannot.

The variants cover the training protocol:

- **`patchtst`** is the supervised model on one channel.
- **`channel_independent`** applies the identical model to every channel with
  one set of weights, so permuting the channels permutes the forecast.
- **`supervised`** masks patches, reconstructs them, then fine-tunes every
  weight on the forecasting loss.
- **`self_supervised`** masks patches, then freezes the representation and fits
  only a linear head — the linear-probe test of what the pretraining learned.

The demo trains on 348 points and holds out 12. The supervised model reaches a
held-out MAE of $0.2956$; fine-tuning after masked pretraining improves it to
$0.2691$, while a frozen linear probe reaches only $0.4015$.

## Variants

| variant | representation | trained | held-out MAE |
|:--|:--|:--|:--|
| `patchtst` | one channel | from scratch | 0.2956 |
| `channel_independent` | per channel, shared | from scratch | 0.9256 joint |
| `supervised` | masked pretraining | all weights | 0.2691 |
| `self_supervised` | masked pretraining | head only | 0.4015 |

**`patchtst`** builds 289 windows from the 348-point training span, cuts each
48-step window into four non-overlapping patches of twelve, and embeds them
into 16 dimensions. The in-sample MSE is $0.0561$ in per-window variance units,
and the held-out MAE of $0.2956$ beats both naive ($1.5527$) and seasonal naive
($0.4627$).

**`channel_independent`** is fitted to two channels, the second an affine
transform of the first plus small noise. The joint MAE of $0.9256$ is dominated
by the second channel, which has three times the scale. Swapping the columns of
the input swaps the columns of the forecast with a maximum difference of
exactly zero, because the channels never meet inside the network.

**`supervised`** pretrains with 30% of the patches hidden and a reconstruction
MSE of $0.0713$, then fine-tunes every weight on the forecast. Fine-tuning moves
the encoder by $1.3013$ in maximum absolute value and lowers the held-out MAE
from $0.2956$ to $0.2691$.

**`self_supervised`** uses the same pretrained weights but freezes the encoder
and the patch embedding, training only the $12\times64$ linear head. The frozen
representation changes by exactly zero, and the probe reaches $0.4015$. On this
small series the representation alone is weaker than supervised training — the
pretraining has only 289 windows to learn from, and the probe cannot fix what
the encoder never learned.

## Derivation

### Patching

A window $x\in\mathbb R^{L}$ is cut into $N$ patches of length $P$ with stride
$S$:

$$
N=\Bigl\lfloor\frac{L-P}{S}\Bigr\rfloor+1,\qquad
p_i=x_{(i-1)S+1:(i-1)S+P},\qquad i=1,\dots,N .
\qquad (1)
$$

With $L=48$, $P=12$ and $S=12$ this gives $N=4$ non-overlapping patches; with
$S=6$ it gives $N=7$ overlapping ones. Non-overlapping patches are cheaper and,
on the demo series, better: each patch then covers exactly one seasonal period.

### Patch embedding

Each patch is projected to a $d$-dimensional token and a learned positional
vector is added:

$$
h_i=W_p p_i+b_p+e_i,\qquad W_p\in\mathbb R^{d\times P},\quad
e_i\in\mathbb R^{d}.
\qquad (2)
$$

The projection is linear, so the model can learn a filter bank — a patch of
twelve points becomes twelve learned projections of that shape. No pooling is
involved, which is why P is small; a large P would make $W_p$ a wide matrix
with little structure.

### Self-attention

With $H_a$ heads and $d_h=d/H_a$, the layer computes

$$
Q=hW_q^{\top}+b_q,\quad K=hW_k^{\top},\quad V=hW_v^{\top}+b_v,
\qquad (3)
$$

and, per head,

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_h}}\Bigr)V .
\qquad (4)
$$

The key bias is omitted deliberately: adding a constant to a row of $K$ adds a
constant to every score in that row, and softmax is invariant to it. Including
it would create a parameter with an identically zero gradient.

The heads are concatenated and projected:

$$
\mathrm{MHA}(h)=\mathrm{Concat}\bigl(\mathrm{head}_1,\dots,
\mathrm{head}_{H_a}\bigr)W_o^{\top}+b_o .
\qquad (5)
$$

### The encoder block

The implementation uses post-norm residual blocks with a ReLU feed-forward
network:

$$
u=h+\mathrm{MHA}(\mathrm{LN}(h)),
\qquad (6)
$$

$$
\mathrm{Block}(h)=u+\mathrm{ReLU}(\mathrm{LN}(u)W_1^{\top}+b_1)W_2^{\top}+b_2 .
\qquad (7)
$$

Layer normalisation is affine-free:

$$
\mathrm{LN}(z)=\frac{z-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
\mu=\frac{1}{d}\sum_j z_j,\quad
\sigma^2=\frac{1}{d}\sum_j (z_j-\mu)^2 .
\qquad (8)
$$

Dropping the learnable gain and bias removes $2d$ parameters per layer and,
more importantly, removes two more gradient terms from an already long backward
pass. On a model this small the gain is mostly in code clarity.

### The forecast head

The encoder output is flattened across patches and mapped to the horizon:

$$
\hat y=W_h\,\mathrm{vec}(h^{(N)})+b_h,\qquad
W_h\in\mathbb R^{H\times Nd}.
\qquad (9)
$$

Flattening rather than pooling keeps the position of each patch. A pooled
representation would force the head to recover where in time each feature came
from, which is exactly the information the head needs to place a forecast.

### The supervised objective

Training minimises the mean squared error over windows:

$$
L_{\mathrm{sup}}=\frac{1}{WH}\sum_{w=1}^{W}\lVert \hat y_w-y_w\rVert_2^2 .
\qquad (10)
$$

Every parameter — patch projection, position, encoder, head — is updated by
Adam on this loss.

### Channel independence

For a $C$-channel series, the channels are stacked into the batch:

$$
\{x^{(1)},\dots,x^{(C)}\}\longrightarrow
\{(x^{(c)},y^{(c)})\}_{c=1}^{C},
\qquad (11)
$$

and the same weights are applied to each. The forecast is therefore equivariant
to permutations of the channels: if $\pi$ reorders the channels then
$\hat y(\pi x)=\pi\hat y(x)$. The demo checks this numerically and gets a
difference of exactly zero, which is the strongest possible version of the
claim.

### Masked pretraining

A mask $m\in\{0,1\}^{N}$ hides a fraction $\rho$ of the patches. The model sees
$x\odot(1-m)$ and must reconstruct the hidden patches:

$$
L_{\mathrm{mask}}=
\frac{1}{|\mathcal M|P}\sum_{(i,j)\in\mathcal M}
\bigl(\hat p_{ij}-p_{ij}\bigr)^2,
\qquad (12)
$$

where $\mathcal M$ is the set of masked (patch, position) pairs. The mask is
redrawn every epoch and every row is forced to hide at least one patch, so no
window is left untouched.

### Linear probing

After pretraining, two protocols are possible. The `supervised` variant keeps
the pretrained weights as an initialisation and fine-tunes everything on
(10). The `self_supervised` variant freezes the encoder and the patch
embedding and fits only $W_h,b_h$:

$$
W_h^{\star}=\arg\min_{W_h}\sum_{w}
\lVert W_h\,\mathrm{vec}(h_w)+b_h-y_w\rVert_2^2 .
\qquad (13)
$$

Because the representation is fixed, (13) is a linear least-squares problem, and
the probe measures the information the pretraining actually put into $h$. The
gap between the probe and fine-tuning is the value of the pretrained
initialisation beyond its linear content.

## Code walkthrough

Patching is an index gather, which is both faster and clearer than a loop:

```python
def _patches(windows: np.ndarray, patch: int, stride: int) -> np.ndarray:
    count = windows.shape[0]
    number = _patch_count(windows.shape[1], patch, stride)
    starts = np.arange(number) * stride
    index = starts[:, None] + np.arange(patch)[None, :]
    return windows[:, index].reshape(count, number, patch)
```

The gather produces `(count, number, patch)` directly, so no later reshape is
needed. The same helper serves training windows and the single prediction
window, which guarantees that the two views of a patch are identical.

Layer normalisation is a free function rather than a layer, because it has no
parameters. The cache keeps the centred input and the inverse standard
deviation, which is everything the backward pass needs:

```python
def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred ** 2).mean(axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt(variance + 1e-5)
    return centred * inverse, (centred, inverse)
```

The backward pass is the standard expression for a normalisation with no
affine term. It has three pieces — the gradient itself, its mean, and its
projection onto the centred input — because the normalisation couples every
coordinate of a token:

```python
def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)
```

Forgetting the `total` term is the usual bug: it still trains, but the gradient
is wrong and the model silently learns a slightly different objective.

The forward pass reads in the order of (3)–(7). Keeping the split heads in the
cache lets the backward pass avoid re-splitting:

```python
        normalised, norm_cache = _layer_norm(x)
        query = normalised @ p["Wq"] + p["bq"]
        key = normalised @ p["Wk"]
        value = normalised @ p["Wv"] + p["bv"]
        qh = _split_heads(query, self.heads)
        kh = _split_heads(key, self.heads)
        vh = _split_heads(value, self.heads)
        scores = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(self.width)
        attention = _softmax(scores)
        mixed = attention @ vh
        joined = _merge_heads(mixed, self.dim)
        attended = joined @ p["Wo"] + p["bo"]
        first = x + attended
```

The attention backward is where most mistakes happen. The softmax Jacobian is
not the elementwise product with `1 - a`; it is
$\partial a_i/\partial s_j=a_i(\delta_{ij}-a_j)$, which collapses to
$a\odot(g-\langle g,a\rangle)$ when contracted with an upstream gradient:

```python
        attention_grad = mixed_grad @ vh.transpose(0, 1, 3, 2)
        value_grad = attention.transpose(0, 1, 3, 2) @ mixed_grad
        centre = (attention_grad * attention).sum(axis=-1, keepdims=True)
        scores_grad = attention * (attention_grad - centre) / np.sqrt(self.width)
        query_grad = scores_grad @ kh
        key_grad = scores_grad.transpose(0, 1, 3, 2) @ qh
```

The numeric gradient check in development matched all eleven parameter arrays
to about $10^{-8}$, which is what the finite-difference tolerance allows.

Masked pretraining hides patches by zeroing their embedding input. A zero
patch is not a special token — the mask is communicated by the position of the
zero, and the reconstruction head learns to invert it:

```python
            mask = self._rng.random((batch, count)) < self.mask_ratio
            empty = ~mask.any(axis=1)
            if empty.any():
                columns = self._rng.integers(0, count, size=int(empty.sum()))
                mask[empty, columns] = True
            visible = np.where(mask[:, :, None], 0.0, patches)
```

Forcing at least one masked patch per row matters when $N$ is small: with four
patches and $\rho=0.3$, the probability that a row hides nothing is
$0.7^4\approx0.24$, so without the guard a quarter of the batch would contribute
no gradient.

The two training protocols differ by one boolean. When it is off, the
optimiser is given only the head, so the encoder is not merely frozen by a
small learning rate — it is absent from the update entirely:

```python
        parameters = {**self.head_parameters}
        if train_encoder:
            parameters = {**parameters, **self.embed_parameters,
                          **self.encoder.parameters}
        combined = _Adam(parameters, self.learning_rate)
```

That is why the demo can assert an exact zero change in the frozen
representation: no code path touches those arrays.

Channel independence is implemented by folding the channel axis into the batch
before patching and unfolding it after the head:

```python
        windows = left.transpose(0, 2, 1).reshape(-1, self.seq_len)
        targets = right.transpose(0, 2, 1).reshape(-1, self.horizon)
```

`transpose(0, 2, 1)` turns `(window, time, channel)` into
`(window, channel, time)`, so the reshape interleaves channels into the batch
rather than mixing them into the patch. Every channel then becomes an
independent row of the same matrix product.

## Complexity

With $W$ windows, $L$ look-back, $N$ patches, patch size $P$, width $d$, $H_a$
heads, feed-forward width $f$ and $E$ epochs:

- **Patching:** $O(WNP)$ memory for the gathered array.
- **Embedding:** $O(WNdP)$ for the projection and $O(WNd)$ for the position.
- **Attention scores:** $O(WN^2d)$ time and $O(WH_aN^2)$ memory for the
  attention maps; this is the term that patching shrinks, since $N\approx L/P$.
- **Feed-forward:** $O(WNdf)$.
- **One epoch:** $O(WN(dP+Nd+df))$, dominated by $dP$ and $df$.
- **Full supervised fit:** the above times $E$.
- **Pretraining:** the same per epoch over `pretrain_epochs`, plus the
  reconstruction head.
- **Prediction:** one window, so $O(NdP+N^2d+Ndf)$ and independent of the
  training size.

With $L=48$, $N=4$, $d=16$ and 289 windows, an epoch is a few hundred
microseconds; the demo's four fits, including pretraining, take about
$4.5$ seconds. The patch count is the lever: stride six doubles $N$ and roughly
quadruples the attention cost for a small accuracy change.

## Failure modes

**A patch shorter than the season is blind to it.** With $P=8$ and a period of
twelve, no patch contains a full cycle, so the first token cannot represent the
seasonal shape and the model must reconstruct it from attention. Patches of
twelve on a period of twelve work well; patches of four do not. Match the patch
length to the shortest structure you expect.

**Channel independence is an assumption, not a fact.** If two channels are
genuinely coupled — a price and its spread, a load and its temperature — the
independent model cannot exploit the coupling. It will still fit, but it leaves
accuracy on the table, and the gap is invisible unless you compare against a
channel-mixing model. Use CI as the default and as a strong baseline, not as a
universal truth.

**Pretrained features can be worse than supervised ones.** The linear probe in
the demo reaches $0.4015$ against $0.2691$ for fine-tuning and $0.2956$ from
scratch. Pretraining on 289 windows does not produce a representation that is
linearly decodable; the encoder has learned features that only become useful
when the head above them can move too. Self-supervision needs scale to pay off.

**The flatten head is large.** $W_h$ has $H\times Nd$ entries, which for a long
look-back and many patches is the biggest matrix in the model. It also has no
structural prior: every output step mixes every patch with its own weights.
Pooling or an attention-based decoder would shrink it, at the cost of losing
positional detail.

**Post-norm blocks need a learning-rate that respects them.** The residual path
passes through a normalisation at every block, so gradients are rescaled on the
way back. With 400 epochs and $10^{-2}$ the demo trains; at $3\times10^{-3}$ it
underfits badly ($0.79$ in-sample MSE after 120 epochs). Pre-norm would be more
forgiving, at the cost of a different residual topology.

**Masking is a weak pretext task on short series.** With four tokens and a 30%
mask, each epoch hides about one patch per row. The reconstruction loss falls
quickly to $0.07$, but the representation it produces is not obviously better
than random features. Masked pretraining is worth it when the unlabelled
corpus is large; on a few hundred windows it is mostly a regulariser.

**One season of history is not enough for a transformer.** Attention can compare
patches, but with four patches there are only six pairs, and most of the
model's capacity goes into the head. The architecture shines on windows of
hundreds of steps; on a 48-step window a linear model with a seasonal term is a
fair fight.

## Applications

PatchTST is used for medium- and long-horizon forecasting where the look-back is
long enough to contain many cycles: electricity load and price, traffic,
weather, retail demand at the aggregate level, and financial volatility. The
channel-independent design makes it a natural fit for datasets with hundreds of
related series — the weights are shared, so adding a series costs nothing but
data.

The self-supervised protocol is used for transfer: pretrain on a large,
unlabelled corpus of series from the same domain, then fine-tune or probe on the
small target series. The paper reports that this beats training from scratch
when the target data are scarce — the opposite of what the demo shows, because
the demo's pretraining corpus is the target itself.

It is the wrong tool for very short series, for intermittent demand, and for
problems where cross-channel structure is the signal. It is also not
probabilistic: the head emits a point forecast. For intervals, add a quantile
head or use a model with a likelihood.

## References

- Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. (2023). *A time series is
  worth one word: Patching time series with transformers.* ICLR. Patching,
  channel independence, and the masked pretraining protocol.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., Polosukhin, I. (2017). *Attention is all you need.* NeurIPS. The
  encoder block in (3)–(7).
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization.*
  arXiv:1607.06450. Equation (8) and its backward pass.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training
  of deep bidirectional transformers for language understanding.* NAACL. The
  masked-reconstruction pretext task adapted in (12).
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). *Are transformers effective for
  time series forecasting?* AAAI. The linear-baseline argument that motivates
  keeping a simple head.
