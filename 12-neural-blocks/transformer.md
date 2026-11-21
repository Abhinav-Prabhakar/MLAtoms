# Transformer blocks — attention, norms, and the order they run in

`transformer.py` · Neural blocks · Tier 1 · 8 variants

## Intuition

A transformer block is two mixing operations with a residual around each.
Attention mixes **across positions**: every token reads every other token and
takes a weighted average of their values. The feed-forward network mixes
**across channels**: each token independently passes through a wide hidden
layer. Neither changes the shape of its input, so both can be added back.

The variants are three decisions taken differently:

1. **Where the normalisation goes.** After each addition (`post_ln`) or before
   the branch (`pre_ln`, `encoder`, `rmsnorm_block`); `rmsnorm_block` also
   changes which normaliser is used.
2. **Which positions attention may see.** All of them, or only the past
   (`pre_ln` and `decoder` carry the causal mask).
3. **What the feed-forward network computes.** A ReLU pair, or a gated SiLU
   product (`swiglu_ffn`).

A fourth axis is how many attention operations a block has: one (`encoder`,
`pre_ln`, `post_ln`, `swiglu_ffn`, `rmsnorm_block`, `parallel_attention`), one
that reads an external memory instead of itself (`cross_attention`), or two —
self then cross — in `decoder`. A fifth is whether the two branches run in
sequence or in parallel (`parallel_attention`).

The demo's task separates these cleanly. Each sequence is eight tokens with two
markers; the label is the sign of the product of their values. A block that can
look both ways sees both markers and learns the product. A causal block cannot,
when the second marker arrives after the first, and sits at chance. Trained for
60 epochs at width 8, two heads and depth 2, the six bidirectional blocks score
between 0.944 and 1.000 and the two causal blocks score 0.506 and 0.469.

## Variants

| variant | layout | attention | parameters |
|:--|:--|:--|--:|
| `post_ln` | add, norm, ffn, add, norm | both ways | 1154 |
| `pre_ln` | norm, add, norm, ffn, add | causal self | 1154 |
| `encoder` | norm, add, norm, ffn, add | both ways | 1154 |
| `decoder` | norm, add, norm, cross, add, norm, ffn, add | causal self + cross | 1730 |
| `cross_attention` | norm, cross, add, norm, ffn, add | cross only | 1154 |
| `swiglu_ffn` | norm, add, norm, gated ffn, add | both ways | 1442 |
| `rmsnorm_block` | norm, add, norm, ffn, add | both ways | 1154 |
| `parallel_attention` | norm, then attn and ffn in parallel | both ways | 1154 |

The counts are for the demo: dimension 8, two heads, hidden width 16, two
blocks, two classes. A block with one attention costs 568 parameters —
$288$ for the four projections and $280$ for the ReLU pair — so two blocks plus
an 18-parameter head give 1154.

**`post_ln`** is the original 2017 layout. Each addition is followed by a layer
norm; attention reads the raw stream and the feed-forward reads the normalised
post-attention stream:

$$
a=\mathrm{LN}\bigl(x+\mathrm{Attn}(x)\bigr),
\qquad
y=\mathrm{LN}\bigl(a+\mathrm{FFN}(a)\bigr).
\qquad (1)
$$

**`pre_ln`** is the GPT layout: normalise, attend causally, add; normalise, run
the feed-forward, add. The residual stream itself is never normalised, only the
copies that branches read.

**`encoder`** is the same layout without the mask, which is the BERT block.

**`decoder`** adds a second attention: causal self-attention over the prefix,
then causal cross-attention into an encoder memory, then the feed-forward. The
demo passes the sequence itself as the memory, so the cross-attention becomes a
second, independently parameterised full attention and the block reaches 1.000
even though its self-attention is masked.

**`cross_attention`** is the decoder's middle section in isolation: no
self-attention at all, only a query from its input against a supplied memory.

**`swiglu_ffn`** keeps the pre-LN layout and replaces the ReLU pair with

$$
h=\mathrm{silu}(xW_1+b_1)\odot(xW_g+b_g),
\qquad
\mathrm{FFN}(x)=hW_2+b_2 .
\qquad (2)
$$

**`rmsnorm_block`** keeps the layout and swaps layer norm for RMSNorm, which
divides by the root mean square and does not subtract the mean.

**`parallel_attention`** computes attention and the feed-forward from the *same*
normalised input and adds both to the stream in one step. There is no sequential
dependency between the two halves, so their matrix products can be issued
together.

## Derivation

### Scaled dot-product attention

For a query $q$, keys $k_j$ and values $v_j$,

$$
\mathrm{Attn}(q,K,V)
=\sum_j \frac{\exp(q\cdot k_j/\sqrt{d_k})}
{\sum_{j'}\exp(q\cdot k_{j'}/\sqrt{d_k})}\,v_j .
\qquad (3)
$$

The softmax weights sum to one, so the output is a convex combination of the
values — attention can only select and blend what is already in the sequence.
The scaling matters because $q\cdot k$ is a sum of $d_k$ products: if the
entries have unit variance the dot product has variance $d_k$, so without the
$1/\sqrt{d_k}$ the spread of the logits grows with width and the softmax
saturates as models get wider. Dividing by $\sqrt{d_k}$ makes the score
distribution width-independent.

### Multi-head

One attention with depth $d$ has a single softmax per token. Splitting the
width into $H$ heads of depth $d_k=d/H$ gives $H$ independent distributions:

$$
\mathrm{MHA}(x)=\Bigl[\mathrm{head}_1;\dots;\mathrm{head}_H\Bigr]W_O,
\qquad
\mathrm{head}_i=\mathrm{Attn}\bigl(xW_i^Q,\,xW_i^K,\,xW_i^V\bigr).
\qquad (4)
$$

Each head is a $d\times d_k$ projection of the same input. The concatenation is
an axis swap followed by a reshape, which is why the implementation transposes
`(batch, length, heads, depth)` to `(batch, heads, length, depth)` once and
never touches the layout again.

### Causal masking

A decoder may not read the future, so scores with $j>i$ are set to a large
negative number before the softmax:

$$
S_{ij}\leftarrow-\infty \quad\text{for } j>i,
\qquad
P=\mathrm{softmax}(S).
\qquad (5)
$$

The implementation uses $-10^{9}$ rather than $-\infty$ to keep the array
finite; after subtracting the row maximum the exponential of a masked entry is
exactly zero in float64, so the weights are exactly zero. The mask must also be
reapplied in the backward pass, otherwise the upstream gradient would flow into
entries the forward pass never used.

### Layer normalisation

$$
\mu=\frac1d\sum_i x_i,
\qquad
\sigma^2=\frac1d\sum_i (x_i-\mu)^2,
\qquad
y_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} .
\qquad (6)
$$

For an upstream gradient $g$ and $\hat y=(x-\mu)/\sigma$,

$$
\frac{\partial L}{\partial x}
=\frac{1}{d\sigma}\Bigl(d\,g-\mathbf 1\sum_i g_i
-\hat y\sum_i g_i\hat y_i\Bigr).
\qquad (7)
$$

The first two terms remove the component of the gradient along the mean, the
third removes the component along the variance. This is the derivative the code
implements, with `centred` holding $x-\mu$ and `inverse` holding $1/\sigma$.

### RMSNorm

RMSNorm drops the mean subtraction and normalises by the root mean square:

$$
y=\frac{x}{r},
\qquad
r=\sqrt{\frac1d\sum_i x_i^2+\epsilon},
\qquad
\frac{\partial L}{\partial x}
=\frac{1}{r}\Bigl(g-\frac{x\,(g\cdot x)}{d\,r^2}\Bigr).
\qquad (8)
$$

Without the mean term there is no $\mathbf 1\sum g_i$ correction, and a constant
offset in the input passes straight through. That is acceptable for a residual
stream, whose mean carries no information, but not for standardised features.

### Post-LN versus pre-LN

Post-LN puts the normaliser on the residual path, so the gradient from layer
$\ell+1$ reaches layer $\ell$ through the layer-norm Jacobian of (7). That
Jacobian has no identity term; it projects the gradient onto the
zero-mean, zero-correlation subspace. Stacked deeply, the product of those
projections shrinks, which is why the original transformer needed a learning-rate
warm-up.

Pre-LN moves the norm off the path:

$$
a=x+\mathrm{Attn}\bigl(\mathrm{LN}(x)\bigr),
\qquad
y=a+\mathrm{FFN}\bigl(\mathrm{LN}(a)\bigr).
\qquad (9)
$$

Now $\partial y/\partial a=I+\dots$, so the identity path is clean at every
depth. The cost is that the stream itself accumulates unscaled branch outputs,
so its variance grows with depth and a final normalisation before the head is
the usual remedy. The demo's two-block stack is too shallow for the growth to
hurt, but the effect is visible: at initialisation the SwiGLU model's loss is
8.28 against 1.15 for post-LN, because the gated branch injects larger values
into the stream.

### The backward pass through the softmax

If $P=\mathrm{softmax}(S)$ and the upstream gradient is $dP$, then

$$
dS_{ij}=P_{ij}\Bigl(dP_{ij}-\sum_{j'}dP_{ij'}P_{ij'}\Bigr).
\qquad (10)
$$

In matrix form the code computes `d_weights = d_context @ v.T`, then applies
(10) row by row, zeroes the masked entries, and only then multiplies by
$1/\sqrt{d_k}$. The order matters: masking after the softmax derivative would
leave a nonzero gradient in positions the forward pass never read.

### Parameter count

For width $d$, $H$ heads, hidden width $h$ and $K$ classes, one block with a
single attention and a ReLU pair has

$$
P_{\mathrm{block}}=4(d^2+d)+(dh+h)+(hd+d)
=4d^2+4d+2dh+d+h .
\qquad (11)
$$

At $d=8$, $h=16$ that is $288+144+136=568$; two blocks plus the $dK+K=18$
parameter head give 1154. SwiGLU adds $dh+h=144$ per block, and the decoder adds
a second attention at $4(d^2+d)=288$ per block.

### Optimiser

$$
\theta\leftarrow\theta-\eta\,
\frac{m/(1-0.9^{t})}{\sqrt{v/(1-0.999^{t})}+\epsilon},
\qquad
m=0.9m+0.1g,\quad v=0.999v+0.001g^2,
\qquad (12)
$$

with $t$ the minibatch step. At 480 sequences and batch 64 the demo takes eight
steps per epoch, 480 in total.

## Code walkthrough

The softmax subtracts the row maximum before exponentiating, which is the only
numerically stable way to write it:

```python
def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)
```

Attention itself is four projections, an axis swap, and a masked softmax:

```python
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = None
        if causal:
            mask = np.triu(np.ones((length, places), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
        weights = _softmax(scores)
        context = weights @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch, length, dim)
```

The mask is built with `triu` one diagonal above the main one, so position $i$
may read positions $0..i$ and nothing later. With `length` and `places` allowed
to differ, the same code serves self-attention and cross-attention.

The backward pass is (10) verbatim, with the mask zeroed before the scale is
applied:

```python
        d_weights = d_context @ v.transpose(0, 1, 3, 2)
        d_v = weights.transpose(0, 1, 3, 2) @ d_context
        d_scores = weights * (d_weights
                              - (d_weights * weights).sum(axis=-1,
                                                          keepdims=True))
        if mask is not None:
            d_scores = np.where(mask, 0.0, d_scores)
        d_scores *= self.scale
```

Layer norm returns both its output and the statistics its derivative needs:

```python
def _layer_norm(x: np.ndarray):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    inverse = 1.0 / np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    return centred * inverse, (centred, inverse)
```

```python
def _layer_norm_backward(gradient: np.ndarray, cache):
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)
```

The two reductions are the two terms of (7): `total` removes the mean direction
and `scaled` removes the variance direction.

Post-LN is the layout of (1), and its backward pass is where the ordering shows
up:

```python
        if self.variant == "post_ln":
            attention, attention_cache = self.self_attention.forward(x)
            normed, norm_cache = self._norm(x + attention)
            branch, ffn_cache = self.feed_forward.forward(normed)
            output, final_cache = self._norm(normed + branch)
            return output, ("post", attention_cache, norm_cache, ffn_cache,
                            final_cache)
```

```python
            d_mixed = self._norm_backward(gradient, final_cache)
            d_normed = d_mixed + self.feed_forward.backward(d_mixed,
                                                           ffn_cache)
            d_residual = self._norm_backward(d_normed, norm_cache)
            d_attention, _ = self.self_attention.backward(d_residual,
                                                          attention_cache)
            return d_residual + d_attention, None
```

The gradient out of the final norm is split: one copy goes into the
feed-forward, the other continues along the skip, and they meet again as the
input to the second layer norm — which is why `d_mixed` appears twice.

Pre-LN keeps the stream clean by normalising only the copies the branches read:

```python
        attention, attention_cache = self.self_attention.forward(
            normed, causal=self.causal)
        mixed = x + attention
        normed2, norm_cache2 = self._norm(mixed)
        branch, ffn_cache = self.feed_forward.forward(normed2)
        return mixed + branch, ("pre", norm_cache, attention_cache,
                                norm_cache2, ffn_cache)
```

The decoder is the same pattern three times: normalise, attend, add, normalise,
cross-attend, add, normalise, feed-forward, add:

```python
        if self.variant == "decoder":
            normed, norm_cache = self._norm(x)
            attention, attention_cache = self.self_attention.forward(
                normed, causal=True)
            mixed = x + attention
            normed2, norm_cache2 = self._norm(mixed)
            branch, cross_cache = self.cross_attention.forward(
                normed2, memory=memory, causal=True)
            mixed2 = mixed + branch
```

The cross-attention takes its queries from `normed2` and its keys and values
from `memory`, so its backward pass returns two gradients: one for the block's
stream and one for the memory. This classifier discards the second — it only
accumulates it and checks that it stays finite — but a model that is itself an
encoder needs it.

SwiGLU replaces the activation with a product of two projections:

```python
            gate_pre = flat @ p[f"{self.name}.gate.W"] \
                + p[f"{self.name}.gate.b"]
            hidden = _silu(pre) * gate_pre
```

```python
            d_pre = d_hidden * gate_pre * _silu_grad(pre)
            d_gate = d_hidden * _silu(pre)
```

The gate is linear, so its derivative is the other factor's value; the activated
branch carries the gate's value and the SiLU derivative. The `_sigmoid` helper
splits at zero so that neither branch ever evaluates `exp` of a large positive
number, which is what makes the gate stable for large pre-activations.

The model counts what it built, rather than hard-coding the number:

```python
        self.n_params_ = int(sum(value.size
                                 for value in self.parameters.values()))
```

which is how the demo can assert that post-LN, pre-LN, the encoder and the
cross-only block all have 1154, that SwiGLU has 1442, and that the decoder has
1730.

## Complexity

For batch $B$, sequence length $L$, width $d$, $H$ heads, hidden width $h$ and
depth $N$:

- **Attention:** $O(BL^2d)$ time for the scores and the weighted sum, $O(BHL^2)$
  memory for the weights when they are kept for the backward pass. This is the
  term that makes long sequences expensive.
- **Feed-forward:** $O(BLdh)$ time, $O(BLh)$ memory for the hidden state.
  With the usual $h=4d$ this is the larger term for short sequences.
- **One block:** $O(BL^2d+BLdh)$; the decoder doubles the attention term.
- **A stack of $N$ blocks:** $N$ times that, with $O(Nd^2+Ndh)$ parameters.
- **Parallel attention:** the same arithmetic as separate attention and
  feed-forward blocks, but the two halves have no dependency, so their matrix
  products can be scheduled together.
- **The demo:** 480 sequences of 8 tokens at width 8 — a few tenths of a
  MFLOP per block per batch, which is why 60 epochs take seconds.

## Failure modes

**Post-LN needs a warm-up.** The normaliser sits on the residual path, so the
gradient is repeatedly projected by (7). At depth 2 the demo trains fine with a
constant learning rate, but that is not evidence for depth 24; the original
transformer's warm-up schedule exists because deep post-LN stacks diverge
without it. Pre-LN removes the need, at the price of an unbounded stream.

**Pre-LN's stream grows with depth.** Each block adds an unscaled branch output,
so the stream's variance is a sum over depth. The demo sees the first symptom at
initialisation: the SwiGLU model starts at loss 8.28 while post-LN starts at
1.15, because the gated branch's values are larger than the ReLU branch's. It
still trains to 0.944, but a deeper stack needs the final norm that this
classifier omits.

**A masked entry must stay masked in the backward pass.** If the `-1e9` entries
are not zeroed after the softmax derivative, the gradient flows into positions
the forward pass never read. The demo tests the forward direction by perturbing
tokens 5 to 7 and asserting that positions 0 to 4 are bit-identical; the same
check on the gradient would catch a missing mask.

**Saturated softmax makes numerical gradient checks lie.** With cross-entropy on
a model whose logits are already confident, the probability of the true class can
be $10^{-7}$, and the relative rounding error of `1 + 1e-7` is $10^{-9}$. A
central difference with step $10^{-6}$ then reports errors of $10^{-3}$ on a
gradient that is exact. The demo avoids this by checking a random linear
projection of the logits instead of the loss; all eight variants agree to at most
$1.3\times10^{-8}$.

**The scale factor is not optional at width.** Without $1/\sqrt{d_k}$ the logits
of a wide model have standard deviation $\sqrt{d_k}$ and the softmax is saturated
from the first step. The failure looks like a dead model, not like a bug.

**The head count must divide the width.** `dim=8, heads=3` is refused rather than
silently truncated. Real implementations pad the width to a multiple of the head
count; hiding the mismatch produces heads of unequal depth.

**Reading one position is a choice.** The classifier reads token 0 and ignores
the other seven. That works because attention lets token 0 aggregate the whole
sequence in one block, but it also means the model has no reason to keep
information anywhere else. Mean pooling is the alternative, and it dilutes a
product: averaging eight positions mixes the two markers with six irrelevant
ones, so the averaged feature is only weakly correlated with the label.

**Parallel attention changes the function class.** In `parallel_attention` the
feed-forward cannot read the attention output, so a task that needs the result of
one before the other is out of reach. The demo measures the cost: 0.956 against
1.000 for the sequential encoder.

**RMSNorm cannot remove an offset.** A constant added to every channel survives
the normalisation and passes into the next block. For a residual stream that
carries no mean information this is harmless; applied to raw features with large
means it is not.

**There is no positional information here.** Attention is permutation
equivariant, so this block cannot tell "a then b" from "b then a". The marker
task is deliberately order-free; the next atom adds positions.

## Applications

The encoder block is BERT and its descendants, and the backbone of vision
transformers once images are cut into patches. The causal pre-LN block is GPT and
essentially every decoder-only language model. The full decoder — masked self
attention plus cross attention — is the T5 and translation half of the original
paper.

Cross-attention alone is the retrieval and multimodal pattern: queries from one
modality, keys and values from another, as in image-conditioned text models and
in retrieval-augmented generators.

Parallel attention is PaLM's and GPT-J's block. It saves the sequential
dependency between the two halves at the cost of some accuracy, which is a good
trade at very large scale where the matrix products dominate.

SwiGLU is the standard feed-forward in modern language models; RMSNorm is the
standard normaliser in the same models, usually paired with pre-LN.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., Polosukhin, I. (2017). *Attention is all you need.* NeurIPS.
  Equations (3), (4) and the post-LN layout of (1).
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization.*
  arXiv:1607.06450. Equations (6) and (7).
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan,
  Y., Wang, L., Liu, T.-Y. (2020). *On layer normalization in the transformer
  architecture.* ICML. Why post-LN needs warm-up.
- Zhang, B., Sennrich, R. (2019). *Root mean square layer normalization.*
  NeurIPS. Equation (8).
- Shazeer, N. (2020). *GLU variants improve transformer.* arXiv:2002.05202.
  Equation (2).
- Chowdhery, A. et al. (2022). *PaLM: scaling language modeling with
  pathways.* arXiv:2204.02311. Parallel attention and feed-forward.
- Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018). *Improving
  language understanding by generative pre-training.* The causal pre-LN block.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: pre-training
  of deep bidirectional transformers.* NAACL. The encoder block.
- Kingma, D. P., Ba, J. (2015). *Adam: a method for stochastic optimization.*
  ICLR. Equation (12).
