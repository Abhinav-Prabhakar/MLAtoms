# Token-mixing blocks — Mixer, gMLP, ConvMixer and ResMLP

`mixer.py` · Neural blocks · Tier 2 · 4 variants

## Intuition

A vision transformer splits an image into patches and mixes information in two
directions: across patches, by attention, and across channels, by an MLP. The
four blocks here keep that two-stage recipe and replace attention with
something cheaper. The question every variant answers is: how do patches talk
to each other, and at what cost?

`mlp_mixer` lets a small MLP do the talking, applied along the token axis. That
is a dense, position-specific linear map followed by a nonlinearity — no
convolution, no attention, $O(T^2)$ weights per block. `gmlp` gates one half of
the channels by a token-mixed version of the other half, with the mixing matrix
initialised at zero so the gate starts as the identity. `convmixer` uses a
depthwise convolution over the token axis, which restores locality: a patch
only sees its neighbours within the kernel radius. `resmlp` drops the
nonlinearity from the token mixer entirely and uses an affine norm, leaving a
linear map across patches and a GELU MLP across channels.

Three of the four are pure token-sequence models; `convmixer` is the only one
with a locality prior. The demo measures the parameter bill, checks the
identity gate, measures the exact receptive field of a two-layer ConvMixer, and
checks every gradient against finite differences.

## Variants

| variant | token mixer | channel mixer | norm | locality |
|:--|:--|:--|:--|:--|
| `mlp_mixer` | $\mathrm{MLP}$ on $X^{\top}$ | $\mathrm{MLP}$ | LayerNorm | none |
| `gmlp` | spatial gating unit | $\mathrm{MLP}$ | LayerNorm | none |
| `convmixer` | depthwise $k$ | pointwise | none | $d\lfloor k/2\rfloor$ |
| `resmlp` | linear map $W_t$ | $\mathrm{MLP}$ | affine | none |

The input is always `(batch, tokens, dim)`. For images, tokens are flattened
patches; for audio, frames; for text, a sequence of embeddings. None of these
blocks carries a position table, because the token mixer itself is
position-specific: weight $W_t[i,j]$ belongs to the pair of positions $i$ and
$j$ and is learned, which is how order enters without an embedding.

## Derivation

### The shared block template

Every variant is a residual stack of two sub-blocks,

$$
X\leftarrow X+\mathcal T\bigl(\mathrm{LN}(X)\bigr),\qquad
X\leftarrow X+\mathcal C\bigl(\mathrm{LN}(X)\bigr),
\qquad (1)
$$

where $\mathcal T$ mixes along the token axis and $\mathcal C$ mixes along the
channel axis. The pre-norm placement keeps the residual path an identity, so a
deep stack starts as a shallow one; the final output is an affine head. The
demo's two-block models at $T=8$, $d=6$, hidden 12 and token-hidden 4 cost 464,
648, 144 and 512 parameters respectively.

### MLP-Mixer

The token step transposes the sequence so that the MLP acts on positions:

$$
U=\sigma\bigl(\mathrm{LN}(X)^{\top}W_1+b_1\bigr),\qquad
M=\bigl(UW_2+b_2\bigr)^{\top},
\qquad (2)
$$

with $W_1\in\mathbb R^{T\times h_t}$ and $W_2\in\mathbb R^{h_t\times T}$. The
two transposes are the whole trick: the same code path that mixes channels can
mix tokens if the axes are swapped. Nothing in (2) is shared across positions —
$W_1[i,:]$ is the feature built for position $i$ — so the parameter count grows
with $T$ and the model is not permutation-equivariant.

The channel step is the usual pre-norm MLP on the last axis. In the demo the
token matrices are $8\times4$ and $4\times8$ for all six channels, which is 64
numbers per block against 162 for the channel MLP.

### gMLP: a gate built from the other half

$$
[A,B]=\sigma\bigl(\mathrm{LN}(X)W+b\bigr),\qquad
S=A\odot\bigl(BW_g+\beta\bigr),\qquad
M=SW_o+b_o,
\qquad (3)
$$

with $W_g\in\mathbb R^{T\times T}$ shared across channels and $\beta$ a
per-token bias. The initialisation is the interesting part: $W_g=0$ and
$\beta=\mathbf 1$, so the gate is exactly one at step zero and the block opens
as a plain channel MLP. The demo checks `gate == 1.0` and
`s == activated` before any training. The gate then learns which positions
should modulate which features, and $W_g$ is the only $T^2$ object in the
block.

### ConvMixer: a depthwise convolution over tokens

$$
X\leftarrow X+\mathrm{DWConv}_k(X),\qquad
X\leftarrow X+W_p\,\sigma(X),
\qquad (4)
$$

where the depthwise convolution runs along the token axis with zero padding
$k/2$ and one $k$-tap filter per channel:

$$
\mathrm{DWConv}_k(X)[b,t,c]=\sum_{j=0}^{k-1}w[c,j]\,
X[b,\,t+j-k/2,\,c].
\qquad (5)
$$

The second step is a $1\times1$ convolution, which is a linear map across
channels. ConvMixer's selling point is that (5) is local: information travels at
most $\lfloor k/2\rfloor$ positions per layer, so after $d$ layers the receptive
field has radius $d\lfloor k/2\rfloor$. The demo perturbs one token with
$d=2$ and $k=3$ and finds a change at tokens 0 through 4 and nowhere else —
exactly radius 2. The reference implementation adds batch norm after both
steps; this version omits it because batch statistics make a single-example
demo ambiguous, and the block algebra is unchanged.

### ResMLP: linear token mixing and an affine norm

$$
X\leftarrow X+(Xa_1+b_1)W_t,\qquad
X\leftarrow X+\mathcal C(Xa_2+b_2),
\qquad (6)
$$

with $W_t\in\mathbb R^{T\times T}$ and per-channel scales and biases. There is
no nonlinearity on the token path, so this mixer is a genuine linear operator
on the token axis. The norm is affine rather than LayerNorm: no mean is
subtracted and no variance divides, which is cheaper and, empirically, enough
when the residual stream carries the normalisation. A consequence is visible
in the demo: adding a constant vector $c$ to every token shifts the normalised
input by exactly $a_1\odot c$, whereas a mean-centred norm would remove $c$
entirely.

### LayerNorm and its backward pass

With $\mu$ and $\sigma^2$ over the channel axis and
$\hat x=(x-\mu)/\sigma$,

$$
\frac{\partial L}{\partial x}
=\frac{1}{\sigma}\Bigl(g-\bar g-\hat x\,\overline{g\hat x}\Bigr),
\qquad (7)
$$

where the bars are means over channels. The three terms are the direct path,
the shift created by moving the mean, and the scale created by moving the
variance. The demo's LayerNorm probe matches central differences to
$7.4\times10^{-10}$.

### Why the depthwise backward is a scatter

For (5), the weight gradient sums over batch, token and channel positions:

$$
\frac{\partial L}{\partial w[c,j]}
=\sum_{b,t}\frac{\partial L}{\partial y[b,t,c]}\,
X[b,t+j-k/2,c],
\qquad
\frac{\partial L}{\partial X[b,s,c]}
=\sum_{j}\frac{\partial L}{\partial y[b,s-j+k/2,c]}\,w[c,j].
\qquad (8)
$$

The second sum is an overlap-add: each of the $k$ taps writes its gradient back
into a shifted copy of the input. The implementation keeps the padded windows
from the forward pass and adds each tap into `d_padded`, which is why the
padding offset reappears on the way out.

### Parameter counts

At $T$ tokens, $d$ channels, channel hidden $h$, token hidden $h_t$ and depth
$L$:

| variant | per block | total, $T{=}8,d{=}6,h{=}12,h_t{=}4,L{=}2$ |
|:--|:--|--:|
| `mlp_mixer` | $2Th_t+(dh+h)+(hd+d)$ | 464 |
| `gmlp` | $(2dh+2h)+T^2+T+(hd+d)$ | 648 |
| `convmixer` | $(dk+d)+(d^2+d)$ | 144 |
| `resmlp` | $2d+T^2+(d^2+d)+(hd+d)$ | 512 |

Every total adds the affine head, $2d$. The ordering is a price list for the
token mixer: a depthwise convolution costs $dk+d$ and buys locality; a Mixer
MLP costs $2Th_t$; a gMLP gate costs $T^2+T$; ResMLP's linear mixer costs
$T^2$ and gives up the nonlinearity.

### The fit

The demo trains one Mixer for 300 steps of plain gradient descent at step size
0.02 on a random target, taking the loss from 160.67 to 0.0060. That is not a
claim about convergence speed — the target is random and the model is
over-parameterised for it — but it does show that the hand-written backward
pass is coherent enough to descend.

## Code walkthrough

The normalisation is shared by both non-affine variants, so it lives at module
level. Storing the mean and inverse standard deviation is enough to rebuild the
backward pass without recomputing anything:

```python
def _layer_norm(x: np.ndarray, eps: float = 1e-5):
    """Normalise over the channel axis, the last one."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    out = (x - mean) * inv
    return out, {"x": x, "mean": mean, "inv": inv}
```

The backward pass is (7) verbatim. The two mean reductions are over channels,
the same axis the forward pass normalised:

```python
def _layer_norm_backward(gradient: np.ndarray, cache) -> np.ndarray:
    x, mean, inv = cache["x"], cache["mean"], cache["inv"]
    centred = x - mean
    shift = gradient.mean(axis=-1, keepdims=True)
    scale = (gradient * centred).mean(axis=-1, keepdims=True)
    return inv * (gradient - shift - centred * inv ** 2 * scale)
```

Mixer's two stages are the template (1) with a transpose on the token path. The
pre-activation is cached, not just the activation, because the GELU derivative
needs its input:

```python
        if self.variant == "mlp_mixer":
            norm, norm_cache = _layer_norm(x)
            flat = norm.transpose(0, 2, 1)
            inner = flat @ p[f"{tag}.token.in"]
            hidden = _gelu(inner)
            mixed = (hidden @ p[f"{tag}.token.out"]).transpose(0, 2, 1)
            first = x + mixed
```

The gMLP gate is one matrix product plus a per-token bias. The bias broadcast
is written explicitly with `[None, None, :]` because the product lives in the
transposed layout, before the transpose back:

```python
            gate = (values.transpose(0, 2, 1) @ p[f"{tag}.gate"]
                    + p[f"{tag}.gate.b"][None, None, :]).transpose(0, 2, 1)
            gated = activated * gate
            mixed = gated @ p[f"{tag}.out"] + p[f"{tag}.out.b"]
            out = x + mixed
```

ResMLP's token mixer is the same transpose without the MLP. Its norm is two
elementwise operations, and the backward pass has no mean to correct:

```python
        scale, bias = p[f"{tag}.token.scale"], p[f"{tag}.token.bias"]
        norm = x * scale + bias
        mixed = (norm.transpose(0, 2, 1)
                 @ p[f"{tag}.token.mix"]).transpose(0, 2, 1)
        first = x + mixed
```

ConvMixer's forward keeps the padded windows, since both the weight gradient
and the input gradient read them. The einsum is (5) with the kernel axis made
explicit:

```python
            _, window = _windows(x, self.kernel)
            depth = np.einsum("btck,ck->btc", window,
                              p[f"{tag}.depth"]) + p[f"{tag}.depth.b"]
            first = x + depth
            norm = _gelu(first)
            point = norm @ p[f"{tag}.point"] + p[f"{tag}.point.b"]
            out = first + point
```

The depthwise backward scatters each of the $k$ taps into a shifted slice of
the padded input. The `d_padded` buffer is one token longer at each end, and
the unpadding happens on the return:

```python
            d_window = d_first[..., None] * p[f"{tag}.depth"][None, None, :, :]
            pad = self.kernel // 2
            d_padded = np.zeros((gradient.shape[0],
                                 self.tokens + 2 * pad, self.dim))
            for start in range(self.kernel):
                d_padded[:, start:start + self.tokens, :] += \
                    d_window[..., start]
            g[f"{tag}.depth"] += np.einsum(
                "btck,btc->ck", cache["window"], d_first)
            g[f"{tag}.depth.b"] += d_first.sum(axis=(0, 1))
            return d_first + d_padded[:, pad:pad + self.tokens, :]
```

## Complexity

For $T$ tokens, $d$ channels, channel hidden $h$, token hidden $h_t$, $L$
blocks and kernel $k$:

- **Mixer token step:** $O(Td h_t + T h_t d)=O(Td h_t)$ per block and
  $O(Th_t+h_tT)$ parameters. The transpose is a view in NumPy.
- **Mixer channel step:** $O(Tdh)$ time and $O(dh)$ parameters per block.
- **gMLP:** the channel projection is $O(Td\cdot 2h)$; the gate is $O(ThT)$
  because $W_g$ is dense over positions. That $T^2h$ term is the expensive one
  for long sequences.
- **ConvMixer:** $O(Tdk)$ for the depthwise convolution and $O(Td^2)$ for the
  pointwise one, with only $O(dk+d^2)$ parameters. Cost grows linearly in $T$,
  which is why it is attractive for high-resolution patches.
- **ResMLP:** $O(T^2d)$ for the linear token mixer and $O(Tdh)$ for the
  channel MLP; the token matrix alone is $T^2$ parameters.
- **Backward:** the same order in every case. ConvMixer's overlap-add is $k$
  slices of $O(Td)$; gMLP's gate backward is another $O(ThT)$ product.
- **The demo:** widths of 6 and 8 tokens, so every product is tiny; the
  finite-difference loop over four variants and the 300-step fit dominate the
  file's runtime, which is still well under a second.

## Failure modes

**Token mixing is position-specific, not permutation-equivariant.** Unlike the
graph blocks, Mixer and ResMLP learn a different weight for every pair of
positions. Feeding a permuted sequence does not permute the output. The model
therefore learns order from the mixer, and it cannot generalise to a longer
sequence without resizing $W_t$ or $W_g$.

**ConvMixer's locality is also a limit.** With $k=3$, information needs
$T/2$ layers to cross a sequence. For long-range tasks this is worse than
attention; for images with local structure it is exactly the prior wanted. The
demo's radius of 2 over two layers is the property, not a bug.

**The gMLP gate can collapse.** Zero-init $W_g$ and a bias of one make the gate
the identity, but nothing keeps it near one during training. If the gate
saturates at zero, the block's token path is switched off and gradients to
$W_g$ vanish. Weight decay or a gate regulariser is common; neither is here.

**ResMLP's affine norm does not remove offsets.** A constant added to every
token survives into the next block, scaled by $a_1$. That is fine in a residual
stack, but it means the block is not invariant to the mean of its input the way
a LayerNorm block is. It also has no variance control, so large activations
pass through untouched.

**ConvMixer without batch norm changes the optimisation problem.** The
reference uses batch norm after both steps, which normalises across the batch
and the token axis. Removing it, as here, keeps the algebra readable but makes
the block sensitive to the input scale. The gradients are still correct for the
block that is implemented.

**The depthwise convolution assumes an odd kernel.** `_windows` pads by
`kernel // 2` on both sides; an even kernel would need asymmetric padding to
keep the token count. The constructor rejects even kernels rather than
silently returning a shorter sequence.

**A dense token mixer is quadratic in the token count.** For $T=1024$ patches,
$W_t$ alone is a million parameters per block. Mixer and ResMLP are practical
at patch counts in the low hundreds; beyond that the token step dominates both
memory and compute.

**The channel MLP is the parameter hog.** At $d=6$, $h=12$ the channel path is
162 of Mixer's 226 parameters per block, and the ratio grows with $d$. The
token path is the architectural novelty; the capacity is still in the MLP.

**The residual stream carries the scale.** Because both sub-blocks are added to
$X$, the output magnitude grows with depth unless a norm sits on the residual
path. Pre-norm blocks mitigate this but do not eliminate it, and the affine head
is the only final normalisation.

**Small batch shapes are easy to get wrong.** A transposed MLP over tokens
silently mixes the batch axis if the reshape is applied to the wrong layout.
The demo checks shapes at every stage, and the validation rejects a
two-dimensional input rather than treating it as a batch of one.

## Applications

MLP-Mixer showed that attention is not required for competitive image
classification at moderate resolution; its token MLP is the direct ancestor of
several efficient vision backbones. gMLP's gating unit is used where a cheap
token interaction is wanted and the $T^2$ gate is affordable. ConvMixer is a
strong patch-based baseline for small images and a common component in
efficient segmentation heads, because its cost is linear in the token count.
ResMLP's affine norm and linear token mixer appear in fast vision models and in
audio taggers, where the residual stream already carries the normalisation.

The same four mixers are used outside vision: on spectrogram frames, on
time-series windows, and as the cross-patch layer in diffusion models, where
the token count is modest and a dense mixer is cheaper than attention.

## References

- Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner,
  T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J., Lucic, M., Dosovitskiy,
  A. (2021). *MLP-Mixer: an all-MLP architecture for vision.* NeurIPS.
  Equation (2).
- Liu, H., Dai, Z., So, D. R., Le, Q. V. (2021). *Pay attention to MLPs.*
  NeurIPS. gMLP and the spatial gating unit, equation (3).
- Trockman, A., Kolter, J. Z. (2022). *Patches are all you need?* arXiv.
  ConvMixer, equations (4) and (5).
- Touvron, H., Bojanowski, P., Caron, M., Cord, M., El-Nouby, A., Grave, E.,
  Izacard, G., Joulin, A., Synnaeve, G., Verbeek, J., Jégou, H. (2021).
  *ResMLP: feedforward networks for image classification with data-efficient
  training.* arXiv. Equations (6) and the affine norm.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization.* arXiv.
  Equation (7).
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
  Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
  Uszkoreit, J., Houlsby, N. (2021). *An image is worth 16x16 words.* ICLR. The
  patch-token view these blocks replace.
