# ConvNeXt blocks — depthwise mixing, GRN and layer scale

`convnext.py` · Neural blocks · Tier 2 · 4 variants

## Intuition

A ConvNeXt block looks like a transformer block that has been rewritten with
convolutions. The token-mixing step is a $7\times7$ depthwise convolution, one
filter per channel, which is the convolutional analogue of attention's
neighbourhood. The channel-mixing step is a $1\times1$ expand, a GELU, and a
$1\times1$ project, exactly the inverted bottleneck of a transformer's FFN.
What remains is a stem that turns the image into a feature map — a $4\times4$
convolution with stride 4, which is precisely patch extraction followed by a
linear layer.

The four variants isolate the pieces. `convnext` is the V1 block with layer
scale. `convnext_v2` replaces layer scale with global response normalisation
(GRN) and drops the scale. `grn` keeps the V1 block but adds GRN, which
separates the normalisation from the rest of the V2 recipe. `patchify_stem`
stops after the stem, showing what the block actually starts from.

The demo checks three structural facts: the stem is exactly a patchwise linear
map, the depthwise convolution touches exactly a $7\times7$ window, and layer
scale makes a V1 block a near-identity at initialisation while V2 acts at full
strength. Every gradient is checked against central differences, with GRN's
learned scale set away from zero so the normalisation path is actually
exercised.

## Variants

| variant | stem | block | norm | scale |
|:--|:--|:--|:--|:--|
| `convnext` | $4\times4$, stride 4 | depthwise 7, LN, MLP | none | layer scale |
| `convnext_v2` | $4\times4$, stride 4 | depthwise 7, LN, MLP | GRN | none |
| `grn` | $4\times4$, stride 4 | depthwise 7, LN, MLP | GRN | layer scale |
| `patchify_stem` | $4\times4$, stride 4 | none | none | none |

The interface is `(batch, height, width, channels)` in and
`(batch, height/4, width/4, dim)` out, so the stem is doing the downsampling
that a patch embedding does in a vision transformer. At an $8\times8$ image with
three input channels and $d=8$, the grid is $2\times2$ and the stem costs 392
parameters.

## Derivation

### The stem is patchify plus a linear map

For stride $s$, patch $(i,j)$ collects the $s^2C_{\text{in}}$ values

$$
P_{ij}=\bigl[X_{si:si+s,\;sj:sj+s,\;:}\bigr]_{\text{flat}},\qquad
Y_{ij}=P_{ij}W_s+b_s,
\qquad (1)
$$

with $W_s\in\mathbb R^{s^2C_{\text{in}}\times d}$. A convolution with kernel
$s$ and stride $s$ computes the same thing with no overlap, so "patchify then
linear" and "strided convolution" are one operation. The demo extracts the
patches by hand, applies $W_s$, and matches the stem output; at $s=4$ and
$C_{\text{in}}=3$ each patch is 48 numbers, which is the fan-in of the stem.

The backward pass is the transposed map: gradients are projected back to patch
space and scattered into the image with the inverse reshape. The implementation
keeps `_patchify` and `_unpatchify` as exact inverses, which is why the stem's
gradient check passes without a special case.

### Depthwise mixing

$$
Z_{ijc}=\sum_{u,v}W^{c}_{uv}\,
X_{i+u-\lfloor k/2\rfloor,\;j+v-\lfloor k/2\rfloor,\;c}+b_c.
\qquad (2)
$$

One filter per channel, shared across space. The kernel is $7\times7$, so a
single block moves information exactly three pixels in each direction; the demo
perturbs one feature and finds the change confined to that $7\times7$ window.
The cost is $O(HWdk^2)$, linear in the image size rather than quadratic as
attention would be.

### The block

$$
H=\mathrm{gelu}\bigl(\mathrm{LN}(Z)W_1+b_1\bigr),\qquad
Y=X+\gamma\odot\bigl(HW_2+b_2\bigr),
\qquad (3)
$$

with $W_1\in\mathbb R^{d\times4d}$ and $W_2\in\mathbb R^{4d\times d}$. The
width ratio is four, as in a transformer FFN; the LayerNorm is over channels at
each spatial position, so it is position-wise and does not mix space. $\gamma$
is layer scale, a per-channel multiplier initialised at $10^{-6}$:

$$
\gamma_c=10^{-6}\ \Rightarrow\ Y\approx X.
\qquad (4)
$$

The demo measures the relative change $\lVert Y-X\rVert/\lVert X\rVert$ through
a V1 block and gets $4.8\times10^{-7}$; a V2 block, which has no scale, changes
its input by a factor of 1.22. That is the whole argument for layer scale: a
deep stack starts as a stack of identities, and each block turns itself on
gradually.

### Global response normalisation

GRN pools each channel over space and divides by the mean pooled norm:

$$
n_c=\sqrt{\sum_{h,w}X_{hwc}^2},\qquad
r_c=\frac{n_c}{\frac1C\sum_k n_k+\epsilon},
\qquad (5)
$$

$$
Y=X+\gamma\odot X\odot r+\beta.
\qquad (6)
$$

The residual form is easy to miss: the normalised term is added to $X$, so
$\gamma=0$ and $\beta=1$ makes the layer a shift by one. The demo checks exactly
that initial state and checks that the normalised norms average one over
channels, which is what the division by the mean enforces. GRN is a *global*
operation — every spatial position affects every channel's norm — which is why
it is not a drop-in replacement for LayerNorm in a convolutional stack.

The backward pass needs the derivative of $r_c$ through both the numerator and
the mean. With $A_c=\sum_{hw}g_{hwc}X_{hwc}$ and
$g=\partial L/\partial Y$:

$$
\frac{\partial L}{\partial X_{hwc}}
=g_{hwc}\bigl(1+\gamma_c r_c\bigr)
+\frac{X_{hwc}}{n_c}\left(
\frac{\gamma_c A_c}{m+\epsilon}
-\frac{\sum_k\gamma_k n_k A_k}{C(m+\epsilon)^2}
\right),
\qquad (7)
$$

where $m$ is the channel mean of $n$. The first bracket term is the direct
effect of channel $c$'s norm; the second is the effect of moving the mean, which
is shared by every channel. Getting the second term wrong is easy because it
looks like a per-position quantity — it is a scalar per example, summed over
space as well as channels.

### V1 and V2 differ by two choices

$$
\text{V1}: Y=X+\gamma\odot P,\qquad
\text{V2}: Y=X+P,\quad P=(\mathrm{GRN}\circ H)W_2+b_2.
\qquad (8)
$$

So V2 removes layer scale and inserts GRN between the GELU and the project. The
`grn` variant keeps the scale and inserts GRN, which is the ablation that
isolates the normalisation from the scale removal. In the demo the three blocks
cost 960, 1016 and 1024 parameters respectively: GRN is two vectors of length
$4d$, and layer scale is one of length $d$.

### Parameter counts

At $C_{\text{in}}=3$, $d=8$, hidden $4d=32$, $k=7$ and $s=4$:

| variant | stem | block | total |
|:--|--:|--:|--:|
| `convnext` | 392 | 960 | 1352 |
| `convnext_v2` | 392 | 1016 | 1408 |
| `grn` | 392 | 1024 | 1416 |
| `patchify_stem` | 392 | 0 | 392 |

The stem is $s^2C_{\text{in}}d+d$. A V1 block is
$(dk^2+d)+(dh+h)+(hd+d)+d$; V2 swaps the last $d$ for $2h$ of GRN parameters.
The block is dominated by the $7\times7$ depthwise kernel when $d$ is small and
by the expanded MLP when $d$ is large.

### Complexity

For an image with $N=HW$ pixels, $d$ channels, hidden $4d$ and kernel $k$:

- **Stem:** $O(Ns^2C_{\text{in}}d)$ and $O(Nd)$ output. The cost falls by
  $s^2$ because the output grid is smaller by that factor.
- **Depthwise:** $O(Ndk^2)$ time and $O(dk^2)$ parameters. For $k=7$ that is 49
  multiply-adds per output element, independent of the image size.
- **Channel MLP:** $O(Nd\cdot4d)=O(Nd^2)$, the dominant term at large $d$.
- **GRN:** $O(Nd)$ to pool, plus $O(d)$ to normalise; negligible next to the
  MLP but it introduces a global dependency that blocks naive parallelism over
  the batch's spatial axis.
- **Backward:** the same order, plus the overlap-add for the depthwise kernel
  and the two scalar reductions in (7).
- **The demo:** $8\times8$ images and $d=8$, so a forward pass is a handful of
  small array operations; the finite-difference loops and the 250-step fit
  dominate, and the file runs in well under a second.

## Code walkthrough

The stem is two reshapes and one transpose. Keeping the inverse next to it is
what makes the gradient a one-liner:

```python
    def _patchify(self, x: np.ndarray) -> np.ndarray:
        batch, height, width, channels = x.shape
        size = self.stride
        rows, columns = height // size, width // size
        patches = x.reshape(batch, rows, size, columns, size, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(batch, rows, columns, size * size * channels)
```

The depthwise convolution materialises its windows once and contracts them with
a per-channel kernel. The einsum names every axis, so the kernel indices `i`
and `j` cannot be confused with the spatial ones:

```python
        padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        windows = np.stack(
            [np.stack([padded[:, i:i + x.shape[1], j:j + x.shape[2], :]
                       for j in range(self.kernel)], axis=-1)
             for i in range(self.kernel)], axis=-2)
        out = np.einsum("bhwcij,cij->bhwc", windows, p["dw.W"])
        return out + p["dw.b"], {"windows": windows}
```

The block is the V1 composition with two optional insertions. Note that
`cache["project_input"]` is captured after the GRN call, because that is what
the project weight actually multiplies:

```python
        depth, dw_cache = self._depthwise(x)
        norm, norm_cache = _layer_norm(depth)
        pre = norm @ p["expand.W"] + p["expand.b"]
        hidden = _gelu(pre)
        cache = {"x": x, "dw": dw_cache, "norm": norm,
                 "norm_cache": norm_cache, "pre": pre, "hidden": hidden}
        if self.variant in ("convnext_v2", "grn"):
            hidden, grn_cache = self._grn(hidden)
            cache["grn"] = grn_cache
        cache["project_input"] = hidden
        projected = hidden @ p["project.W"] + p["project.b"]
        cache["projected"] = projected
        if self.variant in ("convnext", "grn"):
            projected = p["layer_scale"] * projected
        return x + projected, cache
```

GRN pools over both spatial axes with `keepdims=True`, so the result broadcasts
back over the image without an explicit reshape:

```python
    def _grn(self, x: np.ndarray):
        p = self.parameters
        norms = np.sqrt((x ** 2).sum(axis=(1, 2), keepdims=True))
        mean = norms.mean(axis=-1, keepdims=True)
        normalised = norms / (mean + GRN_EPS)
        out = x + p["grn.gamma"] * x * normalised + p["grn.beta"]
        return out, {"x": x, "norms": norms, "mean": mean,
                     "normalised": normalised}
```

Its backward pass is (7). The two reductions have different axes: `totals`
sums over space to get $A_c$, while `share` sums over channels to get the mean
term. Mixing those up is the bug the gradient check is there to catch:

```python
        denominator = mean + GRN_EPS
        totals = (gradient * x).sum(axis=(1, 2), keepdims=True)
        weighted = p["grn.gamma"] * totals
        share = (weighted * norms).sum(axis=-1, keepdims=True)
        bracket = (weighted / denominator
                   - share / denominator ** 2 / self.hidden)
        d_x = gradient * (1.0 + p["grn.gamma"] * normalised)
        d_x += bracket * x / np.maximum(norms, GRN_EPS)
        return d_x
```

The depthwise backward is an overlap-add: each kernel tap writes the gradient
of its shifted window back into a padded buffer, and the unpadding happens on
the way out:

```python
        pad = self.kernel // 2
        batch, height, width = gradient.shape[:3]
        d_padded = np.zeros((batch, height + 2 * pad, width + 2 * pad,
                             self.dim))
        for i in range(self.kernel):
            for j in range(self.kernel):
                d_padded[:, i:i + height, j:j + width, :] += (
                    gradient * p["dw.W"][None, None, None, :, i, j])
        return d_padded[:, pad:pad + height, pad:pad + width, :]
```

Only this tail is variant-specific at the top level: the residual path and the
depthwise path merge, then the stem's transpose sends the gradient back to the
image:

```python
        d_norm = d_pre @ p["expand.W"].T
        d_depth = _layer_norm_backward(d_norm, block["norm_cache"])
        d_stem = gradient + self._depthwise_backward(d_depth, block["dw"])
        return self._stem_backward(d_stem, cache["stem"])
```

## Failure modes

**Layer scale of zero kills the block.** The initialisation is $10^{-6}$, not
zero, so the gradient to the block still flows. Setting it to exactly zero
makes $Y=X$ and every parameter inside the block receives a zero gradient
through the residual; the block never turns on. The demo's gradient check sets
the scale to one precisely so the test cannot pass trivially.

**GRN's gamma at zero hides a wrong backward pass.** With $\gamma=0$ the
normalisation path contributes nothing to the input gradient, and a broken (7)
still passes a gradient check. The demo sets gamma to random values before
checking, which is what exposed an earlier version that summed the mean term
over channels but not over space.

**GRN is global, so it breaks patch independence.** Two images in a batch do
not interact, but all spatial positions of one image do. That is intentional —
it is a response normalisation — yet it means GRN cannot be applied to a canvas
in tiles without a different result per tile.

**The depthwise convolution is not a spatial bottleneck at large kernels.**
$k=7$ costs 49 multiply-adds per element. The original ConvNeXt chose it to
match a $7\times7$ attention window; a $3\times3$ kernel is cheaper and changes
the receptive field. The demo's radius of 3 follows directly from the
constructor's kernel.

**Padding loses the border statistics.** Zero padding at the edges means border
pixels see a different effective window from interior ones. For a single block
this is a small bias; for a deep stack it compounds unless the padding is
reflected or the border is handled explicitly.

**The stem trades resolution for channels.** A stride of 4 throws away 15 of
every 16 pixels. That is fine for classification and bad for dense prediction
unless the network later upsamples; the stem is a design choice, not a neutral
preprocessing step.

**Patch order is fixed.** The stem's reshape defines the token order, and the
block's spatial convolutions assume a grid. A permuted image is a different
image; nothing here is permutation-invariant, which is the opposite of the
graph and set blocks.

**Channel-last is assumed everywhere.** LayerNorm, GRN and the einsum indices
all reduce over the last axis. A channels-first tensor would normalise over
width and still run, producing a plausible but wrong model.

**The V2 and `grn` variants share the block code path.** A change to one can
silently change the other. The variants are selected by membership tests in two
places — the parameter construction and the forward pass — and they must agree;
the demo's parameter counts would catch a mismatch.

**The backward pass assumes a gradient with the stem's output shape.** It
cannot check that the upstream gradient corresponds to the right loss, so a
mis-broadcast gradient produces silent nonsense rather than an error.

## Applications

ConvNeXt is a strong image-classification and detection backbone, competitive
with vision transformers at similar parameter counts while keeping the
convolutional inductive bias. The depthwise $7\times7$ kernel plus inverted
bottleneck is the piece that transfers best: it appears in segmentation heads,
diffusion-model U-Nets and efficient video models.

GRN was introduced for self-supervised pre-training and is used as a cheap
alternative to LayerNorm in convolutional blocks, particularly when the
channel statistics drift between layers. Layer scale is now common well beyond
ConvNeXt: it appears in transformer blocks as a per-channel residual gate and
is the simplest way to make a deep residual stack start near the identity.

## References

- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S. (2022).
  *A ConvNet for the 2020s.* CVPR. The V1 block, layer scale and the
  patchify stem, equations (1)–(4).
- Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., Xie, S.
  (2023). *ConvNeXt V2: co-designing and scaling ConvNets with masked
  autoencoders.* CVPR. GRN and the removal of layer scale, equations (5)–(8).
- Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G., McAuley, J. (2021).
  *ReZero is all you need: fast convergence at large depth.* UAI. The residual
  scaling idea behind layer scale.
- Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H. (2021).
  *Going deeper with image transformers.* ICCV. Layer scale in CaiT, the
  transformer-side origin of the same trick.
