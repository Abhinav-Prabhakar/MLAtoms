# Convolution — one engine, ten variants, and its exact adjoint

`conv.py` · Neural blocks · Tier 1 · 10 variants

## Intuition

A convolution is a dot product applied at every position of a signal, with the
same weights reused at each position. That weight sharing is the whole idea.
It buys two things at once: the parameter count stops depending on the input
length, and the operation commutes with translation, so a feature learned at
one offset is recognised at every other offset.

Everything in this file is one scan. The kernel slides over the input, a
kernel-sized window is flattened to a vector, and that vector is multiplied by
a matrix of flattened kernels. The variants are switches on that scan:

- how many spatial axes are being scanned — one, two or three;
- how the kernel steps over them — `stride`, `dilation`, and where it is
  allowed to read past the border;
- how the channels are wired — dense, grouped, depthwise, or depthwise then
  pointwise;
- whether the scan runs forwards or backwards — `transposed` is the exact
  adjoint of the forward scan, and it is what the backward pass calls.

The demo establishes each switch with a property rather than a picture. A delta
into a kernel returns the kernel's taps. A box kernel reproduces
`numpy.convolve`. The transpose satisfies
$\langle \mathrm{conv}(x), y\rangle = \langle x, \mathrm{conv}^{\top}(y)\rangle$
to machine precision — both sides came out at $15.220855$. Grouped convolutions
equal concatenated smaller ones. Causal padding provably cannot see the future:
perturbing steps 20 onward leaves every earlier output bit-identical, and the
change first appears exactly at step 20. Circular padding commutes with
rolling. And the hand-written backward pass matches finite differences on all
21 probes to $1.49\times10^{-9}$.

## Variants

| variant | what it changes | kernel shape |
|:--|:--|:--|
| `conv1d` | one spatial axis | `(out, in, k)` |
| `conv2d` | two spatial axes | `(out, in, kh, kw)` |
| `conv3d` | three spatial axes | `(out, in, kd, kh, kw)` |
| `dilated` | spaces kernel taps apart | same as the rank |
| `transposed` | runs the adjoint scan | `(out, in, *k)` |
| `depthwise` | one kernel per channel | `(in, 1, *k)` |
| `separable` | depthwise, then pointwise | a pair of the two |
| `group` | independent channel blocks | `(out, in/g, *k)` |
| `causal` | pads only the past | `(out, in, k)` |
| `circular_pad` | wraps instead of zeroing | same as the rank |

**`conv1d`, `conv2d`, `conv3d`** are the same function with a rank check. The
engine already works in $n$ dimensions, so these exist to make the caller's
intent checkable: a 2-D kernel handed to `conv1d` is a bug, not a broadcast.

**`dilated`** multiplies the tap spacing by $d$. A width-3 kernel with
$d=2$ sees five input positions and still has three weights. It refuses
$d=1$, because that is the plain case and silently accepting it hides a
configuration mistake.

**`transposed`** is not an inverse. It is the adjoint of the valid forward
convolution: it scatters each output value back onto the inputs that produced
it. That is exactly the operation the backward pass needs, and the demo checks
the defining inner-product identity.

**`depthwise`** gives every input channel its own kernel and no mixing. It is
the grouped case with $G=C_{\mathrm{in}}$, and it refuses a kernel whose output
count does not equal the channel count.

**`separable`** is depthwise followed by a $1\times1$ convolution, which is the
MobileNet block. At the demo's shapes a dense layer needs $5\cdot6\cdot3=90$
weights and the split form needs $6\cdot3 + 5\cdot6=48$, or $0.53$ of the cost.

**`group`** splits the channels into $G$ blocks that never talk to each other.
The demo verifies the definition literally: three groups of two channels equal
three separate two-channel convolutions, concatenated.

**`causal`** pads $d(k-1)$ zeros on the left and none on the right, so the
output at step $t$ reads only inputs at steps $\le t$. It is one-dimensional by
construction, since "the past" is not a meaningful direction in an image.

**`circular_pad`** wraps the border, which makes the scan commute with cyclic
rolls. That is the right choice for genuinely periodic signals — angles, phase,
a torus — and the wrong one for an image, where the top edge would be blended
into the bottom.

## Derivation

### The scan

For input $x\in\mathbb R^{B\times n_1\times\cdots\times n_D\times C}$, kernel
$w\in\mathbb R^{O\times C\times k_1\times\cdots\times k_D}$, stride $s$ and
dilation $d$,

$$
y_{b,o,\mathbf{p}}
=\sum_{c}\sum_{\mathbf{k}}
w_{o,c,\mathbf{k}}\;
x_{b,\,c,\,\mathbf{p}\odot s+\mathbf{k}\odot d}+b_o .
\qquad (1)
$$

The sum over $\mathbf k$ runs over the kernel's taps and $\odot$ is
elementwise. This is cross-correlation, not the textbook convolution with a
flipped kernel. Every deep learning framework calls it convolution anyway, and
the distinction only matters when you compare against a signal-processing
library or read a proof about commutativity.

### Output size

With $p_{\mathrm b}$ zeros before and $p_{\mathrm a}$ after on some axis,

$$
n_{\mathrm{out}}
=\left\lfloor
\frac{n+p_{\mathrm b}+p_{\mathrm a}-d(k-1)-1}{s}
\right\rfloor+1 .
\qquad (2)
$$

The quantity $d(k-1)$ is the kernel's *reach*, the distance from its first tap
to its last. The demo's nine-by-nine input with a $3\times3$ kernel at
$d=2$ has reach four, so the valid output is $9-4=5$ wide, and the same
padding prepends two and appends two to keep the length at nine.

### Padding

The four modes are one table:

| mode | before | after | length at $s=1$ |
|:--|--:|--:|--:|
| `valid` | 0 | 0 | $n-d(k-1)$ |
| `same` | $\lfloor r/2\rfloor$ | $r-\lfloor r/2\rfloor$ | $n$ |
| `causal` | $r$ | 0 | $n$ |
| `full` | $r$ | $r$ | $n+r$ |

with $r=d(k-1)$. `same` splits the reach as evenly as it can, which is exact
for odd kernels and off by one for even ones — an even kernel cannot be
centred on a sample, so the output is shifted half a sample. That asymmetry is
a real property of the padding, not a rounding bug, and it is why vision
models almost always use odd kernels.

### The adjoint

Let $A$ be the linear map $x\mapsto \mathrm{conv}(x,w)$. Its adjoint
$A^{\top}$ is defined by

$$
\langle Ax,\,y\rangle=\langle x,\,A^{\top}y\rangle
\quad\text{for all }x,y .
\qquad (3)
$$

Written out, the adjoint scatters each output value onto the inputs it read:

$$
\bigl(A^{\top}y\bigr)_{c,\mathbf{q}}
=\sum_{o}\sum_{\mathbf{k}}
w_{o,c,\mathbf{k}}\;
y_{o,\,(\mathbf{q}-\mathbf{k}\odot d)/s},
\qquad (4)
$$

where a term is present only when every coordinate of
$(\mathbf q-\mathbf k\odot d)$ is divisible by $s$ and lands in range. The demo
computes both sides of (3) for random arrays and gets the same float twice.
This identity is the reason a convolution layer needs no separate backward
implementation: the gradient with respect to the input *is* the adjoint.

### Weight gradient

For upstream gradient $g=\partial L/\partial y$,

$$
\frac{\partial L}{\partial w_{o,c,\mathbf{k}}}
=\sum_{b}\sum_{\mathbf{p}}
g_{b,o,\mathbf{p}}\;
x_{b,\,c,\,\mathbf{p}\odot s+\mathbf{k}\odot d},
\qquad (5)
$$

which is the forward scan again, with the upstream gradient in place of the
kernel and the input's windows in place of the input. The implementation
materialises the windows once and contracts them against $g$ with a single
`tensordot`.

### Groups

Split the $C$ input channels and $O$ output channels into $G$ blocks of
$C/G$ and $O/G$. Each output block reads only its own input block, so

$$
P_{\mathrm{group}}
=G\cdot\frac{O}{G}\cdot\frac{C}{G}\cdot\prod_i k_i
=\frac{OC}{G}\prod_i k_i .
\qquad (6)
$$

$G=1$ is the dense case and $G=C=O$ is depthwise. The demo's three groups of
two channels over six inputs and six outputs give
$6\cdot2\cdot3=36$ weights against $6\cdot6\cdot3=108$ dense, and match the
three independent scans exactly.

### Separable

Depthwise convolution followed by a pointwise one is

$$
P_{\mathrm{sep}}
=C\prod_i k_i \;+\; CO .
\qquad (7)
$$

The first term mixes space, the second mixes channels, and the factorisation
assumes the two are separable — an assumption about the data, not a theorem.
At kernel $3$, $C=6$, $O=5$ that is $18+30=48$ against $90$. The saving grows
with the kernel: at $k=7$ a dense layer costs $49CO$ and the split one
$7C+CO$, which tends to $1/7$ of the cost once $C$ is much larger than $k$.

### Dilation and receptive field

A stack of $L$ convolutions with kernel $k_\ell$ and dilation $d_\ell$ sees

$$
R_L = 1+\sum_{\ell=1}^{L} d_\ell\,(k_\ell-1)
\qquad (8)
$$

input positions. With $k=2$ and $d=1,2,4,\dots,2^{L-1}$ the reach is
$2^{L}$, so $L$ layers cover $2^L$ steps with $2L$ weights per channel. That
exponential reach at linear parameter cost is why dilated causal convolutions
replaced recurrent networks for long audio.

A dilated kernel with $k$ taps has the same support as a dense kernel of width
$d(k-1)+1$ with zeros between the taps. It is a *sparse* large kernel, not a
compressed one: the missing positions are not interpolated, they are skipped.

### Cyclic and causal structure

Circular padding makes the input an element of the group
$\mathbb Z_{n_1}\times\cdots\times\mathbb Z_{n_D}$. The scan then commutes
with translation on that group,

$$
T_\tau\bigl(\mathrm{conv}(x)\bigr)
=\mathrm{conv}\bigl(T_\tau x\bigr),
\qquad (9)
$$

which the demo checks by rolling the input four steps and confirming the
output rolls four steps with no seam. Zero padding has no such identity: a
shifted input produces a differently-clipped output.

Causal padding gives the one-sided analogue. With
$p_{\mathrm b}=d(k-1)$ and $p_{\mathrm a}=0$, the output at $t$ satisfies

$$
y_t = f\bigl(x_{t-d(k-1)},\dots,x_t\bigr),
\qquad (10)
$$

so no future input can influence it. Stacking such layers preserves the
property, which is what makes a temporal convolutional network usable for
forecasting at all.

### Transposed convolution and stride

The adjoint of a stride-$s$ scan is a *fractionally strided* convolution: each
output sample of the adjoint corresponds to a position of the original grid,
and the $s-1$ positions in between receive nothing from that sample. In
practice this is implemented by inserting $s-1$ zeros between the inputs and
then running a stride-1 scan — the same scatter as (4), written differently.
It is an upsampling operator with learned weights, not an inverse: applying
$\mathrm{conv}^{\top}$ after $\mathrm{conv}$ smooths and mixes, and does not
recover the original signal.

## Code walkthrough

Padding is a single table driven by the kernel's reach:

```python
    reach = dilation * (kernel - 1)
    if mode == "valid":
        return 0, 0
    if mode == "same":
        before = reach // 2
        return before, reach - before
    if mode == "causal":
        return reach, 0
    if mode == "full":
        return reach, reach
```

The floor division in `same` is the even-kernel asymmetry from the table
above. Writing `reach // 2` on both sides would lose a column for odd reach;
the second term is `reach - before` for exactly that reason.

The window extraction is the honest, slow version of what a GPU does:

```python
    for index in np.ndindex(*shape):
        window = tuple(slice(start * stride[axis],
                             start * stride[axis]
                             + dilation[axis] * (kernel[axis] - 1) + 1,
                             dilation[axis])
                       for axis, start in enumerate(index))
        windows[(slice(None), *index)] = np.moveaxis(
            x[(slice(None), *window)], -1, 1)
```

A slice with step `dilation` is the dilated tap pattern, and `stride` multiplies
the starting offset. The `moveaxis` puts channels before the kernel axes so
that a window's memory layout matches the flattened kernel. The loop over
output positions is why this file is a reference implementation and not a
fast one.

The scan itself is one contraction over three of the four index groups:

```python
    flat = windows.reshape(*windows.shape[:dims + 1], channels, width)
    blocks = flat.reshape(*flat.shape[:dims + 1], groups, channels // groups,
                          width)
    kernels = weight.reshape(groups, weight.shape[0] // groups,
                             channels // groups, width)
    out = np.einsum("...gck,gock->...go", blocks, kernels)
    return out.reshape(*out.shape[:dims + 1], weight.shape[0])
```

Grouping is a reshape, not a loop: both the windows and the kernels are split
along a new `g` axis, and `einsum` refuses to sum across it. That is why the
grouped, depthwise and dense cases are the same code path with a different
`groups` value. The `...` covers batch and every output position at once.

The adjoint is a scatter-add:

```python
        piece = np.einsum("bo,oc...->bc...", y[(slice(None), *index)], weight)
        out[(slice(None), *window)] += piece.transpose(0, *spread, 1)
```

`piece` is the contribution of one output position to the input channels,
placed at the window's offsets. The overlapping windows mean many positions
add into the same input cell, which is why this is `+=` and why the buffer is
zero-initialised. Getting the transpose axes wrong here is the classic bug: the
output is `(batch, out_channels)`, the kernel is `(out, in, *k)`, and the
result must be `(batch, in_channels, *k_window)`.

The weight gradient reuses the same windows:

```python
    windows = _patches(x, kernel, stride, dilation)
    flat = windows.reshape(*windows.shape[:dims + 1], x.shape[-1], -1)
    axes = list(range(dims + 1))
    d_weight = np.tensordot(flat, upstream, axes=(axes, axes))
    d_weight = d_weight.transpose(2, 0, 1)
    d_input = transpose(upstream, weight, stride=stride, dilation=dilation)
```

`tensordot` sums over batch and every output position, leaving
`(channels, kernel_volume, out_channels)`, and the final `transpose` puts it
back into the kernel's layout. `d_input` is literally the call to `transpose`:
equation (4) is the backward pass, so there is nothing to re-derive.

The depthwise guard is where the variant earns its name:

```python
        channels = x.shape[-1]
        if np.shape(weight)[0] != channels or np.shape(weight)[1] != 1:
            raise ValueError("depthwise needs one kernel per input channel")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=channels, padding=padding)
```

One kernel per channel means an output-channel count equal to the input-channel
count and an input-channel count of one inside each group. Both conditions are
checked, because `groups=channels` alone would accept a mismatched kernel and
then reshape it into something silently wrong.

Separable is two calls, and this is the whole implementation:

```python
        depth, point = weight
        mixed = convolve(x, depth, stride=stride, dilation=dilation,
                         groups=x.shape[-1], padding=padding)
        return convolve(mixed, point, padding="same")
```

The unpacking doubles as the type check: a single array cannot be unpacked into
two, so passing a dense kernel to `separable` fails before any arithmetic.
Stride and dilation belong to the depthwise stage; the pointwise stage is a
$1\times1$ scan and has no space to stride through.

Causal padding is one line in the padding table, but it needs a rank guard:

```python
    if variant == "causal":
        if x.ndim != 3:
            raise ValueError("causal is a one-dimensional padding mode")
        return convolve(x, weight, stride=stride, dilation=dilation,
                        groups=groups, padding="causal")
```

Without the guard, a 2-D call would silently pad only the first spatial axis
and produce a filter that sees half the future. The demo's check is behavioural
rather than structural: it adds $100$ to every step from 20 onward and asserts
that outputs 0 through 19 are *bit-identical*, then that the first difference
lands at step 20.

Circular padding is a flag on the padding helper:

```python
def _pad(x: np.ndarray, pads, wrap: bool) -> np.ndarray:
    widths = [(0, 0), *pads, (0, 0)]
    if wrap:
        return np.pad(x, widths, mode="wrap")
    return np.pad(x, widths)
```

`mode="wrap"` repeats the signal cyclically, which is exactly the group
structure of (9). The batch and channel axes get zero-width pads so the wrap
applies only to space; using `np.pad(..., mode="wrap")` without the explicit
widths would wrap the batch dimension too.

## Complexity

For batch $B$, $D$ spatial axes, $C$ input channels, $O$ output channels,
kernel volume $K=\prod_i k_i$ and output volume $N=\prod_i n_{\mathrm{out},i}$:

- **Arithmetic:** $O(B\,N\,O\,C\,K/G)$ multiply-adds, the same count for the
  backward pass. This is the number a hardware spec quotes.
- **Window materialisation:** $O(B\,N\,C\,K)$ floats of temporary storage. For
  a $3\times3$ kernel and 256 channels that is nine times the input, which is
  why real implementations tile the scan instead of materialising it.
- **Python overhead:** the demo loops over $N$ output positions, so the wall
  clock is dominated by the interpreter rather than the arithmetic. A
  vectorised `as_strided` view removes the loop but not the memory cost.
- **Grouped and depthwise:** the arithmetic divides by $G$, but the
  materialised windows do not, so the memory saving is smaller than the FLOP
  saving. This is the usual reason depthwise layers are memory-bound on
  accelerators.
- **Parameter count:** $OCK/G + O$ with bias, from (6); the demo's separable
  example is $48$ against $90$.
- **Receptive field:** $1+\sum_\ell d_\ell(k_\ell-1)$ from (8), independent of
  channel count.

## Failure modes

**`same` is not centred for even kernels.** With $k=2$ the reach is one, so
`before=0` and `after=1`: the output is shifted half a sample relative to the
input. Stacking several such layers drifts the feature map, and the drift is
easy to mistake for a data alignment bug.

**Dilation wider than the input is an error, not a zero.** The engine raises
when the padded input is shorter than the kernel's reach. Some frameworks
silently return an empty or zero output, which turns a configuration mistake
into a model that trains to chance.

**Stride destroys shift equivariance.** A stride-$s$ scan samples the input at
$s$ phases, so a one-pixel shift of the input is not a one-pixel shift of the
output unless the shift is a multiple of $s$. This is aliasing, and it is why
detection models often use stride 1 in the stem and pool afterwards.

**The transpose is not an inverse.** `transpose(convolve(x, w), w)` is a
correlation of a correlation: it smooths and mixes channels. It does not
recover `x`, and a decoder built on the assumption that it does will produce
blurry reconstructions that no amount of training fixes.

**Circular padding wraps the wrong thing.** On a periodic signal it is exact;
on an image it blends the top row into the bottom. The demo's roll identity
holds precisely because the data is treated as cyclic, so it is a check of the
implementation, not evidence that wrapping is right for your problem.

**Zero padding creates a border artefact.** The output near an edge is computed
from fewer real inputs than the interior, so the effective function changes
with position. With `full` padding the artefact moves outside the original
support, which is why `full` is the right mode for analysis and `same` for
stacks.

**Grouped convolutions permute channels.** Group $g$ of the output reads only
group $g$ of the input, so a channel permutation applied to the input is not
undone by the layer. Weight initialisation and channel order therefore matter
more than in the dense case, and a shuffled-channel bug can survive several
layers before it shows.

**The demo's kernel has no bias.** `convolve` is the linear map only. A real
layer adds $b_o$ per output channel, which is $O$ parameters and one broadcast;
leaving it out keeps the adjoint identity clean, since a bias would not be
part of the linear operator.

## Applications

Convolution is the vision backbone. Every image classifier from LeNet through
ResNet is a stack of these scans with a nonlinearity and a normalisation
between them, and the variants here are the standard blocks: $3\times3$ dense
for accuracy, depthwise separable for mobile budgets, grouped for ResNeXt-style
width, dilated for segmentation without losing resolution.

The causal form is the time-series workhorse. WaveNet and the temporal
convolutional network replace recurrence with a stack of dilated causal
convolutions, which parallelise over time and have a fixed receptive field
instead of a vanishing gradient.

The transposed form is the decoder half of every autoencoder, GAN generator and
segmentation network. It is also the operation inside gradient-based
attribution methods, where the adjoint of the forward pass is exactly what is
needed to map a gradient at the output back onto the input.

Circular padding shows up wherever the signal really is periodic: angle
regression, phase estimation, crystallography, and any model over a torus.
Separable and depthwise convolutions are the default on phones, and grouped
convolutions are how mixture-of-experts ideas first entered vision.

## References

- Fukushima, K. (1980). *Neocognitron.* Biological Cybernetics. The
  local-connectivity and weight-sharing architecture.
- LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard, R., Hubbard, W.,
  Jackel, L. (1989). *Backpropagation applied to handwritten zip code
  recognition.* Neural Computation. The gradient of (5) through a shared
  kernel.
- Krizhevsky, A., Sutskever, I., Hinton, G. E. (2012). *ImageNet classification
  with deep convolutional neural networks.* NeurIPS. The modern dense stack.
- Sifre, L., Mallat, S. (2014). *Rigid-motion scattering for image
  classification.* The depthwise-then-pointwise factorisation of (7).
- Yu, F., Koltun, V. (2016). *Multi-scale context aggregation by dilated
  convolutions.* ICLR. Dilation and the receptive field of (8).
- van den Oord, A. et al. (2016). *WaveNet.* arXiv:1609.03499. Dilated causal
  convolutions for audio.
- Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K. (2017). *Aggregated
  residual transformations.* CVPR. Grouped convolutions at scale.
- Dumoulin, V., Visin, F. (2016). *A guide to convolution arithmetic for deep
  learning.* arXiv:1603.07285. The output-size and padding tables of (2).
- Bai, S., Kolter, J. Z., Koltun, V. (2018). *An empirical evaluation of
  generic convolutional and recurrent networks.* arXiv:1803.01271. The causal
  stacking argument.
