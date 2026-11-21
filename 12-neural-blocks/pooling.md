# Pooling — nine ways to throw away spatial resolution

`pooling.py` · Neural blocks · Tier 1 · 9 variants

## Intuition

A pooling layer replaces every window of a feature map with one number. It is
the cheapest way to make a representation smaller, more tolerant of small
shifts, and — in the global forms — independent of the input's spatial size.

The variants differ only in *which number*. The maximum keeps the strongest
response in the window and throws the rest away. The mean keeps a summary of
everything and dilutes the peak. The generalised mean sits between them and is
the same function at $p=1$ and, in the limit, at $p=\infty$. Attention pooling
learns the weights instead of fixing them. Stochastic pooling samples a
location, which is a regulariser with a strange-looking but simple
expectation. The three global variants are the same operations with the window
covering the whole map, and `adaptive` is the mean or max on a window grid
whose sizes are computed from the input rather than fixed.

The demo measures one property per family. Max and mean agree with brute-force
windows on a $2\times5\times6\times3$ input. `adaptive` turns seven samples
into three with the intervals $[0,3)$, $[2,5)$, $[4,7)$ — overlapping at every
seam — and seven into seven as the identity. Attention pooling, trained by
gradient descent on a single scalar, moves its output from $0.924$ to $4.0000$,
the location of the spike, and the learned weight sits on position 6. The
generalised mean is monotone in $p$: geometric $0.7304$, mean $0.8713$, rms
$0.9889$, and $p=64$ reaches $1.6750$ against a true maximum of $1.7433$.
Stochastic pooling's 4000 draws land on the four inputs with shares
$[0.105, 0.240, 0.455, 0.200]$ against exact probabilities
$[0.100, 0.250, 0.450, 0.200]$. And the gradient routes exactly where the
forward pass read: max sends the whole upstream mass to the argmax, mean splits
it evenly, and both conserve the total.

## Variants

| variant | output | window | learned |
|:--|:--|:--|:--:|
| `max` | window maximum | fixed | no |
| `avg` | window mean | fixed | no |
| `global_max` | one value per channel | whole map | no |
| `global_avg` | one value per channel | whole map | no |
| `adaptive` | fixed spatial size | floor/ceil grid | no |
| `attention_pool` | weighted average | whole map | yes |
| `gem` | generalised mean | fixed | power |
| `lp_pool` | unnormalised power sum | fixed | power |
| `stochastic` | one sampled activation | fixed | power |

**`max`** is the default in vision. It keeps the strongest activation in each
window, so a feature detector fires if its pattern appears anywhere in the
window. The cost is that the gradient goes only to the argmax: a max-pooled
layer is a sparse router, not an averaging one.

**`avg`** is the older choice, and it is a box filter divided by its area. It
passes gradient to every input in the window, which makes it better behaved in
very deep networks, at the price of diluting sparse, high-contrast features.

**`global_max` and `global_avg`** pool over every spatial position and return
one value per channel. Global average pooling is what replaced the fully
connected classifier in modern convnets: it has no parameters and no fixed
input size.

**`adaptive`** takes the *output* size and derives the windows. It is how a
network accepts images of different resolutions while keeping a fixed-size
feature vector.

**`attention_pool`** learns a score per position and takes a softmax-weighted
average. It is the set-pooling layer of multiple-instance learning: with a
sharp enough temperature it approaches max pooling, and with a flat one it
approaches the mean.

**`gem`** is $(\frac1K\sum x_i^p)^{1/p}$. At $p=1$ it is exactly the mean, at
$p\to\infty$ it is the max, at $p\to0$ it is the geometric mean, and it is
monotone in $p$ throughout. It interpolates the two classical choices with one
hyperparameter.

**`lp_pool`** is $(\sum x_i^p)^{1/p}$ — the same root without the $1/K$. It
differs from `gem` by exactly $K^{1/p}$, which the demo checks numerically.

**`stochastic`** samples one location per window with probability proportional
to $x_i^p$. It is a regulariser in the style of dropout: at training time it
injects noise, and at test time the expectation is used.

## Derivation

### Windows

Write the window at output position $\mathbf o$ as

$$
W_{\mathbf o}
=\bigl\{\mathbf o\odot s+\mathbf k \;:\; 0\le k_i<k_i\bigr\},
\qquad (1)
$$

with stride $s$ and window shape $\mathbf k$. Every non-adaptive variant is a
reduction over $W_{\mathbf o}$:

$$
y_{b,\mathbf o,c}
=\rho\bigl(x_{b,\mathbf p,c}:\mathbf p\in W_{\mathbf o}\bigr).
\qquad (2)
$$

The output size is $\lfloor (n_i-k_i)/s_i\rfloor+1$ per axis. The demo's
$5\times6$ map with a $2\times2$ window at stride 2 gives $2\times3$; the last
row and column are dropped because the window does not fit, which is the
standard behaviour and a frequent source of off-by-one bugs.

### Max and mean

$$
\rho_{\max}=\max_{\mathbf p\in W}x_{\mathbf p},
\qquad
\rho_{\mathrm{avg}}=\frac{1}{K}\sum_{\mathbf p\in W}x_{\mathbf p},
\qquad K=\prod_i k_i .
\qquad (3)
$$

The mean is a convolution with a box kernel scaled by $1/K$,

$$
\rho_{\mathrm{avg}}(x)
=\frac{1}{K}\,\mathrm{conv}\bigl(x,\mathbf 1_{\mathbf k}\bigr),
\qquad (4)
$$

so it inherits the border behaviour of whatever padding the convolution uses.
The max has no such linear form; it is a nonlinear filter, and that is the
source of both its tolerance to shifts and its sparse gradient.

### Global pooling

$$
\rho_{\mathrm{gmax}}(x)_c=\max_{b,\mathbf p}x_{b,\mathbf p,c},
\qquad
\rho_{\mathrm{gavg}}(x)_c=\frac{1}{N}\sum_{b,\mathbf p}x_{b,\mathbf p,c},
\qquad (5)
$$

with $N$ the number of spatial positions. The batch axis is *not* reduced: the
demo returns shape $(2,3)$ from a batch of two with three channels.

### Adaptive windows

For input length $n$ and target length $m$,

$$
a_i=\left\lfloor\frac{i\,n}{m}\right\rfloor,
\qquad
b_i=\left\lceil\frac{(i+1)\,n}{m}\right\rceil,
\qquad
y_i=\rho\bigl(x_{a_i:b_i}\bigr).
\qquad (6)
$$

For $n=7$, $m=3$ this gives $(0,3)$, $(2,5)$, $(4,7)$: the windows overlap by
one and every input is covered. For $n=m$ it gives $(i,i+1)$, the identity.
The overlap is the price of exact coverage with a fixed output size; it also
means neighbouring outputs are correlated, which is why adaptive pooling is a
poor choice when the outputs feed an independence assumption.

### Attention pooling

With a learned vector $w$ and temperature $\tau$,

$$
u_p=\frac{w^{\top}x_p}{\tau},
\qquad
\alpha_p=\frac{e^{u_p}}{\sum_q e^{u_q}},
\qquad
y=\sum_p \alpha_p x_p .
\qquad (7)
$$

The temperature sets how sharply the weights concentrate. The demo's table has
values $4.0$ at position 6, $1.0$ at 3, $0.5$ at 8 and zeros elsewhere. At
$\tau=0.05$ and a learned $w=0.01$ the initial output is $0.924$ — a weighted
average pulled toward the spike but still mixing in the zeros. Gradient descent
on $w$ against the target $4.0$ raises $w$, sharpens the softmax, and drives
the output to the peak. The temperature is not learned here, but it is the
hyperparameter that decides whether the layer behaves like a mean or a max.

### Generalised mean

$$
M_p=\left(\frac{1}{K}\sum_i x_i^p\right)^{1/p}.
\qquad (8)
$$

$p=1$ is the arithmetic mean. As $p\to\infty$ the largest term dominates and
$M_p\to\max_i x_i$; as $p\to-\infty$ it tends to the minimum. At $p=0$ the
limit is the geometric mean,

$$
M_0=\left(\prod_i x_i\right)^{1/K}
=\exp\left(\frac{1}{K}\sum_i\log x_i\right),
\qquad (9)
$$

which is why the implementation special-cases $p=0$ rather than computing
$1/0$. All of this needs $x_i>0$ for the log form and $x_i\ge0$ in general: a
negative entry with a fractional power is undefined, and the code raises rather
than returning a NaN that would propagate silently.

Monotonicity in $p$ follows from Jensen's inequality applied to the convex
function $t\mapsto t^p$; the demo asserts it on the concrete values.

### LP pooling

$$
L_p=\left(\sum_i x_i^p\right)^{1/p}
=K^{1/p}M_p .
\qquad (10)
$$

The factor $K^{1/p}$ is the only difference. At $p=2$ and $K=16$ it is a factor
of $4$, which the demo multiplies back out and compares to machine precision.
PyTorch's `LPPool` exposes this unnormalised form because it composes with
other scale factors; the generalised mean is the version with a meaningful
absolute scale.

### Stochastic pooling

Choose an index $i$ with probability

$$
P(i)=\frac{x_i^p}{\sum_j x_j^p},
\qquad
\mathbb E[x_I]=\frac{\sum_i x_i^{p+1}}{\sum_i x_i^p}.
\qquad (11)
$$

The expectation is a ratio of consecutive moments, not the mean: for the demo's
four values with $p=1$ it is
$(0.2^2+0.5^2+0.9^2+0.4^2)/(0.2+0.5+0.9+0.4)=1.26/2.0=0.63$, between the mean
$0.5$ and the max $0.9$. At inference one either uses this expectation or
averages many samples; using a single sample at test time is a bug that costs
accuracy quietly.

### Gradients

For max, the derivative is a selector; for mean, it is a constant:

$$
\frac{\partial L}{\partial x_i}
=g\cdot\mathbb 1\bigl[i=\arg\max_j x_j\bigr],
\qquad
\frac{\partial L}{\partial x_i}=\frac{g}{K}.
\qquad (12)
$$

Ties are the subtle case. The demo splits the gradient evenly among tied
maxima, which is one of several conventions; taking the first index is another.
Either is defensible, but the choice has to match between forward and backward,
or the gradient check fails only on the rare tied batch.

For the generalised mean, differentiating (8) gives

$$
\frac{\partial M_p}{\partial x_i}
=\frac{1}{K}\,M_p^{1-p}x_i^{p-1},
\qquad (13)
$$

which reduces to $1/K$ at $p=1$ as it must, and concentrates on the largest
entry as $p$ grows. Stochastic pooling has no pathwise derivative through the
sampled index; it is trained either with REINFORCE or, more commonly, treated
as a noise layer whose gradient is taken as if the chosen value were the mean.

## Code walkthrough

Every non-adaptive variant reads windows from one view:

```python
    dims = len(size)
    view = np.lib.stride_tricks.sliding_window_view(
        x, size, axis=tuple(range(1, dims + 1)))
    steps = (slice(None),) + tuple(slice(None, None, step) for step in stride)
    return view[steps]
```

`sliding_window_view` appends the window axes after the channel axis, so the
result is `(batch, *output, channels, *window)` and the reduction is over the
trailing `dims` axes. The view is read-only and copies nothing; the stride is
applied by subsampling the *output* axes, not by moving the window. Getting
that order wrong produces windows that overlap in the wrong direction, which
still runs and still pools — just not the windows you asked for.

The reduction is one branch per family:

```python
    if how == "gem":
        if np.any(windows < 0.0):
            raise ValueError("gem needs non-negative inputs")
        if power == 0.0:
            if np.any(windows <= 0.0):
                raise ValueError("the geometric mean needs positive inputs")
            return np.exp(np.log(windows).mean(axis=axes))
        moment = np.mean(windows ** power, axis=axes)
        return np.power(moment, 1.0 / power)
```

The guards are not decoration: `0.0 ** -1` is `inf`, `(-1.0) ** 0.5` is `nan`,
and `log(0)` is `-inf`. Each of those propagates through a whole network before
anyone notices, so the entry points refuse them.

Adaptive windows are computed per output index:

```python
def _bounds(size: int, out: int, index: int) -> tuple[int, int]:
    """The torch adaptive rule: floor at the start, ceiling at the end."""
    start = index * size // out
    end = -(-(index + 1) * size // out)
    return start, max(end, start + 1)
```

The `max(end, start + 1)` is defensive: for whole-number inputs the ceiling is
already at least `start + 1`, but stating the invariant where the window is
built beats relying on it. When the target is larger than the input the
windows repeat inputs, which is the defined behaviour of adaptive upsampling.

```python
    for index in np.ndindex(*target):
        window = tuple(slice(*_bounds(x.shape[1 + axis], target[axis], start))
                       for axis, start in enumerate(index))
        piece = x[(slice(None), *window)]
        out[(slice(None), *index)] = piece.max(axis=tuple(range(1, dims + 1))) \
            if reduce == "max" else piece.mean(axis=tuple(range(1, dims + 1)))
    return out
```

The `slice(*_bounds(...))` is (6) written directly. The reduction axes are
`1..dims`, which leaves batch and channels untouched, so the output keeps its
channel dimension even when the spatial size collapses to one.

Attention pooling is a softmax over flattened positions:

```python
    flat = x.reshape(x.shape[0], -1, x.shape[-1])
    scores = (flat @ weight) / temperature
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    pooled = np.einsum("bn,bnc->bc", weights, flat)
```

The shift by the row maximum is the standard stabiliser: softmax is invariant
to adding a constant to every score, and subtracting the largest keeps the
exponentials below one. Without it, a temperature of $0.01$ overflows within
one epoch.

Stochastic pooling builds a cumulative distribution and draws once per output:

```python
    cumulative = np.cumsum(weights / total, axis=-1)
    draw = rng.random(cumulative.shape[:-1] + (1,))
    chosen = np.minimum((cumulative < draw).sum(axis=-1), flat.shape[-1] - 1)
    return np.take_along_axis(flat, chosen[..., None], axis=-1)[..., 0]
```

`(cumulative < draw).sum()` is inverse-transform sampling without a Python
loop: the number of bins below the uniform draw is the chosen index. The
`minimum` guards against a floating-point draw of exactly 1.0, which would
otherwise index one past the end.

The gradient router is the forward pass read backwards:

```python
        if mode == "mean":
            gradient[(slice(None), *window)] += (flows / count)[(slice(None),
                                                                 *([None] * dims))]
            continue
        best = piece.max(axis=tuple(range(1, dims + 1)), keepdims=True)
        hits = piece == best
        hits = hits / hits.sum(axis=tuple(range(1, dims + 1)), keepdims=True)
        gradient[(slice(None), *window)] += \
            hits * flows[(slice(None), *([None] * dims))]
```

For mean, the upstream value is divided by the window size and broadcast over
the window. For max, `hits` is the indicator of (12), normalised so ties share
the gradient instead of each receiving the full amount. The window slices
overlap, so this must be `+=`, exactly as in the convolution adjoint.

The dispatcher is mostly shape bookkeeping:

```python
    if variant in ("global_max", "global_avg"):
        axes = tuple(range(1, dims + 1))
        return x.max(axis=axes) if variant == "global_max" else x.mean(axis=axes)
```

Global pooling is not the fixed-window path with a large window: it has no
window at all, so it must not pay the cost of materialising one. The same
distinction lets it accept any spatial size without a stride calculation.

## Complexity

For batch $B$, $D$ spatial axes, $N$ output positions, $K$ window volume and
$C$ channels:

- **Arithmetic:** $O(B\,N\,K\,C)$ comparisons for max and additions for mean,
  $O(B\,N\,K\,C)$ again for the backward pass. Pooling is cheaper than the
  convolution that precedes it by a factor of the kernel volume, because it has
  no channel mixing.
- **Memory:** the window view is free, but the reduction reads $B\,N\,K\,C$
  elements. A $2\times2$ window at stride 2 reads each input once; overlapping
  windows read some inputs several times.
- **Global pooling:** $O(B\,N\,C)$ and no temporary, since numpy reduces in
  place along the spatial axes.
- **Attention pooling:** $O(B\,N\,C)$ for the scores plus $O(B\,N)$ for the
  softmax and $O(B\,N\,C)$ for the weighted sum — linear in positions, not
  quadratic, because there is no pairwise attention.
- **Adaptive pooling:** $O(B\,N\,K\,C)$ with a Python loop over outputs, so the
  constant is worse than the fixed-window path.
- **Stochastic pooling:** one cumulative sum and one uniform draw per output,
  the same order as mean pooling plus $O(B\,N\,K\,C)$ for the normalisation.

## Failure modes

**Max pooling has a sparse, biased gradient.** Only the argmax of each window
receives gradient, so a unit that never wins its window never learns. In deep
networks this shows up as dead filters, and it is the reason average pooling
and strided convolution are common alternatives.

**Ties are a convention, not a fact.** Two equal maxima have no unique
gradient. Splitting evenly, taking the first, and taking all of them are all
used in practice; the demo splits. A mismatch between the forward selector and
the backward router produces a gradient that looks almost right and fails a
finite-difference check only on tied inputs.

**Adaptive pooling correlates neighbouring outputs.** The windows overlap by
construction, so the outputs are not independent samples of the input. Feeding
them to a model that assumes independence — a naive Bayes head, or a
per-position loss — silently overcounts the shared region.

**Pooling loses localisation.** A global average is invariant to where a
feature is; that is the point, and it is also why a classifier with global
pooling cannot report a bounding box. Global max keeps a little more, attention
pooling keeps the weights, and neither is a detector.

**Stochastic pooling is stochastic at test time too.** A single draw has high
variance, and using it at inference makes predictions non-reproducible. Either
average many draws or use the analytic expectation of (11); the demo's
frequency check is exactly the evidence that sampling matches the intended
distribution.

**Generalised mean is undefined off the positive orthant.** Negative
activations with a fractional power produce NaNs, and $p=0$ needs strictly
positive inputs. ReLU outputs are safe; post-normalisation features with
negative values are not, so GEM is usually placed after a ReLU.

**Large $p$ overflows.** $x^p$ for $p=64$ and $x=10$ is $10^{64}$, close to the
edge of float64; $p=1000$ overflows outright. Implementations either work in
logs or cap $p$; this one caps nothing and relies on the caller's scale.

**A window that does not divide the input drops the tail.** The output size
formula floors, so a $7$-wide map with a $2$-wide window at stride 2 returns
three windows and silently ignores the seventh column. Use `ceil_mode`
padding, or an adaptive target, if the tail matters.

**Pooling is not shift invariant, only shift tolerant.** A shift smaller than
the window changes the pooled values; a shift by a multiple of the stride
permutes the outputs. Calling max pooling "translation invariant" is the most
common overstatement in the literature.

## Applications

Pooling is the downsampling stage of every convolutional classifier. Max
pooling after a block of convolutions enlarges the receptive field and keeps
the strongest evidence; average pooling smooths and is gentler on the
gradient. Modern architectures often replace both with a strided convolution,
which learns the downsample instead of fixing it, but the pooling variants
remain the interpretable baseline.

Global average pooling is the standard classifier head. It removes the large
fully connected layer, makes the model accept arbitrary input sizes, and acts
as a strong regulariser — the ResNet family's parameter count depends on it.

Adaptive pooling is what lets one network serve several resolutions. It is
also the pooling used in detection and segmentation heads, where the output
grid is fixed by the task rather than by the input.

Attention pooling is the multiple-instance-learning layer: a bag of patches,
one label for the bag, and a learned weight per patch. It is used in
histopathology, whole-slide imaging, and any set-valued problem where the
important element is unknown.

Generalised mean pooling is the retrieval standard. GeM with a learned $p$
outperforms both max and average on image retrieval benchmarks, because the
right amount of peakiness depends on the dataset and can be learned.

Stochastic pooling is a training-time regulariser with a long tail of use in
small vision models. It is also a clean example of a layer whose expectation
is tractable but whose samples are not differentiable, which is the setting
that motivates reparameterisation tricks elsewhere.

## References

- Fukushima, K. (1980). *Neocognitron.* Biological Cybernetics. The
  max-pooling hierarchy.
- LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-based
  learning applied to document recognition.* Proceedings of the IEEE. Average
  subsampling in LeNet.
- Boureau, Y.-L., Ponce, J., LeCun, Y. (2010). *A theoretical analysis of
  feature pooling in visual recognition.* CVPR. Why max pooling is tolerant to
  shifts and why its gradient is sparse.
- Zeiler, M. D., Fergus, R. (2013). *Stochastic pooling for regularization of
  deep convolutional neural networks.* ICLR. Equation (11).
- Lin, M., Chen, Q., Yan, S. (2013). *Network in network.* arXiv:1312.4400.
  Global average pooling as a classifier head.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep residual learning for image
  recognition.* CVPR. Global average pooling at scale.
- Radenović, F., Tolias, G., Chum, O. (2018). *Fine-tuning CNN image retrieval
  with no human annotation.* TPAMI. GeM pooling and the learned exponent.
- Ilse, M., Tomczak, J. M., Welling, M. (2018). *Attention-based deep multiple
  instance learning.* ICML. Equation (7) as a set-pooling layer.
