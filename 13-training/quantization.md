# Quantisation — ten ways to store a weight in fewer bits

`quantization.py` · Training · Tier 2 · 10 variants

## Intuition

Quantisation replaces a float by an integer code and a scale. That is the whole
idea, and everything hard about it is in three follow-up questions: how wide is
the grid, how many values share a grid, and how do you choose the grid once the
weights are already trained.

The grid is either symmetric about zero (a sign bit and a magnitude) or
asymmetric with a zero point (so the codes cover an off-centre range exactly).
The granularity is one scale for the whole tensor, one per output channel, or
one per small group of input channels. The procedure is what changes at the low
end: post-training quantisation rounds whatever the trained weights are, QAT
trains through the rounding with a straight-through estimator, GPTQ chooses
each rounding while compensating the weights it has not quantised yet, and AWQ
rescales input channels so the ones that matter survive the grid.

This atom is also about a result that surprises people: 4-bit *weights* are
almost free. On the demo's model the 4-bit recipes lose 0.000 held-out accuracy
against fp32 and use 4.8× fewer bytes. It is 4-bit *activations* that hurt —
`int4` drops to 0.975 — and at 2 bits the procedure finally separates: QAT's
loss is 0.113 against 0.148 for plain rounding. The hard part of quantisation
is not the weights.

## Variants

| variant | grid | granularity | bits | procedure |
|:--|:--|:--|--:|:--|
| `absmax` | symmetric | tensor | 8 | none |
| `zero_point` | asymmetric | tensor | 8 | none |
| `per_channel` | symmetric | output channel | 8 | none |
| `per_group` | symmetric | group of 16 | 4 | none |
| `ptq` | symmetric | group of 16 | 4 | percentile clipping |
| `qat` | symmetric | group of 16 | 4 | straight-through training |
| `gptq` | symmetric | group of 16 | 4 | Hessian compensation |
| `awq` | symmetric | group of 16 | 4 | activation-aware scaling |
| `int4` | symmetric | group of 16 | 4 | weights and activations |
| `int8` | symmetric | output channel | 8 | weights and activations |

The first four are schemes; the next four are procedures layered on a scheme;
the last two switch on activation quantisation and are the recipes people mean
when they say "int8" or "int4" inference.

## Derivation

### Codes and scales

A uniform quantiser maps a weight to an integer and back:

$$
\hat w=s\left(\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w}{s}\right)+z,
0,L\right)-z\right),\qquad L=2^b-1,
\qquad (1)
$$

with bit width $b$, step $s$ and integer zero point $z$. Symmetric
quantisation sets $z=0$ and uses one sign-magnitude grid:

$$
s=\frac{\max|w|}{2^{b-1}-1},\qquad |w-\hat w|\le \frac{s}{2}.
\qquad (2)
$$

Asymmetric quantisation spends the full $2^b$ codes on the observed range:

$$
s=\frac{\max w-\min w}{2^b-1},\qquad
z=\mathrm{round}\!\left(-\frac{\min w}{s}\right).
\qquad (3)
$$

Symmetric is cheaper (no zero point, and the multiply is a single integer
instruction) and is what most weight-only kernels use. Asymmetric wins when the
range is one-sided: the demo's values in $[0,2]$ have an rmse of 0.071 with a
symmetric grid and 0.033 with an asymmetric one, because half the symmetric
codes describe negative numbers that never occur.

### Granularity

One scale per row, and one per group of columns inside a row:

$$
s_i=\frac{\max_j|W_{ij}|}{2^{b-1}-1},\qquad
s_{i,g}=\frac{\max_{j\in g}|W_{ij}|}{2^{b-1}-1}.
\qquad (4)
$$

Under round-to-nearest the error in a block of step $s$ is roughly uniform on
$[-s/2,s/2]$, so

$$
\mathbb E\!\left[(w-\hat w)^2\right]\approx\frac{s^2}{12},
\qquad (5)
$$

which is the entire argument for finer granularity: the error is quadratic in
the block's range. The demo makes the spread explicit with rows scaled from
$10^{-2}$ to $10^{2}$: per-tensor rmse 2.927, per-channel 1.629, per-group
1.147. The cost is the scales themselves, stored in fp16:

$$
B=\frac{nb}{8}+2N_s,
\qquad (6)
$$

so 4-bit groups of 16 cost 0.5 bytes per weight plus 0.125 bytes of scale —
0.625 against 4, a factor of 6.4 before headers. The demo's end-to-end model
reports 448 bytes against 2144, or 4.8×, because the biases and the per-matrix
scales are not compressed.

### Bits and bytes

The same error formula explains the bit-width sweep. Halving the number of
levels doubles $s$ and quadruples the error, and the demo tracks it exactly:
rmse 0.0065 at 8 bits, 0.0282 at 6, 0.1227 at 4, 0.2587 at 3, 0.7984 at 2,
while the bytes fall from 266 to 74. Accuracy is a coarse instrument here; loss
and output error are the sensitive ones.

### Post-training quantisation with clipping

Percentile clipping picks a threshold below the maximum and saturates the tail:

$$
\tau_p=\mathrm{percentile}(|w|,p),\qquad
\hat w=\mathrm{quantise}\big(\mathrm{clip}(w,-\tau_p,\tau_p)\big).
\qquad (7)
$$

The bias from clipping the few largest weights buys a smaller step for
everyone else, and for a Gaussian-ish weight distribution the optimum is
usually around the 99.9th percentile. `ptq` uses 99.5 and the demo's loss
improves from 0.053 to 0.052 — a small effect at 4 bits, a large one at 3.

### QAT and the straight-through estimator

Rounding has zero derivative almost everywhere, so training through it needs a
surrogate:

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial\hat w}\cdot
\frac{\partial\hat w}{\partial w}
\approx\frac{\partial L}{\partial\hat w}.
\qquad (8)
$$

That is, the forward pass uses the quantised weight and the backward pass
pretends the quantiser was the identity. The demo keeps the fp32 weights as the
latent parameters and updates them with gradients evaluated at the quantised
copy:

$$
w\leftarrow w-\eta\,\nabla_{\hat w}L,\qquad
\hat w=\mathrm{quantise}(w).
\qquad (9)
$$

The estimator is biased, but it optimises the objective that deployment
actually sees, which is why it wins the 2-bit loss table: 0.113 against 0.148.

### GPTQ: choose the rounding, not just the grid

GPTQ treats quantisation as a layer-wise reconstruction problem. With
calibration inputs $X$ and a weight matrix $W$ stored as (output, input),

$$
\min_{\hat W}\;\lVert XW^{\top}-X\hat W^{\top}\rVert_F^2
=\min_{\hat W}\;\mathrm{tr}\!\left((W-\hat W)H(W-\hat W)^{\top}\right),
\qquad (10)
$$

where $H=X^{\top}X$ is the input Hessian. Quantising one output row at a time,
the optimal update after committing element $i$ follows from the inverse
Hessian:

$$
\hat w_i=\mathrm{quantise}(w_i),\qquad
w_j\leftarrow w_j-\frac{w_i-\hat w_i}{[H^{-1}]_{ii}}\,[H^{-1}]_{ij},
\quad j>i.
\qquad (11)
$$

Every remaining weight absorbs a share of the error, so the row's *output*
stays close even though individual weights are far from their fp32 values. The
demo's correlated inputs make the effect visible: 9.3e-3 output error for
round-to-nearest against 4.9e-3 for compensated rounding. The inverse is
computed once per layer with damping, and the loop is sequential in the input
dimension, which is why GPTQ is slower to apply than plain rounding.

### AWQ: protect the channels that carry signal

AWQ observes that a few input channels have much larger activations than the
rest, and that quantisation error in those channels matters more. It scales
each input channel before rounding and divides the scale back out afterwards:

$$
\hat W=\mathrm{quantise}(W\,\mathrm{diag}(s))\,\mathrm{diag}(s)^{-1},\qquad
s_j=\left(\mathbb E|X_j|\right)^{\alpha}.
\qquad (12)
$$

The scale is normalised so its geometric mean is one, which keeps the overall
dynamic range fixed. Setting $s_j>1$ for a large-activation channel divides its
effective step by $s_j$, buying resolution exactly where the error is
multiplied by a large $x_j$. On the demo's inputs spanning four orders of
magnitude, plain 4-bit groups leave 5.52 output error and AWQ leaves 3.47.

### Activations are the other operand

`int8` and `int4` quantise the activations as well, per tensor and dynamically
per batch:

$$
a=\mathrm{quantise}(x),\qquad \hat y=f(a;\hat W).
\qquad (13)
$$

Weights are fixed and can be calibrated once; activations change with every
input, so their scale must be computed at runtime, and their outliers are much
harder to control. The demo shows the asymmetry: `int8` matches fp32 at 0.994,
`int4` falls to 0.975 while weight-only 4-bit stays at 0.994. This is why
"4-bit LLM" almost always means 4-bit weights with 16-bit activations, and why
W4A4 is a research result rather than a default.

## Code walkthrough

The quantiser builds one array of blocks — the whole tensor, one row per block,
or one group per block — and then applies a single scale formula to it:

```python
    if mode == "tensor":
        blocks = values.reshape(1, 1, -1)
    elif mode == "channel":
        blocks = values.reshape(rows, 1, cols)
    elif mode == "group":
        groups = (cols + group - 1) // group
        width = groups * group
        padded = np.zeros((rows, width))
        padded[:, :cols] = values
        blocks = padded.reshape(rows, groups, group)
```

The symmetric and asymmetric grids differ only in whether the range is centred,
and the round-trip is the same expression in both cases:

```python
    if symmetric:
        scale = np.abs(blocks).max(axis=-1, keepdims=True) / qmax
        scale = np.where(scale > 0.0, scale, 1.0)
        codes = np.clip(np.round(blocks / scale), -qmax, qmax)
        dequant = codes * scale
        zeros = None
    else:
        low = blocks.min(axis=-1, keepdims=True)
        high = blocks.max(axis=-1, keepdims=True)
        scale = (high - low) / levels
        scale = np.where(scale > 0.0, scale, 1.0)
        zero = np.clip(np.round(-low / scale), 0, levels)
        codes = np.clip(np.round(blocks / scale) + zero, 0, levels)
        dequant = (codes - zero) * scale
        zeros = zero
```

Storage is counted from the codes and the fp16 side information, because a
quantised matrix is the codes plus whatever it takes to decode them:

```python
def quantised_bytes(info: dict) -> int:
    """Codes plus fp16 scales and zero points, plus a small header."""
    code = (info["count"] * info["bits"] + 7) // 8
    total = code + int(info["scale"].size) * 2
    if info["zero"] is not None:
        total += int(info["zero"].size) * 2
    return total + 8
```

GPTQ is the one procedure that cannot be written as a map over blocks. It walks
the input dimension, commits one value at a time, and pushes the residual into
the values it has not reached yet:

```python
    result = np.empty_like(row)
    work = row.copy()
    for index in range(len(row)):
        scale = scales[index // group]
        result[index] = np.clip(np.round(work[index] / scale),
                                -qmax, qmax) * scale
        error = (work[index] - result[index]) / inverse[index, index]
        if index + 1 < len(row):
            work[index + 1:] -= error * inverse[index, index + 1:]
```

The layer version inverts the damped Hessian once and loops over output rows;
damping is scaled by the mean diagonal so it does not depend on the input
scale:

```python
        if self.variant == "gptq":
            hessian = inputs.T @ inputs / inputs.shape[0]
            scale = np.trace(hessian) / hessian.shape[0]
            hessian = hessian + self.damping * scale * np.eye(len(hessian))
            inverse = np.linalg.inv(hessian)
            result = np.empty_like(weight)
            for row in range(weight.shape[0]):
                result[row] = _gptq_row(
                    weight[row], inverse, self.bits, self.group)
            return result
```

AWQ is four lines because the scaling is an ordinary broadcast along the input
axis, applied and then undone around the same `quantise` call:

```python
        if self.variant == "awq":
            activation = np.mean(np.abs(inputs), axis=0)
            activation = np.maximum(activation, 1e-8)
            scale = activation ** self.alpha
            scale = scale / np.exp(np.mean(np.log(scale)))
            scaled = weight * scale[None, :]
            info = quantise(scaled, **self._scheme())
            return info["values"] / scale[None, :]
```

The sequential calibration is what makes GPTQ and AWQ layer-aware: the second
layer is quantised against the activations produced by the *already quantised*
first layer, not against the fp32 activations:

```python
        for name in MATRIX_NAMES:
            inputs = values
            if name == "W2":
                hidden = np.tanh(values @ self.quantised["W1"].T
                                 + self.weights["b1"])
                inputs = hidden
            self.quantised[name] = self._quantise_layer(name, inputs)
```

QAT keeps the fp32 weights and swaps in a quantised copy only for the forward
and backward pass, which is (8) written as code:

```python
            for _ in range(steps - warmup):
                fake = self._fake_weights()
                loss, grads, _ = _loss_and_grads(fake, values, labels)
                for key in self.weights:
                    self.weights[key] -= lr * grads[key]
                self.history.append(loss)
```

Activation quantisation lives in the forward pass, applied to the input and
again to the hidden layer, so the network sees the same grid at every matmul:

```python
    x = np.asarray(x, dtype=float)
    if act_bits is not None:
        x = quantise(x, bits=act_bits)["values"]
    pre = x @ weights["W1"].T + weights["b1"]
    hidden = np.tanh(pre)
    if act_bits is not None:
        hidden = quantise(hidden, bits=act_bits)["values"]
    logits = hidden @ weights["W2"].T + weights["b2"]
```

## Complexity

For a matrix with $n$ entries, $d$ input channels and $c$ output channels:

- **Uniform quantisation:** one pass to find the block ranges, one to round —
  $O(n)$, and the same again to dequantise. Every scheme in the first four rows
  of the table is linear.
- **GPTQ:** the Hessian costs $O(cd^2)$ to form, its inverse $O(d^3)$, and the
  sequential loop $O(cd^2)$. For a 4096×4096 layer that is tens of billions of
  operations against 16.8 million weights — a calibration cost paid once per
  layer, not per token.
- **AWQ:** one pass over the calibration activations plus the quantiser, $O(n)$.
  The scale search over $\alpha$ adds a small constant factor.
- **QAT:** two forward passes' worth of work per step (one for the loss, one
  implicit in the quantise/dequantise) and an unchanged backward pass. The
  cost is training time, not inference time.
- **The demo:** ten recipes on a 4288-byte model, a 2-bit stress run, and four
  matrix-level studies. The whole thing is under a second; the 600-step dense
  training run is most of it.
- **Inference:** the saving is bytes, not necessarily time. A fused integer
  kernel turns 4.8× fewer bytes into real speedup; a dequantise-then-matmul
  loop can be slower than fp32.

## Failure modes

**One outlier sets the per-tensor scale.** A single large weight makes $s$ large
and every other weight coarse. That is the strongest reason to move to
per-channel or per-group, and the demo's row-scaled matrix shows the gap
(2.93 against 1.15).

**Grouping the wrong axis is silent and fatal.** Weights are stored as
(output, input); groups must run along the input dimension, because that is the
axis whose error is weighted by activations. Grouping along outputs instead
still produces valid-looking numbers and makes AWQ look useless — the mistake
is easy to make and invisible unless a reconstruction-error check catches it.

**Calibration data is part of the algorithm.** GPTQ and AWQ both optimise
against a sample of inputs. A calibration set that does not resemble
deployment turns the Hessian and the activation scales into noise. The demo
uses the training split for both training and calibration, which is convenient
and not a recommendation.

**GPTQ's inverse needs damping.** $X^{\top}X$ is singular whenever the
calibration set has fewer examples than input channels, and nearly singular
when features are correlated. Too little damping makes the compensation
explode; too much turns it back into plain rounding. The demo's correlated
inputs are deliberately ill-conditioned for this reason.

**Sequential compensation can accumulate error.** Each committed rounding is
final, so a bad early decision is compensated but never undone. This is why
GPTQ improves over round-to-nearest rather than becoming exact, and why the
gain shrinks as the bit width falls.

**AWQ's scale must travel with the model.** The per-channel factors have to be
folded into the preceding layer or applied at runtime; forgetting to divide
them back out produces a model that is scaled, not quantised. The demo divides
immediately, which is the easy case.

**The straight-through estimator is biased.** Treating the quantiser as the
identity in the backward pass is not the gradient of anything, and QAT can
settle into a worse optimum than fp32 training. Its win is that it optimises
the deployed objective, not that it is a better estimator.

**Fake quantisation must match the kernel.** Rounding mode, saturation,
per-channel axis, and whether zero is exactly representable all differ between
frameworks. A QAT model that matches the training simulator and not the
deployment kernel is a model that gets worse when it ships.

**Weight-only 4-bit is nearly free, so measure before adding machinery.** The
demo's four weight-only 4-bit recipes all score 0.994, the same as fp32. The
gain from GPTQ or AWQ at this scale is in loss and reconstruction error, not
accuracy; the same is true of many small models, where the procedures are
insurance rather than a necessity.

**Activation quantisation is the real constraint.** `int4` drops to 0.975 while
weight-only 4-bit stays at 0.994. The demo's 2-bit stress run is the same story
one level down: QAT keeps the smallest loss (0.113) but GPTQ and AWQ do not
convert their layer-level gains into downstream accuracy.

**Accuracy hides the damage.** A five-class task with 40 examples per class
moves in steps of 0.005. The reconstruction error and the cross-entropy are the
instruments that show a 20% improvement in a layer; accuracy is the headline
that shows whether it mattered.

**Bytes are not latency.** The 4.8× model may run slower than fp32 if the
kernel dequantises on the fly, and activation memory can dominate the weight
saving at long sequence lengths. Count both, and benchmark the kernel you will
actually deploy.

## Applications

Int8 post-training quantisation is the default for CPU and edge inference:
symmetric per-channel weights, asymmetric per-tensor activations, and an
integer kernel that turns the byte saving into throughput. QAT is what mobile
frameworks use when int8 post-training is not accurate enough, because it can
recover most of the gap with a short fine-tune.

GPTQ and AWQ are the two standard weight-only 4-bit recipes for serving large
language models: one calibration pass, a few minutes per model, and 3–4× less
memory with a small perplexity cost. They compose with LoRA — QLoRA keeps the
frozen base in NF4, a 4-bit codebook quantiser, and trains 16-bit adapters on
top, which is how a 65-billion-parameter model fits on one GPU.

Activation quantisation is the frontier: W8A8 is routine on server hardware,
W4A8 is common in recent accelerators, and W4A4 works with careful outlier
handling and rotation-based methods. KV-cache quantisation applies the same
codebook and group ideas to the attention cache, where the memory pressure is
often worse than in the weights.

## References

- Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H.,
  Kalenichenko, D. (2018). *Quantization and training of neural networks for
  efficient integer-arithmetic-only inference.* CVPR. Equations (1)–(3) and
  the integer-only recipe.
- Krishnamoorthi, R. (2018). *Quantizing deep convolutional networks for
  efficient inference: a whitepaper.* arXiv:1806.08342. Per-channel schemes
  and (4).
- Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M.,
  Blankevoort, T. (2021). *A white paper on neural network quantization.*
  arXiv:2106.08295. The PTQ/QAT split and clipping, (7).
- Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). *GPTQ:
  accurate post-training quantization for generative pre-trained
  transformers.* ICLR. Equations (10)–(11).
- Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G.,
  Dang, X., Gan, C., Han, S. (2024). *AWQ: activation-aware weight
  quantization for on-device LLM compression and acceleration.* MLSys.
  Equation (12).
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA:
  efficient finetuning of quantized LLMs.* NeurIPS. The NF4 codebook and
  double quantisation.
- Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or propagating
  gradients through stochastic neurons for conditional computation.*
  arXiv:1308.3432. The straight-through estimator, (8).
- Courbariaux, M., Bengio, Y., David, J.-P. (2015). *BinaryConnect: training
  deep neural networks with binary weights during propagations.* NeurIPS.
  The first modern QAT recipe.
- Choi, Y., El-Khamy, M., Lee, J. (2018). *Towards the limit of network
  quantization.* ICLR. Percentile clipping and the range/rounding trade.
- Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., Modha, D. S.
  (2020). *Learned step size quantization.* ICLR. Making the scale a trained
  parameter.
