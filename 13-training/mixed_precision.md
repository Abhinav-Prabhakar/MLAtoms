# Mixed precision — one exponent, three mantissas, and what breaks

`mixed_precision.py` · Training · Tier 2 · 6 variants

## Intuition

A floating-point number is a sign, an exponent and a mantissa. The exponent
buys range — how small and how large a number can be — and the mantissa buys
resolution — how finely neighbouring values can be told apart. Training needs
both: gradients decay toward $10^{-8}$ and smaller, while weight updates are
tiny differences between numbers near one.

Sixteen-bit formats halve the bytes per number but spend the bits differently.
fp16 keeps five exponent bits and ten mantissa bits, so it has good resolution
over a narrow range. bf16 keeps eight exponent bits and only seven mantissa
bits, so it spans fp32's whole range at a resolution of about three decimal
digits. fp8 keeps four and three, which is enough for a forward pass and not
much else. The three procedures on top — loss scaling, dynamic scaling and
master weights — exist because each format fails in a different place.

The demo emulates each format with explicit rounding, checks the fp16 path
against numpy's native `float16` bit for bit, and trains the same classifier
six ways. fp16, bf16 and the three procedures all reach 1.000 accuracy; fp8
stalls at 0.587, because a 3-bit mantissa cannot resolve the update.

## Variants

| variant | arithmetic | stored weights | scale | survives |
|:--|:--|:--|--:|:--|
| `fp16` | fp16 | fp16 | 1 | normal range |
| `bf16` | bf16 | bf16 | 1 | range, not resolution |
| `fp8` | E4M3 | E4M3 | 1 | forward only |
| `loss_scaling` | fp16 | fp16 | fixed | small gradients |
| `dynamic_scaling` | fp16 | fp16 | adaptive | both |
| `master_weights` | fp16 | fp32 | 1 | weight updates |

All six share the same one-hidden-layer network and the same demo data; only
the cast and the bookkeeping change. `loss_scaling` and `dynamic_scaling` are
`fp16` plus a multiplier applied to the gradient before it is quantised.
`master_weights` keeps the parameters in fp32 and casts a working copy for each
matmul, which is the arrangement every real mixed-precision trainer uses.

## Derivation

### What a format is

With $e$ exponent bits and $p$ mantissa bits, the largest finite value is

$$
x_{\max}=\left(2-2^{-p}\right)2^{\,2^{e-1}-1},\qquad
\epsilon=2^{-p},\qquad
x_{\min}=2^{\,1-\left(2^{e-1}-1\right)},
\qquad (1)
$$

where $\epsilon$ is the gap between 1 and the next representable number. For
fp16 $(e,p)=(5,10)$ this gives $x_{\max}=65504$, $\epsilon=9.77\times10^{-4}$
and $x_{\min}=6.10\times10^{-5}$. For bf16 $(8,7)$ the maximum is
$3.39\times10^{38}$ — fp32's — with $\epsilon=7.81\times10^{-3}$. For fp8
$(4,3)$ the maximum is 448, which is why E4M3 has no infinity encoding and
saturates instead.

Below $x_{\min}$ a format switches to subnormals: the exponent is pinned and
the mantissa counts in units of

$$
\delta=2^{\,1-2^{e-1}-p},
\qquad (2)
$$

which for fp16 is $2^{-24}\approx6\times10^{-8}$. Values smaller than
$\delta/2$ round to zero, and that is the first hard floor: a gradient below it
disappears.

### Rounding

Every cast here rounds to nearest, ties to even. Given frexp's decomposition
$x=m2^{E}$ with $m\in[0.5,1)$, the mantissa is rounded on its own grid:

$$
q(x)=\mathrm{rn}\!\left(m\,2^{p+1}\right)2^{E-p-1},
\qquad (3)
$$

and the subnormal branch replaces $E$ with the pinned exponent and rounds
$x/\delta$. The error is at most half an ulp, so the relative error of a normal
value is bounded by

$$
\frac{|q(x)-x|}{|x|}\le2^{-(p+1)}.
\qquad (4)
$$

For bf16 that is $2^{-8}=3.9\times10^{-3}$; the demo's worst observed error over
a sweep is $2.99\times10^{-3}$, just inside the bound.

### Why gradients vanish, and loss scaling

A gradient component $g$ is representable in fp16 only if $|g|\gtrsim\delta/2$.
Loss scaling multiplies the loss by a power of two $S$ before the backward pass:

$$
\tilde L=S\,L\ \Longrightarrow\ \tilde g=S\,g,
\qquad (5)
$$

so $10^{-8}$ becomes $10^{-5}$ at $S=1024$. Dividing the saved gradient by $S$
is exact because $S$ is a power of two — multiplication by $2^{k}$ only shifts
the exponent, and the rounding has already happened at the larger magnitude.
The demo shows the whole ladder: at fp16, $10^{-9}$ and $10^{-8}$ become zero;
scaled by 1024 and divided back they survive.

The catch is overflow. A gradient that was fine unscaled can exceed 65504 when
multiplied by $S$; fp16 turns it into infinity and the update is lost.

### Dynamic scaling

A fixed $S$ is either too small to help or too large for some step. The dynamic
rule halves on overflow and doubles after a run of clean steps:

$$
S_{t+1}=
\begin{cases}
\max(S_{\min},\,S_t/2), & \text{overflow},\\[2pt]
\min(S_{\max},\,2S_t), & \text{$w$ clean steps},\\[2pt]
S_t, & \text{otherwise}.
\end{cases}
\qquad (6)
$$

After an overflow the good-step counter resets, so it takes another full window
before the scale climbs again. The demo's trace is
$1024,1024,2048,1024,1024,1024,2048,1024$ for a window of three, which is the
rule applied to `F F F T F F F T`.

### Master weights

The update is

$$
w_{t+1}=w_t-\eta\,g_t.
\qquad (7)
$$

If $w_t$ is stored in fp16, the subtraction is rounded back onto the fp16 grid,
whose spacing near $|w|$ is $\epsilon|w|$. The update survives only if

$$
\eta\,|g|>\tfrac12\,\epsilon\,|w|.
\qquad (8)
$$

At $\eta=10^{-5}$ and $|w|\approx1$, the right-hand side is about
$5\times10^{-4}$ in fp16, larger than most single-step updates. The demo runs
100 steps at that rate: 57 of 67 fp16 weights are bit-identical to where they
started, while the same 100 updates move every fp32 master by up to
$5.5\times10^{-4}$. Keeping a fp32 copy and rounding only for the matmuls is
exactly the fix — the accumulator never sees the grid.

### Where the accumulation goes

Mixed precision means *storage and multiplication* in the narrow format, and
*accumulation* in fp32. The forward pass is

$$
h=\tanh\!\left(q(x)\,q(W_1)+q(b_1)\right),\qquad
z=q(h)\,q(W_2)+q(b_2),
\qquad (9)
$$

with every product cast before it enters the sum. The backward pass mirrors it,
and the gradient is quantised only once, after it has been scaled:

$$
\tilde g=q(S\,g),\qquad g=\tilde g/S.
\qquad (10)
$$

That single cast point is what makes overflow detectable: if any entry of
$\tilde g$ is non-finite, the step is dropped and the scale is reduced.

### Why fp8 fails here

The relative error bound (4) with $p=3$ is $2^{-4}=6.25\%$, and the grid near a
weight of order one is spaced by $2^{-3}$. An update of size
$\eta|g|\approx10^{-3}$ is smaller than that spacing, so it rounds to zero or to
a full step depending on where the weight sits. The demo's fp8 run gets to 0.9276
loss and 0.587 accuracy while every other variant reaches zero loss — the format
is fine for a forward pass whose inputs are already scaled, and not fine as the
only storage for a slowly moving weight.

## Code walkthrough

The cast is the whole atom's foundation. `frexp` splits the number, `rint`
rounds the mantissa, `ldexp` puts it back; the subnormal branch counts in units
of $\delta$, and the overflow branch is a comparison against half an ulp past
the maximum:

```python
    ebits, mbits = FORMATS[fmt]
    max_exp = (1 << (ebits - 1)) - 1
    min_exp = 1 - max_exp
    step = 2.0 ** (min_exp - mbits)
    magnitude = np.abs(values)
    mantissa, exponent = np.frexp(magnitude)
    rounded = np.rint(mantissa * 2.0 ** (mbits + 1)) / 2.0 ** (mbits + 1)
    out = np.copysign(np.ldexp(rounded, exponent), values)
    below = magnitude < 2.0 ** min_exp
    out = np.where(below, np.rint(values / step) * step, out)
    limit = MAX_VALUE[fmt]
    if SATURATES[fmt]:
        out = np.clip(out, -limit, limit)
    else:
        edge = limit + 2.0 ** (max_exp - mbits - 1)
        out = np.where(np.abs(out) >= edge, np.copysign(np.inf, values), out)
```

The `edge` comparison is the round-to-nearest rule applied at the top of the
range: 65519 stays finite, 65520 becomes infinity, exactly as IEEE fp16 does.
The demo confirms that agreement against numpy over 1505 draws including
subnormals and overflow.

The scaler is state plus two rules, and the reset of `good` is the part people
forget — without it a single overflow is followed by an immediate doubling:

```python
    def update(self, overflow: bool) -> float:
        if overflow:
            self.scale = max(self.minimum, self.scale * self.backoff)
            self.good = 0
        else:
            self.good += 1
            if self.good >= self.window:
                self.scale = min(self.maximum, self.scale * self.growth)
                self.good = 0
        return self.scale
```

`_working` is the master-weights trick in three lines: the parameters are the
fp32 masters, and every matmul reads a quantised copy that is thrown away after
the step:

```python
    def _working(self) -> dict[str, np.ndarray]:
        return {key: cast(value, self.fmt)
                for key, value in self.parameters.items()}
```

The forward pass casts the input once and keeps its products at the low
precision while the additions stay fp32:

```python
        p = self._working()
        pre = cast(x, self.fmt) @ p["W1"] + p["b1"]
        hidden = cast(np.tanh(pre), self.fmt)
        logits = hidden @ p["W2"] + p["b2"]
```

The backward pass quantises the gradient only after scaling it, and returns the
overflow flag instead of a gradient when any entry is non-finite:

```python
        scale = self.current_scale
        stored = {key: cast(value * scale, self.fmt)
                  for key, value in grads.items()}
        if any(not np.all(np.isfinite(value)) for value in stored.values()):
            return None, True
        return {key: value / scale for key, value in stored.items()}, False
```

`apply` is where master weights change the behaviour: the low-precision variant
rounds the updated weight back onto the grid, the master variant does not:

```python
        for key, gradient in gradients.items():
            value = self.parameters[key] - lr * gradient
            self.parameters[key] = (value if self.master
                                    else cast(value, self.fmt))
```

And the loop treats a dropped step as a skipped step, not an error — the
optimiser simply does not advance that iteration:

```python
            grads, overflow = self.backward(cache)
            if overflow:
                if self.variant == "dynamic_scaling":
                    self.scaler.update(True)
                continue
```

## Complexity

For $n$ parameters, batch $B$ and a model with $P$ parameter entries:

- **Memory:** fp16 and bf16 store $2P$ bytes of weights against $4P$ in fp32;
  master weights store both, so $6P$. Activations halve, unless the framework
  keeps fp32 copies for the backward pass.
- **Arithmetic:** the matmuls cost $O(BP)$ in either format. The cast is $O(P)$
  per step and runs twice for master weights — once to make the working copy,
  once to quantise the gradient.
- **Bandwidth:** the real gain. Half the bytes per tensor means half the traffic
  for weights, activations and gradients, which is why fp16 helps most on
  memory-bound layers.
- **The demo:** 67 parameters, a batch of 75, six variants at 400 steps. Each
  cast touches every entry through `frexp`, `rint` and `ldexp`, so it is a
  handful of vectorised passes; the whole demo is well under a second.

## Failure modes

**Loss scaling and gradient clipping fight.** Clipping must happen after
unscaling; clipping the scaled gradient at a fixed norm clips $S$ times too
hard, and the model stops learning while the loss looks finite.

**A scaler that never resets its window oscillates.** Doubling immediately after
an overflow reproduces the same overflow on the next step. The counter reset in
(6) is what makes the search converge.

**Skipped steps change the schedule.** Every overflow throws away an iteration,
so a run with heavy scaling takes more steps to reach the same point. Learning
rate schedules keyed to the step count must count attempted steps, not applied
ones.

**bf16 has no useful overflow signal.** Its range matches fp32, so the
non-finite check almost never fires — and when it does, the cause is a genuine
blow-up rather than a scale that is too large. bf16 is normally used without a
scaler at all.

**fp8 saturates silently.** E4M3 clamps at 448 instead of producing infinity,
so an overflowing activation becomes a plausible-looking large number and the
loss quietly degrades. Per-tensor scaling is mandatory in practice; this demo
shows the format, not a production recipe.

**Quantising the update is not the same as quantising the weight.** Master
weights help because the accumulator stays wide. If the optimiser state is also
kept narrow, momentum underflows even when the weight does not, and the fix has
to include the state.

**Rounding is biased without stochastic rounding.** Nearest-even always sends
a repeatedly tiny update to the same neighbour, so many small steps in the same
direction can cancel to zero. Stochastic rounding trades that bias for variance
and is what makes low-precision accumulation usable in long runs.

**Accumulating in the narrow format loses the sum.** A dot product of 4096
terms accumulated in fp16 loses most of its low bits; the accumulation must be
fp32 even when the products are fp16. `numpy` here accumulates in float64 by
construction, which hides the issue rather than solving it.

**Reductions are where fp16 hurts most.** LayerNorm, softmax and the loss all
sum over long axes; their intermediate sums overflow or underflow in fp16 long
before the matmuls do, which is why those layers stay in fp32 in real stacks.

**A cast that is not round-to-nearest is a silent bias.** Truncation toward zero
is cheaper and is what some hardware does by default; over a training run it
drags every weight toward zero. The demo's bit-for-bit agreement with numpy's
`float16` is the check that this implementation is not doing that.

## Applications

Automatic mixed precision with fp16 and a dynamic loss scaler is the default on
Volta and later GPUs: matmuls and convolutions in fp16, master weights, norms
and losses in fp32. bf16 is preferred where the hardware supports it — TPUs and
Ampere onward — because it needs no scaler and its range matches fp32, so the
same recipe works without the overflow machinery.

fp8 is the current frontier: E4M3 for forward weights and activations, E5M2 for
gradients, with per-tensor or per-block scaling. The arithmetic in (10) is the
same; the scaling factors become the whole engineering problem. Master weights
appear in every large-scale trainer and in ZeRO, which shards the fp32 copy
along with the optimiser state.

Outside training, the same three-way split shows up in inference quantisation
(int8 with an fp32 scale per channel), in mixed-precision ODE solvers, and in
any pipeline where a wide accumulator sits behind a narrow interface.

## References

- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D.,
  Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., Wu, H. (2018).
  *Mixed precision training.* ICLR. Loss scaling and master weights,
  equations (5)–(8).
- Kalamkar, D., Mudigere, D., Mellempudi, N., Das, D., Banerjee, K., Avancha,
  S., et al. (2019). *A study of BFLOAT16 for deep learning training.*
  arXiv:1905.12322. The bf16 trade-off.
- Wang, N., Choi, J., Brand, D., Chen, C.-Y., Gopalakrishnan, K. (2018).
  *Training deep neural networks with 8-bit floating point numbers.* NeurIPS.
  The case for fp8 with fp32 accumulation.
- Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P., Grisenthwaite,
  R., et al. (2022). *FP8 formats for deep learning.* arXiv:2209.05433. E4M3
  and E5M2, and the saturation rule.
- Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. (2020). *ZeRO: memory
  optimizations towards training a trillion parameter model.* SC. Sharded
  master weights and optimiser state.
- Gupta, S., Agrawal, A., Gopalakrishnan, K., Narayanan, P. (2015). *Deep
  learning with limited numerical precision.* ICML. Stochastic rounding and
  why truncation biases a run.
