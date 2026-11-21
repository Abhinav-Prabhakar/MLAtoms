# Gradient checkpointing — trading compute for memory

`checkpointing.py` · Training · Tier 2 · 5 variants

## Intuition

Backpropagation needs the activations it computed on the way forward. In a deep
chain that is one stored tensor per layer, and at large batch sizes it is often
the thing that runs out of memory first — not the parameters, not the optimiser
state, just the forward pass refusing to fit twice.

Checkpointing keeps a few activations and recomputes the rest during the
backward pass. The gradient is unchanged: the same arithmetic is replayed, so
the numbers are bit-identical to the full-storage version. What changes is the
ledger — a handful of stored states instead of all of them, paid for with one
extra forward pass.

The five variants are five points on that ledger. `full` stores everything and
recomputes nothing. `selective` stores every $k$-th state. `sqrt` chooses $k$
from the depth, which minimises the peak. `offload` still stores everything but
books it as host memory and pays a transfer per state. `reversible` needs no
checkpoints at all: its layers are invertible, so the forward pass can be run
backwards from the output.

The demo runs all five on a 24-layer chain of width 8. The first four produce
bit-identical gradients; the reversible chain, which is a different
architecture, matches finite differences to $1.1\times10^{-10}$ and inverts its
own forward pass to $2.6\times10^{-15}$.

## Variants

| variant | device states | peak | recompute | notes |
|:--|--:|--:|--:|:--|
| `full` | 25 | 25 | 0 | the baseline |
| `selective` | $k+1$ | $k+L/k$ | $L$ | every $k$-th boundary |
| `sqrt` | $\sqrt L+1$ | $2\sqrt L$ | $L$ | $k=\mathrm{round}(\sqrt L)$ |
| `offload` | 1 | 1 | 0 | $L+1$ host states, $L+1$ transfers |
| `reversible` | 1 | 1 | $L$ | coupling layers, no storage |

The chain is $h_{l+1}=h_l+\tanh(h_lW_l+b_l)$. That residual form matters: the
identity path has derivative one, so the adjoint passes through unchanged and
only the nonlinear branch needs the stored input.

## Derivation

### Why the forward pass dominates memory

For a batch $B$, width $d$ and $L$ layers, full backprop keeps

$$
M_{\text{full}}=(L+1)\,B\,d\ \text{floats},
\qquad (1)
$$

because every block needs its input to evaluate $1-\tanh^2$. The parameters are
$Ld^2$, so when $B\gg d$ the activations are the larger term — a batch of 64 at
width 1024 stores more than the weights of a 24-layer block.

### Checkpointed backward

Divide the chain into $k$ segments of length $L/k$. Store only the boundary
states $h_0,h_{L/k},\dots,h_L$. During the backward pass, when a segment is
reached, recompute it from its left boundary, walk it backwards, then discard
it and move to the previous segment. Peak storage is

$$
M(k)=(k+1)+\left\lceil L/k\right\rceil-1=k+\left\lceil L/k\right\rceil,
\qquad (2)
$$

and every layer is evaluated once more, so recomputation is exactly $L$ layer
evaluations regardless of $k$:

$$
C_{\text{checkpoint}}=C_{\text{forward}}+C_{\text{backward}}+C_{\text{forward}}.
\qquad (3)
$$

That is the whole trade: $O(L)$ memory becomes $O(\sqrt L)$ memory for one extra
forward pass.

### The square-root rule

Minimise (2) over $k$:

$$
k+\frac{L}{k}\ \ge\ 2\sqrt L,\qquad\text{equality at }k=\sqrt L,
\qquad (4)
$$

by AM-GM. For $L=24$ the minimum is 10 states, attained at $k=4,5,6$ — the
rounding of $\sqrt{24}=4.9$. The demo prints the whole curve: 25 states at
$k=1$, 14 at $k=2$, 10 across the middle, and back to 25 at $k=24$.

The rule assumes uniform layers. When the layers have different sizes the right
objective is bytes, not states, and the optimum solves a knapsack; the code
here counts states because every block in the demo has the same shape.

### Offloading

Offload keeps the same $L+1$ tensors but moves them out of device memory:

$$
M_{\text{device}}=B\,d,\qquad
M_{\text{host}}=(L+1)B\,d,\qquad
\text{transfers}=L+1.
\qquad (5)
$$

Nothing is recomputed, so the arithmetic is identical to `full`; the new cost
is bandwidth, $2(L+1)Bd$ bytes moved per step at four bytes per float. That is
the right choice when the interconnect is fast and the device is small, and the
wrong one when the host link is the bottleneck.

### Reversible layers

A reversible block splits the state into halves and uses a coupling:

$$
u'=u+F(v),\qquad v'=v,\qquad
u''=u',\qquad v''=v'+G(u').
\qquad (6)
$$

The output determines the input exactly:

$$
v'=v''-G(u''),\qquad u=u'-F(v'),
\qquad (7)
$$

so the forward pass can be run backwards. The backward pass inverts one block,
computes that block's parameter gradient from the recovered input, and
transforms the adjoint:

$$
\frac{\partial L}{\partial u}
=\frac{\partial L}{\partial u'},\qquad
\frac{\partial L}{\partial v}
=\frac{\partial L}{\partial v'}
+\Bigl(\frac{\partial L}{\partial u'}\odot(1-F(v)^2)\Bigr)W^{\top}.
\qquad (8)
$$

The demo's coupling uses $F(v)=\tanh(vW+b)$ on half-width states, so it stores
one state for 24 layers. The price is architectural: only even depths work, and
the model is no longer the plain residual chain of (1).

### Numerical cost of inversion

Each inversion applies $F$ again, so reversible backprop costs the same extra
forward pass as checkpointing. In exact arithmetic the reconstruction is
perfect; in floating point it accumulates roundoff. Twenty-four layers of
$\tanh$ recover the input to $2.6\times10^{-15}$, which is a few ulps — fine
here, but a reason to prefer checkpointing for very deep or ill-conditioned
chains.

## Code walkthrough

The boundaries are the whole policy. `sqrt` and `selective` differ only in the
step; `full` and `offload` keep every state and differ only in the ledger:

```python
    def _bounds(self) -> list[int]:
        if self.variant == "full":
            return list(range(self.layers + 1))
        if self.variant == "offload":
            return list(range(self.layers + 1))
        step = (max(1, int(round(np.sqrt(self.layers))))
                if self.variant == "sqrt" else self.segment)
        bounds = list(range(0, self.layers + 1, step))
        if bounds[-1] != self.layers:
            bounds.append(self.layers)
        return bounds
```

The forward pass always computes every state; the variant only decides which
ones survive. The residual add is where the identity path enters:

```python
        hidden = [x]
        current = x
        for layer in range(self.layers):
            current = current + self._block(current, layer)[0]
            hidden.append(current)
```

The ledger is written at the end of the forward pass, and `largest` is the
longest segment — the temporary buffer that sets the peak:

```python
        if self.variant == "offload":
            self.counts.update(stored=1, host=len(hidden), peak=1,
                               transferred=len(hidden))
        else:
            self.counts.update(stored=len(bounds),
                               peak=len(bounds) + largest - 1)
```

During the backward pass a segment is rebuilt from its left boundary. The
`recomputed` counter is what makes the compute bill visible:

```python
            else:
                segment = [cache["states"][left]]
                current = cache["states"][left]
                for layer in range(left, right):
                    current = current + self._block(current, layer)[0]
                    segment.append(current)
                self.counts["recomputed"] += right - left
```

Inside a segment the arithmetic is the ordinary residual backward, with the
adjoint carried across the identity path by the plain addition:

```python
            for offset in range(right - left - 1, -1, -1):
                layer = left + offset
                hidden = segment[offset]
                tanh, _ = self._block(hidden, layer)
                dpre = adjoint * (1.0 - tanh ** 2)
                self.gradients["W"][layer] += hidden.T @ dpre
                self.gradients["b"][layer] += dpre.sum(axis=0)
                adjoint = adjoint + dpre @ self.parameters["W"][layer].T
```

The reversible backward does both jobs at once: it subtracts the coupling to
recover the input half, and it pushes the adjoint through the same block. No
list of states is ever built:

```python
            if layer % 2 == 0:
                tanh, _ = self._block(second, layer)
                dpre = d_first * (1.0 - tanh ** 2)
                self.gradients["W"][layer] += second.T @ dpre
                self.gradients["b"][layer] += dpre.sum(axis=0)
                d_second = d_second + dpre @ self.parameters["W"][layer].T
                first = first - tanh
```

Inversion is the same loop without the adjoint, and it is what the demo uses to
prove the forward pass is lossless:

```python
        for layer in reversed(range(self.layers)):
            if layer % 2 == 0:
                first = first - self._block(second, layer)[0]
            else:
                second = second - self._block(first, layer)[0]
        return np.concatenate([first, second], axis=1)
```

Note that the order of the two updates is the mirror of the forward pass: the
even layer changes `u`, so the odd layer must be undone first.

## Complexity

For depth $L$, batch $B$, width $d$ and $k$ segments:

- **Full:** $(L+1)Bd$ states, one forward and one backward. The baseline.
- **Selective and sqrt:** $k+\lceil L/k\rceil$ states, two forwards and one
  backward. The recomputation is one extra forward pass, independent of $k$.
- **Offload:** $Bd$ device states, $(L+1)Bd$ host states, $L+1$ transfers of
  $Bd$ floats each.
- **Reversible:** $Bd$ states, one forward plus one inversion pass, no storage.
  The inversion costs one extra forward evaluation per layer.
- **Choosing $k$:** minimise bytes spent on checkpoints rather than states when
  layers differ in width; the optimum is no longer exactly $\sqrt L$.
- **The demo:** 24 layers of width 8 and a batch of 8, so one state is 64
  floats and the full chain is 1600. The finite-difference loop and the 300
  training steps dominate; everything is milliseconds.

## Failure modes

**Recomputation must be deterministic.** If a block draws a random mask,
applies stochastic depth, or advances a batch-norm running statistic, the
recomputed forward differs from the original and the gradient is wrong. Every
random draw must be captured from the first pass, not re-drawn.

**In-place operations break the replay.** A block that overwrites its input
destroys the very activation the checkpoint needs. RevNets avoid this by
construction; ordinary checkpointing requires functional layers.

**The boundaries must include both ends.** Dropping $h_L$ from the checkpoint
list makes the first segment unreachable; dropping $h_0$ makes the last
segment's recomputation start from nothing. The demo's `_bounds` always
contains 0 and $L$.

**Checkpointing does not reduce the parameters.** Optimiser state, gradients
and master weights are untouched. On a model where the weights dominate, the
saving is small, and offloading the optimiser state is a separate technique.

**The peak includes the temporary segment.** Reporting only the boundary count
understates the memory by the segment length, which is why `largest` enters the
ledger. At $k=L$ the boundary count is $L+1$ and the temporary buffer is one
state; at $k=1$ the boundary is one state and the buffer is all of them.

**Square-root is optimal only for a uniform chain.** With layers of different
sizes, or with sub-linear attention blocks, the minimisation is over bytes and
the answer is data-dependent.

**Offload trades one bottleneck for another.** Moving activations to the host
removes the device pressure and adds a transfer per state per step. On a slow
link this is slower than recomputing; on a fast one it is cheaper.

**Reversible layers constrain the architecture.** They need an even number of
blocks, a state that splits in half, and a coupling that is cheap to invert.
The plain residual block is not reversible, so the variant is a different model
with a similar loss surface, not a drop-in flag.

**Inversion accumulates roundoff.** Each subtraction undoes a $\tanh$; deep
chains lose a few bits per layer, and ill-conditioned weights amplify the loss.
Twenty-four layers are fine; a thousand are not.

**The loss gradient must match the loss.** The demo divides by the number of
elements because the loss is a mean; forgetting that factor produces a gradient
that is off by the batch size. Checkpointing does not change that, but the
finite-difference check is what proves both parts at once.

## Applications

Activation checkpointing is standard in large-model training: Megatron,
DeepSpeed and every transformer stack above a certain size use it, usually with
the square-root or per-layer policy. Offloading is the basis of ZeRO-Offload,
which moves optimiser state and sometimes activations to host memory. Reversible
layers appear in RevNets, in the Reformer's attention, and in normalising flows,
where invertibility is required rather than optional.

The same accounting applies outside deep learning: any long computation whose
derivative needs its intermediates can trade storage for replay. Adjoint
methods for ODE solvers, checkpointed simulation, and gradient accumulation
over micro-batches all make the same bargain in different notation.

## References

- Chen, T., Xu, B., Zhang, C., Guestrin, C. (2016). *Training deep nets with
  sublinear memory cost.* arXiv:1604.06174. Checkpointing and the square-root
  policy, equations (1)–(4).
- Griewank, A. (1992). *Achieving logarithmic growth of temporal and spatial
  complexity in dynamic programming.* Optimization Methods and Software.
  The original checkpointing analysis.
- Gomez, A. N., Ren, M., Urtasun, R., Grosse, R. B. (2017). *The reversible
  residual network: backpropagation without storing activations.* NeurIPS.
  Coupling layers, equations (6)–(8).
- Kitaev, N., Kaiser, Ł., Levskaya, A. (2020). *Reformer: the efficient
  transformer.* ICLR. Reversible attention.
- Pudipeddi, B., Meshi, O., Sarawagi, S. (2020). *Activation checkpointing in
  recurrent neural networks.* arXiv:2006.16099. Selective and uneven policies.
- Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. (2020). *ZeRO: memory
  optimizations towards training a trillion parameter model.* SC. Offloading
  as a memory hierarchy.
