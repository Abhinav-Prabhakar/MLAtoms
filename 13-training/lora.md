# LoRA and parameter-efficient fine-tuning — nine ways to train a few numbers

`lora.py` · Training · Tier 2 · 9 variants

## Intuition

Fine-tuning a large model rewrites every weight, and the rewrite is usually
small: the update lives in a much lower-dimensional subspace than the weight
matrix itself. LoRA takes that observation literally and parameterises the
update as a product of two thin matrices, so a $d\times k$ weight becomes
$r(d+k)$ trainable numbers with $r\ll\min(d,k)$.

Everything else in this atom is a different answer to the same question — what
should the small trainable part be? `dora` splits each weight into a magnitude
and a direction and trains them separately. `adalora` gives every rank its own
scale and prunes the useless ones. `rslora` fixes a scaling bug that appears as
the rank grows. `loha` and `lokr` replace the product with a Hadamard or
Kronecker product, which costs more parameters but represents different
updates. `qlora` shrinks the frozen base to four bits so the adapter fits on a
small device. `prefix_tuning` and `prompt_tuning` do not touch weights at all:
they learn extra key-value pairs or extra input tokens.

The demo freezes a small attention block trained on one three-class task,
changes the class centres, and then trains each adapter on the new task. The
frozen model scores 0.283. All seven weight adapters reach 1.000 with between
74 and 460 trainable numbers against a base of 1091. The two token adapters
reach 0.68: they can steer the frozen head but not rewrite it, which is the
honest version of the PEFT trade.

## Variants

| variant | trains | update | parameters per matrix |
|:--|:--|:--|:--|
| `lora` | two thin factors | $BA$ | $r(d+k)$ |
| `qlora` | LoRA on a 4-bit base | $BA$ | $r(d+k)$ |
| `dora` | LoRA plus a magnitude | $m\odot V/\lVert V\rVert$ | $r(d+k)+d$ |
| `adalora` | an SVD triple | $P\Lambda Q$ | $r(d+k)+r$ |
| `rslora` | LoRA, scaled by $\sqrt r$ | $BA$ | $r(d+k)$ |
| `loha` | two products, Hadamard | $(B_1A_1)\odot(B_2A_2)$ | $2r(d+k)$ |
| `lokr` | two rank-one products | $(B_1A_1)\otimes(B_2A_2)$ | $m_1+n_1+m_2+n_2$ |
| `prefix_tuning` | key and value prefixes | $[K_p;XW_k]$ | $2Ld$ in total |
| `prompt_tuning` | input tokens | $[P;X]$ | $Ld$ in total |

There is one class and one training loop. The adapter is selected by name and
everything else — the frozen block, the loss, the optimiser — is shared. That
is the property that matters in practice: swapping LoRA for DoRA is a
constructor argument, not a rewrite.

## Derivation

### Low-rank updates

Let $W_0\in\mathbb R^{d\times k}$ be frozen and let the update be

$$
W=W_0+\frac{\alpha}{r}\,BA,\qquad B\in\mathbb R^{d\times r},\quad
A\in\mathbb R^{r\times k},
\qquad (1)
$$

with $B$ initialised to zero so training starts exactly at the base model. The
trainable count is

$$
\underbrace{r(d+k)}_{\text{adapter}}\quad\text{instead of}\quad
\underbrace{dk}_{\text{dense}},
\qquad (2)
$$

which for $d=k=4096$ and $r=8$ is 65,536 against 16.7 million — a factor of
256. For a loss $L$ and $G=\partial L/\partial W$,

$$
\frac{\partial L}{\partial B}=\frac{\alpha}{r}\,GA^{\top},\qquad
\frac{\partial L}{\partial A}=\frac{\alpha}{r}\,B^{\top}G,
\qquad (3)
$$

so the backward pass costs one $d\times r$ and one $r\times k$ matmul on top of
whatever produced $G$.

### Why the scale is $\alpha/r$, and why that breaks

Take both factors to have entries of standard deviation $\sigma$. Each entry of
the product is a sum of $r$ products,

$$
\mathrm{Var}\!\left(\Delta W_{ij}\right)
=\sum_{k=1}^{r}\mathrm{Var}(B_{ik}A_{kj})=r\sigma^4,\qquad
\mathrm{std}=\sigma^2\sqrt r.
\qquad (4)
$$

Dividing by $r$ leaves a magnitude of $\sigma^2/\sqrt r$, which shrinks as the
rank grows: doubling the rank halves the effective update. `rslora` divides by
$\sqrt r$ instead, leaving $\sigma^2$ constant. The demo prints the two scales
side by side: at $r=16$ with $\alpha=8$, $\alpha/r=0.5$ and
$\alpha/\sqrt r=2.0$.

### QLoRA: a four-bit base

QLoRA leaves $W_0$ in a 4-bit format and learns the same LoRA branch in 16-bit.
The format is NF4, whose 16 levels are the quantiles of a standard normal, so
the codebook is matched to weights rather than to a uniform grid. Each block of
$b$ weights gets its own scale $s=\max|w|$, and

$$
q(w)=\arg\min_{n\in\text{NF4}}\left|w/s-n\right|,\qquad
\hat w=s\,n_{q(w)}.
\qquad (5)
$$

The scales themselves are quantised again to 8 bits — double quantisation —
which costs one byte per block instead of four. For a $256\times256$ matrix with
$b=64$ the demo reports 33,800 bytes against 262,144, a factor of 7.76 and 4.13
bits per weight. The dequantised base differs from the fp32 one by at most
0.122 per weight, and it is frozen, so that error is a constant offset the
adapter can absorb.

### DoRA: magnitude and direction

DoRA writes the weight as a magnitude vector times a unit-norm direction:

$$
W=m\odot\frac{V}{\lVert V\rVert_c},\qquad V=W_0+\frac{\alpha}{r}BA,
\qquad (6)
$$

where $\lVert\cdot\rVert_c$ is the norm of each column and $m$ is initialised
to the column norms of $W_0$, so $W=W_0$ at step zero. The gradients follow
from the quotient rule:

$$
\frac{\partial L}{\partial m_j}=\sum_i G_{ij}\,\hat W_{ij},\qquad
\frac{\partial L}{\partial V_{ij}}
=\frac{m_jG_{ij}-\hat W_{ij}\sum_l m_lG_{lj}\hat W_{lj}}
{\lVert V_j\rVert}.
\qquad (7)
$$

The demo verifies the invariant directly: after training, every column norm of
every adapted matrix equals its learned magnitude to $4.4\times10^{-16}$.

### AdaLoRA: rank as a budget

AdaLoRA keeps an explicit SVD-like triple and a penalty that keeps the factors
orthogonal:

$$
W=W_0+P\Lambda Q,\qquad
\mathcal R=\lVert P^{\top}P-I\rVert_F^2+\lVert QQ^{\top}-I\rVert_F^2,
\qquad (8)
$$

with $P\in\mathbb R^{d\times r}$, $\Lambda=\mathrm{diag}(\lambda)$ and
$Q\in\mathbb R^{r\times k}$. Each rank gets an importance score

$$
s_i=|\lambda_i|\,\lVert P_{:,i}\rVert_2\,\lVert Q_{i,:}\rVert_2,
\qquad (9)
$$

and ranks are pruned by zeroing the smallest. The demo's four scores are 1.82,
1.71, 2.36 and 2.95; dropping the two lowest costs 0.042 of validation loss,
dropping the two highest costs 0.344. Note that the check is on validation
data, not the test set — a pruning decision made on test data is not a decision
you can make in practice.

### LoHa and LoKr

LoHa replaces the sum with a Hadamard product of two low-rank products:

$$
\Delta W=(B_1A_1)\odot(B_2A_2),
\qquad (10)
$$

which costs $2r(d+k)$ and can represent interactions the single product cannot,
because the elementwise product mixes the two rank spaces. LoKr goes further and
uses a Kronecker product of rank-one factors:

$$
\Delta W=(B_1A_1)\otimes(B_2A_2),\qquad
\Delta W_{i m_2+j,\,k n_2+l}=(B_1A_1)_{ik}(B_2A_2)_{jl},
\qquad (11)
$$

with $d=m_1m_2$ and $k=n_1n_2$. Its parameter count is $m_1+n_1+m_2+n_2$ —
74 in the demo, the smallest weight adapter of the nine. The backward pass
reshapes $G$ to $(m_1,m_2,n_1,n_2)$ and contracts with the *other* factor:

$$
\frac{\partial L}{\partial(B_1A_1)}
=\sum_{j,l}G_{ijkl}\,(B_2A_2)_{jl},
\qquad (12)
$$

which is the line that a plain `sum` over the reshaped tensor gets wrong.

### Prefix and prompt tuning

These two change the sequence instead of the weights. Prompt tuning prepends
$L$ learnable vectors to the input, and prefix tuning prepends learnable keys
and values after the projection:

$$
X'=[P;X],\qquad
K=[K_p;X'W_k],\qquad V=[V_p;X'W_v],
\qquad (13)
$$

so attention runs over $L+T$ keys while the queries still come from the real
tokens. The counts are $Ld$ and $2Ld$, the smallest of the nine. Because the
prompt is shared across the batch, its gradient is a sum over examples:

$$
\frac{\partial L}{\partial P}=\sum_{b=1}^{B}
\left.\frac{\partial L}{\partial X'_{b}}\right|_{1:L},
\qquad (14)
$$

a reduction that is easy to forget when the prompt is stored as a batch of
copies.

## Code walkthrough

Every weight adapter has the same shape of implementation: build the effective
matrix, cache what the backward pass needs, and return it. LoRA is the
three-line case:

```python
        if self.variant in ("lora", "qlora", "rslora"):
            left = self.trainable[f"{name}.A"]
            right = self.trainable[f"{name}.B"]
            self._cache[name] = {"A": left, "B": right}
            return base + self.scale * (right @ left)
```

Its backward pass is the pair of products in (3), with the scale applied once:

```python
        if self.variant in ("lora", "qlora", "rslora"):
            self.gradients[f"{name}.A"] += self.scale * (
                cached["B"].T @ dweight)
            self.gradients[f"{name}.B"] += self.scale * (
                dweight @ cached["A"].T)
```

DoRA adds the magnitude and its quotient-rule correction; note that the unit
direction is what the gradients see, and the magnitude is a plain projection:

```python
            unit, norm = cached["unit"], cached["norm"]
            self.gradients[f"{name}.m"] += (dweight * unit).sum(axis=0)
            dunit = dweight * self.trainable[f"{name}.m"]
            dv = (dunit - unit * (unit * dunit).sum(axis=0)) / norm
```

AdaLoRA's three gradients are one contraction each, and the penalty term is the
gradient of the orthogonality loss:

```python
            self.gradients[f"{name}.lam"] += np.einsum(
                "ok,oi,ki->k", left, dweight, right)
            self.gradients[f"{name}.P"] += (dweight @ right.T) * middle
            self.gradients[f"{name}.Q"] += (left.T @ dweight) * middle[:, None]
```

LoKr is where the reshape matters: the transposed four-dimensional view is
contracted with the partner factor, not merely summed:

```python
            shaped = dweight.reshape(rows[0], rows[1], cols[0], cols[1])
            first = (self.trainable[f"{name}.B1"]
                     @ self.trainable[f"{name}.A1"])
            second = (self.trainable[f"{name}.B2"]
                      @ self.trainable[f"{name}.A2"])
            dfirst = np.einsum("ijkl,jl->ik", shaped, second)
            dsecond = np.einsum("ijkl,ik->jl", shaped, first)
```

The NF4 packer is the quantile search of (5) plus the second-level scale
quantisation:

```python
    blocks = flat.reshape(-1, block)
    scales = np.abs(blocks).max(axis=1) / 1.0
    scales = np.where(scales > 0.0, scales, 1.0)
    normalised = blocks / scales[:, None]
    codes = np.abs(normalised[..., None] - NF4).argmin(axis=-1)
```

The block forward is where prefix tuning enters: the prefix is broadcast across
the batch and concatenated in front of the projected keys and values:

```python
    if prefix is not None:
        keys = np.concatenate(
            [np.broadcast_to(prefix["K"], (batch,) + prefix["K"].shape),
             keys], axis=1)
        values = np.concatenate(
            [np.broadcast_to(prefix["V"], (batch,) + prefix["V"].shape),
             values], axis=1)
```

and the backward pass takes the prefix rows, reduces them over the batch, then
removes them before the projection gradients are formed:

```python
    left = cache["prefix"]
    if left:
        dprefix = {"K": dkeys[:, :left].sum(axis=0),
                   "V": dvalues[:, :left].sum(axis=0)}
        dkeys = dkeys[:, left:]
        dvalues = dvalues[:, left:]
```

The shared training loop is small because every adapter exposes the same three
methods — build, backward, step:

```python
        for _ in range(steps):
            loss, state = self.forward(x, y)
            self.backward(state)
            self.step(lr)
            history.append(loss)
```

## Complexity

For a batch $B$, sequence $T$, width $d$, rank $r$ and a target matrix of shape
$d\times k$:

- **Adapter forward:** an extra $O(Brd)$ and $O(Brk)$ matmul per adapted
  matrix, against $O(Bdk)$ for the frozen one. At $r=8$ and $d=k=4096$ the
  adapter is 0.4% of the matmul work.
- **Adapter backward:** the same two matmuls again, so the same fraction.
- **Memory:** $r(d+k)$ trainable numbers and their optimiser state; the frozen
  base in fp32, or 4.13 bits per weight for QLoRA.
- **Token adapters:** $Ld$ or $2Ld$ parameters, and attention cost grows from
  $O(T^2d)$ to $O(T(T+L)d)$ — cheap in parameters, not in compute.
- **The demo:** nine variants at 300 steps over 120 examples, a 1091-parameter
  base, plus a finite-difference check over about 1,600 adapter entries.
  The whole run is about a second, and the gradient check is most of it.

## Failure modes

**The rank can exceed the dense budget.** At $r=4$ in the demo, LoRA already
uses 460 parameters against 816 dense for four matrices, and at $r=16$ it is
more than twice the dense count. The demo's table prints the crossover; the
right rank is the smallest one that fits the task, not the largest that fits
memory.

**The $\alpha/r$ scale shrinks with the rank.** Equation (4) is not a
regularisation trick, it is a bug: the same learning rate produces smaller
updates as $r$ grows. rslora's $\alpha/\sqrt r$ is the fix, and the demo's
scale table is the check.

**Zero initialisation freezes half the adapter for one step.** With $B=0$ the
gradient of $A$ is $B^{\top}G=0$, so $A$ cannot move until $B$ does. The demo
warms every adapter for five steps before its finite-difference check, which is
also why a gradient check at initialisation would pass trivially.

**Quantising the adapter as well as the base wastes the point.** QLoRA keeps
the LoRA branch in 16-bit precisely so the small, precise updates are not
rounded. A four-bit adapter on a four-bit base has nowhere to put an update
smaller than its own grid.

**The NF4 block size is a range/resolution trade.** Small blocks track outliers
better and spend more bytes on scales; large blocks save memory and clip the
tail. The demo's block of 64 is the QLoRA default, not a law.

**Merging an adapter is a one-way door.** $W_0\leftarrow W_0+\Delta W$ removes
the parameter saving and makes the adapter indistinguishable from the base.
Merging two adapters into one base is only exact when they act on the same
subspace; otherwise the cross terms are lost.

**DoRA's norm is over the wrong axis the moment you transpose a weight.**
Convolution weights are usually stored as (out, in, k, k), and the "column
norm" of a flattened matrix is not the per-output-channel norm. The demo checks
the invariant explicitly for this reason.

**AdaLoRA collapses without the orthogonality penalty.** The triple $P\Lambda
Q$ is over-parameterised, so nothing stops $P$ and $Q$ from growing while
$\Lambda$ shrinks. The penalty in (8) is what keeps the importance scores
comparable across ranks.

**Rank pruning must use validation data.** The demo computes both the floor and
the pruned loss on a held-out split; computing them on the test set turns the
pruning decision into test-set tuning.

**A random prefix is not neutral.** Prefix tuning starts with random key-value
vectors, which perturb attention from step zero, unlike the zero-initialised
LoRA. That is a real difference in the optimisation landscape, not just in the
parameter count.

**A soft prompt cannot rewrite a frozen head.** The demo's token adapters reach
0.68 while the weight adapters reach 1.000. If the head's bias is the problem,
only a weight update can fix it; if the task is a matter of conditioning, a
prompt is enough. Choosing between them is a question about where the missing
information lives.

**Adapters on the wrong modules do nothing.** LoRA on the attention projections
and LoRA on the MLP learn different things; the usual recipe is to adapt every
linear layer and check the parameter budget. The demo's `targets` argument is
the same choice, exposed.

## Applications

LoRA is the default way to fine-tune open language models: one base in memory,
one small adapter per task, and a serving stack that swaps adapters per request.
QLoRA made 65-billion-parameter fine-tuning fit on a single 48 GB GPU, and the
NF4 format of (5) is now a standard storage format for frozen weights.

DoRA and AdaLoRA are refinements used when a plain low-rank update underfits:
DoRA when the magnitude of the weights matters, AdaLoRA when the rank budget
should be allocated per layer. LoHa and LoKr target convolutional and
structured layers where a single low-rank product is a poor fit, and they are
common in diffusion-model fine-tuning, where LoRA itself is the standard
customisation mechanism.

Prefix and prompt tuning sit at the other end: they change what the model sees
rather than what it knows, which makes them cheap, composable, and easy to
serve — a prompt is a row of numbers, not a weight file. They are weaker when
the frozen model's decision boundary is wrong rather than merely misdirected,
which is exactly the 0.68 against 1.000 in the demo.

## References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
  Chen, W. (2021). *LoRA: low-rank adaptation of large language models.* ICLR.
  Equations (1)–(3).
- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA:
  efficient finetuning of quantized LLMs.* NeurIPS. NF4, double quantisation
  and (5).
- Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng,
  K.-T., Chen, M.-H. (2024). *DoRA: weight-decomposed low-rank adaptation.*
  ICML. Equations (6)–(7).
- Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W., Zhao, T.
  (2023). *AdaLoRA: adaptive budget allocation for parameter-efficient
  fine-tuning.* ICLR. The SVD parameterisation and (8)–(9).
- Kalajdzievski, D. (2023). *A rank stabilization scaling factor for
  fine-tuning with LoRA.* arXiv:2312.03732. Equation (4) and the
  $\alpha/\sqrt r$ scaling.
- Hyeon-Woo, N., Ye-Bin, M., Oh, T.-H. (2021). *FedPara: low-rank Hadamard
  product for parameter-efficient federated learning.* arXiv:2108.06098. The
  Hadamard parameterisation behind LoHa.
- Yeh, S.-Y., Hsieh, Y.-Y., Gao, Z., Yang, B. B., Oh, G., Gong, Y., Chang,
  Y.-C. (2023). *Navigating text-to-image customization: augmenting
  text-to-image diffusion models with fine-grained ID and attribute control.*
  ECCV. The Kronecker parameterisation behind LoKr.
- Li, X. L., Liang, P. (2021). *Prefix-tuning: optimizing continuous prompts
  for generation.* ACL. Equation (13) for keys and values.
- Lester, B., Al-Rfou, R., Constant, N. (2021). *The power of scale for
  parameter-efficient prompt tuning.* EMNLP. The input-token variant.
- Aghajanyan, A., Zettlemoyer, L., Gupta, S. (2021). *Intrinsic dimensionality
  explains the effectiveness of language model fine-tuning.* ACL. Why a
  low-rank update is enough in the first place.
