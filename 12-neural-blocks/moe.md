# Mixture of experts — sparse capacity without sparse gradients

`moe.py` · Neural blocks · Tier 2 · 8 variants

## Intuition

A mixture of experts replaces one wide feed-forward network with $E$ narrow
ones and a small router that sends each token to a few of them. The capacity is
sparse — a token touches $k$ experts, not $E$ — but the parameter gradients are
not: every expert owns a full array, and the experts that were not selected
simply receive zeros. That combination is the whole engineering story. The
model has $E$ times more parameters but only $k$ times the arithmetic, and the
arrays stay dense so the usual optimiser needs no special case.

The trade is a new failure surface. The router makes a discrete choice, and a
discrete choice has no derivative almost everywhere. A top-1 gate is the
constant one, so the task loss sends **no** gradient to the router at all; only
an auxiliary loss can train it. The demo measures this: with every token routed
to expert 0, the gradients of experts 1–3 are zero *and* the router's gradient
is zero, while the auxiliary loss still has a usable slope.

Capacity is the second half. To keep a batch's expert computation bounded, each
expert accepts at most $C$ tokens; the overflow is dropped and its output is
zero. The variants divide along these two axes — how many experts per token,
who chooses them, and what happens when an expert is full.

## Variants

| variant | token chooses | experts kept | capacity | aux loss |
|:--|:--|--:|:--|:--|
| `top1` | argmax | 1 | no | no |
| `topk` | top-$k$ softmax | $k$ | no | no |
| `expert_choice` | experts choose tokens | variable | yes | no |
| `switch` | argmax | 1 | no | yes |
| `soft_moe` | all, weighted | $E$ | no | optional |
| `capacity_factor` | top-$k$ | $k$ | yes | no |
| `load_balance_loss` | top-$k$ | $k$ | no | yes |
| `shared_expert` | top-$k$ plus a shared expert | $k+1$ | no | no |

The demo uses six input channels, eight hidden units, four experts and
twenty-four tokens.

**`top1`** is the simplest useful mixture: one expert per token, output equal to
that expert's output exactly. It has no router gradient from the task loss.

**`topk`** keeps the $k$ largest logits and renormalises their softmax, so the
gates sum to one. With $k=1$ it is `top1`; with $k=E$ it is a dense mixture.

**`expert_choice`** flips the direction: each expert selects its $C$ favourite
tokens. Every expert is full by construction, so load is perfectly balanced,
but a token may be chosen zero, one or several times.

**`switch`** is top-1 routing plus the load-balance auxiliary loss. It is the
recipe that made sparse mixtures trainable at scale; the capacity mechanism is
separated into `capacity_factor` here so the two effects can be measured
independently.

**`soft_moe`** weights every expert by its router probability, with no hard
assignment. It has no arithmetic saving but no discrete choice either, so the
gradient reaches every expert.

**`capacity_factor`** is top-$k$ routing with a hard quota
$C=\lceil \alpha n/E\rceil$ per expert; assignments beyond the quota are
dropped.

**`load_balance_loss`** is top-$k$ routing with the auxiliary term, without
dropping.

**`shared_expert`** adds one always-on expert to the top-$k$ mixture, the
DeepSeekMoE arrangement: a dense path for common features and sparse experts
for the rest.

## Derivation

### The mixture

For a token $x$ with selected set $S(x)$ and gates $g_e$,

$$
y=\sum_{e\in S(x)} g_e\,E_e(x),
\qquad
E_e(x)=\mathrm{gelu}(xW_1^e+b_1^e)W_2^e+b_2^e .
\qquad (1)
$$

The router produces logits and probabilities,

$$
z=xW_r+b_r,\qquad p=\mathrm{softmax}(z),\qquad
S=\mathrm{TopK}(z),\qquad g=\mathrm{softmax}(z_S),
\qquad (2)
$$

so $g_j=p_j/\sum_{l\in S}p_l$ on the selected set. The token-wise structure is
why the layer is easy to batch: every operation is a matrix product over rows,
and no row mixes with another except through the router's normalisation.

With $k=1$ the selected set has one element and $g=1$ identically. That is the
key observation behind the Switch auxiliary loss: since $y=E_{\arg\max z}(x)$,
the derivative with respect to $z$ is zero wherever the argmax is locally
constant, so the task loss cannot train the router at all.

### Capacity

$$
C=\left\lceil \alpha\,\frac{n}{E}\right\rceil,
\qquad (3)
$$

where $n$ is the number of tokens and $\alpha$ the capacity factor. Each expert
accepts the first $C$ assignments in token order and drops the rest; a dropped
slot contributes nothing, so a top-1 token can disappear entirely. The demo
sends all 24 tokens to expert 0 with $\alpha=0.5$ and $E=4$, so $C=3$: three
outputs survive, twenty-one are exactly zero.

### Load balancing

Let $c_e$ be the number of assignments expert $e$ received before dropping,
$f_e=c_e/(nk)$ the fraction of all assignments, and
$P_e=\frac1n\sum_i p_{ie}$ the mean router probability. The Switch auxiliary
loss is

$$
L_{\mathrm{aux}}
=E\sum_{e=1}^{E} f_e P_e .
\qquad (4)
$$

At uniform routing $f_e=1/E$ and $P_e=1/E$, so $L_{\mathrm{aux}}=1$. At a
collapse onto one expert $f=(1,0,\dots)$ and $P\approx(1,0,\dots)$, so
$L_{\mathrm{aux}}=E$. Hence

$$
1\le L_{\mathrm{aux}}\le E,
\qquad (5)
$$

with equality at the two extremes. Dividing $c_e$ by $nk$ rather than $n$ is
what keeps the lower bound at one when $k>1$; the original Switch statement is
the $k=1$ case.

The gradient passes through $P$ only, because $f$ is a hard count:

$$
\frac{\partial L_{\mathrm{aux}}}{\partial p_{ie}}=\frac{E f_e}{n},
\qquad
\frac{\partial L_{\mathrm{aux}}}{\partial z_{ie}}
=p_{ie}\Bigl(\frac{\partial L}{\partial p_{ie}}
-\sum_l p_{il}\frac{\partial L}{\partial p_{il}}\Bigr).
\qquad (6)
$$

The second factor is the softmax Jacobian. The demo checks (6) against central
differences to $1.4\times10^{-10}$ for `switch` and
$1.8\times10^{-10}$ for `load_balance_loss`.

### Gradients through the gate

For top-$k$, let $d_j=\partial L/\partial g_j$ be the derivative arriving at
the selected gates. The softmax over the selected logits gives

$$
\frac{\partial L}{\partial z_{S_j}}
=g_j\Bigl(d_j-\sum_{l\in S}g_l d_l\Bigr),
\qquad (7)
$$

and the unselected logits receive zero from this path. The expert gradient is
the usual MLP backward pass multiplied by the gate:

$$
\frac{\partial L}{\partial E_e(x)}=g_e\,\frac{\partial L}{\partial y}.
\qquad (8)
$$

For $k=1$, (7) is $1\cdot(d-d)=0$, which is the zero router gradient the demo
measures. For $k>1$ the gates move with the logits and the router does learn
from the task, though the auxiliary term is still what keeps usage even.

### Expert choice

Each expert ranks all tokens by its own logit and takes the top $C$. Let
$m_i=\sum_{e\in S_i}p_{ie}$ be the total probability mass a token received.
The gate is the renormalised probability,

$$
g_{ie}=\frac{p_{ie}}{m_i}\,\mathbf 1[e\in S_i],
\qquad (9)
$$

and the derivative with respect to the *full* probability vector is

$$
\frac{\partial L}{\partial p_{if}}
=\frac{1}{m_i}\Bigl(d_{if}-\sum_{e\in S_i}d_{ie}g_{ie}\Bigr)
\quad\text{for }f\in S_i,
\qquad (10)
$$

with zero outside $S_i$; the softmax Jacobian of (6) then maps it to the
logits. Note that $m_i$ itself depends on $p$, which is why (10) has the
correction term rather than a plain division. The demo's finite differences
agree to $2.4\times10^{-10}$.

Because each expert takes exactly $C$ tokens, the counts are balanced without
any loss term. The price is collisions: with 24 tokens and $E=4$, $C=6$ gives 24
assignment slots for 24 tokens, and the demo finds that 10 tokens are chosen by
no expert at all. Their output is zero.

### Soft routing

With no selection, $S_i$ is every expert and $g=p$:

$$
y_i=\sum_{e=1}^{E}p_{ie}E_e(x_i),
\qquad
\frac{\partial L}{\partial p_{ie}}
=\frac{\partial L}{\partial y_i}\cdot E_e(x_i).
\qquad (11)
$$

Every expert contributes to every token, so no capacity is needed and no
assignment can be wasted. The auxiliary loss still applies, but now
$f_e=P_e$, so (4) becomes $E\sum_e P_e^2$, which is minimised by uniform mean
probability. Its gradient replaces the constant $E f_e/n$ in (6) with
$2E P_e/n$; the demo checks that case to $3.6\times10^{-10}$.

### Shared expert

$$
y=\mathrm{Shared}(x)+\sum_{e\in S(x)}g_eE_e(x).
\qquad (12)
$$

The shared branch sees every token, so it always receives gradient and acts as
a dense residual path. Its parameters are one more expert's worth. The demo
subtracts a plain `topk` model with the same seed and finds the difference is
exactly the shared output, then perturbs the router and confirms the shared
branch does not move.

### Parameter and arithmetic counts

With input width $d$, hidden width $h$ and $E$ experts,

$$
P_{\mathrm{expert}}=2dh+h+d,\qquad
P_{\mathrm{router}}=dE+E,\qquad
P_{\mathrm{total}}=EP_{\mathrm{expert}}+P_{\mathrm{router}} .
\qquad (13)
$$

The small demo model has $d=4$, $h=5$, $E=3$, so each expert owns
$40+5+4=49$ parameters and the whole mixture owns $3\cdot49+15=162$. A forward
pass costs $k$ expert MLPs per token instead of $E$; a dense soft mixture costs
all $E$. The router costs $O(dE)$ per token.

## Code walkthrough

The auxiliary loss is the normalized product of hard usage and soft
probability, and its lower bound is one exactly when both are uniform:

```python
def _load_balance(probability: np.ndarray, fractions: np.ndarray) -> float:
    """Switch-style auxiliary loss, minimal at uniform routing."""
    experts = probability.shape[1]
    return float(experts * np.sum(fractions * probability.mean(axis=0)))
```

The top-$k$ router is three lines: rank, gather, renormalise. Nothing is
computed for an expert that no token selected, which is where the arithmetic
saving lives:

```python
            indices = np.argsort(-logits, axis=1)[:, :width]
            selected = np.take_along_axis(logits, indices, axis=1)
            gates = _softmax(selected)
```

Capacity is a greedy fill in token order, so a token that arrives late can be
dropped even when its gate is large. This is deliberate: it makes the output
independent of the gate magnitudes and keeps the batch order meaningful:

```python
    def _keep(self, n: int, width: int, indices: np.ndarray, capacity: int):
        """Keep at most `capacity` assignments per expert, in token order."""
        used = np.zeros(self.experts, dtype=int)
        kept = np.zeros((n, width), dtype=bool)
        for row in range(n):
            for slot in range(width):
                expert = int(indices[row, slot])
                if used[expert] < capacity:
                    kept[row, slot] = True
                    used[expert] += 1
        return kept, used
```

An expert is a one-hidden-layer GELU network. It is deliberately identical in
every variant, so the differences in the file are routing differences:

```python
def _expert_forward(parameters: dict[str, np.ndarray], name: str,
                    x: np.ndarray):
    pre = x @ parameters[f"{name}.W1"] + parameters[f"{name}.b1"]
    hidden = _gelu(pre)
    out = hidden @ parameters[f"{name}.W2"] + parameters[f"{name}.b2"]
    return out, (x, pre, hidden)
```

Expert choice inverts the direction of selection. Each expert takes its own
top-$C$ tokens, and the gate is the probability renormalised over whichever
experts happened to choose that token:

```python
            capacity = self._capacity(n)
            chosen = np.zeros((n, self.experts), dtype=bool)
            for expert in range(self.experts):
                order = np.argsort(-logits[:, expert])
                chosen[order[:capacity], expert] = True
            mass = (probability * chosen).sum(axis=1, keepdims=True)
            gates = (probability * chosen) / np.maximum(mass, 1e-12)
```

Soft routing is the dense limit: every expert is evaluated and the probability
is the weight. There is no branch, no capacity and no zero:

```python
            for expert in range(self.experts):
                out, cache = _expert_forward(p, f"expert{expert}", x)
                output += probability[:, expert, None] * out
                caches[expert] = (rows, None, cache, out)
```

The backward pass through a softmax is the same identity in every variant. For
top-$k$ the selected gate derivative is scattered back into the full logit
matrix, and the unselected logits get nothing from this path:

```python
            d_selected = gates * (d_gate - (gates * d_gate).sum(
                axis=1, keepdims=True))
            rows = np.arange(n)
            for slot in range(indices.shape[1]):
                np.add.at(dz, (rows, indices[:, slot]), d_selected[:, slot])
```

Expert choice adds the correction for the token-dependent normaliser from
(10). Without the second term the gradient of a gate would act as if the other
gates were fixed, and the finite-difference check would fail by a visible
margin:

```python
            correction = (d_gate * gates).sum(axis=1, keepdims=True)
            d_probability = np.where(
                info["chosen"],
                (d_gate - correction) / np.maximum(mass, 1e-12), 0.0)
```

The auxiliary gradient is analytic rather than autograd: differentiate (4),
apply the softmax Jacobian, and hand the result to the same optimiser update as
the task gradient:

```python
        if self.variant == "soft_moe":
            d_probability = 2.0 * experts * probability.mean(
                axis=0)[None, :] / n
        else:
            d_probability = experts * info["fractions"][None, :] / n
        dz = probability * (d_probability
                            - (probability * d_probability).sum(
                                axis=1, keepdims=True))
```

The shared expert is added after the routed sum, so the backward pass sends the
same upstream gradient down both branches. No gate multiplies it:

```python
        if self.variant == "shared_expert":
            shared, shared_cache = _expert_forward(p, "shared", x)
            output += shared
            info.update(shared=shared, shared_cache=shared_cache)
```

## Complexity

For $n$ tokens, width $d$, hidden width $h$, $E$ experts and $k$ selected
experts:

- **Router:** $O(ndE)$ time for the logits, and $O(nE)$ for the sort or top-$k$
  pass. The router is tiny next to the experts.
- **Experts:** $O(nkdh)$ time and $O(kdh)$ working memory per token. A dense
  mixture is $E$ times the expert cost; `soft_moe` is exactly that.
- **Parameters:** $E(2dh+h+d)+dE+E$, independent of $n$ and $k$. Sparse
  capacity buys parameters, not a smaller state.
- **Capacity:** the greedy fill is $O(nk)$ and the quota is
  $O(\alpha n/E)$ per expert. In this dense implementation dropped tokens are
  still evaluated and then multiplied by zero, so capacity bounds the output
  but not the wall-clock work; a real kernel skips the dropped rows.
- **Expert choice:** an $O(n\log n)$ sort per expert, then $O(E n)$ to build the
  assignment matrix; the gates are one masked row-normalisation.
- **The demo:** eleven sections over at most 24 tokens and 4 experts, plus a few
  thousand finite-difference forward passes on a 7-token model. It is
  deterministic and runs in well under a second.

## Failure modes

**A top-1 router gets no task gradient.** The gate is the constant one, so
$y$ is piecewise constant in $z$; the derivative is zero except at ties. The
demo confirms both the unselected experts' gradients and the router gradient
are exactly zero. Switch's auxiliary loss is not a nicety: it is the only
signal the router receives. A straight-through estimator is the other common
patch.

**Dropped tokens are silently lost.** With capacity $\alpha n/E$ and a router
that favours one expert, most of a batch can vanish. A top-1 token that is
dropped produces a zero row, and the loss then trains on the surviving tokens
only. Capacity factors above one and the auxiliary loss both reduce this, at
the cost of more compute or a weaker router.

**Expert choice collides.** Every expert is full, but tokens are not covered.
At 24 tokens, 4 experts and capacity 6, the demo finds 10 tokens chosen by no
expert. Raising the capacity factor lowers the collision rate, but the total
assignment slots must exceed the token count for coverage, and even then it is
not guaranteed.

**Batch composition changes the answer.** Capacity and expert choice both
depend on the whole batch. A token's output can change when other tokens join
or leave, so per-example inference and training disagree. This is a real
correctness hazard for serving, not just a numerical detail.

**Dead experts never revive.** An expert that is never selected gets zero
gradient forever. The auxiliary loss discourages collapse but cannot guarantee
that every expert is used, and a hard top-1 router can stay stuck in a
self-reinforcing assignment.

**Partial top-$k$ dropping changes the scale.** If one of two selected slots is
dropped, the remaining gate is not renormalised, so the output shrinks. Switch's
top-1 has the cleaner behaviour: the whole token is dropped. A renormalising
variant would keep the scale but change the gradient.

**The shared expert can dominate.** The shared branch is a dense path with a
guaranteed gradient, and a large shared branch lets the routed experts idle.
The DeepSeek recipe uses a small shared expert and a separate balance term; the
demo shows the shared output unchanged when the router moves, which is exactly
why it is a stable but potentially lazy path.

**Soft routing saves no arithmetic.** Every expert runs for every token, so the
parameter count is large and the FLOP count is the dense one. It is a way to
scale parameters and specialisation, not a way to scale compute.

**The auxiliary loss balances probabilities more easily than assignments.**
Equation (4) contains $P$, which is differentiable, and $f$, which is a hard
count. Gradient descent moves $P$ smoothly but $f$ only when an argmax flips;
the demo needs 600 steps to even out the load. A very sharp router makes $f$
change in large jumps.

**A gate can be large and still be dropped.** Capacity fills in token order,
not gate order, so a high-confidence token late in the batch can lose its slot
to a low-confidence one that arrived earlier. This keeps the batching simple
and the result reproducible, but it is not the max-weight assignment.

**The dense implementation wastes work on dropped rows.** `forward` evaluates
every selected expert and multiplies the dropped slots by zero. The output is
correct, but the arithmetic saving of the mixture is not realised; a production
kernel gathers the kept rows first.

## Applications

Switch Transformer, GShard and GLaM made top-1 and top-2 routing with capacity
the standard recipe for scaling language models. Mixtral's decoder layers are
top-2 mixtures with eight experts per block. DeepSeekMoE adds a shared expert
and fine-grained expert segmentation, and its auxiliary-loss-free balancing
replaces the term in (4) with a bias update on the router.

Expert Choice is used where perfect expert balance matters more than token
coverage — vision and speech towers with a fixed per-expert batch. Soft MoE
targets the same setting with a fully differentiable assignment, and the
Perceiver's latent cross-attention is the same idea without the expert
ensemble.

Outside language, V-MoE routes image patches and shows the capacity trade at
inference time; multi-task mixtures route by task or by token. The shared-expert
pattern also appears in adapters and conditional computation, where a dense
path carries the common signal and small branches carry the specialisation.

## References

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G.,
  Dean, J. (2017). *Outrageously large neural networks: the sparsely-gated
  mixture-of-experts layer.* ICLR.
- Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M.,
  Shazeer, N., Chen, Z. (2021). *GShard: scaling giant models with conditional
  computation and automatic sharding.* ICLR.
- Fedus, W., Zoph, B., Shazeer, N. (2022). *Switch Transformers: scaling to
  trillion parameter models with simple and efficient sparsity.* JMLR.
  Equations (4) and (5).
- Du, N. et al. (2022). *GLaM: efficient scaling of language models with
  mixture-of-experts.* ICML.
- Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Le, Q.,
  Laudon, J., et al. (2022). *Mixture-of-experts with expert choice routing.*
  NeurIPS. Equations (9) and (10).
- Puigcerver, J., Riquelme, C., Mustafa, B., Houlsby, N. (2023). *From sparse
  to soft mixture-of-experts.* NeurIPS. Equation (11).
- Jiang, A. Q. et al. (2024). *Mixtral of experts.* arXiv:2401.04088.
- Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Zeng, W.,
  Yu, X., Wu, Y., Xie, Z., Li, Y. K., Huang, P., Luo, F., Ruan, C., Sui, Z.,
  Liang, W., Song, S. (2024). *DeepSeekMoE: towards ultimate expert
  specialization in mixture-of-experts language models.* ACL. Equation (12).
- Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Pinto,
  A. S., Keysers, D., Houlsby, N. (2021). *Scaling vision with sparse
  mixture-of-experts.* NeurIPS.
