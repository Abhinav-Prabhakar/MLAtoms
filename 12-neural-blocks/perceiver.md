# Latent bottlenecks — Perceiver, Set Transformer and the pieces between

`perceiver.py` · Neural blocks · Tier 2 · 4 variants

## Intuition

Attention costs $O(T^2)$ when every token attends to every other. The Perceiver
avoids that by reading the input into a small, fixed array of latents: a
cross-attention layer with $L$ queries and $T$ keys costs $O(LT)$, and once the
sequence is gone, all further processing is $O(L^2)$ — independent of how long
the input was. The latents are the model's working memory, not its input.

That single idea has several arrangements, and the four variants here differ in
where the cross-attention happens. `perceiver` reads once and then runs a stack
of latent self-attention blocks. `latent_array` reads again in every block, so
the latents can revisit the input. `cross_attention_bottleneck` keeps only the
read and a feed-forward update, which isolates what the latent stack buys.
`set_transformer` goes the other way: it first processes the token set with
self-attention, then pools it into learned seeds — attention as a pooling
operator rather than as a compression step.

All four return `(batch, latents, dim)` and are invariant to the order of the
input tokens, because attention with a fixed query is a weighted average over
keys. The demo checks that invariance, checks that permuting the latents
permutes the output, and checks every gradient against finite differences.

## Variants

| variant | cross-attention | latent self-attention | extra |
|:--|:--|:--|:--|
| `perceiver` | once | $N$ shared blocks | cheapest to reason about |
| `latent_array` | every block | $N$ shared blocks | revisits the input |
| `set_transformer` | PMA at the end | SAB over tokens | set input, not a sequence |
| `cross_attention_bottleneck` | once | none | the minimal read |

Every variant shares its projections and FFN across depth, so the depth is free
in parameters. At $d=4$, $L=3$, two heads, hidden 8 and any depth: `perceiver`,
`latent_array` and `set_transformer` cost 248 parameters, and the bottleneck
costs 168.

## Derivation

### Cross-attention is a learned read

With latents $Z\in\mathbb R^{L\times d}$ and tokens $X\in\mathbb R^{T\times d}$,
per head of width $d_h=d/H$:

$$
Q=ZW_q,\quad K=XW_k,\quad V=XW_v,\quad
A=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_h}}\right),
\qquad (1)
$$

$$
\tilde Z=Z+\bigl(AV\bigr)W_o.
\qquad (2)
$$

The output row $i$ is a convex combination of the value rows: $A_{ij}\ge 0$ and
$\sum_j A_{ij}=1$. The demo runs one latent and one head and checks both facts,
plus that the cached context is exactly $AV$. Nothing in (1) depends on the
order of the token axis, so $\tilde Z$ is unchanged by a permutation of $X$ —
the first invariant.

### Four arrangements

Write one latent block as self-attention (3) followed by a feed-forward update
(4), both pre-norm and residual:

$$
Z\leftarrow Z+\mathrm{MHA}(Z,Z),\qquad
Z\leftarrow Z+\mathrm{FFN}(\mathrm{LN}(Z)).
\qquad (3)
$$

Then the variants are

$$
\text{perceiver}:\; Z_0=\mathrm{Cross}(Z,X),\quad
Z_{n+1}=\mathrm{Block}(Z_n);
\qquad (4)
$$

$$
\text{latent\_array}:\quad
Z_{n+1}=\mathrm{Block}\bigl(\mathrm{Cross}(Z_n,X)\bigr);
\qquad (5)
$$

$$
\text{bottleneck}:\quad
Z_1=\mathrm{FFN}\bigl(\mathrm{Cross}(Z_0,X)\bigr);
\qquad (6)
$$

where Cross includes the residual around the read. The difference between (4)
and (5) is the difference between reading the input once and reading it at
every step: (4) is cheaper and cannot recover a detail it missed; (5) pays
$O(NTLd)$ and can.

### Set Transformer: attention as pooling

For a set, the natural layer is self-attention with no positional term:

$$
\mathrm{SAB}(X)=X+\mathrm{MHA}(X,X),\qquad X\leftarrow X+\mathrm{FFN}(\mathrm{LN}(X)).
\qquad (7)
$$

The output is permutation-equivariant, so it is still a set. To produce a fixed
number of outputs, pooling by multi-head attention (PMA) uses $L$ learned seeds
$Z$ as queries against the encoded set:

$$
Z\leftarrow Z+\mathrm{MHA}(Z,\,X_{\text{encoded}}).
\qquad (8)
$$

That is exactly the cross-attention of (1) with the seeds in place of the
latents, and it is why `set_transformer` shares the latent machinery: seeds and
latents play the same role, one after the other. The demo permutes the seeds
and confirms the output is permuted the same way.

### Permutation invariance, precisely

For a permutation matrix $P$ acting on the token axis,

$$
\mathrm{Cross}(Z,PX)=\mathrm{Cross}(Z,X),\qquad
\mathrm{SAB}(PX)=P\,\mathrm{SAB}(X).
\qquad (9)
$$

The first holds because the softmax normalises over keys and the sum
$\sum_j A_{ij}V_{j:}$ is over a set. The second holds because self-attention is
applied identically to every token; composing the two, PMA is invariant. The
demo's worst deviation over the four variants is $6.2\times10^{-15}$, which is
floating-point noise in the softmax sums.

Note what invariance costs: no position information can enter, so these blocks
are for sets and bags, not for sequences with order. The Perceiver adds
position embeddings to the input when order matters; that is a separate,
explicit choice, and it is not in this file.

### Parameter counts and depth sharing

Each attention block owns four $d\times d$ projections with biases,
$4(d^2+d)$. The FFN owns $(dh+h)+(hd+d)$. The latent array owns $Ld$. So

$$
P_{\text{perceiver}}=2\cdot4(d^2+d)+(dh+h)+(hd+d)+Ld,
\qquad (10)
$$

with one attention block for the cross-attention and one for the shared latent
self-attention; `latent_array` has the same count because its cross-attention
is the same module reused. `set_transformer` swaps the self-attention for a SAB
and adds a PMA, which is again $4(d^2+d)$ — hence the same total. The bottleneck
drops the self-attention, which is the 80-parameter difference in the demo
table.

Because depth reuses these modules, a depth-6 Perceiver owns exactly the same
248 parameters as a depth-1 one. The demo checks that and also checks that the
two outputs differ, so the depth is doing something even though it costs
nothing.

### The backward pass

Only the attention block is new relative to the transformer chapter. For the
softmax matrix $A$ and upstream $dC$ on the context,

$$
dA=dC\,V^{\top},\qquad dV=A^{\top}dC,\qquad
dS=A\odot\Bigl(dA-\sum_j A_{ij}dA_{ij}\Bigr),
\qquad (11)
$$

and then, since $S=QK^{\top}/\sqrt{d_h}$,

$$
dQ=\frac{dS\,K}{\sqrt{d_h}},\qquad
dK=\frac{dS^{\top}Q}{\sqrt{d_h}}.
\qquad (12)
$$

The projections follow: $dW_q=Q_{\text{in}}^{\top}dQ$ and so on. When the query
and key streams are the same tensor, as in self-attention, the two gradients
add before the LayerNorm backward pass; when they differ, as in cross-attention
and PMA, they go to different places. The demo's worst errors are
$2.7\times10^{-8}$ for the three full variants and $1.3\times10^{-9}$ for the
bottleneck.

### Complexity

For $T$ tokens, $L$ latents, $d$ channels and $N$ blocks:

- **Perceiver:** $O(TLd)$ for the one read, then $O(NL^2d)$ for the latent
  stack. Once $L$ is fixed, the cost is linear in the input length.
- **latent_array:** $O(NTLd)$ for the repeated reads, plus the latent stack.
  Still linear in $T$, with a larger constant.
- **bottleneck:** $O(TLd + Ldh)$; the cheapest variant and the least
  expressive.
- **set_transformer:** $O(NT^2d)$ for the SAB stack, then $O(TLd)$ for PMA. It
  is the only variant that pays the quadratic token cost, which is the price of
  letting tokens interact before pooling.
- **The demo:** 5 tokens, 3 latents, width 4 and depth 2, so every product is
  tiny; the finite-difference loops dominate, and the file runs in well under a
  second.

## Code walkthrough

The attention helper is the whole model in miniature. Queries and keys are
projected separately, scores are scaled, and the context is merged before the
output projection:

```python
    def _attend(self, queries: np.ndarray, keys: np.ndarray, prefix: str):
        p = self.parameters
        q = self._heads(queries @ p[f"{prefix}.q.W"] + p[f"{prefix}.q.b"])
        k = self._heads(keys @ p[f"{prefix}.k.W"] + p[f"{prefix}.k.b"])
        v = self._heads(keys @ p[f"{prefix}.v.W"] + p[f"{prefix}.v.b"])
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attention = _softmax(scores)
        context = attention @ v
        out = (self._merge(context) @ p[f"{prefix}.o.W"]
               + p[f"{prefix}.o.b"])
```

The Perceiver forward is then three lines of arrangement: one read, then the
shared latent loop. The caches are per step because the parameters are shared
but the activations are not:

```python
        if self.variant == "perceiver":
            latent, cache["cross"] = self._cross_layer(latent, tokens)
            for _ in range(self.depth):
                latent, self_cache = self._self_layer(latent, "self")
                latent, ffn = self._ffn(latent)
                cache["steps"].append({"self": self_cache, "ffn": ffn})
            return latent, cache
```

PMA is the same attention call with two different normalised streams. Keeping
the query and key caches separate is what lets the backward pass send the two
gradients to the seeds and to the encoded set respectively:

```python
    def _pool_layer(self, queries: np.ndarray, keys: np.ndarray):
        """A sublayer whose query and key streams differ, as in PMA."""
        norm_q, cache_q = _layer_norm(queries)
        norm_k, cache_k = _layer_norm(keys)
        out, attention = self._attend(norm_q, norm_k, "pma")
        return queries + out, {"norm_q": cache_q, "norm_k": cache_k,
                               "attend": attention}
```

The FFN caches both the normalised input and the LayerNorm statistics. The
normalised tensor is what the weight gradient multiplies; the statistics are
what the backward pass through the norm needs:

```python
    def _ffn(self, x: np.ndarray):
        p = self.parameters
        norm, norm_cache = _layer_norm(x)
        pre = norm @ p["ffn.in"] + p["ffn.in.b"]
        hidden = _gelu(pre)
        out = hidden @ p["ffn.out"] + p["ffn.out.b"]
        return x + out, {"norm": norm, "norm_cache": norm_cache,
                         "pre": pre, "hidden": hidden}
```

The softmax backward is (11) and (12) in four lines. The subtraction of the
weighted mean is what makes the Jacobian act on the zero-sum subspace:

```python
        d_attention = d_context @ v.transpose(0, 1, 3, 2)
        d_v = attention.transpose(0, 1, 3, 2) @ d_context
        d_scores = attention * (
            d_attention - (attention * d_attention).sum(
                axis=-1, keepdims=True))
        d_scores = d_scores * self.scale
```

Walking the latent-array stack backwards requires accumulating the token
gradient across blocks, because every block reads the same input:

```python
        elif self.variant == "latent_array":
            d_latent = gradient
            for step in reversed(cache["steps"]):
                d_latent = self._ffn_backward(d_latent, step["ffn"])
                d_latent = self._same_backward(
                    d_latent, step["self"], "self")
                d_latent, d_step = self._cross_backward(
                    d_latent, step["cross"])
                d_tokens += d_step
```

The latent parameter is shared across the batch, so its gradient is a sum over
the batch axis, exactly as an embedding table sums over occurrences:

```python
    def _scatter_latent(self, d_latent: np.ndarray) -> None:
        self.gradients["latent"] += d_latent.sum(axis=0)
```

## Failure modes

**The bottleneck can throw away the answer.** A single read with $L$ latents is
a projection onto an $L$-dimensional summary; if the task depends on a detail
that no latent query looks at, no later block can recover it. `latent_array`
exists to mitigate this, at a linear cost in $T$ per block.

**Fixed latents are a learned prior, not data.** The initial latent array is a
parameter, so it does not adapt per example. That is the point — it is a fixed
set of queries — but it also means the model cannot decide, per input, how many
latents to use.

**Permutation invariance forbids order.** Feeding a sequence to any of these
blocks without adding positional information gives a bag-of-tokens model. The
`set_transformer` is explicit about this; the Perceiver's input embedding is
where positions must be added, and this file does not have one.

**Softmax over many keys saturates.** With $T$ large and scores of moderate
scale, the attention rows can become nearly one-hot, and the gradient through
the softmax shrinks. The demo's finite-difference errors of $10^{-8}$ rather
than $10^{-10}$ are a hint of that flatness; a real model uses temperature or
QK normalisation.

**Depth sharing saves parameters and limits capacity.** Reusing one block $N$
times gives a recurrent computation, not $N$ independent layers. That is
cheaper and often enough, but it cannot represent a different function at each
depth.

**PMA's seeds are unordered.** Permuting the seeds permutes the pooled output,
which is fine when the output is a set but wrong if the caller assumes a fixed
semantic order. Nothing in the layer assigns meaning to seed $i$.

**Cross-attention keys are unnormalised in this implementation.** `_cross_layer`
normalises the queries only. That matches the reference Perceiver, but it means
the key stream must already be on a reasonable scale; adding a key norm is a
one-line change and a different model.

**LayerNorm over channels assumes the width is the feature axis.** Every norm
here reduces over `dim`. A layout with channels first would normalise the wrong
axis and still run, producing plausible but wrong numbers.

**A wrong gradient shape is the only guard on the output.** The backward pass
checks the latent shape but cannot check that the upstream gradient came from
the right loss; a transposed or mis-broadcast gradient produces silent
nonsense.

**The number of latents is a hyperparameter with no schedule.** Too few and the
read is lossy; too many and the self-attention cost grows quadratically in $L$.
The demo's $L=3$ against $T=5$ compresses almost nothing, which is convenient
for checking shapes and misleading as a capacity guide.

## Applications

The Perceiver family is used for modalities whose inputs are large and
unstructured: images at high resolution, audio spectrograms, point clouds and
multi-modal arrays, where the token count makes full self-attention
prohibitive. Perceiver IO extends the same latent array to arbitrary outputs by
cross-attending from latents to output queries.

The Set Transformer is used wherever the input is genuinely a set — multiple
instance learning, point-cloud classification, amortised clustering, and
attention-based pooling in retrieval models. PMA in particular is a drop-in
replacement for mean or max pooling when the pooled representation should be
learned rather than fixed.

## References

- Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., Carreira, J.
  (2021). *Perceiver: general perception with iterative attention.* ICML.
  Equations (1), (2) and the latent stack.
- Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C., Ding, D.,
  Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénaff, O.,
  Botvinick, M., Zisserman, A., Vinyals, O., Carreira, J. (2022).
  *Perceiver IO: a general architecture for structured inputs and outputs.*
  ICLR. The output-query extension of the same bottleneck.
- Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., Teh, Y. W. (2019). *Set
  Transformer: a framework for attention-based permutation-invariant neural
  networks.* ICML. SAB, PMA, equations (7) and (8).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., Polosukhin, I. (2017). *Attention is all you need.* NeurIPS. The
  attention block and its backward pass, equations (11) and (12).
