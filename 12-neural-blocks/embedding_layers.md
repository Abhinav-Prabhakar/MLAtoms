# Embedding layers — tables, compositions and the weights they share

`embedding_layers.py` · Neural blocks · Tier 2 · 7 variants

## Intuition

An embedding is a table lookup: an integer id selects one row of a matrix. It
is the only place in a network where a discrete symbol becomes a vector, and
the backward pass is correspondingly simple — every occurrence of an id adds
its gradient to that row. Everything else in this file is a variation on how
many tables there are, how they are composed, and which matrices are allowed to
be the same object.

The variants answer four practical questions. Where does order come from? Add a
position table, or a position plus a segment table. How do you shrink a large
vocabulary? Factor the table into two low-rank factors. How do you avoid a
second huge matrix at the output? Reuse the input table as the projection. How
do you adapt a pretrained table without touching it? Freeze it and learn a
small bottleneck correction. The demo measures the parameter bill for each
answer and checks that the gradients land in the right rows.

## Variants

| variant | tables | output tied | trainable |
|:--|:--|:--|--:|
| `token` | one token table | no | $Vd$ |
| `position` | token plus position | no | $Vd+Nd$ |
| `type` | token, position, segment | no | $Vd+Nd+Sd$ |
| `factorized` | two factors | no | $Vr+rd$ |
| `weight_tying` | one table, reused at output | yes | $Vd$ |
| `tied_embeddings` | one table, two input streams | no | $Vd$ |
| `adapter_embedding` | frozen token plus bottleneck | no | $dr+rd$ |

$V$ is the vocabulary, $d$ the width, $N$ the number of positions, $S$ the
number of segments and $r$ the factorization or adapter rank.

**`token`** is the base case: one $V\times d$ table and a gather.

**`position`** adds a learned $N\times d$ position table, one row per index. The
default positions are $0,\dots,L-1$, so a batch of equal-length sequences shares
one vector per slot.

**`type`** adds a third table for the segment id — BERT's sentence A or
sentence B — on top of token and position.

**`factorized`** writes the token table as $E=UV^{\top}$ with
$U\in\mathbb R^{V\times r}$ and $V\in\mathbb R^{r\times d}$. For large $V$ and
small $r$ this is a fraction of the full table.

**`weight_tying`** uses the token table as the output projection as well, so
the logits are $hE^{\top}$ and there is no second $V\times d$ matrix.

**`tied_embeddings`** uses one table for two input streams, such as the source
and target of a translation model, which must share a vocabulary.

**`adapter_embedding`** keeps a pretrained table frozen and learns a low-rank
correction to each lookup. Only the two adapter matrices receive gradient.

## Derivation

### The lookup and its gradient

With a one-hot row $o_i$ for id $i$,

$$
e=o_i^{\top}E,\qquad
\frac{\partial L}{\partial E}
=\sum_{(b,t)}o_{i_{bt}}\Bigl(\frac{\partial L}{\partial e_{bt}}\Bigr)^{\top}.
\qquad (1)
$$

In code the sum over one-hots is a scatter-add: `np.add.at(table, ids, grad)`.
Repeated ids accumulate, which the demo checks by looking up id 1 three times
and finding a row of three counts after a backward pass of ones.

### Position and type

$$
e_{bt}=E[i_{bt}]+P[t]+T[s_{bt}].
\qquad (2)
$$

The position index need not equal the token index: the demo passes
`positions=[2, 3, 5]` and checks that the output moves by exactly
$P[2]-P[0]$, $P[3]-P[1]$, $P[5]-P[2]$. When a position vector is shared across
the batch, its gradient is the sum over the batch; the implementation
broadcasts the stored positions to the ids' shape and scatters into that sum
with `np.add.at`, so no separate reduction is needed.

Type is the same argument one table further. At $V=10$, $d=4$, $N=6$ and
$S=2$, the three tables cost $40+24+8=72$ parameters.

### Factorized tables

$$
E=UV^{\top},\qquad U\in\mathbb R^{V\times r},\quad
V\in\mathbb R^{r\times d}.
\qquad (3)
$$

The parameter count is $Vr+rd$ instead of $Vd$, which is smaller whenever
$r<Vd/(V+d)\approx d$. At $V=1000$, $d=16$, $r=4$: $4064$ against $16000$, a
quarter of the table. The lookup is $U[i]\,V$, one row times an $r\times d$
matrix.

The gradients follow from the product rule:

$$
\frac{\partial L}{\partial U[i]}=\frac{\partial L}{\partial e}V^{\top},
\qquad
\frac{\partial L}{\partial V}=U[i]^{\top}\frac{\partial L}{\partial e}.
\qquad (4)
$$

Both factors receive gradient on every lookup, so neither is frozen. The demo
checks (4) against central differences to $5.0\times10^{-11}$.

### Weight tying

For a hidden vector $h$,

$$
\mathrm{logits}=hE^{\top},\qquad e=E[i].
\qquad (5)
$$

One matrix serves two very different roles: a gather at the input and a
$d\times V$ projection at the output. The parameter count drops from $2Vd$ to
$Vd$; at $V=1000$, $d=16$ that saves $16000$ parameters. The gradient is the
sum of the two paths, so a row receives the input scatter plus the output
projection's column.

The demo edits row 2 of the table and confirms that exactly column 2 of the
logits moves, by $h\cdot\mathbf 1$. That is the tie made visible: there is no
separate output matrix to edit.

### Two tied streams

If two streams share a vocabulary, one table can serve both:

$$
e^{(0)}_{bt}=E[i_{bt}],\qquad e^{(1)}_{bt}=E[j_{bt}].
\qquad (6)
$$

The demo verifies that the same ids give identical outputs on stream 0 and
stream 1, that the model owns 40 parameters instead of 80, and that a backward
pass through either stream writes the same rows of the same table.

Tying two streams is not the same as tying input to output. The first shares a
table between two reads; the second shares it between a read and a write. Both
are implemented here as separate variants because they have different failure
modes.

### Adapters

$$
e=E[i]+\mathrm{gelu}\bigl(E[i]W_{\downarrow}\bigr)W_{\uparrow},
\qquad
W_{\downarrow}\in\mathbb R^{d\times r},\quad
W_{\uparrow}\in\mathbb R^{r\times d},
\qquad (7)
$$

with $E$ frozen. The correction is low-rank, so the trainable count is
$2dr$ — 256 against a 16000-row base at $V=1000$, $d=16$, $r=8$. The backward
pass differentiates the bottleneck and never touches $E$:

$$
\frac{\partial L}{\partial W_{\uparrow}}
=H^{\top}\frac{\partial L}{\partial e},
\qquad
\frac{\partial L}{\partial W_{\downarrow}}
=E^{\top}\Bigl(\Bigl(\frac{\partial L}{\partial e}\Bigr)
W_{\uparrow}^{\top}\odot\mathrm{gelu}'(EW_{\downarrow})\Bigr),
\qquad (8)
$$

where $H=\mathrm{gelu}(EW_{\downarrow})$. The demo checks (8) below
$10^{-7}$ and then trains the adapter to halve its input, taking the
loss from 0.6057 to 0.0192 in 300 steps while the frozen table stays
bit-identical.

### Parameter counts

$$
P_{\mathrm{token}}=Vd,\quad
P_{\mathrm{position}}=Vd+Nd,\quad
P_{\mathrm{type}}=Vd+Nd+Sd,
\qquad (9)
$$

$$
P_{\mathrm{factorized}}=Vr+rd,\quad
P_{\mathrm{tying}}=Vd,\quad
P_{\mathrm{adapter}}=2dr \text{ trainable}.
\qquad (10)
$$

At the demo's realistic size, $V=1000$, $d=16$, $N=32$, $S=2$, $r=4$ and
adapter rank 8:

| variant | total | trainable |
|:--|--:|--:|
| `token` | 16000 | 16000 |
| `position` | 16512 | 16512 |
| `type` | 16544 | 16544 |
| `factorized` | 4064 | 4064 |
| `weight_tying` | 16000 | 16000 |
| `tied_embeddings` | 16000 | 16000 |
| `adapter_embedding` | 16256 | 256 |

The table is also a reminder that "total" and "trainable" are different
questions: the adapter owns a large frozen base and a small learnable delta.

### Initialization

The tables are drawn from a normal with standard deviation $1/\sqrt d$, and the
position and type tables at 0.02. The small position scale matters: positions
start as a slight perturbation of the token embedding, so the model learns to
use them rather than being dominated by them at step zero. A zero-initialised
position table is also common and has the same effect at the start.

## Code walkthrough

The lookup has three shapes: a plain gather, a product of two factors, and a
frozen base plus a bottleneck. Keeping them in one function is what makes the
variants comparable:

```python
    def _lookup(self, ids: np.ndarray):
        p = self.parameters
        if self.variant == "factorized":
            out = p["token.left"][ids] @ p["token.right"]
            return out, {"ids": ids, "kind": "factorized"}
        if self.variant == "adapter_embedding":
            base = p["token.table"][ids]
            pre = base @ p["adapter.down"]
            hidden = _gelu(pre)
            out = base + hidden @ p["adapter.up"]
            return out, {"ids": ids, "kind": "adapter", "base": base,
                         "pre": pre, "hidden": hidden}
        return p["token.table"][ids], {"ids": ids, "kind": "table"}
```

Ids are validated once, at the boundary. A float id is rejected rather than
silently truncated, and a negative id is rejected rather than wrapping around
the way a raw NumPy index would:

```python
    def _check_ids(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu":
            raise ValueError("ids must be integers")
        if ids.ndim < 1 or ids.size == 0:
            raise ValueError("ids must be a non-empty array")
        if np.any(ids < 0) or np.any(ids >= self.vocab):
            raise ValueError("an id is out of vocabulary")
        return ids.astype(int)
```

Positions and segments get the same treatment plus a broadcast check, because a
position vector of length $L$ is allowed to apply to every row of a batch:

```python
    def _check_indices(self, values, limit: int, name: str,
                       shape: tuple[int, ...]) -> np.ndarray:
        values = np.asarray(values)
        if values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be integers")
        try:
            broadcast = np.broadcast_to(values, shape)
        except ValueError as error:
            raise ValueError(f"{name} does not broadcast to the ids") from error
        if np.any(broadcast < 0) or np.any(broadcast >= limit):
            raise ValueError(f"{name} out of range")
        return values
```

The compositions are two lines because the tables have the same width. Order
matters for interpretation but not for the arithmetic of addition:

```python
        if positions is not None:
            output = output + self.parameters["position.table"][positions]
        if segments is not None:
            output = output + self.parameters["type.table"][segments]
```

The table gradient is a scatter-add of the flattened upstream gradient. This is
the line that makes repeated ids accumulate instead of overwrite:

```python
        else:
            np.add.at(g["token.table"], ids.ravel(), flat)
```

The factorized backward applies the two product-rule terms from (4), scattering
into the left factor by row and accumulating the right factor over every
lookup:

```python
        if cache["kind"] == "factorized":
            p = self.parameters
            np.add.at(g["token.left"], ids.ravel(),
                      flat @ p["token.right"].T)
            g["token.right"] += p["token.left"][ids.ravel()].T @ flat
```

The adapter backward differentiates the bottleneck and deliberately has no
branch for the base table, because that table is not in `self.gradients`:

```python
        elif cache["kind"] == "adapter":
            p = self.parameters
            pre = cache["pre"].reshape(-1, self.adapter_rank)
            hidden = cache["hidden"].reshape(-1, self.adapter_rank)
            base = cache["base"].reshape(-1, self.dim)
            d_pre = (flat @ p["adapter.up"].T) * _gelu_grad(pre)
            g["adapter.down"] += base.T @ d_pre
            g["adapter.up"] += hidden.T @ flat
```

Position gradients need the batch sum, and broadcasting the stored positions to
the ids' shape is the cheapest way to get it:

```python
        if cache.get("positions") is not None:
            positions = np.broadcast_to(cache["positions"], ids.shape)
            np.add.at(g["position.table"], positions.ravel(), flat)
```

Weight tying is a method rather than a flag because the output projection has a
different input shape. It refuses to run on variants that do not own the tie:

```python
    def logits(self, hidden: np.ndarray) -> np.ndarray:
        if self.variant != "weight_tying":
            raise ValueError("only weight_tying exposes tied logits")
        hidden = np.asarray(hidden, dtype=float)
        if hidden.shape[-1] != self.dim:
            raise ValueError("the hidden width must equal dim")
        return hidden @ self.parameters["token.table"].T
```

## Complexity

For a batch with $T$ ids, width $d$, vocabulary $V$, positions $N$ and rank
$r$:

- **Lookup:** $O(Td)$ time and memory for the output; the gather itself is
  $O(T)$ index arithmetic. The backward scatter is $O(Td)$ plus the number of
  distinct rows touched.
- **Position and type:** one extra $O(Td)$ addition per table, and one scatter
  per table. The tables cost $O(Nd)$ and $O(Sd)$ memory.
- **Factorized:** $O(Trd)$ for the lookup and $O(r(V+d))$ parameters. The saving
  is in memory, not in the per-token arithmetic, which grows with $r$.
- **Weight tying:** saves $Vd$ parameters and one $d\times V$ matrix; the output
  projection itself is still $O(TVd)$.
- **Tied streams:** saves $Vd$ parameters; the shared table's gradient is the
  sum of the streams' scatters.
- **Adapter:** $O(Tdr)$ extra arithmetic and $O(dr)$ parameters; the base table
  is read-only.
- **The demo:** tables of at most 1000 rows and a 300-step adapter fit. Every
  count is small and the file runs in a fraction of a second.

## Failure modes

**An out-of-vocabulary id is a silent wrap in raw NumPy.** Negative indices
index from the end and large ones raise only when the gather runs. The guard
checks both ends and rejects non-integer dtypes, so a float id cannot be
truncated into a plausible row.

**A padding id is a real row.** If padding uses id 0, its gradient accumulates
from every padded position unless the loss mask is applied. The table learns a
padding vector and the row consumes capacity; masking the ids or the gradient
is the caller's job.

**Factorized rank can be too small.** The parameter saving is real, but the
table can no longer represent arbitrary rows. At rank $r$ the row space is
$r$-dimensional; a rank of one is a table of scalar profiles. The demo's $r=4$
against $d=16$ is a quarter-size table, not a free lunch.

**Tying assumes one width.** The input table and the output projection must
share $d$, and the output must have no separate bias unless it is also tied.
A model that changes the hidden width after the trunk cannot use the same
matrix without a projection.

**Tying can make the logits scale with the embedding norm.** The logit for token
$j$ is $h\cdot E[j]$, so frequent tokens with large-norm rows can dominate.
Scaling the logits, normalising the rows, or adding a bias are the usual
patches; none of them is in this minimal implementation.

**Two tied streams need a shared vocabulary.** If source and target ids have
different meanings at the same index, one table is wrong. Tying is a statement
about the vocabulary, not just about the parameter count.

**The adapter's frozen table must be excluded from the optimizer.** This
implementation leaves it out of `self.gradients`, but an external optimizer
that iterates `self.parameters` would still update it once weight decay is
applied. Freezing is a property of the update, not only of the backward pass.

**A tiny adapter can be a no-op at initialization.** If both adapter matrices
start near zero, the correction is negligible and the first steps are spent
learning to leave zero. The demo's random initialization is small but not zero,
which is why the loss moves immediately.

**The type table starts as a small perturbation.** With a large random type
table, the segment identity would dominate token and position at step zero.
The 0.02 initialization is a choice, not a law; zero is the other common
choice.

**Position broadcasting hides a shape bug.** A position vector of length $L$
applies to every row, which is usually intended, but a per-example position
matrix must have the same leading shape as the ids. The broadcast check
rejects the mismatch instead of silently repeating the wrong row.

**A tied weight decay is counted once per parameter, not once per use.** The
gradient has two paths, but the optimiser sees one array. That is the intended
behaviour; the common bug is to register the table twice and decay it twice.

## Applications

Token, position and type tables are the BERT recipe; decoder-only language
models usually keep token and learned position and skip the segment table.
Factorized embeddings are ALBERT's parameter reduction, and weight tying is
standard in language models from Press and Wolf onward, as well as in GPT-2's
input-output matrix.

Tied source-target embeddings are common in multilingual translation when the
tokenizer is shared. Adapter embeddings and bottleneck adapters more generally
are used for domain and language adaptation, where a large pretrained table
stays frozen and a small delta is trained per task.

## References

- Bengio, Y., Ducharme, R., Vincent, P., Jauvin, C. (2003). *A neural
  probabilistic language model.* JMLR. The original learned token table.
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: pre-training
  of deep bidirectional transformers.* NAACL. Token, position and segment
  tables, equation (2).
- Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R. (2020).
  *ALBERT: a lite BERT for self-supervised learning of language
  representations.* ICLR. Factorized embeddings, equations (3) and (4).
- Press, O., Wolf, L. (2017). *Using the output embedding to improve language
  models.* EACL. Weight tying, equation (5).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019).
  *Language models are unsupervised multitask learners.* GPT-2, a tied
  input-output table in practice.
- Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q.,
  Gesmundo, A., Attariyan, M., Gelly, S. (2019). *Parameter-efficient transfer
  learning for NLP.* ICML. Bottleneck adapters.
- Pfeiffer, J., Vulić, I., Gurevych, I., Ruder, S. (2020). *MAD-X: an adapter-
  based framework for multi-task cross-lingual transfer.* ACL. Adapters on
  embeddings.
