# Graph neural blocks — eight ways to pass messages along edges

`gnn_blocks.py` · Neural blocks · Tier 2 · 8 variants

## Intuition

A graph layer replaces the fixed neighbour window of a convolution with an
edge list. Every node collects something from its neighbours, and every variant
in this file is a different answer to two questions: what is a message, and how
are messages combined?

The eight answers span the design space. `gcn` combines by a degree-normalised
sum, which is a low-pass filter on the graph. `gat` combines by learned
attention, so the weights depend on the endpoints. `gin` sums and says the sum
must be injective, which makes the layer as discriminating as the
Weisfeiler–Lehman test. `sage` concatenates the node with the mean of its
neighbours, trading sum's degree information for scale stability. `mpnn`
separates the edge message from the node update so edge features enter the
first stage. `edge_conv` takes a per-coordinate maximum over messages. The
`transformer_conv` adds an edge term to the attention score. `pna` concatenates
four aggregators at three degree scales.

All eight are permutation-equivariant: relabel the nodes and the output is
relabelled the same way. The demo checks that for every variant, then checks
gradients against central differences.

## Variants

| variant | message | combine | edge features |
|:--|:--|:--|:--|
| `gcn` | $Wx_j$ | normalised sum | no |
| `gat` | $W x_j$ | attention | no |
| `gin` | $x_j$ | sum | no |
| `sage` | $W_n x_j$ | mean, then concat | no |
| `mpnn` | $\phi(x_j, x_i, e_{ji})$ | mean | yes |
| `edge_conv` | $\mathrm{gelu}(x_i+\Theta e_{ji})$ | per-coordinate max | yes |
| `transformer_conv` | $Wx_j$ | edge-biased attention | yes |
| `pna` | $x_j$ | 4 aggregators, 3 scales | no |

Edges are directed `(source, target)` pairs. A message from $j$ to $i$ is
written $j\to i$, and a symmetric graph must list both directions; the demo
does exactly that and ends with 24 directed edges over 6 nodes.

## Derivation

### One template

Write the layer as

$$
m_i=\square_{j\in N(i)}\phi\bigl(x_j,x_i,e_{ji}\bigr),\qquad
x_i'=\psi\bigl(x_i,m_i\bigr),
\qquad (1)
$$

where $\square$ is a permutation-invariant pool — sum, mean, max, or an
attention-weighted sum. This is the message-passing framework of Gilmer et
al., and every variant below is a choice of $\phi$, $\square$ and $\psi$.

### GCN: a normalised sum

$$
H'=\sigma\Bigl(\hat D^{-1/2}\hat A\hat D^{-1/2}XW+b\Bigr),\qquad
\hat A=A+I,
\qquad (2)
$$

with $\hat D$ the degree of $\hat A$. The self loop guarantees degree at least
one, so no node is dropped, and the two-sided scaling keeps the row sums near
one. The matrix $\hat D^{-1/2}\hat A\hat D^{-1/2}$ is similar to the random-walk
matrix $\hat D^{-1}\hat A$, so its eigenvalues lie in $[-1,1]$; the demo's graph
gives $[-0.200,1.000]$. A single layer therefore cannot amplify a signal, which
is the stability argument for (2); stacking many layers still drives every
node towards the same vector — the oversmoothing failure.

The backward pass needs only two transposes:

$$
\frac{\partial L}{\partial W}=X^{\top}\hat A^{\top}dP,\qquad
\frac{\partial L}{\partial X}=\hat A^{\top}dP\,W^{\top},\qquad
dP=\nabla\sigma\odot G.
\qquad (3)
$$

### GAT: attention over neighbours

$$
e_{ij}=\mathrm{LeakyReLU}\bigl(a_s^{\top}Wx_j+a_t^{\top}Wx_i\bigr),\quad
\alpha_{ij}=\frac{\exp e_{ij}}{\sum_{k\in N(i)\cup\{i\}}\exp e_{ik}},
\qquad (4)
$$

$$
x_i'=\sigma\Bigl(\sum_{j}\alpha_{ij}Wx_j\Bigr).
\qquad (5)
$$

The score is additive in a source term and a target term, so it factorises into
one $n\times n$ matrix per head without building $n^2$ feature vectors. Scores
of non-edges are set to $-\infty$ before the softmax, which makes their weights
exactly zero. With two heads of width 2 on the demo graph, each row sums to one
over five allowed entries on average — four neighbours plus the self loop.

### GIN: sum, because sum is injective

$$
h_i'=\mathrm{MLP}\Bigl((1+\epsilon)x_i+\sum_{j\in N(i)}x_j\Bigr).
\qquad (6)
$$

The claim behind GIN is that a sum over a multiset is injective up to
permutation, so with a sufficiently expressive MLP the layer separates any two
neighbourhoods the Weisfeiler–Lehman test separates. Mean does not: it forgets
the degree. The demo makes that concrete — node 0 has four neighbours, so the
sum is exactly four times the mean, and the two aggregators differ only by a
scalar the next layer must undo. $\epsilon$ is a learned scalar; its gradient
is the inner product of the upstream gradient with $x$.

### GraphSAGE: keep the node and the mean apart

$$
h_i'=\sigma\Bigl(W_o\,\bigl[\,W_s x_i\;\|\;
W_n\,\mathrm{mean}_{j\in N(i)}x_j\,\bigr]\Bigr).
\qquad (7)
$$

Concatenation rather than addition is the point: the layer can learn a
different weight for "self" and "neighbourhood" instead of forcing them through
one matrix. The neighbour term is scaled by the degree, so a node with 100
neighbours does not produce a 100× larger vector.

### MPNN: an edge network, then a node network

$$
m_{ji}=\phi_e\bigl(x_j,x_i,e_{ji}\bigr),\qquad
\bar m_i=\frac{1}{|N(i)|}\sum_{j} m_{ji},\qquad
x_i'=\phi_v\bigl(x_i,\bar m_i\bigr).
\qquad (8)
$$

Here $\phi_e$ and $\phi_v$ are small MLPs. The edge features enter before the
pool, which is what `gcn` and `gat` cannot do; the cost is a per-edge forward
pass. The demo's 24 edges produce 24 message vectors, and the cache keeps all
of them for the backward pass. Note that this implementation pools over
*sources*, so the mean is taken over outgoing edges; the naming follows the
edge direction rather than the usual "incoming" convention.

### EdgeConv: a per-coordinate maximum

$$
m_{ji}=\mathrm{gelu}\bigl(x_i+\Theta e_{ji}\bigr),\qquad
x_i'=\sigma\Bigl(W_sx_i+W_o\max_{j}m_{ji}\Bigr),
\qquad (9)
$$

with the max taken coordinate by coordinate. The max is not injective: it
keeps one message per feature and discards the rest, and the winner can differ
between coordinates. That is deliberate for point clouds, where the nearest
neighbours are samples of a surface and the extremal direction is the
informative one. The backward pass routes each coordinate's gradient to the
edge that won it.

### TransformerConv: attention with an edge term

$$
e_{ij}=\frac{a_s^{\top}Wx_j+a_t^{\top}Wx_i+w^{\top}e_{ji}}{\sqrt{d_h}},
\qquad (10)
$$

and the rest of (4)–(5). Dividing by $\sqrt{d_h}$ keeps the score variance
independent of the head width. The demo feeds zero edge features and checks
that both the attention matrix and the output change — the edge term is live,
not multiplied away.

### PNA: four aggregators at three scales

$$
\mathrm{agg}=\Bigl[\;\mu,\;\max,\;\min,\;\sigma\;\Bigr],\qquad
s_{\text{amp}}=\frac{\log(1+d)}{\overline{\log(1+d)}},\quad
s_{\text{att}}=\frac{\overline{\log(1+d)}}{\log(1+d)},
\qquad (11)
$$

$$
h_i'=\sigma\Bigl(W_sx_i+W_o\bigl[\mathrm{agg}\cdot s\bigr]_{4\times3}\Bigr).
\qquad (12)
$$

Twelve blocks per node, each of width $F$: 48 features at $F=4$. The scalers
are what make the layer degree-aware without discarding the degree — the same
mean is presented amplified, unchanged, and attenuated. The mean of the logged
degrees is taken over the batch, so the scalers depend on the graph, not on a
global constant.

### Backward passes

Two derivatives carry most of the work. A scatter-add is self-adjoint: if
$m_i=\sum_{j\to i}\phi_j$ then $\partial L/\partial\phi_j$ is the upstream
gradient at $i$. The softmax with a LeakyReLU inside has

$$
dS=\alpha\odot\Bigl(dA-\sum_{k}\alpha_{ik}dA_{ik}\Bigr)\odot
\mathrm{LeakyReLU}'(S),
\qquad (13)
$$

and the max in (9) and the standard deviation in (12) route to their winners:

$$
\frac{\partial\sigma_i}{\partial x_j}=\frac{x_j-\mu_i}{n\,\sigma_i}.
\qquad (14)
$$

The demo checks every parameter and the input against central differences for
all eight variants, with worst errors between $6.2\times10^{-11}$ and
$4.9\times10^{-10}$. Max and argmax are piecewise linear, so as long as no two
messages tie they are differentiable in the ordinary sense.

### Permutation equivariance

For a permutation matrix $P$,

$$
f\bigl(PX,\,PAP^{\top}\bigr)=P\,f(X,A).
\qquad (15)
$$

Every operation used here is index-free: gathers by edge endpoint, row sums,
elementwise maxima, and linear maps applied per node. The demo relabels the
demo graph with `perm` and `argsort(perm)` and measures the largest output
difference per variant. All eight are below $5\times10^{-16}$, and `mpnn` and
`edge_conv` are exactly zero because their aggregation is a sum and a max over
an unchanged multiset.

### Parameter counts

At input width $F$, hidden width $H$, $E$ edge features and $k$ heads of width
$d_h=H/k$:

| variant | parameters | at $F=4,H=4,k=2,E=3$ |
|:--|:--|--:|
| `gcn` | $FH+H$ | 20 |
| `gat` | $k(Fd_h+2d_h)+H$ | 28 |
| `gin` | $1+(FH+H)+(HH+H)$ | 41 |
| `sage` | $3(FH+H)+HH+H$ | 76 |
| `mpnn` | $2(FH)+EH+H+2(HH+H)$ | 88 |
| `edge_conv` | $EH+H+(FH+H)+FH$ | 52 |
| `transformer_conv` | $k(Fd_h+2d_h)+Ek+(HH+H)$ | 50 |
| `pna` | $FH+H+12FH+H$ | 216 |

The ordering is the price of expressivity: a normalised sum is 20 numbers, a
twelve-feature aggregator is 216. None of these counts includes the MLPs that
real implementations stack between layers.

## Code walkthrough

The whole file is one class with a `forward` that dispatches on the variant and
a `backward` that mirrors it. Validation happens once, at the boundary, and
rejects an edge endpoint that would silently wrap in NumPy:

```python
        edges = np.asarray(edges)
        if edges.dtype.kind not in "iu" or edges.ndim != 2 \
                or edges.shape[1] != 2 or edges.shape[0] < 1:
            raise ValueError("edges must be a non-empty (m, 2) integer array")
        if np.any(edges < 0) or np.any(edges >= nodes):
            raise ValueError("an edge endpoint is out of range")
```

The normalisations are three lines each, and keeping them separate is what
makes (2) and (7) comparable. `_normalized` is the symmetric GCN form,
`_row_mean` is the GraphSAGE mean:

```python
def _normalized(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    return inverse[:, None] * adjacency * inverse[None, :]
```

GCN is then a dense matrix product. Building $\hat A$ explicitly is $O(n^2)$
and fine for the demo; a sparse implementation would gather rows instead:

```python
        if self.variant == "gcn":
            adjacency = _normalized(_adjacency(edges, nodes, True))
            pre = adjacency @ x @ p["gcn.W"] + p["gcn.b"]
            out = _gelu(pre)
            return out, {"adjacency": adjacency, "x": x, "pre": pre}
```

GAT computes the score as an outer sum of two projections per head. The mask is
applied before the softmax, so non-edges get weight exactly zero rather than a
small one:

```python
            for head in range(self.heads):
                projected = x @ p[f"attn{head}.W"]
                scores = (projected @ p[f"attn{head}.src"])[:, None] \
                    + (projected @ p[f"attn{head}.dst"])[None, :]
                masked = np.where(allowed, _leaky(scores), -np.inf)
                attention = _softmax(masked)
                parts.append(attention @ projected)
                heads.append({"projected": projected, "scores": scores,
                              "attention": attention})
```

PNA's aggregation is a loop over nodes because each node needs four reductions
and the argmax indices for two of them. The winners are stored as node ids so
the backward pass can scatter into them directly:

```python
        for node, rows in enumerate(neighbours):
            if rows.size == 0:
                continue
            block = x[rows]
            mean[node] = block.mean(axis=0)
            maximum[node] = block.max(axis=0)
            minimum[node] = block.min(axis=0)
            deviation[node] = block.std(axis=0)
            amax[node] = rows[block.argmax(axis=0)]
            amin[node] = rows[block.argmin(axis=0)]
```

The GCN backward is the two transposes of (3). The `.T` on the adjacency is the
only place the graph enters the backward pass at all:

```python
            d_pre = gradient * _gelu_grad(cache["pre"])
            g["gcn.W"] += cache["x"].T @ (cache["adjacency"].T @ d_pre)
            g["gcn.b"] += d_pre.sum(axis=0)
            return cache["adjacency"].T @ d_pre @ p["gcn.W"].T
```

The attention backward applies the softmax Jacobian (13) once per head. Note
that `d_projected` receives two paths: through the attention output and through
the two score projections:

```python
                d_attention = d_head @ projected.T
                d_projected = attention.T @ d_head
                d_scores = attention * (
                    d_attention - (attention * d_attention).sum(
                        axis=1, keepdims=True))
                d_scores = d_scores * _leaky_grad(data["scores"])
                d_projected += (d_scores.sum(axis=1)[:, None]
                                * p[f"attn{head}.src"][None, :])
                d_projected += (d_scores.sum(axis=0)[:, None]
                                * p[f"attn{head}.dst"][None, :])
```

## Complexity

For $n$ nodes, $m$ directed edges, input width $F$, hidden width $H$ and $k$
heads:

- **Dense GCN:** $O(n^2F+n^2H)$ time for the adjacency product and
  $O(n^2)$ memory for $\hat A$. A sparse implementation is $O(mH)$.
- **GAT and TransformerConv:** $O(n^2H)$ for the score matrix even though only
  $m$ entries survive the mask, plus $O(mH)$ for the value product. The demo
  builds the dense matrix because it is short; production code uses a sparse
  neighbourhood.
- **GIN, SAGE, PNA:** $O(mF)$ for the aggregation and $O(nFH)$ for the
  projection. PNA's loop is $O(mF)$ plus a Python iteration over nodes.
- **MPNN and EdgeConv:** $O(mFH)$ for the per-edge messages, the dominant cost
  when $H\gg F$. Both cache every message, so memory is $O(mH)$.
- **Backward:** the same order as the forward pass, with one extra gather or
  scatter per edge. The demo's finite-difference loop is the slow part: it
  perturbs every parameter of every variant one at a time.
- **The demo:** 6 nodes, 24 edges and widths of 4; the whole file runs in well
  under a second.

## Failure modes

**A directed edge list silently halves a symmetric graph.** Every variant here
treats `(j, i)` as one direction. Passing each undirected edge once gives a
different, asymmetric operator, and the normalisation in (2) is then wrong for
the missing side. The demo adds both directions explicitly, which is why node 0
ends up with four neighbours.

**GCN oversmooths.** Repeated application of (2) is a power of a low-pass
filter; after enough layers every node converges to the same vector and the
readout cannot tell them apart. Residual connections, jumping knowledge, or
keeping the layer count small are the usual mitigations — none of them is in
this single layer.

**GIN's sum grows with degree.** A node with 1000 neighbours produces a
neighbour term 1000× larger than a leaf's, so deep stacks need normalisation or
a learned degree scale. Mean avoids the growth and loses the degree; the
trade-off is a modelling choice, not a bug.

**Max aggregators are non-injective.** `edge_conv` and the `max` row of PNA
keep one message per coordinate and throw the rest away. Two different
neighbourhoods can produce the same maximum, and the layer cannot distinguish
them. Ties make the argmax gradient ambiguous: NumPy's `argmax` picks the first
winner, so the tie is broken by edge order.

**Isolated nodes lose their neighbourhood.** The guards here avoid division by
zero — GCN adds self loops, SAGE and PNA divide by a clipped degree — but an
isolated node's aggregate is zero, so it contributes only through the self
path. In PNA the node's degree scaler is also zero-log, which is why
`log1p(degree)` is used instead of `log(degree)`.

**PNA's scalers are batch statistics.** The mean logged degree is computed over
the nodes present, so the same node in a different graph gets different
features. That is intended, but it means the layer is not a function of the
node alone.

**Standard deviation has a singular derivative.** When every neighbour has the
same value the spread is zero and (14) divides by zero. The guard skips that
term, which is the correct limit only when the upstream gradient on the
deviation is also zero; otherwise the true derivative is undefined.

**Attention needs a non-empty allowed set.** A row with every score masked is a
softmax over all $-\infty$, which is NaN. The self loop in `gat` and
`transformer_conv` guarantees at least one allowed entry; remove it and the
guard has to change.

**An edge feature on a variant that has no edge path is a dropped argument.**
`_check` rejects it rather than ignoring it, because silently discarding edge
information is the kind of bug that only shows up as a worse model.

**Two heads must fit the hidden width.** `hidden % heads` is checked, so a
width of 3 with 2 heads is refused instead of leaving a dead dimension.

**The dense adjacency is a demo shortcut.** Building $n^2$ entries for a graph
with $m\ll n^2$ edges wastes memory and time; the derivations are sparse, and a
real implementation should be too.

## Applications

GCN and GAT are the citation-network and knowledge-graph baselines. GIN's
injectivity argument made it the standard for molecular property prediction and
for graph classification benchmarks where WL discrimination matters. GraphSAGE
was designed for inductive recommendation, where new nodes appear after
training. MPNN is the chemistry workhorse — bond type and distance are edge
features, and the edge network is where they enter. EdgeConv and its DGCNN
descendant are point-cloud and mesh layers, where the "graph" is a k-nearest-
neighbour graph rebuilt at every layer. TransformerConv and its Graphormer
relatives bring attention and edge bias to molecular and long-range tasks. PNA
targets molecular datasets whose degree distributions vary by orders of
magnitude across graphs.

## References

- Kipf, T. N., Welling, M. (2017). *Semi-supervised classification with graph
  convolutional networks.* ICLR. Equations (2) and (3).
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y.
  (2018). *Graph attention networks.* ICLR. Equations (4) and (5).
- Xu, K., Hu, W., Leskovec, J., Jegelka, S. (2019). *How powerful are graph
  neural networks?* ICLR. GIN and the Weisfeiler–Lehman argument, equation (6).
- Hamilton, W. L., Ying, R., Leskovec, J. (2017). *Inductive representation
  learning on large graphs.* NeurIPS. GraphSAGE, equation (7).
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., Dahl, G. E.
  (2017). *Neural message passing for quantum chemistry.* ICML. The
  template, equation (1), and MPNN, equation (8).
- Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., Solomon, J. M.
  (2019). *Dynamic graph CNN for learning on point clouds.* ACM TOG. EdgeConv,
  equation (9).
- Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., Sun, Y. (2021). *Masked
  label prediction: unified message passing model for semi-supervised
  classification.* IJCAI. TransformerConv, equation (10).
- Corso, G., Cavalleri, L., Beaini, D., Liò, P., Veličković, P. (2020).
  *Principal neighbourhood aggregation for graph nets.* NeurIPS. PNA,
  equations (11) and (12).
