# Positional encoding — how a bag of tokens learns about order

`positional.py` · Neural blocks · Tier 2 · 8 variants

## Intuition

Attention is a weighted average over a set. Permute the tokens and every weight
permutes with them, because nothing in $q\cdot k$ refers to an index. A
transformer without positional information therefore cannot tell "a then b" from
"b then a". The demo measures exactly that: swap the values of two markers and
the readout of the `no_position` model is bit-identical in 48 pairs out of 48.

There are three places to put order back, and the variants divide along them:

- **An additive table.** Add a vector to each token before the first projection
  (`sinusoidal` with a formula, `learned` with a parameter).
- **A rotation.** Rotate queries and keys by an angle proportional to the
  position, which makes the score depend on the distance (`rope`, `rope_ntk`).
- **A score bias.** Add a term that depends only on the two positions
  (`alibi`, `relative_bias`, `t5_bias`).

`no_position` does none of them and is the control. All three channels are
visible in one line:

$$
s_{ij}=\frac{\bigl((x_i+t_i)W_q\bigr)\cdot\bigl((x_j+t_j)W_k\bigr)}
{\sqrt d}+b_{ij},
\qquad (1)
$$

where $t$ is the table (zero for the bias variants) and $b$ is the bias (zero
for the table variants). RoPE replaces the two projections with rotated copies,
which is a different but equivalent way of making the score depend on $i-j$.

The demo applies each scheme to a fixed random attention readout and asks how
much the readout moves when two marker values are exchanged. Six schemes move
on every pair. `t5_bias` collides on 19 of 48 pairs, because two distances can
fall in the same bucket. `no_position` never moves: the readout is a function of
the multiset, so no downstream model can recover the order from it.

## Variants

| variant | mechanism | applied to | parameters |
|:--|:--|:--|--:|
| `sinusoidal` | fixed sin/cos table | tokens | 0 |
| `learned` | trainable table | tokens | 128 |
| `rope` | per-pair rotation | queries and keys | 0 |
| `rope_ntk` | rotation, stretched base | queries and keys | 0 |
| `alibi` | linear distance penalty | scores | 0 |
| `relative_bias` | learned per-distance bias | scores | 27 |
| `t5_bias` | shared log-bucket table | scores | 8 |
| `no_position` | nothing | — | 0 |

The counts are for the demo: the learned table is 16 positions by 8 channels,
the relative bias has 3 heads and 9 distances, and the T5 table has 8 buckets.

**`sinusoidal`** uses no parameters and can be evaluated at any position, so it
extrapolates past the longest sequence it was designed for. Its rows have
constant norm and its dot products depend only on the gap.

**`learned`** is a $N\times d$ table, one row per position, trained with
everything else. It cannot extrapolate: a position beyond $N$ has no row. The
demo's table is initialised at standard deviation 0.02, which is why its readout
moves the least of the exact schemes.

**`rope`** leaves the token features alone and rotates each query and key by an
angle proportional to its position, two coordinates at a time. The value is not
rotated. The score between positions $i$ and $j$ then depends only on $j-i$.

**`rope_ntk`** is the same rotation with the base stretched by
$s^{d/(d-2)}$ for a context-extension factor $s$. At dimension 8 and $s=8$ the
base goes from 10000 to 160000, and the frequency ratios from slowest to fastest
become $1, 1/2, 1/4, 1/8$: the longest wave is untouched, the fastest is
stretched eightfold.

**`alibi`** adds $-m_h|i-j|$ to the score, with one slope per head and the
slopes spaced geometrically. It costs no parameters and no memory beyond the
distance.

**`relative_bias`** learns one bias per head and per relative distance, clipped
at a maximum distance. It costs $H(2D+1)$ parameters.

**`t5_bias`** learns one shared table over $B$ buckets. Small distances get
their own bucket and large distances are log-spaced into the rest; the sign of
the offset uses the other half of the table.

**`no_position`** returns zeros from every channel, so the rest of the code can
treat it uniformly.

## Derivation

### Sinusoidal

With $d$ even and $i=0,\dots,d/2-1$,

$$
\theta_i=\mathrm{base}^{-2i/d},
\qquad
\mathrm{PE}(p,2i)=\sin(p\theta_i),
\qquad
\mathrm{PE}(p,2i+1)=\cos(p\theta_i).
\qquad (2)
$$

Each pair of coordinates is a point on the unit circle, turning at a different
speed: the first pair turns one radian per token, the last pair turns
$1/10000$ of a radian per token. Two useful facts follow immediately.

The norm of every row is the same, because every pair contributes
$\sin^2+\cos^2=1$:

$$
\lVert \mathrm{PE}(p)\rVert^2=\sum_i 1=\frac d2 .
\qquad (3)
$$

And the dot product of two rows is a function of the gap only:

$$
\mathrm{PE}(p)\cdot \mathrm{PE}(q)
=\sum_i \cos\bigl((p-q)\theta_i\bigr),
\qquad (4)
$$

using $\sin a\sin b+\cos a\cos b=\cos(a-b)$. The demo checks (4) against the
table for all sixteen positions to $10^{-12}$ and the norm in (3) to $10^{-12}$.

### Learned

The table is a parameter, and the lookup is a slice:

$$
t_p=T[p], \qquad T\in\mathbb R^{N\times d}.
\qquad (5)
$$

Gradients flow only to the rows that were looked up; row $p$ receives the sum of
the gradients of every token at position $p$. Positions beyond $N$ have no row,
which is why the constructor takes a maximum length and why `table(length)`
raises beyond it. The demo's table is written in place through the slice, which
is what makes a gradient step possible.

### Rotary embeddings

Split each projected vector into pairs and rotate pair $i$ by $p\theta_i$:

$$
R_i(p)=
\begin{pmatrix}
\cos p\theta_i & -\sin p\theta_i\\
\sin p\theta_i & \cos p\theta_i
\end{pmatrix}.
\qquad (6)
$$

The score between a query at $p$ and a key at $q$ is

$$
q_p\cdot k_q
=\bigl(R(p)W_qx_p\bigr)\cdot\bigl(R(q)W_kx_q\bigr)
=\bigl(W_qx_p\bigr)^{\top}R(q-p)\bigl(W_kx_q\bigr),
\qquad (7)
$$

because $R(p)^{\top}R(q)=R(q-p)$: rotations commute and their angles add. The
absolute positions disappear and only the gap survives. That is the whole point
of RoPE, and the demo verifies it three ways: scores at gaps 1, 3 and 7 are
constant along the sequence to $10^{-12}$, the score at $(3,7)$ equals the score
at $(0,4)$, and the rotation preserves every row norm.

The catch is the period. The slowest pair turns $\theta_0=1$ radian per token
at dimension 8, so its period is $2\pi\approx6.3$ tokens: positions 0 and 6
differ by only $0.28$ radians in that coordinate. Rotation does not alias
outright, but the low frequencies stop being informative well before the model's
context length. This is the reason context extension is a research topic rather
than a multiplication.

### NTK-aware scaling

The base stretch multiplies the frequency of pair $i$ by

$$
\frac{\theta_i'}{\theta_i}
=s^{-2i/(d-2)},
\qquad
\theta_i'=\bigl(\mathrm{base}\,s^{d/(d-2)}\bigr)^{-2i/d}.
\qquad (8)
$$

At $i=0$ the ratio is one; at $i=d/2-1$ it is $1/s$. The interpolation is
deliberately non-uniform: the slowest pair keeps its wavelength, so absolute
positions remain anchored, while the fastest pairs are stretched so that a long
sequence does not rotate them many times over. The demo prints the four ratios
for $d=8,s=8$ and asserts both endpoints and monotonicity.

### ALiBi

$$
b_h(i,j)=-m_h\,|i-j|,
\qquad
m_h=2^{-8h/H},
\qquad h=1,\dots,H .
\qquad (9)
$$

The slopes are fixed, geometric, and per head: the first head is almost flat,
the last decays by $2^{-8}$ per token. The paper defines the penalty on the
causal side only, with future entries masked to $-\infty$; the demo uses
$|i-j|$ so the same bias is usable in a bidirectional block. The matrix is
Toeplitz, which is what lets a real implementation generate it from one vector
of distances instead of storing $H L^2$ numbers.

### Learned relative bias

$$
b_h(i,j)=\beta_h\bigl[\mathrm{clip}(i-j,\,-D,\,D)\bigr],
\qquad
\beta\in\mathbb R^{H\times(2D+1)} .
\qquad (10)
$$

Every pair with the same offset shares a parameter, so the bias is constant on
diagonals and all distances beyond $D$ share the edge value. The demo checks the
diagonal property directly, with 3 heads and $D=4$, giving 27 parameters.

### T5 buckets

T5 keeps a shared table and assigns an index to each signed offset. With $B$
buckets split into two halves for the sign, $E=(B/2)/2$ exact distances, and a
maximum distance $M$,

$$
\mathrm{bucket}(i-j)=
\begin{cases}
|i-j|, & |i-j|<E,\\[2pt]
E+\Bigl\lfloor \dfrac{\log(|i-j|/E)}{\log(M/E)}\Bigl(\dfrac B2-E\Bigr)\Bigr\rfloor,
& \text{otherwise},
\end{cases}
\qquad (11)
$$

clipped to the half and offset by $B/2$ when $i-j<0$. The table has $B$ entries
and is shared by every head. With $B=8$, $M=16$ and length 12, the forward
distances use buckets $[0,1,2,2,2,2,3,3,3,3,3,3]$ and the backward ones
$[0,5,6,6,6,6,7,7,7,7,7,7]$ — the demo prints both lists. Note the collision:
distances 3 and 4 share bucket 2, which is a real loss of resolution and the
reason 19 of the 48 swapped pairs produced an identical readout.

### Parameter counts

$$
P_{\mathrm{learned}}=Nd,
\qquad
P_{\mathrm{relative}}=H(2D+1),
\qquad
P_{\mathrm{t5}}=B,
\qquad (12)
$$

and zero for the fixed schemes. At $N=16$, $d=8$: 128; at $H=3$, $D=4$: 27; at
$B=8$: 8.

## Code walkthrough

The frequencies are the only formula the fixed schemes need:

```python
def _frequencies(dim: int, base: float) -> np.ndarray:
    """The d/2 inverse wavelengths of the sinusoidal and rotary tables."""
    return base ** (-np.arange(0, dim, 2) / dim)
```

Multiplying them by the position gives the angles, and the sinusoidal table
interleaves the sine and cosine of those angles:

```python
            angles = _rotations(length, self.dim, self.base)
            table = np.zeros((length, self.dim))
            table[:, 0::2] = np.sin(angles)
            table[:, 1::2] = np.cos(angles)
            return table
```

The rotation is four multiplications on reshaped pairs, and it works for any
leading shape, so one function serves a single vector and a whole sequence:

```python
    pairs = x.reshape(shape[:-1] + (shape[-1] // 2, 2))
    cosine = np.cos(angles)[None]
    sine = np.sin(angles)[None]
    out = np.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cosine - pairs[..., 1] * sine
    out[..., 1] = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return out.reshape(shape)
```

The learned table is a slice of a parameter, not a copy, which is what makes the
gradient step in the demo land in the model:

```python
        if self.variant == "learned":
            if length > self.max_positions:
                raise ValueError("length exceeds max_positions")
            return self.parameters["table"][:length]
```

ALiBi is one outer product of slopes and distances:

```python
        if self.variant == "alibi":
            slopes = 2.0 ** (-8.0 * np.arange(1, self.heads + 1)
                             / self.heads)
            distance = np.abs(np.arange(length)[:, None]
                              - np.arange(length)[None, :])
            return -slopes[:, None, None] * distance[None]
```

The learned relative bias is a table lookup on the clipped diagonal index:

```python
        if self.variant == "relative_bias":
            table = self.parameters["bias"]
            signed = np.arange(length)[:, None] - np.arange(length)[None, :]
            signed = np.clip(signed, -self.max_distance, self.max_distance)
            return table[:, signed + self.max_distance]
```

and the T5 bucket map is the only branchy piece:

```python
    half = buckets // 2
    exact = half // 2
    distance = np.minimum(np.abs(signed), max_distance)
    index = np.zeros_like(distance)
    small = distance < exact
    index[small] = distance[small]
    large = ~small
    index[large] = exact + (np.log(distance[large] / exact)
                            / np.log(max_distance / exact) * (half - exact))
    index = np.minimum(index.astype(int), half - 1)
    return index + (signed < 0) * half
```

The `small` mask is what keeps `log` away from zero; the constructor refuses a
maximum distance that does not exceed the exact bucket count, because that is
the case where the logarithm's denominator vanishes.

The readout that the demo measures applies all three channels in the order (1)
prescribes:

```python
    hidden = x + pos.table(length)
    query = pos.rotate(hidden @ wq)
    key = pos.rotate(hidden @ wk)
    value = hidden @ wv
    scores = query[0] @ key.T / np.sqrt(pos.dim)
    weights_ = _softmax(scores + pos.bias(length)[0, 0])
    return weights_ @ value
```

Note that the value is not rotated: rotation is a unitary change of basis, and
applying it to the values would rotate the output without adding information.
The table is added before the projection and the bias after the score, which is
the standard placement for each.

The backward pass is short because the readout is one attention layer, and it is
what the learned-table demo trains:

```python
    d_scores = weights_ * (value @ residual
                           - float(weights_ @ (value @ residual)))
    d_query = d_scores @ key / np.sqrt(dim)
    d_key = d_scores[:, None] * query[0][None, :] / np.sqrt(dim)
    d_value = weights_[:, None] * residual[None, :]
```

`d_scores` is the softmax Jacobian from the transformer atom, and the three
projections each receive one term. The table's gradient is the sum of
`d_query @ wq.T`, `d_key @ wk.T` and `d_value @ wv.T`, which is why it has
support only on the rows that were used.

## Complexity

For sequence length $L$, dimension $d$, $H$ heads, $N$ stored positions, $D$
maximum relative distance and $B$ buckets:

- **Sinusoidal:** $O(Ld)$ time and memory to build the table, and no
  parameters. Because it is a formula, it can be evaluated at $L$ larger than
  any length seen in training.
- **Learned:** $O(Nd)$ parameters, $O(Ld)$ lookup. Training touches only the
  $L$ used rows per sequence.
- **RoPE:** $O(Ld)$ time to rotate queries and keys, no parameters, and no extra
  memory if the rotation is fused into the projection.
- **ALiBi:** $O(HL^2)$ if the matrix is materialised and $O(L)$ if it is
  generated per row. The arithmetic is one multiply-add per score.
- **Relative bias and T5:** the same $O(HL^2)$ score-matrix cost, but with a
  table lookup and, for T5, one shared table instead of $H$.
- **The demo:** one attention readout over eight tokens, so every scheme costs
  a handful of matrix products and the whole file runs in a tenth of a second.

## Failure modes

**No position is exactly order-blind.** The readout is a weighted average over
the multiset of tokens, so exchanging two rows leaves it unchanged up to
summation order — $2.6\times10^{-17}$ in the demo's 48 pairs. No classifier on
top of that readout can do better than chance on an order task, no matter how
large it is.

**Buckets collide.** T5's log spacing maps distances 3 and 4 to the same bucket,
so 19 of 48 swapped pairs produced an identical readout. Coarse buckets save
parameters and lose resolution exactly where the model might need it; the exact
relative table has no such collision but costs $H(2D+1)$ parameters.

**Clipping is the same loss at the tail.** A relative bias with maximum distance
$D$ gives every offset beyond $D$ the same value. A model that needs to
distinguish "ten tokens back" from "a hundred tokens back" cannot.

**The learned table cannot extrapolate.** A sequence longer than
`max_positions` has no rows to look up. Sinusoidal and rotary schemes have no
such bound, which is why they are the usual choice for context extension.

**RoPE's slowest frequency has a short period at small dimension.** At $d=8$,
the first pair turns one radian per token and repeats every $2\pi\approx6.3$
tokens. NTK scaling does not change that pair — the ratio in (8) is one at
$i=0$ — so it buys high-frequency resolution, not a longer absolute range.

**Rotating the values is a bug.** RoPE rotates queries and keys to make the
score relative; rotating the values too would rotate the attention output by the
query's position, mixing the readout with the encoding. The demo's readout
rotates `query` and `key` and leaves `value` alone.

**ALiBi's causal form is not symmetric.** The paper's bias is defined on the
past side and masks the future. Using $|i-j|$ in a bidirectional block is a
different function; it is the one this file implements, and the doc says so.

**A zero-initialised bias learns slowly at first.** The relative and T5 tables
start near zero, so the bias has no effect until the gradients move it. That is
the intended behaviour, but it means the first few steps see an effectively
unpositioned model.

**A table added before layer norm is partly erased.** If the first block
normalises its input, the mean and scale of the table are removed and only its
direction survives. Pre-LN transformers add the table to the embedding before
the block, so the table is re-normalised at every block; that is fine, but it
means the absolute magnitude of the table is not what the model sees.

**Buckets and distances must agree at construction.** The constructor refuses a
maximum distance that does not exceed the exact bucket count, because the log
spacing divides by $\log(M/E)$ and $M=E$ would be a zero denominator — the kind
of bug that only appears when someone configures a small model.

**The readout is one layer, not a trained model.** The demo measures whether the
information is present in a feature, not whether gradient descent will use it.
The learned-table block closes that gap for one scheme by checking the gradient
against finite differences ($2.3\times10^{-11}$) and training the table until
the loss falls from 0.0467 to 0.0006.

## Applications

Learned absolute tables are BERT's and GPT-2's choice: simple, fast, and
bounded by the training length. Sinusoidal tables are the original transformer's
and still appear where extrapolation matters.

RoPE is the standard in open decoder-only models — LLaMA, GPT-NeoX, Qwen — and
NTK-aware scaling is one of the standard tricks for extending their context.
ALiBi is used by BLOOM and Mosaic's models, and its fixed geometric slopes make
it cheap at long lengths.

Relative position in the score is the Transformer-XL and T5 answer: position
enters through the offset rather than through the token, which keeps the table
shared across positions and, in T5's case, across heads.

No positional encoding is the right choice for set-valued inputs — the
Perceiver's latents, point clouds, molecules — where order is genuinely absent
and adding a position would be a lie.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., Polosukhin, I. (2017). *Attention is all you need.* NeurIPS.
  Equation (2).
- Gehring, J., Auli, M., Grangier, D., Yarats, D., Dauphin, Y. N. (2017).
  *Convolutional sequence to sequence learning.* ICML. Learned absolute
  positions.
- Shaw, P., Uszkoreit, J., Vaswani, A. (2018). *Self-attention with relative
  position representations.* NAACL. Equation (10).
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). *RoFormer:
  enhanced transformer with rotary position embedding.* arXiv:2104.09864.
  Equations (6) and (7).
- bloc97 (2023). *NTK-aware scaled RoPE allows for efficient fine-tuning of
  large language models.* The base stretch of (8).
- Press, O., Smith, N. A., Lewis, M. (2022). *Train short, test long: attention
  with linear biases input length.* ICLR. Equation (9).
- Raffel, C. et al. (2020). *Exploring the limits of transfer learning with a
  unified text-to-text transformer.* JMLR. The buckets of (11).
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: pre-training
  of deep bidirectional transformers.* NAACL. A learned absolute table in
  practice.
