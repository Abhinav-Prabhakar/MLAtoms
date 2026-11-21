# Gated memory cells — LSTM and GRU, gate by gate

`lstm_gru.py` · Sequences · Tier 1 · 6 variants

## Intuition

The recurrent cell of the previous atom multiplies the state by the same matrix
at every step. A gradient crossing $T-t$ steps therefore crosses $T-t$ copies of
that matrix times a tanh derivative, and the `rnn` atom measured the result:
eleven orders of magnitude over thirty-two steps. The fix is not a better
matrix. It is an additive path that the network can open or close.

An LSTM keeps two states: a hidden state $h_t$ that it exposes, and a cell state
$c_t$ that it protects. The cell state is updated by a convex combination,

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t,
$$

where $f_t$ is a forget gate, $i_t$ an input gate, and $g_t$ a candidate. If
$f_t \approx 1$ and $i_t \approx 0$, the cell state is copied unchanged and the
gradient crosses the step without shrinking. The gates are themselves functions
of the input and the hidden state, so the network learns when to copy and when
to write.

```
                            c_(t-1)
                               │
                               ▼
      ┌──────┐   ┌────┐   ┌─────────┐   ┌────┐
 x_t ─┤ gates├──▶│ i,f├──▶│ f c+i g │──▶│ tanh├──▶ h_t
h_(t-1)└──────┘   └────┘   └─────────┘   └────┘
```

A GRU reaches the same effect with fewer parts. It drops the separate cell state
and uses an update gate $z_t$ to interpolate between the previous hidden state
and a candidate, which makes the state both the memory and the output. That is
cheaper and often just as good; the price is that the memory and the readout are
the same vector, so there is no protected cell to read from later.

The variants are budgets on this idea. `lstm` is the full cell. `gru` removes
one gate and the cell state. `peephole` lets the gates read the cell state
directly, which helps when the timing of a count matters. `lstm_projection`
compresses the output through a matrix, which is how large recurrent layers were
made affordable. `cifg` ties the input gate to the forget gate, $i_t=1-f_t$, and
drops a quarter of the gate parameters. `bidirectional` runs two cells in
opposite directions.

The demos put numbers on the claims. On a symbol delayed by two steps — the kind
of memory task a tanh recurrence memorises rather than learns — the LSTM reaches
$0.984$ held-out and the GRU $0.983$.
The coupled-gate cell reaches $0.989$ with $76\%$ of the LSTM's parameters, and
the projection variant $0.953$ with $51\%$. Every gate branch is checked against
central differences to $2.6\times10^{-8}$. On a next-symbol target the forward
LSTM is at chance ($0.257$) while the bidirectional model reaches $0.938$.

## Variants

| variant | gates | state carried | extra parameters |
|:--|:--|:--|:--|
| `lstm` | input, forget, output, candidate | cell and hidden | none |
| `gru` | update, reset, candidate | hidden only | `W_hn_` |
| `peephole` | LSTM gates plus peepholes | cell and hidden | `w_i_`, `w_f_`, `w_o_` |
| `lstm_projection` | LSTM gates | cell, then projected hidden | `W_p_` |
| `cifg` | forget, output, candidate ($i=1-f$) | cell and hidden | none |
| `bidirectional` | two LSTM cells | two hidden states, concatenated | second cell |

**`lstm`** is the reference. Four gate blocks are computed from the input and
the previous hidden state; the input and candidate gates write, the forget gate
erases, and the output gate decides how much of the cell state is exposed.

**`gru`** merges the cell state into the hidden state. An update gate mixes the
previous state with a candidate, and a reset gate controls how much history the
candidate sees. It has three gate blocks instead of four and no output gate.

**`peephole`** adds a diagonal connection from the cell state to each gate. The
input and forget gates read $c_{t-1}$; the output gate reads $c_t$. The weights
are vectors, not matrices, so the cost is three extra numbers per unit.

**`lstm_projection`** computes the usual cell state and output vector $m_t$, then
projects it to a narrower state $h_t=W_p m_t$. The recurrent matrix and the
output layer both shrink, which is the point when the hidden width is large.

**`cifg`** couples the input and forget gates: whatever is not forgotten is
written. It is one linear constraint, not an approximation, and it removes one
gate block and its parameters.

**`bidirectional`** runs one LSTM over the sequence and another over its reverse,
concatenates the two hidden states at every step, and reads the output from the
concatenation. It is the only variant here that can depend on the future.

## Derivation

### The LSTM cell

Let $x_t\in\mathbb{R}^{F}$, $h_{t-1}\in\mathbb{R}^{H}$, and let
$[W_x,W_h,b]$ be the stacked parameters of the four gate blocks. With $\sigma$
the logistic function,

$$
\begin{aligned}
i_t &= \sigma(W_{xi}x_t+W_{hi}h_{t-1}+b_i), &
f_t &= \sigma(W_{xf}x_t+W_{hf}h_{t-1}+b_f), \\
o_t &= \sigma(W_{xo}x_t+W_{ho}h_{t-1}+b_o), &
g_t &= \tanh(W_{xg}x_t+W_{hg}h_{t-1}+b_g),
\end{aligned}
\qquad (1)
$$

and then

$$
c_t = f_t\odot c_{t-1} + i_t\odot g_t, \qquad h_t = o_t\odot\tanh(c_t).
$$

The implementation stacks the four blocks along the first axis, so one matrix
product produces every gate, and slices the result:

```python
            i_pre, f_pre, o_pre, g_pre = (pre[:, 0:hidden], pre[:, hidden:2 * hidden],
                                          pre[:, 2 * hidden:3 * hidden],
                                          pre[:, 3 * hidden:])
            if self.variant == "peephole":
                i_pre = i_pre + c_prev * self.w_i_
                f_pre = f_pre + c_prev * self.w_f_
            i, f, g = _sigmoid(i_pre), _sigmoid(f_pre), np.tanh(g_pre)
        c = f * c_prev + i * g
```

The forget-gate bias is initialised to one, not zero. At the start of training
the cell state is empty, and a forget gate at $0.5$ would halve whatever the
input gate writes before it can be read; starting it open lets the gradient
reach the early steps and lets the model learn to close it.

### The GRU

The GRU removes the cell state. With update and reset gates,

$$
\begin{aligned}
z_t &= \sigma(W_{xz}x_t+W_{hz}h_{t-1}+b_z), &
r_t &= \sigma(W_{xr}x_t+W_{hr}h_{t-1}+b_r), \\
n_t &= \tanh(W_{xn}x_t+W_{hn}(r_t\odot h_{t-1})+b_n), &
h_t &= (1-z_t)\odot n_t + z_t\odot h_{t-1}.
\end{aligned}
\qquad (2)
$$

The update gate plays the role of the forget gate and its complement in one
number: $z_t$ keeps the old state, $1-z_t$ admits the candidate. The reset gate
is inside the candidate, so it can hide the past from the new proposal without
erasing it. The code is the same two matrix products, with the candidate
computed from the reset state:

```python
            z = _sigmoid(pre[:, :hidden] + h_prev @ W_h[:hidden].T)
            r = _sigmoid(pre[:, hidden:2 * hidden] + h_prev @ W_h[hidden:].T)
            candidate = np.tanh(pre[:, 2 * hidden:] + (r * h_prev) @ self.W_hn_.T)
            state = (1.0 - z) * candidate + z * h_prev
```

The candidate needs its own recurrent matrix `W_hn_`, because it reads the reset
state rather than the full hidden state. That is the one place where the GRU
does not fit the stacked-block pattern of the LSTM.

### Peepholes

The gates can read the cell state directly:

$$
i_t=\sigma(\cdot+w_i\odot c_{t-1}), \qquad
f_t=\sigma(\cdot+w_f\odot c_{t-1}), \qquad
o_t=\sigma(\cdot+w_o\odot c_t).
\qquad (3)
$$

The output gate reads $c_t$, the value it is about to scale, which is why the
forward pass computes $c_t$ before $o_t$:

```python
        o = (_sigmoid(o_pre + c * self.w_o_) if self.variant == "peephole"
             else _sigmoid(o_pre))
```

In the backward pass the peepholes add three terms: the gradient of $c_t$ picks
up $do_t\odot w_o$, and the gradient of $c_{t-1}$ picks up
$di_t\odot w_i+df_t\odot w_f$. The weights themselves receive the correlations
$do_t\odot c_t$, $di_t\odot c_{t-1}$ and $df_t\odot c_{t-1}$.

### Projection

For a large hidden width $H$, the recurrent matrix $4H\times H$ and the output
layer $C\times H$ dominate. The projected cell computes

$$
m_t=o_t\odot\tanh(c_t)\in\mathbb{R}^{H}, \qquad h_t=W_p m_t\in\mathbb{R}^{P},
\qquad (4)
$$

and feeds $h_t$, not $m_t$, back into the gates and out to the classifier. The
recurrent matrix becomes $4H\times P$ and the output layer $C\times P$. The
backward pass is one extra linear map:

```python
        if self.variant == "lstm_projection":
            grads["W_p_"] += dh.T @ m
            dm = dh @ self.W_p_
```

The projection is not free: it is a bottleneck, so information that does not fit
in $P$ dimensions is lost before the next step. In the demo, $H=24$ and $P=8$
cut the parameter count from $2884$ to $1476$ and the held-out accuracy from
$0.984$ to $0.953$.

### Coupled gates

The CIFG sets $i_t=1-f_t$. The cell update becomes a two-way interpolation,

$$
c_t=f_t\odot c_{t-1}+(1-f_t)\odot g_t,
\qquad (5)
$$

so the cell state is always a convex combination of the old state and the
candidate. The backward pass follows by substituting $i=1-f$:

```python
        if self.variant == "cifg":
            dz_f = dc * (cache["c_prev"] - g) * f * (1.0 - f)
            dz_g = dc * (1.0 - f) * (1.0 - g * g)
            dpre = np.concatenate([dz_f, dz_o, dz_g], axis=1)
```

This removes one gate block, a quarter of the gate parameters, and the ability
to write and erase independently. The demo's coupled cell is the most accurate
of the six ($0.989$) and uses $76\%$ of the reference parameters, which is a
reminder that the full cell's extra freedom is not always used.

### Bidirectional passes

Two LSTM cells read the sequence from opposite ends:

$$
h_t^{f}=\mathrm{LSTM}^{f}(x_t,h_{t-1}^{f}), \qquad
h_t^{b}=\mathrm{LSTM}^{b}(x_t,h_{t+1}^{b}),
\qquad (6)
$$

and the output reads the concatenation
$p_t=\mathrm{softmax}(W_o[h_t^{f};h_t^{b}]+b_o)$. The gradient of the output
layer splits into the two blocks; each block drives its own backward pass, and
the two cells share only $W_o$:

```python
        split = dlogits @ self.W_o_
        first, _, _ = self._backward_direction(
            features, forward, cache_f, split[..., :self.hidden], None)
        second, _, _ = self._backward_direction(
            features[:, ::-1], backward, cache_b,
            split[..., self.hidden:][:, ::-1], None, suffix="b_")
```

### The loss and the backward pass

Every step emits a class distribution, and the loss is the mean cross-entropy
over the batch and the sequence:

$$
p_t=\mathrm{softmax}(W_o h_t+b_o), \qquad
L=-\frac{1}{BT}\sum_{b,t}\log p_{b,t}[y_{b,t}].
\qquad (7)
$$

The backward pass of the cell is the chain rule applied to (1). With
$dm_t=\partial L/\partial m_t$ (equal to $\partial L/\partial h_t$ unless the
state is projected) and $dc_{t+1}$ the gradient carried from the next step,

$$
\begin{aligned}
dc_t &= dm_t\odot o_t\odot\big(1-\tanh^2 c_t\big) + dc_{t+1}, \\
do_t &= dm_t\odot\tanh(c_t)\odot o_t(1-o_t), \\
di_t &= dc_t\odot g_t\odot i_t(1-i_t), \\
df_t &= dc_t\odot c_{t-1}\odot f_t(1-f_t), \\
dg_t &= dc_t\odot i_t\odot(1-g_t^2), \\
dc_{t-1} &= dc_t\odot f_t.
\end{aligned}
$$

The implementation makes the carried cell gradient an explicit argument, because
forgetting it is the classic bug in a hand-written LSTM:

```python
        dc = dm * o * (1.0 - tanh_c * tanh_c) + carry_c
        dz_o = dm * tanh_c * o * (1.0 - o)
        ...
            dz_i = dc * g * i * (1.0 - i)
            dz_f = dc * cache["c_prev"] * f * (1.0 - f)
            dz_g = dc * i * (1.0 - g * g)
            dc_prev = dc * f
```

The three gate pre-activations are then concatenated and multiplied by the input
and recurrent matrices, exactly as in a feedforward layer. The demo perturbs the
largest entry of every parameter and compares with a central difference; the
worst relative error across all six variants is $2.6\times10^{-8}$.

### Why the cell state helps

The gradient of $c_T$ with respect to $c_t$ is a product of the cell-state
Jacobians. Differentiating $c_s=f_s\odot c_{s-1}+i_s\odot g_s$,

$$
\frac{\partial c_s}{\partial c_{s-1}}
=\mathrm{diag}(f_s)
+\mathrm{diag}\Big(\frac{\partial f_s}{\partial c_{s-1}}\Big)c_{s-1}
+\mathrm{diag}\Big(\frac{\partial i_s}{\partial c_{s-1}}\Big)g_s.
\qquad (8)
$$

The leading term is diagonal and equals the forget gate; the other two vanish
unless the gates have peepholes, which is why that variant has a slightly
different analysis. When $f_s\approx1$ the product is close to the identity, so
the gradient crosses the step unchanged; a vanilla recurrence has $D_sW_h$
instead, and its norm is below one almost everywhere. The gates learn which
regime to be in, which is the whole mechanism.

The demo's fitted forget gate is $0.60$, not $1$. A two-step copy does not need a
perfect integrator, and forcing $f$ to one would make the cell unable to forget.
The advantage of the additive path is that it exists and is controllable; it is
not that the fitted gate is always saturated.

## Code walkthrough

### The logistic function

```python
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function without overflow on either tail."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    tail = np.exp(x[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out
```

Four gates per unit per step means four chances for an exponential to overflow.
The split form evaluates $\exp$ only on the tail where the argument is negative,
so neither branch can overflow, and it is the same function as
$1/(1+e^{-x})$. It is written once and used by every LSTM variant.

### The peephole backward

```python
        if self.variant == "peephole":
            dc = dc + dz_o * self.w_o_
            grads["w_o_"] += (dz_o * c).sum(axis=0)
```

The output gate was computed from $c_t$, so the gradient of $c_t$ has to include
the path back through that gate before the cell-state gradient is propagated.
The weight gradient is the sum over the batch of $do_t\odot c_t$; the peephole
weight is shared by every step, so the time axis is summed as well.

### The GRU backward

```python
        dz = dh * (h_prev - candidate) * z * (1.0 - z)
        dn = dh * (1.0 - z) * (1.0 - candidate * candidate)
        d_reset_hidden = dn @ self.W_hn_
        d_r = d_reset_hidden * h_prev
        dr = d_r * r * (1.0 - r)
```

The update gate has no separate cell state, so its gradient is the difference
between the two things it mixes. The reset gate receives the candidate gradient
through `W_hn_`, multiplied by the previous state; the previous state receives
the same quantity multiplied by the reset gate. Both terms are needed, and
leaving out either is a silent half-gradient that still trains.

### The carried cell state

```python
        for t in range(length - 1, -1, -1):
            cache = caches[t]
            dh = dh_out[:, t] + carry_h
            if self.variant == "gru":
                dpre, dcarry_h = self._backward_gru(cache, dh, grads)
            else:
                dpre, dcarry_h, carry_c = self._backward_lstm(
                    cache, dh, carry_c, grads, suffix)
```

`carry_h` is the hidden-state gradient from step $t+1$; `carry_c` is the
cell-state gradient. The LSTM needs both, the GRU only the first. The loop runs
backwards because each step's gate depends on the next step's state, which is
the definition of backpropagation through time.

### Parameter sharing

```python
            grads[f"W_x_{suffix}"] += dpre.T @ cache["x"]
            if self.variant == "gru":
                grads[f"W_h_{suffix}"] += dpre[:, :2 * hidden].T @ cache["h_prev"]
            else:
                grads[f"W_h_{suffix}"] += dpre.T @ cache["h_prev"]
```

The same gate matrices are used at every step, so the gradient is a sum over
time. The GRU is the exception: its candidate has a separate recurrent matrix,
so only the update and reset blocks multiply `W_h_`. Getting this slice wrong is
the reason the first version of this file failed its own gradient check.

## Complexity

Let $B$ be the batch size, $T$ the sequence length, $H$ the hidden width, $P$
the projected width, $F$ the input width, $C$ the classes and $G$ the number of
gate blocks ($4$ for the LSTM family, $3$ for the GRU and CIFG).

| operation | time | memory |
|:--|:--|:--|
| one step, LSTM | $O(BG H(F+H))$ | $O(BGH)$ |
| one step, projected | $O(BG H(F+P))$ | $O(BGH)$ |
| one step, GRU | $O(BH(F+H))$ | $O(BH)$ |
| one epoch | $O(BTG H(F+H))$ | $O(BTH)$ |
| parameter count, LSTM | $GH(F+H+1)+C(H+1)$ | |
| parameter count, projected | $GH(F+P+1)+CP+PH+C$ | |
| parameter count, CIFG | $3H(F+H+1)+C(H+1)$ | |

The gate width is the multiplier that makes gated cells expensive: an LSTM step
costs four times a vanilla step at the same hidden width. Projection attacks
that multiplier from the other side, replacing $H$ with $P$ in the recurrent
term. The demo's numbers make the trade concrete: $2884$ parameters and $0.984$
for the LSTM, $1476$ and $0.953$ for the projected cell at $P=8$.

## Failure modes

**Saturated gates.** A sigmoid driven far into its tail has derivative near
zero, and the gate stops moving. If both the input and forget gates saturate in
opposite directions the cell becomes a constant, and the model looks like it has
stopped learning while the loss is merely flat. Gradient clipping and a
sensible learning rate are the first defences; inspecting `gate_means` is the
diagnostic.

**A forget bias that is too open.** Initialising the forget bias to one is a
good default for learning long dependencies, but on a task that needs fast
forgetting it delays the fit: the gate has to travel down from one before the
cell can discard stale state. The demo's copy task ends at $0.60$, so the gate
does move; a model stuck near $1.0$ after training is a symptom, not a success.

**The cell state is not a hard guarantee.** The product of forget gates still
decays if the gates settle below one, and an LSTM does not solve a dependency of
ten thousand steps. It moves the practical horizon from tens to hundreds of
steps, which is why truncation and attention remain relevant. `jacobian_norms`
in the `rnn` atom is the tool for measuring where a particular fit actually
sits.

**Projection is a bottleneck.** The projected state is a linear image of the
cell output, so dimensions discarded by $W_p$ cannot be recovered downstream. A
projection that is too narrow costs accuracy, and the cost is not visible in the
training loss until the model has fit everything else. Tune $P$ against held-out
data, not against parameters saved.

**Coupling is a constraint.** The CIFG cannot write and erase independently:
whatever fraction of the old state survives, the complementary fraction of the
candidate enters. That is a good prior on many sequence tasks and a bad one on
tasks that need to add to memory while keeping it. The demo's coupled cell wins
by five thousandths, which is within noise of the reference; the parameter
saving is the reliable part.

**Peepholes add variance.** Three extra vectors per unit are cheap, but they
also give the optimiser more ways to fit noise, and on small data the peephole
variant is the easiest of the six to overfit. The demo's peephole cell uses
$102\%$ of the reference parameters for a $0.982$ that is indistinguishable from
$0.984$; on this task the extra connections buy nothing.

**Bidirectional leakage.** A bidirectional model reads the future by
construction. It is correct for tagging a complete sequence and wrong for
streaming, online control, or any evaluation where the future would not be
available at prediction time. Using one in a live system is a data leak, not an
architecture choice.

**Small batches memorise.** With sixty-four sequences, these cells can fit the
training set by storing sequence identity rather than the rule; the demo's
held-out split is what distinguishes the two. A gated cell has more capacity to
memorise than a vanilla one, so the gap between training and held-out accuracy
is the first thing to inspect.

**No masking.** Every sequence is assumed to start at step zero and to have a
label at every step. Padded batches need a mask in the loss and in the carried
state, or the model learns the pad symbol and the state is polluted at the
boundary. This file refuses ragged labels rather than guessing an alignment.

## Applications

**Speech recognition.** LSTM and GRU acoustic models replaced HMM-hybrid
systems before attention did, and the bidirectional variant was standard for
offline transcription. The `ctc` atom is the alignment-free loss that made
frame-level labels unnecessary.

**Machine translation.** The sequence-to-sequence model of 2014 was two LSTMs,
one reading the source and one writing the target, and the projection variant
was introduced to keep the output softmax affordable on large vocabularies. The
`attention` atom is what removed the fixed-size bottleneck between them.

**Language modelling.** A gated recurrent network trained on next-token
prediction was the strongest language model before the transformer, and the
forget gate's ability to carry information across sentences is why it worked.
The training objective here is the same one.

**Time series and sensor streams.** Gated cells are the standard recurrent
baseline for forecasting and anomaly detection, and the GRU is often preferred
on small data because it has fewer parameters. The `tcn` and `ssm` atoms are the
convolutional and state-space alternatives.

**Handwriting and gesture recognition.** Pen trajectories and inertial streams
are variable-length sequences with long-range style; gated cells were the state
of the art before attention, and the `dtw` atom is the template-matching
alternative when there is too little data to train one.

**Reinforcement learning.** A gated policy carries memory through partial
observability, and the `actor-critic` and `PPO` atoms use exactly this cell when
the observation is not Markov. The GRU is common because the update is cheap
enough for a rollout loop.

## References

- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation* 9(8), 1735–1780. The cell and the constant error carousel.
- Gers, F. A., Schmidhuber, J. & Cummins, F. (2000). Learning to forget:
  continual prediction with LSTM. *Neural Computation* 12(10), 2451–2471. The
  forget gate and the forget-bias initialisation.
- Gers, F. A. & Schmidhuber, J. (2001). Recurrent nets that time and count.
  *IJCNN*, 189–194. Peephole connections and counting tasks.
- Cho, K. et al. (2014). Learning phrase representations using RNN
  encoder–decoder for statistical machine translation. *EMNLP*, 1724–1734. The
  GRU and the sequence-to-sequence setting.
- Sak, H., Senior, A. & Beaufays, F. (2014). Long short-term memory recurrent
  neural network architectures for large scale acoustic modeling.
  *Interspeech*, 338–342. The projected LSTM.
- Chung, J., Gülçehre, Ç., Cho, K. & Bengio, Y. (2014). Empirical evaluation of
  gated recurrent neural networks on sequence modeling. *arXiv:1412.3555*. GRU
  against LSTM on a common benchmark.
- Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R. & Schmidhuber,
  J. (2017). LSTM: a search space odyssey. *IEEE Transactions on Neural
  Networks and Learning Systems* 28(10), 2222–2232. Which LSTM variants matter.
- Jozefowicz, R., Zaremba, W. & Sutskever, I. (2015). An empirical exploration
  of recurrent network architectures. *ICML*, 2342–2350. The evolutionary
  search that produced the coupled-gate cell.
- Karpathy, A., Johnson, J. & Fei-Fei, L. (2015). Visualizing and understanding
  recurrent networks. *ICLR*. What the gates actually do on character-level
  language modelling.
