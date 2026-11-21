# Recurrent networks — backpropagation through time, by hand

`rnn.py` · Sequences · Tier 1 · 7 variants

## Intuition

A recurrent network is a feedforward network applied to a moving window of
history, with the window replaced by a state. At every step the same cell reads
the current input and the previous state, writes a new state, and emits a
prediction. The weights are shared across time, so the network has one set of
parameters regardless of how long the sequence is.

```
   x_0      x_1      x_2            x_T
    │        │        │              │
    ▼        ▼        ▼              ▼
  ┌────┐   ┌────┐   ┌────┐        ┌────┐
  │ h0 │──▶│ h1 │──▶│ h2 │── . . .▶│ hT │
  └────┘   └────┘   └────┘        └────┘
    │        │        │              │
    ▼        ▼        ▼              ▼
   p_0      p_1      p_2            p_T
```

Training this is not a new algorithm. Unroll the loop into a chain of copies,
share the weights, and run ordinary backpropagation on the resulting graph. The
only thing that makes it "through time" is that the chain is as long as the
sequence, and the same parameter appears at every link. Everything difficult
about recurrent networks follows from that length.

The seven variants are one cell and one schedule. The cell is the Elman
recurrence, the same with a learned initial state, Jordan's output recurrence,
or a layer-normalised cell. The schedule is full backpropagation, windows that
cut the gradient at their boundaries, or two passes in opposite directions.
They are independent axes, which is why the file has one backward routine with
flags rather than seven copies of the chain rule.

The numbers in this doc are the demo's output. Analytic gradients agree with
central differences to between $10^{-9}$ and $3.4\times10^{-9}$ for every cell,
including the layer-normalised and bidirectional branches. Cumulative parity is
learned to $0.980$ on held-out sequences. A next-symbol target is at chance
($0.257$) for a forward network and $0.937$ for a bidirectional one. And the
recurrent Jacobian loses $11.75$ orders of magnitude over thirty-two steps,
which is the vanishing gradient measured directly rather than inferred from a
training curve.

## Variants

| variant | cell | schedule | extra parameters |
|:--|:--|:--|:--|
| `vanilla` | Elman, zero start | full BPTT | none |
| `elman` | Elman, learned start | full BPTT | `h0_` |
| `jordan` | output recurrence | full BPTT | `W_y_` |
| `bptt` | Elman, zero start | full BPTT | none; exposes Jacobian norms |
| `truncated_bptt` | Elman, zero start | windows of `chunk` | none |
| `layer_norm_rnn` | normalised pre-activation | full BPTT | `ln_gain_`, `ln_bias_` |
| `bidirectional` | two Elman passes | full BPTT, both directions | second cell |

**`vanilla`** is the reference: the tanh recurrence of Elman with a zero initial
state, trained by full backpropagation through time with global-norm clipping.

**`elman`** makes the initial state a parameter instead of a constant. This is
the cleanest way to give the network a learned "resting" context, and it is the
part of Elman's original model that survives when the context units are folded
into the recurrence. In the demo the learned norm is $1.50$, far from the
$0.1$-scale initialisation.

**`jordan`** feeds the previous *output* back into the hidden layer instead of
(or as well as) the previous state. It is the natural cell when the task is
autoregressive — the next symbol is predicted from the last emitted symbol — and
it needs a softmax Jacobian in the backward pass, which is the extra branch the
gradient check covers.

**`bptt`** is the same cell and the same descent as `vanilla`, with the
diagnostic surface turned on: `jacobian_norms` returns
$\lVert \partial h_{T-1}/\partial h_t\rVert$ for every $t$. The two variants
train identically; the point of the separate name is that the vanishing
gradient is a property of the recurrence, not of the optimiser.

**`truncated_bptt`** runs the forward pass over the whole sequence but cuts the
backward pass at window boundaries of length `chunk`. The state is carried
forward as a constant, so the model can still use long-range information; what
it cannot do is attribute credit across a boundary.

**`layer_norm_rnn`** normalises the pre-activation before the tanh and learns a
gain and a bias for it. It changes the conditioning of the optimisation, not the
functional family, and the demo's gains move from $1$ to a range of
$0.24$–$3.54$, so the extra parameters are doing work.

**`bidirectional`** runs one cell left-to-right and another right-to-left,
concatenates the two states at every step, and reads the output from the
concatenation. It is the right model whenever the whole sequence is available at
prediction time, and it is the only variant here that can exploit a dependency
on the future.

## Derivation

### The cell

For a batch of $B$ sequences of length $T$, with inputs
$x_t\in\mathbb{R}^{F}$ and states $h_t\in\mathbb{R}^{H}$,

$$
a_t = W_x x_t + W_h h_{t-1} + b_h, \qquad h_t = \tanh(a_t),
\qquad (1)
$$

with $h_{-1}=0$ for `vanilla` and $h_{-1}=h_0$ (a parameter) for `elman`.

The code computes the whole batch at once. One step is two matrix products and
a broadcast bias:

```python
        pre = x @ W_x.T + h_prev @ W_h.T + b_h
        if self.variant == "jordan":
            pre = pre + y_prev @ self.W_y_.T
```

The state is stored in a ``(batch, length, hidden)`` array, so no Python loop
over sequences is needed — only over time:

```python
        for t in range(length):
            state, caches[t] = self._cell(features[:, t], state, previous_y, suffix)
            hidden[:, t] = state
```

Each step keeps the pre-activation, the normalised value, the standard
deviation and the previous state. That cache is the entire backward pass's
input; without it the chain rule would have to be re-derived numerically.

### Jordan's cell

Jordan's cell adds the previous output distribution to the pre-activation, on
top of the hidden recurrence:

$$
a_t = W_x x_t + W_h h_{t-1} + W_y y_{t-1} + b_h, \qquad
y_{t-1}=\mathrm{softmax}(W_o h_{t-1}+b_o).
\qquad (2)
$$

The gradient with respect to $y_{t-1}$ is $W_y^{\top}d_t$, and it has to pass
through the softmax that produced $y_{t-1}$. The Jacobian of the softmax is
$J=\mathrm{diag}(p)-pp^{\top}$, so the feedback into the logits at $t-1$ is
$J\,W_y^{\top}d_t$. That contribution has two effects: it is a gradient for
$W_o$ through $h_{t-1}$, and it adds to the gradient reaching $h_{t-1}$:

```python
    def _jordan_feedback(self, dpre: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Push a gradient on ``y_(t-1)`` back through its softmax."""
        dy = dpre @ self.W_y_
        jacobian = np.einsum("nc,nd->ncd", probs, probs)
        jacobian = -jacobian + np.einsum("nc,cd->ncd", probs,
                                         np.eye(self.n_classes_))
        return np.einsum("ncd,nd->nc", jacobian, dy)
```

The `einsum` calls build $J$ per sample without a Python loop, and the last
one applies it. This is the one place where a cell couples two time steps
through a non-linearity other than the tanh, and the finite-difference check is
what makes it trustworthy.

### The loss

The output layer reads the state at every step,

$$
p_t = \mathrm{softmax}(W_o h_t + b_o), \qquad
L = -\frac{1}{BT}\sum_{b,t}\log p_{b,t}[y_{b,t}].
\qquad (3)
$$

The mean over $B$ sequences and $T$ steps keeps the gradient scale independent
of the batch size, which is the division by `labels.size` in `_loss_gradient`.

### Backpropagation through time

Write $d_t = \partial L/\partial a_t$ for the gradient with respect to the
pre-activation. The output layer contributes directly, and the next step
contributes through $W_h$:

$$
d_t = \Big(W_o^{\top}(p_t-e_t) + W_h^{\top} d_{t+1}\Big)\odot\big(1-h_t^{2}\big),
\qquad (4)
$$

where $e_t$ is the one-hot label and $d_{T}=0$, so the recursion runs backwards
from $T-1$. The parameter gradients are then sums over time and over the batch:

$$
\frac{\partial L}{\partial W_x}=\sum_{t} d_t x_t^{\top}, \qquad
\frac{\partial L}{\partial W_h}=\sum_{t} d_t h_{t-1}^{\top}, \qquad
\frac{\partial L}{\partial b_h}=\sum_{t} d_t.
$$

The implementation is that recursion with the batch axis kept and a `carry`
variable standing in for $W_h^{\top}d_{t+1}$:

```python
        for t in range(length - 1, -1, -1):
            normalised, centred, deviation, h_prev = caches[t]
            dh = dh_out[:, t] + carry
```

and the parameter updates are outer products averaged over the batch:

```python
            grads[f"W_x_{suffix}"] += dpre.T @ features[:, t]
            grads[f"W_h_{suffix}"] += dpre.T @ h_prev
            grads[f"b_h_{suffix}"] += dpre.sum(axis=0)
            ...
            carry = dpre @ getattr(self, f"W_h_{suffix}")
```

The `carry` that comes back from the loop is the gradient with respect to the
start state. For `elman` it is summed over the batch and becomes `grads["h0_"]`,
which is exactly what a parameter shared by every sequence should receive.

### Layer normalisation

The normalised cell computes, per sample and per step,

$$
c_t=\frac{a_t-\mu_t}{\sigma_t+\epsilon}, \qquad
z_t=g\odot c_t+\beta, \qquad h_t=\tanh(z_t),
\qquad (5)
$$

with $\mu_t$ and $\sigma_t$ the mean and standard deviation over the hidden
axis. Backpropagation through the normalisation uses the standard identity

$$
\frac{\partial L}{\partial a_t}
=\frac{1}{\sigma_t+\epsilon}\Big(
\delta_t-\overline{\delta}_t-c_t\,\overline{\delta_t c_t}\Big),
\qquad \delta_t=\frac{\partial L}{\partial c_t},
$$

where the bar is the mean over the hidden axis. The code writes it out in one
expression, with the tanh derivative folded in:

```python
                dz = dh * (1.0 - np.tanh(normalised) ** 2)
                grads["ln_gain_"] += (dz * centred).sum(axis=0)
                grads["ln_bias_"] += dz.sum(axis=0)
                dcentred = dz * self.ln_gain_
                dpre = (dcentred - dcentred.mean(axis=1, keepdims=True)
                        - centred * (dcentred * centred).mean(axis=1, keepdims=True)
                        ) / (deviation + EPS)
```

The gain and bias are shared across time and the batch, so their gradients are
sums over both axes. The demo's worst relative error for this branch is
$3.4\times10^{-9}$.

### Truncation

Full BPTT differentiates through every boundary. Truncated BPTT partitions the
sequence into windows $[s,s+k)$ and, for each window, treats the incoming state
as a constant:

$$
\frac{\partial L}{\partial\theta}
\approx\sum_{\text{windows}}\frac{\partial L_{\text{window}}}{\partial\theta},
\qquad \frac{\partial h_s}{\partial\theta}=0.
\qquad (6)
$$

The forward pass still carries $h_s$ into the next window; only the gradient is
cut. The code makes the cut explicit by starting each window from the previous
window's last state and never backpropagating through it:

```python
                hidden, probs, caches = self._forward(window, start=state)
                dlogits = self._loss_gradient(labels[n:n + 1, begin:end], probs)
                grads, _ = self._backward_direction(
                    window, probs, hidden, caches, dlogits @ self.W_o_, dlogits)
                ...
                state = hidden[:, -1]
```

The demonstration is exact: a window that covers the whole sequence
reproduces the full gradient to $3\times10^{-17}$, and shorter windows differ
only by the terms that cross a boundary.

### Bidirectional passes

Two independent cells read the sequence from opposite ends:

$$
h_t^{f}=\phi(h_{t-1}^{f},x_t), \qquad
h_t^{b}=\phi(h_{t+1}^{b},x_t), \qquad
p_t=\mathrm{softmax}\big(W_o[h_t^{f};h_t^{b}]+b_o\big).
\qquad (7)
$$

At prediction time, step $t$ can use information from both sides, so a target
that depends on the future becomes learnable. The gradient of the output layer
splits into a forward block and a reverse block, and each block is fed to its
own backward pass:

```python
        split = dlogits @ self.W_o_
        first, _ = self._backward_direction(
            features, probs, forward, cache_f, split[..., :self.hidden], None)
        second, _ = self._backward_direction(
            features[:, ::-1], probs[:, ::-1], backward, cache_b,
            split[..., self.hidden:][:, ::-1], None, suffix="b_")
```

The two directions share no recurrent parameters (the second cell is
`W_x_b_`, `W_h_b_`, `b_h_b_`), and only the output layer is shared. The demo's
next-symbol task is the clean separation: the forward pass is at chance because
the label is independent of everything it has seen, while the bidirectional
model reaches $0.937$.

### The recurrent Jacobian

The gradient of the last state with respect to an earlier one is a product of
single-step Jacobians:

$$
\frac{\partial h_{T-1}}{\partial h_t}
=\prod_{s=t+1}^{T-1} D_s W_h, \qquad D_s=\mathrm{diag}\big(1-h_s^{2}\big).
\qquad (8)
$$

Its norm is what limits how far credit can travel. If every factor had norm one
the product would be neutral; with $\tanh'\le 1$ and a spectral radius near one,
the product typically decays geometrically. `jacobian_norms` accumulates the
product from the end, so it measures exactly this quantity:

```python
        steps = [np.diag(1.0 - hidden[0, t] ** 2) @ self.W_h_
                 for t in range(length)]
        product = np.eye(self.hidden)
        norms = np.empty(length)
        norms[-1] = 1.0
        for t in range(length - 1, 0, -1):
            product = product @ steps[t]
            norms[t - 1] = float(np.linalg.norm(product))
```

On a fitted parity model of length $32$, the norm at the first step is
$1.78\times10^{-12}$ against $1$ at the last: $11.75$ orders of magnitude, or
one order every $2.64$ steps. That is the phenomenon, not a training artefact.

## Code walkthrough

### One cell, one batch

```python
        pre = x @ W_x.T + h_prev @ W_h.T + b_h
        if self.variant == "jordan":
            pre = pre + y_prev @ self.W_y_.T
        if self.variant == "layer_norm_rnn":
```

Every variant differs only in these few lines: an extra term for Jordan, a
normalisation for `layer_norm_rnn`, and the choice of parameter suffix for the
second direction. The shape is ``(batch, hidden)`` throughout, so the batch is
never a loop.

### Starting state

```python
        if start is not None:
            state = start
        elif self.variant == "elman" and suffix == "":
            state = np.repeat(self.h0_[None, :], batch, axis=0)
        else:
            state = np.zeros((batch, self.hidden))
```

The `start` branch is what makes truncation possible: a window can be given the
previous state without pretending it is the beginning of a sequence. The
`self.repeat` is the learned context of `elman`, broadcast over the batch. In
the backward pass it receives the sum of the per-sequence start gradients,
because the parameter is shared.

### Clipped Adam

```python
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = CLIP / norm if norm > CLIP else 1.0
```

The global norm is clipped at $5$ before the Adam update. Recurrent gradients
can explode when the spectral radius of the recurrence exceeds one, and clipping
is the cheapest guard. It is also a diagnostic risk: a model that only trains
with clipping has a recurrence that wants to explode, and the clip hides the
symptom. `jacobian_norms` is the instrument that shows which regime the fitted
model is in.

### Windowing

```python
            for begin in range(0, length, self.chunk):
                end = min(length, begin + self.chunk)
                window = features[n:n + 1, begin:end]
```

The window loop is the only place where the batch is indexed one sequence at a
time. It has to be: each sequence has its own carry state, and the state is the
thing that must not be differentiated across the boundary. Window gradients are
summed and divided by the batch size, so an epoch is still one step on one
objective, with each window contributing only its local term.

### The Jacobian product

```python
        for t in range(length - 1, 0, -1):
            product = product @ steps[t]
            norms[t - 1] = float(np.linalg.norm(product))
```

`steps[t]` is the Jacobian of step $t$ with respect to step $t-1$. Multiplying
on the right accumulates the product in the correct order, and the norm at each
$t$ is the influence of $h_t$ on the final state. The loop is $O(TH^3)$, which
is why it is a diagnostic and not part of training.

## Complexity

Let $B$ be the batch size, $T$ the sequence length, $H$ the hidden width, $F$
the input width, $C$ the number of classes and $I$ the number of epochs.

| operation | time | memory |
|:--|:--|:--|
| one epoch, forward | $O(BT(HF+H^2+HC))$ | $O(BTH)$ |
| one epoch, backward | same, with a larger constant | $O(BTH)$ |
| `fit` | $O(I\,BT(HF+H^2+HC))$ | $O(BTH)$ |
| truncated, per epoch | same asymptotics, $BT/k$ windows | $O(BkH)$ |
| bidirectional | twice the forward and backward work | $O(BTH)$ |
| `jacobian_norms` | $O(TH^3)$ | $O(H^2)$ |

The quadratic term in $H$ is the recurrence; the $HF$ term is the input
projection. For long sequences the memory is the binding constraint: the cache
stores every intermediate state, so training memory grows linearly in $T$ even
though the parameters do not. Truncation bounds that memory as well as the
gradient horizon, which is the practical reason it is used on long streams.

## Failure modes

**Vanishing gradients.** The demo measures $11.75$ orders of magnitude over
thirty-two steps. A dependency a hundred steps back contributes a gradient
smaller than floating-point noise, so the model cannot learn it no matter how
long it trains. The gated cells of `lstm_gru` and the state-space models of
`ssm` are the two standard fixes; layer normalisation is not one of them, since
it rescales the pre-activation but leaves the product of Jacobians in place.

**Exploding gradients.** The other side of the same product: a spectral radius
above one makes the norm grow exponentially, and one bad batch produces a step
that destroys the weights. Global-norm clipping at $5$ prevents the destruction
but not the underlying instability. If a model only trains with clipping, check
the Jacobian norms before trusting the result.

**Memorisation instead of generalisation.** With a small batch, a vanilla cell
will fit the training sequences by encoding their identities in the state rather
than by learning the rule. The demo's parity task is chosen because it
generalises ($0.980$ held-out); a delayed-copy task with the same model and
thirty-two sequences reaches high training accuracy and near-chance held-out
accuracy. The cure is more data or a smaller model, not more epochs.

**Truncation is a bias, not a bug.** A window that covers the sequence
reproduces the exact gradient ($3\times10^{-17}$), and a window of four changes
it by twice the gradient scale. On parity the truncated model is not worse
($0.873$ against $0.729$) because the forward state carries the answer; on a
task where the difficulty is attributing credit to an early step, the bias is
exactly what fails.

**One direction cannot see the future.** On the next-symbol task the forward
model is at $0.257$ — chance — while the bidirectional model is at $0.937$. If
a target depends on later inputs, no amount of training will help a causal
model; the model class, not the optimiser, is wrong. The price of the
bidirectional fix is that it cannot be used for streaming or autoregressive
decoding, where the future is not available.

**Saturation.** $\tanh$ saturates, so a large pre-activation has a derivative
near zero and the state stops responding. Orthogonal initialisation and
normalisation reduce the chance of starting there, but a learning rate that is
too high can drive the cell into saturation in a few steps. The symptom is a
loss that stalls while the gradients are small but not vanishing with depth.

**Padding and boundaries.** Every sequence in a batch is assumed to start at
step $0$ and to have a label at every step. Real batches contain padding, and
the loss must mask it; otherwise the model learns to predict the pad symbol and
the state is polluted at the boundary. The code refuses ragged labels rather
than guessing an alignment, which is the honest default for a single-file
implementation.

**Layer normalisation is not a memory fix.** It improves the conditioning of
the optimisation and the demo's gains move a long way from one, but it does not
change the Jacobian product of (8). A layer-normalised vanilla RNN still cannot
learn a dependency of a hundred steps.

## Applications

**Speech recognition.** Before CTC and attention, hybrid systems used recurrent
networks over acoustic frames, and the bidirectional variant was standard
because the whole utterance is available at transcription time. The `ctc` atom
in this chapter is the alignment-free successor, and the `hmm` atom is the
generative model it replaced.

**Language modelling.** A recurrent network trained to predict the next token is
the direct ancestor of the transformer. The `ngram_lm` atom is the counting
baseline, and the `attention` atom is the architecture that removed the
recurrence — but the training objective, next-token cross-entropy, is the same
one in (3).

**Time series and sensor streams.** A state that carries history is the natural
model for a stream, and the truncated schedule is how it is trained when the
stream is longer than memory. The `tcn` and `ssm` atoms are the convolutional
and state-space answers to the same problem.

**Control and reinforcement learning.** A policy with memory is a recurrent
network whose input is the observation and whose output is an action
distribution; the `actor-critic` and `PPO` atoms in the reinforcement chapter use
exactly this cell when the environment is partially observed.

**Handwriting and gesture recognition.** Pen trajectories are variable-length
sequences with long-range style, and bidirectional recurrent models were the
state of the art before sequence-to-sequence attention. The `dtw` atom is the
template-matching alternative for the same data.

**Clinical and industrial monitoring.** A patient's observations arrive as an
irregular stream, and a recurrent model with a carried state is the standard
baseline for deterioration and fault detection. Truncation and masking are the
two engineering details that decide whether it works.

## References

- Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning
  representations by back-propagating errors. *Nature* 323, 533–536. The
  backpropagation algorithm that BPTT unrolls.
- Jordan, M. I. (1986). Attractor dynamics and parallelism in a connectionist
  sequential machine. *Proceedings of the Eighth Annual Conference of the
  Cognitive Science Society*, 531–546. The output recurrence.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science* 14(2),
  179–211. The hidden recurrence and the context units.
- Werbos, P. J. (1990). Backpropagation through time: what it does and how to do
  it. *Proceedings of the IEEE* 78(10), 1550–1560. The name and the derivation.
- Bengio, Y., Simard, P. & Frasconi, P. (1994). Learning long-term dependencies
  with gradient descent is difficult. *IEEE Transactions on Neural Networks*
  5(2), 157–166. Why the Jacobian product decays.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation* 9(8), 1735–1780. The gated fix, implemented in `lstm_gru`.
- Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training
  recurrent neural networks. *ICML*, 1310–1318. Clipping and the Jacobian
  analysis.
- Sutskever, I. (2013). *Training Recurrent Neural Networks*. PhD thesis,
  University of Toronto. Truncation, momentum and the practical schedule.
- Cho, K. et al. (2014). Learning phrase representations using RNN
  encoder–decoder for statistical machine translation. *EMNLP*, 1724–1734. The
  gated recurrent unit and the sequence-to-sequence setting.
- Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016). Layer normalization.
  *arXiv:1607.06450*. The normalised cell of (5).
